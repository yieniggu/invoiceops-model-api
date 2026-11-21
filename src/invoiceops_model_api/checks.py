"""Deploy check: format, signature, model interface and the server-side AUC gate.

The HTTP process only does the cheap, model-free parts here (signature, gate arithmetic). Loading
the model and running ``predict_proba`` on the official validation features happens in a
resource-limited child process (``isolation.CheckRunner``); the parent receives only scores,
validates them and computes AUC itself, so the validation labels never leave this process.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from invoiceops_ml.contract import FEATURE_SCHEMA_VERSION, FEATURES, TARGET
from invoiceops_ml.synthetic import generate_invoices
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

from invoiceops_model_api.artifacts import ArtifactError, FetchedModel
from invoiceops_model_api.isolation import CheckRunner
from invoiceops_model_api.settings import Settings

# Returned metrics are rounded: with full precision a student could use many deploy checks as
# an oracle to recover the validation labels. (MLOPS-05 must also rate-limit deploy-checks
# per group; the gate itself compares unrounded values.)
METRIC_DECIMALS = 2


class GateUnavailable(Exception):
    """The official validation data cannot be produced; the gate must fail closed."""


@dataclass
class Reason:
    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


@dataclass
class CheckResult:
    passed: bool
    reasons: list[Reason]
    metrics: dict[str, Any] | None = None
    columns: list[str] | None = None
    positive_index: int = 1


def check_signature(mlmodel: dict) -> list[str]:
    """Return the input columns in signature order; raise ``ArtifactError`` on mismatch."""
    signature = mlmodel.get("signature")
    if not isinstance(signature, dict) or not signature.get("inputs"):
        raise ArtifactError("SIGNATURE_MISSING", "The model has no input signature")
    try:
        inputs = json.loads(signature["inputs"])
    except (TypeError, ValueError):
        raise ArtifactError("SIGNATURE_MISMATCH", "Input signature is unreadable") from None
    if not isinstance(inputs, list) or not all(isinstance(c, dict) for c in inputs):
        raise ArtifactError("SIGNATURE_MISMATCH", "Input signature is unreadable")
    names = [c.get("name") for c in inputs]
    bad_types = sorted({str(c.get("type")) for c in inputs if c.get("type") != "double"})
    if sorted(map(str, names)) != sorted(FEATURES) or len(set(names)) != len(names) or bad_types:
        raise ArtifactError(
            "SIGNATURE_MISMATCH",
            f"Inputs must be exactly the {FEATURE_SCHEMA_VERSION} features as double",
            {
                "expected": list(FEATURES),
                "found": [str(n) for n in names][:32],
                "nonDoubleTypes": bad_types[:8],
                "schemaVersion": FEATURE_SCHEMA_VERSION,
            },
        )
    return [str(n) for n in names]


@lru_cache(maxsize=4)
def _official_frame(seed: int, rows: int) -> pd.DataFrame:
    frame = generate_invoices(n_invoices=rows, seed=seed)
    frame = frame[[*FEATURES, TARGET]].astype({f: "float64" for f in FEATURES}).copy()
    if frame[TARGET].nunique() < 2:
        raise GateUnavailable("validation set has a single class")
    frame.attrs["readonly"] = True
    return frame


def official_frame(settings: Settings) -> pd.DataFrame:
    if settings.official_validation_seed is None:
        raise GateUnavailable("OFFICIAL_VALIDATION_SEED is not configured")
    return _official_frame(settings.official_validation_seed, settings.official_validation_rows)


def dummy_auc(frame: pd.DataFrame) -> float:
    dummy = DummyClassifier(strategy="prior").fit(frame[list(FEATURES)], frame[TARGET])
    scores = dummy.predict_proba(frame[list(FEATURES)])[:, 1]
    return float(roc_auc_score(frame[TARGET], scores))


def run_check(fetched: FetchedModel, settings: Settings, runner: CheckRunner) -> CheckResult:
    """Validate a downloaded model. Student-logged metrics are never read."""
    frame = official_frame(settings)  # raises GateUnavailable (fail closed) before any loading
    reasons: list[Reason] = []
    try:
        columns = check_signature(fetched.mlmodel)
    except ArtifactError as error:
        return CheckResult(False, [Reason(error.code, str(error), error.detail)])
    matrix = frame[list(FEATURES)].to_numpy(dtype="float64")
    outcome = runner.run(fetched.skops_path, matrix, columns)
    if not outcome.ok:
        return CheckResult(False, [Reason(outcome.code, outcome.message, outcome.detail or {})])
    scores = outcome.scores
    if (
        scores is None
        or scores.shape != (len(frame),)
        or not np.all(np.isfinite(scores))
        or scores.min() < 0
        or scores.max() > 1
    ):
        return CheckResult(
            False, [Reason("INFERENCE_FAILED", "predict_proba failed on the validation set")]
        )
    auc = float(roc_auc_score(frame[TARGET], scores))
    baseline = dummy_auc(frame)
    lift = auc - baseline
    if not math.isfinite(auc):
        return CheckResult(False, [Reason("INFERENCE_FAILED", "AUC is not finite")])
    metrics = {
        "auc": round(auc, METRIC_DECIMALS),
        "dummyAuc": round(baseline, METRIC_DECIMALS),
        "lift": round(lift, METRIC_DECIMALS),
        "minAuc": settings.gate_min_auc,
        "minLift": settings.gate_min_lift,
        "validationRows": len(frame),
    }
    if auc < settings.gate_min_auc:
        reasons.append(
            Reason(
                "AUC_BELOW_MIN",
                "AUC is below the minimum",
                {"auc": metrics["auc"], "minAuc": settings.gate_min_auc},
            )
        )
    if lift < settings.gate_min_lift:
        reasons.append(
            Reason(
                "AUC_LIFT_BELOW_MIN",
                "AUC does not beat the dummy baseline by the required lift",
                {
                    "auc": metrics["auc"],
                    "dummyAuc": metrics["dummyAuc"],
                    "lift": metrics["lift"],
                    "minLift": settings.gate_min_lift,
                },
            )
        )
    return CheckResult(not reasons, reasons, metrics, columns, outcome.positive_index)
