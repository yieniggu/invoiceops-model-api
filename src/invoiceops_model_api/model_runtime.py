"""Loading and scoring of student models. IMPORTED ONLY BY CHILD PROCESSES.

The HTTP process never imports this module and never deserialises or calls a student model:
everything that touches model bytes runs in ``_child`` (deploy check) or ``_worker``
(serving), which are separate, resource-limited processes.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from invoiceops_ml.contract import FEATURES

from invoiceops_model_api.skops_policy import SkopsPolicyError, load_allowlisted
from invoiceops_model_api.structure import StructureError, validate_model_structure


class ModelRejected(Exception):
    """The model is rejected with a stable reason code."""

    def __init__(self, code: str, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail or {}


def positive_index(model: object) -> int:
    """Index of class 1 in a binary 0/1 classifier; anything else is ``INVALID_CLASSES``."""
    invalid = ModelRejected("INVALID_CLASSES", "The model must be a binary classifier (0/1)")
    try:
        classes = list(getattr(model, "classes_", []))
        if len(classes) != 2:
            raise invalid
        ints: list[int] = []
        for c in classes:
            # Strings (even "0"/"1"), objects, NaN and fractional values are not 0/1 labels.
            if isinstance(c, bool | np.bool_):
                ints.append(int(c))
            elif isinstance(c, int | float | np.integer | np.floating):
                value = float(c)
                if not math.isfinite(value) or not value.is_integer():
                    raise invalid
                ints.append(int(value))
            else:
                raise invalid
    except ModelRejected:
        raise
    except Exception:  # noqa: BLE001
        raise invalid from None
    if sorted(ints) != [0, 1]:
        raise invalid
    return ints.index(1)


def predict_positive(model: object, matrix: np.ndarray, columns: list[str], pos: int) -> np.ndarray:
    """Probability of class 1 for ``matrix`` (rows in ``FEATURES`` order), validated."""
    frame = pd.DataFrame(matrix, columns=list(FEATURES))
    proba = np.asarray(model.predict_proba(frame[columns]), dtype=float)  # type: ignore[attr-defined]
    if proba.ndim != 2 or proba.shape != (len(frame), 2):
        raise ValueError("predict_proba must return (n, 2)")
    out = proba[:, pos]
    if not np.all(np.isfinite(out)) or out.min() < 0 or out.max() > 1:
        raise ValueError("invalid probabilities")
    return out


def load_checked(
    path: Path, max_uncompressed_bytes: int, *, validate_structure: bool = True
) -> tuple[object, int]:
    """Allowlisted skops load + structural validation + class check -> (model, positive index).

    ``validate_structure=False`` exists ONLY so tests can prove the surrounding process
    isolation contains a crafted model; production entry points always validate.
    """
    try:
        model = load_allowlisted(path, max_uncompressed_bytes)
    except SkopsPolicyError as error:
        raise ModelRejected(error.code, str(error), error.detail) from None
    except MemoryError:
        raise
    except Exception:  # noqa: BLE001 - never echo loader internals
        raise ModelRejected("MODEL_LOAD_FAILED", "The model could not be loaded") from None
    if validate_structure:
        try:
            validate_model_structure(model)
        except StructureError as error:
            raise ModelRejected(
                "MODEL_STRUCTURE_INVALID",
                "The model contains an invalid tree structure",
                error.detail,
            ) from None
    if not hasattr(model, "predict_proba"):
        raise ModelRejected("NO_PREDICT_PROBA", "The model has no predict_proba")
    return model, positive_index(model)
