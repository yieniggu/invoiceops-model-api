"""Crafted tree arrays: rejected by structural validation, and contained by process isolation."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import make_settings
from evil import HGB_ATTACKS, TREE_ATTACKS, dump, tamper, training_data
from invoiceops_ml.contract import FEATURES
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from invoiceops_model_api.isolation import CheckRunner
from invoiceops_model_api.model_runtime import ModelRejected, load_checked
from invoiceops_model_api.skops_policy import load_allowlisted
from invoiceops_model_api.structure import StructureError, validate_model_structure

HELPERS = Path(__file__).parent / "helpers"
UNVALIDATED_CHILD = (sys.executable, "-I", str(HELPERS / "unvalidated_child.py"))
LIMIT = 256 * 1024 * 1024
COLUMNS = list(FEATURES)

TREE_MODELS = {
    "tree": lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
    "forest": lambda: RandomForestClassifier(n_estimators=4, max_depth=3, random_state=0),
    "extra": lambda: ExtraTreesClassifier(n_estimators=3, max_depth=3, random_state=0),
    "gb": lambda: GradientBoostingClassifier(n_estimators=3, max_depth=2, random_state=0),
    "calibrated_forest": lambda: CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0), cv=3
    ),
    "pipeline_tree": lambda: make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=3)),
}
HGB_MODELS = {
    "hgb": lambda: HistGradientBoostingClassifier(max_iter=4, random_state=0),
    "pipeline_hgb": lambda: make_pipeline(
        StandardScaler(), HistGradientBoostingClassifier(max_iter=3, random_state=0)
    ),
    "calibrated_hgb": lambda: CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_iter=3, random_state=0), cv=3
    ),
}


@pytest.fixture(scope="module")
def dumped(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("structure")
    return {
        name: dump(factory(), root / f"{name}.skops")
        for name, factory in {**TREE_MODELS, **HGB_MODELS}.items()
    }


@pytest.mark.parametrize("name", [*TREE_MODELS, *HGB_MODELS])
def test_valid_models_pass_structure_validation(dumped, name: str) -> None:
    model = load_allowlisted(dumped[name], LIMIT)
    validate_model_structure(model)  # does not raise
    load_checked(dumped[name], LIMIT)


@pytest.mark.parametrize("attack", sorted(TREE_ATTACKS))
@pytest.mark.parametrize("name", sorted(TREE_MODELS))
def test_evil_tree_nodes_are_rejected(dumped, tmp_path: Path, name: str, attack: str) -> None:
    evil = tamper(dumped[name], tmp_path / "evil.skops", TREE_ATTACKS[attack])
    model = load_allowlisted(evil, LIMIT)  # skops itself happily loads the crafted arrays
    with pytest.raises(StructureError):
        validate_model_structure(model)
    with pytest.raises(ModelRejected) as info:
        load_checked(evil, LIMIT)
    assert info.value.code == "MODEL_STRUCTURE_INVALID"


@pytest.mark.parametrize("attack", sorted(HGB_ATTACKS))
@pytest.mark.parametrize("name", sorted(HGB_MODELS))
def test_evil_hist_gradient_boosting_nodes_are_rejected(
    dumped, tmp_path: Path, name: str, attack: str
) -> None:
    evil = tamper(dumped[name], tmp_path / "evil.skops", HGB_ATTACKS[attack])
    with pytest.raises(ModelRejected) as info:
        load_checked(evil, LIMIT)
    assert info.value.code == "MODEL_STRUCTURE_INVALID"


def test_tree_feature_count_must_match_the_estimator(dumped, tmp_path: Path) -> None:
    model = load_allowlisted(dumped["forest"], LIMIT)
    model.n_features_in_ = 3  # sklearn would validate X against this, the Cython tree would not
    with pytest.raises(StructureError):
        validate_model_structure(model)


# ---- process isolation: a crafted model that BYPASSES validation cannot hurt the parent ------


def runner(tmp_path: Path, *, timeout: float = 120.0, memory_mb: int = 3072) -> CheckRunner:
    settings = make_settings(
        "http://127.0.0.1:1", check_timeout_seconds=timeout, check_memory_mb=memory_mb
    )
    return CheckRunner(settings, entrypoint=UNVALIDATED_CHILD)


def matrix(rows: int = 50) -> np.ndarray:
    X, _ = training_data()
    return X.to_numpy(dtype="float64")[:rows]


def test_unvalidated_child_survives_a_valid_model(dumped, tmp_path: Path) -> None:
    """Control: the test entry point works, so crashes below are caused by the evil arrays."""
    outcome = runner(tmp_path).run(dumped["tree"], matrix(), COLUMNS)
    assert outcome.ok and outcome.scores is not None and len(outcome.scores) == 50


def test_out_of_range_child_crashes_only_the_child(dumped, tmp_path: Path) -> None:
    evil = tamper(dumped["tree"], tmp_path / "evil.skops", TREE_ATTACKS["child_out_of_range"])
    outcome = runner(tmp_path).run(evil, matrix(), COLUMNS)
    assert not outcome.ok
    assert outcome.code == "MODEL_CHECK_CRASHED"
    assert outcome.detail and outcome.detail.get("signal") in {"SIGSEGV", "SIGBUS", "SIGABRT"}


def test_cyclic_child_hangs_only_the_child_and_times_out(dumped, tmp_path: Path) -> None:
    evil = tamper(dumped["tree"], tmp_path / "evil.skops", TREE_ATTACKS["child_cycle"])
    started = time.monotonic()
    outcome = runner(tmp_path, timeout=5).run(evil, matrix(), COLUMNS)
    assert outcome.code == "MODEL_CHECK_TIMEOUT"
    assert time.monotonic() - started >= 5  # it was the timeout that ended the run, not a crash


def test_default_child_rejects_the_same_models_with_a_reason(dumped, tmp_path: Path) -> None:
    default = CheckRunner(make_settings("http://127.0.0.1:1"))
    for attack in ("child_out_of_range", "child_cycle", "bad_feature"):
        evil = tamper(dumped["tree"], tmp_path / f"{attack}.skops", TREE_ATTACKS[attack])
        outcome = default.run(evil, matrix(), COLUMNS)
        assert outcome.code == "MODEL_STRUCTURE_INVALID", attack
