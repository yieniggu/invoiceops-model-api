"""The allowlist is explicit: every supported estimator passes, anything else is rejected."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import skops.io as sio
from invoiceops_ml.contract import FEATURES, TARGET
from invoiceops_ml.synthetic import generate_invoices
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from invoiceops_model_api.skops_policy import (
    ALLOWED_SKOPS_TYPES,
    SkopsPolicyError,
    check_allowlist,
    load_allowlisted,
)

LIMIT = 256 * 1024 * 1024
F = list(FEATURES)

ESTIMATORS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)),
    "rf": lambda: RandomForestClassifier(n_estimators=5, random_state=0),
    "extra": lambda: ExtraTreesClassifier(n_estimators=5, random_state=0),
    "gb": lambda: GradientBoostingClassifier(n_estimators=5, random_state=0),
    "hgb": lambda: HistGradientBoostingClassifier(max_iter=5, random_state=0),
    "tree": lambda: DecisionTreeClassifier(max_depth=3, random_state=0),
    "dummy": lambda: DummyClassifier(strategy="prior"),
    "calibrated": lambda: CalibratedClassifierCV(LogisticRegression(max_iter=300), cv=3),
    "column_transformer": lambda: Pipeline(
        [
            (
                "ct",
                ColumnTransformer(
                    [
                        ("s", StandardScaler(), F[:3]),
                        ("m", MinMaxScaler(), F[3:5]),
                        ("r", RobustScaler(), F[5:7]),
                        ("o", OneHotEncoder(handle_unknown="ignore"), [F[2]]),
                    ],
                    remainder="passthrough",
                ),
            ),
            ("imp", SimpleImputer()),
            ("clf", HistGradientBoostingClassifier(max_iter=5, random_state=0)),
        ]
    ),
}


@pytest.fixture(scope="module")
def data():
    d = generate_invoices(n_invoices=800, seed=3)
    return d[F], d[TARGET]


@pytest.mark.parametrize("name", sorted(ESTIMATORS))
def test_supported_estimators_are_allowlisted_and_load(name: str, data, tmp_path: Path) -> None:
    X, y = data
    model = ESTIMATORS[name]().fit(X, y)
    path = tmp_path / "model.skops"
    sio.dump(model, path)
    check_allowlist(path, LIMIT)
    loaded = load_allowlisted(path, LIMIT)
    assert loaded.predict_proba(X.head(3)).shape == (3, 2)


def test_everything_skops_reports_untrusted_is_in_the_allowlist(data, tmp_path: Path) -> None:
    X, y = data
    seen: set[str] = set()
    for build in ESTIMATORS.values():
        path = tmp_path / "m.skops"
        sio.dump(build().fit(X, y), path)
        seen |= set(sio.get_untrusted_types(file=path))
    assert seen <= ALLOWED_SKOPS_TYPES


def _archive(path: Path, schema: dict) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("schema.json", json.dumps(schema))


def test_function_node_pointing_at_os_system_is_rejected_without_import(tmp_path: Path) -> None:
    path = tmp_path / "evil.skops"
    _archive(
        path,
        {
            "__class__": "system",
            "__module__": "os",
            "__loader__": "FunctionNode",
            "content": {"module_path": "os", "function": "system"},
        },
    )
    with pytest.raises(SkopsPolicyError) as info:
        check_allowlist(path, LIMIT)
    assert info.value.code == "UNTRUSTED_TYPE"


def test_unexpected_archive_members_and_traversal_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.skops"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("schema.json", "{}")
        z.writestr("../evil.py", "x")
    with pytest.raises(SkopsPolicyError) as info:
        check_allowlist(path, LIMIT)
    assert info.value.code == "SKOPS_ARCHIVE_INVALID"


def test_archive_expanding_beyond_limit_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "big.skops"
    _archive(path, {"__class__": "dict", "__module__": "builtins", "__loader__": "DictNode"})
    with pytest.raises(SkopsPolicyError) as info:
        check_allowlist(path, 10)
    assert info.value.code == "ARTIFACT_TOO_LARGE"


def test_not_a_zip_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "x.skops"
    path.write_bytes(b"\x80\x04pickle-ish")
    with pytest.raises(SkopsPolicyError):
        check_allowlist(path, LIMIT)
