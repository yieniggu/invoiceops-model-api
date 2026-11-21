"""Explicit skops allowlist and archive inspection.

skops is only ever loaded after every node type in the archive is on ALLOWED_SKOPS_TYPES,
and only the subset of those that skops itself reports as untrusted is passed as
``trusted``. ``trusted=True`` is never used. The allowlist is checked against the archive's
own ``schema.json`` (all nodes, not only skops' "untrusted" ones), so a type that skops
trusts by default but is not enumerated here (for example an SVM) is rejected too.

The list was produced empirically: each supported estimator/pipeline was fitted on the
synthetic invoice data, saved with skops 0.14.0 and its schema walked (see
``tests/test_skops_policy.py``, which re-derives it).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

ALLOWED_SKOPS_TYPES: frozenset[str] = frozenset(
    {
        # containers / python builtins
        "builtins.dict",
        "builtins.list",
        "builtins.slice",
        "builtins.str",
        "builtins.tuple",
        # numpy
        "numpy.dtype",
        "numpy.float64",
        "numpy.int64",
        "numpy.uint64",
        "numpy.ndarray",
        "numpy.random._generator.Generator",
        "numpy.random.mtrand.RandomState",
        # estimators and transformers
        "sklearn.pipeline.Pipeline",
        "sklearn.compose._column_transformer.ColumnTransformer",
        "sklearn.preprocessing._data.StandardScaler",
        "sklearn.preprocessing._data.MinMaxScaler",
        "sklearn.preprocessing._data.RobustScaler",
        "sklearn.preprocessing._encoders.OneHotEncoder",
        "sklearn.preprocessing._function_transformer.FunctionTransformer",
        "sklearn.preprocessing._label.LabelEncoder",
        "sklearn.impute._base.SimpleImputer",
        "sklearn.linear_model._logistic.LogisticRegression",
        "sklearn.ensemble._forest.RandomForestClassifier",
        "sklearn.ensemble._forest.ExtraTreesClassifier",
        "sklearn.ensemble._gb.GradientBoostingClassifier",
        "sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier",
        "sklearn.tree._classes.DecisionTreeClassifier",
        "sklearn.tree._classes.DecisionTreeRegressor",
        "sklearn.tree._classes.ExtraTreeClassifier",
        "sklearn.dummy.DummyClassifier",
        "sklearn.calibration.CalibratedClassifierCV",
        # internal types skops reports for the estimators above
        "sklearn.calibration._CalibratedClassifier",
        "sklearn.calibration._SigmoidCalibration",
        "sklearn.ensemble._hist_gradient_boosting.binning._BinMapper",
        "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
        "sklearn.tree._tree.Tree",
        "sklearn._loss._loss.CyHalfBinomialLoss",
        "sklearn._loss.link.Interval",
        "sklearn._loss.link.LogitLink",
        "sklearn._loss.loss.HalfBinomialLoss",
    }
)

# skops node loaders that may appear. Anything else (function, method, operator, partial,
# reduce...) can reference arbitrary callables and is rejected.
ALLOWED_SKOPS_LOADERS: frozenset[str] = frozenset(
    {
        "DTypeNode",
        "DictNode",
        "JsonNode",
        "ListNode",
        "LossNode",
        "NdArrayNode",
        "ObjectNode",
        "RandomGeneratorNode",
        "RandomStateNode",
        "SliceNode",
        "TreeNode",
        "TupleNode",
        "TypeNode",
    }
)

_ALLOWED_MEMBER_SUFFIXES = (".npy", "schema.json")


class SkopsPolicyError(ValueError):
    def __init__(self, code: str, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail or {}


def _walk(node: Any, types: set[str], loaders: set[str]) -> None:
    if isinstance(node, dict):
        if "__module__" in node and "__class__" in node:
            types.add(f"{node['__module__']}.{node['__class__']}")
        if "__loader__" in node:
            loaders.add(str(node["__loader__"]))
        for value in node.values():
            _walk(value, types, loaders)
    elif isinstance(node, list):
        for value in node:
            _walk(value, types, loaders)


def inspect_archive(path: Path, max_uncompressed_bytes: int) -> tuple[set[str], set[str]]:
    """Return (types, loaders) found in a skops archive without deserialising any object."""
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            total = 0
            for info in infos:
                name = info.filename
                if name.startswith("/") or ".." in name.split("/") or "\\" in name:
                    raise SkopsPolicyError("SKOPS_ARCHIVE_INVALID", "Unsafe member path")
                if not name.endswith(_ALLOWED_MEMBER_SUFFIXES):
                    raise SkopsPolicyError("SKOPS_ARCHIVE_INVALID", "Unexpected archive member")
                total += info.file_size
                if total > max_uncompressed_bytes:
                    raise SkopsPolicyError("ARTIFACT_TOO_LARGE", "Archive expands beyond the limit")
            try:
                schema = json.loads(archive.read("schema.json"))
            except KeyError:
                raise SkopsPolicyError("SKOPS_ARCHIVE_INVALID", "schema.json is missing") from None
    except zipfile.BadZipFile:
        raise SkopsPolicyError("SKOPS_ARCHIVE_INVALID", "Not a skops archive") from None
    types: set[str] = set()
    loaders: set[str] = set()
    _walk(schema, types, loaders)
    return types, loaders


def check_allowlist(path: Path, max_uncompressed_bytes: int) -> list[str]:
    """Raise ``SkopsPolicyError`` unless every type is allowlisted.

    Returns the types skops reports as untrusted (all of them are allowlisted), to be passed
    as the explicit ``trusted`` list to ``skops.io.load``.
    """
    import skops.io as sio

    types, loaders = inspect_archive(path, max_uncompressed_bytes)
    bad_loaders = sorted(loaders - ALLOWED_SKOPS_LOADERS)
    if bad_loaders:
        raise SkopsPolicyError(
            "UNTRUSTED_TYPE", "Disallowed skops node kinds", {"types": bad_loaders}
        )
    disallowed = sorted(types - ALLOWED_SKOPS_TYPES)
    if disallowed:
        raise SkopsPolicyError("UNTRUSTED_TYPE", "Disallowed types", {"types": disallowed[:20]})
    untrusted = sorted(sio.get_untrusted_types(file=path))
    extra = sorted(set(untrusted) - ALLOWED_SKOPS_TYPES)
    if extra:
        raise SkopsPolicyError("UNTRUSTED_TYPE", "Disallowed types", {"types": extra[:20]})
    return untrusted


def load_allowlisted(path: Path, max_uncompressed_bytes: int) -> object:
    trusted = check_allowlist(path, max_uncompressed_bytes)
    import skops.io as sio

    return sio.load(path, trusted=trusted)
