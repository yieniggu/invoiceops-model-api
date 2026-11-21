"""Helpers that build hostile skops archives by tampering with the .npy members of real ones.

skops only instantiates allowlisted types, but the arrays inside (tree node tables) are
attacker-controlled bytes: these helpers rewrite them the way a student would.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import skops.io as sio
from invoiceops_ml.contract import FEATURES, TARGET
from invoiceops_ml.synthetic import generate_invoices

Mutator = Callable[[np.ndarray], np.ndarray | None]


def tamper(src: Path, dst: Path, mutate: Mutator) -> Path:
    """Copy a skops archive applying ``mutate`` to every .npy array (None = keep)."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename.endswith(".npy"):
                array = np.load(io.BytesIO(data), allow_pickle=False)
                changed = mutate(array.copy())
                if changed is not None:
                    buffer = io.BytesIO()
                    np.save(buffer, changed, allow_pickle=False)
                    data = buffer.getvalue()
            zout.writestr(info.filename, data)
    return dst


def _tree_nodes(array: np.ndarray) -> bool:
    return array.dtype.names is not None and "left_child" in array.dtype.names


def _hgb_nodes(array: np.ndarray) -> bool:
    return array.dtype.names is not None and "is_leaf" in array.dtype.names


def tree_child_out_of_range(array: np.ndarray) -> np.ndarray | None:
    if not _tree_nodes(array):
        return None
    array["left_child"][0] = 10**9
    array["right_child"][0] = 10**9 + 1
    return array


def tree_child_cycle(array: np.ndarray) -> np.ndarray | None:
    if not _tree_nodes(array):
        return None
    array["left_child"][0] = 0
    array["right_child"][0] = 0
    return array


def tree_bad_feature(array: np.ndarray) -> np.ndarray | None:
    if not _tree_nodes(array):
        return None
    array["feature"][0] = 10**6
    return array


def tree_single_child(array: np.ndarray) -> np.ndarray | None:
    if not _tree_nodes(array):
        return None
    array["right_child"][0] = -1
    return array


def hgb_child_out_of_range(array: np.ndarray) -> np.ndarray | None:
    if not _hgb_nodes(array):
        return None
    array["left"][0] = 10**6
    array["right"][0] = 10**6 + 1
    return array


def hgb_child_cycle(array: np.ndarray) -> np.ndarray | None:
    if not _hgb_nodes(array):
        return None
    array["left"][0] = 0
    array["right"][0] = 0
    return array


def hgb_bad_feature(array: np.ndarray) -> np.ndarray | None:
    if not _hgb_nodes(array):
        return None
    array["feature_idx"][0] = 10**6
    return array


def hgb_bad_leaf_flag(array: np.ndarray) -> np.ndarray | None:
    if not _hgb_nodes(array):
        return None
    array["is_leaf"][0] = 2
    return array


def hgb_categorical(array: np.ndarray) -> np.ndarray | None:
    if not _hgb_nodes(array):
        return None
    array["is_categorical"][0] = 1
    return array


TREE_ATTACKS: dict[str, Mutator] = {
    "child_out_of_range": tree_child_out_of_range,
    "child_cycle": tree_child_cycle,
    "bad_feature": tree_bad_feature,
    "single_child": tree_single_child,
}
HGB_ATTACKS: dict[str, Mutator] = {
    "child_out_of_range": hgb_child_out_of_range,
    "child_cycle": hgb_child_cycle,
    "bad_feature": hgb_bad_feature,
    "bad_leaf_flag": hgb_bad_leaf_flag,
    "categorical": hgb_categorical,
}


def training_data() -> tuple[object, object]:
    data = generate_invoices(n_invoices=800, seed=3)
    return data[list(FEATURES)], data[TARGET]


def dump(estimator, path: Path) -> Path:
    X, y = training_data()
    estimator.fit(X, y)
    sio.dump(estimator, path)
    return path
