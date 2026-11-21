"""Structural validation of a loaded model's tree arrays (runs ONLY inside child processes).

skops guarantees that only allowlisted *types* are instantiated, but the numpy arrays inside
(``sklearn.tree._tree.Tree`` nodes, ``TreePredictor.nodes``) are attacker-controlled bytes that
sklearn's Cython code indexes WITHOUT bounds checks. A crafted child index or feature index
reads out of bounds (SIGBUS/SIGSEGV) and a cyclic child makes prediction loop forever.

This validator walks the estimator object graph (pipelines, forests, gradient boosting,
calibrated wrappers, ...) and checks every tree *before any predict*:

* ``Tree``: ``1 <= node_count <= capacity``; every internal node has both children in
  ``(node_id, node_count)`` (strictly greater than the parent, so the graph is acyclic and every
  traversal terminates) and leaves have both children ``-1``; internal ``feature`` is in
  ``[0, n_features)``; ``values`` shape is ``(capacity, n_outputs, max_n_classes)``; the tree's
  ``n_features`` equals the ``n_features_in_`` of the tree-family estimators that own it
  (sklearn validates that against the real input width, the Cython tree code does not);
* ``TreePredictor`` (HistGradientBoosting): ``is_leaf`` is 0/1, internal nodes have both
  children in ``(node_id, n)`` and ``feature_idx < n_features``; categorical splits (which index
  extra bitset arrays) are refused because the feature contract is numeric only.

Any violation raises ``StructureError`` (reason code ``MODEL_STRUCTURE_INVALID``).
"""

from __future__ import annotations

import numpy as np

MAX_OBJECTS = 2_000_000
MAX_DEPTH = 200


class StructureError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.detail = {"check": message}


def _fail(message: str) -> StructureError:
    return StructureError(message)


def validate_tree(tree: object, owner_n_features: tuple[int, ...] = ()) -> None:
    """Validate one ``sklearn.tree._tree.Tree``."""
    try:
        state = tree.__getstate__()  # type: ignore[attr-defined]
        nodes = state["nodes"]
        values = state["values"]
        node_count = int(state["node_count"])
        n_features = int(tree.n_features)  # type: ignore[attr-defined]
        n_outputs = int(tree.n_outputs)  # type: ignore[attr-defined]
        max_n_classes = int(tree.max_n_classes)  # type: ignore[attr-defined]
        left_all = np.asarray(nodes["left_child"])
        right_all = np.asarray(nodes["right_child"])
        feature_all = np.asarray(nodes["feature"])
    except Exception:  # noqa: BLE001
        raise _fail("tree arrays are unreadable") from None
    capacity = int(nodes.shape[0])
    if node_count < 1 or node_count > capacity:
        raise _fail("tree node_count is inconsistent with the node array")
    if n_features < 1 or n_outputs < 1 or max_n_classes < 1:
        raise _fail("tree dimensions are invalid")
    if tuple(values.shape) != (capacity, n_outputs, max_n_classes):
        raise _fail("tree value array shape is inconsistent")
    for expected in owner_n_features:
        if n_features != expected:
            raise _fail("tree feature count differs from the estimator's n_features_in_")
    left = left_all[:node_count].astype(np.int64)
    right = right_all[:node_count].astype(np.int64)
    feature = feature_all[:node_count].astype(np.int64)
    ids = np.arange(node_count, dtype=np.int64)
    leaf = (left == -1) & (right == -1)
    internal = (left != -1) & (right != -1)
    if not np.all(leaf | internal):
        raise _fail("tree node has exactly one child")
    if np.any(internal & ((left <= ids) | (left >= node_count) | (right <= ids))):
        raise _fail("tree child index is out of range or not after its parent")
    if np.any(internal & (right >= node_count)):
        raise _fail("tree child index is out of range or not after its parent")
    if np.any(internal & ((feature < 0) | (feature >= n_features))):
        raise _fail("tree split feature is out of range")


def validate_tree_predictor(predictor: object, owner_n_features: tuple[int, ...] = ()) -> None:
    """Validate one ``TreePredictor`` (HistGradientBoosting)."""
    try:
        nodes = predictor.nodes  # type: ignore[attr-defined]
        left = np.asarray(nodes["left"]).astype(np.int64)
        right = np.asarray(nodes["right"]).astype(np.int64)
        feature = np.asarray(nodes["feature_idx"]).astype(np.int64)
        is_leaf = np.asarray(nodes["is_leaf"]).astype(np.int64)
        categorical = np.asarray(nodes["is_categorical"]).astype(np.int64)
    except Exception:  # noqa: BLE001
        raise _fail("hist-gradient-boosting nodes are unreadable") from None
    n = int(left.shape[0])
    if n < 1 or not (left.shape == right.shape == feature.shape == is_leaf.shape):
        raise _fail("hist-gradient-boosting node array is empty or inconsistent")
    if np.any((is_leaf != 0) & (is_leaf != 1)):
        raise _fail("hist-gradient-boosting is_leaf flag is invalid")
    if np.any(categorical != 0):
        raise _fail("categorical splits are not supported")
    ids = np.arange(n, dtype=np.int64)
    internal = is_leaf == 0
    if np.any(internal & ((left <= ids) | (left >= n) | (right <= ids) | (right >= n))):
        raise _fail("hist-gradient-boosting child index is out of range or not after its parent")
    if not owner_n_features:
        raise _fail("hist-gradient-boosting predictor has no owning estimator")
    limit = min(owner_n_features)
    if np.any(internal & ((feature < 0) | (feature >= limit))):
        raise _fail("hist-gradient-boosting split feature is out of range")


def _owner_features(obj: object, current: tuple[int, ...]) -> tuple[int, ...]:
    """Extend the n_features_in_ context when ``obj`` is a tree-family estimator."""
    module = type(obj).__module__
    if not module.startswith("sklearn.") or not any(key in module for key in ("tree", "ensemble")):
        return current
    n = getattr(obj, "n_features_in_", None)
    if isinstance(n, int | np.integer) and not isinstance(n, bool):
        return (*current, int(n))
    return current


def _children(obj: object) -> list[object]:
    if isinstance(obj, list | tuple | set | frozenset):
        return list(obj)
    if isinstance(obj, dict):
        return list(obj.values())
    if isinstance(obj, np.ndarray):
        return list(obj.ravel()) if obj.dtype == object else []
    state = getattr(obj, "__dict__", None)
    if isinstance(state, dict):
        return list(state.values())
    return []


def validate_model_structure(model: object) -> None:
    """Walk the estimator object graph and validate every tree. Raises ``StructureError``."""
    from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
    from sklearn.tree._tree import Tree

    seen: set[tuple[int, tuple[int, ...]]] = set()
    stack: list[tuple[object, tuple[int, ...], int]] = [(model, (), 0)]
    while stack:
        obj, ctx, depth = stack.pop()
        if obj is None or isinstance(obj, str | bytes | int | float | bool | complex | type):
            continue
        if isinstance(obj, np.generic):
            continue
        if depth > MAX_DEPTH:
            raise _fail("model object graph is too deep")
        key = (id(obj), ctx)
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > MAX_OBJECTS:
            raise _fail("model object graph is too large")
        if isinstance(obj, Tree):
            validate_tree(obj, ctx)
            continue
        if isinstance(obj, TreePredictor):
            validate_tree_predictor(obj, ctx)
            continue
        ctx = _owner_features(obj, ctx)
        for child in _children(obj):
            stack.append((child, ctx, depth + 1))
