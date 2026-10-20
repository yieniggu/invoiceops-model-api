"""Picklable, offline estimator fixtures for real spawn-process tests."""

import os
import struct
import time

from invoiceops_model_api.app import LoadedModel


class Estimator:
    classes_ = (0, 1)

    def predict_proba(self, rows):
        return [[0.2, 0.8] for _ in rows]

    def __reduce__(self):
        raise AssertionError("Estimator must never leave its owner")


class SlowEstimator(Estimator):
    def predict_proba(self, rows):
        time.sleep(float(rows[0][0]))
        probability = float(rows[0][1])
        return [[1 - probability, probability]]


class ExitingEstimator(Estimator):
    def predict_proba(self, rows):
        os._exit(7)


class SelectiveEstimator(Estimator):
    def predict_proba(self, rows):
        if rows[0][7] == "unknown":
            raise ValueError("private-provider-password")
        return super().predict_proba(rows)


def load_estimator(*_):
    return LoadedModel(Estimator(), "7", "run-123")


def load_new_estimator(*_):
    return LoadedModel(Estimator(), "8", "run-456")


def load_slow_estimator(*_):
    return LoadedModel(SlowEstimator(), "7", "run-123")


def load_exiting_estimator(*_):
    return LoadedModel(ExitingEstimator(), "7", "run-123")


def load_selective_estimator(*_):
    return LoadedModel(SelectiveEstimator(), "7", "run-123")


def load_blocked(*_):
    time.sleep(30)
    return load_estimator()


def load_partial_then_block(entry, uri):
    if entry.registered_model == "blocked":
        return load_blocked()
    return load_estimator(entry, uri)


def malformed_inference_worker(incoming, outgoing, entry, _uri, _loader, deadline):
    from invoiceops_model_api.worker import receive_frame, send_frame

    os.set_blocking(incoming.fileno(), False)
    os.set_blocking(outgoing.fileno(), False)
    send_frame(outgoing.fileno(), ["ready", "7", "run-123", [0, 1]], deadline)
    receive_frame(incoming.fileno(), time.monotonic() + 10)
    if entry.alias == "partial-result":
        os.write(outgoing.fileno(), struct.pack("!I", 100) + b"[")
    elif entry.alias == "wrong-result":
        send_frame(outgoing.fileno(), ["result", [[1.5, -0.5]]], time.monotonic() + 1)
    elif entry.alias == "corrupt-result":
        os.write(outgoing.fileno(), struct.pack("!I", 3) + b"???")
    elif entry.alias == "late-result":
        time.sleep(1)
        send_frame(outgoing.fileno(), ["result", [[0.2, 0.8]]], time.monotonic() + 1)
    time.sleep(30)


def malformed_worker(incoming, outgoing, entry, _uri, _loader, _deadline):
    """An actual child that stalls or corrupts the bootstrap protocol."""
    fd = outgoing.fileno()
    if entry.alias == "partial-header":
        os.write(fd, b"\x00\x00")
    elif entry.alias == "partial-body":
        os.write(fd, struct.pack("!I", 100) + b"[")
    elif entry.alias == "oversize":
        os.write(fd, struct.pack("!I", 16385))
    elif entry.alias == "corrupt":
        os.write(fd, struct.pack("!I", 3) + b"???")
    elif entry.alias == "deep":
        payload = b"[" * 100 + b"]" * 100
        os.write(fd, struct.pack("!I", len(payload)) + payload)
    elif entry.alias == "late":
        time.sleep(1.2)
        payload = b'["ready","7","run-123",[0,1]]'
        os.write(fd, struct.pack("!I", len(payload)) + payload)
    time.sleep(30)
    incoming.close()
    outgoing.close()


def load_deserialization_hang(*_):
    # The trusted loader deserializes an artifact only inside the owned worker.
    import pickle

    class Artifact:
        def __reduce__(self):
            return time.sleep, (30,)

    return pickle.loads(pickle.dumps(Artifact()))


def load_pipeline(*_):
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

    rows = pd.DataFrame(
        [
            [
                125000 + i * 1000,
                365 + i,
                i % 2,
                1.1,
                True,
                True,
                False,
                ("low", "medium", "high")[i % 3],
            ]
            for i in range(64)
        ]
    )
    pipeline = Pipeline(
        [
            (
                "encode",
                ColumnTransformer(
                    [
                        ("numeric", "passthrough", [0, 1, 2, 3]),
                        (
                            "flags",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            [4, 5, 6],
                        ),
                        ("risk", OrdinalEncoder(), [7]),
                    ]
                ),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(min_samples_leaf=2, max_iter=10, random_state=0),
            ),
        ]
    )
    pipeline.fit(rows, [i % 2 for i in range(64)])
    return LoadedModel(pipeline, "7", "run-123")
