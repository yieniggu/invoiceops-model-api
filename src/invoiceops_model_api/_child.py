"""One-shot deploy-check child process: load + structural validation + validation predict.

Spawned by ``isolation.CheckRunner`` as ``python -I -m invoiceops_model_api._child <fd> <MB>``
with a minimal environment (no service token, no seed, no MLflow credentials). It applies its
memory rlimit before importing numpy/sklearn, reads one request from the inherited socket,
answers with JSON and exits. A crafted model that crashes or hangs it only kills/times out this
process; the parent maps that to ``MODEL_CHECK_CRASHED`` / ``MODEL_CHECK_TIMEOUT``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from invoiceops_model_api.procutil import apply_limits

MAX_REQUEST_BYTES = 64 * 1024


def serve(
    conn: Connection,
    *,
    validate_structure: bool = True,
    predict_fn: Callable[..., Any] | None = None,
) -> None:
    import numpy as np

    from invoiceops_model_api import model_runtime
    from invoiceops_model_api.model_runtime import ModelRejected

    predict = predict_fn or model_runtime.predict_positive
    request = json.loads(conn.recv_bytes(MAX_REQUEST_BYTES))
    try:
        model, pos = model_runtime.load_checked(
            Path(request["model_path"]),
            int(request["max_uncompressed_bytes"]),
            validate_structure=validate_structure,
        )
        matrix = np.load(request["frame_path"], allow_pickle=False)
        try:
            scores = predict(model, matrix, list(request["columns"]), pos)
        except Exception:  # noqa: BLE001
            raise ModelRejected(
                "INFERENCE_FAILED", "predict_proba failed on the validation set"
            ) from None
        reply: dict[str, Any] = {
            "ok": True,
            "positive_index": pos,
            "scores": [float(s) for s in np.asarray(scores, dtype=float)],
        }
    except ModelRejected as error:
        reply = {"ok": False, "code": error.code, "message": str(error), "detail": error.detail}
    except MemoryError:
        reply = {"ok": False, "code": "MODEL_CHECK_CRASHED", "message": "Out of memory"}
    conn.send_bytes(json.dumps(reply).encode())


def main(*, validate_structure: bool = True, predict_fn: Callable[..., Any] | None = None) -> None:
    fd, memory_mb = int(sys.argv[1]), int(sys.argv[2])
    apply_limits(memory_mb)
    conn = Connection(fd)
    try:
        serve(conn, validate_structure=validate_structure, predict_fn=predict_fn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
