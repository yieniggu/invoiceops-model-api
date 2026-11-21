"""Supervised inference worker: the only process that holds loaded student models.

One long-lived worker holds ALL active deployments (models are small; a pool would multiply the
~300 MB interpreter+sklearn cost per process). It reads JSON requests from the inherited socket
one at a time:

* ``load``    {slot, path, columns, max_uncompressed_bytes}  (allowlist + structure validation)
* ``predict`` {slot, rows}                                    (rows in FEATURES order)
* ``unload``  {slot}

It never trusts the parent's cache blindly either: ``load`` re-runs the same validation as the
deploy check. If it crashes, hangs or runs out of memory, ``inference.InferenceSupervisor``
kills and restarts it and reloads deployments from the cached artifact bytes on demand.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from invoiceops_model_api.procutil import apply_limits

MAX_REQUEST_BYTES = 8 * 1024 * 1024


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
    models: dict[str, tuple[object, list[str], int]] = {}
    while True:
        try:
            request = json.loads(conn.recv_bytes(MAX_REQUEST_BYTES))
        except (EOFError, OSError):
            return  # parent gone or closed: exit
        op = request.get("op")
        reply: dict[str, Any]
        try:
            if op == "load":
                model, pos = model_runtime.load_checked(
                    Path(request["path"]),
                    int(request["max_uncompressed_bytes"]),
                    validate_structure=validate_structure,
                )
                models[request["slot"]] = (model, list(request["columns"]), pos)
                reply = {"ok": True}
            elif op == "predict":
                model, columns, pos = models[request["slot"]]
                matrix = np.asarray(request["rows"], dtype=float)
                values = predict(model, matrix, columns, pos)
                reply = {"ok": True, "values": [float(v) for v in values]}
            elif op == "unload":
                models.pop(request["slot"], None)
                reply = {"ok": True}
            else:
                reply = {"ok": False, "code": "BAD_REQUEST"}
        except ModelRejected as error:
            reply = {"ok": False, "code": error.code, "message": str(error)}
        except MemoryError:
            reply = {"ok": False, "code": "OUT_OF_MEMORY"}
        except Exception:  # noqa: BLE001 - never include feature values in the error
            reply = {"ok": False, "code": "INFERENCE_FAILED"}
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
