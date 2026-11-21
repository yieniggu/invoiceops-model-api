"""Process isolation for everything that touches student model bytes.

The HTTP process never imports or unpickles a student model. A deploy check runs in a one-shot
child (``_child``) with a hard timeout and a memory rlimit; serving runs in a supervised worker
(``inference``). Children get a minimal environment (no service token, seed or MLflow
credentials), their own session, no stdio, and one inherited socket carrying small JSON
messages; the parent bounds every message it reads.
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any

import numpy as np

from invoiceops_model_api.procutil import popen_tracked, untrack
from invoiceops_model_api.settings import Settings

MAX_REPLY_BYTES = 4 * 1024 * 1024
CHILD_ENTRYPOINT: tuple[str, ...] = (sys.executable, "-I", "-m", "invoiceops_model_api._child")
WORKER_ENTRYPOINT: tuple[str, ...] = (sys.executable, "-I", "-m", "invoiceops_model_api._worker")


SECRET_ENV_MARKERS = ("TOKEN", "SEED", "SECRET", "PASSWORD", "CREDENTIAL")


def child_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Minimal environment: bounded native threads, nothing secret.

    Children never receive the service token, the validation seed or MLflow/gateway tokens:
    not through the environment, not through argv. What they need arrives on the inherited
    socket (paths and the validation FEATURES, never the labels). ``extra`` is for tests and is
    refused when it looks like a secret.
    """
    env = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key in extra or {}:
        if any(marker in key.upper() for marker in SECRET_ENV_MARKERS):
            raise ValueError(f"refusing to pass {key} to a child process")
    env.update(extra or {})
    return env


@dataclass
class ChildProcess:
    process: subprocess.Popen
    conn: Connection
    _reaped: bool = False
    _code: int | None = None

    def kill(self) -> int | None:
        """SIGKILL the child's whole process group, reap it and close the channel (idempotent)."""
        if self._reaped:
            return self._code
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.process.kill()
        except ProcessLookupError:
            pass
        try:
            self._code = self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            self._code = None
        self._reaped = True
        untrack(self.process)
        self.conn.close()
        return self._code

    def alive(self) -> bool:
        return self.process.poll() is None


def spawn(
    entrypoint: Sequence[str], memory_mb: int, extra_env: Mapping[str, str] | None = None
) -> ChildProcess:
    parent_sock, child_sock = socket.socketpair()
    try:
        process = popen_tracked(
            [*entrypoint, str(child_sock.fileno()), str(memory_mb)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=child_env(extra_env),
            pass_fds=(child_sock.fileno(),),
            close_fds=True,
            start_new_session=True,
        )
    except BaseException:
        parent_sock.close()
        raise
    finally:
        child_sock.close()
    return ChildProcess(process, Connection(parent_sock.detach()))


def exit_detail(code: int | None) -> dict[str, Any]:
    if code is None:
        return {}
    if code < 0:
        try:
            return {"signal": signal.Signals(-code).name}
        except ValueError:
            return {"signal": str(-code)}
    return {"exitCode": code}


# What a check child may report. Everything it sends is student-influenced (a crafted model can
# run code in it), so only a closed set of codes with FIXED messages and a fixed detail schema
# reaches the API response: no free text, no unknown keys, bounded sizes. (The returned metrics
# are rounded for the same reason: the response must not be a channel for the validation set.)
CHILD_MESSAGES: dict[str, str] = {
    "SKOPS_ARCHIVE_INVALID": "The model archive is not a valid skops archive",
    "ARTIFACT_TOO_LARGE": "The model archive expands beyond the limit",
    "UNTRUSTED_TYPE": "The model contains types that are not allowed",
    "MODEL_LOAD_FAILED": "The model could not be loaded",
    "MODEL_STRUCTURE_INVALID": "The model contains an invalid tree structure",
    "NO_PREDICT_PROBA": "The model has no predict_proba",
    "INVALID_CLASSES": "The model must be a binary classifier (0/1)",
    "INFERENCE_FAILED": "predict_proba failed on the validation set",
    "MODEL_CHECK_CRASHED": "The model check process terminated abnormally",
}
_TYPE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,119}$")
MAX_TYPE_NAMES = 10


def sanitize_failure(reply: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """(code, fixed message, detail restricted to the expected keys) for a child failure."""
    code = reply.get("code")
    if not isinstance(code, str) or code not in CHILD_MESSAGES:
        code = "MODEL_CHECK_CRASHED"
    detail: dict[str, Any] = {}
    raw = reply.get("detail")
    if code == "UNTRUSTED_TYPE" and isinstance(raw, dict) and isinstance(raw.get("types"), list):
        types = [t for t in raw["types"] if isinstance(t, str) and _TYPE_NAME.fullmatch(t)]
        detail["types"] = types[:MAX_TYPE_NAMES]
    return code, CHILD_MESSAGES[code], detail


class CheckBusy(Exception):
    """No check slot became free within ``check_wait_seconds`` (the API answers 503 BUSY)."""


@dataclass
class ChildOutcome:
    ok: bool
    code: str = ""
    message: str = ""
    detail: dict[str, Any] | None = None
    positive_index: int = 1
    scores: np.ndarray | None = None


class CheckRunner:
    """Runs the load + validation-predict of one model in a one-shot, limited child process."""

    def __init__(
        self,
        settings: Settings,
        entrypoint: Sequence[str] = CHILD_ENTRYPOINT,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        self._settings = settings
        self._entrypoint = tuple(entrypoint)
        self._extra_env = dict(extra_env or {})
        # Bounds concurrent children (and so peak memory); excess checks wait their turn.
        self._slots = BoundedSemaphore(settings.max_concurrent_checks)

    def run(self, skops_path: Path, matrix: np.ndarray, columns: list[str]) -> ChildOutcome:
        settings = self._settings
        if not self._slots.acquire(timeout=settings.check_wait_seconds):
            raise CheckBusy
        try:
            return self._run_in_slot(skops_path, matrix, columns)
        finally:
            self._slots.release()

    def _run_in_slot(
        self, skops_path: Path, matrix: np.ndarray, columns: list[str]
    ) -> ChildOutcome:
        settings = self._settings
        with tempfile.TemporaryDirectory(prefix="invoiceops-frame-") as tmp:
            frame_path = Path(tmp) / "frame.npy"
            np.save(frame_path, matrix, allow_pickle=False)
            deadline = time.monotonic() + settings.check_timeout_seconds
            child = spawn(self._entrypoint, settings.check_memory_mb, self._extra_env)
            try:
                child.conn.send_bytes(
                    json.dumps(
                        {
                            "model_path": str(skops_path),
                            "frame_path": str(frame_path),
                            "columns": columns,
                            "max_uncompressed_bytes": settings.max_skops_uncompressed_bytes,
                        }
                    ).encode()
                )
                if not child.conn.poll(max(0.0, deadline - time.monotonic())):
                    child.kill()
                    return ChildOutcome(
                        False,
                        "MODEL_CHECK_TIMEOUT",
                        "The model check exceeded its time limit",
                        {"timeoutSeconds": settings.check_timeout_seconds},
                    )
                raw = child.conn.recv_bytes(MAX_REPLY_BYTES)
                reply = json.loads(raw)
            except (EOFError, OSError, ValueError):
                code = child.kill()
                return ChildOutcome(
                    False,
                    "MODEL_CHECK_CRASHED",
                    "The model check process terminated abnormally",
                    exit_detail(code),
                )
            finally:
                child.kill()
        if not isinstance(reply, dict):
            return ChildOutcome(False, "MODEL_CHECK_CRASHED", "Invalid reply from the check")
        if reply.get("ok") is True:
            try:
                scores = np.asarray(reply["scores"], dtype=float)
                pos = int(reply["positive_index"])
                if pos not in (0, 1):
                    raise ValueError("positive_index")
            except (KeyError, TypeError, ValueError):
                return ChildOutcome(False, "MODEL_CHECK_CRASHED", "Invalid reply from the check")
            return ChildOutcome(True, positive_index=pos, scores=scores)
        code, message, detail = sanitize_failure(reply)
        return ChildOutcome(False, code, message, detail)
