"""Supervised inference worker.

Design (documented choice): ONE worker process holds all active deployments. A pool would pay
the ~300 MB interpreter+sklearn cost per process for no benefit at this scale (single-row
predictions take microseconds), and one worker keeps memory low. Requests are serialised
through a lock with a bounded wait. The HTTP process never imports or calls a student model.

Failure handling (per request, never taking the HTTP process down):

* worker crash (SIGSEGV/SIGBUS/OOM/exit), hang past ``inference_timeout_seconds`` or a protocol
  error: the worker is killed; the request gets ``INFERENCE_UNAVAILABLE`` (503); the next
  request starts a fresh worker and RELOADS the deployments it needs from the cached artifact
  bytes (lazily, per deployment, so a bad model cannot take the others down on restart);
* activating a model never blocks the serving worker: while other deployments are loaded, the
  new model is loaded (together with the existing ones, which are cheap to reload from the cached
  bytes) into a CANDIDATE worker outside the serving lock, and the candidate replaces the serving
  worker under the lock in an instant (blue/green). Other models keep serving from the old worker
  until the swap; one activation at a time;
* the deployment in flight when the worker died is charged one crash; at
  ``quarantine_after_crashes`` (2) it is quarantined: unloaded from the registry, its cached
  bytes deleted, and reported by ``GET /deployments``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from invoiceops_model_api.isolation import (
    MAX_REPLY_BYTES,
    WORKER_ENTRYPOINT,
    ChildProcess,
    exit_detail,
    spawn,
)
from invoiceops_model_api.settings import Settings

logger = logging.getLogger("invoiceops_model_api")


class InferenceUnavailable(Exception):
    """The worker died, timed out or is busy; the request can be retried."""

    def __init__(self, message: str, *, quarantined: bool = False) -> None:
        super().__init__(message)
        self.quarantined = quarantined


class InferenceFailed(Exception):
    """The model raised or returned invalid probabilities (the worker is fine)."""


class _WorkerDied(Exception):
    def __init__(self, kind: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(kind)
        self.kind = kind
        self.detail = detail or {}


@dataclass
class _Slot:
    path: Path
    columns: list[str]
    crashes: int = 0


class InferenceSupervisor:
    def __init__(
        self,
        settings: Settings,
        *,
        entrypoint: Sequence[str] = WORKER_ENTRYPOINT,
        extra_env: Mapping[str, str] | None = None,
        cache_dir: Path | None = None,
        on_quarantine: Callable[[str, str], None] | None = None,
    ) -> None:
        self._settings = settings
        self._entrypoint = tuple(entrypoint)
        self._extra_env = dict(extra_env or {})
        self._cache_dir = cache_dir
        self._own_cache = False
        self._lock = Lock()  # one request at a time to the worker
        self._activation_lock = Lock()  # one candidate worker at a time (never held to serve)
        self._candidate: ChildProcess | None = None
        self._state_lock = Lock()  # guards the slot table
        self._worker: ChildProcess | None = None
        self._loaded: set[str] = set()  # slots loaded in the CURRENT worker
        self._stale: set[str] = set()  # slots to unload from the worker when next possible
        self._slots: dict[str, _Slot] = {}
        self.on_quarantine = on_quarantine
        self.restarts = 0

    # ---- artifact cache -----------------------------------------------------------------

    def stage(self, source: Path) -> tuple[str, Path]:
        """Copy verified artifact bytes into the private cache; returns (slot, cached path)."""
        with self._state_lock:
            if self._cache_dir is None:
                self._cache_dir = Path(tempfile.mkdtemp(prefix="invoiceops-cache-"))
                self._own_cache = True
            self._cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            slot = uuid.uuid4().hex
            target = self._cache_dir / f"{slot}.skops"
        try:
            shutil.copyfile(source, target)
            os.chmod(target, 0o600)
        except BaseException:
            target.unlink(missing_ok=True)  # never leave a partial copy behind (disk full)
            raise
        return slot, target

    # ---- worker lifecycle ----------------------------------------------------------------

    def _acquire(self, timeout: float) -> None:
        if not self._lock.acquire(timeout=timeout):
            raise InferenceUnavailable("The inference worker is busy")

    def _spawn(self) -> ChildProcess:
        """Spawn a worker; a failed spawn (EAGAIN, EMFILE, ENOMEM) is an unavailable worker."""
        try:
            return spawn(self._entrypoint, self._settings.inference_memory_mb, self._extra_env)
        except OSError as exc:
            logger.error("inference_worker_spawn_failed errno=%s", exc.errno)
            raise InferenceUnavailable("The inference worker could not be started") from None

    def _start(self) -> ChildProcess:
        worker = self._spawn()
        self._worker = worker
        self._loaded = set()
        self.restarts += 1
        return worker

    def _kill_worker(self) -> int | None:
        worker, self._worker = self._worker, None
        self._loaded = set()
        return worker.kill() if worker else None

    @staticmethod
    def _exchange(worker: ChildProcess, request: dict[str, Any], timeout: float) -> dict[str, Any]:
        """One request/reply on ``worker``'s channel; the caller decides what to kill."""
        try:
            worker.conn.send_bytes(json.dumps(request).encode())
            if not worker.conn.poll(timeout):
                raise _WorkerDied("timeout", {"timeoutSeconds": timeout})
            reply = json.loads(worker.conn.recv_bytes(MAX_REPLY_BYTES))
        except (EOFError, OSError, ValueError):
            raise _WorkerDied("crashed") from None
        if not isinstance(reply, dict):
            raise _WorkerDied("crashed")
        return reply

    def _roundtrip(self, request: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Send one request to the serving worker and wait for its reply. Caller holds ``_lock``."""
        worker = self._worker or self._start()
        try:
            return self._exchange(worker, request, timeout)
        except _WorkerDied as died:
            code = self._kill_worker()
            if died.kind == "crashed":
                died.detail = exit_detail(code)
            raise

    def _flush_stale(self) -> None:
        with self._state_lock:
            stale, self._stale = self._stale, set()
        for slot in stale:
            if self._worker is not None and slot in self._loaded:
                try:
                    self._roundtrip({"op": "unload", "slot": slot}, 5.0)
                except _WorkerDied:
                    pass
                self._loaded.discard(slot)

    def _queue_unload(self, slot: str) -> None:
        with self._state_lock:
            self._stale.add(slot)
        if self._lock.acquire(timeout=1.0):
            try:
                self._flush_stale()
            finally:
                self._lock.release()

    def _ensure_loaded(self, slot: str) -> None:
        if slot in self._loaded and self._worker is not None:
            return
        with self._state_lock:
            info = self._slots.get(slot)
        if info is None:
            raise InferenceUnavailable("The deployment is not available")
        reply = self._roundtrip(
            {
                "op": "load",
                "slot": slot,
                "path": str(info.path),
                "columns": info.columns,
                "max_uncompressed_bytes": self._settings.max_skops_uncompressed_bytes,
            },
            self._settings.check_timeout_seconds,
        )
        if not reply.get("ok"):
            raise InferenceFailed(str(reply.get("code", "MODEL_LOAD_FAILED")))
        self._loaded.add(slot)

    # ---- public API ----------------------------------------------------------------------

    def activate(self, slot: str, path: Path, columns: Sequence[str]) -> None:
        """Register a staged artifact and load it into the worker (before it is served)."""
        with self._state_lock:
            self._slots[slot] = _Slot(path, list(columns))
        try:
            with self._activation_lock:
                if not self._activate_in_place(slot):
                    self._activate_in_candidate(slot)
        except (InferenceFailed, InferenceUnavailable):
            self._forget(slot)
            raise
        except OSError:
            self._forget(slot)
            raise InferenceUnavailable("The model could not be activated") from None

    def _activate_in_place(self, slot: str) -> bool:
        """Load straight into the serving worker when it serves nothing yet (nobody to starve)."""
        self._acquire(self._settings.check_timeout_seconds)
        try:
            self._flush_stale()
            if self._loaded:
                return False
            try:
                self._ensure_loaded(slot)
            except _WorkerDied as died:
                raise InferenceUnavailable(f"worker {died.kind} while loading") from None
            return True
        finally:
            self._lock.release()

    def _activate_in_candidate(self, slot: str) -> None:
        timeout = self._settings.check_timeout_seconds
        candidate = self._spawn()
        self._candidate = candidate
        swapped = False
        try:
            with self._state_lock:
                slots = dict(self._slots)
            loaded: set[str] = set()
            for name in [slot, *(n for n in slots if n != slot)]:
                info = slots[name]
                try:
                    reply = self._exchange(
                        candidate,
                        {
                            "op": "load",
                            "slot": name,
                            "path": str(info.path),
                            "columns": info.columns,
                            "max_uncompressed_bytes": self._settings.max_skops_uncompressed_bytes,
                        },
                        timeout,
                    )
                except _WorkerDied as died:
                    if name == slot:
                        raise InferenceUnavailable(f"worker {died.kind} while loading") from None
                    break  # the candidate is gone: the old worker keeps serving the others
                if reply.get("ok"):
                    loaded.add(name)
                elif name == slot:
                    raise InferenceFailed(str(reply.get("code", "MODEL_LOAD_FAILED")))
                # another deployment that fails to reload here is simply loaded lazily later
            self._acquire(timeout)  # waits for the request in flight, then swaps in an instant
            try:
                old, self._worker = self._worker, candidate
                self._loaded = loaded
                self.restarts += 1
                swapped = True
            finally:
                self._lock.release()
            if old is not None:
                old.kill()
            with self._state_lock:  # unloaded while the candidate was loading
                self._stale.update(n for n in loaded if n not in self._slots)
        finally:
            self._candidate = None
            if not swapped:
                candidate.kill()

    def _forget(self, slot: str) -> None:
        with self._state_lock:
            self._slots.pop(slot, None)

    def predict(self, slot: str, rows: list[list[float]]) -> list[float]:
        self._acquire(self._settings.inference_timeout_seconds)
        try:
            self._flush_stale()
            try:
                self._ensure_loaded(slot)
                reply = self._roundtrip(
                    {"op": "predict", "slot": slot, "rows": rows},
                    self._settings.inference_timeout_seconds,
                )
            except _WorkerDied as died:
                quarantined = self._charge_crash(slot, died)
                raise InferenceUnavailable(
                    f"inference worker {died.kind}", quarantined=quarantined
                ) from None
            except OSError:
                raise InferenceUnavailable("The inference worker is unavailable") from None
        finally:
            self._lock.release()
        if not reply.get("ok"):
            raise InferenceFailed(str(reply.get("code", "INFERENCE_FAILED")))
        try:
            return [float(v) for v in reply["values"]]
        except (KeyError, TypeError, ValueError):
            raise InferenceFailed("INFERENCE_FAILED") from None

    def _charge_crash(self, slot: str, died: _WorkerDied) -> bool:
        with self._state_lock:
            info = self._slots.get(slot)
            if info is None:
                return False
            info.crashes += 1
            quarantine = info.crashes >= self._settings.quarantine_after_crashes
            if quarantine:
                self._slots.pop(slot, None)
        logger.error("inference_worker_%s slot=%s crashes=%s", died.kind, slot[:8], info.crashes)
        if quarantine:
            info.path.unlink(missing_ok=True)
            if self.on_quarantine is not None:
                self.on_quarantine(slot, f"inference worker {died.kind} {info.crashes} times")
        return quarantine

    def unload(self, slot: str, path: Path | None = None) -> None:
        with self._state_lock:
            info = self._slots.pop(slot, None)
        target = path or (info.path if info else None)
        if target is not None:
            target.unlink(missing_ok=True)
        self._queue_unload(slot)

    def shutdown(self) -> None:
        candidate = self._candidate
        if candidate is not None:
            candidate.kill()
        with self._lock:
            self._kill_worker()
        with self._state_lock:
            self._slots.clear()
            if self._own_cache and self._cache_dir is not None:
                shutil.rmtree(self._cache_dir, ignore_errors=True)
                self._cache_dir = None
                self._own_cache = False

    def worker_pid(self) -> int | None:
        worker = self._worker
        return worker.process.pid if worker else None

    def warm(self) -> None:
        """Start the worker now (optional; otherwise it starts on first use)."""
        self._acquire(self._settings.check_timeout_seconds)
        try:
            self._worker or self._start()
        finally:
            self._lock.release()
