"""Owned estimator process and bounded, non-pickle inference transport.

Connection objects only transport trusted spawn handles. Application messages use
nonblocking anonymous pipe descriptors, never Connection.recv/send or pickle.
"""

import json
import logging
import math
import os
import select
import struct
from threading import Event, Lock, Timer
from time import monotonic

MAX_FRAME_BYTES = 16384
INFERENCE_TIMEOUT_SECONDS = 10.0
RETIRE_TIMEOUT_SECONDS = 10.0
PROCESS_CLEANUP_TIMEOUT_SECONDS = 0.1
logger = logging.getLogger("invoiceops_model_api.refresh")


class ModelLoadFailure(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ModelPredictionFailure(ModelLoadFailure):
    """A completed estimator error, not transport corruption or an expired RPC."""


def check_deadline(deadline, cancelled=None):
    if cancelled is not None and cancelled.is_set():
        raise ModelLoadFailure("cancelled")
    if monotonic() >= deadline:
        raise ModelLoadFailure("timeout")


def _transfer(fd, value, deadline, cancelled=None):
    """Read exactly N bytes or write bytes; every syscall is nonblocking."""
    reading = isinstance(value, int)
    size = value if reading else len(value)
    result = bytearray()
    offset = 0
    while offset < size:
        check_deadline(deadline, cancelled)
        try:
            if reading:
                chunk = os.read(fd, size - offset)
                if not chunk:
                    raise ModelLoadFailure("child_exit")
                result.extend(chunk)
                offset += len(chunk)
            else:
                offset += os.write(fd, value[offset:])
        except BlockingIOError:
            remaining = min(0.01, max(0, deadline - monotonic()))
            select.select([fd] if reading else [], [] if reading else [fd], [], remaining)
    check_deadline(deadline, cancelled)
    return bytes(result)


def _validate(value, depth=0, budget=None):
    # Limit nesting and scalar sizes before/after JSON; no object hooks or arbitrary types.
    if budget is None:
        budget = [MAX_FRAME_BYTES]
    budget[0] -= 6
    if budget[0] < 0:
        raise ModelLoadFailure("protocol_failure")
    if depth > 3:
        raise ModelLoadFailure("protocol_failure")
    if value is None or type(value) is bool:
        return
    if type(value) in (int, float):
        if (
            (type(value) is int and value.bit_length() > 333)
            or not math.isfinite(value)
            or abs(value) > 1e100
        ):
            raise ModelLoadFailure("protocol_failure")
        budget[0] -= 104
        if budget[0] < 0:
            raise ModelLoadFailure("protocol_failure")
        return
    if type(value) is str and len(value) <= 2048:
        budget[0] -= len(value) * 6  # Worst-case JSON escaping, before allocating encoded bytes.
        if budget[0] < 0:
            raise ModelLoadFailure("protocol_failure")
        return
    if type(value) is list and len(value) <= 64:
        for item in value:
            _validate(item, depth + 1, budget)
        return
    raise ModelLoadFailure("protocol_failure")


def _encode_frame(value, deadline, cancelled=None):
    check_deadline(deadline, cancelled)
    _validate(value)
    payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_FRAME_BYTES:
        raise ModelLoadFailure("protocol_failure")
    check_deadline(deadline, cancelled)
    return struct.pack("!I", len(payload)) + payload


def send_frame(fd, value, deadline, cancelled=None):
    _transfer(fd, _encode_frame(value, deadline, cancelled), deadline, cancelled)


def receive_frame(fd, deadline, cancelled=None):
    length = struct.unpack("!I", _transfer(fd, 4, deadline, cancelled))[0]
    if not 0 < length <= MAX_FRAME_BYTES:
        raise ModelLoadFailure("protocol_failure")
    raw = _transfer(fd, length, deadline, cancelled)
    # A byte-level nesting check bounds JSON parser recursion before decoding.
    depth, quoted, escaped = 0, False, False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > 4:
                raise ModelLoadFailure("protocol_failure")
        elif byte in (93, 125):
            depth -= 1
    try:
        value = json.loads(raw)
        _validate(value)
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        raise ModelLoadFailure("protocol_failure") from None
    check_deadline(deadline, cancelled)
    return value


def _scalar(value):
    # numpy classes/probabilities are converted only inside the estimator owner.
    return value.item() if hasattr(value, "item") else value


def serve_model(incoming, outgoing, entry, uri, loader, startup_deadline):
    reader, writer = incoming.fileno(), outgoing.fileno()
    os.set_blocking(reader, False)
    os.set_blocking(writer, False)
    try:
        loaded = loader(entry, uri)
        classes = [_scalar(value) for value in loaded.model.classes_]
        if not 1 <= len(classes) <= 64 or 1 not in classes:
            raise ValueError("Invalid classes")
        send_frame(
            writer, ["ready", loaded.model_version, loaded.run_id, classes], startup_deadline
        )
        while True:
            # An unused active model has no expiry. EOF/owned termination ends idle.
            if not select.select([reader], [], [], 1)[0]:
                continue
            message = receive_frame(reader, monotonic() + INFERENCE_TIMEOUT_SECONDS)
            if not isinstance(message, list) or len(message) != 3 or message[0] != "predict":
                raise ValueError("Invalid request")
            _, deadline, rows = message
            if type(deadline) not in (float, int):
                raise ValueError("Invalid deadline")
            check_deadline(deadline)
            if not isinstance(rows, list) or len(rows) != 1 or len(rows[0]) != 8:
                raise ValueError("Invalid features")
            try:
                result = [
                    [float(value) for value in row] for row in loaded.model.predict_proba(rows)
                ]
                if len(result) != 1 or len(result[0]) != len(classes):
                    raise ValueError("Invalid probabilities")
                if any(not math.isfinite(p) or not 0 <= p <= 1 for p in result[0]):
                    raise ValueError("Invalid probabilities")
            except Exception:  # noqa: BLE001
                # A completed model validation error does not poison a healthy owner.
                send_frame(writer, ["error"], deadline)
                continue
            send_frame(writer, ["result", result], deadline)
    except (KeyboardInterrupt, SystemExit):
        # A provider requesting process exit must not print its private exit message.
        return
    except Exception:  # noqa: BLE001
        # Provider errors never cross the process boundary, including traceback text.
        try:
            send_frame(writer, ["failure"], min(startup_deadline, monotonic() + 0.05))
        except Exception:  # noqa: BLE001
            # Broken/expired transport: closing the pipe is the failure signal.
            return
    finally:
        incoming.close()
        outgoing.close()


def stop_process(process):
    try:
        if process.is_alive():
            process.terminate()
            process.join(PROCESS_CLEANUP_TIMEOUT_SECONDS)
        if process.is_alive():
            process.kill()
            process.join(PROCESS_CLEANUP_TIMEOUT_SECONDS)
        if process.is_alive():
            raise ModelLoadFailure("cleanup_unverified")
        process.join(0)
        process.close()
    except (OSError, ValueError):
        raise ModelLoadFailure("cleanup_unverified") from None


class WorkerPool:
    """Reserve ownership before allocation; only confirmed cleanup frees capacity."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.workers = []
        self._lock = Lock()
        self._closing = False

    def reserve(self, stop):
        with self._lock:
            self.workers[:] = [owner for owner in self.workers if not owner.cleaned]
            if self._closing:
                raise ModelLoadFailure("cancelled")
            if len(self.workers) >= self.capacity:
                raise ModelLoadFailure("worker_capacity")
            owner = ModelProxy(stop)
            self.workers.append(owner)
            return owner

    def begin_shutdown(self):
        with self._lock:
            self._closing = True

    def close(self):
        with self._lock:
            self._closing = True
            owners = list(self.workers)
        # Never hold the pool lock across a process stop or owner callback.
        failed = False
        for owner in owners:
            try:
                owner.close()
            except ModelLoadFailure:
                failed = True
        with self._lock:
            self.workers[:] = [owner for owner in self.workers if not owner.cleaned]
            failed = failed or bool(self.workers)
        if failed:
            raise ModelLoadFailure("cleanup_unverified")


class _Cancellation:
    def __init__(self, *events):
        self.events = events

    def is_set(self):
        return any(event is not None and event.is_set() for event in self.events)


class ModelProxy:
    """One correlated RPC at a time; leases bind snapshots across registry swaps."""

    def __init__(self, stop):
        self.process = None
        self.incoming = None
        self.outgoing = None
        self._connections = []
        self._stop = stop
        self._rpc = Lock()
        self._state = Lock()
        self._cleanup = Lock()
        self._closed = Event()
        self._leases = 0
        self._retired = False
        self._timer = None
        self.cleaned = False
        self._process_stopped = False
        self.classes_ = []

    def launch(self, context, target, args):
        # Shutdown may close a reservation before launch. Allocation and spawn
        # cannot race its cleanup, and every acquired handle is already owned.
        with self._cleanup:
            if self._closed.is_set():
                raise ModelLoadFailure("cancelled")
            self.incoming, child_output = context.Pipe(duplex=False)
            self._connections.extend((self.incoming, child_output))
            child_input, self.outgoing = context.Pipe(duplex=False)
            self._connections.extend((child_input, self.outgoing))
            os.set_blocking(self.incoming.fileno(), False)
            os.set_blocking(self.outgoing.fileno(), False)
            self.process = context.Process(target=target, args=(child_input, child_output, *args))
            if self._closed.is_set():
                raise ModelLoadFailure("cancelled")
            try:
                self.process.start()
            except Exception:  # noqa: BLE001
                raise ModelLoadFailure("process_start_failure") from None
            child_input.close()
            child_output.close()

    def bootstrap(self, deadline, cancelled):
        cancellation = _Cancellation(cancelled, self._closed)
        while True:
            check_deadline(deadline, cancellation)
            if self._rpc.acquire(timeout=min(0.01, max(0, deadline - monotonic()))):
                break
        try:
            check_deadline(deadline, cancellation)
            return receive_frame(self.incoming.fileno(), deadline, cancellation)
        finally:
            self._rpc.release()

    @property
    def available(self):
        with self._cleanup:
            return (
                not self._closed.is_set() and self.process is not None and self.process.is_alive()
            )

    def acquire(self):
        with self._state:
            if self._closed.is_set() or self._retired:
                raise ModelLoadFailure("child_exit")
            self._leases += 1

    def release(self):
        with self._state:
            self._leases -= 1
            close = self._retired and self._leases == 0
        if close:
            self.close()

    def retire(self):
        with self._state:
            self._retired = True
            close = self._closed.is_set() or self._leases == 0
            if not close:
                self._timer = Timer(RETIRE_TIMEOUT_SECONDS, self._close_retired)
                self._timer.daemon = True
                self._timer.start()
        if close:
            self._close_retired()

    def _close_retired(self):
        try:
            self.close()
        except ModelLoadFailure:
            # Keep this unverified owner in the application's bounded owner list;
            # shutdown retries it. Publication has already committed, not rolled back.
            logger.error("model_cleanup_failed", extra={"reason": "cleanup_unverified"})

    def close(self):
        # Never acquire the RPC mutex to terminate a hung estimator.
        deadline = monotonic() + 0.3
        self._closed.set()
        if not self._cleanup.acquire(timeout=0.3):
            raise ModelLoadFailure("cleanup_unverified")
        try:
            if self.cleaned:
                return
            if self._timer is not None:
                self._timer.cancel()
            if not self._process_stopped:
                if self.process is not None:
                    if self.process.pid is None:
                        self.process.close()
                    else:
                        # Leave enough budget for the two bounded stop joins.
                        if deadline - monotonic() < 2 * PROCESS_CLEANUP_TIMEOUT_SECONDS:
                            raise ModelLoadFailure("cleanup_unverified")
                        self._stop(self.process)
                self._process_stopped = True
            if not self._rpc.acquire(timeout=max(0, deadline - monotonic())):
                raise ModelLoadFailure("cleanup_unverified")
            try:
                for connection in self._connections:
                    connection.close()
                self.cleaned = True
            finally:
                self._rpc.release()
        except (OSError, ValueError):
            raise ModelLoadFailure("cleanup_unverified") from None
        finally:
            self._cleanup.release()

    def predict_proba(self, rows):
        deadline = monotonic() + INFERENCE_TIMEOUT_SECONDS
        # Reject local transport-limit violations before touching/poisoning the
        # healthy worker. No HTTP schema or estimator object crosses this boundary.
        if (
            type(rows) is not list
            or len(rows) != 1
            or type(rows[0]) is not list
            or len(rows[0]) != 8
        ):
            raise ModelLoadFailure("inference_failure")
        packet = _encode_frame(["predict", deadline, rows], deadline, self._closed)
        acquired = False
        try:
            while not acquired:
                check_deadline(deadline, self._closed)
                acquired = self._rpc.acquire(timeout=min(0.01, max(0, deadline - monotonic())))
            _transfer(self.outgoing.fileno(), packet, deadline, self._closed)
            result = receive_frame(self.incoming.fileno(), deadline, self._closed)
            if result == ["error"]:
                raise ModelPredictionFailure("inference_failure")
            if not isinstance(result, list) or len(result) != 2 or result[0] != "result":
                raise ModelLoadFailure("inference_failure")
            values = result[1]
            if (
                not isinstance(values, list)
                or len(values) != 1
                or not isinstance(values[0], list)
                or len(values[0]) != len(self.classes_)
                or any(type(p) not in (float, int) or not 0 <= p <= 1 for p in values[0])
            ):
                raise ModelLoadFailure("protocol_failure")
            check_deadline(deadline, self._closed)
            return values
        except ModelPredictionFailure:
            raise
        except Exception:  # noqa: BLE001
            if acquired:
                self._closed.set()
            raise ModelLoadFailure("inference_failure") from None
        finally:
            if acquired:
                self._rpc.release()
            if self._closed.is_set():
                self.close()
