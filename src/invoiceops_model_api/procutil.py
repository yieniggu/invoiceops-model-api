"""Helpers shared by the parent and its child processes (no model or sklearn imports)."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from typing import Any


def apply_limits(memory_mb: int) -> bool:
    """Best-effort hard limits for a child process. Returns True if RLIMIT_AS was applied.

    ``RLIMIT_AS`` is enforced on Linux (the container). macOS ignores or rejects it, so the
    limit is a no-op there (development only). Core dumps are disabled so a crash never writes
    model or validation data to disk.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return False
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass
    limit = memory_mb * 1024 * 1024
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError):
        print("rlimit: RLIMIT_AS not applied on this platform", file=sys.stderr)
        return False
    return True


PR_SET_DUMPABLE = 4


def make_undumpable() -> bool:
    """Linux only: ``prctl(PR_SET_DUMPABLE, 0)`` on THIS process. Returns True when applied.

    The HTTP process keeps the service token, the validation seed and the gateway token in its
    environment. A non-dumpable process cannot be read through ``/proc/<pid>/environ`` or
    ``/proc/<pid>/mem`` or attached to with ptrace by another process of the same user, which is
    what a compromised check child or inference worker would be. The children get none of those
    secrets in the first place (see ``isolation.child_env``); this closes the parent's side.
    Core dumps of the parent are disabled by the same flag.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) == 0
    except (OSError, AttributeError):  # pragma: no cover - exotic libc
        return False


# ---- zombie reaping when this process is PID 1 --------------------------------------------------
#
# The production container runs uvicorn directly as PID 1 (no ``init: true``: docker-init would be
# PID 1 with the FULL environment, readable through /proc/1/environ by any child of the same
# user). The kernel then re-parents the orphans of the killed check/worker children (a student
# model may fork) to this process, and nobody would ``wait`` for them. ``reap_orphans`` reaps
# exactly those zombies and never a child that ``spawn`` tracks (its owner still needs the exit
# status), so it is safe to run next to ``subprocess``.

logger = logging.getLogger("invoiceops_model_api")
_TRACKED: dict[int, subprocess.Popen] = {}
_TRACK_LOCK = threading.Lock()
MAX_REAP_PER_PASS = 256
REAP_INTERVAL_SECONDS = 2.0
_reaper_started = threading.Event()


def popen_tracked(*args: Any, **kwargs: Any) -> subprocess.Popen:
    """``subprocess.Popen`` whose pid the orphan reaper leaves alone (until it is waited for)."""
    with _TRACK_LOCK:  # registered before the reaper can look at the new child
        process = subprocess.Popen(*args, **kwargs)
        _TRACKED[process.pid] = process
    return process


def untrack(process: subprocess.Popen) -> None:
    with _TRACK_LOCK:
        _TRACKED.pop(process.pid, None)


def _zombie_children() -> list[int]:
    """Pids of this process' children in state Z (Linux, from /proc; bounded by pids_limit)."""
    me = os.getpid()
    found: list[int] = []
    try:
        entries = [e.name for e in os.scandir("/proc") if e.name.isdigit()]
    except OSError:
        return found
    for name in entries:
        try:
            with open(f"/proc/{name}/stat", encoding="utf-8", errors="replace") as handle:
                data = handle.read()
        except OSError:
            continue
        fields = data.rpartition(")")[2].split()  # "pid (comm) state ppid ..."; comm may hold ")"
        if len(fields) > 1 and fields[0] == "Z" and fields[1].isdigit() and int(fields[1]) == me:
            found.append(int(name))
    return found


def reap_orphans() -> int:
    """Wait for zombie children that nobody owns. Returns how many were reaped (bounded)."""
    if not sys.platform.startswith("linux"):
        return 0
    reaped = 0
    with _TRACK_LOCK:
        for pid in [p for p, proc in _TRACKED.items() if proc.returncode is not None]:
            _TRACKED.pop(pid, None)  # already waited for by its owner
        for pid in _zombie_children():
            if pid in _TRACKED:
                continue
            try:
                done, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                continue
            reaped += 1 if done else 0
            if reaped >= MAX_REAP_PER_PASS:
                break
    return reaped


def start_zombie_reaper(*, force: bool = False, interval: float = REAP_INTERVAL_SECONDS) -> bool:
    """Start the reaper thread when this process is PID 1 (or ``force``). Idempotent."""
    if (os.getpid() != 1 and not force) or _reaper_started.is_set():
        return False
    _reaper_started.set()

    def loop() -> None:
        stop = threading.Event()  # never set: the thread is a daemon
        while not stop.wait(interval):
            try:
                reap_orphans()
            except Exception:  # the reaper must never die
                logger.exception("zombie reaper pass failed")

    threading.Thread(target=loop, name="zombie-reaper", daemon=True).start()
    return True
