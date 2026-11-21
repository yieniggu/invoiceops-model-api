"""Orphan reaping when the service is PID 1. Linux only: it needs /proc and a subreaper."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="/proc and PR_SET_CHILD_SUBREAPER are Linux"
)

# A subreaper stands in for PID 1: orphans of its descendants are re-parented to it.
SCRIPT = textwrap.dedent(
    """
    import ctypes, os, subprocess, sys, time
    from invoiceops_model_api import procutil

    assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER

    def zombies():
        return len(procutil._zombie_children())

    # An orphaned grandchild: the middle process exits, the grandchild is re-parented to us.
    pid = os.fork()
    if pid == 0:
        if os.fork() == 0:
            os._exit(0)
        os._exit(0)
    os.waitpid(pid, 0)
    # A tracked child that exits with a status its owner still needs.
    owned = procutil.popen_tracked([sys.executable, "-c", "raise SystemExit(3)"])
    deadline = time.monotonic() + 10
    while zombies() < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    before = zombies()
    reaped = procutil.reap_orphans()
    after = zombies()
    print(before, reaped, after, owned.wait(), flush=True)
    """
)


def test_orphans_are_reaped_but_tracked_children_keep_their_exit_status() -> None:
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    before, reaped, after, owned_status = result.stdout.split()
    assert int(before) == 2  # the orphan and the tracked child, both zombies
    assert int(reaped) == 1  # only the orphan
    assert int(after) == 1  # the tracked one waits for its owner
    assert int(owned_status) == 3  # ...which still gets the real exit status


def test_the_reaper_only_starts_as_pid_1_unless_forced() -> None:
    from invoiceops_model_api import procutil

    assert procutil.start_zombie_reaper() is False  # pytest is not PID 1
