"""Linux container test: the production flags, uvicorn as PID 1, secrets unreadable.

Runs the real image with the flags of ``compose.production.yml`` (read-only, cap_drop ALL,
no-new-privileges, pids/memory limits, tmpfs) and, if the compose file ever asks for ``init``,
with ``--init`` too, which makes this test fail. Set ``MODEL_API_TEST_IMAGE`` to the built image
(CI builds ``invoiceops-model-api:ci`` first); without Docker or the variable it is skipped.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from composeenv import compose_default

ROOT = Path(__file__).resolve().parents[1]
IMAGE = os.environ.get("MODEL_API_TEST_IMAGE", "")
pytestmark = pytest.mark.skipif(
    not IMAGE or shutil.which("docker") is None,
    reason="set MODEL_API_TEST_IMAGE to a built image and have docker available",
)

CANARY_ENV = {
    "MODEL_API_SERVICE_TOKEN": "pid1-test-service-token-" + "a" * 24,
    "OFFICIAL_VALIDATION_SEED": "9f2c5d71e0a84b36c7d1f0958e2a6b43",
    "GATEWAY_SERVICE_READ_TOKEN": "pid1-test-gateway-token-" + "b" * 24,
    "MLFLOW_TRACKING_URI": "http://127.0.0.1:9",
    "MODEL_API_MLFLOW_AUTH": "gateway",
    "HOME": "/tmp",
}


def production_flags() -> list[str]:
    service = yaml.safe_load((ROOT / "compose.production.yml").read_text())["services"]["model-api"]
    flags = [
        "--read-only",
        "--memory",
        service["mem_limit"],
        "--pids-limit",
        str(service["pids_limit"]),
    ]
    for capability in service["cap_drop"]:
        flags += ["--cap-drop", capability]
    for option in service["security_opt"]:
        flags += ["--security-opt", option]
    for tmpfs in service["tmpfs"]:
        flags += ["--tmpfs", tmpfs]
    if service.get("init"):
        flags.append("--init")
    for key in ("MODEL_CHECK_MEMORY_MB", "MODEL_WORKER_MEMORY_MB"):
        flags += ["-e", f"{key}={compose_default(service['environment'][key])}"]
    for key, value in CANARY_ENV.items():
        flags += ["-e", f"{key}={value}"]
    return flags


def docker(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, check=False
    )


def wait_healthy(name: str) -> None:
    deadline = time.monotonic() + 90
    probe = (
        "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
    )
    while docker("exec", name, "python", "-c", probe).returncode != 0:
        assert time.monotonic() < deadline, docker("logs", name).stdout
        time.sleep(1)


@pytest.fixture()
def container() -> Iterator[str]:
    name = f"invoiceops-pid1-test-{uuid.uuid4().hex[:12]}"
    started = docker("run", "-d", "--name", name, *production_flags(), IMAGE)
    assert started.returncode == 0, started.stderr
    try:
        wait_healthy(name)
        yield name
    finally:
        docker("rm", "-f", name)


READ_ENVIRON = textwrap.dedent(
    """
    import json, os
    out = {"uid": os.getuid(), "pid1": open("/proc/1/cmdline", "rb").read().split(b"\\0")[:3]}
    out["pid1"] = [part.decode() for part in out["pid1"]]
    try:
        data = open("/proc/1/environ", "rb").read()
        out["environ"] = "READ" if b"MODEL_API_SERVICE_TOKEN" in data else "EMPTY"
    except PermissionError:
        out["environ"] = "DENIED"
    print(json.dumps(out))
    """
)


def test_uvicorn_is_pid_1_and_a_process_of_the_same_user_cannot_read_its_environment(
    container: str,
) -> None:
    result = docker("exec", container, "python", "-c", READ_ENVIRON)
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout)
    assert seen["uid"] != 0
    assert any("uvicorn" in part for part in seen["pid1"]), seen["pid1"]  # not docker-init
    assert seen["environ"] == "DENIED"


# The app is created exactly as in production (create_app -> make_undumpable, reaper), as PID 1 of
# its own container, and a child is spawned with the production spawn() (new session, minimal
# env, no stdio). The child tries /proc/1/environ and /proc/<ppid>/environ.
PID1_SCRIPT = textwrap.dedent(
    """
    import json, os, sys, time
    from invoiceops_model_api import isolation
    from invoiceops_model_api.app import create_app

    assert os.getpid() == 1
    create_app()
    child_code = (
        "import json, os\\n"
        "result = {}\\n"
        "for label, pid in (('pid1', 1), ('ppid', os.getppid())):\\n"
        "    try:\\n"
        "        data = open(f'/proc/{pid}/environ', 'rb').read()\\n"
        "        result[label] = 'READ' if b'MODEL_API_SERVICE_TOKEN' in data else 'EMPTY'\\n"
        "    except PermissionError:\\n"
        "        result[label] = 'DENIED'\\n"
        "result['ppid_value'] = os.getppid()\\n"
        "open('/tmp/child-result.json', 'w').write(json.dumps(result))\\n"
    )
    child = isolation.spawn((sys.executable, "-I", "-c", child_code), 64)
    child.process.wait(timeout=30)
    print(open("/tmp/child-result.json").read())
    """
)


def test_a_child_of_the_service_cannot_read_pid_1_or_its_parent_environment() -> None:
    name = f"invoiceops-pid1-child-{uuid.uuid4().hex[:12]}"
    try:
        result = docker(
            "run",
            "--name",
            name,
            *production_flags(),
            "--entrypoint",
            "python",
            IMAGE,
            "-c",
            PID1_SCRIPT,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        seen = json.loads(result.stdout.strip().splitlines()[-1])
        assert seen["ppid_value"] == 1  # the parent of the child IS the service
        assert seen["pid1"] == "DENIED"
        assert seen["ppid"] == "DENIED"
    finally:
        docker("rm", "-f", name)


# Orphans of a killed child (a model may fork) are re-parented to PID 1: the service must reap them.
REAP_SCRIPT = textwrap.dedent(
    """
    import os, subprocess, sys, time
    from invoiceops_model_api import procutil

    assert os.getpid() == 1
    pid = os.fork()
    if pid == 0:
        if os.fork() == 0:
            os._exit(0)
        os._exit(0)
    os.waitpid(pid, 0)
    owned = procutil.popen_tracked([sys.executable, "-c", "raise SystemExit(3)"])
    deadline = time.monotonic() + 10
    while len(procutil._zombie_children()) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    before = len(procutil._zombie_children())
    assert procutil.start_zombie_reaper(interval=0.2)
    time.sleep(1.5)
    left = len(procutil._zombie_children())
    print(before, left, owned.wait())
    """
)


def test_pid_1_reaps_orphans_and_leaves_tracked_children_to_their_owner() -> None:
    name = f"invoiceops-pid1-reap-{uuid.uuid4().hex[:12]}"
    try:
        result = docker(
            "run",
            "--name",
            name,
            *production_flags(),
            "--entrypoint",
            "python",
            IMAGE,
            "-c",
            REAP_SCRIPT,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        before, left, status = result.stdout.split()[-3:]
        assert (int(before), int(left), int(status)) == (2, 1, 3)
    finally:
        docker("rm", "-f", name)
