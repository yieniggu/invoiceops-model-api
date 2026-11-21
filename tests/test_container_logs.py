"""Container test: the REAL image logs audit lines on stdout.

Runs the image built from the Dockerfile with its real CMD (no create_app by hand), performs a
DELETE and a PUT and reads the lines back with ``docker logs``. Set ``MODEL_API_TEST_IMAGE``;
without Docker or the variable it is skipped.
"""

from __future__ import annotations

import re
import textwrap
import uuid
from collections.abc import Iterator

import pytest
from test_container_pid1 import CANARY_ENV, IMAGE, docker, production_flags, wait_healthy

pytestmark = pytest.mark.skipif(not IMAGE, reason="set MODEL_API_TEST_IMAGE to a built image")

REQUEST = textwrap.dedent(
    """
    import json, sys, urllib.error, urllib.request

    method, path, token = sys.argv[1:4]
    body = json.dumps({"workspace": "w", "registeredModel": "m", "version": "1"}).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:8000" + path,
        data=body if method == "PUT" else None,
        method=method,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    try:
        print(urllib.request.urlopen(request, timeout=30).status)
    except urllib.error.HTTPError as error:
        print(error.code)
    """
)


@pytest.fixture()
def running() -> Iterator[str]:
    name = f"invoiceops-logs-test-{uuid.uuid4().hex[:12]}"
    started = docker("run", "-d", "--name", name, *production_flags(), IMAGE)
    assert started.returncode == 0, started.stderr
    try:
        wait_healthy(name)
        yield name
    finally:
        docker("rm", "-f", name)


def call(name: str, method: str, path: str, token: str) -> int:
    result = docker("exec", name, "python", "-c", REQUEST, method, path, token)
    assert result.returncode == 0, result.stderr
    return int(result.stdout.strip())


def test_the_real_image_writes_audit_and_deployment_lines_to_stdout(running: str) -> None:
    token = CANARY_ENV["MODEL_API_SERVICE_TOKEN"]
    assert call(running, "DELETE", "/deployments/logs-test", token) == 200
    assert call(running, "PUT", "/deployments/logs-test", token) in (
        422,
        502,
        503,
    )  # no MLflow here
    assert call(running, "DELETE", "/deployments/logs-test", "x" * 40) == 401
    logs = docker("logs", running)
    text = logs.stdout  # stdout only: the lines must not depend on stderr's last-resort handler
    assert re.search(
        r'^ts=\S+ level=INFO logger=invoiceops_model_api msg="audit principal=[0-9a-f]{16} '
        r'method=DELETE route=/deployments/\{deployment_id\} status=200"$',
        text,
        re.MULTILINE,
    ), logs.stdout + logs.stderr
    assert 'msg="deployment_unloaded id=logs-test removed=False"' in text
    assert "method=PUT" in text and "principal=anonymous method=DELETE" in text
    assert token not in text + logs.stderr
    assert CANARY_ENV["GATEWAY_SERVICE_READ_TOKEN"] not in text + logs.stderr


HEALTH = (
    "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').status)"
)


def test_health_requests_leave_no_unstructured_access_lines(running: str) -> None:
    for _ in range(3):
        result = docker("exec", running, "python", "-c", HEALTH)
        assert result.returncode == 0, result.stderr
    logs = docker("logs", running)
    for line in (logs.stdout + logs.stderr).splitlines():
        assert "GET /health" not in line, line  # uvicorn access format: '127.0.0.1:1 - "GET ..."'
        assert ' - "' not in line, line
