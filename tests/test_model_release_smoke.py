import importlib.util
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from typing import ClassVar, Self

import pytest

_smoke_spec = importlib.util.spec_from_file_location(
    "model_release_smoke", Path(__file__).resolve().parents[1] / "scripts/model_release_smoke.py"
)
assert _smoke_spec is not None and _smoke_spec.loader is not None
smoke = importlib.util.module_from_spec(_smoke_spec)
sys.modules[_smoke_spec.name] = smoke
_smoke_spec.loader.exec_module(smoke)


@pytest.mark.parametrize(
    ("override", "expected"),
    [(None, 10), ("45", 45), ("120", 120), ("125", 125)],
)
def test_smoke_http_timeout_configuration(
    monkeypatch: pytest.MonkeyPatch, override: str | None, expected: int
) -> None:
    monkeypatch.setenv("MODEL_API_BASE_URL", "http://127.0.0.1:5800")
    monkeypatch.setenv("MODEL_API_ADMIN_TOKEN", "test-token")
    monkeypatch.setenv("MODEL_RELEASE_EXPECTED_MODEL_VERSION", "8")
    monkeypatch.setenv("MODEL_RELEASE_EXPECTED_RUN_ID", "run-456")
    if override is None:
        monkeypatch.delenv("MODEL_RELEASE_HTTP_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("MODEL_RELEASE_HTTP_TIMEOUT_SECONDS", override)

    assert smoke.read_configuration().http_timeout_seconds == expected


@pytest.mark.parametrize("override", ["", "0", "-1", "126", "inf", "-inf", "nan", "1e309", "abc"])
def test_smoke_http_timeout_rejects_invalid_override(
    monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    monkeypatch.setenv("MODEL_API_BASE_URL", "http://127.0.0.1:5800")
    monkeypatch.setenv("MODEL_API_ADMIN_TOKEN", "test-token")
    monkeypatch.setenv("MODEL_RELEASE_EXPECTED_MODEL_VERSION", "8")
    monkeypatch.setenv("MODEL_RELEASE_EXPECTED_RUN_ID", "run-456")
    monkeypatch.setenv("MODEL_RELEASE_HTTP_TIMEOUT_SECONDS", override)

    with pytest.raises(smoke.SmokeFailure, match="MODEL_RELEASE_HTTP_TIMEOUT_SECONDS"):
        smoke.read_configuration()


@pytest.mark.parametrize("timeout", [10, 45])
def test_smoke_request_uses_timeout_for_child_and_watchdog(
    monkeypatch: pytest.MonkeyPatch, timeout: int
) -> None:
    class FinishedProcess:
        exitcode = 0

        def __init__(self, *, target: object, args: tuple[object, ...], daemon: bool) -> None:
            assert target is smoke.execute_request
            assert daemon is True
            assert args[-1] == timeout
            self.path = str(args[-2])

        def start(self) -> None:
            pass

        def join(self, seconds: float) -> None:
            assert timeout - 2 < seconds <= timeout
            with open(self.path, "w", encoding="utf-8") as result_file:
                result_file.write('{"outcome":"response","body":"e30="}')

        def is_alive(self) -> bool:
            return False

    context = type("Context", (), {"Process": FinishedProcess})()
    monkeypatch.setattr(smoke.multiprocessing, "get_context", lambda _: context)
    configuration = smoke.SmokeConfiguration(
        "http://127.0.0.1:5800", "test-token", "8", "run-456", timeout
    )
    assert smoke.request_json(configuration, "/models/invoice-review/health", "GET") == {}


@pytest.mark.parametrize("timeout", [10, 45])
def test_smoke_child_urlopen_uses_configured_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, timeout: int
) -> None:
    class Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(_request: object, *, timeout: float) -> Response:
        assert timeout == expected_timeout
        return Response()

    expected_timeout = timeout
    monkeypatch.setattr(smoke, "urlopen", fake_urlopen)
    result_path = str(tmp_path / "result.json")
    smoke.execute_request("http://127.0.0.1:5800/health", None, {}, "GET", result_path, timeout)
    assert json.loads((tmp_path / "result.json").read_text()) == {
        "outcome": "response",
        "body": "e30=",
    }


class SmokeHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[tuple[str, dict[str, str], dict[str, object] | None]]] = []
    model_version: ClassVar[str] = "8"
    response_overrides: ClassVar[dict[str, tuple[int, dict[str, object] | bytes]]] = {}

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(content_length)) if content_length else None
        self.requests.append((self.path, dict(self.headers), body))
        if response := self.response_overrides.get(self.path):
            self._respond(*response)
            return
        if self.path == "/admin/models/invoice-review/refresh":
            self._respond(200, self._metadata())
        elif self.path == "/models/invoice-review/predict":
            self._respond(200, {**self._metadata(), "probability": 0.42})
        else:
            self._respond(404, {"detail": "not found"})

    def do_GET(self) -> None:
        self.requests.append((self.path, dict(self.headers), None))
        if response := self.response_overrides.get(self.path):
            self._respond(*response)
            return
        if self.path == "/models/invoice-review/health":
            self._respond(200, self._metadata())
        else:
            self._respond(404, {"detail": "not found"})

    @staticmethod
    def _metadata() -> dict[str, str]:
        return {
            "model_id": "invoice-review",
            "registered_model": "invoice-review-production",
            "alias": "champion",
            "model_version": SmokeHandler.model_version,
            "run_id": "run-456",
        }

    def _respond(self, status: int, body: dict[str, object] | bytes) -> None:
        encoded = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_: object) -> None:
        pass


class SlowDripHandler(BaseHTTPRequestHandler):
    stop_event: ClassVar[Event]

    def do_POST(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        try:
            while not self.stop_event.wait(0.1):
                self.wfile.write(b" ")
                self.wfile.flush()
        except BrokenPipeError:
            pass

    def log_message(self, *_: object) -> None:
        pass


def run_smoke(
    url: str, token: str, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/model_release_smoke.py"],
        cwd=os.getcwd(),
        env={
            **os.environ,
            "MODEL_API_BASE_URL": url,
            "MODEL_API_ADMIN_TOKEN": token,
            "MODEL_RELEASE_EXPECTED_MODEL_VERSION": "8",
            "MODEL_RELEASE_EXPECTED_RUN_ID": "run-456",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def assert_safe_failure(result: subprocess.CompletedProcess[str], url: str, token: str) -> None:
    assert result.returncode != 0
    assert token not in result.stdout
    assert token not in result.stderr
    assert url not in result.stdout
    assert url not in result.stderr


def test_model_release_smoke_refreshes_then_validates_health_and_prediction() -> None:
    SmokeHandler.requests = []
    SmokeHandler.model_version = "8"
    server = ThreadingHTTPServer(("127.0.0.1", 0), SmokeHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    token = "operator-token-that-must-not-be-printed"
    try:
        result = subprocess.run(
            [sys.executable, "scripts/model_release_smoke.py"],
            cwd=os.getcwd(),
            env={
                **os.environ,
                "MODEL_API_BASE_URL": url,
                "MODEL_API_ADMIN_TOKEN": token,
                "MODEL_RELEASE_EXPECTED_MODEL_VERSION": "8",
                "MODEL_RELEASE_EXPECTED_RUN_ID": "run-456",
            },
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert result.returncode == 0, result.stderr
    assert token not in result.stdout
    assert token not in result.stderr
    assert url not in result.stdout
    assert url not in result.stderr
    assert [request[0] for request in SmokeHandler.requests] == [
        "/admin/models/invoice-review/refresh",
        "/models/invoice-review/health",
        "/models/invoice-review/predict",
    ]
    assert SmokeHandler.requests[0][1]["X-Model-Admin-Token"] == token
    prediction = SmokeHandler.requests[2][2]
    assert prediction == {
        "invoice_amount_cents": 125000,
        "vendor_tenure_days": 730,
        "previous_incidents_12m": 0,
        "amount_vs_vendor_median": 1.1,
        "has_purchase_order": True,
        "three_way_match": True,
        "bank_account_recently_changed": False,
        "country_risk": "low",
    }


def test_model_release_smoke_fails_safely_for_missing_config_and_unexpected_metadata() -> None:
    missing_config = subprocess.run(
        [sys.executable, "scripts/model_release_smoke.py"],
        cwd=os.getcwd(),
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )

    SmokeHandler.requests = []
    SmokeHandler.model_version = "unexpected-version"
    server = ThreadingHTTPServer(("127.0.0.1", 0), SmokeHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    token = "operator-token-that-must-not-be-printed"
    try:
        unexpected_metadata = subprocess.run(
            [sys.executable, "scripts/model_release_smoke.py"],
            cwd=os.getcwd(),
            env={
                **os.environ,
                "MODEL_API_BASE_URL": url,
                "MODEL_API_ADMIN_TOKEN": token,
                "MODEL_RELEASE_EXPECTED_MODEL_VERSION": "8",
                "MODEL_RELEASE_EXPECTED_RUN_ID": "run-456",
            },
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        SmokeHandler.model_version = "8"

    assert missing_config.returncode == 1
    assert "Missing required environment variable" in missing_config.stderr
    assert unexpected_metadata.returncode == 1
    assert "model_version did not match" in unexpected_metadata.stderr
    assert token not in unexpected_metadata.stdout
    assert token not in unexpected_metadata.stderr
    assert url not in unexpected_metadata.stdout
    assert url not in unexpected_metadata.stderr
    assert [request[0] for request in SmokeHandler.requests] == [
        "/admin/models/invoice-review/refresh",
    ]


def test_model_release_smoke_rejects_invalid_admin_header_without_leaking_it() -> None:
    SmokeHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), SmokeHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    token = "header-secret\r\ninjected-header"
    try:
        result = run_smoke(url, token)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert_safe_failure(result, url, token)
    assert "header-secret" not in result.stderr
    assert "injected-header" not in result.stderr
    assert SmokeHandler.requests == []


def test_model_release_smoke_enforces_a_total_deadline_for_slow_drip_responses() -> None:
    SlowDripHandler.stop_event = Event()
    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowDripHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    token = "operator-token-that-must-not-be-printed"
    started_at = time.monotonic()
    try:
        result = run_smoke(url, token, timeout=12)
    finally:
        SlowDripHandler.stop_event.set()
        server.shutdown()
        thread.join()
        server.server_close()

    assert_safe_failure(result, url, token)
    assert "timed out" in result.stderr
    assert time.monotonic() - started_at < 12


@pytest.mark.parametrize(
    ("path", "status", "body", "expected_error"),
    [
        ("/admin/models/invoice-review/refresh", 503, {"detail": "unavailable"}, None),
        ("/admin/models/invoice-review/refresh", 200, b"not-json", None),
        (
            "/models/invoice-review/predict",
            200,
            {**SmokeHandler._metadata(), "probability": float("nan")},
            "predict returned an invalid probability",
        ),
        (
            "/models/invoice-review/predict",
            200,
            {**SmokeHandler._metadata(), "probability": 1.01},
            "predict returned an invalid probability",
        ),
    ],
)
def test_model_release_smoke_fails_safely_for_invalid_server_responses(
    path: str, status: int, body: dict[str, object] | bytes, expected_error: str | None
) -> None:
    SmokeHandler.requests = []
    SmokeHandler.response_overrides = {path: (status, body)}
    server = ThreadingHTTPServer(("127.0.0.1", 0), SmokeHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    token = "operator-token-that-must-not-be-printed"
    try:
        result = run_smoke(url, token)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        SmokeHandler.response_overrides = {}

    assert_safe_failure(result, url, token)
    if expected_error:
        assert expected_error in result.stderr
