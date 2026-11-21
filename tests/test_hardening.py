"""Bounded MLflow traffic, body/field limits, child reply schema, secrets and observability."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from conftest import AUTH, TOKEN, make_settings, ref_body, sample_request
from fastapi.testclient import TestClient

import invoiceops_model_api.app as app_module
from invoiceops_model_api import isolation
from invoiceops_model_api.app import create_app, principal_hash
from invoiceops_model_api.artifacts import MlflowUnavailable, _Tracking
from invoiceops_model_api.inference import InferenceSupervisor
from invoiceops_model_api.isolation import CheckRunner, sanitize_failure
from invoiceops_model_api.procutil import make_undumpable
from invoiceops_model_api.settings import ConfigError, Settings

HELPERS = Path(__file__).parent / "helpers"
REF = {"workspace": "group-a", "registeredModel": "m-good", "version": "1"}


class Upstream:
    """A tracking server that answers every request with a fixed status, optionally slowly."""

    def __init__(self, status: int = 503, delay: float = 0.0, body: bytes = b"{}") -> None:
        self.count = 0
        self.paths: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.count += 1
                outer.paths.append(self.path)
                time.sleep(delay)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def failing() -> Iterator[Upstream]:
    server = Upstream(503)
    yield server
    server.close()


def bounded_settings(url: str, **overrides) -> Settings:
    base = {
        "mlflow_max_retries": 1,
        "mlflow_backoff_factor": 0.1,
        "mlflow_request_timeout_seconds": 3.0,
        "mlflow_total_budget_seconds": 8.0,
    }
    return make_settings(url, **{**base, **overrides})


# ---- explicit retries, timeouts, budget; liveness independent of MLflow -------------


def test_always_503_fails_fast_with_a_bounded_number_of_attempts(failing: Upstream) -> None:
    settings = bounded_settings(failing.url)
    app = create_app(settings)
    client = TestClient(app)
    started = time.monotonic()
    response = client.post("/deploy-check", json=REF, headers=AUTH)
    elapsed = time.monotonic() - started
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "MLFLOW_UNAVAILABLE"
    # The MLflow defaults (7 retries, exponential backoff) take ~4 minutes. The call is bounded by
    # the configured total budget (twice that value allows for a loaded host); the exact attempt
    # count below is the real, clock-free assertion.
    assert elapsed < 2 * settings.mlflow_total_budget_seconds
    assert failing.count == 2  # one attempt + exactly one retry, then give up
    app.state.supervisor.shutdown()


def test_health_keeps_answering_while_deploy_checks_hit_a_failing_mlflow() -> None:
    slow_failure = Upstream(503, delay=0.4)
    app = create_app(bounded_settings(slow_failure.url, mlflow_max_retries=2))
    client = TestClient(app)
    results: list[int] = []

    def hammer() -> None:
        results.append(client.post("/deploy-check", json=REF, headers=AUTH).status_code)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    began = time.monotonic()
    for thread in threads:
        thread.start()
    latencies = []
    while any(thread.is_alive() for thread in threads):
        started = time.monotonic()
        assert client.get("/health").json() == {"status": "ok"}
        latencies.append(time.monotonic() - started)
        time.sleep(0.05)
    for thread in threads:
        thread.join()
    hammered = time.monotonic() - began
    # No absolute bound. A liveness probe blocked behind the checks would wait for (most
    # of) the whole burst, so it must stay far below the burst's own duration, whatever the host.
    assert len(latencies) >= 2
    assert max(latencies) < hammered / 2, (latencies, hammered)
    assert results and set(results) <= {502, 503}  # 503 BUSY past the download slots
    app.state.supervisor.shutdown()
    slow_failure.close()


def test_the_liveness_handler_is_a_coroutine() -> None:
    import inspect

    app = create_app(make_settings("http://127.0.0.1:1"))
    handler = next(r.endpoint for r in app.routes if getattr(r, "path", "") == "/health")
    assert inspect.iscoroutinefunction(handler)
    app.state.supervisor.shutdown()


def test_the_request_budget_is_shared_by_all_calls_of_one_check() -> None:
    server = Upstream(200, delay=0.6, body=json.dumps({"model_version": {}}).encode())
    settings = bounded_settings(
        server.url,
        mlflow_max_retries=0,
        mlflow_total_budget_seconds=1.0,
        mlflow_request_timeout_seconds=1.0,
    )
    tracking = _Tracking(settings, "group-a")
    endpoint = "/api/2.0/mlflow/model-versions/get"
    assert tracking._json(endpoint, {})[0] == 200  # 0.6 s of the 1 s budget
    with pytest.raises(MlflowUnavailable):  # the remaining 0.4 s clamp the timeout
        tracking._json(endpoint, {})
    count = server.count
    with pytest.raises(MlflowUnavailable, match="budget"):  # nothing left: no request at all
        tracking._json(endpoint, {})
    assert server.count == count
    server.close()


def test_explicit_bounds_are_validated() -> None:
    for bad in ({"mlflow_max_retries": 7}, {"mlflow_request_timeout_seconds": 120.0}):
        with pytest.raises(ConfigError):
            make_settings("http://127.0.0.1:1", **bad)
    with pytest.raises(ConfigError):
        make_settings(
            "http://127.0.0.1:1", mlflow_request_timeout_seconds=20.0, mlflow_total_budget_seconds=5
        )


def test_downloads_are_capped_and_the_excess_gets_503_busy() -> None:
    slow = Upstream(404, delay=1.5)
    app = create_app(
        bounded_settings(slow.url, max_concurrent_downloads=1, download_wait_seconds=0.3)
    )
    client = TestClient(app)
    first: list[int] = []
    holder = threading.Thread(
        target=lambda: first.append(
            client.post("/deploy-check", json=REF, headers=AUTH).status_code
        )
    )
    holder.start()
    arrival = time.monotonic() + 30
    while slow.count < 1 and time.monotonic() < arrival:  # wait for the event, not 0.4 s
        time.sleep(0.01)
    assert slow.count >= 1
    busy = client.post("/deploy-check", json=REF, headers=AUTH)
    assert busy.status_code == 503 and busy.json()["detail"]["code"] == "BUSY"
    holder.join()
    assert first  # the first one was served (its own outcome is irrelevant here)
    app.state.supervisor.shutdown()
    slow.close()


# ---- parent secrets out of reach, children never get them -----------------------


def test_children_environment_holds_no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MODEL_API_SERVICE_TOKEN",
        "OFFICIAL_VALIDATION_SEED",
        "GATEWAY_SERVICE_READ_TOKEN",
    ):
        monkeypatch.setenv(name, "super-secret-value-" + name)
    echo = (
        sys.executable,
        "-I",
        "-c",
        (
            "import json, os, sys\n"
            "from multiprocessing.connection import Connection\n"
            "Connection(int(sys.argv[1])).send_bytes(json.dumps(dict(os.environ)).encode())"
        ),
    )
    child = isolation.spawn(echo, 64)
    try:
        assert child.conn.poll(20)
        seen = json.loads(child.conn.recv_bytes())
    finally:
        child.kill()
    assert "super-secret-value" not in json.dumps(seen)
    assert set(isolation.child_env()) <= set(seen)
    assert not [k for k in seen if any(m in k.upper() for m in isolation.SECRET_ENV_MARKERS)]
    with pytest.raises(ValueError):
        isolation.child_env({"MODEL_API_SERVICE_TOKEN": "x"})
    with pytest.raises(ValueError):
        isolation.child_env({"SOME_PASSWORD": "x"})


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc/<pid>/environ is Linux")
def test_a_child_cannot_read_the_parents_environment_through_proc() -> None:
    script = textwrap.dedent(
        """
        import os, subprocess, sys
        from invoiceops_model_api.procutil import make_undumpable

        if sys.argv[1] == "harden":
            assert make_undumpable()
        os.environ["PARENT_SECRET_CANARY"] = "canary"  # also present in /proc/<pid>/environ below
        child = subprocess.run(
            [sys.executable, "-I", "-c",
             "import os, sys\\n"
             "try:\\n"
             "    data = open(f'/proc/{os.getppid()}/environ', 'rb').read()\\n"
             "    print('READ', b'PARENT_SECRET_CANARY' in data)\\n"
             "except PermissionError:\\n"
             "    print('DENIED')"],
            capture_output=True, text=True, check=True,
        )
        print(child.stdout.strip())
        """
    )

    def run(mode: str) -> str:
        env = {**os.environ, "PARENT_SECRET_CANARY": "canary"}  # really in the initial environment
        result = subprocess.run(
            [sys.executable, "-c", script, mode],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        return result.stdout.strip()

    assert run("open").startswith("READ")  # control: without the flag the child can read it
    assert run("harden") == "DENIED"


def test_make_undumpable_is_a_noop_off_linux() -> None:
    applied = make_undumpable()
    assert applied == sys.platform.startswith("linux") or not applied


# ---- the child's reply is reduced to a fixed schema ---------------------------------


def test_only_expected_keys_and_fixed_messages_survive_sanitizing() -> None:
    hostile = {
        "ok": False,
        "code": "UNTRUSTED_TYPE",
        "message": "LEAK " + "feature-bits " * 100,
        "detail": {
            "types": [
                "os.system",
                "not a type name!",
                7,
                "a" * 500,
                *[f"t{i}.X" for i in range(30)],
            ],
            "leak": [1.5, 2.5],
            "check": "x" * 500,
        },
        "extra": "field",
    }
    code, message, detail = sanitize_failure(hostile)
    assert code == "UNTRUSTED_TYPE"
    assert message == isolation.CHILD_MESSAGES["UNTRUSTED_TYPE"]
    assert set(detail) == {"types"}
    assert len(detail["types"]) == isolation.MAX_TYPE_NAMES
    assert all(isinstance(t, str) and len(t) <= 120 and " " not in t for t in detail["types"])


@pytest.mark.parametrize(
    "reply",
    [
        {"ok": False, "code": "MADE_UP", "message": "secret", "detail": {"a": 1}},
        {"ok": False, "code": ["list"], "message": "x"},
        {"ok": False, "code": None},
        {"ok": False},
    ],
)
def test_unknown_codes_become_a_generic_crash_without_detail(reply: dict) -> None:
    code, message, detail = sanitize_failure(reply)
    assert code == "MODEL_CHECK_CRASHED"
    assert message == isolation.CHILD_MESSAGES["MODEL_CHECK_CRASHED"]
    assert detail == {}


def test_other_codes_never_carry_detail() -> None:
    for known in isolation.CHILD_MESSAGES:
        if known == "UNTRUSTED_TYPE":
            continue
        _, _, detail = sanitize_failure({"code": known, "detail": {"check": "x", "types": ["a.B"]}})
        assert detail == {}


def test_a_compromised_child_cannot_smuggle_text_through_the_runner(tmp_path: Path) -> None:
    reply = tmp_path / "reply.json"
    reply.write_text(
        json.dumps(
            {
                "ok": False,
                "code": "MODEL_STRUCTURE_INVALID",
                "message": "validation features: " + "0.123456 " * 50,
                "detail": {"check": "0.5 0.25 0.125", "values": [0.1] * 100},
            }
        )
    )
    runner = CheckRunner(
        make_settings("http://127.0.0.1:1"),
        entrypoint=(sys.executable, "-I", str(HELPERS / "chatty_child.py")),
        extra_env={"CHATTY_REPLY_FILE": str(reply)},
    )
    outcome = runner.run(tmp_path / "model.skops", np.zeros((2, 3)), ["a", "b", "c"])
    assert not outcome.ok
    assert outcome.code == "MODEL_STRUCTURE_INVALID"
    assert outcome.message == isolation.CHILD_MESSAGES["MODEL_STRUCTURE_INVALID"]
    assert outcome.detail == {}


def test_a_scores_reply_with_a_bad_positive_index_is_a_crash(tmp_path: Path) -> None:
    reply = tmp_path / "reply.json"
    reply.write_text(json.dumps({"ok": True, "positive_index": 7, "scores": [0.1, 0.2]}))
    runner = CheckRunner(
        make_settings("http://127.0.0.1:1"),
        entrypoint=(sys.executable, "-I", str(HELPERS / "chatty_child.py")),
        extra_env={"CHATTY_REPLY_FILE": str(reply)},
    )
    outcome = runner.run(tmp_path / "model.skops", np.zeros((2, 3)), ["a", "b", "c"])
    assert not outcome.ok and outcome.code == "MODEL_CHECK_CRASHED"


# ---- field bounds, overflow, total priors, body size ------------------------------


def with_amount(amount) -> dict:
    body = sample_request()
    body["invoice"]["amountPesos"] = amount
    return body


@pytest.mark.parametrize("amount", [10**13 + 1, 10**400, -1])
def test_out_of_range_amounts_are_422_not_500(client: TestClient, versions, amount) -> None:
    client.put("/deployments/d1", json=ref_body(versions, "good"), headers=AUTH)
    raw = json.dumps(with_amount(amount))
    response = client.post(
        "/deployments/d1/predict",
        content=raw,
        headers={**AUTH, "Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_REQUEST"
    assert "10000000" not in response.text  # values are never echoed


def test_prior_amounts_and_names_are_bounded_too(client: TestClient, versions) -> None:
    client.put("/deployments/d1", json=ref_body(versions, "good"), headers=AUTH)
    body = sample_request()
    body["priorInvoices"][0]["amountPesos"] = 10**20
    assert client.post("/deployments/d1/predict", json=body, headers=AUTH).status_code == 422
    body = sample_request()
    body["invoice"]["supplierName"] = "x" * 201
    assert client.post("/deployments/d1/predict", json=body, headers=AUTH).status_code == 422


def test_an_overflow_while_computing_features_is_422(
    client: TestClient, versions, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.put("/deployments/d1", json=ref_body(versions, "good"), headers=AUTH)

    def explode(invoice, priors):
        raise OverflowError("int too large to convert to float")

    monkeypatch.setattr(app_module, "compute_features", explode)
    response = client.post("/deployments/d1/predict", json=sample_request(), headers=AUTH)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_INVOICE"


def test_total_priors_per_batch_request_are_capped(store, versions) -> None:
    app = create_app(make_settings(store.uri, max_total_priors=10))
    client = TestClient(app)
    try:
        client.put("/deployments/d1", json=ref_body(versions, "good"), headers=AUTH)
        item = sample_request()
        item["priorInvoices"] = item["priorInvoices"] * 3  # 6 priors per item
        ok = client.post("/deployments/d1/predict-batch", json={"items": [item]}, headers=AUTH)
        assert ok.status_code == 200
        over = client.post(
            "/deployments/d1/predict-batch", json={"items": [item, item]}, headers=AUTH
        )
        assert over.status_code == 422
        assert over.json()["detail"]["code"] == "TOO_MANY_PRIORS"
        assert over.json()["detail"]["max"] == 10
    finally:
        app.state.supervisor.shutdown()


def test_request_bodies_over_the_limit_are_413(store) -> None:
    app = create_app(make_settings(store.uri, max_body_bytes=2048))
    client = TestClient(app)
    big = {"invoice": {"supplierName": "x" * 100}, "pad": "y" * 5000}
    declared = client.post("/deployments/d1/predict", json=big, headers=AUTH)
    assert declared.status_code == 413
    assert declared.json()["detail"]["code"] == "PAYLOAD_TOO_LARGE"

    def chunks() -> Iterator[bytes]:  # no Content-Length: counted while streaming
        for _ in range(10):
            yield b"z" * 500

    streamed = client.post(
        "/deployments/d1/predict",
        content=chunks(),
        headers={**AUTH, "Content-Type": "application/json"},
    )
    assert streamed.status_code == 413
    small = client.post("/deployments/d1/predict", json={"a": 1}, headers=AUTH)
    assert small.status_code != 413
    app.state.supervisor.shutdown()


def test_limits_are_configurable_from_the_environment() -> None:
    env = {
        "MODEL_API_SERVICE_TOKEN": TOKEN,
        "MLFLOW_TRACKING_URI": "http://127.0.0.1:5000",
        "OFFICIAL_VALIDATION_SEED": "9f2c5d71e0a84b36c7d1f0958e2a6b43",
        "MAX_TOTAL_PRIORS": "123",
        "MAX_BODY_BYTES": "4096",
        "MODEL_API_MLFLOW_MAX_RETRIES": "1",
        "MODEL_API_MLFLOW_TIMEOUT_SECONDS": "5",
        "MODEL_API_MLFLOW_BUDGET_SECONDS": "20",
    }
    settings = Settings.from_env(env)
    assert (settings.max_total_priors, settings.max_body_bytes) == (123, 4096)
    assert (settings.mlflow_max_retries, settings.mlflow_total_budget_seconds) == (1, 20.0)


# ---- activating a model does not stop the others from serving -------------------


def test_activation_does_not_block_other_deployments(store, versions, tmp_path: Path) -> None:
    flag = tmp_path / "slow-load"
    settings = make_settings(store.uri, inference_timeout_seconds=1.0, check_timeout_seconds=60.0)
    supervisor = InferenceSupervisor(
        settings,
        entrypoint=(sys.executable, "-I", str(HELPERS / "slow_load_worker.py")),
        extra_env={"SLOW_LOAD_FLAG": str(flag)},
    )
    client = TestClient(create_app(settings, supervisor=supervisor))
    try:
        assert client.put(
            "/deployments/a", json=ref_body(versions, "good"), headers=AUTH
        ).is_success
        before = client.post("/deployments/a/predict", json=sample_request(), headers=AUTH).json()
        flag.write_text(
            "4"
        )  # the next load (the new model's) takes 4 s, well over the 1 s predict timeout
        outcome: dict[str, int] = {}
        activation = threading.Thread(
            target=lambda: outcome.update(
                status=client.put(
                    "/deployments/b", json=ref_body(versions, "hgb"), headers=AUTH
                ).status_code
            )
        )
        started = time.monotonic()
        activation.start()
        time.sleep(1.5)  # B's load is in progress now (the check child already finished)
        served = []
        while activation.is_alive() and time.monotonic() - started < 30:
            r = client.post("/deployments/a/predict", json=sample_request(), headers=AUTH)
            served.append((r.status_code, r.json().get("probability")))
            time.sleep(0.1)
        activation.join()
        assert outcome["status"] == 200
        assert served, "the activation finished before any prediction was attempted"
        assert all(code == 200 for code, _ in served), served
        assert all(p == pytest.approx(before["probability"]) for _, p in served)
        assert (
            client.post("/deployments/b/predict", json=sample_request(), headers=AUTH).status_code
            == 200
        )
        assert (
            client.post("/deployments/a/predict", json=sample_request(), headers=AUTH).status_code
            == 200
        )
    finally:
        supervisor.shutdown()


def test_a_model_that_cannot_be_activated_leaves_the_others_serving(
    store, versions, tmp_path: Path
) -> None:
    settings = make_settings(store.uri)
    supervisor = InferenceSupervisor(settings)
    client = TestClient(create_app(settings, supervisor=supervisor))
    try:
        client.put("/deployments/a", json=ref_body(versions, "good"), headers=AUTH)
        gone = tmp_path / "missing.skops"
        from invoiceops_model_api.inference import InferenceFailed

        with pytest.raises(InferenceFailed):
            supervisor.activate("ghost-slot", gone, ["f"])
        assert (
            client.post("/deployments/a/predict", json=sample_request(), headers=AUTH).status_code
            == 200
        )
        assert "ghost-slot" not in supervisor._slots
    finally:
        supervisor.shutdown()


# ---- audit hash and counters --------------------------------------------------------


def test_principal_hash_is_stable_short_and_does_not_reveal_the_secret() -> None:
    key = make_settings("http://127.0.0.1:1").audit_key
    first, again = principal_hash(TOKEN, key), principal_hash(TOKEN, key)
    assert first == again and len(first) == 16 and int(first, 16) >= 0
    assert TOKEN not in first and first not in TOKEN
    assert principal_hash(TOKEN + "x", key) != first
    import hashlib

    assert first != hashlib.sha256(TOKEN.encode()).hexdigest()[:16]  # not the plain digest


# ---- the audit key is never a constant in the source --------------------------------


def test_audit_key_is_derived_from_the_service_token_or_dedicated() -> None:
    a = make_settings("http://127.0.0.1:1", service_token="a" * 40)
    b = make_settings("http://127.0.0.1:1", service_token="b" * 40)
    assert a.audit_key != b.audit_key  # per deployment, not a fixed key
    assert a.audit_key != a.service_token.encode()  # derived, not the token itself
    assert principal_hash("t", a.audit_key) != principal_hash("t", b.audit_key)
    dedicated = make_settings("http://127.0.0.1:1", service_token="a" * 40, audit_hmac_key="k" * 48)
    assert dedicated.audit_key == b"k" * 48
    assert principal_hash("t", dedicated.audit_key) != principal_hash("t", a.audit_key)


def test_audit_key_env_var_is_read_and_validated() -> None:
    env = {
        "MODEL_API_SERVICE_TOKEN": TOKEN,
        "MLFLOW_TRACKING_URI": "http://127.0.0.1:5000",
        "OFFICIAL_VALIDATION_SEED": "9f2c5d71e0a84b36c7d1f0958e2a6b43",
    }
    assert Settings.from_env(env).audit_hmac_key is None
    assert Settings.from_env({**env, "MODEL_API_AUDIT_HMAC_KEY": "z" * 40}).audit_key == b"z" * 40
    with pytest.raises(ConfigError, match="MODEL_API_AUDIT_HMAC_KEY"):
        Settings.from_env({**env, "MODEL_API_AUDIT_HMAC_KEY": "short"})


def test_no_fixed_audit_key_remains_in_the_source() -> None:
    source = (Path(app_module.__file__)).read_text()
    assert "invoiceops-model-api/principal/v1" not in source


def test_mutations_are_audited_with_the_hash_and_counted(
    client: TestClient, versions, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="invoiceops_model_api")
    client.put("/deployments/audit", json=ref_body(versions, "good"), headers=AUTH)
    client.get("/deployments", headers=AUTH)
    client.delete("/deployments/audit", headers=AUTH)
    client.put("/deployments/audit", json=REF, headers={"Authorization": "Bearer " + "w" * 40})
    audit = [r.getMessage() for r in caplog.records if r.getMessage().startswith("audit ")]
    assert any(
        f"principal={principal_hash(TOKEN, make_settings('http://x').audit_key)} method=PUT route=/deployments/{{deployment_id}} status=200"
        in line
        for line in audit
    )
    assert any("method=DELETE" in line for line in audit)
    assert any("principal=anonymous" in line and "status=401" in line for line in audit)
    assert not any(
        r.getMessage().startswith("audit ") and "GET" in r.getMessage() for r in caplog.records
    )
    assert TOKEN not in caplog.text and "w" * 40 not in caplog.text
    counts = client.get("/metrics", headers=AUTH).json()["requests"]
    assert counts["PUT /deployments/{deployment_id} 200"] == 1
    assert counts["PUT /deployments/{deployment_id} 401"] == 1
    assert counts["GET /deployments 200"] == 1
    assert "GET /metrics 200" not in counts or counts["GET /metrics 200"] >= 0


def test_an_unhandled_exception_in_a_mutation_is_still_audited_and_counted(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="invoiceops_model_api")

    def boom(deployment_id: str):
        raise RuntimeError("registry exploded")

    monkeypatch.setattr(client.app.state.registry, "remove", boom)
    crashing = TestClient(client.app, raise_server_exceptions=False)
    response = crashing.delete("/deployments/x", headers=AUTH)
    assert response.status_code == 500
    audit = [r.getMessage() for r in caplog.records if r.getMessage().startswith("audit ")]
    assert any("method=DELETE" in line and "status=500" in line for line in audit), audit
    counts = crashing.get("/metrics", headers=AUTH).json()["requests"]
    assert counts["DELETE /deployments/{deployment_id} 500"] == 1


def test_metrics_need_the_service_credential(client: TestClient) -> None:
    assert client.get("/metrics").status_code == 401


# ---- request counters have a bounded key space --------------------------------------


def test_unknown_methods_and_unmatched_routes_do_not_grow_the_counters(client: TestClient) -> None:
    for number in range(100):
        client.request(f"METHOD{number}", "/deployments", headers=AUTH)
        client.get(f"/no/such/route/{number}", headers=AUTH)
    counts = client.get("/metrics", headers=AUTH).json()["requests"]
    assert counts["OTHER /deployments 405"] == 100
    assert counts["GET unmatched 404"] == 100
    assert len(counts) <= 6, counts
    assert not any("METHOD" in key or "no/such" in key for key in counts)


# ---- a failed prctl is visible ---------------------------------------------------------


def test_a_failed_hardening_on_linux_is_logged_and_exposed_in_metrics(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(app_module, "make_undumpable", lambda: False)
    monkeypatch.setattr(app_module, "ON_LINUX", True)
    caplog.set_level(logging.ERROR, logger="invoiceops_model_api")
    app = create_app(make_settings("http://127.0.0.1:9"))
    assert any(
        r.levelno == logging.ERROR and "process_hardening_failed" in r.getMessage()
        for r in caplog.records
    )
    with TestClient(app) as client:
        assert client.get("/metrics", headers=AUTH).json()["processHardened"] is False


def test_a_failed_hardening_off_linux_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(app_module, "make_undumpable", lambda: False)
    monkeypatch.setattr(app_module, "ON_LINUX", False)
    caplog.set_level(logging.ERROR, logger="invoiceops_model_api")
    create_app(make_settings("http://127.0.0.1:9"))
    assert not [r for r in caplog.records if "process_hardening_failed" in r.getMessage()]


def test_metrics_report_a_hardened_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "make_undumpable", lambda: True)
    app = create_app(make_settings("http://127.0.0.1:9"))
    with TestClient(app) as client:
        assert client.get("/metrics", headers=AUTH).json()["processHardened"] is True
