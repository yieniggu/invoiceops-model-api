"""Supervised inference worker: crash/hang -> 503 then recovery, quarantine, HTTP process safety."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import AUTH, make_settings, ref_body, sample_request
from fastapi.testclient import TestClient

from invoiceops_model_api.app import create_app
from invoiceops_model_api.inference import InferenceSupervisor
from invoiceops_model_api.isolation import CheckRunner

HELPERS = Path(__file__).parent / "helpers"
SRC = Path(__file__).resolve().parents[1] / "src" / "invoiceops_model_api"


@pytest.fixture()
def flag(tmp_path: Path) -> Path:
    return tmp_path / "fault"


@pytest.fixture()
def faulty(store, flag: Path) -> Iterator[TestClient]:
    settings = make_settings(store.uri, inference_timeout_seconds=3.0)
    supervisor = InferenceSupervisor(
        settings,
        entrypoint=(sys.executable, "-I", str(HELPERS / "faulty_worker.py")),
        extra_env={"FAULT_FLAG_FILE": str(flag)},
    )
    app = create_app(settings, supervisor=supervisor)
    yield TestClient(app)
    supervisor.shutdown()


def predict(client: TestClient, deployment: str):
    return client.post(f"/deployments/{deployment}/predict", json=sample_request(), headers=AUTH)


def deploy(client: TestClient, versions, deployment: str, key: str = "good") -> None:
    r = client.put(f"/deployments/{deployment}", json=ref_body(versions, key), headers=AUTH)
    assert r.status_code == 200, r.text


def test_models_live_in_a_separate_process(faulty: TestClient, versions) -> None:
    deploy(faulty, versions, "a")
    pid = faulty.app.state.supervisor.worker_pid()
    assert pid is not None and pid != os.getpid()
    assert predict(faulty, "a").status_code == 200


def test_worker_crash_returns_503_then_recovers(faulty: TestClient, versions, flag: Path) -> None:
    supervisor = faulty.app.state.supervisor
    deploy(faulty, versions, "a")
    deploy(faulty, versions, "b", "hgb")
    good = predict(faulty, "a").json()["probability"]
    first_pid = supervisor.worker_pid()

    flag.write_text("crash")
    crashed = predict(faulty, "a")
    assert crashed.status_code == 503
    assert crashed.json()["detail"]["code"] == "INFERENCE_UNAVAILABLE"
    assert faulty.get("/health").json() == {"status": "ok"}  # the HTTP process is alive
    assert faulty.get("/deployments", headers=AUTH).json()["loaded"] == 2

    flag.unlink()
    recovered = predict(faulty, "a")  # fresh worker, model reloaded from the cached bytes
    assert recovered.status_code == 200
    assert recovered.json()["probability"] == pytest.approx(good)
    assert supervisor.worker_pid() not in (None, first_pid)
    assert predict(faulty, "b").status_code == 200  # the other deployment reloads on demand too


def test_worker_hang_times_out_with_503_then_recovers(
    faulty: TestClient, versions, flag: Path
) -> None:
    deploy(faulty, versions, "a")
    flag.write_text("hang")
    hung = predict(faulty, "a")
    assert hung.status_code == 503 and hung.json()["detail"]["code"] == "INFERENCE_UNAVAILABLE"
    flag.unlink()
    assert predict(faulty, "a").status_code == 200


def test_deployment_that_crashes_the_worker_twice_is_quarantined(
    faulty: TestClient, versions, flag: Path
) -> None:
    supervisor = faulty.app.state.supervisor
    deploy(faulty, versions, "bad")
    deploy(faulty, versions, "fine", "hgb")
    cached = faulty.app.state.registry.get("bad").artifact_path

    flag.write_text("crash")
    first = predict(faulty, "bad")
    assert first.status_code == 503 and "quarantined" not in first.json()["detail"]
    flag.unlink()
    assert predict(faulty, "bad").status_code == 200  # one crash is forgiven
    flag.write_text("crash")
    second = predict(faulty, "bad")
    assert second.status_code == 503 and second.json()["detail"]["quarantined"] is True
    flag.unlink()

    assert predict(faulty, "bad").status_code == 404  # unloaded
    assert not cached.exists()  # cached bytes deleted
    listing = faulty.get("/deployments", headers=AUTH).json()
    assert [d["deploymentId"] for d in listing["deployments"]] == ["fine"]
    assert [q["deploymentId"] for q in listing["quarantined"]] == ["bad"]
    assert "crashed" in listing["quarantined"][0]["reason"]
    assert predict(faulty, "fine").status_code == 200  # innocent deployments are unaffected
    assert supervisor.worker_pid() is not None

    deploy(faulty, versions, "bad")  # an explicit redeploy clears the quarantine
    assert faulty.get("/deployments", headers=AUTH).json()["quarantined"] == []
    assert predict(faulty, "bad").status_code == 200


def test_unload_and_replace_free_the_cached_artifacts(faulty: TestClient, versions) -> None:
    deploy(faulty, versions, "a")
    first = faulty.app.state.registry.get("a").artifact_path
    assert first.exists()
    deploy(faulty, versions, "a", "hgb")  # replacement
    second = faulty.app.state.registry.get("a").artifact_path
    assert second != first and second.exists() and not first.exists()
    assert faulty.delete("/deployments/a", headers=AUTH).json()["unloaded"] is True
    assert not second.exists()


def test_shutdown_kills_the_worker_and_removes_the_cache(faulty: TestClient, versions) -> None:
    deploy(faulty, versions, "a")
    supervisor = faulty.app.state.supervisor
    pid = supervisor.worker_pid()
    cached = faulty.app.state.registry.get("a").artifact_path
    supervisor.shutdown()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # reaped
    assert not cached.exists()


# ---- the HTTP process never touches student models ---------------------------------------


def test_http_process_modules_never_import_the_model_loader() -> None:
    forbidden_modules = {"skops", "model_runtime", "structure", "_child", "_worker", "skops_policy"}
    for name in ("app", "checks", "artifacts", "registry", "inference", "isolation", "settings"):
        tree = ast.parse((SRC / f"{name}.py").read_text())
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or "", *[alias.name for alias in node.names]]
            for module in imported:
                parts = set(module.split("."))
                assert not parts & forbidden_modules, f"{name}.py imports {module}"
            if isinstance(node, ast.Attribute | ast.Name):
                ident = node.attr if isinstance(node, ast.Attribute) else node.id
                assert ident not in {"load_allowlisted", "load_checked"}, f"{name}.py uses {ident}"


def test_creating_the_app_and_serving_does_not_import_skops_in_the_http_process() -> None:
    code = (
        "import sys\n"
        "from invoiceops_model_api.app import create_app\n"
        "from invoiceops_model_api.settings import Settings\n"
        "create_app(Settings(service_token='t'*40, tracking_uri='http://127.0.0.1:1'))\n"
        "bad = [m for m in sys.modules if m == 'skops' or m.startswith('skops.')"
        " or m.endswith(('.model_runtime', '.skops_policy', '.structure'))]\n"
        "assert not bad, bad\n"
    )
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    done = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert done.returncode == 0, done.stderr


# ---- no MLflow lock: concurrent checks run side by side ---------------------------------


class BarrierRunner(CheckRunner):
    """Both checks must be inside the check stage at the same time (nothing serializes them)."""

    def __init__(self, settings, parties: int) -> None:
        super().__init__(settings)
        self.barrier = threading.Barrier(parties, timeout=60)

    def run(self, skops_path, matrix, columns):
        self.barrier.wait()
        return super().run(skops_path, matrix, columns)


def test_concurrent_deploy_checks_do_not_serialize(store, versions) -> None:
    settings = make_settings(store.uri, max_concurrent_checks=2)
    runner = BarrierRunner(settings, parties=2)
    app = create_app(settings, check_runner=runner)
    client = TestClient(app)
    results: list = []

    def check(key: str) -> None:
        results.append(
            client.post("/deploy-check", json=ref_body(versions, key), headers=AUTH).status_code
        )

    threads = [threading.Thread(target=check, args=(key,)) for key in ("good", "hgb")]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
    finally:
        app.state.supervisor.shutdown()
    assert not runner.barrier.broken
    assert sorted(results) == [200, 200]
