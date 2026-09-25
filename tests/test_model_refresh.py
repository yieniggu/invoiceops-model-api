import logging
import os
import time
from pathlib import Path
from threading import Lock, Thread
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import invoiceops_model_api.app as application

ADMIN_HEADERS = {"X-Model-Admin-Token": "test-admin-token"}


def configure_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.setenv("MODEL_API_ADMIN_TOKEN", "test-admin-token")


def loaded_model(model: object, version: str, run_id: str) -> application.LoadedModel:
    return application.LoadedModel(model=model, model_version=version, run_id=run_id)


def refreshed_model_loader(*_: object) -> application.LoadedModel:
    return loaded_model("refreshed", "8", "run-456")


def slow_model_loader(*_: object) -> application.LoadedModel:
    child_pid_file = os.getenv("MODEL_REFRESH_TEST_CHILD_PID_FILE")
    if child_pid_file:
        Path(child_pid_file).write_text(str(os.getpid()))
    time.sleep(3)
    return loaded_model("late", "8", "run-456")


def delayed_model_loader(*_: object) -> application.LoadedModel:
    time.sleep(0.2)
    return refreshed_model_loader()


def failed_model_loader(*_: object) -> application.LoadedModel:
    raise RuntimeError("provider token=not-for-clients")


def untransferable_model_loader(*_: object) -> application.LoadedModel:
    return loaded_model(Lock(), "8", "run-456")


def abnormal_model_loader(*_: object) -> application.LoadedModel:
    os._exit(9)


def test_refresh_timeout_preserves_the_active_model_and_logs_a_safe_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    configure_refresh(monkeypatch)
    monkeypatch.setenv("MODEL_REFRESH_TIMEOUT_SECONDS", "0.8")
    child_pid_file = tmp_path / "loader.pid"
    monkeypatch.setenv("MODEL_REFRESH_TEST_CHILD_PID_FILE", str(child_pid_file))
    monkeypatch.setattr(
        application,
        "load_model_from_mlflow",
        lambda *_: SimpleNamespace(model="active", model_version="7", run_id="run-123"),
    )
    caplog.set_level(logging.INFO, logger="invoiceops_model_api.refresh")

    with TestClient(application.create_app()) as client:
        monkeypatch.setattr(application, "load_model_from_mlflow", slow_model_loader)
        started_at = time.monotonic()
        refresh = client.post("/admin/models/invoice-review/refresh", headers=ADMIN_HEADERS)
        elapsed = time.monotonic() - started_at
        health = client.get("/models/invoice-review/health")

    assert elapsed < 1.2
    assert refresh.status_code == 503
    assert health.status_code == 200
    assert health.json()["model_version"] == "7"
    assert [record.message for record in caplog.records] == [
        "model_refresh_attempt",
        "model_refresh_failed",
    ]
    assert caplog.records[1].reason == "timeout"
    assert "MODEL_API_ADMIN_TOKEN" not in caplog.text
    assert child_pid_file.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(child_pid_file.read_text()), 0)


def test_refresh_requires_an_administrative_operational_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_refresh(monkeypatch)
    monkeypatch.setattr(
        application,
        "load_model_from_mlflow",
        lambda *_: SimpleNamespace(model=object(), model_version="7", run_id="run-123"),
    )

    with TestClient(application.create_app()) as client:
        missing = client.post("/admin/models/invoice-review/refresh")
        invalid = client.post(
            "/admin/models/invoice-review/refresh",
            headers={"X-Model-Admin-Token": "incorrect"},
        )

    assert missing.status_code == 401
    assert missing.json() == {"detail": "Administrative credentials are invalid"}
    assert invalid.status_code == 401
    assert invalid.json() == {"detail": "Administrative credentials are invalid"}


def test_refresh_loads_the_current_champion_and_health_reports_the_active_version(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    configure_refresh(monkeypatch)
    monkeypatch.setattr(
        application,
        "load_model_from_mlflow",
        lambda *_: SimpleNamespace(model="active", model_version="7", run_id="run-123"),
    )
    caplog.set_level(logging.INFO, logger="invoiceops_model_api.refresh")

    with TestClient(application.create_app()) as client:
        monkeypatch.setattr(
            application,
            "load_model_from_mlflow",
            refreshed_model_loader,
        )
        refresh = client.post("/admin/models/invoice-review/refresh", headers=ADMIN_HEADERS)
        health = client.get("/models/invoice-review/health")

    expected = {
        "model_id": "invoice-review",
        "registered_model": "invoice-review-production",
        "alias": "champion",
        "model_version": "8",
        "run_id": "run-456",
    }
    assert refresh.status_code == 200
    assert refresh.json() == expected
    assert health.status_code == 200
    assert health.json() == expected
    assert [record.message for record in caplog.records] == [
        "model_refresh_attempt",
        "model_refresh_succeeded",
    ]


def test_failed_refresh_retains_the_previously_active_model(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    configure_refresh(monkeypatch)
    active_model = object()
    monkeypatch.setattr(
        application,
        "load_model_from_mlflow",
        lambda *_: SimpleNamespace(model=active_model, model_version="7", run_id="run-123"),
    )
    caplog.set_level(logging.INFO, logger="invoiceops_model_api.refresh")

    with TestClient(application.create_app()) as client:
        monkeypatch.setattr(application, "load_model_from_mlflow", failed_model_loader)
        refresh = client.post("/admin/models/invoice-review/refresh", headers=ADMIN_HEADERS)
        health = client.get("/models/invoice-review/health")

    assert refresh.status_code == 503
    assert refresh.json() == {"detail": "Model refresh is unavailable"}
    assert "provider token=not-for-clients" not in refresh.text
    assert "provider token=not-for-clients" not in caplog.text
    assert [record.message for record in caplog.records] == [
        "model_refresh_attempt",
        "model_refresh_failed",
    ]
    assert caplog.records[1].reason == "load_failure"
    assert health.status_code == 200
    assert health.json()["model_version"] == "7"
    assert health.json()["run_id"] == "run-123"


def test_health_reads_the_previous_complete_model_while_refresh_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_refresh(monkeypatch)
    monkeypatch.setattr(
        application,
        "load_model_from_mlflow",
        lambda *_: SimpleNamespace(model="active", model_version="7", run_id="run-123"),
    )
    with TestClient(application.create_app()) as client:
        monkeypatch.setattr(application, "load_model_from_mlflow", delayed_model_loader)
        refresh_response: list[object] = []
        refresh_thread = Thread(
            target=lambda: refresh_response.append(
                client.post("/admin/models/invoice-review/refresh", headers=ADMIN_HEADERS)
            )
        )
        refresh_thread.start()
        time.sleep(0.05)
        health = client.get("/models/invoice-review/health")
        refresh_thread.join(timeout=1)

    assert not refresh_thread.is_alive()
    assert health.status_code == 200
    assert health.json()["model_version"] == "7"
    assert refresh_response[0].status_code == 200


def test_refresh_rejects_an_untransferable_model_and_preserves_the_active_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_refresh(monkeypatch)
    monkeypatch.setattr(
        application,
        "load_model_from_mlflow",
        lambda *_: SimpleNamespace(model="active", model_version="7", run_id="run-123"),
    )

    with TestClient(application.create_app()) as client:
        monkeypatch.setattr(
            application,
            "load_model_from_mlflow",
            untransferable_model_loader,
        )
        refresh = client.post("/admin/models/invoice-review/refresh", headers=ADMIN_HEADERS)
        health = client.get("/models/invoice-review/health")

    assert refresh.status_code == 503
    assert health.status_code == 200
    assert health.json()["model_version"] == "7"


def test_refresh_handles_an_abnormal_loader_exit_and_preserves_the_active_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_refresh(monkeypatch)
    monkeypatch.setattr(
        application,
        "load_model_from_mlflow",
        lambda *_: SimpleNamespace(model="active", model_version="7", run_id="run-123"),
    )

    with TestClient(application.create_app()) as client:
        monkeypatch.setattr(application, "load_model_from_mlflow", abnormal_model_loader)
        refresh = client.post("/admin/models/invoice-review/refresh", headers=ADMIN_HEADERS)
        health = client.get("/models/invoice-review/health")

    assert refresh.status_code == 503
    assert health.status_code == 200
    assert health.json()["model_version"] == "7"


def test_refresh_returns_not_found_for_an_unknown_model_after_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_refresh(monkeypatch)
    monkeypatch.setattr(
        application,
        "load_model_from_mlflow",
        lambda *_: SimpleNamespace(model=object(), model_version="7", run_id="run-123"),
    )

    with TestClient(application.create_app()) as client:
        response = client.post("/admin/models/unknown/refresh", headers=ADMIN_HEADERS)

    assert response.status_code == 404
    assert response.json() == {"detail": "Model not found"}


def test_missing_administrative_credential_fails_closed_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.delenv("MODEL_API_ADMIN_TOKEN", raising=False)
    calls: list[object] = []
    monkeypatch.setattr(application, "load_model_from_mlflow", lambda *_: calls.append(None))

    with (
        pytest.raises(RuntimeError, match="MODEL_API_ADMIN_TOKEN must be configured"),
        TestClient(application.create_app()),
    ):
        pass

    assert calls == []


@pytest.mark.parametrize("timeout", ["0", "NaN", "not-a-number"])
def test_invalid_refresh_timeout_fails_closed_at_startup(
    monkeypatch: pytest.MonkeyPatch, timeout: str
) -> None:
    configure_refresh(monkeypatch)
    monkeypatch.setenv("MODEL_REFRESH_TIMEOUT_SECONDS", timeout)
    calls: list[object] = []
    monkeypatch.setattr(application, "load_model_from_mlflow", lambda *_: calls.append(None))

    with (
        pytest.raises(
            RuntimeError,
            match="MODEL_REFRESH_TIMEOUT_SECONDS must be a finite positive number",
        ),
        TestClient(application.create_app()),
    ):
        pass

    assert calls == []
