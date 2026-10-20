"""Exercise lifespan with real owned spawn processes, without MLflow or sockets."""

import asyncio
import logging
import multiprocessing
import os
import time
from threading import Event

import pytest
from fastapi.testclient import TestClient
from worker_fixtures import Estimator

import invoiceops_model_api.app as application


def delayed_startup_loader(*_: object) -> application.LoadedModel:
    time.sleep(0.15)
    return application.LoadedModel(Estimator(), "7", "run-123")


def blocked_startup_loader(*_: object) -> application.LoadedModel:
    time.sleep(2)
    return application.LoadedModel(Estimator(), "8", "run-late")


def missing_startup_loader(*_: object) -> application.LoadedModel:
    raise RuntimeError("private-provider-password")


def partial_startup_loader(entry, _uri):
    if entry.registered_model == "missing-model":
        raise RuntimeError("private-provider-password")
    return application.LoadedModel(Estimator(), "7", "run-123")


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.setenv("MODEL_API_ADMIN_TOKEN", "test-admin-token")
    monkeypatch.setenv("MODEL_STARTUP_TIMEOUT_SECONDS", "0.8")


def test_startup_deadline_terminates_and_reaps_owned_loader(configured, monkeypatch, caplog):
    monkeypatch.setattr(application, "load_model_from_mlflow", blocked_startup_loader)
    process_type = application.model_load_process_context().Process
    actual_start, actual_close = process_type.start, process_type.close
    started, closed = [], []

    def observe_start(process):
        actual_start(process)
        started.append(process.pid)

    def observe_close(process):
        closed.append((process.exitcode, process.is_alive()))
        actual_close(process)

    monkeypatch.setattr(process_type, "start", observe_start)
    monkeypatch.setattr(process_type, "close", observe_close)
    caplog.set_level(logging.INFO, logger="invoiceops_model_api.refresh")
    before = time.monotonic()
    with TestClient(application.create_app()) as client:
        assert time.monotonic() - before < 1.5
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/models/invoice-review/health").status_code == 503
    assert len(started) == 1
    assert len(closed) == 1 and closed[0][0] is not None and closed[0][1] is False
    assert not any(p.pid in started for p in multiprocessing.active_children())
    with pytest.raises(ProcessLookupError):
        os.kill(started[0], 0)
    failure = next(r for r in caplog.records if r.message == "model_startup_failed")
    assert failure.reason == "timeout"


def test_slow_startup_within_budget_publishes_complete_champion(configured, monkeypatch):
    monkeypatch.setenv("MODEL_STARTUP_TIMEOUT_SECONDS", "3")
    monkeypatch.setattr(application, "load_model_from_mlflow", delayed_startup_loader)
    with TestClient(application.create_app()) as client:
        health = client.get("/models/invoice-review/health")
        assert health.status_code == 200
        assert health.json() == {
            "model_id": "invoice-review",
            "registered_model": "invoice-review-production",
            "alias": "champion",
            "model_version": "7",
            "run_id": "run-123",
        }


def test_missing_champion_is_safe_and_process_health_remains_available(
    configured,
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(application, "load_model_from_mlflow", missing_startup_loader)
    with TestClient(application.create_app()) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/models/invoice-review/health")
        assert response.status_code == 503
        assert "private-provider-password" not in response.text + caplog.text
    assert any(r.message == "model_startup_failed" for r in caplog.records)


def test_partial_startup_failure_preserves_other_complete_model(configured, monkeypatch):
    monkeypatch.setenv("MODEL_STARTUP_TIMEOUT_SECONDS", "3")
    monkeypatch.setattr(application, "load_model_from_mlflow", partial_startup_loader)
    monkeypatch.setattr(
        application,
        "MODEL_CATALOG",
        {
            **application.MODEL_CATALOG,
            "missing": application.ModelCatalogEntry("missing-model", "champion"),
        },
    )
    app = application.create_app()
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/models/invoice-review/health").status_code == 200
        assert client.get("/models/missing/health").status_code == 503
        assert app.state.model_registry["invoice-review"].model_version == "7"
        assert app.state.model_registry["missing"] is None


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_invalid_startup_deadline_fails_before_starting_loader(configured, monkeypatch, value):
    monkeypatch.setenv("MODEL_STARTUP_TIMEOUT_SECONDS", value)
    monkeypatch.setattr(application, "load_model_from_mlflow", missing_startup_loader)
    with (
        pytest.raises(RuntimeError, match="MODEL_STARTUP_TIMEOUT_SECONDS.*finite positive"),
        TestClient(application.create_app()),
    ):
        pass


def test_cancelled_lifespan_waits_for_owned_child_cleanup(configured, monkeypatch):
    monkeypatch.setenv("MODEL_STARTUP_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(application, "load_model_from_mlflow", blocked_startup_loader)
    process_type = application.model_load_process_context().Process
    actual_start = process_type.start
    started = []

    def observe_start(process):
        actual_start(process)
        started.append(process.pid)

    monkeypatch.setattr(process_type, "start", observe_start)

    async def cancel_startup():
        app = application.create_app()
        context = application.model_lifespan(app)
        task = asyncio.create_task(context.__aenter__())
        deadline = time.monotonic() + 2
        while not started and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert started
        before = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert time.monotonic() - before < 1
        assert app.state.model_registry["invoice-review"] is None

    asyncio.run(cancel_startup())
    assert not any(p.pid in started for p in multiprocessing.active_children())
    for pid in started:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_startup_default_is_distinct_and_does_not_change_refresh_default(configured, monkeypatch):
    monkeypatch.delenv("MODEL_STARTUP_TIMEOUT_SECONDS")
    monkeypatch.delenv("MODEL_REFRESH_TIMEOUT_SECONDS", raising=False)
    configuration = application.read_mlflow_configuration()
    assert configuration.startup_timeout_seconds == 180
    assert configuration.refresh_timeout_seconds == 30


def test_repeated_cancellation_still_waits_for_child_reaping(configured, monkeypatch):
    monkeypatch.setenv("MODEL_STARTUP_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(application, "load_model_from_mlflow", blocked_startup_loader)
    stopping, reaped = Event(), Event()
    original_stop = application._stop_loader_process

    def observe_stop(process):
        stopping.set()
        time.sleep(0.05)
        original_stop(process)
        reaped.set()

    monkeypatch.setattr(application, "_stop_loader_process", observe_stop)

    async def cancel_twice():
        context = application.model_lifespan(application.create_app())
        task = asyncio.create_task(context.__aenter__())
        await asyncio.sleep(0.05)
        task.cancel()
        deadline = time.monotonic() + 1
        while not stopping.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.001)
        assert stopping.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert reaped.is_set()

    asyncio.run(cancel_twice())
