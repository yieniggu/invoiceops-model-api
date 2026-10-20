"""Real owned children and anonymous pipes; no network, provider or Docker."""

import asyncio
import json
import multiprocessing
import os
import statistics
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Thread, Timer
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from worker_fixtures import (
    load_blocked,
    load_deserialization_hang,
    load_estimator,
    load_exiting_estimator,
    load_new_estimator,
    load_partial_then_block,
    load_pipeline,
    load_selective_estimator,
    load_slow_estimator,
    malformed_inference_worker,
    malformed_worker,
)

import invoiceops_model_api.app as application
from invoiceops_model_api import worker

ENTRY = application.MODEL_CATALOG["invoice-review"]
FEATURES = [125000, 365, 1, 1.25, True, True, False, "low"]
BODY = dict(zip(application.PredictRequest.model_fields, FEATURES, strict=True))
HEADERS = {"X-Model-Admin-Token": "test-admin-token"}


@pytest.fixture(autouse=True)
def owned_children_only():
    before = {p.pid for p in multiprocessing.active_children()}
    yield
    assert not [p.pid for p in multiprocessing.active_children() if p.pid not in before]


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://offline.invalid")
    monkeypatch.setenv("MODEL_API_ADMIN_TOKEN", "test-admin-token")
    monkeypatch.setattr(application, "load_model_from_mlflow", load_estimator)


def start(loader=load_estimator):
    return application.load_model_in_isolated_process(ENTRY, "offline", 10, loader)


def assert_gone(pid):
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_non_picklable_estimator_stays_in_child_and_proxy_preserves_interface():
    loaded = start()
    pid = loaded.model.process.pid
    try:
        assert loaded.model.classes_ == [0, 1]
        assert loaded.model.predict_proba([FEATURES]) == [[0.2, 0.8]]
        assert application.positive_class_probability(loaded.model, FEATURES) == 0.8
        assert (loaded.model_version, loaded.run_id) == ("7", "run-123")
    finally:
        loaded.model.close()
    assert_gone(pid)


@pytest.mark.parametrize(
    "mode",
    [
        "partial-header",
        "partial-body",
        "oversize",
        "corrupt",
        "deep",
        "late",
    ],
)
def test_partial_corrupt_oversize_deep_and_late_bootstrap_frames_are_bounded(monkeypatch, mode):
    monkeypatch.setattr(application, "serve_model", malformed_worker)
    before = time.monotonic()
    with pytest.raises(worker.ModelLoadFailure) as caught:
        application.load_model_in_isolated_process(
            application.ModelCatalogEntry("offline", mode), "offline", 1, load_estimator
        )
    assert caught.value.reason in ("timeout", "protocol_failure")
    assert time.monotonic() - before < 1.5


def test_partial_body_cancellation_reaps_actual_child(monkeypatch):
    monkeypatch.setattr(application, "serve_model", malformed_worker)
    cancel = Event()
    timer = Timer(0.7, cancel.set)
    timer.start()
    try:
        before = time.monotonic()
        with pytest.raises(worker.ModelLoadFailure, match="cancelled"):
            application.load_model_in_isolated_process(
                application.ModelCatalogEntry("offline", "partial-body"),
                "offline",
                10,
                load_estimator,
                cancelled=cancel,
            )
        assert time.monotonic() - before < 1.2
    finally:
        timer.cancel()
        timer.join()


def test_blocked_pipe_write_obeys_absolute_deadline():
    reader, writer_fd = os.pipe()
    os.set_blocking(writer_fd, False)
    try:
        while True:
            try:
                os.write(writer_fd, b"x" * 4096)
            except BlockingIOError:
                break
        before = time.monotonic()
        with pytest.raises(worker.ModelLoadFailure, match="timeout"):
            worker.send_frame(writer_fd, ["predict"], before + 0.05)
        assert time.monotonic() - before < 0.2
    finally:
        os.close(reader)
        os.close(writer_fd)


def test_late_already_buffered_frame_is_not_accepted():
    reader, writer_fd = os.pipe()
    os.set_blocking(reader, False)
    os.set_blocking(writer_fd, False)
    try:
        worker.send_frame(writer_fd, ["ready"], time.monotonic() + 1)
        with pytest.raises(worker.ModelLoadFailure, match="timeout"):
            worker.receive_frame(reader, time.monotonic() - 0.01)
    finally:
        os.close(reader)
        os.close(writer_fd)


def test_deserialization_hang_is_contained_in_worker():
    before = time.monotonic()
    with pytest.raises(worker.ModelLoadFailure, match="timeout"):
        application.load_model_in_isolated_process(ENTRY, "offline", 1, load_deserialization_hang)
    assert time.monotonic() - before < 1.5


def test_two_concurrent_rpcs_are_correlated_and_serialize():
    loaded = start(load_slow_estimator)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(loaded.model.predict_proba, [[0.15, 0.25, *FEATURES[2:]]])
            second = pool.submit(loaded.model.predict_proba, [[0.05, 0.75, *FEATURES[2:]]])
            assert first.result(timeout=2) == [[0.75, 0.25]]
            assert second.result(timeout=2) == [[0.25, 0.75]]
    finally:
        loaded.model.close()


def test_hung_inference_and_lock_wait_are_bounded_and_worker_is_reaped(monkeypatch):
    loaded = start(load_slow_estimator)
    pid = loaded.model.process.pid
    monkeypatch.setattr(worker, "INFERENCE_TIMEOUT_SECONDS", 0.15)
    before = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(loaded.model.predict_proba, [[30, 0.2, *FEATURES[2:]]])
                for _ in range(2)
            ]
            for future in futures:
                with pytest.raises(worker.ModelLoadFailure):
                    future.result(timeout=1)
        assert time.monotonic() - before < 0.8
        assert not loaded.model.available
    finally:
        loaded.model.close()
    assert_gone(pid)


@pytest.mark.parametrize(
    "mode", ["partial-result", "wrong-result", "corrupt-result", "late-result"]
)
def test_bad_inference_frames_expire_and_dispose_owner(monkeypatch, mode):
    monkeypatch.setattr(application, "serve_model", malformed_inference_worker)
    loaded = application.load_model_in_isolated_process(
        application.ModelCatalogEntry("offline", mode),
        "offline",
        3,
        load_estimator,
    )
    pid = loaded.model.process.pid
    monkeypatch.setattr(worker, "INFERENCE_TIMEOUT_SECONDS", 0.1)
    try:
        before = time.monotonic()
        with pytest.raises(worker.ModelLoadFailure):
            loaded.model.predict_proba([FEATURES])
        assert time.monotonic() - before < 0.6
        assert loaded.model.cleaned
    finally:
        loaded.model.close()
    assert_gone(pid)


def test_cleanup_reports_unverifiable_os_termination_without_claiming_reap():
    class UnreapableProcess:
        def __init__(self):
            self.signals = []

        def is_alive(self):
            return True

        def terminate(self):
            self.signals.append("terminate")

        def kill(self):
            self.signals.append("kill")

        def join(self, timeout):
            assert timeout == 0.1

        def close(self):
            pytest.fail("Must not claim an unreaped OS process is closed")

    process = UnreapableProcess()
    with pytest.raises(worker.ModelLoadFailure, match="cleanup_unverified"):
        worker.stop_process(process)
    assert process.signals == ["terminate", "kill"]


def test_worker_exit_does_not_leave_health_ready(configured, monkeypatch):
    monkeypatch.setattr(application, "load_model_from_mlflow", load_exiting_estimator)
    with TestClient(application.create_app()) as client:
        assert client.get("/models/invoice-review/health").status_code == 200
        response = client.post("/models/invoice-review/predict", json=BODY)
        assert response.status_code == 503
        assert client.get("/models/invoice-review/health").status_code == 503


def test_actual_worker_http_startup_refresh_and_predict(configured, monkeypatch):
    with TestClient(application.create_app()) as client:
        assert client.post("/models/invoice-review/predict", json=BODY).json() == {
            "model_id": "invoice-review",
            "model_version": "7",
            "run_id": "run-123",
            "probability": 0.8,
        }
        monkeypatch.setattr(application, "load_model_from_mlflow", load_new_estimator)
        assert (
            client.post("/admin/models/invoice-review/refresh", headers=HEADERS).status_code == 200
        )
        response = client.post("/models/invoice-review/predict", json=BODY)
        assert response.status_code == 200
        assert response.json()["run_id"] == "run-456"


def test_failed_refresh_preserves_real_worker_inference(configured, monkeypatch):
    monkeypatch.setenv("MODEL_REFRESH_TIMEOUT_SECONDS", "0.1")
    with TestClient(application.create_app()) as client:
        monkeypatch.setattr(application, "load_model_from_mlflow", load_blocked)
        assert (
            client.post("/admin/models/invoice-review/refresh", headers=HEADERS).status_code == 503
        )
        response = client.post("/models/invoice-review/predict", json=BODY)
        assert response.status_code == 200
        assert response.json()["run_id"] == "run-123"
        assert response.json()["probability"] == 0.8


def test_concurrent_refresh_rejects_without_queueing_or_spawning(configured):
    app = application.create_app()
    with TestClient(app) as client:
        with app.state.model_refresh_lock:
            before = time.monotonic()
            assert (
                client.post("/admin/models/invoice-review/refresh", headers=HEADERS).status_code
                == 503
            )
            assert time.monotonic() - before < 0.5
        assert len(app.state.owned_workers) == 1
        assert client.post("/models/invoice-review/predict", json=BODY).status_code == 200


def test_oversize_input_does_not_kill_healthy_worker_or_change_http_schema(configured):
    with TestClient(application.create_app()) as client:
        response = client.post(
            "/models/invoice-review/predict", json={**BODY, "country_risk": "x" * 2049}
        )
        assert response.status_code == 503
        assert response.json() == {"detail": "Model inference is unavailable"}
        assert client.get("/models/invoice-review/health").status_code == 200
        assert client.post("/models/invoice-review/predict", json=BODY).json()["probability"] == 0.8


def test_completed_estimator_error_is_sanitized_and_preserves_worker(
    configured, monkeypatch, caplog
):
    monkeypatch.setattr(application, "load_model_from_mlflow", load_selective_estimator)
    with TestClient(application.create_app()) as client:
        response = client.post(
            "/models/invoice-review/predict", json={**BODY, "country_risk": "unknown"}
        )
        assert response.status_code == 503
        assert "private-provider-password" not in response.text + caplog.text
        assert client.get("/models/invoice-review/health").status_code == 200
        assert client.post("/models/invoice-review/predict", json=BODY).json()["probability"] == 0.8


def test_health_rejects_worker_that_exits_without_an_inference_request(configured):
    app = application.create_app()
    with TestClient(app) as client:
        proxy = app.state.model_registry["invoice-review"].model
        proxy.process.kill()
        proxy.process.join(timeout=0.5)
        assert client.get("/models/invoice-review/health").status_code == 503


def test_snapshot_lease_before_rpc_survives_refresh(configured, monkeypatch):
    app = application.create_app()
    request = SimpleNamespace(app=app)
    with TestClient(app) as client:
        with application.prediction_snapshot(request, "invoice-review") as old:
            old_pid = old.model.process.pid
            monkeypatch.setattr(application, "load_model_from_mlflow", load_new_estimator)
            assert (
                client.post("/admin/models/invoice-review/refresh", headers=HEADERS).status_code
                == 200
            )
            assert old.model.available
            assert application.positive_class_probability(old.model, FEATURES) == 0.8
            assert old.run_id == "run-123"
            assert client.get("/models/invoice-review/health").json()["run_id"] == "run-456"
        assert_gone(old_pid)


def test_inflight_prediction_keeps_old_metadata_during_refresh(configured, monkeypatch):
    monkeypatch.setattr(application, "load_model_from_mlflow", load_slow_estimator)
    app = application.create_app()
    with TestClient(app) as client:
        replies = []
        thread = Thread(
            target=lambda: replies.append(
                client.post(
                    "/models/invoice-review/predict",
                    json={**BODY, "invoice_amount_cents": 1, "vendor_tenure_days": 0},
                )
            )
        )
        thread.start()
        try:
            deadline = time.monotonic() + 1
            while not app.state.model_registry["invoice-review"].model._leases:
                assert time.monotonic() < deadline
                time.sleep(0.005)
            monkeypatch.setattr(application, "load_model_from_mlflow", load_new_estimator)
            assert (
                client.post("/admin/models/invoice-review/refresh", headers=HEADERS).status_code
                == 200
            )
        finally:
            thread.join(timeout=2)
        assert not thread.is_alive()
        assert replies[0].status_code == 200
        assert replies[0].json()["run_id"] == "run-123"
        assert (
            client.post("/models/invoice-review/predict", json=BODY).json()["run_id"] == "run-456"
        )


def test_retirement_deadline_and_shutdown_reap_leased_workers(configured, monkeypatch):
    monkeypatch.setattr(worker, "RETIRE_TIMEOUT_SECONDS", 0.1)
    app = application.create_app()
    with TestClient(app) as client:
        snapshot = application.prediction_snapshot(SimpleNamespace(app=app), "invoice-review")
        old = snapshot.__enter__()
        pid = old.model.process.pid
        monkeypatch.setattr(application, "load_model_from_mlflow", load_new_estimator)
        assert (
            client.post("/admin/models/invoice-review/refresh", headers=HEADERS).status_code == 200
        )
        deadline = time.monotonic() + 1
        while not old.model.cleaned:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert_gone(pid)
        snapshot.__exit__(None, None, None)
        # A second retired lease is forcibly reclaimed by shutdown, not GC.
        with application.prediction_snapshot(SimpleNamespace(app=app), "invoice-review") as current:
            current_pid = current.model.process.pid
        assert (
            client.post("/admin/models/invoice-review/refresh", headers=HEADERS).status_code == 200
        )
    assert_gone(current_pid)


def test_repeated_refresh_bounds_owned_workers_and_shutdown(configured, monkeypatch):
    app = application.create_app()
    snapshots = []
    with TestClient(app) as client:
        try:
            for _ in range(2):
                context = application.prediction_snapshot(
                    SimpleNamespace(app=app), "invoice-review"
                )
                context.__enter__()
                snapshots.append(context)
                assert (
                    client.post("/admin/models/invoice-review/refresh", headers=HEADERS).status_code
                    == 200
                )
            # At most three live owners; another candidate is rejected without spawning.
            assert (
                client.post("/admin/models/invoice-review/refresh", headers=HEADERS).status_code
                == 503
            )
            assert len(app.state.owned_workers) == 3
        finally:
            for context in snapshots:
                context.__exit__(None, None, None)
        assert (
            client.post("/admin/models/invoice-review/refresh", headers=HEADERS).status_code == 200
        )
    assert all(owner.cleaned for owner in app.state.owned_workers)


def test_shutdown_closes_current_and_retired_workers_with_outstanding_leases(configured):
    app = application.create_app()
    contexts, pids = [], []
    try:
        with TestClient(app) as client:
            for _ in range(2):
                context = application.prediction_snapshot(
                    SimpleNamespace(app=app), "invoice-review"
                )
                loaded = context.__enter__()
                contexts.append(context)
                pids.append(loaded.model.process.pid)
                assert (
                    client.post("/admin/models/invoice-review/refresh", headers=HEADERS).status_code
                    == 200
                )
            current = app.state.model_registry["invoice-review"]
            pids.append(current.model.process.pid)
        for pid in pids:
            assert_gone(pid)
    finally:
        for context in contexts:
            context.__exit__(None, None, None)


def test_shutdown_cancels_candidate_refresh_and_reaps_before_return(configured, monkeypatch):
    candidate_started = Event()
    process_type = application.model_load_process_context().Process
    original_start = process_type.start
    pids, replies = [], []

    def record_start(process):
        original_start(process)
        pids.append(process.pid)
        candidate_started.set()

    async def run():
        app = application.create_app()
        context = application.model_lifespan(app)
        await context.__aenter__()
        pids.append(app.state.model_registry["invoice-review"].model.process.pid)
        monkeypatch.setattr(application, "load_model_from_mlflow", load_blocked)
        monkeypatch.setattr(process_type, "start", record_start)

        def refresh():
            try:
                application.refresh_model(
                    "invoice-review",
                    SimpleNamespace(app=app, headers=HEADERS),
                )
            except application.HTTPException as error:
                replies.append(error.status_code)

        thread = Thread(target=refresh)
        thread.start()
        try:
            assert candidate_started.wait(timeout=1)
            before = time.monotonic()
            await context.__aexit__(None, None, None)
            assert time.monotonic() - before < 1
        finally:
            thread.join(timeout=1)
        assert not thread.is_alive()
        assert replies == [503]
        for pid in pids:
            assert_gone(pid)

    asyncio.run(run())


def test_shutdown_interrupts_current_inference_without_rpc_mutex_deadlock(configured, monkeypatch):
    monkeypatch.setattr(application, "load_model_from_mlflow", load_slow_estimator)
    failures = []

    async def run():
        app = application.create_app()
        context = application.model_lifespan(app)
        await context.__aenter__()
        proxy = app.state.model_registry["invoice-review"].model
        pid = proxy.process.pid

        def infer():
            try:
                proxy.predict_proba([[30, 0.2, *FEATURES[2:]]])
            except worker.ModelLoadFailure:
                failures.append(True)

        thread = Thread(target=infer)
        thread.start()
        try:
            await asyncio.sleep(0.05)
            before = time.monotonic()
            await context.__aexit__(None, None, None)
            assert time.monotonic() - before < 1
        finally:
            thread.join(timeout=1)
        assert not thread.is_alive()
        assert failures == [True]
        assert_gone(pid)

    asyncio.run(run())


def test_partial_frame_repeated_async_cancel_does_not_detach_cleanup(configured, monkeypatch):
    monkeypatch.setattr(application, "serve_model", malformed_worker)
    monkeypatch.setattr(
        application,
        "MODEL_CATALOG",
        {
            "invoice-review": application.ModelCatalogEntry("offline", "partial-body"),
        },
    )

    async def run():
        context = application.model_lifespan(application.create_app())
        task = asyncio.create_task(context.__aenter__())
        await asyncio.sleep(0.6)
        task.cancel()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


def test_cancel_partial_catalog_reaps_already_loaded_and_loading_workers(configured, monkeypatch):
    monkeypatch.setattr(application, "load_model_from_mlflow", load_partial_then_block)
    monkeypatch.setattr(
        application,
        "MODEL_CATALOG",
        {
            **application.MODEL_CATALOG,
            "second": application.ModelCatalogEntry("blocked", "champion"),
        },
    )
    process_type = application.model_load_process_context().Process
    original_start = process_type.start
    pids = []

    def record_start(process):
        original_start(process)
        pids.append(process.pid)

    monkeypatch.setattr(process_type, "start", record_start)

    async def run():
        app = application.create_app()
        context = application.model_lifespan(app)
        task = asyncio.create_task(context.__aenter__())
        deadline = time.monotonic() + 3
        while len(pids) < 2:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.01)
        assert app.state.model_registry["invoice-review"] is not None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert all(value is None for value in app.state.model_registry.values())

    asyncio.run(run())
    for pid in pids:
        assert_gone(pid)


@pytest.fixture
def cleanup_fault(monkeypatch):
    """Inject only the owned-process stop boundary; always recover probe children."""
    real_stop = application._stop_loader_process
    state = SimpleNamespace(
        enabled=False, protected=set(), processes={}, errors=[], apps=[], delay=0
    )
    state.entered = Event()

    def stop(process):
        if state.enabled and process.pid not in state.protected:
            state.processes[process.pid] = process
            error = worker.ModelLoadFailure("cleanup_unverified")
            state.errors.append(error)
            state.entered.set()
            time.sleep(state.delay)
            raise error
        real_stop(process)

    monkeypatch.setattr(application, "_stop_loader_process", stop)
    try:
        yield state
    finally:
        state.enabled = False
        owners = [owner for app in state.apps for owner in app.state.owned_workers]
        # The RED implementation loses candidates. Keep its failure tracebacks
        # solely so test teardown can recover those otherwise-unreachable pipes.
        for error in state.errors:
            for frame, _ in traceback.walk_tb(error.__traceback__):
                owner = frame.f_locals.get("self")
                if isinstance(owner, worker.ModelProxy):
                    owners.append(owner)
        unique = {id(owner): owner for owner in owners}
        for owner in unique.values():
            owner._stop = real_stop
            owner.close()
        state.errors.clear()


def test_ownership_failed_bootstrap_retained_and_capacity_blocks_fourth_attempt(
    configured,
    monkeypatch,
    cleanup_fault,
):
    app = application.create_app()
    cleanup_fault.apps.append(app)
    monkeypatch.setenv("MODEL_REFRESH_TIMEOUT_SECONDS", "0.08")
    with TestClient(app) as client:
        cleanup_fault.protected.add(app.state.model_registry["invoice-review"].model.process.pid)
        cleanup_fault.enabled = True
        monkeypatch.setattr(application, "load_model_from_mlflow", load_blocked)
        try:
            for _ in range(4):
                assert (
                    client.post("/admin/models/invoice-review/refresh", headers=HEADERS).status_code
                    == 503
                )
            # Active + two unreaped candidates fill all three slots. The third
            # and fourth refresh attempts must not create another process.
            assert len(app.state.owned_workers) == 3
            assert len(cleanup_fault.processes) == 2
            assert all(process.is_alive() for process in cleanup_fault.processes.values())
            assert (
                client.post("/models/invoice-review/predict", json=BODY).json()["run_id"]
                == "run-123"
            )
        finally:
            cleanup_fault.enabled = False
    assert all(owner.cleaned for owner in app.state.owned_workers)


def test_ownership_shutdown_retries_unpublished_candidate_before_clean_return(
    configured,
    monkeypatch,
    cleanup_fault,
):
    app = application.create_app()
    cleanup_fault.apps.append(app)
    monkeypatch.setenv("MODEL_REFRESH_TIMEOUT_SECONDS", "0.08")
    with TestClient(app) as client:
        cleanup_fault.protected.add(app.state.model_registry["invoice-review"].model.process.pid)
        cleanup_fault.enabled = True
        monkeypatch.setattr(application, "load_model_from_mlflow", load_blocked)
        assert (
            client.post("/admin/models/invoice-review/refresh", headers=HEADERS).status_code == 503
        )
        pids = list(cleanup_fault.processes)
        cleanup_fault.enabled = False
    for pid in pids:
        assert_gone(pid)
    assert all(owner.cleaned for owner in app.state.owned_workers)


def test_ownership_shutdown_reports_unverified_candidate_not_clean_success(
    configured,
    monkeypatch,
    cleanup_fault,
):
    app = application.create_app()
    cleanup_fault.apps.append(app)
    monkeypatch.setenv("MODEL_REFRESH_TIMEOUT_SECONDS", "0.08")
    client = TestClient(app)
    client.__enter__()
    try:
        cleanup_fault.protected.add(app.state.model_registry["invoice-review"].model.process.pid)
        cleanup_fault.enabled = True
        monkeypatch.setattr(application, "load_model_from_mlflow", load_blocked)
        assert (
            client.post("/admin/models/invoice-review/refresh", headers=HEADERS).status_code == 503
        )
    finally:
        with pytest.raises(worker.ModelLoadFailure, match="cleanup_unverified"):
            client.__exit__(None, None, None)
    assert any(not owner.cleaned for owner in app.state.owned_workers)
    cleanup_fault.enabled = False
    app.state.worker_pool.close()
    assert app.state.owned_workers == []


def test_ownership_startup_failure_retains_candidate_when_cleanup_cannot_confirm(
    configured,
    monkeypatch,
    cleanup_fault,
):
    app = application.create_app()
    cleanup_fault.apps.append(app)
    monkeypatch.setenv("MODEL_STARTUP_TIMEOUT_SECONDS", "0.08")
    monkeypatch.setattr(application, "load_model_from_mlflow", load_blocked)
    cleanup_fault.enabled = True
    with pytest.raises(worker.ModelLoadFailure, match="cleanup_unverified"), TestClient(app):
        pass
    assert len(app.state.owned_workers) == 1
    assert not app.state.owned_workers[0].cleaned
    assert len(cleanup_fault.processes) == 1
    cleanup_fault.enabled = False
    app.state.worker_pool.close()
    assert app.state.owned_workers == []


def test_ownership_repeated_startup_cancel_retains_failed_cleanup_candidate(
    configured,
    monkeypatch,
    cleanup_fault,
):
    app = application.create_app()
    cleanup_fault.apps.append(app)
    monkeypatch.setattr(application, "load_model_from_mlflow", load_blocked)
    cleanup_fault.enabled = True
    cleanup_fault.delay = 0.05

    async def run():
        context = application.model_lifespan(app)
        task = asyncio.create_task(context.__aenter__())
        await asyncio.sleep(0.1)
        task.cancel()
        deadline = time.monotonic() + 1
        while not cleanup_fault.entered.is_set():
            assert time.monotonic() < deadline
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(worker.ModelLoadFailure, match="cleanup_unverified"):
            await task
        assert len(app.state.owned_workers) == 1
        assert not app.state.owned_workers[0].cleaned

    asyncio.run(run())
    cleanup_fault.enabled = False
    app.state.worker_pool.close()
    assert app.state.owned_workers == []


def test_ownership_concurrent_reservations_include_unverified_candidates(
    configured,
    cleanup_fault,
):
    app = application.create_app()
    cleanup_fault.apps.append(app)
    cleanup_fault.enabled = True
    barrier = Barrier(8)

    def attempt():
        barrier.wait(timeout=2)
        try:
            application.load_model_in_isolated_process(
                ENTRY,
                "offline",
                0.1,
                load_blocked,
                owner_pool=app.state.worker_pool,
            )
        except worker.ModelLoadFailure as error:
            return error.reason
        pytest.fail("A blocked bootstrap must not succeed")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: attempt(), range(8)))
    assert results.count("worker_capacity") == 5
    assert results.count("cleanup_unverified") == 3
    assert len(app.state.owned_workers) == 3
    assert len(cleanup_fault.processes) == 3


def test_ownership_publication_and_retirement_reuse_the_reserved_owner(configured):
    app = application.create_app()
    pids = []
    with TestClient(app) as client:
        with application.prediction_snapshot(SimpleNamespace(app=app), "invoice-review") as old:
            assert app.state.owned_workers == [old.model]
            pids.append(old.model.process.pid)
            assert (
                client.post("/admin/models/invoice-review/refresh", headers=HEADERS).status_code
                == 200
            )
            current = app.state.model_registry["invoice-review"].model
            pids.append(current.process.pid)
            assert len(app.state.owned_workers) == 2
            assert {id(owner) for owner in app.state.owned_workers} == {id(old.model), id(current)}
            assert all(owner.available for owner in app.state.owned_workers)
        assert old.model.cleaned
        assert_gone(pids[0])
        assert current.available
    assert app.state.owned_workers == []
    for pid in pids:
        assert_gone(pid)


def test_ownership_cleanup_callbacks_never_hold_pool_or_publication_lock(configured, monkeypatch):
    app = application.create_app()
    real_stop = application._stop_loader_process
    stopped = []

    def stop_boundary(process):
        assert app.state.worker_pool._lock.acquire(blocking=False)
        app.state.worker_pool._lock.release()
        assert app.state.model_registry_lock.acquire(blocking=False)
        app.state.model_registry_lock.release()
        stopped.append(process.pid)
        real_stop(process)

    monkeypatch.setattr(application, "_stop_loader_process", stop_boundary)
    try:
        with TestClient(app) as client:
            assert (
                client.post("/admin/models/invoice-review/refresh", headers=HEADERS).status_code
                == 200
            )
            assert len(stopped) == 1
            assert client.post("/models/invoice-review/predict", json=BODY).status_code == 200
        assert len(stopped) == 2
        assert app.state.owned_workers == []
    finally:
        for owner in app.state.owned_workers:
            owner._stop = real_stop
        app.state.worker_pool.close()


def test_ownership_closing_between_reservation_and_launch_prevents_spawn(configured, monkeypatch):
    app = application.create_app()
    actual_context = application.model_load_process_context()
    entered, release = Event(), Event()

    def context_boundary():
        entered.set()
        assert release.wait(timeout=2)
        return actual_context

    monkeypatch.setattr(application, "model_load_process_context", context_boundary)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            application.load_model_in_isolated_process,
            ENTRY,
            "offline",
            3,
            load_blocked,
            owner_pool=app.state.worker_pool,
        )
        try:
            assert entered.wait(timeout=1)
            assert len(app.state.owned_workers) == 1
            reservation = app.state.owned_workers[0]
            assert reservation.process is None
            app.state.worker_pool.close()
            assert reservation.cleaned
            assert app.state.owned_workers == []
        finally:
            release.set()
        with pytest.raises(worker.ModelLoadFailure, match="cancelled"):
            future.result(timeout=1)
    with pytest.raises(worker.ModelLoadFailure, match="cancelled"):
        application.load_model_in_isolated_process(
            ENTRY,
            "offline",
            1,
            load_estimator,
            owner_pool=app.state.worker_pool,
        )
    assert app.state.owned_workers == []


@pytest.mark.parametrize(
    "failure", ["first_pipe", "second_pipe", "process", "start", "after_start"]
)
def test_ownership_allocation_and_spawn_errors_close_all_acquired_handles(
    configured,
    monkeypatch,
    failure,
):
    app = application.create_app()
    context = application.model_load_process_context()
    connections, pids = [], []
    calls = 0

    class AllocationBoundary:
        def Pipe(self, *, duplex):
            nonlocal calls
            calls += 1
            if failure == ("first_pipe" if calls == 1 else "second_pipe"):
                raise OSError("private-provider-password")
            pair = context.Pipe(duplex=duplex)
            connections.extend(pair)
            return pair

        def Process(self, *, target, args):
            if failure == "process":
                raise OSError("private-provider-password")
            process = context.Process(target=target, args=args)
            original_start = process.start

            def start_boundary():
                if failure == "after_start":
                    original_start()
                    pids.append(process.pid)
                raise OSError("private-provider-password")

            process.start = start_boundary
            return process

    monkeypatch.setattr(application, "model_load_process_context", AllocationBoundary)
    try:
        with pytest.raises(worker.ModelLoadFailure) as caught:
            application.load_model_in_isolated_process(
                ENTRY,
                "offline",
                1,
                load_blocked,
                owner_pool=app.state.worker_pool,
            )
        assert caught.value.reason in ("child_exit", "process_start_failure")
        assert "private-provider-password" not in str(caught.value)
        assert connections == [] or all(connection.closed for connection in connections)
        assert all(owner.cleaned for owner in app.state.owned_workers)
        app.state.worker_pool.close()
        assert app.state.owned_workers == []
        for pid in pids:
            assert_gone(pid)
    finally:
        app.state.worker_pool.close()


def test_pipeline_output_and_offline_cost_measurements(capsys):
    # RSS is sampled with local ps, not summed PSS: shared pages may be counted twice.
    reference_started = time.monotonic()
    reference = load_pipeline().model
    reference_load = time.monotonic() - reference_started
    before_rss = int(
        subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        ).strip()
    )
    started = time.monotonic()
    loaded = start(load_pipeline)
    startup = time.monotonic() - started
    try:
        expected = reference.predict_proba([FEATURES]).tolist()
        assert loaded.model.predict_proba([FEATURES]) == expected
        serial, direct = [], []
        for _ in range(20):
            before = time.monotonic()
            reference.predict_proba([FEATURES])
            direct.append(time.monotonic() - before)
            before = time.monotonic()
            assert loaded.model.predict_proba([FEATURES]) == expected
            serial.append(time.monotonic() - before)
        before = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: loaded.model.predict_proba([FEATURES]), range(20)))
        concurrent = time.monotonic() - before
        assert results == [expected] * 20
        child_rss = int(
            subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(loaded.model.process.pid)],
                text=True,
            ).strip()
        )
        after_rss = int(
            subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                text=True,
            ).strip()
        )
        with capsys.disabled():
            print(
                "OFFLINE_WORKER_COST "
                + json.dumps(
                    {
                        "fixture": "64-row positional HGB pipeline; 20 warm calls; four callers",
                        "reference_load_s": reference_load,
                        "worker_startup_s": startup,
                        "reference_median_ms": statistics.median(direct) * 1000,
                        "proxy_median_ms": statistics.median(serial) * 1000,
                        "proxy_max_ms": max(serial) * 1000,
                        "concurrent_20_s": concurrent,
                        "parent_rss_before_kib": before_rss,
                        "parent_rss_after_kib": after_rss,
                        "child_rss_kib": child_rss,
                    },
                    sort_keys=True,
                )
            )
    finally:
        loaded.model.close()
