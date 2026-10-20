"""Capture real supervisor descriptors, including pre-target provider imports."""

import multiprocessing
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor

import pytest
from worker_fixtures import load_blocked, load_estimator

import invoiceops_model_api.app as application
from invoiceops_model_api import worker, worker_launch


def load_with_parent_liveness(*args):
    assert multiprocessing.parent_process().is_alive()
    return load_estimator(*args)


def open_descriptors():
    # Listing opens a transient directory FD; fstat removes that closed entry.
    result = set()
    for entry in os.listdir("/dev/fd"):
        try:
            os.fstat(int(entry))
        except OSError:
            continue
        result.add(int(entry))
    return result


@pytest.mark.parametrize("failure", ["sink", "exec", "first_launch_pipe", "second_launch_pipe"])
def test_quiet_launcher_failures_close_resources_and_fail_closed(monkeypatch, failure):
    # Warm only stdlib's persistent tracker before comparing owned descriptors.
    worker_launch.resource_tracker.getfd()
    before = open_descriptors()
    pool = worker.WorkerPool(1)
    actual_open = os.open
    actual_pipe = os.pipe
    calls = 0

    def open_boundary(path, *args, **kwargs):
        if path == os.devnull:
            raise OSError("SYNTHETIC_SINK_FAILURE")
        return actual_open(path, *args, **kwargs)

    def exec_boundary(*args, **kwargs):
        raise OSError("SYNTHETIC_EXEC_FAILURE")

    def pipe_boundary():
        nonlocal calls
        calls += 1
        # Two application transport pipes precede the two launcher pipes.
        if calls == (3 if failure == "first_launch_pipe" else 4):
            raise OSError("SYNTHETIC_PIPE_FAILURE")
        return actual_pipe()

    if failure == "sink":
        monkeypatch.setattr(os, "open", open_boundary)
    elif failure == "exec":
        monkeypatch.setattr(subprocess.Popen, "_execute_child", exec_boundary)
    else:
        monkeypatch.setattr(os, "pipe", pipe_boundary)
    try:
        for _ in range(3):
            calls = 0
            with pytest.raises(worker.ModelLoadFailure, match="process_start_failure"):
                application.load_model_in_isolated_process(
                    application.MODEL_CATALOG["invoice-review"],
                    "offline",
                    3,
                    load_estimator,
                    owner_pool=pool,
                )
            assert all(owner.cleaned for owner in pool.workers)
            assert open_descriptors() == before
    finally:
        pool.close()
    assert pool.workers == []


def test_concurrent_quiet_workers_and_cancel_reap_without_descriptor_growth():
    from threading import Event, Timer

    worker_launch.resource_tracker.getfd()
    before = open_descriptors()
    pool = worker.WorkerPool(3)
    entry = application.MODEL_CATALOG["invoice-review"]
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    application.load_model_in_isolated_process,
                    entry,
                    "offline",
                    3,
                    load_with_parent_liveness,
                    owner_pool=pool,
                )
                for _ in range(3)
            ]
            loaded = [future.result(timeout=5) for future in futures]
        pids = [item.model.process.pid for item in loaded]
        assert len(set(pids)) == 3
        for item in loaded:
            assert item.model.predict_proba([[0] * 8]) == [[0.2, 0.8]]
            item.model.close()
        cancelled = Event()
        timer = Timer(0.1, cancelled.set)
        timer.start()
        try:
            with pytest.raises(worker.ModelLoadFailure, match="cancelled"):
                application.load_model_in_isolated_process(
                    entry, "offline", 3, load_blocked, cancelled=cancelled, owner_pool=pool
                )
        finally:
            timer.cancel()
            timer.join()
    finally:
        pool.close()
    assert not pool.workers
    assert open_descriptors() == before
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_post_spawn_bootstrap_failure_retains_owner_until_reaping_is_confirmed(monkeypatch):
    worker_launch.resource_tracker.getfd()
    before = open_descriptors()
    pool = worker.WorkerPool(1)
    actual_stop = application._stop_loader_process

    def write_boundary(*args, **kwargs):
        raise OSError("SYNTHETIC_BOOTSTRAP_WRITE_FAILURE")

    def stop_boundary(process):
        raise worker.ModelLoadFailure("cleanup_unverified")

    monkeypatch.setattr(worker_launch, "open", write_boundary, raising=False)
    monkeypatch.setattr(application, "_stop_loader_process", stop_boundary)
    try:
        with pytest.raises(worker.ModelLoadFailure, match="cleanup_unverified"):
            application.load_model_in_isolated_process(
                application.MODEL_CATALOG["invoice-review"],
                "offline",
                3,
                load_estimator,
                owner_pool=pool,
            )
        assert len(pool.workers) == 1
        owner = pool.workers[0]
        pid = owner.process.pid
        assert pid is not None
        assert not owner.cleaned
        with pytest.raises(worker.ModelLoadFailure, match="worker_capacity"):
            application.load_model_in_isolated_process(
                application.MODEL_CATALOG["invoice-review"],
                "offline",
                3,
                load_estimator,
                owner_pool=pool,
            )
    finally:
        for owner in pool.workers:
            owner._stop = actual_stop
        pool.close()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert open_descriptors() == before


@pytest.mark.parametrize(
    "mode", ["success", "load_error", "load_exit", "predict_error", "predict_exit", "import_error"]
)
def test_provider_output_never_reaches_supervisor_descriptors(tmp_path, mode):
    provider = tmp_path / "output_provider.py"
    # Only the child import emits: importing this picklable fixture in the
    # supervisor cannot create a false positive in the outer capture.
    provider.write_text(
        textwrap.dedent(
            f"""
            import atexit, logging, os, sys
            from invoiceops_model_api.app import LoadedModel
            def noise(stage):
                value = 'SYNTHETIC_OUTPUT_CANARY_' + stage
                print(value, flush=True)
                print(value, file=sys.stderr, flush=True)
                sys.__stdout__.write(value + '\\n'); sys.__stdout__.flush()
                sys.__stderr__.write(value + '\\n'); sys.__stderr__.flush()
                os.write(1, (value + '\\n').encode())
                os.write(2, (value + '\\n').encode())
                logging.warning(value)
                if {mode!r} == 'success':
                    os.write(1, b'SYNTHETIC_OUTPUT_CANARY_volume' * 16384)
                    os.write(2, b'SYNTHETIC_OUTPUT_CANARY_volume' * 16384)
            def late_output():
                noise('atexit')
                with open({str(tmp_path / "exit-marker")!r}, 'w') as marker:
                    marker.write('completed')
            def finish():
                return
            if os.getpid() != PARENT_PID:
                noise('import')
                atexit.register(late_output)
                if {mode!r} == 'import_error':
                    raise RuntimeError('SYNTHETIC_OUTPUT_CANARY_import_failure')
            class Estimator:
                classes_ = [0, 1]
                def __reduce__(self):
                    raise AssertionError('Estimator must stay in child')
                def predict_proba(self, rows):
                    noise('predict')
                    if {mode!r} == 'predict_error':
                        raise ValueError('SYNTHETIC_OUTPUT_CANARY_predict_failure')
                    if {mode!r} == 'predict_exit':
                        raise SystemExit('SYNTHETIC_OUTPUT_CANARY_predict_exit')
                    return [[0.2, 0.8]]
            def load(*args):
                noise('load')
                if {mode!r} == 'load_error':
                    raise ValueError('SYNTHETIC_OUTPUT_CANARY_load_failure')
                if {mode!r} == 'load_exit':
                    raise SystemExit('SYNTHETIC_OUTPUT_CANARY_load_exit')
                return LoadedModel(Estimator(), '7', 'run-123')
            """
        )
    )
    code = textwrap.dedent(
        f"""
        import logging, os, sys, threading, time
        sys.path.insert(0, {str(tmp_path)!r})
        from pathlib import Path
        provider = Path({str(provider)!r})
        provider.write_text(provider.read_text().replace('PARENT_PID', str(os.getpid())))
        import output_provider
        import invoiceops_model_api.app as app
        from fastapi.testclient import TestClient
        from invoiceops_model_api.worker import ModelLoadFailure
        logging.basicConfig(level=logging.INFO)
        stop = threading.Event()
        counts = [0]
        def controls():
            while not stop.is_set():
                i = counts[0]
                os.write(1, f'PARENT_OUT_{{i}}\\n'.encode())
                os.write(2, f'PARENT_ERR_{{i}}\\n'.encode())
                counts[0] += 1
                time.sleep(0.001)
        thread = threading.Thread(target=controls)
        thread.start()
        os.environ['MLFLOW_TRACKING_URI'] = 'offline'
        os.environ['MODEL_API_ADMIN_TOKEN'] = 'synthetic-admin'
        app.load_model_from_mlflow = output_provider.load
        service = app.create_app()
        try:
            with TestClient(service) as client:
                assert client.get('/health').status_code == 200
                health = client.get('/models/invoice-review/health')
                if {mode!r} in ('load_error', 'load_exit', 'import_error'):
                    assert health.status_code == 503
                    assert service.state.model_registry['invoice-review'] is None
                else:
                    assert health.status_code == 200
                    assert health.json()['run_id'] == 'run-123'
                    assert health.json()['model_version'] == '7'
                    assert service.state.model_registry['invoice-review'].model.classes_ == [0, 1]
                    body = dict(zip(app.PredictRequest.model_fields,
                        [125000, 365, 1, 1.25, True, True, False, 'low'], strict=True))
                    reply = client.post('/models/invoice-review/predict', json=body)
                    if {mode!r} == 'success':
                        assert reply.status_code == 200
                        assert reply.json()['probability'] == 0.8
                        refreshed = client.post('/admin/models/invoice-review/refresh',
                            headers={{'X-Model-Admin-Token': 'synthetic-admin'}})
                        assert refreshed.status_code == 200
                        assert refreshed.json()['run_id'] == 'run-123'
                    else:
                        assert reply.status_code == 503
                        assert reply.json() == {{'detail': 'Model inference is unavailable'}}
                    assert 'SYNTHETIC_OUTPUT_CANARY' not in reply.text
            assert not service.state.owned_workers
            if {mode!r} == 'load_error':
                try:
                    app.load_model_in_isolated_process(app.MODEL_CATALOG['invoice-review'],
                        'offline', 3, output_provider.load)
                except ModelLoadFailure as error:
                    assert error.reason == 'load_failure'
                    assert str(error) == 'load_failure'
                else:
                    raise AssertionError('Failing provider must not publish')
            if {mode!r} == 'success':
                child = app.model_load_process_context().Process(target=output_provider.finish)
                child.start()
                try:
                    child.join(3)
                    assert child.exitcode == 0
                    assert Path({str(tmp_path / "exit-marker")!r}).read_text() == 'completed'
                finally:
                    if child.is_alive():
                        child.kill()
                        child.join(1)
                    child.close()
        finally:
            stop.set()
            thread.join()
        print('CONTROL_COUNT', counts[0], flush=True)
        """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    count = int(result.stdout.rsplit("CONTROL_COUNT ", 1)[1].strip())
    assert count > 0
    for i in range(count):
        assert f"PARENT_OUT_{i}\n" in result.stdout
        assert f"PARENT_ERR_{i}\n" in result.stderr
    assert "model_startup_attempt" in result.stderr
    assert "SYNTHETIC_OUTPUT_CANARY" not in result.stdout + result.stderr
