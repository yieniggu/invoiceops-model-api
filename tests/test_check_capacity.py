"""Deploy-check capacity: bounded waits, fail-closed storage errors, a separate pool."""

from __future__ import annotations

import errno
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import AUTH, make_settings, ref_body
from fastapi.testclient import TestClient

from invoiceops_model_api import isolation
from invoiceops_model_api.app import create_app
from invoiceops_model_api.isolation import CheckBusy, CheckRunner
from invoiceops_model_api.settings import Settings

HELPERS = Path(__file__).parent / "helpers"


def slow_runner(settings: Settings) -> CheckRunner:
    return CheckRunner(settings, entrypoint=(sys.executable, "-I", str(HELPERS / "slow_child.py")))


def test_three_concurrent_checks_with_one_slot_give_a_bounded_busy(tmp_path: Path) -> None:
    settings = make_settings(
        "http://127.0.0.1:1",
        max_concurrent_checks=1,
        check_wait_seconds=0.4,
        check_timeout_seconds=2.0,
    )
    runner = slow_runner(settings)
    matrix, results = np.zeros((2, 3)), []

    def attempt() -> None:
        try:
            outcome = runner.run(tmp_path / "m.skops", matrix, ["a", "b", "c"])
            results.append(("ran", outcome.code, time.monotonic()))
        except CheckBusy:
            results.append(("busy", "", time.monotonic()))

    threads = [threading.Thread(target=attempt) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    busy = [r for r in results if r[0] == "busy"]
    ran = [r for r in results if r[0] == "ran"]
    assert len(busy) == 2 and len(ran) == 1, results
    # No absolute bounds. "Bounded by check_wait_seconds, not by the check" means the
    # refused callers return BEFORE the running check ends (it only ends at its 2 s timeout).
    assert all(r[2] < ran[0][2] for r in busy), results
    assert ran[0][1] == "MODEL_CHECK_TIMEOUT"
    assert not any(thread.is_alive() for thread in threads)


def test_a_released_slot_can_be_taken_again(tmp_path: Path) -> None:
    settings = make_settings("http://127.0.0.1:1", max_concurrent_checks=1, check_wait_seconds=0.2)
    runner = CheckRunner(settings, entrypoint=(sys.executable, "-I", "-c", "raise SystemExit(0)"))
    for _ in range(3):  # the slot is released even when the child crashes
        outcome = runner.run(tmp_path / "m.skops", np.zeros((1, 1)), ["a"])
        assert outcome.code == "MODEL_CHECK_CRASHED"


class StubRunner:
    def __init__(self, behaviour) -> None:
        self.behaviour, self.threads = behaviour, []

    def run(self, *args, **kwargs):
        self.threads.append(threading.current_thread().name)
        return self.behaviour()


def deploy_check(store, versions, runner, **overrides) -> tuple[TestClient, object]:
    app = create_app(make_settings(store.uri, **overrides), check_runner=runner)
    return TestClient(app), app


def test_busy_checks_answer_503_busy(store, versions) -> None:
    def busy():
        raise CheckBusy

    client, app = deploy_check(store, versions, StubRunner(busy))
    response = client.post("/deploy-check", json=ref_body(versions, "good"), headers=AUTH)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "BUSY"
    app.state.supervisor.shutdown()


@pytest.mark.parametrize("number", [errno.ENOSPC, errno.EMFILE, errno.EAGAIN])
def test_storage_and_spawn_errors_fail_closed_with_503(store, versions, number) -> None:
    def broken():
        raise OSError(number, "boom")

    client, app = deploy_check(store, versions, StubRunner(broken))
    for method, path in (("post", "/deploy-check"), ("put", "/deployments/disk")):
        response = getattr(client, method)(path, json=ref_body(versions, "good"), headers=AUTH)
        assert response.status_code == 503, (path, response.text)
        assert response.json()["detail"]["code"] == "CHECK_UNAVAILABLE"
    assert client.get("/deployments", headers=AUTH).json()["loaded"] == 0  # nothing got loaded
    app.state.supervisor.shutdown()


def test_checks_run_on_their_own_pool(store, versions) -> None:
    runner = StubRunner(lambda: isolation.ChildOutcome(False, "MODEL_CHECK_CRASHED", "x"))
    client, app = deploy_check(store, versions, runner)
    client.post("/deploy-check", json=ref_body(versions, "good"), headers=AUTH)
    assert runner.threads and all(name.startswith("check") for name in runner.threads)
    app.state.supervisor.shutdown()


def test_waiting_checks_cannot_starve_predict_and_excess_gets_busy(store, versions) -> None:
    release = threading.Event()

    def hold():
        release.wait(30)
        return isolation.ChildOutcome(False, "MODEL_CHECK_CRASHED", "x")

    runner = StubRunner(hold)
    client, app = deploy_check(
        store, versions, runner, max_concurrent_checks=1, max_concurrent_downloads=1
    )
    workers = app.state.check_executor._max_workers
    statuses: list[int] = []
    body = ref_body(versions, "good")

    def fire() -> None:
        statuses.append(client.post("/deploy-check", json=body, headers=AUTH).status_code)

    # More requests than the admission bound: the surplus is refused at once (503 BUSY).
    threads = [threading.Thread(target=fire) for _ in range(workers * 2 + 4)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 20
    while len([s for s in statuses if s == 503]) < 4 and time.monotonic() < deadline:
        time.sleep(0.1)
    # While checks are blocked, the default pool still answers (predict shares it).
    assert client.get("/deployments", headers=AUTH).status_code == 200  # TestClient has no timeout
    assert len([s for s in statuses if s == 503]) >= 4
    release.set()
    for thread in threads:
        thread.join(timeout=60)
    app.state.supervisor.shutdown()
