"""An OS error while spawning the worker is a 503, never a 500, and leaves no residue."""

from __future__ import annotations

import errno
from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import AUTH, make_settings, ref_body, sample_request
from fastapi.testclient import TestClient

from invoiceops_model_api import inference
from invoiceops_model_api.app import create_app
from invoiceops_model_api.inference import InferenceSupervisor, InferenceUnavailable


def eagain(*_args, **_kwargs):
    raise OSError(errno.EAGAIN, "Resource temporarily unavailable")


@pytest.fixture()
def cache(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture()
def client(store, cache: Path) -> Iterator[TestClient]:
    settings = make_settings(store.uri)
    supervisor = InferenceSupervisor(settings, cache_dir=cache)
    yield TestClient(create_app(settings, supervisor=supervisor))
    supervisor.shutdown()


def staged_files(cache: Path) -> list[str]:
    return sorted(p.name for p in cache.glob("*.skops")) if cache.exists() else []


def test_a_failed_spawn_while_activating_is_503_and_removes_the_staged_file(
    client: TestClient, versions, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(inference, "spawn", eagain)
    response = client.put("/deployments/a", json=ref_body(versions, "good"), headers=AUTH)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "INFERENCE_UNAVAILABLE"
    assert staged_files(cache) == []
    assert client.app.state.supervisor._slots == {}
    assert client.get("/deployments", headers=AUTH).json()["loaded"] == 0


def test_a_failed_candidate_spawn_keeps_the_serving_deployment_and_cleans_the_new_one(
    client: TestClient, versions, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = client.put("/deployments/a", json=ref_body(versions, "good"), headers=AUTH)
    assert first.status_code == 200
    kept = staged_files(cache)
    assert len(kept) == 1
    monkeypatch.setattr(inference, "spawn", eagain)
    second = client.put("/deployments/b", json=ref_body(versions, "good"), headers=AUTH)
    assert second.status_code == 503
    assert staged_files(cache) == kept
    assert len(client.app.state.supervisor._slots) == 1
    monkeypatch.undo()
    served = client.post("/deployments/a/predict", json=sample_request(), headers=AUTH)
    assert served.status_code == 200


def test_activate_forgets_the_slot_when_spawn_fails(
    store, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = InferenceSupervisor(make_settings(store.uri), cache_dir=cache)
    source = cache.parent / "model.skops"
    source.write_bytes(b"x")
    slot, staged = supervisor.stage(source)
    monkeypatch.setattr(inference, "spawn", eagain)
    with pytest.raises(InferenceUnavailable):
        supervisor.activate(slot, staged, ["a"])
    assert slot not in supervisor._slots
    supervisor.shutdown()


def test_a_failed_spawn_while_predicting_is_503_then_the_next_request_recovers(
    client: TestClient, versions, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (
        client.put("/deployments/a", json=ref_body(versions, "good"), headers=AUTH).status_code
        == 200
    )
    client.app.state.supervisor._kill_worker()  # the worker is gone: the next request respawns it
    with monkeypatch.context() as patched:
        patched.setattr(inference, "spawn", eagain)
        response = client.post("/deployments/a/predict", json=sample_request(), headers=AUTH)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "INFERENCE_UNAVAILABLE"
    recovered = client.post("/deployments/a/predict", json=sample_request(), headers=AUTH)
    assert recovered.status_code == 200


def test_the_staged_file_is_removed_when_the_registry_fails_unexpectedly(
    client: TestClient, versions, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("registry exploded")

    monkeypatch.setattr(client.app.state.registry, "put", boom)
    unsafe = TestClient(client.app, raise_server_exceptions=False)
    response = unsafe.put("/deployments/a", json=ref_body(versions, "good"), headers=AUTH)
    assert response.status_code == 500
    assert staged_files(cache) == []
    assert client.app.state.supervisor._slots == {}


def test_a_partial_copy_is_removed_when_staging_fails(
    store, cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = InferenceSupervisor(make_settings(store.uri), cache_dir=cache)
    source = tmp_path / "model.skops"
    source.write_bytes(b"x")

    def full_disk(src, dst):
        Path(dst).write_bytes(b"par")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(inference.shutil, "copyfile", full_disk)
    with pytest.raises(OSError):
        supervisor.stage(source)
    assert staged_files(cache) == []
    supervisor.shutdown()
