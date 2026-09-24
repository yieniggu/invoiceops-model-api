import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import invoiceops_model_api.app as application


def test_mlflow_loader_loads_resolved_version_when_alias_changes_and_retains_metadata(
    monkeypatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    loaded_model = object()

    class FakeMlflowClient:
        def __init__(self, *, tracking_uri: str) -> None:
            calls.append(("client", tracking_uri))

        def get_model_version_by_alias(self, name: str, alias: str) -> object:
            calls.append(("alias", name, alias))
            # The alias changes after resolution; the loader must retain version 7.
            calls.append(("alias_moved", name, alias, "8"))
            return SimpleNamespace(version="7", run_id="run-123")

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: calls.append(("tracking_uri", uri)),
        MlflowClient=FakeMlflowClient,
        pyfunc=SimpleNamespace(
            load_model=lambda uri: calls.append(("load", uri)) or loaded_model,
        ),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    loaded = application.load_model_from_mlflow(
        application.MODEL_CATALOG["invoice-review"],
        "https://mlflow.example.test",
    )

    assert calls == [
        ("tracking_uri", "https://mlflow.example.test"),
        ("client", "https://mlflow.example.test"),
        ("alias", "invoice-review-production", "champion"),
        ("alias_moved", "invoice-review-production", "champion", "8"),
        ("load", "models:/invoice-review-production/7"),
    ]
    assert loaded.model is loaded_model
    assert loaded.model_version == "7"
    assert loaded.run_id == "run-123"


def test_lifespan_loads_each_model_once_and_requests_do_not_call_mlflow(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    calls: list[tuple[str, str]] = []

    def fake_loader(entry: object, tracking_uri: str) -> object:
        calls.append((entry.registered_model, tracking_uri))
        return SimpleNamespace(model=object(), model_version="7", run_id="run-123")

    monkeypatch.setattr(application, "load_model_from_mlflow", fake_loader)

    with TestClient(application.create_app()) as client:
        assert client.get("/models/invoice-review/health").status_code == 503
        assert client.get("/models/invoice-review/health").status_code == 503
        assert client.get("/models").json() == {"models": ["invoice-review"]}

    assert calls == [("invoice-review-production", "https://mlflow.example.test")]


@pytest.mark.parametrize("tracking_uri", [None, "", "   "])
def test_missing_or_blank_tracking_uri_fails_closed_at_startup_without_provider_call(
    monkeypatch, tracking_uri: str | None
) -> None:
    if tracking_uri is None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    else:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    calls: list[object] = []
    monkeypatch.setattr(application, "load_model_from_mlflow", lambda *_: calls.append(None))

    with (
        pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI must be configured"),
        TestClient(application.create_app()),
    ):
        pass

    assert calls == []


def test_partial_tracking_credentials_fail_closed_at_startup_without_provider_call(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "operator")
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    calls: list[object] = []
    monkeypatch.setattr(application, "load_model_from_mlflow", lambda *_: calls.append(None))

    with (
        pytest.raises(
            RuntimeError, match="MLflow tracking credentials must be configured together"
        ),
        TestClient(application.create_app()),
    ):
        pass

    assert calls == []


def test_loader_failure_degrades_only_the_affected_model_without_leaking_error(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.setattr(
        application,
        "MODEL_CATALOG",
        {
            "invoice-review": application.MODEL_CATALOG["invoice-review"],
            "healthy": application.ModelCatalogEntry(
                registered_model="healthy-production",
                alias="champion",
            ),
        },
    )

    def fake_loader(entry: object, _: str) -> object:
        if entry.registered_model == "invoice-review-production":
            raise RuntimeError("credential=not-for-clients")
        return SimpleNamespace(model=object(), model_version="8", run_id="run-456")

    monkeypatch.setattr(application, "load_model_from_mlflow", fake_loader)

    with TestClient(application.create_app()) as client:
        response = client.get("/models/invoice-review/health")
        assert client.get("/models/healthy/health").status_code == 503
        assert client.get("/health").json() == {"status": "ok"}

    assert response.status_code == 503
    assert "credential=not-for-clients" not in response.text


def test_incomplete_mlflow_metadata_degrades_registered_model(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")

    class FakeMlflowClient:
        def __init__(self, *, tracking_uri: str) -> None:
            pass

        def get_model_version_by_alias(self, name: str, alias: str) -> object:
            return SimpleNamespace(version="7", run_id=None)

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda _: None,
        MlflowClient=FakeMlflowClient,
        pyfunc=SimpleNamespace(load_model=lambda _: object()),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    with TestClient(application.create_app()) as client:
        response = client.get("/models/invoice-review/health")

    assert response.status_code == 503
    assert response.json()["detail"]["message"] == "Model loader is unavailable"


def test_loaded_model_routes_remain_unavailable_before_api_04_and_api_05(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.setattr(
        application,
        "load_model_from_mlflow",
        lambda *_: SimpleNamespace(model=object(), model_version="7", run_id="run-123"),
    )

    with TestClient(application.create_app()) as client:
        health_response = client.get("/models/invoice-review/health")
        predict_response = client.post(
            "/models/invoice-review/predict",
            json={"payload": {"invoice_id": "inv-1"}},
        )

    assert health_response.status_code == 503
    assert health_response.json() == {
        "detail": "Model health is unavailable until API-04"
    }
    assert predict_response.status_code == 503
    assert predict_response.json() == {"detail": "Inference is unavailable until API-05"}
