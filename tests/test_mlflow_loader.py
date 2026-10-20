import sys
from base64 import b64encode
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import invoiceops_model_api.app as application

LOCAL_TREE_PREDICTOR = "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"


@pytest.fixture(autouse=True)
def local_loader_executor(monkeypatch):
    """Keep provider-boundary stubs local; real spawn/lifespan is tested separately."""

    def execute(entry, uri, timeout, loader, *, cancelled=None, owner_pool=None):
        assert timeout > 0
        try:
            return loader(entry, uri)
        except Exception as error:
            raise application.ModelLoadFailure("load_failure") from error

    monkeypatch.setattr(application, "load_model_in_isolated_process", execute)


def proxy_loader_fixture(
    monkeypatch, *, artifact_uri="mlflow-artifacts:/1/run-123/artifacts", files=None
):
    calls = []
    files = (
        files
        if files is not None
        else [
            SimpleNamespace(path="model/MLmodel", is_dir=False),
            SimpleNamespace(path="model/model.skops", is_dir=False),
        ]
    )
    model = object()

    class Client:
        def __init__(self, *, tracking_uri):
            calls.append(("client", tracking_uri))

        def get_model_version_by_alias(self, name, alias):
            return SimpleNamespace(version="7", run_id="run-123")

        def get_run(self, run_id):
            calls.append(("run", run_id))
            return SimpleNamespace(info=SimpleNamespace(artifact_uri=artifact_uri))

        def list_artifacts(self, run_id, path):
            calls.append(("list", run_id, path))
            return files if path == "model" else []

        def download_artifacts(self, *args):
            raise AssertionError("SDK artifact download must not run")

    def load(uri):
        path = Path(uri)
        assert path.is_dir()
        assert (path / "MLmodel").read_bytes() == b"manifest"
        assert (path / "model.skops").read_bytes() == b"model"
        calls.append(("load", uri))
        return model

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(
            set_tracking_uri=lambda uri: calls.append(("tracking", uri)),
            MlflowClient=Client,
            sklearn=SimpleNamespace(load_model=load),
        ),
    )
    monkeypatch.setenv("MODEL_API_ARTIFACT_DOWNLOAD_MODE", "proxy-http")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "local-user")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "local-password")
    return calls, model


def fake_skops(monkeypatch, calls, model, untrusted=None):
    def inspect(*, file):
        assert Path(file).read_bytes() == b"model"
        calls.append(("inspect", file))
        return [LOCAL_TREE_PREDICTOR] if untrusted is None else untrusted

    def load(file, *, trusted):
        assert Path(file).read_bytes() == b"model"
        calls.append(("skops_load", file, trusted))
        return model

    io = ModuleType("skops.io")
    io.get_untrusted_types = inspect
    io.load = load
    monkeypatch.setitem(sys.modules, "skops.io", io)


def test_proxy_http_downloads_run_files_with_basic_auth_and_cleans_up(monkeypatch):
    calls, model = proxy_loader_fixture(monkeypatch)
    fake_skops(monkeypatch, calls, model)
    seen = []

    class Response:
        status = 200

        def __init__(self, body):
            self.body = body

        def getheader(self, name):
            return None

        def read(self, amount):
            chunk, self.body = self.body[:amount], self.body[amount:]
            return chunk

    class Connection:
        sock = SimpleNamespace(settimeout=lambda _: None)

        def __init__(self, host, port, timeout):
            seen.append((host, port, timeout))

        def request(self, method, path, headers):
            seen.append((method, path, headers))
            self.path = path

        def getresponse(self):
            return Response(b"manifest" if self.path.endswith("MLmodel") else b"model")

        def close(self):
            pass

    monkeypatch.setattr(application.http.client, "HTTPConnection", Connection)
    result = application.load_model_from_mlflow(
        application.MODEL_CATALOG["invoice-review"], "http://mlflow:5000"
    )
    assert result == application.LoadedModel(model, "7", "run-123")
    assert ("run", "run-123") in calls
    assert ("list", "run-123", "model") in calls
    assert not any(call[0] == "load" for call in calls)
    inspection = next(call for call in calls if call[0] == "inspect")
    deserialization = next(call for call in calls if call[0] == "skops_load")
    assert calls.index(inspection) < calls.index(deserialization)
    assert deserialization == ("skops_load", inspection[1], [LOCAL_TREE_PREDICTOR])
    assert not Path(inspection[1]).exists()
    requests = [call for call in seen if call[0] == "GET"]
    assert [call[1] for call in requests] == [
        "/api/2.0/mlflow-artifacts/artifacts/1/run-123/artifacts/model/MLmodel",
        "/api/2.0/mlflow-artifacts/artifacts/1/run-123/artifacts/model/model.skops",
    ]
    assert all(
        call[2]["Authorization"] == "Basic " + b64encode(b"local-user:local-password").decode()
        for call in requests
    )
    assert all(call[:2] == ("mlflow", 5000) for call in seen if call[0] != "GET")


def test_proxy_rejects_additional_untrusted_type_without_deserializing(monkeypatch):
    calls, model = proxy_loader_fixture(monkeypatch)
    fake_skops(monkeypatch, calls, model, [LOCAL_TREE_PREDICTOR, "unapproved.CustomType"])

    def download(host, port, path, destination, authorization, deadline, remaining):
        content = b"model" if path.endswith("model.skops") else b"manifest"
        destination.write_bytes(content)
        return len(content)

    monkeypatch.setattr(application, "_proxy_download_file", download)
    with pytest.raises(ValueError, match="Local model artifact is not trusted"):
        application.load_model_from_mlflow(
            application.MODEL_CATALOG["invoice-review"], "http://mlflow:5000"
        )
    assert len([call for call in calls if call[0] == "inspect"]) == 1
    assert not any(call[0] in ("skops_load", "load") for call in calls)


def test_proxy_rejects_missing_expected_untrusted_type(monkeypatch):
    calls, model = proxy_loader_fixture(monkeypatch)
    fake_skops(monkeypatch, calls, model, [])

    def download(host, port, path, destination, authorization, deadline, remaining):
        content = b"model" if path.endswith("model.skops") else b"manifest"
        destination.write_bytes(content)
        return len(content)

    monkeypatch.setattr(application, "_proxy_download_file", download)
    with pytest.raises(ValueError, match="Local model artifact is not trusted"):
        application.load_model_from_mlflow(
            application.MODEL_CATALOG["invoice-review"], "http://mlflow:5000"
        )
    assert not any(call[0] in ("skops_load", "load") for call in calls)


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/1/run-123/artifacts",
        "mlflow-artifacts://evil.test/1/run-123/artifacts",
        "mlflow-artifacts:/1/../artifacts",
    ],
)
def test_proxy_rejects_untrusted_artifact_uri(monkeypatch, uri):
    proxy_loader_fixture(monkeypatch, artifact_uri=uri)
    with pytest.raises(ValueError):
        application.load_model_from_mlflow(
            application.MODEL_CATALOG["invoice-review"], "http://mlflow:5000"
        )


@pytest.mark.parametrize(
    "tracking_uri",
    [
        "http://evil.test:5000",
        "http://mlflow:5000/path",
        "http://user@mlflow:5000",
    ],
)
def test_proxy_rejects_unapproved_tracking_origin(monkeypatch, tracking_uri):
    proxy_loader_fixture(monkeypatch)
    with pytest.raises(ValueError):
        application.load_model_from_mlflow(
            application.MODEL_CATALOG["invoice-review"], tracking_uri
        )


@pytest.mark.parametrize("path", ["model/../other", "model/%2e%2e/other", "model/x\\y", "model//x"])
def test_proxy_rejects_unsafe_listed_paths(monkeypatch, path):
    proxy_loader_fixture(monkeypatch, files=[SimpleNamespace(path=path, is_dir=False)])
    with pytest.raises(ValueError):
        application.load_model_from_mlflow(
            application.MODEL_CATALOG["invoice-review"], "http://mlflow:5000"
        )


def test_proxy_rejects_redirect_and_cleans_up_partial_download(monkeypatch):
    proxy_loader_fixture(monkeypatch)

    class Response:
        status = 302

    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(application.http.client, "HTTPConnection", Connection)
    with pytest.raises(ValueError):
        application.load_model_from_mlflow(
            application.MODEL_CATALOG["invoice-review"], "http://mlflow:5000"
        )


@pytest.mark.parametrize(
    "files",
    [
        [],
        [SimpleNamespace(path=f"model/{i}", is_dir=False) for i in range(65)],
    ],
)
def test_proxy_rejects_empty_or_excessive_artifact_listing(monkeypatch, files):
    proxy_loader_fixture(monkeypatch, files=files)
    with pytest.raises(ValueError):
        application.load_model_from_mlflow(
            application.MODEL_CATALOG["invoice-review"], "http://mlflow:5000"
        )


def test_proxy_cleans_up_partial_download_on_failure(monkeypatch):
    proxy_loader_fixture(monkeypatch)
    paths = []

    def fail_download(host, port, path, destination, authorization, deadline, remaining):
        destination.write_bytes(b"partial")
        paths.append(destination)
        raise TimeoutError("local proxy timed out")

    monkeypatch.setattr(application, "_proxy_download_file", fail_download)
    with pytest.raises(TimeoutError):
        application.load_model_from_mlflow(
            application.MODEL_CATALOG["invoice-review"], "http://mlflow:5000"
        )
    assert len(paths) == 1
    assert not paths[0].exists()
    assert not paths[0].parent.exists()


def test_proxy_cleans_up_when_local_model_deserialization_fails(monkeypatch):
    calls, _ = proxy_loader_fixture(monkeypatch)
    model_dir = []

    def download(host, port, path, destination, authorization, deadline, remaining):
        content = b"model" if path.endswith("model.skops") else b"manifest"
        destination.write_bytes(content)
        return len(content)

    def fail_load(file, *, trusted):
        assert trusted == [LOCAL_TREE_PREDICTOR]
        model_dir.append(Path(file))
        raise ValueError("invalid model")

    monkeypatch.setattr(application, "_proxy_download_file", download)
    io = ModuleType("skops.io")
    io.get_untrusted_types = lambda *, file: [LOCAL_TREE_PREDICTOR]
    io.load = fail_load
    monkeypatch.setitem(sys.modules, "skops.io", io)
    with pytest.raises(ValueError, match="invalid model"):
        application.load_model_from_mlflow(
            application.MODEL_CATALOG["invoice-review"], "http://mlflow:5000"
        )
    assert len(model_dir) == 1
    assert not model_dir[0].exists()
    assert not any(call[0] == "load" for call in calls)


def test_proxy_rejects_total_budget_after_first_file(monkeypatch):
    proxy_loader_fixture(monkeypatch)
    remaining = []

    def download(host, port, path, destination, authorization, deadline, budget):
        remaining.append(budget)
        destination.write_bytes(b"x")
        if len(remaining) == 2:
            raise ValueError("Artifact exceeds download limit")
        return application.MAX_PROXY_TOTAL_BYTES

    monkeypatch.setattr(application, "_proxy_download_file", download)
    with pytest.raises(ValueError, match="limit"):
        application.load_model_from_mlflow(
            application.MODEL_CATALOG["invoice-review"], "http://mlflow:5000"
        )
    assert remaining == [application.MAX_PROXY_TOTAL_BYTES, 0]


def test_proxy_rejects_oversized_response_before_writing(monkeypatch, tmp_path):
    class Response:
        status = 200

        def getheader(self, name):
            return str(application.MAX_PROXY_FILE_BYTES + 1)

    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(application.http.client, "HTTPConnection", Connection)
    destination = tmp_path / "artifact"
    with pytest.raises(ValueError, match="limit"):
        application._proxy_download_file(
            "mlflow",
            5000,
            "/api/2.0/mlflow-artifacts/artifacts/1/run-123/artifacts/model/MLmodel",
            destination,
            "Basic test",
            application.monotonic() + 5,
            application.MAX_PROXY_TOTAL_BYTES,
        )
    assert not destination.exists()


def test_proxy_rejects_stream_exceeding_limit_and_closes_connection(monkeypatch, tmp_path):
    closed = []

    class Response:
        status = 200

        def getheader(self, name):
            return None

        def read(self, amount):
            return b"x" * amount

    class Connection:
        sock = SimpleNamespace(settimeout=lambda _: None)

        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            closed.append(True)

    monkeypatch.setattr(application.http.client, "HTTPConnection", Connection)
    with pytest.raises(ValueError, match="limit"):
        application._proxy_download_file(
            "mlflow",
            5000,
            "/artifact",
            tmp_path / "artifact",
            "Basic test",
            application.monotonic() + 5,
            1,
        )
    assert closed == [True]


def test_proxy_rejects_expired_overall_deadline_without_request(monkeypatch, tmp_path):
    calls = []

    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            calls.append("request")

        def close(self):
            pass

    monkeypatch.setattr(application.http.client, "HTTPConnection", Connection)
    with pytest.raises(TimeoutError):
        application._proxy_download_file(
            "mlflow",
            5000,
            "/artifact",
            tmp_path / "artifact",
            "Basic test",
            application.monotonic() - 1,
            1,
        )
    assert calls == []


def test_invalid_artifact_mode_fails_at_startup(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("MODEL_API_ARTIFACT_DOWNLOAD_MODE", "unexpected")
    with pytest.raises(RuntimeError, match="MODEL_API_ARTIFACT_DOWNLOAD_MODE"):
        application.read_mlflow_configuration()


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
        sklearn=SimpleNamespace(
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


def test_proxy_mode_preserves_model_routes_and_refresh_auth(monkeypatch):
    monkeypatch.setenv("MODEL_API_ARTIFACT_DOWNLOAD_MODE", "proxy-http")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("MODEL_API_ADMIN_TOKEN", "test-admin-token")

    class Model:
        classes_ = (0, 1)

        def predict_proba(self, rows):
            assert len(rows) == 1
            return [[0.3, 0.7]]

    class Client:
        def __init__(self, *, tracking_uri):
            assert tracking_uri == "http://mlflow:5000"

        def get_model_version_by_alias(self, name, alias):
            assert (name, alias) == ("invoice-review-production", "champion")
            return SimpleNamespace(version="7", run_id="run-123")

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(
            set_tracking_uri=lambda uri: None,
            MlflowClient=Client,
            sklearn=SimpleNamespace(load_model=lambda uri: pytest.fail("SDK loader called")),
        ),
    )
    monkeypatch.setattr(application, "_load_model_via_local_proxy", lambda *args: Model())

    with TestClient(application.create_app()) as client:
        assert client.get("/models/invoice-review/health").json()["model_version"] == "7"
        response = client.post(
            "/models/invoice-review/predict",
            json={
                "invoice_amount_cents": 125000,
                "vendor_tenure_days": 365,
                "previous_incidents_12m": 1,
                "amount_vs_vendor_median": 1.25,
                "has_purchase_order": True,
                "three_way_match": True,
                "bank_account_recently_changed": False,
                "country_risk": "CL",
            },
        )
        assert response.status_code == 200
        assert response.json()["probability"] == 0.7
        assert client.post("/admin/models/invoice-review/refresh").status_code == 401


def test_lifespan_loads_each_model_once_and_requests_do_not_call_mlflow(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.setenv("MODEL_API_ADMIN_TOKEN", "test-admin-token")
    calls: list[tuple[str, str]] = []

    class FakeModel:
        classes_ = (0, 1)

        def predict_proba(self, _: list[list[object]]) -> list[list[float]]:
            return [[0.3, 0.7]]

    def fake_loader(entry: object, tracking_uri: str) -> object:
        calls.append((entry.registered_model, tracking_uri))
        return SimpleNamespace(model=FakeModel(), model_version="7", run_id="run-123")

    monkeypatch.setattr(application, "load_model_from_mlflow", fake_loader)

    with TestClient(application.create_app()) as client:
        assert client.get("/models/invoice-review/health").status_code == 200
        assert client.get("/models/invoice-review/health").status_code == 200
        assert client.get("/models").json() == {"models": ["invoice-review"]}
        predict_response = client.post(
            "/models/invoice-review/predict",
            json={
                "invoice_amount_cents": 125000,
                "vendor_tenure_days": 365,
                "previous_incidents_12m": 1,
                "amount_vs_vendor_median": 1.25,
                "has_purchase_order": True,
                "three_way_match": True,
                "bank_account_recently_changed": False,
                "country_risk": "CL",
            },
        )

    assert predict_response.status_code == 200
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
    monkeypatch.setenv("MODEL_API_ADMIN_TOKEN", "test-admin-token")
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
        assert client.get("/models/healthy/health").status_code == 200
        assert client.get("/health").json() == {"status": "ok"}

    assert response.status_code == 503
    assert "credential=not-for-clients" not in response.text


def test_incomplete_mlflow_metadata_degrades_registered_model(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.setenv("MODEL_API_ADMIN_TOKEN", "test-admin-token")

    class FakeMlflowClient:
        def __init__(self, *, tracking_uri: str) -> None:
            pass

        def get_model_version_by_alias(self, name: str, alias: str) -> object:
            return SimpleNamespace(version="7", run_id=None)

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda _: None,
        MlflowClient=FakeMlflowClient,
        sklearn=SimpleNamespace(load_model=lambda _: object()),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    with TestClient(application.create_app()) as client:
        response = client.get("/models/invoice-review/health")

    assert response.status_code == 503
    assert response.json()["detail"]["message"] == "Model loader is unavailable"


def test_loaded_model_health_reports_the_active_model_after_api_04(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.setenv("MODEL_API_ADMIN_TOKEN", "test-admin-token")
    monkeypatch.setattr(
        application,
        "load_model_from_mlflow",
        lambda *_: SimpleNamespace(model=object(), model_version="7", run_id="run-123"),
    )

    with TestClient(application.create_app()) as client:
        health_response = client.get("/models/invoice-review/health")

    assert health_response.status_code == 200
    assert health_response.json() == {
        "model_id": "invoice-review",
        "registered_model": "invoice-review-production",
        "alias": "champion",
        "model_version": "7",
        "run_id": "run-123",
    }
