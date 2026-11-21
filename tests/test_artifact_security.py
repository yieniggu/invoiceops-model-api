"""SSRF / credential exfiltration regressions: the version `source` is student-controlled."""

from __future__ import annotations

import os

import mlflow
import pytest
from conftest import AUTH, WORKSPACE, make_settings
from fakeservers import FakeTracking, attacker, fixed_body, gzip_bomb, redirect_to
from fastapi.testclient import TestClient
from mlflow.utils.workspace_context import WorkspaceContext

from invoiceops_model_api.app import create_app
from invoiceops_model_api.artifacts import (
    ArtifactError,
    ModelRef,
    fetch_model,
    validate_artifact_source,
)
from invoiceops_model_api.settings import ConfigError, Settings

TRACKING = "http://tracking.internal:5000"
MLMODEL = (
    b"flavors:\n  python_function:\n    loader_module: mlflow.sklearn\n    model_path: model.skops\n"
    b"  sklearn:\n    pickled_model: model.skops\n    serialization_format: skops\n"
)


# ---- source validation --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "path"),
    [
        ("mlflow-artifacts:/0/models/m-abc/artifacts", "0/models/m-abc/artifacts"),
        ("mlflow-artifacts:///1/run/artifacts/model", "1/run/artifacts/model"),
        (
            "mlflow-artifacts:/workspaces/group-a/1/models/m-1/artifacts",
            "workspaces/group-a/1/models/m-1/artifacts",
        ),
        ("mlflow-artifacts://tracking.internal:5000/1/x", "1/x"),
        ("MLFLOW-ARTIFACTS://TRACKING.INTERNAL:5000/1/x", "1/x"),
    ],
)
def test_legitimate_sources_are_normalized(source: str, path: str) -> None:
    assert validate_artifact_source(source, TRACKING, "group-a") == path


@pytest.mark.parametrize(
    "source",
    [
        "mlflow-artifacts://attacker.example:80/x",
        "mlflow-artifacts://127.0.0.1:9/x",
        "mlflow-artifacts://tracking.internal:5001/x",  # same host, other port
        "mlflow-artifacts://user:pw@tracking.internal:5000/x",
        "mlflow-artifacts://tracking.internal.evil.example:5000/x",
        "http://tracking.internal:5000/api/2.0/mlflow-artifacts/artifacts/x",
        "https://attacker.example/x",
        "s3://bucket/key",
        "file:///etc/passwd",
        "/etc/passwd",
        "models:/m-1",
        "",
        "mlflow-artifacts:/",
        "mlflow-artifacts:/0/../1/x",
        "mlflow-artifacts:/0/./x",
        "mlflow-artifacts:/0//x",
        "mlflow-artifacts:/0/%2e%2e/x",
        "mlflow-artifacts:/0/x?redirect=http://attacker.example",
        "mlflow-artifacts:/0/x#frag",
        "mlflow-artifacts:/0/x;params",
        "mlflow-artifacts:/0\\x",
        "mlflow-artifacts://[::1/x",  # malformed: urllib/mlflow raise ValueError
        "mlflow-artifacts://host:notaport/x",
        "mlflow-artifacts:/workspaces/group-b/1/x",  # another workspace's prefix
        "mlflow-artifacts:/0/x\n",
    ],
)
def test_hostile_sources_are_rejected(source: str) -> None:
    with pytest.raises(ArtifactError) as info:
        validate_artifact_source(source, TRACKING, "group-a")
    assert info.value.code == "ARTIFACT_SOURCE_NOT_ALLOWED"


def test_settings_cannot_reenable_other_schemes_or_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not hasattr(Settings, "allowed_artifact_schemes")
    env = {
        "MODEL_API_SERVICE_TOKEN": "t" * 40,
        "MLFLOW_TRACKING_URI": TRACKING,
        "OFFICIAL_VALIDATION_SEED": "9f2c5d71e0a84b36c7d1f0958e2a6b43",
        "MODEL_API_ALLOWED_ARTIFACT_SCHEMES": "mlflow-artifacts,https,http",
    }
    settings = Settings.from_env(env)
    assert not any("scheme" in field for field in vars(settings))
    with pytest.raises(ArtifactError):
        validate_artifact_source("https://attacker.example/x", settings.tracking_uri)


@pytest.mark.parametrize("uri", ["sqlite:///x.db", "/local/mlruns", "file:///x", "http://", ""])
def test_tracking_uri_must_be_an_http_url(uri: str) -> None:
    with pytest.raises(ConfigError):
        Settings(service_token="t" * 40, tracking_uri=uri)


# ---- end to end against a real MLflow server ----------------------------------------------


def register_source(store, name: str, source: str) -> str:
    client = mlflow.MlflowClient(store.uri)
    with WorkspaceContext(WORKSPACE):
        client.create_registered_model(name)
        return str(client.create_model_version(name, source=source, await_creation_for=0).version)


def body(name: str, version: str) -> dict:
    return {"workspace": WORKSPACE, "registeredModel": name, "version": version}


@pytest.mark.parametrize(
    ("slug", "template"),
    [
        ("netloc", "mlflow-artifacts://{netloc}/x"),
        ("userinfo", "mlflow-artifacts://user:pw@{netloc}/x"),
        ("http", "http://{netloc}/x"),
        ("https", "https://{netloc}/x"),
        ("http-api", "http://{netloc}/api/2.0/mlflow-artifacts/artifacts/x"),
    ],
)
def test_foreign_source_never_reaches_the_foreign_host_with_credentials(
    store, monkeypatch: pytest.MonkeyPatch, slug: str, template: str
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "SERVICE-IDENTITY-SECRET")
    with attacker() as evil:
        baseline = len(evil.requests)  # the MLflow server itself must not contact it either
        slug = f"ssrf-{slug}"
        version = register_source(store, f"evil-{slug}", template.format(netloc=evil.netloc))
        client = TestClient(create_app(make_settings(store.uri)))
        try:
            r = client.post("/deploy-check", json=body(f"evil-{slug}", version), headers=AUTH)
            r2 = client.put("/deployments/evil", json=body(f"evil-{slug}", version), headers=AUTH)
        finally:
            client.app.state.supervisor.shutdown()
        assert r.status_code == 200, r.text
        assert [x["code"] for x in r.json()["reasons"]] == ["ARTIFACT_SOURCE_NOT_ALLOWED"]
        assert r2.status_code == 422
        assert evil.requests[baseline:] == []  # nothing ever hit the foreign host
        assert "SERVICE-IDENTITY-SECRET" not in r.text + r2.text


# ---- bounded, tracking-only downloads -----------------------------------------------------


def fetch(tracking: str, **overrides):
    settings = make_settings(tracking, **overrides)
    return fetch_model(ModelRef(WORKSPACE, "m", "1"), settings)


def routes(source: str = "mlflow-artifacts:/1/r/artifacts") -> dict:
    return {
        "/api/2.0/mlflow/model-versions/get": {"model_version": {"run_id": "r" * 32}},
        "/api/2.0/mlflow/model-versions/get-download-uri": {"artifact_uri": source},
    }


def listing(skops_size: int) -> list[dict]:
    return [
        {"path": "MLmodel", "is_dir": False, "file_size": len(MLMODEL)},
        {"path": "model.skops", "is_dir": False, "file_size": skops_size},
    ]


def test_streaming_cap_is_enforced_while_reading_even_if_the_listing_lies() -> None:
    sent = [0]

    def endless(handler) -> None:
        handler.send_response(200)
        handler.end_headers()  # no Content-Length: only the bytes actually read can be capped
        chunk = b"x" * 65536
        while True:
            handler.wfile.write(chunk)
            sent[0] += len(chunk)

    server = FakeTracking(
        listing(10), {"MLmodel": fixed_body(MLMODEL), "model.skops": endless}, routes()
    )
    try:
        with pytest.raises(ArtifactError) as info, fetch(server.url, max_model_bytes=1024 * 1024):
            pass
        assert info.value.code == "ARTIFACT_TOO_LARGE"
        # The client stopped reading at the cap (plus socket buffers), it did not drain the stream.
        assert sent[0] < 16 * 1024 * 1024
    finally:
        server.close()


def test_listing_size_is_not_trusted_in_the_other_direction() -> None:
    big = b"y" * (2 * 1024 * 1024)
    server = FakeTracking(
        listing(5),
        {"MLmodel": fixed_body(MLMODEL), "model.skops": fixed_body(big)},
        routes(),
    )
    try:
        with pytest.raises(ArtifactError) as info, fetch(server.url, max_model_bytes=1024 * 1024):
            pass
        assert info.value.code == "ARTIFACT_TOO_LARGE"
    finally:
        server.close()


def test_compressed_bomb_is_capped_on_decoded_bytes() -> None:
    server = FakeTracking(
        listing(10),
        {"MLmodel": fixed_body(MLMODEL), "model.skops": gzip_bomb(64 * 1024 * 1024)},
        routes(),
    )
    try:
        with pytest.raises(ArtifactError) as info, fetch(server.url, max_model_bytes=1024 * 1024):
            pass
        assert info.value.code == "ARTIFACT_TOO_LARGE"
    finally:
        server.close()


def test_downloads_only_go_to_the_tracking_server_and_never_follow_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "SERVICE-IDENTITY-SECRET")
    with attacker() as evil:
        server = FakeTracking(
            listing(10),
            {
                "MLmodel": fixed_body(MLMODEL),
                "model.skops": redirect_to(f"http://{evil.netloc}/steal"),
            },
            routes(),
        )
        try:
            with pytest.raises(ArtifactError) as info, fetch(server.url):
                pass
            assert info.value.code == "ARTIFACT_SOURCE_NOT_ALLOWED"
            assert evil.requests == []
            # Credentials and workspace go to the tracking host, which is the only host used.
            assert server.requests and all(
                r["headers"].get("Authorization") == "Bearer SERVICE-IDENTITY-SECRET"
                and r["headers"].get("X-MLFLOW-WORKSPACE") == WORKSPACE
                for r in server.requests
            )
            assert all(r["path"].startswith("/api/2.0/mlflow") for r in server.requests)
            assert any("1/r/artifacts/model.skops" in r["path"] for r in server.requests)
        finally:
            server.close()


@pytest.mark.parametrize(
    "source",
    [
        "http://attacker.example/x",
        "mlflow-artifacts://attacker.example/x",
        "models:/name/3",
        "mlflow-artifacts://[::1/x",  # MLflow's own URL parsing raises ValueError on this one
    ],
)
def test_unresolved_or_foreign_sources_never_trigger_artifact_requests(source: str) -> None:
    server = FakeTracking(listing(10), {}, routes(source))
    try:
        with pytest.raises(ArtifactError) as info, fetch(server.url):
            pass
        assert info.value.code == "ARTIFACT_SOURCE_NOT_ALLOWED"
        assert not any("mlflow-artifacts/artifacts" in r["path"] for r in server.requests)
    finally:
        server.close()


# ---- auth modes: credentials only ever go to the tracking URI -----------------------------


def served_ok() -> dict:
    return {"MLmodel": fixed_body(MLMODEL), "model.skops": fixed_body(b"bytes")}


def test_gateway_mode_sends_bearer_and_workspace_header_to_the_tracking_uri_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "should-not-be-used")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "should-not-be-used")
    monkeypatch.delenv("MLFLOW_WORKSPACE", raising=False)
    token = "g" * 48
    with attacker() as evil:
        server = FakeTracking(
            listing(5), served_ok(), routes(f"mlflow-artifacts://{evil.netloc}/x")
        )
        try:
            # a foreign host in the source is rejected before any artifact request
            with (
                pytest.raises(ArtifactError),
                fetch(server.url, mlflow_auth_mode="gateway", gateway_read_token=token),
            ):
                pass
        finally:
            server.close()
        server = FakeTracking(listing(5), served_ok(), routes())
        try:
            with fetch(server.url, mlflow_auth_mode="gateway", gateway_read_token=token) as got:
                assert got.skops_path.read_bytes() == b"bytes"
            assert server.requests
            for request in server.requests:
                assert request["headers"]["Authorization"] == f"Bearer {token}"
                assert request["headers"]["X-InvoiceOps-Workspace"] == WORKSPACE
                assert "X-MLFLOW-WORKSPACE" not in request["headers"]
        finally:
            server.close()
        assert evil.requests == []
    assert "MLFLOW_WORKSPACE" not in os.environ  # no process-global state touched


def test_mlflow_mode_uses_native_credentials_and_the_mlflow_workspace_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "svc")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "pw")
    server = FakeTracking(listing(5), served_ok(), routes())
    try:
        with fetch(server.url):
            pass
        for request in server.requests:
            assert request["headers"]["Authorization"].startswith("Basic ")
            assert request["headers"]["X-MLFLOW-WORKSPACE"] == WORKSPACE
    finally:
        server.close()


def test_gateway_mode_requires_a_token_and_env_selects_it() -> None:
    with pytest.raises(ConfigError):
        Settings(service_token="t" * 40, tracking_uri=TRACKING, mlflow_auth_mode="gateway")
    with pytest.raises(ConfigError):
        Settings(service_token="t" * 40, tracking_uri=TRACKING, mlflow_auth_mode="digest")
    env = {
        "MODEL_API_SERVICE_TOKEN": "t" * 40,
        "MLFLOW_TRACKING_URI": TRACKING,
        "OFFICIAL_VALIDATION_SEED": "9f2c5d71e0a84b36c7d1f0958e2a6b43",
        "GATEWAY_SERVICE_READ_TOKEN": "g" * 48,
    }
    settings = Settings.from_env(env)
    assert settings.mlflow_auth_mode == "gateway" and settings.gateway_read_token == "g" * 48
    assert "g" * 48 not in repr(settings)
    assert (
        Settings.from_env({**env, "MODEL_API_MLFLOW_AUTH": "mlflow"}).mlflow_auth_mode == "mlflow"
    )


# ---- MLflow 3 sources: models:/m-..., runs:/... resolve through the tracking server ------


def test_logged_model_source_models_m_passes_deploy_check(store, versions) -> None:
    name, version = versions["good"]
    source = mlflow.MlflowClient(store.uri)
    with WorkspaceContext(WORKSPACE):
        mv = source.get_model_version(name, version)
    assert mv.source.startswith("models:/m-")  # what log_model + register produces in 3.16.1
    client = TestClient(create_app(make_settings(store.uri)))
    try:
        r = client.post("/deploy-check", json=body(name, version), headers=AUTH)
    finally:
        client.app.state.supervisor.shutdown()
    assert r.status_code == 200 and r.json()["passed"] is True, r.text


def test_runs_source_resolves_through_the_run_in_the_deployments_workspace(store, tmp_path) -> None:
    from invoiceops_ml.contract import FEATURES
    from mlflow.models import infer_signature
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = store.train_X[list(FEATURES)]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, store.train_y)
    local = tmp_path / "saved"
    mlflow.sklearn.save_model(
        model,
        str(local),
        serialization_format="skops",
        signature=infer_signature(X, model.predict_proba(X)),
        pip_requirements=["scikit-learn"],
    )
    client = mlflow.MlflowClient(store.uri)
    with WorkspaceContext(WORKSPACE):
        run = client.create_run(store.experiment_id)
        client.log_artifacts(run.info.run_id, str(local), "model")
        version = register_source(store, "runs-source", f"runs:/{run.info.run_id}/model")
    app = create_app(make_settings(store.uri))
    web = TestClient(app)
    try:
        ok = web.post("/deploy-check", json=body("runs-source", version), headers=AUTH)
        assert ok.status_code == 200 and ok.json()["passed"] is True, ok.text
        # the same run id registered from ANOTHER workspace is not resolvable there
        with WorkspaceContext("group-b"):
            client.create_registered_model("runs-source-b")
            other = client.create_model_version(
                "runs-source-b", source=f"runs:/{run.info.run_id}/model", await_creation_for=0
            ).version
        r = web.post(
            "/deploy-check",
            json={"workspace": "group-b", "registeredModel": "runs-source-b", "version": other},
            headers=AUTH,
        )
        assert [x["code"] for x in r.json()["reasons"]] == ["ARTIFACT_SOURCE_NOT_ALLOWED"]
    finally:
        app.state.supervisor.shutdown()
