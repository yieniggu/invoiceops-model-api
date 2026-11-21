"""Fixtures: a real local MLflow server (workspaces + artifact proxy) and real skops models.

The service only downloads from the tracking server's proxied artifact endpoint over HTTP, so
the tests run an actual ``mlflow server --enable-workspaces`` on a free localhost port.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pytest
from fastapi.testclient import TestClient
from invoiceops_ml.contract import FEATURES, TARGET
from invoiceops_ml.synthetic import generate_invoices
from mlflow.models import infer_signature
from mlflow.utils.workspace_context import WorkspaceContext
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from invoiceops_model_api.app import create_app
from invoiceops_model_api.settings import Settings

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

TOKEN = "t" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}
WORKSPACE = "group-a"
SEED = 424242  # official validation seed used by the tests (not the students' seed)


def make_settings(tracking_uri: str, **overrides) -> Settings:
    base = {
        "service_token": TOKEN,
        "tracking_uri": tracking_uri,
        "official_validation_seed": SEED,
        "official_validation_rows": 2500,
    }
    base.update(overrides)
    return Settings(**base)


@dataclass
class Store:
    uri: str
    experiment_id: str
    train_X: object
    train_y: object
    artifact_root: Path

    def log(
        self,
        estimator,
        name: str,
        *,
        fmt: str = "skops",
        columns: list[str] | None = None,
        fake_auc: float | None = None,
        signature_override=None,
        workspace: str = WORKSPACE,
        y=None,
    ) -> str:
        cols = columns or list(FEATURES)
        X = self.train_X[cols]
        estimator.fit(X, self.train_y if y is None else y)
        signature = signature_override or infer_signature(X, estimator.predict_proba(X))
        client = mlflow.MlflowClient(self.uri)
        mlflow.set_tracking_uri(self.uri)
        with WorkspaceContext(workspace):
            with mlflow.start_run(experiment_id=self.experiment_id):
                if fake_auc is not None:
                    mlflow.log_metric("auc", fake_auc)
                    mlflow.log_metric("validation_auc", fake_auc)
                mlflow.sklearn.log_model(
                    estimator,
                    name="model",
                    serialization_format=fmt,
                    signature=signature,
                    input_example=X.head(2),
                    registered_model_name=name,
                    pip_requirements=["scikit-learn"],
                )
            versions = client.search_model_versions(f"name='{name}'")
        return str(max(int(v.version) for v in versions))


@dataclass(frozen=True)
class ServerInfo:
    uri: str
    artifact_root: Path  # where the server stores artifacts on disk (tests tamper with it)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def mlflow_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ServerInfo]:
    root: Path = tmp_path_factory.mktemp("mlflow")
    port = _free_port()
    uri = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--enable-workspaces",
            "--backend-store-uri",
            f"sqlite:///{root}/mlflow.db",
            "--artifacts-destination",
            str(root / "artifacts"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 90
        while True:
            try:
                with urllib.request.urlopen(f"{uri}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("mlflow server did not start")
            time.sleep(0.5)
        yield ServerInfo(uri, root / "artifacts")
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)


@pytest.fixture(scope="session")
def store(mlflow_server: ServerInfo) -> Store:
    mlflow.set_tracking_uri(mlflow_server.uri)
    admin = mlflow.MlflowClient(mlflow_server.uri)
    for workspace in (WORKSPACE, "group-b"):
        admin.create_workspace(workspace)
    with WorkspaceContext(WORKSPACE):
        experiment_id = mlflow.create_experiment("tests")
    train = generate_invoices(n_invoices=3000, seed=1)
    return Store(
        mlflow_server.uri,
        experiment_id,
        train[list(FEATURES)],
        train[TARGET],
        mlflow_server.artifact_root,
    )


@pytest.fixture(scope="session")
def versions(store: Store) -> dict[str, tuple[str, str]]:
    """name -> (registered model name, version) for each scenario."""
    out: dict[str, tuple[str, str]] = {}

    def add(key: str, estimator, **kw):
        out[key] = (f"m-{key}", store.log(estimator, f"m-{key}", **kw))

    add("good", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))
    add("hgb", HistGradientBoostingClassifier(max_iter=30, random_state=0))
    add("dummy_fake", DummyClassifier(strategy="prior"), fake_auc=0.99)
    add("pickle", LogisticRegression(max_iter=1000), fmt="cloudpickle")
    add("svc", SVC(probability=True, random_state=0))  # skops-trusted by default, not allowlisted
    cols7 = list(FEATURES)[:-1]
    add("seven", LogisticRegression(max_iter=1000), columns=cols7)
    labels = np.where(store.train_y == 1, "rej", "ok")  # string classes_
    add("strclasses", LogisticRegression(max_iter=1000), y=labels)
    return out


@pytest.fixture()
def client(store: Store) -> Iterator[TestClient]:
    app = create_app(make_settings(store.uri))
    yield TestClient(app)
    app.state.supervisor.shutdown()


def ref_body(versions, key: str, version: str | None = None, workspace: str = WORKSPACE) -> dict:
    name, ver = versions[key]
    return {"workspace": workspace, "registeredModel": name, "version": version or ver}


def sample_request() -> dict:
    return {
        "invoice": {
            "supplierName": "Proveedor 001",
            "amountPesos": 1250000,
            "issuedOn": "2025-06-01",
            "dueOn": "2025-07-01",
            "createdAt": "2025-06-03T10:00:00Z",
        },
        "priorInvoices": [
            {
                "supplierName": "proveedor 001",
                "amountPesos": 1000000,
                "createdAt": "2025-05-01T10:00:00Z",
                "businessStatus": "REJECTED",
            },
            {
                "supplierName": "Proveedor 001",
                "amountPesos": 900000,
                "createdAt": "2025-05-10T10:00:00Z",
                "businessStatus": "APPROVED",
            },
        ],
    }
