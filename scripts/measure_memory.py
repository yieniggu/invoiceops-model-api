"""Measure resident memory of the service with N loaded deployments of a typical model.

Usage: uv run python scripts/measure_memory.py [N]   (default 10)

Starts a temporary local MLflow server (workspaces + artifact proxy), logs three typical skops
models (logistic regression pipeline, histogram gradient boosting, random forest), loads each as
N deployments through the real API and prints the RSS (via ``ps``) of the HTTP process and of the
inference worker process (which holds the models) at rest and after each batch of loads, plus the
peak RSS of a one-shot deploy-check child.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
from fastapi.testclient import TestClient
from invoiceops_ml.contract import FEATURES, TARGET
from invoiceops_ml.synthetic import generate_invoices
from mlflow.models import infer_signature
from mlflow.utils.workspace_context import WorkspaceContext
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from invoiceops_model_api.app import create_app
from invoiceops_model_api.settings import Settings

TOKEN = "m" * 40


def rss_mb(pid: int | None = None) -> float:
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid or os.getpid())],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip()) / 1024


def start_server(root: Path) -> tuple[subprocess.Popen, str]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
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
            f"sqlite:///{root}/m.db",
            "--artifacts-destination",
            str(root / "art"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"{uri}/health", timeout=2):
                return server, uri
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("mlflow server did not start")


def main(n: int) -> None:
    root = Path(tempfile.mkdtemp())
    server, uri = start_server(root)
    try:
        run(n, root, uri)
    finally:
        os.killpg(server.pid, signal.SIGTERM)


def run(n: int, root: Path, uri: str) -> None:
    mlflow.set_tracking_uri(uri)
    mlflow.MlflowClient(uri).create_workspace("mem-ws")
    with WorkspaceContext("mem-ws"):
        exp = mlflow.create_experiment("mem")
    data = generate_invoices(n_invoices=4000, seed=1)
    X, y = data[list(FEATURES)], data[TARGET]
    models = {
        "logreg-pipeline": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "hist-gb-200": HistGradientBoostingClassifier(max_iter=200, random_state=0),
        "random-forest-200": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, random_state=0
        ),
    }
    for name, est in models.items():
        est.fit(X, y)
        with WorkspaceContext("mem-ws"), mlflow.start_run(experiment_id=exp):
            mlflow.sklearn.log_model(
                est,
                name="model",
                serialization_format="skops",
                signature=infer_signature(X, est.predict_proba(X)),
                input_example=X.head(2),
                registered_model_name=name,
                pip_requirements=["scikit-learn"],
            )
    settings = Settings(
        service_token=TOKEN,
        tracking_uri=uri,
        official_validation_seed=987654,
        max_deployments=3 * n,
    )
    app = create_app(settings)
    client = TestClient(app)
    supervisor = app.state.supervisor
    headers = {"Authorization": f"Bearer {TOKEN}"}

    def worker_rss() -> float:
        pid = supervisor.worker_pid()
        return rss_mb(pid) if pid else 0.0

    def report(label: str) -> None:
        http, worker = rss_mb(), worker_rss()
        print(f"{label}: http {http:.0f} MB + worker {worker:.0f} MB = {http + worker:.0f} MB")

    # Warm the validation set and imports with one deploy-check (spawns a one-shot child).
    started = time.monotonic()
    client.post(
        "/deploy-check",
        headers=headers,
        json={"workspace": "mem-ws", "registeredModel": "logreg-pipeline", "version": "1"},
    )
    print(f"one deploy-check (download + child spawn + gate): {time.monotonic() - started:.1f} s")
    supervisor.warm()  # start the (empty) inference worker so its idle size is measured
    report("rest (0 deployments, empty worker)")
    for name in models:
        before = worker_rss()
        for i in range(n):
            r = client.put(
                f"/deployments/{name}-{i}",
                headers=headers,
                json={"workspace": "mem-ws", "registeredModel": name, "version": "1"},
            )
            r.raise_for_status()
        after = worker_rss()
        size = next((root / "art").rglob("model.skops")).stat().st_size
        print(
            f"{name}: +{n} deployments -> worker {after:.0f} MB, "
            f"+{(after - before) / n:.1f} MB each (model.skops {size / 1024:.0f} KiB)"
        )
    report(f"all {3 * n} deployments loaded")
    supervisor.shutdown()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
