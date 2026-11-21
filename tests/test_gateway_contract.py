"""Contract test: the Model API in ``gateway`` mode against a stub of the MLOPS-02 gateway.

The stub enforces what the real gateway enforces on its read path (Bearer service token, the
``X-InvoiceOps-Workspace`` selector, read-only methods, only listed headers reach MLflow) and
forwards to the real MLflow of the test session, so a model is deployed and scored end to end.
"""

from __future__ import annotations

import re
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import AUTH, WORKSPACE, make_settings, ref_body, sample_request
from fastapi.testclient import TestClient

from invoiceops_model_api.app import create_app

GATEWAY_TOKEN = "gateway-read-token-" + "g" * 24
WORKSPACE_RE = re.compile(r"^[a-z0-9][-a-z0-9]{1,62}$")


class StubGateway:
    def __init__(self, upstream: str) -> None:
        self.seen: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _deny(self, status: int) -> None:
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self) -> None:
                headers = {k.lower(): v for k, v in self.headers.items()}
                outer.seen.append({"path": self.path, "headers": headers})
                if headers.get("authorization") != f"Bearer {GATEWAY_TOKEN}":
                    return self._deny(401)
                workspace = headers.get("x-invoiceops-workspace", "")
                if not WORKSPACE_RE.match(workspace):
                    return self._deny(400)
                request = urllib.request.Request(
                    upstream + self.path, headers={"X-MLFLOW-WORKSPACE": workspace}
                )
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        status, body = response.status, response.read()
                        ctype = response.headers.get("Content-Type", "application/octet-stream")
                except urllib.error.HTTPError as error:
                    status, body, ctype = error.code, error.read(), "application/json"
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_only(self) -> None:
                self._deny(403)

            do_POST = do_PUT = do_DELETE = do_PATCH = _read_only

            def log_message(self, *args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def gateway(store) -> Iterator[StubGateway]:
    stub = StubGateway(store.uri)
    yield stub
    stub.close()


def api(gateway: StubGateway, token: str = GATEWAY_TOKEN) -> Iterator[TestClient]:
    app = create_app(
        make_settings(gateway.url, mlflow_auth_mode="gateway", gateway_read_token=token)
    )
    yield TestClient(app)
    app.state.supervisor.shutdown()


@pytest.fixture()
def through_gateway(gateway: StubGateway) -> Iterator[TestClient]:
    yield from api(gateway)


def test_deploy_and_predict_through_the_gateway_contract(
    through_gateway: TestClient, gateway: StubGateway, versions
) -> None:
    check = through_gateway.post("/deploy-check", json=ref_body(versions, "good"), headers=AUTH)
    assert check.status_code == 200 and check.json()["passed"] is True
    loaded = through_gateway.put("/deployments/d1", json=ref_body(versions, "good"), headers=AUTH)
    assert loaded.status_code == 200
    scored = through_gateway.post("/deployments/d1/predict", json=sample_request(), headers=AUTH)
    assert 0 <= scored.json()["probability"] <= 1

    assert gateway.seen
    for request in gateway.seen:
        headers = request["headers"]
        assert headers["authorization"] == f"Bearer {GATEWAY_TOKEN}"
        assert headers["x-invoiceops-workspace"] == WORKSPACE
        # Nothing MLflow-native leaks into the gateway contract: the gateway alone chooses the
        # MLflow workspace and identity.
        assert "x-mlflow-workspace" not in headers
        assert not headers["authorization"].lower().startswith("basic")
        assert request["path"].startswith("/api/2.0/mlflow")
    assert not [r for r in gateway.seen if "cookie" in r["headers"]]


def test_a_rejected_service_token_is_mlflow_unavailable_not_a_crash(
    gateway: StubGateway, versions
) -> None:
    for client in api(gateway, token="wrong-read-token-" + "w" * 24):
        response = client.post("/deploy-check", json=ref_body(versions, "good"), headers=AUTH)
        assert response.status_code == 502
        assert response.json()["detail"]["code"] == "MLFLOW_UNAVAILABLE"
    assert {r["headers"]["authorization"] for r in gateway.seen} == {
        "Bearer wrong-read-token-" + "w" * 24
    }


def test_another_workspace_is_not_found_through_the_gateway(
    through_gateway: TestClient, versions
) -> None:
    body = ref_body(versions, "good", workspace="group-b")
    result = through_gateway.post("/deploy-check", json=body, headers=AUTH).json()
    assert result["passed"] is False
    assert [r["code"] for r in result["reasons"]] == ["MODEL_VERSION_NOT_FOUND"]


def test_the_stub_enforces_a_read_only_contract(gateway: StubGateway) -> None:
    request = urllib.request.Request(
        gateway.url + "/api/2.0/mlflow/experiments/create",
        data=b"{}",
        method="POST",
        headers={"Authorization": f"Bearer {GATEWAY_TOKEN}", "X-InvoiceOps-Workspace": WORKSPACE},
    )
    with pytest.raises(urllib.error.HTTPError) as denied:
        urllib.request.urlopen(request, timeout=10)
    assert denied.value.code == 403
