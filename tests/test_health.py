from fastapi.testclient import TestClient

from invoiceops_model_api.app import app

client = TestClient(app)


def test_health_returns_deterministic_status() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
