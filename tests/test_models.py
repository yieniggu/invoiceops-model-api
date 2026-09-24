from fastapi.testclient import TestClient

from invoiceops_model_api.app import MODEL_CATALOG, app

client = TestClient(app)


def test_models_lists_registered_logical_ids() -> None:
    response = client.get("/models")

    assert response.status_code == 200
    assert response.json() == {"models": ["invoice-review"]}


def test_invoice_review_catalog_maps_to_production_champion() -> None:
    model = MODEL_CATALOG["invoice-review"]

    assert model.registered_model == "invoice-review-production"
    assert model.alias == "champion"
    assert model.alias


def test_registered_model_health_is_unavailable_without_loader() -> None:
    response = client.get("/models/invoice-review/health")

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "model_id": "invoice-review",
            "registered_model": "invoice-review-production",
            "alias": "champion",
            "message": "Model loader is unavailable",
        }
    }


def test_unknown_model_health_returns_not_found() -> None:
    response = client.get("/models/unknown/health")

    assert response.status_code == 404


def test_predict_rejects_invalid_payload() -> None:
    response = client.post("/models/invoice-review/predict", json={})

    assert response.status_code == 422
    assert "detail" in response.json()


def test_registered_model_predict_is_unavailable_without_loader() -> None:
    response = client.post(
        "/models/invoice-review/predict",
        json={"payload": {"invoice_id": "inv-1"}},
    )

    assert response.status_code == 503


def test_unknown_model_predict_returns_not_found() -> None:
    response = client.post(
        "/models/unknown/predict",
        json={"payload": {"invoice_id": "inv-1"}},
    )

    assert response.status_code == 404
