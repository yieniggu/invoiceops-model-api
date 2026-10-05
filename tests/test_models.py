import math

import pytest
from fastapi.testclient import TestClient

from invoiceops_model_api.app import MODEL_CATALOG, LoadedModel, app, create_app

client = TestClient(app)

VALID_PREDICT_REQUEST = {
    "invoice_amount_cents": 125000,
    "vendor_tenure_days": 365,
    "previous_incidents_12m": 1,
    "amount_vs_vendor_median": 1.25,
    "has_purchase_order": True,
    "three_way_match": True,
    "bank_account_recently_changed": False,
    "country_risk": "low",
}


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


def test_predict_rejects_unexpected_features() -> None:
    response = client.post(
        "/models/invoice-review/predict",
        json={**VALID_PREDICT_REQUEST, "payload": {"invoice_id": "inv-1"}},
    )

    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("invoice_amount_cents", "125000"),
        ("amount_vs_vendor_median", "1.25"),
        ("has_purchase_order", "true"),
    ],
)
def test_predict_rejects_coercible_string_features(field: str, value: str) -> None:
    response = client.post(
        "/models/invoice-review/predict",
        json={**VALID_PREDICT_REQUEST, field: value},
    )

    assert response.status_code == 422
    assert "detail" in response.json()


def test_registered_model_predict_is_unavailable_without_loader() -> None:
    response = client.post(
        "/models/invoice-review/predict",
        json=VALID_PREDICT_REQUEST,
    )

    assert response.status_code == 503


def test_unknown_model_predict_returns_not_found() -> None:
    response = client.post(
        "/models/unknown/predict",
        json=VALID_PREDICT_REQUEST,
    )

    assert response.status_code == 404


@pytest.mark.parametrize("country_risk", ["low", "medium", "high"])
def test_predict_returns_positive_class_probability_and_active_model_metadata(
    country_risk: str,
) -> None:
    inputs: list[list[list[object]]] = []

    class FakeModel:
        classes_ = (0, 1)

        def predict_proba(self, features: list[list[object]]) -> list[list[float]]:
            inputs.append(features)
            return [[0.2, 0.8]]

    application = create_app()
    application.state.model_registry["invoice-review"] = LoadedModel(
        model=FakeModel(), model_version="7", run_id="run-123"
    )

    response = TestClient(application).post(
        "/models/invoice-review/predict",
        json={**VALID_PREDICT_REQUEST, "country_risk": country_risk},
    )

    assert response.status_code == 200
    assert response.json() == {
        "model_id": "invoice-review",
        "model_version": "7",
        "run_id": "run-123",
        "probability": 0.8,
    }
    assert inputs == [
        [[125000, 365, 1, 1.25, True, True, False, country_risk]]
    ]


def test_predict_returns_safe_unavailable_response_for_invalid_model_probability() -> None:
    class FakeModel:
        classes_ = (0, 1)

        def predict_proba(self, _: list[list[object]]) -> list[list[float]]:
            return [[0.2, math.nan]]

    application = create_app()
    application.state.model_registry["invoice-review"] = LoadedModel(
        model=FakeModel(), model_version="7", run_id="run-123"
    )

    response = TestClient(application).post(
        "/models/invoice-review/predict", json=VALID_PREDICT_REQUEST
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Model inference is unavailable"}


def test_predict_returns_safe_unavailable_response_for_out_of_range_probability() -> None:
    class FakeModel:
        classes_ = (0, 1)

        def predict_proba(self, _: list[list[object]]) -> list[list[float]]:
            return [[0.2, 1.1]]

    application = create_app()
    application.state.model_registry["invoice-review"] = LoadedModel(
        model=FakeModel(), model_version="7", run_id="run-123"
    )

    response = TestClient(application).post(
        "/models/invoice-review/predict", json=VALID_PREDICT_REQUEST
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Model inference is unavailable"}


def test_predict_does_not_leak_model_provider_errors() -> None:
    class FakeModel:
        def predict_proba(self, _: list[list[object]]) -> list[list[float]]:
            raise RuntimeError("provider token=not-for-clients")

    application = create_app()
    application.state.model_registry["invoice-review"] = LoadedModel(
        model=FakeModel(), model_version="7", run_id="run-123"
    )

    response = TestClient(application).post(
        "/models/invoice-review/predict", json=VALID_PREDICT_REQUEST
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Model inference is unavailable"}
    assert "provider token=not-for-clients" not in response.text
