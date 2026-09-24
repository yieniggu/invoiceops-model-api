import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from math import isfinite

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict


@dataclass(frozen=True)
class ModelCatalogEntry:
    registered_model: str
    alias: str


MODEL_CATALOG: dict[str, ModelCatalogEntry] = {
    "invoice-review": ModelCatalogEntry(
        registered_model="invoice-review-production",
        alias="champion",
    )
}
router = APIRouter()


@dataclass(frozen=True)
class MlflowConfiguration:
    tracking_uri: str


@dataclass(frozen=True)
class LoadedModel:
    model: object
    model_version: str
    run_id: str


def read_mlflow_configuration() -> MlflowConfiguration:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if not tracking_uri or not tracking_uri.strip():
        raise RuntimeError("MLFLOW_TRACKING_URI must be configured")
    if (username is None) != (password is None):
        raise RuntimeError("MLflow tracking credentials must be configured together")
    return MlflowConfiguration(tracking_uri=tracking_uri)


def load_model_from_mlflow(
    catalog_entry: ModelCatalogEntry, tracking_uri: str
) -> LoadedModel:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    model_version = client.get_model_version_by_alias(
        catalog_entry.registered_model, catalog_entry.alias
    )
    version = getattr(model_version, "version", None)
    run_id = getattr(model_version, "run_id", None)
    if not version or not run_id:
        raise ValueError("MLflow model version metadata is incomplete")
    model = mlflow.sklearn.load_model(
        f"models:/{catalog_entry.registered_model}/{version}"
    )
    return LoadedModel(model=model, model_version=str(version), run_id=str(run_id))


@asynccontextmanager
async def model_lifespan(app: FastAPI) -> AsyncIterator[None]:
    configuration = read_mlflow_configuration()
    for model_id, catalog_entry in MODEL_CATALOG.items():
        try:
            app.state.model_registry[model_id] = load_model_from_mlflow(
                catalog_entry, configuration.tracking_uri
            )
        except Exception:  # noqa: BLE001
            # MLflow can surface provider-specific exceptions; do not let one stop startup.
            app.state.model_registry[model_id] = None
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=model_lifespan,
    )
    application.state.model_registry = {model_id: None for model_id in MODEL_CATALOG}
    application.include_router(router)
    return application


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    invoice_amount_cents: int
    vendor_tenure_days: int
    previous_incidents_12m: int
    amount_vs_vendor_median: float
    has_purchase_order: bool
    three_way_match: bool
    bank_account_recently_changed: bool
    country_risk: str

    def as_training_features(self) -> list[object]:
        return [
            self.invoice_amount_cents,
            self.vendor_tenure_days,
            self.previous_incidents_12m,
            self.amount_vs_vendor_median,
            self.has_purchase_order,
            self.three_way_match,
            self.bank_account_recently_changed,
            self.country_risk,
        ]


class PredictResponse(BaseModel):
    model_id: str
    model_version: str
    run_id: str
    probability: float


def positive_class_probability(model: object, features: list[object]) -> float:
    probabilities = model.predict_proba([features])
    classes = list(model.classes_)
    positive_class_index = classes.index(1)
    rows = list(probabilities)
    if len(rows) != 1:
        raise ValueError("Model returned an unexpected number of predictions")
    probability = float(list(rows[0])[positive_class_index])
    if not isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Model returned an invalid probability")
    return probability


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/models")
def list_models() -> dict[str, list[str]]:
    return {"models": sorted(MODEL_CATALOG)}


@router.get("/models/{model_id}/health")
def model_health(model_id: str, request: Request) -> None:
    if model_id not in MODEL_CATALOG:
        raise HTTPException(status_code=404, detail="Model not found")
    if request.app.state.model_registry[model_id] is None:
        model = MODEL_CATALOG[model_id]
        raise HTTPException(
            status_code=503,
            detail={
                "model_id": model_id,
                "registered_model": model.registered_model,
                "alias": model.alias,
                "message": "Model loader is unavailable",
            },
        )
    raise HTTPException(status_code=503, detail="Model health is unavailable until API-04")


@router.post("/models/{model_id}/predict", response_model=PredictResponse)
def predict(
    model_id: str, request: PredictRequest, app_request: Request
) -> PredictResponse:
    if model_id not in MODEL_CATALOG:
        raise HTTPException(status_code=404, detail="Model not found")
    loaded_model = app_request.app.state.model_registry[model_id]
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model loader is unavailable")
    try:
        probability = positive_class_probability(loaded_model.model, request.as_training_features())
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=503, detail="Model inference is unavailable") from None
    return PredictResponse(
        model_id=model_id,
        model_version=loaded_model.model_version,
        run_id=loaded_model.run_id,
        probability=probability,
    )


app = create_app()
