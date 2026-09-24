import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel


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
    model = mlflow.pyfunc.load_model(
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
    payload: dict[str, object]


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


@router.post("/models/{model_id}/predict")
def predict(model_id: str, request: PredictRequest, app_request: Request) -> None:
    if model_id not in MODEL_CATALOG:
        raise HTTPException(status_code=404, detail="Model not found")
    if app_request.app.state.model_registry[model_id] is None:
        raise HTTPException(status_code=503, detail="Model loader is unavailable")
    raise HTTPException(status_code=503, detail="Inference is unavailable until API-05")


app = create_app()
