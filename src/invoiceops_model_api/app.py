import logging
import multiprocessing
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hmac import compare_digest
from math import isfinite
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from threading import Lock
from time import monotonic

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
logger = logging.getLogger("invoiceops_model_api.refresh")
DEFAULT_MODEL_REFRESH_TIMEOUT_SECONDS = 30.0
PROCESS_CLEANUP_TIMEOUT_SECONDS = 0.1


@dataclass(frozen=True)
class MlflowConfiguration:
    tracking_uri: str
    admin_token: str
    refresh_timeout_seconds: float


@dataclass(frozen=True)
class LoadedModel:
    model: object
    model_version: str
    run_id: str


class ModelLoadFailure(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


def read_mlflow_configuration() -> MlflowConfiguration:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    admin_token = os.getenv("MODEL_API_ADMIN_TOKEN")
    if not tracking_uri or not tracking_uri.strip():
        raise RuntimeError("MLFLOW_TRACKING_URI must be configured")
    if (username is None) != (password is None):
        raise RuntimeError("MLflow tracking credentials must be configured together")
    if not admin_token or not admin_token.strip():
        raise RuntimeError("MODEL_API_ADMIN_TOKEN must be configured")
    timeout_value = os.getenv(
        "MODEL_REFRESH_TIMEOUT_SECONDS", str(DEFAULT_MODEL_REFRESH_TIMEOUT_SECONDS)
    )
    try:
        refresh_timeout_seconds = float(timeout_value)
    except ValueError as error:
        raise RuntimeError("MODEL_REFRESH_TIMEOUT_SECONDS must be a finite positive number") from error
    if not isfinite(refresh_timeout_seconds) or refresh_timeout_seconds <= 0:
        raise RuntimeError("MODEL_REFRESH_TIMEOUT_SECONDS must be a finite positive number")
    return MlflowConfiguration(
        tracking_uri=tracking_uri,
        admin_token=admin_token,
        refresh_timeout_seconds=refresh_timeout_seconds,
    )


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


def model_load_process_context() -> BaseContext:
    # spawn avoids inheriting the HTTP server's threads and locks into the loader process.
    return multiprocessing.get_context("spawn")


def _send_child_result(connection: Connection, status: str, payload: object | None) -> bool:
    try:
        connection.send((status, payload))
    except Exception:  # noqa: BLE001
        return False
    return True


def _load_model_in_child(
    connection: Connection,
    catalog_entry: ModelCatalogEntry,
    tracking_uri: str,
    loader: Callable[[ModelCatalogEntry, str], LoadedModel],
) -> None:
    try:
        loaded_model = loader(catalog_entry, tracking_uri)
    except Exception:  # noqa: BLE001
        _send_child_result(connection, "load_failure", None)
    else:
        if not _send_child_result(connection, "success", loaded_model):
            _send_child_result(connection, "transfer_failure", None)
    finally:
        connection.close()


def _stop_loader_process(process: multiprocessing.Process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(PROCESS_CLEANUP_TIMEOUT_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(PROCESS_CLEANUP_TIMEOUT_SECONDS)
    if not process.is_alive():
        process.close()


def load_model_in_isolated_process(
    catalog_entry: ModelCatalogEntry,
    tracking_uri: str,
    timeout_seconds: float,
    loader: Callable[[ModelCatalogEntry, str], LoadedModel],
) -> LoadedModel:
    context = model_load_process_context()
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_load_model_in_child,
        args=(child_connection, catalog_entry, tracking_uri, loader),
    )
    try:
        process.start()
    except Exception as error:
        parent_connection.close()
        child_connection.close()
        process.close()
        raise ModelLoadFailure("process_start_failure") from error
    child_connection.close()

    try:
        deadline = monotonic() + timeout_seconds
        while monotonic() < deadline:
            remaining_seconds = deadline - monotonic()
            if parent_connection.poll(min(remaining_seconds, 0.01)):
                try:
                    status, payload = parent_connection.recv()
                except (EOFError, OSError) as error:
                    raise ModelLoadFailure("child_exit") from error
                if status == "success":
                    if isinstance(payload, LoadedModel):
                        return payload
                    raise ModelLoadFailure("transfer_failure")
                raise ModelLoadFailure(
                    "transfer_failure" if status == "transfer_failure" else "load_failure"
                )
            if process.exitcode is not None:
                raise ModelLoadFailure("child_exit")
        raise ModelLoadFailure("timeout")
    finally:
        parent_connection.close()
        _stop_loader_process(process)


@asynccontextmanager
async def model_lifespan(app: FastAPI) -> AsyncIterator[None]:
    configuration = read_mlflow_configuration()
    app.state.mlflow_configuration = configuration
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
    application.state.model_registry_lock = Lock()
    application.state.model_refresh_lock = Lock()
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


class ModelHealthResponse(BaseModel):
    model_id: str
    registered_model: str
    alias: str
    model_version: str
    run_id: str


def active_model(request: Request, model_id: str) -> LoadedModel | None:
    # A request snapshots one fully loaded model; inference continues on that instance.
    with request.app.state.model_registry_lock:
        return request.app.state.model_registry[model_id]


def model_health_response(model_id: str, loaded_model: LoadedModel) -> ModelHealthResponse:
    catalog_entry = MODEL_CATALOG[model_id]
    return ModelHealthResponse(
        model_id=model_id,
        registered_model=catalog_entry.registered_model,
        alias=catalog_entry.alias,
        model_version=loaded_model.model_version,
        run_id=loaded_model.run_id,
    )


def require_administrative_credential(request: Request) -> None:
    provided_token = request.headers.get("X-Model-Admin-Token")
    expected_token = request.app.state.mlflow_configuration.admin_token
    if provided_token is None or not compare_digest(provided_token, expected_token):
        raise HTTPException(status_code=401, detail="Administrative credentials are invalid")


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


@router.get("/models/{model_id}/health", response_model=ModelHealthResponse)
def model_health(model_id: str, request: Request) -> ModelHealthResponse:
    if model_id not in MODEL_CATALOG:
        raise HTTPException(status_code=404, detail="Model not found")
    loaded_model = active_model(request, model_id)
    if loaded_model is None:
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
    return model_health_response(model_id, loaded_model)


@router.post(
    "/admin/models/{model_id}/refresh", response_model=ModelHealthResponse
)
def refresh_model(model_id: str, request: Request) -> ModelHealthResponse:
    require_administrative_credential(request)
    if model_id not in MODEL_CATALOG:
        raise HTTPException(status_code=404, detail="Model not found")

    with request.app.state.model_refresh_lock:
        logger.info("model_refresh_attempt", extra={"model_id": model_id})
        try:
            loaded_model = load_model_in_isolated_process(
                MODEL_CATALOG[model_id],
                request.app.state.mlflow_configuration.tracking_uri,
                request.app.state.mlflow_configuration.refresh_timeout_seconds,
                load_model_from_mlflow,
            )
        except ModelLoadFailure as error:
            logger.warning(
                "model_refresh_failed",
                extra={"model_id": model_id, "reason": error.reason},
            )
            raise HTTPException(status_code=503, detail="Model refresh is unavailable") from None
        with request.app.state.model_registry_lock:
            request.app.state.model_registry[model_id] = loaded_model
        logger.info("model_refresh_succeeded", extra={"model_id": model_id})
    return model_health_response(model_id, loaded_model)


@router.post("/models/{model_id}/predict", response_model=PredictResponse)
def predict(
    model_id: str, request: PredictRequest, app_request: Request
) -> PredictResponse:
    if model_id not in MODEL_CATALOG:
        raise HTTPException(status_code=404, detail="Model not found")
    loaded_model = active_model(app_request, model_id)
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
