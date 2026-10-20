import asyncio
import http.client
import logging
import multiprocessing
import os
from base64 import b64encode
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from hmac import compare_digest
from math import isfinite
from multiprocessing.context import BaseContext
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from time import monotonic
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from invoiceops_model_api.worker import (
    ModelLoadFailure,
    ModelProxy,
    WorkerPool,
    check_deadline,
    serve_model,
    stop_process,
)
from invoiceops_model_api.worker_launch import QuietSpawnContext


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
# Startup has no active model to preserve and includes cold SDK/model imports.
DEFAULT_MODEL_STARTUP_TIMEOUT_SECONDS = 180.0
MAX_PROXY_ARTIFACTS = 64
MAX_PROXY_FILE_BYTES = 32 * 1024 * 1024
MAX_PROXY_TOTAL_BYTES = 64 * 1024 * 1024
PROXY_REQUEST_TIMEOUT_SECONDS = 8.0
PROXY_OVERALL_TIMEOUT_SECONDS = 25.0
LOCAL_TRUSTED_SKOPS_TYPE = "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"


@dataclass(frozen=True)
class MlflowConfiguration:
    tracking_uri: str
    admin_token: str
    refresh_timeout_seconds: float
    startup_timeout_seconds: float = DEFAULT_MODEL_STARTUP_TIMEOUT_SECONDS


@dataclass(frozen=True)
class LoadedModel:
    model: object
    model_version: str
    run_id: str


def read_mlflow_configuration() -> MlflowConfiguration:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    admin_token = os.getenv("MODEL_API_ADMIN_TOKEN")
    if not tracking_uri or not tracking_uri.strip():
        raise RuntimeError("MLFLOW_TRACKING_URI must be configured")
    if (username is None) != (password is None):
        raise RuntimeError("MLflow tracking credentials must be configured together")
    if os.getenv("MODEL_API_ARTIFACT_DOWNLOAD_MODE", "sdk") not in ("sdk", "proxy-http"):
        raise RuntimeError("MODEL_API_ARTIFACT_DOWNLOAD_MODE must be sdk or proxy-http")
    if not admin_token or not admin_token.strip():
        raise RuntimeError("MODEL_API_ADMIN_TOKEN must be configured")
    timeout_value = os.getenv(
        "MODEL_REFRESH_TIMEOUT_SECONDS", str(DEFAULT_MODEL_REFRESH_TIMEOUT_SECONDS)
    )
    try:
        refresh_timeout_seconds = float(timeout_value)
    except ValueError as error:
        raise RuntimeError(
            "MODEL_REFRESH_TIMEOUT_SECONDS must be a finite positive number"
        ) from error
    if not isfinite(refresh_timeout_seconds) or refresh_timeout_seconds <= 0:
        raise RuntimeError("MODEL_REFRESH_TIMEOUT_SECONDS must be a finite positive number")
    try:
        startup_timeout_seconds = float(
            os.getenv("MODEL_STARTUP_TIMEOUT_SECONDS", str(DEFAULT_MODEL_STARTUP_TIMEOUT_SECONDS))
        )
    except ValueError:
        raise RuntimeError(
            "MODEL_STARTUP_TIMEOUT_SECONDS must be a finite positive number"
        ) from None
    if not isfinite(startup_timeout_seconds) or startup_timeout_seconds <= 0:
        raise RuntimeError("MODEL_STARTUP_TIMEOUT_SECONDS must be a finite positive number")
    return MlflowConfiguration(
        tracking_uri=tracking_uri,
        admin_token=admin_token,
        refresh_timeout_seconds=refresh_timeout_seconds,
        startup_timeout_seconds=startup_timeout_seconds,
    )


def _safe_proxy_path(path: str) -> list[str]:
    if not isinstance(path, str) or len(path) > 1024 or not path:
        raise ValueError("Invalid artifact path")
    parts = path.split("/")
    if any(
        part in ("", ".", "..")
        or len(part) > 255
        or any(char in part for char in ("\\", "%", "?", "#", ":"))
        or any(ord(char) < 32 or ord(char) == 127 for char in part)
        for part in parts
    ):
        raise ValueError("Invalid artifact path")
    return parts


def _proxy_origin(tracking_uri: str) -> tuple[str, int]:
    parsed = urlsplit(tracking_uri)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("Invalid local MLflow origin") from error
    host = parsed.hostname
    if (
        parsed.scheme != "http"
        or host not in ("mlflow", "localhost", "127.0.0.1")
        or port is None
        or (host == "mlflow" and port != 5000)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or parsed.netloc != f"{host}:{port}"
    ):
        raise ValueError("Invalid local MLflow origin")
    return host, port


def _proxy_download_file(
    host: str,
    port: int,
    path: str,
    destination: Path,
    authorization: str,
    deadline: float,
    remaining_bytes: int,
) -> int:
    connection = http.client.HTTPConnection(
        host, port, timeout=min(PROXY_REQUEST_TIMEOUT_SECONDS, max(0.001, deadline - monotonic()))
    )
    started = monotonic()
    try:
        if started >= deadline:
            raise TimeoutError("Artifact download deadline exceeded")
        connection.request("GET", path, headers={"Authorization": authorization})
        response = connection.getresponse()
        if response.status != 200:  # Includes all redirects; http.client never follows them.
            raise ValueError("Artifact proxy returned a non-success response")
        length = response.getheader("Content-Length")
        if length is not None and (
            not length.isdecimal() or int(length) > min(MAX_PROXY_FILE_BYTES, remaining_bytes)
        ):
            raise ValueError("Artifact exceeds download limit")
        count = 0
        with destination.open("xb") as output:
            while True:
                remaining_time = min(
                    PROXY_REQUEST_TIMEOUT_SECONDS - (monotonic() - started),
                    deadline - monotonic(),
                )
                if remaining_time <= 0:
                    raise TimeoutError("Artifact download deadline exceeded")
                if connection.sock is not None:
                    connection.sock.settimeout(remaining_time)
                chunk = response.read(
                    min(65536, min(MAX_PROXY_FILE_BYTES, remaining_bytes) - count + 1)
                )
                if not chunk:
                    break
                count += len(chunk)
                if count > min(MAX_PROXY_FILE_BYTES, remaining_bytes):
                    raise ValueError("Artifact exceeds download limit")
                output.write(chunk)
        if length is not None and count != int(length):
            raise ValueError("Incomplete artifact download")
        return count
    finally:
        connection.close()


def _load_model_via_local_proxy(
    mlflow: object, client: object, run_id: str, tracking_uri: str
) -> object:
    host, port = _proxy_origin(tracking_uri)
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if not username or not password:
        raise ValueError("Local proxy credentials are required")
    artifact_uri = client.get_run(run_id).info.artifact_uri
    parsed = urlsplit(artifact_uri)
    if (
        parsed.scheme != "mlflow-artifacts"
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise ValueError("Run does not use the local artifact proxy")
    artifact_parts = _safe_proxy_path(parsed.path[1:])
    if len(artifact_parts) < 3 or artifact_parts[-1] != "artifacts" or artifact_parts[-2] != run_id:
        raise ValueError("Run artifact URI does not match the resolved run")

    pending = ["model"]
    files: list[str] = []
    seen: set[str] = set()
    while pending:
        directory = pending.pop()
        if directory in seen or len(seen) + len(files) >= MAX_PROXY_ARTIFACTS:
            raise ValueError("Artifact listing exceeds limit")
        seen.add(directory)
        for entry in client.list_artifacts(run_id, directory):
            parts = _safe_proxy_path(entry.path)
            if parts[0] != "model" or "/".join(parts[:-1]) != directory:
                raise ValueError("Artifact listing escapes the model directory")
            if entry.path in seen or entry.path in pending or entry.path in files:
                raise ValueError("Duplicate artifact path")
            if len(seen) + len(pending) + len(files) >= MAX_PROXY_ARTIFACTS:
                raise ValueError("Artifact listing exceeds limit")
            if entry.is_dir:
                pending.append(entry.path)
            else:
                files.append(entry.path)
    if "model/MLmodel" not in files or "model/model.skops" not in files:
        raise ValueError("Model artifact is incomplete")
    authorization = "Basic " + b64encode(f"{username}:{password}".encode()).decode()
    deadline = monotonic() + PROXY_OVERALL_TIMEOUT_SECONDS
    total = 0
    with TemporaryDirectory(prefix="invoiceops-model-") as temporary:
        root = Path(temporary)
        for file in sorted(files):
            relative = _safe_proxy_path(file)
            destination = root.joinpath(*relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            proxy_path = "/api/2.0/mlflow-artifacts/artifacts/" + "/".join(
                quote(part, safe="") for part in [*artifact_parts, *relative]
            )
            total += _proxy_download_file(
                host,
                port,
                proxy_path,
                destination,
                authorization,
                deadline,
                MAX_PROXY_TOTAL_BYTES - total,
            )
        import skops.io as sio

        model_file = root / "model" / "model.skops"
        untrusted = sio.get_untrusted_types(file=model_file)
        if untrusted != [LOCAL_TRUSTED_SKOPS_TYPE]:
            raise ValueError("Local model artifact is not trusted")
        return sio.load(model_file, trusted=[LOCAL_TRUSTED_SKOPS_TYPE])


def load_model_from_mlflow(catalog_entry: ModelCatalogEntry, tracking_uri: str) -> LoadedModel:
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
    if os.getenv("MODEL_API_ARTIFACT_DOWNLOAD_MODE", "sdk") == "proxy-http":
        model = _load_model_via_local_proxy(mlflow, client, str(run_id), tracking_uri)
    else:
        model = mlflow.sklearn.load_model(f"models:/{catalog_entry.registered_model}/{version}")
    return LoadedModel(model=model, model_version=str(version), run_id=str(run_id))


def model_load_process_context() -> BaseContext:
    # spawn avoids inheriting the HTTP server's threads and locks into the loader process.
    return QuietSpawnContext()


def _stop_loader_process(process: multiprocessing.Process) -> None:
    stop_process(process)


def load_model_in_isolated_process(
    catalog_entry: ModelCatalogEntry,
    tracking_uri: str,
    timeout_seconds: float,
    loader: Callable[[ModelCatalogEntry, str], LoadedModel],
    *,
    cancelled: Event | None = None,
    owner_pool: WorkerPool | None = None,
) -> LoadedModel:
    deadline = monotonic() + timeout_seconds
    # Applications pass their persistent pool. Standalone callers retain a
    # successful proxy, or can recover an unverified owner through the exception.
    owners = owner_pool if owner_pool is not None else WorkerPool(1)
    proxy = owners.reserve(_stop_loader_process)
    try:
        check_deadline(deadline, cancelled)
        proxy.launch(
            model_load_process_context(),
            serve_model,
            (catalog_entry, tracking_uri, loader, deadline),
        )
        result = proxy.bootstrap(deadline, cancelled)
        if result == ["failure"]:
            raise ModelLoadFailure("load_failure")
        if (
            not isinstance(result, list)
            or len(result) != 4
            or result[0] != "ready"
            or any(type(value) is not str or not value for value in result[1:3])
            or not isinstance(result[3], list)
            or not result[3]
            or 1 not in result[3]
            or any(type(value) not in (str, int, float, bool) for value in result[3])
        ):
            raise ModelLoadFailure("protocol_failure")
        proxy.classes_ = result[3]
        check_deadline(deadline, cancelled)
        return LoadedModel(proxy, result[1], result[2])
    except BaseException as error:
        try:
            proxy.close()
        except ModelLoadFailure as cleanup_error:
            cleanup_error.owner_pool = owners
            raise cleanup_error from None
        if isinstance(error, Exception) and not isinstance(error, ModelLoadFailure):
            raise ModelLoadFailure("child_exit") from None
        raise


async def load_startup_model(
    catalog_entry: ModelCatalogEntry,
    configuration: MlflowConfiguration,
    owner_pool: WorkerPool,
) -> LoadedModel:
    # The thread only owns the bounded process supervisor, never the model loader.
    # Keep the event loop cancellable and wait for owned-child cleanup on cancellation.
    cancelled = Event()
    task = asyncio.create_task(
        asyncio.to_thread(
            load_model_in_isolated_process,
            catalog_entry,
            configuration.tracking_uri,
            configuration.startup_timeout_seconds,
            load_model_from_mlflow,
            cancelled=cancelled,
            owner_pool=owner_pool,
        )
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancelled.set()
        # Repeated shutdown cancellation must not detach the cleanup supervisor.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except ModelLoadFailure:
                break
        # Retrieve a completed failure without leaking an unobserved task exception.
        if task.done() and not task.cancelled():
            error = task.exception()
            if error is None:
                # Cancellation can race a successfully completed bootstrap.
                close_loaded_model(task.result())
            elif isinstance(error, ModelLoadFailure) and error.reason == "cleanup_unverified":
                raise error
        raise


@asynccontextmanager
async def model_lifespan(app: FastAPI) -> AsyncIterator[None]:
    configuration = read_mlflow_configuration()
    app.state.mlflow_configuration = configuration
    try:
        for model_id, catalog_entry in MODEL_CATALOG.items():
            logger.info("model_startup_attempt", extra={"model_id": model_id})
            try:
                loaded = await load_startup_model(
                    catalog_entry, configuration, app.state.worker_pool
                )
                app.state.model_registry[model_id] = loaded
            except ModelLoadFailure as error:
                app.state.model_registry[model_id] = None
                if error.reason == "cleanup_unverified":
                    raise
                logger.warning(
                    "model_startup_failed", extra={"model_id": model_id, "reason": error.reason}
                )
            else:
                logger.info("model_startup_succeeded", extra={"model_id": model_id})
        yield
    finally:
        # The publication lock prevents a refresh from publishing after shutdown.
        with app.state.model_registry_lock:
            app.state.stopping = True
            app.state.refresh_cancelled.set()
            app.state.worker_pool.begin_shutdown()
            app.state.model_registry = {model_id: None for model_id in MODEL_CATALOG}
        failures = []
        if app.state.model_refresh_lock.acquire(timeout=0.5):
            app.state.model_refresh_lock.release()
        else:
            failures.append(ModelLoadFailure("cleanup_unverified"))
        try:
            app.state.worker_pool.close()
        except ModelLoadFailure as error:
            failures.append(error)
        if failures:
            raise ModelLoadFailure("cleanup_unverified")


def close_loaded_model(loaded: LoadedModel) -> None:
    if isinstance(loaded.model, ModelProxy):
        loaded.model.close()


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
    application.state.worker_pool = WorkerPool(3 * len(MODEL_CATALOG))
    application.state.owned_workers = application.state.worker_pool.workers
    application.state.stopping = False
    application.state.refresh_cancelled = Event()
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
        loaded = request.app.state.model_registry[model_id]
    # Availability can wait for cleanup; never nest that lock under publication.
    if loaded is not None and isinstance(loaded.model, ModelProxy) and not loaded.model.available:
        return None
    return loaded


@contextmanager
def prediction_snapshot(request: Request, model_id: str):
    with request.app.state.model_registry_lock:
        loaded = request.app.state.model_registry[model_id]
        if loaded is None:
            raise HTTPException(status_code=503, detail="Model loader is unavailable")
        proxy = loaded.model if isinstance(loaded.model, ModelProxy) else None
        if proxy is not None:
            try:
                proxy.acquire()
            except ModelLoadFailure:
                raise HTTPException(
                    status_code=503, detail="Model inference is unavailable"
                ) from None
    try:
        yield loaded
    finally:
        if proxy is not None:
            try:
                proxy.release()
            except ModelLoadFailure:
                logger.error("model_cleanup_failed", extra={"reason": "cleanup_unverified"})
                raise HTTPException(
                    status_code=503, detail="Model inference is unavailable"
                ) from None


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


@router.post("/admin/models/{model_id}/refresh", response_model=ModelHealthResponse)
def refresh_model(model_id: str, request: Request) -> ModelHealthResponse:
    require_administrative_credential(request)
    if model_id not in MODEL_CATALOG:
        raise HTTPException(status_code=404, detail="Model not found")

    logger.info("model_refresh_attempt", extra={"model_id": model_id})
    if not request.app.state.model_refresh_lock.acquire(blocking=False):
        logger.warning(
            "model_refresh_failed", extra={"model_id": model_id, "reason": "refresh_busy"}
        )
        raise HTTPException(status_code=503, detail="Model refresh is unavailable")
    try:
        try:
            with request.app.state.model_registry_lock:
                if request.app.state.stopping:
                    raise ModelLoadFailure("cancelled")
            loaded_model = load_model_in_isolated_process(
                MODEL_CATALOG[model_id],
                request.app.state.mlflow_configuration.tracking_uri,
                request.app.state.mlflow_configuration.refresh_timeout_seconds,
                load_model_from_mlflow,
                cancelled=request.app.state.refresh_cancelled,
                owner_pool=request.app.state.worker_pool,
            )
        except ModelLoadFailure as error:
            logger.warning(
                "model_refresh_failed",
                extra={"model_id": model_id, "reason": error.reason},
            )
            raise HTTPException(status_code=503, detail="Model refresh is unavailable") from None
        with request.app.state.model_registry_lock:
            stopping = request.app.state.stopping
            old = request.app.state.model_registry[model_id]
            if not stopping:
                request.app.state.model_registry[model_id] = loaded_model
        if stopping:
            try:
                close_loaded_model(loaded_model)
            except ModelLoadFailure:
                logger.error("model_cleanup_failed", extra={"reason": "cleanup_unverified"})
            raise HTTPException(status_code=503, detail="Model refresh is unavailable")
        if old is not None and isinstance(old.model, ModelProxy):
            old.model.retire()
        logger.info("model_refresh_succeeded", extra={"model_id": model_id})
    finally:
        request.app.state.model_refresh_lock.release()
    return model_health_response(model_id, loaded_model)


@router.post("/models/{model_id}/predict", response_model=PredictResponse)
def predict(model_id: str, request: PredictRequest, app_request: Request) -> PredictResponse:
    if model_id not in MODEL_CATALOG:
        raise HTTPException(status_code=404, detail="Model not found")
    with prediction_snapshot(app_request, model_id) as loaded_model:
        try:
            probability = positive_class_probability(
                loaded_model.model, request.as_training_features()
            )
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=503, detail="Model inference is unavailable") from None
        return PredictResponse(
            model_id=model_id,
            model_version=loaded_model.model_version,
            run_id=loaded_model.run_id,
            probability=probability,
        )


app = create_app()
