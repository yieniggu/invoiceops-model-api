"""InvoiceOps Model API: internal, multi-model scoring service.

Every route except ``GET /health`` (liveness) requires ``Authorization: Bearer <service
token>``. Models are loaded by deployment from a fixed MLflow model version. No invoice
data or feature values are ever logged.

This (HTTP) process NEVER imports, unpickles or calls a student model: deploy checks run in a
one-shot limited child process and predictions in a supervised inference worker process.
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import hmac
import json
import logging
import sys
import threading
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime
from hmac import compare_digest
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from invoiceops_ml.contract import FEATURE_SCHEMA_VERSION, FEATURES, compute_features
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from invoiceops_model_api.artifacts import (
    ArtifactError,
    DownloadBusy,
    DownloadGate,
    MlflowUnavailable,
    ModelRef,
    fetch_model,
    validate_ref,
)
from invoiceops_model_api.checks import CheckResult, GateUnavailable, Reason, run_check
from invoiceops_model_api.inference import (
    InferenceFailed,
    InferenceSupervisor,
    InferenceUnavailable,
)
from invoiceops_model_api.isolation import CheckBusy, CheckRunner
from invoiceops_model_api.logs import configure_logging, log_value
from invoiceops_model_api.procutil import make_undumpable, start_zombie_reaper
from invoiceops_model_api.registry import (
    CapacityExceeded,
    DeploymentRegistry,
    LoadedDeployment,
    now_iso,
)
from invoiceops_model_api.settings import Settings

logger = logging.getLogger("invoiceops_model_api")

DeploymentId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
# Largest invoice amount accepted (pesos). Far above any real invoice, far below float overflow.
MAX_AMOUNT_PESOS = 10**13
SupplierName = Annotated[str, StringConstraints(min_length=1, max_length=200)]
AmountPesos = Annotated[int, Field(ge=0, le=MAX_AMOUNT_PESOS)]
BusinessStatus = Annotated[str, StringConstraints(min_length=1, max_length=64)]
MUTATING_METHODS = frozenset({"PUT", "DELETE"})


class ModelReference(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    workspace: str
    registered_model: str = Field(alias="registeredModel")
    version: str

    def to_ref(self) -> ModelRef:
        return ModelRef(self.workspace, self.registered_model, self.version)


class InvoiceIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    supplierName: SupplierName
    amountPesos: AmountPesos
    issuedOn: date
    dueOn: date
    createdAt: datetime
    businessStatus: BusinessStatus | None = None


DECIDED_AT_REJECTED = "/predict acepta solo estados vigentes"
# Validation error types whose fixed message is safe to return (never a submitted value).
SAFE_ERROR_MESSAGES = frozenset({"decided_at_not_accepted"})


class PriorInvoiceIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def reject_decided_at(cls, data: Any) -> Any:
        # The caller must send each prior with its status AS OF the invoice's createdAt;
        # silently ignoring ``decidedAt`` would score a different history than the caller meant.
        if isinstance(data, dict) and "decidedAt" in data:
            raise PydanticCustomError("decided_at_not_accepted", DECIDED_AT_REJECTED)
        return data

    supplierName: SupplierName
    amountPesos: AmountPesos
    createdAt: datetime
    businessStatus: BusinessStatus


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    invoice: InvoiceIn
    prior_invoices: list[PriorInvoiceIn] = Field(
        default_factory=list, alias="priorInvoices", max_length=10_000
    )


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PredictRequest]


class BodyLimitMiddleware:
    """Refuses request bodies over ``limit`` bytes (declared or streamed) with 413."""

    def __init__(self, app: Callable[..., Awaitable[None]], limit: int) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.limit:
            await self._refuse(send)
            return
        total = 0
        started = False
        refused = False

        async def limited_receive() -> Any:
            nonlocal total, refused
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.limit:
                    # Answer 413 ourselves and tell the application the client is gone: the
                    # framework would otherwise turn a failed body read into a 400.
                    if not started and not refused:
                        refused = True
                        await self._refuse(send)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Any) -> None:
            nonlocal started
            if refused:
                return  # the 413 is already on the wire
            started = True
            await send(message)

        await self.app(scope, limited_receive, guarded_send)

    async def _refuse(self, send: Any) -> None:
        body = json.dumps(
            {"detail": {"code": "PAYLOAD_TOO_LARGE", "message": "Request body too large"}}
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


KNOWN_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})
UNMATCHED_ROUTE = "unmatched"


class RequestCounters:
    """Requests by route template and status (thread-safe, in-memory, reset on restart).

    The key space is bounded whatever a client sends: unknown HTTP methods share ``OTHER`` and
    every path that matches no route shares ``unmatched``.
    """

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def record(self, method: str, route: str, status: int) -> None:
        method = method.upper()
        if method not in KNOWN_METHODS:
            method = "OTHER"
        with self._lock:
            self._counts[f"{method} {route or UNMATCHED_ROUTE} {status}"] += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(sorted(self._counts.items()))


def principal_hash(token: str, key: bytes) -> str:
    """Non-reversible, log-safe identifier of the caller (the service credential).

    HMAC-SHA-256 under ``Settings.audit_key`` (a dedicated key or one derived from the service
    token, never a constant in the source), truncated to 64 bits: it correlates the requests of
    one credential across log lines and reveals nothing usable about the secret.
    """
    return hmac.new(key, token.encode(), hashlib.sha256).hexdigest()[:16]


def error(status: int, code: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message, **extra})


def report(result: CheckResult, ref: ModelRef, run_id: str | None) -> dict[str, Any]:
    return {
        "passed": result.passed,
        "reasons": [r.as_dict() for r in result.reasons],
        "metrics": result.metrics,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "model": {
            "workspace": ref.workspace,
            "registeredModel": ref.registered_model,
            "version": ref.version,
            "runId": run_id,
        },
    }


@dataclass
class CheckOutcome:
    result: CheckResult
    run_id: str | None
    slot: str | None = None  # staged artifact (only when requested and the check passed)
    staged: Path | None = None


def check_reference(
    settings: Settings,
    ref: ModelRef,
    runner: CheckRunner,
    supervisor: InferenceSupervisor | None = None,
    gate: DownloadGate | None = None,
) -> CheckOutcome:
    """Download + validate one version. With ``supervisor``, a passing model's bytes are staged."""
    try:
        validate_ref(ref)
        with fetch_model(ref, settings, gate) as fetched:
            result = run_check(fetched, settings, runner)
            if result.passed and supervisor is not None:
                slot, staged = supervisor.stage(fetched.skops_path)
                return CheckOutcome(result, fetched.run_id, slot, staged)
            return CheckOutcome(result, fetched.run_id)
    except ArtifactError as e:
        return CheckOutcome(CheckResult(False, [Reason(e.code, str(e), e.detail)]), None)
    except DownloadBusy:
        raise error(503, "BUSY", "Too many model downloads in progress; retry shortly") from None
    except MlflowUnavailable:
        raise error(502, "MLFLOW_UNAVAILABLE", "MLflow is unavailable") from None
    except GateUnavailable:
        raise error(503, "GATE_UNAVAILABLE", "The validation gate is not available") from None
    except CheckBusy:
        raise error(503, "BUSY", "Too many model checks in progress; retry shortly") from None
    except OSError:
        # Disk full / tmpfs exhausted / cannot spawn: the gate cannot run, so it fails closed.
        logger.error("check_storage_unavailable id=%s", log_value(ref.registered_model))
        raise error(
            503, "CHECK_UNAVAILABLE", "The model check could not run; retry later"
        ) from None


def feature_row(item: PredictRequest) -> list[float]:
    invoice = item.invoice.model_dump()
    priors = [p.model_dump() for p in item.prior_invoices]
    try:
        features = compute_features(invoice, priors)
    except (ValueError, OverflowError):
        raise error(422, "INVALID_INVOICE", "Invoice or prior invoice fields are invalid") from None
    return [float(features[f]) for f in FEATURES]


ON_LINUX = sys.platform.startswith("linux")


def apply_process_hardening() -> bool:
    """Make this process non-dumpable so the secrets in its environment stay out of children's reach.

    On Linux a False result means the protection is NOT active (the container then relies on the
    other layers only): it is logged as an error and exposed as ``processHardened`` in /metrics.
    """
    applied = make_undumpable()
    if not applied and ON_LINUX:
        logger.error(
            "process_hardening_failed prctl_set_dumpable=0 secrets_readable_by_children=true"
        )
    return applied


def create_app(
    settings: Settings | None = None,
    *,
    check_runner: CheckRunner | None = None,
    supervisor: InferenceSupervisor | None = None,
) -> FastAPI:
    """Build the app. ``check_runner``/``supervisor`` are seams for tests (fault injection)."""
    settings = settings or Settings.from_env()
    configure_logging()  # INFO to stdout: audit, deploy_check and deployment_* lines
    hardened = apply_process_hardening()
    start_zombie_reaper()  # no-op unless this process is PID 1 (production: no docker-init)
    registry = DeploymentRegistry(settings.max_deployments)
    download_gate = DownloadGate(settings.max_concurrent_downloads, settings.download_wait_seconds)
    counters = RequestCounters()
    runner = check_runner or CheckRunner(settings)
    supervisor = supervisor or InferenceSupervisor(settings)
    supervisor.on_quarantine = registry.quarantine
    audit_key = settings.audit_key
    # Deploy checks and loads block (download, check child, activation). They run on their OWN
    # bounded pool, never on the default threadpool that serves /predict, and at most
    # ``2 x workers`` of them are admitted at once; the rest get 503 BUSY immediately.
    workers = max(2, settings.max_concurrent_checks + settings.max_concurrent_downloads + 2)
    check_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="check")
    admission = threading.BoundedSemaphore(workers * 2)

    async def offload(function: Callable[..., Any], *args: Any) -> Any:
        if not admission.acquire(blocking=False):
            raise error(503, "BUSY", "Too many model checks in progress; retry shortly")

        def guarded() -> Any:
            try:
                return function(*args)
            finally:
                admission.release()

        try:
            future = check_executor.submit(guarded)
        except BaseException:
            admission.release()
            raise
        return await asyncio.wrap_future(future)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        check_executor.shutdown(wait=False, cancel_futures=True)
        supervisor.shutdown()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.settings = settings
    app.state.registry = registry
    app.state.supervisor = supervisor
    app.state.counters = counters
    app.state.process_hardened = hardened
    app.state.check_executor = check_executor
    app.add_middleware(BodyLimitMiddleware, limit=settings.max_body_bytes)
    atexit.register(supervisor.shutdown)  # also covers processes without a lifespan (tests)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Never echo the submitted values (they may be invoice data).
        issues = [
            {
                "loc": [str(p) for p in e["loc"]],
                "type": e["type"],
                **({"message": e["msg"]} if e["type"] in SAFE_ERROR_MESSAGES else {}),
            }
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422, content={"detail": {"code": "INVALID_REQUEST", "issues": issues}}
        )

    def require_service(request: Request) -> None:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        expected = settings.service_token
        if (
            len(expected) < 32
            or scheme.lower() != "bearer"
            or not compare_digest(token.encode(), expected.encode())
        ):
            raise HTTPException(status_code=401, detail={"code": "UNAUTHORIZED"})

    @app.middleware("http")
    async def observe(request: Request, call_next: Callable[[Request], Awaitable[Any]]) -> Any:
        status = 500  # what an unhandled exception becomes; replaced once a response exists
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            template = getattr(route, "path", UNMATCHED_ROUTE)
            counters.record(request.method, template, status)
            if request.method in MUTATING_METHODS:
                # Audit trail of mutations: who (non-reversible digest), what, outcome.
                header = request.headers.get("authorization", "")
                token = header.partition(" ")[2]
                who = principal_hash(token, audit_key) if status != 401 and token else "anonymous"
                logger.info(
                    "audit principal=%s method=%s route=%s status=%s",
                    who,
                    request.method,
                    template,
                    status,
                )

    router = APIRouter(dependencies=[Depends(require_service)])

    @app.get("/health")
    async def liveness() -> dict[str, str]:
        # async on purpose: it runs on the event loop, never behind the (possibly exhausted)
        # worker thread pool that serves the blocking routes.
        return {"status": "ok"}

    @router.get("/metrics")
    def metrics() -> dict[str, Any]:
        return {
            "status": "ok",
            "processHardened": hardened,
            "requests": counters.snapshot(),
        }

    @router.get("/deployments")
    def list_deployments() -> dict[str, Any]:
        items = registry.list()
        return {
            "status": "ok",
            "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
            "capacity": registry.capacity,
            "loaded": len(items),
            "deployments": [d.summary() for d in items],
            "quarantined": registry.quarantined(),
        }

    @router.post("/deploy-check")
    async def deploy_check(body: ModelReference) -> dict[str, Any]:
        return await offload(run_deploy_check, body)

    def run_deploy_check(body: ModelReference) -> dict[str, Any]:
        ref = body.to_ref()
        checked = check_reference(settings, ref, runner, gate=download_gate)
        result, run_id = checked.result, checked.run_id
        logger.info(
            "deploy_check model=%s version=%s passed=%s codes=%s",
            log_value(ref.registered_model),
            ref.version,
            result.passed,
            [r.code for r in result.reasons],
        )
        return report(result, ref, run_id)

    @router.put("/deployments/{deployment_id}")
    async def load_deployment(deployment_id: DeploymentId, body: ModelReference) -> dict[str, Any]:
        return await offload(run_load_deployment, deployment_id, body)

    def run_load_deployment(deployment_id: str, body: ModelReference) -> dict[str, Any]:
        ref = body.to_ref()
        if not registry.has_room_for(deployment_id):
            raise error(
                409,
                "CAPACITY_EXCEEDED",
                "Maximum loaded deployments reached",
                capacity=registry.capacity,
            )
        existing = registry.get(deployment_id)
        if existing is not None and existing.ref == ref:
            return existing.summary()
        checked = check_reference(settings, ref, runner, supervisor, download_gate)
        try:
            return activate_checked(deployment_id, ref, checked)
        except BaseException:
            # Every failure path (gate failed, worker unavailable, capacity, anything unexpected)
            # removes the staged bytes: a failed deploy never leaves a .skops file in the cache.
            if checked.slot is not None and checked.staged is not None:
                supervisor.unload(checked.slot, checked.staged)
            raise

    def activate_checked(
        deployment_id: str, ref: ModelRef, checked: CheckOutcome
    ) -> dict[str, Any]:
        result, run_id = checked.result, checked.run_id
        if (
            not result.passed
            or run_id is None
            or checked.slot is None
            or checked.staged is None
            or result.columns is None
        ):
            raise HTTPException(
                status_code=422, detail={"code": "GATE_FAILED", **report(result, ref, run_id)}
            )
        slot, staged = checked.slot, checked.staged
        loaded = LoadedDeployment(
            deployment_id=deployment_id,
            ref=ref,
            run_id=run_id,
            slot=slot,
            artifact_path=staged,
            columns=tuple(result.columns),
            positive_index=result.positive_index,
            metrics=result.metrics or {},
            loaded_at=now_iso(),
        )
        try:
            # Load into the inference worker BEFORE publishing; the old model keeps serving.
            supervisor.activate(slot, staged, loaded.columns)
        except (InferenceUnavailable, InferenceFailed):
            raise error(503, "INFERENCE_UNAVAILABLE", "The model could not be activated") from None
        try:
            replaced = registry.put(loaded)  # replaces atomically; the old model served until now
        except CapacityExceeded:
            raise error(
                409,
                "CAPACITY_EXCEEDED",
                "Maximum loaded deployments reached",
                capacity=registry.capacity,
            ) from None
        if replaced is not None:
            supervisor.unload(replaced.slot, replaced.artifact_path)
        logger.info(
            "deployment_loaded id=%s model=%s version=%s",
            deployment_id,
            log_value(ref.registered_model),
            ref.version,
        )
        return loaded.summary()

    @router.delete("/deployments/{deployment_id}")
    def unload_deployment(deployment_id: DeploymentId) -> dict[str, Any]:
        removed = registry.remove(deployment_id)
        if removed is not None:
            supervisor.unload(removed.slot, removed.artifact_path)
        logger.info("deployment_unloaded id=%s removed=%s", deployment_id, removed is not None)
        return {"deploymentId": deployment_id, "unloaded": removed is not None}

    def score(
        deployment: LoadedDeployment, items: list[PredictRequest]
    ) -> tuple[LoadedDeployment, list[float]]:
        """Score ``items``; returns the deployment that actually served them.

        A request that raced a redeploy of the same ``deploymentId`` can hold a slot that was just
        replaced (and unloaded). It is retried ONCE against the current slot instead of
        failing with a 503 for a deployment that is, in fact, loaded.
        """
        for item in items:
            if len(item.prior_invoices) > settings.max_prior_invoices:
                raise error(
                    422,
                    "TOO_MANY_PRIORS",
                    "Too many prior invoices",
                    max=settings.max_prior_invoices,
                )
        if sum(len(item.prior_invoices) for item in items) > settings.max_total_priors:
            raise error(
                422,
                "TOO_MANY_PRIORS",
                "Too many prior invoices in the request",
                max=settings.max_total_priors,
            )
        rows = [feature_row(item) for item in items]
        try:
            try:
                return deployment, supervisor.predict(deployment.slot, rows)
            except InferenceUnavailable:
                current = registry.get(deployment.deployment_id)
                if current is None or current.slot == deployment.slot:
                    raise
                logger.info("predict_retry_on_replaced_slot id=%s", deployment.deployment_id)
                return current, supervisor.predict(current.slot, rows)
        except InferenceUnavailable as unavailable:
            logger.error("inference_unavailable id=%s", deployment.deployment_id)
            raise error(
                503,
                "INFERENCE_UNAVAILABLE",
                "The inference worker is unavailable; retry shortly",
                **({"quarantined": True} if unavailable.quarantined else {}),
            ) from None
        except InferenceFailed:
            logger.error("inference_failed id=%s", deployment.deployment_id)
            raise error(503, "INFERENCE_FAILED", "Model inference failed") from None

    def prediction_payload(deployment: LoadedDeployment) -> dict[str, Any]:
        return {
            "deploymentId": deployment.deployment_id,
            "workspace": deployment.ref.workspace,
            "registeredModel": deployment.ref.registered_model,
            "modelVersion": deployment.ref.version,
            "runId": deployment.run_id,
            "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        }

    def get_loaded(deployment_id: str) -> LoadedDeployment:
        deployment = registry.get(deployment_id)
        if deployment is None:
            raise error(404, "DEPLOYMENT_NOT_LOADED", "Deployment is not loaded")
        return deployment

    @router.post("/deployments/{deployment_id}/predict")
    def predict(deployment_id: DeploymentId, body: PredictRequest) -> dict[str, Any]:
        deployment = get_loaded(deployment_id)
        deployment, (probability,) = score(deployment, [body])
        return {**prediction_payload(deployment), "probability": probability}

    @router.post("/deployments/{deployment_id}/predict-batch")
    def predict_batch(deployment_id: DeploymentId, body: BatchPredictRequest) -> dict[str, Any]:
        deployment = get_loaded(deployment_id)
        if not body.items or len(body.items) > settings.max_batch_items:
            raise error(
                422, "INVALID_BATCH_SIZE", "Batch size out of range", max=settings.max_batch_items
            )
        deployment, probabilities = score(deployment, body.items)
        return {**prediction_payload(deployment), "probabilities": probabilities}

    app.include_router(router)
    return app


def _default_app() -> FastAPI:
    return create_app()


def __getattr__(name: str) -> Any:
    # ``uvicorn invoiceops_model_api.app:app`` builds the app lazily, from the environment,
    # so importing this module (tests, tooling) never requires secrets.
    if name == "app":
        application = _default_app()
        globals()["app"] = application
        return application
    raise AttributeError(name)
