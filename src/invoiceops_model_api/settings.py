"""Service configuration read from the environment. Fails closed when a secret is missing."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isnan
from urllib.parse import urlsplit

MIN_SERVICE_TOKEN_LENGTH = 32
# 128 bits of entropy: 32 hex chars, or 22 base64/base64url chars (22 * 6 = 132 bits).
MIN_SEED_HEX_CHARS = 32
MIN_SEED_BASE64_CHARS = 22
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_B64_RE = re.compile(r"^[A-Za-z0-9+/_-]+={0,2}$")


class ConfigError(RuntimeError):
    """The service is not safely configured; it must not start."""


def _number(env: Mapping[str, str], name: str, default: float, *, integer: bool = False) -> float:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw) if integer else float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number") from None
    if isnan(value) or value <= 0:
        raise ConfigError(f"{name} must be a positive number")
    return value


def derive_validation_seed(secret: str) -> int:
    """Hash a high-entropy secret into the integer seed of the official validation set.

    The seed is 128+ bits so the validation set cannot be recovered by enumerating seeds from
    the (rounded) metrics a deploy check returns.
    """
    secret = secret.strip()
    if _HEX_RE.fullmatch(secret):  # hex (or all digits): 4 bits per char
        long_enough = len(secret) >= MIN_SEED_HEX_CHARS
    else:  # base64/base64url: 6 bits per char
        long_enough = bool(_B64_RE.fullmatch(secret)) and len(secret) >= MIN_SEED_BASE64_CHARS
    # Cheap sanity check against obvious placeholders such as "aaaa..." or "abab...".
    if not long_enough or len(set(secret)) < 10:
        raise ConfigError(
            "OFFICIAL_VALIDATION_SEED must be a secret with at least 128 bits of entropy "
            f"({MIN_SEED_HEX_CHARS}+ hex chars or {MIN_SEED_BASE64_CHARS}+ base64 chars, "
            "e.g. `openssl rand -hex 32`)"
        )
    digest = hashlib.sha256(b"invoiceops-official-validation-seed-v1:" + secret.encode()).digest()
    return int.from_bytes(digest, "big")


def validate_tracking_uri(uri: str) -> str:
    """The tracking server is the ONLY host artifacts are ever downloaded from."""
    try:
        parts = urlsplit(uri)
        host = parts.hostname
    except ValueError:
        raise ConfigError("MLFLOW_TRACKING_URI is not a valid URL") from None
    if parts.scheme not in ("http", "https") or not host or parts.query or parts.fragment:
        raise ConfigError("MLFLOW_TRACKING_URI must be an http(s) URL of the MLflow server")
    return uri.rstrip("/")


@dataclass(frozen=True)
class Settings:
    service_token: str = field(repr=False)
    tracking_uri: str
    # None disables the gate: deploy-check/load then fail closed with GATE_UNAVAILABLE.
    official_validation_seed: int | None = field(default=None, repr=False)
    gate_min_auc: float = 0.70
    gate_min_lift: float = 0.02
    official_validation_rows: int = 4000
    max_deployments: int = 30
    max_model_bytes: int = 64 * 1024 * 1024
    max_skops_uncompressed_bytes: int = 256 * 1024 * 1024
    max_batch_items: int = 100
    max_prior_invoices: int = 5000
    # Total prior invoices accepted in ONE batch request (items x priors would otherwise allow
    # 100 x 5000 rows of work and ~tens of MB of JSON) and the largest request body at all.
    max_total_priors: int = 20_000
    max_body_bytes: int = 4 * 1024 * 1024
    # MLflow/gateway traffic is bounded explicitly (the MLflow client defaults are 7 retries and a
    # 120 s timeout, i.e. minutes of blocked threads per call against a failing server):
    # per-request timeout, retries with backoff, a budget for ALL calls of one deploy check and
    # a cap on concurrent artifact downloads (excess callers wait briefly, then get 503).
    mlflow_max_retries: int = 2
    mlflow_backoff_factor: float = 0.5
    mlflow_request_timeout_seconds: float = 15.0
    mlflow_total_budget_seconds: float = 45.0
    max_concurrent_downloads: int = 2
    download_wait_seconds: float = 10.0
    # Process isolation (see isolation.py / inference.py). Student-controlled model bytes are
    # only ever parsed inside child processes bounded by these limits.
    check_timeout_seconds: float = 60.0
    check_memory_mb: int = 1024
    max_concurrent_checks: int = 2
    # A check waits at most this long for a free slot, then the caller gets 503 BUSY.
    check_wait_seconds: float = 10.0
    inference_timeout_seconds: float = 10.0
    inference_memory_mb: int = 1024
    quarantine_after_crashes: int = 2
    # How the service authenticates to the tracking URI. "gateway": Authorization Bearer
    # GATEWAY_SERVICE_READ_TOKEN + X-InvoiceOps-Workspace (MLOPS-02 gateway). "mlflow": MLflow's
    # native credentials from the environment (MLFLOW_TRACKING_TOKEN or USERNAME/PASSWORD) +
    # X-MLFLOW-WORKSPACE, for a direct MLflow server / local tests. Either way credentials only
    # ever go to tracking_uri.
    mlflow_auth_mode: str = "mlflow"
    gateway_read_token: str | None = field(default=None, repr=False)
    # Dedicated HMAC key of the audit hash (MODEL_API_AUDIT_HMAC_KEY). None: derived from the
    # service token (see ``audit_key``).
    audit_hmac_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        validate_tracking_uri(self.tracking_uri)
        if not 0 <= self.mlflow_max_retries <= 2:
            raise ConfigError("MODEL_API_MLFLOW_MAX_RETRIES must be between 0 and 2")
        if not 1 <= self.mlflow_request_timeout_seconds <= 30:
            raise ConfigError("MODEL_API_MLFLOW_TIMEOUT_SECONDS must be between 1 and 30")
        if self.mlflow_total_budget_seconds < self.mlflow_request_timeout_seconds:
            raise ConfigError(
                "MODEL_API_MLFLOW_BUDGET_SECONDS must be at least the request timeout"
            )
        if self.mlflow_auth_mode not in ("mlflow", "gateway"):
            raise ConfigError("MODEL_API_MLFLOW_AUTH must be 'mlflow' or 'gateway'")
        if self.mlflow_auth_mode == "gateway" and len(self.gateway_read_token or "") < (
            MIN_SERVICE_TOKEN_LENGTH
        ):
            raise ConfigError(
                "GATEWAY_SERVICE_READ_TOKEN (>= 32 chars) is required when "
                "MODEL_API_MLFLOW_AUTH=gateway"
            )

    @property
    def audit_key(self) -> bytes:
        """Key of the audit digest: the dedicated one, else derived from the service token.

        Never a constant: ``HMAC-SHA-256(service token, "invoiceops-model-api/audit-key/v1")`` is
        different per deployment and is not the token itself.
        """
        if self.audit_hmac_key:
            return self.audit_hmac_key.encode()
        return hmac.new(
            self.service_token.encode(), b"invoiceops-model-api/audit-key/v1", hashlib.sha256
        ).digest()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        token = (env.get("MODEL_API_SERVICE_TOKEN") or "").strip()
        if len(token) < MIN_SERVICE_TOKEN_LENGTH:
            raise ConfigError(
                f"MODEL_API_SERVICE_TOKEN must be set and at least {MIN_SERVICE_TOKEN_LENGTH} chars"
            )
        uri = (env.get("MLFLOW_TRACKING_URI") or "").strip()
        if not uri:
            raise ConfigError("MLFLOW_TRACKING_URI must be configured")
        seed_raw = (env.get("OFFICIAL_VALIDATION_SEED") or "").strip()
        if not seed_raw:
            raise ConfigError("OFFICIAL_VALIDATION_SEED must be configured")
        seed = derive_validation_seed(seed_raw)
        gateway_token = (env.get("GATEWAY_SERVICE_READ_TOKEN") or "").strip() or None
        auth_mode = (env.get("MODEL_API_MLFLOW_AUTH") or "").strip().lower() or (
            "gateway" if gateway_token else "mlflow"
        )
        audit_key = (env.get("MODEL_API_AUDIT_HMAC_KEY") or "").strip() or None
        if audit_key is not None and len(audit_key) < MIN_SERVICE_TOKEN_LENGTH:
            raise ConfigError(
                f"MODEL_API_AUDIT_HMAC_KEY must be at least {MIN_SERVICE_TOKEN_LENGTH} chars"
            )
        min_auc = _number(env, "GATE_MIN_AUC", 0.70)
        if min_auc >= 1:
            raise ConfigError("GATE_MIN_AUC must be below 1")
        return cls(
            service_token=token,
            tracking_uri=validate_tracking_uri(uri),
            official_validation_seed=seed,
            gate_min_auc=min_auc,
            gate_min_lift=_number(env, "GATE_MIN_LIFT", 0.02),
            official_validation_rows=int(
                _number(env, "OFFICIAL_VALIDATION_ROWS", 4000, integer=True)
            ),
            max_deployments=int(_number(env, "MAX_LOADED_DEPLOYMENTS", 30, integer=True)),
            max_model_bytes=int(_number(env, "MAX_MODEL_BYTES", 64 * 1024 * 1024, integer=True)),
            check_timeout_seconds=_number(env, "MODEL_CHECK_TIMEOUT_SECONDS", 60.0),
            check_memory_mb=int(_number(env, "MODEL_CHECK_MEMORY_MB", 1024, integer=True)),
            max_concurrent_checks=int(_number(env, "MODEL_CHECK_CONCURRENCY", 2, integer=True)),
            check_wait_seconds=_number(env, "MODEL_CHECK_WAIT_SECONDS", 10.0),
            inference_timeout_seconds=_number(env, "MODEL_INFERENCE_TIMEOUT_SECONDS", 10.0),
            inference_memory_mb=int(_number(env, "MODEL_WORKER_MEMORY_MB", 1024, integer=True)),
            max_total_priors=int(_number(env, "MAX_TOTAL_PRIORS", 20_000, integer=True)),
            max_body_bytes=int(_number(env, "MAX_BODY_BYTES", 4 * 1024 * 1024, integer=True)),
            mlflow_max_retries=int(_number(env, "MODEL_API_MLFLOW_MAX_RETRIES", 2, integer=True)),
            mlflow_request_timeout_seconds=_number(env, "MODEL_API_MLFLOW_TIMEOUT_SECONDS", 15.0),
            mlflow_total_budget_seconds=_number(env, "MODEL_API_MLFLOW_BUDGET_SECONDS", 45.0),
            max_concurrent_downloads=int(_number(env, "MAX_CONCURRENT_DOWNLOADS", 2, integer=True)),
            mlflow_auth_mode=auth_mode,
            gateway_read_token=gateway_token,
            audit_hmac_key=audit_key,
        )
