"""Download of a registered model version from MLflow, with workspace selection and limits.

Threat model: students fully control a registered model version, including its ``source`` URI
and the bytes of every artifact. Therefore:

* ALL traffic (version metadata, run lookup, artifact bytes) goes to the configured tracking
  server only, through one small client (``_Tracking``) that builds every URL from
  ``settings.tracking_uri``. The service never lets the MLflow client or ``requests`` resolve a
  host from a version's source. The source (as resolved server-side by ``get-download-uri``) is
  only *parsed*: ``mlflow-artifacts:`` with an empty netloc (or the tracking host's own) and a
  normalized path without ``..``; ``runs:/<run id>/<path>`` is resolved through the tracking
  server's own run record, in the deployment's workspace. Anything else (any other scheme,
  any foreign host, ``models:/name/version`` left unresolved...) is
  ``ARTIFACT_SOURCE_NOT_ALLOWED``.
* Credentials and the workspace header only ever go to the tracking URI, redirects are never
  followed, and every download is streamed with a hard byte cap enforced while reading
  (listing sizes are never trusted).
* Two auth modes (``Settings.mlflow_auth_mode``): ``gateway`` (``Authorization: Bearer
  <GATEWAY_SERVICE_READ_TOKEN>`` + ``X-InvoiceOps-Workspace``) for the MLOPS-02 gateway, and
  ``mlflow`` (MLflow's native env credentials + ``X-MLFLOW-WORKSPACE``) for a direct server.

The workspace is sent explicitly as a request header on every call, so no process-global MLflow
state (``WorkspaceContext``/``MLFLOW_WORKSPACE``) is touched and no lock is needed: concurrent
checks never serialize on MLflow, and nothing is held during load/predict/gate.
"""

from __future__ import annotations

import json
import re
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit

import yaml

from invoiceops_model_api.settings import Settings

WORKSPACE_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,255}$")
VERSION_RE = re.compile(r"^[1-9][0-9]{0,8}$")
MAX_MLMODEL_BYTES = 1024 * 1024
MAX_LISTING_BYTES = 1024 * 1024
DOWNLOAD_DEADLINE_SECONDS = 120.0
ARTIFACT_SCHEME = "mlflow-artifacts"
ARTIFACTS_ENDPOINT = "/api/2.0/mlflow-artifacts/artifacts"
GATEWAY_WORKSPACE_HEADER = "X-InvoiceOps-Workspace"
_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_FORBIDDEN_SUFFIXES = (".pkl", ".pickle", ".cloudpickle", ".joblib")
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._~@=+, -]{1,255}$")
_MAX_SOURCE_PATH = 1024


class ArtifactError(Exception):
    """A rejection with a stable code (surfaced as a deploy-check reason)."""

    def __init__(self, code: str, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail or {}


class MlflowUnavailable(Exception):
    """MLflow could not be reached or refused the service identity."""


class DownloadBusy(Exception):
    """Too many artifact downloads are already running; the caller should retry shortly."""


class DownloadGate:
    """Bounds concurrent artifact downloads (memory, disk and MLflow connections)."""

    def __init__(self, slots: int, wait_seconds: float) -> None:
        self._slots = threading.BoundedSemaphore(slots)
        self._wait = wait_seconds

    def acquire(self) -> None:
        if not self._slots.acquire(timeout=self._wait):
            raise DownloadBusy

    def release(self) -> None:
        self._slots.release()


@dataclass(frozen=True)
class ModelRef:
    workspace: str
    registered_model: str
    version: str


@dataclass(frozen=True)
class FetchedModel:
    ref: ModelRef
    run_id: str
    mlmodel: dict
    skops_path: Path


def validate_ref(ref: ModelRef) -> None:
    if not WORKSPACE_RE.fullmatch(ref.workspace) or not 2 <= len(ref.workspace) <= 63:
        raise ArtifactError("INVALID_REFERENCE", "Invalid workspace name")
    if not MODEL_NAME_RE.fullmatch(ref.registered_model):
        raise ArtifactError("INVALID_REFERENCE", "Invalid registered model name")
    if not VERSION_RE.fullmatch(ref.version):
        raise ArtifactError("INVALID_REFERENCE", "Version must be a fixed integer, not an alias")


def _not_allowed(message: str = "Artifact source is not allowed") -> ArtifactError:
    return ArtifactError("ARTIFACT_SOURCE_NOT_ALLOWED", message)


def validate_artifact_source(uri: str, tracking_uri: str, workspace: str | None = None) -> str:
    """Return the normalized relative artifact path of a version's source, or raise.

    Accepts ONLY ``mlflow-artifacts:`` URIs with an empty netloc, or a netloc exactly equal to
    the tracking server's. The returned path is only ever appended to the tracking URI; the
    host of ``uri`` is never contacted.
    """
    try:
        if not isinstance(uri, str) or not uri or len(uri) > 4096 or not uri.isprintable():
            raise _not_allowed()
        parts = urlsplit(uri)
        # Accessing hostname/port forces urllib's remaining validation (IPv6 brackets, ports).
        _ = parts.hostname, parts.port
        tracking = urlsplit(tracking_uri)
    except ValueError:
        raise _not_allowed("Artifact source is malformed") from None
    if parts.scheme.lower() != ARTIFACT_SCHEME:
        raise _not_allowed("Artifact source scheme is not allowed")
    if parts.netloc and parts.netloc.lower() != tracking.netloc.lower():
        raise _not_allowed("Artifact source host is not the tracking server")
    if parts.query or parts.fragment or ";" in parts.path or "%" in parts.path or "\\" in uri:
        raise _not_allowed("Artifact source is malformed")
    path = parts.path.strip("/")
    if not path or len(path) > _MAX_SOURCE_PATH:
        raise _not_allowed("Artifact source path is invalid")
    segments = path.split("/")
    if any(seg in ("", ".", "..") or not _SEGMENT_RE.fullmatch(seg) for seg in segments):
        raise _not_allowed("Artifact source path is invalid")
    if (
        workspace
        and segments[0] == "workspaces"
        and (len(segments) < 2 or segments[1] != workspace)
    ):
        raise _not_allowed("Artifact source points at another workspace")
    return "/".join(segments)


def _safe_name(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    if not base or base in (".", "..") or "\\" in base or "\x00" in base:
        raise ArtifactError("ARTIFACT_LAYOUT_INVALID", "Unsafe artifact name")
    return base


@dataclass(frozen=True)
class _Entry:
    name: str
    is_dir: bool
    size: int | None


class _Tracking:
    """The only network client: tracking server metadata + artifact proxy, for one workspace."""

    def __init__(self, settings: Settings, workspace: str) -> None:
        from mlflow.utils.credentials import get_default_host_creds
        from mlflow.utils.rest_utils import MlflowHostCreds
        from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME

        if settings.mlflow_auth_mode == "gateway":
            self._creds = MlflowHostCreds(
                host=settings.tracking_uri, token=settings.gateway_read_token
            )
            self._headers = {GATEWAY_WORKSPACE_HEADER: workspace}
        else:
            self._creds = get_default_host_creds(settings.tracking_uri)
            self._headers = {WORKSPACE_HEADER_NAME: workspace}
        self._settings = settings
        self._spent = 0.0  # seconds spent inside MLflow requests (retries and backoff included)

    def _get(self, endpoint: str, **kwargs):
        from mlflow.exceptions import MlflowException
        from mlflow.utils.rest_utils import http_request

        settings = self._settings
        remaining = settings.mlflow_total_budget_seconds - self._spent
        if remaining <= 0:
            raise MlflowUnavailable("request budget exhausted")
        started = time.monotonic()
        try:
            # Explicit bounds: the MLflow defaults (7 retries, 120 s timeout) would block a worker
            # thread for minutes per call when MLflow or the gateway keeps answering 5xx.
            response = http_request(
                self._creds,
                endpoint,
                "GET",
                max_retries=settings.mlflow_max_retries,
                backoff_factor=settings.mlflow_backoff_factor,
                backoff_jitter=0.0,
                timeout=min(settings.mlflow_request_timeout_seconds, remaining),
                extra_headers=self._headers,
                allow_redirects=False,  # a redirect would forward credentials/workspace
                **kwargs,
            )
        except MlflowException as error:
            raise MlflowUnavailable(type(error).__name__) from None
        finally:
            self._spent += time.monotonic() - started
        if 300 <= response.status_code < 400:
            response.close()
            raise _not_allowed("The tracking server redirected the request")
        return response

    def _json(self, endpoint: str, params: dict) -> tuple[int, dict]:
        response = self._get(endpoint, params=params, stream=True)
        try:
            body = _read_capped(
                response,
                MAX_LISTING_BYTES,
                time.monotonic() + self._settings.mlflow_request_timeout_seconds,
            )
            status = response.status_code
        finally:
            response.close()
        try:
            data = json.loads(body) if body else {}
        except ValueError:
            data = {}
        return status, data if isinstance(data, dict) else {}

    # ---- metadata ---------------------------------------------------------------------

    def version(self, ref: ModelRef) -> tuple[str, str]:
        """(run id, server-resolved download URI) of one fixed version."""
        params = {"name": ref.registered_model, "version": ref.version}
        status, data = self._json("/api/2.0/mlflow/model-versions/get", params)
        if status == 404 or data.get("error_code") == "RESOURCE_DOES_NOT_EXIST":
            raise ArtifactError("MODEL_VERSION_NOT_FOUND", "Registered model version not found")
        if status != 200:
            raise MlflowUnavailable(f"HTTP {status}")
        run_id = str((data.get("model_version") or {}).get("run_id") or "")
        status, data = self._json("/api/2.0/mlflow/model-versions/get-download-uri", params)
        if status == 404:
            raise ArtifactError("MODEL_VERSION_NOT_FOUND", "Registered model version not found")
        uri = data.get("artifact_uri")
        if status != 200 or not isinstance(uri, str):
            raise MlflowUnavailable(f"HTTP {status}")
        return run_id, uri

    def run_artifact_uri(self, run_id: str) -> str:
        status, data = self._json("/api/2.0/mlflow/runs/get", {"run_id": run_id})
        if status == 404:
            raise _not_allowed("The run is not in this workspace")
        uri = ((data.get("run") or {}).get("info") or {}).get("artifact_uri")
        if status != 200 or not isinstance(uri, str):
            raise MlflowUnavailable(f"HTTP {status}")
        return uri

    def artifact_path(self, source: str, settings: Settings, workspace: str) -> str:
        """Normalized relative artifact path of a version's source (never a host)."""
        if source.lower().startswith("runs:/"):
            return self._runs_source(source, settings, workspace)
        return validate_artifact_source(source, settings.tracking_uri, workspace)

    def _runs_source(self, source: str, settings: Settings, workspace: str) -> str:
        # runs:/<run id>/<path>, resolved through THIS tracking server in THIS workspace.
        rest = source[len("runs:/") :]
        run_id, _, sub = rest.partition("/")
        segments = [seg for seg in sub.split("/")] if sub else []
        if (
            not _RUN_ID_RE.fullmatch(run_id)
            or not source.isprintable()
            or any(seg in ("", ".", "..") or not _SEGMENT_RE.fullmatch(seg) for seg in segments)
        ):
            raise _not_allowed("Artifact source is malformed")
        base = validate_artifact_source(
            self.run_artifact_uri(run_id), settings.tracking_uri, workspace
        )
        return "/".join([base, *segments])

    # ---- artifacts --------------------------------------------------------------------

    def list(self, path: str) -> dict[str, _Entry]:
        status, data = self._json(ARTIFACTS_ENDPOINT, {"path": path})
        if status != 200:
            raise MlflowUnavailable(f"HTTP {status}")
        try:
            entries: dict[str, _Entry] = {}
            for info in data.get("files", []):
                name = _safe_name(str(info["path"]))
                size = int(info["file_size"]) if "file_size" in info else None
                entries[name] = _Entry(name, bool(info.get("is_dir")), size)
        except (ValueError, TypeError, KeyError, AttributeError):
            raise ArtifactError("ARTIFACT_LAYOUT_INVALID", "Unreadable artifact listing") from None
        return entries

    def download(self, path: str, name: str, dest: Path, max_bytes: int) -> Path:
        """Stream ``path/name`` to ``dest/name``; abort as soon as ``max_bytes`` is exceeded."""
        quoted = quote(f"{path}/{name}", safe="/")
        response = self._get(f"{ARTIFACTS_ENDPOINT}/{quoted}", stream=True)
        target = dest / name
        try:
            if response.status_code == 404:
                raise ArtifactError("ARTIFACT_LAYOUT_INVALID", f"{name} is missing")
            if response.status_code != 200:
                raise MlflowUnavailable(f"HTTP {response.status_code}")
            declared = response.headers.get("Content-Length", "")
            if declared.isdigit() and int(declared) > max_bytes:
                raise ArtifactError("ARTIFACT_TOO_LARGE", f"{name} exceeds the size limit")
            deadline = time.monotonic() + DOWNLOAD_DEADLINE_SECONDS
            with open(target, "wb") as out:
                _read_capped(response, max_bytes, deadline, out)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        finally:
            response.close()
        return target


def _read_capped(response, max_bytes: int, deadline: float, out=None) -> bytes:
    """Read the (decoded) body, never accepting more than ``max_bytes``; sizes are not trusted."""
    import requests

    total = 0
    chunks: list[bytes] = []
    try:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise ArtifactError("ARTIFACT_TOO_LARGE", "Artifact exceeds the size limit")
            if time.monotonic() > deadline:
                raise MlflowUnavailable("download deadline exceeded")
            if out is not None:
                out.write(chunk)
            else:
                chunks.append(chunk)
    except requests.exceptions.RequestException as error:
        raise MlflowUnavailable(type(error).__name__) from None
    return b"".join(chunks)


@contextmanager
def fetch_model(
    ref: ModelRef, settings: Settings, gate: DownloadGate | None = None
) -> Iterator[FetchedModel]:
    """Download ``MLmodel`` and ``model.skops`` of one fixed version into a temp dir.

    The temp dir is removed on exit. Raises ``ArtifactError`` for content problems and
    ``MlflowUnavailable`` for transport/auth problems.
    """
    validate_ref(ref)
    if gate is not None:
        gate.acquire()  # raises DownloadBusy; released as soon as the bytes are on disk
    held = gate is not None
    try:
        with _download(ref, settings) as fetched:
            if held:
                gate.release()  # type: ignore[union-attr]
                held = False
            yield fetched
    finally:
        if held:
            gate.release()  # type: ignore[union-attr]


@contextmanager
def _download(ref: ModelRef, settings: Settings) -> Iterator[FetchedModel]:
    tracking = _Tracking(settings, ref.workspace)
    try:
        run_id, uri = tracking.version(ref)
        path = tracking.artifact_path(uri, settings, ref.workspace)
    except ValueError:
        # Defensive: any parser ValueError on a student-controlled source is a rejection.
        raise _not_allowed("Artifact source is malformed") from None
    with tempfile.TemporaryDirectory(prefix="invoiceops-model-") as tmp:
        root = Path(tmp).resolve()
        files = tracking.list(path)
        for name in files:
            if name.lower().endswith(_FORBIDDEN_SUFFIXES):
                raise ArtifactError(
                    "PICKLE_FORMAT_NOT_ALLOWED", "Pickle-based artifacts are not allowed"
                )
        for required in ("MLmodel", "model.skops"):
            if required not in files or files[required].is_dir:
                raise ArtifactError("ARTIFACT_LAYOUT_INVALID", f"{required} is missing")
        mlmodel_path = tracking.download(path, "MLmodel", root, MAX_MLMODEL_BYTES)
        mlmodel = _parse_mlmodel(mlmodel_path)
        _check_flavors(mlmodel)
        listed = files["model.skops"].size
        if listed is not None and listed > settings.max_model_bytes:
            raise ArtifactError(
                "ARTIFACT_TOO_LARGE",
                "Model artifact exceeds the size limit",
                {"sizeBytes": listed, "maxBytes": settings.max_model_bytes},
            )
        # The listing is only a fast path: the cap below is enforced on the bytes received.
        skops_path = tracking.download(path, "model.skops", root, settings.max_model_bytes)
        yield FetchedModel(ref=ref, run_id=run_id, mlmodel=mlmodel, skops_path=skops_path)


def _parse_mlmodel(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError):
        raise ArtifactError("ARTIFACT_LAYOUT_INVALID", "MLmodel is not valid YAML") from None
    if not isinstance(data, dict) or not isinstance(data.get("flavors"), dict):
        raise ArtifactError("ARTIFACT_LAYOUT_INVALID", "MLmodel has no flavors")
    return data


def _check_flavors(mlmodel: dict) -> None:
    flavors = mlmodel["flavors"]
    sk = flavors.get("sklearn")
    if not isinstance(sk, dict):
        raise ArtifactError("UNSUPPORTED_FLAVOR", "Only the sklearn flavor is supported")
    if sk.get("serialization_format") != "skops" or sk.get("pickled_model") != "model.skops":
        raise ArtifactError(
            "PICKLE_FORMAT_NOT_ALLOWED",
            "Model must be serialized with skops",
            {"serializationFormat": str(sk.get("serialization_format"))[:32]},
        )
    extra = sorted(set(flavors) - {"sklearn", "python_function"})
    if extra:
        raise ArtifactError("UNSUPPORTED_FLAVOR", "Unsupported flavors", {"flavors": extra[:10]})
    pyfunc = flavors.get("python_function")
    if isinstance(pyfunc, dict) and (
        pyfunc.get("loader_module") != "mlflow.sklearn" or pyfunc.get("model_path") != "model.skops"
    ):
        raise ArtifactError(
            "PICKLE_FORMAT_NOT_ALLOWED", "python_function flavor is not skops-based"
        )
