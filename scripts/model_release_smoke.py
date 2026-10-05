"""Opt-in verification of a manually operated model release."""

import base64
import json
import math
import multiprocessing
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

MODEL_ID = "invoice-review"
HTTP_TIMEOUT_SECONDS = 10
MAX_HTTP_TIMEOUT_SECONDS = 120
PROCESS_CLEANUP_SECONDS = 1
PREDICT_PAYLOAD = {
    "invoice_amount_cents": 125000,
    "vendor_tenure_days": 730,
    "previous_incidents_12m": 0,
    "amount_vs_vendor_median": 1.1,
    "has_purchase_order": True,
    "three_way_match": True,
    "bank_account_recently_changed": False,
    "country_risk": "low",
}


class SmokeFailure(Exception):
    pass


@dataclass(frozen=True)
class SmokeConfiguration:
    base_url: str
    admin_token: str
    expected_model_version: str
    expected_run_id: str
    http_timeout_seconds: float


def execute_request(
    url: str,
    body: bytes | None,
    headers: dict[str, str],
    method: str,
    result_path: str,
    http_timeout_seconds: float,
) -> None:
    try:
        request = Request(url, data=body, headers=headers, method=method)
        with urlopen(request, timeout=http_timeout_seconds) as response:
            result = {"outcome": "response", "body": base64.b64encode(response.read()).decode("ascii")}
    except HTTPError as error:
        result = {"outcome": "http", "status": error.code}
    except URLError:
        result = {"outcome": "unreachable"}
    except TimeoutError:
        result = {"outcome": "timeout"}
    except (HTTPException, ValueError):
        result = {"outcome": "failure"}

    with open(result_path, "w", encoding="utf-8") as result_file:
        json.dump(result, result_file)


def stop_request_process(process: multiprocessing.Process) -> None:
    process.terminate()
    process.join(PROCESS_CLEANUP_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(PROCESS_CLEANUP_SECONDS)


def required_environment(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise SmokeFailure(f"Missing required environment variable: {name}")
    return value


def required_header_environment(name: str) -> str:
    value = required_environment(name)
    if "\r" in value or "\n" in value:
        raise SmokeFailure(f"{name} must not contain CR or LF")
    return value


def read_http_timeout() -> float:
    value = os.getenv("MODEL_RELEASE_HTTP_TIMEOUT_SECONDS")
    if value is None:
        return HTTP_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except ValueError:
        timeout = 0
    if not math.isfinite(timeout) or not 0 < timeout <= MAX_HTTP_TIMEOUT_SECONDS:
        raise SmokeFailure("MODEL_RELEASE_HTTP_TIMEOUT_SECONDS must be finite and in (0, 120]")
    return timeout


def read_configuration() -> SmokeConfiguration:
    base_url = required_environment("MODEL_API_BASE_URL").rstrip("/")
    parsed_url = urlsplit(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise SmokeFailure("MODEL_API_BASE_URL must be an absolute HTTP(S) URL")
    return SmokeConfiguration(
        base_url=base_url,
        admin_token=required_header_environment("MODEL_API_ADMIN_TOKEN"),
        expected_model_version=required_environment("MODEL_RELEASE_EXPECTED_MODEL_VERSION"),
        expected_run_id=required_environment("MODEL_RELEASE_EXPECTED_RUN_ID"),
        http_timeout_seconds=read_http_timeout(),
    )


def request_json(
    configuration: SmokeConfiguration, path: str, method: str, payload: dict[str, object] | None = None
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    if path.startswith("/admin/"):
        headers["X-Model-Admin-Token"] = configuration.admin_token
    url = f"{configuration.base_url}{path}"
    try:
        Request(url, data=body, headers=headers, method=method)
    except ValueError:
        raise SmokeFailure(f"{method} {path} could not be constructed") from None

    result_descriptor, result_path = tempfile.mkstemp(prefix="model-release-smoke-", suffix=".json")
    os.close(result_descriptor)
    process: multiprocessing.Process | None = None
    try:
        process = multiprocessing.get_context("spawn").Process(
            target=execute_request,
            args=(url, body, headers, method, result_path, configuration.http_timeout_seconds),
            daemon=True,
        )
        started_at = time.monotonic()
        process.start()
        process.join(max(0, configuration.http_timeout_seconds - (time.monotonic() - started_at)))
        if process.is_alive() or time.monotonic() - started_at >= configuration.http_timeout_seconds:
            if process.is_alive():
                stop_request_process(process)
            raise SmokeFailure(f"{method} {path} timed out")
        if process.exitcode != 0:
            raise SmokeFailure(f"{method} {path} failed to complete the request")
        with open(result_path, encoding="utf-8") as result_file:
            result = json.load(result_file)
    except (OSError, ValueError, json.JSONDecodeError):
        raise SmokeFailure(f"{method} {path} failed to complete the request") from None
    finally:
        if process is not None and process.is_alive():
            stop_request_process(process)
        os.unlink(result_path)

    outcome = result.get("outcome")
    if outcome == "http":
        raise SmokeFailure(f"{method} {path} failed with HTTP {result.get('status')}")
    if outcome == "unreachable":
        raise SmokeFailure(f"{method} {path} failed to reach the Model API")
    if outcome == "timeout":
        raise SmokeFailure(f"{method} {path} timed out")
    if outcome != "response" or not isinstance(result.get("body"), str):
        raise SmokeFailure(f"{method} {path} failed to complete the request")

    try:
        response_body = base64.b64decode(result["body"], validate=True)
        decoded = json.loads(response_body)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise SmokeFailure(f"{method} {path} returned invalid JSON") from None
    if not isinstance(decoded, dict):
        raise SmokeFailure(f"{method} {path} returned an unexpected JSON response")
    return decoded


def validate_metadata(
    response: dict[str, Any], configuration: SmokeConfiguration, endpoint: str
) -> None:
    if response.get("model_id") != MODEL_ID:
        raise SmokeFailure(f"{endpoint} returned an unexpected model_id")
    if response.get("model_version") != configuration.expected_model_version:
        raise SmokeFailure(f"{endpoint} model_version did not match the expected value")
    if response.get("run_id") != configuration.expected_run_id:
        raise SmokeFailure(f"{endpoint} run_id did not match the expected value")


def run_smoke(configuration: SmokeConfiguration) -> None:
    refresh = request_json(
        configuration,
        f"/admin/models/{MODEL_ID}/refresh",
        "POST",
    )
    validate_metadata(refresh, configuration, "refresh")

    health = request_json(configuration, f"/models/{MODEL_ID}/health", "GET")
    validate_metadata(health, configuration, "health")

    prediction = request_json(
        configuration,
        f"/models/{MODEL_ID}/predict",
        "POST",
        PREDICT_PAYLOAD,
    )
    validate_metadata(prediction, configuration, "predict")
    probability = prediction.get("probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise SmokeFailure("predict returned an invalid probability")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise SmokeFailure("predict returned an invalid probability")


def main() -> int:
    try:
        run_smoke(read_configuration())
    except SmokeFailure as error:
        print(f"Model release smoke failed: {error}", file=sys.stderr)
        return 1
    print("Model release smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
