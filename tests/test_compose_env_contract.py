"""compose.production.yml forwards exactly the variables settings.py reads."""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import yaml
from composeenv import compose_default, is_forwarded

from invoiceops_model_api.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_SOURCE = (ROOT / "src" / "invoiceops_model_api" / "settings.py").read_text()
READ_BY_SETTINGS = set(
    re.findall(r'(?:env\.get\(|_number\(env, )"([A-Z][A-Z0-9_]+)"', SETTINGS_SOURCE)
)
ENVIRONMENT = yaml.safe_load((ROOT / "compose.production.yml").read_text())["services"][
    "model-api"
]["environment"]
# Set by compose itself, not by settings.py: HOME for the read-only filesystem; the tracking URI
# comes from MODEL_API_MLFLOW_URI and the auth mode is fixed to the gateway in production.
FIXED = {"HOME": "/tmp", "MODEL_API_MLFLOW_AUTH": "gateway"}
REQUIRED = {"MODEL_API_SERVICE_TOKEN", "OFFICIAL_VALIDATION_SEED", "GATEWAY_SERVICE_READ_TOKEN"}


def test_the_scan_of_settings_py_finds_the_known_variables() -> None:
    assert {"MAX_BODY_BYTES", "MODEL_API_MLFLOW_BUDGET_SECONDS", "GATE_MIN_AUC"} <= READ_BY_SETTINGS
    assert len(READ_BY_SETTINGS) >= 23


def test_compose_sets_exactly_the_variables_settings_py_reads() -> None:
    assert set(ENVIRONMENT) - {"HOME"} == READ_BY_SETTINGS


def test_every_optional_variable_is_forwarded_with_an_empty_or_explicit_default() -> None:
    optional = READ_BY_SETTINGS - REQUIRED - {"MLFLOW_TRACKING_URI", "MODEL_API_MLFLOW_AUTH"}
    for name in sorted(optional):
        assert is_forwarded(ENVIRONMENT[name], name), f"{name}: {ENVIRONMENT[name]!r}"
    for name in REQUIRED:
        assert str(ENVIRONMENT[name]).startswith(f"${{{name}:?"), name
    for name, value in FIXED.items():
        assert ENVIRONMENT[name] == value


def test_every_variable_is_documented_in_the_env_example_and_the_readme() -> None:
    example = (ROOT / "model-api.env.example").read_text()
    readme = (ROOT / "README.md").read_text()
    expected = READ_BY_SETTINGS - {"MLFLOW_TRACKING_URI", "MODEL_API_MLFLOW_AUTH"} | {
        "MODEL_API_MLFLOW_URI"
    }
    for name in sorted(expected):
        assert re.search(rf"^#?\s*{name}=", example, re.MULTILINE), (
            f"{name} missing in model-api.env.example"
        )
    for name in sorted(READ_BY_SETTINGS):
        assert f"`{name}`" in readme or re.search(rf"`[^`]*\b{name}\b[^`]*`", readme), (
            f"{name} missing in README"
        )


def test_the_compose_defaults_resolve_to_the_service_defaults() -> None:
    """With nothing set in the host, the container behaves exactly like the code defaults."""
    env = {name: compose_default(value) for name, value in ENVIRONMENT.items()}
    env.update(
        MODEL_API_SERVICE_TOKEN="s" * 40,
        OFFICIAL_VALIDATION_SEED="9f2c5d71e0a84b36c7d1f0958e2a6b43",
        GATEWAY_SERVICE_READ_TOKEN="g" * 40,
        MLFLOW_TRACKING_URI="http://gateway:8080",
    )
    resolved = Settings.from_env(env)
    defaults = Settings(service_token="s" * 40, tracking_uri="http://gateway:8080")
    secrets = {
        "service_token",
        "official_validation_seed",
        "mlflow_auth_mode",
        "gateway_read_token",
    }
    for field in fields(Settings):
        if field.name in secrets:
            continue
        assert getattr(resolved, field.name) == getattr(defaults, field.name), field.name
    assert resolved.audit_hmac_key is None


def test_a_forwarded_value_reaches_settings() -> None:
    env = {name: compose_default(value) for name, value in ENVIRONMENT.items()}
    env.update(
        MODEL_API_SERVICE_TOKEN="s" * 40,
        OFFICIAL_VALIDATION_SEED="9f2c5d71e0a84b36c7d1f0958e2a6b43",
        GATEWAY_SERVICE_READ_TOKEN="g" * 40,
        MLFLOW_TRACKING_URI="http://gateway:8080",
        MAX_BODY_BYTES="1000",
        MODEL_CHECK_WAIT_SECONDS="3",
    )
    settings = Settings.from_env(env)
    assert (settings.max_body_bytes, settings.check_wait_seconds) == (1000, 3.0)
