"""Helpers to read ``compose.production.yml`` without running docker compose."""

from __future__ import annotations

import re


def compose_default(value: object) -> str:
    """The value compose resolves when the variable is unset: ``${VAR:-default}`` -> ``default``."""
    match = re.fullmatch(r"\$\{[A-Z0-9_]+:-([^}]*)\}", str(value))
    return match.group(1) if match else str(value)


def is_forwarded(value: object, name: str) -> bool:
    """True when the value is the interpolation ``${name:-...}`` of that same variable."""
    return re.fullmatch(r"\$\{" + re.escape(name) + r":-[^}]*\}", str(value)) is not None
