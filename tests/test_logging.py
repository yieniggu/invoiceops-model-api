"""The service logger writes INFO lines to stdout."""

from __future__ import annotations

import logging

import pytest
from conftest import AUTH, TOKEN, make_settings
from fastapi.testclient import TestClient

from invoiceops_model_api.app import create_app
from invoiceops_model_api.logs import LOGGER_NAME, KeyValueFormatter, configure_logging, log_value


@pytest.fixture()
def clean_logger():
    logger = logging.getLogger(LOGGER_NAME)
    saved = (list(logger.handlers), logger.level)
    logger.handlers = []
    logger.setLevel(logging.NOTSET)
    yield logger
    logger.handlers, level = saved
    logger.setLevel(level)


def test_the_logger_has_no_output_until_configured(clean_logger, capsys) -> None:
    clean_logger.info("before")
    assert "before" not in capsys.readouterr().out  # regression: INFO was dropped


def test_configure_logging_writes_info_lines_to_stdout_in_key_value_format(
    clean_logger, capsys
) -> None:
    configure_logging()
    clean_logger.info("audit principal=abc method=DELETE route=/x status=200")
    out = capsys.readouterr().out.strip()
    assert out.startswith("ts=") and " level=INFO logger=invoiceops_model_api " in out
    assert out.endswith('msg="audit principal=abc method=DELETE route=/x status=200"')


def test_configure_logging_is_idempotent(clean_logger, capsys) -> None:
    configure_logging()
    configure_logging()
    clean_logger.info("once")
    assert capsys.readouterr().out.count("once") == 1


def test_formatter_escapes_quotes_and_newlines() -> None:
    record = logging.LogRecord("n", logging.ERROR, "f", 1, 'a "b"', None, None)
    line = KeyValueFormatter().format(record)
    assert 'msg="a \\"b\\""' in line and "\n" not in line


def test_create_app_configures_logging_so_the_audit_line_reaches_stdout(
    clean_logger, capsys
) -> None:
    app = create_app(make_settings("http://127.0.0.1:9"))
    with TestClient(app) as client:
        assert client.delete("/deployments/some-id", headers=AUTH).status_code == 200
    out = capsys.readouterr().out
    assert "audit principal=" in out and "method=DELETE" in out
    assert "deployment_unloaded id=some-id removed=False" in out
    assert TOKEN not in out


def format_line(message: str) -> str:
    record = logging.LogRecord("n", logging.INFO, "f", 1, message, None, None)
    return KeyValueFormatter().format(record)


def test_formatter_escapes_every_control_character_and_line_separator() -> None:
    line = format_line("a\nb\rc\td\x00e\x1bf\x85g h i\x7fj")
    assert "\n" not in line and "\r" not in line
    assert len(line.splitlines()) == 1
    assert all(ord(char) >= 0x20 and not 0x7F <= ord(char) <= 0x9F for char in line)
    assert "a\\nb\\rc\\td\\u0000e\\u001bf\\u0085g\\u2028h\\u2029i\\u007fj" in line


def test_formatter_caps_the_length_of_one_line() -> None:
    line = format_line("x" * 100_000)
    assert len(line) < 3000


def test_log_value_caps_a_value_and_says_how_much_was_cut() -> None:
    assert log_value("short") == "short"
    capped = log_value("m" * 5000)
    assert capped.startswith("m" * 200) and "m" * 201 not in capped
    assert capped.endswith("(+4800)")


def test_a_registered_model_with_a_newline_is_one_escaped_log_line(
    clean_logger, capsys, monkeypatch
) -> None:
    app = create_app(make_settings("http://127.0.0.1:9"))
    name = "evil\nlevel=ERROR forged=1\r" + "z" * 500
    with TestClient(app) as client:
        client.post(
            "/deploy-check",
            json={"workspace": "w", "registeredModel": name, "version": "1"},
            headers=AUTH,
        )
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "deploy_check" in ln]
    assert len(lines) == 1, lines
    assert "evil\\nlevel=ERROR forged=1\\r" in lines[0]
    assert "z" * 201 not in lines[0]
