"""Keep single-invoice inference bounded in the container runtime."""

from pathlib import Path


def test_model_api_runtime_bounds_native_numeric_threads() -> None:
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text()
    runtime = dockerfile.split("FROM python:3.12-slim\n", 1)[1]
    assert "OMP_NUM_THREADS=1" in runtime
    assert "OPENBLAS_NUM_THREADS=1" in runtime
