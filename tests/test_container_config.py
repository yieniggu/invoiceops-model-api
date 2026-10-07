"""Static guardrails for the container build, not an image execution test."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def instructions(text: str) -> list[list[str]]:
    return [
        line.strip().split()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_runtime_stage_copies_only_app_and_frozen_production_dependencies() -> None:
    stages = instructions((ROOT / "Dockerfile").read_text())
    builder_start = next(
        i
        for i, parts in enumerate(stages)
        if parts[0].upper() == "FROM"
        and len(parts) >= 4
        and parts[-2].upper() == "AS"
        and parts[-1].lower() == "builder"
    )
    runtime_start = next(
        i for i in range(builder_start + 1, len(stages)) if stages[i][0].upper() == "FROM"
    )
    builder = stages[builder_start:runtime_start]
    runtime = stages[runtime_start:]

    sync_steps = [
        parts for parts in builder if parts[0].upper() == "RUN" and parts[1:3] == ["uv", "sync"]
    ]
    assert sync_steps
    assert all("--frozen" in parts and "--no-dev" in parts for parts in sync_steps)
    copies = [parts for parts in runtime if parts[0].upper() in ("COPY", "ADD")]
    assert {(parts[-2], parts[-1]) for parts in copies} == {
        ("/app/.venv", ".venv"),
        ("/app/src", "./src"),
    }
    assert len(copies) == 2
    assert all(parts[0].upper() == "COPY" and "--from=builder" in parts for parts in copies)
    assert any(parts[0].upper() == "USER" and parts[1:] == ["app"] for parts in runtime)
    assert not any(
        key in " ".join(parts[1:])
        for parts in stages
        if parts[0].upper() in ("ARG", "ENV")
        for key in ("MODEL_API_ADMIN_TOKEN", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_USERNAME")
    )


def test_build_context_excludes_local_credentials_and_development_artifacts() -> None:
    exclusions = set((ROOT / ".dockerignore").read_text().splitlines())

    assert {".env", ".env.*", ".venv/", "tests/", ".git/"} <= exclusions
    # A later exception can re-include credentials despite the exclusions above.
    assert not any(pattern.strip().startswith("!") for pattern in exclusions)


@pytest.mark.parametrize(
    "extra_instruction",
    [
        "copy . /app/unsafe",
        "  ADD . /app/unsafe",
        "env MODEL_API_ADMIN_TOKEN=exposed",
        "  arg MLFLOW_TRACKING_PASSWORD=exposed",
    ],
)
def test_container_guard_rejects_case_and_indentation_variants(
    extra_instruction: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    dockerfile = ROOT / "Dockerfile"
    original = dockerfile.read_text()
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path: original + "\n" + extra_instruction if path == dockerfile else original,
    )
    with pytest.raises(AssertionError):
        test_runtime_stage_copies_only_app_and_frozen_production_dependencies()
