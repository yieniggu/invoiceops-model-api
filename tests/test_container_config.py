"""Static guardrails for the container build, not an image execution test."""

from pathlib import Path

import pytest
from composeenv import compose_default

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
    # The shared package comes from the "ml" named build context, only src + metadata.
    builder_copies = [parts for parts in builder if parts[0].upper() == "COPY"]
    assert any("--from=ml" in parts for parts in builder_copies)
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
        for key in (
            "MODEL_API_SERVICE_TOKEN",
            "OFFICIAL_VALIDATION_SEED",
            "MLFLOW_TRACKING_PASSWORD",
            "MLFLOW_TRACKING_USERNAME",
            "MLFLOW_TRACKING_TOKEN",
        )
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
        "env MODEL_API_SERVICE_TOKEN=exposed",
        "env OFFICIAL_VALIDATION_SEED=1",
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


def test_compose_hardening_and_memory_alignment() -> None:
    import yaml

    compose = yaml.safe_load((ROOT / "compose.production.yml").read_text())
    service = compose["services"]["model-api"]
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]
    assert service["read_only"] is True
    assert any(str(t).startswith("/tmp") for t in service["tmpfs"])
    assert 0 < int(service["pids_limit"]) <= 1024
    memory = service["mem_limit"]
    limit_mb = int(memory[:-1]) * 1024 if memory.endswith("g") else int(memory[:-1])
    for variable in ("MODEL_CHECK_MEMORY_MB", "MODEL_WORKER_MEMORY_MB"):
        # RLIMIT_AS of a child must stay below the cgroup limit it shares with the HTTP process.
        assert 0 < int(compose_default(service["environment"][variable])) < limit_mb
    assert service["environment"]["HOME"] == "/tmp"


def test_base_images_are_pinned_by_digest() -> None:
    froms = [
        parts[1] for parts in instructions((ROOT / "Dockerfile").read_text()) if parts[0] == "FROM"
    ]
    assert froms and all("@sha256:" in image for image in froms), froms


def test_ci_requires_a_pinned_invoiceops_ml_ref() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "INVOICEOPS_ML_REF: main" not in workflow
    assert "vars.INVOICEOPS_ML_REF" in workflow
    assert workflow.count("Require a pinned invoiceops-ml ref") == 2  # both jobs


def test_ci_runs_every_container_test_with_the_image_variable() -> None:
    import yaml

    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    steps = workflow["jobs"]["image"]["steps"]
    step = next(s for s in steps if "test_container_pid1.py" in s.get("run", ""))
    assert step["env"]["MODEL_API_TEST_IMAGE"]
    listed = {part for part in step["run"].split() if part.startswith("tests/test_container_")}
    on_disk = {f"tests/{path.name}" for path in (ROOT / "tests").glob("test_container_*.py")}
    assert on_disk - {"tests/test_container_config.py"} <= listed, on_disk - listed
    assert listed <= on_disk, listed - on_disk


def test_ci_documents_that_a_sha_is_preferred_over_a_tag() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "commit SHA (preferred" in workflow
    assert "Prefer the SHA" in (ROOT / "README.md").read_text()


def test_the_image_disables_the_unstructured_uvicorn_access_log() -> None:
    cmd = [parts for parts in instructions((ROOT / "Dockerfile").read_text()) if parts[0] == "CMD"]
    assert "--no-access-log" in " ".join(cmd[-1])


def test_skops_pin_documents_its_reason_and_matches_invoiceops_ml() -> None:
    text = (ROOT / "pyproject.toml").read_text()
    assert "must stay equal to the pins of" in text
    ml = ROOT.parent / "invoiceops-ml" / "pyproject.toml"
    if ml.exists():
        for package in ("skops", "scikit-learn", "numpy", "pandas"):
            pin = next(
                line.strip().strip(",").strip('"')
                for line in text.splitlines()
                if f'"{package}==' in line
            )
            assert pin in ml.read_text(), pin


# ---- no docker-init as PID 1 while secrets are in the environment ---------------------

SECRET_MARKERS = ("TOKEN", "SEED", "SECRET", "PASSWORD", "CREDENTIAL")


def assert_no_init_with_secrets(service: dict) -> None:
    secrets = [
        k for k in service.get("environment", {}) if any(m in k.upper() for m in SECRET_MARKERS)
    ]
    assert secrets, "the service is expected to receive its secrets through the environment"
    assert not service.get("init", False), (
        "init: true makes docker-init PID 1 with the full environment, readable through "
        f"/proc/1/environ by every child; secrets in the environment: {secrets}"
    )


def test_compose_does_not_enable_init_while_secrets_are_in_the_environment() -> None:
    import yaml

    compose = yaml.safe_load((ROOT / "compose.production.yml").read_text())
    assert_no_init_with_secrets(compose["services"]["model-api"])


def test_the_init_guard_fails_when_init_is_enabled() -> None:
    import yaml

    compose = yaml.safe_load((ROOT / "compose.production.yml").read_text())
    service = {**compose["services"]["model-api"], "init": True}
    with pytest.raises(AssertionError, match="docker-init"):
        assert_no_init_with_secrets(service)


def test_uvicorn_is_pid_1_in_the_image() -> None:
    cmd = [parts for parts in instructions((ROOT / "Dockerfile").read_text()) if parts[0] == "CMD"]
    assert cmd and cmd[-1][1].startswith('["uvicorn"'), "CMD must exec uvicorn (exec form)"
    assert not any(
        parts[0] in ("ENTRYPOINT",) for parts in instructions((ROOT / "Dockerfile").read_text())
    )


# ---- the shared package version must equal the lockfile version ------------------------


def load_check_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_ml_version", ROOT / "scripts" / "check_ml_version.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOCK = '[[package]]\nname = "invoiceops-ml"\nversion = "0.2.0"\nsource = { directory = "../ml" }\n'


def test_the_build_verifies_the_ml_version_against_the_lockfile_after_the_last_sync() -> None:
    stages = instructions((ROOT / "Dockerfile").read_text())
    start = next(
        i for i, p in enumerate(stages) if p[0].upper() == "FROM" and p[-1].lower() == "builder"
    )
    end = next(i for i in range(start + 1, len(stages)) if stages[i][0].upper() == "FROM")
    builder = stages[start:end]
    syncs = [i for i, p in enumerate(builder) if p[0] == "RUN" and p[1:3] == ["uv", "sync"]]
    checks = [
        i
        for i, p in enumerate(builder)
        if p[0] == "RUN" and any(part.endswith("check_ml_version.py") for part in p)
    ]
    assert checks and checks[-1] > syncs[-1]
    assert any(p[0] == "COPY" and p[1] == "scripts/check_ml_version.py" for p in builder), (
        "the check script must be copied into the builder"
    )


def test_the_ml_version_check_passes_on_a_match_and_fails_the_build_on_drift() -> None:
    module = load_check_script()
    module.check(LOCK, "0.2.0")  # no exception
    with pytest.raises(SystemExit, match="0.3.0.*pins 0.2.0"):
        module.check(LOCK, "0.3.0")
    with pytest.raises(SystemExit, match="not pinned"):
        module.check('[[package]]\nname = "other"\nversion = "1"\n', "0.2.0")


def test_the_real_lockfile_pins_the_version_of_the_sibling_checkout() -> None:
    module = load_check_script()
    locked = module.locked_version((ROOT / "uv.lock").read_text())
    ml = ROOT.parent / "invoiceops-ml" / "pyproject.toml"
    if ml.exists():
        assert f'version = "{locked}"' in ml.read_text()


# ---- the interpreter is pinned for local runs too --------------------------------------


def test_python_version_file_pins_the_supported_interpreter() -> None:
    assert (ROOT / ".python-version").read_text().strip() == "3.12"
    assert 'requires-python = ">=3.12,<3.13"' in (ROOT / "pyproject.toml").read_text()
