"""CI workflow contract: a step never runs in a directory that does not exist yet."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def normalise(path: str) -> str:
    return path.strip().rstrip("/").removeprefix("./") or "."


def checkout_covers(step: dict, directory: str) -> bool:
    if not str(step.get("uses", "")).startswith("actions/checkout@"):
        return False
    path = normalise(str((step.get("with") or {}).get("path", ".")))
    return path == "." or directory == path or directory.startswith(path + "/")


def steps_running_before_their_directory_exists(workflow: dict) -> list[str]:
    """Names of `run` steps whose working directory is not yet created by a prior checkout."""
    offenders = []
    workflow_default = ((workflow.get("defaults") or {}).get("run") or {}).get("working-directory")
    for job_name, job in workflow["jobs"].items():
        job_default = ((job.get("defaults") or {}).get("run") or {}).get("working-directory")
        checked_out: list[dict] = []
        for step in job["steps"]:
            if "uses" in step:
                checked_out.append(step)
            if "run" not in step:
                continue
            directory = normalise(
                str(step.get("working-directory") or job_default or workflow_default or ".")
            )
            if directory != "." and not any(checkout_covers(c, directory) for c in checked_out):
                offenders.append(f"{job_name}: {step.get('name', step['run'][:40])}")
    return offenders


def test_no_run_step_precedes_the_checkout_of_its_working_directory() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    assert steps_running_before_their_directory_exists(workflow) == []


def test_the_contract_detects_a_step_that_inherits_a_default_before_the_checkout() -> None:
    broken = yaml.safe_load(
        """
jobs:
  checks:
    defaults: {run: {working-directory: app}}
    steps:
      - name: guard
        run: echo guard
      - uses: actions/checkout@v1
        with: {path: app}
      - name: after
        run: echo after
"""
    )
    assert steps_running_before_their_directory_exists(broken) == ["checks: guard"]


def test_an_explicit_working_directory_opts_a_step_out_of_the_inherited_default() -> None:
    fixed = yaml.safe_load(
        """
jobs:
  checks:
    defaults: {run: {working-directory: app}}
    steps:
      - name: guard
        working-directory: .
        run: echo guard
      - uses: actions/checkout@v1
        with: {path: app}
"""
    )
    assert steps_running_before_their_directory_exists(fixed) == []
