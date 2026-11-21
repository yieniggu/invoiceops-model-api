"""Fail the image build when the installed invoiceops-ml is not the version pinned in uv.lock.

``uv sync --frozen`` installs the shared package from the ``ml`` build context (a path source), so
a context checked out at another version would build without any error. This check compares the
version recorded in ``uv.lock`` with the one installed in the virtual environment.

    python scripts/check_ml_version.py uv.lock
"""

from __future__ import annotations

import importlib.metadata
import re
import sys

PACKAGE = "invoiceops-ml"


def locked_version(lock_text: str, package: str = PACKAGE) -> str:
    pattern = rf'^\[\[package\]\]\nname = "{re.escape(package)}"\nversion = "([^"]+)"'
    match = re.search(pattern, lock_text, re.MULTILINE)
    if match is None:
        raise SystemExit(f"{package} is not pinned in the lockfile")
    return match.group(1)


def check(lock_text: str, installed: str, package: str = PACKAGE) -> None:
    locked = locked_version(lock_text, package)
    if installed != locked:
        raise SystemExit(
            f"{package} {installed} is installed but uv.lock pins {locked}: "
            "check out the matching invoiceops-ml ref for the 'ml' build context"
        )


def main(argv: list[str]) -> None:
    lock_path = argv[1] if len(argv) > 1 else "uv.lock"
    with open(lock_path, encoding="utf-8") as handle:
        check(handle.read(), importlib.metadata.version(PACKAGE))
    print(f"{PACKAGE} matches uv.lock")


if __name__ == "__main__":
    main(sys.argv)
