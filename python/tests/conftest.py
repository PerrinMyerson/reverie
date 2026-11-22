"""Fixtures that mirror the `reverie` CLI's test setup."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

import reverie

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"


@pytest.fixture
def examples() -> Path:
    """The repository's `examples/` directory."""
    return EXAMPLES


@pytest.fixture
def example() -> Callable[..., reverie.Program]:
    """Load `examples/<name>.rev`, as `reverie <cmd> examples/<name>.rev` would."""

    def load(name: str, **kwargs: Any) -> reverie.Program:
        return reverie.load(EXAMPLES / f"{name}.rev", **kwargs)

    return load


def _cli_binary() -> Path | None:
    configured = os.environ.get("REVERIE_CLI")
    if configured:
        return Path(configured)
    for profile in ("debug", "release"):
        for name in ("reverie", "reverie.exe"):
            candidate = REPO_ROOT / "target" / profile / name
            if candidate.exists():
                return candidate
    found = shutil.which("reverie")
    return Path(found) if found else None


@pytest.fixture
def reverie_cli() -> Callable[..., Dict[str, Any]]:
    """Run the `reverie` CLI binary (from `target/` or `$REVERIE_CLI`) with
    `--json`-style output and return the parsed JSON. Skips when the binary is
    not built."""
    binary = _cli_binary()
    if binary is None:
        pytest.skip("reverie CLI binary not built (cargo build -p reverie-cli)")

    def run(*args: str, cwd: Path = REPO_ROOT) -> Dict[str, Any]:
        completed = subprocess.run(
            [str(binary), *args], cwd=cwd, check=True, capture_output=True, text=True
        )
        return json.loads(completed.stdout)

    return run


def var_args(store: Dict[str, Any]) -> List[str]:
    """`{"n": 7}` -> `["--var", "n=7"]`, like the CLI's `--var` flags."""
    args: List[str] = []
    for name, value in store.items():
        args += ["--var", f"{name}={json.dumps(value).replace(' ', '')}"]
    return args
