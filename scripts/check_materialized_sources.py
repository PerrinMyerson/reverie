#!/usr/bin/env python3
"""Fail fast when source files are still APFS dataless placeholders."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = (
    REPO_ROOT / "Cargo.toml",
    REPO_ROOT / "Cargo.lock",
    REPO_ROOT / "crates",
    REPO_ROOT / "docs",
    REPO_ROOT / "examples",
    REPO_ROOT / "benchmarks",
    REPO_ROOT / "scripts",
)
SOURCE_SUFFIXES = {
    ".gif",
    ".ja",
    ".janus",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".rev",
    ".rs",
    ".toml",
    ".txt",
    ".yml",
}
SKIP_PARTS = {
    ".git",
    "target",
    "__pycache__",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report APFS dataless placeholders before broad verification hangs."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=list(DEFAULT_PATHS),
        help="Files or directories to scan. Defaults to source, docs, examples, and scripts.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent parser self-tests and exit.",
    )
    parser.add_argument(
        "--include-quarantine",
        action="store_true",
        help=(
            "Also scan *.dataless-quarantine-* files. By default the checker "
            "ignores quarantined placeholders so active source gates are not "
            "blocked by archived, unreadable originals."
        ),
    )
    return parser.parse_args()


def iter_files(paths: list[Path], *, include_quarantine: bool = False) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        path = path if path.is_absolute() else REPO_ROOT / path
        if not path.exists():
            continue
        if path.is_file():
            if should_scan(path, include_quarantine=include_quarantine):
                files.append(path)
            continue
        for candidate in path.rglob("*"):
            if candidate.is_file() and should_scan(
                candidate, include_quarantine=include_quarantine
            ):
                files.append(candidate)
    return sorted(set(files))


def should_scan(path: Path, *, include_quarantine: bool = False) -> bool:
    if any(part in SKIP_PARTS for part in path.parts):
        return False
    if "dataless-quarantine" in path.name:
        return include_quarantine
    return path.suffix in SOURCE_SUFFIXES or path.name in {"Cargo.toml", "Cargo.lock"}


def ls_metadata(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["ls", "-ldO@", str(path)],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"could not inspect `{path}`: {error}") from error
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"could not inspect `{path}`: {stderr}")
    return completed.stdout


def is_dataless_listing(text: str) -> bool:
    first_line = text.splitlines()[0] if text.splitlines() else text
    fields = first_line.split()
    return len(fields) >= 5 and any(
        flag == "dataless" or flag.endswith(",dataless") or flag.startswith("dataless,")
        for flag in fields
    )


def dataless_files(
    paths: list[Path], *, include_quarantine: bool = False
) -> list[Path]:
    results = []
    for path in iter_files(paths, include_quarantine=include_quarantine):
        if is_dataless_listing(ls_metadata(path)):
            results.append(path)
    return results


def run_self_tests() -> int:
    dataless = "-rw-r--r--@ 1 user staff compressed,dataless 38 May 28 file.rev\n"
    dataless_first = "-rw-r--r--@ 1 user staff dataless,compressed 38 May 28 file.rev\n"
    dataless_only = "-rw-r--r--@ 1 user staff dataless 38 May 28 file.rev\n"
    normal = "-rw-r--r--@ 1 user staff - 38 May 28 file.rev\n"
    compressed = "-rw-r--r--@ 1 user staff compressed 38 May 28 file.rev\n"
    if not is_dataless_listing(dataless):
        raise AssertionError("failed to detect compressed,dataless flags")
    if not is_dataless_listing(dataless_first):
        raise AssertionError("failed to detect dataless,compressed flags")
    if not is_dataless_listing(dataless_only):
        raise AssertionError("failed to detect dataless-only flags")
    if is_dataless_listing(normal):
        raise AssertionError("plain file detected as dataless")
    if is_dataless_listing(compressed):
        raise AssertionError("compressed materialized file detected as dataless")
    active = REPO_ROOT / "examples" / "fib.rev"
    quarantined = REPO_ROOT / "examples" / "fib.rev.dataless-quarantine-20260531"
    if not should_scan(active):
        raise AssertionError("active source file was not selected for scanning")
    if should_scan(quarantined):
        raise AssertionError("quarantined source file selected by default")
    if not should_scan(quarantined, include_quarantine=True):
        raise AssertionError("quarantined source file not selected when requested")
    print("ok: materialized source checker self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()

    try:
        missing = dataless_files(
            args.paths, include_quarantine=args.include_quarantine
        )
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if missing:
        print(
            "error: source file(s) are still APFS dataless placeholders; "
            "materialize them before running broad Cargo tests:",
            file=sys.stderr,
        )
        for path in missing:
            print(f"  {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1
    print("ok: all scanned source files are materialized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
