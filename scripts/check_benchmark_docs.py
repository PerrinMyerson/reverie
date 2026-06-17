#!/usr/bin/env python3
"""Validate that the benchmark workload corpus is documented."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_jana_vs_reverie.py"
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "jana-workload-corpus.txt"
DEFAULT_DOCS = [
    REPO_ROOT / "docs" / "performance.md",
    REPO_ROOT / "benchmarks" / "README.md",
]
REQUIRED_DOC_SNIPPETS = {
    "minimum observed speedup floor": "--min-observed-speedup 2.0",
    "median speedup floor": "--min-median-speedup 3.0",
    "geomean speedup floor": "--min-geomean-speedup 3.0",
    "forward direction count": "--expect-direction-count forward:19",
    "reverse direction count": "--expect-direction-count reverse:12",
    "roundtrip direction count": "--expect-direction-count roundtrip:10",
    "Markdown summary consistency": "--expect-markdown-summary benchmarks/results/jana-vs-reverie-smoke.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that each benchmark workload appears in benchmark docs."
    )
    parser.add_argument(
        "--doc",
        action="append",
        type=Path,
        default=[],
        help=(
            "Documentation file to search. Can be repeated; defaults to "
            "docs/performance.md and benchmarks/README.md."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Benchmark workload manifest that must match the harness order.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent benchmark-doc checker self-tests and exit.",
    )
    return parser.parse_args()


def current_workloads() -> list[str]:
    completed = subprocess.run(
        [sys.executable, str(BENCH_SCRIPT), "--list-workloads"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise RuntimeError("could not list benchmark workloads")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def read_docs(paths: list[Path]) -> str:
    chunks = []
    for path in paths:
        resolved = path if path.is_absolute() else REPO_ROOT / path
        chunks.append(resolved.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def read_manifest(path: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped in seen:
            raise ValueError(f"{path}:{line_number}: duplicate workload `{stripped}`")
        seen.add(stripped)
        names.append(stripped)
    return names


def validate_manifest(harness_workloads: list[str], manifest_workloads: list[str]) -> list[str]:
    errors = []
    harness_set = set(harness_workloads)
    manifest_set = set(manifest_workloads)
    missing = sorted(harness_set - manifest_set)
    extra = sorted(manifest_set - harness_set)
    if missing:
        errors.append("manifest missing workload(s): " + ", ".join(missing))
    if extra:
        errors.append("manifest has unexpected workload(s): " + ", ".join(extra))
    if not missing and not extra and manifest_workloads != harness_workloads:
        errors.append(
            "manifest workload order does not match harness order: "
            f"expected {', '.join(harness_workloads)}; found {', '.join(manifest_workloads)}"
        )
    return errors


def validate_required_doc_snippets(text: str) -> list[str]:
    missing = [
        label
        for label, snippet in REQUIRED_DOC_SNIPPETS.items()
        if snippet not in text
    ]
    if not missing:
        return []
    return ["benchmark docs missing required gate/direction detail(s): " + ", ".join(missing)]


def run_self_tests() -> int:
    harness = ["alpha_forward", "alpha_reverse", "alpha_roundtrip"]
    cases = [
        ("missing workload", ["alpha_forward", "alpha_reverse"], "manifest missing workload(s)"),
        (
            "extra workload",
            ["alpha_forward", "alpha_reverse", "alpha_roundtrip", "beta"],
            "manifest has unexpected workload(s)",
        ),
        (
            "wrong order",
            ["alpha_reverse", "alpha_forward", "alpha_roundtrip"],
            "manifest workload order does not match harness order",
        ),
    ]
    if validate_manifest(harness, list(harness)):
        print("error: self-test valid manifest failed", file=sys.stderr)
        return 1
    valid_docs = "\n".join(REQUIRED_DOC_SNIPPETS.values())
    doc_errors = validate_required_doc_snippets(valid_docs)
    if doc_errors:
        for error in doc_errors:
            print(f"error: self-test valid docs failed: {error}", file=sys.stderr)
        return 1
    doc_errors = validate_required_doc_snippets("--min-observed-speedup 2.0")
    if not any("benchmark docs missing required gate/direction detail" in error for error in doc_errors):
        print(
            "error: self-test stale docs did not report missing gate/direction details",
            file=sys.stderr,
        )
        return 1
    for label, manifest, needle in cases:
        errors = validate_manifest(harness, manifest)
        if not any(needle in error for error in errors):
            print(
                f"error: {label} did not report `{needle}`; errors were {errors}",
                file=sys.stderr,
            )
            return 1
    print("ok: benchmark docs checker self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    docs = args.doc or DEFAULT_DOCS
    try:
        workloads = current_workloads()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    try:
        manifest_workloads = read_manifest(manifest)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    manifest_errors = validate_manifest(workloads, manifest_workloads)
    if manifest_errors:
        for error in manifest_errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    text = read_docs(docs)
    doc_errors = validate_required_doc_snippets(text)
    if doc_errors:
        for error in doc_errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    missing = [workload for workload in workloads if workload not in text]
    if missing:
        for workload in missing:
            print(f"error: missing documented workload `{workload}`", file=sys.stderr)
        return 1

    print(f"ok: documented {len(workloads)} benchmark workload(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
