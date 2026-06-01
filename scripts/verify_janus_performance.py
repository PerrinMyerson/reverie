#!/usr/bin/env python3
"""Run the checked Janus-vs-Reverie smoke performance gate."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_jana_vs_reverie.py"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "jana-vs-reverie-smoke.json"
DEFAULT_MARKDOWN_OUTPUT = (
    REPO_ROOT / "benchmarks" / "results" / "jana-vs-reverie-smoke.md"
)
SUMMARY_SCRIPT = REPO_ROOT / "scripts" / "summarize_janus_performance.py"
CHECK_SCRIPT = REPO_ROOT / "scripts" / "check_benchmark_artifact.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the checked Janus-vs-Reverie smoke performance gate."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Measured runs per workload and implementation.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Untimed warmup runs per workload and implementation.",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.25,
        help="Required median Jana/Reverie speedup for every workload.",
    )
    parser.add_argument(
        "--min-observed-speedup",
        type=float,
        default=2.0,
        help="Required floor for every observed workload speedup in the JSON artifact.",
    )
    parser.add_argument(
        "--min-median-speedup",
        type=float,
        default=4.0,
        help="Required floor for the suite median observed speedup.",
    )
    parser.add_argument(
        "--min-geomean-speedup",
        type=float,
        default=4.0,
        help="Required floor for the geometric mean observed speedup.",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=30.0,
        help="Seconds to allow each verification or timed command.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the machine-readable benchmark artifact.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
        help="Path for the human-readable Markdown summary.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Reuse an existing target/release/reverie binary.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Benchmark name to run. Can be repeated.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent wrapper validation self-tests and exit.",
    )
    return parser.parse_args()


def benchmark_workloads() -> list[str]:
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


def annotate_performance_gate(
    path: Path,
    min_observed_speedup: float,
    min_median_speedup: float,
    min_geomean_speedup: float,
) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("benchmark artifact root must be a JSON object")
    data["performance_gate"] = {
        "min_geomean_speedup": min_geomean_speedup,
        "min_observed_speedup": min_observed_speedup,
        "min_median_speedup": min_median_speedup,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validation_errors(args: argparse.Namespace) -> list[str]:
    errors = []
    if args.runs <= 0:
        errors.append("--runs must be positive")
    if args.warmup < 0:
        errors.append("--warmup must be non-negative")
    if not math.isfinite(args.min_speedup) or args.min_speedup <= 0:
        errors.append("--min-speedup must be finite and positive")
    if not math.isfinite(args.min_observed_speedup) or args.min_observed_speedup <= 0:
        errors.append("--min-observed-speedup must be finite and positive")
    if not math.isfinite(args.min_median_speedup) or args.min_median_speedup <= 0:
        errors.append("--min-median-speedup must be finite and positive")
    if not math.isfinite(args.min_geomean_speedup) or args.min_geomean_speedup <= 0:
        errors.append("--min-geomean-speedup must be finite and positive")
    if not math.isfinite(args.command_timeout) or args.command_timeout <= 0:
        errors.append("--command-timeout must be finite and positive")
    duplicates = sorted(
        name for name in set(args.only) if args.only.count(name) > 1
    )
    if duplicates:
        errors.append(f"duplicate --only workload(s): {', '.join(duplicates)}")
    return errors


def expected_workloads(all_workloads: list[str], only: list[str]) -> list[str]:
    if not only:
        return all_workloads
    unknown = sorted(set(only) - set(all_workloads))
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"unknown benchmark workload(s): {names}")
    return only


def self_test_args(**overrides: Any) -> argparse.Namespace:
    values = {
        "runs": 5,
        "warmup": 1,
        "min_speedup": 1.25,
        "min_observed_speedup": 2.0,
        "min_median_speedup": 4.0,
        "min_geomean_speedup": 4.0,
        "command_timeout": 30.0,
        "only": [],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def expect_error(label: str, args: argparse.Namespace, expected: str) -> None:
    errors = validation_errors(args)
    if expected not in errors:
        raise AssertionError(f"{label}: expected {expected!r}, got {errors!r}")


def run_self_tests() -> int:
    valid = self_test_args()
    errors = validation_errors(valid)
    if errors:
        raise AssertionError(f"valid defaults rejected: {errors!r}")

    expect_error("runs", self_test_args(runs=0), "--runs must be positive")
    expect_error("warmup", self_test_args(warmup=-1), "--warmup must be non-negative")
    expect_error(
        "min speedup",
        self_test_args(min_speedup=0),
        "--min-speedup must be finite and positive",
    )
    expect_error(
        "min speedup nan",
        self_test_args(min_speedup=float("nan")),
        "--min-speedup must be finite and positive",
    )
    expect_error(
        "min speedup inf",
        self_test_args(min_speedup=float("inf")),
        "--min-speedup must be finite and positive",
    )
    expect_error(
        "observed floor",
        self_test_args(min_observed_speedup=0),
        "--min-observed-speedup must be finite and positive",
    )
    expect_error(
        "observed floor nan",
        self_test_args(min_observed_speedup=float("nan")),
        "--min-observed-speedup must be finite and positive",
    )
    expect_error(
        "median floor",
        self_test_args(min_median_speedup=0),
        "--min-median-speedup must be finite and positive",
    )
    expect_error(
        "median floor nan",
        self_test_args(min_median_speedup=float("nan")),
        "--min-median-speedup must be finite and positive",
    )
    expect_error(
        "geomean floor",
        self_test_args(min_geomean_speedup=0),
        "--min-geomean-speedup must be finite and positive",
    )
    expect_error(
        "geomean floor nan",
        self_test_args(min_geomean_speedup=float("nan")),
        "--min-geomean-speedup must be finite and positive",
    )
    expect_error(
        "timeout",
        self_test_args(command_timeout=0),
        "--command-timeout must be finite and positive",
    )
    expect_error(
        "timeout nan",
        self_test_args(command_timeout=float("nan")),
        "--command-timeout must be finite and positive",
    )
    expect_error(
        "duplicate workload",
        self_test_args(only=["fib", "fib"]),
        "duplicate --only workload(s): fib",
    )

    workloads = ["fib", "sort", "turing"]
    if expected_workloads(workloads, []) != workloads:
        raise AssertionError("empty --only should select the full workload list")
    if expected_workloads(workloads, ["sort"]) != ["sort"]:
        raise AssertionError("--only should preserve requested workload order")
    try:
        expected_workloads(workloads, ["missing"])
    except ValueError as error:
        if str(error) != "unknown benchmark workload(s): missing":
            raise AssertionError(f"unexpected unknown-workload error: {error}") from error
    else:
        raise AssertionError("unknown --only workload was accepted")

    print("ok: Janus performance wrapper self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    errors = validation_errors(args)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    try:
        all_workloads = benchmark_workloads()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    try:
        selected_workloads = expected_workloads(all_workloads, args.only)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    command = [
        sys.executable,
        str(BENCH_SCRIPT),
        "--runs",
        str(args.runs),
        "--warmup",
        str(args.warmup),
        "--min-speedup",
        str(args.min_speedup),
        "--command-timeout",
        str(args.command_timeout),
        "--json-output",
        str(args.json_output),
    ]
    if args.no_build:
        command.append("--no-build")
    for name in args.only:
        command.extend(["--only", name])

    completed = subprocess.run(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        return completed.returncode
    try:
        annotate_performance_gate(
            args.json_output,
            args.min_observed_speedup,
            args.min_median_speedup,
            args.min_geomean_speedup,
        )
    except (OSError, json.JSONDecodeError, RuntimeError) as error:
        print(f"error: could not annotate benchmark artifact: {error}", file=sys.stderr)
        return 1

    check_command = [
        sys.executable,
        str(CHECK_SCRIPT),
        str(args.json_output),
        "--min-workloads",
        str(len(set(selected_workloads))),
        "--exact-workloads",
        "--min-observed-speedup",
        str(args.min_observed_speedup),
        "--min-median-speedup",
        str(args.min_median_speedup),
        "--min-geomean-speedup",
        str(args.min_geomean_speedup),
        "--expect-performance-gate",
        "--expect-source-digests",
        "--verify-file-digests",
        "--expect-jana-bin-suffix",
        "target/jana-baseline/bin/janus",
        "--expect-reverie-bin-suffix",
        "target/release/reverie",
    ]
    if not args.only:
        check_command.extend(
            [
                "--require-direction",
                "forward",
                "--require-direction",
                "reverse",
                "--require-direction",
                "roundtrip",
            ]
        )
    for workload in selected_workloads:
        check_command.extend(["--expect-workload", workload])
    completed = subprocess.run(check_command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        return completed.returncode

    if args.markdown_output is not None:
        summary_command = [
            sys.executable,
            str(SUMMARY_SCRIPT),
            str(args.json_output),
            "--output",
            str(args.markdown_output),
            "--min-observed-speedup",
            str(args.min_observed_speedup),
            "--min-median-speedup",
            str(args.min_median_speedup),
            "--min-geomean-speedup",
            str(args.min_geomean_speedup),
        ]
        completed = subprocess.run(summary_command, cwd=REPO_ROOT)
        if completed.returncode != 0:
            return completed.returncode
        summary_check_command = check_command + [
            "--expect-markdown-summary",
            str(args.markdown_output),
        ]
        return subprocess.run(summary_check_command, cwd=REPO_ROOT).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
