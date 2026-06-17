#!/usr/bin/env python3
"""Validate a Jana-vs-Reverie benchmark artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from summarize_janus_performance import render_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "benchmarks" / "results" / "jana-vs-reverie-smoke.json"
BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_jana_vs_reverie.py"
DEFAULT_DIRECTIONS = ("forward", "reverse", "roundtrip")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a checked Jana-vs-Reverie benchmark JSON artifact."
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--min-workloads",
        type=int,
        default=1,
        help="Require at least this many benchmark workloads.",
    )
    parser.add_argument(
        "--require-direction",
        action="append",
        choices=DEFAULT_DIRECTIONS,
        default=[],
        help=(
            "Require at least one workload in this direction. Can be repeated; "
            "defaults to no direction requirement."
        ),
    )
    parser.add_argument(
        "--expect-workload",
        action="append",
        default=[],
        help=(
            "Require a named workload to be present. Can be repeated to pin "
            "the expected benchmark corpus."
        ),
    )
    parser.add_argument(
        "--exact-workloads",
        action="store_true",
        help="Reject workloads not listed with --expect-workload.",
    )
    parser.add_argument(
        "--expect-workload-order",
        action="store_true",
        help=(
            "Require benchmark rows to appear in the same order as the combined "
            "--expect-workload, --expect-current-workloads, and "
            "--exact-workloads-file list."
        ),
    )
    parser.add_argument(
        "--expect-current-workloads",
        action="store_true",
        help=(
            "Require the artifact's workload names to exactly match "
            "scripts/bench_jana_vs_reverie.py --list-workloads."
        ),
    )
    parser.add_argument(
        "--exact-workloads-file",
        type=Path,
        help=(
            "Require the artifact's workload names to exactly match non-empty, "
            "non-comment lines from this file."
        ),
    )
    parser.add_argument(
        "--min-observed-speedup",
        type=float,
        default=None,
        help="Require every workload's observed median speedup to meet this floor.",
    )
    parser.add_argument(
        "--min-median-speedup",
        type=float,
        default=None,
        help="Require the suite median of observed workload speedups to meet this floor.",
    )
    parser.add_argument(
        "--min-geomean-speedup",
        type=float,
        default=None,
        help="Require the geometric mean of observed workload speedups to meet this floor.",
    )
    parser.add_argument(
        "--expect-performance-gate",
        action="store_true",
        help=(
            "Require performance_gate.min_observed_speedup, "
            "performance_gate.min_median_speedup, and "
            "performance_gate.min_geomean_speedup to be recorded in the artifact."
        ),
    )
    parser.add_argument(
        "--expect-source-digests",
        action="store_true",
        help=(
            "Require every benchmark side to record source_files entries with "
            "SHA-256 digests for every .ja, .janus, or .rev command argument."
        ),
    )
    parser.add_argument(
        "--verify-file-digests",
        action="store_true",
        help=(
            "Recompute recorded binary and source-file SHA-256 digests from "
            "the local filesystem. Use this for freshly generated artifacts."
        ),
    )
    parser.add_argument(
        "--expect-runs",
        type=int,
        help="Require the artifact to contain exactly this many measured runs.",
    )
    parser.add_argument(
        "--expect-warmup",
        type=int,
        help="Require the artifact to record exactly this many warmup runs.",
    )
    parser.add_argument(
        "--expect-command-timeout",
        type=float,
        help="Require the artifact to record exactly this command timeout in seconds.",
    )
    parser.add_argument(
        "--expect-min-speedup",
        type=float,
        help="Require the artifact to record exactly this per-workload speedup gate.",
    )
    parser.add_argument(
        "--expect-direction-count",
        action="append",
        type=parse_direction_count,
        default=[],
        metavar="DIRECTION:COUNT",
        help=(
            "Require exactly COUNT workloads for DIRECTION. Can be repeated, "
            "for example forward:18."
        ),
    )
    parser.add_argument(
        "--expect-markdown-summary",
        type=Path,
        help=(
            "Require this Markdown summary to exactly match the summary "
            "rendered from the JSON artifact."
        ),
    )
    parser.add_argument(
        "--expect-jana-bin-suffix",
        help="Require jana_baseline and metadata.jana.binary.path to end with this suffix.",
    )
    parser.add_argument(
        "--expect-reverie-bin-suffix",
        help="Require reverie_binary and metadata.reverie.binary.path to end with this suffix.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent benchmark-validator self-tests and exit.",
    )
    return parser.parse_args()


def parse_direction_count(value: str) -> tuple[str, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected DIRECTION:COUNT, such as forward:18")
    direction, count_text = parts
    if direction not in DEFAULT_DIRECTIONS:
        raise argparse.ArgumentTypeError(
            f"direction must be one of {', '.join(DEFAULT_DIRECTIONS)}"
        )
    try:
        count = int(count_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("COUNT must be an integer") from error
    if count < 0:
        raise argparse.ArgumentTypeError("COUNT must be non-negative")
    return direction, count


def workload_direction(name: str) -> str:
    if name.endswith("_roundtrip"):
        return "roundtrip"
    if name.endswith("_reverse"):
        return "reverse"
    return "forward"


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_allowed_keys(
    value: dict[str, Any],
    name: str,
    allowed: set[str],
    errors: list[str],
) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        errors.append(f"{name} has unexpected field(s): {', '.join(unexpected)}")


def number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int)


def synthetic_side(binary: str, median_seconds: float) -> dict[str, Any]:
    samples = [median_seconds, median_seconds]
    rss_samples = [20 * 1024 * 1024, 22 * 1024 * 1024]
    command = [binary, "/tmp/reverie-benchmark-fixture.rev"]
    return {
        "command": command,
        "commands": [command],
        "expected_stdout": [["ok"]],
        "max_rss_bytes": max(rss_samples),
        "mean_seconds": statistics.mean(samples),
        "mean_max_rss_bytes": statistics.mean(rss_samples),
        "median_seconds": statistics.median(samples),
        "median_max_rss_bytes": statistics.median(rss_samples),
        "min_seconds": min(samples),
        "samples_max_rss_bytes": rss_samples,
        "samples_seconds": samples,
    }


def synthetic_binary(path: str) -> dict[str, Any]:
    return {
        "exists": True,
        "mtime_utc": "2026-05-30T00:00:00+00:00",
        "path": path,
        "sha256": "0" * 64,
        "size_bytes": 1,
    }


def synthetic_source(path: str) -> dict[str, Any]:
    return {
        "dirty": False,
        "path": path,
        "revision": "synthetic",
        "status": [],
    }


def valid_synthetic_artifact() -> dict[str, Any]:
    jana_bin = "/tmp/jana/bin/janus"
    reverie_bin = "/tmp/reverie"
    jana = synthetic_side(jana_bin, 2.0)
    reverie = synthetic_side(reverie_bin, 1.0)
    return {
        "benchmarks": [
            {
                "direction": "forward",
                "jana": jana,
                "name": "synthetic_forward",
                "passes_min_speedup": True,
                "reverie": reverie,
                "speedup": 2.0,
                "memory_ratio": 1.0,
            }
        ],
        "command_timeout_seconds": 1.0,
        "generated_at": "2026-05-30T00:00:00+00:00",
        "jana_baseline": jana_bin,
        "metadata": {
            "host": {
                "cpu_count": 1,
                "machine": "synthetic",
                "platform": "synthetic",
                "python": "synthetic",
            },
            "jana": {
                "binary": synthetic_binary(jana_bin),
                "source": synthetic_source("/tmp/jana"),
            },
            "reverie": {
                "binary": synthetic_binary(reverie_bin),
                "source": synthetic_source(str(REPO_ROOT)),
            },
        },
        "min_speedup": 1.25,
        "performance_gate": {
            "min_geomean_speedup": 1.5,
            "min_median_speedup": 1.5,
            "min_observed_speedup": 1.5,
        },
        "reverie_binary": reverie_bin,
        "runs": 2,
        "selected_workloads": ["synthetic_forward"],
        "warmup": 0,
        "working_directory": str(REPO_ROOT),
    }


def validate_synthetic_artifact(data: dict[str, Any]) -> list[str]:
    return validate_artifact(
        data,
        1,
        [],
        ["synthetic_forward"],
        True,
        1.5,
        1.5,
        1.5,
        2,
        0,
        1.0,
        1.25,
        [("forward", 1)],
        ["synthetic_forward"],
        True,
        True,
        False,
        False,
        "bin/janus",
        "reverie",
    )


def validate_markdown_summary(data: dict[str, Any], markdown: str) -> list[str]:
    errors = []
    benchmarks = data.get("benchmarks")
    if not isinstance(benchmarks, list):
        errors.append("Markdown summary cannot be checked without benchmark rows")
        return errors
    direction_counts = {direction: 0 for direction in DEFAULT_DIRECTIONS}
    speedups = []
    for benchmark in benchmarks:
        if not isinstance(benchmark, dict):
            continue
        direction = benchmark.get("direction")
        if direction in direction_counts:
            direction_counts[direction] += 1
        speedup = benchmark.get("speedup")
        if number(speedup):
            speedups.append(float(speedup))
    expected_snippets = [
        f"Workloads: `{len(benchmarks)}`",
        (
            "Directions: "
            f"`{direction_counts['forward']}` forward, "
            f"`{direction_counts['reverse']}` reverse, "
            f"`{direction_counts['roundtrip']}` roundtrip"
        ),
    ]
    failed = [
        benchmark
        for benchmark in benchmarks
        if isinstance(benchmark, dict) and benchmark.get("passes_min_speedup") is False
    ]
    if speedups:
        gate = data.get("performance_gate")
        min_observed_speedup = (
            gate.get("min_observed_speedup") if isinstance(gate, dict) else None
        )
        min_median_speedup = (
            gate.get("min_median_speedup") if isinstance(gate, dict) else None
        )
        min_geomean_speedup = (
            gate.get("min_geomean_speedup") if isinstance(gate, dict) else None
        )
        aggregate_failed = (
            number(min_observed_speedup) and min(speedups) < min_observed_speedup
        ) or (
            number(min_median_speedup)
            and statistics.median(speedups) < min_median_speedup
        ) or (
            number(min_geomean_speedup)
            and statistics.geometric_mean(speedups) < min_geomean_speedup
        )
        expected_snippets.append(
            f"Gate: `{'FAIL' if failed or aggregate_failed else 'PASS'}`"
        )
    for snippet in expected_snippets:
        if snippet not in markdown:
            errors.append(f"Markdown summary missing `{snippet}`")
    expected_markdown = render_markdown(data)
    if markdown != expected_markdown:
        errors.append("Markdown summary does not match benchmark artifact")
    return errors


def expect_self_test_error(label: str, data: dict[str, Any], needle: str) -> None:
    errors = validate_synthetic_artifact(data)
    if not any(needle in error for error in errors):
        raise AssertionError(f"{label} did not report `{needle}`; errors were {errors}")


def run_self_tests() -> int:
    numeric_cases = [
        ("positive int", 1, True),
        ("positive float", 1.25, True),
        ("zero", 0, True),
        ("true bool", True, False),
        ("false bool", False, False),
        ("nan", math.nan, False),
        ("infinity", math.inf, False),
        ("string", "1", False),
    ]
    failures = [
        f"{label}: expected {expected}, found {observed}"
        for label, value, expected in numeric_cases
        for observed in [number(value)]
        if observed != expected
    ]
    integer_cases = [
        ("positive int", 1, True),
        ("zero", 0, True),
        ("negative int", -1, True),
        ("true bool", True, False),
        ("false bool", False, False),
        ("float", 1.0, False),
        ("string", "1", False),
    ]
    failures.extend(
        f"integer {label}: expected {expected}, found {observed}"
        for label, value, expected in integer_cases
        for observed in [integer(value)]
        if observed != expected
    )
    if failures:
        for failure in failures:
            print(f"error: self-test number() failed: {failure}", file=sys.stderr)
        return 1

    valid = valid_synthetic_artifact()
    errors = validate_synthetic_artifact(valid)
    if errors:
        for error in errors:
            print(f"error: self-test valid artifact failed: {error}", file=sys.stderr)
        return 1
    markdown_errors = validate_markdown_summary(valid, render_markdown(valid))
    if markdown_errors:
        for error in markdown_errors:
            print(
                f"error: self-test valid Markdown summary failed: {error}",
                file=sys.stderr,
            )
        return 1

    duplicate = copy.deepcopy(valid)
    duplicate["benchmarks"].append(copy.deepcopy(duplicate["benchmarks"][0]))
    duplicate["selected_workloads"].append("synthetic_forward")
    stale_selected = copy.deepcopy(valid)
    stale_selected["selected_workloads"] = ["synthetic_reverse"]
    bad_speedup = copy.deepcopy(valid)
    bad_speedup["benchmarks"][0]["speedup"] = 1.0
    missing_gate = copy.deepcopy(valid)
    del missing_gate["performance_gate"]["min_geomean_speedup"]
    bad_command_binary = copy.deepcopy(valid)
    bad_command_binary["benchmarks"][0]["reverie"]["commands"][0][0] = "/tmp/other-reverie"
    bad_direction = copy.deepcopy(valid)
    bad_direction["benchmarks"][0]["direction"] = "reverse"
    stale_direction_markdown = render_markdown(valid).replace(
        "Directions: `1` forward, `0` reverse, `0` roundtrip",
        "Directions: `0` forward, `1` reverse, `0` roundtrip",
    )
    stale_gate_markdown = render_markdown(valid).replace("Gate: `PASS`", "Gate: `FAIL`")

    try:
        expect_self_test_error("duplicate benchmark", duplicate, "duplicate benchmark name")
        expect_self_test_error(
            "stale selected workloads",
            stale_selected,
            "selected_workloads must match benchmark row order",
        )
        expect_self_test_error("bad speedup", bad_speedup, "speedup does not match")
        expect_self_test_error(
            "missing performance gate",
            missing_gate,
            "performance_gate.min_geomean_speedup must be recorded",
        )
        expect_self_test_error(
            "bad command binary",
            bad_command_binary,
            "does not match reverie binary path",
        )
        expect_self_test_error(
            "bad direction",
            bad_direction,
            "direction `reverse` does not match workload name",
        )
        markdown_errors = validate_markdown_summary(valid, stale_direction_markdown)
        if not any("Directions:" in error for error in markdown_errors):
            raise AssertionError(
                "stale Markdown direction summary did not report direction mismatch"
            )
        markdown_errors = validate_markdown_summary(valid, stale_gate_markdown)
        if not any("Gate:" in error for error in markdown_errors):
            raise AssertionError(
                "stale Markdown gate summary did not report gate mismatch"
            )
    except AssertionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("ok: benchmark artifact validator self-tests passed")
    return 0


def gate_value(data: dict[str, Any], key: str, errors: list[str]) -> Optional[float]:
    gate = data.get("performance_gate")
    if gate is None:
        return None
    if not isinstance(gate, dict):
        errors.append("performance_gate must be an object")
        return None
    value = gate.get(key)
    if value is None:
        return None
    require(
        number(value) and value > 0,
        f"performance_gate.{key} must be a positive finite number",
        errors,
    )
    return float(value) if number(value) else None


def effective_gate(
    cli_value: Optional[float],
    artifact_value: Optional[float],
    key: str,
    errors: list[str],
) -> Optional[float]:
    if cli_value is None:
        return artifact_value
    if artifact_value is not None:
        require(
            cli_value == artifact_value,
            (
                f"--{key.replace('_', '-')} {cli_value:.2f}x does not match "
                f"artifact performance_gate.{key} {artifact_value:.2f}x"
            ),
            errors,
        )
    return cli_value


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


def read_expected_names(path: Path) -> list[str]:
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


def validate_sample_summary(
    benchmark_name: str,
    side_name: str,
    runs: int,
    side: dict[str, Any],
    expected_binary_path: Optional[str],
    expect_source_digests: bool,
    verify_file_digests: bool,
    errors: list[str],
) -> None:
    validate_allowed_keys(
        side,
        f"{benchmark_name}.{side_name}",
        {
            "command",
            "commands",
            "expected_stdout",
            "max_rss_bytes",
            "mean_seconds",
            "mean_max_rss_bytes",
            "median_seconds",
            "median_max_rss_bytes",
            "min_seconds",
            "samples_max_rss_bytes",
            "samples_seconds",
            "source_files",
        },
        errors,
    )
    samples = side.get("samples_seconds")
    if not isinstance(samples, list):
        errors.append(f"{benchmark_name}.{side_name}: samples_seconds must be a list")
        return
    require(
        len(samples) == runs,
        f"{benchmark_name}.{side_name}: expected {runs} sample(s), found {len(samples)}",
        errors,
    )
    if not samples:
        return
    bad_samples = [sample for sample in samples if not number(sample) or sample <= 0]
    require(
        not bad_samples,
        f"{benchmark_name}.{side_name}: samples must be positive finite numbers",
        errors,
    )
    if bad_samples:
        return
    require(
        side.get("median_seconds") == statistics.median(samples),
        f"{benchmark_name}.{side_name}: median_seconds does not match samples",
        errors,
    )
    require(
        side.get("mean_seconds") == statistics.mean(samples),
        f"{benchmark_name}.{side_name}: mean_seconds does not match samples",
        errors,
    )
    require(
        side.get("min_seconds") == min(samples),
        f"{benchmark_name}.{side_name}: min_seconds does not match samples",
        errors,
    )
    rss_samples = side.get("samples_max_rss_bytes")
    if rss_samples is not None:
        if not isinstance(rss_samples, list):
            errors.append(
                f"{benchmark_name}.{side_name}: samples_max_rss_bytes must be a list"
            )
            return
        require(
            len(rss_samples) == runs,
            (
                f"{benchmark_name}.{side_name}: expected {runs} RSS sample(s), "
                f"found {len(rss_samples)}"
            ),
            errors,
        )
        bad_rss_samples = [
            sample
            for sample in rss_samples
            if not number(sample) or sample < 0
        ]
        require(
            not bad_rss_samples,
            (
                f"{benchmark_name}.{side_name}: RSS samples must be "
                "non-negative finite numbers"
            ),
            errors,
        )
        if bad_rss_samples:
            return
        require(
            side.get("median_max_rss_bytes") == statistics.median(rss_samples),
            (
                f"{benchmark_name}.{side_name}: median_max_rss_bytes "
                "does not match RSS samples"
            ),
            errors,
        )
        require(
            side.get("mean_max_rss_bytes") == statistics.mean(rss_samples),
            (
                f"{benchmark_name}.{side_name}: mean_max_rss_bytes "
                "does not match RSS samples"
            ),
            errors,
        )
        require(
            side.get("max_rss_bytes") == max(rss_samples),
            f"{benchmark_name}.{side_name}: max_rss_bytes does not match RSS samples",
            errors,
        )
    commands = side.get("commands")
    if not isinstance(commands, list) or len(commands) < 1:
        errors.append(f"{benchmark_name}.{side_name}: commands must be a non-empty list")
        return
    expected_stdout = side.get("expected_stdout")
    if not isinstance(expected_stdout, list):
        errors.append(f"{benchmark_name}.{side_name}: expected_stdout must be a list")
        return
    require(
        len(expected_stdout) == len(commands),
        (
            f"{benchmark_name}.{side_name}: expected_stdout length "
            "must match commands length"
        ),
        errors,
    )
    for command_index, command in enumerate(commands):
        command_name = f"{benchmark_name}.{side_name}.commands[{command_index}]"
        if not isinstance(command, list) or len(command) < 1:
            errors.append(f"{command_name} must be a non-empty argv list")
            continue
        require(
            all(isinstance(arg, str) and arg for arg in command),
            f"{command_name} must contain only non-empty strings",
            errors,
        )
        for arg_index, arg in enumerate(command):
            if isinstance(arg, str) and arg.endswith((".ja", ".janus", ".rev")):
                validate_absolute_path(arg, f"{command_name}[{arg_index}]", errors)
        if expected_binary_path is not None:
            require(
                command[0] == expected_binary_path,
                (
                    f"{command_name}[0] `{command[0]}` does not match "
                    f"{side_name} binary path `{expected_binary_path}`"
                ),
                errors,
            )
    validate_side_source_files(
        benchmark_name,
        side_name,
        side.get("source_files"),
        commands,
        expect_source_digests,
        verify_file_digests,
        errors,
    )
    for stdout_index, expected in enumerate(expected_stdout):
        expected_name = f"{benchmark_name}.{side_name}.expected_stdout[{stdout_index}]"
        if not isinstance(expected, list):
            errors.append(f"{expected_name} must be a list")
            continue
        require(
            all(isinstance(line, str) and line for line in expected),
            f"{expected_name} must contain only non-empty strings",
            errors,
        )
    legacy_command = side.get("command")
    if len(commands) == 1:
        require(
            legacy_command == commands[0],
            f"{benchmark_name}.{side_name}.command must match commands[0]",
            errors,
        )
    else:
        require(
            legacy_command is None,
            f"{benchmark_name}.{side_name}.command must be omitted for command sequences",
            errors,
        )


def command_source_paths(commands: list[Any]) -> list[str]:
    paths: set[str] = set()
    for command in commands:
        if not isinstance(command, list):
            continue
        for arg in command:
            if isinstance(arg, str) and arg.endswith((".ja", ".janus", ".rev")):
                paths.add(arg)
    return sorted(paths)


def validate_side_source_files(
    benchmark_name: str,
    side_name: str,
    source_files: Any,
    commands: list[Any],
    expect_source_digests: bool,
    verify_file_digests: bool,
    errors: list[str],
) -> None:
    expected_paths = command_source_paths(commands)
    if source_files is None:
        if expect_source_digests and expected_paths:
            errors.append(
                f"{benchmark_name}.{side_name}.source_files must be recorded "
                "when --expect-source-digests is set"
            )
        return
    source_name = f"{benchmark_name}.{side_name}.source_files"
    if not isinstance(source_files, list):
        errors.append(f"{source_name} must be a list")
        return
    seen_paths: set[str] = set()
    actual_paths: list[str] = []
    for index, entry in enumerate(source_files):
        entry_name = f"{source_name}[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{entry_name} must be an object")
            continue
        validate_allowed_keys(entry, entry_name, {"path", "sha256"}, errors)
        path = validate_string(entry.get("path"), f"{entry_name}.path", errors)
        validate_absolute_path(path, f"{entry_name}.path", errors)
        if path is not None:
            if path in seen_paths:
                errors.append(f"{source_name} contains duplicate path `{path}`")
            seen_paths.add(path)
            actual_paths.append(path)
        sha256 = entry.get("sha256")
        require(
            isinstance(sha256, str) and bool(SHA256_RE.fullmatch(sha256)),
            f"{entry_name}.sha256 must be a lowercase 64-character hex digest",
            errors,
        )
        if (
            verify_file_digests
            and path is not None
            and isinstance(sha256, str)
            and SHA256_RE.fullmatch(sha256)
        ):
            validate_recorded_sha256(Path(path), sha256, f"{entry_name}.sha256", errors)
    require(
        sorted(actual_paths) == expected_paths,
        (
            f"{source_name} paths must match source file arguments; "
            f"expected {expected_paths}, found {sorted(actual_paths)}"
        ),
        errors,
    )


def validate_string(value: Any, name: str, errors: list[str]) -> Optional[str]:
    if not isinstance(value, str) or not value:
        errors.append(f"{name} must be a non-empty string")
        return None
    return value


def validate_iso_datetime(value: Any, name: str, errors: list[str]) -> None:
    text = validate_string(value, name, errors)
    if text is None:
        return
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        errors.append(f"{name} must be an ISO-8601 datetime")
        return
    require(
        parsed.tzinfo is not None,
        f"{name} must include a timezone offset",
        errors,
    )


def validate_path_suffix(
    value: Optional[str],
    name: str,
    expected_suffix: Optional[str],
    errors: list[str],
) -> None:
    if expected_suffix is not None and isinstance(value, str):
        require(
            value.endswith(expected_suffix),
            f"{name} `{value}` does not end with `{expected_suffix}`",
            errors,
        )


def validate_absolute_path(value: Optional[str], name: str, errors: list[str]) -> None:
    if value is not None:
        require(Path(value).is_absolute(), f"{name} must be an absolute path", errors)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_recorded_sha256(
    path: Path,
    expected: str,
    name: str,
    errors: list[str],
) -> None:
    try:
        actual = file_sha256(path)
    except OSError as error:
        errors.append(f"{name} could not be verified from `{path}`: {error}")
        return
    require(
        actual == expected,
        f"{name} does not match current file `{path}`",
        errors,
    )


def validate_binary_metadata(
    metadata: dict[str, Any],
    name: str,
    expected_suffix: Optional[str],
    expected_path: Optional[str],
    verify_file_digests: bool,
    errors: list[str],
) -> None:
    validate_allowed_keys(
        metadata,
        f"metadata.{name}.binary",
        {"exists", "mtime_utc", "path", "sha256", "size_bytes"},
        errors,
    )
    path = validate_string(metadata.get("path"), f"metadata.{name}.binary.path", errors)
    validate_absolute_path(path, f"metadata.{name}.binary.path", errors)
    validate_path_suffix(path, f"metadata.{name}.binary.path", expected_suffix, errors)
    if expected_path is not None and path is not None:
        require(
            path == expected_path,
            (
                f"metadata.{name}.binary.path `{path}` does not match "
                f"top-level {name} binary path `{expected_path}`"
            ),
            errors,
        )
    require(
        metadata.get("exists") is True,
        f"metadata.{name}.binary.exists must be true",
        errors,
    )
    require(
        integer(metadata.get("size_bytes")) and metadata["size_bytes"] > 0,
        f"metadata.{name}.binary.size_bytes must be a positive integer",
        errors,
    )
    validate_iso_datetime(
        metadata.get("mtime_utc"),
        f"metadata.{name}.binary.mtime_utc",
        errors,
    )
    if "sha256" in metadata:
        sha256 = metadata.get("sha256")
        require(
            isinstance(sha256, str) and bool(SHA256_RE.fullmatch(sha256)),
            f"metadata.{name}.binary.sha256 must be a lowercase 64-character hex digest",
            errors,
        )
        if (
            verify_file_digests
            and path is not None
            and isinstance(sha256, str)
            and SHA256_RE.fullmatch(sha256)
        ):
            validate_recorded_sha256(
                Path(path),
                sha256,
                f"metadata.{name}.binary.sha256",
                errors,
            )


def validate_source_metadata(
    metadata: dict[str, Any],
    name: str,
    errors: list[str],
) -> None:
    validate_allowed_keys(
        metadata,
        f"metadata.{name}.source",
        {"dirty", "path", "revision", "status"},
        errors,
    )
    path = validate_string(metadata.get("path"), f"metadata.{name}.source.path", errors)
    validate_absolute_path(path, f"metadata.{name}.source.path", errors)
    require(
        "revision" in metadata,
        f"metadata.{name}.source.revision must be recorded",
        errors,
    )
    revision = metadata.get("revision")
    require(
        revision is None or (isinstance(revision, str) and bool(revision)),
        f"metadata.{name}.source.revision must be null or a non-empty string",
        errors,
    )
    require(
        isinstance(metadata.get("dirty"), bool),
        f"metadata.{name}.source.dirty must be a boolean",
        errors,
    )
    status = metadata.get("status")
    require(
        isinstance(status, list)
        and all(isinstance(entry, str) and entry for entry in status),
        f"metadata.{name}.source.status must be a list of non-empty strings",
        errors,
    )


def validate_side_metadata(
    metadata: Any,
    name: str,
    expected_bin_suffix: Optional[str],
    expected_bin_path: Optional[str],
    verify_file_digests: bool,
    errors: list[str],
) -> None:
    if not isinstance(metadata, dict):
        errors.append(f"metadata.{name} must be an object")
        return
    validate_allowed_keys(metadata, f"metadata.{name}", {"binary", "source"}, errors)
    source = metadata.get("source")
    binary = metadata.get("binary")
    if not isinstance(source, dict):
        errors.append(f"metadata.{name}.source must be an object")
    else:
        validate_source_metadata(source, name, errors)
    if not isinstance(binary, dict):
        errors.append(f"metadata.{name}.binary must be an object")
    else:
        validate_binary_metadata(
            binary,
            name,
            expected_bin_suffix,
            expected_bin_path,
            verify_file_digests,
            errors,
        )


def validate_metadata(
    data: dict[str, Any],
    expected_jana_bin_suffix: Optional[str],
    expected_reverie_bin_suffix: Optional[str],
    verify_file_digests: bool,
    errors: list[str],
) -> None:
    validate_iso_datetime(data.get("generated_at"), "generated_at", errors)
    jana_baseline = validate_string(data.get("jana_baseline"), "jana_baseline", errors)
    reverie_binary = validate_string(data.get("reverie_binary"), "reverie_binary", errors)
    working_directory = validate_string(
        data.get("working_directory"),
        "working_directory",
        errors,
    )
    validate_absolute_path(jana_baseline, "jana_baseline", errors)
    validate_absolute_path(reverie_binary, "reverie_binary", errors)
    validate_absolute_path(working_directory, "working_directory", errors)
    if working_directory is not None:
        require(
            working_directory == str(REPO_ROOT),
            (
                f"working_directory `{working_directory}` does not match "
                f"repository root `{REPO_ROOT}`"
            ),
            errors,
        )
    validate_path_suffix(
        jana_baseline,
        "jana_baseline",
        expected_jana_bin_suffix,
        errors,
    )
    validate_path_suffix(
        reverie_binary,
        "reverie_binary",
        expected_reverie_bin_suffix,
        errors,
    )

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
        return
    validate_allowed_keys(metadata, "metadata", {"host", "jana", "reverie"}, errors)
    host = metadata.get("host")
    if not isinstance(host, dict):
        errors.append("metadata.host must be an object")
    else:
        validate_allowed_keys(
            host,
            "metadata.host",
            {"cpu_count", "machine", "platform", "python"},
            errors,
        )
        require(
            integer(host.get("cpu_count")) and host["cpu_count"] >= 1,
            "metadata.host.cpu_count must be a positive integer",
            errors,
        )
        validate_string(host.get("platform"), "metadata.host.platform", errors)
        validate_string(host.get("machine"), "metadata.host.machine", errors)
        validate_string(host.get("python"), "metadata.host.python", errors)
    validate_side_metadata(
        metadata.get("jana"),
        "jana",
        expected_jana_bin_suffix,
        jana_baseline,
        verify_file_digests,
        errors,
    )
    validate_side_metadata(
        metadata.get("reverie"),
        "reverie",
        expected_reverie_bin_suffix,
        reverie_binary,
        verify_file_digests,
        errors,
    )


def validate_artifact(
    data: dict[str, Any],
    min_workloads: int,
    required_directions: list[str],
    expected_workloads: list[str],
    exact_workloads: bool,
    min_observed_speedup: Optional[float],
    min_median_speedup: Optional[float],
    min_geomean_speedup: Optional[float],
    expected_runs: Optional[int],
    expected_warmup: Optional[int],
    expected_command_timeout: Optional[float],
    expected_min_speedup: Optional[float],
    expected_direction_counts: list[tuple[str, int]],
    exact_workload_names: Optional[list[str]],
    expect_workload_order: bool,
    expect_performance_gate: bool,
    expect_source_digests: bool,
    verify_file_digests: bool,
    expected_jana_bin_suffix: Optional[str],
    expected_reverie_bin_suffix: Optional[str],
) -> list[str]:
    errors: list[str] = []
    validate_allowed_keys(
        data,
        "artifact",
        {
            "benchmarks",
            "command_timeout_seconds",
            "generated_at",
            "jana_baseline",
            "metadata",
            "min_speedup",
            "performance_gate",
            "reverie_binary",
            "runs",
            "selected_workloads",
            "warmup",
            "working_directory",
        },
        errors,
    )
    validate_metadata(
        data,
        expected_jana_bin_suffix,
        expected_reverie_bin_suffix,
        verify_file_digests,
        errors,
    )
    benchmarks = data.get("benchmarks")
    jana_baseline = data.get("jana_baseline")
    reverie_binary = data.get("reverie_binary")
    runs = data.get("runs")
    warmup = data.get("warmup")
    command_timeout = data.get("command_timeout_seconds")
    min_speedup = data.get("min_speedup")
    selected_workloads = data.get("selected_workloads")
    performance_gate = data.get("performance_gate")
    if expect_performance_gate and not isinstance(performance_gate, dict):
        errors.append("performance_gate must be recorded when --expect-performance-gate is set")
    if isinstance(performance_gate, dict):
        validate_allowed_keys(
            performance_gate,
            "performance_gate",
            {
                "min_observed_speedup",
                "min_median_speedup",
                "min_geomean_speedup",
            },
            errors,
        )
        for key in (
            "min_observed_speedup",
            "min_median_speedup",
            "min_geomean_speedup",
        ):
            require(
                key in performance_gate,
                f"performance_gate.{key} must be recorded",
                errors,
            )
    artifact_min_observed_speedup = gate_value(data, "min_observed_speedup", errors)
    artifact_min_median_speedup = gate_value(data, "min_median_speedup", errors)
    artifact_min_geomean_speedup = gate_value(data, "min_geomean_speedup", errors)
    min_observed_speedup = effective_gate(
        min_observed_speedup,
        artifact_min_observed_speedup,
        "min_observed_speedup",
        errors,
    )
    min_median_speedup = effective_gate(
        min_median_speedup,
        artifact_min_median_speedup,
        "min_median_speedup",
        errors,
    )
    min_geomean_speedup = effective_gate(
        min_geomean_speedup,
        artifact_min_geomean_speedup,
        "min_geomean_speedup",
        errors,
    )

    require(integer(runs) and runs >= 1, "runs must be a positive integer", errors)
    require(
        integer(warmup) and warmup >= 0,
        "warmup must be a non-negative integer",
        errors,
    )
    require(
        isinstance(benchmarks, list),
        "benchmarks must be a list",
        errors,
    )
    if not isinstance(benchmarks, list):
        return errors
    require(
        len(benchmarks) >= min_workloads,
        f"expected at least {min_workloads} workload(s), found {len(benchmarks)}",
        errors,
    )
    require(
        min_speedup is None or (number(min_speedup) and min_speedup > 0),
        "min_speedup must be null or a positive finite number",
        errors,
    )
    require(
        number(command_timeout) and command_timeout > 0,
        "command_timeout_seconds must be a positive finite number",
        errors,
    )
    if expected_runs is not None:
        require(
            runs == expected_runs,
            f"runs `{runs}` does not match expected `{expected_runs}`",
            errors,
        )
    if expected_warmup is not None:
        require(
            warmup == expected_warmup,
            f"warmup `{warmup}` does not match expected `{expected_warmup}`",
            errors,
        )
    if expected_command_timeout is not None:
        require(
            command_timeout == expected_command_timeout,
            (
                f"command_timeout_seconds `{command_timeout}` does not match "
                f"expected `{expected_command_timeout}`"
            ),
            errors,
        )
    if expected_min_speedup is not None:
        require(
            min_speedup == expected_min_speedup,
            f"min_speedup `{min_speedup}` does not match expected `{expected_min_speedup}`",
            errors,
        )

    names: set[str] = set()
    ordered_names: list[str] = []
    directions = {direction: 0 for direction in DEFAULT_DIRECTIONS}
    speedups: list[float] = []
    for index, benchmark in enumerate(benchmarks):
        if not isinstance(benchmark, dict):
            errors.append(f"benchmarks[{index}] must be an object")
            continue
        validate_allowed_keys(
            benchmark,
            f"benchmarks[{index}]",
            {
                "direction",
                "jana",
                "name",
                "passes_min_speedup",
                "reverie",
                "speedup",
                "memory_ratio",
            },
            errors,
        )
        name = benchmark.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"benchmarks[{index}].name must be a non-empty string")
            continue
        require(name not in names, f"duplicate benchmark name `{name}`", errors)
        names.add(name)
        ordered_names.append(name)
        direction = benchmark.get("direction")
        require(
            direction in DEFAULT_DIRECTIONS,
            f"{name}: direction must be one of {', '.join(DEFAULT_DIRECTIONS)}",
            errors,
        )
        if direction in DEFAULT_DIRECTIONS:
            require(
                direction == workload_direction(name),
                f"{name}: direction `{direction}` does not match workload name",
                errors,
            )
            directions[direction] += 1

        jana = benchmark.get("jana")
        reverie = benchmark.get("reverie")
        if not isinstance(jana, dict):
            errors.append(f"{name}.jana must be an object")
            continue
        if not isinstance(reverie, dict):
            errors.append(f"{name}.reverie must be an object")
            continue
        if integer(runs):
            validate_sample_summary(
                name,
                "jana",
                runs,
                jana,
                jana_baseline if isinstance(jana_baseline, str) else None,
                expect_source_digests,
                verify_file_digests,
                errors,
            )
            validate_sample_summary(
                name,
                "reverie",
                runs,
                reverie,
                reverie_binary if isinstance(reverie_binary, str) else None,
                expect_source_digests,
                verify_file_digests,
                errors,
            )

        jana_median = jana.get("median_seconds")
        reverie_median = reverie.get("median_seconds")
        speedup = benchmark.get("speedup")
        if number(jana_median) and number(reverie_median) and reverie_median > 0:
            require(
                speedup == jana_median / reverie_median,
                f"{name}: speedup does not match Jana/Reverie medians",
                errors,
            )
        else:
            errors.append(f"{name}: medians must be positive finite numbers")
        jana_rss_median = jana.get("median_max_rss_bytes")
        reverie_rss_median = reverie.get("median_max_rss_bytes")
        memory_ratio = benchmark.get("memory_ratio")
        if (
            number(jana_rss_median)
            and number(reverie_rss_median)
            and reverie_rss_median > 0
        ):
            require(
                memory_ratio == jana_rss_median / reverie_rss_median,
                f"{name}: memory_ratio does not match Jana/Reverie median RSS",
                errors,
            )
        elif memory_ratio is not None:
            errors.append(
                f"{name}: memory_ratio must be null unless median RSS is recorded"
            )
        if min_speedup is not None and number(speedup):
            require(
                isinstance(benchmark.get("passes_min_speedup"), bool),
                f"{name}: passes_min_speedup must be a boolean when min_speedup is set",
                errors,
            )
            require(
                benchmark.get("passes_min_speedup") == (speedup >= min_speedup),
                f"{name}: passes_min_speedup does not match min_speedup gate",
                errors,
            )
        elif min_speedup is None:
            require(
                benchmark.get("passes_min_speedup") is None,
                f"{name}: passes_min_speedup must be null when min_speedup is not set",
                errors,
            )
        if number(speedup):
            speedups.append(float(speedup))

    if not isinstance(selected_workloads, list):
        errors.append("selected_workloads must be a list")
    else:
        require(
            all(isinstance(name, str) and name for name in selected_workloads),
            "selected_workloads must contain only non-empty strings",
            errors,
        )
        require(
            len(set(selected_workloads)) == len(selected_workloads),
            "selected_workloads must not contain duplicates",
            errors,
        )
        require(
            selected_workloads == ordered_names,
            "selected_workloads must match benchmark row order",
            errors,
        )

    for direction in required_directions:
        require(
            directions[direction] > 0,
            f"missing required {direction} workload",
            errors,
        )
    for direction, count in expected_direction_counts:
        observed = directions[direction]
        require(
            observed == count,
            f"{direction} workload count `{observed}` does not match expected `{count}`",
            errors,
        )
    for workload in expected_workloads:
        require(
            workload in names,
            f"missing expected workload `{workload}`",
            errors,
        )
    if exact_workload_names is not None:
        exact_workload_set = set(exact_workload_names)
        missing = sorted(exact_workload_set - names)
        unexpected = sorted(names - exact_workload_set)
        if missing:
            errors.append("missing manifest workload(s): " + ", ".join(missing))
        if unexpected:
            errors.append("unexpected manifest workload(s): " + ", ".join(unexpected))
    if exact_workloads:
        expected = set(expected_workloads)
        unexpected = sorted(names - expected)
        require(
            not unexpected,
            f"unexpected workload(s): {', '.join(unexpected)}",
            errors,
        )
    if expect_workload_order:
        checked_order = False
        expected_order = list(dict.fromkeys(expected_workloads))
        if expected_order and set(expected_order) == names:
            checked_order = True
            if ordered_names != expected_order:
                errors.append(
                    "benchmark workload order does not match expected workload order: "
                    f"expected {', '.join(expected_order)}; found {', '.join(ordered_names)}"
                )
        if exact_workload_names is not None and set(exact_workload_names) == names:
            checked_order = True
            if ordered_names != exact_workload_names:
                errors.append(
                    "benchmark workload order does not match manifest order: "
                    f"expected {', '.join(exact_workload_names)}; "
                    f"found {', '.join(ordered_names)}"
                )
        if not checked_order:
            errors.append(
                "--expect-workload-order requires --expect-workload, "
                "--expect-current-workloads, or --exact-workloads-file entries "
                "covering the artifact exactly"
            )
    if min_observed_speedup is not None and speedups:
        observed_min = min(speedups)
        require(
            observed_min >= min_observed_speedup,
            (
                "minimum observed speedup "
                f"{observed_min:.2f}x is below required {min_observed_speedup:.2f}x"
            ),
            errors,
        )
    if min_median_speedup is not None and speedups:
        observed_median = statistics.median(speedups)
        require(
            observed_median >= min_median_speedup,
            (
                "median observed speedup "
                f"{observed_median:.2f}x is below required {min_median_speedup:.2f}x"
            ),
            errors,
        )
    if min_geomean_speedup is not None and speedups:
        observed_geomean = statistics.geometric_mean(speedups)
        require(
            observed_geomean >= min_geomean_speedup,
            (
                "geometric mean observed speedup "
                f"{observed_geomean:.2f}x is below required {min_geomean_speedup:.2f}x"
            ),
            errors,
        )
    return errors


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    if args.expect_runs is not None and args.expect_runs < 1:
        print("error: --expect-runs must be at least 1", file=sys.stderr)
        return 1
    if args.expect_warmup is not None and args.expect_warmup < 0:
        print("error: --expect-warmup must be non-negative", file=sys.stderr)
        return 1
    if args.expect_command_timeout is not None and args.expect_command_timeout <= 0:
        print("error: --expect-command-timeout must be positive", file=sys.stderr)
        return 1
    if args.expect_min_speedup is not None and args.expect_min_speedup <= 0:
        print("error: --expect-min-speedup must be positive", file=sys.stderr)
        return 1
    if args.min_observed_speedup is not None and args.min_observed_speedup <= 0:
        print("error: --min-observed-speedup must be positive", file=sys.stderr)
        return 1
    if args.min_median_speedup is not None and args.min_median_speedup <= 0:
        print("error: --min-median-speedup must be positive", file=sys.stderr)
        return 1
    if args.min_geomean_speedup is not None and args.min_geomean_speedup <= 0:
        print("error: --min-geomean-speedup must be positive", file=sys.stderr)
        return 1
    expected_workloads = list(args.expect_workload)
    exact_workloads = args.exact_workloads
    exact_workload_names = None
    if args.exact_workloads_file is not None:
        try:
            exact_workload_names = read_expected_names(args.exact_workloads_file)
        except OSError as error:
            print(f"error: failed to read exact workloads file: {error}", file=sys.stderr)
            return 1
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    if args.expect_current_workloads:
        try:
            expected_workloads.extend(current_workloads())
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        exact_workloads = True

    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
    except OSError as error:
        print(f"error: failed to read benchmark artifact: {error}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as error:
        print(f"error: failed to parse benchmark artifact: {error}", file=sys.stderr)
        return 1
    if not isinstance(data, dict):
        print("error: artifact root must be a JSON object", file=sys.stderr)
        return 1
    errors = validate_artifact(
        data,
        args.min_workloads,
        args.require_direction,
        expected_workloads,
        exact_workloads,
        args.min_observed_speedup,
        args.min_median_speedup,
        args.min_geomean_speedup,
        args.expect_runs,
        args.expect_warmup,
        args.expect_command_timeout,
        args.expect_min_speedup,
        args.expect_direction_count,
        exact_workload_names,
        args.expect_workload_order,
        args.expect_performance_gate,
        args.expect_source_digests,
        args.verify_file_digests,
        args.expect_jana_bin_suffix,
        args.expect_reverie_bin_suffix,
    )
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    if args.expect_markdown_summary is not None:
        try:
            markdown = args.expect_markdown_summary.read_text(encoding="utf-8")
        except OSError as error:
            print(f"error: failed to read Markdown summary: {error}", file=sys.stderr)
            return 1
        markdown_errors = validate_markdown_summary(data, markdown)
        if markdown_errors:
            for error in markdown_errors:
                print(f"error: {error}", file=sys.stderr)
            return 1
    print(f"ok: validated {len(data.get('benchmarks', []))} benchmark workload(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
