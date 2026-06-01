#!/usr/bin/env python3
"""Validate an upstream Jana example audit artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "benchmarks" / "results" / "jana-example-audit.json"
TARGETS = ("jana", "reverie", "reverie_legacy_janus")
STATUSES = ("error", "failed", "ok", "ok_with_diagnostics", "timeout")


def parse_min_status(value: str) -> tuple[str, str, int]:
    return parse_status_count(value, "expected TARGET:STATUS:COUNT, such as reverie:ok:20")


def parse_status_count(value: str, message: str) -> tuple[str, str, int]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(message)
    target, status, count_text = parts
    if target not in TARGETS:
        raise argparse.ArgumentTypeError(
            f"target must be one of {', '.join(TARGETS)}"
        )
    if status not in STATUSES:
        raise argparse.ArgumentTypeError(
            f"status must be one of {', '.join(STATUSES)}"
        )
    try:
        count = int(count_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("COUNT must be an integer") from error
    if count < 0:
        raise argparse.ArgumentTypeError("COUNT must be non-negative")
    return target, status, count


def parse_expect_total(value: str) -> tuple[str, str, int]:
    return parse_status_count(
        value,
        "expected TARGET:STATUS:COUNT, such as jana:ok_with_diagnostics:2",
    )


def parse_expect_status(value: str) -> tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "expected TARGET:PATH:STATUS, such as reverie:basicExamples/turing.janus:failed"
        )
    target, path, status = parts
    if target not in TARGETS:
        raise argparse.ArgumentTypeError(
            f"target must be one of {', '.join(TARGETS)}"
        )
    if not path:
        raise argparse.ArgumentTypeError("PATH must be non-empty")
    if not status:
        raise argparse.ArgumentTypeError("STATUS must be non-empty")
    if status not in STATUSES:
        raise argparse.ArgumentTypeError(
            f"STATUS must be one of {', '.join(STATUSES)}"
        )
    return target, path, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Jana example compatibility audit JSON artifact."
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--min-examples",
        type=int,
        default=1,
        help="Require at least this many audited upstream files.",
    )
    parser.add_argument(
        "--min-status",
        action="append",
        type=parse_min_status,
        default=[],
        metavar="TARGET:STATUS:COUNT",
        help="Require at least COUNT examples with STATUS for TARGET. Can be repeated.",
    )
    parser.add_argument(
        "--expect-status",
        action="append",
        type=parse_expect_status,
        default=[],
        metavar="TARGET:PATH:STATUS",
        help="Require one audited PATH to have STATUS for TARGET. Can be repeated.",
    )
    parser.add_argument(
        "--expect-total",
        action="append",
        type=parse_expect_total,
        default=[],
        metavar="TARGET:STATUS:COUNT",
        help="Require exactly COUNT examples with STATUS for TARGET. Can be repeated.",
    )
    parser.add_argument(
        "--exact-paths-file",
        type=Path,
        help=(
            "Require audited paths to exactly match non-empty, non-comment "
            "lines from this file."
        ),
    )
    parser.add_argument(
        "--expect-timeout",
        type=float,
        help="Require the audit artifact to record exactly this timeout in seconds.",
    )
    parser.add_argument(
        "--expect-jana-bin-suffix",
        help="Require jana_bin to end with this path suffix.",
    )
    parser.add_argument(
        "--expect-reverie-bin-suffix",
        help="Require reverie_bin to end with this path suffix.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent audit-validator self-tests and exit.",
    )
    return parser.parse_args()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int)


def validate_allowed_keys(
    value: dict[str, Any],
    name: str,
    allowed: set[str],
    errors: list[str],
) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        errors.append(f"{name} has unexpected field(s): {', '.join(unexpected)}")


def status_counts(rows: list[dict[str, Any]], target: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        result = row.get(target)
        if not isinstance(result, dict):
            continue
        status = result.get("status")
        if not isinstance(status, str) or not status:
            continue
        counts[status] = counts.get(status, 0) + 1
    return counts


def validate_totals(totals: dict[str, Any], errors: list[str]) -> None:
    validate_allowed_keys(totals, "totals", set(TARGETS), errors)
    for target in TARGETS:
        counts = totals.get(target)
        if not isinstance(counts, dict):
            errors.append(f"totals.{target} must be an object")
            continue
        validate_allowed_keys(counts, f"totals.{target}", set(STATUSES), errors)
        for status, count in counts.items():
            require(
                integer(count) and count >= 0,
                f"totals.{target}.{status} must be a non-negative integer",
                errors,
            )


def read_expected_paths(path: Path) -> set[str]:
    paths: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped in paths:
            raise ValueError(f"{path}:{line_number}: duplicate path `{stripped}`")
        paths.add(stripped)
    return paths


def validate_artifact(
    data: dict[str, Any],
    min_examples: int,
    min_statuses: list[tuple[str, str, int]],
    expected_totals: list[tuple[str, str, int]],
    expected_statuses: list[tuple[str, str, str]],
    exact_paths: Optional[set[str]] = None,
    expected_timeout: Optional[float] = None,
    expected_jana_bin_suffix: Optional[str] = None,
    expected_reverie_bin_suffix: Optional[str] = None,
) -> list[str]:
    errors: list[str] = []
    validate_allowed_keys(
        data,
        "artifact",
        {
            "examples",
            "jana_bin",
            "jana_dir",
            "reverie_bin",
            "timeout_seconds",
            "totals",
        },
        errors,
    )
    examples = data.get("examples")
    totals = data.get("totals")
    timeout = data.get("timeout_seconds")
    jana_bin = data.get("jana_bin")
    reverie_bin = data.get("reverie_bin")

    require(isinstance(examples, list), "examples must be a list", errors)
    require(isinstance(totals, dict), "totals must be an object", errors)
    if isinstance(totals, dict):
        validate_totals(totals, errors)
    require(
        number(timeout) and timeout > 0,
        "timeout_seconds must be a positive number",
        errors,
    )
    require(isinstance(jana_bin, str) and jana_bin, "jana_bin must be a non-empty string", errors)
    require(
        isinstance(reverie_bin, str) and reverie_bin,
        "reverie_bin must be a non-empty string",
        errors,
    )
    if expected_timeout is not None:
        require(
            timeout == expected_timeout,
            f"timeout_seconds `{timeout}` does not match expected `{expected_timeout}`",
            errors,
        )
    if expected_jana_bin_suffix is not None and isinstance(jana_bin, str):
        require(
            jana_bin.endswith(expected_jana_bin_suffix),
            f"jana_bin `{jana_bin}` does not end with `{expected_jana_bin_suffix}`",
            errors,
        )
    if expected_reverie_bin_suffix is not None and isinstance(reverie_bin, str):
        require(
            reverie_bin.endswith(expected_reverie_bin_suffix),
            (
                f"reverie_bin `{reverie_bin}` does not end with "
                f"`{expected_reverie_bin_suffix}`"
            ),
            errors,
        )
    if not isinstance(examples, list):
        return errors
    require(
        len(examples) >= min_examples,
        f"expected at least {min_examples} example(s), found {len(examples)}",
        errors,
    )

    paths: set[str] = set()
    rows_by_path: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(examples):
        if not isinstance(row, dict):
            errors.append(f"examples[{index}] must be an object")
            continue
        validate_allowed_keys(
            row,
            f"examples[{index}]",
            {"jana", "path", "reverie", "reverie_legacy_janus"},
            errors,
        )
        path = row.get("path")
        if not isinstance(path, str) or not path:
            errors.append(f"examples[{index}].path must be a non-empty string")
            continue
        require(path not in paths, f"duplicate example path `{path}`", errors)
        paths.add(path)
        rows_by_path[path] = row
        for target in TARGETS:
            result = row.get(target)
            if not isinstance(result, dict):
                errors.append(f"{path}.{target} must be an object")
                continue
            validate_allowed_keys(
                result,
                f"{path}.{target}",
                {"exit_code", "message", "status", "stderr_lines", "stdout_lines"},
                errors,
            )
            status = result.get("status")
            require(
                isinstance(status, str) and status,
                f"{path}.{target}.status must be a non-empty string",
                errors,
            )
            if isinstance(status, str) and status not in STATUSES:
                errors.append(
                    f"{path}.{target}.status `{status}` must be one of "
                    + ", ".join(STATUSES)
                )
            exit_code = result.get("exit_code")
            require(
                exit_code is None or integer(exit_code),
                f"{path}.{target}.exit_code must be an integer or null",
                errors,
            )
            for key in ("stdout_lines", "stderr_lines"):
                require(
                    integer(result.get(key)) and result[key] >= 0,
                    f"{path}.{target}.{key} must be a non-negative integer",
                    errors,
                )

    if exact_paths is not None:
        missing = sorted(exact_paths - paths)
        unexpected = sorted(paths - exact_paths)
        if missing:
            errors.append("missing audited path(s): " + ", ".join(missing))
        if unexpected:
            errors.append("unexpected audited path(s): " + ", ".join(unexpected))

    if isinstance(totals, dict):
        for target in TARGETS:
            observed = status_counts(
                [row for row in examples if isinstance(row, dict)],
                target,
            )
            require(
                totals.get(target) == dict(sorted(observed.items())),
                f"totals.{target} does not match example statuses",
                errors,
            )

    observed_by_target = {
        target: status_counts(
            [row for row in examples if isinstance(row, dict)],
            target,
        )
        for target in TARGETS
    }
    for target, status, count in min_statuses:
        observed = observed_by_target[target].get(status, 0)
        require(
            observed >= count,
            f"{target} has {observed} `{status}` example(s), expected at least {count}",
            errors,
        )

    for target, status, count in expected_totals:
        observed = observed_by_target[target].get(status, 0)
        require(
            observed == count,
            f"{target} has {observed} `{status}` example(s), expected exactly {count}",
            errors,
        )

    for target, path, status in expected_statuses:
        row = rows_by_path.get(path)
        if row is None:
            errors.append(f"missing expected example `{path}`")
            continue
        result = row.get(target)
        observed = result.get("status") if isinstance(result, dict) else None
        require(
            observed == status,
            f"{path}.{target} status `{observed}` does not match expected `{status}`",
            errors,
        )
    return errors


def valid_synthetic_artifact() -> dict[str, Any]:
    return {
        "examples": [
            {
                "path": "examples/fib.ja",
                "jana": {
                    "exit_code": 0,
                    "message": "ok",
                    "status": "ok",
                    "stderr_lines": 0,
                    "stdout_lines": 2,
                },
                "reverie": {
                    "exit_code": 0,
                    "message": "ok",
                    "status": "ok",
                    "stderr_lines": 0,
                    "stdout_lines": 1,
                },
                "reverie_legacy_janus": {
                    "exit_code": 0,
                    "message": "ok",
                    "status": "ok",
                    "stderr_lines": 0,
                    "stdout_lines": 1,
                },
            }
        ],
        "jana_bin": "/tmp/jana/bin/janus",
        "jana_dir": "/tmp/jana",
        "reverie_bin": "/tmp/reverie",
        "timeout_seconds": 10.0,
        "totals": {
            "jana": {"ok": 1},
            "reverie": {"ok": 1},
            "reverie_legacy_janus": {"ok": 1},
        },
    }


def expect_self_test_error(label: str, data: dict[str, Any], needle: str) -> None:
    errors = validate_artifact(data, 1, [], [], [])
    if not any(needle in error for error in errors):
        raise AssertionError(f"{label} did not report `{needle}`; errors were {errors}")


def run_self_tests() -> int:
    valid = valid_synthetic_artifact()
    errors = validate_artifact(valid, 1, [], [], [])
    if errors:
        for error in errors:
            print(f"error: self-test valid fixture failed: {error}", file=sys.stderr)
        return 1

    bool_timeout = valid_synthetic_artifact()
    bool_timeout["timeout_seconds"] = True
    bool_exit_code = valid_synthetic_artifact()
    bool_exit_code["examples"][0]["jana"]["exit_code"] = False
    bool_line_count = valid_synthetic_artifact()
    bool_line_count["examples"][0]["jana"]["stdout_lines"] = True
    negative_line_count = valid_synthetic_artifact()
    negative_line_count["examples"][0]["jana"]["stderr_lines"] = -1
    stale_total = valid_synthetic_artifact()
    stale_total["totals"]["jana"]["ok"] = 2
    bool_total = valid_synthetic_artifact()
    bool_total["totals"]["jana"]["ok"] = True
    unknown_status = valid_synthetic_artifact()
    unknown_status["examples"][0]["jana"]["status"] = "maybe"
    unknown_total_status = valid_synthetic_artifact()
    unknown_total_status["totals"]["jana"]["maybe"] = 1
    extra_result_key = valid_synthetic_artifact()
    extra_result_key["examples"][0]["jana"]["duration_seconds"] = 0.1

    try:
        expect_self_test_error(
            "bool timeout",
            bool_timeout,
            "timeout_seconds must be a positive number",
        )
        expect_self_test_error(
            "bool exit code",
            bool_exit_code,
            "exit_code must be an integer or null",
        )
        expect_self_test_error(
            "bool line count",
            bool_line_count,
            "stdout_lines must be a non-negative integer",
        )
        expect_self_test_error(
            "negative line count",
            negative_line_count,
            "stderr_lines must be a non-negative integer",
        )
        expect_self_test_error(
            "stale total",
            stale_total,
            "totals.jana does not match example statuses",
        )
        expect_self_test_error(
            "bool total",
            bool_total,
            "totals.jana.ok must be a non-negative integer",
        )
        expect_self_test_error(
            "unknown status",
            unknown_status,
            "must be one of",
        )
        expect_self_test_error(
            "unknown total status",
            unknown_total_status,
            "totals.jana has unexpected field(s)",
        )
        expect_self_test_error(
            "extra result key",
            extra_result_key,
            "has unexpected field(s)",
        )
    except AssertionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print("ok: Jana audit validator self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    if args.min_examples < 1:
        print("error: --min-examples must be positive", file=sys.stderr)
        return 1
    if args.expect_timeout is not None and args.expect_timeout <= 0:
        print("error: --expect-timeout must be positive", file=sys.stderr)
        return 1

    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
    except OSError as error:
        print(f"error: failed to read audit artifact: {error}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as error:
        print(f"error: failed to parse audit artifact: {error}", file=sys.stderr)
        return 1
    if not isinstance(data, dict):
        print("error: artifact root must be a JSON object", file=sys.stderr)
        return 1
    exact_paths = None
    if args.exact_paths_file is not None:
        try:
            exact_paths = read_expected_paths(args.exact_paths_file)
        except OSError as error:
            print(f"error: failed to read exact paths file: {error}", file=sys.stderr)
            return 1
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    errors = validate_artifact(
        data,
        args.min_examples,
        args.min_status,
        args.expect_total,
        args.expect_status,
        exact_paths,
        args.expect_timeout,
        args.expect_jana_bin_suffix,
        args.expect_reverie_bin_suffix,
    )
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"ok: validated {len(data.get('examples', []))} audited Jana example(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
