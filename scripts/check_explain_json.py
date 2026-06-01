#!/usr/bin/env python3
"""Validate the machine-readable `reverie explain --json` schema."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAMPLES = [
    REPO_ROOT / "examples" / "fib.rev",
    REPO_ROOT / "examples" / "globals.rev",
    REPO_ROOT / "examples" / "array.rev",
]
REQUIRED_KEYS = {
    "file",
    "status",
    "globals",
    "procedures",
    "statements",
    "expressions",
    "features",
    "safety_checks",
    "safety_check_counts",
    "external_store",
    "declared_store",
    "run_template",
    "declared_override_template",
    "inverse_template",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run `reverie explain --json` and validate its stable schema."
    )
    parser.add_argument(
        "examples",
        nargs="*",
        type=Path,
        default=DEFAULT_EXAMPLES,
        help="Example source files to validate. Defaults to fib.rev, globals.rev, and array.rev.",
    )
    parser.add_argument(
        "--reverie-bin",
        type=Path,
        default=os.environ.get("REVERIE_BIN"),
        help="Use an existing reverie binary instead of `cargo run`.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent schema-validator self-tests and exit.",
    )
    return parser.parse_args()


def explain_command(path: Path, reverie_bin: Optional[Path]) -> list[str]:
    if reverie_bin is not None:
        return [str(reverie_bin), "explain", "--json", str(path)]
    return [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "reverie-cli",
        "--",
        "explain",
        "--json",
        str(path),
    ]


def explain_json(path: Path, reverie_bin: Optional[Path]) -> dict[str, Any]:
    completed = subprocess.run(
        explain_command(path, reverie_bin),
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise RuntimeError(f"`reverie explain --json {path}` failed")
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"`{path}` did not emit valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise RuntimeError(f"`{path}` JSON root must be an object")
    return data


def matches_type(value: Any, expected: type) -> bool:
    return isinstance(value, expected) and not (
        expected is int and isinstance(value, bool)
    )


def require_type(errors: list[str], label: str, value: Any, expected: type) -> None:
    if not matches_type(value, expected):
        errors.append(f"{label} must be {expected.__name__}, found {type(value).__name__}")


def validate_store_entries(errors: list[str], label: str, entries: Any) -> None:
    if not isinstance(entries, list):
        errors.append(f"{label} must be list, found {type(entries).__name__}")
        return
    for index, entry in enumerate(entries):
        entry_label = f"{label}[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{entry_label} must be object, found {type(entry).__name__}")
            continue
        if "name" not in entry:
            errors.append(f"{entry_label} missing name")
        else:
            require_type(errors, f"{entry_label}.name", entry["name"], str)
        if "template" not in entry:
            errors.append(f"{entry_label} missing template")
        else:
            require_type(errors, f"{entry_label}.template", entry["template"], str)
        if label == "external_store" and "type" in entry and entry["type"] is not None:
            require_type(errors, f"{entry_label}.type", entry["type"], str)


def validate_safety_check_counts(
    errors: list[str], checks: Any, counts: Any
) -> None:
    if not isinstance(counts, dict):
        errors.append(f"safety_check_counts must be dict, found {type(counts).__name__}")
        return
    if not isinstance(checks, list):
        return
    check_set = {check for check in checks if isinstance(check, str)}
    count_keys = set(counts)
    missing = check_set - count_keys
    extra = count_keys - check_set
    if missing:
        errors.append(
            "safety_check_counts missing key(s): " + ", ".join(sorted(missing))
        )
    if extra:
        errors.append(
            "safety_check_counts has unexpected key(s): " + ", ".join(sorted(extra))
        )
    for key, value in counts.items():
        if not isinstance(key, str):
            errors.append("safety_check_counts key must be str")
        if not matches_type(value, int):
            errors.append(
                f"safety_check_counts[{key!r}] must be int, found {type(value).__name__}"
            )
        elif value < 1:
            errors.append(f"safety_check_counts[{key!r}] must be positive")


def expected_file_path(path: Path) -> str:
    return str(path if path.is_absolute() else REPO_ROOT / path)


def validate(path: Path, data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    keys = set(data)
    missing = REQUIRED_KEYS - keys
    extra = keys - REQUIRED_KEYS
    if missing:
        errors.append("missing key(s): " + ", ".join(sorted(missing)))
    if extra:
        errors.append("unexpected key(s): " + ", ".join(sorted(extra)))

    for key in ("file", "status", "run_template", "declared_override_template", "inverse_template"):
        if key in data:
            require_type(errors, key, data[key], str)
    for key in ("globals", "procedures", "statements", "expressions"):
        if key in data:
            require_type(errors, key, data[key], int)
    if data.get("status") != "reversible program checks":
        errors.append("status must be `reversible program checks`")
    if "features" in data:
        require_type(errors, "features", data["features"], list)
        if isinstance(data["features"], list):
            for index, feature in enumerate(data["features"]):
                require_type(errors, f"features[{index}]", feature, str)
    if "safety_checks" in data:
        require_type(errors, "safety_checks", data["safety_checks"], list)
        if isinstance(data["safety_checks"], list):
            for index, check in enumerate(data["safety_checks"]):
                require_type(errors, f"safety_checks[{index}]", check, str)
            if path.name == "array.rev":
                expected_checks = {
                    "constant array indexes checked before runtime",
                    "same-root update aliases rejected before runtime",
                }
                missing_checks = expected_checks - set(data["safety_checks"])
                if missing_checks:
                    errors.append(
                        "array.rev missing safety check(s): "
                        + ", ".join(sorted(missing_checks))
                    )
    if "safety_check_counts" in data:
        validate_safety_check_counts(
            errors, data.get("safety_checks"), data["safety_check_counts"]
        )
        if path.name == "array.rev" and isinstance(data["safety_check_counts"], dict):
            index_count = data["safety_check_counts"].get(
                "constant array indexes checked before runtime"
            )
            if not matches_type(index_count, int) or index_count < 3:
                errors.append("array.rev should count at least 3 constant array indexes")
    validate_store_entries(errors, "external_store", data.get("external_store"))
    validate_store_entries(errors, "declared_store", data.get("declared_store"))

    expected_file = expected_file_path(path)
    if data.get("file") != expected_file:
        errors.append(f"file should be `{expected_file}`, found `{data.get('file')}`")
    for key in ("run_template", "declared_override_template", "inverse_template"):
        value = data.get(key)
        if isinstance(value, str) and value != value.rstrip():
            errors.append(f"{key} must not have trailing whitespace")
    return errors


def valid_synthetic_array_summary() -> dict[str, Any]:
    return {
        "file": str(REPO_ROOT / "examples" / "array.rev"),
        "status": "reversible program checks",
        "globals": 0,
        "procedures": 0,
        "statements": 5,
        "expressions": 14,
        "features": ["array literals", "fixed-size arrays"],
        "safety_checks": [
            "constant array indexes checked before runtime",
            "same-root update aliases rejected before runtime",
        ],
        "safety_check_counts": {
            "constant array indexes checked before runtime": 4,
            "same-root update aliases rejected before runtime": 2,
        },
        "external_store": [],
        "declared_store": [],
        "run_template": "reverie run examples/array.rev",
        "declared_override_template": "reverie run examples/array.rev",
        "inverse_template": "reverie invert examples/array.rev",
    }


def expect_self_test_error(label: str, data: dict[str, Any], needle: str) -> None:
    errors = validate(Path("examples/array.rev"), data)
    if not any(needle in error for error in errors):
        raise AssertionError(f"{label} did not report `{needle}`; errors were {errors}")


def run_self_tests() -> int:
    valid = valid_synthetic_array_summary()
    errors = validate(Path("examples/array.rev"), valid)
    if errors:
        for error in errors:
            print(f"error: self-test valid fixture failed: {error}", file=sys.stderr)
        return 1
    absolute = valid_synthetic_array_summary()
    errors = validate(REPO_ROOT / "examples" / "array.rev", absolute)
    if errors:
        for error in errors:
            print(f"error: self-test absolute fixture failed: {error}", file=sys.stderr)
        return 1

    relative_file = valid_synthetic_array_summary()
    relative_file["file"] = "examples/array.rev"
    missing_count = valid_synthetic_array_summary()
    del missing_count["safety_check_counts"][
        "same-root update aliases rejected before runtime"
    ]
    missing_key = valid_synthetic_array_summary()
    del missing_key["safety_check_counts"]
    bool_statement_count = valid_synthetic_array_summary()
    bool_statement_count["statements"] = True
    zero_count = valid_synthetic_array_summary()
    zero_count["safety_check_counts"][
        "constant array indexes checked before runtime"
    ] = 0
    bool_count = valid_synthetic_array_summary()
    bool_count["safety_check_counts"][
        "constant array indexes checked before runtime"
    ] = True
    string_count = valid_synthetic_array_summary()
    string_count["safety_check_counts"][
        "constant array indexes checked before runtime"
    ] = "4"
    extra_count = valid_synthetic_array_summary()
    extra_count["safety_check_counts"][
        "dynamic array indexes checked at runtime"
    ] = 1
    weak_array_count = valid_synthetic_array_summary()
    weak_array_count["safety_check_counts"][
        "constant array indexes checked before runtime"
    ] = 2

    try:
        expect_self_test_error(
            "relative file",
            relative_file,
            "file should be",
        )
        expect_self_test_error(
            "missing count",
            missing_count,
            "safety_check_counts missing key(s)",
        )
        expect_self_test_error(
            "missing safety_check_counts",
            missing_key,
            "missing key(s): safety_check_counts",
        )
        expect_self_test_error(
            "bool statement count",
            bool_statement_count,
            "statements must be int",
        )
        expect_self_test_error(
            "zero count",
            zero_count,
            "must be positive",
        )
        expect_self_test_error(
            "bool count",
            bool_count,
            "must be int",
        )
        expect_self_test_error(
            "string count",
            string_count,
            "must be int",
        )
        expect_self_test_error(
            "extra count",
            extra_count,
            "safety_check_counts has unexpected key(s)",
        )
        expect_self_test_error(
            "weak array count",
            weak_array_count,
            "array.rev should count at least 3 constant array indexes",
        )
    except AssertionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print("ok: explain JSON schema validator self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    reverie_bin = args.reverie_bin
    if reverie_bin is not None and not reverie_bin.exists():
        print(f"error: reverie binary `{reverie_bin}` does not exist", file=sys.stderr)
        return 1
    total = 0
    for path in args.examples:
        resolved = path if path.is_absolute() else REPO_ROOT / path
        try:
            data = explain_json(resolved, reverie_bin)
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        errors = validate(path, data)
        if errors:
            for error in errors:
                print(f"error: {path}: {error}", file=sys.stderr)
            return 1
        total += 1
    print(f"ok: validated {total} explain JSON schema(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
