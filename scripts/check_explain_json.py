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
    "dataset_loops",
    "witness_store",
    "witness_metrics",
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
    parser.add_argument(
        "--ml",
        action="store_true",
        help="Validate the optional `reverie explain --ml --json` profile.",
    )
    return parser.parse_args()


def explain_command(path: Path, reverie_bin: Optional[Path], ml: bool) -> list[str]:
    if reverie_bin is not None:
        command = [str(reverie_bin), "explain", "--json"]
    else:
        command = [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "reverie-cli",
            "--",
            "explain",
            "--json",
        ]
    if ml:
        command.append("--ml")
    command.append(str(path))
    return command


def explain_json(path: Path, reverie_bin: Optional[Path], ml: bool) -> dict[str, Any]:
    completed = subprocess.run(
        explain_command(path, reverie_bin, ml),
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        profile_flag = " --ml" if ml else ""
        raise RuntimeError(f"`reverie explain{profile_flag} --json {path}` failed")
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


def validate_dataset_loops(errors: list[str], loops: Any) -> None:
    if not isinstance(loops, list):
        errors.append(f"dataset_loops must be list, found {type(loops).__name__}")
        return
    for index, entry in enumerate(loops):
        entry_label = f"dataset_loops[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{entry_label} must be object, found {type(entry).__name__}")
            continue
        if "index" not in entry:
            errors.append(f"{entry_label} missing index")
        else:
            require_type(errors, f"{entry_label}.index", entry["index"], str)
        sources = entry.get("size_sources")
        if not isinstance(sources, list):
            errors.append(
                f"{entry_label}.size_sources must be list, found {type(sources).__name__}"
            )
            continue
        if not sources:
            errors.append(f"{entry_label}.size_sources must not be empty")
        for source_index, source in enumerate(sources):
            require_type(
                errors,
                f"{entry_label}.size_sources[{source_index}]",
                source,
                str,
            )


def validate_witness_store(errors: list[str], names: Any) -> None:
    if not isinstance(names, list):
        errors.append(f"witness_store must be list, found {type(names).__name__}")
        return
    for index, name in enumerate(names):
        require_type(errors, f"witness_store[{index}]", name, str)


def validate_witness_metrics(errors: list[str], metrics: Any) -> None:
    if not isinstance(metrics, dict):
        errors.append(f"witness_metrics must be dict, found {type(metrics).__name__}")
        return
    for key in ("variables", "known_cells", "known_payload_bytes"):
        if key in metrics:
            require_type(errors, f"witness_metrics.{key}", metrics[key], int)
        else:
            errors.append(f"witness_metrics missing {key}")
    unknown = metrics.get("unknown_variables")
    if not isinstance(unknown, list):
        errors.append(
            f"witness_metrics.unknown_variables must be list, found {type(unknown).__name__}"
        )
    else:
        for index, name in enumerate(unknown):
            require_type(errors, f"witness_metrics.unknown_variables[{index}]", name, str)
    entries = metrics.get("entries")
    if not isinstance(entries, list):
        errors.append(
            f"witness_metrics.entries must be list, found {type(entries).__name__}"
        )
        return
    for index, entry in enumerate(entries):
        entry_label = f"witness_metrics.entries[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{entry_label} must be object, found {type(entry).__name__}")
            continue
        if "name" not in entry:
            errors.append(f"{entry_label} missing name")
        else:
            require_type(errors, f"{entry_label}.name", entry["name"], str)
        for key in ("cells", "payload_bytes"):
            if key not in entry:
                errors.append(f"{entry_label} missing {key}")
            elif entry[key] is not None:
                require_type(errors, f"{entry_label}.{key}", entry[key], int)


def validate_string_list(errors: list[str], label: str, value: Any) -> None:
    if not isinstance(value, list):
        errors.append(f"{label} must be list, found {type(value).__name__}")
        return
    for index, item in enumerate(value):
        require_type(errors, f"{label}[{index}]", item, str)


def validate_count_object(errors: list[str], label: str, value: Any) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be dict, found {type(value).__name__}")
        return
    for key, count in value.items():
        if not isinstance(key, str):
            errors.append(f"{label} key must be str")
        elif not key:
            errors.append(f"{label} key must not be empty")
        if not matches_type(count, int):
            errors.append(f"{label}.{key} must be int, found {type(count).__name__}")
        elif count < 0:
            errors.append(f"{label}.{key} must be non-negative")


def validate_ml_replay_cost(errors: list[str], value: Any) -> None:
    label = "ml_profile.replay_cost"
    if not isinstance(value, dict):
        errors.append(f"{label} must be dict, found {type(value).__name__}")
        return
    integer_keys = (
        "forward_statement_count",
        "inverse_statement_count",
        "roundtrip_statement_count",
        "forward_expression_count",
        "inverse_expression_count",
        "roundtrip_expression_count",
        "known_state_tensor_payload_bytes",
        "known_witness_payload_bytes",
        "known_replay_payload_bytes",
    )
    for key in integer_keys:
        if key not in value:
            errors.append(f"{label} missing {key}")
        else:
            require_type(errors, f"{label}.{key}", value[key], int)
            if matches_type(value[key], int) and value[key] < 0:
                errors.append(f"{label}.{key} must be non-negative")
    validate_string_list(
        errors,
        f"{label}.unknown_state_tensor_variables",
        value.get("unknown_state_tensor_variables"),
    )
    validate_string_list(
        errors,
        f"{label}.unknown_witness_variables",
        value.get("unknown_witness_variables"),
    )
    ratio = value.get("witness_to_state_payload_ratio")
    if ratio is not None and (
        isinstance(ratio, bool) or not isinstance(ratio, (int, float))
    ):
        errors.append(f"{label}.witness_to_state_payload_ratio must be number or null")
    if (
        matches_type(value.get("forward_statement_count"), int)
        and matches_type(value.get("inverse_statement_count"), int)
        and matches_type(value.get("roundtrip_statement_count"), int)
        and value["roundtrip_statement_count"]
        != value["forward_statement_count"] + value["inverse_statement_count"]
    ):
        errors.append(f"{label}.roundtrip_statement_count must equal forward + inverse")
    if (
        matches_type(value.get("forward_expression_count"), int)
        and matches_type(value.get("inverse_expression_count"), int)
        and matches_type(value.get("roundtrip_expression_count"), int)
        and value["roundtrip_expression_count"]
        != value["forward_expression_count"] + value["inverse_expression_count"]
    ):
        errors.append(f"{label}.roundtrip_expression_count must equal forward + inverse")
    if (
        matches_type(value.get("known_state_tensor_payload_bytes"), int)
        and matches_type(value.get("known_witness_payload_bytes"), int)
        and matches_type(value.get("known_replay_payload_bytes"), int)
        and value["known_replay_payload_bytes"]
        != value["known_state_tensor_payload_bytes"]
        + value["known_witness_payload_bytes"]
    ):
        errors.append(
            f"{label}.known_replay_payload_bytes must equal state tensor + witness bytes"
        )


def validate_ml_profile(errors: list[str], profile: Any) -> None:
    if not isinstance(profile, dict):
        errors.append(f"ml_profile must be dict, found {type(profile).__name__}")
        return
    required = {
        "schema",
        "goal_fit",
        "capabilities",
        "tensor_variables",
        "tensor_entries",
        "tensor_metrics",
        "replay_cost",
        "update_counts",
        "builtin_counts",
        "tensor_builtin_calls",
        "q31_builtin_calls",
        "fixed_point_binary_ops",
        "non_injective_signal_calls",
        "dataset_loop_count",
        "roundtrip_template",
    }
    missing = required - set(profile)
    if missing:
        errors.append("ml_profile missing key(s): " + ", ".join(sorted(missing)))
    if profile.get("schema") != "reverie_explain_ml_profile_v1":
        errors.append("ml_profile.schema must be reverie_explain_ml_profile_v1")
    if profile.get("goal_fit") not in {
        "general_reversible_program",
        "tensor_reversible_kernel",
        "auditable_ml_kernel",
    }:
        errors.append("ml_profile.goal_fit has unsupported value")
    validate_string_list(errors, "ml_profile.capabilities", profile.get("capabilities"))
    validate_string_list(
        errors, "ml_profile.tensor_variables", profile.get("tensor_variables")
    )
    entries = profile.get("tensor_entries")
    if not isinstance(entries, list):
        errors.append(
            f"ml_profile.tensor_entries must be list, found {type(entries).__name__}"
        )
    else:
        for index, entry in enumerate(entries):
            entry_label = f"ml_profile.tensor_entries[{index}]"
            if not isinstance(entry, dict):
                errors.append(f"{entry_label} must be dict, found {type(entry).__name__}")
                continue
            for key in ("name", "role"):
                if key not in entry:
                    errors.append(f"{entry_label} missing {key}")
                else:
                    require_type(errors, f"{entry_label}.{key}", entry[key], str)
            if entry.get("role") not in {"state", "witness"}:
                errors.append(f"{entry_label}.role must be state or witness")
            for key in ("cells", "payload_bytes"):
                if key not in entry:
                    errors.append(f"{entry_label} missing {key}")
                elif entry[key] is not None:
                    require_type(errors, f"{entry_label}.{key}", entry[key], int)
    metrics = profile.get("tensor_metrics")
    if not isinstance(metrics, dict):
        errors.append(
            f"ml_profile.tensor_metrics must be dict, found {type(metrics).__name__}"
        )
    else:
        for key in (
            "variables",
            "state_variables",
            "witness_variables",
            "known_cells",
            "known_payload_bytes",
            "known_state_cells",
            "known_state_payload_bytes",
            "known_witness_cells",
            "known_witness_payload_bytes",
        ):
            if key not in metrics:
                errors.append(f"ml_profile.tensor_metrics missing {key}")
            else:
                require_type(errors, f"ml_profile.tensor_metrics.{key}", metrics[key], int)
        for key in (
            "unknown_variables",
            "unknown_state_variables",
            "unknown_witness_variables",
        ):
            validate_string_list(
                errors, f"ml_profile.tensor_metrics.{key}", metrics.get(key)
            )
    validate_ml_replay_cost(errors, profile.get("replay_cost"))
    update_counts = profile.get("update_counts")
    if not isinstance(update_counts, dict):
        errors.append(
            f"ml_profile.update_counts must be dict, found {type(update_counts).__name__}"
        )
    else:
        for key in ("tensor_update_statements", "witness_update_statements"):
            if key not in update_counts:
                errors.append(f"ml_profile.update_counts missing {key}")
            else:
                require_type(errors, f"ml_profile.update_counts.{key}", update_counts[key], int)
        validate_count_object(
            errors,
            "ml_profile.update_counts.all_update_targets",
            update_counts.get("all_update_targets"),
        )
    validate_count_object(errors, "ml_profile.builtin_counts", profile.get("builtin_counts"))
    validate_count_object(
        errors,
        "ml_profile.non_injective_signal_calls",
        profile.get("non_injective_signal_calls"),
    )
    for key in (
        "tensor_builtin_calls",
        "q31_builtin_calls",
        "fixed_point_binary_ops",
        "dataset_loop_count",
    ):
        if key in profile:
            require_type(errors, f"ml_profile.{key}", profile[key], int)
        else:
            errors.append(f"ml_profile missing {key}")
    if "roundtrip_template" in profile:
        require_type(errors, "ml_profile.roundtrip_template", profile["roundtrip_template"], str)


def expected_file_path(path: Path) -> str:
    return str(path if path.is_absolute() else REPO_ROOT / path)


def validate(path: Path, data: dict[str, Any], allow_ml_profile: bool = False) -> list[str]:
    errors: list[str] = []
    required_keys = set(REQUIRED_KEYS)
    if allow_ml_profile:
        required_keys.add("ml_profile")
    keys = set(data)
    missing = required_keys - keys
    extra = keys - required_keys
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
    if "dataset_loops" in data:
        validate_dataset_loops(errors, data["dataset_loops"])
    if "witness_store" in data:
        validate_witness_store(errors, data["witness_store"])
    if "witness_metrics" in data:
        validate_witness_metrics(errors, data["witness_metrics"])
    if "ml_profile" in data:
        validate_ml_profile(errors, data["ml_profile"])
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
        "dataset_loops": [],
        "witness_store": [],
        "witness_metrics": {
            "variables": 0,
            "known_cells": 0,
            "known_payload_bytes": 0,
            "unknown_variables": [],
            "entries": [],
        },
        "external_store": [],
        "declared_store": [],
        "run_template": "reverie run examples/array.rev",
        "declared_override_template": "reverie run examples/array.rev",
        "inverse_template": "reverie invert examples/array.rev",
    }


def valid_synthetic_ml_summary() -> dict[str, Any]:
    data = valid_synthetic_array_summary()
    data["features"] = data["features"] + [
        "array-backed tensors",
        "tensor builtins",
        "witness tapes",
        "Janus fixed-point multiply",
    ]
    data["dataset_loops"] = [{"index": "sample", "size_sources": ["labels"]}]
    data["witness_store"] = ["logits_tape"]
    data["witness_metrics"] = {
        "variables": 1,
        "known_cells": 20,
        "known_payload_bytes": 160,
        "unknown_variables": [],
        "entries": [
            {"name": "logits_tape", "cells": 20, "payload_bytes": 160},
        ],
    }
    data["ml_profile"] = {
        "schema": "reverie_explain_ml_profile_v1",
        "goal_fit": "auditable_ml_kernel",
        "capabilities": [
            "dataset_shaped_iteration",
            "deterministic_q31_fixed_point",
            "explicit_witness_tapes",
            "known_witness_budget",
            "ml_linear_algebra_builtins",
            "non_injective_signals_captured_as_state",
            "reversible_tensor_accumulation",
            "roundtrip_proof_ready",
            "static_replay_cost_model",
            "static_reversibility_check",
            "static_tensor_shapes",
        ],
        "tensor_variables": ["images", "logits_tape", "weights"],
        "tensor_entries": [
            {"name": "images", "role": "state", "cells": 1568, "payload_bytes": 12544},
            {
                "name": "logits_tape",
                "role": "witness",
                "cells": 20,
                "payload_bytes": 160,
            },
            {"name": "weights", "role": "state", "cells": 7840, "payload_bytes": 62720},
        ],
        "tensor_metrics": {
            "variables": 3,
            "state_variables": 2,
            "witness_variables": 1,
            "known_cells": 9428,
            "known_payload_bytes": 75424,
            "known_state_cells": 9408,
            "known_state_payload_bytes": 75264,
            "known_witness_cells": 20,
            "known_witness_payload_bytes": 160,
            "unknown_variables": [],
            "unknown_state_variables": [],
            "unknown_witness_variables": [],
        },
        "replay_cost": {
            "forward_statement_count": 11,
            "inverse_statement_count": 11,
            "roundtrip_statement_count": 22,
            "forward_expression_count": 33,
            "inverse_expression_count": 33,
            "roundtrip_expression_count": 66,
            "known_state_tensor_payload_bytes": 75264,
            "known_witness_payload_bytes": 160,
            "known_replay_payload_bytes": 75424,
            "unknown_state_tensor_variables": [],
            "unknown_witness_variables": [],
            "witness_to_state_payload_ratio": 160 / 75264,
        },
        "update_counts": {
            "tensor_update_statements": 3,
            "witness_update_statements": 1,
            "all_update_targets": {"logits_tape": 1, "weights": 1, "bias": 1},
        },
        "builtin_counts": {"argmax": 1, "vecmat_q31": 1},
        "tensor_builtin_calls": 2,
        "q31_builtin_calls": 1,
        "fixed_point_binary_ops": 0,
        "non_injective_signal_calls": {"argmax": 1},
        "dataset_loop_count": 1,
        "roundtrip_template": "reverie roundtrip <FILE> --json",
    }
    return data


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
    ml_valid = valid_synthetic_ml_summary()
    errors = validate(Path("examples/array.rev"), ml_valid, allow_ml_profile=True)
    if errors:
        for error in errors:
            print(f"error: self-test ML fixture failed: {error}", file=sys.stderr)
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
    missing_dataset_loops = valid_synthetic_array_summary()
    del missing_dataset_loops["dataset_loops"]
    bad_dataset_loop = valid_synthetic_array_summary()
    bad_dataset_loop["dataset_loops"] = [{"index": "i", "size_sources": []}]
    missing_witness_metrics = valid_synthetic_array_summary()
    del missing_witness_metrics["witness_metrics"]
    bad_witness_metric = valid_synthetic_array_summary()
    bad_witness_metric["witness_metrics"]["variables"] = True
    extra_count = valid_synthetic_array_summary()
    extra_count["safety_check_counts"][
        "dynamic array indexes checked at runtime"
    ] = 1
    weak_array_count = valid_synthetic_array_summary()
    weak_array_count["safety_check_counts"][
        "constant array indexes checked before runtime"
    ] = 2
    unexpected_ml_profile = valid_synthetic_ml_summary()
    bad_ml_profile = valid_synthetic_ml_summary()
    bad_ml_profile["ml_profile"]["tensor_metrics"]["known_cells"] = True

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
            "missing dataset loops",
            missing_dataset_loops,
            "missing key(s): dataset_loops",
        )
        expect_self_test_error(
            "bad dataset loop",
            bad_dataset_loop,
            "dataset_loops[0].size_sources must not be empty",
        )
        expect_self_test_error(
            "missing witness metrics",
            missing_witness_metrics,
            "missing key(s): witness_metrics",
        )
        expect_self_test_error(
            "bad witness metric",
            bad_witness_metric,
            "witness_metrics.variables must be int",
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
        expect_self_test_error(
            "unexpected ml profile",
            unexpected_ml_profile,
            "unexpected key(s): ml_profile",
        )
        errors = validate(
            Path("examples/array.rev"),
            bad_ml_profile,
            allow_ml_profile=True,
        )
        if not any(
            "ml_profile.tensor_metrics.known_cells must be int" in error
            for error in errors
        ):
            raise AssertionError(
                f"bad ml profile did not report tensor metric type; errors were {errors}"
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
            data = explain_json(resolved, reverie_bin, args.ml)
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        errors = validate(path, data, allow_ml_profile=args.ml)
        if errors:
            for error in errors:
                print(f"error: {path}: {error}", file=sys.stderr)
            return 1
        total += 1
    print(f"ok: validated {total} explain JSON schema(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
