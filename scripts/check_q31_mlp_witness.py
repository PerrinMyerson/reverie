#!/usr/bin/env python3
"""Reference-check Reverie's MNIST-shaped Q31 MLP witness trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional


SAMPLES = 2
IMAGE_PIXELS = 784
HIDDEN = 16
DIGITS = 10
Q31_ONE = 1 << 31
DEFAULT_LR = 2_147_483
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
U64_MOD = 1 << 64
I64_BYTES = 8
REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_RELATIVE_PATH = "examples/mnist_mlp_witness.rev"
PROGRAM_PATH = REPO_ROOT / PROGRAM_RELATIVE_PATH
CHECKED_STORE_FIELDS = (
    "images",
    "labels",
    "lr",
    "w1",
    "b1",
    "hidden_pre_tape",
    "hidden_mask_tape",
    "hidden_tape",
    "w2",
    "b2",
    "logits_tape",
    "out_error_tape",
    "hidden_back_tape",
    "hidden_delta_tape",
    "prediction_tape",
    "correct_tape",
)
WITNESS_TAPE_FIELDS = (
    "hidden_pre_tape",
    "hidden_mask_tape",
    "hidden_tape",
    "logits_tape",
    "out_error_tape",
    "hidden_back_tape",
    "hidden_delta_tape",
    "prediction_tape",
    "correct_tape",
)
WITNESS_METADATA_TYPES = {
    "hidden_pre_tape": "witness<tensor<int, 2, 16>>",
    "hidden_mask_tape": "witness<tensor<int, 2, 16>>",
    "hidden_tape": "witness<tensor<int, 2, 16>>",
    "logits_tape": "witness<tensor<int, 2, 10>>",
    "out_error_tape": "witness<tensor<int, 2, 10>>",
    "hidden_back_tape": "witness<tensor<int, 2, 16>>",
    "hidden_delta_tape": "witness<tensor<int, 2, 16>>",
    "prediction_tape": "witness<tensor<int, 2>>",
    "correct_tape": "witness<tensor<int, 2>>",
}
EXPECTED_DATASET_LOOPS = [{"index": "sample", "size_sources": ["labels"]}]


class MlpCheckError(ValueError):
    """Raised when an MLP seed, output, or expectation is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reference-check examples/mnist_mlp_witness.rev against exact Q31 "
            "MLP witness semantics."
        )
    )
    parser.add_argument(
        "--vars-json",
        type=Path,
        help="Seed JSON passed to `reverie run examples/mnist_mlp_witness.rev --vars-json`.",
    )
    parser.add_argument(
        "--run-output-json",
        type=Path,
        help="JSON output from `reverie run ... --json` to compare against.",
    )
    parser.add_argument(
        "--expect-predictions",
        help="JSON array of two expected predicted digits.",
    )
    parser.add_argument(
        "--expect-correct",
        help="JSON array of two expected booleans.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable report.",
    )
    parser.add_argument(
        "--expect-report-json",
        type=Path,
        help="Require the recomputed machine-readable report to match this saved JSON report.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Write a human-readable MLP witness proof card.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent Q31 MLP checker self-tests and exit.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise MlpCheckError(f"failed to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise MlpCheckError(f"failed to parse {path}: {error}") from error


def wrap_i64(value: int) -> int:
    value %= U64_MOD
    if value >= (1 << 63):
        value -= U64_MOD
    return value


def checked_i64(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MlpCheckError(f"{label} must be a signed integer")
    if not I64_MIN <= value <= I64_MAX:
        raise MlpCheckError(f"{label} is outside signed i64 range")
    return value


def fixed_mul_q31(left: int, right: int) -> int:
    return wrap_i64((left * right) >> 31)


def add_i64(left: int, right: int) -> int:
    return wrap_i64(left + right)


def sub_i64(left: int, right: int) -> int:
    return wrap_i64(left - right)


def argmax_first(values: list[int]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def expect_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MlpCheckError(f"{context} must be a JSON object")
    return value


def vector_from_json(
    value: Any,
    length: int,
    label: str,
    default: Optional[list[int]] = None,
) -> list[int]:
    if value is None:
        if default is None:
            raise MlpCheckError(f"{label} is required")
        value = default
    if not isinstance(value, list) or len(value) != length:
        raise MlpCheckError(f"{label} must have {length} entries")
    return [checked_i64(item, f"{label}[{index}]") for index, item in enumerate(value)]


def matrix_from_json(
    value: Any,
    rows: int,
    cols: int,
    label: str,
    default_zero: bool = False,
) -> list[list[int]]:
    if value is None:
        if not default_zero:
            raise MlpCheckError(f"{label} is required")
        value = [[0 for _ in range(cols)] for _ in range(rows)]
    if not isinstance(value, list) or len(value) != rows:
        raise MlpCheckError(f"{label} must have {rows} rows")
    matrix: list[list[int]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != cols:
            raise MlpCheckError(f"{label}[{row_index}] must have {cols} entries")
        matrix.append(
            [
                checked_i64(item, f"{label}[{row_index}][{col_index}]")
                for col_index, item in enumerate(row)
            ]
        )
    return matrix


def parse_seed_store(seed: Any) -> dict[str, Any]:
    store = expect_object(seed, "vars JSON")
    return {
        "images": matrix_from_json(
            store.get("images"),
            SAMPLES,
            IMAGE_PIXELS,
            "images",
            default_zero=True,
        ),
        "labels": vector_from_json(store.get("labels"), SAMPLES, "labels", [0, 1]),
        "lr": checked_i64(store.get("lr", DEFAULT_LR), "lr"),
        "w1": matrix_from_json(store.get("w1"), IMAGE_PIXELS, HIDDEN, "w1"),
        "b1": vector_from_json(store.get("b1"), HIDDEN, "b1"),
        "w2": matrix_from_json(store.get("w2"), HIDDEN, DIGITS, "w2"),
        "b2": vector_from_json(store.get("b2"), DIGITS, "b2"),
    }


def vecmat_q31(vector: list[int], matrix: list[list[int]], cols: int) -> list[int]:
    out = []
    for col in range(cols):
        acc = 0
        for row, value in enumerate(vector):
            acc = add_i64(acc, fixed_mul_q31(value, matrix[row][col]))
        out.append(acc)
    return out


def matvec_q31(matrix: list[list[int]], vector: list[int]) -> list[int]:
    out = []
    for row in matrix:
        acc = 0
        for left, right in zip(row, vector):
            acc = add_i64(acc, fixed_mul_q31(left, right))
        out.append(acc)
    return out


def one_hot_q31(label: int, classes: int) -> list[int]:
    if not 0 <= label < classes:
        raise MlpCheckError(f"label {label} is outside 0..{classes - 1}")
    return [Q31_ONE if index == label else 0 for index in range(classes)]


def hadamard_q31(left: list[int], right: list[int]) -> list[int]:
    return [fixed_mul_q31(a, b) for a, b in zip(left, right)]


def scale_q31_vector(values: list[int], scalar: int) -> list[int]:
    return [fixed_mul_q31(value, scalar) for value in values]


def scale_q31_outer(left: list[int], right: list[int], scalar: int) -> list[list[int]]:
    rows = []
    for left_value in left:
        rows.append(
            [
                fixed_mul_q31(fixed_mul_q31(left_value, right_value), scalar)
                for right_value in right
            ]
        )
    return rows


def add_vector(left: list[int], right: list[int]) -> list[int]:
    return [add_i64(a, b) for a, b in zip(left, right)]


def sub_vector(left: list[int], right: list[int]) -> list[int]:
    return [sub_i64(a, b) for a, b in zip(left, right)]


def sub_matrix(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    return [sub_vector(left_row, right_row) for left_row, right_row in zip(left, right)]


def simulate(seed: dict[str, Any]) -> dict[str, Any]:
    images = deepcopy(seed["images"])
    labels = deepcopy(seed["labels"])
    lr = seed["lr"]
    w1 = deepcopy(seed["w1"])
    b1 = deepcopy(seed["b1"])
    w2 = deepcopy(seed["w2"])
    b2 = deepcopy(seed["b2"])

    hidden_pre_tape = []
    hidden_mask_tape = []
    hidden_tape = []
    logits_tape = []
    out_error_tape = []
    hidden_back_tape = []
    hidden_delta_tape = []
    prediction_tape = []
    correct_tape = []

    for sample in range(SAMPLES):
        image = images[sample]
        label = labels[sample]
        if not 0 <= label < DIGITS:
            raise MlpCheckError(f"labels[{sample}] must be in 0..9")

        hidden_pre = add_vector(vecmat_q31(image, w1, HIDDEN), b1)
        hidden_mask = [Q31_ONE if value > 0 else 0 for value in hidden_pre]
        hidden = [max(value, 0) for value in hidden_pre]
        logits = add_vector(vecmat_q31(hidden, w2, DIGITS), b2)
        prediction = argmax_first(logits)
        correct = 1 if prediction == label else 0
        out_error = sub_vector(logits, one_hot_q31(label, DIGITS))
        hidden_back = matvec_q31(w2, out_error)
        hidden_delta = hadamard_q31(hidden_back, hidden_mask)

        hidden_pre_tape.append(hidden_pre)
        hidden_mask_tape.append(hidden_mask)
        hidden_tape.append(hidden)
        logits_tape.append(logits)
        out_error_tape.append(out_error)
        hidden_back_tape.append(hidden_back)
        hidden_delta_tape.append(hidden_delta)
        prediction_tape.append(prediction)
        correct_tape.append(correct)

        w2 = sub_matrix(w2, scale_q31_outer(hidden, out_error, lr))
        b2 = sub_vector(b2, scale_q31_vector(out_error, lr))
        w1 = sub_matrix(w1, scale_q31_outer(image, hidden_delta, lr))
        b1 = sub_vector(b1, scale_q31_vector(hidden_delta, lr))

    return {
        "images": images,
        "labels": labels,
        "lr": lr,
        "w1": w1,
        "b1": b1,
        "hidden_pre_tape": hidden_pre_tape,
        "hidden_mask_tape": hidden_mask_tape,
        "hidden_tape": hidden_tape,
        "w2": w2,
        "b2": b2,
        "logits_tape": logits_tape,
        "out_error_tape": out_error_tape,
        "hidden_back_tape": hidden_back_tape,
        "hidden_delta_tape": hidden_delta_tape,
        "prediction_tape": prediction_tape,
        "correct_tape": correct_tape,
    }


def parse_run_output(value: Any) -> dict[str, Any]:
    output = expect_object(value, "run output JSON")
    kind = output.get("kind")
    if kind != "reverie_run_result":
        raise MlpCheckError("run output JSON kind must be reverie_run_result")
    validate_run_output_provenance(output)
    dataset_loops = validate_run_output_dataset_loops(output)
    validate_run_output_witness_metadata(output)
    store = expect_object(output.get("store"), "run output JSON store")
    witness_proof = validate_run_output_witness_proof(output, store)
    return {
        "store": store,
        "dataset_loops": dataset_loops,
        "witness_proof": witness_proof,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def comma(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.6g}"
    return str(value)


def short_hash(value: Any) -> str:
    if isinstance(value, str) and len(value) >= 12:
        return value[:12]
    return ""


def json_cell(value: Any) -> str:
    return "`{}`".format(json.dumps(value, separators=(",", ":")))


def validate_sha256_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise MlpCheckError(f"{context} must be a lowercase SHA-256 hex digest")
    return value


def validate_run_output_provenance(output: dict[str, Any]) -> None:
    program = expect_object(output.get("program"), "run output JSON program")
    file = program.get("file")
    if not isinstance(file, str) or not file:
        raise MlpCheckError("run output JSON program.file must be a non-empty string")
    if program.get("direction") != "run":
        raise MlpCheckError("run output JSON program.direction must be run")
    engine = program.get("engine")
    if engine not in ("slot", "tree"):
        raise MlpCheckError("run output JSON program.engine must be slot or tree")
    if program.get("legacy_janus") is not False:
        raise MlpCheckError("run output JSON program.legacy_janus must be false")
    actual = validate_sha256_text(
        program.get("source_sha256"),
        "run output JSON program.source_sha256",
    )
    expected = sha256_file(PROGRAM_PATH)
    if actual != expected:
        raise MlpCheckError(
            "run output JSON program.source_sha256 does not match "
            f"{PROGRAM_RELATIVE_PATH}"
        )


def validate_run_output_dataset_loops(output: dict[str, Any]) -> list[dict[str, Any]]:
    loops = output.get("dataset_loops")
    if loops != EXPECTED_DATASET_LOOPS:
        raise MlpCheckError(
            "run output JSON dataset_loops must prove sample is bounded by labels"
        )
    return deepcopy(EXPECTED_DATASET_LOOPS)


def witness_metadata_cells() -> dict[str, int]:
    return {
        "hidden_pre_tape": SAMPLES * HIDDEN,
        "hidden_mask_tape": SAMPLES * HIDDEN,
        "hidden_tape": SAMPLES * HIDDEN,
        "logits_tape": SAMPLES * DIGITS,
        "out_error_tape": SAMPLES * DIGITS,
        "hidden_back_tape": SAMPLES * HIDDEN,
        "hidden_delta_tape": SAMPLES * HIDDEN,
        "prediction_tape": SAMPLES,
        "correct_tape": SAMPLES,
    }


def validate_run_output_witness_metadata(output: dict[str, Any]) -> None:
    expected_cells = witness_metadata_cells()
    expected_payload_bytes = {
        name: cells * I64_BYTES for name, cells in expected_cells.items()
    }
    expected_fields = set(WITNESS_TAPE_FIELDS)

    witness_store = output.get("witness_store")
    if not isinstance(witness_store, list):
        raise MlpCheckError("run output JSON witness_store must be an array")
    witness_names = set()
    for index, item in enumerate(witness_store):
        if not isinstance(item, str):
            raise MlpCheckError(f"run output JSON witness_store[{index}] must be a string")
        witness_names.add(item)
    if witness_names != expected_fields:
        missing = sorted(expected_fields - witness_names)
        extra = sorted(witness_names - expected_fields)
        raise MlpCheckError(
            "run output JSON witness_store mismatch: missing={} extra={}".format(
                missing,
                extra,
            )
        )

    metrics = expect_object(output.get("witness_metrics"), "run output JSON witness_metrics")
    if checked_i64(metrics.get("variables"), "witness_metrics.variables") != len(WITNESS_TAPE_FIELDS):
        raise MlpCheckError("witness_metrics.variables does not match MLP witness tapes")
    total_cells = sum(expected_cells.values())
    total_payload_bytes = sum(expected_payload_bytes.values())
    if checked_i64(metrics.get("known_cells"), "witness_metrics.known_cells") != total_cells:
        raise MlpCheckError("witness_metrics.known_cells does not match MLP witness cells")
    if (
        checked_i64(metrics.get("known_payload_bytes"), "witness_metrics.known_payload_bytes")
        != total_payload_bytes
    ):
        raise MlpCheckError(
            "witness_metrics.known_payload_bytes does not match MLP witness payload bytes"
        )
    unknown = metrics.get("unknown_variables")
    if unknown != []:
        raise MlpCheckError("witness_metrics.unknown_variables must be empty")
    metric_entries = expect_metadata_entries(
        metrics.get("entries"),
        "witness_metrics.entries",
    )

    store_entries = expect_metadata_entries(
        output.get("store_metadata"),
        "run output JSON store_metadata",
    )
    for name in WITNESS_TAPE_FIELDS:
        metric = metric_entries.get(name)
        if metric is None:
            raise MlpCheckError(f"witness_metrics.entries missing {name}")
        if metric.get("cells") != expected_cells[name]:
            raise MlpCheckError(f"witness_metrics entry {name} has wrong cells")
        if metric.get("payload_bytes") != expected_payload_bytes[name]:
            raise MlpCheckError(f"witness_metrics entry {name} has wrong payload_bytes")

        store_entry = store_entries.get(name)
        if store_entry is None:
            raise MlpCheckError(f"store_metadata missing witness {name}")
        if store_entry.get("role") != "witness":
            raise MlpCheckError(f"store_metadata {name}.role must be witness")
        if store_entry.get("type") != WITNESS_METADATA_TYPES[name]:
            raise MlpCheckError(f"store_metadata {name}.type does not match declaration")
        if store_entry.get("cells") != expected_cells[name]:
            raise MlpCheckError(f"store_metadata {name}.cells has wrong value")
        if store_entry.get("payload_bytes") != expected_payload_bytes[name]:
            raise MlpCheckError(f"store_metadata {name}.payload_bytes has wrong value")


def validate_run_output_witness_proof(output: dict[str, Any], store: dict[str, Any]) -> dict[str, Any]:
    expected_cells = witness_metadata_cells()
    expected_payload_bytes = {
        name: cells * I64_BYTES for name, cells in expected_cells.items()
    }
    expected_fields = set(WITNESS_TAPE_FIELDS)
    proof = expect_object(output.get("witness_proof"), "run output JSON witness_proof")
    if proof.get("schema") != "reverie_witness_store_proof_v1":
        raise MlpCheckError("witness_proof.schema must be reverie_witness_store_proof_v1")
    if proof.get("algorithm") != "sha256":
        raise MlpCheckError("witness_proof.algorithm must be sha256")
    fingerprint = validate_sha256_text(proof.get("fingerprint"), "witness_proof.fingerprint")
    payload = expect_object(proof.get("payload"), "witness_proof.payload")
    if fingerprint != sha256_json(payload):
        raise MlpCheckError("witness_proof.fingerprint does not match payload")
    if payload.get("schema") != "reverie_witness_store_proof_v1":
        raise MlpCheckError("witness_proof.payload.schema must be reverie_witness_store_proof_v1")
    if payload.get("algorithm") != "sha256":
        raise MlpCheckError("witness_proof.payload.algorithm must be sha256")
    if checked_i64(payload.get("variables"), "witness_proof.payload.variables") != len(
        WITNESS_TAPE_FIELDS
    ):
        raise MlpCheckError("witness_proof.payload.variables does not match MLP witness tapes")
    if checked_i64(payload.get("known_cells"), "witness_proof.payload.known_cells") != sum(
        expected_cells.values()
    ):
        raise MlpCheckError("witness_proof.payload.known_cells does not match MLP witness cells")
    if checked_i64(
        payload.get("known_payload_bytes"),
        "witness_proof.payload.known_payload_bytes",
    ) != sum(expected_payload_bytes.values()):
        raise MlpCheckError(
            "witness_proof.payload.known_payload_bytes does not match MLP witness payload bytes"
        )
    if payload.get("unknown_variables") != []:
        raise MlpCheckError("witness_proof.payload.unknown_variables must be empty")
    entries = expect_metadata_entries(
        payload.get("entries"),
        "witness_proof.payload.entries",
    )
    entry_names = set(entries)
    if entry_names != expected_fields:
        missing = sorted(expected_fields - entry_names)
        extra = sorted(entry_names - expected_fields)
        raise MlpCheckError(
            "witness_proof.payload.entries mismatch: missing={} extra={}".format(
                missing,
                extra,
            )
        )
    for name in WITNESS_TAPE_FIELDS:
        if name not in store:
            raise MlpCheckError(f"store.{name} missing for witness_proof validation")
        entry = entries[name]
        if entry.get("present") is not True:
            raise MlpCheckError(f"witness_proof entry {name}.present must be true")
        if entry.get("cells") != expected_cells[name]:
            raise MlpCheckError(f"witness_proof entry {name}.cells has wrong value")
        if entry.get("payload_bytes") != expected_payload_bytes[name]:
            raise MlpCheckError(f"witness_proof entry {name}.payload_bytes has wrong value")
        actual_fingerprint = validate_sha256_text(
            entry.get("value_fingerprint"),
            f"witness_proof entry {name}.value_fingerprint",
        )
        expected_fingerprint = sha256_json(store[name])
        if actual_fingerprint != expected_fingerprint:
            raise MlpCheckError(f"witness_proof entry {name}.value_fingerprint does not match store")
    return proof


def expect_metadata_entries(value: Any, context: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise MlpCheckError(f"{context} must be an array")
    entries: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(value):
        entry = expect_object(item, f"{context}[{index}]")
        name = entry.get("name")
        if not isinstance(name, str):
            raise MlpCheckError(f"{context}[{index}].name must be a string")
        if name in entries:
            raise MlpCheckError(f"{context} has duplicate entry {name}")
        entries[name] = entry
    return entries


def compare_values(expected: Any, actual: Any, path: str, mismatches: list[str], limit: int) -> None:
    if len(mismatches) >= limit:
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            mismatches.append(f"{path} expected array, found {type(actual).__name__}")
            return
        if len(expected) != len(actual):
            mismatches.append(f"{path} expected length {len(expected)}, found {len(actual)}")
            return
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            compare_values(expected_item, actual_item, f"{path}[{index}]", mismatches, limit)
            if len(mismatches) >= limit:
                return
        return
    if expected != actual:
        mismatches.append(f"{path} expected {expected}, found {actual}")


def compare_store(expected: dict[str, Any], actual: dict[str, Any], limit: int = 20) -> list[str]:
    mismatches: list[str] = []
    for field in CHECKED_STORE_FIELDS:
        if field not in actual:
            mismatches.append(f"store.{field} missing from Reverie output")
            if len(mismatches) >= limit:
                break
            continue
        compare_values(expected[field], actual[field], f"store.{field}", mismatches, limit)
    return mismatches


def parse_expected_predictions(text: Optional[str]) -> Optional[list[int]]:
    if text is None:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise MlpCheckError(f"--expect-predictions must be a JSON array: {error}") from error
    return vector_from_json(value, SAMPLES, "--expect-predictions")


def parse_expected_correct(text: Optional[str]) -> Optional[list[bool]]:
    if text is None:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise MlpCheckError(f"--expect-correct must be a JSON array: {error}") from error
    if not isinstance(value, list) or len(value) != SAMPLES:
        raise MlpCheckError(f"--expect-correct must have {SAMPLES} entries")
    for index, item in enumerate(value):
        if not isinstance(item, bool):
            raise MlpCheckError(f"--expect-correct[{index}] must be a boolean")
    return value


def validate_expectations(
    expected_store: dict[str, Any],
    expected_predictions: Optional[list[int]],
    expected_correct: Optional[list[bool]],
) -> list[str]:
    errors = []
    if (
        expected_predictions is not None
        and expected_store["prediction_tape"] != expected_predictions
    ):
        errors.append(
            "prediction_tape expected {}, found {}".format(
                expected_predictions, expected_store["prediction_tape"]
            )
        )
    if expected_correct is not None:
        actual_correct = [value != 0 for value in expected_store["correct_tape"]]
        if actual_correct != expected_correct:
            errors.append(
                f"correct_tape expected {expected_correct}, found {actual_correct}"
            )
    return errors


def vector_payload_bytes(length: int) -> int:
    return length * I64_BYTES


def matrix_payload_bytes(rows: int, cols: int) -> int:
    return rows * cols * I64_BYTES


def mlp_proof_cost() -> dict[str, Any]:
    model_payload_bytes = (
        matrix_payload_bytes(IMAGE_PIXELS, HIDDEN)
        + vector_payload_bytes(HIDDEN)
        + matrix_payload_bytes(HIDDEN, DIGITS)
        + vector_payload_bytes(DIGITS)
    )
    sample_payload_bytes = (
        matrix_payload_bytes(SAMPLES, IMAGE_PIXELS) + vector_payload_bytes(SAMPLES)
    )
    witness_bytes_per_sample = (
        5 * vector_payload_bytes(HIDDEN)
        + 2 * vector_payload_bytes(DIGITS)
        + 2 * vector_payload_bytes(1)
    )
    witness_payload_bytes = SAMPLES * witness_bytes_per_sample
    trace_payload_bytes = sample_payload_bytes + witness_payload_bytes
    recomputed_update_bytes_per_sample = model_payload_bytes
    recomputed_update_payload_bytes = SAMPLES * recomputed_update_bytes_per_sample
    replay_payload_bytes = model_payload_bytes + trace_payload_bytes
    total_recompute_steps = SAMPLES * 2
    return {
        "claim": "deterministic_q31_mlp_witness_replay",
        "model_payload_bytes": model_payload_bytes,
        "sample_payload_bytes": sample_payload_bytes,
        "witness_payload_bytes": witness_payload_bytes,
        "witness_bytes_per_sample": witness_bytes_per_sample,
        "trace_payload_bytes": trace_payload_bytes,
        "trace_bytes_per_sample": trace_payload_bytes // SAMPLES,
        "replay_payload_bytes": replay_payload_bytes,
        "stored_derived_update_payload_bytes": 0,
        "recomputed_update_payload_bytes": recomputed_update_payload_bytes,
        "recomputed_update_bytes_per_sample": recomputed_update_bytes_per_sample,
        "forward_recompute_steps": SAMPLES,
        "inverse_recompute_steps": SAMPLES,
        "total_recompute_steps": total_recompute_steps,
        "witness_to_model_payload_ratio": witness_payload_bytes / model_payload_bytes,
        "trace_to_model_payload_ratio": trace_payload_bytes / model_payload_bytes,
        "recomputed_update_to_witness_payload_ratio": (
            recomputed_update_payload_bytes / witness_payload_bytes
        ),
        "checked_store_fields": len(CHECKED_STORE_FIELDS),
        "witness_tape_fields": list(WITNESS_TAPE_FIELDS),
    }


def count_nonzero(values: list[int]) -> int:
    return sum(1 for value in values if value != 0)


def witness_summary(expected_store: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for sample in range(SAMPLES):
        rows.append(
            {
                "sample": sample,
                "label": expected_store["labels"][sample],
                "prediction": expected_store["prediction_tape"][sample],
                "correct": expected_store["correct_tape"][sample] != 0,
                "active_pixels": count_nonzero(expected_store["images"][sample]),
                "active_hidden_mask": count_nonzero(expected_store["hidden_mask_tape"][sample]),
                "active_hidden": count_nonzero(expected_store["hidden_tape"][sample]),
                "nonzero_output_error": count_nonzero(expected_store["out_error_tape"][sample]),
                "nonzero_hidden_delta": count_nonzero(expected_store["hidden_delta_tape"][sample]),
            }
        )
    return {
        "samples": rows,
        "total_active_pixels": sum(row["active_pixels"] for row in rows),
        "total_active_hidden": sum(row["active_hidden"] for row in rows),
        "total_nonzero_output_error": sum(row["nonzero_output_error"] for row in rows),
        "total_nonzero_hidden_delta": sum(row["nonzero_hidden_delta"] for row in rows),
    }


def witness_value_fingerprints(witness_proof: Optional[dict[str, Any]]) -> dict[str, str]:
    if witness_proof is None:
        return {}
    payload = witness_proof.get("payload")
    if not isinstance(payload, dict):
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return {}
    fingerprints = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        fingerprint = entry.get("value_fingerprint")
        if isinstance(name, str) and isinstance(fingerprint, str):
            fingerprints[name] = fingerprint
    return fingerprints


def witness_tape_records(witness_proof: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    cells = witness_metadata_cells()
    fingerprints = witness_value_fingerprints(witness_proof)
    return [
        {
            "name": name,
            "type": WITNESS_METADATA_TYPES[name],
            "cells": cells[name],
            "payload_bytes": cells[name] * I64_BYTES,
            "value_fingerprint": fingerprints.get(name),
        }
        for name in WITNESS_TAPE_FIELDS
    ]


def report_json(
    vars_path: Optional[Path],
    run_output_path: Optional[Path],
    expected_store: dict[str, Any],
    mismatches: list[str],
    witness_proof: Optional[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "kind": "reverie_q31_mlp_witness_reference",
        "program": "examples/mnist_mlp_witness.rev",
        "vars_json": str(vars_path) if vars_path is not None else None,
        "run_output_json": str(run_output_path) if run_output_path is not None else None,
        "samples": SAMPLES,
        "q31_one": Q31_ONE,
        "lr": expected_store["lr"],
        "labels": expected_store["labels"],
        "dataset_loops": deepcopy(EXPECTED_DATASET_LOOPS),
        "predictions": expected_store["prediction_tape"],
        "correct": [value != 0 for value in expected_store["correct_tape"]],
        "witness_summary": witness_summary(expected_store),
        "witness_tapes": witness_tape_records(witness_proof),
        "checked_store_fields": list(CHECKED_STORE_FIELDS),
        "witness_proof": witness_proof,
        "proof": mlp_proof_cost(),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def render_witness_rows(rows: list[dict[str, Any]]) -> list[str]:
    rendered = []
    for row in rows:
        rendered.append(
            "| `{}` | `{}` | {} | {} | `{}` |".format(
                row["name"],
                row["type"],
                comma(row["cells"]),
                comma(row["payload_bytes"]),
                short_hash(row.get("value_fingerprint")) or "-",
            )
        )
    return rendered


def render_sample_rows(rows: list[dict[str, Any]]) -> list[str]:
    rendered = []
    for row in rows:
        rendered.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["sample"],
                row["label"],
                row["prediction"],
                str(row["correct"]).lower(),
                comma(row["active_pixels"]),
                comma(row["active_hidden_mask"]),
                comma(row["active_hidden"]),
                comma(row["nonzero_output_error"]),
                comma(row["nonzero_hidden_delta"]),
            )
        )
    return rendered


def render_markdown(report: dict[str, Any]) -> str:
    proof = report["proof"]
    witness_proof = report.get("witness_proof")
    witness_fingerprint = (
        short_hash(witness_proof.get("fingerprint"))
        if isinstance(witness_proof, dict)
        else "-"
    )
    summary = report["witness_summary"]
    dataset_loop = report["dataset_loops"][0]
    lines = [
        "# Reverie MLP Witness Proof",
        "",
        "| Passed | Samples | Predictions | Correct | LR | Witness proof | Replay bytes | Witness bytes | Trace bytes |",
        "| --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: |",
        "| {} | {} | {} | {} | {} | `{}` | {} | {} | {} |".format(
            str(report["passed"]).lower(),
            comma(report["samples"]),
            json_cell(report["predictions"]),
            json_cell(report["correct"]),
            comma(report["lr"]),
            witness_fingerprint,
            comma(proof["replay_payload_bytes"]),
            comma(proof["witness_payload_bytes"]),
            comma(proof["trace_payload_bytes"]),
        ),
        "",
        "## Dataset Loop",
        "",
        "| Index | Size source |",
        "| --- | --- |",
        "| `{}` | `{}` |".format(
            dataset_loop["index"],
            ",".join(dataset_loop["size_sources"]),
        ),
        "",
        "## Sample Witnesses",
        "",
        "| Sample | Label | Prediction | Correct | Active pixels | Hidden mask | Hidden active | Output error | Hidden delta |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        *render_sample_rows(summary["samples"]),
        "",
        "## Witness Tapes",
        "",
        "| Tape | Type | Cells | Payload bytes | Value fingerprint |",
        "| --- | --- | ---: | ---: | --- |",
        *render_witness_rows(report["witness_tapes"]),
        "",
        "## Replay Cost",
        "",
        "| Model | Sample | Witness | Trace | Replay | Recomputed update | Forward recompute | Inverse recompute |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            comma(proof["model_payload_bytes"]),
            comma(proof["sample_payload_bytes"]),
            comma(proof["witness_payload_bytes"]),
            comma(proof["trace_payload_bytes"]),
            comma(proof["replay_payload_bytes"]),
            comma(proof["recomputed_update_payload_bytes"]),
            comma(proof["forward_recompute_steps"]),
            comma(proof["inverse_recompute_steps"]),
        ),
        "",
        "## Tradeoffs",
        "",
        "| Witness/model | Trace/model | Recomputed update/witness | Stored update bytes | Checked fields |",
        "| ---: | ---: | ---: | ---: | ---: |",
        "| {:.6f} | {:.6f} | {:.6f} | {} | {} |".format(
            proof["witness_to_model_payload_ratio"],
            proof["trace_to_model_payload_ratio"],
            proof["recomputed_update_to_witness_payload_ratio"],
            comma(proof["stored_derived_update_payload_bytes"]),
            comma(proof["checked_store_fields"]),
        ),
    ]
    if report["mismatches"]:
        lines.extend(["", "## Mismatches", "", "| Mismatch |", "| --- |"])
        lines.extend(f"| `{mismatch}` |" for mismatch in report["mismatches"])
    lines.append("")
    return "\n".join(lines)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(report), encoding="utf-8")
    except OSError as error:
        raise MlpCheckError(f"failed to write {path}: {error}") from error


def validate_expected_report(path: Path, report: dict[str, Any]) -> None:
    expected = load_json(path)
    if expected != report:
        raise MlpCheckError(f"recomputed report does not match {path}")


def self_test_seed() -> dict[str, Any]:
    images = [[0 for _ in range(IMAGE_PIXELS)] for _ in range(SAMPLES)]
    images[0][0] = Q31_ONE
    images[1][1] = Q31_ONE
    w1 = [[0 for _ in range(HIDDEN)] for _ in range(IMAGE_PIXELS)]
    b1 = [0 for _ in range(HIDDEN)]
    w2 = [[0 for _ in range(DIGITS)] for _ in range(HIDDEN)]
    b2 = [0 for _ in range(DIGITS)]
    w1[0][0] = Q31_ONE
    w1[1][1] = Q31_ONE
    w2[0][0] = Q31_ONE
    w2[1][1] = Q31_ONE
    return {
        "images": images,
        "labels": [0, 1],
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "lr": DEFAULT_LR,
    }


def valid_run_output_for_tests(store: dict[str, Any]) -> dict[str, Any]:
    cells = witness_metadata_cells()
    entries = [
        {
            "name": name,
            "cells": cells[name],
            "payload_bytes": cells[name] * I64_BYTES,
        }
        for name in sorted(WITNESS_TAPE_FIELDS)
    ]
    store_metadata = [
        {
            "name": name,
            "type": WITNESS_METADATA_TYPES[name],
            "role": "witness",
            "source": "declaration",
            "cells": cells[name],
            "payload_bytes": cells[name] * I64_BYTES,
        }
        for name in sorted(WITNESS_TAPE_FIELDS)
    ]
    proof_entries = [
        {
            "name": name,
            "present": True,
            "cells": cells[name],
            "payload_bytes": cells[name] * I64_BYTES,
            "value_fingerprint": sha256_json(store[name]),
        }
        for name in sorted(WITNESS_TAPE_FIELDS)
    ]
    proof_payload = {
        "schema": "reverie_witness_store_proof_v1",
        "algorithm": "sha256",
        "variables": len(WITNESS_TAPE_FIELDS),
        "known_cells": sum(cells.values()),
        "known_payload_bytes": sum(cells[name] * I64_BYTES for name in cells),
        "unknown_variables": [],
        "entries": proof_entries,
    }
    return {
        "kind": "reverie_run_result",
        "program": {
            "file": PROGRAM_RELATIVE_PATH,
            "source_sha256": sha256_file(PROGRAM_PATH),
            "direction": "run",
            "engine": "slot",
            "legacy_janus": False,
        },
        "store": store,
        "input": [],
        "output": [],
        "observations": [],
        "dataset_loops": deepcopy(EXPECTED_DATASET_LOOPS),
        "store_metadata": store_metadata,
        "witness_store": sorted(WITNESS_TAPE_FIELDS),
        "witness_metrics": {
            "variables": len(WITNESS_TAPE_FIELDS),
            "known_cells": sum(cells.values()),
            "known_payload_bytes": sum(cells[name] * I64_BYTES for name in cells),
            "unknown_variables": [],
            "entries": entries,
        },
        "witness_proof": {
            "schema": "reverie_witness_store_proof_v1",
            "algorithm": "sha256",
            "fingerprint": sha256_json(proof_payload),
            "payload": proof_payload,
        },
    }


def run_self_tests() -> int:
    try:
        seed = parse_seed_store(self_test_seed())
        expected = simulate(seed)
        proof = mlp_proof_cost()
        if expected["prediction_tape"] != [0, 1] or expected["correct_tape"] != [1, 1]:
            raise AssertionError("self-test MLP predictions did not match expected labels")
        if proof["witness_payload_bytes"] != 1_632:
            raise AssertionError("self-test MLP witness payload changed unexpectedly")
        if proof["trace_payload_bytes"] != 14_192:
            raise AssertionError("self-test MLP trace payload changed unexpectedly")
        if proof["recomputed_update_payload_bytes"] != 203_680:
            raise AssertionError("self-test MLP recomputed update payload changed unexpectedly")
        run_output = valid_run_output_for_tests(expected)
        parsed = parse_run_output(run_output)
        if parsed["witness_proof"]["fingerprint"] != sha256_json(parsed["witness_proof"]["payload"]):
            raise AssertionError("self-test witness proof fingerprint did not match payload")
        mismatches = compare_store(expected, parsed["store"])
        if mismatches:
            raise AssertionError("self-test comparison unexpectedly failed")
        report = report_json(None, None, expected, [], parsed["witness_proof"])
        markdown = render_markdown(report)
        for snippet in (
            "# Reverie MLP Witness Proof",
            "## Sample Witnesses",
            "## Witness Tapes",
            "## Replay Cost",
            short_hash(parsed["witness_proof"]["fingerprint"]),
        ):
            if snippet not in markdown:
                raise AssertionError(f"rendered Markdown missing {snippet}")
        tampered = deepcopy(run_output)
        tampered["store"]["b2"][0] = 123
        mismatches = compare_store(expected, parse_run_output(tampered)["store"])
        if not mismatches or "b2" not in mismatches[0]:
            raise AssertionError("self-test tampered model state was not detected")
        tampered_witness = deepcopy(run_output)
        tampered_witness["store"]["logits_tape"][0][0] = 123
        try:
            parse_run_output(tampered_witness)
        except MlpCheckError as error:
            if "value_fingerprint" not in str(error):
                raise AssertionError(f"wrong bad-witness-proof error: {error}") from error
        else:
            raise AssertionError("self-test tampered witness proof was not rejected")
        tampered_metadata = deepcopy(run_output)
        tampered_metadata["witness_metrics"]["known_payload_bytes"] = 1
        try:
            parse_run_output(tampered_metadata)
        except MlpCheckError as error:
            if "known_payload_bytes" not in str(error):
                raise AssertionError(f"wrong bad-metadata error: {error}") from error
        else:
            raise AssertionError("self-test bad witness metadata was not rejected")
        tampered_dataset_loop = deepcopy(run_output)
        tampered_dataset_loop["dataset_loops"] = []
        try:
            parse_run_output(tampered_dataset_loop)
        except MlpCheckError as error:
            if "dataset_loops" not in str(error):
                raise AssertionError(f"wrong bad-dataset-loop error: {error}") from error
        else:
            raise AssertionError("self-test bad dataset loop metadata was not rejected")
        tampered_provenance = deepcopy(run_output)
        tampered_provenance["program"]["source_sha256"] = "0" * 64
        try:
            parse_run_output(tampered_provenance)
        except MlpCheckError as error:
            if "source_sha256" not in str(error):
                raise AssertionError(f"wrong bad-provenance error: {error}") from error
        else:
            raise AssertionError("self-test bad source provenance was not rejected")
        try:
            parse_seed_store({"w1": [[0]], "b1": [0] * HIDDEN, "w2": seed["w2"], "b2": seed["b2"]})
        except MlpCheckError as error:
            if "w1" not in str(error):
                raise AssertionError(f"wrong bad-shape error: {error}") from error
        else:
            raise AssertionError("self-test bad shape was not rejected")
    except (AssertionError, MlpCheckError) as error:
        print(f"error: self-test failed: {error}", file=sys.stderr)
        return 1

    print("ok: Q31 MLP witness checker self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    if args.vars_json is None:
        print("error: --vars-json is required", file=sys.stderr)
        return 2

    try:
        seed = parse_seed_store(load_json(args.vars_json))
        expected_store = simulate(seed)
        mismatches: list[str] = []
        witness_proof: Optional[dict[str, Any]] = None
        if args.run_output_json is not None:
            parsed_output = parse_run_output(load_json(args.run_output_json))
            actual_store = parsed_output["store"]
            witness_proof = parsed_output["witness_proof"]
            mismatches.extend(compare_store(expected_store, actual_store))
        mismatches.extend(
            validate_expectations(
                expected_store,
                parse_expected_predictions(args.expect_predictions),
                parse_expected_correct(args.expect_correct),
            )
        )
    except MlpCheckError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    report = report_json(
        args.vars_json,
        args.run_output_json,
        expected_store,
        mismatches,
        witness_proof,
    )
    try:
        if args.expect_report_json is not None:
            validate_expected_report(args.expect_report_json, report)
        if args.markdown_output is not None:
            write_markdown(args.markdown_output, report)
    except MlpCheckError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
            )
        )
    elif mismatches:
        for mismatch in mismatches:
            print(f"error: {mismatch}", file=sys.stderr)
    else:
        print(
            "ok: predictions={} correct={} checked_fields={}".format(
                expected_store["prediction_tape"],
                [value != 0 for value in expected_store["correct_tape"]],
                len(CHECKED_STORE_FIELDS),
            )
            + " witness_bytes={} trace_bytes={} recompute_steps={}".format(
                mlp_proof_cost()["witness_payload_bytes"],
                mlp_proof_cost()["trace_payload_bytes"],
                mlp_proof_cost()["total_recompute_steps"],
            )
        )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
