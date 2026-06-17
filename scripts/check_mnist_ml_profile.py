#!/usr/bin/env python3
"""Validate Reverie MNIST ML speed, memory, and proof-cost reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import check_invertible_coupling as coupling_check
import check_triangular_residual as residual_check
import check_reversible_inference_trace as inference_trace_check
import check_reversible_preprocess as preprocess_check


RUN_KIND = "reverie_mnist_linear_q31"
COMPARISON_KIND = "reverie_mnist_linear_q31_artifact_comparison"
MLP_WITNESS_KIND = "reverie_q31_mlp_witness_reference"
MLP_WITNESS_PROOF_CLAIM = "deterministic_q31_mlp_witness_replay"
INVERTIBLE_COUPLING_KIND = "reverie_q31_invertible_coupling_reference"
INVERTIBLE_COUPLING_PROOF_CLAIM = "deterministic_q31_invertible_coupling_replay"
TRIANGULAR_RESIDUAL_KIND = "reverie_q31_triangular_residual_reference"
TRIANGULAR_RESIDUAL_PROOF_CLAIM = "deterministic_q31_triangular_residual_replay"
REVERSIBLE_PREPROCESS_KIND = "reverie_q31_reversible_preprocess_reference"
REVERSIBLE_PREPROCESS_PROOF_CLAIM = "deterministic_q31_reversible_preprocess_replay"
REVERSIBLE_INFERENCE_TRACE_KIND = "reverie_q31_reversible_inference_trace_reference"
REVERSIBLE_INFERENCE_TRACE_PROOF_CLAIM = "deterministic_q31_reversible_inference_trace"
Q31_REFERENCE_INFERENCE_KIND = "reverie_q31_linear_reference_inference"
Q31_REFERENCE_EVALUATION_KIND = "reverie_q31_linear_reference_evaluation"
AUDIT_STEP_KIND = "reverie_mnist_linear_q31_audit_step"
AUDIT_SCAN_KIND = "reverie_mnist_linear_q31_audit_scan"
AUDIT_VERIFICATION_KIND = "reverie_mnist_linear_q31_audit_verification"
STEP_VERIFICATION_KIND = "reverie_mnist_linear_q31_step_verification"
TRAINING_STEP_PROOF_CLAIM = "deterministic_q31_training_step_replay"
TRAINING_CAUSE_LEDGER_SCHEMA = "q31_training_update_cause_ledger_v1"
TRAINING_LINEAGE_LEDGER_SCHEMA = "q31_training_lineage_ledger_v1"
AUDIT_FINGERPRINT_FIELDS = (
    "train_source",
    "computation",
    "final_model",
    "witness_trace",
    "report",
    "proof",
    "lineage_ledger",
    "payload",
)
INFERENCE_AUDIT_KIND = "reverie_mnist_linear_q31_inference_audit"
MODEL_INFERENCE_AUDIT_KIND = "reverie_mnist_linear_q31_model_inference_audit"
MODEL_EVALUATION_ROW_KIND = "reverie_mnist_linear_q31_model_evaluation_row"
INFERENCE_VERIFICATION_KIND = "reverie_mnist_linear_q31_inference_verification"
INFERENCE_EXPLANATION_CLAIM = "q31_inference_prediction_explanation"
MODEL_EVALUATION_KIND = "reverie_mnist_linear_q31_model_evaluation"
MODEL_EVALUATION_SCAN_KIND = "reverie_mnist_linear_q31_model_evaluation_scan"
MODEL_EVALUATION_VERIFICATION_KIND = "reverie_mnist_linear_q31_model_evaluation_verification"
MODEL_IMPORT_KIND = "reverie_mnist_linear_q31_model_import"
MODEL_VERIFICATION_KIND = "reverie_mnist_linear_q31_model_verification"
PIPELINE_KIND = "reverie_mnist_ml_audit_pipeline"
PIPELINE_SUMMARY_KIND = "reverie_mnist_ml_audit_pipeline_summary"
PIPELINE_CLAIMS_KIND = "reverie_reversible_ml_audit_claims"
MODEL_CAPSULE_KIND = "reverie_reversible_ml_model_capsule"
MODEL_CAPSULE_SCHEMA = "reverie_reversible_ml_model_capsule_v1"
ML_CAPABILITY_KIND = "reverie_mnist_ml_capability_map"
ML_GOAL_READINESS_KIND = "reverie_reversible_ml_goal_readiness"
RECOMPUTE_FRONTIER_KIND = "reverie_mnist_ml_recompute_frontier"
SCALING_PROJECTION_KIND = "reverie_mnist_ml_scaling_projection"
INFERENCE_TRACE_PROFILE_KIND = "reverie_mnist_ml_inference_trace_profile"
INFERENCE_PROOF_CLAIM = "deterministic_q31_inference_replay"
IMAGE_PIXELS = 784
DIGITS = 10
Q31_ONE = 1 << 31
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
I64_BYTES = 8
U64_MOD = 1 << 64
AUDIT_STEP_STRATEGIES = ("explicit", "lowest-margin", "largest-update", "top-suspicious")
EVALUATION_ROW_STRATEGIES = ("explicit", "lowest-margin", "top-incorrect")
PROFILE_FIELDS = (
    "total_model_payload_bytes",
    "total_sample_payload_bytes",
    "total_witness_payload_bytes",
    "total_trace_payload_bytes",
    "total_derived_update_payload_bytes",
    "total_steps",
    "total_forward_recompute_steps",
    "total_inverse_recompute_steps",
    "total_recompute_steps",
)
ARTIFACT_FIELDS = (
    "file_bytes",
    "logical_payload_bytes",
    "model_payload_bytes",
    "sample_payload_bytes",
    "witness_payload_bytes",
    "trace_payload_bytes",
    "derived_update_payload_bytes",
    "steps",
    "forward_recompute_steps",
    "inverse_recompute_steps",
)
AUDIT_CONTRACT_CLAIM = "reversible_inspectable_deterministic_q31_ml_kernel"
AUDIT_CONTRACT_METRICS = (
    "training_trace",
    "model_bundle",
    "sample_set",
    "training_step_replay",
    "inference_replay",
    "evaluation_replay",
    "balanced_recompute",
    "bounded_payload_ratios",
    "fingerprint_coverage",
    "proof_or_provenance_fingerprints",
)
PIPELINE_REPORT_KEYS = (
    "run_report",
    "artifact_comparison",
    "audit_verification",
    "audit_scan",
    "audit_step",
    "step_verification",
    "imported_model_import",
    "imported_model_verification",
    "imported_model_inference",
    "imported_model_inference_verification",
    "native_inference",
    "native_inference_verification",
    "model_evaluation",
    "model_evaluation_verification",
    "model_evaluation_scan",
    "model_evaluation_row",
    "row_inference_verification",
    "mlp_witness",
    "invertible_coupling",
    "triangular_residual",
    "reversible_preprocess",
    "reversible_inference_trace",
    "q31_reference_inference",
    "q31_reference_evaluation",
)
PIPELINE_BUNDLE_KEYS = (
    "training_audit",
    "training_step",
    "model",
    "imported_model_source",
    "imported_model",
    "imported_model_inference",
    "samples",
    "native_inference",
    "model_evaluation",
    "evaluation_row_inference",
    "mlp_vars",
    "mlp_run_output",
    "coupling_final_vars",
    "coupling_forward",
    "coupling_reverse",
    "residual_final_vars",
    "residual_forward",
    "residual_reverse",
    "preprocess_final_vars",
    "preprocess_forward",
    "preprocess_reverse",
    "inference_trace_final_vars",
    "inference_trace_forward",
    "inference_trace_reverse",
    "inference_trace_roundtrip_proof",
    "inference_trace_roundtrip_verification",
)
PIPELINE_STANDALONE_KEYS = (
    "native_standalone_rev_classifier",
    "native_standalone_rev_run",
    "native_standalone_rev_roundtrip",
    "native_standalone_rev_roundtrip_verification",
)
PIPELINE_GATE_METRICS = (
    "train_accuracy_percent",
    "built_in_eval_accuracy_percent",
    "training_audit_accuracy_percent",
    "model_evaluation_accuracy_percent",
    "q31_reference_accuracy_percent",
    "reverse_restored_initial_model",
    "reverse_checked_all_training_steps",
    "run_peak_rss_bytes",
    "training_audit_steps",
    "training_audit_lineage_replay",
    "training_audit_witness_mismatches",
    "training_step_selection_traceable",
    "model_evaluation_samples",
    "q31_reference_samples",
    "model_evaluation_scan_matches_report",
    "evaluation_row_selection_traceable",
    "q31_reference_matches_native_inference",
    "q31_reference_matches_native_evaluation",
    "inference_trace_profile_complete",
    "audit_contract",
    "trace_to_model_payload_ratio",
    "witness_to_model_payload_ratio",
    "balanced_recompute_steps",
    "recompute_frontier_complete",
    "scaling_projection_complete",
    "reverse_check_cost_measured",
    "reverse_check_elapsed_ratio",
    "training_step_replay",
    "training_step_debug_contract",
    "training_update_ledger_fingerprints",
    "model_import_provenance",
    "imported_model_inference_replay",
    "native_inference_replay",
    "native_inference_explanation_contract",
    "native_standalone_rev_classifier",
    "model_evaluation_replay",
    "evaluation_row_inference_replay",
    "evaluation_row_inference_explanation_contract",
    "mlp_witness_replay",
    "invertible_coupling_replay",
    "triangular_residual_replay",
    "reversible_preprocess_replay",
    "reversible_inference_trace_replay",
    "v6_scorecard_complete",
    "ml_roadmap_capability_map_complete",
    "ml_goal_readiness_complete",
)
PIPELINE_CLAIMS = (
    "debug_training_update",
    "auditable_model_lineage",
    "deterministic_q31_inference",
    "memory_recompute_profile",
    "mlp_activation_mask_witnesses",
    "invertible_layer_without_witness",
    "artifact_evidence_integrity",
    "v6_ml_audit_scorecard",
    "ml_roadmap_capabilities",
    "north_star_reversible_ml_kernels",
)
PIPELINE_CAPABILITIES = (
    (
        "V1",
        "reversible_linear_mnist",
        (
            "train_accuracy_percent",
            "reverse_restored_initial_model",
            "reverse_checked_all_training_steps",
            "training_audit_lineage_replay",
            "training_step_replay",
            "model_import_provenance",
            "imported_model_inference_replay",
            "native_inference_replay",
            "inference_trace_profile_complete",
            "reversible_inference_trace_replay",
        ),
    ),
    (
        "V2",
        "witness_tapes",
        (
            "training_audit_witness_mismatches",
            "training_step_selection_traceable",
            "training_step_debug_contract",
            "training_update_ledger_fingerprints",
        ),
    ),
    (
        "V3",
        "batched_tensor_iteration",
        (
            "model_evaluation_samples",
            "model_evaluation_replay",
            "q31_reference_samples",
            "q31_reference_matches_native_evaluation",
            "evaluation_row_selection_traceable",
        ),
    ),
    ("V4", "reversible_mlp_witnesses", ("mlp_witness_replay",)),
    (
        "V5",
        "invertible_layer_pattern",
        (
            "invertible_coupling_replay",
            "triangular_residual_replay",
            "reversible_preprocess_replay",
        ),
    ),
    (
        "V6",
        "speed_memory_trace_reverse_scorecard",
        (
            "run_peak_rss_bytes",
            "balanced_recompute_steps",
            "recompute_frontier_complete",
            "scaling_projection_complete",
            "reverse_check_cost_measured",
            "reverse_check_elapsed_ratio",
            "v6_scorecard_complete",
        ),
    ),
)
PIPELINE_GOAL_READINESS = (
    (
        "debug_training",
        (
            "training_step_selection_traceable",
            "training_step_replay",
            "training_step_debug_contract",
            "training_update_ledger_fingerprints",
        ),
    ),
    (
        "audit_model_lineage",
        (
            "reverse_restored_initial_model",
            "reverse_checked_all_training_steps",
            "training_audit_lineage_replay",
            "audit_contract",
            "ml_roadmap_capability_map_complete",
        ),
    ),
    (
        "deterministic_inference",
        (
            "native_inference_replay",
            "native_inference_explanation_contract",
            "native_standalone_rev_classifier",
            "model_import_provenance",
            "imported_model_inference_replay",
            "evaluation_row_inference_replay",
            "evaluation_row_inference_explanation_contract",
            "q31_reference_matches_native_inference",
            "q31_reference_matches_native_evaluation",
            "inference_trace_profile_complete",
            "reversible_inference_trace_replay",
        ),
    ),
    (
        "memory_recompute_tradeoffs",
        (
            "run_peak_rss_bytes",
            "balanced_recompute_steps",
            "recompute_frontier_complete",
            "scaling_projection_complete",
            "reverse_check_cost_measured",
            "reverse_check_elapsed_ratio",
            "v6_scorecard_complete",
        ),
    ),
    (
        "invertible_model_patterns",
        (
            "mlp_witness_replay",
            "invertible_coupling_replay",
            "triangular_residual_replay",
            "reversible_preprocess_replay",
        ),
    ),
)
INFERENCE_ACTION_OPERATIONS = (
    "reproduce_prediction",
    "explain_margin",
    "replay_imported_model_inference",
    "replay_native_inference",
    "run_standalone_rev_classifier",
    "reverse_reversible_trace",
)
RECOMPUTE_FRONTIER_ROWS = (
    "training_trace",
    "training_step_debug",
    "imported_model_inference",
    "native_inference",
    "model_evaluation_batch",
    "evaluation_row_inference",
    "mlp_witness_trace",
    "invertible_coupling",
    "triangular_residual",
    "reversible_preprocess",
    "reversible_inference_trace",
)
SCALING_PROJECTION_FAMILIES = (
    "training_trace",
    "model_evaluation_batch",
    "mlp_witness_trace",
)
SCALING_PROJECTION_SCALES = (1, 10, 100)


def wrap_i64(value: int) -> int:
    value %= U64_MOD
    if value >= (1 << 63):
        value -= U64_MOD
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Reverie MNIST ML speed, memory, and proof-cost reports."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="JSON report(s) to validate.")
    parser.add_argument(
        "--require-reverse-check",
        action="store_true",
        help="Require run reports to include an enabled reverse check.",
    )
    parser.add_argument(
        "--require-peak-rss",
        action="store_true",
        help="Require memory.peak_rss_bytes to be present and positive.",
    )
    parser.add_argument(
        "--require-ml-profile",
        action="store_true",
        help="Require every report to be an artifact comparison with ml_profile.",
    )
    parser.add_argument(
        "--require-audit-contract",
        action="store_true",
        help="Require artifact comparisons to satisfy the reversible ML audit contract.",
    )
    parser.add_argument(
        "--verify-pipeline-files",
        action="store_true",
        help="For pipeline summary/manifest reports, recompute evidence SHA-256 and file byte sizes.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent ML profile checker self-tests and exit.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"failed to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"failed to parse {path}: {error}") from error


def json_field(value: Any, name: str, context: str) -> Any:
    if not isinstance(value, dict) or name not in value:
        raise ValueError(f"{context} missing `{name}`")
    return value[name]


def expect_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean")
    return value


def expect_nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def expect_positive_int(value: Any, context: str) -> int:
    actual = expect_nonnegative_int(value, context)
    if actual <= 0:
        raise ValueError(f"{context} must be positive")
    return actual


def expect_finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    actual = float(value)
    if not math.isfinite(actual):
        raise ValueError(f"{context} must be finite")
    return actual


def expect_optional_peak_rss(value: Any, context: str, required: bool) -> None:
    if value is None:
        if required:
            raise ValueError(f"{context} is required")
        return
    expect_positive_int(value, context)


def expect_ratio(value: Any, numerator: int, denominator: int, context: str) -> None:
    if denominator == 0:
        if value is not None:
            raise ValueError(f"{context} must be null when denominator is zero")
        return
    actual = expect_finite_number(value, context)
    expected = numerator / denominator
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{context} expected {expected}, found {actual}")


def expect_float_ratio(value: Any, numerator: float, denominator: float, context: str) -> None:
    if denominator == 0.0:
        if value is not None:
            raise ValueError(f"{context} must be null when denominator is zero")
        return
    actual = expect_finite_number(value, context)
    expected = numerator / denominator
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{context} expected {expected}, found {actual}")


def expect_i64(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be a signed integer")
    if not I64_MIN <= value <= I64_MAX:
        raise ValueError(f"{context} is outside signed i64 range")
    return value


def expect_digit(value: Any, context: str) -> int:
    digit = expect_nonnegative_int(value, context)
    if digit >= DIGITS:
        raise ValueError(f"{context} must be in 0..9")
    return digit


def expect_byte(value: Any, context: str) -> int:
    byte = expect_nonnegative_int(value, context)
    if byte > 255:
        raise ValueError(f"{context} must be a byte")
    return byte


def fixed_mul_q31(left: int, right: int) -> int:
    return wrap_i64((left * right) >> 31)


def validate_q31_constant(value: Any, context: str) -> None:
    actual = expect_positive_int(value, context)
    if actual != Q31_ONE:
        raise ValueError(f"{context} must equal 1 << 31")


def validate_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def validate_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{context} must be a 64-character SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{context} must be hex") from error
    return value


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_fingerprints(value: Any, fields: tuple[str, ...], context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    algorithm = json_field(value, "algorithm", context)
    if algorithm != "sha256":
        raise ValueError(f"{context}.algorithm must be sha256")
    for field in fields:
        validate_sha256(json_field(value, field, context), f"{context}.{field}")


def validate_logits(value: Any, context: str) -> list[int]:
    if not isinstance(value, list) or len(value) != DIGITS:
        raise ValueError(f"{context} must have {DIGITS} entries")
    return [expect_i64(item, f"{context}[{index}]") for index, item in enumerate(value)]


def validate_top_logits(value: Any, logits: list[int], prediction: int, context: str) -> list[dict[str, int]]:
    if not isinstance(value, list) or len(value) != min(3, DIGITS):
        raise ValueError(f"{context} must contain the top {min(3, DIGITS)} logits")
    expected = [
        {"digit": digit, "value": logit}
        for digit, logit in enumerate(logits)
    ]
    expected.sort(key=lambda item: (-item["value"], item["digit"]))
    expected = expected[: len(value)]
    seen: set[int] = set()
    checked: list[dict[str, int]] = []
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        digit = expect_digit(json_field(item, "digit", item_context), f"{item_context}.digit")
        logit = expect_i64(json_field(item, "value", item_context), f"{item_context}.value")
        if digit in seen:
            raise ValueError(f"{context} has duplicate digit {digit}")
        seen.add(digit)
        if logit != logits[digit]:
            raise ValueError(f"{item_context}.value must match logits[{digit}]")
        checked.append({"digit": digit, "value": logit})
    if checked != expected:
        raise ValueError(f"{context} must be sorted by value descending and digit ascending")
    if checked[0]["digit"] != prediction:
        raise ValueError(f"{context}[0].digit must match prediction")
    return checked


def validate_sample_source(value: Any, context: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    for field in ("sample_index", "sample_count", "audit_step", "source_sample_index"):
        if field in value:
            expect_nonnegative_int(value[field], f"{context}.{field}")
    for field in ("sample_kind", "sample_set_kind"):
        if field in value and not isinstance(value[field], str):
            raise ValueError(f"{context}.{field} must be a string")
    if "sample_index" in value and "sample_count" in value:
        sample_index = int(value["sample_index"])
        sample_count = int(value["sample_count"])
        if sample_count == 0 or sample_index >= sample_count:
            raise ValueError(f"{context}.sample_index must be within sample_count")


def contribution_sort_key(item: dict[str, int]) -> tuple[int, int]:
    return (-abs(item["contribution"]), item["pixel"])


def validate_contribution_rows(value: Any, context: str, *, margin: bool) -> list[dict[str, int]]:
    if not isinstance(value, list) or len(value) > 10:
        raise ValueError(f"{context} must be an array with at most 10 entries")
    rows: list[dict[str, int]] = []
    seen: set[int] = set()
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        pixel = expect_nonnegative_int(json_field(item, "pixel", item_context), f"{item_context}.pixel")
        if pixel >= IMAGE_PIXELS:
            raise ValueError(f"{item_context}.pixel must be less than {IMAGE_PIXELS}")
        if pixel in seen:
            raise ValueError(f"{context} has duplicate pixel {pixel}")
        seen.add(pixel)
        byte = expect_byte(json_field(item, "u8", item_context), f"{item_context}.u8")
        q31 = expect_i64(json_field(item, "q31", item_context), f"{item_context}.q31")
        if q31 != (byte * Q31_ONE) // 255:
            raise ValueError(f"{item_context}.q31 must match u8 scaled to Q31")
        if margin:
            predicted_weight = expect_i64(
                json_field(item, "predicted_weight", item_context),
                f"{item_context}.predicted_weight",
            )
            runner_up_weight = expect_i64(
                json_field(item, "runner_up_weight", item_context),
                f"{item_context}.runner_up_weight",
            )
            weight_delta = expect_i64(
                json_field(item, "weight_delta", item_context),
                f"{item_context}.weight_delta",
            )
            if weight_delta != wrap_i64(predicted_weight - runner_up_weight):
                raise ValueError(f"{item_context}.weight_delta must equal predicted - runner-up")
            contribution = expect_i64(
                json_field(item, "contribution", item_context),
                f"{item_context}.contribution",
            )
            expected_contribution = wrap_i64(
                fixed_mul_q31(q31, predicted_weight)
                - fixed_mul_q31(q31, runner_up_weight)
            )
            if contribution != expected_contribution:
                raise ValueError(f"{item_context}.contribution must equal predicted product - runner-up product")
        else:
            weight = expect_i64(json_field(item, "weight", item_context), f"{item_context}.weight")
            contribution = expect_i64(
                json_field(item, "contribution", item_context),
                f"{item_context}.contribution",
            )
            if contribution != fixed_mul_q31(q31, weight):
                raise ValueError(f"{item_context}.contribution must equal q31 * weight")
        rows.append({"pixel": pixel, "contribution": contribution})
    if rows != sorted(rows, key=contribution_sort_key):
        raise ValueError(f"{context} must be sorted by absolute contribution descending")
    return rows


def validate_attribution(
    value: Any,
    logits: list[int],
    top_logits: list[dict[str, int]],
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    predicted_digit = expect_digit(
        json_field(value, "predicted_digit", context),
        f"{context}.predicted_digit",
    )
    runner_up_digit = expect_digit(
        json_field(value, "runner_up_digit", context),
        f"{context}.runner_up_digit",
    )
    if predicted_digit != top_logits[0]["digit"]:
        raise ValueError(f"{context}.predicted_digit must match the top logit")
    expected_runner_up = top_logits[1]["digit"] if len(top_logits) > 1 else predicted_digit
    if runner_up_digit != expected_runner_up:
        raise ValueError(f"{context}.runner_up_digit must match the runner-up logit")

    predicted_logit = expect_i64(
        json_field(value, "predicted_logit", context),
        f"{context}.predicted_logit",
    )
    runner_up_logit = expect_i64(
        json_field(value, "runner_up_logit", context),
        f"{context}.runner_up_logit",
    )
    if predicted_logit != logits[predicted_digit]:
        raise ValueError(f"{context}.predicted_logit must match logits[predicted_digit]")
    if runner_up_logit != logits[runner_up_digit]:
        raise ValueError(f"{context}.runner_up_logit must match logits[runner_up_digit]")
    margin = expect_i64(json_field(value, "margin", context), f"{context}.margin")
    if margin != wrap_i64(predicted_logit - runner_up_logit):
        raise ValueError(f"{context}.margin must equal predicted_logit - runner_up_logit")

    bias = expect_i64(json_field(value, "bias", context), f"{context}.bias")
    runner_up_bias = expect_i64(
        json_field(value, "runner_up_bias", context),
        f"{context}.runner_up_bias",
    )
    margin_bias = expect_i64(
        json_field(value, "margin_bias", context),
        f"{context}.margin_bias",
    )
    if margin_bias != wrap_i64(bias - runner_up_bias):
        raise ValueError(f"{context}.margin_bias must equal bias - runner_up_bias")
    contribution_sum = expect_i64(
        json_field(value, "contribution_sum", context),
        f"{context}.contribution_sum",
    )
    margin_contribution_sum = expect_i64(
        json_field(value, "margin_contribution_sum", context),
        f"{context}.margin_contribution_sum",
    )
    reconstructed_logit = expect_i64(
        json_field(value, "reconstructed_logit", context),
        f"{context}.reconstructed_logit",
    )
    reconstructed_margin = expect_i64(
        json_field(value, "reconstructed_margin", context),
        f"{context}.reconstructed_margin",
    )
    if reconstructed_logit != wrap_i64(bias + contribution_sum):
        raise ValueError(f"{context}.reconstructed_logit must equal bias + contribution_sum")
    if reconstructed_margin != wrap_i64(margin_bias + margin_contribution_sum):
        raise ValueError(f"{context}.reconstructed_margin must equal margin_bias + margin contributions")
    matches_logit = expect_bool(
        json_field(value, "matches_logit", context),
        f"{context}.matches_logit",
    )
    matches_margin = expect_bool(
        json_field(value, "matches_margin", context),
        f"{context}.matches_margin",
    )
    if matches_logit != (reconstructed_logit == predicted_logit) or not matches_logit:
        raise ValueError(f"{context}.matches_logit must prove reconstructed_logit")
    if matches_margin != (reconstructed_margin == margin) or not matches_margin:
        raise ValueError(f"{context}.matches_margin must prove reconstructed_margin")
    contribution_count = expect_nonnegative_int(
        json_field(value, "contribution_count", context),
        f"{context}.contribution_count",
    )
    margin_contribution_count = expect_nonnegative_int(
        json_field(value, "margin_contribution_count", context),
        f"{context}.margin_contribution_count",
    )
    top_contribution_count = expect_nonnegative_int(
        json_field(value, "top_contribution_count", context),
        f"{context}.top_contribution_count",
    )
    top_margin_contribution_count = expect_nonnegative_int(
        json_field(value, "top_margin_contribution_count", context),
        f"{context}.top_margin_contribution_count",
    )
    if contribution_count > IMAGE_PIXELS or margin_contribution_count > IMAGE_PIXELS:
        raise ValueError(f"{context}.contribution_count must be no more than active pixel capacity")
    validate_sha256(
        json_field(value, "contribution_ledger_fingerprint", context),
        f"{context}.contribution_ledger_fingerprint",
    )
    validate_sha256(
        json_field(value, "margin_contribution_ledger_fingerprint", context),
        f"{context}.margin_contribution_ledger_fingerprint",
    )
    top_rows = validate_contribution_rows(
        json_field(value, "top_contributions", context),
        f"{context}.top_contributions",
        margin=False,
    )
    top_margin_rows = validate_contribution_rows(
        json_field(value, "top_margin_contributions", context),
        f"{context}.top_margin_contributions",
        margin=True,
    )
    if top_contribution_count != len(top_rows):
        raise ValueError(f"{context}.top_contribution_count must match top_contributions")
    if top_margin_contribution_count != len(top_margin_rows):
        raise ValueError(f"{context}.top_margin_contribution_count must match top_margin_contributions")
    if contribution_count < top_contribution_count:
        raise ValueError(f"{context}.contribution_count must cover top contributions")
    if margin_contribution_count < top_margin_contribution_count:
        raise ValueError(f"{context}.margin_contribution_count must cover top margin contributions")
    return {
        "contribution_count": contribution_count,
        "margin_contribution_count": margin_contribution_count,
        "contribution_ledger_fingerprint": json_field(value, "contribution_ledger_fingerprint", context),
        "margin_contribution_ledger_fingerprint": json_field(
            value,
            "margin_contribution_ledger_fingerprint",
            context,
        ),
    }


def validate_q31_reference_inference_report(report: dict[str, Any]) -> None:
    validate_nonempty_string(json_field(report, "model", "Q31 inference report"), "model")
    validate_nonempty_string(json_field(report, "sample", "Q31 inference report"), "sample")
    validate_q31_constant(json_field(report, "q31_one", "Q31 inference report"), "q31_one")
    prediction = expect_digit(json_field(report, "prediction", "Q31 inference report"), "prediction")
    label = expect_digit(json_field(report, "label", "Q31 inference report"), "label")
    correct = expect_bool(json_field(report, "correct", "Q31 inference report"), "correct")
    if correct != (prediction == label):
        raise ValueError("correct must match prediction == label")
    active_pixels = expect_nonnegative_int(
        json_field(report, "active_pixels", "Q31 inference report"),
        "active_pixels",
    )
    if active_pixels > IMAGE_PIXELS:
        raise ValueError(f"active_pixels must be no more than {IMAGE_PIXELS}")
    logits = validate_logits(json_field(report, "logits", "Q31 inference report"), "logits")
    top_logits = validate_top_logits(
        json_field(report, "top_logits", "Q31 inference report"),
        logits,
        prediction,
        "top_logits",
    )
    validate_attribution(
        json_field(report, "attribution", "Q31 inference report"),
        logits,
        top_logits,
        "attribution",
    )
    validate_sample_source(report.get("sample_source"), "sample_source")


def validate_evaluation_row(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    index = expect_nonnegative_int(json_field(value, "index", context), f"{context}.index")
    label = expect_digit(json_field(value, "label", context), f"{context}.label")
    prediction = expect_digit(json_field(value, "prediction", context), f"{context}.prediction")
    correct = expect_bool(json_field(value, "correct", context), f"{context}.correct")
    if correct != (prediction == label):
        raise ValueError(f"{context}.correct must match prediction == label")
    margin = expect_i64(json_field(value, "margin", context), f"{context}.margin")
    runner_up_digit = expect_digit(
        json_field(value, "runner_up_digit", context),
        f"{context}.runner_up_digit",
    )
    active_pixels = expect_nonnegative_int(
        json_field(value, "active_pixels", context),
        f"{context}.active_pixels",
    )
    if active_pixels > IMAGE_PIXELS:
        raise ValueError(f"{context}.active_pixels must be no more than {IMAGE_PIXELS}")
    validate_sample_source(value.get("sample_source"), f"{context}.sample_source")
    return {
        "index": index,
        "label": label,
        "prediction": prediction,
        "correct": correct,
        "margin": margin,
        "runner_up_digit": runner_up_digit,
        "active_pixels": active_pixels,
        **({"sample_source": value["sample_source"]} if "sample_source" in value else {}),
    }


def validate_accuracy(value: Any, correct: int, samples: int, context: str) -> None:
    actual = expect_finite_number(value, context)
    expected = (100.0 * correct / samples) if samples else 0.0
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{context} expected {expected}, found {actual}")


def validate_q31_reference_evaluation_report(report: dict[str, Any]) -> None:
    validate_nonempty_string(json_field(report, "model", "Q31 evaluation report"), "model")
    validate_nonempty_string(json_field(report, "sample", "Q31 evaluation report"), "sample")
    validate_q31_constant(json_field(report, "q31_one", "Q31 evaluation report"), "q31_one")
    report_limit = expect_nonnegative_int(
        json_field(report, "report_limit", "Q31 evaluation report"),
        "report_limit",
    )

    rows_value = json_field(report, "rows", "Q31 evaluation report")
    if not isinstance(rows_value, list) or not rows_value:
        raise ValueError("rows must be a non-empty array")
    rows = [
        validate_evaluation_row(row, f"rows[{index}]")
        for index, row in enumerate(rows_value)
    ]
    for expected_index, row in enumerate(rows):
        if row["index"] != expected_index:
            raise ValueError("rows index fields must match row order")

    summary = json_field(report, "summary", "Q31 evaluation report")
    samples = expect_positive_int(json_field(summary, "samples", "summary"), "summary.samples")
    if samples != len(rows):
        raise ValueError("summary.samples must match rows length")
    correct = expect_nonnegative_int(json_field(summary, "correct", "summary"), "summary.correct")
    incorrect = expect_nonnegative_int(
        json_field(summary, "incorrect", "summary"),
        "summary.incorrect",
    )
    observed_correct = sum(1 for row in rows if row["correct"])
    if correct != observed_correct:
        raise ValueError("summary.correct must match rows")
    if incorrect != samples - correct:
        raise ValueError("summary.incorrect must equal samples - correct")
    validate_accuracy(
        json_field(summary, "accuracy_percent", "summary"),
        correct,
        samples,
        "summary.accuracy_percent",
    )
    lowest = min(rows, key=lambda row: (row["margin"], row["index"]))
    lowest_margin_index = json_field(summary, "lowest_margin_index", "summary")
    lowest_margin = json_field(summary, "lowest_margin", "summary")
    if expect_nonnegative_int(lowest_margin_index, "summary.lowest_margin_index") != lowest["index"]:
        raise ValueError("summary.lowest_margin_index must identify the lowest margin row")
    if expect_i64(lowest_margin, "summary.lowest_margin") != lowest["margin"]:
        raise ValueError("summary.lowest_margin must match the lowest margin row")

    by_label = json_field(report, "by_label", "Q31 evaluation report")
    if not isinstance(by_label, list) or len(by_label) != DIGITS:
        raise ValueError(f"by_label must have {DIGITS} entries")
    for label, item in enumerate(by_label):
        item_context = f"by_label[{label}]"
        if expect_digit(json_field(item, "label", item_context), f"{item_context}.label") != label:
            raise ValueError(f"{item_context}.label must match its position")
        matching = [row for row in rows if row["label"] == label]
        label_samples = expect_nonnegative_int(
            json_field(item, "samples", item_context),
            f"{item_context}.samples",
        )
        label_correct = expect_nonnegative_int(
            json_field(item, "correct", item_context),
            f"{item_context}.correct",
        )
        label_incorrect = expect_nonnegative_int(
            json_field(item, "incorrect", item_context),
            f"{item_context}.incorrect",
        )
        observed_label_correct = sum(1 for row in matching if row["correct"])
        if label_samples != len(matching):
            raise ValueError(f"{item_context}.samples must match rows")
        if label_correct != observed_label_correct:
            raise ValueError(f"{item_context}.correct must match rows")
        if label_incorrect != label_samples - label_correct:
            raise ValueError(f"{item_context}.incorrect must equal samples - correct")
        validate_accuracy(
            json_field(item, "accuracy_percent", item_context),
            label_correct,
            label_samples,
            f"{item_context}.accuracy_percent",
        )
        label_lowest_index = json_field(item, "lowest_margin_index", item_context)
        label_lowest_margin = json_field(item, "lowest_margin", item_context)
        if matching:
            label_lowest = min(matching, key=lambda row: (row["margin"], row["index"]))
            if (
                expect_nonnegative_int(label_lowest_index, f"{item_context}.lowest_margin_index")
                != label_lowest["index"]
            ):
                raise ValueError(f"{item_context}.lowest_margin_index must identify the label's lowest row")
            if expect_i64(label_lowest_margin, f"{item_context}.lowest_margin") != label_lowest["margin"]:
                raise ValueError(f"{item_context}.lowest_margin must match the label's lowest row")
        elif label_lowest_index is not None or label_lowest_margin is not None:
            raise ValueError(f"{item_context}.lowest_margin fields must be null without samples")

    top_low_margin_value = json_field(report, "top_low_margin", "Q31 evaluation report")
    if not isinstance(top_low_margin_value, list) or len(top_low_margin_value) > report_limit:
        raise ValueError("top_low_margin must honor report_limit")
    top_low_margin = [
        validate_evaluation_row(row, f"top_low_margin[{index}]")
        for index, row in enumerate(top_low_margin_value)
    ]
    expected_low_margin = sorted(rows, key=lambda row: (row["margin"], row["index"]))[:report_limit]
    if top_low_margin != expected_low_margin:
        raise ValueError("top_low_margin must match rows sorted by margin")

    top_incorrect_value = json_field(report, "top_incorrect", "Q31 evaluation report")
    if not isinstance(top_incorrect_value, list) or len(top_incorrect_value) > report_limit:
        raise ValueError("top_incorrect must honor report_limit")
    top_incorrect = [
        validate_evaluation_row(row, f"top_incorrect[{index}]")
        for index, row in enumerate(top_incorrect_value)
    ]
    expected_incorrect = sorted(
        [row for row in rows if not row["correct"]],
        key=lambda row: (row["margin"], row["index"]),
    )[:report_limit]
    if top_incorrect != expected_incorrect:
        raise ValueError("top_incorrect must match incorrect rows sorted by margin")


def validate_audit_top_logits(value: Any, logits: list[int], context: str) -> list[dict[str, int]]:
    if not isinstance(value, list) or len(value) != min(3, DIGITS):
        raise ValueError(f"{context} must contain the top {min(3, DIGITS)} logits")
    expected = [
        {"digit": digit, "logit": logit}
        for digit, logit in enumerate(logits)
    ]
    expected.sort(key=lambda item: (-item["logit"], item["digit"]))
    expected = expected[: len(value)]
    checked: list[dict[str, int]] = []
    seen: set[int] = set()
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        digit = expect_digit(json_field(item, "digit", item_context), f"{item_context}.digit")
        logit = expect_i64(json_field(item, "logit", item_context), f"{item_context}.logit")
        if digit in seen:
            raise ValueError(f"{context} has duplicate digit {digit}")
        seen.add(digit)
        if logit != logits[digit]:
            raise ValueError(f"{item_context}.logit must match logits[{digit}]")
        checked.append({"digit": digit, "logit": logit})
    if checked != expected:
        raise ValueError(f"{context} must be sorted by logit descending and digit ascending")
    return checked


def validate_logit_margin(value: Any, top_logits: list[dict[str, int]], context: str) -> int:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    predicted_digit = expect_digit(
        json_field(value, "predicted_digit", context),
        f"{context}.predicted_digit",
    )
    predicted_logit = expect_i64(
        json_field(value, "predicted_logit", context),
        f"{context}.predicted_logit",
    )
    runner_up_digit = expect_digit(
        json_field(value, "runner_up_digit", context),
        f"{context}.runner_up_digit",
    )
    runner_up_logit = expect_i64(
        json_field(value, "runner_up_logit", context),
        f"{context}.runner_up_logit",
    )
    if predicted_digit != top_logits[0]["digit"] or predicted_logit != top_logits[0]["logit"]:
        raise ValueError(f"{context}.predicted_* must match top_logits[0]")
    expected_runner_up = top_logits[1] if len(top_logits) > 1 else top_logits[0]
    if runner_up_digit != expected_runner_up["digit"] or runner_up_logit != expected_runner_up["logit"]:
        raise ValueError(f"{context}.runner_up_* must match top_logits[1]")
    margin = expect_i64(json_field(value, "margin", context), f"{context}.margin")
    if margin != wrap_i64(predicted_logit - runner_up_logit):
        raise ValueError(f"{context}.margin must equal predicted_logit - runner_up_logit")
    return margin


def validate_active_pixel_rows(value: Any, context: str) -> int:
    if not isinstance(value, list) or len(value) > IMAGE_PIXELS:
        raise ValueError(f"{context} must be an array with at most {IMAGE_PIXELS} entries")
    previous = -1
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        pixel = expect_nonnegative_int(json_field(item, "index", item_context), f"{item_context}.index")
        if pixel >= IMAGE_PIXELS:
            raise ValueError(f"{item_context}.index must be less than {IMAGE_PIXELS}")
        if pixel <= previous:
            raise ValueError(f"{context} must be sorted by index with no duplicates")
        previous = pixel
        byte = expect_byte(json_field(item, "u8", item_context), f"{item_context}.u8")
        if byte == 0:
            raise ValueError(f"{item_context}.u8 must be non-zero")
        q31 = expect_i64(json_field(item, "q31", item_context), f"{item_context}.q31")
        if q31 != (byte * Q31_ONE) // 255:
            raise ValueError(f"{item_context}.q31 must match u8 scaled to Q31")
    return len(value)


def validate_weight_delta_rows(value: Any, context: str) -> list[dict[str, int]]:
    if not isinstance(value, list) or len(value) > 10:
        raise ValueError(f"{context} must be an array with at most 10 entries")
    rows = []
    seen: set[tuple[int, int]] = set()
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        pixel = expect_nonnegative_int(json_field(item, "pixel", item_context), f"{item_context}.pixel")
        if pixel >= IMAGE_PIXELS:
            raise ValueError(f"{item_context}.pixel must be less than {IMAGE_PIXELS}")
        digit = expect_digit(json_field(item, "digit", item_context), f"{item_context}.digit")
        delta = expect_i64(json_field(item, "delta", item_context), f"{item_context}.delta")
        if delta == 0:
            raise ValueError(f"{item_context}.delta must be non-zero")
        key = (pixel, digit)
        if key in seen:
            raise ValueError(f"{context} has duplicate pixel/digit pair")
        seen.add(key)
        rows.append({"pixel": pixel, "digit": digit, "delta": delta})
    expected = sorted(rows, key=lambda row: (-abs(row["delta"]), row["pixel"], row["digit"]))
    if rows != expected:
        raise ValueError(f"{context} must be sorted by absolute delta descending")
    return rows


def validate_update_block(value: Any, active_pixel_count: int, context: str) -> list[dict[str, int]]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    validate_nonempty_string(json_field(value, "formula", context), f"{context}.formula")
    bias_delta = validate_logits(json_field(value, "bias_delta", context), f"{context}.bias_delta")
    max_abs_bias_delta = max((abs(delta) for delta in bias_delta), default=0)
    nonzero_bias_delta_count = expect_nonnegative_int(
        json_field(value, "nonzero_bias_delta_count", context),
        f"{context}.nonzero_bias_delta_count",
    )
    expected_bias_delta_count = sum(1 for delta in bias_delta if delta != 0)
    if nonzero_bias_delta_count != expected_bias_delta_count:
        raise ValueError(f"{context}.nonzero_bias_delta_count must match bias_delta")
    bias_delta_ledger = validate_sha256(
        json_field(value, "bias_delta_ledger_fingerprint", context),
        f"{context}.bias_delta_ledger_fingerprint",
    )
    nonzero_weight_delta_count = expect_nonnegative_int(
        json_field(value, "nonzero_weight_delta_count", context),
        f"{context}.nonzero_weight_delta_count",
    )
    if nonzero_weight_delta_count > active_pixel_count * DIGITS:
        raise ValueError(f"{context}.nonzero_weight_delta_count exceeds active_pixels * digits")
    max_abs_weight_delta = expect_nonnegative_int(
        json_field(value, "max_abs_weight_delta", context),
        f"{context}.max_abs_weight_delta",
    )
    top_weight_deltas = validate_weight_delta_rows(
        json_field(value, "top_weight_deltas", context),
        f"{context}.top_weight_deltas",
    )
    if nonzero_weight_delta_count < len(top_weight_deltas):
        raise ValueError(f"{context}.nonzero_weight_delta_count must cover top_weight_deltas")
    if top_weight_deltas and max_abs_weight_delta < abs(top_weight_deltas[0]["delta"]):
        raise ValueError(f"{context}.max_abs_weight_delta must cover top_weight_deltas")
    if nonzero_weight_delta_count == 0 and max_abs_weight_delta != 0:
        raise ValueError(f"{context}.max_abs_weight_delta must be zero without weight deltas")
    if max_abs_bias_delta > I64_MAX:
        raise ValueError(f"{context}.bias_delta contains an invalid absolute value")
    weight_delta_ledger = validate_sha256(
        json_field(value, "weight_delta_ledger_fingerprint", context),
        f"{context}.weight_delta_ledger_fingerprint",
    )
    ledger_fingerprints = validate_exact_key_object(
        json_field(value, "ledger_fingerprints", context),
        ("algorithm", "bias_delta", "weight_delta"),
        f"{context}.ledger_fingerprints",
    )
    if json_field(ledger_fingerprints, "algorithm", f"{context}.ledger_fingerprints") != "sha256":
        raise ValueError(f"{context}.ledger_fingerprints.algorithm must be sha256")
    if validate_sha256(
        json_field(ledger_fingerprints, "bias_delta", f"{context}.ledger_fingerprints"),
        f"{context}.ledger_fingerprints.bias_delta",
    ) != bias_delta_ledger:
        raise ValueError(f"{context}.ledger_fingerprints.bias_delta must match bias_delta_ledger_fingerprint")
    if validate_sha256(
        json_field(ledger_fingerprints, "weight_delta", f"{context}.ledger_fingerprints"),
        f"{context}.ledger_fingerprints.weight_delta",
    ) != weight_delta_ledger:
            raise ValueError(f"{context}.ledger_fingerprints.weight_delta must match weight_delta_ledger_fingerprint")
    return top_weight_deltas


def validate_cause_ledger(value: Any, context: str) -> dict[str, Any]:
    ledger = validate_exact_key_object(
        value,
        ("algorithm", "schema", "fingerprint", "payload"),
        context,
    )
    if json_field(ledger, "algorithm", context) != "sha256":
        raise ValueError(f"{context}.algorithm must be sha256")
    if json_field(ledger, "schema", context) != TRAINING_CAUSE_LEDGER_SCHEMA:
        raise ValueError(f"{context}.schema must be {TRAINING_CAUSE_LEDGER_SCHEMA}")
    validate_sha256(json_field(ledger, "fingerprint", context), f"{context}.fingerprint")
    payload = validate_exact_key_object(
        json_field(ledger, "payload", context),
        (
            "schema",
            "step",
            "sample_index",
            "sample_fingerprint",
            "witness_fingerprint",
            "update_fingerprint",
            "label",
            "lr",
            "logits",
            "error",
            "prediction",
            "correct",
            "active_pixels",
            "top_logits",
            "logit_margin",
            "update_ledger_fingerprints",
            "top_weight_deltas",
        ),
        f"{context}.payload",
    )
    if json_field(payload, "schema", f"{context}.payload") != TRAINING_CAUSE_LEDGER_SCHEMA:
        raise ValueError(f"{context}.payload.schema must be {TRAINING_CAUSE_LEDGER_SCHEMA}")
    expect_nonnegative_int(json_field(payload, "step", f"{context}.payload"), f"{context}.payload.step")
    expect_nonnegative_int(
        json_field(payload, "sample_index", f"{context}.payload"),
        f"{context}.payload.sample_index",
    )
    for field in ("sample_fingerprint", "witness_fingerprint", "update_fingerprint"):
        validate_sha256(json_field(payload, field, f"{context}.payload"), f"{context}.payload.{field}")
    expect_digit(json_field(payload, "label", f"{context}.payload"), f"{context}.payload.label")
    expect_i64(json_field(payload, "lr", f"{context}.payload"), f"{context}.payload.lr")
    logits = validate_logits(json_field(payload, "logits", f"{context}.payload"), f"{context}.payload.logits")
    validate_logits(json_field(payload, "error", f"{context}.payload"), f"{context}.payload.error")
    expect_digit(json_field(payload, "prediction", f"{context}.payload"), f"{context}.payload.prediction")
    expect_bool(json_field(payload, "correct", f"{context}.payload"), f"{context}.payload.correct")
    active_pixel_count = validate_active_pixel_rows(
        json_field(payload, "active_pixels", f"{context}.payload"),
        f"{context}.payload.active_pixels",
    )
    top_logits = validate_audit_top_logits(
        json_field(payload, "top_logits", f"{context}.payload"),
        logits,
        f"{context}.payload.top_logits",
    )
    validate_logit_margin(
        json_field(payload, "logit_margin", f"{context}.payload"),
        top_logits,
        f"{context}.payload.logit_margin",
    )
    update_ledgers = validate_exact_key_object(
        json_field(payload, "update_ledger_fingerprints", f"{context}.payload"),
        ("algorithm", "bias_delta", "weight_delta"),
        f"{context}.payload.update_ledger_fingerprints",
    )
    if json_field(update_ledgers, "algorithm", f"{context}.payload.update_ledger_fingerprints") != "sha256":
        raise ValueError(f"{context}.payload.update_ledger_fingerprints.algorithm must be sha256")
    validate_sha256(
        json_field(update_ledgers, "bias_delta", f"{context}.payload.update_ledger_fingerprints"),
        f"{context}.payload.update_ledger_fingerprints.bias_delta",
    )
    validate_sha256(
        json_field(update_ledgers, "weight_delta", f"{context}.payload.update_ledger_fingerprints"),
        f"{context}.payload.update_ledger_fingerprints.weight_delta",
    )
    validate_update_rows = validate_weight_delta_rows(
        json_field(payload, "top_weight_deltas", f"{context}.payload"),
        f"{context}.payload.top_weight_deltas",
    )
    if active_pixel_count == 0 and validate_update_rows:
        raise ValueError(f"{context}.payload.top_weight_deltas require active_pixels")
    return payload


def validate_lineage_transition(value: Any, context: str, *, steps: int) -> dict[str, Any]:
    row = validate_exact_key_object(
        value,
        (
            "step",
            "sample_index",
            "label",
            "prediction",
            "correct",
            "lr",
            "sample_fingerprint",
            "witness_fingerprint",
            "update_fingerprint",
            "cause_ledger_fingerprint",
            "before_chain",
            "transition_fingerprint",
            "after_chain",
        ),
        context,
    )
    step = expect_nonnegative_int(json_field(row, "step", context), f"{context}.step")
    if step >= steps:
        raise ValueError(f"{context}.step must be less than lineage steps")
    expect_nonnegative_int(json_field(row, "sample_index", context), f"{context}.sample_index")
    expect_digit(json_field(row, "label", context), f"{context}.label")
    expect_digit(json_field(row, "prediction", context), f"{context}.prediction")
    expect_bool(json_field(row, "correct", context), f"{context}.correct")
    expect_i64(json_field(row, "lr", context), f"{context}.lr")
    for field in (
        "sample_fingerprint",
        "witness_fingerprint",
        "update_fingerprint",
        "cause_ledger_fingerprint",
        "before_chain",
        "transition_fingerprint",
        "after_chain",
    ):
        validate_sha256(json_field(row, field, context), f"{context}.{field}")
    return row


def validate_lineage_ledger(value: Any, context: str) -> dict[str, Any]:
    ledger = validate_exact_key_object(
        value,
        ("algorithm", "schema", "fingerprint", "payload"),
        context,
    )
    if json_field(ledger, "algorithm", context) != "sha256":
        raise ValueError(f"{context}.algorithm must be sha256")
    if json_field(ledger, "schema", context) != TRAINING_LINEAGE_LEDGER_SCHEMA:
        raise ValueError(f"{context}.schema must be {TRAINING_LINEAGE_LEDGER_SCHEMA}")
    validate_sha256(json_field(ledger, "fingerprint", context), f"{context}.fingerprint")
    payload = validate_exact_key_object(
        json_field(ledger, "payload", context),
        (
            "schema",
            "steps",
            "initial_model_fingerprint",
            "final_model_fingerprint",
            "witness_trace_fingerprint",
            "sample_order_fingerprint",
            "transition_ledger_fingerprint",
            "initial_chain",
            "final_chain",
            "first_transition",
            "last_transition",
        ),
        f"{context}.payload",
    )
    if json_field(payload, "schema", f"{context}.payload") != TRAINING_LINEAGE_LEDGER_SCHEMA:
        raise ValueError(f"{context}.payload.schema must be {TRAINING_LINEAGE_LEDGER_SCHEMA}")
    steps = expect_positive_int(json_field(payload, "steps", f"{context}.payload"), f"{context}.payload.steps")
    for field in (
        "initial_model_fingerprint",
        "final_model_fingerprint",
        "witness_trace_fingerprint",
        "sample_order_fingerprint",
        "transition_ledger_fingerprint",
        "initial_chain",
        "final_chain",
    ):
        validate_sha256(json_field(payload, field, f"{context}.payload"), f"{context}.payload.{field}")
    first = validate_lineage_transition(
        json_field(payload, "first_transition", f"{context}.payload"),
        f"{context}.payload.first_transition",
        steps=steps,
    )
    last = validate_lineage_transition(
        json_field(payload, "last_transition", f"{context}.payload"),
        f"{context}.payload.last_transition",
        steps=steps,
    )
    if first["step"] != 0:
        raise ValueError(f"{context}.payload.first_transition.step must be 0")
    if last["step"] != steps - 1:
        raise ValueError(f"{context}.payload.last_transition.step must be steps - 1")
    if first["before_chain"] != payload["initial_chain"]:
        raise ValueError(f"{context}.payload.first_transition.before_chain must match initial_chain")
    if last["after_chain"] != payload["final_chain"]:
        raise ValueError(f"{context}.payload.last_transition.after_chain must match final_chain")
    return payload


def validate_model_window(value: Any, update_rows: list[dict[str, int]], context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    reconstructed = expect_bool(
        json_field(value, "reconstructed", context),
        f"{context}.reconstructed",
    )
    if not reconstructed:
        raise ValueError(f"{context}.reconstructed must be true")
    expect_nonnegative_int(
        json_field(value, "reversed_later_steps", context),
        f"{context}.reversed_later_steps",
    )
    bias_before = validate_logits(json_field(value, "bias_before", context), f"{context}.bias_before")
    bias_after = validate_logits(json_field(value, "bias_after", context), f"{context}.bias_after")
    observed = validate_logits(
        json_field(value, "bias_observed_delta", context),
        f"{context}.bias_observed_delta",
    )
    expected_observed = [
        wrap_i64(after - before)
        for before, after in zip(bias_before, bias_after)
    ]
    if observed != expected_observed:
        raise ValueError(f"{context}.bias_observed_delta must equal bias_after - bias_before")
    expect_bool(
        json_field(value, "bias_delta_matches", context),
        f"{context}.bias_delta_matches",
    )
    weight_delta_matches = expect_bool(
        json_field(value, "weight_delta_matches", context),
        f"{context}.weight_delta_matches",
    )
    windows = json_field(value, "top_weight_windows", context)
    if not isinstance(windows, list) or len(windows) != len(update_rows):
        raise ValueError(f"{context}.top_weight_windows must match top_weight_deltas length")
    all_match = True
    for index, item in enumerate(windows):
        item_context = f"{context}.top_weight_windows[{index}]"
        update = update_rows[index]
        pixel = expect_nonnegative_int(json_field(item, "pixel", item_context), f"{item_context}.pixel")
        digit = expect_digit(json_field(item, "digit", item_context), f"{item_context}.digit")
        if pixel != update["pixel"] or digit != update["digit"]:
            raise ValueError(f"{item_context} must match top_weight_deltas[{index}]")
        before = expect_i64(json_field(item, "before", item_context), f"{item_context}.before")
        after = expect_i64(json_field(item, "after", item_context), f"{item_context}.after")
        observed_delta = expect_i64(
            json_field(item, "observed_delta", item_context),
            f"{item_context}.observed_delta",
        )
        computed_delta = expect_i64(
            json_field(item, "computed_delta", item_context),
            f"{item_context}.computed_delta",
        )
        delta_matches = expect_bool(
            json_field(item, "delta_matches", item_context),
            f"{item_context}.delta_matches",
        )
        if observed_delta != wrap_i64(after - before):
            raise ValueError(f"{item_context}.observed_delta must equal after - before")
        if computed_delta != update["delta"]:
            raise ValueError(f"{item_context}.computed_delta must match top_weight_deltas")
        if delta_matches != (observed_delta == computed_delta):
            raise ValueError(f"{item_context}.delta_matches must match observed/computed equality")
        all_match = all_match and delta_matches
    if weight_delta_matches != all_match:
        raise ValueError(f"{context}.weight_delta_matches must summarize top_weight_windows")


def validate_witness_checks(
    value: Any,
    logits: list[int],
    label: int,
    prediction: int,
    correct: bool,
    context: str,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    computed_prediction = expect_digit(
        json_field(value, "computed_prediction", context),
        f"{context}.computed_prediction",
    )
    expected_prediction = max(range(DIGITS), key=lambda digit: (logits[digit], -digit))
    if computed_prediction != expected_prediction:
        raise ValueError(f"{context}.computed_prediction must equal argmax(logits)")
    computed_correct = expect_bool(
        json_field(value, "computed_correct", context),
        f"{context}.computed_correct",
    )
    if computed_correct != (computed_prediction == label):
        raise ValueError(f"{context}.computed_correct must match computed_prediction == label")
    prediction_matches = expect_bool(
        json_field(value, "prediction_matches_logits", context),
        f"{context}.prediction_matches_logits",
    )
    correct_matches = expect_bool(
        json_field(value, "correct_matches_logits", context),
        f"{context}.correct_matches_logits",
    )
    if prediction_matches != (prediction == computed_prediction):
        raise ValueError(f"{context}.prediction_matches_logits must match prediction")
    if correct_matches != (correct == computed_correct):
        raise ValueError(f"{context}.correct_matches_logits must match correct")


def validate_audit_step_debug_contract(
    report: dict[str, Any],
    active_pixel_count: int,
    update_rows: list[dict[str, int]],
) -> None:
    contract = json_field(report, "debug_contract", "audit step report")
    if not isinstance(contract, dict):
        raise ValueError("debug_contract must be an object")
    claim = json_field(contract, "claim", "debug_contract")
    if claim != "step_backward_from_model_update":
        raise ValueError("debug_contract.claim must be `step_backward_from_model_update`")
    passed = expect_bool(json_field(contract, "passed", "debug_contract"), "debug_contract.passed")
    witness_checks = json_field(report, "witness_checks", "audit step report")
    model_window = json_field(report, "model_window", "audit step report")
    update = json_field(report, "update", "audit step report")
    expected_results = {
        "witness_recomputes_prediction": (
            bool(witness_checks["prediction_matches_logits"])
            and bool(witness_checks["correct_matches_logits"])
        ),
        "model_window_reconstructed": bool(model_window["reconstructed"]),
        "update_matches_model_window": (
            bool(model_window["bias_delta_matches"])
            and bool(model_window["weight_delta_matches"])
        ),
        "explanatory_state_present": (
            active_pixel_count > 0
            and int(update["nonzero_weight_delta_count"]) > 0
            and int(report["lr"]) != 0
            and bool(update_rows)
        ),
        "update_ledger_fingerprints": (
            isinstance(update.get("bias_delta_ledger_fingerprint"), str)
            and isinstance(update.get("weight_delta_ledger_fingerprint"), str)
            and len(update["bias_delta_ledger_fingerprint"]) == 64
            and len(update["weight_delta_ledger_fingerprint"]) == 64
        ),
    }
    checks = json_field(contract, "checks", "debug_contract")
    if not isinstance(checks, list):
        raise ValueError("debug_contract.checks must be an array")
    seen: dict[str, bool] = {}
    for index, check in enumerate(checks):
        context = f"debug_contract.checks[{index}]"
        metric = json_field(check, "metric", context)
        if not isinstance(metric, str):
            raise ValueError(f"{context}.metric must be a string")
        if metric in seen:
            raise ValueError(f"debug_contract has duplicate check `{metric}`")
        actual = json_field(check, "actual", context)
        if not isinstance(actual, str) or not actual:
            raise ValueError(f"{context}.actual must be a non-empty string")
        requirement = json_field(check, "requirement", context)
        if not isinstance(requirement, str) or not requirement:
            raise ValueError(f"{context}.requirement must be a non-empty string")
        check_passed = expect_bool(json_field(check, "passed", context), f"{context}.passed")
        if metric not in expected_results:
            raise ValueError(f"debug_contract has unknown check `{metric}`")
        if check_passed != expected_results[metric]:
            raise ValueError(f"debug_contract check `{metric}` does not match audit step")
        seen[metric] = check_passed
    missing = [metric for metric in expected_results if metric not in seen]
    if missing:
        raise ValueError("debug_contract missing check(s): " + ", ".join(missing))
    expected_passed = all(expected_results.values())
    if passed != expected_passed:
        raise ValueError("debug_contract.passed does not match audit step")
    replay_direction = json_field(contract, "replay_direction", "debug_contract")
    if not isinstance(replay_direction, dict):
        raise ValueError("debug_contract.replay_direction must be an object")
    if expect_nonnegative_int(
        json_field(replay_direction, "backward_from_final_model_steps", "debug_contract.replay_direction"),
        "debug_contract.replay_direction.backward_from_final_model_steps",
    ) != int(model_window["reversed_later_steps"]):
        raise ValueError("debug_contract replay direction must match model_window.reversed_later_steps")
    for field in ("selected_transition", "reverse_transition"):
        validate_nonempty_string(
            json_field(replay_direction, field, "debug_contract.replay_direction"),
            f"debug_contract.replay_direction.{field}",
        )
    debug_focus = json_field(contract, "debug_focus", "debug_contract")
    if not isinstance(debug_focus, dict):
        raise ValueError("debug_contract.debug_focus must be an object")
    for field in ("prediction", "update", "reversibility"):
        validate_nonempty_string(
            json_field(debug_focus, field, "debug_contract.debug_focus"),
            f"debug_contract.debug_focus.{field}",
        )


def validate_audit_step_report(report: dict[str, Any]) -> None:
    validate_nonempty_string(json_field(report, "path", "audit step report"), "path")
    validate_fingerprints(
        json_field(report, "fingerprints", "audit step report"),
        AUDIT_FINGERPRINT_FIELDS,
        "fingerprints",
    )
    step = expect_nonnegative_int(json_field(report, "step", "audit step report"), "step")
    sample_index = expect_nonnegative_int(
        json_field(report, "sample_index", "audit step report"),
        "sample_index",
    )
    label = expect_digit(json_field(report, "label", "audit step report"), "label")
    prediction = expect_digit(json_field(report, "prediction", "audit step report"), "prediction")
    correct = expect_bool(json_field(report, "correct", "audit step report"), "correct")
    expect_i64(json_field(report, "lr", "audit step report"), "lr")
    logits = validate_logits(json_field(report, "logits", "audit step report"), "logits")
    top_logits = validate_audit_top_logits(
        json_field(report, "top_logits", "audit step report"),
        logits,
        "top_logits",
    )
    validate_logit_margin(json_field(report, "logit_margin", "audit step report"), top_logits, "logit_margin")
    error = validate_logits(json_field(report, "error", "audit step report"), "error")
    if sum(1 for value in error if value != 0) == 0:
        raise ValueError("error must include at least one non-zero entry")
    validate_witness_checks(
        json_field(report, "witness_checks", "audit step report"),
        logits,
        label,
        prediction,
        correct,
        "witness_checks",
    )
    active_pixel_count = validate_active_pixel_rows(
        json_field(report, "active_pixels", "audit step report"),
        "active_pixels",
    )
    update_rows = validate_update_block(
        json_field(report, "update", "audit step report"),
        active_pixel_count,
        "update",
    )
    update = json_field(report, "update", "audit step report")
    cause_payload = validate_cause_ledger(
        json_field(report, "cause_ledger", "audit step report"),
        "cause_ledger",
    )
    if cause_payload["step"] != step:
        raise ValueError("cause_ledger.payload.step must match audit step")
    if cause_payload["sample_index"] != sample_index:
        raise ValueError("cause_ledger.payload.sample_index must match audit sample_index")
    if cause_payload["label"] != label:
        raise ValueError("cause_ledger.payload.label must match audit label")
    if cause_payload["lr"] != report["lr"]:
        raise ValueError("cause_ledger.payload.lr must match audit lr")
    if cause_payload["logits"] != logits:
        raise ValueError("cause_ledger.payload.logits must match audit logits")
    if cause_payload["error"] != error:
        raise ValueError("cause_ledger.payload.error must match audit error")
    if cause_payload["prediction"] != prediction:
        raise ValueError("cause_ledger.payload.prediction must match audit prediction")
    if cause_payload["correct"] != correct:
        raise ValueError("cause_ledger.payload.correct must match audit correctness")
    if cause_payload["active_pixels"] != report["active_pixels"]:
        raise ValueError("cause_ledger.payload.active_pixels must match audit active_pixels")
    if cause_payload["top_logits"] != top_logits:
        raise ValueError("cause_ledger.payload.top_logits must match audit top_logits")
    if cause_payload["logit_margin"] != report["logit_margin"]:
        raise ValueError("cause_ledger.payload.logit_margin must match audit logit_margin")
    if cause_payload["update_ledger_fingerprints"] != update["ledger_fingerprints"]:
        raise ValueError("cause_ledger.payload.update_ledger_fingerprints must match update")
    if cause_payload["top_weight_deltas"] != update["top_weight_deltas"]:
        raise ValueError("cause_ledger.payload.top_weight_deltas must match update")
    validate_model_window(json_field(report, "model_window", "audit step report"), update_rows, "model_window")
    validate_audit_step_debug_contract(report, active_pixel_count, update_rows)
    if "step_output" in report and report["step_output"] is not None:
        validate_nonempty_string(report["step_output"], "step_output")


def validate_audit_scan_row(value: Any, context: str, *, steps: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    step = expect_nonnegative_int(json_field(value, "step", context), f"{context}.step")
    if step >= steps:
        raise ValueError(f"{context}.step must be less than summary.steps")
    sample_index = expect_nonnegative_int(
        json_field(value, "sample_index", context),
        f"{context}.sample_index",
    )
    label = expect_digit(json_field(value, "label", context), f"{context}.label")
    prediction = expect_digit(json_field(value, "prediction", context), f"{context}.prediction")
    correct = expect_bool(json_field(value, "correct", context), f"{context}.correct")
    lr = expect_i64(json_field(value, "lr", context), f"{context}.lr")
    computed_prediction = expect_digit(
        json_field(value, "computed_prediction", context),
        f"{context}.computed_prediction",
    )
    predicted_logit = expect_i64(
        json_field(value, "predicted_logit", context),
        f"{context}.predicted_logit",
    )
    runner_up_digit = expect_digit(
        json_field(value, "runner_up_digit", context),
        f"{context}.runner_up_digit",
    )
    runner_up_logit = expect_i64(
        json_field(value, "runner_up_logit", context),
        f"{context}.runner_up_logit",
    )
    margin = expect_i64(json_field(value, "margin", context), f"{context}.margin")
    if margin != wrap_i64(predicted_logit - runner_up_logit):
        raise ValueError(f"{context}.margin must equal predicted_logit - runner_up_logit")
    prediction_matches = expect_bool(
        json_field(value, "prediction_matches_logits", context),
        f"{context}.prediction_matches_logits",
    )
    correct_matches = expect_bool(
        json_field(value, "correct_matches_logits", context),
        f"{context}.correct_matches_logits",
    )
    if prediction_matches != (prediction == computed_prediction):
        raise ValueError(f"{context}.prediction_matches_logits must match prediction")
    if correct_matches != (correct == (computed_prediction == label)):
        raise ValueError(f"{context}.correct_matches_logits must match correct")
    active_pixel_count = expect_nonnegative_int(
        json_field(value, "active_pixel_count", context),
        f"{context}.active_pixel_count",
    )
    if active_pixel_count > IMAGE_PIXELS:
        raise ValueError(f"{context}.active_pixel_count must be no more than {IMAGE_PIXELS}")
    error_nonzero_count = expect_nonnegative_int(
        json_field(value, "error_nonzero_count", context),
        f"{context}.error_nonzero_count",
    )
    if error_nonzero_count > DIGITS:
        raise ValueError(f"{context}.error_nonzero_count must be no more than {DIGITS}")
    max_abs_error = expect_nonnegative_int(
        json_field(value, "max_abs_error", context),
        f"{context}.max_abs_error",
    )
    max_abs_bias_delta = expect_nonnegative_int(
        json_field(value, "max_abs_bias_delta", context),
        f"{context}.max_abs_bias_delta",
    )
    nonzero_weight_delta_count = expect_nonnegative_int(
        json_field(value, "nonzero_weight_delta_count", context),
        f"{context}.nonzero_weight_delta_count",
    )
    if nonzero_weight_delta_count > active_pixel_count * DIGITS:
        raise ValueError(f"{context}.nonzero_weight_delta_count exceeds active pixels * digits")
    max_abs_weight_delta = expect_nonnegative_int(
        json_field(value, "max_abs_weight_delta", context),
        f"{context}.max_abs_weight_delta",
    )
    return {
        "step": step,
        "sample_index": sample_index,
        "label": label,
        "prediction": prediction,
        "correct": correct,
        "lr": lr,
        "computed_prediction": computed_prediction,
        "predicted_logit": predicted_logit,
        "runner_up_digit": runner_up_digit,
        "runner_up_logit": runner_up_logit,
        "margin": margin,
        "prediction_matches_logits": prediction_matches,
        "correct_matches_logits": correct_matches,
        "active_pixel_count": active_pixel_count,
        "error_nonzero_count": error_nonzero_count,
        "max_abs_error": max_abs_error,
        "max_abs_bias_delta": max_abs_bias_delta,
        "nonzero_weight_delta_count": nonzero_weight_delta_count,
        "max_abs_weight_delta": max_abs_weight_delta,
    }


def validate_audit_scan_rows(value: Any, context: str, *, steps: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    seen: set[int] = set()
    rows = []
    for index, item in enumerate(value):
        row = validate_audit_scan_row(item, f"{context}[{index}]", steps=steps)
        if row["step"] in seen:
            raise ValueError(f"{context} has duplicate step {row['step']}")
        seen.add(row["step"])
        rows.append(row)
    return rows


def validate_audit_scan_order(rows: list[dict[str, Any]], context: str, order: str) -> None:
    if order == "suspicious":
        expected = sorted(
            rows,
            key=lambda row: (
                row["correct"],
                -row["max_abs_weight_delta"],
                -row["max_abs_bias_delta"],
                -row["max_abs_error"],
                row["step"],
            ),
        )
    elif order == "low_margin":
        expected = sorted(
            rows,
            key=lambda row: (
                row["margin"],
                row["correct"],
                -row["max_abs_weight_delta"],
                row["step"],
            ),
        )
    elif order == "large_update":
        expected = sorted(
            rows,
            key=lambda row: (
                -row["max_abs_weight_delta"],
                -row["max_abs_bias_delta"],
                -row["max_abs_error"],
                row["step"],
            ),
        )
    else:
        raise AssertionError(f"unknown audit scan order {order}")
    if rows != expected:
        raise ValueError(f"{context} is not sorted as {order}")


def validate_audit_scan_report(report: dict[str, Any]) -> None:
    validate_nonempty_string(json_field(report, "path", "audit scan report"), "path")
    validate_fingerprints(
        json_field(report, "fingerprints", "audit scan report"),
        AUDIT_FINGERPRINT_FIELDS,
        "fingerprints",
    )
    summary = json_field(report, "summary", "audit scan report")
    steps = expect_positive_int(json_field(summary, "steps", "summary"), "summary.steps")
    correct = expect_nonnegative_int(json_field(summary, "correct", "summary"), "summary.correct")
    incorrect = expect_nonnegative_int(
        json_field(summary, "incorrect", "summary"),
        "summary.incorrect",
    )
    if correct + incorrect != steps:
        raise ValueError("summary.correct + summary.incorrect must equal summary.steps")
    validate_accuracy(json_field(summary, "accuracy_percent", "summary"), correct, steps, "summary.accuracy_percent")
    witness_mismatches = expect_nonnegative_int(
        json_field(summary, "witness_mismatches", "summary"),
        "summary.witness_mismatches",
    )
    if witness_mismatches > steps:
        raise ValueError("summary.witness_mismatches cannot exceed summary.steps")
    largest_update_step = json_field(summary, "largest_update_step", "summary")
    if largest_update_step is not None and expect_nonnegative_int(largest_update_step, "summary.largest_update_step") >= steps:
        raise ValueError("summary.largest_update_step must be less than summary.steps")
    expect_nonnegative_int(
        json_field(summary, "max_abs_weight_delta", "summary"),
        "summary.max_abs_weight_delta",
    )
    lowest_margin_step = json_field(summary, "lowest_margin_step", "summary")
    if lowest_margin_step is None:
        raise ValueError("summary.lowest_margin_step must be present when steps are positive")
    if expect_nonnegative_int(lowest_margin_step, "summary.lowest_margin_step") >= steps:
        raise ValueError("summary.lowest_margin_step must be less than summary.steps")
    expect_i64(json_field(summary, "lowest_margin", "summary"), "summary.lowest_margin")
    trace_payload_bytes = expect_nonnegative_int(
        json_field(summary, "trace_payload_bytes", "summary"),
        "summary.trace_payload_bytes",
    )
    bytes_per_step = expect_positive_int(
        json_field(summary, "bytes_per_step", "summary"),
        "summary.bytes_per_step",
    )
    if trace_payload_bytes != steps * bytes_per_step:
        raise ValueError("summary.trace_payload_bytes must equal steps * bytes_per_step")

    by_label = json_field(report, "by_label", "audit scan report")
    if not isinstance(by_label, list) or len(by_label) != DIGITS:
        raise ValueError(f"by_label must have {DIGITS} entries")
    label_sample_sum = 0
    label_correct_sum = 0
    label_incorrect_sum = 0
    for label, item in enumerate(by_label):
        item_context = f"by_label[{label}]"
        if expect_digit(json_field(item, "label", item_context), f"{item_context}.label") != label:
            raise ValueError(f"{item_context}.label must match its position")
        label_samples = expect_nonnegative_int(
            json_field(item, "samples", item_context),
            f"{item_context}.samples",
        )
        label_correct = expect_nonnegative_int(
            json_field(item, "correct", item_context),
            f"{item_context}.correct",
        )
        label_incorrect = expect_nonnegative_int(
            json_field(item, "incorrect", item_context),
            f"{item_context}.incorrect",
        )
        if label_correct + label_incorrect != label_samples:
            raise ValueError(f"{item_context}.correct + incorrect must equal samples")
        validate_accuracy(
            json_field(item, "accuracy_percent", item_context),
            label_correct,
            label_samples,
            f"{item_context}.accuracy_percent",
        )
        min_margin_step = json_field(item, "min_margin_step", item_context)
        min_margin = json_field(item, "min_margin", item_context)
        largest_step = json_field(item, "largest_update_step", item_context)
        max_delta = expect_nonnegative_int(
            json_field(item, "max_abs_weight_delta", item_context),
            f"{item_context}.max_abs_weight_delta",
        )
        if label_samples:
            if min_margin_step is None or min_margin is None:
                raise ValueError(f"{item_context}.min_margin fields are required with samples")
            if expect_nonnegative_int(min_margin_step, f"{item_context}.min_margin_step") >= steps:
                raise ValueError(f"{item_context}.min_margin_step must be less than summary.steps")
            expect_i64(min_margin, f"{item_context}.min_margin")
            if largest_step is None:
                raise ValueError(f"{item_context}.largest_update_step is required with samples")
            if expect_nonnegative_int(largest_step, f"{item_context}.largest_update_step") >= steps:
                raise ValueError(f"{item_context}.largest_update_step must be less than summary.steps")
        elif min_margin_step is not None or min_margin is not None or largest_step is not None or max_delta != 0:
            raise ValueError(f"{item_context} empty labels must have null steps/margins and zero delta")
        label_sample_sum += label_samples
        label_correct_sum += label_correct
        label_incorrect_sum += label_incorrect
    if (label_sample_sum, label_correct_sum, label_incorrect_sum) != (steps, correct, incorrect):
        raise ValueError("by_label totals must match summary")

    top_suspicious = validate_audit_scan_rows(
        json_field(report, "top_suspicious", "audit scan report"),
        "top_suspicious",
        steps=steps,
    )
    validate_audit_scan_order(top_suspicious, "top_suspicious", "suspicious")
    top_low_margin = validate_audit_scan_rows(
        json_field(report, "top_low_margin", "audit scan report"),
        "top_low_margin",
        steps=steps,
    )
    validate_audit_scan_order(top_low_margin, "top_low_margin", "low_margin")
    top_large_updates = validate_audit_scan_rows(
        json_field(report, "top_large_updates", "audit scan report"),
        "top_large_updates",
        steps=steps,
    )
    validate_audit_scan_order(top_large_updates, "top_large_updates", "large_update")

    top_confusions = json_field(report, "top_confusions", "audit scan report")
    if not isinstance(top_confusions, list):
        raise ValueError("top_confusions must be an array")
    for index, item in enumerate(top_confusions):
        item_context = f"top_confusions[{index}]"
        label = expect_digit(json_field(item, "label", item_context), f"{item_context}.label")
        prediction = expect_digit(
            json_field(item, "prediction", item_context),
            f"{item_context}.prediction",
        )
        if label == prediction:
            raise ValueError(f"{item_context} must describe a misclassification")
        expect_positive_int(json_field(item, "count", item_context), f"{item_context}.count")
        if expect_nonnegative_int(json_field(item, "first_step", item_context), f"{item_context}.first_step") >= steps:
            raise ValueError(f"{item_context}.first_step must be less than summary.steps")
        expect_i64(json_field(item, "min_margin", item_context), f"{item_context}.min_margin")
        expect_nonnegative_int(
            json_field(item, "max_abs_weight_delta", item_context),
            f"{item_context}.max_abs_weight_delta",
        )

    gate = report.get("gate")
    if gate is not None:
        if not isinstance(gate, dict):
            raise ValueError("gate must be an object")
        gate_passed = expect_bool(json_field(gate, "passed", "gate"), "gate.passed")
        checks = json_field(gate, "checks", "gate")
        if not isinstance(checks, list) or not checks:
            raise ValueError("gate.checks must be a non-empty array")
        check_results = []
        for index, check in enumerate(checks):
            check_context = f"gate.checks[{index}]"
            validate_nonempty_string(json_field(check, "metric", check_context), f"{check_context}.metric")
            check_results.append(expect_bool(json_field(check, "passed", check_context), f"{check_context}.passed"))
            validate_nonempty_string(json_field(check, "actual", check_context), f"{check_context}.actual")
            validate_nonempty_string(
                json_field(check, "expectation", check_context),
                f"{check_context}.expectation",
            )
        if gate_passed != all(check_results):
            raise ValueError("gate.passed must match gate checks")


def validate_training_trace_proof(value: Any, entries: int, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if not expect_bool(json_field(value, "enabled", context), f"{context}.enabled"):
        raise ValueError(f"{context}.enabled must be true")
    proof_entries = expect_nonnegative_int(json_field(value, "entries", context), f"{context}.entries")
    if proof_entries != entries:
        raise ValueError(f"{context}.entries must match checked steps")
    model_payload_bytes = expect_positive_int(
        json_field(value, "model_payload_bytes", context),
        f"{context}.model_payload_bytes",
    )
    sample_payload_bytes = expect_nonnegative_int(
        json_field(value, "sample_payload_bytes", context),
        f"{context}.sample_payload_bytes",
    )
    witness_payload_bytes = expect_nonnegative_int(
        json_field(value, "witness_payload_bytes", context),
        f"{context}.witness_payload_bytes",
    )
    trace_replay_payload_bytes = expect_nonnegative_int(
        json_field(value, "trace_replay_payload_bytes", context),
        f"{context}.trace_replay_payload_bytes",
    )
    full_replay_payload_bytes = expect_positive_int(
        json_field(value, "full_replay_payload_bytes", context),
        f"{context}.full_replay_payload_bytes",
    )
    sample_bytes_per_step = expect_nonnegative_int(
        json_field(value, "sample_bytes_per_step", context),
        f"{context}.sample_bytes_per_step",
    )
    witness_bytes_per_step = expect_nonnegative_int(
        json_field(value, "witness_bytes_per_step", context),
        f"{context}.witness_bytes_per_step",
    )
    trace_replay_bytes_per_step = expect_nonnegative_int(
        json_field(value, "trace_replay_bytes_per_step", context),
        f"{context}.trace_replay_bytes_per_step",
    )
    if sample_payload_bytes != entries * sample_bytes_per_step:
        raise ValueError(f"{context}.sample_payload_bytes must equal entries * sample_bytes_per_step")
    if witness_payload_bytes != entries * witness_bytes_per_step:
        raise ValueError(f"{context}.witness_payload_bytes must equal entries * witness_bytes_per_step")
    if trace_replay_bytes_per_step != sample_bytes_per_step + witness_bytes_per_step:
        raise ValueError(f"{context}.trace_replay_bytes_per_step must equal sample + witness bytes per step")
    if trace_replay_payload_bytes != entries * trace_replay_bytes_per_step:
        raise ValueError(f"{context}.trace_replay_payload_bytes must equal entries * trace replay bytes per step")
    if full_replay_payload_bytes != model_payload_bytes + trace_replay_payload_bytes:
        raise ValueError(f"{context}.full_replay_payload_bytes must equal model + trace replay bytes")
    if expect_nonnegative_int(
        json_field(value, "forward_recompute_steps", context),
        f"{context}.forward_recompute_steps",
    ) != entries:
        raise ValueError(f"{context}.forward_recompute_steps must equal entries")
    if expect_nonnegative_int(
        json_field(value, "inverse_recompute_steps", context),
        f"{context}.inverse_recompute_steps",
    ) != entries:
        raise ValueError(f"{context}.inverse_recompute_steps must equal entries")


def validate_replay_timing(value: Any, context: str, *, checked: int, flag: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if expect_nonnegative_int(json_field(value, "checked", context), f"{context}.checked") != checked:
        raise ValueError(f"{context}.checked must match audit verification checked")
    if not expect_bool(json_field(value, flag, context), f"{context}.{flag}"):
        raise ValueError(f"{context}.{flag} must be true")
    elapsed = expect_finite_number(json_field(value, "elapsed_seconds", context), f"{context}.elapsed_seconds")
    if elapsed < 0.0:
        raise ValueError(f"{context}.elapsed_seconds must be non-negative")
    rate = expect_finite_number(json_field(value, "steps_per_second", context), f"{context}.steps_per_second")
    if rate < 0.0:
        raise ValueError(f"{context}.steps_per_second must be non-negative")


def validate_audit_verification_report(report: dict[str, Any]) -> None:
    validate_nonempty_string(json_field(report, "path", "audit verification report"), "path")
    validate_fingerprints(
        json_field(report, "fingerprints", "audit verification report"),
        AUDIT_FINGERPRINT_FIELDS,
        "fingerprints",
    )
    checked = expect_positive_int(json_field(report, "checked", "audit verification report"), "checked")
    for field in (
        "witnesses_match_forward_replay",
        "final_model_replayed",
        "restored_initial_model",
        "proof_matches",
        "lineage_ledger_matches",
    ):
        if not expect_bool(json_field(report, field, "audit verification report"), field):
            raise ValueError(f"{field} must be true")
    validate_training_trace_proof(
        json_field(report, "proof", "audit verification report"),
        checked,
        "proof",
    )
    lineage_payload = validate_lineage_ledger(
        json_field(report, "lineage_ledger", "audit verification report"),
        "lineage_ledger",
    )
    if lineage_payload["steps"] != checked:
        raise ValueError("lineage_ledger.payload.steps must match checked")
    validate_replay_timing(
        json_field(report, "forward", "audit verification report"),
        "forward",
        checked=checked,
        flag="witnesses_match",
    )
    validate_replay_timing(
        json_field(report, "reverse", "audit verification report"),
        "reverse",
        checked=checked,
        flag="restored_initial_model",
    )


def validate_step_proof(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    claim = json_field(value, "claim", context)
    if claim != TRAINING_STEP_PROOF_CLAIM:
        raise ValueError(f"{context}.claim must be `{TRAINING_STEP_PROOF_CLAIM}`")
    kernel = json_field(value, "kernel", context)
    if kernel != "examples/mnist_reversible_step.rev":
        raise ValueError(f"{context}.kernel must be examples/mnist_reversible_step.rev")
    arithmetic = json_field(value, "arithmetic", context)
    if arithmetic != "q31_wrapping_i64":
        raise ValueError(f"{context}.arithmetic must be q31_wrapping_i64")
    witnesses = json_field(value, "witnesses", context)
    if witnesses != ["logits", "error", "prediction", "correct", "lr"]:
        raise ValueError(f"{context}.witnesses must list the training-step witnesses")
    model_payload_bytes = expect_positive_int(
        json_field(value, "model_payload_bytes", context),
        f"{context}.model_payload_bytes",
    )
    sample_payload_bytes = expect_positive_int(
        json_field(value, "sample_payload_bytes", context),
        f"{context}.sample_payload_bytes",
    )
    witness_payload_bytes = expect_positive_int(
        json_field(value, "witness_payload_bytes", context),
        f"{context}.witness_payload_bytes",
    )
    derived_update_payload_bytes = expect_positive_int(
        json_field(value, "derived_update_payload_bytes", context),
        f"{context}.derived_update_payload_bytes",
    )
    expect_positive_int(
        json_field(value, "nonzero_bias_delta_count", context),
        f"{context}.nonzero_bias_delta_count",
    )
    expect_positive_int(
        json_field(value, "nonzero_weight_delta_count", context),
        f"{context}.nonzero_weight_delta_count",
    )
    update_ledgers = validate_exact_key_object(
        json_field(value, "update_ledger_fingerprints", context),
        ("algorithm", "bias_delta", "weight_delta"),
        f"{context}.update_ledger_fingerprints",
    )
    if json_field(update_ledgers, "algorithm", f"{context}.update_ledger_fingerprints") != "sha256":
        raise ValueError(f"{context}.update_ledger_fingerprints.algorithm must be sha256")
    validate_sha256(
        json_field(update_ledgers, "bias_delta", f"{context}.update_ledger_fingerprints"),
        f"{context}.update_ledger_fingerprints.bias_delta",
    )
    validate_sha256(
        json_field(update_ledgers, "weight_delta", f"{context}.update_ledger_fingerprints"),
        f"{context}.update_ledger_fingerprints.weight_delta",
    )
    cause_ledger = json_field(value, "cause_ledger", context)
    cause_payload = validate_cause_ledger(cause_ledger, f"{context}.cause_ledger")
    if cause_payload["update_ledger_fingerprints"] != update_ledgers:
        raise ValueError(f"{context}.cause_ledger.payload.update_ledger_fingerprints must match proof")
    trace_payload_bytes = expect_nonnegative_int(
        json_field(value, "trace_payload_bytes", context),
        f"{context}.trace_payload_bytes",
    )
    if trace_payload_bytes != 0:
        raise ValueError(f"{context}.trace_payload_bytes must be zero for standalone step replay")
    replay_payload_bytes = expect_positive_int(
        json_field(value, "replay_payload_bytes", context),
        f"{context}.replay_payload_bytes",
    )
    if replay_payload_bytes != (
        model_payload_bytes + sample_payload_bytes + witness_payload_bytes + derived_update_payload_bytes
    ):
        raise ValueError(f"{context}.replay_payload_bytes must equal model + sample + witness + update")
    if expect_nonnegative_int(
        json_field(value, "forward_recompute_steps", context),
        f"{context}.forward_recompute_steps",
    ) != 1:
        raise ValueError(f"{context}.forward_recompute_steps must be 1")
    if expect_nonnegative_int(
        json_field(value, "inverse_recompute_steps", context),
        f"{context}.inverse_recompute_steps",
    ) != 1:
        raise ValueError(f"{context}.inverse_recompute_steps must be 1")
    checks = json_field(value, "checks", context)
    if not isinstance(checks, dict):
        raise ValueError(f"{context}.checks must be an object")
    for field in (
        "witnesses_match_forward_replay",
        "after_model_matches",
        "before_model_restored",
        "update_matches_witnesses",
    ):
        if not expect_bool(json_field(checks, field, f"{context}.checks"), f"{context}.checks.{field}"):
            raise ValueError(f"{context}.checks.{field} must be true")
    validate_fingerprints(
        json_field(value, "fingerprints", context),
        (
            "train_source",
            "before_model",
            "after_model",
            "sample",
            "witness",
            "update",
            "cause_ledger",
        ),
        f"{context}.fingerprints",
    )
    if json_field(value["fingerprints"], "cause_ledger", f"{context}.fingerprints") != json_field(
        cause_ledger,
        "fingerprint",
        f"{context}.cause_ledger",
    ):
        raise ValueError(f"{context}.fingerprints.cause_ledger must match cause_ledger.fingerprint")


def validate_step_verification_report(report: dict[str, Any]) -> None:
    validate_nonempty_string(json_field(report, "path", "step verification report"), "path")
    validate_fingerprints(
        json_field(report, "fingerprints", "step verification report"),
        (
            "train_source",
            "computation",
            "before_model",
            "after_model",
            "sample",
            "witness",
            "update",
            "proof",
            "payload",
        ),
        "fingerprints",
    )
    if not expect_bool(json_field(report, "proof_matches", "step verification report"), "proof_matches"):
        raise ValueError("proof_matches must be true")
    validate_step_proof(json_field(report, "proof", "step verification report"), "proof")
    forward = json_field(report, "forward", "step verification report")
    if not expect_bool(json_field(forward, "witnesses_match", "forward"), "forward.witnesses_match"):
        raise ValueError("forward.witnesses_match must be true")
    if not expect_bool(json_field(forward, "after_model_matches", "forward"), "forward.after_model_matches"):
        raise ValueError("forward.after_model_matches must be true")
    if expect_finite_number(json_field(forward, "elapsed_seconds", "forward"), "forward.elapsed_seconds") < 0.0:
        raise ValueError("forward.elapsed_seconds must be non-negative")
    reverse = json_field(report, "reverse", "step verification report")
    if not expect_bool(json_field(reverse, "before_model_restored", "reverse"), "reverse.before_model_restored"):
        raise ValueError("reverse.before_model_restored must be true")
    if expect_finite_number(json_field(reverse, "elapsed_seconds", "reverse"), "reverse.elapsed_seconds") < 0.0:
        raise ValueError("reverse.elapsed_seconds must be non-negative")


def validate_model_fingerprints(value: Any, context: str) -> None:
    validate_fingerprints(
        value,
        (
            "train_source",
            "identify_source",
            "computation",
            "model",
            "provenance",
            "report",
            "payload",
        ),
        context,
    )


def validate_inference_fingerprints(value: Any, context: str) -> None:
    validate_fingerprints(
        value,
        (
            "identify_source",
            "computation",
            "model",
            "sample",
            "result",
            "proof",
            "report",
            "payload",
        ),
        context,
    )


def validate_model_evaluation_fingerprints(value: Any, context: str) -> None:
    validate_fingerprints(
        value,
        (
            "identify_source",
            "computation",
            "model",
            "samples",
            "summary",
            "proof",
            "rows",
            "gate_policy",
            "gate",
            "report",
            "payload",
        ),
        context,
    )


def validate_model_bundle_ref(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    validate_nonempty_string(json_field(value, "path", context), f"{context}.path")
    validate_model_fingerprints(json_field(value, "fingerprints", context), f"{context}.fingerprints")


def validate_imported_model_source_check(value: Any, context: str) -> None:
    if value is None:
        raise ValueError(f"{context} must be an object")
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    validate_nonempty_string(json_field(value, "path", context), f"{context}.path")
    validate_nonempty_string(json_field(value, "resolved_path", context), f"{context}.resolved_path")
    checked = expect_bool(json_field(value, "checked", context), f"{context}.checked")
    validate_sha256(json_field(value, "fingerprint", context), f"{context}.fingerprint")
    validate_sha256(json_field(value, "file_sha256", context), f"{context}.file_sha256")
    if not checked:
        validate_nonempty_string(
            json_field(value, "unavailable_reason", context),
            f"{context}.unavailable_reason",
        )


def validate_model_import_report(report: dict[str, Any]) -> None:
    validate_nonempty_string(
        json_field(report, "source_model_json_path", "model import"),
        "source_model_json_path",
    )
    validate_nonempty_string(
        json_field(report, "model_output", "model import"),
        "model_output",
    )
    validate_sha256(
        json_field(report, "source_model_json_fingerprint", "model import"),
        "source_model_json_fingerprint",
    )
    validate_sha256(
        json_field(report, "source_model_json_file_sha256", "model import"),
        "source_model_json_file_sha256",
    )
    validate_model_fingerprints(json_field(report, "fingerprints", "model import"), "fingerprints")
    if not expect_bool(json_field(report, "shape_matches", "model import"), "shape_matches"):
        raise ValueError("model import shape_matches must be true")
    if json_field(report, "provenance_kind", "model import") != "external_import":
        raise ValueError("model import provenance_kind must be external_import")
    storage = json_field(report, "storage", "model import")
    if not isinstance(storage, dict):
        raise ValueError("model import storage must be an object")
    expect_positive_int(
        json_field(storage, "model_payload_bytes", "model import storage"),
        "storage.model_payload_bytes",
    )


def validate_model_verification_report(report: dict[str, Any]) -> None:
    validate_nonempty_string(json_field(report, "path", "model verification"), "path")
    validate_model_fingerprints(
        json_field(report, "fingerprints", "model verification"),
        "fingerprints",
    )
    expect_positive_int(
        json_field(report, "model_payload_bytes", "model verification"),
        "model_payload_bytes",
    )
    for field in ("shape_matches", "provenance_matches"):
        if not expect_bool(json_field(report, field, "model verification"), field):
            raise ValueError(f"model verification {field} must be true")
    provenance_kind = validate_nonempty_string(
        json_field(report, "provenance_kind", "model verification"),
        "provenance_kind",
    )
    if provenance_kind not in ("training_audit", "external_import"):
        raise ValueError("model verification provenance_kind is unsupported")
    proof_matches = json_field(report, "proof_matches", "model verification")
    training_steps = json_field(report, "training_steps", "model verification")
    if provenance_kind == "external_import":
        if proof_matches is not None:
            raise ValueError("external imported model proof_matches must be null")
        if training_steps is not None:
            raise ValueError("external imported model training_steps must be null")
        if not expect_bool(
            json_field(report, "source_model_json_checked", "model verification"),
            "source_model_json_checked",
        ):
            raise ValueError("external imported model source JSON must be checked")
        validate_imported_model_source_check(
            json_field(report, "source_model_json", "model verification"),
            "source_model_json",
        )
    else:
        if proof_matches is not True:
            raise ValueError("training model proof_matches must be true")
        expect_positive_int(training_steps, "training_steps")
        validate_sha256(
            json_field(report, "source_audit_payload", "model verification"),
            "source_audit_payload",
        )


def validate_model_evaluation_bundle_ref(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    validate_nonempty_string(json_field(value, "path", context), f"{context}.path")
    validate_model_evaluation_fingerprints(json_field(value, "fingerprints", context), f"{context}.fingerprints")


def validate_optional_source_check(value: Any, context: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object or null")
    validate_nonempty_string(json_field(value, "path", context), f"{context}.path")
    if "resolved_path" in value:
        validate_nonempty_string(value["resolved_path"], f"{context}.resolved_path")
    checked = expect_bool(json_field(value, "checked", context), f"{context}.checked")
    validate_sha256(json_field(value, "payload_fingerprint", context), f"{context}.payload_fingerprint")
    items_checked = expect_nonnegative_int(
        json_field(value, "items_checked", context),
        f"{context}.items_checked",
    )
    if checked and items_checked == 0:
        raise ValueError(f"{context}.items_checked must be positive when checked")
    if not checked and "unavailable_reason" in value:
        validate_nonempty_string(value["unavailable_reason"], f"{context}.unavailable_reason")


def validate_optional_source_evaluation_check(value: Any, context: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object or null")
    validate_nonempty_string(json_field(value, "path", context), f"{context}.path")
    if "resolved_path" in value:
        validate_nonempty_string(value["resolved_path"], f"{context}.resolved_path")
    checked = expect_bool(json_field(value, "checked", context), f"{context}.checked")
    validate_sha256(json_field(value, "payload_fingerprint", context), f"{context}.payload_fingerprint")
    if checked:
        expect_nonnegative_int(json_field(value, "row_index", context), f"{context}.row_index")
        validate_sha256(json_field(value, "sample_fingerprint", context), f"{context}.sample_fingerprint")
    elif "unavailable_reason" in value:
        validate_nonempty_string(value["unavailable_reason"], f"{context}.unavailable_reason")


def validate_inference_memory(value: Any, context: str, *, expected_steps: int = 1) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    model_payload_bytes = expect_positive_int(
        json_field(value, "model_payload_bytes", context),
        f"{context}.model_payload_bytes",
    )
    sample_payload_bytes = expect_positive_int(
        json_field(value, "sample_payload_bytes", context),
        f"{context}.sample_payload_bytes",
    )
    witness_payload_bytes = expect_positive_int(
        json_field(value, "witness_payload_bytes", context),
        f"{context}.witness_payload_bytes",
    )
    trace_payload_bytes = expect_nonnegative_int(
        json_field(value, "trace_payload_bytes", context),
        f"{context}.trace_payload_bytes",
    )
    if trace_payload_bytes != 0:
        raise ValueError(f"{context}.trace_payload_bytes must be zero for inference replay")
    replay_payload_bytes = expect_positive_int(
        json_field(value, "replay_payload_bytes", context),
        f"{context}.replay_payload_bytes",
    )
    if replay_payload_bytes != model_payload_bytes + sample_payload_bytes + witness_payload_bytes:
        raise ValueError(f"{context}.replay_payload_bytes must equal model + sample + witness")
    expect_positive_int(
        json_field(value, "runtime_state_payload_bytes", context),
        f"{context}.runtime_state_payload_bytes",
    )
    if expect_nonnegative_int(
        json_field(value, "forward_recompute_steps", context),
        f"{context}.forward_recompute_steps",
    ) != expected_steps:
        raise ValueError(f"{context}.forward_recompute_steps must match sample count")
    if expect_nonnegative_int(
        json_field(value, "inverse_recompute_steps", context),
        f"{context}.inverse_recompute_steps",
    ) != expected_steps:
        raise ValueError(f"{context}.inverse_recompute_steps must match sample count")
    if "peak_rss_bytes" in value:
        expect_optional_peak_rss(value["peak_rss_bytes"], f"{context}.peak_rss_bytes", required=False)


def validate_inference_proof(
    value: Any,
    context: str,
    *,
    prediction: int,
    correct: bool,
    margin: int,
    active_pixels: int,
    attribution_metadata: Optional[dict[str, Any]] = None,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if json_field(value, "claim", context) != INFERENCE_PROOF_CLAIM:
        raise ValueError(f"{context}.claim must be `{INFERENCE_PROOF_CLAIM}`")
    if json_field(value, "kernel", context) != "examples/mnist_identify.rev":
        raise ValueError(f"{context}.kernel must be examples/mnist_identify.rev")
    if json_field(value, "arithmetic", context) != "q31_wrapping_i64":
        raise ValueError(f"{context}.arithmetic must be q31_wrapping_i64")
    if json_field(value, "witnesses", context) != ["logits", "prediction", "correct"]:
        raise ValueError(f"{context}.witnesses must list inference witnesses")
    if expect_digit(json_field(value, "prediction", context), f"{context}.prediction") != prediction:
        raise ValueError(f"{context}.prediction must match report")
    if expect_bool(json_field(value, "correct", context), f"{context}.correct") != correct:
        raise ValueError(f"{context}.correct must match report")
    expect_digit(json_field(value, "predicted_digit", context), f"{context}.predicted_digit")
    expect_digit(json_field(value, "runner_up_digit", context), f"{context}.runner_up_digit")
    if expect_i64(json_field(value, "margin", context), f"{context}.margin") != margin:
        raise ValueError(f"{context}.margin must match attribution")
    if expect_nonnegative_int(json_field(value, "active_pixels", context), f"{context}.active_pixels") != active_pixels:
        raise ValueError(f"{context}.active_pixels must match report")
    contribution_count = expect_nonnegative_int(
        json_field(value, "contribution_count", context),
        f"{context}.contribution_count",
    )
    margin_contribution_count = expect_nonnegative_int(
        json_field(value, "margin_contribution_count", context),
        f"{context}.margin_contribution_count",
    )
    contribution_fingerprint = validate_sha256(
        json_field(value, "contribution_ledger_fingerprint", context),
        f"{context}.contribution_ledger_fingerprint",
    )
    margin_contribution_fingerprint = validate_sha256(
        json_field(value, "margin_contribution_ledger_fingerprint", context),
        f"{context}.margin_contribution_ledger_fingerprint",
    )
    if attribution_metadata is not None:
        if contribution_count != attribution_metadata["contribution_count"]:
            raise ValueError(f"{context}.contribution_count must match attribution")
        if margin_contribution_count != attribution_metadata["margin_contribution_count"]:
            raise ValueError(f"{context}.margin_contribution_count must match attribution")
        if contribution_fingerprint != attribution_metadata["contribution_ledger_fingerprint"]:
            raise ValueError(f"{context}.contribution_ledger_fingerprint must match attribution")
        if margin_contribution_fingerprint != attribution_metadata["margin_contribution_ledger_fingerprint"]:
            raise ValueError(f"{context}.margin_contribution_ledger_fingerprint must match attribution")
    validate_inference_memory(value, context)
    checks = json_field(value, "checks", context)
    if not isinstance(checks, dict):
        raise ValueError(f"{context}.checks must be an object")
    for field in (
        "prediction_matches_logits",
        "correct_matches_logits",
        "attribution_matches_logit",
        "attribution_matches_margin",
        "restored_initial_state",
    ):
        if not expect_bool(json_field(checks, field, f"{context}.checks"), f"{context}.checks.{field}"):
            raise ValueError(f"{context}.checks.{field} must be true")
    if "fingerprints" in value:
        validate_fingerprints(
            value["fingerprints"],
            ("identify_source", "model", "sample", "result"),
            f"{context}.fingerprints",
        )


def validate_inference_explanation_contract(
    value: Any,
    context: str,
    *,
    prediction: int,
    correct: bool,
    margin: int,
    active_pixels: int,
    require_verification_checks: bool,
    attribution_metadata: Optional[dict[str, Any]] = None,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if json_field(value, "claim", context) != INFERENCE_EXPLANATION_CLAIM:
        raise ValueError(f"{context}.claim must be `{INFERENCE_EXPLANATION_CLAIM}`")
    passed = expect_bool(json_field(value, "passed", context), f"{context}.passed")
    if expect_digit(json_field(value, "prediction", context), f"{context}.prediction") != prediction:
        raise ValueError(f"{context}.prediction must match report")
    if expect_bool(json_field(value, "correct", context), f"{context}.correct") != correct:
        raise ValueError(f"{context}.correct must match report")
    expect_digit(json_field(value, "predicted_digit", context), f"{context}.predicted_digit")
    expect_digit(json_field(value, "runner_up_digit", context), f"{context}.runner_up_digit")
    if expect_i64(json_field(value, "margin", context), f"{context}.margin") != margin:
        raise ValueError(f"{context}.margin must match attribution")
    if expect_nonnegative_int(
        json_field(value, "active_pixel_count", context),
        f"{context}.active_pixel_count",
    ) != active_pixels:
        raise ValueError(f"{context}.active_pixel_count must match report")
    top_logit_count = expect_positive_int(
        json_field(value, "top_logit_count", context),
        f"{context}.top_logit_count",
    )
    if top_logit_count > DIGITS:
        raise ValueError(f"{context}.top_logit_count must be no more than {DIGITS}")
    contribution_count = expect_nonnegative_int(
        json_field(value, "contribution_count", context),
        f"{context}.contribution_count",
    )
    margin_contribution_count = expect_nonnegative_int(
        json_field(value, "margin_contribution_count", context),
        f"{context}.margin_contribution_count",
    )
    ledger_fingerprints = json_field(value, "ledger_fingerprints", context)
    if not isinstance(ledger_fingerprints, dict):
        raise ValueError(f"{context}.ledger_fingerprints must be an object")
    if json_field(ledger_fingerprints, "algorithm", f"{context}.ledger_fingerprints") != "sha256":
        raise ValueError(f"{context}.ledger_fingerprints.algorithm must be sha256")
    contribution_fingerprint = validate_sha256(
        json_field(ledger_fingerprints, "contribution", f"{context}.ledger_fingerprints"),
        f"{context}.ledger_fingerprints.contribution",
    )
    margin_contribution_fingerprint = validate_sha256(
        json_field(ledger_fingerprints, "margin_contribution", f"{context}.ledger_fingerprints"),
        f"{context}.ledger_fingerprints.margin_contribution",
    )
    if attribution_metadata is not None:
        if contribution_count != attribution_metadata["contribution_count"]:
            raise ValueError(f"{context}.contribution_count must match attribution")
        if margin_contribution_count != attribution_metadata["margin_contribution_count"]:
            raise ValueError(f"{context}.margin_contribution_count must match attribution")
        if contribution_fingerprint != attribution_metadata["contribution_ledger_fingerprint"]:
            raise ValueError(f"{context}.ledger_fingerprints.contribution must match attribution")
        if margin_contribution_fingerprint != attribution_metadata["margin_contribution_ledger_fingerprint"]:
            raise ValueError(f"{context}.ledger_fingerprints.margin_contribution must match attribution")
    replay_direction = json_field(value, "replay_direction", context)
    if not isinstance(replay_direction, dict):
        raise ValueError(f"{context}.replay_direction must be an object")
    for field in ("forward_transition", "reverse_transition"):
        validate_nonempty_string(
            json_field(replay_direction, field, f"{context}.replay_direction"),
            f"{context}.replay_direction.{field}",
        )
    if not expect_bool(
        json_field(replay_direction, "inverse_restores_initial_state", f"{context}.replay_direction"),
        f"{context}.replay_direction.inverse_restores_initial_state",
    ):
        raise ValueError(f"{context}.replay_direction.inverse_restores_initial_state must be true")

    checks = json_field(value, "checks", context)
    if not isinstance(checks, list) or not checks:
        raise ValueError(f"{context}.checks must be a non-empty array")
    required = {
        "logits_determine_prediction",
        "correct_matches_label",
        "attribution_reconstructs_logit",
        "attribution_reconstructs_margin",
        "reverse_restores_initial_state",
    }
    if require_verification_checks:
        required.update(
            {
                "proof_recomputed",
                "result_recomputed",
                "source_inputs_checked",
                "verification_restores_initial_state",
            }
        )
    seen: set[str] = set()
    all_passed = True
    for index, check in enumerate(checks):
        check_context = f"{context}.checks[{index}]"
        metric = json_field(check, "metric", check_context)
        if not isinstance(metric, str) or not metric:
            raise ValueError(f"{check_context}.metric must be a non-empty string")
        if metric in seen:
            raise ValueError(f"{context}.checks has duplicate metric `{metric}`")
        seen.add(metric)
        validate_nonempty_string(json_field(check, "actual", check_context), f"{check_context}.actual")
        validate_nonempty_string(
            json_field(check, "requirement", check_context),
            f"{check_context}.requirement",
        )
        all_passed = all_passed and expect_bool(
            json_field(check, "passed", check_context),
            f"{check_context}.passed",
        )
    missing = sorted(required - seen)
    if missing:
        raise ValueError(f"{context}.checks missing metric(s): " + ", ".join(missing))
    if passed != all_passed:
        raise ValueError(f"{context}.passed must summarize checks")
    if not passed:
        raise ValueError(f"{context}.passed must be true")


def validate_native_inference_report(report: dict[str, Any]) -> None:
    kind = report["kind"]
    if kind == INFERENCE_AUDIT_KIND:
        if report.get("path") is not None:
            validate_nonempty_string(report["path"], "path")
        if report.get("fingerprints") is not None:
            validate_fingerprints(
                report["fingerprints"],
                AUDIT_FINGERPRINT_FIELDS,
                "fingerprints",
            )
    elif kind in (MODEL_INFERENCE_AUDIT_KIND, MODEL_EVALUATION_ROW_KIND):
        validate_model_bundle_ref(json_field(report, "model_bundle", "inference report"), "model_bundle")
        sample_source = json_field(report, "sample_source", "inference report")
        if not isinstance(sample_source, dict):
            raise ValueError("sample_source must be an object")
        validate_nonempty_string(json_field(sample_source, "kind", "sample_source"), "sample_source.kind")
        if kind == MODEL_EVALUATION_ROW_KIND:
            validate_model_evaluation_bundle_ref(
                json_field(report, "evaluation_bundle", "inference report"),
                "evaluation_bundle",
            )
            if not expect_bool(
                json_field(report, "row_matches_recomputed_inference", "inference report"),
                "row_matches_recomputed_inference",
            ):
                raise ValueError("row_matches_recomputed_inference must be true")
            validate_model_evaluation_row(
                json_field(report, "row", "inference report"),
                "row",
                samples=None,
            )
    else:
        raise ValueError(f"unsupported native inference report kind `{kind}`")

    label = expect_digit(json_field(report, "label", "inference report"), "label")
    prediction = expect_digit(json_field(report, "prediction", "inference report"), "prediction")
    correct = expect_bool(json_field(report, "correct", "inference report"), "correct")
    if correct != (prediction == label):
        raise ValueError("correct must match prediction == label")
    logits = validate_logits(json_field(report, "logits", "inference report"), "logits")
    top_logits = validate_audit_top_logits(
        json_field(report, "top_logits", "inference report"),
        logits,
        "top_logits",
    )
    top_logits_for_attribution = [
        {"digit": item["digit"], "value": item["logit"]}
        for item in top_logits
    ]
    attribution_metadata = validate_attribution(
        json_field(report, "attribution", "inference report"),
        logits,
        top_logits_for_attribution,
        "attribution",
    )
    validate_witness_checks(
        json_field(report, "witness_checks", "inference report"),
        logits,
        label,
        prediction,
        correct,
        "witness_checks",
    )
    active_pixel_count = validate_active_pixel_rows(
        json_field(report, "active_pixels", "inference report"),
        "active_pixels",
    )
    forward = json_field(report, "forward", "inference report")
    if expect_finite_number(json_field(forward, "elapsed_seconds", "forward"), "forward.elapsed_seconds") < 0.0:
        raise ValueError("forward.elapsed_seconds must be non-negative")
    inverse = json_field(report, "inverse", "inference report")
    if not expect_bool(json_field(inverse, "restored_initial_state", "inverse"), "inverse.restored_initial_state"):
        raise ValueError("inverse.restored_initial_state must be true")
    if expect_finite_number(json_field(inverse, "elapsed_seconds", "inverse"), "inverse.elapsed_seconds") < 0.0:
        raise ValueError("inverse.elapsed_seconds must be non-negative")
    validate_inference_memory(json_field(report, "memory", "inference report"), "memory")
    validate_inference_proof(
        json_field(report, "proof", "inference report"),
        "proof",
        prediction=prediction,
        correct=correct,
        margin=int(report["attribution"]["margin"]),
        active_pixels=active_pixel_count,
        attribution_metadata=attribution_metadata,
    )
    validate_inference_explanation_contract(
        json_field(report, "explanation_contract", "inference report"),
        "explanation_contract",
        prediction=prediction,
        correct=correct,
        margin=int(report["attribution"]["margin"]),
        active_pixels=active_pixel_count,
        require_verification_checks=False,
        attribution_metadata=attribution_metadata,
    )
    if "inference_output" in report and report["inference_output"] is not None:
        validate_nonempty_string(report["inference_output"], "inference_output")
    if "audit_step" in report and report["audit_step"] is not None:
        expect_nonnegative_int(report["audit_step"], "audit_step")
    if "sample_index" in report and report["sample_index"] is not None:
        expect_nonnegative_int(report["sample_index"], "sample_index")


def validate_model_evaluation_row(value: Any, context: str, *, samples: Optional[int]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    index = expect_nonnegative_int(json_field(value, "index", context), f"{context}.index")
    if samples is not None and index >= samples:
        raise ValueError(f"{context}.index must be less than sample count")
    validate_sha256(json_field(value, "sample_fingerprint", context), f"{context}.sample_fingerprint")
    label = expect_digit(json_field(value, "label", context), f"{context}.label")
    prediction = expect_digit(json_field(value, "prediction", context), f"{context}.prediction")
    correct = expect_bool(json_field(value, "correct", context), f"{context}.correct")
    if correct != (prediction == label):
        raise ValueError(f"{context}.correct must match prediction == label")
    runner_up_digit = expect_digit(
        json_field(value, "runner_up_digit", context),
        f"{context}.runner_up_digit",
    )
    margin = expect_i64(json_field(value, "margin", context), f"{context}.margin")
    active_pixels = expect_nonnegative_int(
        json_field(value, "active_pixels", context),
        f"{context}.active_pixels",
    )
    if active_pixels > IMAGE_PIXELS:
        raise ValueError(f"{context}.active_pixels must be no more than {IMAGE_PIXELS}")
    if "audit_step" in value:
        expect_nonnegative_int(value["audit_step"], f"{context}.audit_step")
    if "source_sample_index" in value:
        expect_nonnegative_int(value["source_sample_index"], f"{context}.source_sample_index")
    return {
        "index": index,
        "label": label,
        "prediction": prediction,
        "correct": correct,
        "runner_up_digit": runner_up_digit,
        "margin": margin,
        "active_pixels": active_pixels,
    }


def validate_model_evaluation_rows(value: Any, context: str, *, samples: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != samples:
        raise ValueError(f"{context} length must match sample count")
    rows = [
        validate_model_evaluation_row(row, f"{context}[{index}]", samples=samples)
        for index, row in enumerate(value)
    ]
    for index, row in enumerate(rows):
        if row["index"] != index:
            raise ValueError(f"{context}[{index}].index must match row order")
    return rows


def validate_model_evaluation_row_subset(value: Any, context: str, *, samples: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > samples:
        raise ValueError(f"{context} must be an array no longer than sample count")
    rows = [
        validate_model_evaluation_row(row, f"{context}[{index}]", samples=samples)
        for index, row in enumerate(value)
    ]
    seen: set[int] = set()
    for row in rows:
        if row["index"] in seen:
            raise ValueError(f"{context} has duplicate row index {row['index']}")
        seen.add(row["index"])
    return rows


def validate_model_evaluation_summary(
    value: Any,
    context: str,
    *,
    rows: Optional[list[dict[str, Any]]] = None,
    require_elapsed: bool,
) -> tuple[int, int, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    samples = expect_positive_int(json_field(value, "samples", context), f"{context}.samples")
    correct = expect_nonnegative_int(json_field(value, "correct", context), f"{context}.correct")
    incorrect = expect_nonnegative_int(json_field(value, "incorrect", context), f"{context}.incorrect")
    if correct + incorrect != samples:
        raise ValueError(f"{context}.correct + incorrect must equal samples")
    validate_accuracy(json_field(value, "accuracy_percent", context), correct, samples, f"{context}.accuracy_percent")
    lowest_index = json_field(value, "lowest_margin_index", context)
    lowest_margin = json_field(value, "lowest_margin", context)
    if rows is not None:
        if len(rows) != samples:
            raise ValueError(f"{context}.samples must match rows length")
        observed_correct = sum(1 for row in rows if row["correct"])
        if correct != observed_correct:
            raise ValueError(f"{context}.correct must match rows")
        observed_lowest = min(rows, key=lambda row: (row["margin"], row["index"]))
        if expect_nonnegative_int(lowest_index, f"{context}.lowest_margin_index") != observed_lowest["index"]:
            raise ValueError(f"{context}.lowest_margin_index must identify the lowest margin row")
        if expect_i64(lowest_margin, f"{context}.lowest_margin") != observed_lowest["margin"]:
            raise ValueError(f"{context}.lowest_margin must match the lowest margin row")
    else:
        if lowest_index is not None:
            expect_nonnegative_int(lowest_index, f"{context}.lowest_margin_index")
        if lowest_margin is not None:
            expect_i64(lowest_margin, f"{context}.lowest_margin")
    if require_elapsed:
        elapsed = expect_finite_number(json_field(value, "elapsed_seconds", context), f"{context}.elapsed_seconds")
        if elapsed < 0.0:
            raise ValueError(f"{context}.elapsed_seconds must be non-negative")
        samples_per_second = expect_finite_number(
            json_field(value, "samples_per_second", context),
            f"{context}.samples_per_second",
        )
        if samples_per_second < 0.0:
            raise ValueError(f"{context}.samples_per_second must be non-negative")
        for field in ("forward_elapsed_seconds", "reverse_elapsed_seconds"):
            if expect_finite_number(json_field(value, field, context), f"{context}.{field}") < 0.0:
                raise ValueError(f"{context}.{field} must be non-negative")
    return samples, correct, incorrect


def validate_batch_inference_proof(value: Any, samples: int, context: str) -> None:
    validate_inference_memory(value, context, expected_steps=samples)


def validate_model_evaluation_gate(value: Any, context: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    gate_passed = expect_bool(json_field(value, "passed", context), f"{context}.passed")
    checks = json_field(value, "checks", context)
    if not isinstance(checks, list) or not checks:
        raise ValueError(f"{context}.checks must be a non-empty array")
    results = []
    for index, check in enumerate(checks):
        check_context = f"{context}.checks[{index}]"
        validate_nonempty_string(json_field(check, "metric", check_context), f"{check_context}.metric")
        results.append(expect_bool(json_field(check, "passed", check_context), f"{check_context}.passed"))
        validate_nonempty_string(json_field(check, "actual", check_context), f"{check_context}.actual")
        validate_nonempty_string(json_field(check, "expectation", check_context), f"{check_context}.expectation")
    if gate_passed != all(results):
        raise ValueError(f"{context}.passed must match checks")


def validate_model_evaluation_report(report: dict[str, Any]) -> None:
    validate_model_bundle_ref(json_field(report, "model_bundle", "model evaluation report"), "model_bundle")
    samples_json = json_field(report, "samples_json", "model evaluation report")
    if not isinstance(samples_json, dict):
        raise ValueError("samples_json must be an object")
    validate_nonempty_string(json_field(samples_json, "path", "samples_json"), "samples_json.path")
    validate_sha256(json_field(samples_json, "fingerprint", "samples_json"), "samples_json.fingerprint")
    if "fingerprints" in report:
        validate_model_evaluation_fingerprints(report["fingerprints"], "fingerprints")
    rows_value = json_field(report, "rows", "model evaluation report")
    summary_value = json_field(report, "summary", "model evaluation report")
    samples = expect_positive_int(json_field(summary_value, "samples", "summary"), "summary.samples")
    rows = validate_model_evaluation_rows(rows_value, "rows", samples=samples)
    validate_model_evaluation_summary(summary_value, "summary", rows=rows, require_elapsed=True)
    validate_batch_inference_proof(json_field(report, "proof", "model evaluation report"), samples, "proof")
    if "evaluation_output" in report and report["evaluation_output"] is not None:
        validate_nonempty_string(report["evaluation_output"], "evaluation_output")
    if "gate" in report:
        validate_model_evaluation_gate(report["gate"], "gate")
    if "gate_policy" in report and not isinstance(report["gate_policy"], dict):
        raise ValueError("gate_policy must be an object")


def validate_model_evaluation_label_summaries(
    value: Any,
    rows: list[dict[str, Any]],
    context: str,
) -> None:
    if not isinstance(value, list) or len(value) != DIGITS:
        raise ValueError(f"{context} must have {DIGITS} entries")
    for label, item in enumerate(value):
        item_context = f"{context}[{label}]"
        if expect_digit(json_field(item, "label", item_context), f"{item_context}.label") != label:
            raise ValueError(f"{item_context}.label must match its position")
        matching = [row for row in rows if row["label"] == label]
        correct = sum(1 for row in matching if row["correct"])
        samples = expect_nonnegative_int(json_field(item, "samples", item_context), f"{item_context}.samples")
        if samples != len(matching):
            raise ValueError(f"{item_context}.samples must match rows")
        if expect_nonnegative_int(json_field(item, "correct", item_context), f"{item_context}.correct") != correct:
            raise ValueError(f"{item_context}.correct must match rows")
        incorrect = expect_nonnegative_int(
            json_field(item, "incorrect", item_context),
            f"{item_context}.incorrect",
        )
        if incorrect != samples - correct:
            raise ValueError(f"{item_context}.incorrect must equal samples - correct")
        validate_accuracy(
            json_field(item, "accuracy_percent", item_context),
            correct,
            samples,
            f"{item_context}.accuracy_percent",
        )
        lowest_index = json_field(item, "lowest_margin_index", item_context)
        lowest_margin = json_field(item, "lowest_margin", item_context)
        if matching:
            lowest = min(matching, key=lambda row: (row["margin"], row["index"]))
            if expect_nonnegative_int(lowest_index, f"{item_context}.lowest_margin_index") != lowest["index"]:
                raise ValueError(f"{item_context}.lowest_margin_index must match rows")
            if expect_i64(lowest_margin, f"{item_context}.lowest_margin") != lowest["margin"]:
                raise ValueError(f"{item_context}.lowest_margin must match rows")
        elif lowest_index is not None or lowest_margin is not None:
            raise ValueError(f"{item_context}.lowest_margin fields must be null without samples")


def validate_model_evaluation_label_summary_totals(
    value: Any,
    *,
    samples: int,
    correct: int,
    incorrect: int,
    context: str,
) -> None:
    if not isinstance(value, list) or len(value) != DIGITS:
        raise ValueError(f"{context} must have {DIGITS} entries")
    sample_total = 0
    correct_total = 0
    incorrect_total = 0
    for label, item in enumerate(value):
        item_context = f"{context}[{label}]"
        if expect_digit(json_field(item, "label", item_context), f"{item_context}.label") != label:
            raise ValueError(f"{item_context}.label must match its position")
        label_samples = expect_nonnegative_int(
            json_field(item, "samples", item_context),
            f"{item_context}.samples",
        )
        label_correct = expect_nonnegative_int(
            json_field(item, "correct", item_context),
            f"{item_context}.correct",
        )
        label_incorrect = expect_nonnegative_int(
            json_field(item, "incorrect", item_context),
            f"{item_context}.incorrect",
        )
        if label_correct + label_incorrect != label_samples:
            raise ValueError(f"{item_context}.correct + incorrect must equal samples")
        validate_accuracy(
            json_field(item, "accuracy_percent", item_context),
            label_correct,
            label_samples,
            f"{item_context}.accuracy_percent",
        )
        lowest_index = json_field(item, "lowest_margin_index", item_context)
        lowest_margin = json_field(item, "lowest_margin", item_context)
        if label_samples:
            if lowest_index is None or lowest_margin is None:
                raise ValueError(f"{item_context}.lowest_margin fields are required with samples")
            if expect_nonnegative_int(lowest_index, f"{item_context}.lowest_margin_index") >= samples:
                raise ValueError(f"{item_context}.lowest_margin_index must be less than sample count")
            expect_i64(lowest_margin, f"{item_context}.lowest_margin")
        elif lowest_index is not None or lowest_margin is not None:
            raise ValueError(f"{item_context}.lowest_margin fields must be null without samples")
        sample_total += label_samples
        correct_total += label_correct
        incorrect_total += label_incorrect
    if (sample_total, correct_total, incorrect_total) != (samples, correct, incorrect):
        raise ValueError(f"{context} totals must match summary")


def validate_model_evaluation_row_order(rows: list[dict[str, Any]], context: str, order: str) -> None:
    if order == "low_margin" or order == "incorrect":
        expected = sorted(rows, key=lambda row: (row["margin"], row["index"]))
    else:
        raise AssertionError(f"unknown model evaluation row order {order}")
    if rows != expected:
        raise ValueError(f"{context} must be sorted by margin and index")
    if order == "incorrect" and any(row["correct"] for row in rows):
        raise ValueError(f"{context} must contain only incorrect rows")


def validate_model_evaluation_scan_report(report: dict[str, Any]) -> None:
    validate_nonempty_string(json_field(report, "path", "model evaluation scan"), "path")
    validate_model_evaluation_fingerprints(json_field(report, "fingerprints", "model evaluation scan"), "fingerprints")
    summary_value = json_field(report, "summary", "model evaluation scan")
    samples, correct, incorrect = validate_model_evaluation_summary(
        summary_value,
        "summary",
        rows=None,
        require_elapsed=False,
    )
    top_low_margin = validate_model_evaluation_row_subset(
        json_field(report, "top_low_margin", "model evaluation scan"),
        "top_low_margin",
        samples=samples,
    )
    validate_model_evaluation_row_order(top_low_margin, "top_low_margin", "low_margin")
    top_incorrect = validate_model_evaluation_row_subset(
        json_field(report, "top_incorrect", "model evaluation scan"),
        "top_incorrect",
        samples=samples,
    )
    validate_model_evaluation_row_order(top_incorrect, "top_incorrect", "incorrect")
    validate_model_evaluation_label_summary_totals(
        json_field(report, "by_label", "model evaluation scan"),
        samples=samples,
        correct=correct,
        incorrect=incorrect,
        context="by_label",
    )
    top_confusions = json_field(report, "top_confusions", "model evaluation scan")
    if not isinstance(top_confusions, list):
        raise ValueError("top_confusions must be an array")
    for index, item in enumerate(top_confusions):
        item_context = f"top_confusions[{index}]"
        label = expect_digit(json_field(item, "label", item_context), f"{item_context}.label")
        prediction = expect_digit(json_field(item, "prediction", item_context), f"{item_context}.prediction")
        if label == prediction:
            raise ValueError(f"{item_context} must describe a confusion")
        expect_positive_int(json_field(item, "count", item_context), f"{item_context}.count")
        first_index = expect_nonnegative_int(json_field(item, "first_index", item_context), f"{item_context}.first_index")
        if first_index >= samples:
            raise ValueError(f"{item_context}.first_index must be less than sample count")
        expect_i64(json_field(item, "lowest_margin", item_context), f"{item_context}.lowest_margin")


def validate_model_evaluation_verification_report(report: dict[str, Any]) -> None:
    validate_nonempty_string(json_field(report, "path", "model evaluation verification"), "path")
    validate_model_evaluation_fingerprints(json_field(report, "fingerprints", "model evaluation verification"), "fingerprints")
    summary_value = json_field(report, "summary", "model evaluation verification")
    samples, _, _ = validate_model_evaluation_summary(summary_value, "summary", rows=None, require_elapsed=True)
    for field in ("rows_match", "proof_matches", "restored_initial_state"):
        if not expect_bool(json_field(report, field, "model evaluation verification"), field):
            raise ValueError(f"{field} must be true")
    for field in ("source_model_checked", "source_samples_checked"):
        expect_bool(json_field(report, field, "model evaluation verification"), field)
    validate_optional_source_check(report.get("source_model"), "source_model")
    validate_optional_source_check(report.get("source_samples_json"), "source_samples_json")
    validate_model_evaluation_gate(report.get("gate"), "gate")
    if expect_finite_number(
        json_field(report, "elapsed_seconds", "model evaluation verification"),
        "elapsed_seconds",
    ) < 0.0:
        raise ValueError("elapsed_seconds must be non-negative")
    # The report summary establishes the replay sample count for the matching proof in the bundle.
    if samples <= 0:
        raise ValueError("summary.samples must be positive")


def validate_inference_verification_report(report: dict[str, Any]) -> None:
    validate_nonempty_string(json_field(report, "path", "inference verification"), "path")
    validate_inference_fingerprints(json_field(report, "fingerprints", "inference verification"), "fingerprints")
    prediction = expect_digit(json_field(report, "prediction", "inference verification"), "prediction")
    correct = expect_bool(json_field(report, "correct", "inference verification"), "correct")
    for field in ("result_matches", "restored_initial_state", "proof_matches"):
        if not expect_bool(json_field(report, field, "inference verification"), field):
            raise ValueError(f"{field} must be true")
    for field in (
        "source_evaluation_checked",
        "source_model_checked",
        "source_training_checked",
        "source_sample_checked",
    ):
        expect_bool(json_field(report, field, "inference verification"), field)
    validate_optional_source_check(report.get("source_model"), "source_model")
    validate_optional_source_check(report.get("source_training_bundle"), "source_training_bundle")
    validate_optional_source_check(report.get("source_sample"), "source_sample")
    source_evaluation = report.get("source_model_evaluation")
    if source_evaluation is not None:
        validate_optional_source_evaluation_check(source_evaluation, "source_model_evaluation")
    attribution = json_field(report, "attribution", "inference verification")
    if not isinstance(attribution, dict):
        raise ValueError("attribution must be an object")
    margin = expect_i64(json_field(attribution, "margin", "attribution"), "attribution.margin")
    attribution_metadata = {
        "contribution_count": expect_nonnegative_int(
            json_field(attribution, "contribution_count", "attribution"),
            "attribution.contribution_count",
        ),
        "margin_contribution_count": expect_nonnegative_int(
            json_field(attribution, "margin_contribution_count", "attribution"),
            "attribution.margin_contribution_count",
        ),
        "contribution_ledger_fingerprint": validate_sha256(
            json_field(attribution, "contribution_ledger_fingerprint", "attribution"),
            "attribution.contribution_ledger_fingerprint",
        ),
        "margin_contribution_ledger_fingerprint": validate_sha256(
            json_field(attribution, "margin_contribution_ledger_fingerprint", "attribution"),
            "attribution.margin_contribution_ledger_fingerprint",
        ),
    }
    active_pixels = expect_nonnegative_int(
        json_field(report["proof"], "active_pixels", "proof"),
        "proof.active_pixels",
    )
    validate_inference_memory(json_field(report, "memory", "inference verification"), "memory")
    validate_inference_proof(
        json_field(report, "proof", "inference verification"),
        "proof",
        prediction=prediction,
        correct=correct,
        margin=margin,
        active_pixels=active_pixels,
        attribution_metadata=attribution_metadata,
    )
    validate_inference_explanation_contract(
        json_field(report, "explanation_contract", "inference verification"),
        "explanation_contract",
        prediction=prediction,
        correct=correct,
        margin=margin,
        active_pixels=active_pixels,
        require_verification_checks=True,
        attribution_metadata=attribution_metadata,
    )
    if expect_finite_number(json_field(report, "elapsed_seconds", "inference verification"), "elapsed_seconds") < 0.0:
        raise ValueError("elapsed_seconds must be non-negative")


def validate_throughput_block(block: Any, context: str, rate_field: str) -> None:
    samples = expect_positive_int(json_field(block, "samples", context), f"{context}.samples")
    correct = expect_nonnegative_int(json_field(block, "correct", context), f"{context}.correct")
    if correct > samples:
        raise ValueError(f"{context}.correct cannot exceed samples")
    accuracy = expect_finite_number(
        json_field(block, "accuracy_percent", context),
        f"{context}.accuracy_percent",
    )
    if not 0.0 <= accuracy <= 100.0:
        raise ValueError(f"{context}.accuracy_percent must be in 0..100")
    elapsed = expect_finite_number(
        json_field(block, "elapsed_seconds", context),
        f"{context}.elapsed_seconds",
    )
    if elapsed < 0.0:
        raise ValueError(f"{context}.elapsed_seconds must be non-negative")
    rate = expect_finite_number(json_field(block, rate_field, context), f"{context}.{rate_field}")
    if rate < 0.0:
        raise ValueError(f"{context}.{rate_field} must be non-negative")


def validate_run_report(
    report: dict[str, Any],
    require_reverse_check: bool,
    require_peak_rss: bool,
) -> None:
    config = json_field(report, "config", "run report")
    reverse_check = expect_bool(
        json_field(config, "reverse_check", "run report.config"),
        "run report.config.reverse_check",
    )
    if require_reverse_check and not reverse_check:
        raise ValueError("run report reverse_check is required")

    validate_throughput_block(json_field(report, "train", "run report"), "train", "updates_per_second")
    validate_throughput_block(json_field(report, "eval", "run report"), "eval", "samples_per_second")

    trace = json_field(report, "trace", "run report")
    trace_enabled = expect_bool(json_field(trace, "enabled", "trace"), "trace.enabled")
    trace_entries = expect_nonnegative_int(json_field(trace, "entries", "trace"), "trace.entries")
    trace_bytes_per_step = expect_nonnegative_int(
        json_field(trace, "bytes_per_step", "trace"), "trace.bytes_per_step"
    )
    trace_payload_bytes = expect_nonnegative_int(
        json_field(trace, "payload_bytes", "trace"), "trace.payload_bytes"
    )
    if trace_enabled and trace_payload_bytes != trace_entries * trace_bytes_per_step:
        raise ValueError("trace payload_bytes must equal entries * bytes_per_step")

    memory = json_field(report, "memory", "run report")
    model_payload_bytes = expect_positive_int(
        json_field(memory, "model_payload_bytes", "memory"),
        "memory.model_payload_bytes",
    )
    train_dataset_payload_bytes = expect_nonnegative_int(
        json_field(memory, "train_dataset_payload_bytes", "memory"),
        "memory.train_dataset_payload_bytes",
    )
    eval_dataset_payload_bytes = expect_nonnegative_int(
        json_field(memory, "eval_dataset_payload_bytes", "memory"),
        "memory.eval_dataset_payload_bytes",
    )
    dataset_payload_bytes = expect_nonnegative_int(
        json_field(memory, "dataset_payload_bytes", "memory"),
        "memory.dataset_payload_bytes",
    )
    memory_trace_payload_bytes = expect_nonnegative_int(
        json_field(memory, "trace_payload_bytes", "memory"),
        "memory.trace_payload_bytes",
    )
    estimated_payload_bytes = expect_nonnegative_int(
        json_field(memory, "estimated_payload_bytes", "memory"),
        "memory.estimated_payload_bytes",
    )
    expect_optional_peak_rss(
        json_field(memory, "peak_rss_bytes", "memory"),
        "memory.peak_rss_bytes",
        require_peak_rss,
    )
    if dataset_payload_bytes != train_dataset_payload_bytes + eval_dataset_payload_bytes:
        raise ValueError("memory.dataset_payload_bytes must equal train + eval dataset bytes")
    if memory_trace_payload_bytes != trace_payload_bytes:
        raise ValueError("memory.trace_payload_bytes must match trace.payload_bytes")
    expected_estimated = model_payload_bytes + dataset_payload_bytes + trace_payload_bytes
    if estimated_payload_bytes != expected_estimated:
        raise ValueError("memory.estimated_payload_bytes must equal model + dataset + trace bytes")

    proof = json_field(report, "proof", "run report")
    proof_enabled = expect_bool(json_field(proof, "enabled", "proof"), "proof.enabled")
    if proof_enabled != reverse_check:
        raise ValueError("proof.enabled must match config.reverse_check")
    if proof_enabled:
        entries = expect_nonnegative_int(json_field(proof, "entries", "proof"), "proof.entries")
        sample_bytes_per_step = expect_nonnegative_int(
            json_field(proof, "sample_bytes_per_step", "proof"),
            "proof.sample_bytes_per_step",
        )
        witness_bytes_per_step = expect_nonnegative_int(
            json_field(proof, "witness_bytes_per_step", "proof"),
            "proof.witness_bytes_per_step",
        )
        trace_replay_bytes_per_step = expect_nonnegative_int(
            json_field(proof, "trace_replay_bytes_per_step", "proof"),
            "proof.trace_replay_bytes_per_step",
        )
        sample_payload_bytes = expect_nonnegative_int(
            json_field(proof, "sample_payload_bytes", "proof"),
            "proof.sample_payload_bytes",
        )
        witness_payload_bytes = expect_nonnegative_int(
            json_field(proof, "witness_payload_bytes", "proof"),
            "proof.witness_payload_bytes",
        )
        trace_replay_payload_bytes = expect_nonnegative_int(
            json_field(proof, "trace_replay_payload_bytes", "proof"),
            "proof.trace_replay_payload_bytes",
        )
        full_replay_payload_bytes = expect_nonnegative_int(
            json_field(proof, "full_replay_payload_bytes", "proof"),
            "proof.full_replay_payload_bytes",
        )
        proof_model_payload_bytes = expect_positive_int(
            json_field(proof, "model_payload_bytes", "proof"),
            "proof.model_payload_bytes",
        )
        if entries != trace_entries:
            raise ValueError("proof.entries must match trace.entries")
        if proof_model_payload_bytes != model_payload_bytes:
            raise ValueError("proof.model_payload_bytes must match memory.model_payload_bytes")
        if sample_payload_bytes != entries * sample_bytes_per_step:
            raise ValueError("proof.sample_payload_bytes must equal entries * sample_bytes_per_step")
        if witness_payload_bytes != entries * witness_bytes_per_step:
            raise ValueError("proof.witness_payload_bytes must equal entries * witness_bytes_per_step")
        if witness_payload_bytes != trace_payload_bytes:
            raise ValueError("proof.witness_payload_bytes must match trace.payload_bytes")
        if trace_replay_bytes_per_step != sample_bytes_per_step + witness_bytes_per_step:
            raise ValueError("proof.trace_replay_bytes_per_step must equal sample + witness bytes per step")
        if trace_replay_payload_bytes != entries * trace_replay_bytes_per_step:
            raise ValueError("proof.trace_replay_payload_bytes must equal entries * trace_replay_bytes_per_step")
        if full_replay_payload_bytes != proof_model_payload_bytes + trace_replay_payload_bytes:
            raise ValueError("proof.full_replay_payload_bytes must equal model + trace replay bytes")
        forward_recompute_steps = expect_nonnegative_int(
            json_field(proof, "forward_recompute_steps", "proof"),
            "proof.forward_recompute_steps",
        )
        inverse_recompute_steps = expect_nonnegative_int(
            json_field(proof, "inverse_recompute_steps", "proof"),
            "proof.inverse_recompute_steps",
        )
        if forward_recompute_steps != entries or inverse_recompute_steps != entries:
            raise ValueError("proof recompute steps must match proof.entries")

    reverse = json_field(report, "reverse", "run report")
    reverse_enabled = expect_bool(json_field(reverse, "enabled", "reverse"), "reverse.enabled")
    if reverse_enabled != reverse_check:
        raise ValueError("reverse.enabled must match config.reverse_check")
    if reverse_enabled:
        checked = expect_nonnegative_int(json_field(reverse, "checked", "reverse"), "reverse.checked")
        if checked != trace_entries:
            raise ValueError("reverse.checked must match trace.entries")
        restored = expect_bool(
            json_field(reverse, "restored_initial_model", "reverse"),
            "reverse.restored_initial_model",
        )
        if not restored:
            raise ValueError("reverse.restored_initial_model must be true")
        elapsed = expect_finite_number(
            json_field(reverse, "elapsed_seconds", "reverse"),
            "reverse.elapsed_seconds",
        )
        if elapsed < 0.0:
            raise ValueError("reverse.elapsed_seconds must be non-negative")
        steps_per_second = expect_finite_number(
            json_field(reverse, "steps_per_second", "reverse"),
            "reverse.steps_per_second",
        )
        if steps_per_second < 0.0:
            raise ValueError("reverse.steps_per_second must be non-negative")


def validate_artifact_comparison(report: dict[str, Any]) -> None:
    artifacts = json_field(report, "artifacts", "artifact comparison")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("artifact comparison artifacts must be a non-empty array")
    profile = json_field(report, "ml_profile", "artifact comparison")
    totals = json_field(report, "totals", "artifact comparison")
    count = expect_nonnegative_int(json_field(totals, "count", "totals"), "totals.count")
    if count != len(artifacts):
        raise ValueError("totals.count must match artifact count")

    sums = {field: 0 for field in ARTIFACT_FIELDS}
    for index, artifact in enumerate(artifacts):
        context = f"artifacts[{index}]"
        json_field(artifact, "kind", context)
        json_field(artifact, "path", context)
        json_field(artifact, "payload_fingerprint", context)
        validate_artifact_fingerprint_summary(
            json_field(artifact, "fingerprints", context),
            context,
        )
        for field in ARTIFACT_FIELDS:
            value = expect_nonnegative_int(
                json_field(artifact, field, context),
                f"{context}.{field}",
            )
            sums[field] += value

    if expect_nonnegative_int(json_field(totals, "file_bytes", "totals"), "totals.file_bytes") != sums["file_bytes"]:
        raise ValueError("totals.file_bytes must match artifact file byte sum")
    if (
        expect_nonnegative_int(
            json_field(totals, "logical_payload_bytes", "totals"),
            "totals.logical_payload_bytes",
        )
        != sums["logical_payload_bytes"]
    ):
        raise ValueError("totals.logical_payload_bytes must match artifact logical payload sum")

    expected_profile = {
        "total_model_payload_bytes": sums["model_payload_bytes"],
        "total_sample_payload_bytes": sums["sample_payload_bytes"],
        "total_witness_payload_bytes": sums["witness_payload_bytes"],
        "total_trace_payload_bytes": sums["trace_payload_bytes"],
        "total_derived_update_payload_bytes": sums["derived_update_payload_bytes"],
        "total_steps": sums["steps"],
        "total_forward_recompute_steps": sums["forward_recompute_steps"],
        "total_inverse_recompute_steps": sums["inverse_recompute_steps"],
    }
    expected_profile["total_recompute_steps"] = (
        expected_profile["total_forward_recompute_steps"]
        + expected_profile["total_inverse_recompute_steps"]
    )
    for field in PROFILE_FIELDS:
        actual = expect_nonnegative_int(json_field(profile, field, "ml_profile"), f"ml_profile.{field}")
        if actual != expected_profile[field]:
            raise ValueError(f"ml_profile.{field} does not match artifact sum")
    expect_nonnegative_int(
        json_field(profile, "total_file_bytes", "ml_profile"),
        "ml_profile.total_file_bytes",
    )
    expect_nonnegative_int(
        json_field(profile, "total_logical_payload_bytes", "ml_profile"),
        "ml_profile.total_logical_payload_bytes",
    )
    expect_ratio(
        json_field(profile, "trace_to_model_payload_ratio", "ml_profile"),
        expected_profile["total_trace_payload_bytes"],
        expected_profile["total_model_payload_bytes"],
        "ml_profile.trace_to_model_payload_ratio",
    )
    expect_ratio(
        json_field(profile, "witness_to_model_payload_ratio", "ml_profile"),
        expected_profile["total_witness_payload_bytes"],
        expected_profile["total_model_payload_bytes"],
        "ml_profile.witness_to_model_payload_ratio",
    )
    expect_ratio(
        json_field(profile, "logical_to_file_ratio", "ml_profile"),
        sums["logical_payload_bytes"],
        sums["file_bytes"],
        "ml_profile.logical_to_file_ratio",
    )


def first_artifact_by_kind(artifacts: list[Any], kind: str) -> Optional[dict[str, Any]]:
    for artifact in artifacts:
        if isinstance(artifact, dict) and artifact.get("kind") == kind:
            return artifact
    return None


def validate_artifact_fingerprint_summary(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context}.fingerprints must be an object")
    algorithm = json_field(value, "algorithm", f"{context}.fingerprints")
    if algorithm != "sha256":
        raise ValueError(f"{context}.fingerprints.algorithm must be sha256")
    count = expect_nonnegative_int(
        json_field(value, "count", f"{context}.fingerprints"),
        f"{context}.fingerprints.count",
    )
    source_count = expect_nonnegative_int(
        json_field(value, "source_count", f"{context}.fingerprints"),
        f"{context}.fingerprints.source_count",
    )
    if source_count > count:
        raise ValueError(f"{context}.fingerprints.source_count must not exceed count")
    for field in ("has_computation", "has_payload", "has_proof", "has_provenance"):
        expect_bool(
            json_field(value, field, f"{context}.fingerprints"),
            f"{context}.fingerprints.{field}",
        )


def artifact_fingerprint_summary(
    artifact: Optional[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    if artifact is None:
        return None
    fingerprints = artifact.get("fingerprints")
    if isinstance(fingerprints, dict):
        return fingerprints
    return None


def expected_audit_contract_results(report: dict[str, Any]) -> dict[str, bool]:
    artifacts = json_field(report, "artifacts", "artifact comparison")
    profile = json_field(report, "ml_profile", "artifact comparison")
    training_trace = first_artifact_by_kind(artifacts, "training_trace")
    model = first_artifact_by_kind(artifacts, "model")
    sample_set = first_artifact_by_kind(artifacts, "sample_set")
    training_step = first_artifact_by_kind(artifacts, "training_step")
    inference = first_artifact_by_kind(artifacts, "inference")
    evaluation = first_artifact_by_kind(artifacts, "model_evaluation")

    def positive(artifact: Optional[dict[str, Any]], field: str) -> bool:
        return artifact is not None and int(artifact[field]) > 0

    def steps(artifact: Optional[dict[str, Any]]) -> int:
        return int(artifact["steps"]) if artifact is not None else 0

    def balanced_artifact_recompute(artifact: Optional[dict[str, Any]]) -> bool:
        if artifact is None:
            return False
        step_count = steps(artifact)
        return (
            step_count > 0
            and int(artifact["forward_recompute_steps"]) == step_count
            and int(artifact["inverse_recompute_steps"]) == step_count
        )

    def fingerprinted(artifact: Optional[dict[str, Any]]) -> bool:
        fingerprints = artifact_fingerprint_summary(artifact)
        return (
            fingerprints is not None
            and int(fingerprints["count"]) >= 3
            and int(fingerprints["source_count"]) > 0
            and bool(fingerprints["has_computation"])
            and bool(fingerprints["has_payload"])
        )

    def has_proof(artifact: Optional[dict[str, Any]]) -> bool:
        fingerprints = artifact_fingerprint_summary(artifact)
        return fingerprints is not None and bool(fingerprints["has_proof"])

    def has_provenance(artifact: Optional[dict[str, Any]]) -> bool:
        fingerprints = artifact_fingerprint_summary(artifact)
        return fingerprints is not None and bool(fingerprints["has_provenance"])

    known_artifacts = [
        artifact
        for artifact in (
            training_trace,
            model,
            sample_set,
            training_step,
            inference,
            evaluation,
        )
        if artifact is not None
    ]

    return {
        "training_trace": (
            positive(training_trace, "trace_payload_bytes")
            and positive(training_trace, "witness_payload_bytes")
            and balanced_artifact_recompute(training_trace)
        ),
        "model_bundle": positive(model, "model_payload_bytes"),
        "sample_set": positive(sample_set, "sample_payload_bytes"),
        "training_step_replay": (
            steps(training_step) == 1
            and positive(training_step, "witness_payload_bytes")
            and positive(training_step, "derived_update_payload_bytes")
            and int(training_step["forward_recompute_steps"]) == 1
            and int(training_step["inverse_recompute_steps"]) == 1
            if training_step is not None
            else False
        ),
        "inference_replay": (
            positive(inference, "model_payload_bytes")
            and positive(inference, "sample_payload_bytes")
            and positive(inference, "witness_payload_bytes")
            and balanced_artifact_recompute(inference)
        ),
        "evaluation_replay": (
            positive(evaluation, "model_payload_bytes")
            and positive(evaluation, "sample_payload_bytes")
            and positive(evaluation, "witness_payload_bytes")
            and balanced_artifact_recompute(evaluation)
        ),
        "balanced_recompute": (
            int(profile["total_forward_recompute_steps"]) > 0
            and int(profile["total_forward_recompute_steps"])
            == int(profile["total_inverse_recompute_steps"])
        ),
        "bounded_payload_ratios": (
            int(profile["total_model_payload_bytes"]) > 0
            and int(profile["total_witness_payload_bytes"]) > 0
            and int(profile["total_trace_payload_bytes"]) > 0
        ),
        "fingerprint_coverage": (
            bool(known_artifacts) and all(fingerprinted(artifact) for artifact in known_artifacts)
        ),
        "proof_or_provenance_fingerprints": (
            has_proof(training_trace)
            and has_provenance(model)
            and has_proof(sample_set)
            and has_proof(training_step)
            and has_proof(inference)
            and has_proof(evaluation)
        ),
    }


def validate_audit_contract(report: dict[str, Any], required: bool) -> None:
    contract = report.get("audit_contract")
    if contract is None:
        if required:
            raise ValueError("artifact comparison audit_contract is required")
        return
    if not isinstance(contract, dict):
        raise ValueError("audit_contract must be an object")
    claim = json_field(contract, "claim", "audit_contract")
    if claim != AUDIT_CONTRACT_CLAIM:
        raise ValueError(f"audit_contract.claim expected `{AUDIT_CONTRACT_CLAIM}`")
    passed = expect_bool(
        json_field(contract, "passed", "audit_contract"),
        "audit_contract.passed",
    )
    checks = json_field(contract, "checks", "audit_contract")
    if not isinstance(checks, list):
        raise ValueError("audit_contract.checks must be an array")
    expected_results = expected_audit_contract_results(report)
    seen: dict[str, bool] = {}
    for index, check in enumerate(checks):
        context = f"audit_contract.checks[{index}]"
        metric = json_field(check, "metric", context)
        if not isinstance(metric, str):
            raise ValueError(f"{context}.metric must be a string")
        if metric in seen:
            raise ValueError(f"audit_contract has duplicate check `{metric}`")
        actual = json_field(check, "actual", context)
        if not isinstance(actual, str):
            raise ValueError(f"{context}.actual must be a string")
        requirement = json_field(check, "requirement", context)
        if not isinstance(requirement, str) or not requirement:
            raise ValueError(f"{context}.requirement must be a non-empty string")
        check_passed = expect_bool(json_field(check, "passed", context), f"{context}.passed")
        if metric not in expected_results:
            raise ValueError(f"audit_contract has unknown check `{metric}`")
        if check_passed != expected_results[metric]:
            raise ValueError(f"audit_contract check `{metric}` does not match artifact rows")
        seen[metric] = check_passed
    missing = [metric for metric in AUDIT_CONTRACT_METRICS if metric not in seen]
    if missing:
        raise ValueError("audit_contract missing check(s): " + ", ".join(missing))
    expected_passed = all(expected_results.values())
    if passed != expected_passed:
        raise ValueError("audit_contract.passed does not match artifact rows")
    if required and not passed:
        failed = [metric for metric, result in expected_results.items() if not result]
        raise ValueError("audit_contract failed check(s): " + ", ".join(failed))


def validate_string_list(value: Any, context: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    if not allow_empty and not value:
        raise ValueError(f"{context} must not be empty")
    for index, item in enumerate(value):
        if not isinstance(item, str):
                raise ValueError(f"{context}[{index}] must be a string")
    return value


def expected_mlp_dataset_loops() -> list[dict[str, Any]]:
    return [{"index": "sample", "size_sources": ["labels"]}]


def validate_mlp_dataset_loops(value: Any, context: str) -> None:
    if value != expected_mlp_dataset_loops():
        raise ValueError(f"{context} must identify sample bounded by labels")


def validate_mlp_witness_report(report: dict[str, Any]) -> None:
    samples = expect_positive_int(json_field(report, "samples", "MLP witness report"), "samples")
    q31_one = expect_positive_int(json_field(report, "q31_one", "MLP witness report"), "q31_one")
    if q31_one != 1 << 31:
        raise ValueError("MLP witness report q31_one must equal 1 << 31")
    json_field(report, "program", "MLP witness report")
    validate_mlp_dataset_loops(
        json_field(report, "dataset_loops", "MLP witness report"),
        "MLP witness report.dataset_loops",
    )
    checked_fields = validate_string_list(
        json_field(report, "checked_store_fields", "MLP witness report"),
        "checked_store_fields",
    )
    predictions = json_field(report, "predictions", "MLP witness report")
    if not isinstance(predictions, list) or len(predictions) != samples:
        raise ValueError("MLP witness report predictions length must match samples")
    for index, prediction in enumerate(predictions):
        actual = expect_nonnegative_int(prediction, f"predictions[{index}]")
        if actual >= 10:
            raise ValueError(f"predictions[{index}] must be in 0..9")
    correct = json_field(report, "correct", "MLP witness report")
    if not isinstance(correct, list) or len(correct) != samples:
        raise ValueError("MLP witness report correct length must match samples")
    for index, item in enumerate(correct):
        expect_bool(item, f"correct[{index}]")
    mismatches = validate_string_list(
        json_field(report, "mismatches", "MLP witness report"),
        "mismatches",
        allow_empty=True,
    )
    passed = expect_bool(json_field(report, "passed", "MLP witness report"), "passed")
    if passed != (not mismatches):
        raise ValueError("MLP witness report passed must match mismatches emptiness")

    proof = json_field(report, "proof", "MLP witness report")
    claim = json_field(proof, "claim", "MLP proof")
    if claim != MLP_WITNESS_PROOF_CLAIM:
        raise ValueError(f"MLP proof claim must be `{MLP_WITNESS_PROOF_CLAIM}`")
    proof_checked_fields = expect_positive_int(
        json_field(proof, "checked_store_fields", "MLP proof"),
        "proof.checked_store_fields",
    )
    if proof_checked_fields != len(checked_fields):
        raise ValueError("MLP proof checked_store_fields must match checked_store_fields length")
    witness_tape_fields = validate_string_list(
        json_field(proof, "witness_tape_fields", "MLP proof"),
        "proof.witness_tape_fields",
    )

    model_payload_bytes = expect_positive_int(
        json_field(proof, "model_payload_bytes", "MLP proof"),
        "proof.model_payload_bytes",
    )
    sample_payload_bytes = expect_positive_int(
        json_field(proof, "sample_payload_bytes", "MLP proof"),
        "proof.sample_payload_bytes",
    )
    witness_payload_bytes = expect_positive_int(
        json_field(proof, "witness_payload_bytes", "MLP proof"),
        "proof.witness_payload_bytes",
    )
    trace_payload_bytes = expect_positive_int(
        json_field(proof, "trace_payload_bytes", "MLP proof"),
        "proof.trace_payload_bytes",
    )
    replay_payload_bytes = expect_positive_int(
        json_field(proof, "replay_payload_bytes", "MLP proof"),
        "proof.replay_payload_bytes",
    )
    witness_bytes_per_sample = expect_positive_int(
        json_field(proof, "witness_bytes_per_sample", "MLP proof"),
        "proof.witness_bytes_per_sample",
    )
    trace_bytes_per_sample = expect_positive_int(
        json_field(proof, "trace_bytes_per_sample", "MLP proof"),
        "proof.trace_bytes_per_sample",
    )
    stored_update_payload_bytes = expect_nonnegative_int(
        json_field(proof, "stored_derived_update_payload_bytes", "MLP proof"),
        "proof.stored_derived_update_payload_bytes",
    )
    recomputed_update_payload_bytes = expect_positive_int(
        json_field(proof, "recomputed_update_payload_bytes", "MLP proof"),
        "proof.recomputed_update_payload_bytes",
    )
    recomputed_update_bytes_per_sample = expect_positive_int(
        json_field(proof, "recomputed_update_bytes_per_sample", "MLP proof"),
        "proof.recomputed_update_bytes_per_sample",
    )
    forward_recompute_steps = expect_nonnegative_int(
        json_field(proof, "forward_recompute_steps", "MLP proof"),
        "proof.forward_recompute_steps",
    )
    inverse_recompute_steps = expect_nonnegative_int(
        json_field(proof, "inverse_recompute_steps", "MLP proof"),
        "proof.inverse_recompute_steps",
    )
    total_recompute_steps = expect_nonnegative_int(
        json_field(proof, "total_recompute_steps", "MLP proof"),
        "proof.total_recompute_steps",
    )
    if witness_payload_bytes != samples * witness_bytes_per_sample:
        raise ValueError("MLP proof witness_payload_bytes must equal samples * witness_bytes_per_sample")
    if trace_payload_bytes != sample_payload_bytes + witness_payload_bytes:
        raise ValueError("MLP proof trace_payload_bytes must equal sample + witness bytes")
    if trace_payload_bytes != samples * trace_bytes_per_sample:
        raise ValueError("MLP proof trace_payload_bytes must equal samples * trace_bytes_per_sample")
    if replay_payload_bytes != model_payload_bytes + trace_payload_bytes:
        raise ValueError("MLP proof replay_payload_bytes must equal model + trace bytes")
    if stored_update_payload_bytes != 0:
        raise ValueError("MLP proof stored_derived_update_payload_bytes must be zero")
    if recomputed_update_payload_bytes != samples * recomputed_update_bytes_per_sample:
        raise ValueError("MLP proof recomputed_update_payload_bytes must equal samples * per-sample bytes")
    if forward_recompute_steps != samples or inverse_recompute_steps != samples:
        raise ValueError("MLP proof forward/inverse recompute steps must match samples")
    if total_recompute_steps != forward_recompute_steps + inverse_recompute_steps:
        raise ValueError("MLP proof total_recompute_steps must equal forward + inverse")
    expect_ratio(
        json_field(proof, "witness_to_model_payload_ratio", "MLP proof"),
        witness_payload_bytes,
        model_payload_bytes,
        "proof.witness_to_model_payload_ratio",
    )
    expect_ratio(
        json_field(proof, "trace_to_model_payload_ratio", "MLP proof"),
        trace_payload_bytes,
        model_payload_bytes,
        "proof.trace_to_model_payload_ratio",
    )
    expect_ratio(
        json_field(proof, "recomputed_update_to_witness_payload_ratio", "MLP proof"),
        recomputed_update_payload_bytes,
        witness_payload_bytes,
        "proof.recomputed_update_to_witness_payload_ratio",
    )
    witness_proof = report.get("witness_proof")
    if witness_proof is None:
        if report.get("run_output_json") is not None:
            raise ValueError("MLP witness report with run_output_json must include witness_proof")
    else:
        validate_mlp_witness_store_proof(
            witness_proof,
            "MLP witness report.witness_proof",
            witness_tape_fields=witness_tape_fields,
            expected_payload_bytes=witness_payload_bytes,
        )


def validate_mlp_witness_store_proof(
    value: Any,
    context: str,
    *,
    witness_tape_fields: list[str],
    expected_payload_bytes: int,
) -> str:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if json_field(value, "schema", context) != "reverie_witness_store_proof_v1":
        raise ValueError(f"{context}.schema must be reverie_witness_store_proof_v1")
    if json_field(value, "algorithm", context) != "sha256":
        raise ValueError(f"{context}.algorithm must be sha256")
    fingerprint = validate_sha256(json_field(value, "fingerprint", context), f"{context}.fingerprint")
    payload = json_field(value, "payload", context)
    if not isinstance(payload, dict):
        raise ValueError(f"{context}.payload must be an object")
    if fingerprint != sha256_json(payload):
        raise ValueError(f"{context}.fingerprint must match payload")
    if json_field(payload, "schema", f"{context}.payload") != "reverie_witness_store_proof_v1":
        raise ValueError(f"{context}.payload.schema must be reverie_witness_store_proof_v1")
    if json_field(payload, "algorithm", f"{context}.payload") != "sha256":
        raise ValueError(f"{context}.payload.algorithm must be sha256")
    variables = expect_positive_int(json_field(payload, "variables", f"{context}.payload"), f"{context}.payload.variables")
    if variables != len(witness_tape_fields):
        raise ValueError(f"{context}.payload.variables must match witness tape field count")
    known_payload_bytes = expect_positive_int(
        json_field(payload, "known_payload_bytes", f"{context}.payload"),
        f"{context}.payload.known_payload_bytes",
    )
    if known_payload_bytes != expected_payload_bytes:
        raise ValueError(f"{context}.payload.known_payload_bytes must match MLP witness payload bytes")
    known_cells = expect_positive_int(
        json_field(payload, "known_cells", f"{context}.payload"),
        f"{context}.payload.known_cells",
    )
    if known_cells * I64_BYTES != known_payload_bytes:
        raise ValueError(f"{context}.payload.known_cells must match known payload bytes")
    unknown = json_field(payload, "unknown_variables", f"{context}.payload")
    if unknown != []:
        raise ValueError(f"{context}.payload.unknown_variables must be empty")
    entries = json_field(payload, "entries", f"{context}.payload")
    if not isinstance(entries, list) or len(entries) != variables:
        raise ValueError(f"{context}.payload.entries must match variables")
    expected_names = set(witness_tape_fields)
    seen: set[str] = set()
    entry_cells = 0
    entry_payload_bytes = 0
    for index, entry in enumerate(entries):
        entry_context = f"{context}.payload.entries[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{entry_context} must be an object")
        name = json_field(entry, "name", entry_context)
        if not isinstance(name, str) or not name:
            raise ValueError(f"{entry_context}.name must be a non-empty string")
        if name in seen:
            raise ValueError(f"{context}.payload.entries duplicate witness `{name}`")
        seen.add(name)
        if not expect_bool(json_field(entry, "present", entry_context), f"{entry_context}.present"):
            raise ValueError(f"{entry_context}.present must be true")
        cells = expect_positive_int(json_field(entry, "cells", entry_context), f"{entry_context}.cells")
        payload_bytes = expect_positive_int(
            json_field(entry, "payload_bytes", entry_context),
            f"{entry_context}.payload_bytes",
        )
        if cells * I64_BYTES != payload_bytes:
            raise ValueError(f"{entry_context}.cells must match payload_bytes")
        validate_sha256(
            json_field(entry, "value_fingerprint", entry_context),
            f"{entry_context}.value_fingerprint",
        )
        entry_cells += cells
        entry_payload_bytes += payload_bytes
    if seen != expected_names:
        raise ValueError(f"{context}.payload.entries must match MLP witness tape fields")
    if entry_cells != known_cells or entry_payload_bytes != known_payload_bytes:
        raise ValueError(f"{context}.payload.entries must sum to known witness size")
    return fingerprint


def validate_coupling_vector(value: Any, context: str) -> list[int]:
    if not isinstance(value, list) or len(value) != coupling_check.WIDTH:
        raise ValueError(f"{context} must have {coupling_check.WIDTH} entries")
    return [expect_i64(item, f"{context}[{index}]") for index, item in enumerate(value)]


def validate_coupling_state(value: Any, context: str) -> dict[str, list[int]]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return {
        "left": validate_coupling_vector(json_field(value, "left", context), f"{context}.left"),
        "right": validate_coupling_vector(json_field(value, "right", context), f"{context}.right"),
    }


def validate_invertible_coupling_proof(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    claim = json_field(value, "claim", context)
    if claim != INVERTIBLE_COUPLING_PROOF_CLAIM:
        raise ValueError(f"{context}.claim must be `{INVERTIBLE_COUPLING_PROOF_CLAIM}`")
    if json_field(value, "arithmetic", context) != "q31_wrapping_i64":
        raise ValueError(f"{context}.arithmetic must be q31_wrapping_i64")
    model_payload_bytes = expect_positive_int(
        json_field(value, "model_payload_bytes", context),
        f"{context}.model_payload_bytes",
    )
    state_payload_bytes = expect_positive_int(
        json_field(value, "state_payload_bytes", context),
        f"{context}.state_payload_bytes",
    )
    witness_payload_bytes = expect_nonnegative_int(
        json_field(value, "witness_payload_bytes", context),
        f"{context}.witness_payload_bytes",
    )
    trace_payload_bytes = expect_nonnegative_int(
        json_field(value, "trace_payload_bytes", context),
        f"{context}.trace_payload_bytes",
    )
    if witness_payload_bytes != 0 or trace_payload_bytes != 0:
        raise ValueError(f"{context} witness and trace payload bytes must be zero")
    replay_payload_bytes = expect_positive_int(
        json_field(value, "replay_payload_bytes", context),
        f"{context}.replay_payload_bytes",
    )
    if replay_payload_bytes != model_payload_bytes + state_payload_bytes:
        raise ValueError(f"{context}.replay_payload_bytes must equal model + state bytes")
    forward = expect_positive_int(
        json_field(value, "forward_recompute_steps", context),
        f"{context}.forward_recompute_steps",
    )
    inverse = expect_positive_int(
        json_field(value, "inverse_recompute_steps", context),
        f"{context}.inverse_recompute_steps",
    )
    total = expect_positive_int(
        json_field(value, "total_recompute_steps", context),
        f"{context}.total_recompute_steps",
    )
    if forward != inverse or total != forward + inverse:
        raise ValueError(f"{context} recompute steps must be balanced")
    expect_ratio(
        json_field(value, "witness_to_model_payload_ratio", context),
        witness_payload_bytes,
        model_payload_bytes,
        f"{context}.witness_to_model_payload_ratio",
    )
    expect_ratio(
        json_field(value, "trace_to_model_payload_ratio", context),
        trace_payload_bytes,
        model_payload_bytes,
        f"{context}.trace_to_model_payload_ratio",
    )
    checks = json_field(value, "checks", context)
    if not isinstance(checks, dict):
        raise ValueError(f"{context}.checks must be an object")
    for field in (
        "forward_matches_reference",
        "reverse_restores_initial_state",
        "no_witness_tape",
        "balanced_recompute",
    ):
        if not expect_bool(json_field(checks, field, f"{context}.checks"), f"{context}.checks.{field}"):
            raise ValueError(f"{context}.checks.{field} must be true")


def validate_invertible_coupling_report(report: dict[str, Any]) -> None:
    if json_field(report, "program", "coupling report") != coupling_check.PROGRAM:
        raise ValueError(f"program must be `{coupling_check.PROGRAM}`")
    validate_q31_constant(json_field(report, "q31_one", "coupling report"), "q31_one")
    initial = validate_coupling_state(json_field(report, "initial", "coupling report"), "initial")
    forward = validate_coupling_state(json_field(report, "forward", "coupling report"), "forward")
    expected_initial = coupling_check.initial_store()
    expected_forward = coupling_check.forward_store(expected_initial)
    if initial != {"left": expected_initial["left"], "right": expected_initial["right"]}:
        raise ValueError("initial coupling state does not match reference")
    if forward != {"left": expected_forward["left"], "right": expected_forward["right"]}:
        raise ValueError("forward coupling state does not match reference")
    validate_nonempty_string(
        json_field(report, "forward_output_json", "coupling report"),
        "forward_output_json",
    )
    validate_nonempty_string(
        json_field(report, "reverse_output_json", "coupling report"),
        "reverse_output_json",
    )
    mismatches = validate_string_list(
        json_field(report, "mismatches", "coupling report"),
        "mismatches",
        allow_empty=True,
    )
    passed = expect_bool(json_field(report, "passed", "coupling report"), "passed")
    if passed != (not mismatches):
        raise ValueError("coupling report passed must match mismatches emptiness")
    if not passed:
        raise ValueError("coupling report must pass")
    validate_invertible_coupling_proof(json_field(report, "proof", "coupling report"), "proof")


def validate_residual_vector(value: Any, context: str) -> list[int]:
    if not isinstance(value, list) or len(value) != residual_check.WIDTH:
        raise ValueError(f"{context} must have {residual_check.WIDTH} entries")
    return [expect_i64(item, f"{context}[{index}]") for index, item in enumerate(value)]


def validate_residual_parameters(value: Any, context: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return {
        field: expect_i64(json_field(value, field, context), f"{context}.{field}")
        for field in residual_check.PARAM_FIELDS
    }


def validate_triangular_residual_proof(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    claim = json_field(value, "claim", context)
    if claim != TRIANGULAR_RESIDUAL_PROOF_CLAIM:
        raise ValueError(f"{context}.claim must be `{TRIANGULAR_RESIDUAL_PROOF_CLAIM}`")
    if json_field(value, "arithmetic", context) != "q31_wrapping_i64":
        raise ValueError(f"{context}.arithmetic must be q31_wrapping_i64")
    parameter_payload_bytes = expect_positive_int(
        json_field(value, "parameter_payload_bytes", context),
        f"{context}.parameter_payload_bytes",
    )
    state_payload_bytes = expect_positive_int(
        json_field(value, "state_payload_bytes", context),
        f"{context}.state_payload_bytes",
    )
    witness_payload_bytes = expect_nonnegative_int(
        json_field(value, "witness_payload_bytes", context),
        f"{context}.witness_payload_bytes",
    )
    trace_payload_bytes = expect_nonnegative_int(
        json_field(value, "trace_payload_bytes", context),
        f"{context}.trace_payload_bytes",
    )
    if witness_payload_bytes != 0 or trace_payload_bytes != 0:
        raise ValueError(f"{context} witness and trace payload bytes must be zero")
    replay_payload_bytes = expect_positive_int(
        json_field(value, "replay_payload_bytes", context),
        f"{context}.replay_payload_bytes",
    )
    if replay_payload_bytes != parameter_payload_bytes + state_payload_bytes:
        raise ValueError(f"{context}.replay_payload_bytes must equal parameter + state bytes")
    forward = expect_positive_int(
        json_field(value, "forward_recompute_steps", context),
        f"{context}.forward_recompute_steps",
    )
    inverse = expect_positive_int(
        json_field(value, "inverse_recompute_steps", context),
        f"{context}.inverse_recompute_steps",
    )
    total = expect_positive_int(
        json_field(value, "total_recompute_steps", context),
        f"{context}.total_recompute_steps",
    )
    if forward != inverse or total != forward + inverse:
        raise ValueError(f"{context} recompute steps must be balanced")
    expect_ratio(
        json_field(value, "witness_to_parameter_payload_ratio", context),
        witness_payload_bytes,
        parameter_payload_bytes,
        f"{context}.witness_to_parameter_payload_ratio",
    )
    expect_ratio(
        json_field(value, "trace_to_parameter_payload_ratio", context),
        trace_payload_bytes,
        parameter_payload_bytes,
        f"{context}.trace_to_parameter_payload_ratio",
    )
    checks = json_field(value, "checks", context)
    if not isinstance(checks, dict):
        raise ValueError(f"{context}.checks must be an object")
    for field in (
        "forward_matches_reference",
        "reverse_restores_initial_state",
        "triangular_source_order",
        "no_witness_tape",
        "balanced_recompute",
    ):
        if not expect_bool(json_field(checks, field, f"{context}.checks"), f"{context}.checks.{field}"):
            raise ValueError(f"{context}.checks.{field} must be true")


def validate_triangular_residual_report(report: dict[str, Any]) -> None:
    if json_field(report, "program", "residual report") != residual_check.PROGRAM:
        raise ValueError(f"program must be `{residual_check.PROGRAM}`")
    validate_q31_constant(json_field(report, "q31_one", "residual report"), "q31_one")
    initial = json_field(report, "initial", "residual report")
    if not isinstance(initial, dict):
        raise ValueError("residual report.initial must be an object")
    initial_x = validate_residual_vector(json_field(initial, "x", "initial"), "initial.x")
    parameters = validate_residual_parameters(
        json_field(initial, "parameters", "initial"),
        "initial.parameters",
    )
    forward_x = validate_residual_vector(
        json_field(json_field(report, "forward", "residual report"), "x", "forward"),
        "forward.x",
    )
    expected_initial = residual_check.initial_store()
    expected_forward = residual_check.forward_store(expected_initial)
    if initial_x != expected_initial["x"]:
        raise ValueError("initial residual state does not match reference")
    if parameters != {field: expected_initial[field] for field in residual_check.PARAM_FIELDS}:
        raise ValueError("initial residual parameters do not match reference")
    if forward_x != expected_forward["x"]:
        raise ValueError("forward residual state does not match reference")
    validate_nonempty_string(
        json_field(report, "forward_output_json", "residual report"),
        "forward_output_json",
    )
    validate_nonempty_string(
        json_field(report, "reverse_output_json", "residual report"),
        "reverse_output_json",
    )
    mismatches = validate_string_list(
        json_field(report, "mismatches", "residual report"),
        "mismatches",
        allow_empty=True,
    )
    passed = expect_bool(json_field(report, "passed", "residual report"), "passed")
    if passed != (not mismatches):
        raise ValueError("residual report passed must match mismatches emptiness")
    if not passed:
        raise ValueError("residual report must pass")
    validate_triangular_residual_proof(json_field(report, "proof", "residual report"), "proof")


def validate_preprocess_vector(value: Any, context: str) -> list[int]:
    if not isinstance(value, list) or len(value) != preprocess_check.WIDTH:
        raise ValueError(f"{context} must have {preprocess_check.WIDTH} entries")
    return [expect_i64(item, f"{context}[{index}]") for index, item in enumerate(value)]


def validate_preprocess_state(value: Any, context: str) -> dict[str, list[int]]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return {
        "raw": validate_preprocess_vector(json_field(value, "raw", context), f"{context}.raw"),
        "mean": validate_preprocess_vector(json_field(value, "mean", context), f"{context}.mean"),
        "features": validate_preprocess_vector(
            json_field(value, "features", context),
            f"{context}.features",
        ),
    }


def validate_reversible_preprocess_proof(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    claim = json_field(value, "claim", context)
    if claim != REVERSIBLE_PREPROCESS_PROOF_CLAIM:
        raise ValueError(f"{context}.claim must be `{REVERSIBLE_PREPROCESS_PROOF_CLAIM}`")
    if json_field(value, "arithmetic", context) != "q31_wrapping_i64":
        raise ValueError(f"{context}.arithmetic must be q31_wrapping_i64")
    raw_payload_bytes = expect_positive_int(
        json_field(value, "raw_payload_bytes", context),
        f"{context}.raw_payload_bytes",
    )
    mean_payload_bytes = expect_positive_int(
        json_field(value, "mean_payload_bytes", context),
        f"{context}.mean_payload_bytes",
    )
    feature_payload_bytes = expect_positive_int(
        json_field(value, "feature_payload_bytes", context),
        f"{context}.feature_payload_bytes",
    )
    witness_payload_bytes = expect_nonnegative_int(
        json_field(value, "witness_payload_bytes", context),
        f"{context}.witness_payload_bytes",
    )
    trace_payload_bytes = expect_nonnegative_int(
        json_field(value, "trace_payload_bytes", context),
        f"{context}.trace_payload_bytes",
    )
    if witness_payload_bytes != 0 or trace_payload_bytes != 0:
        raise ValueError(f"{context} witness and trace payload bytes must be zero")
    replay_payload_bytes = expect_positive_int(
        json_field(value, "replay_payload_bytes", context),
        f"{context}.replay_payload_bytes",
    )
    state_payload_bytes = raw_payload_bytes + mean_payload_bytes + feature_payload_bytes
    if replay_payload_bytes != state_payload_bytes:
        raise ValueError(f"{context}.replay_payload_bytes must equal raw + mean + feature bytes")
    forward = expect_positive_int(
        json_field(value, "forward_recompute_steps", context),
        f"{context}.forward_recompute_steps",
    )
    inverse = expect_positive_int(
        json_field(value, "inverse_recompute_steps", context),
        f"{context}.inverse_recompute_steps",
    )
    total = expect_positive_int(
        json_field(value, "total_recompute_steps", context),
        f"{context}.total_recompute_steps",
    )
    if forward != inverse or total != forward + inverse:
        raise ValueError(f"{context} recompute steps must be balanced")
    expect_ratio(
        json_field(value, "witness_to_state_payload_ratio", context),
        witness_payload_bytes,
        state_payload_bytes,
        f"{context}.witness_to_state_payload_ratio",
    )
    expect_ratio(
        json_field(value, "trace_to_state_payload_ratio", context),
        trace_payload_bytes,
        state_payload_bytes,
        f"{context}.trace_to_state_payload_ratio",
    )
    checks = json_field(value, "checks", context)
    if not isinstance(checks, dict):
        raise ValueError(f"{context}.checks must be an object")
    for field in (
        "forward_matches_reference",
        "reverse_restores_initial_state",
        "raw_preserved",
        "mean_preserved",
        "no_witness_tape",
        "balanced_recompute",
    ):
        if not expect_bool(json_field(checks, field, f"{context}.checks"), f"{context}.checks.{field}"):
            raise ValueError(f"{context}.checks.{field} must be true")


def validate_reversible_preprocess_report(report: dict[str, Any]) -> None:
    if json_field(report, "program", "preprocess report") != preprocess_check.PROGRAM:
        raise ValueError(f"program must be `{preprocess_check.PROGRAM}`")
    validate_q31_constant(json_field(report, "q31_one", "preprocess report"), "q31_one")
    initial = validate_preprocess_state(json_field(report, "initial", "preprocess report"), "initial")
    forward = validate_preprocess_state(json_field(report, "forward", "preprocess report"), "forward")
    expected_initial = preprocess_check.initial_store()
    expected_forward = preprocess_check.forward_store(expected_initial)
    if initial != expected_initial:
        raise ValueError("initial preprocessing state does not match reference")
    if forward != expected_forward:
        raise ValueError("forward preprocessing state does not match reference")
    validate_nonempty_string(
        json_field(report, "forward_output_json", "preprocess report"),
        "forward_output_json",
    )
    validate_nonempty_string(
        json_field(report, "reverse_output_json", "preprocess report"),
        "reverse_output_json",
    )
    mismatches = validate_string_list(
        json_field(report, "mismatches", "preprocess report"),
        "mismatches",
        allow_empty=True,
    )
    passed = expect_bool(json_field(report, "passed", "preprocess report"), "passed")
    if passed != (not mismatches):
        raise ValueError("preprocess report passed must match mismatches emptiness")
    if not passed:
        raise ValueError("preprocess report must pass")
    validate_reversible_preprocess_proof(json_field(report, "proof", "preprocess report"), "proof")


def validate_inference_trace_vector(value: Any, length: int, context: str) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{context} must have {length} entries")
    return [expect_i64(item, f"{context}[{index}]") for index, item in enumerate(value)]


def validate_inference_trace_matrix(value: Any, rows: int, cols: int, context: str) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{context} must have {rows} rows")
    return [
        validate_inference_trace_vector(row, cols, f"{context}[{index}]")
        for index, row in enumerate(value)
    ]


def validate_inference_trace_state(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return {
        "raw": validate_inference_trace_vector(
            json_field(value, "raw", context),
            inference_trace_check.FEATURES,
            f"{context}.raw",
        ),
        "mean": validate_inference_trace_vector(
            json_field(value, "mean", context),
            inference_trace_check.FEATURES,
            f"{context}.mean",
        ),
        "features": validate_inference_trace_vector(
            json_field(value, "features", context),
            inference_trace_check.FEATURES,
            f"{context}.features",
        ),
        "weights": validate_inference_trace_matrix(
            json_field(value, "weights", context),
            inference_trace_check.FEATURES,
            inference_trace_check.CLASSES,
            f"{context}.weights",
        ),
        "bias": validate_inference_trace_vector(
            json_field(value, "bias", context),
            inference_trace_check.CLASSES,
            f"{context}.bias",
        ),
        "logits": validate_inference_trace_vector(
            json_field(value, "logits", context),
            inference_trace_check.CLASSES,
            f"{context}.logits",
        ),
        "top_classes": validate_inference_trace_vector(
            json_field(value, "top_classes", context),
            inference_trace_check.CLASSES,
            f"{context}.top_classes",
        ),
        "top_logit_values": validate_inference_trace_vector(
            json_field(value, "top_logit_values", context),
            inference_trace_check.CLASSES,
            f"{context}.top_logit_values",
        ),
        "prediction": expect_i64(json_field(value, "prediction", context), f"{context}.prediction"),
        "runner_up_class": expect_i64(
            json_field(value, "runner_up_class", context),
            f"{context}.runner_up_class",
        ),
        "margin": expect_i64(json_field(value, "margin", context), f"{context}.margin"),
        "label_rank": expect_i64(
            json_field(value, "label_rank", context),
            f"{context}.label_rank",
        ),
        "correct": expect_i64(json_field(value, "correct", context), f"{context}.correct"),
        "top2_correct": expect_i64(
            json_field(value, "top2_correct", context),
            f"{context}.top2_correct",
        ),
        "label": expect_i64(json_field(value, "label", context), f"{context}.label"),
    }


def validate_inference_trace_top_logits(
    value: Any,
    logits: list[int],
    context: str,
) -> list[dict[str, int]]:
    if not isinstance(value, list) or len(value) != inference_trace_check.CLASSES:
        raise ValueError(f"{context} must contain every ranked class")
    checked = []
    seen = set()
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{item_context} must be an object")
        class_index = expect_nonnegative_int(json_field(item, "class", item_context), f"{item_context}.class")
        if class_index >= inference_trace_check.CLASSES:
            raise ValueError(f"{item_context}.class must be less than class count")
        if class_index in seen:
            raise ValueError(f"{context} contains duplicate class {class_index}")
        seen.add(class_index)
        checked.append(
            {
                "class": class_index,
                "value": expect_i64(json_field(item, "value", item_context), f"{item_context}.value"),
            }
        )
    expected = inference_trace_check.top_logits(logits)
    if checked != expected:
        raise ValueError(f"{context} must match reference top-logit ordering")
    return checked


def validate_inference_trace_attribution(
    value: Any,
    forward: dict[str, Any],
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    expected = inference_trace_check.inference_attribution(forward)
    if value != expected:
        raise ValueError(f"{context} must match reference contribution ledger")
    validate_sha256(
        json_field(value, "contribution_ledger_fingerprint", context),
        f"{context}.contribution_ledger_fingerprint",
    )
    validate_sha256(
        json_field(value, "margin_contribution_ledger_fingerprint", context),
        f"{context}.margin_contribution_ledger_fingerprint",
    )
    return expected


def validate_reversible_inference_trace_proof(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    claim = json_field(value, "claim", context)
    if claim != REVERSIBLE_INFERENCE_TRACE_PROOF_CLAIM:
        raise ValueError(f"{context}.claim must be `{REVERSIBLE_INFERENCE_TRACE_PROOF_CLAIM}`")
    if json_field(value, "arithmetic", context) != "q31_wrapping_i64":
        raise ValueError(f"{context}.arithmetic must be q31_wrapping_i64")

    raw_payload_bytes = expect_positive_int(json_field(value, "raw_payload_bytes", context), f"{context}.raw_payload_bytes")
    mean_payload_bytes = expect_positive_int(json_field(value, "mean_payload_bytes", context), f"{context}.mean_payload_bytes")
    feature_payload_bytes = expect_positive_int(json_field(value, "feature_payload_bytes", context), f"{context}.feature_payload_bytes")
    weight_payload_bytes = expect_positive_int(json_field(value, "weight_payload_bytes", context), f"{context}.weight_payload_bytes")
    bias_payload_bytes = expect_positive_int(json_field(value, "bias_payload_bytes", context), f"{context}.bias_payload_bytes")
    model_payload_bytes = expect_positive_int(json_field(value, "model_payload_bytes", context), f"{context}.model_payload_bytes")
    logit_payload_bytes = expect_positive_int(json_field(value, "logit_payload_bytes", context), f"{context}.logit_payload_bytes")
    top_class_payload_bytes = expect_positive_int(
        json_field(value, "top_class_payload_bytes", context),
        f"{context}.top_class_payload_bytes",
    )
    top_logit_value_payload_bytes = expect_positive_int(
        json_field(value, "top_logit_value_payload_bytes", context),
        f"{context}.top_logit_value_payload_bytes",
    )
    prediction_payload_bytes = expect_positive_int(
        json_field(value, "prediction_payload_bytes", context),
        f"{context}.prediction_payload_bytes",
    )
    runner_up_payload_bytes = expect_positive_int(
        json_field(value, "runner_up_payload_bytes", context),
        f"{context}.runner_up_payload_bytes",
    )
    margin_payload_bytes = expect_positive_int(
        json_field(value, "margin_payload_bytes", context),
        f"{context}.margin_payload_bytes",
    )
    label_rank_payload_bytes = expect_positive_int(
        json_field(value, "label_rank_payload_bytes", context),
        f"{context}.label_rank_payload_bytes",
    )
    correct_payload_bytes = expect_positive_int(
        json_field(value, "correct_payload_bytes", context),
        f"{context}.correct_payload_bytes",
    )
    top_k_correct_payload_bytes = expect_positive_int(
        json_field(value, "top_k_correct_payload_bytes", context),
        f"{context}.top_k_correct_payload_bytes",
    )
    label_payload_bytes = expect_positive_int(json_field(value, "label_payload_bytes", context), f"{context}.label_payload_bytes")
    state_payload_bytes = expect_positive_int(json_field(value, "state_payload_bytes", context), f"{context}.state_payload_bytes")
    witness_payload_bytes = expect_positive_int(
        json_field(value, "witness_payload_bytes", context),
        f"{context}.witness_payload_bytes",
    )
    trace_payload_bytes = expect_nonnegative_int(
        json_field(value, "trace_payload_bytes", context),
        f"{context}.trace_payload_bytes",
    )
    replay_payload_bytes = expect_positive_int(
        json_field(value, "replay_payload_bytes", context),
        f"{context}.replay_payload_bytes",
    )
    if model_payload_bytes != weight_payload_bytes + bias_payload_bytes:
        raise ValueError(f"{context}.model_payload_bytes must equal weight + bias bytes")
    if state_payload_bytes != raw_payload_bytes + mean_payload_bytes + feature_payload_bytes + label_payload_bytes:
        raise ValueError(f"{context}.state_payload_bytes must equal raw + mean + feature + label bytes")
    if witness_payload_bytes != (
        logit_payload_bytes
        + top_class_payload_bytes
        + top_logit_value_payload_bytes
        + prediction_payload_bytes
        + runner_up_payload_bytes
        + margin_payload_bytes
        + label_rank_payload_bytes
        + correct_payload_bytes
        + top_k_correct_payload_bytes
    ):
        raise ValueError(
            f"{context}.witness_payload_bytes must equal logits + top-k classes + top-k values + prediction + runner-up + margin + label rank + correct + top-k correct bytes"
        )
    if replay_payload_bytes != state_payload_bytes + model_payload_bytes + witness_payload_bytes + trace_payload_bytes:
        raise ValueError(f"{context}.replay_payload_bytes must equal state + model + witness + trace bytes")
    if trace_payload_bytes != 0:
        raise ValueError(f"{context}.trace_payload_bytes must be zero")

    forward = expect_positive_int(json_field(value, "forward_recompute_steps", context), f"{context}.forward_recompute_steps")
    inverse = expect_positive_int(json_field(value, "inverse_recompute_steps", context), f"{context}.inverse_recompute_steps")
    total = expect_positive_int(json_field(value, "total_recompute_steps", context), f"{context}.total_recompute_steps")
    if forward != inverse or total != forward + inverse:
        raise ValueError(f"{context} recompute steps must be balanced")
    expect_ratio(
        json_field(value, "witness_to_model_payload_ratio", context),
        witness_payload_bytes,
        model_payload_bytes,
        f"{context}.witness_to_model_payload_ratio",
    )
    expect_ratio(
        json_field(value, "trace_to_model_payload_ratio", context),
        trace_payload_bytes,
        model_payload_bytes,
        f"{context}.trace_to_model_payload_ratio",
    )
    expect_ratio(
        json_field(value, "witness_to_state_payload_ratio", context),
        witness_payload_bytes,
        state_payload_bytes,
        f"{context}.witness_to_state_payload_ratio",
    )
    expect_ratio(
        json_field(value, "trace_to_state_payload_ratio", context),
        trace_payload_bytes,
        state_payload_bytes,
        f"{context}.trace_to_state_payload_ratio",
    )
    checks = json_field(value, "checks", context)
    if not isinstance(checks, dict):
        raise ValueError(f"{context}.checks must be an object")
    for field in (
        "preprocess_matches_reference",
        "logits_match_reference",
        "top_classes_match_reference",
        "top_logit_values_match_reference",
        "prediction_matches_reference",
        "runner_up_matches_reference",
        "margin_matches_reference",
        "label_rank_matches_reference",
        "correctness_matches_reference",
        "top_k_correctness_matches_reference",
        "reverse_restores_initial_state",
        "raw_preserved",
        "model_preserved",
        "balanced_recompute",
    ):
        if not expect_bool(json_field(checks, field, f"{context}.checks"), f"{context}.checks.{field}"):
            raise ValueError(f"{context}.checks.{field} must be true")


def validate_reversible_inference_trace_report(report: dict[str, Any]) -> None:
    if json_field(report, "program", "inference trace report") != inference_trace_check.PROGRAM:
        raise ValueError(f"program must be `{inference_trace_check.PROGRAM}`")
    validate_q31_constant(json_field(report, "q31_one", "inference trace report"), "q31_one")
    initial = validate_inference_trace_state(
        json_field(report, "initial", "inference trace report"),
        "initial",
    )
    forward = validate_inference_trace_state(
        json_field(report, "forward", "inference trace report"),
        "forward",
    )
    expected_initial = inference_trace_check.initial_store()
    expected_forward = inference_trace_check.forward_store(expected_initial)
    if initial != expected_initial:
        raise ValueError("initial inference trace state does not match reference")
    if forward != expected_forward:
        raise ValueError("forward inference trace state does not match reference")
    validate_inference_trace_top_logits(
        json_field(report, "top_logits", "inference trace report"),
        forward["logits"],
        "top_logits",
    )
    validate_inference_trace_attribution(
        json_field(report, "attribution", "inference trace report"),
        forward,
        "attribution",
    )
    validate_nonempty_string(
        json_field(report, "forward_output_json", "inference trace report"),
        "forward_output_json",
    )
    validate_nonempty_string(
        json_field(report, "reverse_output_json", "inference trace report"),
        "reverse_output_json",
    )
    mismatches = validate_string_list(
        json_field(report, "mismatches", "inference trace report"),
        "mismatches",
        allow_empty=True,
    )
    passed = expect_bool(json_field(report, "passed", "inference trace report"), "passed")
    if passed != (not mismatches):
        raise ValueError("inference trace report passed must match mismatches emptiness")
    if not passed:
        raise ValueError("inference trace report must pass")
    validate_reversible_inference_trace_proof(
        json_field(report, "proof", "inference trace report"),
        "proof",
    )


def validate_exact_key_object(value: Any, expected: tuple[str, ...], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    actual_keys = set(value)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing:
        raise ValueError(f"{context} missing key(s): {', '.join(missing)}")
    if extra:
        raise ValueError(f"{context} has unexpected key(s): {', '.join(extra)}")
    return value


def validate_string_map_keys(value: Any, expected: tuple[str, ...], context: str) -> dict[str, Any]:
    checked = validate_exact_key_object(value, expected, context)
    for key in expected:
        validate_nonempty_string(checked[key], f"{context}.{key}")
    return checked


def validate_pipeline_evidence_index(
    value: Any,
    expected_paths: dict[str, str],
    context: str,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if json_field(value, "algorithm", context) != "sha256":
        raise ValueError(f"{context}.algorithm must be sha256")
    files = validate_exact_key_object(json_field(value, "files", context), tuple(expected_paths), f"{context}.files")
    for key, expected_path in expected_paths.items():
        record = files[key]
        record_context = f"{context}.files.{key}"
        if not isinstance(record, dict):
            raise ValueError(f"{record_context} must be an object")
        path = validate_nonempty_string(json_field(record, "path", record_context), f"{record_context}.path")
        if path != expected_path:
            raise ValueError(f"{record_context}.path expected `{expected_path}`, found `{path}`")
        role = validate_nonempty_string(json_field(record, "role", record_context), f"{record_context}.role")
        if key == "profile_markdown":
            expected_role = "markdown"
        elif key == "summary":
            expected_role = "summary"
        elif key.startswith("report."):
            expected_role = "report"
        elif key.startswith("bundle."):
            expected_role = "bundle"
        elif key.startswith("standalone."):
            expected_role = "standalone"
        else:
            raise ValueError(f"{record_context} has unsupported evidence key")
        if role != expected_role:
            raise ValueError(f"{record_context}.role expected `{expected_role}`, found `{role}`")
        expect_positive_int(json_field(record, "bytes", record_context), f"{record_context}.bytes")
        validate_sha256(json_field(record, "sha256", record_context), f"{record_context}.sha256")


def pipeline_standalone_evidence_paths(value: Any, context: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    files = json_field(value, "files", context)
    if not isinstance(files, dict):
        raise ValueError(f"{context}.files must be an object")
    paths: dict[str, str] = {}
    for key in PIPELINE_STANDALONE_KEYS:
        evidence_key = f"standalone.{key}"
        record_context = f"{context}.files.{evidence_key}"
        record = json_field(files, evidence_key, f"{context}.files")
        if not isinstance(record, dict):
            raise ValueError(f"{record_context} must be an object")
        paths[evidence_key] = validate_nonempty_string(
            json_field(record, "path", record_context),
            f"{record_context}.path",
        )
    return paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"failed to read evidence file {path}: {error}") from error
    return digest.hexdigest()


def verify_pipeline_file_evidence(report: Any) -> None:
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    if report.get("kind") == MODEL_CAPSULE_KIND:
        evidence = json_field(
            json_field(report, "payload", "model capsule"),
            "evidence",
            "model capsule.payload",
        )
    elif report.get("kind") in (PIPELINE_SUMMARY_KIND, PIPELINE_KIND):
        evidence = json_field(report, "evidence", "pipeline report")
    else:
        return
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be an object")
    files = json_field(evidence, "files", "evidence")
    if not isinstance(files, dict):
        raise ValueError("evidence.files must be an object")
    for key, record in files.items():
        context = f"evidence.files.{key}"
        if not isinstance(record, dict):
            raise ValueError(f"{context} must be an object")
        path = Path(validate_nonempty_string(json_field(record, "path", context), f"{context}.path"))
        expected_bytes = expect_positive_int(json_field(record, "bytes", context), f"{context}.bytes")
        expected_sha = validate_sha256(json_field(record, "sha256", context), f"{context}.sha256")
        try:
            actual_bytes = path.stat().st_size
        except OSError as error:
            raise ValueError(f"failed to stat evidence file {path}: {error}") from error
        if actual_bytes != expected_bytes:
            raise ValueError(f"{context}.bytes expected {expected_bytes}, found {actual_bytes}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"{context}.sha256 expected {expected_sha}, found {actual_sha}")


def validate_pipeline_gate_policy(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    for field in (
        "min_train_accuracy_percent",
        "min_eval_accuracy_percent",
        "min_audit_accuracy_percent",
        "min_model_evaluation_accuracy_percent",
        "min_reference_accuracy_percent",
    ):
        actual = expect_finite_number(json_field(value, field, context), f"{context}.{field}")
        if not 0.0 <= actual <= 100.0:
            raise ValueError(f"{context}.{field} must be in 0..100")
    max_mismatches = expect_nonnegative_int(
        json_field(value, "max_witness_mismatches", context),
        f"{context}.max_witness_mismatches",
    )
    for field in (
        "max_trace_to_model_payload_ratio",
        "max_witness_to_model_payload_ratio",
        "max_reverse_train_elapsed_ratio",
    ):
        if expect_finite_number(json_field(value, field, context), f"{context}.{field}") < 0.0:
            raise ValueError(f"{context}.{field} must be non-negative")
    max_replay = json_field(value, "max_replay_payload_bytes", context)
    if max_replay is not None:
        expect_positive_int(max_replay, f"{context}.max_replay_payload_bytes")
    audit_step_strategy = validate_nonempty_string(
        json_field(value, "audit_step_strategy", context),
        f"{context}.audit_step_strategy",
    )
    if audit_step_strategy not in AUDIT_STEP_STRATEGIES:
        raise ValueError(f"{context}.audit_step_strategy is invalid")
    requested_audit_step = expect_nonnegative_int(
        json_field(value, "requested_audit_step", context),
        f"{context}.requested_audit_step",
    )
    evaluation_row_strategy = validate_nonempty_string(
        json_field(value, "evaluation_row_strategy", context),
        f"{context}.evaluation_row_strategy",
    )
    if evaluation_row_strategy not in EVALUATION_ROW_STRATEGIES:
        raise ValueError(f"{context}.evaluation_row_strategy is invalid")
    requested_evaluation_row = expect_nonnegative_int(
        json_field(value, "requested_evaluation_row", context),
        f"{context}.requested_evaluation_row",
    )
    return {
        **value,
        "max_witness_mismatches": max_mismatches,
        "audit_step_strategy": audit_step_strategy,
        "requested_audit_step": requested_audit_step,
        "evaluation_row_strategy": evaluation_row_strategy,
        "requested_evaluation_row": requested_evaluation_row,
    }


def validate_pipeline_gates(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    gate_passed = expect_bool(json_field(value, "passed", context), f"{context}.passed")
    policy = validate_pipeline_gate_policy(json_field(value, "policy", context), f"{context}.policy")
    checks = json_field(value, "checks", context)
    if not isinstance(checks, list) or not checks:
        raise ValueError(f"{context}.checks must be a non-empty array")
    seen: set[str] = set()
    passed_values = []
    for index, check in enumerate(checks):
        check_context = f"{context}.checks[{index}]"
        if not isinstance(check, dict):
            raise ValueError(f"{check_context} must be an object")
        metric = validate_nonempty_string(
            json_field(check, "metric", check_context),
            f"{check_context}.metric",
        )
        if metric in seen:
            raise ValueError(f"{context}.checks has duplicate metric `{metric}`")
        seen.add(metric)
        json_field(check, "actual", check_context)
        validate_nonempty_string(
            json_field(check, "requirement", check_context),
            f"{check_context}.requirement",
        )
        passed_values.append(expect_bool(json_field(check, "passed", check_context), f"{check_context}.passed"))
    missing = [metric for metric in PIPELINE_GATE_METRICS if metric not in seen]
    if missing:
        raise ValueError(f"{context}.checks missing required metric(s): {', '.join(missing)}")
    if policy["max_replay_payload_bytes"] is not None and "max_replay_payload_bytes" not in seen:
        raise ValueError(f"{context}.checks missing max_replay_payload_bytes gate")
    if gate_passed != all(passed_values):
        raise ValueError(f"{context}.passed must equal all check results")
    if not gate_passed:
        failed = [
            check["metric"]
            for check in checks
            if isinstance(check, dict) and check.get("passed") is False
        ]
        raise ValueError(f"{context} failed gate(s): {', '.join(failed)}")
    return value


def validate_pipeline_claims(
    value: Any,
    *,
    gates: dict[str, Any],
    metrics: dict[str, Any],
    evidence_files: dict[str, Any],
    context: str,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if json_field(value, "kind", context) != PIPELINE_CLAIMS_KIND:
        raise ValueError(f"{context}.kind must be `{PIPELINE_CLAIMS_KIND}`")
    claims_passed = expect_bool(json_field(value, "passed", context), f"{context}.passed")
    checks = json_field(value, "checks", context)
    if not isinstance(checks, list) or len(checks) != len(PIPELINE_CLAIMS):
        raise ValueError(f"{context}.checks must contain every pipeline claim")
    gate_results = {
        check["metric"]: check["passed"]
        for check in gates["checks"]
        if isinstance(check, dict) and isinstance(check.get("metric"), str)
    }
    seen: set[str] = set()
    passed_values = []
    for index, check in enumerate(checks):
        check_context = f"{context}.checks[{index}]"
        if not isinstance(check, dict):
            raise ValueError(f"{check_context} must be an object")
        claim = validate_nonempty_string(
            json_field(check, "claim", check_context),
            f"{check_context}.claim",
        )
        if claim in seen:
            raise ValueError(f"{context}.checks has duplicate claim `{claim}`")
        seen.add(claim)
        validate_nonempty_string(
            json_field(check, "description", check_context),
            f"{check_context}.description",
        )
        gate_metrics = validate_string_list(
            json_field(check, "gate_metrics", check_context),
            f"{check_context}.gate_metrics",
            allow_empty=(claim == "artifact_evidence_integrity"),
        )
        evidence = validate_string_list(
            json_field(check, "evidence", check_context),
            f"{check_context}.evidence",
        )
        metric_refs = validate_string_list(
            json_field(check, "metrics", check_context),
            f"{check_context}.metrics",
            allow_empty=(claim == "artifact_evidence_integrity"),
        )
        expected_passed = True
        for metric in gate_metrics:
            if metric not in gate_results:
                raise ValueError(f"{check_context}.gate_metrics references unknown gate `{metric}`")
            if gate_results[metric] is not True:
                expected_passed = False
        for evidence_key in evidence:
            if evidence_key not in evidence_files:
                raise ValueError(f"{check_context}.evidence references unknown file `{evidence_key}`")
        if claim == "artifact_evidence_integrity":
            expected_evidence = set(evidence_files)
            # The pipeline summary cannot cite itself before it is written. Capsule-level
            # evidence may include the summary file, while the claims matrix is copied
            # from the already-validated summary packet.
            expected_evidence.discard("summary")
            actual_evidence = set(evidence)
            missing_evidence = sorted(expected_evidence - actual_evidence)
            extra_evidence = sorted(actual_evidence - expected_evidence)
            if missing_evidence or extra_evidence:
                message = f"{check_context}.evidence for artifact_evidence_integrity must cite every digest-indexed evidence file"
                details = []
                if missing_evidence:
                    details.append("missing: " + ", ".join(missing_evidence))
                if extra_evidence:
                    details.append("extra: " + ", ".join(extra_evidence))
                raise ValueError(message + " (" + "; ".join(details) + ")")
        for metric_ref in metric_refs:
            if metric_ref not in metrics:
                raise ValueError(f"{check_context}.metrics references unknown metric `{metric_ref}`")
        passed = expect_bool(json_field(check, "passed", check_context), f"{check_context}.passed")
        if passed != expected_passed:
            raise ValueError(f"{check_context}.passed does not match referenced gate results")
        if not passed:
            raise ValueError(f"{check_context} claim `{claim}` failed")
        passed_values.append(passed)
    missing = [claim for claim in PIPELINE_CLAIMS if claim not in seen]
    if missing:
        raise ValueError(f"{context}.checks missing required claim(s): {', '.join(missing)}")
    extra = [claim for claim in seen if claim not in PIPELINE_CLAIMS]
    if extra:
        raise ValueError(f"{context}.checks has unexpected claim(s): {', '.join(sorted(extra))}")
    if claims_passed != all(passed_values):
        raise ValueError(f"{context}.passed must equal all claim results")
    if not claims_passed:
        raise ValueError(f"{context}.passed must be true")


def validate_pipeline_accuracy_block(
    value: Any,
    context: str,
    *,
    rate_field: Optional[str] = None,
    require_incorrect: bool = False,
) -> tuple[int, int, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    samples = expect_positive_int(json_field(value, "samples", context), f"{context}.samples")
    correct = expect_nonnegative_int(json_field(value, "correct", context), f"{context}.correct")
    if correct > samples:
        raise ValueError(f"{context}.correct cannot exceed samples")
    if require_incorrect:
        incorrect = expect_nonnegative_int(json_field(value, "incorrect", context), f"{context}.incorrect")
        if correct + incorrect != samples:
            raise ValueError(f"{context}.correct + incorrect must equal samples")
    else:
        incorrect = samples - correct
    validate_accuracy(
        json_field(value, "accuracy_percent", context),
        correct,
        samples,
        f"{context}.accuracy_percent",
    )
    if rate_field is not None:
        if expect_finite_number(json_field(value, rate_field, context), f"{context}.{rate_field}") < 0.0:
            raise ValueError(f"{context}.{rate_field} must be non-negative")
    return samples, correct, incorrect


def validate_pipeline_replay_block(value: Any, context: str, fields: tuple[str, ...]) -> int:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    for field in fields:
        if not expect_bool(json_field(value, field, context), f"{context}.{field}"):
            raise ValueError(f"{context}.{field} must be true")
    return expect_positive_int(
        json_field(value, "replay_payload_bytes", context),
        f"{context}.replay_payload_bytes",
    )


def validate_pipeline_bool_block(value: Any, context: str, fields: tuple[str, ...]) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    for field in fields:
        if not expect_bool(json_field(value, field, context), f"{context}.{field}"):
            raise ValueError(f"{context}.{field} must be true")


def validate_pipeline_q31_reference_inference_metrics(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    prediction = expect_digit(json_field(value, "prediction", context), f"{context}.prediction")
    correct = expect_bool(json_field(value, "correct", context), f"{context}.correct")
    margin = expect_i64(json_field(value, "margin", context), f"{context}.margin")
    active_pixels = expect_nonnegative_int(
        json_field(value, "active_pixels", context),
        f"{context}.active_pixels",
    )
    native_prediction = expect_digit(
        json_field(value, "native_prediction", context),
        f"{context}.native_prediction",
    )
    native_correct = expect_bool(
        json_field(value, "native_correct", context),
        f"{context}.native_correct",
    )
    native_margin = expect_i64(
        json_field(value, "native_margin", context),
        f"{context}.native_margin",
    )
    native_active_pixels = expect_nonnegative_int(
        json_field(value, "native_active_pixels", context),
        f"{context}.native_active_pixels",
    )
    contribution_count = expect_nonnegative_int(
        json_field(value, "contribution_count", context),
        f"{context}.contribution_count",
    )
    margin_contribution_count = expect_nonnegative_int(
        json_field(value, "margin_contribution_count", context),
        f"{context}.margin_contribution_count",
    )
    contribution_fingerprint = validate_sha256(
        json_field(value, "contribution_ledger_fingerprint", context),
        f"{context}.contribution_ledger_fingerprint",
    )
    margin_contribution_fingerprint = validate_sha256(
        json_field(value, "margin_contribution_ledger_fingerprint", context),
        f"{context}.margin_contribution_ledger_fingerprint",
    )
    native_contribution_count = expect_nonnegative_int(
        json_field(value, "native_contribution_count", context),
        f"{context}.native_contribution_count",
    )
    native_margin_contribution_count = expect_nonnegative_int(
        json_field(value, "native_margin_contribution_count", context),
        f"{context}.native_margin_contribution_count",
    )
    native_contribution_fingerprint = validate_sha256(
        json_field(value, "native_contribution_ledger_fingerprint", context),
        f"{context}.native_contribution_ledger_fingerprint",
    )
    native_margin_contribution_fingerprint = validate_sha256(
        json_field(value, "native_margin_contribution_ledger_fingerprint", context),
        f"{context}.native_margin_contribution_ledger_fingerprint",
    )
    if active_pixels > IMAGE_PIXELS:
        raise ValueError(f"{context}.active_pixels must be no more than {IMAGE_PIXELS}")
    if native_active_pixels > IMAGE_PIXELS:
        raise ValueError(f"{context}.native_active_pixels must be no more than {IMAGE_PIXELS}")
    comparisons = {
        "prediction_matches_native": prediction == native_prediction,
        "correct_matches_native": correct == native_correct,
        "margin_matches_native": margin == native_margin,
        "active_pixels_match_native": active_pixels == native_active_pixels,
        "contribution_ledger_matches_native": (
            contribution_count == native_contribution_count
            and contribution_fingerprint == native_contribution_fingerprint
        ),
        "margin_contribution_ledger_matches_native": (
            margin_contribution_count == native_margin_contribution_count
            and margin_contribution_fingerprint == native_margin_contribution_fingerprint
        ),
        "attribution_matches_logit": True,
        "attribution_matches_margin": True,
    }
    for field, expected in comparisons.items():
        actual = expect_bool(json_field(value, field, context), f"{context}.{field}")
        if actual is not True:
            raise ValueError(f"{context}.{field} must be true")
        if actual != expected:
            raise ValueError(f"{context}.{field} must summarize native/reference parity")


def validate_pipeline_inference_ledger(value: Any, context: str) -> dict[str, Any]:
    ledger = validate_exact_key_object(
        value,
        (
            "contribution_count",
            "margin_contribution_count",
            "contribution_ledger_fingerprint",
            "margin_contribution_ledger_fingerprint",
        ),
        context,
    )
    return {
        "contribution_count": expect_nonnegative_int(
            json_field(ledger, "contribution_count", context),
            f"{context}.contribution_count",
        ),
        "margin_contribution_count": expect_nonnegative_int(
            json_field(ledger, "margin_contribution_count", context),
            f"{context}.margin_contribution_count",
        ),
        "contribution_ledger_fingerprint": validate_sha256(
            json_field(ledger, "contribution_ledger_fingerprint", context),
            f"{context}.contribution_ledger_fingerprint",
        ),
        "margin_contribution_ledger_fingerprint": validate_sha256(
            json_field(ledger, "margin_contribution_ledger_fingerprint", context),
            f"{context}.margin_contribution_ledger_fingerprint",
        ),
    }


def validate_pipeline_inference_trace_source(
    value: Any,
    context: str,
    *,
    role: str,
) -> dict[str, Any]:
    source = validate_exact_key_object(
        value,
        (
            "role",
            "prediction",
            "correct",
            "margin",
            "active_pixels",
            "ledger",
            "explanation_passed",
            "replay_payload_bytes",
            "forward_recompute_steps",
            "inverse_recompute_steps",
        ),
        context,
    )
    if json_field(source, "role", context) != role:
        raise ValueError(f"{context}.role must be `{role}`")
    active_pixels = expect_nonnegative_int(
        json_field(source, "active_pixels", context),
        f"{context}.active_pixels",
    )
    if active_pixels > IMAGE_PIXELS:
        raise ValueError(f"{context}.active_pixels must be no more than {IMAGE_PIXELS}")
    if not expect_bool(json_field(source, "explanation_passed", context), f"{context}.explanation_passed"):
        raise ValueError(f"{context}.explanation_passed must be true")
    return {
        "prediction": expect_digit(json_field(source, "prediction", context), f"{context}.prediction"),
        "correct": expect_bool(json_field(source, "correct", context), f"{context}.correct"),
        "margin": expect_i64(json_field(source, "margin", context), f"{context}.margin"),
        "active_pixels": active_pixels,
        "ledger": validate_pipeline_inference_ledger(json_field(source, "ledger", context), f"{context}.ledger"),
        "replay_payload_bytes": expect_positive_int(
            json_field(source, "replay_payload_bytes", context),
            f"{context}.replay_payload_bytes",
        ),
        "forward_recompute_steps": expect_positive_int(
            json_field(source, "forward_recompute_steps", context),
            f"{context}.forward_recompute_steps",
        ),
        "inverse_recompute_steps": expect_positive_int(
            json_field(source, "inverse_recompute_steps", context),
            f"{context}.inverse_recompute_steps",
        ),
    }


def validate_pipeline_inference_trace_reference(value: Any, context: str) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    available = expect_bool(json_field(value, "available", context), f"{context}.available")
    if json_field(value, "role", context) != "q31_reference":
        raise ValueError(f"{context}.role must be `q31_reference`")
    if not available:
        for field in ("prediction", "correct", "margin", "active_pixels", "ledger"):
            if json_field(value, field, context) is not None:
                raise ValueError(f"{context}.{field} must be null when reference is unavailable")
        return None
    active_pixels = expect_nonnegative_int(
        json_field(value, "active_pixels", context),
        f"{context}.active_pixels",
    )
    if active_pixels > IMAGE_PIXELS:
        raise ValueError(f"{context}.active_pixels must be no more than {IMAGE_PIXELS}")
    for field in ("explanation_passed", "replay_payload_bytes", "forward_recompute_steps", "inverse_recompute_steps"):
        if json_field(value, field, context) is not None:
            raise ValueError(f"{context}.{field} must be null for the independent reference")
    return {
        "prediction": expect_digit(json_field(value, "prediction", context), f"{context}.prediction"),
        "correct": expect_bool(json_field(value, "correct", context), f"{context}.correct"),
        "margin": expect_i64(json_field(value, "margin", context), f"{context}.margin"),
        "active_pixels": active_pixels,
        "ledger": validate_pipeline_inference_ledger(json_field(value, "ledger", context), f"{context}.ledger"),
    }


def inference_trace_result_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left[field] == right[field] for field in ("prediction", "correct", "margin", "active_pixels"))


def validate_pipeline_inference_trace_profile(
    value: Any,
    context: str,
    *,
    metrics: dict[str, Any],
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if json_field(value, "kind", context) != INFERENCE_TRACE_PROFILE_KIND:
        raise ValueError(f"{context}.kind must be `{INFERENCE_TRACE_PROFILE_KIND}`")
    traces = json_field(value, "traces", context)
    if not isinstance(traces, list) or len(traces) != 2:
        raise ValueError(f"{context}.traces must contain native and evaluation-row traces")
    expected_trace_ids = ("native_selected_sample", "evaluation_row")
    passed_values = []
    reference_checked = 0
    report_verification_results = []
    report_verification_ledgers = []
    required_reference_results = []
    replay_results = []
    explanation_results = []
    source_results = []
    for index, trace in enumerate(traces):
        trace_context = f"{context}.traces[{index}]"
        trace = validate_exact_key_object(
            trace,
            (
                "id",
                "label",
                "source_kind",
                "sample_index",
                "row_index",
                "prediction",
                "correct",
                "margin",
                "active_pixels",
                "report",
                "verification",
                "reference",
                "checks",
                "passed",
            ),
            trace_context,
        )
        trace_id = json_field(trace, "id", trace_context)
        if trace_id != expected_trace_ids[index]:
            raise ValueError(f"{trace_context}.id must be `{expected_trace_ids[index]}`")
        validate_nonempty_string(json_field(trace, "label", trace_context), f"{trace_context}.label")
        validate_nonempty_string(json_field(trace, "source_kind", trace_context), f"{trace_context}.source_kind")
        expect_nonnegative_int(json_field(trace, "sample_index", trace_context), f"{trace_context}.sample_index")
        row_index = json_field(trace, "row_index", trace_context)
        if trace_id == "native_selected_sample":
            if row_index is not None:
                raise ValueError(f"{trace_context}.row_index must be null")
        else:
            expect_nonnegative_int(row_index, f"{trace_context}.row_index")
        report = validate_pipeline_inference_trace_source(
            json_field(trace, "report", trace_context),
            f"{trace_context}.report",
            role="native_report",
        )
        verification = validate_pipeline_inference_trace_source(
            json_field(trace, "verification", trace_context),
            f"{trace_context}.verification",
            role="replay_verification",
        )
        reference = validate_pipeline_inference_trace_reference(
            json_field(trace, "reference", trace_context),
            f"{trace_context}.reference",
        )
        expected_top = {
            "prediction": report["prediction"],
            "correct": report["correct"],
            "margin": report["margin"],
            "active_pixels": report["active_pixels"],
        }
        for field, expected in expected_top.items():
            actual = json_field(trace, field, trace_context)
            if actual != expected:
                raise ValueError(f"{trace_context}.{field} must match report")
        checks = validate_exact_key_object(
            json_field(trace, "checks", trace_context),
            (
                "report_verification_result_matches",
                "report_verification_ledgers_match",
                "report_explanation_passed",
                "verification_explanation_passed",
                "verification_replay_passed",
                "source_inputs_checked",
                "reference_required",
                "reference_available",
                "reference_result_matches_report",
                "reference_ledgers_match_report",
            ),
            f"{trace_context}.checks",
        )
        result_match = inference_trace_result_matches(report, verification)
        ledger_match = report["ledger"] == verification["ledger"]
        if expect_bool(
            json_field(checks, "report_verification_result_matches", f"{trace_context}.checks"),
            f"{trace_context}.checks.report_verification_result_matches",
        ) != result_match:
            raise ValueError(f"{trace_context}.checks.report_verification_result_matches must summarize trace")
        if expect_bool(
            json_field(checks, "report_verification_ledgers_match", f"{trace_context}.checks"),
            f"{trace_context}.checks.report_verification_ledgers_match",
        ) != ledger_match:
            raise ValueError(f"{trace_context}.checks.report_verification_ledgers_match must summarize trace")
        reference_required = expect_bool(
            json_field(checks, "reference_required", f"{trace_context}.checks"),
            f"{trace_context}.checks.reference_required",
        )
        expected_reference_required = trace_id == "native_selected_sample"
        if reference_required != expected_reference_required:
            raise ValueError(f"{trace_context}.checks.reference_required must match trace role")
        reference_available = expect_bool(
            json_field(checks, "reference_available", f"{trace_context}.checks"),
            f"{trace_context}.checks.reference_available",
        )
        if reference_available != (reference is not None):
            raise ValueError(f"{trace_context}.checks.reference_available must match reference")
        reference_result_match = reference is not None and inference_trace_result_matches(report, reference)
        reference_ledger_match = reference is not None and report["ledger"] == reference["ledger"]
        if expect_bool(
            json_field(checks, "reference_result_matches_report", f"{trace_context}.checks"),
            f"{trace_context}.checks.reference_result_matches_report",
        ) != reference_result_match:
            raise ValueError(f"{trace_context}.checks.reference_result_matches_report must summarize trace")
        if expect_bool(
            json_field(checks, "reference_ledgers_match_report", f"{trace_context}.checks"),
            f"{trace_context}.checks.reference_ledgers_match_report",
        ) != reference_ledger_match:
            raise ValueError(f"{trace_context}.checks.reference_ledgers_match_report must summarize trace")
        for field in (
            "report_explanation_passed",
            "verification_explanation_passed",
            "verification_replay_passed",
            "source_inputs_checked",
        ):
            if not expect_bool(json_field(checks, field, f"{trace_context}.checks"), f"{trace_context}.checks.{field}"):
                raise ValueError(f"{trace_context}.checks.{field} must be true")
        expected_passed = (
            result_match
            and ledger_match
            and checks["report_explanation_passed"] is True
            and checks["verification_explanation_passed"] is True
            and checks["verification_replay_passed"] is True
            and checks["source_inputs_checked"] is True
            and (
                not reference_required
                or (reference_available and reference_result_match and reference_ledger_match)
            )
        )
        passed = expect_bool(json_field(trace, "passed", trace_context), f"{trace_context}.passed")
        if passed != expected_passed:
            raise ValueError(f"{trace_context}.passed must summarize trace checks")
        if not passed:
            raise ValueError(f"{trace_context} must pass")
        if trace_id == "native_selected_sample":
            reference_metrics = json_field(metrics, "q31_reference_inference", "metrics")
            if reference is None:
                raise ValueError(f"{trace_context}.reference must be available")
            if reference["ledger"]["contribution_ledger_fingerprint"] != json_field(
                reference_metrics,
                "contribution_ledger_fingerprint",
                "metrics.q31_reference_inference",
            ):
                raise ValueError(f"{trace_context}.reference ledger must match q31_reference_inference")
        else:
            if reference is not None:
                raise ValueError(f"{trace_context}.reference must be unavailable until per-row reference output is materialized")
        passed_values.append(passed)
        reference_checked += 1 if reference_available else 0
        report_verification_results.append(result_match)
        report_verification_ledgers.append(ledger_match)
        required_reference_results.append(
            (not reference_required) or (reference_available and reference_result_match and reference_ledger_match)
        )
        replay_results.append(checks["verification_replay_passed"] is True)
        explanation_results.append(
            checks["report_explanation_passed"] is True
            and checks["verification_explanation_passed"] is True
        )
        source_results.append(checks["source_inputs_checked"] is True)
    summary = json_field(value, "summary", context)
    if not isinstance(summary, dict):
        raise ValueError(f"{context}.summary must be an object")
    expected_summary = {
        "trace_count": len(traces),
        "reference_checked_traces": reference_checked,
        "all_report_verification_results_match": all(report_verification_results),
        "all_report_verification_ledgers_match": all(report_verification_ledgers),
        "all_required_references_match": all(required_reference_results),
        "all_replay_verified": all(replay_results),
        "all_explanations_passed": all(explanation_results),
        "all_sources_checked": all(source_results),
        "all_passed": all(passed_values),
    }
    for field, expected in expected_summary.items():
        actual = json_field(summary, field, f"{context}.summary")
        if isinstance(expected, bool):
            actual = expect_bool(actual, f"{context}.summary.{field}")
        else:
            actual = expect_nonnegative_int(actual, f"{context}.summary.{field}")
        if actual != expected:
            raise ValueError(f"{context}.summary.{field} must summarize traces")
    if not expected_summary["all_passed"]:
        raise ValueError(f"{context}.summary.all_passed must be true")


def validate_pipeline_training_step_debug(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    claim = json_field(value, "claim", context)
    if claim != "step_backward_from_model_update":
        raise ValueError(f"{context}.claim must be `step_backward_from_model_update`")
    passed = expect_bool(json_field(value, "passed", context), f"{context}.passed")
    step = expect_nonnegative_int(json_field(value, "step", context), f"{context}.step")
    expect_nonnegative_int(json_field(value, "sample_index", context), f"{context}.sample_index")
    expect_digit(json_field(value, "prediction", context), f"{context}.prediction")
    expect_bool(json_field(value, "correct", context), f"{context}.correct")
    expect_i64(json_field(value, "margin", context), f"{context}.margin")
    active_pixel_count = expect_positive_int(
        json_field(value, "active_pixel_count", context),
        f"{context}.active_pixel_count",
    )
    if active_pixel_count > IMAGE_PIXELS:
        raise ValueError(f"{context}.active_pixel_count must be no more than {IMAGE_PIXELS}")
    expect_positive_int(
        json_field(value, "top_weight_deltas", context),
        f"{context}.top_weight_deltas",
    )
    expect_positive_int(
        json_field(value, "nonzero_bias_delta_count", context),
        f"{context}.nonzero_bias_delta_count",
    )
    expect_positive_int(
        json_field(value, "nonzero_weight_delta_count", context),
        f"{context}.nonzero_weight_delta_count",
    )
    validate_sha256(
        json_field(value, "bias_delta_ledger_fingerprint", context),
        f"{context}.bias_delta_ledger_fingerprint",
    )
    validate_sha256(
        json_field(value, "weight_delta_ledger_fingerprint", context),
        f"{context}.weight_delta_ledger_fingerprint",
    )
    validate_sha256(
        json_field(value, "cause_ledger_fingerprint", context),
        f"{context}.cause_ledger_fingerprint",
    )
    expect_nonnegative_int(
        json_field(value, "reversed_later_steps", context),
        f"{context}.reversed_later_steps",
    )
    checks = json_field(value, "checks", context)
    if not isinstance(checks, list) or not checks:
        raise ValueError(f"{context}.checks must be a non-empty array")
    seen: set[str] = set()
    all_passed = True
    for index, check in enumerate(checks):
        check_context = f"{context}.checks[{index}]"
        metric = json_field(check, "metric", check_context)
        if not isinstance(metric, str) or not metric:
            raise ValueError(f"{check_context}.metric must be a non-empty string")
        if metric in seen:
            raise ValueError(f"{context}.checks has duplicate metric `{metric}`")
        seen.add(metric)
        validate_nonempty_string(json_field(check, "actual", check_context), f"{check_context}.actual")
        validate_nonempty_string(
            json_field(check, "requirement", check_context),
            f"{check_context}.requirement",
        )
        all_passed = all_passed and expect_bool(
            json_field(check, "passed", check_context),
            f"{check_context}.passed",
        )
    required = {
        "witness_recomputes_prediction",
        "model_window_reconstructed",
        "update_matches_model_window",
        "explanatory_state_present",
    }
    missing = sorted(required - seen)
    if missing:
        raise ValueError(f"{context}.checks missing metric(s): " + ", ".join(missing))
    if passed != all_passed:
        raise ValueError(f"{context}.passed must summarize checks")
    if not passed:
        raise ValueError(f"{context}.passed must be true")


def validate_pipeline_training_step_selection(
    value: Any,
    context: str,
    *,
    requested_step: int,
    selected_pipeline_step: int,
    selection_strategy: str,
    audit_steps: int,
    audit_scan: dict[str, Any],
    training_step_debug: dict[str, Any],
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    selected_step = expect_nonnegative_int(json_field(value, "selected_step", context), f"{context}.selected_step")
    observed_requested = expect_nonnegative_int(
        json_field(value, "requested_step", context),
        f"{context}.requested_step",
    )
    if observed_requested != requested_step:
        raise ValueError(f"{context}.requested_step must match pipeline requested_audit_step")
    if selected_step != selected_pipeline_step:
        raise ValueError(f"{context}.selected_step must match pipeline audit_step")
    if selected_step >= audit_steps:
        raise ValueError(f"{context}.selected_step must be less than audit scan steps")
    if selected_step != json_field(training_step_debug, "step", "metrics.training_step_debug"):
        raise ValueError(f"{context}.selected_step must match training_step_debug.step")
    observed_strategy = validate_nonempty_string(
        json_field(value, "selection_strategy", context),
        f"{context}.selection_strategy",
    )
    if observed_strategy != selection_strategy:
        raise ValueError(f"{context}.selection_strategy must match pipeline audit_step_strategy")
    if observed_strategy not in AUDIT_STEP_STRATEGIES:
        raise ValueError(f"{context}.selection_strategy is invalid")
    expect_nonnegative_int(
        json_field(value, "selected_sample_index", context),
        f"{context}.selected_sample_index",
    )
    selected_margin = expect_i64(json_field(value, "selected_margin", context), f"{context}.selected_margin")
    if selected_margin != json_field(training_step_debug, "margin", "metrics.training_step_debug"):
        raise ValueError(f"{context}.selected_margin must match training_step_debug.margin")
    expect_nonnegative_int(
        json_field(value, "selected_max_abs_weight_delta", context),
        f"{context}.selected_max_abs_weight_delta",
    )
    scan_lowest_step = expect_nonnegative_int(
        json_field(value, "scan_lowest_margin_step", context),
        f"{context}.scan_lowest_margin_step",
    )
    scan_lowest_margin = expect_i64(
        json_field(value, "scan_lowest_margin", context),
        f"{context}.scan_lowest_margin",
    )
    scan_largest_step = expect_nonnegative_int(
        json_field(value, "scan_largest_update_step", context),
        f"{context}.scan_largest_update_step",
    )
    scan_max_delta = expect_nonnegative_int(
        json_field(value, "scan_max_abs_weight_delta", context),
        f"{context}.scan_max_abs_weight_delta",
    )
    scan_top_suspicious_step = expect_nonnegative_int(
        json_field(value, "scan_top_suspicious_step", context),
        f"{context}.scan_top_suspicious_step",
    )
    scan_top_low_margin_step = expect_nonnegative_int(
        json_field(value, "scan_top_low_margin_step", context),
        f"{context}.scan_top_low_margin_step",
    )
    scan_top_large_updates_step = expect_nonnegative_int(
        json_field(value, "scan_top_large_updates_step", context),
        f"{context}.scan_top_large_updates_step",
    )
    for field, step in (
        ("scan_top_suspicious_step", scan_top_suspicious_step),
        ("scan_top_low_margin_step", scan_top_low_margin_step),
        ("scan_top_large_updates_step", scan_top_large_updates_step),
    ):
        if step >= audit_steps:
            raise ValueError(f"{context}.{field} must be less than audit scan steps")
    if scan_lowest_step != json_field(audit_scan, "lowest_margin_step", "metrics.training_audit_scan"):
        raise ValueError(f"{context}.scan_lowest_margin_step must match training_audit_scan")
    if scan_lowest_margin != json_field(audit_scan, "lowest_margin", "metrics.training_audit_scan"):
        raise ValueError(f"{context}.scan_lowest_margin must match training_audit_scan")
    if scan_largest_step != json_field(audit_scan, "largest_update_step", "metrics.training_audit_scan"):
        raise ValueError(f"{context}.scan_largest_update_step must match training_audit_scan")
    if scan_max_delta != json_field(audit_scan, "max_abs_weight_delta", "metrics.training_audit_scan"):
        raise ValueError(f"{context}.scan_max_abs_weight_delta must match training_audit_scan")
    expectations = {
        "matches_requested_step": selected_step == requested_step,
        "matches_scan_lowest_margin": selected_step == scan_lowest_step,
        "matches_scan_largest_update": selected_step == scan_largest_step,
    }
    expected_strategy_steps = {
        "explicit": requested_step,
        "lowest-margin": scan_lowest_step,
        "largest-update": scan_largest_step,
        "top-suspicious": scan_top_suspicious_step,
    }
    expected_strategy_step = expected_strategy_steps[observed_strategy]
    selection_strategy_step = expect_nonnegative_int(
        json_field(value, "selection_strategy_step", context),
        f"{context}.selection_strategy_step",
    )
    if selection_strategy_step != expected_strategy_step:
        raise ValueError(f"{context}.selection_strategy_step must match selection_strategy")
    matches_strategy = expect_bool(
        json_field(value, "matches_selection_strategy", context),
        f"{context}.matches_selection_strategy",
    )
    if matches_strategy != (selected_step == expected_strategy_step):
        raise ValueError(f"{context}.matches_selection_strategy must summarize selected step")
    if not matches_strategy:
        raise ValueError(f"{context}.matches_selection_strategy must be true")
    for field, expected in expectations.items():
        actual = expect_bool(json_field(value, field, context), f"{context}.{field}")
        if actual != expected:
            raise ValueError(f"{context}.{field} must summarize selected step")
    for field in (
        "present_in_top_suspicious",
        "present_in_top_low_margin",
        "present_in_top_large_updates",
    ):
        expect_bool(json_field(value, field, context), f"{context}.{field}")
    reasons = validate_string_list(
        json_field(value, "selection_reasons", context),
        f"{context}.selection_reasons",
    )
    expected_reasons = []
    if expectations["matches_requested_step"]:
        expected_reasons.append("requested")
    if expectations["matches_scan_lowest_margin"]:
        expected_reasons.append("lowest_margin")
    if expectations["matches_scan_largest_update"]:
        expected_reasons.append("largest_update")
    if value["present_in_top_suspicious"]:
        expected_reasons.append("top_suspicious")
    if value["present_in_top_low_margin"]:
        expected_reasons.append("top_low_margin")
    if value["present_in_top_large_updates"]:
        expected_reasons.append("top_large_updates")
    if reasons != expected_reasons:
        raise ValueError(f"{context}.selection_reasons must summarize scan evidence")


def validate_optional_row_index(value: Any, context: str, *, samples: int) -> Optional[int]:
    if value is None:
        return None
    row = expect_nonnegative_int(value, context)
    if row >= samples:
        raise ValueError(f"{context} must be less than model evaluation samples")
    return row


def validate_pipeline_evaluation_row_selection(
    value: Any,
    context: str,
    *,
    requested_row: int,
    selected_pipeline_row: int,
    selection_strategy: str,
    model_samples: int,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    observed_strategy = validate_nonempty_string(
        json_field(value, "selection_strategy", context),
        f"{context}.selection_strategy",
    )
    if observed_strategy != selection_strategy:
        raise ValueError(f"{context}.selection_strategy must match pipeline evaluation_row_strategy")
    if observed_strategy not in EVALUATION_ROW_STRATEGIES:
        raise ValueError(f"{context}.selection_strategy is invalid")
    observed_requested = expect_nonnegative_int(
        json_field(value, "requested_row", context),
        f"{context}.requested_row",
    )
    if observed_requested != requested_row:
        raise ValueError(f"{context}.requested_row must match pipeline requested_evaluation_row")
    selected_row = expect_nonnegative_int(
        json_field(value, "selected_row", context),
        f"{context}.selected_row",
    )
    if selected_row != selected_pipeline_row:
        raise ValueError(f"{context}.selected_row must match pipeline evaluation_row")
    if selected_row >= model_samples:
        raise ValueError(f"{context}.selected_row must be less than model evaluation samples")
    source_index = json_field(value, "selected_source_sample_index", context)
    if source_index is not None:
        validate_optional_row_index(
            source_index,
            f"{context}.selected_source_sample_index",
            samples=model_samples,
        )
    expect_digit(json_field(value, "selected_prediction", context), f"{context}.selected_prediction")
    expect_bool(json_field(value, "selected_correct", context), f"{context}.selected_correct")
    expect_i64(json_field(value, "selected_margin", context), f"{context}.selected_margin")
    scan_lowest_row = expect_nonnegative_int(
        json_field(value, "scan_lowest_margin_row", context),
        f"{context}.scan_lowest_margin_row",
    )
    if scan_lowest_row >= model_samples:
        raise ValueError(f"{context}.scan_lowest_margin_row must be less than model evaluation samples")
    expect_i64(json_field(value, "scan_lowest_margin", context), f"{context}.scan_lowest_margin")
    scan_top_low_margin_row = validate_optional_row_index(
        json_field(value, "scan_top_low_margin_row", context),
        f"{context}.scan_top_low_margin_row",
        samples=model_samples,
    )
    scan_top_incorrect_row = validate_optional_row_index(
        json_field(value, "scan_top_incorrect_row", context),
        f"{context}.scan_top_incorrect_row",
        samples=model_samples,
    )
    scan_top_low_margin_rows_value = json_field(value, "scan_top_low_margin_rows", context)
    if not isinstance(scan_top_low_margin_rows_value, list) or not scan_top_low_margin_rows_value:
        raise ValueError(f"{context}.scan_top_low_margin_rows must be a non-empty array")
    scan_top_low_margin_rows = [
        validate_optional_row_index(item, f"{context}.scan_top_low_margin_rows[{index}]", samples=model_samples)
        for index, item in enumerate(scan_top_low_margin_rows_value)
    ]
    scan_top_incorrect_rows_value = json_field(value, "scan_top_incorrect_rows", context)
    if not isinstance(scan_top_incorrect_rows_value, list):
        raise ValueError(f"{context}.scan_top_incorrect_rows must be an array")
    scan_top_incorrect_rows = [
        validate_optional_row_index(item, f"{context}.scan_top_incorrect_rows[{index}]", samples=model_samples)
        for index, item in enumerate(scan_top_incorrect_rows_value)
    ]
    if scan_top_low_margin_row != scan_top_low_margin_rows[0]:
        raise ValueError(f"{context}.scan_top_low_margin_row must match first top-low-margin row")
    if scan_top_incorrect_rows and scan_top_incorrect_row != scan_top_incorrect_rows[0]:
        raise ValueError(f"{context}.scan_top_incorrect_row must match first top-incorrect row")
    if not scan_top_incorrect_rows and scan_top_incorrect_row is not None:
        raise ValueError(f"{context}.scan_top_incorrect_row must be null without top-incorrect rows")
    expected_strategy_rows = {
        "explicit": requested_row,
        "lowest-margin": scan_lowest_row,
        "top-incorrect": scan_top_incorrect_row,
    }
    expected_strategy_row = expected_strategy_rows[observed_strategy]
    if expected_strategy_row is None:
        raise ValueError(f"{context}.selection_strategy has no matching scan row")
    selection_strategy_row = expect_nonnegative_int(
        json_field(value, "selection_strategy_row", context),
        f"{context}.selection_strategy_row",
    )
    if selection_strategy_row != expected_strategy_row:
        raise ValueError(f"{context}.selection_strategy_row must match selection_strategy")
    expected = {
        "matches_selection_strategy": selected_row == expected_strategy_row,
        "matches_requested_row": selected_row == requested_row,
        "matches_scan_lowest_margin": selected_row == scan_lowest_row,
        "present_in_top_low_margin": selected_row in scan_top_low_margin_rows,
        "present_in_top_incorrect": selected_row in scan_top_incorrect_rows,
    }
    for field, expected_value in expected.items():
        actual = expect_bool(json_field(value, field, context), f"{context}.{field}")
        if actual != expected_value:
            raise ValueError(f"{context}.{field} must summarize selected row")
    if not expected["matches_selection_strategy"]:
        raise ValueError(f"{context}.matches_selection_strategy must be true")
    reasons = validate_string_list(
        json_field(value, "selection_reasons", context),
        f"{context}.selection_reasons",
    )
    expected_reasons = []
    if expected["matches_requested_row"]:
        expected_reasons.append("requested")
    if expected["matches_scan_lowest_margin"]:
        expected_reasons.append("lowest_margin")
    if expected["present_in_top_low_margin"]:
        expected_reasons.append("top_low_margin")
    if expected["present_in_top_incorrect"]:
        expected_reasons.append("top_incorrect")
    if reasons != expected_reasons:
        raise ValueError(f"{context}.selection_reasons must summarize scan evidence")


def validate_pipeline_reverse_check_cost(value: Any, context: str, train_samples: int) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if not expect_bool(json_field(value, "enabled", context), f"{context}.enabled"):
        raise ValueError(f"{context}.enabled must be true")
    train_updates = expect_positive_int(
        json_field(value, "train_updates", context),
        f"{context}.train_updates",
    )
    checked_steps = expect_positive_int(
        json_field(value, "checked_steps", context),
        f"{context}.checked_steps",
    )
    if train_updates != train_samples:
        raise ValueError(f"{context}.train_updates must equal train samples")
    if checked_steps != train_samples:
        raise ValueError(f"{context}.checked_steps must equal train samples")
    train_elapsed = expect_finite_number(
        json_field(value, "train_elapsed_seconds", context),
        f"{context}.train_elapsed_seconds",
    )
    reverse_elapsed = expect_finite_number(
        json_field(value, "reverse_elapsed_seconds", context),
        f"{context}.reverse_elapsed_seconds",
    )
    if train_elapsed <= 0.0:
        raise ValueError(f"{context}.train_elapsed_seconds must be positive")
    if reverse_elapsed <= 0.0:
        raise ValueError(f"{context}.reverse_elapsed_seconds must be positive")
    train_rate = expect_finite_number(
        json_field(value, "train_updates_per_second", context),
        f"{context}.train_updates_per_second",
    )
    reverse_rate = expect_finite_number(
        json_field(value, "reverse_steps_per_second", context),
        f"{context}.reverse_steps_per_second",
    )
    if train_rate <= 0.0:
        raise ValueError(f"{context}.train_updates_per_second must be positive")
    if reverse_rate <= 0.0:
        raise ValueError(f"{context}.reverse_steps_per_second must be positive")
    expect_float_ratio(
        json_field(value, "train_seconds_per_update", context),
        train_elapsed,
        float(train_updates),
        f"{context}.train_seconds_per_update",
    )
    expect_float_ratio(
        json_field(value, "reverse_seconds_per_step", context),
        reverse_elapsed,
        float(checked_steps),
        f"{context}.reverse_seconds_per_step",
    )
    expect_float_ratio(
        json_field(value, "reverse_to_train_elapsed_ratio", context),
        reverse_elapsed,
        train_elapsed,
        f"{context}.reverse_to_train_elapsed_ratio",
    )
    expect_float_ratio(
        json_field(value, "reverse_to_train_throughput_ratio", context),
        reverse_rate,
        train_rate,
        f"{context}.reverse_to_train_throughput_ratio",
    )
    forward = expect_positive_int(
        json_field(value, "proof_forward_recompute_steps", context),
        f"{context}.proof_forward_recompute_steps",
    )
    inverse = expect_positive_int(
        json_field(value, "proof_inverse_recompute_steps", context),
        f"{context}.proof_inverse_recompute_steps",
    )
    if forward != checked_steps or inverse != checked_steps:
        raise ValueError(f"{context} proof recompute steps must match checked steps")
    replay_payload = expect_positive_int(
        json_field(value, "proof_replay_payload_bytes", context),
        f"{context}.proof_replay_payload_bytes",
    )
    trace_replay_payload = expect_positive_int(
        json_field(value, "trace_replay_payload_bytes", context),
        f"{context}.trace_replay_payload_bytes",
    )
    if trace_replay_payload > replay_payload:
        raise ValueError(f"{context}.trace_replay_payload_bytes must not exceed proof replay bytes")


def validate_pipeline_artifact_profile(value: Any, context: str) -> tuple[int, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    model_bytes = expect_positive_int(
        json_field(value, "total_model_payload_bytes", context),
        f"{context}.total_model_payload_bytes",
    )
    for field in (
        "total_file_bytes",
        "total_logical_payload_bytes",
        "total_witness_payload_bytes",
        "total_trace_payload_bytes",
    ):
        expect_nonnegative_int(json_field(value, field, context), f"{context}.{field}")
    forward = expect_positive_int(
        json_field(value, "total_forward_recompute_steps", context),
        f"{context}.total_forward_recompute_steps",
    )
    inverse = expect_positive_int(
        json_field(value, "total_inverse_recompute_steps", context),
        f"{context}.total_inverse_recompute_steps",
    )
    total = expect_positive_int(
        json_field(value, "total_recompute_steps", context),
        f"{context}.total_recompute_steps",
    )
    if total != forward + inverse:
        raise ValueError(f"{context}.total_recompute_steps must equal forward + inverse")
    trace_bytes = expect_nonnegative_int(
        json_field(value, "total_trace_payload_bytes", context),
        f"{context}.total_trace_payload_bytes",
    )
    witness_bytes = expect_nonnegative_int(
        json_field(value, "total_witness_payload_bytes", context),
        f"{context}.total_witness_payload_bytes",
    )
    expect_ratio(
        json_field(value, "trace_to_model_payload_ratio", context),
        trace_bytes,
        model_bytes,
        f"{context}.trace_to_model_payload_ratio",
    )
    expect_ratio(
        json_field(value, "witness_to_model_payload_ratio", context),
        witness_bytes,
        model_bytes,
        f"{context}.witness_to_model_payload_ratio",
    )
    return forward, inverse


def validate_pipeline_mlp_witness_metrics(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if not expect_bool(json_field(value, "passed", context), f"{context}.passed"):
        raise ValueError(f"{context}.passed must be true")
    samples = expect_positive_int(json_field(value, "samples", context), f"{context}.samples")
    validate_mlp_dataset_loops(
        json_field(value, "dataset_loops", context),
        f"{context}.dataset_loops",
    )
    predictions = json_field(value, "predictions", context)
    if not isinstance(predictions, list) or len(predictions) != samples:
        raise ValueError(f"{context}.predictions must match samples")
    for index, item in enumerate(predictions):
        expect_digit(item, f"{context}.predictions[{index}]")
    correct = json_field(value, "correct", context)
    if not isinstance(correct, list) or len(correct) != samples:
        raise ValueError(f"{context}.correct must match samples")
    for index, item in enumerate(correct):
        if not expect_bool(item, f"{context}.correct[{index}]"):
            raise ValueError(f"{context}.correct[{index}] must be true")
    validate_sha256(
        json_field(value, "witness_proof_fingerprint", context),
        f"{context}.witness_proof_fingerprint",
    )

    witness_payload_bytes = expect_positive_int(
        json_field(value, "witness_payload_bytes", context),
        f"{context}.witness_payload_bytes",
    )
    trace_payload_bytes = expect_positive_int(
        json_field(value, "trace_payload_bytes", context),
        f"{context}.trace_payload_bytes",
    )
    replay_payload_bytes = expect_positive_int(
        json_field(value, "replay_payload_bytes", context),
        f"{context}.replay_payload_bytes",
    )
    recomputed_update_payload_bytes = expect_positive_int(
        json_field(value, "recomputed_update_payload_bytes", context),
        f"{context}.recomputed_update_payload_bytes",
    )
    forward = expect_positive_int(
        json_field(value, "forward_recompute_steps", context),
        f"{context}.forward_recompute_steps",
    )
    inverse = expect_positive_int(
        json_field(value, "inverse_recompute_steps", context),
        f"{context}.inverse_recompute_steps",
    )
    total = expect_positive_int(
        json_field(value, "total_recompute_steps", context),
        f"{context}.total_recompute_steps",
    )
    if forward != samples or inverse != samples:
        raise ValueError(f"{context} forward/inverse recompute steps must match samples")
    if total != forward + inverse:
        raise ValueError(f"{context}.total_recompute_steps must equal forward + inverse")
    if replay_payload_bytes <= trace_payload_bytes:
        raise ValueError(f"{context}.replay_payload_bytes must include model plus trace bytes")
    model_payload_bytes = replay_payload_bytes - trace_payload_bytes
    expect_ratio(
        json_field(value, "witness_to_model_payload_ratio", context),
        witness_payload_bytes,
        model_payload_bytes,
        f"{context}.witness_to_model_payload_ratio",
    )
    expect_ratio(
        json_field(value, "trace_to_model_payload_ratio", context),
        trace_payload_bytes,
        model_payload_bytes,
        f"{context}.trace_to_model_payload_ratio",
    )
    expect_ratio(
        json_field(value, "recomputed_update_to_witness_payload_ratio", context),
        recomputed_update_payload_bytes,
        witness_payload_bytes,
        f"{context}.recomputed_update_to_witness_payload_ratio",
    )


def validate_pipeline_coupling_metrics(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if not expect_bool(json_field(value, "passed", context), f"{context}.passed"):
        raise ValueError(f"{context}.passed must be true")
    initial = {
        "left": validate_coupling_vector(json_field(value, "initial_left", context), f"{context}.initial_left"),
        "right": validate_coupling_vector(json_field(value, "initial_right", context), f"{context}.initial_right"),
    }
    forward = {
        "left": validate_coupling_vector(json_field(value, "forward_left", context), f"{context}.forward_left"),
        "right": validate_coupling_vector(json_field(value, "forward_right", context), f"{context}.forward_right"),
    }
    expected_initial = coupling_check.initial_store()
    expected_forward = coupling_check.forward_store(expected_initial)
    if initial != {"left": expected_initial["left"], "right": expected_initial["right"]}:
        raise ValueError(f"{context} initial state does not match reference")
    if forward != {"left": expected_forward["left"], "right": expected_forward["right"]}:
        raise ValueError(f"{context} forward state does not match reference")
    proof = {
        "claim": INVERTIBLE_COUPLING_PROOF_CLAIM,
        "arithmetic": "q31_wrapping_i64",
        "model_payload_bytes": json_field(value, "model_payload_bytes", context),
        "state_payload_bytes": json_field(value, "state_payload_bytes", context),
        "witness_payload_bytes": json_field(value, "witness_payload_bytes", context),
        "trace_payload_bytes": json_field(value, "trace_payload_bytes", context),
        "replay_payload_bytes": json_field(value, "replay_payload_bytes", context),
        "forward_recompute_steps": json_field(value, "forward_recompute_steps", context),
        "inverse_recompute_steps": json_field(value, "inverse_recompute_steps", context),
        "total_recompute_steps": json_field(value, "total_recompute_steps", context),
        "witness_to_model_payload_ratio": json_field(value, "witness_to_model_payload_ratio", context),
        "trace_to_model_payload_ratio": json_field(value, "trace_to_model_payload_ratio", context),
        "checks": {
            "forward_matches_reference": True,
            "reverse_restores_initial_state": True,
            "no_witness_tape": True,
            "balanced_recompute": True,
        },
    }
    validate_invertible_coupling_proof(proof, context)


def validate_pipeline_residual_metrics(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if not expect_bool(json_field(value, "passed", context), f"{context}.passed"):
        raise ValueError(f"{context}.passed must be true")
    initial_x = validate_residual_vector(
        json_field(value, "initial_x", context),
        f"{context}.initial_x",
    )
    forward_x = validate_residual_vector(
        json_field(value, "forward_x", context),
        f"{context}.forward_x",
    )
    expected_initial = residual_check.initial_store()
    expected_forward = residual_check.forward_store(expected_initial)
    if initial_x != expected_initial["x"]:
        raise ValueError(f"{context}.initial_x does not match reference")
    if forward_x != expected_forward["x"]:
        raise ValueError(f"{context}.forward_x does not match reference")
    proof = {
        "claim": TRIANGULAR_RESIDUAL_PROOF_CLAIM,
        "arithmetic": "q31_wrapping_i64",
        "parameter_payload_bytes": json_field(value, "parameter_payload_bytes", context),
        "state_payload_bytes": json_field(value, "state_payload_bytes", context),
        "witness_payload_bytes": json_field(value, "witness_payload_bytes", context),
        "trace_payload_bytes": json_field(value, "trace_payload_bytes", context),
        "replay_payload_bytes": json_field(value, "replay_payload_bytes", context),
        "forward_recompute_steps": json_field(value, "forward_recompute_steps", context),
        "inverse_recompute_steps": json_field(value, "inverse_recompute_steps", context),
        "total_recompute_steps": json_field(value, "total_recompute_steps", context),
        "witness_to_parameter_payload_ratio": json_field(
            value,
            "witness_to_parameter_payload_ratio",
            context,
        ),
        "trace_to_parameter_payload_ratio": json_field(
            value,
            "trace_to_parameter_payload_ratio",
            context,
        ),
        "checks": {
            "forward_matches_reference": True,
            "reverse_restores_initial_state": True,
            "triangular_source_order": True,
            "no_witness_tape": True,
            "balanced_recompute": True,
        },
    }
    validate_triangular_residual_proof(proof, context)


def validate_pipeline_preprocess_metrics(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if not expect_bool(json_field(value, "passed", context), f"{context}.passed"):
        raise ValueError(f"{context}.passed must be true")
    initial_features = validate_preprocess_vector(
        json_field(value, "initial_features", context),
        f"{context}.initial_features",
    )
    forward_features = validate_preprocess_vector(
        json_field(value, "forward_features", context),
        f"{context}.forward_features",
    )
    expected_initial = preprocess_check.initial_store()
    expected_forward = preprocess_check.forward_store(expected_initial)
    if initial_features != expected_initial["features"]:
        raise ValueError(f"{context}.initial_features does not match reference")
    if forward_features != expected_forward["features"]:
        raise ValueError(f"{context}.forward_features does not match reference")
    proof = {
        "claim": REVERSIBLE_PREPROCESS_PROOF_CLAIM,
        "arithmetic": "q31_wrapping_i64",
        "raw_payload_bytes": json_field(value, "raw_payload_bytes", context),
        "mean_payload_bytes": json_field(value, "mean_payload_bytes", context),
        "feature_payload_bytes": json_field(value, "feature_payload_bytes", context),
        "witness_payload_bytes": json_field(value, "witness_payload_bytes", context),
        "trace_payload_bytes": json_field(value, "trace_payload_bytes", context),
        "replay_payload_bytes": json_field(value, "replay_payload_bytes", context),
        "forward_recompute_steps": json_field(value, "forward_recompute_steps", context),
        "inverse_recompute_steps": json_field(value, "inverse_recompute_steps", context),
        "total_recompute_steps": json_field(value, "total_recompute_steps", context),
        "witness_to_state_payload_ratio": json_field(value, "witness_to_state_payload_ratio", context),
        "trace_to_state_payload_ratio": json_field(value, "trace_to_state_payload_ratio", context),
        "checks": {
            "forward_matches_reference": True,
            "reverse_restores_initial_state": True,
            "raw_preserved": True,
            "mean_preserved": True,
            "no_witness_tape": True,
            "balanced_recompute": True,
        },
    }
    validate_reversible_preprocess_proof(proof, context)


def validate_pipeline_reversible_inference_trace_metrics(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if not expect_bool(json_field(value, "passed", context), f"{context}.passed"):
        raise ValueError(f"{context}.passed must be true")
    initial_features = validate_inference_trace_vector(
        json_field(value, "initial_features", context),
        inference_trace_check.FEATURES,
        f"{context}.initial_features",
    )
    forward_features = validate_inference_trace_vector(
        json_field(value, "forward_features", context),
        inference_trace_check.FEATURES,
        f"{context}.forward_features",
    )
    forward_logits = validate_inference_trace_vector(
        json_field(value, "forward_logits", context),
        inference_trace_check.CLASSES,
        f"{context}.forward_logits",
    )
    top_classes = validate_inference_trace_vector(
        json_field(value, "top_classes", context),
        inference_trace_check.CLASSES,
        f"{context}.top_classes",
    )
    top_logit_values = validate_inference_trace_vector(
        json_field(value, "top_logit_values", context),
        inference_trace_check.CLASSES,
        f"{context}.top_logit_values",
    )
    expected_initial = inference_trace_check.initial_store()
    expected_forward = inference_trace_check.forward_store(expected_initial)
    if initial_features != expected_initial["features"]:
        raise ValueError(f"{context}.initial_features does not match reference")
    if forward_features != expected_forward["features"]:
        raise ValueError(f"{context}.forward_features does not match reference")
    if forward_logits != expected_forward["logits"]:
        raise ValueError(f"{context}.forward_logits does not match reference")
    if (
        top_classes != expected_forward["top_classes"]
        or top_logit_values != expected_forward["top_logit_values"]
    ):
        raise ValueError(f"{context}.top_* fields do not match reference")
    validate_inference_trace_top_logits(
        json_field(value, "top_logits", context),
        forward_logits,
        f"{context}.top_logits",
    )
    validate_inference_trace_attribution(
        json_field(value, "attribution", context),
        expected_forward,
        f"{context}.attribution",
    )
    prediction = expect_i64(json_field(value, "prediction", context), f"{context}.prediction")
    runner_up_class = expect_i64(
        json_field(value, "runner_up_class", context),
        f"{context}.runner_up_class",
    )
    margin = expect_i64(json_field(value, "margin", context), f"{context}.margin")
    label_rank = expect_i64(json_field(value, "label_rank", context), f"{context}.label_rank")
    correct = expect_i64(json_field(value, "correct", context), f"{context}.correct")
    top2_correct = expect_i64(
        json_field(value, "top2_correct", context),
        f"{context}.top2_correct",
    )
    if (
        prediction != expected_forward["prediction"]
        or runner_up_class != expected_forward["runner_up_class"]
        or margin != expected_forward["margin"]
        or label_rank != expected_forward["label_rank"]
        or correct != expected_forward["correct"]
        or top2_correct != expected_forward["top2_correct"]
    ):
        raise ValueError(f"{context} prediction/margin/correctness does not match reference")
    proof = {
        "claim": REVERSIBLE_INFERENCE_TRACE_PROOF_CLAIM,
        "arithmetic": "q31_wrapping_i64",
        "raw_payload_bytes": json_field(value, "raw_payload_bytes", context),
        "mean_payload_bytes": json_field(value, "mean_payload_bytes", context),
        "feature_payload_bytes": json_field(value, "feature_payload_bytes", context),
        "weight_payload_bytes": json_field(value, "weight_payload_bytes", context),
        "bias_payload_bytes": json_field(value, "bias_payload_bytes", context),
        "model_payload_bytes": json_field(value, "model_payload_bytes", context),
        "logit_payload_bytes": json_field(value, "logit_payload_bytes", context),
        "top_class_payload_bytes": json_field(value, "top_class_payload_bytes", context),
        "top_logit_value_payload_bytes": json_field(
            value,
            "top_logit_value_payload_bytes",
            context,
        ),
        "prediction_payload_bytes": json_field(value, "prediction_payload_bytes", context),
        "runner_up_payload_bytes": json_field(value, "runner_up_payload_bytes", context),
        "margin_payload_bytes": json_field(value, "margin_payload_bytes", context),
        "label_rank_payload_bytes": json_field(value, "label_rank_payload_bytes", context),
        "correct_payload_bytes": json_field(value, "correct_payload_bytes", context),
        "top_k_correct_payload_bytes": json_field(
            value,
            "top_k_correct_payload_bytes",
            context,
        ),
        "label_payload_bytes": json_field(value, "label_payload_bytes", context),
        "state_payload_bytes": json_field(value, "state_payload_bytes", context),
        "witness_payload_bytes": json_field(value, "witness_payload_bytes", context),
        "trace_payload_bytes": json_field(value, "trace_payload_bytes", context),
        "replay_payload_bytes": json_field(value, "replay_payload_bytes", context),
        "forward_recompute_steps": json_field(value, "forward_recompute_steps", context),
        "inverse_recompute_steps": json_field(value, "inverse_recompute_steps", context),
        "total_recompute_steps": json_field(value, "total_recompute_steps", context),
        "witness_to_model_payload_ratio": json_field(value, "witness_to_model_payload_ratio", context),
        "trace_to_model_payload_ratio": json_field(value, "trace_to_model_payload_ratio", context),
        "witness_to_state_payload_ratio": json_field(value, "witness_to_state_payload_ratio", context),
        "trace_to_state_payload_ratio": json_field(value, "trace_to_state_payload_ratio", context),
        "checks": {
            "preprocess_matches_reference": True,
            "logits_match_reference": True,
            "top_classes_match_reference": True,
            "top_logit_values_match_reference": True,
            "prediction_matches_reference": True,
            "runner_up_matches_reference": True,
            "margin_matches_reference": True,
            "label_rank_matches_reference": True,
            "correctness_matches_reference": True,
            "top_k_correctness_matches_reference": True,
            "reverse_restores_initial_state": True,
            "raw_preserved": True,
            "model_preserved": True,
            "balanced_recompute": True,
        },
    }
    validate_reversible_inference_trace_proof(proof, context)


def validate_frontier_ratio(value: Any, numerator: int, denominator: int, context: str) -> None:
    if denominator <= 0:
        if value is not None:
            raise ValueError(f"{context} must be null when denominator is zero")
        return
    expect_ratio(value, numerator, denominator, context)


def validate_pipeline_recompute_frontier(value: Any, context: str, *, metrics: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if json_field(value, "kind", context) != RECOMPUTE_FRONTIER_KIND:
        raise ValueError(f"{context}.kind must be `{RECOMPUTE_FRONTIER_KIND}`")
    rows = json_field(value, "rows", context)
    if not isinstance(rows, list) or len(rows) != len(RECOMPUTE_FRONTIER_ROWS):
        raise ValueError(f"{context}.rows must contain every recompute frontier row")
    expected_replay = {
        "training_trace": json_field(
            json_field(metrics, "reverse_check_cost", "metrics"),
            "proof_replay_payload_bytes",
            "metrics.reverse_check_cost",
        ),
        "training_step_debug": json_field(
            json_field(metrics, "training_step_replay", "metrics"),
            "replay_payload_bytes",
            "metrics.training_step_replay",
        ),
        "imported_model_inference": json_field(
            json_field(metrics, "imported_model_inference", "metrics"),
            "replay_payload_bytes",
            "metrics.imported_model_inference",
        ),
        "native_inference": json_field(
            json_field(metrics, "native_inference_replay", "metrics"),
            "replay_payload_bytes",
            "metrics.native_inference_replay",
        ),
        "model_evaluation_batch": json_field(
            json_field(metrics, "model_evaluation", "metrics"),
            "replay_payload_bytes",
            "metrics.model_evaluation",
        ),
        "evaluation_row_inference": json_field(
            json_field(metrics, "evaluation_row_inference", "metrics"),
            "replay_payload_bytes",
            "metrics.evaluation_row_inference",
        ),
        "mlp_witness_trace": json_field(
            json_field(metrics, "mlp_witness", "metrics"),
            "replay_payload_bytes",
            "metrics.mlp_witness",
        ),
        "invertible_coupling": json_field(
            json_field(metrics, "invertible_coupling", "metrics"),
            "replay_payload_bytes",
            "metrics.invertible_coupling",
        ),
        "triangular_residual": json_field(
            json_field(metrics, "triangular_residual", "metrics"),
            "replay_payload_bytes",
            "metrics.triangular_residual",
        ),
        "reversible_preprocess": json_field(
            json_field(metrics, "reversible_preprocess", "metrics"),
            "replay_payload_bytes",
            "metrics.reversible_preprocess",
        ),
        "reversible_inference_trace": json_field(
            json_field(metrics, "reversible_inference_trace", "metrics"),
            "replay_payload_bytes",
            "metrics.reversible_inference_trace",
        ),
    }
    expected_forward = {
        "training_trace": json_field(
            json_field(metrics, "reverse_check_cost", "metrics"),
            "proof_forward_recompute_steps",
            "metrics.reverse_check_cost",
        ),
        "training_step_debug": 1,
        "imported_model_inference": json_field(
            json_field(metrics, "imported_model_inference", "metrics"),
            "forward_recompute_steps",
            "metrics.imported_model_inference",
        ),
        "native_inference": 1,
        "model_evaluation_batch": json_field(
            json_field(metrics, "model_evaluation", "metrics"),
            "samples",
            "metrics.model_evaluation",
        ),
        "evaluation_row_inference": 1,
        "mlp_witness_trace": json_field(
            json_field(metrics, "mlp_witness", "metrics"),
            "forward_recompute_steps",
            "metrics.mlp_witness",
        ),
        "invertible_coupling": json_field(
            json_field(metrics, "invertible_coupling", "metrics"),
            "forward_recompute_steps",
            "metrics.invertible_coupling",
        ),
        "triangular_residual": json_field(
            json_field(metrics, "triangular_residual", "metrics"),
            "forward_recompute_steps",
            "metrics.triangular_residual",
        ),
        "reversible_preprocess": json_field(
            json_field(metrics, "reversible_preprocess", "metrics"),
            "forward_recompute_steps",
            "metrics.reversible_preprocess",
        ),
        "reversible_inference_trace": json_field(
            json_field(metrics, "reversible_inference_trace", "metrics"),
            "forward_recompute_steps",
            "metrics.reversible_inference_trace",
        ),
    }
    expected_inverse = {
        "training_trace": json_field(
            json_field(metrics, "reverse_check_cost", "metrics"),
            "proof_inverse_recompute_steps",
            "metrics.reverse_check_cost",
        ),
        "training_step_debug": 1,
        "imported_model_inference": json_field(
            json_field(metrics, "imported_model_inference", "metrics"),
            "inverse_recompute_steps",
            "metrics.imported_model_inference",
        ),
        "native_inference": 1,
        "model_evaluation_batch": json_field(
            json_field(metrics, "model_evaluation", "metrics"),
            "samples",
            "metrics.model_evaluation",
        ),
        "evaluation_row_inference": 1,
        "mlp_witness_trace": json_field(
            json_field(metrics, "mlp_witness", "metrics"),
            "inverse_recompute_steps",
            "metrics.mlp_witness",
        ),
        "invertible_coupling": json_field(
            json_field(metrics, "invertible_coupling", "metrics"),
            "inverse_recompute_steps",
            "metrics.invertible_coupling",
        ),
        "triangular_residual": json_field(
            json_field(metrics, "triangular_residual", "metrics"),
            "inverse_recompute_steps",
            "metrics.triangular_residual",
        ),
        "reversible_preprocess": json_field(
            json_field(metrics, "reversible_preprocess", "metrics"),
            "inverse_recompute_steps",
            "metrics.reversible_preprocess",
        ),
        "reversible_inference_trace": json_field(
            json_field(metrics, "reversible_inference_trace", "metrics"),
            "inverse_recompute_steps",
            "metrics.reversible_inference_trace",
        ),
    }
    checked_rows = []
    for index, row in enumerate(rows):
        row_context = f"{context}.rows[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{row_context} must be an object")
        row_id = validate_nonempty_string(json_field(row, "id", row_context), f"{row_context}.id")
        if row_id != RECOMPUTE_FRONTIER_ROWS[index]:
            raise ValueError(f"{row_context}.id must be `{RECOMPUTE_FRONTIER_ROWS[index]}`")
        for field in ("label", "scope", "kept_state", "reversible_transition"):
            validate_nonempty_string(json_field(row, field, row_context), f"{row_context}.{field}")
        replay_payload = expect_positive_int(
            json_field(row, "replay_payload_bytes", row_context),
            f"{row_context}.replay_payload_bytes",
        )
        if replay_payload != expected_replay[row_id]:
            raise ValueError(f"{row_context}.replay_payload_bytes must match source replay metric")
        model_payload = expect_positive_int(
            json_field(row, "model_payload_bytes", row_context),
            f"{row_context}.model_payload_bytes",
        )
        sample_payload = expect_nonnegative_int(
            json_field(row, "sample_payload_bytes", row_context),
            f"{row_context}.sample_payload_bytes",
        )
        witness_payload = expect_nonnegative_int(
            json_field(row, "witness_payload_bytes", row_context),
            f"{row_context}.witness_payload_bytes",
        )
        trace_payload = expect_nonnegative_int(
            json_field(row, "trace_payload_bytes", row_context),
            f"{row_context}.trace_payload_bytes",
        )
        derived_update_payload = expect_nonnegative_int(
            json_field(row, "derived_update_payload_bytes", row_context),
            f"{row_context}.derived_update_payload_bytes",
        )
        recomputed_update_payload = expect_nonnegative_int(
            json_field(row, "recomputed_update_payload_bytes", row_context),
            f"{row_context}.recomputed_update_payload_bytes",
        )
        state_payload = expect_nonnegative_int(
            json_field(row, "state_payload_bytes", row_context),
            f"{row_context}.state_payload_bytes",
        )
        forward = expect_positive_int(
            json_field(row, "forward_recompute_steps", row_context),
            f"{row_context}.forward_recompute_steps",
        )
        inverse = expect_positive_int(
            json_field(row, "inverse_recompute_steps", row_context),
            f"{row_context}.inverse_recompute_steps",
        )
        if forward != expected_forward[row_id]:
            raise ValueError(f"{row_context}.forward_recompute_steps must match source metric")
        if inverse != expected_inverse[row_id]:
            raise ValueError(f"{row_context}.inverse_recompute_steps must match source metric")
        total = expect_positive_int(
            json_field(row, "total_recompute_steps", row_context),
            f"{row_context}.total_recompute_steps",
        )
        if total != forward + inverse:
            raise ValueError(f"{row_context}.total_recompute_steps must equal forward + inverse")
        balanced = expect_bool(
            json_field(row, "balanced_recompute", row_context),
            f"{row_context}.balanced_recompute",
        )
        if balanced != (forward == inverse and forward > 0):
            raise ValueError(f"{row_context}.balanced_recompute must summarize recompute steps")
        zero_witness = expect_bool(json_field(row, "zero_witness", row_context), f"{row_context}.zero_witness")
        if zero_witness != (witness_payload == 0 and trace_payload == 0):
            raise ValueError(f"{row_context}.zero_witness must summarize witness and trace bytes")
        witness_backed = expect_bool(
            json_field(row, "witness_backed", row_context),
            f"{row_context}.witness_backed",
        )
        if witness_backed != (witness_payload > 0):
            raise ValueError(f"{row_context}.witness_backed must summarize witness bytes")
        validate_frontier_ratio(
            json_field(row, "bytes_per_inverse_step", row_context),
            replay_payload,
            inverse,
            f"{row_context}.bytes_per_inverse_step",
        )
        validate_frontier_ratio(
            json_field(row, "witness_to_model_payload_ratio", row_context),
            witness_payload,
            model_payload,
            f"{row_context}.witness_to_model_payload_ratio",
        )
        validate_frontier_ratio(
            json_field(row, "trace_to_model_payload_ratio", row_context),
            trace_payload,
            model_payload,
            f"{row_context}.trace_to_model_payload_ratio",
        )
        if row_id == "training_trace":
            expected_trace = json_field(
                json_field(metrics, "reverse_check_cost", "metrics"),
                "trace_replay_payload_bytes",
                "metrics.reverse_check_cost",
            )
            if trace_payload != expected_trace:
                raise ValueError(f"{row_context}.trace_payload_bytes must match reverse-check trace replay bytes")
            if replay_payload != model_payload + trace_payload:
                raise ValueError(f"{row_context}.replay_payload_bytes must equal model + trace")
        elif row_id == "mlp_witness_trace":
            mlp_metrics = json_field(metrics, "mlp_witness", "metrics")
            if recomputed_update_payload != json_field(mlp_metrics, "recomputed_update_payload_bytes", "metrics.mlp_witness"):
                raise ValueError(f"{row_context}.recomputed_update_payload_bytes must match MLP metrics")
            if replay_payload != model_payload + trace_payload:
                raise ValueError(f"{row_context}.replay_payload_bytes must equal model + trace")
        else:
            if replay_payload != model_payload + sample_payload + witness_payload + trace_payload + derived_update_payload + state_payload:
                raise ValueError(f"{row_context}.replay_payload_bytes must equal retained payload components")
            if recomputed_update_payload < derived_update_payload:
                raise ValueError(f"{row_context}.recomputed_update_payload_bytes cannot be less than stored update bytes")
        checked_rows.append(
            {
                "id": row_id,
                "replay_payload_bytes": replay_payload,
                "forward_recompute_steps": forward,
                "inverse_recompute_steps": inverse,
                "zero_witness": zero_witness,
                "witness_backed": witness_backed,
            }
        )
    summary = json_field(value, "summary", context)
    if not isinstance(summary, dict):
        raise ValueError(f"{context}.summary must be an object")
    replay_values = [row["replay_payload_bytes"] for row in checked_rows]
    if expect_positive_int(json_field(summary, "rows", f"{context}.summary"), f"{context}.summary.rows") != len(RECOMPUTE_FRONTIER_ROWS):
        raise ValueError(f"{context}.summary.rows must match row count")
    if expect_positive_int(
        json_field(summary, "total_replay_payload_bytes", f"{context}.summary"),
        f"{context}.summary.total_replay_payload_bytes",
    ) != sum(replay_values):
        raise ValueError(f"{context}.summary.total_replay_payload_bytes must match rows")
    max_replay = max(replay_values)
    min_replay = min(replay_values)
    if expect_positive_int(
        json_field(summary, "max_replay_payload_bytes", f"{context}.summary"),
        f"{context}.summary.max_replay_payload_bytes",
    ) != max_replay:
        raise ValueError(f"{context}.summary.max_replay_payload_bytes must match rows")
    if expect_positive_int(
        json_field(summary, "min_replay_payload_bytes", f"{context}.summary"),
        f"{context}.summary.min_replay_payload_bytes",
    ) != min_replay:
        raise ValueError(f"{context}.summary.min_replay_payload_bytes must match rows")
    max_id = json_field(summary, "max_replay_id", f"{context}.summary")
    min_id = json_field(summary, "min_replay_id", f"{context}.summary")
    if max_id != checked_rows[replay_values.index(max_replay)]["id"]:
        raise ValueError(f"{context}.summary.max_replay_id must match rows")
    if min_id != checked_rows[replay_values.index(min_replay)]["id"]:
        raise ValueError(f"{context}.summary.min_replay_id must match rows")
    if expect_positive_int(
        json_field(summary, "total_forward_recompute_steps", f"{context}.summary"),
        f"{context}.summary.total_forward_recompute_steps",
    ) != sum(row["forward_recompute_steps"] for row in checked_rows):
        raise ValueError(f"{context}.summary.total_forward_recompute_steps must match rows")
    if expect_positive_int(
        json_field(summary, "total_inverse_recompute_steps", f"{context}.summary"),
        f"{context}.summary.total_inverse_recompute_steps",
    ) != sum(row["inverse_recompute_steps"] for row in checked_rows):
        raise ValueError(f"{context}.summary.total_inverse_recompute_steps must match rows")
    zero_rows = validate_string_list(
        json_field(summary, "zero_witness_rows", f"{context}.summary"),
        f"{context}.summary.zero_witness_rows",
    )
    expected_zero_rows = [row["id"] for row in checked_rows if row["zero_witness"]]
    if zero_rows != expected_zero_rows:
        raise ValueError(f"{context}.summary.zero_witness_rows must match rows")
    witness_rows = validate_string_list(
        json_field(summary, "witness_backed_rows", f"{context}.summary"),
        f"{context}.summary.witness_backed_rows",
    )
    expected_witness_rows = [row["id"] for row in checked_rows if row["witness_backed"]]
    if witness_rows != expected_witness_rows:
        raise ValueError(f"{context}.summary.witness_backed_rows must match rows")


def exact_projection_unit(total: int, count: int, context: str) -> int:
    if count <= 0:
        raise ValueError(f"{context} count must be positive")
    if total % count != 0:
        raise ValueError(f"{context} total must divide evenly by count")
    return total // count


def validate_pipeline_scaling_projection(value: Any, context: str, *, metrics: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if json_field(value, "kind", context) != SCALING_PROJECTION_KIND:
        raise ValueError(f"{context}.kind must be `{SCALING_PROJECTION_KIND}`")
    recompute_frontier = json_field(metrics, "recompute_frontier", "metrics")
    frontier_rows = json_field(recompute_frontier, "rows", "metrics.recompute_frontier")
    if not isinstance(frontier_rows, list):
        raise ValueError("metrics.recompute_frontier.rows must be a list")
    frontier_by_id = {
        row.get("id"): row
        for row in frontier_rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    reverse_cost = json_field(metrics, "reverse_check_cost", "metrics")
    model_evaluation = json_field(metrics, "model_evaluation", "metrics")
    mlp_witness = json_field(metrics, "mlp_witness", "metrics")
    observed_counts = {
        "training_trace": expect_positive_int(
            json_field(reverse_cost, "train_updates", "metrics.reverse_check_cost"),
            "metrics.reverse_check_cost.train_updates",
        ),
        "model_evaluation_batch": expect_positive_int(
            json_field(model_evaluation, "samples", "metrics.model_evaluation"),
            "metrics.model_evaluation.samples",
        ),
        "mlp_witness_trace": expect_positive_int(
            json_field(mlp_witness, "samples", "metrics.mlp_witness"),
            "metrics.mlp_witness.samples",
        ),
    }
    families = json_field(value, "families", context)
    if not isinstance(families, list) or len(families) != len(SCALING_PROJECTION_FAMILIES):
        raise ValueError(f"{context}.families must contain every scaling projection family")
    checked_families = []
    for index, family in enumerate(families):
        family_context = f"{context}.families[{index}]"
        if not isinstance(family, dict):
            raise ValueError(f"{family_context} must be an object")
        family_id = validate_nonempty_string(
            json_field(family, "id", family_context),
            f"{family_context}.id",
        )
        if family_id != SCALING_PROJECTION_FAMILIES[index]:
            raise ValueError(f"{family_context}.id must be `{SCALING_PROJECTION_FAMILIES[index]}`")
        validate_nonempty_string(json_field(family, "label", family_context), f"{family_context}.label")
        validate_nonempty_string(json_field(family, "unit", family_context), f"{family_context}.unit")
        observed_count = expect_positive_int(
            json_field(family, "observed_count", family_context),
            f"{family_context}.observed_count",
        )
        if observed_count != observed_counts[family_id]:
            raise ValueError(f"{family_context}.observed_count must match source metrics")
        if family_id not in frontier_by_id:
            raise ValueError(f"{family_context}.id is missing from recompute frontier")
        frontier = frontier_by_id[family_id]
        model_payload = expect_positive_int(
            json_field(family, "model_payload_bytes", family_context),
            f"{family_context}.model_payload_bytes",
        )
        if model_payload != json_field(frontier, "model_payload_bytes", "frontier row"):
            raise ValueError(f"{family_context}.model_payload_bytes must match recompute frontier")
        sample_per = expect_nonnegative_int(
            json_field(family, "sample_payload_bytes_per_unit", family_context),
            f"{family_context}.sample_payload_bytes_per_unit",
        )
        witness_per = expect_nonnegative_int(
            json_field(family, "witness_payload_bytes_per_unit", family_context),
            f"{family_context}.witness_payload_bytes_per_unit",
        )
        trace_per = expect_nonnegative_int(
            json_field(family, "trace_payload_bytes_per_unit", family_context),
            f"{family_context}.trace_payload_bytes_per_unit",
        )
        variable_per = expect_nonnegative_int(
            json_field(family, "variable_replay_payload_bytes_per_unit", family_context),
            f"{family_context}.variable_replay_payload_bytes_per_unit",
        )
        recomputed_update_per = expect_nonnegative_int(
            json_field(family, "recomputed_update_payload_bytes_per_unit", family_context),
            f"{family_context}.recomputed_update_payload_bytes_per_unit",
        )
        forward_per = expect_positive_int(
            json_field(family, "forward_recompute_steps_per_unit", family_context),
            f"{family_context}.forward_recompute_steps_per_unit",
        )
        inverse_per = expect_positive_int(
            json_field(family, "inverse_recompute_steps_per_unit", family_context),
            f"{family_context}.inverse_recompute_steps_per_unit",
        )
        expected_sample_per = exact_projection_unit(
            expect_nonnegative_int(
                json_field(frontier, "sample_payload_bytes", "frontier row"),
                "frontier row.sample_payload_bytes",
            ),
            observed_count,
            f"{family_context}.sample_payload_bytes_per_unit",
        )
        expected_witness_per = exact_projection_unit(
            expect_nonnegative_int(
                json_field(frontier, "witness_payload_bytes", "frontier row"),
                "frontier row.witness_payload_bytes",
            ),
            observed_count,
            f"{family_context}.witness_payload_bytes_per_unit",
        )
        expected_trace_per = exact_projection_unit(
            expect_nonnegative_int(
                json_field(frontier, "trace_payload_bytes", "frontier row"),
                "frontier row.trace_payload_bytes",
            ),
            observed_count,
            f"{family_context}.trace_payload_bytes_per_unit",
        )
        expected_recomputed_update_per = exact_projection_unit(
            expect_nonnegative_int(
                json_field(frontier, "recomputed_update_payload_bytes", "frontier row"),
                "frontier row.recomputed_update_payload_bytes",
            ),
            observed_count,
            f"{family_context}.recomputed_update_payload_bytes_per_unit",
        )
        expected_forward_per = exact_projection_unit(
            expect_positive_int(
                json_field(frontier, "forward_recompute_steps", "frontier row"),
                "frontier row.forward_recompute_steps",
            ),
            observed_count,
            f"{family_context}.forward_recompute_steps_per_unit",
        )
        expected_inverse_per = exact_projection_unit(
            expect_positive_int(
                json_field(frontier, "inverse_recompute_steps", "frontier row"),
                "frontier row.inverse_recompute_steps",
            ),
            observed_count,
            f"{family_context}.inverse_recompute_steps_per_unit",
        )
        if sample_per != expected_sample_per:
            raise ValueError(f"{family_context}.sample_payload_bytes_per_unit must match recompute frontier")
        if witness_per != expected_witness_per:
            raise ValueError(f"{family_context}.witness_payload_bytes_per_unit must match recompute frontier")
        if trace_per != expected_trace_per:
            raise ValueError(f"{family_context}.trace_payload_bytes_per_unit must match recompute frontier")
        if recomputed_update_per != expected_recomputed_update_per:
            raise ValueError(
                f"{family_context}.recomputed_update_payload_bytes_per_unit must match recompute frontier"
            )
        if forward_per != expected_forward_per or inverse_per != expected_inverse_per:
            raise ValueError(f"{family_context}.recompute_steps_per_unit must match recompute frontier")
        if family_id in ("training_trace", "mlp_witness_trace"):
            expected_variable_per = trace_per
        else:
            expected_variable_per = sample_per + witness_per + trace_per
        if variable_per != expected_variable_per:
            raise ValueError(f"{family_context}.variable_replay_payload_bytes_per_unit must match replay model")
        observed_replay = expect_positive_int(
            json_field(family, "observed_replay_payload_bytes", family_context),
            f"{family_context}.observed_replay_payload_bytes",
        )
        frontier_replay = expect_positive_int(
            json_field(frontier, "replay_payload_bytes", "frontier row"),
            "frontier row.replay_payload_bytes",
        )
        if observed_replay != frontier_replay:
            raise ValueError(f"{family_context}.observed_replay_payload_bytes must match recompute frontier")
        projections = json_field(family, "projections", family_context)
        if not isinstance(projections, list) or len(projections) != len(SCALING_PROJECTION_SCALES):
            raise ValueError(f"{family_context}.projections must contain observed, 10x, and 100x rows")
        checked_projections = []
        for projection_index, projection in enumerate(projections):
            projection_context = f"{family_context}.projections[{projection_index}]"
            if not isinstance(projection, dict):
                raise ValueError(f"{projection_context} must be an object")
            scale = expect_positive_int(
                json_field(projection, "scale_factor", projection_context),
                f"{projection_context}.scale_factor",
            )
            if scale != SCALING_PROJECTION_SCALES[projection_index]:
                raise ValueError(f"{projection_context}.scale_factor must match projection scale")
            count = expect_positive_int(
                json_field(projection, "count", projection_context),
                f"{projection_context}.count",
            )
            if count != observed_count * scale:
                raise ValueError(f"{projection_context}.count must equal observed count times scale")
            expected_replay = model_payload + count * variable_per
            if expect_positive_int(
                json_field(projection, "projected_replay_payload_bytes", projection_context),
                f"{projection_context}.projected_replay_payload_bytes",
            ) != expected_replay:
                raise ValueError(f"{projection_context}.projected_replay_payload_bytes must match formula")
            if expect_nonnegative_int(
                json_field(projection, "projected_sample_payload_bytes", projection_context),
                f"{projection_context}.projected_sample_payload_bytes",
            ) != count * sample_per:
                raise ValueError(f"{projection_context}.projected_sample_payload_bytes must match formula")
            if expect_nonnegative_int(
                json_field(projection, "projected_witness_payload_bytes", projection_context),
                f"{projection_context}.projected_witness_payload_bytes",
            ) != count * witness_per:
                raise ValueError(f"{projection_context}.projected_witness_payload_bytes must match formula")
            if expect_nonnegative_int(
                json_field(projection, "projected_trace_payload_bytes", projection_context),
                f"{projection_context}.projected_trace_payload_bytes",
            ) != count * trace_per:
                raise ValueError(f"{projection_context}.projected_trace_payload_bytes must match formula")
            if expect_nonnegative_int(
                json_field(projection, "projected_recomputed_update_payload_bytes", projection_context),
                f"{projection_context}.projected_recomputed_update_payload_bytes",
            ) != count * recomputed_update_per:
                raise ValueError(
                    f"{projection_context}.projected_recomputed_update_payload_bytes must match formula"
                )
            forward = expect_positive_int(
                json_field(projection, "projected_forward_recompute_steps", projection_context),
                f"{projection_context}.projected_forward_recompute_steps",
            )
            inverse = expect_positive_int(
                json_field(projection, "projected_inverse_recompute_steps", projection_context),
                f"{projection_context}.projected_inverse_recompute_steps",
            )
            if forward != count * forward_per:
                raise ValueError(f"{projection_context}.projected_forward_recompute_steps must match formula")
            if inverse != count * inverse_per:
                raise ValueError(f"{projection_context}.projected_inverse_recompute_steps must match formula")
            balanced = expect_bool(
                json_field(projection, "balanced_recompute", projection_context),
                f"{projection_context}.balanced_recompute",
            )
            if balanced != (forward == inverse and forward > 0):
                raise ValueError(f"{projection_context}.balanced_recompute must summarize recompute steps")
            validate_frontier_ratio(
                json_field(projection, "projected_bytes_per_inverse_step", projection_context),
                expected_replay,
                inverse,
                f"{projection_context}.projected_bytes_per_inverse_step",
            )
            checked_projections.append(
                {
                    "count": count,
                    "scale_factor": scale,
                    "projected_replay_payload_bytes": expected_replay,
                    "balanced_recompute": balanced,
                }
            )
        if observed_replay != checked_projections[0]["projected_replay_payload_bytes"]:
            raise ValueError(f"{family_context}.observed_replay_payload_bytes must match observed projection")
        checked_families.append(
            {
                "id": family_id,
                "observed_replay_payload_bytes": observed_replay,
                "projections": checked_projections,
            }
        )
    summary = json_field(value, "summary", context)
    if not isinstance(summary, dict):
        raise ValueError(f"{context}.summary must be an object")
    if expect_positive_int(json_field(summary, "families", f"{context}.summary"), f"{context}.summary.families") != len(SCALING_PROJECTION_FAMILIES):
        raise ValueError(f"{context}.summary.families must match family count")
    scales = [
        expect_positive_int(item, f"{context}.summary.projection_scales[{index}]")
        for index, item in enumerate(json_field(summary, "projection_scales", f"{context}.summary"))
    ]
    if scales != list(SCALING_PROJECTION_SCALES):
        raise ValueError(f"{context}.summary.projection_scales must match supported scales")
    observed_total = sum(family["observed_replay_payload_bytes"] for family in checked_families)
    if expect_positive_int(
        json_field(summary, "total_observed_replay_payload_bytes", f"{context}.summary"),
        f"{context}.summary.total_observed_replay_payload_bytes",
    ) != observed_total:
        raise ValueError(f"{context}.summary.total_observed_replay_payload_bytes must match families")
    hundred_x = [
        projection
        for family in checked_families
        for projection in family["projections"]
        if projection["scale_factor"] == 100
    ]
    hundred_x_total = sum(projection["projected_replay_payload_bytes"] for projection in hundred_x)
    if expect_positive_int(
        json_field(summary, "total_projected_replay_payload_bytes_at_100x", f"{context}.summary"),
        f"{context}.summary.total_projected_replay_payload_bytes_at_100x",
    ) != hundred_x_total:
        raise ValueError(
            f"{context}.summary.total_projected_replay_payload_bytes_at_100x must match projections"
        )
    all_projections = [
        (family, projection)
        for family in checked_families
        for projection in family["projections"]
    ]
    largest = max(all_projections, key=lambda item: item[1]["projected_replay_payload_bytes"])
    if expect_positive_int(
        json_field(summary, "max_projected_replay_payload_bytes", f"{context}.summary"),
        f"{context}.summary.max_projected_replay_payload_bytes",
    ) != largest[1]["projected_replay_payload_bytes"]:
        raise ValueError(f"{context}.summary.max_projected_replay_payload_bytes must match projections")
    if json_field(summary, "max_projected_replay_family", f"{context}.summary") != largest[0]["id"]:
        raise ValueError(f"{context}.summary.max_projected_replay_family must match projections")
    if expect_positive_int(
        json_field(summary, "max_projected_count", f"{context}.summary"),
        f"{context}.summary.max_projected_count",
    ) != largest[1]["count"]:
        raise ValueError(f"{context}.summary.max_projected_count must match projections")
    all_balanced = expect_bool(
        json_field(summary, "all_balanced", f"{context}.summary"),
        f"{context}.summary.all_balanced",
    )
    if all_balanced != all(projection["balanced_recompute"] for _, projection in all_projections):
        raise ValueError(f"{context}.summary.all_balanced must match projections")


def validate_pipeline_scorecard(
    value: Any,
    context: str,
    *,
    metrics: dict[str, Any],
    gate_checks: dict[str, Any],
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if json_field(value, "kind", context) != "reverie_mnist_ml_v6_scorecard":
        raise ValueError(f"{context}.kind must be reverie_mnist_ml_v6_scorecard")
    train = json_field(metrics, "train", "metrics")
    built_in_eval = json_field(metrics, "built_in_eval", "metrics")
    reverse = json_field(metrics, "reverse", "metrics")
    reverse_cost = json_field(metrics, "reverse_check_cost", "metrics")
    memory = json_field(metrics, "memory", "metrics")
    artifact_profile = json_field(metrics, "artifact_profile", "metrics")
    model_eval = json_field(metrics, "model_evaluation", "metrics")
    replay_payloads = json_field(metrics, "replay_payload_bytes", "metrics")
    for field, expected in (
        ("train_updates_per_second", json_field(train, "updates_per_second", "metrics.train")),
        (
            "built_in_eval_samples_per_second",
            json_field(built_in_eval, "samples_per_second", "metrics.built_in_eval"),
        ),
        (
            "model_evaluation_samples_per_second",
            json_field(model_eval, "samples_per_second", "metrics.model_evaluation"),
        ),
        ("reverse_steps_per_second", json_field(reverse, "steps_per_second", "metrics.reverse")),
        (
            "reverse_to_train_elapsed_ratio",
            json_field(reverse_cost, "reverse_to_train_elapsed_ratio", "metrics.reverse_check_cost"),
        ),
        ("run_peak_rss_bytes", json_field(memory, "run_peak_rss_bytes", "metrics.memory")),
        ("estimated_payload_bytes", json_field(memory, "estimated_payload_bytes", "metrics.memory")),
        (
            "total_model_payload_bytes",
            json_field(artifact_profile, "total_model_payload_bytes", "metrics.artifact_profile"),
        ),
        (
            "total_witness_payload_bytes",
            json_field(artifact_profile, "total_witness_payload_bytes", "metrics.artifact_profile"),
        ),
        (
            "total_trace_payload_bytes",
            json_field(artifact_profile, "total_trace_payload_bytes", "metrics.artifact_profile"),
        ),
        (
            "trace_to_model_payload_ratio",
            json_field(artifact_profile, "trace_to_model_payload_ratio", "metrics.artifact_profile"),
        ),
        (
            "witness_to_model_payload_ratio",
            json_field(artifact_profile, "witness_to_model_payload_ratio", "metrics.artifact_profile"),
        ),
        ("max_replay_payload_bytes", max(replay_payloads.values())),
        (
            "total_recompute_steps",
            json_field(artifact_profile, "total_recompute_steps", "metrics.artifact_profile"),
        ),
        (
            "total_forward_recompute_steps",
            json_field(artifact_profile, "total_forward_recompute_steps", "metrics.artifact_profile"),
        ),
        (
            "total_inverse_recompute_steps",
            json_field(artifact_profile, "total_inverse_recompute_steps", "metrics.artifact_profile"),
        ),
    ):
        actual = json_field(value, field, context)
        if isinstance(expected, float):
            if not math.isclose(
                expect_finite_number(actual, f"{context}.{field}"),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{context}.{field} must match metrics")
        elif actual != expected:
            raise ValueError(f"{context}.{field} must match metrics")
    if expect_finite_number(
        json_field(value, "max_reverse_train_elapsed_ratio", context),
        f"{context}.max_reverse_train_elapsed_ratio",
    ) < 0.0:
        raise ValueError(f"{context}.max_reverse_train_elapsed_ratio must be non-negative")
    balanced = expect_bool(json_field(value, "balanced_recompute", context), f"{context}.balanced_recompute")
    if balanced != (
        value["total_forward_recompute_steps"] == value["total_inverse_recompute_steps"]
        and value["total_forward_recompute_steps"] > 0
    ):
        raise ValueError(f"{context}.balanced_recompute must match recompute steps")
    contracts = validate_exact_key_object(
        json_field(value, "contracts", context),
        (
            "training_step_debug",
            "model_import",
            "native_inference_explanation",
            "evaluation_row_inference_explanation",
            "q31_reference_inference",
            "inference_trace_profile",
            "mlp_witness",
            "invertible_coupling",
            "triangular_residual",
            "reversible_preprocess",
            "reversible_inference_trace",
        ),
        f"{context}.contracts",
    )
    for name, actual in contracts.items():
        if not expect_bool(actual, f"{context}.contracts.{name}"):
            raise ValueError(f"{context}.contracts.{name} must be true")
    for field in ("base_gates_passed", "gates_passed"):
        if not expect_bool(json_field(value, field, context), f"{context}.{field}"):
            raise ValueError(f"{context}.{field} must be true")
    for field in ("base_failed_gates", "failed_gates"):
        failed = json_field(value, field, context)
        if failed != []:
            raise ValueError(f"{context}.{field} must be empty")
    scorecard_gate = json_field(gate_checks, "v6_scorecard_complete", "gates.checks")
    if not expect_bool(json_field(scorecard_gate, "passed", "v6_scorecard_complete"), "v6_scorecard_complete.passed"):
        raise ValueError("v6_scorecard_complete gate must pass")


def validate_pipeline_ml_capability_map(
    value: Any,
    context: str,
    *,
    gate_checks: dict[str, Any],
    metrics: dict[str, Any],
    evidence_files: dict[str, Any],
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if json_field(value, "kind", context) != ML_CAPABILITY_KIND:
        raise ValueError(f"{context}.kind must be `{ML_CAPABILITY_KIND}`")
    if json_field(value, "north_star", context) != "reversible_inspectable_deterministic_ml_kernels":
        raise ValueError(f"{context}.north_star must describe the reversible ML kernel goal")
    capabilities = json_field(value, "capabilities", context)
    if not isinstance(capabilities, list) or len(capabilities) != len(PIPELINE_CAPABILITIES):
        raise ValueError(f"{context}.capabilities must contain the V1-V6 capability rows")
    summary = json_field(value, "summary", context)
    if not isinstance(summary, dict):
        raise ValueError(f"{context}.summary must be an object")
    passed_values = []
    for index, (row, expected) in enumerate(zip(capabilities, PIPELINE_CAPABILITIES)):
        row_context = f"{context}.capabilities[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{row_context} must be an object")
        expected_phase, expected_id, expected_gates = expected
        if json_field(row, "phase", row_context) != expected_phase:
            raise ValueError(f"{row_context}.phase must be `{expected_phase}`")
        if json_field(row, "id", row_context) != expected_id:
            raise ValueError(f"{row_context}.id must be `{expected_id}`")
        validate_nonempty_string(json_field(row, "goal", row_context), f"{row_context}.goal")
        validate_nonempty_string(json_field(row, "description", row_context), f"{row_context}.description")
        gate_metrics = validate_string_list(
            json_field(row, "gate_metrics", row_context),
            f"{row_context}.gate_metrics",
        )
        if tuple(gate_metrics) != expected_gates:
            raise ValueError(f"{row_context}.gate_metrics must match the roadmap gate catalog")
        expected_blocking = []
        for metric in gate_metrics:
            gate = json_field(gate_checks, metric, "gates.checks")
            if expect_bool(json_field(gate, "passed", f"gates.checks.{metric}"), f"gates.checks.{metric}.passed") is not True:
                expected_blocking.append(metric)
        blocking_gates = validate_string_list(
            json_field(row, "blocking_gates", row_context),
            f"{row_context}.blocking_gates",
            allow_empty=True,
        )
        if blocking_gates != expected_blocking:
            raise ValueError(f"{row_context}.blocking_gates must match failed gate metrics")
        evidence = validate_string_list(json_field(row, "evidence", row_context), f"{row_context}.evidence")
        for evidence_key in evidence:
            if evidence_key not in evidence_files:
                raise ValueError(f"{row_context}.evidence references unknown file `{evidence_key}`")
        metric_refs = validate_string_list(json_field(row, "metrics", row_context), f"{row_context}.metrics")
        for metric_ref in metric_refs:
            if metric_ref not in metrics:
                raise ValueError(f"{row_context}.metrics references unknown metric `{metric_ref}`")
        passed = expect_bool(json_field(row, "passed", row_context), f"{row_context}.passed")
        expected_passed = not expected_blocking
        if passed != expected_passed:
            raise ValueError(f"{row_context}.passed must summarize its gate metrics")
        expected_status = "passed" if expected_passed else "failed"
        if json_field(row, "status", row_context) != expected_status:
            raise ValueError(f"{row_context}.status must match passed")
        passed_values.append(passed)
    total = expect_positive_int(json_field(summary, "total", f"{context}.summary"), f"{context}.summary.total")
    if total != len(PIPELINE_CAPABILITIES):
        raise ValueError(f"{context}.summary.total must equal the V1-V6 row count")
    passed_count = expect_nonnegative_int(
        json_field(summary, "passed", f"{context}.summary"),
        f"{context}.summary.passed",
    )
    failed_count = expect_nonnegative_int(
        json_field(summary, "failed", f"{context}.summary"),
        f"{context}.summary.failed",
    )
    if passed_count != sum(1 for passed in passed_values if passed):
        raise ValueError(f"{context}.summary.passed must match capability rows")
    if failed_count != total - passed_count:
        raise ValueError(f"{context}.summary.failed must match capability rows")
    failed_capabilities = validate_string_list(
        json_field(summary, "failed_capabilities", f"{context}.summary"),
        f"{context}.summary.failed_capabilities",
        allow_empty=True,
    )
    expected_failed = [
        row["id"]
        for row, passed in zip(capabilities, passed_values)
        if not passed
    ]
    if failed_capabilities != expected_failed:
        raise ValueError(f"{context}.summary.failed_capabilities must match failed rows")
    map_passed = expect_bool(json_field(value, "passed", context), f"{context}.passed")
    if map_passed != all(passed_values):
        raise ValueError(f"{context}.passed must summarize capability rows")
    if not map_passed:
        raise ValueError(f"{context}.passed must be true")
    capability_gate = json_field(gate_checks, "ml_roadmap_capability_map_complete", "gates.checks")
    if not expect_bool(
        json_field(capability_gate, "passed", "ml_roadmap_capability_map_complete"),
        "ml_roadmap_capability_map_complete.passed",
    ):
        raise ValueError("ml_roadmap_capability_map_complete gate must pass")


def validate_pipeline_ml_goal_readiness(
    value: Any,
    context: str,
    *,
    gate_checks: dict[str, Any],
    metrics: dict[str, Any],
    evidence_files: dict[str, Any],
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if json_field(value, "kind", context) != ML_GOAL_READINESS_KIND:
        raise ValueError(f"{context}.kind must be `{ML_GOAL_READINESS_KIND}`")
    if (
        json_field(value, "north_star", context)
        != "best_small_language_for_reversible_inspectable_deterministic_ml_kernels"
    ):
        raise ValueError(f"{context}.north_star must describe the reversible ML kernel goal")
    if (
        json_field(value, "non_goal", context)
        != "general_purpose_pytorch_tensorflow_training_replacement"
    ):
        raise ValueError(f"{context}.non_goal must preserve the non-goal")
    goals = json_field(value, "goals", context)
    if not isinstance(goals, list) or len(goals) != len(PIPELINE_GOAL_READINESS):
        raise ValueError(f"{context}.goals must contain the north-star readiness rows")
    summary = json_field(value, "summary", context)
    if not isinstance(summary, dict):
        raise ValueError(f"{context}.summary must be an object")
    passed_values = []
    for index, (row, expected) in enumerate(zip(goals, PIPELINE_GOAL_READINESS)):
        row_context = f"{context}.goals[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{row_context} must be an object")
        expected_id, expected_gates = expected
        if json_field(row, "id", row_context) != expected_id:
            raise ValueError(f"{row_context}.id must be `{expected_id}`")
        validate_nonempty_string(json_field(row, "goal", row_context), f"{row_context}.goal")
        validate_nonempty_string(
            json_field(row, "description", row_context),
            f"{row_context}.description",
        )
        gate_metrics = validate_string_list(
            json_field(row, "gate_metrics", row_context),
            f"{row_context}.gate_metrics",
        )
        if tuple(gate_metrics) != expected_gates:
            raise ValueError(f"{row_context}.gate_metrics must match the readiness gate catalog")
        expected_blocking = []
        for metric in gate_metrics:
            gate = json_field(gate_checks, metric, "gates.checks")
            if expect_bool(json_field(gate, "passed", f"gates.checks.{metric}"), f"gates.checks.{metric}.passed") is not True:
                expected_blocking.append(metric)
        blocking_gates = validate_string_list(
            json_field(row, "blocking_gates", row_context),
            f"{row_context}.blocking_gates",
            allow_empty=True,
        )
        if blocking_gates != expected_blocking:
            raise ValueError(f"{row_context}.blocking_gates must match failed gate metrics")
        evidence = validate_string_list(json_field(row, "evidence", row_context), f"{row_context}.evidence")
        for evidence_key in evidence:
            if evidence_key not in evidence_files:
                raise ValueError(f"{row_context}.evidence references unknown file `{evidence_key}`")
        metric_refs = validate_string_list(json_field(row, "metrics", row_context), f"{row_context}.metrics")
        for metric_ref in metric_refs:
            if metric_ref not in metrics:
                raise ValueError(f"{row_context}.metrics references unknown metric `{metric_ref}`")
        passed = expect_bool(json_field(row, "passed", row_context), f"{row_context}.passed")
        expected_passed = not expected_blocking
        if passed != expected_passed:
            raise ValueError(f"{row_context}.passed must summarize its gate metrics")
        expected_status = "passed" if expected_passed else "failed"
        if json_field(row, "status", row_context) != expected_status:
            raise ValueError(f"{row_context}.status must match passed")
        passed_values.append(passed)
    total = expect_positive_int(json_field(summary, "total", f"{context}.summary"), f"{context}.summary.total")
    if total != len(PIPELINE_GOAL_READINESS):
        raise ValueError(f"{context}.summary.total must equal the readiness row count")
    passed_count = expect_nonnegative_int(
        json_field(summary, "passed", f"{context}.summary"),
        f"{context}.summary.passed",
    )
    failed_count = expect_nonnegative_int(
        json_field(summary, "failed", f"{context}.summary"),
        f"{context}.summary.failed",
    )
    if passed_count != sum(1 for passed in passed_values if passed):
        raise ValueError(f"{context}.summary.passed must match readiness rows")
    if failed_count != total - passed_count:
        raise ValueError(f"{context}.summary.failed must match readiness rows")
    failed_goals = validate_string_list(
        json_field(summary, "failed_goals", f"{context}.summary"),
        f"{context}.summary.failed_goals",
        allow_empty=True,
    )
    expected_failed = [
        row["id"]
        for row, passed in zip(goals, passed_values)
        if not passed
    ]
    if failed_goals != expected_failed:
        raise ValueError(f"{context}.summary.failed_goals must match failed rows")
    readiness_passed = expect_bool(json_field(value, "passed", context), f"{context}.passed")
    if readiness_passed != all(passed_values):
        raise ValueError(f"{context}.passed must summarize readiness rows")
    if not readiness_passed:
        raise ValueError(f"{context}.passed must be true")
    readiness_gate = json_field(gate_checks, "ml_goal_readiness_complete", "gates.checks")
    if not expect_bool(
        json_field(readiness_gate, "passed", "ml_goal_readiness_complete"),
        "ml_goal_readiness_complete.passed",
    ):
        raise ValueError("ml_goal_readiness_complete gate must pass")


def validate_pipeline_summary_report(report: dict[str, Any]) -> None:
    if json_field(report, "pipeline_kind", "pipeline summary") != PIPELINE_KIND:
        raise ValueError(f"pipeline_kind must be `{PIPELINE_KIND}`")
    sample_limit = expect_positive_int(
        json_field(report, "sample_limit", "pipeline summary"),
        "sample_limit",
    )
    requested_audit_step = expect_nonnegative_int(
        json_field(report, "requested_audit_step", "pipeline summary"),
        "requested_audit_step",
    )
    audit_step = expect_nonnegative_int(json_field(report, "audit_step", "pipeline summary"), "audit_step")
    audit_step_strategy = validate_nonempty_string(
        json_field(report, "audit_step_strategy", "pipeline summary"),
        "audit_step_strategy",
    )
    if audit_step_strategy not in AUDIT_STEP_STRATEGIES:
        raise ValueError("audit_step_strategy is invalid")
    evaluation_row = expect_nonnegative_int(
        json_field(report, "evaluation_row", "pipeline summary"),
        "evaluation_row",
    )
    if requested_audit_step >= sample_limit:
        raise ValueError("requested_audit_step must be less than sample_limit")
    if audit_step >= sample_limit:
        raise ValueError("audit_step must be less than sample_limit")
    requested_evaluation_row = expect_nonnegative_int(
        json_field(report, "requested_evaluation_row", "pipeline summary"),
        "requested_evaluation_row",
    )
    if evaluation_row >= sample_limit:
        raise ValueError("evaluation_row must be less than sample_limit")
    evaluation_row_strategy = validate_nonempty_string(
        json_field(report, "evaluation_row_strategy", "pipeline summary"),
        "evaluation_row_strategy",
    )
    if evaluation_row_strategy not in EVALUATION_ROW_STRATEGIES:
        raise ValueError("evaluation_row_strategy is invalid")
    if requested_evaluation_row >= sample_limit:
        raise ValueError("requested_evaluation_row must be less than sample_limit")
    validate_nonempty_string(json_field(report, "profile_markdown", "pipeline summary"), "profile_markdown")
    reports = validate_string_map_keys(json_field(report, "reports", "pipeline summary"), PIPELINE_REPORT_KEYS, "reports")
    bundles = validate_string_map_keys(json_field(report, "bundles", "pipeline summary"), PIPELINE_BUNDLE_KEYS, "bundles")
    evidence = json_field(report, "evidence", "pipeline summary")
    expected_evidence_paths = {
        "profile_markdown": report["profile_markdown"],
        **{f"report.{key}": reports[key] for key in PIPELINE_REPORT_KEYS},
        **{f"bundle.{key}": bundles[key] for key in PIPELINE_BUNDLE_KEYS},
        **pipeline_standalone_evidence_paths(evidence, "evidence"),
    }
    validate_pipeline_evidence_index(
        evidence,
        expected_evidence_paths,
        "evidence",
    )

    metrics = json_field(report, "metrics", "pipeline summary")
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an object")
    train_samples, _, _ = validate_pipeline_accuracy_block(
        json_field(metrics, "train", "metrics"),
        "metrics.train",
        rate_field="updates_per_second",
    )
    validate_pipeline_accuracy_block(
        json_field(metrics, "built_in_eval", "metrics"),
        "metrics.built_in_eval",
        rate_field="samples_per_second",
    )
    reverse = json_field(metrics, "reverse", "metrics")
    if not isinstance(reverse, dict):
        raise ValueError("metrics.reverse must be an object")
    if not expect_bool(json_field(reverse, "enabled", "reverse"), "reverse.enabled"):
        raise ValueError("reverse.enabled must be true")
    if not expect_bool(
        json_field(reverse, "restored_initial_model", "reverse"),
        "reverse.restored_initial_model",
    ):
        raise ValueError("reverse.restored_initial_model must be true")
    checked = expect_positive_int(json_field(reverse, "checked", "reverse"), "reverse.checked")
    if checked != train_samples:
        raise ValueError("reverse.checked must equal train samples")
    if expect_finite_number(json_field(reverse, "steps_per_second", "reverse"), "reverse.steps_per_second") < 0.0:
        raise ValueError("reverse.steps_per_second must be non-negative")
    reverse_check_cost = json_field(metrics, "reverse_check_cost", "metrics")
    validate_pipeline_reverse_check_cost(
        reverse_check_cost,
        "metrics.reverse_check_cost",
        train_samples,
    )

    memory = json_field(metrics, "memory", "metrics")
    if not isinstance(memory, dict):
        raise ValueError("metrics.memory must be an object")
    expect_positive_int(json_field(memory, "run_peak_rss_bytes", "memory"), "memory.run_peak_rss_bytes")
    expect_optional_peak_rss(
        json_field(memory, "model_evaluation_peak_rss_bytes", "memory"),
        "memory.model_evaluation_peak_rss_bytes",
        required=False,
    )
    expect_positive_int(
        json_field(memory, "estimated_payload_bytes", "memory"),
        "memory.estimated_payload_bytes",
    )

    audit_scan = json_field(metrics, "training_audit_scan", "metrics")
    if not isinstance(audit_scan, dict):
        raise ValueError("metrics.training_audit_scan must be an object")
    audit_steps = expect_positive_int(
        json_field(audit_scan, "steps", "training_audit_scan"),
        "training_audit_scan.steps",
    )
    audit_correct = expect_nonnegative_int(
        json_field(audit_scan, "correct", "training_audit_scan"),
        "training_audit_scan.correct",
    )
    audit_incorrect = expect_nonnegative_int(
        json_field(audit_scan, "incorrect", "training_audit_scan"),
        "training_audit_scan.incorrect",
    )
    if audit_correct + audit_incorrect != audit_steps:
        raise ValueError("training_audit_scan.correct + incorrect must equal steps")
    validate_accuracy(
        json_field(audit_scan, "accuracy_percent", "training_audit_scan"),
        audit_correct,
        audit_steps,
        "training_audit_scan.accuracy_percent",
    )
    expect_nonnegative_int(
        json_field(audit_scan, "witness_mismatches", "training_audit_scan"),
        "training_audit_scan.witness_mismatches",
    )
    expect_positive_int(
        json_field(audit_scan, "trace_payload_bytes", "training_audit_scan"),
        "training_audit_scan.trace_payload_bytes",
    )
    lowest_margin_step = expect_nonnegative_int(
        json_field(audit_scan, "lowest_margin_step", "training_audit_scan"),
        "training_audit_scan.lowest_margin_step",
    )
    if lowest_margin_step >= audit_steps:
        raise ValueError("training_audit_scan.lowest_margin_step must be less than steps")
    expect_i64(
        json_field(audit_scan, "lowest_margin", "training_audit_scan"),
        "training_audit_scan.lowest_margin",
    )
    largest_update_step = expect_nonnegative_int(
        json_field(audit_scan, "largest_update_step", "training_audit_scan"),
        "training_audit_scan.largest_update_step",
    )
    if largest_update_step >= audit_steps:
        raise ValueError("training_audit_scan.largest_update_step must be less than steps")
    expect_nonnegative_int(
        json_field(audit_scan, "max_abs_weight_delta", "training_audit_scan"),
        "training_audit_scan.max_abs_weight_delta",
    )
    if audit_steps != train_samples:
        raise ValueError("training_audit_scan.steps must equal train samples")

    audit_replay = json_field(metrics, "training_audit_replay", "metrics")
    if not isinstance(audit_replay, dict):
        raise ValueError("metrics.training_audit_replay must be an object")
    if expect_positive_int(
        json_field(audit_replay, "checked", "training_audit_replay"),
        "training_audit_replay.checked",
    ) != audit_steps:
        raise ValueError("training_audit_replay.checked must equal audit steps")
    for field in (
        "witnesses_match_forward_replay",
        "final_model_replayed",
        "restored_initial_model",
        "proof_matches",
        "lineage_ledger_matches",
    ):
        if not expect_bool(
            json_field(audit_replay, field, "training_audit_replay"),
            f"training_audit_replay.{field}",
        ):
            raise ValueError(f"training_audit_replay.{field} must be true")
    for field in ("lineage_ledger_fingerprint", "transition_ledger_fingerprint", "final_chain"):
        validate_sha256(
            json_field(audit_replay, field, "training_audit_replay"),
            f"training_audit_replay.{field}",
        )

    forward, inverse = validate_pipeline_artifact_profile(
        json_field(metrics, "artifact_profile", "metrics"),
        "metrics.artifact_profile",
    )
    if forward != inverse:
        raise ValueError("artifact_profile recompute steps must be balanced")

    validate_pipeline_replay_block(
        json_field(metrics, "training_step_replay", "metrics"),
        "metrics.training_step_replay",
        ("proof_matches", "witnesses_match", "after_model_matches", "before_model_restored"),
    )
    validate_pipeline_training_step_debug(
        json_field(metrics, "training_step_debug", "metrics"),
        "metrics.training_step_debug",
    )
    validate_pipeline_training_step_selection(
        json_field(metrics, "training_step_selection", "metrics"),
        "metrics.training_step_selection",
        requested_step=requested_audit_step,
        selected_pipeline_step=audit_step,
        selection_strategy=audit_step_strategy,
        audit_steps=audit_steps,
        audit_scan=audit_scan,
        training_step_debug=json_field(metrics, "training_step_debug", "metrics"),
    )
    model_import = json_field(metrics, "model_import", "metrics")
    if not isinstance(model_import, dict):
        raise ValueError("metrics.model_import must be an object")
    for field in (
        "source_model_json_path",
        "model_output",
        "import_provenance_kind",
        "verification_provenance_kind",
    ):
        validate_nonempty_string(
            json_field(model_import, field, "metrics.model_import"),
            f"metrics.model_import.{field}",
        )
    for field in (
        "source_model_json_fingerprint",
        "source_model_json_file_sha256",
        "import_payload_fingerprint",
        "verification_payload_fingerprint",
    ):
        validate_sha256(
            json_field(model_import, field, "metrics.model_import"),
            f"metrics.model_import.{field}",
        )
    expect_positive_int(
        json_field(model_import, "model_payload_bytes", "metrics.model_import"),
        "metrics.model_import.model_payload_bytes",
    )
    for field in (
        "import_shape_matches",
        "verification_shape_matches",
        "verification_provenance_matches",
        "verification_source_checked",
    ):
        if not expect_bool(
            json_field(model_import, field, "metrics.model_import"),
            f"metrics.model_import.{field}",
        ):
            raise ValueError(f"metrics.model_import.{field} must be true")
    if model_import["import_provenance_kind"] != "external_import":
        raise ValueError("metrics.model_import.import_provenance_kind must be external_import")
    if model_import["verification_provenance_kind"] != "external_import":
        raise ValueError("metrics.model_import.verification_provenance_kind must be external_import")
    if model_import["proof_matches"] is not None:
        raise ValueError("metrics.model_import.proof_matches must be null")
    if model_import["training_steps"] is not None:
        raise ValueError("metrics.model_import.training_steps must be null")
    if model_import["import_payload_fingerprint"] != model_import["verification_payload_fingerprint"]:
        raise ValueError("metrics.model_import payload fingerprints must match")
    validate_imported_model_source_check(
        json_field(model_import, "source_model_json", "metrics.model_import"),
        "metrics.model_import.source_model_json",
    )
    imported_inference = json_field(metrics, "imported_model_inference", "metrics")
    if not isinstance(imported_inference, dict):
        raise ValueError("metrics.imported_model_inference must be an object")
    expect_digit(
        json_field(imported_inference, "prediction", "metrics.imported_model_inference"),
        "metrics.imported_model_inference.prediction",
    )
    expect_digit(
        json_field(
            imported_inference,
            "native_prediction",
            "metrics.imported_model_inference",
        ),
        "metrics.imported_model_inference.native_prediction",
    )
    for field in (
        "correct",
        "correct_matches_native",
        "prediction_matches_native",
        "margin_matches_native",
        "contribution_ledger_matches_native",
        "margin_contribution_ledger_matches_native",
        "proof_matches",
        "result_matches",
        "restored_initial_state",
        "source_model_checked",
        "source_sample_checked",
    ):
        if not expect_bool(
            json_field(imported_inference, field, "metrics.imported_model_inference"),
            f"metrics.imported_model_inference.{field}",
        ):
            raise ValueError(f"metrics.imported_model_inference.{field} must be true")
    expect_positive_int(
        json_field(
            imported_inference,
            "replay_payload_bytes",
            "metrics.imported_model_inference",
        ),
        "metrics.imported_model_inference.replay_payload_bytes",
    )
    expect_positive_int(
        json_field(
            imported_inference,
            "forward_recompute_steps",
            "metrics.imported_model_inference",
        ),
        "metrics.imported_model_inference.forward_recompute_steps",
    )
    expect_positive_int(
        json_field(
            imported_inference,
            "inverse_recompute_steps",
            "metrics.imported_model_inference",
        ),
        "metrics.imported_model_inference.inverse_recompute_steps",
    )
    validate_inference_explanation_contract(
        json_field(metrics, "imported_model_inference_explanation", "metrics"),
        "metrics.imported_model_inference_explanation",
        prediction=expect_digit(
            json_field(
                metrics["imported_model_inference_explanation"],
                "prediction",
                "imported_model_inference_explanation",
            ),
            "imported_model_inference_explanation.prediction",
        ),
        correct=expect_bool(
            json_field(
                metrics["imported_model_inference_explanation"],
                "correct",
                "imported_model_inference_explanation",
            ),
            "imported_model_inference_explanation.correct",
        ),
        margin=expect_i64(
            json_field(
                metrics["imported_model_inference_explanation"],
                "margin",
                "imported_model_inference_explanation",
            ),
            "imported_model_inference_explanation.margin",
        ),
        active_pixels=expect_nonnegative_int(
            json_field(
                metrics["imported_model_inference_explanation"],
                "active_pixel_count",
                "imported_model_inference_explanation",
            ),
            "imported_model_inference_explanation.active_pixel_count",
        ),
        require_verification_checks=True,
    )
    validate_pipeline_replay_block(
        json_field(metrics, "native_inference_replay", "metrics"),
        "metrics.native_inference_replay",
        (
            "proof_matches",
            "result_matches",
            "restored_initial_state",
            "source_model_checked",
            "source_sample_checked",
        ),
    )
    validate_inference_explanation_contract(
        json_field(metrics, "native_inference_explanation", "metrics"),
        "metrics.native_inference_explanation",
        prediction=expect_digit(
            json_field(metrics["native_inference_explanation"], "prediction", "native_inference_explanation"),
            "native_inference_explanation.prediction",
        ),
        correct=expect_bool(
            json_field(metrics["native_inference_explanation"], "correct", "native_inference_explanation"),
            "native_inference_explanation.correct",
        ),
        margin=expect_i64(
            json_field(metrics["native_inference_explanation"], "margin", "native_inference_explanation"),
            "native_inference_explanation.margin",
        ),
        active_pixels=expect_nonnegative_int(
            json_field(
                metrics["native_inference_explanation"],
                "active_pixel_count",
                "native_inference_explanation",
            ),
            "native_inference_explanation.active_pixel_count",
        ),
        require_verification_checks=True,
    )
    model_eval = json_field(metrics, "model_evaluation", "metrics")
    model_samples, model_correct, model_incorrect = validate_pipeline_accuracy_block(
        model_eval,
        "metrics.model_evaluation",
        rate_field="samples_per_second",
        require_incorrect=True,
    )
    if model_samples != sample_limit:
        raise ValueError("model_evaluation.samples must equal sample_limit")
    expect_i64(json_field(model_eval, "lowest_margin", "metrics.model_evaluation"), "metrics.model_evaluation.lowest_margin")
    model_replay = expect_positive_int(
        json_field(model_eval, "replay_payload_bytes", "metrics.model_evaluation"),
        "metrics.model_evaluation.replay_payload_bytes",
    )
    validate_pipeline_bool_block(
        json_field(metrics, "model_evaluation_replay", "metrics"),
        "metrics.model_evaluation_replay",
        (
            "rows_match",
            "proof_matches",
            "restored_initial_state",
            "source_model_checked",
            "source_samples_checked",
        ),
    )
    validate_pipeline_evaluation_row_selection(
        json_field(metrics, "evaluation_row_selection", "metrics"),
        "metrics.evaluation_row_selection",
        requested_row=requested_evaluation_row,
        selected_pipeline_row=evaluation_row,
        selection_strategy=evaluation_row_strategy,
        model_samples=model_samples,
    )
    validate_pipeline_replay_block(
        json_field(metrics, "evaluation_row_inference", "metrics"),
        "metrics.evaluation_row_inference",
        (
            "proof_matches",
            "result_matches",
            "restored_initial_state",
            "source_model_checked",
            "source_evaluation_checked",
        ),
    )
    validate_inference_explanation_contract(
        json_field(metrics, "evaluation_row_inference_explanation", "metrics"),
        "metrics.evaluation_row_inference_explanation",
        prediction=expect_digit(
            json_field(
                metrics["evaluation_row_inference_explanation"],
                "prediction",
                "evaluation_row_inference_explanation",
            ),
            "evaluation_row_inference_explanation.prediction",
        ),
        correct=expect_bool(
            json_field(
                metrics["evaluation_row_inference_explanation"],
                "correct",
                "evaluation_row_inference_explanation",
            ),
            "evaluation_row_inference_explanation.correct",
        ),
        margin=expect_i64(
            json_field(
                metrics["evaluation_row_inference_explanation"],
                "margin",
                "evaluation_row_inference_explanation",
            ),
            "evaluation_row_inference_explanation.margin",
        ),
        active_pixels=expect_nonnegative_int(
            json_field(
                metrics["evaluation_row_inference_explanation"],
                "active_pixel_count",
                "evaluation_row_inference_explanation",
            ),
            "evaluation_row_inference_explanation.active_pixel_count",
        ),
        require_verification_checks=True,
    )
    validate_pipeline_q31_reference_inference_metrics(
        json_field(metrics, "q31_reference_inference", "metrics"),
        "metrics.q31_reference_inference",
    )
    validate_pipeline_inference_trace_profile(
        json_field(metrics, "inference_trace_profile", "metrics"),
        "metrics.inference_trace_profile",
        metrics=metrics,
    )
    reference_eval = json_field(metrics, "q31_reference_evaluation", "metrics")
    reference_samples, reference_correct, reference_incorrect = validate_pipeline_accuracy_block(
        reference_eval,
        "metrics.q31_reference_evaluation",
        require_incorrect=True,
    )
    if reference_samples != model_samples:
        raise ValueError("q31_reference_evaluation.samples must equal model_evaluation.samples")
    if (reference_correct, reference_incorrect) != (model_correct, model_incorrect):
        raise ValueError("Q31 reference evaluation totals must match native model evaluation")
    expect_i64(
        json_field(reference_eval, "lowest_margin", "metrics.q31_reference_evaluation"),
        "metrics.q31_reference_evaluation.lowest_margin",
    )
    validate_pipeline_mlp_witness_metrics(
        json_field(metrics, "mlp_witness", "metrics"),
        "metrics.mlp_witness",
    )
    validate_pipeline_coupling_metrics(
        json_field(metrics, "invertible_coupling", "metrics"),
        "metrics.invertible_coupling",
    )
    validate_pipeline_residual_metrics(
        json_field(metrics, "triangular_residual", "metrics"),
        "metrics.triangular_residual",
    )
    validate_pipeline_preprocess_metrics(
        json_field(metrics, "reversible_preprocess", "metrics"),
        "metrics.reversible_preprocess",
    )
    validate_pipeline_reversible_inference_trace_metrics(
        json_field(metrics, "reversible_inference_trace", "metrics"),
        "metrics.reversible_inference_trace",
    )

    replay_payloads = validate_exact_key_object(
        json_field(metrics, "replay_payload_bytes", "metrics"),
        (
            "training_trace",
            "training_step",
            "imported_model_inference",
            "native_inference",
            "model_evaluation",
            "evaluation_row_inference",
        ),
        "metrics.replay_payload_bytes",
    )
    replay_values = [
        expect_positive_int(replay_payloads[key], f"metrics.replay_payload_bytes.{key}")
        for key in replay_payloads
    ]
    if model_replay != replay_payloads["model_evaluation"]:
        raise ValueError("model_evaluation replay payload must match replay_payload_bytes")
    if imported_inference["replay_payload_bytes"] != replay_payloads["imported_model_inference"]:
        raise ValueError("imported model replay payload must match replay_payload_bytes")
    max_replay = expect_positive_int(
        json_field(metrics, "max_replay_payload_bytes", "metrics"),
        "metrics.max_replay_payload_bytes",
    )
    if max_replay != max(replay_values):
        raise ValueError("max_replay_payload_bytes must equal replay payload maximum")
    gates = validate_pipeline_gates(json_field(report, "gates", "pipeline summary"), "gates")
    gate_checks = {
        check["metric"]: check
        for check in gates["checks"]
        if isinstance(check, dict) and isinstance(check.get("metric"), str)
    }
    reverse_ratio = expect_finite_number(
        json_field(reverse_check_cost, "reverse_to_train_elapsed_ratio", "metrics.reverse_check_cost"),
        "metrics.reverse_check_cost.reverse_to_train_elapsed_ratio",
    )
    max_reverse_ratio = expect_finite_number(
        json_field(gates["policy"], "max_reverse_train_elapsed_ratio", "gates.policy"),
        "gates.policy.max_reverse_train_elapsed_ratio",
    )
    reverse_ratio_gate = json_field(gate_checks, "reverse_check_elapsed_ratio", "gates.checks")
    if expect_finite_number(
        json_field(reverse_ratio_gate, "actual", "reverse_check_elapsed_ratio"),
        "reverse_check_elapsed_ratio.actual",
    ) != round(reverse_ratio, 6):
        raise ValueError("reverse_check_elapsed_ratio actual must match reverse_check_cost ratio")
    expected_reverse_ratio_passed = reverse_ratio <= max_reverse_ratio
    if expect_bool(
        json_field(reverse_ratio_gate, "passed", "reverse_check_elapsed_ratio"),
        "reverse_check_elapsed_ratio.passed",
    ) != expected_reverse_ratio_passed:
        raise ValueError("reverse_check_elapsed_ratio gate result must match policy")
    if json_field(gates["policy"], "audit_step_strategy", "gates.policy") != audit_step_strategy:
        raise ValueError("gate policy audit_step_strategy must match summary")
    if json_field(gates["policy"], "requested_audit_step", "gates.policy") != requested_audit_step:
        raise ValueError("gate policy requested_audit_step must match summary")
    if json_field(gates["policy"], "evaluation_row_strategy", "gates.policy") != evaluation_row_strategy:
        raise ValueError("gate policy evaluation_row_strategy must match summary")
    if json_field(gates["policy"], "requested_evaluation_row", "gates.policy") != requested_evaluation_row:
        raise ValueError("gate policy requested_evaluation_row must match summary")
    validate_pipeline_recompute_frontier(
        json_field(metrics, "recompute_frontier", "metrics"),
        "metrics.recompute_frontier",
        metrics=metrics,
    )
    validate_pipeline_scaling_projection(
        json_field(metrics, "scaling_projection", "metrics"),
        "metrics.scaling_projection",
        metrics=metrics,
    )
    validate_pipeline_scorecard(
        json_field(metrics, "scorecard", "metrics"),
        "metrics.scorecard",
        metrics=metrics,
        gate_checks=gate_checks,
    )
    if not math.isclose(
        expect_finite_number(
            json_field(metrics["scorecard"], "max_reverse_train_elapsed_ratio", "metrics.scorecard"),
            "metrics.scorecard.max_reverse_train_elapsed_ratio",
        ),
        max_reverse_ratio,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("scorecard max_reverse_train_elapsed_ratio must match gate policy")
    evidence = json_field(report, "evidence", "pipeline summary")
    validate_pipeline_ml_capability_map(
        json_field(metrics, "ml_capability_map", "metrics"),
        "metrics.ml_capability_map",
        gate_checks=gate_checks,
        metrics=metrics,
        evidence_files=json_field(evidence, "files", "evidence"),
    )
    validate_pipeline_ml_goal_readiness(
        json_field(metrics, "ml_goal_readiness", "metrics"),
        "metrics.ml_goal_readiness",
        gate_checks=gate_checks,
        metrics=metrics,
        evidence_files=json_field(evidence, "files", "evidence"),
    )
    validate_pipeline_claims(
        json_field(report, "claims", "pipeline summary"),
        gates=gates,
        metrics=metrics,
        evidence_files=json_field(evidence, "files", "evidence"),
        context="claims",
    )


def validate_pipeline_manifest_report(report: dict[str, Any]) -> None:
    sample_limit = expect_positive_int(
        json_field(report, "sample_limit", "pipeline manifest"),
        "sample_limit",
    )
    requested_audit_step = expect_nonnegative_int(
        json_field(report, "requested_audit_step", "pipeline manifest"),
        "requested_audit_step",
    )
    audit_step = expect_nonnegative_int(json_field(report, "audit_step", "pipeline manifest"), "audit_step")
    audit_step_strategy = validate_nonempty_string(
        json_field(report, "audit_step_strategy", "pipeline manifest"),
        "audit_step_strategy",
    )
    if audit_step_strategy not in AUDIT_STEP_STRATEGIES:
        raise ValueError("audit_step_strategy is invalid")
    evaluation_row = expect_nonnegative_int(
        json_field(report, "evaluation_row", "pipeline manifest"),
        "evaluation_row",
    )
    if requested_audit_step >= sample_limit:
        raise ValueError("requested_audit_step must be less than sample_limit")
    if audit_step >= sample_limit:
        raise ValueError("audit_step must be less than sample_limit")
    requested_evaluation_row = expect_nonnegative_int(
        json_field(report, "requested_evaluation_row", "pipeline manifest"),
        "requested_evaluation_row",
    )
    evaluation_row_strategy = validate_nonempty_string(
        json_field(report, "evaluation_row_strategy", "pipeline manifest"),
        "evaluation_row_strategy",
    )
    if evaluation_row_strategy not in EVALUATION_ROW_STRATEGIES:
        raise ValueError("evaluation_row_strategy is invalid")
    if evaluation_row >= sample_limit:
        raise ValueError("evaluation_row must be less than sample_limit")
    if requested_evaluation_row >= sample_limit:
        raise ValueError("requested_evaluation_row must be less than sample_limit")
    validate_nonempty_string(json_field(report, "profile_markdown", "pipeline manifest"), "profile_markdown")
    validate_nonempty_string(json_field(report, "summary", "pipeline manifest"), "summary")
    validate_nonempty_string(json_field(report, "model_capsule", "pipeline manifest"), "model_capsule")
    validate_sha256(
        json_field(report, "model_capsule_fingerprint", "pipeline manifest"),
        "model_capsule_fingerprint",
    )
    if not expect_bool(json_field(report, "gates_passed", "pipeline manifest"), "gates_passed"):
        raise ValueError("gates_passed must be true")
    gate_policy = validate_pipeline_gate_policy(json_field(report, "gate_policy", "pipeline manifest"), "gate_policy")
    if gate_policy["audit_step_strategy"] != audit_step_strategy:
        raise ValueError("gate_policy audit_step_strategy must match manifest")
    if gate_policy["requested_audit_step"] != requested_audit_step:
        raise ValueError("gate_policy requested_audit_step must match manifest")
    if gate_policy["evaluation_row_strategy"] != evaluation_row_strategy:
        raise ValueError("gate_policy evaluation_row_strategy must match manifest")
    if gate_policy["requested_evaluation_row"] != requested_evaluation_row:
        raise ValueError("gate_policy requested_evaluation_row must match manifest")

    reports = json_field(report, "reports", "pipeline manifest")
    if not isinstance(reports, list) or len(reports) != len(PIPELINE_REPORT_KEYS):
        raise ValueError("reports must list every pipeline profile report")
    if len(set(reports)) != len(reports):
        raise ValueError("reports must be unique")
    for index, path in enumerate(reports):
        validate_nonempty_string(path, f"reports[{index}]")
    bundles = json_field(report, "bundles", "pipeline manifest")
    if not isinstance(bundles, list) or len(bundles) != len(PIPELINE_BUNDLE_KEYS):
        raise ValueError("bundles must list every pipeline bundle")
    if len(set(bundles)) != len(bundles):
        raise ValueError("bundles must be unique")
    for index, path in enumerate(bundles):
        validate_nonempty_string(path, f"bundles[{index}]")
    evidence = json_field(report, "evidence", "pipeline manifest")
    expected_evidence_paths = {
        "profile_markdown": report["profile_markdown"],
        "summary": report["summary"],
        **{f"report.{key}": reports[index] for index, key in enumerate(PIPELINE_REPORT_KEYS)},
        **{f"bundle.{key}": bundles[index] for index, key in enumerate(PIPELINE_BUNDLE_KEYS)},
        **pipeline_standalone_evidence_paths(evidence, "evidence"),
    }
    validate_pipeline_evidence_index(
        evidence,
        expected_evidence_paths,
        "evidence",
    )

    steps = json_field(report, "steps", "pipeline manifest")
    if not isinstance(steps, list) or not steps:
        raise ValueError("steps must be a non-empty array")
    for index, step in enumerate(steps):
        context = f"steps[{index}]"
        if not isinstance(step, dict):
            raise ValueError(f"{context} must be an object")
        validate_nonempty_string(json_field(step, "label", context), f"{context}.label")
        output = json_field(step, "output", context)
        if output is not None:
            validate_nonempty_string(output, f"{context}.output")
        command = json_field(step, "command", context)
        if not isinstance(command, list) or not command:
            raise ValueError(f"{context}.command must be a non-empty array")
        for arg_index, argument in enumerate(command):
            validate_nonempty_string(argument, f"{context}.command[{arg_index}]")


def validate_capsule_passed_catalog(value: Any, kind: str, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if json_field(value, "kind", context) != kind:
        raise ValueError(f"{context}.kind must be `{kind}`")
    if not expect_bool(json_field(value, "passed", context), f"{context}.passed"):
        raise ValueError(f"{context}.passed must be true")
    summary = json_field(value, "summary", context)
    if not isinstance(summary, dict):
        raise ValueError(f"{context}.summary must be an object")
    total = expect_positive_int(json_field(summary, "total", f"{context}.summary"), f"{context}.summary.total")
    passed = expect_positive_int(json_field(summary, "passed", f"{context}.summary"), f"{context}.summary.passed")
    failed = expect_nonnegative_int(json_field(summary, "failed", f"{context}.summary"), f"{context}.summary.failed")
    if passed != total or failed != 0:
        raise ValueError(f"{context}.summary must report all rows passed")


def validate_capsule_claims(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if json_field(value, "kind", context) != PIPELINE_CLAIMS_KIND:
        raise ValueError(f"{context}.kind must be `{PIPELINE_CLAIMS_KIND}`")
    if not expect_bool(json_field(value, "passed", context), f"{context}.passed"):
        raise ValueError(f"{context}.passed must be true")
    checks = json_field(value, "checks", context)
    if not isinstance(checks, list) or len(checks) != len(PIPELINE_CLAIMS):
        raise ValueError(f"{context}.checks must contain every pipeline claim")
    seen: set[str] = set()
    for index, check in enumerate(checks):
        check_context = f"{context}.checks[{index}]"
        if not isinstance(check, dict):
            raise ValueError(f"{check_context} must be an object")
        claim = validate_nonempty_string(json_field(check, "claim", check_context), f"{check_context}.claim")
        if claim in seen:
            raise ValueError(f"{context}.checks has duplicate claim `{claim}`")
        seen.add(claim)
        if not expect_bool(json_field(check, "passed", check_context), f"{check_context}.passed"):
            raise ValueError(f"{check_context}.passed must be true")
    missing = [claim for claim in PIPELINE_CLAIMS if claim not in seen]
    if missing:
        raise ValueError(f"{context}.checks missing required claim(s): {', '.join(missing)}")
    extra = [claim for claim in seen if claim not in PIPELINE_CLAIMS]
    if extra:
        raise ValueError(f"{context}.checks has unexpected claim(s): {', '.join(sorted(extra))}")


def expected_inference_action_contract(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    q31 = json_field(metrics, "q31_reference_inference", "metrics")
    imported = json_field(metrics, "imported_model_inference", "metrics")
    native = json_field(metrics, "native_inference_replay", "metrics")
    standalone = json_field(metrics, "native_standalone_rev_classifier", "metrics")
    trace = json_field(metrics, "reversible_inference_trace", "metrics")
    return [
        {
            "operation": "reproduce_prediction",
            "supported": True,
            "evidence": "q31_reference_matches_native_inference",
            "result": {
                "prediction": q31["prediction"],
                "correct": q31["correct"],
                "margin": q31["margin"],
            },
        },
        {
            "operation": "explain_margin",
            "supported": True,
            "evidence": "native_inference_explanation_contract",
            "ledgers": {
                "contribution": q31["contribution_ledger_fingerprint"],
                "margin_contribution": q31["margin_contribution_ledger_fingerprint"],
            },
        },
        {
            "operation": "replay_imported_model_inference",
            "supported": True,
            "evidence": "imported_model_inference_replay",
            "payload_bytes": imported["replay_payload_bytes"],
            "prediction": imported["prediction"],
            "matches_native": (
                imported["prediction_matches_native"]
                and imported["correct_matches_native"]
                and imported["margin_matches_native"]
            ),
        },
        {
            "operation": "replay_native_inference",
            "supported": True,
            "evidence": "native_inference_replay",
            "payload_bytes": native["replay_payload_bytes"],
        },
        {
            "operation": "run_standalone_rev_classifier",
            "supported": True,
            "evidence": "native_standalone_rev_classifier",
            "source_bytes": standalone["bytes"],
            "source_sha256": standalone["sha256"],
            "prediction": standalone["run_prediction"],
            "correct": bool(standalone["run_correct"]),
            "roundtrip_passed": standalone["roundtrip_passed"],
            "verification_passed": standalone["verification_passed"],
        },
        {
            "operation": "reverse_reversible_trace",
            "supported": True,
            "evidence": "reversible_inference_trace_replay",
            "payload_bytes": trace["replay_payload_bytes"],
            "witness_payload_bytes": trace["witness_payload_bytes"],
            "trace_payload_bytes": trace["trace_payload_bytes"],
            "recompute_steps": trace["total_recompute_steps"],
        },
    ]


def validate_inference_action_contract(value: Any, context: str) -> None:
    if not isinstance(value, list) or len(value) != len(INFERENCE_ACTION_OPERATIONS):
        raise ValueError(f"{context} must list every deterministic inference action")
    seen: set[str] = set()
    for index, raw_action in enumerate(value):
        action_context = f"{context}[{index}]"
        if not isinstance(raw_action, dict):
            raise ValueError(f"{action_context} must be an object")
        operation = json_field(raw_action, "operation", action_context)
        if operation in seen:
            raise ValueError(f"{context} has duplicate operation `{operation}`")
        seen.add(operation)
        if operation == "reproduce_prediction":
            action = validate_exact_key_object(
                raw_action,
                ("operation", "supported", "evidence", "result"),
                action_context,
            )
            if json_field(action, "evidence", action_context) != "q31_reference_matches_native_inference":
                raise ValueError(f"{action_context}.evidence must identify the Q31/native parity gate")
            result = validate_exact_key_object(
                json_field(action, "result", action_context),
                ("prediction", "correct", "margin"),
                f"{action_context}.result",
            )
            expect_digit(json_field(result, "prediction", f"{action_context}.result"), f"{action_context}.result.prediction")
            expect_bool(json_field(result, "correct", f"{action_context}.result"), f"{action_context}.result.correct")
            expect_i64(json_field(result, "margin", f"{action_context}.result"), f"{action_context}.result.margin")
        elif operation == "explain_margin":
            action = validate_exact_key_object(
                raw_action,
                ("operation", "supported", "evidence", "ledgers"),
                action_context,
            )
            if json_field(action, "evidence", action_context) != "native_inference_explanation_contract":
                raise ValueError(f"{action_context}.evidence must identify the native explanation gate")
            ledgers = validate_exact_key_object(
                json_field(action, "ledgers", action_context),
                ("contribution", "margin_contribution"),
                f"{action_context}.ledgers",
            )
            validate_sha256(json_field(ledgers, "contribution", f"{action_context}.ledgers"), f"{action_context}.ledgers.contribution")
            validate_sha256(
                json_field(ledgers, "margin_contribution", f"{action_context}.ledgers"),
                f"{action_context}.ledgers.margin_contribution",
            )
        elif operation == "replay_native_inference":
            action = validate_exact_key_object(
                raw_action,
                ("operation", "supported", "evidence", "payload_bytes"),
                action_context,
            )
            if json_field(action, "evidence", action_context) != "native_inference_replay":
                raise ValueError(f"{action_context}.evidence must identify the native replay gate")
            expect_positive_int(json_field(action, "payload_bytes", action_context), f"{action_context}.payload_bytes")
        elif operation == "replay_imported_model_inference":
            action = validate_exact_key_object(
                raw_action,
                (
                    "operation",
                    "supported",
                    "evidence",
                    "payload_bytes",
                    "prediction",
                    "matches_native",
                ),
                action_context,
            )
            if json_field(action, "evidence", action_context) != "imported_model_inference_replay":
                raise ValueError(f"{action_context}.evidence must identify the imported model replay gate")
            expect_positive_int(json_field(action, "payload_bytes", action_context), f"{action_context}.payload_bytes")
            expect_digit(json_field(action, "prediction", action_context), f"{action_context}.prediction")
            if not expect_bool(json_field(action, "matches_native", action_context), f"{action_context}.matches_native"):
                raise ValueError(f"{action_context}.matches_native must be true")
        elif operation == "run_standalone_rev_classifier":
            action = validate_exact_key_object(
                raw_action,
                (
                    "operation",
                    "supported",
                    "evidence",
                    "source_bytes",
                    "source_sha256",
                    "prediction",
                    "correct",
                    "roundtrip_passed",
                    "verification_passed",
                ),
                action_context,
            )
            if json_field(action, "evidence", action_context) != "native_standalone_rev_classifier":
                raise ValueError(f"{action_context}.evidence must identify the standalone Reverie classifier gate")
            expect_positive_int(json_field(action, "source_bytes", action_context), f"{action_context}.source_bytes")
            validate_sha256(json_field(action, "source_sha256", action_context), f"{action_context}.source_sha256")
            expect_digit(json_field(action, "prediction", action_context), f"{action_context}.prediction")
            expect_bool(json_field(action, "correct", action_context), f"{action_context}.correct")
            if not expect_bool(json_field(action, "roundtrip_passed", action_context), f"{action_context}.roundtrip_passed"):
                raise ValueError(f"{action_context}.roundtrip_passed must be true")
            if not expect_bool(json_field(action, "verification_passed", action_context), f"{action_context}.verification_passed"):
                raise ValueError(f"{action_context}.verification_passed must be true")
        elif operation == "reverse_reversible_trace":
            action = validate_exact_key_object(
                raw_action,
                (
                    "operation",
                    "supported",
                    "evidence",
                    "payload_bytes",
                    "witness_payload_bytes",
                    "trace_payload_bytes",
                    "recompute_steps",
                ),
                action_context,
            )
            if json_field(action, "evidence", action_context) != "reversible_inference_trace_replay":
                raise ValueError(f"{action_context}.evidence must identify the reversible trace gate")
            expect_positive_int(json_field(action, "payload_bytes", action_context), f"{action_context}.payload_bytes")
            expect_positive_int(
                json_field(action, "witness_payload_bytes", action_context),
                f"{action_context}.witness_payload_bytes",
            )
            trace_bytes = expect_nonnegative_int(
                json_field(action, "trace_payload_bytes", action_context),
                f"{action_context}.trace_payload_bytes",
            )
            if trace_bytes != 0:
                raise ValueError(f"{action_context}.trace_payload_bytes must be zero for the v1 inference trace")
            expect_positive_int(json_field(action, "recompute_steps", action_context), f"{action_context}.recompute_steps")
        else:
            raise ValueError(f"{action_context}.operation is unsupported")
        if not expect_bool(json_field(raw_action, "supported", action_context), f"{action_context}.supported"):
            raise ValueError(f"{action_context}.supported must be true")
    if tuple(action["operation"] for action in value) != INFERENCE_ACTION_OPERATIONS:
        raise ValueError(f"{context} must preserve the deterministic inference action order")


def validate_model_capsule_report(report: dict[str, Any]) -> None:
    capsule = validate_exact_key_object(
        report,
        ("kind", "algorithm", "fingerprint", "payload"),
        "model capsule",
    )
    if json_field(capsule, "kind", "model capsule") != MODEL_CAPSULE_KIND:
        raise ValueError(f"model capsule kind must be `{MODEL_CAPSULE_KIND}`")
    if json_field(capsule, "algorithm", "model capsule") != "sha256":
        raise ValueError("model capsule algorithm must be sha256")
    fingerprint = validate_sha256(json_field(capsule, "fingerprint", "model capsule"), "model capsule.fingerprint")
    payload = json_field(capsule, "payload", "model capsule")
    if not isinstance(payload, dict):
        raise ValueError("model capsule.payload must be an object")
    if sha256_json(payload) != fingerprint:
        raise ValueError("model capsule fingerprint must match canonical payload JSON")
    if json_field(payload, "schema", "model capsule.payload") != MODEL_CAPSULE_SCHEMA:
        raise ValueError(f"model capsule schema must be `{MODEL_CAPSULE_SCHEMA}`")
    if json_field(payload, "pipeline_kind", "model capsule.payload") != PIPELINE_KIND:
        raise ValueError(f"model capsule pipeline_kind must be `{PIPELINE_KIND}`")

    sample_limit = expect_positive_int(
        json_field(payload, "sample_limit", "model capsule.payload"),
        "model capsule.payload.sample_limit",
    )
    audit_step = expect_nonnegative_int(
        json_field(payload, "audit_step", "model capsule.payload"),
        "model capsule.payload.audit_step",
    )
    if audit_step >= sample_limit:
        raise ValueError("model capsule audit_step must be less than sample_limit")
    evaluation_row = expect_nonnegative_int(
        json_field(payload, "evaluation_row", "model capsule.payload"),
        "model capsule.payload.evaluation_row",
    )
    if evaluation_row >= sample_limit:
        raise ValueError("model capsule evaluation_row must be less than sample_limit")
    if json_field(payload, "audit_step_strategy", "model capsule.payload") not in AUDIT_STEP_STRATEGIES:
        raise ValueError("model capsule audit_step_strategy is invalid")
    if json_field(payload, "evaluation_row_strategy", "model capsule.payload") not in EVALUATION_ROW_STRATEGIES:
        raise ValueError("model capsule evaluation_row_strategy is invalid")

    profile_markdown = validate_nonempty_string(
        json_field(payload, "profile_markdown", "model capsule.payload"),
        "model capsule.payload.profile_markdown",
    )
    pipeline_summary = validate_nonempty_string(
        json_field(payload, "pipeline_summary", "model capsule.payload"),
        "model capsule.payload.pipeline_summary",
    )
    reports = validate_string_map_keys(
        json_field(payload, "reports", "model capsule.payload"),
        PIPELINE_REPORT_KEYS,
        "model capsule.payload.reports",
    )
    bundles = validate_string_map_keys(
        json_field(payload, "bundles", "model capsule.payload"),
        PIPELINE_BUNDLE_KEYS,
        "model capsule.payload.bundles",
    )
    evidence = json_field(payload, "evidence", "model capsule.payload")
    expected_evidence_paths = {
        "profile_markdown": profile_markdown,
        "summary": pipeline_summary,
        **{f"report.{key}": reports[key] for key in PIPELINE_REPORT_KEYS},
        **{f"bundle.{key}": bundles[key] for key in PIPELINE_BUNDLE_KEYS},
        **pipeline_standalone_evidence_paths(evidence, "model capsule.payload.evidence"),
    }
    validate_pipeline_evidence_index(evidence, expected_evidence_paths, "model capsule.payload.evidence")
    evidence_files = json_field(evidence, "files", "model capsule.payload.evidence")

    gates = validate_exact_key_object(
        json_field(payload, "gates", "model capsule.payload"),
        ("passed", "total", "passed_count", "failed_metrics", "policy"),
        "model capsule.payload.gates",
    )
    if not expect_bool(json_field(gates, "passed", "model capsule.payload.gates"), "model capsule.payload.gates.passed"):
        raise ValueError("model capsule gates must pass")
    total = expect_positive_int(json_field(gates, "total", "model capsule.payload.gates"), "model capsule.payload.gates.total")
    passed_count = expect_positive_int(
        json_field(gates, "passed_count", "model capsule.payload.gates"),
        "model capsule.payload.gates.passed_count",
    )
    if total < len(PIPELINE_GATE_METRICS):
        raise ValueError("model capsule gates.total must cover the pipeline gate catalog")
    if passed_count != total:
        raise ValueError("model capsule gates.passed_count must equal total")
    failed_metrics = validate_string_list(
        json_field(gates, "failed_metrics", "model capsule.payload.gates"),
        "model capsule.payload.gates.failed_metrics",
        allow_empty=True,
    )
    if failed_metrics:
        raise ValueError("model capsule gates.failed_metrics must be empty")
    validate_pipeline_gate_policy(json_field(gates, "policy", "model capsule.payload.gates"), "model capsule.payload.gates.policy")

    model = validate_exact_key_object(
        json_field(payload, "model", "model capsule.payload"),
        ("bundle", "sha256", "payload_bytes"),
        "model capsule.payload.model",
    )
    if json_field(model, "bundle", "model capsule.payload.model") != bundles["model"]:
        raise ValueError("model capsule model.bundle must match bundles.model")
    if (
        validate_sha256(json_field(model, "sha256", "model capsule.payload.model"), "model capsule.payload.model.sha256")
        != evidence_files["bundle.model"]["sha256"]
    ):
        raise ValueError("model capsule model.sha256 must match evidence")
    expect_positive_int(json_field(model, "payload_bytes", "model capsule.payload.model"), "model capsule.payload.model.payload_bytes")

    imported_model = validate_exact_key_object(
        json_field(payload, "imported_model", "model capsule.payload"),
        (
            "source_json",
            "source_json_sha256",
            "bundle",
            "bundle_sha256",
            "source_model_json_checked",
            "provenance_kind",
            "source_model_json_fingerprint",
            "inference_replay_passed",
            "prediction_matches_native",
            "margin_matches_native",
        ),
        "model capsule.payload.imported_model",
    )
    if json_field(imported_model, "source_json", "model capsule.payload.imported_model") != bundles["imported_model_source"]:
        raise ValueError("model capsule imported_model.source_json must match bundles.imported_model_source")
    if json_field(imported_model, "bundle", "model capsule.payload.imported_model") != bundles["imported_model"]:
        raise ValueError("model capsule imported_model.bundle must match bundles.imported_model")
    if (
        validate_sha256(
            json_field(imported_model, "source_json_sha256", "model capsule.payload.imported_model"),
            "model capsule.payload.imported_model.source_json_sha256",
        )
        != evidence_files["bundle.imported_model_source"]["sha256"]
    ):
        raise ValueError("model capsule imported_model.source_json_sha256 must match evidence")
    if (
        validate_sha256(
            json_field(imported_model, "bundle_sha256", "model capsule.payload.imported_model"),
            "model capsule.payload.imported_model.bundle_sha256",
        )
        != evidence_files["bundle.imported_model"]["sha256"]
    ):
        raise ValueError("model capsule imported_model.bundle_sha256 must match evidence")
    if json_field(imported_model, "provenance_kind", "model capsule.payload.imported_model") != "external_import":
        raise ValueError("model capsule imported_model.provenance_kind must be external_import")
    validate_sha256(
        json_field(imported_model, "source_model_json_fingerprint", "model capsule.payload.imported_model"),
        "model capsule.payload.imported_model.source_model_json_fingerprint",
    )
    for field in (
        "source_model_json_checked",
        "inference_replay_passed",
        "prediction_matches_native",
        "margin_matches_native",
    ):
        if not expect_bool(
            json_field(imported_model, field, "model capsule.payload.imported_model"),
            f"model capsule.payload.imported_model.{field}",
        ):
            raise ValueError(f"model capsule imported_model.{field} must be true")

    samples = validate_exact_key_object(
        json_field(payload, "samples", "model capsule.payload"),
        ("bundle", "sha256", "count", "accuracy_percent"),
        "model capsule.payload.samples",
    )
    if json_field(samples, "bundle", "model capsule.payload.samples") != bundles["samples"]:
        raise ValueError("model capsule samples.bundle must match bundles.samples")
    if (
        validate_sha256(json_field(samples, "sha256", "model capsule.payload.samples"), "model capsule.payload.samples.sha256")
        != evidence_files["bundle.samples"]["sha256"]
    ):
        raise ValueError("model capsule samples.sha256 must match evidence")
    sample_count = expect_positive_int(json_field(samples, "count", "model capsule.payload.samples"), "model capsule.payload.samples.count")
    if sample_count != sample_limit:
        raise ValueError("model capsule samples.count must match sample_limit")
    accuracy = expect_finite_number(
        json_field(samples, "accuracy_percent", "model capsule.payload.samples"),
        "model capsule.payload.samples.accuracy_percent",
    )
    if not 0.0 <= accuracy <= 100.0:
        raise ValueError("model capsule samples.accuracy_percent must be in 0..100")

    lineage = validate_exact_key_object(
        json_field(payload, "training_lineage", "model capsule.payload"),
        (
            "checked_steps",
            "restored_initial_model",
            "final_model_replayed",
            "lineage_ledger_matches",
            "lineage_ledger_fingerprint",
            "transition_ledger_fingerprint",
            "final_chain",
        ),
        "model capsule.payload.training_lineage",
    )
    if expect_positive_int(json_field(lineage, "checked_steps", "model capsule.payload.training_lineage"), "model capsule.payload.training_lineage.checked_steps") < sample_limit:
        raise ValueError("model capsule training_lineage.checked_steps must cover sample_limit")
    for field in ("restored_initial_model", "final_model_replayed", "lineage_ledger_matches"):
        if not expect_bool(json_field(lineage, field, "model capsule.payload.training_lineage"), f"model capsule.payload.training_lineage.{field}"):
            raise ValueError(f"model capsule training_lineage.{field} must be true")
    for field in ("lineage_ledger_fingerprint", "transition_ledger_fingerprint", "final_chain"):
        validate_sha256(json_field(lineage, field, "model capsule.payload.training_lineage"), f"model capsule.payload.training_lineage.{field}")

    inference = validate_exact_key_object(
        json_field(payload, "inference", "model capsule.payload"),
        (
            "imported_model_replay_passed",
            "imported_model_prediction",
            "imported_model_matches_native",
            "native_replay_passed",
            "native_prediction",
            "native_correct",
            "native_margin",
            "evaluation_row_replay_passed",
            "evaluation_row_prediction",
            "evaluation_row_correct",
            "evaluation_row_margin",
            "q31_reference_prediction",
            "q31_reference_matches_native",
            "action_contract",
        ),
        "model capsule.payload.inference",
    )
    for field in (
        "imported_model_replay_passed",
        "imported_model_matches_native",
        "native_replay_passed",
        "evaluation_row_replay_passed",
        "q31_reference_matches_native",
    ):
        if not expect_bool(json_field(inference, field, "model capsule.payload.inference"), f"model capsule.payload.inference.{field}"):
            raise ValueError(f"model capsule inference.{field} must be true")
    for field in (
        "imported_model_prediction",
        "native_prediction",
        "evaluation_row_prediction",
        "q31_reference_prediction",
    ):
        expect_digit(json_field(inference, field, "model capsule.payload.inference"), f"model capsule.payload.inference.{field}")
    for field in ("native_correct", "evaluation_row_correct"):
        expect_bool(json_field(inference, field, "model capsule.payload.inference"), f"model capsule.payload.inference.{field}")
    for field in ("native_margin", "evaluation_row_margin"):
        expect_i64(json_field(inference, field, "model capsule.payload.inference"), f"model capsule.payload.inference.{field}")
    validate_inference_action_contract(
        json_field(inference, "action_contract", "model capsule.payload.inference"),
        "model capsule.payload.inference.action_contract",
    )

    reversible_trace = validate_exact_key_object(
        json_field(payload, "reversible_inference_trace", "model capsule.payload"),
        (
            "passed",
            "logits",
            "top_classes",
            "top_logit_values",
            "top_logits",
            "prediction",
            "runner_up_class",
            "margin",
            "label_rank",
            "correct",
            "top2_correct",
            "contribution_ledger_fingerprint",
            "margin_contribution_ledger_fingerprint",
            "witness_payload_bytes",
            "trace_payload_bytes",
            "replay_payload_bytes",
            "total_recompute_steps",
        ),
        "model capsule.payload.reversible_inference_trace",
    )
    if not expect_bool(
        json_field(
            reversible_trace,
            "passed",
            "model capsule.payload.reversible_inference_trace",
        ),
        "model capsule.payload.reversible_inference_trace.passed",
    ):
        raise ValueError("model capsule reversible_inference_trace.passed must be true")
    trace_logits = validate_inference_trace_vector(
        json_field(
            reversible_trace,
            "logits",
            "model capsule.payload.reversible_inference_trace",
        ),
        inference_trace_check.CLASSES,
        "model capsule.payload.reversible_inference_trace.logits",
    )
    trace_top_classes = validate_inference_trace_vector(
        json_field(
            reversible_trace,
            "top_classes",
            "model capsule.payload.reversible_inference_trace",
        ),
        inference_trace_check.CLASSES,
        "model capsule.payload.reversible_inference_trace.top_classes",
    )
    trace_top_logit_values = validate_inference_trace_vector(
        json_field(
            reversible_trace,
            "top_logit_values",
            "model capsule.payload.reversible_inference_trace",
        ),
        inference_trace_check.CLASSES,
        "model capsule.payload.reversible_inference_trace.top_logit_values",
    )
    expected_forward = inference_trace_check.forward_store(
        inference_trace_check.initial_store()
    )
    if (
        trace_logits != expected_forward["logits"]
        or trace_top_classes != expected_forward["top_classes"]
        or trace_top_logit_values != expected_forward["top_logit_values"]
    ):
        raise ValueError("model capsule reversible inference trace logits/top-k fields must match reference")
    validate_inference_trace_top_logits(
        json_field(
            reversible_trace,
            "top_logits",
            "model capsule.payload.reversible_inference_trace",
        ),
        trace_logits,
        "model capsule.payload.reversible_inference_trace.top_logits",
    )
    for field in (
        "prediction",
        "runner_up_class",
        "margin",
        "label_rank",
        "correct",
        "top2_correct",
    ):
        actual = expect_i64(
            json_field(
                reversible_trace,
                field,
                "model capsule.payload.reversible_inference_trace",
            ),
            f"model capsule.payload.reversible_inference_trace.{field}",
        )
        if actual != expected_forward[field]:
            raise ValueError(
                f"model capsule reversible_inference_trace.{field} must match reference"
            )
    validate_sha256(
        json_field(
            reversible_trace,
            "contribution_ledger_fingerprint",
            "model capsule.payload.reversible_inference_trace",
        ),
        "model capsule.payload.reversible_inference_trace.contribution_ledger_fingerprint",
    )
    validate_sha256(
        json_field(
            reversible_trace,
            "margin_contribution_ledger_fingerprint",
            "model capsule.payload.reversible_inference_trace",
        ),
        "model capsule.payload.reversible_inference_trace.margin_contribution_ledger_fingerprint",
    )
    expect_positive_int(
        json_field(
            reversible_trace,
            "witness_payload_bytes",
            "model capsule.payload.reversible_inference_trace",
        ),
        "model capsule.payload.reversible_inference_trace.witness_payload_bytes",
    )
    expect_nonnegative_int(
        json_field(
            reversible_trace,
            "trace_payload_bytes",
            "model capsule.payload.reversible_inference_trace",
        ),
        "model capsule.payload.reversible_inference_trace.trace_payload_bytes",
    )
    expect_positive_int(
        json_field(
            reversible_trace,
            "replay_payload_bytes",
            "model capsule.payload.reversible_inference_trace",
        ),
        "model capsule.payload.reversible_inference_trace.replay_payload_bytes",
    )
    expect_positive_int(
        json_field(
            reversible_trace,
            "total_recompute_steps",
            "model capsule.payload.reversible_inference_trace",
        ),
        "model capsule.payload.reversible_inference_trace.total_recompute_steps",
    )

    witnesses = validate_exact_key_object(
        json_field(payload, "witnesses", "model capsule.payload"),
        (
            "mlp_passed",
            "mlp_samples",
            "mlp_dataset_loops",
            "mlp_witness_proof_fingerprint",
            "witness_payload_bytes",
            "witness_to_model_payload_ratio",
        ),
        "model capsule.payload.witnesses",
    )
    if not expect_bool(json_field(witnesses, "mlp_passed", "model capsule.payload.witnesses"), "model capsule.payload.witnesses.mlp_passed"):
        raise ValueError("model capsule witnesses.mlp_passed must be true")
    expect_positive_int(json_field(witnesses, "mlp_samples", "model capsule.payload.witnesses"), "model capsule.payload.witnesses.mlp_samples")
    validate_mlp_dataset_loops(
        json_field(witnesses, "mlp_dataset_loops", "model capsule.payload.witnesses"),
        "model capsule.payload.witnesses.mlp_dataset_loops",
    )
    validate_sha256(
        json_field(witnesses, "mlp_witness_proof_fingerprint", "model capsule.payload.witnesses"),
        "model capsule.payload.witnesses.mlp_witness_proof_fingerprint",
    )
    expect_positive_int(
        json_field(witnesses, "witness_payload_bytes", "model capsule.payload.witnesses"),
        "model capsule.payload.witnesses.witness_payload_bytes",
    )
    if expect_finite_number(
        json_field(witnesses, "witness_to_model_payload_ratio", "model capsule.payload.witnesses"),
        "model capsule.payload.witnesses.witness_to_model_payload_ratio",
    ) < 0.0:
        raise ValueError("model capsule witnesses.witness_to_model_payload_ratio must be non-negative")

    invertible = validate_exact_key_object(
        json_field(payload, "invertible_patterns", "model capsule.payload"),
        (
            "coupling_passed",
            "triangular_residual_passed",
            "reversible_preprocess_passed",
            "reversible_inference_trace_passed",
        ),
        "model capsule.payload.invertible_patterns",
    )
    for field in invertible:
        if not expect_bool(json_field(invertible, field, "model capsule.payload.invertible_patterns"), f"model capsule.payload.invertible_patterns.{field}"):
            raise ValueError(f"model capsule invertible_patterns.{field} must be true")

    scorecard = validate_exact_key_object(
        json_field(payload, "scorecard", "model capsule.payload"),
        (
            "train_updates_per_second",
            "model_evaluation_samples_per_second",
            "reverse_steps_per_second",
            "reverse_to_train_elapsed_ratio",
            "run_peak_rss_bytes",
            "max_replay_payload_bytes",
            "trace_to_model_payload_ratio",
            "witness_to_model_payload_ratio",
            "total_recompute_steps",
            "balanced_recompute",
        ),
        "model capsule.payload.scorecard",
    )
    for field in ("train_updates_per_second", "model_evaluation_samples_per_second", "reverse_steps_per_second"):
        if expect_finite_number(json_field(scorecard, field, "model capsule.payload.scorecard"), f"model capsule.payload.scorecard.{field}") <= 0.0:
            raise ValueError(f"model capsule scorecard.{field} must be positive")
    for field in ("reverse_to_train_elapsed_ratio", "trace_to_model_payload_ratio", "witness_to_model_payload_ratio"):
        if expect_finite_number(json_field(scorecard, field, "model capsule.payload.scorecard"), f"model capsule.payload.scorecard.{field}") < 0.0:
            raise ValueError(f"model capsule scorecard.{field} must be non-negative")
    expect_positive_int(json_field(scorecard, "run_peak_rss_bytes", "model capsule.payload.scorecard"), "model capsule.payload.scorecard.run_peak_rss_bytes")
    expect_positive_int(json_field(scorecard, "max_replay_payload_bytes", "model capsule.payload.scorecard"), "model capsule.payload.scorecard.max_replay_payload_bytes")
    expect_positive_int(json_field(scorecard, "total_recompute_steps", "model capsule.payload.scorecard"), "model capsule.payload.scorecard.total_recompute_steps")
    if not expect_bool(json_field(scorecard, "balanced_recompute", "model capsule.payload.scorecard"), "model capsule.payload.scorecard.balanced_recompute"):
        raise ValueError("model capsule scorecard.balanced_recompute must be true")

    validate_capsule_passed_catalog(
        json_field(payload, "capabilities", "model capsule.payload"),
        ML_CAPABILITY_KIND,
        "model capsule.payload.capabilities",
    )
    validate_capsule_passed_catalog(
        json_field(payload, "readiness", "model capsule.payload"),
        ML_GOAL_READINESS_KIND,
        "model capsule.payload.readiness",
    )
    readiness = json_field(payload, "readiness", "model capsule.payload")
    if (
        json_field(readiness, "north_star", "model capsule.payload.readiness")
        != "best_small_language_for_reversible_inspectable_deterministic_ml_kernels"
    ):
        raise ValueError("model capsule readiness north_star is not the reversible ML north star")
    if (
        json_field(readiness, "non_goal", "model capsule.payload.readiness")
        != "general_purpose_pytorch_tensorflow_training_replacement"
    ):
        raise ValueError("model capsule readiness non_goal must preserve the PyTorch/TensorFlow non-goal")
    validate_capsule_claims(json_field(payload, "claims", "model capsule.payload"), "model capsule.payload.claims")


def artifact_reference_matches(reference: str, path: Path) -> bool:
    if reference == str(path):
        return True
    try:
        return Path(reference).resolve() == path.resolve()
    except OSError:
        return False


def find_referenced_report(
    reference: str,
    candidates: list[tuple[Path, dict[str, Any]]],
    context: str,
) -> Optional[tuple[Path, dict[str, Any]]]:
    matches = [
        item
        for item in candidates
        if artifact_reference_matches(reference, item[0])
    ]
    if len(matches) > 1:
        raise ValueError(f"{context} matches multiple input paths")
    if matches:
        return matches[0]
    if len(candidates) == 1:
        return candidates[0]
    return None


def validate_model_capsule_matches_summary(
    capsule_report: dict[str, Any],
    summary_report: dict[str, Any],
    context: str,
) -> None:
    payload = json_field(capsule_report, "payload", context)
    metrics = json_field(summary_report, "metrics", f"{context}.summary")
    for field in (
        "sample_limit",
        "audit_step",
        "audit_step_strategy",
        "evaluation_row",
        "evaluation_row_strategy",
        "profile_markdown",
        "reports",
        "bundles",
    ):
        if json_field(payload, field, f"{context}.payload") != json_field(summary_report, field, f"{context}.summary"):
            raise ValueError(f"{context}.payload.{field} must match pipeline summary")

    summary_checks = json_field(json_field(summary_report, "gates", f"{context}.summary"), "checks", f"{context}.summary.gates")
    failed_metrics = [
        check["metric"]
        for check in summary_checks
        if not check["passed"]
    ]
    expected_gates = {
        "passed": summary_report["gates"]["passed"],
        "total": len(summary_checks),
        "passed_count": len(summary_checks) - len(failed_metrics),
        "failed_metrics": failed_metrics,
        "policy": summary_report["gates"]["policy"],
    }
    if json_field(payload, "gates", f"{context}.payload") != expected_gates:
        raise ValueError(f"{context}.payload.gates must summarize pipeline summary gates")

    artifact_profile = json_field(metrics, "artifact_profile", f"{context}.summary.metrics")
    model = json_field(payload, "model", f"{context}.payload")
    if json_field(model, "payload_bytes", f"{context}.payload.model") != json_field(
        artifact_profile,
        "total_model_payload_bytes",
        f"{context}.summary.metrics.artifact_profile",
    ):
        raise ValueError(f"{context}.payload.model.payload_bytes must match summary artifact profile")
    samples = json_field(payload, "samples", f"{context}.payload")
    model_evaluation = json_field(metrics, "model_evaluation", f"{context}.summary.metrics")
    if json_field(samples, "count", f"{context}.payload.samples") != json_field(model_evaluation, "samples", f"{context}.summary.metrics.model_evaluation"):
        raise ValueError(f"{context}.payload.samples.count must match summary model evaluation")
    if json_field(samples, "accuracy_percent", f"{context}.payload.samples") != json_field(
        model_evaluation,
        "accuracy_percent",
        f"{context}.summary.metrics.model_evaluation",
    ):
        raise ValueError(f"{context}.payload.samples.accuracy_percent must match summary model evaluation")
    model_import = json_field(metrics, "model_import", f"{context}.summary.metrics")
    imported_inference = json_field(
        metrics,
        "imported_model_inference",
        f"{context}.summary.metrics",
    )
    expected_imported_model = {
        "source_json": summary_report["bundles"]["imported_model_source"],
        "source_json_sha256": payload["evidence"]["files"]["bundle.imported_model_source"][
            "sha256"
        ],
        "bundle": summary_report["bundles"]["imported_model"],
        "bundle_sha256": payload["evidence"]["files"]["bundle.imported_model"]["sha256"],
        "source_model_json_checked": model_import["verification_source_checked"],
        "provenance_kind": model_import["verification_provenance_kind"],
        "source_model_json_fingerprint": model_import["source_model_json_fingerprint"],
        "inference_replay_passed": (
            imported_inference["proof_matches"]
            and imported_inference["result_matches"]
        ),
        "prediction_matches_native": imported_inference["prediction_matches_native"],
        "margin_matches_native": imported_inference["margin_matches_native"],
    }
    if json_field(payload, "imported_model", f"{context}.payload") != expected_imported_model:
        raise ValueError(f"{context}.payload.imported_model must match summary imported model metrics")

    replay = json_field(metrics, "training_audit_replay", f"{context}.summary.metrics")
    expected_lineage = {
        "checked_steps": replay["checked"],
        "restored_initial_model": replay["restored_initial_model"],
        "final_model_replayed": replay["final_model_replayed"],
        "lineage_ledger_matches": replay["lineage_ledger_matches"],
        "lineage_ledger_fingerprint": replay["lineage_ledger_fingerprint"],
        "transition_ledger_fingerprint": replay["transition_ledger_fingerprint"],
        "final_chain": replay["final_chain"],
    }
    if json_field(payload, "training_lineage", f"{context}.payload") != expected_lineage:
        raise ValueError(f"{context}.payload.training_lineage must match summary replay lineage")

    q31 = json_field(metrics, "q31_reference_inference", f"{context}.summary.metrics")
    native = json_field(metrics, "native_inference_replay", f"{context}.summary.metrics")
    native_explanation = json_field(metrics, "native_inference_explanation", f"{context}.summary.metrics")
    row = json_field(metrics, "evaluation_row_inference", f"{context}.summary.metrics")
    row_explanation = json_field(metrics, "evaluation_row_inference_explanation", f"{context}.summary.metrics")
    expected_inference = {
        "imported_model_replay_passed": (
            imported_inference["proof_matches"]
            and imported_inference["result_matches"]
        ),
        "imported_model_prediction": imported_inference["prediction"],
        "imported_model_matches_native": (
            imported_inference["prediction_matches_native"]
            and imported_inference["correct_matches_native"]
            and imported_inference["margin_matches_native"]
            and imported_inference["contribution_ledger_matches_native"]
            and imported_inference["margin_contribution_ledger_matches_native"]
        ),
        "native_replay_passed": native["proof_matches"] and native["result_matches"],
        "native_prediction": native_explanation["prediction"],
        "native_correct": native_explanation["correct"],
        "native_margin": native_explanation["margin"],
        "evaluation_row_replay_passed": row["proof_matches"] and row["result_matches"],
        "evaluation_row_prediction": row_explanation["prediction"],
        "evaluation_row_correct": row_explanation["correct"],
        "evaluation_row_margin": row_explanation["margin"],
        "q31_reference_prediction": q31["prediction"],
        "q31_reference_matches_native": (
            q31["prediction_matches_native"]
            and q31["correct_matches_native"]
            and q31["margin_matches_native"]
            and q31["contribution_ledger_matches_native"]
            and q31["margin_contribution_ledger_matches_native"]
        ),
        "action_contract": expected_inference_action_contract(metrics),
    }
    if json_field(payload, "inference", f"{context}.payload") != expected_inference:
        raise ValueError(f"{context}.payload.inference must match summary replay and Q31 reference metrics")

    trace = json_field(metrics, "reversible_inference_trace", f"{context}.summary.metrics")
    expected_reversible_trace = {
        "passed": trace["passed"],
        "logits": trace["forward_logits"],
        "top_classes": trace["top_classes"],
        "top_logit_values": trace["top_logit_values"],
        "top_logits": trace["top_logits"],
        "prediction": trace["prediction"],
        "runner_up_class": trace["runner_up_class"],
        "margin": trace["margin"],
        "label_rank": trace["label_rank"],
        "correct": trace["correct"],
        "top2_correct": trace["top2_correct"],
        "contribution_ledger_fingerprint": trace["attribution"][
            "contribution_ledger_fingerprint"
        ],
        "margin_contribution_ledger_fingerprint": trace["attribution"][
            "margin_contribution_ledger_fingerprint"
        ],
        "witness_payload_bytes": trace["witness_payload_bytes"],
        "trace_payload_bytes": trace["trace_payload_bytes"],
        "replay_payload_bytes": trace["replay_payload_bytes"],
        "total_recompute_steps": trace["total_recompute_steps"],
    }
    if (
        json_field(payload, "reversible_inference_trace", f"{context}.payload")
        != expected_reversible_trace
    ):
        raise ValueError(
            f"{context}.payload.reversible_inference_trace must match summary reversible trace metrics"
        )

    mlp = json_field(metrics, "mlp_witness", f"{context}.summary.metrics")
    expected_witnesses = {
        "mlp_passed": mlp["passed"],
        "mlp_samples": mlp["samples"],
        "mlp_dataset_loops": mlp["dataset_loops"],
        "mlp_witness_proof_fingerprint": mlp["witness_proof_fingerprint"],
        "witness_payload_bytes": artifact_profile["total_witness_payload_bytes"],
        "witness_to_model_payload_ratio": artifact_profile["witness_to_model_payload_ratio"],
    }
    if json_field(payload, "witnesses", f"{context}.payload") != expected_witnesses:
        raise ValueError(f"{context}.payload.witnesses must match summary witness metrics")

    expected_invertible = {
        "coupling_passed": metrics["invertible_coupling"]["passed"],
        "triangular_residual_passed": metrics["triangular_residual"]["passed"],
        "reversible_preprocess_passed": metrics["reversible_preprocess"]["passed"],
        "reversible_inference_trace_passed": metrics["reversible_inference_trace"]["passed"],
    }
    if json_field(payload, "invertible_patterns", f"{context}.payload") != expected_invertible:
        raise ValueError(f"{context}.payload.invertible_patterns must match summary metrics")

    scorecard = json_field(metrics, "scorecard", f"{context}.summary.metrics")
    expected_scorecard = {
        "train_updates_per_second": scorecard["train_updates_per_second"],
        "model_evaluation_samples_per_second": scorecard["model_evaluation_samples_per_second"],
        "reverse_steps_per_second": scorecard["reverse_steps_per_second"],
        "reverse_to_train_elapsed_ratio": scorecard["reverse_to_train_elapsed_ratio"],
        "run_peak_rss_bytes": scorecard["run_peak_rss_bytes"],
        "max_replay_payload_bytes": scorecard["max_replay_payload_bytes"],
        "trace_to_model_payload_ratio": scorecard["trace_to_model_payload_ratio"],
        "witness_to_model_payload_ratio": scorecard["witness_to_model_payload_ratio"],
        "total_recompute_steps": scorecard["total_recompute_steps"],
        "balanced_recompute": scorecard["balanced_recompute"],
    }
    if json_field(payload, "scorecard", f"{context}.payload") != expected_scorecard:
        raise ValueError(f"{context}.payload.scorecard must match summary scorecard")
    if json_field(payload, "capabilities", f"{context}.payload") != metrics["ml_capability_map"]:
        raise ValueError(f"{context}.payload.capabilities must match summary capability map")
    if json_field(payload, "readiness", f"{context}.payload") != metrics["ml_goal_readiness"]:
        raise ValueError(f"{context}.payload.readiness must match summary goal readiness")
    if json_field(payload, "claims", f"{context}.payload") != json_field(summary_report, "claims", f"{context}.summary"):
        raise ValueError(f"{context}.payload.claims must match summary claims")


def validate_manifest_matches_summary(
    manifest: dict[str, Any],
    summary: dict[str, Any],
    context: str,
) -> None:
    if json_field(manifest, "sample_limit", context) != json_field(summary, "sample_limit", f"{context}.summary"):
        raise ValueError(f"{context}.sample_limit must match summary")
    for field in (
        "audit_step",
        "audit_step_strategy",
        "evaluation_row",
        "evaluation_row_strategy",
        "profile_markdown",
    ):
        if json_field(manifest, field, context) != json_field(summary, field, f"{context}.summary"):
            raise ValueError(f"{context}.{field} must match summary")
    if json_field(manifest, "gate_policy", context) != json_field(summary["gates"], "policy", f"{context}.summary.gates"):
        raise ValueError(f"{context}.gate_policy must match summary gates policy")
    if json_field(manifest, "gates_passed", context) != json_field(summary["gates"], "passed", f"{context}.summary.gates"):
        raise ValueError(f"{context}.gates_passed must match summary gates")
    if json_field(manifest, "reports", context) != [summary["reports"][key] for key in PIPELINE_REPORT_KEYS]:
        raise ValueError(f"{context}.reports must match summary reports")
    if json_field(manifest, "bundles", context) != [summary["bundles"][key] for key in PIPELINE_BUNDLE_KEYS]:
        raise ValueError(f"{context}.bundles must match summary bundles")


def validate_report_set_consistency(reports: list[tuple[Path, dict[str, Any]]]) -> None:
    summaries = [(path, report) for path, report in reports if report.get("kind") == PIPELINE_SUMMARY_KIND]
    capsules = [(path, report) for path, report in reports if report.get("kind") == MODEL_CAPSULE_KIND]
    manifests = [(path, report) for path, report in reports if report.get("kind") == PIPELINE_KIND]
    for capsule_path, capsule in capsules:
        payload = json_field(capsule, "payload", f"{capsule_path}")
        summary_ref = validate_nonempty_string(
            json_field(payload, "pipeline_summary", f"{capsule_path}.payload"),
            f"{capsule_path}.payload.pipeline_summary",
        )
        summary_match = find_referenced_report(summary_ref, summaries, f"{capsule_path}.payload.pipeline_summary")
        if summary_match is not None:
            validate_model_capsule_matches_summary(
                capsule,
                summary_match[1],
                f"{capsule_path} vs {summary_match[0]}",
            )
        for manifest_path, manifest in manifests:
            manifest_capsule_ref = validate_nonempty_string(
                json_field(manifest, "model_capsule", f"{manifest_path}"),
                f"{manifest_path}.model_capsule",
            )
            if artifact_reference_matches(manifest_capsule_ref, capsule_path) or len(capsules) == 1:
                if json_field(manifest, "model_capsule_fingerprint", f"{manifest_path}") != json_field(
                    capsule,
                    "fingerprint",
                    f"{capsule_path}",
                ):
                    raise ValueError(f"{manifest_path}.model_capsule_fingerprint must match {capsule_path}")
                if summary_match is not None:
                    validate_manifest_matches_summary(
                        manifest,
                        summary_match[1],
                        f"{manifest_path} vs {summary_match[0]}",
                    )
    for manifest_path, manifest in manifests:
        summary_ref = validate_nonempty_string(
            json_field(manifest, "summary", f"{manifest_path}"),
            f"{manifest_path}.summary",
        )
        summary_match = find_referenced_report(summary_ref, summaries, f"{manifest_path}.summary")
        if summary_match is not None:
            validate_manifest_matches_summary(
                manifest,
                summary_match[1],
                f"{manifest_path} vs {summary_match[0]}",
            )


def validate_report(
    report: Any,
    require_reverse_check: bool,
    require_peak_rss: bool,
    require_ml_profile: bool,
    require_audit_contract: bool,
) -> None:
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    kind = json_field(report, "kind", "report")
    if kind == RUN_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_run_report(report, require_reverse_check, require_peak_rss)
        return
    if kind == COMPARISON_KIND:
        validate_artifact_comparison(report)
        validate_audit_contract(report, require_audit_contract)
        return
    if kind == MLP_WITNESS_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_mlp_witness_report(report)
        return
    if kind == INVERTIBLE_COUPLING_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_invertible_coupling_report(report)
        return
    if kind == TRIANGULAR_RESIDUAL_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_triangular_residual_report(report)
        return
    if kind == REVERSIBLE_PREPROCESS_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_reversible_preprocess_report(report)
        return
    if kind == REVERSIBLE_INFERENCE_TRACE_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_reversible_inference_trace_report(report)
        return
    if kind == Q31_REFERENCE_INFERENCE_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_q31_reference_inference_report(report)
        return
    if kind == Q31_REFERENCE_EVALUATION_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_q31_reference_evaluation_report(report)
        return
    if kind == AUDIT_STEP_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_audit_step_report(report)
        return
    if kind == AUDIT_SCAN_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_audit_scan_report(report)
        return
    if kind == AUDIT_VERIFICATION_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_audit_verification_report(report)
        return
    if kind == STEP_VERIFICATION_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_step_verification_report(report)
        return
    if kind == MODEL_IMPORT_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_model_import_report(report)
        return
    if kind == MODEL_VERIFICATION_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_model_verification_report(report)
        return
    if kind in (INFERENCE_AUDIT_KIND, MODEL_INFERENCE_AUDIT_KIND, MODEL_EVALUATION_ROW_KIND):
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_native_inference_report(report)
        return
    if kind == INFERENCE_VERIFICATION_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_inference_verification_report(report)
        return
    if kind == MODEL_EVALUATION_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_model_evaluation_report(report)
        return
    if kind == MODEL_EVALUATION_SCAN_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_model_evaluation_scan_report(report)
        return
    if kind == MODEL_EVALUATION_VERIFICATION_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_model_evaluation_verification_report(report)
        return
    if kind == PIPELINE_SUMMARY_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_pipeline_summary_report(report)
        return
    if kind == PIPELINE_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_pipeline_manifest_report(report)
        return
    if kind == MODEL_CAPSULE_KIND:
        if require_ml_profile:
            raise ValueError("expected artifact comparison report with ml_profile")
        validate_model_capsule_report(report)
        return
    raise ValueError(f"unsupported report kind `{kind}`")


def valid_run_report() -> dict[str, Any]:
    return {
        "kind": RUN_KIND,
        "config": {"reverse_check": True},
        "train": {
            "samples": 2,
            "correct": 1,
            "accuracy_percent": 50.0,
            "elapsed_seconds": 0.2,
            "updates_per_second": 10.0,
        },
        "eval": {
            "samples": 1,
            "correct": 1,
            "accuracy_percent": 100.0,
            "elapsed_seconds": 0.1,
            "samples_per_second": 10.0,
        },
        "trace": {"enabled": True, "entries": 2, "payload_bytes": 400, "bytes_per_step": 200},
        "reverse": {
            "enabled": True,
            "checked": 2,
            "restored_initial_model": True,
            "elapsed_seconds": 0.2,
            "steps_per_second": 10.0,
        },
        "proof": {
            "enabled": True,
            "entries": 2,
            "model_payload_bytes": 62800,
            "sample_bytes_per_step": 784,
            "sample_payload_bytes": 1568,
            "witness_bytes_per_step": 200,
            "witness_payload_bytes": 400,
            "trace_replay_bytes_per_step": 984,
            "trace_replay_payload_bytes": 1968,
            "full_replay_payload_bytes": 64768,
            "forward_recompute_steps": 2,
            "inverse_recompute_steps": 2,
        },
        "memory": {
            "model_payload_bytes": 62800,
            "train_dataset_payload_bytes": 1568,
            "eval_dataset_payload_bytes": 784,
            "dataset_payload_bytes": 2352,
            "trace_payload_bytes": 400,
            "estimated_payload_bytes": 65552,
            "peak_rss_bytes": 123456,
        },
    }


def valid_artifact_fingerprint_summary(kind: str) -> dict[str, Any]:
    has_provenance = kind == "model"
    has_proof = kind in {
        "training_trace",
        "sample_set",
        "training_step",
        "inference",
        "model_evaluation",
    }
    return {
        "algorithm": "sha256",
        "count": 7 if has_provenance or has_proof else 5,
        "source_count": 1,
        "has_computation": True,
        "has_payload": True,
        "has_proof": has_proof,
        "has_provenance": has_provenance,
    }


def valid_comparison_report() -> dict[str, Any]:
    artifacts = [
        {
            "kind": "model",
            "path": "model.json",
            "file_bytes": 1000,
            "logical_payload_bytes": 62800,
            "model_payload_bytes": 62800,
            "sample_payload_bytes": 0,
            "witness_payload_bytes": 0,
            "trace_payload_bytes": 0,
            "derived_update_payload_bytes": 0,
            "steps": 0,
            "forward_recompute_steps": 0,
            "inverse_recompute_steps": 0,
            "payload_fingerprint": "abc",
            "fingerprints": valid_artifact_fingerprint_summary("model"),
        },
        {
            "kind": "inference",
            "path": "inference.json",
            "file_bytes": 2000,
            "logical_payload_bytes": 63681,
            "model_payload_bytes": 62800,
            "sample_payload_bytes": 785,
            "witness_payload_bytes": 96,
            "trace_payload_bytes": 0,
            "derived_update_payload_bytes": 0,
            "steps": 1,
            "forward_recompute_steps": 1,
            "inverse_recompute_steps": 1,
            "payload_fingerprint": "def",
            "fingerprints": valid_artifact_fingerprint_summary("inference"),
        },
    ]
    total_model = 125600
    total_witness = 96
    total_logical = 126481
    total_file = 3000
    report = {
        "kind": COMPARISON_KIND,
        "artifacts": artifacts,
        "totals": {"count": 2, "file_bytes": total_file, "logical_payload_bytes": total_logical},
        "ml_profile": {
            "total_file_bytes": total_file,
            "total_logical_payload_bytes": total_logical,
            "total_model_payload_bytes": total_model,
            "total_sample_payload_bytes": 785,
            "total_witness_payload_bytes": total_witness,
            "total_trace_payload_bytes": 0,
            "total_derived_update_payload_bytes": 0,
            "total_steps": 1,
            "total_forward_recompute_steps": 1,
            "total_inverse_recompute_steps": 1,
            "total_recompute_steps": 2,
            "trace_to_model_payload_ratio": 0.0,
            "witness_to_model_payload_ratio": total_witness / total_model,
            "logical_to_file_ratio": total_logical / total_file,
        },
    }
    report["audit_contract"] = audit_contract_from_expected_results(
        expected_audit_contract_results(report)
    )
    return report


def valid_audit_contract_report() -> dict[str, Any]:
    artifacts = [
        {
            "kind": "training_trace",
            "path": "training.json",
            "file_bytes": 1000,
            "logical_payload_bytes": 64768,
            "model_payload_bytes": 62800,
            "sample_payload_bytes": 1568,
            "witness_payload_bytes": 400,
            "trace_payload_bytes": 1968,
            "derived_update_payload_bytes": 0,
            "steps": 2,
            "forward_recompute_steps": 2,
            "inverse_recompute_steps": 2,
            "payload_fingerprint": "abc",
            "fingerprints": valid_artifact_fingerprint_summary("training_trace"),
        },
        {
            "kind": "model",
            "path": "model.json",
            "file_bytes": 1000,
            "logical_payload_bytes": 62800,
            "model_payload_bytes": 62800,
            "sample_payload_bytes": 0,
            "witness_payload_bytes": 0,
            "trace_payload_bytes": 0,
            "derived_update_payload_bytes": 0,
            "steps": 0,
            "forward_recompute_steps": 0,
            "inverse_recompute_steps": 0,
            "payload_fingerprint": "def",
            "fingerprints": valid_artifact_fingerprint_summary("model"),
        },
        {
            "kind": "sample_set",
            "path": "samples.json",
            "file_bytes": 1000,
            "logical_payload_bytes": 785,
            "model_payload_bytes": 0,
            "sample_payload_bytes": 785,
            "witness_payload_bytes": 0,
            "trace_payload_bytes": 0,
            "derived_update_payload_bytes": 0,
            "steps": 1,
            "forward_recompute_steps": 0,
            "inverse_recompute_steps": 0,
            "payload_fingerprint": "ghi",
            "fingerprints": valid_artifact_fingerprint_summary("sample_set"),
        },
        {
            "kind": "training_step",
            "path": "step.json",
            "file_bytes": 1000,
            "logical_payload_bytes": 126625,
            "model_payload_bytes": 125600,
            "sample_payload_bytes": 785,
            "witness_payload_bytes": 96,
            "trace_payload_bytes": 0,
            "derived_update_payload_bytes": 216,
            "steps": 1,
            "forward_recompute_steps": 1,
            "inverse_recompute_steps": 1,
            "payload_fingerprint": "jkl",
            "fingerprints": valid_artifact_fingerprint_summary("training_step"),
        },
        {
            "kind": "inference",
            "path": "inference.json",
            "file_bytes": 1000,
            "logical_payload_bytes": 63681,
            "model_payload_bytes": 62800,
            "sample_payload_bytes": 785,
            "witness_payload_bytes": 96,
            "trace_payload_bytes": 0,
            "derived_update_payload_bytes": 0,
            "steps": 1,
            "forward_recompute_steps": 1,
            "inverse_recompute_steps": 1,
            "payload_fingerprint": "mno",
            "fingerprints": valid_artifact_fingerprint_summary("inference"),
        },
        {
            "kind": "model_evaluation",
            "path": "evaluation.json",
            "file_bytes": 1000,
            "logical_payload_bytes": 63681,
            "model_payload_bytes": 62800,
            "sample_payload_bytes": 785,
            "witness_payload_bytes": 96,
            "trace_payload_bytes": 0,
            "derived_update_payload_bytes": 0,
            "steps": 1,
            "forward_recompute_steps": 1,
            "inverse_recompute_steps": 1,
            "payload_fingerprint": "pqr",
            "fingerprints": valid_artifact_fingerprint_summary("model_evaluation"),
        },
    ]
    total_file = sum(int(artifact["file_bytes"]) for artifact in artifacts)
    total_logical = sum(int(artifact["logical_payload_bytes"]) for artifact in artifacts)
    total_model = sum(int(artifact["model_payload_bytes"]) for artifact in artifacts)
    total_sample = sum(int(artifact["sample_payload_bytes"]) for artifact in artifacts)
    total_witness = sum(int(artifact["witness_payload_bytes"]) for artifact in artifacts)
    total_trace = sum(int(artifact["trace_payload_bytes"]) for artifact in artifacts)
    total_update = sum(int(artifact["derived_update_payload_bytes"]) for artifact in artifacts)
    total_steps = sum(int(artifact["steps"]) for artifact in artifacts)
    total_forward = sum(int(artifact["forward_recompute_steps"]) for artifact in artifacts)
    total_inverse = sum(int(artifact["inverse_recompute_steps"]) for artifact in artifacts)
    report = {
        "kind": COMPARISON_KIND,
        "artifacts": artifacts,
        "totals": {
            "count": len(artifacts),
            "file_bytes": total_file,
            "logical_payload_bytes": total_logical,
        },
        "ml_profile": {
            "total_file_bytes": total_file,
            "total_logical_payload_bytes": total_logical,
            "total_model_payload_bytes": total_model,
            "total_sample_payload_bytes": total_sample,
            "total_witness_payload_bytes": total_witness,
            "total_trace_payload_bytes": total_trace,
            "total_derived_update_payload_bytes": total_update,
            "total_steps": total_steps,
            "total_forward_recompute_steps": total_forward,
            "total_inverse_recompute_steps": total_inverse,
            "total_recompute_steps": total_forward + total_inverse,
            "trace_to_model_payload_ratio": total_trace / total_model,
            "witness_to_model_payload_ratio": total_witness / total_model,
            "logical_to_file_ratio": total_logical / total_file,
        },
    }
    report["audit_contract"] = audit_contract_from_expected_results(
        expected_audit_contract_results(report)
    )
    return report


def valid_mlp_witness_report() -> dict[str, Any]:
    witness_tape_fields = [
        "hidden_pre_tape",
        "hidden_mask_tape",
        "hidden_tape",
        "logits_tape",
        "out_error_tape",
        "hidden_back_tape",
        "hidden_delta_tape",
        "prediction_tape",
        "correct_tape",
    ]
    witness_cells = {
        "hidden_pre_tape": 32,
        "hidden_mask_tape": 32,
        "hidden_tape": 32,
        "logits_tape": 20,
        "out_error_tape": 20,
        "hidden_back_tape": 32,
        "hidden_delta_tape": 32,
        "prediction_tape": 2,
        "correct_tape": 2,
    }
    witness_proof_payload = {
        "schema": "reverie_witness_store_proof_v1",
        "algorithm": "sha256",
        "variables": len(witness_tape_fields),
        "known_cells": sum(witness_cells.values()),
        "known_payload_bytes": 1632,
        "unknown_variables": [],
        "entries": [
            {
                "name": name,
                "present": True,
                "cells": witness_cells[name],
                "payload_bytes": witness_cells[name] * I64_BYTES,
                "value_fingerprint": str(index) * 64,
            }
            for index, name in enumerate(witness_tape_fields)
        ],
    }
    return {
        "kind": MLP_WITNESS_KIND,
        "program": "examples/mnist_mlp_witness.rev",
        "vars_json": "target/mlp-check-seed.json",
        "run_output_json": "target/mlp-check-run.json",
        "samples": 2,
        "q31_one": 1 << 31,
        "lr": 2147483,
        "dataset_loops": expected_mlp_dataset_loops(),
        "predictions": [0, 1],
        "correct": [True, True],
        "checked_store_fields": [
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
        ],
        "witness_proof": {
            "schema": "reverie_witness_store_proof_v1",
            "algorithm": "sha256",
            "fingerprint": sha256_json(witness_proof_payload),
            "payload": witness_proof_payload,
        },
        "mismatches": [],
        "passed": True,
        "proof": {
            "claim": MLP_WITNESS_PROOF_CLAIM,
            "model_payload_bytes": 101840,
            "sample_payload_bytes": 12560,
            "witness_payload_bytes": 1632,
            "witness_bytes_per_sample": 816,
            "trace_payload_bytes": 14192,
            "trace_bytes_per_sample": 7096,
            "replay_payload_bytes": 116032,
            "stored_derived_update_payload_bytes": 0,
            "recomputed_update_payload_bytes": 203680,
            "recomputed_update_bytes_per_sample": 101840,
            "forward_recompute_steps": 2,
            "inverse_recompute_steps": 2,
            "total_recompute_steps": 4,
            "witness_to_model_payload_ratio": 1632 / 101840,
            "trace_to_model_payload_ratio": 14192 / 101840,
            "recomputed_update_to_witness_payload_ratio": 203680 / 1632,
            "checked_store_fields": 16,
            "witness_tape_fields": witness_tape_fields,
        },
    }


def valid_invertible_coupling_report() -> dict[str, Any]:
    return coupling_check.build_report(
        Path("target/invertible-coupling-forward.json"),
        Path("target/invertible-coupling-reverse.json"),
        [],
    )


def valid_triangular_residual_report() -> dict[str, Any]:
    return residual_check.build_report(
        Path("target/triangular-residual-forward.json"),
        Path("target/triangular-residual-reverse.json"),
        [],
    )


def valid_reversible_preprocess_report() -> dict[str, Any]:
    return preprocess_check.build_report(
        Path("target/reversible-preprocess-forward.json"),
        Path("target/reversible-preprocess-reverse.json"),
        [],
    )


def valid_reversible_inference_trace_report() -> dict[str, Any]:
    return inference_trace_check.build_report(
        Path("target/reversible-inference-trace-forward.json"),
        Path("target/reversible-inference-trace-reverse.json"),
        [],
    )


def valid_q31_reference_inference_report() -> dict[str, Any]:
    logits = [0 for _ in range(DIGITS)]
    logits[3] = Q31_ONE
    logits[4] = Q31_ONE // 2
    return {
        "kind": Q31_REFERENCE_INFERENCE_KIND,
        "model": "target/imported-q31-linear-model-bundle.json",
        "sample": "target/mnist-samples.json",
        "prediction": 3,
        "correct": True,
        "label": 3,
        "logits": logits,
        "top_logits": [
            {"digit": 3, "value": Q31_ONE},
            {"digit": 4, "value": Q31_ONE // 2},
            {"digit": 0, "value": 0},
        ],
        "attribution": {
            "predicted_digit": 3,
            "runner_up_digit": 4,
            "predicted_logit": Q31_ONE,
            "runner_up_logit": Q31_ONE // 2,
            "margin": Q31_ONE // 2,
            "bias": 0,
            "runner_up_bias": 0,
            "margin_bias": 0,
            "contribution_sum": Q31_ONE,
            "margin_contribution_sum": Q31_ONE // 2,
            "reconstructed_logit": Q31_ONE,
            "reconstructed_margin": Q31_ONE // 2,
            "matches_logit": True,
            "matches_margin": True,
            "contribution_count": 1,
            "margin_contribution_count": 2,
            "top_contribution_count": 1,
            "top_margin_contribution_count": 2,
            "contribution_ledger_fingerprint": "0" * 64,
            "margin_contribution_ledger_fingerprint": "1" * 64,
            "top_contributions": [
                {
                    "pixel": 0,
                    "u8": 255,
                    "q31": Q31_ONE,
                    "weight": Q31_ONE,
                    "contribution": Q31_ONE,
                }
            ],
            "top_margin_contributions": [
                {
                    "pixel": 0,
                    "u8": 255,
                    "q31": Q31_ONE,
                    "predicted_weight": Q31_ONE,
                    "runner_up_weight": 0,
                    "weight_delta": Q31_ONE,
                    "contribution": Q31_ONE,
                },
                {
                    "pixel": 1,
                    "u8": 255,
                    "q31": Q31_ONE,
                    "predicted_weight": 0,
                    "runner_up_weight": Q31_ONE // 2,
                    "weight_delta": -(Q31_ONE // 2),
                    "contribution": -(Q31_ONE // 2),
                },
            ],
        },
        "active_pixels": 2,
        "q31_one": Q31_ONE,
        "sample_source": {
            "sample_index": 1,
            "sample_count": 2,
            "sample_set_kind": "reverie_mnist_linear_q31_samples",
            "sample_kind": "reverie_mnist_linear_q31_sample",
            "audit_step": 7,
            "source_sample_index": 11,
        },
    }


def valid_q31_reference_evaluation_report() -> dict[str, Any]:
    rows = [
        {
            "index": 0,
            "label": 0,
            "prediction": 0,
            "correct": True,
            "margin": 0,
            "runner_up_digit": 1,
            "active_pixels": 0,
            "sample_source": {
                "sample_index": 0,
                "sample_count": 2,
                "sample_set_kind": "reverie_mnist_linear_q31_samples",
                "audit_step": 0,
            },
        },
        {
            "index": 1,
            "label": 3,
            "prediction": 3,
            "correct": True,
            "margin": Q31_ONE // 2,
            "runner_up_digit": 4,
            "active_pixels": 2,
            "sample_source": {
                "sample_index": 1,
                "sample_count": 2,
                "sample_set_kind": "reverie_mnist_linear_q31_samples",
                "sample_kind": "reverie_mnist_linear_q31_sample",
                "audit_step": 7,
                "source_sample_index": 11,
            },
        },
    ]
    by_label = []
    for label in range(DIGITS):
        matching = [row for row in rows if row["label"] == label]
        if matching:
            lowest = min(matching, key=lambda row: (row["margin"], row["index"]))
            correct = sum(1 for row in matching if row["correct"])
            by_label.append(
                {
                    "label": label,
                    "samples": len(matching),
                    "correct": correct,
                    "incorrect": len(matching) - correct,
                    "accuracy_percent": 100.0 * correct / len(matching),
                    "lowest_margin_index": lowest["index"],
                    "lowest_margin": lowest["margin"],
                }
            )
        else:
            by_label.append(
                {
                    "label": label,
                    "samples": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "accuracy_percent": 0.0,
                    "lowest_margin_index": None,
                    "lowest_margin": None,
                }
            )
    return {
        "kind": Q31_REFERENCE_EVALUATION_KIND,
        "model": "target/imported-q31-linear-model-bundle.json",
        "sample": "target/mnist-samples.json",
        "q31_one": Q31_ONE,
        "report_limit": 10,
        "summary": {
            "samples": 2,
            "correct": 2,
            "incorrect": 0,
            "accuracy_percent": 100.0,
            "lowest_margin_index": 0,
            "lowest_margin": 0,
        },
        "by_label": by_label,
        "rows": rows,
        "top_low_margin": rows,
        "top_incorrect": [],
    }


def fake_fingerprints(*fields: str) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        **{field: "0" * 64 for field in fields},
    }


def valid_training_cause_ledger(
    *,
    step: int = 0,
    sample_index: int = 0,
    label: int = 0,
    lr: int = 2147483,
    logits: Optional[list[int]] = None,
    error: Optional[list[int]] = None,
    prediction: int = 0,
    correct: bool = True,
    active_pixels: Optional[list[dict[str, int]]] = None,
    top_logits: Optional[list[dict[str, int]]] = None,
    logit_margin: Optional[dict[str, int]] = None,
    update_ledger_fingerprints: Optional[dict[str, str]] = None,
    top_weight_deltas: Optional[list[dict[str, int]]] = None,
) -> dict[str, Any]:
    logits = logits if logits is not None else [Q31_ONE, Q31_ONE // 2] + [0 for _ in range(DIGITS - 2)]
    error = error if error is not None else [-Q31_ONE] + [0 for _ in range(DIGITS - 1)]
    active_pixels = active_pixels if active_pixels is not None else [
        {"index": 0, "u8": 255, "q31": Q31_ONE},
        {"index": 10, "u8": 128, "q31": (128 * Q31_ONE) // 255},
    ]
    top_logits = top_logits if top_logits is not None else [
        {"digit": 0, "logit": Q31_ONE},
        {"digit": 1, "logit": Q31_ONE // 2},
        {"digit": 2, "logit": 0},
    ]
    logit_margin = logit_margin if logit_margin is not None else {
        "predicted_digit": 0,
        "predicted_logit": Q31_ONE,
        "runner_up_digit": 1,
        "runner_up_logit": Q31_ONE // 2,
        "margin": Q31_ONE // 2,
    }
    update_ledger_fingerprints = update_ledger_fingerprints if update_ledger_fingerprints is not None else {
        "algorithm": "sha256",
        "bias_delta": "2" * 64,
        "weight_delta": "3" * 64,
    }
    top_weight_deltas = top_weight_deltas if top_weight_deltas is not None else [
        {"pixel": 0, "digit": 0, "delta": 100},
        {"pixel": 10, "digit": 0, "delta": 50},
    ]
    return {
        "algorithm": "sha256",
        "schema": TRAINING_CAUSE_LEDGER_SCHEMA,
        "fingerprint": "4" * 64,
        "payload": {
            "schema": TRAINING_CAUSE_LEDGER_SCHEMA,
            "step": step,
            "sample_index": sample_index,
            "sample_fingerprint": "a" * 64,
            "witness_fingerprint": "b" * 64,
            "update_fingerprint": "c" * 64,
            "label": label,
            "lr": lr,
            "logits": logits,
            "error": error,
            "prediction": prediction,
            "correct": correct,
            "active_pixels": active_pixels,
            "top_logits": top_logits,
            "logit_margin": logit_margin,
            "update_ledger_fingerprints": update_ledger_fingerprints,
            "top_weight_deltas": top_weight_deltas,
        },
    }


def valid_lineage_transition(step: int, *, steps: int = 2) -> dict[str, Any]:
    return {
        "step": step,
        "sample_index": step,
        "label": step % DIGITS,
        "prediction": step % DIGITS,
        "correct": True,
        "lr": 2147483,
        "sample_fingerprint": "a" * 64,
        "witness_fingerprint": "b" * 64,
        "update_fingerprint": "c" * 64,
        "cause_ledger_fingerprint": "4" * 64,
        "before_chain": "5" * 64 if step == 0 else "6" * 64,
        "transition_fingerprint": "7" * 64,
        "after_chain": "8" * 64 if step == steps - 1 else "6" * 64,
    }


def valid_training_lineage_ledger(*, steps: int = 2) -> dict[str, Any]:
    return {
        "algorithm": "sha256",
        "schema": TRAINING_LINEAGE_LEDGER_SCHEMA,
        "fingerprint": "9" * 64,
        "payload": {
            "schema": TRAINING_LINEAGE_LEDGER_SCHEMA,
            "steps": steps,
            "initial_model_fingerprint": "0" * 64,
            "final_model_fingerprint": "1" * 64,
            "witness_trace_fingerprint": "2" * 64,
            "sample_order_fingerprint": "3" * 64,
            "transition_ledger_fingerprint": "4" * 64,
            "initial_chain": "5" * 64,
            "final_chain": "8" * 64,
            "first_transition": valid_lineage_transition(0, steps=steps),
            "last_transition": valid_lineage_transition(steps - 1, steps=steps),
        },
    }


def valid_audit_step_report() -> dict[str, Any]:
    logits = [0 for _ in range(DIGITS)]
    logits[0] = Q31_ONE
    logits[1] = Q31_ONE // 2
    bias_before = [0 for _ in range(DIGITS)]
    bias_after = [0 for _ in range(DIGITS)]
    bias_after[0] = 2147483
    bias_delta = [0 for _ in range(DIGITS)]
    bias_delta[0] = 2147483
    top_logits = [
        {"digit": 0, "logit": Q31_ONE},
        {"digit": 1, "logit": Q31_ONE // 2},
        {"digit": 2, "logit": 0},
    ]
    logit_margin = {
        "predicted_digit": 0,
        "predicted_logit": Q31_ONE,
        "runner_up_digit": 1,
        "runner_up_logit": Q31_ONE // 2,
        "margin": Q31_ONE // 2,
    }
    error = [-Q31_ONE] + [0 for _ in range(DIGITS - 1)]
    active_pixels = [
        {"index": 0, "u8": 255, "q31": Q31_ONE},
        {"index": 10, "u8": 128, "q31": (128 * Q31_ONE) // 255},
    ]
    update = {
        "formula": "weights -= scale_q31(outer_q31(image, error), lr); bias -= scale_q31(error, lr)",
        "bias_delta": bias_delta,
        "nonzero_bias_delta_count": 1,
        "nonzero_weight_delta_count": 2,
        "max_abs_weight_delta": 100,
        "bias_delta_ledger_fingerprint": "2" * 64,
        "weight_delta_ledger_fingerprint": "3" * 64,
        "ledger_fingerprints": {
            "algorithm": "sha256",
            "bias_delta": "2" * 64,
            "weight_delta": "3" * 64,
        },
        "top_weight_deltas": [
            {"pixel": 0, "digit": 0, "delta": 100},
            {"pixel": 10, "digit": 0, "delta": 50},
        ],
    }
    return {
        "kind": AUDIT_STEP_KIND,
        "path": "target/mnist-self-test-replay-bundle.json",
        "fingerprints": fake_fingerprints(*AUDIT_FINGERPRINT_FIELDS),
        "step_output": "target/mnist-self-test-step-bundle.json",
        "step": 0,
        "sample_index": 0,
        "label": 0,
        "prediction": 0,
        "correct": True,
        "lr": 2147483,
        "logits": logits,
        "top_logits": top_logits,
        "logit_margin": logit_margin,
        "error": error,
        "witness_checks": {
            "computed_prediction": 0,
            "computed_correct": True,
            "prediction_matches_logits": True,
            "correct_matches_logits": True,
        },
        "active_pixels": active_pixels,
        "update": update,
        "cause_ledger": valid_training_cause_ledger(
            logits=logits,
            error=error,
            active_pixels=active_pixels,
            top_logits=top_logits,
            logit_margin=logit_margin,
            update_ledger_fingerprints=update["ledger_fingerprints"],
            top_weight_deltas=update["top_weight_deltas"],
        ),
        "model_window": {
            "reconstructed": True,
            "reversed_later_steps": 1,
            "bias_before": bias_before,
            "bias_after": bias_after,
            "bias_observed_delta": bias_delta,
            "bias_delta_matches": True,
            "weight_delta_matches": True,
            "top_weight_windows": [
                {
                    "pixel": 0,
                    "digit": 0,
                    "before": 0,
                    "after": 100,
                    "observed_delta": 100,
                    "computed_delta": 100,
                    "delta_matches": True,
                },
                {
                    "pixel": 10,
                    "digit": 0,
                    "before": 0,
                    "after": 50,
                    "observed_delta": 50,
                    "computed_delta": 50,
                    "delta_matches": True,
                },
            ],
        },
        "debug_contract": {
            "claim": "step_backward_from_model_update",
            "passed": True,
            "checks": [
                {
                    "metric": "witness_recomputes_prediction",
                    "passed": True,
                    "actual": "self-test",
                    "requirement": "self-test",
                },
                {
                    "metric": "model_window_reconstructed",
                    "passed": True,
                    "actual": "self-test",
                    "requirement": "self-test",
                },
                {
                    "metric": "update_matches_model_window",
                    "passed": True,
                    "actual": "self-test",
                    "requirement": "self-test",
                },
                {
                    "metric": "explanatory_state_present",
                    "passed": True,
                    "actual": "self-test",
                    "requirement": "self-test",
                },
                {
                    "metric": "update_ledger_fingerprints",
                    "passed": True,
                    "actual": "self-test",
                    "requirement": "self-test",
                },
            ],
            "replay_direction": {
                "backward_from_final_model_steps": 1,
                "selected_transition": "before_model + sample + witness -> after_model",
                "reverse_transition": "after_model + sample + witness -> before_model",
            },
            "debug_focus": {
                "prediction": "top_logits and logit_margin",
                "update": "error, active_pixels, top_weight_deltas, and ledger_fingerprints",
                "reversibility": "model_window and witness_checks",
            },
        },
    }


def valid_audit_scan_report() -> dict[str, Any]:
    rows = [
        {
            "step": 0,
            "sample_index": 0,
            "label": 0,
            "prediction": 0,
            "correct": True,
            "lr": 2147483,
            "computed_prediction": 0,
            "predicted_logit": Q31_ONE,
            "runner_up_digit": 1,
            "runner_up_logit": Q31_ONE // 2,
            "margin": Q31_ONE // 2,
            "prediction_matches_logits": True,
            "correct_matches_logits": True,
            "active_pixel_count": 2,
            "error_nonzero_count": 1,
            "max_abs_error": Q31_ONE,
            "max_abs_bias_delta": 2147483,
            "nonzero_weight_delta_count": 2,
            "max_abs_weight_delta": 100,
        },
        {
            "step": 1,
            "sample_index": 1,
            "label": 1,
            "prediction": 1,
            "correct": True,
            "lr": 2147483,
            "computed_prediction": 1,
            "predicted_logit": Q31_ONE // 2,
            "runner_up_digit": 0,
            "runner_up_logit": (Q31_ONE // 2) - 100,
            "margin": 100,
            "prediction_matches_logits": True,
            "correct_matches_logits": True,
            "active_pixel_count": 2,
            "error_nonzero_count": 2,
            "max_abs_error": Q31_ONE // 2,
            "max_abs_bias_delta": 1000,
            "nonzero_weight_delta_count": 2,
            "max_abs_weight_delta": 50,
        },
    ]
    by_label = []
    for label in range(DIGITS):
        row = next((item for item in rows if item["label"] == label), None)
        if row is None:
            by_label.append(
                {
                    "label": label,
                    "samples": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "accuracy_percent": 0.0,
                    "min_margin_step": None,
                    "min_margin": None,
                    "largest_update_step": None,
                    "max_abs_weight_delta": 0,
                }
            )
        else:
            by_label.append(
                {
                    "label": label,
                    "samples": 1,
                    "correct": 1,
                    "incorrect": 0,
                    "accuracy_percent": 100.0,
                    "min_margin_step": row["step"],
                    "min_margin": row["margin"],
                    "largest_update_step": row["step"],
                    "max_abs_weight_delta": row["max_abs_weight_delta"],
                }
            )
    return {
        "kind": AUDIT_SCAN_KIND,
        "path": "target/mnist-self-test-replay-bundle.json",
        "fingerprints": fake_fingerprints(*AUDIT_FINGERPRINT_FIELDS),
        "summary": {
            "steps": 2,
            "correct": 2,
            "incorrect": 0,
            "accuracy_percent": 100.0,
            "witness_mismatches": 0,
            "largest_update_step": 0,
            "max_abs_weight_delta": 100,
            "lowest_margin_step": 1,
            "lowest_margin": 100,
            "trace_payload_bytes": 400,
            "bytes_per_step": 200,
        },
        "by_label": by_label,
        "top_confusions": [],
        "top_suspicious": rows,
        "top_low_margin": [rows[1], rows[0]],
        "top_large_updates": rows,
        "gate": {
            "passed": True,
            "checks": [
                {
                    "metric": "accuracy_percent",
                    "passed": True,
                    "actual": "100.00",
                    "expectation": ">= 100.00",
                }
            ],
        },
    }


def valid_training_trace_proof(entries: int = 2) -> dict[str, Any]:
    model_payload_bytes = 62800
    sample_bytes_per_step = 784
    witness_bytes_per_step = 200
    trace_replay_bytes_per_step = sample_bytes_per_step + witness_bytes_per_step
    sample_payload_bytes = entries * sample_bytes_per_step
    witness_payload_bytes = entries * witness_bytes_per_step
    trace_replay_payload_bytes = entries * trace_replay_bytes_per_step
    return {
        "enabled": True,
        "entries": entries,
        "model_payload_bytes": model_payload_bytes,
        "sample_payload_bytes": sample_payload_bytes,
        "witness_payload_bytes": witness_payload_bytes,
        "trace_replay_payload_bytes": trace_replay_payload_bytes,
        "full_replay_payload_bytes": model_payload_bytes + trace_replay_payload_bytes,
        "sample_bytes_per_step": sample_bytes_per_step,
        "witness_bytes_per_step": witness_bytes_per_step,
        "trace_replay_bytes_per_step": trace_replay_bytes_per_step,
        "forward_recompute_steps": entries,
        "inverse_recompute_steps": entries,
    }


def valid_audit_verification_report() -> dict[str, Any]:
    entries = 2
    return {
        "kind": AUDIT_VERIFICATION_KIND,
        "path": "target/mnist-self-test-replay-bundle.json",
        "fingerprints": fake_fingerprints(*AUDIT_FINGERPRINT_FIELDS),
        "checked": entries,
        "witnesses_match_forward_replay": True,
        "final_model_replayed": True,
        "restored_initial_model": True,
        "proof_matches": True,
        "lineage_ledger_matches": True,
        "elapsed_seconds": 0.01,
        "steps_per_second": 200.0,
        "proof": valid_training_trace_proof(entries),
        "lineage_ledger": valid_training_lineage_ledger(steps=entries),
        "forward": {
            "checked": entries,
            "witnesses_match": True,
            "final_model_matches": True,
            "elapsed_seconds": 0.005,
            "steps_per_second": 400.0,
        },
        "reverse": {
            "checked": entries,
            "restored_initial_model": True,
            "elapsed_seconds": 0.005,
            "steps_per_second": 400.0,
        },
    }


def valid_step_proof() -> dict[str, Any]:
    model_payload_bytes = 125600
    sample_payload_bytes = 785
    witness_payload_bytes = 176
    update_payload_bytes = 216
    cause_ledger = valid_training_cause_ledger()
    fingerprints = fake_fingerprints(
        "train_source",
        "before_model",
        "after_model",
        "sample",
        "witness",
        "update",
        "cause_ledger",
    )
    fingerprints["cause_ledger"] = cause_ledger["fingerprint"]
    return {
        "claim": TRAINING_STEP_PROOF_CLAIM,
        "kernel": "examples/mnist_reversible_step.rev",
        "arithmetic": "q31_wrapping_i64",
        "witnesses": ["logits", "error", "prediction", "correct", "lr"],
        "model_payload_bytes": model_payload_bytes,
        "sample_payload_bytes": sample_payload_bytes,
        "witness_payload_bytes": witness_payload_bytes,
        "derived_update_payload_bytes": update_payload_bytes,
        "nonzero_bias_delta_count": 1,
        "nonzero_weight_delta_count": 2,
        "update_ledger_fingerprints": {
            "algorithm": "sha256",
            "bias_delta": "2" * 64,
            "weight_delta": "3" * 64,
        },
        "cause_ledger": cause_ledger,
        "trace_payload_bytes": 0,
        "replay_payload_bytes": model_payload_bytes
        + sample_payload_bytes
        + witness_payload_bytes
        + update_payload_bytes,
        "forward_recompute_steps": 1,
        "inverse_recompute_steps": 1,
        "checks": {
            "witnesses_match_forward_replay": True,
            "after_model_matches": True,
            "before_model_restored": True,
            "update_matches_witnesses": True,
        },
        "fingerprints": fingerprints,
    }


def valid_step_verification_report() -> dict[str, Any]:
    return {
        "kind": STEP_VERIFICATION_KIND,
        "path": "target/mnist-self-test-step-bundle.json",
        "fingerprints": fake_fingerprints(
            "train_source",
            "computation",
            "before_model",
            "after_model",
            "sample",
            "witness",
            "update",
            "proof",
            "payload",
        ),
        "proof_matches": True,
        "proof": valid_step_proof(),
        "forward": {
            "witnesses_match": True,
            "after_model_matches": True,
            "elapsed_seconds": 0.001,
        },
        "reverse": {
            "before_model_restored": True,
            "elapsed_seconds": 0.001,
        },
    }


def valid_inference_memory(expected_steps: int = 1) -> dict[str, Any]:
    model_payload_bytes = 62800
    sample_payload_bytes = 785 * expected_steps
    witness_payload_bytes = 96 * expected_steps
    return {
        "model_payload_bytes": model_payload_bytes,
        "sample_payload_bytes": sample_payload_bytes,
        "witness_payload_bytes": witness_payload_bytes,
        "trace_payload_bytes": 0,
        "replay_payload_bytes": model_payload_bytes + sample_payload_bytes + witness_payload_bytes,
        "runtime_state_payload_bytes": 63681,
        "forward_recompute_steps": expected_steps,
        "inverse_recompute_steps": expected_steps,
        "peak_rss_bytes": None,
    }


def valid_inference_proof(prediction: int = 3, correct: bool = True, margin: int = Q31_ONE // 2) -> dict[str, Any]:
    return {
        "claim": INFERENCE_PROOF_CLAIM,
        "kernel": "examples/mnist_identify.rev",
        "arithmetic": "q31_wrapping_i64",
        "witnesses": ["logits", "prediction", "correct"],
        "prediction": prediction,
        "correct": correct,
        "predicted_digit": prediction,
        "runner_up_digit": 4,
        "margin": margin,
        "active_pixels": 2,
        "contribution_count": 1,
        "margin_contribution_count": 2,
        "contribution_ledger_fingerprint": "0" * 64,
        "margin_contribution_ledger_fingerprint": "1" * 64,
        **valid_inference_memory(),
        "checks": {
            "prediction_matches_logits": True,
            "correct_matches_logits": True,
            "attribution_matches_logit": True,
            "attribution_matches_margin": True,
            "restored_initial_state": True,
        },
        "fingerprints": fake_fingerprints(
            "identify_source",
            "model",
            "sample",
            "result",
        ),
    }


def valid_inference_explanation_contract(
    *,
    verification: bool = False,
    prediction: int = 3,
    correct: bool = True,
    margin: int = Q31_ONE // 2,
) -> dict[str, Any]:
    checks = [
        {
            "metric": "logits_determine_prediction",
            "passed": True,
            "actual": "self-test",
            "requirement": "self-test",
        },
        {
            "metric": "correct_matches_label",
            "passed": True,
            "actual": "self-test",
            "requirement": "self-test",
        },
        {
            "metric": "attribution_reconstructs_logit",
            "passed": True,
            "actual": "self-test",
            "requirement": "self-test",
        },
        {
            "metric": "attribution_reconstructs_margin",
            "passed": True,
            "actual": "self-test",
            "requirement": "self-test",
        },
        {
            "metric": "reverse_restores_initial_state",
            "passed": True,
            "actual": "self-test",
            "requirement": "self-test",
        },
    ]
    if verification:
        checks.extend(
            [
                {
                    "metric": "proof_recomputed",
                    "passed": True,
                    "actual": "self-test",
                    "requirement": "self-test",
                },
                {
                    "metric": "result_recomputed",
                    "passed": True,
                    "actual": "self-test",
                    "requirement": "self-test",
                },
                {
                    "metric": "source_inputs_checked",
                    "passed": True,
                    "actual": "self-test",
                    "requirement": "self-test",
                },
                {
                    "metric": "verification_restores_initial_state",
                    "passed": True,
                    "actual": "self-test",
                    "requirement": "self-test",
                },
            ]
        )
    return {
        "claim": INFERENCE_EXPLANATION_CLAIM,
        "passed": True,
        "prediction": prediction,
        "correct": correct,
        "predicted_digit": prediction,
        "runner_up_digit": 4,
        "margin": margin,
        "active_pixel_count": 2,
        "contribution_count": 1,
        "margin_contribution_count": 2,
        "top_logit_count": DIGITS,
        "ledger_fingerprints": {
            "algorithm": "sha256",
            "contribution": "0" * 64,
            "margin_contribution": "1" * 64,
        },
        "replay_direction": {
            "forward_transition": "model + sample -> logits + prediction",
            "reverse_transition": "model + sample + result -> initial state",
            "inverse_restores_initial_state": True,
        },
        "checks": checks,
    }


def valid_model_fingerprint_block() -> dict[str, str]:
    return fake_fingerprints(
        "train_source",
        "identify_source",
        "computation",
        "model",
        "provenance",
        "report",
        "payload",
    )


def valid_imported_model_source_check() -> dict[str, Any]:
    return {
        "path": "target/imported-q31-linear-model.json",
        "resolved_path": "target/imported-q31-linear-model.json",
        "checked": True,
        "fingerprint": "a" * 64,
        "file_sha256": "b" * 64,
    }


def valid_model_import_report() -> dict[str, Any]:
    return {
        "kind": MODEL_IMPORT_KIND,
        "source_model_json_path": "target/imported-q31-linear-model.json",
        "model_output": "target/imported-model-bundle.json",
        "source_model_json_fingerprint": "a" * 64,
        "source_model_json_file_sha256": "b" * 64,
        "fingerprints": valid_model_fingerprint_block(),
        "shape_matches": True,
        "provenance_kind": "external_import",
        "storage": {
            "model_payload_bytes": 62760,
        },
    }


def valid_model_verification_report() -> dict[str, Any]:
    return {
        "kind": MODEL_VERIFICATION_KIND,
        "path": "target/imported-model-bundle.json",
        "fingerprints": valid_model_fingerprint_block(),
        "model_payload_bytes": 62760,
        "shape_matches": True,
        "provenance_matches": True,
        "proof_matches": None,
        "provenance_kind": "external_import",
        "training_steps": None,
        "source_audit_payload": None,
        "source_model_json_checked": True,
        "source_model_json": valid_imported_model_source_check(),
    }


def valid_model_evaluation_fingerprint_block() -> dict[str, str]:
    return fake_fingerprints(
        "identify_source",
        "computation",
        "model",
        "samples",
        "summary",
        "proof",
        "rows",
        "gate_policy",
        "gate",
        "report",
        "payload",
    )


def valid_inference_fingerprint_block() -> dict[str, str]:
    return fake_fingerprints(
        "identify_source",
        "computation",
        "model",
        "sample",
        "result",
        "proof",
        "report",
        "payload",
    )


def valid_native_inference_report(kind: str = MODEL_INFERENCE_AUDIT_KIND) -> dict[str, Any]:
    base = valid_q31_reference_inference_report()
    logits = base["logits"]
    report = {
        "kind": kind,
        "label": base["label"],
        "prediction": base["prediction"],
        "correct": base["correct"],
        "logits": logits,
        "top_logits": [
            {"digit": item["digit"], "logit": item["value"]}
            for item in base["top_logits"]
        ],
        "attribution": base["attribution"],
        "witness_checks": {
            "computed_prediction": base["prediction"],
            "computed_correct": True,
            "prediction_matches_logits": True,
            "correct_matches_logits": True,
        },
        "active_pixels": [
            {"index": 0, "u8": 255, "q31": Q31_ONE},
            {"index": 1, "u8": 255, "q31": Q31_ONE},
        ],
        "forward": {"elapsed_seconds": 0.001},
        "inverse": {"restored_initial_state": True, "elapsed_seconds": 0.001},
        "memory": valid_inference_memory(),
        "proof": valid_inference_proof(),
        "explanation_contract": valid_inference_explanation_contract(),
        "inference_output": "target/inference-bundle.json",
    }
    if kind == INFERENCE_AUDIT_KIND:
        report.update(
            {
                "path": "target/mnist-debug-replay-bundle.json",
                "fingerprints": fake_fingerprints(*AUDIT_FINGERPRINT_FIELDS),
                "audit_step": 0,
                "sample_index": 0,
                "model": {
                    "source": "final_model",
                    "weights_shape": [IMAGE_PIXELS, DIGITS],
                    "bias_shape": [DIGITS],
                },
            }
        )
    else:
        report.update(
            {
                "model_bundle": {
                    "path": "target/model-bundle.json",
                    "fingerprints": valid_model_fingerprint_block(),
                },
                "sample_source": {
                    "kind": "sample_json",
                    "path": "target/sample.json",
                    "fingerprint": "0" * 64,
                },
                "sample_audit": None,
                "sample_json": {"path": "target/sample.json", "fingerprint": "0" * 64},
                "audit_step": None,
                "sample_index": None,
                "model": {
                    "source": "model_bundle",
                    "weights_shape": [IMAGE_PIXELS, DIGITS],
                    "bias_shape": [DIGITS],
                },
            }
        )
    if kind == MODEL_EVALUATION_ROW_KIND:
        report["evaluation_bundle"] = {
            "path": "target/evaluation-bundle.json",
            "fingerprints": valid_model_evaluation_fingerprint_block(),
        }
        report["row"] = valid_model_evaluation_rows_for_tests()[1]
        report["row_matches_recomputed_inference"] = True
    return report


def valid_model_evaluation_rows_for_tests() -> list[dict[str, Any]]:
    return [
        {
            "index": 0,
            "sample_fingerprint": "0" * 64,
            "label": 0,
            "prediction": 0,
            "correct": True,
            "runner_up_digit": 1,
            "margin": 100,
            "active_pixels": 2,
            "audit_step": 0,
            "source_sample_index": 0,
        },
        {
            "index": 1,
            "sample_fingerprint": "1" * 64,
            "label": 3,
            "prediction": 3,
            "correct": True,
            "runner_up_digit": 4,
            "margin": Q31_ONE // 2,
            "active_pixels": 2,
            "audit_step": 1,
            "source_sample_index": 1,
        },
    ]


def valid_model_evaluation_summary_for_tests(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(1 for row in rows if row["correct"])
    lowest = min(rows, key=lambda row: (row["margin"], row["index"]))
    return {
        "samples": len(rows),
        "correct": correct,
        "incorrect": len(rows) - correct,
        "accuracy_percent": 100.0 * correct / len(rows),
        "lowest_margin_index": lowest["index"],
        "lowest_margin": lowest["margin"],
        "elapsed_seconds": 0.002,
        "samples_per_second": 1000.0,
        "forward_elapsed_seconds": 0.001,
        "reverse_elapsed_seconds": 0.001,
    }


def valid_batch_inference_proof(samples: int = 2) -> dict[str, Any]:
    return valid_inference_memory(expected_steps=samples)


def valid_model_evaluation_report() -> dict[str, Any]:
    rows = valid_model_evaluation_rows_for_tests()
    return {
        "kind": MODEL_EVALUATION_KIND,
        "model_bundle": {
            "path": "target/model-bundle.json",
            "fingerprints": valid_model_fingerprint_block(),
        },
        "samples_json": {"path": "target/samples.json", "fingerprint": "0" * 64},
        "evaluation_output": "target/evaluation-bundle.json",
        "fingerprints": valid_model_evaluation_fingerprint_block(),
        "summary": valid_model_evaluation_summary_for_tests(rows),
        "proof": valid_batch_inference_proof(len(rows)),
        "rows": rows,
        "gate": {
            "passed": True,
            "checks": [
                {
                    "metric": "accuracy_percent",
                    "passed": True,
                    "actual": "100.00",
                    "expectation": ">= 100.00",
                }
            ],
        },
        "gate_policy": {"min_accuracy": 100.0},
    }


def valid_model_evaluation_scan_report() -> dict[str, Any]:
    rows = valid_model_evaluation_rows_for_tests()
    by_label = []
    for label in range(DIGITS):
        matching = [row for row in rows if row["label"] == label]
        if matching:
            lowest = min(matching, key=lambda row: (row["margin"], row["index"]))
            correct = sum(1 for row in matching if row["correct"])
            by_label.append(
                {
                    "label": label,
                    "samples": len(matching),
                    "correct": correct,
                    "incorrect": len(matching) - correct,
                    "accuracy_percent": 100.0 * correct / len(matching),
                    "lowest_margin_index": lowest["index"],
                    "lowest_margin": lowest["margin"],
                }
            )
        else:
            by_label.append(
                {
                    "label": label,
                    "samples": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "accuracy_percent": 0.0,
                    "lowest_margin_index": None,
                    "lowest_margin": None,
                }
            )
    return {
        "kind": MODEL_EVALUATION_SCAN_KIND,
        "path": "target/evaluation-bundle.json",
        "fingerprints": valid_model_evaluation_fingerprint_block(),
        "summary": {
            key: value
            for key, value in valid_model_evaluation_summary_for_tests(rows).items()
            if key
            not in (
                "elapsed_seconds",
                "samples_per_second",
                "forward_elapsed_seconds",
                "reverse_elapsed_seconds",
            )
        },
        "by_label": by_label,
        "top_confusions": [],
        "top_incorrect": [],
        "top_low_margin": [rows[0], rows[1]],
    }


def valid_model_evaluation_verification_report() -> dict[str, Any]:
    rows = valid_model_evaluation_rows_for_tests()
    return {
        "kind": MODEL_EVALUATION_VERIFICATION_KIND,
        "path": "target/evaluation-bundle.json",
        "fingerprints": valid_model_evaluation_fingerprint_block(),
        "summary": valid_model_evaluation_summary_for_tests(rows),
        "rows_match": True,
        "proof_matches": True,
        "source_model_checked": True,
        "source_samples_checked": True,
        "source_model": {
            "path": "target/model-bundle.json",
            "resolved_path": "target/model-bundle.json",
            "checked": True,
            "payload_fingerprint": "0" * 64,
            "items_checked": 1,
        },
        "source_samples_json": {
            "path": "target/samples.json",
            "resolved_path": "target/samples.json",
            "checked": True,
            "payload_fingerprint": "0" * 64,
            "items_checked": 2,
        },
        "restored_initial_state": True,
        "elapsed_seconds": 0.001,
    }


def valid_inference_verification_report() -> dict[str, Any]:
    native = valid_native_inference_report()
    return {
        "kind": INFERENCE_VERIFICATION_KIND,
        "path": "target/inference-bundle.json",
        "fingerprints": valid_inference_fingerprint_block(),
        "prediction": native["prediction"],
        "correct": native["correct"],
        "result_matches": True,
        "restored_initial_state": True,
        "proof_matches": True,
        "source_evaluation_checked": False,
        "source_model_evaluation": None,
        "source_model_checked": True,
        "source_training_checked": False,
        "source_sample_checked": True,
        "source_model": {
            "path": "target/model-bundle.json",
            "resolved_path": "target/model-bundle.json",
            "checked": True,
            "payload_fingerprint": "0" * 64,
            "items_checked": 1,
        },
        "source_training_bundle": None,
        "source_sample": {
            "path": "target/sample.json",
            "resolved_path": "target/sample.json",
            "checked": True,
            "payload_fingerprint": "0" * 64,
            "items_checked": 1,
        },
        "elapsed_seconds": 0.001,
        "attribution": native["attribution"],
        "memory": native["memory"],
        "proof": native["proof"],
        "explanation_contract": valid_inference_explanation_contract(verification=True),
    }


def valid_pipeline_gate_policy_for_tests() -> dict[str, Any]:
    return {
        "min_train_accuracy_percent": 90.0,
        "min_eval_accuracy_percent": 90.0,
        "min_audit_accuracy_percent": 90.0,
        "min_model_evaluation_accuracy_percent": 90.0,
        "min_reference_accuracy_percent": 90.0,
        "max_witness_mismatches": 0,
        "max_trace_to_model_payload_ratio": 1.0,
        "max_witness_to_model_payload_ratio": 0.25,
        "max_reverse_train_elapsed_ratio": 5.0,
        "max_replay_payload_bytes": None,
        "audit_step_strategy": "explicit",
        "requested_audit_step": 0,
        "evaluation_row_strategy": "explicit",
        "requested_evaluation_row": 1,
    }


def fake_evidence_record(path: str, role: str) -> dict[str, Any]:
    return {
        "path": path,
        "role": role,
        "bytes": 1,
        "sha256": "0" * 64,
    }


def valid_pipeline_evidence_for_tests(*, include_summary: bool) -> dict[str, Any]:
    files = {
        "profile_markdown": fake_evidence_record("target/mnist-ml-audit-profile.md", "markdown"),
        **{
            f"report.{key}": fake_evidence_record(f"target/{key}.json", "report")
            for key in PIPELINE_REPORT_KEYS
        },
        **{
            f"bundle.{key}": fake_evidence_record(f"target/{key}.json", "bundle")
            for key in PIPELINE_BUNDLE_KEYS
        },
        **{
            f"standalone.{key}": fake_evidence_record(f"target/{key}.json", "standalone")
            for key in PIPELINE_STANDALONE_KEYS
        },
    }
    if include_summary:
        files["summary"] = fake_evidence_record("target/pipeline-summary.json", "summary")
    return {"algorithm": "sha256", "files": files}


def valid_pipeline_claims_for_tests() -> dict[str, Any]:
    rows = [
        (
            "debug_training_update",
            [
                "training_audit_steps",
                "training_audit_witness_mismatches",
                "training_step_selection_traceable",
                "training_step_replay",
                "training_step_debug_contract",
                "training_update_ledger_fingerprints",
            ],
            ["report.audit_scan", "report.audit_step", "report.step_verification", "bundle.training_audit", "bundle.training_step"],
            ["training_audit_scan", "training_step_selection", "training_step_replay", "training_step_debug"],
        ),
        (
            "auditable_model_lineage",
            [
                "reverse_restored_initial_model",
                "reverse_checked_all_training_steps",
                "training_audit_lineage_replay",
                "audit_contract",
            ],
            [
                "report.run_report",
                "report.audit_verification",
                "report.artifact_comparison",
                "bundle.training_audit",
                "bundle.model",
                "bundle.samples",
            ],
            ["reverse", "training_audit_replay", "artifact_profile"],
        ),
        (
            "deterministic_q31_inference",
            [
                "model_import_provenance",
                "imported_model_inference_replay",
                "native_inference_replay",
                "native_inference_explanation_contract",
                "native_standalone_rev_classifier",
                "model_evaluation_replay",
                "evaluation_row_selection_traceable",
                "evaluation_row_inference_replay",
                "evaluation_row_inference_explanation_contract",
                "q31_reference_matches_native_inference",
                "q31_reference_matches_native_evaluation",
                "inference_trace_profile_complete",
                "reversible_inference_trace_replay",
            ],
            [
                "report.imported_model_import",
                "report.imported_model_verification",
                "report.imported_model_inference",
                "report.imported_model_inference_verification",
                "report.native_inference",
                "report.native_inference_verification",
                "standalone.native_standalone_rev_classifier",
                "standalone.native_standalone_rev_run",
                "standalone.native_standalone_rev_roundtrip",
                "standalone.native_standalone_rev_roundtrip_verification",
                "report.model_evaluation",
                "report.model_evaluation_verification",
                "report.row_inference_verification",
                "report.reversible_inference_trace",
                "report.q31_reference_inference",
                "report.q31_reference_evaluation",
                "bundle.imported_model_source",
                "bundle.imported_model",
                "bundle.imported_model_inference",
                "bundle.native_inference",
                "bundle.model_evaluation",
                "bundle.evaluation_row_inference",
                "bundle.inference_trace_forward",
                "bundle.inference_trace_reverse",
            ],
            [
                "model_import",
                "imported_model_inference",
                "native_inference_replay",
                "native_inference_explanation",
                "native_standalone_rev_classifier",
                "model_evaluation",
                "model_evaluation_replay",
                "evaluation_row_selection",
                "evaluation_row_inference",
                "evaluation_row_inference_explanation",
                "q31_reference_inference",
                "q31_reference_evaluation",
                "inference_trace_profile",
                "reversible_inference_trace",
            ],
        ),
        (
            "memory_recompute_profile",
            [
                "run_peak_rss_bytes",
                "trace_to_model_payload_ratio",
                "witness_to_model_payload_ratio",
                "balanced_recompute_steps",
                "recompute_frontier_complete",
                "scaling_projection_complete",
                "reverse_check_cost_measured",
                "reverse_check_elapsed_ratio",
            ],
            ["report.run_report", "report.artifact_comparison", "profile_markdown"],
            [
                "train",
                "built_in_eval",
                "memory",
                "artifact_profile",
                "recompute_frontier",
                "scaling_projection",
                "reverse_check_cost",
                "replay_payload_bytes",
            ],
        ),
        (
            "mlp_activation_mask_witnesses",
            ["mlp_witness_replay"],
            ["report.mlp_witness", "bundle.mlp_vars", "bundle.mlp_run_output"],
            ["mlp_witness"],
        ),
        (
            "invertible_layer_without_witness",
            [
                "invertible_coupling_replay",
                "triangular_residual_replay",
                "reversible_preprocess_replay",
            ],
            [
                "report.invertible_coupling",
                "report.triangular_residual",
                "report.reversible_preprocess",
                "bundle.coupling_final_vars",
                "bundle.coupling_forward",
                "bundle.coupling_reverse",
                "bundle.residual_final_vars",
                "bundle.residual_forward",
                "bundle.residual_reverse",
                "bundle.preprocess_final_vars",
                "bundle.preprocess_forward",
                "bundle.preprocess_reverse",
            ],
            ["invertible_coupling", "triangular_residual", "reversible_preprocess"],
        ),
        (
            "artifact_evidence_integrity",
            [],
            [
                "profile_markdown",
                *[f"report.{key}" for key in PIPELINE_REPORT_KEYS],
                *[f"bundle.{key}" for key in PIPELINE_BUNDLE_KEYS],
                *[f"standalone.{key}" for key in PIPELINE_STANDALONE_KEYS],
            ],
            [],
        ),
        (
            "v6_ml_audit_scorecard",
            ["v6_scorecard_complete"],
            [
                "profile_markdown",
                "report.run_report",
                "report.artifact_comparison",
                "report.native_inference_verification",
                "report.row_inference_verification",
            ],
            ["scorecard"],
        ),
        (
            "ml_roadmap_capabilities",
            ["ml_roadmap_capability_map_complete"],
            [
                "profile_markdown",
                "report.run_report",
                "report.audit_scan",
                "report.audit_step",
                "report.model_evaluation",
                "report.mlp_witness",
                "report.invertible_coupling",
                "report.triangular_residual",
                "report.reversible_preprocess",
                "report.artifact_comparison",
            ],
            ["ml_capability_map"],
        ),
        (
            "north_star_reversible_ml_kernels",
            ["ml_goal_readiness_complete"],
            [
                "profile_markdown",
                "report.audit_step",
                "report.step_verification",
                "report.native_inference_verification",
                "report.row_inference_verification",
                "report.mlp_witness",
                "report.invertible_coupling",
                "report.triangular_residual",
                "report.reversible_preprocess",
                "report.artifact_comparison",
            ],
            ["ml_goal_readiness"],
        ),
    ]
    return {
        "kind": PIPELINE_CLAIMS_KIND,
        "passed": True,
        "checks": [
            {
                "claim": claim,
                "description": "self-test",
                "gate_metrics": gate_metrics,
                "evidence": evidence,
                "metrics": metrics,
                "passed": True,
            }
            for claim, gate_metrics, evidence, metrics in rows
        ],
    }


def valid_pipeline_gates_for_tests() -> dict[str, Any]:
    checks = [
        {
            "metric": metric,
            "actual": "self-test",
            "requirement": "self-test",
            "passed": True,
        }
        for metric in PIPELINE_GATE_METRICS
    ]
    for check in checks:
        if check["metric"] == "reverse_check_elapsed_ratio":
            check["actual"] = 1.0
            check["requirement"] = "<= 5"
    return {
        "passed": True,
        "policy": valid_pipeline_gate_policy_for_tests(),
        "checks": checks,
    }


def valid_pipeline_capability_map_for_tests() -> dict[str, Any]:
    rows = [
        (
            "V1",
            "reversible_linear_mnist",
            "reversible linear MNIST",
            (
                "train_accuracy_percent",
                "reverse_restored_initial_model",
                "reverse_checked_all_training_steps",
                "training_audit_lineage_replay",
                "training_step_replay",
                "model_import_provenance",
                "imported_model_inference_replay",
                "native_inference_replay",
                "inference_trace_profile_complete",
                "reversible_inference_trace_replay",
            ),
            (
                "report.run_report",
                "report.audit_verification",
                "report.step_verification",
                "report.imported_model_verification",
                "report.imported_model_inference_verification",
                "report.native_inference_verification",
                "report.q31_reference_inference",
                "report.reversible_inference_trace",
                "bundle.training_audit",
                "bundle.training_step",
                "bundle.imported_model_source",
                "bundle.imported_model",
                "bundle.imported_model_inference",
                "bundle.native_inference",
                "bundle.inference_trace_forward",
                "bundle.inference_trace_reverse",
            ),
            (
                "train",
                "reverse",
                "training_audit_replay",
                "training_step_replay",
                "model_import",
                "imported_model_inference",
                "native_inference_replay",
                "inference_trace_profile",
                "reversible_inference_trace",
            ),
        ),
        (
            "V2",
            "witness_tapes",
            "first-class witness tapes",
            (
                "training_audit_witness_mismatches",
                "training_step_selection_traceable",
                "training_step_debug_contract",
                "training_update_ledger_fingerprints",
            ),
            (
                "report.audit_scan",
                "report.audit_step",
                "bundle.training_audit",
                "bundle.training_step",
            ),
            ("training_audit_scan", "training_step_selection", "training_step_debug"),
        ),
        (
            "V3",
            "batched_tensor_iteration",
            "batched tensor dataset iteration",
            (
                "model_evaluation_samples",
                "model_evaluation_replay",
                "q31_reference_samples",
                "q31_reference_matches_native_evaluation",
                "evaluation_row_selection_traceable",
            ),
            (
                "report.model_evaluation",
                "report.model_evaluation_verification",
                "report.model_evaluation_scan",
                "report.q31_reference_evaluation",
                "bundle.model_evaluation",
            ),
            (
                "model_evaluation",
                "model_evaluation_replay",
                "evaluation_row_selection",
                "q31_reference_evaluation",
            ),
        ),
        (
            "V4",
            "reversible_mlp_witnesses",
            "reversible MLP activation and mask witnesses",
            ("mlp_witness_replay",),
            ("report.mlp_witness", "bundle.mlp_vars", "bundle.mlp_run_output"),
            ("mlp_witness",),
        ),
        (
            "V5",
            "invertible_layer_pattern",
            "invertible layer pattern",
            (
                "invertible_coupling_replay",
                "triangular_residual_replay",
                "reversible_preprocess_replay",
            ),
            (
                "report.invertible_coupling",
                "report.triangular_residual",
                "report.reversible_preprocess",
                "bundle.coupling_final_vars",
                "bundle.coupling_forward",
                "bundle.coupling_reverse",
                "bundle.residual_final_vars",
                "bundle.residual_forward",
                "bundle.residual_reverse",
                "bundle.preprocess_final_vars",
                "bundle.preprocess_forward",
                "bundle.preprocess_reverse",
            ),
            ("invertible_coupling", "triangular_residual", "reversible_preprocess"),
        ),
        (
            "V6",
            "speed_memory_trace_reverse_scorecard",
            "speed, memory, trace, and reverse-cost scorecard",
            (
                "run_peak_rss_bytes",
                "balanced_recompute_steps",
                "recompute_frontier_complete",
                "scaling_projection_complete",
                "reverse_check_cost_measured",
                "reverse_check_elapsed_ratio",
                "v6_scorecard_complete",
            ),
            (
                "profile_markdown",
                "report.run_report",
                "report.artifact_comparison",
                "report.native_inference_verification",
                "report.row_inference_verification",
            ),
            (
                "memory",
                "artifact_profile",
                "reverse_check_cost",
                "recompute_frontier",
                "scaling_projection",
                "scorecard",
            ),
        ),
    ]
    return {
        "kind": ML_CAPABILITY_KIND,
        "north_star": "reversible_inspectable_deterministic_ml_kernels",
        "passed": True,
        "summary": {
            "total": len(rows),
            "passed": len(rows),
            "failed": 0,
            "failed_capabilities": [],
        },
        "capabilities": [
            {
                "phase": phase,
                "id": capability_id,
                "goal": goal,
                "description": "self-test",
                "status": "passed",
                "passed": True,
                "gate_metrics": list(gate_metrics),
                "blocking_gates": [],
                "evidence": list(evidence),
                "metrics": list(metrics),
            }
            for phase, capability_id, goal, gate_metrics, evidence, metrics in rows
        ],
    }


def valid_pipeline_goal_readiness_for_tests() -> dict[str, Any]:
    rows = [
        (
            "debug_training",
            "Debug training updates backward",
            (
                "training_step_selection_traceable",
                "training_step_replay",
                "training_step_debug_contract",
                "training_update_ledger_fingerprints",
            ),
            (
                "report.audit_scan",
                "report.audit_step",
                "report.step_verification",
                "bundle.training_audit",
                "bundle.training_step",
            ),
            (
                "training_step_selection",
                "training_step_replay",
                "training_step_debug",
            ),
        ),
        (
            "audit_model_lineage",
            "Prove auditable model lineage",
            (
                "reverse_restored_initial_model",
                "reverse_checked_all_training_steps",
                "training_audit_lineage_replay",
                "audit_contract",
                "ml_roadmap_capability_map_complete",
            ),
            (
                "report.run_report",
                "report.audit_verification",
                "report.artifact_comparison",
                "bundle.training_audit",
                "bundle.model",
                "bundle.samples",
            ),
            (
                "reverse",
                "training_audit_replay",
                "artifact_profile",
                "ml_capability_map",
            ),
        ),
        (
            "deterministic_inference",
            "Run deterministic reversible inference traces",
            (
                "native_inference_replay",
                "native_inference_explanation_contract",
                "native_standalone_rev_classifier",
                "model_import_provenance",
                "imported_model_inference_replay",
                "evaluation_row_inference_replay",
                "evaluation_row_inference_explanation_contract",
                "q31_reference_matches_native_inference",
                "q31_reference_matches_native_evaluation",
                "inference_trace_profile_complete",
                "reversible_inference_trace_replay",
            ),
            (
                "report.native_inference_verification",
                "standalone.native_standalone_rev_classifier",
                "standalone.native_standalone_rev_run",
                "standalone.native_standalone_rev_roundtrip",
                "standalone.native_standalone_rev_roundtrip_verification",
                "report.imported_model_verification",
                "report.imported_model_inference_verification",
                "report.row_inference_verification",
                "report.q31_reference_inference",
                "report.q31_reference_evaluation",
                "report.reversible_inference_trace",
                "bundle.native_inference",
                "bundle.imported_model_source",
                "bundle.imported_model",
                "bundle.imported_model_inference",
                "bundle.evaluation_row_inference",
                "bundle.inference_trace_forward",
                "bundle.inference_trace_reverse",
            ),
            (
                "native_inference_replay",
                "native_inference_explanation",
                "native_standalone_rev_classifier",
                "model_import",
                "imported_model_inference",
                "evaluation_row_inference",
                "evaluation_row_inference_explanation",
                "q31_reference_inference",
                "q31_reference_evaluation",
                "inference_trace_profile",
                "reversible_inference_trace",
            ),
        ),
        (
            "memory_recompute_tradeoffs",
            "Measure memory, trace, witness, and recompute tradeoffs",
            (
                "run_peak_rss_bytes",
                "balanced_recompute_steps",
                "recompute_frontier_complete",
                "scaling_projection_complete",
                "reverse_check_cost_measured",
                "reverse_check_elapsed_ratio",
                "v6_scorecard_complete",
            ),
            (
                "profile_markdown",
                "report.run_report",
                "report.artifact_comparison",
            ),
            (
                "memory",
                "recompute_frontier",
                "scaling_projection",
                "reverse_check_cost",
                "scorecard",
            ),
        ),
        (
            "invertible_model_patterns",
            "Show reversible and invertible model families",
            (
                "mlp_witness_replay",
                "invertible_coupling_replay",
                "triangular_residual_replay",
                "reversible_preprocess_replay",
            ),
            (
                "report.mlp_witness",
                "report.invertible_coupling",
                "report.triangular_residual",
                "report.reversible_preprocess",
                "bundle.mlp_vars",
                "bundle.coupling_forward",
                "bundle.coupling_reverse",
                "bundle.residual_forward",
                "bundle.residual_reverse",
                "bundle.preprocess_forward",
                "bundle.preprocess_reverse",
            ),
            (
                "mlp_witness",
                "invertible_coupling",
                "triangular_residual",
                "reversible_preprocess",
            ),
        ),
    ]
    return {
        "kind": ML_GOAL_READINESS_KIND,
        "north_star": "best_small_language_for_reversible_inspectable_deterministic_ml_kernels",
        "non_goal": "general_purpose_pytorch_tensorflow_training_replacement",
        "passed": True,
        "summary": {
            "total": len(rows),
            "passed": len(rows),
            "failed": 0,
            "failed_goals": [],
        },
        "goals": [
            {
                "id": readiness_id,
                "goal": goal,
                "description": "self-test",
                "status": "passed",
                "passed": True,
                "gate_metrics": list(gate_metrics),
                "blocking_gates": [],
                "evidence": list(evidence),
                "metrics": list(metrics),
            }
            for readiness_id, goal, gate_metrics, evidence, metrics in rows
        ],
    }


def valid_pipeline_recompute_frontier_for_tests(
    replay_payloads: dict[str, int],
    mlp_proof: dict[str, Any],
    coupling_proof: dict[str, Any],
    residual_proof: dict[str, Any],
    preprocess_proof: dict[str, Any],
    inference_trace_proof: dict[str, Any],
) -> dict[str, Any]:
    def ratio(numerator: int, denominator: int) -> Optional[float]:
        if denominator == 0:
            return None
        return numerator / denominator

    def row(
        row_id: str,
        label: str,
        scope: str,
        kept_state: str,
        reversible_transition: str,
        replay_payload_bytes: int,
        model_payload_bytes: int,
        sample_payload_bytes: int,
        witness_payload_bytes: int,
        trace_payload_bytes: int,
        derived_update_payload_bytes: int,
        recomputed_update_payload_bytes: int,
        state_payload_bytes: int,
        forward_recompute_steps: int,
        inverse_recompute_steps: int,
    ) -> dict[str, Any]:
        return {
            "id": row_id,
            "label": label,
            "scope": scope,
            "kept_state": kept_state,
            "reversible_transition": reversible_transition,
            "replay_payload_bytes": replay_payload_bytes,
            "model_payload_bytes": model_payload_bytes,
            "sample_payload_bytes": sample_payload_bytes,
            "witness_payload_bytes": witness_payload_bytes,
            "trace_payload_bytes": trace_payload_bytes,
            "derived_update_payload_bytes": derived_update_payload_bytes,
            "recomputed_update_payload_bytes": recomputed_update_payload_bytes,
            "state_payload_bytes": state_payload_bytes,
            "forward_recompute_steps": forward_recompute_steps,
            "inverse_recompute_steps": inverse_recompute_steps,
            "total_recompute_steps": forward_recompute_steps + inverse_recompute_steps,
            "balanced_recompute": forward_recompute_steps == inverse_recompute_steps
            and forward_recompute_steps > 0,
            "zero_witness": witness_payload_bytes == 0 and trace_payload_bytes == 0,
            "witness_backed": witness_payload_bytes > 0,
            "bytes_per_inverse_step": ratio(replay_payload_bytes, inverse_recompute_steps),
            "witness_to_model_payload_ratio": ratio(witness_payload_bytes, model_payload_bytes),
            "trace_to_model_payload_ratio": ratio(trace_payload_bytes, model_payload_bytes),
        }

    rows = [
        row(
            "training_trace",
            "Training trace replay",
            "all training updates",
            "final model plus per-sample logits/error/prediction witnesses",
            "final model + witness trace -> initial model",
            replay_payloads["training_trace"],
            62800,
            1570,
            398,
            1968,
            0,
            0,
            0,
            2,
            2,
        ),
        row(
            "training_step_debug",
            "Selected training-step replay",
            "one model update",
            "before/after model window, sample, witnesses, and derived update summary",
            "after model + sample + witnesses -> before model",
            replay_payloads["training_step"],
            125600,
            785,
            184,
            0,
            216,
            216,
            0,
            1,
            1,
        ),
        row(
            "imported_model_inference",
            "Imported model inference replay",
            "one external-import model sample",
            "imported model, image, logits, prediction, and correctness witnesses",
            "imported inference result -> zeroed inference state",
            replay_payloads["imported_model_inference"],
            62800,
            785,
            96,
            0,
            0,
            0,
            0,
            1,
            1,
        ),
        row(
            "native_inference",
            "Single native inference replay",
            "one signed sample",
            "model, image, logits, prediction, and correctness witnesses",
            "inference result -> zeroed inference state",
            replay_payloads["native_inference"],
            62800,
            785,
            96,
            0,
            0,
            0,
            0,
            1,
            1,
        ),
        row(
            "model_evaluation_batch",
            "Batch model-evaluation replay",
            "signed sample batch",
            "model, sample batch, logits/prediction/correctness witnesses",
            "batch evaluation rows -> zeroed evaluation state",
            replay_payloads["model_evaluation"],
            62800,
            7850,
            960,
            0,
            0,
            0,
            0,
            2,
            2,
        ),
        row(
            "evaluation_row_inference",
            "Evaluation-row inference replay",
            "one scan-selected evaluation row",
            "signed model evaluation plus row inference witnesses",
            "row prediction explanation -> zeroed inference state",
            replay_payloads["evaluation_row_inference"],
            62800,
            785,
            96,
            0,
            0,
            0,
            0,
            1,
            1,
        ),
        row(
            "mlp_witness_trace",
            "MLP witness trace replay",
            "two-sample MLP update",
            "model plus hidden preactivation/ReLU/error/backprop witness tapes",
            "final MLP weights + witness tapes -> prior weights",
            mlp_proof["replay_payload_bytes"],
            mlp_proof["model_payload_bytes"],
            mlp_proof["sample_payload_bytes"],
            mlp_proof["witness_payload_bytes"],
            mlp_proof["trace_payload_bytes"],
            mlp_proof["stored_derived_update_payload_bytes"],
            mlp_proof["recomputed_update_payload_bytes"],
            0,
            mlp_proof["forward_recompute_steps"],
            mlp_proof["inverse_recompute_steps"],
        ),
        row(
            "invertible_coupling",
            "Invertible coupling replay",
            "one additive coupling layer",
            "activation halves only",
            "coupled activations -> original activations",
            coupling_proof["replay_payload_bytes"],
            coupling_proof["model_payload_bytes"],
            0,
            coupling_proof["witness_payload_bytes"],
            coupling_proof["trace_payload_bytes"],
            0,
            0,
            coupling_proof["state_payload_bytes"],
            coupling_proof["forward_recompute_steps"],
            coupling_proof["inverse_recompute_steps"],
        ),
        row(
            "triangular_residual",
            "Triangular residual replay",
            "one triangular residual layer",
            "activation vector and residual parameters",
            "residual activations -> original activations",
            residual_proof["replay_payload_bytes"],
            residual_proof["parameter_payload_bytes"],
            0,
            residual_proof["witness_payload_bytes"],
            residual_proof["trace_payload_bytes"],
            0,
            0,
            residual_proof["state_payload_bytes"],
            residual_proof["forward_recompute_steps"],
            residual_proof["inverse_recompute_steps"],
        ),
        row(
            "reversible_preprocess",
            "Reversible preprocessing replay",
            "one centered/permuted feature block",
            "raw sample, mean vector, and feature buffer",
            "preprocessed features -> zeroed feature buffer",
            preprocess_proof["replay_payload_bytes"],
            preprocess_proof["mean_payload_bytes"],
            preprocess_proof["raw_payload_bytes"],
            preprocess_proof["witness_payload_bytes"],
            preprocess_proof["trace_payload_bytes"],
            0,
            0,
            preprocess_proof["feature_payload_bytes"],
            preprocess_proof["forward_recompute_steps"],
            preprocess_proof["inverse_recompute_steps"],
        ),
        row(
            "reversible_inference_trace",
            "Reversible inference trace replay",
            "one preprocessed Q31 classifier trace",
            "raw sample, mean vector, model, features, logits, prediction, and correctness",
            "prediction trace -> zeroed feature/logit/prediction state",
            inference_trace_proof["replay_payload_bytes"],
            inference_trace_proof["model_payload_bytes"],
            0,
            inference_trace_proof["witness_payload_bytes"],
            inference_trace_proof["trace_payload_bytes"],
            0,
            0,
            inference_trace_proof["state_payload_bytes"],
            inference_trace_proof["forward_recompute_steps"],
            inference_trace_proof["inverse_recompute_steps"],
        ),
    ]
    replay_values = [item["replay_payload_bytes"] for item in rows]
    largest = max(rows, key=lambda item: item["replay_payload_bytes"])
    smallest = min(rows, key=lambda item: item["replay_payload_bytes"])
    return {
        "kind": RECOMPUTE_FRONTIER_KIND,
        "rows": rows,
        "summary": {
            "rows": len(rows),
            "total_replay_payload_bytes": sum(replay_values),
            "max_replay_payload_bytes": largest["replay_payload_bytes"],
            "max_replay_id": largest["id"],
            "min_replay_payload_bytes": smallest["replay_payload_bytes"],
            "min_replay_id": smallest["id"],
            "total_forward_recompute_steps": sum(item["forward_recompute_steps"] for item in rows),
            "total_inverse_recompute_steps": sum(item["inverse_recompute_steps"] for item in rows),
            "zero_witness_rows": [
                item["id"]
                for item in rows
                if item["zero_witness"]
            ],
            "witness_backed_rows": [
                item["id"]
                for item in rows
                if item["witness_backed"]
            ],
        },
    }


def valid_pipeline_scaling_projection_for_tests(
    recompute_frontier: dict[str, Any],
    *,
    train_updates: int,
    model_evaluation_samples: int,
    mlp_samples: int,
) -> dict[str, Any]:
    def ratio(numerator: int, denominator: int) -> Optional[float]:
        if denominator == 0:
            return None
        return numerator / denominator

    frontier_by_id = {
        row["id"]: row
        for row in recompute_frontier["rows"]
    }
    observed_counts = {
        "training_trace": train_updates,
        "model_evaluation_batch": model_evaluation_samples,
        "mlp_witness_trace": mlp_samples,
    }
    labels = {
        "training_trace": ("Training trace replay", "training update"),
        "model_evaluation_batch": ("Batch model-evaluation replay", "evaluated sample"),
        "mlp_witness_trace": ("MLP witness trace replay", "MLP sample update"),
    }

    def per_unit(row: dict[str, Any], field: str, count: int) -> int:
        value = int(row[field])
        if value % count != 0:
            raise AssertionError(f"{field} does not divide evenly")
        return value // count

    def family(family_id: str) -> dict[str, Any]:
        row = frontier_by_id[family_id]
        count = observed_counts[family_id]
        sample_per = per_unit(row, "sample_payload_bytes", count)
        witness_per = per_unit(row, "witness_payload_bytes", count)
        trace_per = per_unit(row, "trace_payload_bytes", count)
        recomputed_per = per_unit(row, "recomputed_update_payload_bytes", count)
        forward_per = per_unit(row, "forward_recompute_steps", count)
        inverse_per = per_unit(row, "inverse_recompute_steps", count)
        variable_per = (
            trace_per
            if family_id in ("training_trace", "mlp_witness_trace")
            else sample_per + witness_per + trace_per
        )
        projections = []
        for scale in SCALING_PROJECTION_SCALES:
            projected_count = count * scale
            replay = row["model_payload_bytes"] + projected_count * variable_per
            forward = projected_count * forward_per
            inverse = projected_count * inverse_per
            projections.append(
                {
                    "scale_factor": scale,
                    "count": projected_count,
                    "projected_replay_payload_bytes": replay,
                    "projected_sample_payload_bytes": projected_count * sample_per,
                    "projected_witness_payload_bytes": projected_count * witness_per,
                    "projected_trace_payload_bytes": projected_count * trace_per,
                    "projected_recomputed_update_payload_bytes": projected_count * recomputed_per,
                    "projected_forward_recompute_steps": forward,
                    "projected_inverse_recompute_steps": inverse,
                    "balanced_recompute": forward == inverse and forward > 0,
                    "projected_bytes_per_inverse_step": ratio(replay, inverse),
                }
            )
        label, unit = labels[family_id]
        return {
            "id": family_id,
            "label": label,
            "unit": unit,
            "observed_count": count,
            "model_payload_bytes": row["model_payload_bytes"],
            "sample_payload_bytes_per_unit": sample_per,
            "witness_payload_bytes_per_unit": witness_per,
            "trace_payload_bytes_per_unit": trace_per,
            "variable_replay_payload_bytes_per_unit": variable_per,
            "recomputed_update_payload_bytes_per_unit": recomputed_per,
            "forward_recompute_steps_per_unit": forward_per,
            "inverse_recompute_steps_per_unit": inverse_per,
            "observed_replay_payload_bytes": projections[0]["projected_replay_payload_bytes"],
            "projections": projections,
        }

    families = [family(family_id) for family_id in SCALING_PROJECTION_FAMILIES]
    all_projections = [
        (item, projection)
        for item in families
        for projection in item["projections"]
    ]
    largest = max(all_projections, key=lambda item: item[1]["projected_replay_payload_bytes"])
    hundred_x = [
        projection
        for _, projection in all_projections
        if projection["scale_factor"] == 100
    ]
    return {
        "kind": SCALING_PROJECTION_KIND,
        "families": families,
        "summary": {
            "families": len(families),
            "projection_scales": list(SCALING_PROJECTION_SCALES),
            "total_observed_replay_payload_bytes": sum(
                item["observed_replay_payload_bytes"] for item in families
            ),
            "total_projected_replay_payload_bytes_at_100x": sum(
                item["projected_replay_payload_bytes"] for item in hundred_x
            ),
            "max_projected_replay_payload_bytes": largest[1]["projected_replay_payload_bytes"],
            "max_projected_replay_family": largest[0]["id"],
            "max_projected_count": largest[1]["count"],
            "all_balanced": all(
                projection["balanced_recompute"]
                for _, projection in all_projections
            ),
        },
    }


def valid_pipeline_inference_ledger_for_tests() -> dict[str, Any]:
    return {
        "contribution_count": 1,
        "margin_contribution_count": 2,
        "contribution_ledger_fingerprint": "0" * 64,
        "margin_contribution_ledger_fingerprint": "1" * 64,
    }


def valid_pipeline_inference_trace_source_for_tests(role: str) -> dict[str, Any]:
    return {
        "role": role,
        "prediction": 3,
        "correct": True,
        "margin": Q31_ONE // 2,
        "active_pixels": 2,
        "ledger": valid_pipeline_inference_ledger_for_tests(),
        "explanation_passed": True,
        "replay_payload_bytes": 63681,
        "forward_recompute_steps": 1,
        "inverse_recompute_steps": 1,
    }


def valid_pipeline_inference_trace_for_tests(
    *,
    trace_id: str,
    label: str,
    source_kind: str,
    sample_index: int,
    row_index: Optional[int],
    reference_available: bool,
    reference_required: bool,
) -> dict[str, Any]:
    reference = {
        "available": reference_available,
        "role": "q31_reference",
        "prediction": 3 if reference_available else None,
        "correct": True if reference_available else None,
        "margin": Q31_ONE // 2 if reference_available else None,
        "active_pixels": 2 if reference_available else None,
        "ledger": valid_pipeline_inference_ledger_for_tests() if reference_available else None,
    }
    if reference_available:
        reference.update(
            {
                "explanation_passed": None,
                "replay_payload_bytes": None,
                "forward_recompute_steps": None,
                "inverse_recompute_steps": None,
            }
        )
    return {
        "id": trace_id,
        "label": label,
        "source_kind": source_kind,
        "sample_index": sample_index,
        "row_index": row_index,
        "prediction": 3,
        "correct": True,
        "margin": Q31_ONE // 2,
        "active_pixels": 2,
        "report": valid_pipeline_inference_trace_source_for_tests("native_report"),
        "verification": valid_pipeline_inference_trace_source_for_tests("replay_verification"),
        "reference": reference,
        "checks": {
            "report_verification_result_matches": True,
            "report_verification_ledgers_match": True,
            "report_explanation_passed": True,
            "verification_explanation_passed": True,
            "verification_replay_passed": True,
            "source_inputs_checked": True,
            "reference_required": reference_required,
            "reference_available": reference_available,
            "reference_result_matches_report": reference_available,
            "reference_ledgers_match_report": reference_available,
        },
        "passed": True,
    }


def valid_pipeline_inference_trace_profile_for_tests() -> dict[str, Any]:
    traces = [
        valid_pipeline_inference_trace_for_tests(
            trace_id="native_selected_sample",
            label="Selected audit-sample inference",
            source_kind="audit_bundle",
            sample_index=1,
            row_index=None,
            reference_available=True,
            reference_required=True,
        ),
        valid_pipeline_inference_trace_for_tests(
            trace_id="evaluation_row",
            label="Selected model-evaluation row inference",
            source_kind="model_evaluation_bundle",
            sample_index=1,
            row_index=1,
            reference_available=False,
            reference_required=False,
        ),
    ]
    return {
        "kind": INFERENCE_TRACE_PROFILE_KIND,
        "traces": traces,
        "summary": {
            "trace_count": len(traces),
            "reference_checked_traces": 1,
            "all_report_verification_results_match": True,
            "all_report_verification_ledgers_match": True,
            "all_required_references_match": True,
            "all_replay_verified": True,
            "all_explanations_passed": True,
            "all_sources_checked": True,
            "all_passed": True,
        },
    }


def valid_pipeline_summary_report() -> dict[str, Any]:
    model_bytes = 376800
    trace_bytes = 98400
    witness_bytes = 21240
    mlp_witness = valid_mlp_witness_report()
    mlp_proof = mlp_witness["proof"]
    coupling = valid_invertible_coupling_report()
    coupling_proof = coupling["proof"]
    residual = valid_triangular_residual_report()
    residual_proof = residual["proof"]
    preprocess = valid_reversible_preprocess_report()
    preprocess_proof = preprocess["proof"]
    inference_trace = valid_reversible_inference_trace_report()
    inference_trace_proof = inference_trace["proof"]
    replay_payloads = {
        "training_trace": 64768,
        "training_step": 126785,
        "imported_model_inference": 63681,
        "native_inference": 63681,
        "model_evaluation": 71610,
        "evaluation_row_inference": 63681,
    }
    recompute_frontier = valid_pipeline_recompute_frontier_for_tests(
        replay_payloads,
        mlp_proof,
        coupling_proof,
        residual_proof,
        preprocess_proof,
        inference_trace_proof,
    )
    scaling_projection = valid_pipeline_scaling_projection_for_tests(
        recompute_frontier,
        train_updates=2,
        model_evaluation_samples=2,
        mlp_samples=mlp_witness["samples"],
    )
    return {
        "kind": PIPELINE_SUMMARY_KIND,
        "pipeline_kind": PIPELINE_KIND,
        "sample_limit": 2,
        "requested_audit_step": 0,
        "audit_step": 0,
        "audit_step_strategy": "explicit",
        "requested_evaluation_row": 1,
        "evaluation_row": 1,
        "evaluation_row_strategy": "explicit",
        "profile_markdown": "target/mnist-ml-audit-profile.md",
        "reports": {key: f"target/{key}.json" for key in PIPELINE_REPORT_KEYS},
        "bundles": {key: f"target/{key}.json" for key in PIPELINE_BUNDLE_KEYS},
        "metrics": {
            "train": {
                "samples": 2,
                "correct": 2,
                "accuracy_percent": 100.0,
                "updates_per_second": 10.0,
            },
            "built_in_eval": {
                "samples": 1,
                "correct": 1,
                "accuracy_percent": 100.0,
                "samples_per_second": 10.0,
            },
            "reverse": {
                "enabled": True,
                "checked": 2,
                "restored_initial_model": True,
                "steps_per_second": 10.0,
            },
            "reverse_check_cost": {
                "enabled": True,
                "train_updates": 2,
                "checked_steps": 2,
                "train_elapsed_seconds": 0.2,
                "reverse_elapsed_seconds": 0.2,
                "train_seconds_per_update": 0.1,
                "reverse_seconds_per_step": 0.1,
                "reverse_to_train_elapsed_ratio": 1.0,
                "reverse_to_train_throughput_ratio": 1.0,
                "train_updates_per_second": 10.0,
                "reverse_steps_per_second": 10.0,
                "proof_forward_recompute_steps": 2,
                "proof_inverse_recompute_steps": 2,
                "proof_replay_payload_bytes": 64768,
                "trace_replay_payload_bytes": 1968,
            },
            "memory": {
                "run_peak_rss_bytes": 123456,
                "model_evaluation_peak_rss_bytes": 234567,
                "estimated_payload_bytes": 65552,
            },
            "training_audit_scan": {
                "steps": 2,
                "correct": 2,
                "incorrect": 0,
                "accuracy_percent": 100.0,
                "witness_mismatches": 0,
                "trace_payload_bytes": 400,
                "lowest_margin_step": 1,
                "lowest_margin": 100,
                "largest_update_step": 0,
                "max_abs_weight_delta": 100,
            },
            "training_step_selection": {
                "selection_strategy": "explicit",
                "selection_strategy_step": 0,
                "requested_step": 0,
                "selected_step": 0,
                "selected_sample_index": 0,
                "selected_margin": Q31_ONE // 2,
                "selected_max_abs_weight_delta": 100,
                "scan_lowest_margin_step": 1,
                "scan_lowest_margin": 100,
                "scan_largest_update_step": 0,
                "scan_max_abs_weight_delta": 100,
                "scan_top_suspicious_step": 0,
                "scan_top_low_margin_step": 1,
                "scan_top_large_updates_step": 0,
                "matches_selection_strategy": True,
                "matches_requested_step": True,
                "matches_scan_lowest_margin": False,
                "matches_scan_largest_update": True,
                "present_in_top_suspicious": True,
                "present_in_top_low_margin": True,
                "present_in_top_large_updates": True,
                "selection_reasons": [
                    "requested",
                    "largest_update",
                    "top_suspicious",
                    "top_low_margin",
                    "top_large_updates",
                ],
            },
            "artifact_profile": {
                "total_file_bytes": 1768437,
                "total_logical_payload_bytes": 493854,
                "total_model_payload_bytes": model_bytes,
                "total_witness_payload_bytes": witness_bytes,
                "total_trace_payload_bytes": trace_bytes,
                "total_recompute_steps": 4,
                "total_forward_recompute_steps": 2,
                "total_inverse_recompute_steps": 2,
                "trace_to_model_payload_ratio": trace_bytes / model_bytes,
                "witness_to_model_payload_ratio": witness_bytes / model_bytes,
            },
            "training_step_replay": {
                "proof_matches": True,
                "witnesses_match": True,
                "after_model_matches": True,
                "before_model_restored": True,
                "replay_payload_bytes": replay_payloads["training_step"],
            },
            "training_audit_replay": {
                "checked": 2,
                "witnesses_match_forward_replay": True,
                "final_model_replayed": True,
                "restored_initial_model": True,
                "proof_matches": True,
                "lineage_ledger_matches": True,
                "lineage_ledger_fingerprint": "9" * 64,
                "transition_ledger_fingerprint": "4" * 64,
                "final_chain": "8" * 64,
            },
            "training_step_debug": {
                "claim": "step_backward_from_model_update",
                "passed": True,
                "step": 0,
                "sample_index": 0,
                "prediction": 0,
                "correct": True,
                "margin": Q31_ONE // 2,
                "active_pixel_count": 2,
                "top_weight_deltas": 2,
                "nonzero_bias_delta_count": 1,
                "nonzero_weight_delta_count": 2,
                "bias_delta_ledger_fingerprint": "2" * 64,
                "weight_delta_ledger_fingerprint": "3" * 64,
                "cause_ledger_fingerprint": "4" * 64,
                "reversed_later_steps": 1,
                "checks": [
                    {
                        "metric": "witness_recomputes_prediction",
                        "passed": True,
                        "actual": "self-test",
                        "requirement": "self-test",
                    },
                    {
                        "metric": "model_window_reconstructed",
                        "passed": True,
                        "actual": "self-test",
                        "requirement": "self-test",
                    },
                    {
                        "metric": "update_matches_model_window",
                        "passed": True,
                        "actual": "self-test",
                        "requirement": "self-test",
                    },
                    {
                        "metric": "explanatory_state_present",
                        "passed": True,
                        "actual": "self-test",
                        "requirement": "self-test",
                    },
                    {
                        "metric": "update_ledger_fingerprints",
                        "passed": True,
                        "actual": "self-test",
                        "requirement": "self-test",
                    },
                ],
            },
            "model_import": {
                "source_model_json_path": "target/imported-q31-linear-model.json",
                "source_model_json_fingerprint": "a" * 64,
                "source_model_json_file_sha256": "b" * 64,
                "model_output": "target/imported-model-bundle.json",
                "model_payload_bytes": 62760,
                "import_shape_matches": True,
                "import_provenance_kind": "external_import",
                "import_payload_fingerprint": "0" * 64,
                "verification_shape_matches": True,
                "verification_provenance_matches": True,
                "verification_provenance_kind": "external_import",
                "verification_source_checked": True,
                "verification_payload_fingerprint": "0" * 64,
                "proof_matches": None,
                "training_steps": None,
                "source_model_json": valid_imported_model_source_check(),
            },
            "imported_model_inference": {
                "correct": True,
                "prediction": 3,
                "native_prediction": 3,
                "correct_matches_native": True,
                "prediction_matches_native": True,
                "margin_matches_native": True,
                "contribution_ledger_matches_native": True,
                "margin_contribution_ledger_matches_native": True,
                "proof_matches": True,
                "result_matches": True,
                "restored_initial_state": True,
                "source_model_checked": True,
                "source_sample_checked": True,
                "replay_payload_bytes": replay_payloads["imported_model_inference"],
                "forward_recompute_steps": 1,
                "inverse_recompute_steps": 1,
            },
            "imported_model_inference_explanation": (
                valid_inference_explanation_contract(verification=True)
            ),
            "native_inference_replay": {
                "correct": True,
                "proof_matches": True,
                "result_matches": True,
                "restored_initial_state": True,
                "source_model_checked": True,
                "source_sample_checked": True,
                "replay_payload_bytes": replay_payloads["native_inference"],
            },
            "native_inference_explanation": valid_inference_explanation_contract(verification=True),
            "native_standalone_rev_classifier": {
                "path": "target/mnist-standalone-classifier.rev",
                "bytes": 1,
                "sha256": "2" * 64,
                "line_count": 1,
                "contains_vecmat_q31": True,
                "contains_prediction_assert": True,
                "contains_correct_assert": True,
                "run_kind": "reverie_run_result",
                "run_prediction": 3,
                "run_correct": 1,
                "run_label": 3,
                "expected_prediction": 3,
                "expected_correct": 1,
                "run_uses_vecmat_q31": True,
                "matches_native_prediction": True,
                "matches_native_correct": True,
                "matches_native_label": True,
                "roundtrip_kind": "reverie_roundtrip_result",
                "roundtrip_passed": True,
                "roundtrip_prediction": 3,
                "roundtrip_correct": 1,
                "roundtrip_restored_prediction": 0,
                "roundtrip_restored_correct": 0,
                "roundtrip_fingerprint": "3" * 64,
                "verification_kind": "reverie_roundtrip_verification",
                "verification_passed": True,
                "verification_source_hash_matches": True,
                "verification_replay_restoration_passed": True,
                "verification_replayed": True,
            },
            "model_evaluation": {
                "samples": 2,
                "correct": 2,
                "incorrect": 0,
                "accuracy_percent": 100.0,
                "samples_per_second": 10.0,
                "lowest_margin": 0,
                "replay_payload_bytes": replay_payloads["model_evaluation"],
            },
            "model_evaluation_replay": {
                "rows_match": True,
                "proof_matches": True,
                "restored_initial_state": True,
                "source_model_checked": True,
                "source_samples_checked": True,
            },
            "evaluation_row_selection": {
                "selection_strategy": "explicit",
                "selection_strategy_row": 1,
                "requested_row": 1,
                "selected_row": 1,
                "selected_source_sample_index": 1,
                "selected_prediction": 3,
                "selected_correct": True,
                "selected_margin": Q31_ONE // 2,
                "scan_lowest_margin_row": 0,
                "scan_lowest_margin": 0,
                "scan_top_low_margin_row": 0,
                "scan_top_incorrect_row": None,
                "scan_top_low_margin_rows": [0, 1],
                "scan_top_incorrect_rows": [],
                "matches_selection_strategy": True,
                "matches_requested_row": True,
                "matches_scan_lowest_margin": False,
                "present_in_top_low_margin": True,
                "present_in_top_incorrect": False,
                "selection_reasons": ["requested", "top_low_margin"],
            },
            "evaluation_row_inference": {
                "correct": True,
                "proof_matches": True,
                "result_matches": True,
                "restored_initial_state": True,
                "source_model_checked": True,
                "source_evaluation_checked": True,
                "replay_payload_bytes": replay_payloads["evaluation_row_inference"],
            },
            "evaluation_row_inference_explanation": valid_inference_explanation_contract(verification=True),
            "q31_reference_inference": {
                "prediction": 3,
                "correct": True,
                "margin": Q31_ONE // 2,
                "active_pixels": 2,
                "contribution_count": 1,
                "margin_contribution_count": 2,
                "contribution_ledger_fingerprint": "0" * 64,
                "margin_contribution_ledger_fingerprint": "1" * 64,
                "native_prediction": 3,
                "native_correct": True,
                "native_margin": Q31_ONE // 2,
                "native_active_pixels": 2,
                "native_contribution_count": 1,
                "native_margin_contribution_count": 2,
                "native_contribution_ledger_fingerprint": "0" * 64,
                "native_margin_contribution_ledger_fingerprint": "1" * 64,
                "prediction_matches_native": True,
                "correct_matches_native": True,
                "margin_matches_native": True,
                "active_pixels_match_native": True,
                "contribution_ledger_matches_native": True,
                "margin_contribution_ledger_matches_native": True,
                "attribution_matches_logit": True,
                "attribution_matches_margin": True,
            },
            "q31_reference_evaluation": {
                "samples": 2,
                "correct": 2,
                "incorrect": 0,
                "accuracy_percent": 100.0,
                "lowest_margin": 0,
            },
            "inference_trace_profile": valid_pipeline_inference_trace_profile_for_tests(),
            "mlp_witness": {
                "passed": mlp_witness["passed"],
                "samples": mlp_witness["samples"],
                "dataset_loops": expected_mlp_dataset_loops(),
                "predictions": mlp_witness["predictions"],
                "correct": mlp_witness["correct"],
                "witness_proof_fingerprint": mlp_witness["witness_proof"]["fingerprint"],
                "witness_payload_bytes": mlp_proof["witness_payload_bytes"],
                "trace_payload_bytes": mlp_proof["trace_payload_bytes"],
                "replay_payload_bytes": mlp_proof["replay_payload_bytes"],
                "recomputed_update_payload_bytes": mlp_proof["recomputed_update_payload_bytes"],
                "forward_recompute_steps": mlp_proof["forward_recompute_steps"],
                "inverse_recompute_steps": mlp_proof["inverse_recompute_steps"],
                "total_recompute_steps": mlp_proof["total_recompute_steps"],
                "witness_to_model_payload_ratio": mlp_proof["witness_to_model_payload_ratio"],
                "trace_to_model_payload_ratio": mlp_proof["trace_to_model_payload_ratio"],
                "recomputed_update_to_witness_payload_ratio": (
                    mlp_proof["recomputed_update_to_witness_payload_ratio"]
                ),
            },
            "invertible_coupling": {
                "passed": coupling["passed"],
                "initial_left": coupling["initial"]["left"],
                "initial_right": coupling["initial"]["right"],
                "forward_left": coupling["forward"]["left"],
                "forward_right": coupling["forward"]["right"],
                "model_payload_bytes": coupling_proof["model_payload_bytes"],
                "state_payload_bytes": coupling_proof["state_payload_bytes"],
                "witness_payload_bytes": coupling_proof["witness_payload_bytes"],
                "trace_payload_bytes": coupling_proof["trace_payload_bytes"],
                "replay_payload_bytes": coupling_proof["replay_payload_bytes"],
                "forward_recompute_steps": coupling_proof["forward_recompute_steps"],
                "inverse_recompute_steps": coupling_proof["inverse_recompute_steps"],
                "total_recompute_steps": coupling_proof["total_recompute_steps"],
                "witness_to_model_payload_ratio": coupling_proof["witness_to_model_payload_ratio"],
                "trace_to_model_payload_ratio": coupling_proof["trace_to_model_payload_ratio"],
            },
            "triangular_residual": {
                "passed": residual["passed"],
                "initial_x": residual["initial"]["x"],
                "forward_x": residual["forward"]["x"],
                "parameter_payload_bytes": residual_proof["parameter_payload_bytes"],
                "state_payload_bytes": residual_proof["state_payload_bytes"],
                "witness_payload_bytes": residual_proof["witness_payload_bytes"],
                "trace_payload_bytes": residual_proof["trace_payload_bytes"],
                "replay_payload_bytes": residual_proof["replay_payload_bytes"],
                "forward_recompute_steps": residual_proof["forward_recompute_steps"],
                "inverse_recompute_steps": residual_proof["inverse_recompute_steps"],
                "total_recompute_steps": residual_proof["total_recompute_steps"],
                "witness_to_parameter_payload_ratio": residual_proof[
                    "witness_to_parameter_payload_ratio"
                ],
                "trace_to_parameter_payload_ratio": residual_proof[
                    "trace_to_parameter_payload_ratio"
                ],
            },
            "reversible_preprocess": {
                "passed": preprocess["passed"],
                "initial_features": preprocess["initial"]["features"],
                "forward_features": preprocess["forward"]["features"],
                "raw_payload_bytes": preprocess_proof["raw_payload_bytes"],
                "mean_payload_bytes": preprocess_proof["mean_payload_bytes"],
                "feature_payload_bytes": preprocess_proof["feature_payload_bytes"],
                "witness_payload_bytes": preprocess_proof["witness_payload_bytes"],
                "trace_payload_bytes": preprocess_proof["trace_payload_bytes"],
                "replay_payload_bytes": preprocess_proof["replay_payload_bytes"],
                "forward_recompute_steps": preprocess_proof["forward_recompute_steps"],
                "inverse_recompute_steps": preprocess_proof["inverse_recompute_steps"],
                "total_recompute_steps": preprocess_proof["total_recompute_steps"],
                "witness_to_state_payload_ratio": preprocess_proof["witness_to_state_payload_ratio"],
                "trace_to_state_payload_ratio": preprocess_proof["trace_to_state_payload_ratio"],
            },
            "reversible_inference_trace": {
                "passed": inference_trace["passed"],
                "initial_features": inference_trace["initial"]["features"],
                "forward_features": inference_trace["forward"]["features"],
                "forward_logits": inference_trace["forward"]["logits"],
                "top_classes": inference_trace["forward"]["top_classes"],
                "top_logit_values": inference_trace["forward"]["top_logit_values"],
                "top_logits": inference_trace["top_logits"],
                "attribution": inference_trace["attribution"],
                "prediction": inference_trace["forward"]["prediction"],
                "runner_up_class": inference_trace["forward"]["runner_up_class"],
                "margin": inference_trace["forward"]["margin"],
                "label_rank": inference_trace["forward"]["label_rank"],
                "correct": inference_trace["forward"]["correct"],
                "top2_correct": inference_trace["forward"]["top2_correct"],
                "raw_payload_bytes": inference_trace_proof["raw_payload_bytes"],
                "mean_payload_bytes": inference_trace_proof["mean_payload_bytes"],
                "feature_payload_bytes": inference_trace_proof["feature_payload_bytes"],
                "weight_payload_bytes": inference_trace_proof["weight_payload_bytes"],
                "bias_payload_bytes": inference_trace_proof["bias_payload_bytes"],
                "model_payload_bytes": inference_trace_proof["model_payload_bytes"],
                "logit_payload_bytes": inference_trace_proof["logit_payload_bytes"],
                "top_class_payload_bytes": inference_trace_proof["top_class_payload_bytes"],
                "top_logit_value_payload_bytes": inference_trace_proof[
                    "top_logit_value_payload_bytes"
                ],
                "prediction_payload_bytes": inference_trace_proof["prediction_payload_bytes"],
                "runner_up_payload_bytes": inference_trace_proof["runner_up_payload_bytes"],
                "margin_payload_bytes": inference_trace_proof["margin_payload_bytes"],
                "label_rank_payload_bytes": inference_trace_proof["label_rank_payload_bytes"],
                "correct_payload_bytes": inference_trace_proof["correct_payload_bytes"],
                "top_k_correct_payload_bytes": inference_trace_proof[
                    "top_k_correct_payload_bytes"
                ],
                "label_payload_bytes": inference_trace_proof["label_payload_bytes"],
                "state_payload_bytes": inference_trace_proof["state_payload_bytes"],
                "witness_payload_bytes": inference_trace_proof["witness_payload_bytes"],
                "trace_payload_bytes": inference_trace_proof["trace_payload_bytes"],
                "replay_payload_bytes": inference_trace_proof["replay_payload_bytes"],
                "forward_recompute_steps": inference_trace_proof["forward_recompute_steps"],
                "inverse_recompute_steps": inference_trace_proof["inverse_recompute_steps"],
                "total_recompute_steps": inference_trace_proof["total_recompute_steps"],
                "witness_to_model_payload_ratio": inference_trace_proof[
                    "witness_to_model_payload_ratio"
                ],
                "trace_to_model_payload_ratio": inference_trace_proof[
                    "trace_to_model_payload_ratio"
                ],
                "witness_to_state_payload_ratio": inference_trace_proof[
                    "witness_to_state_payload_ratio"
                ],
                "trace_to_state_payload_ratio": inference_trace_proof[
                    "trace_to_state_payload_ratio"
                ],
            },
            "scorecard": {
                "kind": "reverie_mnist_ml_v6_scorecard",
                "train_updates_per_second": 10.0,
                "built_in_eval_samples_per_second": 10.0,
                "model_evaluation_samples_per_second": 10.0,
                "reverse_steps_per_second": 10.0,
                "reverse_to_train_elapsed_ratio": 1.0,
                "max_reverse_train_elapsed_ratio": 5.0,
                "run_peak_rss_bytes": 123456,
                "estimated_payload_bytes": 65552,
                "total_model_payload_bytes": model_bytes,
                "total_witness_payload_bytes": witness_bytes,
                "total_trace_payload_bytes": trace_bytes,
                "trace_to_model_payload_ratio": trace_bytes / model_bytes,
                "witness_to_model_payload_ratio": witness_bytes / model_bytes,
                "max_replay_payload_bytes": max(replay_payloads.values()),
                "total_recompute_steps": 4,
                "total_forward_recompute_steps": 2,
                "total_inverse_recompute_steps": 2,
                "balanced_recompute": True,
                "contracts": {
                    "training_step_debug": True,
                    "model_import": True,
                    "native_inference_explanation": True,
                    "evaluation_row_inference_explanation": True,
                    "q31_reference_inference": True,
                    "inference_trace_profile": True,
                    "mlp_witness": True,
                    "invertible_coupling": True,
                    "triangular_residual": True,
                    "reversible_preprocess": True,
                    "reversible_inference_trace": True,
                },
                "base_gates_passed": True,
                "base_failed_gates": [],
                "gates_passed": True,
                "failed_gates": [],
            },
            "recompute_frontier": recompute_frontier,
            "scaling_projection": scaling_projection,
            "ml_capability_map": valid_pipeline_capability_map_for_tests(),
            "ml_goal_readiness": valid_pipeline_goal_readiness_for_tests(),
            "replay_payload_bytes": replay_payloads,
            "max_replay_payload_bytes": max(replay_payloads.values()),
        },
        "gates": valid_pipeline_gates_for_tests(),
        "evidence": valid_pipeline_evidence_for_tests(include_summary=False),
        "claims": valid_pipeline_claims_for_tests(),
    }


def valid_model_capsule_report() -> dict[str, Any]:
    summary = valid_pipeline_summary_report()
    metrics = summary["metrics"]
    evidence = valid_pipeline_evidence_for_tests(include_summary=True)
    evidence_files = evidence["files"]
    payload = {
        "schema": MODEL_CAPSULE_SCHEMA,
        "pipeline_kind": PIPELINE_KIND,
        "pipeline_summary": "target/pipeline-summary.json",
        "profile_markdown": summary["profile_markdown"],
        "sample_limit": summary["sample_limit"],
        "audit_step": summary["audit_step"],
        "audit_step_strategy": summary["audit_step_strategy"],
        "evaluation_row": summary["evaluation_row"],
        "evaluation_row_strategy": summary["evaluation_row_strategy"],
        "reports": summary["reports"],
        "bundles": summary["bundles"],
        "evidence": evidence,
        "gates": {
            "passed": True,
            "total": len(PIPELINE_GATE_METRICS),
            "passed_count": len(PIPELINE_GATE_METRICS),
            "failed_metrics": [],
            "policy": valid_pipeline_gate_policy_for_tests(),
        },
        "model": {
            "bundle": summary["bundles"]["model"],
            "sha256": evidence_files["bundle.model"]["sha256"],
            "payload_bytes": metrics["artifact_profile"]["total_model_payload_bytes"],
        },
        "imported_model": {
            "source_json": summary["bundles"]["imported_model_source"],
            "source_json_sha256": evidence_files["bundle.imported_model_source"]["sha256"],
            "bundle": summary["bundles"]["imported_model"],
            "bundle_sha256": evidence_files["bundle.imported_model"]["sha256"],
            "source_model_json_checked": metrics["model_import"]["verification_source_checked"],
            "provenance_kind": metrics["model_import"]["verification_provenance_kind"],
            "source_model_json_fingerprint": metrics["model_import"][
                "source_model_json_fingerprint"
            ],
            "inference_replay_passed": metrics["imported_model_inference"][
                "proof_matches"
            ]
            and metrics["imported_model_inference"]["result_matches"],
            "prediction_matches_native": metrics["imported_model_inference"][
                "prediction_matches_native"
            ],
            "margin_matches_native": metrics["imported_model_inference"][
                "margin_matches_native"
            ],
        },
        "samples": {
            "bundle": summary["bundles"]["samples"],
            "sha256": evidence_files["bundle.samples"]["sha256"],
            "count": metrics["model_evaluation"]["samples"],
            "accuracy_percent": metrics["model_evaluation"]["accuracy_percent"],
        },
        "training_lineage": {
            "checked_steps": metrics["training_audit_replay"]["checked"],
            "restored_initial_model": metrics["training_audit_replay"]["restored_initial_model"],
            "final_model_replayed": metrics["training_audit_replay"]["final_model_replayed"],
            "lineage_ledger_matches": metrics["training_audit_replay"]["lineage_ledger_matches"],
            "lineage_ledger_fingerprint": metrics["training_audit_replay"]["lineage_ledger_fingerprint"],
            "transition_ledger_fingerprint": metrics["training_audit_replay"]["transition_ledger_fingerprint"],
            "final_chain": metrics["training_audit_replay"]["final_chain"],
        },
        "inference": {
            "imported_model_replay_passed": True,
            "imported_model_prediction": metrics["imported_model_inference"]["prediction"],
            "imported_model_matches_native": True,
            "native_replay_passed": True,
            "native_prediction": metrics["native_inference_explanation"]["prediction"],
            "native_correct": metrics["native_inference_explanation"]["correct"],
            "native_margin": metrics["native_inference_explanation"]["margin"],
            "evaluation_row_replay_passed": True,
            "evaluation_row_prediction": metrics["evaluation_row_inference_explanation"]["prediction"],
            "evaluation_row_correct": metrics["evaluation_row_inference_explanation"]["correct"],
            "evaluation_row_margin": metrics["evaluation_row_inference_explanation"]["margin"],
            "q31_reference_prediction": metrics["q31_reference_inference"]["prediction"],
            "q31_reference_matches_native": True,
            "action_contract": expected_inference_action_contract(metrics),
        },
        "reversible_inference_trace": {
            "passed": metrics["reversible_inference_trace"]["passed"],
            "logits": metrics["reversible_inference_trace"]["forward_logits"],
            "top_classes": metrics["reversible_inference_trace"]["top_classes"],
            "top_logit_values": metrics["reversible_inference_trace"]["top_logit_values"],
            "top_logits": metrics["reversible_inference_trace"]["top_logits"],
            "prediction": metrics["reversible_inference_trace"]["prediction"],
            "runner_up_class": metrics["reversible_inference_trace"]["runner_up_class"],
            "margin": metrics["reversible_inference_trace"]["margin"],
            "label_rank": metrics["reversible_inference_trace"]["label_rank"],
            "correct": metrics["reversible_inference_trace"]["correct"],
            "top2_correct": metrics["reversible_inference_trace"]["top2_correct"],
            "contribution_ledger_fingerprint": metrics["reversible_inference_trace"][
                "attribution"
            ]["contribution_ledger_fingerprint"],
            "margin_contribution_ledger_fingerprint": metrics["reversible_inference_trace"][
                "attribution"
            ]["margin_contribution_ledger_fingerprint"],
            "witness_payload_bytes": metrics["reversible_inference_trace"][
                "witness_payload_bytes"
            ],
            "trace_payload_bytes": metrics["reversible_inference_trace"]["trace_payload_bytes"],
            "replay_payload_bytes": metrics["reversible_inference_trace"]["replay_payload_bytes"],
            "total_recompute_steps": metrics["reversible_inference_trace"][
                "total_recompute_steps"
            ],
        },
        "witnesses": {
            "mlp_passed": metrics["mlp_witness"]["passed"],
            "mlp_samples": metrics["mlp_witness"]["samples"],
            "mlp_dataset_loops": metrics["mlp_witness"]["dataset_loops"],
            "mlp_witness_proof_fingerprint": metrics["mlp_witness"]["witness_proof_fingerprint"],
            "witness_payload_bytes": metrics["artifact_profile"]["total_witness_payload_bytes"],
            "witness_to_model_payload_ratio": metrics["artifact_profile"]["witness_to_model_payload_ratio"],
        },
        "invertible_patterns": {
            "coupling_passed": metrics["invertible_coupling"]["passed"],
            "triangular_residual_passed": metrics["triangular_residual"]["passed"],
            "reversible_preprocess_passed": metrics["reversible_preprocess"]["passed"],
            "reversible_inference_trace_passed": metrics["reversible_inference_trace"]["passed"],
        },
        "scorecard": {
            "train_updates_per_second": metrics["scorecard"]["train_updates_per_second"],
            "model_evaluation_samples_per_second": metrics["scorecard"]["model_evaluation_samples_per_second"],
            "reverse_steps_per_second": metrics["scorecard"]["reverse_steps_per_second"],
            "reverse_to_train_elapsed_ratio": metrics["scorecard"]["reverse_to_train_elapsed_ratio"],
            "run_peak_rss_bytes": metrics["scorecard"]["run_peak_rss_bytes"],
            "max_replay_payload_bytes": metrics["scorecard"]["max_replay_payload_bytes"],
            "trace_to_model_payload_ratio": metrics["scorecard"]["trace_to_model_payload_ratio"],
            "witness_to_model_payload_ratio": metrics["scorecard"]["witness_to_model_payload_ratio"],
            "total_recompute_steps": metrics["scorecard"]["total_recompute_steps"],
            "balanced_recompute": metrics["scorecard"]["balanced_recompute"],
        },
        "capabilities": metrics["ml_capability_map"],
        "readiness": metrics["ml_goal_readiness"],
        "claims": summary["claims"],
    }
    return {
        "kind": MODEL_CAPSULE_KIND,
        "algorithm": "sha256",
        "fingerprint": sha256_json(payload),
        "payload": payload,
    }


def valid_pipeline_manifest_report() -> dict[str, Any]:
    return {
        "kind": PIPELINE_KIND,
        "sample_limit": 2,
        "requested_audit_step": 0,
        "audit_step": 0,
        "audit_step_strategy": "explicit",
        "requested_evaluation_row": 1,
        "evaluation_row": 1,
        "evaluation_row_strategy": "explicit",
        "profile_markdown": "target/mnist-ml-audit-profile.md",
        "summary": "target/pipeline-summary.json",
        "model_capsule": "target/model-capsule.json",
        "model_capsule_fingerprint": "1" * 64,
        "gates_passed": True,
        "gate_policy": valid_pipeline_gate_policy_for_tests(),
        "reports": [f"target/{key}.json" for key in PIPELINE_REPORT_KEYS],
        "bundles": [f"target/{key}.json" for key in PIPELINE_BUNDLE_KEYS],
        "evidence": valid_pipeline_evidence_for_tests(include_summary=True),
        "steps": [
            {
                "label": "self-test",
                "output": "target/run-report.json",
                "command": ["reverie-mnist-linear", "--self-test", "--json"],
            }
        ],
    }


def audit_contract_from_expected_results(results: dict[str, bool]) -> dict[str, Any]:
    return {
        "claim": AUDIT_CONTRACT_CLAIM,
        "passed": all(results.values()),
        "checks": [
            {
                "metric": metric,
                "passed": results[metric],
                "actual": "self-test",
                "requirement": "self-test",
            }
            for metric in AUDIT_CONTRACT_METRICS
        ],
    }


def expect_self_test_error(label: str, report: dict[str, Any], needle: str) -> None:
    try:
        validate_report(
            report,
            require_reverse_check=True,
            require_peak_rss=True,
            require_ml_profile=False,
            require_audit_contract=False,
        )
    except ValueError as error:
        if needle not in str(error):
            raise AssertionError(f"{label} reported `{error}`, expected `{needle}`") from error
        return
    raise AssertionError(f"{label} did not fail")


def expect_report_set_error(
    label: str,
    reports: list[tuple[Path, dict[str, Any]]],
    needle: str,
) -> None:
    try:
        validate_report_set_consistency(reports)
    except ValueError as error:
        if needle not in str(error):
            raise AssertionError(f"{label} reported `{error}`, expected `{needle}`") from error
        return
    raise AssertionError(f"{label} did not fail")


def run_self_tests() -> int:
    try:
        validate_report(
            valid_run_report(),
            require_reverse_check=True,
            require_peak_rss=True,
            require_ml_profile=False,
            require_audit_contract=False,
        )
        validate_report(
            valid_comparison_report(),
            require_reverse_check=False,
            require_peak_rss=False,
            require_ml_profile=True,
            require_audit_contract=False,
        )
        validate_report(
            valid_audit_contract_report(),
            require_reverse_check=False,
            require_peak_rss=False,
            require_ml_profile=True,
            require_audit_contract=True,
        )
        validate_report(
            valid_mlp_witness_report(),
            require_reverse_check=False,
            require_peak_rss=False,
            require_ml_profile=False,
            require_audit_contract=False,
        )
        validate_report(
            valid_q31_reference_inference_report(),
            require_reverse_check=False,
            require_peak_rss=False,
            require_ml_profile=False,
            require_audit_contract=False,
        )
        validate_report(
            valid_q31_reference_evaluation_report(),
            require_reverse_check=False,
            require_peak_rss=False,
            require_ml_profile=False,
            require_audit_contract=False,
        )
        validate_report(
            valid_audit_step_report(),
            require_reverse_check=False,
            require_peak_rss=False,
            require_ml_profile=False,
            require_audit_contract=False,
        )
        validate_report(
            valid_audit_scan_report(),
            require_reverse_check=False,
            require_peak_rss=False,
            require_ml_profile=False,
            require_audit_contract=False,
        )
        validate_report(
            valid_step_verification_report(),
            require_reverse_check=False,
            require_peak_rss=False,
            require_ml_profile=False,
            require_audit_contract=False,
        )
        for report in (
            valid_model_import_report(),
            valid_model_verification_report(),
            valid_native_inference_report(INFERENCE_AUDIT_KIND),
            valid_native_inference_report(MODEL_INFERENCE_AUDIT_KIND),
            valid_native_inference_report(MODEL_EVALUATION_ROW_KIND),
            valid_invertible_coupling_report(),
            valid_triangular_residual_report(),
            valid_reversible_preprocess_report(),
            valid_reversible_inference_trace_report(),
            valid_inference_verification_report(),
            valid_model_evaluation_report(),
            valid_model_evaluation_scan_report(),
            valid_model_evaluation_verification_report(),
            valid_pipeline_summary_report(),
            valid_model_capsule_report(),
            valid_pipeline_manifest_report(),
        ):
            validate_report(
                report,
                require_reverse_check=False,
                require_peak_rss=False,
                require_ml_profile=False,
                require_audit_contract=False,
            )

        valid_summary_set = valid_pipeline_summary_report()
        valid_capsule_set = valid_model_capsule_report()
        valid_manifest_set = valid_pipeline_manifest_report()
        valid_manifest_set["model_capsule_fingerprint"] = valid_capsule_set["fingerprint"]
        validate_report_set_consistency(
            [
                (Path("target/pipeline-summary.json"), valid_summary_set),
                (Path("target/model-capsule.json"), valid_capsule_set),
                (Path("target/pipeline-manifest.json"), valid_manifest_set),
            ]
        )
        stale_capsule = json.loads(json.dumps(valid_capsule_set))
        stale_capsule["payload"]["witnesses"]["mlp_witness_proof_fingerprint"] = "f" * 64
        stale_capsule["fingerprint"] = sha256_json(stale_capsule["payload"])
        expect_report_set_error(
            "stale model capsule summary link",
            [
                (Path("target/pipeline-summary.json"), valid_summary_set),
                (Path("target/model-capsule.json"), stale_capsule),
            ],
            "payload.witnesses",
        )
        stale_manifest = json.loads(json.dumps(valid_manifest_set))
        stale_manifest["model_capsule_fingerprint"] = "2" * 64
        expect_report_set_error(
            "stale manifest capsule fingerprint",
            [
                (Path("target/model-capsule.json"), valid_capsule_set),
                (Path("target/pipeline-manifest.json"), stale_manifest),
            ],
            "model_capsule_fingerprint",
        )

        bad_trace = valid_run_report()
        bad_trace["trace"]["payload_bytes"] += 1
        expect_self_test_error("bad trace", bad_trace, "trace payload_bytes")
        bad_mlp = valid_mlp_witness_report()
        bad_mlp["proof"]["trace_payload_bytes"] += 1
        expect_self_test_error("bad MLP witness", bad_mlp, "trace_payload_bytes")
        bad_coupling = valid_invertible_coupling_report()
        bad_coupling["proof"]["witness_payload_bytes"] = 8
        expect_self_test_error("bad invertible coupling", bad_coupling, "witness and trace")
        bad_residual = valid_triangular_residual_report()
        bad_residual["proof"]["witness_payload_bytes"] = 8
        expect_self_test_error("bad triangular residual", bad_residual, "witness and trace")
        bad_preprocess = valid_reversible_preprocess_report()
        bad_preprocess["proof"]["trace_payload_bytes"] = 8
        expect_self_test_error("bad reversible preprocessing", bad_preprocess, "witness and trace")
        bad_inference_trace = valid_reversible_inference_trace_report()
        bad_inference_trace["proof"]["trace_payload_bytes"] = 8
        bad_inference_trace["proof"]["replay_payload_bytes"] += 8
        expect_self_test_error(
            "bad reversible inference trace",
            bad_inference_trace,
            "trace_payload_bytes must be zero",
        )
        bad_inference_trace_attribution = valid_reversible_inference_trace_report()
        bad_inference_trace_attribution["attribution"]["matches_margin"] = False
        expect_self_test_error(
            "bad reversible inference trace attribution",
            bad_inference_trace_attribution,
            "attribution must match reference contribution ledger",
        )
        bad_q31_inference = valid_q31_reference_inference_report()
        bad_q31_inference["attribution"]["reconstructed_margin"] += 1
        expect_self_test_error(
            "bad Q31 reference inference",
            bad_q31_inference,
            "reconstructed_margin",
        )
        bad_capsule_fingerprint = valid_model_capsule_report()
        bad_capsule_fingerprint["payload"]["sample_limit"] += 1
        expect_self_test_error(
            "bad model capsule fingerprint",
            bad_capsule_fingerprint,
            "model capsule fingerprint",
        )
        bad_capsule_witness = valid_model_capsule_report()
        bad_capsule_witness["payload"]["witnesses"]["mlp_passed"] = False
        bad_capsule_witness["fingerprint"] = sha256_json(bad_capsule_witness["payload"])
        expect_self_test_error(
            "bad model capsule witness",
            bad_capsule_witness,
            "mlp_passed",
        )
        bad_q31_evaluation = valid_q31_reference_evaluation_report()
        bad_q31_evaluation["summary"]["correct"] = 1
        expect_self_test_error(
            "bad Q31 reference evaluation",
            bad_q31_evaluation,
            "summary.correct",
        )
        bad_audit_step = valid_audit_step_report()
        bad_audit_step["witness_checks"]["computed_prediction"] = 1
        expect_self_test_error(
            "bad audit step",
            bad_audit_step,
            "computed_prediction",
        )
        bad_audit_step_contract = valid_audit_step_report()
        bad_audit_step_contract["debug_contract"]["checks"][0]["passed"] = False
        expect_self_test_error(
            "bad audit step debug contract",
            bad_audit_step_contract,
            "debug_contract check `witness_recomputes_prediction`",
        )
        bad_audit_scan = valid_audit_scan_report()
        bad_audit_scan["summary"]["correct"] = 1
        expect_self_test_error(
            "bad audit scan",
            bad_audit_scan,
            "summary.correct",
        )
        bad_step_verification = valid_step_verification_report()
        bad_step_verification["proof"]["replay_payload_bytes"] += 1
        expect_self_test_error(
            "bad step verification",
            bad_step_verification,
            "replay_payload_bytes",
        )
        bad_native_inference = valid_native_inference_report()
        bad_native_inference["proof"]["checks"]["restored_initial_state"] = False
        expect_self_test_error(
            "bad native inference",
            bad_native_inference,
            "restored_initial_state",
        )
        bad_inference_explanation = valid_native_inference_report()
        bad_inference_explanation["explanation_contract"]["checks"][0]["passed"] = False
        expect_self_test_error(
            "bad inference explanation contract",
            bad_inference_explanation,
            "explanation_contract.passed must summarize checks",
        )
        bad_model_evaluation = valid_model_evaluation_report()
        bad_model_evaluation["summary"]["correct"] = 1
        expect_self_test_error(
            "bad model evaluation",
            bad_model_evaluation,
            "summary.correct",
        )
        bad_inference_verification = valid_inference_verification_report()
        bad_inference_verification["proof_matches"] = False
        expect_self_test_error(
            "bad inference verification",
            bad_inference_verification,
            "proof_matches",
        )
        bad_pipeline_summary = valid_pipeline_summary_report()
        bad_pipeline_summary["gates"]["passed"] = False
        bad_pipeline_summary["gates"]["checks"][0]["passed"] = False
        expect_self_test_error(
            "bad pipeline summary",
            bad_pipeline_summary,
            "failed gate",
        )
        bad_pipeline_claim = valid_pipeline_summary_report()
        bad_pipeline_claim["claims"]["checks"][0]["evidence"][0] = "report.missing"
        expect_self_test_error(
            "bad pipeline claims",
            bad_pipeline_claim,
            "references unknown file",
        )
        bad_pipeline_integrity_claim = valid_pipeline_summary_report()
        for claim in bad_pipeline_integrity_claim["claims"]["checks"]:
            if claim["claim"] == "artifact_evidence_integrity":
                claim["evidence"].remove("bundle.inference_trace_roundtrip_proof")
                break
        expect_self_test_error(
            "bad pipeline integrity claim",
            bad_pipeline_integrity_claim,
            "must cite every digest-indexed evidence file",
        )
        bad_pipeline_debug = valid_pipeline_summary_report()
        bad_pipeline_debug["metrics"]["training_step_debug"]["checks"][0]["passed"] = False
        expect_self_test_error(
            "bad pipeline debug contract",
            bad_pipeline_debug,
            "metrics.training_step_debug.passed must summarize checks",
        )
        bad_pipeline_selection = valid_pipeline_summary_report()
        bad_pipeline_selection["metrics"]["training_step_selection"]["matches_requested_step"] = False
        expect_self_test_error(
            "bad pipeline training-step selection",
            bad_pipeline_selection,
            "metrics.training_step_selection.matches_requested_step must summarize selected step",
        )
        bad_pipeline_row_selection = valid_pipeline_summary_report()
        bad_pipeline_row_selection["metrics"]["evaluation_row_selection"]["matches_selection_strategy"] = False
        expect_self_test_error(
            "bad pipeline evaluation-row selection",
            bad_pipeline_row_selection,
            "metrics.evaluation_row_selection.matches_selection_strategy must summarize selected row",
        )
        bad_pipeline_reverse_cost = valid_pipeline_summary_report()
        bad_pipeline_reverse_cost["metrics"]["reverse_check_cost"]["reverse_to_train_elapsed_ratio"] = 2.0
        expect_self_test_error(
            "bad pipeline reverse-check cost",
            bad_pipeline_reverse_cost,
            "metrics.reverse_check_cost.reverse_to_train_elapsed_ratio expected",
        )
        bad_pipeline_reverse_gate = valid_pipeline_summary_report()
        bad_pipeline_reverse_gate["gates"]["policy"]["max_reverse_train_elapsed_ratio"] = 0.5
        expect_self_test_error(
            "bad pipeline reverse-check ratio gate",
            bad_pipeline_reverse_gate,
            "reverse_check_elapsed_ratio gate result must match policy",
        )
        bad_pipeline_reference_inference = valid_pipeline_summary_report()
        bad_pipeline_reference_inference["metrics"]["q31_reference_inference"]["margin_matches_native"] = False
        expect_self_test_error(
            "bad pipeline Q31 reference inference",
            bad_pipeline_reference_inference,
            "metrics.q31_reference_inference.margin_matches_native must be true",
        )
        bad_pipeline_scorecard = valid_pipeline_summary_report()
        bad_pipeline_scorecard["metrics"]["scorecard"]["max_replay_payload_bytes"] += 1
        expect_self_test_error(
            "bad pipeline scorecard",
            bad_pipeline_scorecard,
            "metrics.scorecard.max_replay_payload_bytes must match metrics",
        )
        bad_pipeline_frontier = valid_pipeline_summary_report()
        bad_pipeline_frontier["metrics"]["recompute_frontier"]["rows"][0]["trace_payload_bytes"] += 1
        expect_self_test_error(
            "bad pipeline recompute frontier",
            bad_pipeline_frontier,
            "metrics.recompute_frontier.rows[0].trace_to_model_payload_ratio expected",
        )
        bad_pipeline_scaling = valid_pipeline_summary_report()
        bad_pipeline_scaling["metrics"]["scaling_projection"]["families"][0]["projections"][1][
            "projected_replay_payload_bytes"
        ] += 1
        expect_self_test_error(
            "bad pipeline scaling projection",
            bad_pipeline_scaling,
            "metrics.scaling_projection.families[0].projections[1].projected_replay_payload_bytes",
        )
        bad_pipeline_capability = valid_pipeline_summary_report()
        bad_pipeline_capability["metrics"]["ml_capability_map"]["capabilities"][0]["passed"] = False
        expect_self_test_error(
            "bad pipeline capability map",
            bad_pipeline_capability,
            "metrics.ml_capability_map.capabilities[0].passed must summarize its gate metrics",
        )
        bad_pipeline_manifest = valid_pipeline_manifest_report()
        bad_pipeline_manifest["gates_passed"] = False
        expect_self_test_error(
            "bad pipeline manifest",
            bad_pipeline_manifest,
            "gates_passed",
        )
    except TypeError:
        # Keep self-test failures readable if the helper call shape regresses.
        raise
    except (AssertionError, ValueError) as error:
        print(f"error: self-test failed: {error}", file=sys.stderr)
        return 1

    bad_profile = valid_comparison_report()
    bad_profile["ml_profile"]["total_recompute_steps"] = 99
    try:
        validate_report(
            bad_profile,
            require_reverse_check=False,
            require_peak_rss=False,
            require_ml_profile=True,
            require_audit_contract=False,
        )
    except ValueError as error:
        if "total_recompute_steps" not in str(error):
            print(f"error: wrong bad-profile failure: {error}", file=sys.stderr)
            return 1
    else:
        print("error: bad-profile self-test did not fail", file=sys.stderr)
        return 1

    bad_contract = valid_audit_contract_report()
    bad_contract["audit_contract"]["checks"][0]["passed"] = False
    try:
        validate_report(
            bad_contract,
            require_reverse_check=False,
            require_peak_rss=False,
            require_ml_profile=True,
            require_audit_contract=True,
        )
    except ValueError as error:
        if "audit_contract check `training_trace`" not in str(error):
            print(f"error: wrong bad-contract failure: {error}", file=sys.stderr)
            return 1
    else:
        print("error: bad-contract self-test did not fail", file=sys.stderr)
        return 1

    print("ok: MNIST ML profile checker self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    if not args.paths:
        print("error: at least one report path is required", file=sys.stderr)
        return 2

    loaded_reports: list[tuple[Path, dict[str, Any]]] = []
    for path in args.paths:
        try:
            report = load_json(path)
            validate_report(
                report,
                require_reverse_check=args.require_reverse_check,
                require_peak_rss=args.require_peak_rss,
                require_ml_profile=args.require_ml_profile,
                require_audit_contract=args.require_audit_contract,
            )
            if args.verify_pipeline_files:
                verify_pipeline_file_evidence(report)
            if isinstance(report, dict):
                loaded_reports.append((path, report))
        except ValueError as error:
            print(f"error: {path}: {error}", file=sys.stderr)
            return 1
    try:
        validate_report_set_consistency(loaded_reports)
    except ValueError as error:
        print(f"error: report-set consistency: {error}", file=sys.stderr)
        return 1

    print(f"ok: validated {len(args.paths)} MNIST ML profile report(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
