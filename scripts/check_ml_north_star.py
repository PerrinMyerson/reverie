#!/usr/bin/env python3
"""Validate the Reverie reversible ML north-star release packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import check_benchmark_artifact as benchmark_check
import check_mnist_ml_profile as profile_check
import run_mnist_ml_audit_pipeline as audit_pipeline
import summarize_janus_performance
import verify_model_capsule as capsule_check


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = REPO_ROOT / "target" / "mnist-ml-audit-pipeline-release"
DEFAULT_BENCHMARK_JSON = (
    REPO_ROOT / "benchmarks" / "results" / "jana-vs-reverie-local-arm64-memory.json"
)
DEFAULT_BENCHMARK_MARKDOWN = (
    REPO_ROOT / "benchmarks" / "results" / "jana-vs-reverie-local-arm64-memory.md"
)
DEFAULT_REPORT_NAME = "ml-north-star-gate.json"
DEFAULT_MARKDOWN_NAME = "ml-north-star-gate.md"
DEFAULT_TRANSCRIPT_NAME = "ml-reviewer-replay.json"
DEFAULT_TRANSCRIPT_MARKDOWN_NAME = "ml-reviewer-replay.md"
DEFAULT_RELEASE_VERIFICATION_NAME = "ml-release-verification.json"
DEFAULT_RELEASE_VERIFICATION_MARKDOWN_NAME = "ml-release-verification.md"
GATE_KIND = "reverie_ml_north_star_release_gate"
GATE_SCHEMA = "reverie_ml_north_star_release_gate_v1"
TRANSCRIPT_KIND = "reverie_ml_north_star_reviewer_replay"
TRANSCRIPT_SCHEMA = "reverie_ml_north_star_reviewer_replay_v1"
RELEASE_VERIFICATION_KIND = "reverie_ml_north_star_release_verification"
RELEASE_VERIFICATION_SCHEMA = "reverie_ml_north_star_release_verification_v1"
NORTH_STAR = "best_small_language_for_reversible_inspectable_deterministic_ml_kernels"
NON_GOAL = "general_purpose_pytorch_tensorflow_training_replacement"
EXPECTED_PHASES = ("V1", "V2", "V3", "V4", "V5", "V6")
GOAL_CONTRACT_CLAIMS = (
    "deterministic_q31_inference_replay",
    "reversible_inference_roundtrip",
    "training_lineage_replay",
    "training_step_debug_replay",
    "witness_taped_mlp_update_replay",
    "invertible_zero_witness_layer_patterns",
    "speed_memory_trace_reverse_scorecard",
)
GOAL_CONTRACT_CLAIM_REQUIREMENTS = {
    "deterministic_q31_inference_replay": {
        "pipeline_claims": ("deterministic_q31_inference",),
        "gate_metrics": (
            "model_import_provenance",
            "imported_model_inference_replay",
            "native_inference_replay",
            "native_standalone_rev_classifier",
            "q31_reference_matches_native_inference",
            "reversible_inference_trace_replay",
        ),
        "release_checks": (),
    },
    "reversible_inference_roundtrip": {
        "pipeline_claims": ("deterministic_q31_inference",),
        "gate_metrics": ("reversible_inference_trace_replay",),
        "release_checks": ("roundtrip",),
    },
    "training_lineage_replay": {
        "pipeline_claims": ("auditable_model_lineage",),
        "gate_metrics": ("training_audit_lineage_replay",),
        "release_checks": (),
    },
    "training_step_debug_replay": {
        "pipeline_claims": ("debug_training_update",),
        "gate_metrics": (
            "training_step_replay",
            "training_step_debug_contract",
            "training_update_ledger_fingerprints",
        ),
        "release_checks": (),
    },
    "witness_taped_mlp_update_replay": {
        "pipeline_claims": ("mlp_activation_mask_witnesses",),
        "gate_metrics": ("mlp_witness_replay",),
        "release_checks": (),
    },
    "invertible_zero_witness_layer_patterns": {
        "pipeline_claims": ("invertible_layer_without_witness",),
        "gate_metrics": (
            "invertible_coupling_replay",
            "triangular_residual_replay",
            "reversible_preprocess_replay",
        ),
        "release_checks": (),
    },
    "speed_memory_trace_reverse_scorecard": {
        "pipeline_claims": ("memory_recompute_profile", "v6_ml_audit_scorecard"),
        "gate_metrics": (
            "run_peak_rss_bytes",
            "trace_to_model_payload_ratio",
            "witness_to_model_payload_ratio",
            "reverse_check_cost_measured",
            "v6_scorecard_complete",
        ),
        "release_checks": ("benchmark_speed_memory",),
    },
}
GOAL_CONTRACT_CLAIM_REPLAY_COMMANDS = {
    "deterministic_q31_inference_replay": (
        "inference_action_receipt",
        "capsule_handoff",
        "reversible_inference_trace_proof",
    ),
    "reversible_inference_roundtrip": (
        "roundtrip_proof",
        "reversible_inference_trace_proof",
    ),
    "training_lineage_replay": (
        "profile_evidence",
        "capsule_handoff",
        "training_audit_lineage",
    ),
    "training_step_debug_replay": ("training_update_receipt", "training_step_debug"),
    "witness_taped_mlp_update_replay": ("mlp_witness_proof",),
    "invertible_zero_witness_layer_patterns": (
        "invertible_coupling_proof",
        "triangular_residual_proof",
        "reversible_preprocess_proof",
    ),
    "speed_memory_trace_reverse_scorecard": (
        "profile_evidence",
        "benchmark_speed_memory",
    ),
}
GOAL_CONTRACT_CLAIM_REPLAY_ARTIFACTS = {
    "deterministic_q31_inference_replay": (
        "inference_action_review_receipt",
        "inference_action_review_receipt_markdown",
        "q31_reference_inference",
        "q31_reference_inference_markdown",
        "imported_model_source",
        "imported_model_bundle",
        "imported_model_import",
        "imported_model_verification",
        "imported_model_inference",
        "imported_model_inference_bundle",
        "imported_model_inference_verification",
        "native_inference_audit_markdown",
        "native_inference_verification_markdown",
        "evaluation_row_inference_audit_markdown",
        "inference_trace_forward",
        "inference_trace_reverse",
        "reversible_inference_trace",
        "reversible_inference_trace_markdown",
    ),
    "reversible_inference_roundtrip": (
        "roundtrip_proof",
        "roundtrip_verification",
        "roundtrip_markdown",
        "inference_trace_forward",
        "inference_trace_reverse",
        "reversible_inference_trace",
        "reversible_inference_trace_markdown",
    ),
    "training_lineage_replay": (
        "run_report",
        "audit_bundle",
        "audit_verification",
        "training_audit_verification_markdown",
        "summary",
        "capsule",
        "manifest",
    ),
    "training_step_debug_replay": (
        "training_update_review_receipt",
        "training_update_review_receipt_markdown",
        "training_step_bundle",
        "training_step_debug",
    ),
    "witness_taped_mlp_update_replay": (
        "mlp_vars",
        "mlp_run_output",
        "mlp_witness",
        "mlp_witness_markdown",
    ),
    "invertible_zero_witness_layer_patterns": (
        "coupling_forward",
        "coupling_reverse",
        "invertible_coupling",
        "invertible_coupling_markdown",
        "residual_forward",
        "residual_reverse",
        "triangular_residual",
        "triangular_residual_markdown",
        "preprocess_forward",
        "preprocess_reverse",
        "reversible_preprocess",
        "reversible_preprocess_markdown",
    ),
    "speed_memory_trace_reverse_scorecard": (
        "summary",
        "capsule",
        "manifest",
        "benchmark_json",
        "benchmark_markdown",
    ),
}
REVIEWER_COMMAND_SCHEMA = "reverie_ml_north_star_reviewer_commands_v1"
REQUIRED_RELEASE_REPLAY_COMMANDS = (
    "profile_evidence",
    "capsule_handoff",
    "training_audit_lineage",
    "benchmark_speed_memory",
    "roundtrip_proof",
    "training_step_debug",
    "training_update_receipt",
    "mlp_witness_proof",
    "invertible_coupling_proof",
    "triangular_residual_proof",
    "reversible_preprocess_proof",
    "reversible_inference_trace_proof",
    "inference_action_receipt",
)
RELEASE_VERIFIER_SOURCE_FILES = (
    "scripts/check_ml_north_star.py",
    "scripts/verify_model_capsule.py",
    "scripts/check_mnist_ml_profile.py",
    "scripts/check_triangular_residual.py",
    "scripts/check_benchmark_artifact.py",
)
RELEASE_PACKET_ARTIFACT_IDS = (
    "gate_json",
    "gate_markdown",
    "reviewer_transcript_json",
    "reviewer_transcript_markdown",
    "benchmark_json",
    "benchmark_markdown",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the ML audit capsule, handoff, reversible inference "
            "roundtrip, and Jana/Reverie speed+memory benchmark as one release gate."
        )
    )
    parser.add_argument(
        "audit_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_AUDIT_DIR,
        help="MNIST ML audit pipeline output directory.",
    )
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        default=DEFAULT_BENCHMARK_JSON,
        help="Jana-vs-Reverie speed+memory benchmark JSON artifact.",
    )
    parser.add_argument(
        "--benchmark-markdown",
        type=Path,
        default=DEFAULT_BENCHMARK_MARKDOWN,
        help="Rendered benchmark Markdown summary to compare against the JSON artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"Write the machine-readable gate report. Defaults to {DEFAULT_REPORT_NAME} in audit_dir.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help=f"Write a human-readable gate report. Defaults to {DEFAULT_MARKDOWN_NAME} in audit_dir.",
    )
    parser.add_argument(
        "--min-workloads",
        type=int,
        default=41,
        help="Minimum benchmark workloads required.",
    )
    parser.add_argument(
        "--expect-runs",
        type=int,
        default=5,
        help="Expected measured benchmark runs per workload.",
    )
    parser.add_argument(
        "--expect-warmup",
        type=int,
        default=1,
        help="Expected benchmark warmup runs per workload.",
    )
    parser.add_argument(
        "--min-observed-speedup",
        type=float,
        default=2.0,
        help="Minimum speedup required for every benchmark workload.",
    )
    parser.add_argument(
        "--min-median-speedup",
        type=float,
        default=3.0,
        help="Minimum median suite speedup.",
    )
    parser.add_argument(
        "--min-geomean-speedup",
        type=float,
        default=3.0,
        help="Minimum geometric-mean suite speedup.",
    )
    parser.add_argument(
        "--min-observed-rss-ratio",
        type=float,
        default=1.0,
        help="Minimum Jana/Reverie median peak RSS ratio for every workload.",
    )
    parser.add_argument(
        "--min-median-rss-ratio",
        type=float,
        default=1.0,
        help="Minimum median Jana/Reverie peak RSS ratio.",
    )
    parser.add_argument(
        "--skip-file-digests",
        action="store_true",
        help="Do not recompute benchmark binary/source digests or pipeline file evidence.",
    )
    parser.add_argument(
        "--allow-stale-workload-list",
        action="store_true",
        help="Do not require the benchmark artifact to match the current harness workload list.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the machine-readable gate report.",
    )
    parser.add_argument(
        "--list-reviewer-commands",
        action="store_true",
        help="List reviewer replay command ids and exit after validating the gate.",
    )
    parser.add_argument(
        "--run-reviewer-command",
        action="append",
        default=[],
        metavar="ID",
        help="Run a named reviewer replay command from the validated gate. Can be repeated.",
    )
    parser.add_argument(
        "--reviewer-transcript-output",
        type=Path,
        help=(
            "Write reviewer replay transcript JSON. Defaults to "
            f"{DEFAULT_TRANSCRIPT_NAME} in audit_dir when reviewer commands run."
        ),
    )
    parser.add_argument(
        "--reviewer-transcript-markdown",
        type=Path,
        help=(
            "Write reviewer replay transcript Markdown. Defaults to "
            f"{DEFAULT_TRANSCRIPT_MARKDOWN_NAME} in audit_dir when reviewer commands run."
        ),
    )
    parser.add_argument(
        "--verify-reviewer-transcript",
        action="store_true",
        help=(
            "Validate the saved reviewer replay transcript JSON and Markdown "
            "against the current gate and reviewer command index."
        ),
    )
    parser.add_argument(
        "--release-verification-output",
        type=Path,
        help=(
            "Write a combined gate+transcript release verification JSON. Defaults to "
            f"{DEFAULT_RELEASE_VERIFICATION_NAME} in audit_dir when --verify-reviewer-transcript is used."
        ),
    )
    parser.add_argument(
        "--release-verification-markdown",
        type=Path,
        help=(
            "Write a combined gate+transcript release verification Markdown summary. Defaults to "
            f"{DEFAULT_RELEASE_VERIFICATION_MARKDOWN_NAME} in audit_dir when --verify-reviewer-transcript is used."
        ),
    )
    parser.add_argument(
        "--verify-release-verification",
        action="store_true",
        help=(
            "Validate the saved release verification JSON and Markdown against "
            "the current gate and saved reviewer transcript."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent gate self-tests and exit.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"failed to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"failed to parse {path}: {error}") from error


def number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def resolve_audit_paths(audit_dir: Path) -> dict[str, Optional[Path]]:
    return {
        "summary": audit_dir / capsule_check.DEFAULT_SUMMARY_NAME,
        "capsule": audit_dir / capsule_check.DEFAULT_CAPSULE_NAME,
        "manifest": audit_dir / capsule_check.DEFAULT_MANIFEST_NAME,
        "profile": audit_dir / capsule_check.DEFAULT_PROFILE_NAME,
        "verification_markdown": audit_dir / capsule_check.DEFAULT_VERIFICATION_MARKDOWN_NAME,
        "handoff": audit_dir / capsule_check.DEFAULT_HANDOFF_NAME,
        "handoff_markdown": audit_dir / capsule_check.DEFAULT_HANDOFF_MARKDOWN_NAME,
    }


def validate_capsule_gate(
    audit_dir: Path,
    *,
    verify_file_evidence: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = resolve_audit_paths(audit_dir)
    result = capsule_check.verify_capsule(
        paths,
        verify_file_evidence=verify_file_evidence,
        allow_missing_profile=False,
    )
    report = capsule_check.verification_report(
        result,
        verify_file_evidence=verify_file_evidence,
        allow_missing_profile=False,
        verification_markdown_path=paths["verification_markdown"],
        require_verification_markdown=True,
        handoff_path=paths["handoff"],
        handoff_markdown_path=paths["handoff_markdown"],
        require_handoff=True,
    )
    if report["passed"] is not True:
        raise ValueError("model capsule trust certificate failed")
    return result, report


def validate_inference_action_review_receipt(
    audit_dir: Path,
    capsule_report: dict[str, Any],
) -> dict[str, Any]:
    receipt_path = audit_dir / capsule_check.DEFAULT_ACTION_REVIEW_RECEIPT_NAME
    markdown_path = audit_dir / capsule_check.DEFAULT_ACTION_REVIEW_RECEIPT_MARKDOWN_NAME
    receipt = load_json(receipt_path)
    if not isinstance(receipt, dict):
        raise ValueError("inference action receipt must be a JSON object")
    if receipt.get("kind") != capsule_check.ACTION_REVIEW_RECEIPT_KIND:
        raise ValueError(
            f"inference action receipt kind must be {capsule_check.ACTION_REVIEW_RECEIPT_KIND}"
        )
    if receipt.get("algorithm") != "sha256":
        raise ValueError("inference action receipt algorithm must be sha256")
    payload = receipt.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("inference action receipt payload must be an object")
    if payload.get("schema") != capsule_check.ACTION_REVIEW_RECEIPT_SCHEMA:
        raise ValueError(
            "inference action receipt schema must be "
            f"{capsule_check.ACTION_REVIEW_RECEIPT_SCHEMA}"
        )
    if receipt.get("fingerprint") != sha256_json(payload):
        raise ValueError("inference action receipt fingerprint does not match payload")
    semantic_fingerprint = payload.get("semantic_fingerprint")
    if not is_sha256_hex(semantic_fingerprint):
        raise ValueError("inference action receipt semantic_fingerprint must be SHA-256")
    if semantic_fingerprint != capsule_check.inference_action_review_semantic_fingerprint(payload):
        raise ValueError("inference action receipt semantic_fingerprint does not match payload")
    if receipt.get("passed") is not True:
        raise ValueError("inference action receipt did not pass")
    if payload.get("handoff_fingerprint") != capsule_report["handoff"]["fingerprint"]:
        raise ValueError("inference action receipt handoff fingerprint mismatch")
    if payload.get("capsule_fingerprint") != capsule_report["capsule"]["fingerprint"]:
        raise ValueError("inference action receipt capsule fingerprint mismatch")
    if (
        payload.get("trust_certificate_fingerprint")
        != capsule_report["trust_certificate"]["fingerprint"]
    ):
        raise ValueError("inference action receipt trust-certificate fingerprint mismatch")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("inference action receipt summary must be an object")
    if summary.get("passed") is not True:
        raise ValueError("inference action receipt summary did not pass")
    operations = payload.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("inference action receipt operations must be a non-empty list")
    operation_names = [row.get("operation") for row in operations if isinstance(row, dict)]
    expected_operations = [
        "reproduce_prediction",
        "explain_margin",
        "replay_imported_model_inference",
        "replay_native_inference",
        "run_standalone_rev_classifier",
        "reverse_reversible_trace",
    ]
    if operation_names != expected_operations:
        raise ValueError("inference action receipt operations changed")
    if summary.get("command_count") != len(expected_operations):
        raise ValueError("inference action receipt command count changed")
    if summary.get("passed_count") != len(expected_operations) or summary.get("failed_count") != 0:
        raise ValueError("inference action receipt must pass every command")
    if summary.get("failed_operations") not in ([], ()):
        raise ValueError("inference action receipt lists failed operations")
    for index, row in enumerate(operations):
        context = f"inference action receipt operations[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{context} must be an object")
        if row.get("passed") is not True:
            raise ValueError(f"{context} did not pass")
        if row.get("exit_code") != 0:
            raise ValueError(f"{context} exit code must be 0")
        for field in ("stdout_sha256", "stderr_sha256"):
            if not is_sha256_hex(row.get(field)):
                raise ValueError(f"{context}.{field} must be a SHA-256 hex digest")
        if not number(row.get("elapsed_seconds")) or row["elapsed_seconds"] < 0:
            raise ValueError(f"{context}.elapsed_seconds must be non-negative")
    try:
        markdown = markdown_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"failed to read inference action receipt Markdown: {error}") from error
    expected_markdown = capsule_check.render_inference_action_review_receipt_markdown(receipt)
    if markdown != expected_markdown:
        raise ValueError("inference action receipt Markdown is stale")
    return {
        "path": str(receipt_path),
        "markdown": str(markdown_path),
        "fingerprint": receipt["fingerprint"],
        "semantic_fingerprint": semantic_fingerprint,
        "markdown_sha256": sha256_text(markdown),
        "passed": True,
        "command_count": summary["command_count"],
        "passed_count": summary["passed_count"],
        "failed_count": summary["failed_count"],
        "operations": operation_names,
    }


def validate_training_update_review_receipt(
    audit_dir: Path,
    capsule_report: dict[str, Any],
    debug_update: dict[str, Any],
) -> dict[str, Any]:
    receipt_path = audit_dir / audit_pipeline.DEFAULT_TRAINING_UPDATE_REVIEW_RECEIPT_NAME
    markdown_path = (
        audit_dir / audit_pipeline.DEFAULT_TRAINING_UPDATE_REVIEW_RECEIPT_MARKDOWN_NAME
    )
    receipt = load_json(receipt_path)
    if not isinstance(receipt, dict):
        raise ValueError("training update receipt must be a JSON object")
    if receipt.get("kind") != audit_pipeline.TRAINING_UPDATE_REVIEW_RECEIPT_KIND:
        raise ValueError(
            "training update receipt kind must be "
            f"{audit_pipeline.TRAINING_UPDATE_REVIEW_RECEIPT_KIND}"
        )
    if receipt.get("algorithm") != "sha256":
        raise ValueError("training update receipt algorithm must be sha256")
    payload = receipt.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("training update receipt payload must be an object")
    if payload.get("schema") != audit_pipeline.TRAINING_UPDATE_REVIEW_RECEIPT_SCHEMA:
        raise ValueError(
            "training update receipt schema must be "
            f"{audit_pipeline.TRAINING_UPDATE_REVIEW_RECEIPT_SCHEMA}"
        )
    if receipt.get("fingerprint") != sha256_json(payload):
        raise ValueError("training update receipt fingerprint does not match payload")
    semantic_fingerprint = payload.get("semantic_fingerprint")
    if not is_sha256_hex(semantic_fingerprint):
        raise ValueError("training update receipt semantic_fingerprint must be SHA-256")
    if semantic_fingerprint != audit_pipeline.training_update_review_semantic_fingerprint(payload):
        raise ValueError("training update receipt semantic_fingerprint does not match payload")
    if receipt.get("passed") is not True:
        raise ValueError("training update receipt did not pass")
    if payload.get("capsule_fingerprint") != capsule_report["capsule"]["fingerprint"]:
        raise ValueError("training update receipt capsule fingerprint mismatch")
    if payload.get("handoff_fingerprint") != capsule_report["handoff"]["fingerprint"]:
        raise ValueError("training update receipt handoff fingerprint mismatch")
    selected = payload.get("selected_training_update")
    if not isinstance(selected, dict):
        raise ValueError("training update receipt selected_training_update must be an object")
    for field in ("step", "sample_index", "prediction", "active_pixels"):
        if selected.get(field) != debug_update.get(field if field != "active_pixels" else "active_pixel_count"):
            raise ValueError(f"training update receipt selected {field} mismatch")
    if selected.get("correct") != debug_update.get("correct"):
        raise ValueError("training update receipt selected correct mismatch")
    for field in (
        "cause_ledger_fingerprint",
        "bias_delta_ledger_fingerprint",
        "weight_delta_ledger_fingerprint",
        "step_verification_fingerprint",
        "proof_fingerprint",
    ):
        if not is_sha256_hex(selected.get(field)):
            raise ValueError(f"training update receipt selected {field} must be SHA-256")
    for field in (
        "cause_ledger_fingerprint",
        "bias_delta_ledger_fingerprint",
        "weight_delta_ledger_fingerprint",
    ):
        if selected.get(field) != debug_update.get(field):
            raise ValueError(f"training update receipt selected {field} mismatch")
    summary = payload.get("summary")
    if not isinstance(summary, dict) or summary.get("passed") is not True:
        raise ValueError("training update receipt summary did not pass")
    operations = payload.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("training update receipt operations must be a non-empty list")
    operation_names = [row.get("operation") for row in operations if isinstance(row, dict)]
    expected_operations = [
        "replay_training_lineage",
        "inspect_training_update",
        "reverse_training_update",
    ]
    if operation_names != expected_operations:
        raise ValueError("training update receipt operations changed")
    if summary.get("command_count") != len(expected_operations):
        raise ValueError("training update receipt command count changed")
    if summary.get("passed_count") != len(expected_operations) or summary.get("failed_count") != 0:
        raise ValueError("training update receipt must pass every command")
    if summary.get("failed_operations") not in ([], ()):
        raise ValueError("training update receipt lists failed operations")
    for index, row in enumerate(operations):
        context = f"training update receipt operations[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{context} must be an object")
        if row.get("passed") is not True:
            raise ValueError(f"{context} did not pass")
        if row.get("exit_code") != 0:
            raise ValueError(f"{context} exit code must be 0")
        for field in ("stdout_sha256", "stderr_sha256"):
            if not is_sha256_hex(row.get(field)):
                raise ValueError(f"{context}.{field} must be a SHA-256 hex digest")
        if not number(row.get("elapsed_seconds")) or row["elapsed_seconds"] < 0:
            raise ValueError(f"{context}.elapsed_seconds must be non-negative")
    try:
        markdown = markdown_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"failed to read training update receipt Markdown: {error}") from error
    expected_markdown = audit_pipeline.render_training_update_review_receipt_markdown(receipt)
    if markdown != expected_markdown:
        raise ValueError("training update receipt Markdown is stale")
    return {
        "path": str(receipt_path),
        "markdown": str(markdown_path),
        "fingerprint": receipt["fingerprint"],
        "semantic_fingerprint": semantic_fingerprint,
        "markdown_sha256": sha256_text(markdown),
        "passed": True,
        "command_count": summary["command_count"],
        "passed_count": summary["passed_count"],
        "failed_count": summary["failed_count"],
        "operations": operation_names,
        "selected_step": selected["step"],
        "sample_index": selected["sample_index"],
    }


def validate_summary_goal(summary: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    metrics = summary.get("metrics")
    gates = summary.get("gates")
    claims = summary.get("claims")
    if not isinstance(metrics, dict):
        raise ValueError("pipeline summary metrics must be an object")
    if not isinstance(gates, dict):
        raise ValueError("pipeline summary gates must be an object")
    if not isinstance(claims, dict):
        raise ValueError("pipeline summary claims must be an object")

    capability = metrics.get("ml_capability_map")
    readiness = metrics.get("ml_goal_readiness")
    scorecard = metrics.get("scorecard")
    require(isinstance(capability, dict), "ml_capability_map must be present", errors)
    require(isinstance(readiness, dict), "ml_goal_readiness must be present", errors)
    require(isinstance(scorecard, dict), "scorecard must be present", errors)
    if errors:
        raise ValueError("; ".join(errors))
    assert isinstance(capability, dict)
    assert isinstance(readiness, dict)
    assert isinstance(scorecard, dict)

    capability_summary = capability.get("summary")
    readiness_summary = readiness.get("summary")
    require(capability.get("passed") is True, "V1-V6 capability map did not pass", errors)
    require(readiness.get("passed") is True, "ML goal readiness did not pass", errors)
    require(scorecard.get("gates_passed") is True, "V6 scorecard gates did not pass", errors)
    require(gates.get("passed") is True, "pipeline summary gates did not pass", errors)
    require(claims.get("passed") is True, "pipeline summary claims did not pass", errors)
    require(readiness.get("north_star") == NORTH_STAR, "north-star string changed", errors)
    require(readiness.get("non_goal") == NON_GOAL, "non-goal string changed", errors)
    if isinstance(capability_summary, dict):
        require(
            capability_summary.get("passed") == capability_summary.get("total") == 6,
            "expected all 6 roadmap capabilities to pass",
            errors,
        )
        require(capability_summary.get("failed") == 0, "roadmap capabilities have failures", errors)
    else:
        errors.append("ml_capability_map.summary must be an object")
    if isinstance(readiness_summary, dict):
        require(
            readiness_summary.get("passed") == readiness_summary.get("total") == 5,
            "expected all 5 north-star readiness goals to pass",
            errors,
        )
        require(readiness_summary.get("failed") == 0, "north-star readiness has failures", errors)
    else:
        errors.append("ml_goal_readiness.summary must be an object")

    capabilities = capability.get("capabilities")
    if not isinstance(capabilities, list):
        errors.append("ml_capability_map.capabilities must be a list")
        phases: list[str] = []
    else:
        phases = []
        for entry in capabilities:
            if not isinstance(entry, dict):
                errors.append("each roadmap capability must be an object")
                continue
            phases.append(str(entry.get("phase")))
            require(entry.get("passed") is True, f"{entry.get('phase')} capability failed", errors)
            require(entry.get("blocking_gates") == [], f"{entry.get('phase')} has blocking gates", errors)
        require(tuple(phases) == EXPECTED_PHASES, "roadmap phases must be V1 through V6", errors)

    for field in (
        "run_peak_rss_bytes",
        "max_replay_payload_bytes",
        "total_recompute_steps",
        "total_forward_recompute_steps",
        "total_inverse_recompute_steps",
    ):
        require(
            isinstance(scorecard.get(field), int) and scorecard[field] > 0,
            f"scorecard.{field} must be positive",
            errors,
        )
    require(scorecard.get("balanced_recompute") is True, "scorecard recompute is not balanced", errors)
    require(
        number(scorecard.get("trace_to_model_payload_ratio"))
        and scorecard["trace_to_model_payload_ratio"] <= gates["policy"]["max_trace_to_model_payload_ratio"],
        "trace/model ratio exceeds policy",
        errors,
    )
    require(
        number(scorecard.get("witness_to_model_payload_ratio"))
        and scorecard["witness_to_model_payload_ratio"] <= gates["policy"]["max_witness_to_model_payload_ratio"],
        "witness/model ratio exceeds policy",
        errors,
    )
    require(
        number(scorecard.get("reverse_to_train_elapsed_ratio"))
        and scorecard["reverse_to_train_elapsed_ratio"] <= gates["policy"]["max_reverse_train_elapsed_ratio"],
        "reverse/train elapsed ratio exceeds policy",
        errors,
    )
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "capabilities": {
            "passed": capability_summary["passed"],
            "total": capability_summary["total"],
            "phases": phases,
        },
        "readiness": {
            "passed": readiness_summary["passed"],
            "total": readiness_summary["total"],
        },
        "scorecard": {
            "run_peak_rss_bytes": scorecard["run_peak_rss_bytes"],
            "max_replay_payload_bytes": scorecard["max_replay_payload_bytes"],
            "trace_to_model_payload_ratio": scorecard["trace_to_model_payload_ratio"],
            "witness_to_model_payload_ratio": scorecard["witness_to_model_payload_ratio"],
            "reverse_to_train_elapsed_ratio": scorecard["reverse_to_train_elapsed_ratio"],
            "total_recompute_steps": scorecard["total_recompute_steps"],
        },
    }


def validate_debug_training_update(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics")
    gates = summary.get("gates")
    claims = summary.get("claims")
    if not isinstance(metrics, dict):
        raise ValueError("pipeline summary metrics must be an object")
    if not isinstance(gates, dict):
        raise ValueError("pipeline summary gates must be an object")
    if not isinstance(claims, dict):
        raise ValueError("pipeline summary claims must be an object")
    debug = metrics.get("training_step_debug")
    selection = metrics.get("training_step_selection")
    replay = metrics.get("training_step_replay")
    if not isinstance(debug, dict):
        raise ValueError("metrics.training_step_debug must be an object")
    if not isinstance(selection, dict):
        raise ValueError("metrics.training_step_selection must be an object")
    if not isinstance(replay, dict):
        raise ValueError("metrics.training_step_replay must be an object")

    errors: list[str] = []
    gate_checks = gates.get("checks")
    claim_checks = claims.get("checks")
    require(isinstance(gate_checks, list), "pipeline gates.checks must be a list", errors)
    require(isinstance(claim_checks, list), "pipeline claims.checks must be a list", errors)
    if isinstance(gate_checks, list):
        gate_results = {
            check["metric"]: check["passed"]
            for check in gate_checks
            if isinstance(check, dict)
            and isinstance(check.get("metric"), str)
            and isinstance(check.get("passed"), bool)
        }
        for metric in (
            "training_step_selection_traceable",
            "training_step_replay",
            "training_step_debug_contract",
            "training_update_ledger_fingerprints",
        ):
            require(gate_results.get(metric) is True, f"debug update gate `{metric}` did not pass", errors)
    if isinstance(claim_checks, list):
        debug_claim = next(
            (
                check
                for check in claim_checks
                if isinstance(check, dict) and check.get("claim") == "debug_training_update"
            ),
            None,
        )
        require(
            isinstance(debug_claim, dict) and debug_claim.get("passed") is True,
            "debug_training_update claim did not pass",
            errors,
        )

    require(debug.get("passed") is True, "training_step_debug did not pass", errors)
    require(
        debug.get("claim") == "step_backward_from_model_update",
        "training_step_debug claim changed",
        errors,
    )
    require(
        selection.get("matches_selection_strategy") is True
        and selection.get("matches_requested_step") is True,
        "training step selection no longer matches request/strategy",
        errors,
    )
    require(
        selection.get("selected_step") == debug.get("step")
        and selection.get("selected_sample_index") == debug.get("sample_index"),
        "training step selection does not match debug record",
        errors,
    )
    require(
        isinstance(selection.get("selection_reasons"), list)
        and bool(selection["selection_reasons"]),
        "training step selection reasons must be present",
        errors,
    )
    require(
        isinstance(selection.get("selected_max_abs_weight_delta"), int)
        and not isinstance(selection.get("selected_max_abs_weight_delta"), bool)
        and selection["selected_max_abs_weight_delta"] > 0,
        "training_step_selection.selected_max_abs_weight_delta must be positive",
        errors,
    )
    for field in (
        "proof_matches",
        "witnesses_match",
        "after_model_matches",
        "before_model_restored",
    ):
        require(replay.get(field) is True, f"training_step_replay.{field} did not pass", errors)
    require(
        isinstance(replay.get("replay_payload_bytes"), int)
        and replay["replay_payload_bytes"] > 0,
        "training_step_replay.replay_payload_bytes must be positive",
        errors,
    )

    for field in ("step", "sample_index", "prediction", "active_pixel_count"):
        require(
            isinstance(debug.get(field), int) and not isinstance(debug.get(field), bool),
            f"training_step_debug.{field} must be int",
            errors,
        )
    for field in (
        "top_weight_deltas",
        "nonzero_bias_delta_count",
        "nonzero_weight_delta_count",
    ):
        require(
            isinstance(debug.get(field), int)
            and not isinstance(debug.get(field), bool)
            and debug[field] > 0,
            f"training_step_debug.{field} must be positive",
            errors,
        )
    require(
        isinstance(debug.get("reversed_later_steps"), int)
        and not isinstance(debug.get("reversed_later_steps"), bool)
        and debug["reversed_later_steps"] >= 0,
        "training_step_debug.reversed_later_steps must be non-negative",
        errors,
    )
    require(isinstance(debug.get("correct"), bool), "training_step_debug.correct must be bool", errors)
    for field in (
        "bias_delta_ledger_fingerprint",
        "weight_delta_ledger_fingerprint",
        "cause_ledger_fingerprint",
    ):
        require(is_sha256_hex(debug.get(field)), f"training_step_debug.{field} must be SHA-256", errors)
    required_checks = {
        "witness_recomputes_prediction",
        "model_window_reconstructed",
        "update_matches_model_window",
        "explanatory_state_present",
        "update_ledger_fingerprints",
    }
    checks = debug.get("checks")
    if not isinstance(checks, list):
        errors.append("training_step_debug.checks must be a list")
        passed_checks: list[str] = []
    else:
        passed_checks = [
            check["metric"]
            for check in checks
            if isinstance(check, dict)
            and isinstance(check.get("metric"), str)
            and check.get("passed") is True
        ]
        missing_checks = sorted(required_checks - set(passed_checks))
        if missing_checks:
            errors.append(
                "training_step_debug missing passing check(s): " + ", ".join(missing_checks)
            )
    if errors:
        raise ValueError("; ".join(errors))

    return {
        "claim": debug["claim"],
        "step": debug["step"],
        "sample_index": debug["sample_index"],
        "prediction": debug["prediction"],
        "correct": debug["correct"],
        "margin": debug.get("margin"),
        "active_pixel_count": debug["active_pixel_count"],
        "top_weight_deltas": debug["top_weight_deltas"],
        "nonzero_bias_delta_count": debug["nonzero_bias_delta_count"],
        "nonzero_weight_delta_count": debug["nonzero_weight_delta_count"],
        "reversed_later_steps": debug["reversed_later_steps"],
        "replay_payload_bytes": replay["replay_payload_bytes"],
        "selected_max_abs_weight_delta": selection.get("selected_max_abs_weight_delta"),
        "selection_reasons": selection["selection_reasons"],
        "checks": sorted(passed_checks),
        "cause_ledger_fingerprint": debug["cause_ledger_fingerprint"],
        "bias_delta_ledger_fingerprint": debug["bias_delta_ledger_fingerprint"],
        "weight_delta_ledger_fingerprint": debug["weight_delta_ledger_fingerprint"],
    }


def validate_training_lineage(audit_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics")
    gates = summary.get("gates")
    claims = summary.get("claims")
    if not isinstance(metrics, dict):
        raise ValueError("pipeline summary metrics must be an object")
    if not isinstance(gates, dict):
        raise ValueError("pipeline summary gates must be an object")
    if not isinstance(claims, dict):
        raise ValueError("pipeline summary claims must be an object")
    replay = metrics.get("training_audit_replay")
    reverse = metrics.get("reverse")
    train = metrics.get("train")
    artifact_profile = metrics.get("artifact_profile")
    if not isinstance(replay, dict):
        raise ValueError("metrics.training_audit_replay must be an object")
    if not isinstance(reverse, dict):
        raise ValueError("metrics.reverse must be an object")
    if not isinstance(train, dict):
        raise ValueError("metrics.train must be an object")
    if not isinstance(artifact_profile, dict):
        raise ValueError("metrics.artifact_profile must be an object")

    verification_path = audit_dir / "training-audit-verification.json"
    verification = load_json(verification_path)
    if not isinstance(verification, dict):
        raise ValueError("training-audit-verification.json must be an object")
    lineage_ledger = verification.get("lineage_ledger")
    if not isinstance(lineage_ledger, dict):
        raise ValueError("training audit verification lineage_ledger must be an object")
    lineage_payload = lineage_ledger.get("payload")
    if not isinstance(lineage_payload, dict):
        raise ValueError("training audit verification lineage_ledger.payload must be an object")

    errors: list[str] = []
    gate_checks = gates.get("checks")
    claim_checks = claims.get("checks")
    require(isinstance(gate_checks, list), "pipeline gates.checks must be a list", errors)
    require(isinstance(claim_checks, list), "pipeline claims.checks must be a list", errors)
    if isinstance(gate_checks, list):
        gate_results = {
            check["metric"]: check["passed"]
            for check in gate_checks
            if isinstance(check, dict)
            and isinstance(check.get("metric"), str)
            and isinstance(check.get("passed"), bool)
        }
        for metric in (
            "reverse_restored_initial_model",
            "reverse_checked_all_training_steps",
            "training_audit_lineage_replay",
        ):
            require(gate_results.get(metric) is True, f"lineage gate `{metric}` did not pass", errors)
    if isinstance(claim_checks, list):
        lineage_claim = next(
            (
                check
                for check in claim_checks
                if isinstance(check, dict) and check.get("claim") == "auditable_model_lineage"
            ),
            None,
        )
        require(
            isinstance(lineage_claim, dict) and lineage_claim.get("passed") is True,
            "auditable_model_lineage claim did not pass",
            errors,
        )

    require(
        isinstance(train.get("samples"), int)
        and not isinstance(train.get("samples"), bool)
        and train["samples"] > 0,
        "train.samples must be positive",
        errors,
    )
    if isinstance(train.get("samples"), int) and not isinstance(train.get("samples"), bool):
        require(replay.get("checked") == train["samples"], "training replay must cover all train samples", errors)
        require(reverse.get("checked") == train["samples"], "reverse replay must cover all train samples", errors)
    for field in (
        "witnesses_match_forward_replay",
        "final_model_replayed",
        "restored_initial_model",
        "proof_matches",
        "lineage_ledger_matches",
    ):
        require(replay.get(field) is True, f"training_audit_replay.{field} must be true", errors)
        require(verification.get(field) is True, f"training audit verification {field} must be true", errors)
    require(reverse.get("restored_initial_model") is True, "reverse restored initial model must be true", errors)
    for field in ("lineage_ledger_fingerprint", "transition_ledger_fingerprint", "final_chain"):
        require(is_sha256_hex(replay.get(field)), f"training_audit_replay.{field} must be SHA-256", errors)
    for field in (
        "initial_model_fingerprint",
        "final_model_fingerprint",
        "sample_order_fingerprint",
        "witness_trace_fingerprint",
    ):
        require(is_sha256_hex(lineage_payload.get(field)), f"lineage_ledger.payload.{field} must be SHA-256", errors)
    require(
        lineage_ledger.get("fingerprint") == replay.get("lineage_ledger_fingerprint"),
        "lineage ledger fingerprint must match summary",
        errors,
    )
    require(
        lineage_payload.get("transition_ledger_fingerprint")
        == replay.get("transition_ledger_fingerprint"),
        "transition ledger fingerprint must match summary",
        errors,
    )
    require(
        lineage_payload.get("final_chain") == replay.get("final_chain"),
        "final chain must match summary",
        errors,
    )
    require(
        lineage_payload.get("steps") == replay.get("checked"),
        "lineage ledger steps must match checked replay steps",
        errors,
    )
    require(
        isinstance(artifact_profile.get("total_model_payload_bytes"), int)
        and not isinstance(artifact_profile.get("total_model_payload_bytes"), bool)
        and artifact_profile["total_model_payload_bytes"] > 0,
        "artifact_profile.total_model_payload_bytes must be positive",
        errors,
    )
    require(
        isinstance(artifact_profile.get("total_witness_payload_bytes"), int)
        and not isinstance(artifact_profile.get("total_witness_payload_bytes"), bool)
        and artifact_profile["total_witness_payload_bytes"] > 0,
        "artifact_profile.total_witness_payload_bytes must be positive",
        errors,
    )
    if errors:
        raise ValueError("; ".join(errors))

    return {
        "checked_steps": replay["checked"],
        "train_samples": train["samples"],
        "final_model_replayed": replay["final_model_replayed"],
        "restored_initial_model": replay["restored_initial_model"],
        "witnesses_match_forward_replay": replay["witnesses_match_forward_replay"],
        "proof_matches": replay["proof_matches"],
        "lineage_ledger_matches": replay["lineage_ledger_matches"],
        "model_payload_bytes": artifact_profile["total_model_payload_bytes"],
        "witness_payload_bytes": artifact_profile["total_witness_payload_bytes"],
        "lineage_ledger_fingerprint": replay["lineage_ledger_fingerprint"],
        "transition_ledger_fingerprint": replay["transition_ledger_fingerprint"],
        "final_chain": replay["final_chain"],
        "initial_model_fingerprint": lineage_payload["initial_model_fingerprint"],
        "final_model_fingerprint": lineage_payload["final_model_fingerprint"],
        "sample_order_fingerprint": lineage_payload["sample_order_fingerprint"],
        "witness_trace_fingerprint": lineage_payload["witness_trace_fingerprint"],
    }


def validate_deterministic_inference(audit_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics")
    gates = summary.get("gates")
    claims = summary.get("claims")
    if not isinstance(metrics, dict):
        raise ValueError("pipeline summary metrics must be an object")
    if not isinstance(gates, dict):
        raise ValueError("pipeline summary gates must be an object")
    if not isinstance(claims, dict):
        raise ValueError("pipeline summary claims must be an object")
    q31 = metrics.get("q31_reference_inference")
    model_import = metrics.get("model_import")
    imported = metrics.get("imported_model_inference")
    native = metrics.get("native_inference_replay")
    standalone = metrics.get("native_standalone_rev_classifier")
    explanation = metrics.get("native_inference_explanation")
    profile = metrics.get("inference_trace_profile")
    trace = metrics.get("reversible_inference_trace")
    for name, value in (
        ("q31_reference_inference", q31),
        ("model_import", model_import),
        ("imported_model_inference", imported),
        ("native_inference_replay", native),
        ("native_standalone_rev_classifier", standalone),
        ("native_inference_explanation", explanation),
        ("inference_trace_profile", profile),
        ("reversible_inference_trace", trace),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"metrics.{name} must be an object")
    assert isinstance(q31, dict)
    assert isinstance(model_import, dict)
    assert isinstance(imported, dict)
    assert isinstance(native, dict)
    assert isinstance(standalone, dict)
    assert isinstance(explanation, dict)
    assert isinstance(profile, dict)
    assert isinstance(trace, dict)

    q31_report = load_json(audit_dir / "q31-reference-inference.json")
    imported_model_import_report = load_json(audit_dir / "imported-model-import-report.json")
    imported_model_verification_report = load_json(audit_dir / "imported-model-verification.json")
    imported_inference_report = load_json(audit_dir / "imported-model-inference-report.json")
    imported_verification_report = load_json(audit_dir / "imported-model-inference-verification.json")
    native_report = load_json(audit_dir / "native-inference-verification.json")
    trace_report = load_json(audit_dir / "reversible-inference-trace-report.json")
    if not isinstance(q31_report, dict):
        raise ValueError("q31-reference-inference.json must be an object")
    if not isinstance(imported_model_import_report, dict):
        raise ValueError("imported-model-import-report.json must be an object")
    if not isinstance(imported_model_verification_report, dict):
        raise ValueError("imported-model-verification.json must be an object")
    if not isinstance(imported_inference_report, dict):
        raise ValueError("imported-model-inference-report.json must be an object")
    if not isinstance(imported_verification_report, dict):
        raise ValueError("imported-model-inference-verification.json must be an object")
    if not isinstance(native_report, dict):
        raise ValueError("native-inference-verification.json must be an object")
    if not isinstance(trace_report, dict):
        raise ValueError("reversible-inference-trace-report.json must be an object")
    native_proof = native_report.get("proof")
    trace_proof = trace_report.get("proof")
    if not isinstance(native_proof, dict):
        raise ValueError("native inference verification proof must be an object")
    if not isinstance(trace_proof, dict):
        raise ValueError("reversible inference trace proof must be an object")

    errors: list[str] = []
    gate_checks = gates.get("checks")
    claim_checks = claims.get("checks")
    require(isinstance(gate_checks, list), "pipeline gates.checks must be a list", errors)
    require(isinstance(claim_checks, list), "pipeline claims.checks must be a list", errors)
    if isinstance(gate_checks, list):
        gate_results = {
            check["metric"]: check["passed"]
            for check in gate_checks
            if isinstance(check, dict)
            and isinstance(check.get("metric"), str)
            and isinstance(check.get("passed"), bool)
        }
        for metric in (
            "model_import_provenance",
            "imported_model_inference_replay",
            "native_inference_replay",
            "native_inference_explanation_contract",
            "q31_reference_matches_native_inference",
            "inference_trace_profile_complete",
            "reversible_inference_trace_replay",
        ):
            require(gate_results.get(metric) is True, f"inference gate `{metric}` did not pass", errors)
    if isinstance(claim_checks, list):
        inference_claim = next(
            (
                check
                for check in claim_checks
                if isinstance(check, dict) and check.get("claim") == "deterministic_q31_inference"
            ),
            None,
        )
        require(
            isinstance(inference_claim, dict) and inference_claim.get("passed") is True,
            "deterministic_q31_inference claim did not pass",
            errors,
        )

    for field in (
        "prediction_matches_native",
        "correct_matches_native",
        "margin_matches_native",
        "active_pixels_match_native",
        "contribution_ledger_matches_native",
        "margin_contribution_ledger_matches_native",
        "attribution_matches_logit",
        "attribution_matches_margin",
    ):
        require(q31.get(field) is True, f"q31_reference_inference.{field} must be true", errors)
    for field in ("prediction", "native_prediction", "margin", "native_margin", "active_pixels", "native_active_pixels"):
        require(
            isinstance(q31.get(field), int) and not isinstance(q31.get(field), bool),
            f"q31_reference_inference.{field} must be int",
            errors,
        )
    require(q31.get("prediction") == q31.get("native_prediction"), "q31 prediction must match native", errors)
    require(q31.get("margin") == q31.get("native_margin"), "q31 margin must match native", errors)
    require(q31.get("active_pixels") == q31.get("native_active_pixels"), "q31 active pixels must match native", errors)
    require(q31.get("correct") is True and q31.get("native_correct") is True, "q31/native correctness must be true", errors)
    for field in ("contribution_ledger_fingerprint", "margin_contribution_ledger_fingerprint"):
        require(is_sha256_hex(q31.get(field)), f"q31_reference_inference.{field} must be SHA-256", errors)
        require(q31.get(field) == q31.get(f"native_{field}"), f"q31/native {field} must match", errors)

    for field in (
        "import_shape_matches",
        "verification_shape_matches",
        "verification_provenance_matches",
        "verification_source_checked",
    ):
        require(model_import.get(field) is True, f"model_import.{field} must be true", errors)
    require(model_import.get("import_provenance_kind") == "external_import", "model import provenance must be external_import", errors)
    require(
        model_import.get("verification_provenance_kind") == "external_import",
        "model import verification provenance must be external_import",
        errors,
    )
    require(model_import.get("proof_matches") is None, "external model import proof_matches must be null", errors)
    require(model_import.get("training_steps") is None, "external model import training_steps must be null", errors)
    for field in (
        "source_model_json_fingerprint",
        "source_model_json_file_sha256",
        "import_payload_fingerprint",
        "verification_payload_fingerprint",
    ):
        require(is_sha256_hex(model_import.get(field)), f"model_import.{field} must be SHA-256", errors)
    require(
        model_import.get("import_payload_fingerprint") == model_import.get("verification_payload_fingerprint"),
        "model import payload fingerprints must match",
        errors,
    )
    require(
        imported_model_import_report.get("provenance_kind") == "external_import",
        "import report provenance must be external_import",
        errors,
    )
    require(
        imported_model_import_report.get("source_model_json_fingerprint")
        == model_import.get("source_model_json_fingerprint"),
        "import report source JSON fingerprint must match summary",
        errors,
    )
    require(
        imported_model_verification_report.get("provenance_kind") == "external_import",
        "import verification provenance must be external_import",
        errors,
    )
    require(
        imported_model_verification_report.get("source_model_json_checked") is True,
        "import verification source JSON check must pass",
        errors,
    )
    require(
        imported_model_verification_report.get("training_steps") is None,
        "import verification training_steps must be null",
        errors,
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
        require(imported.get(field) is True, f"imported_model_inference.{field} must be true", errors)
    require(imported.get("prediction") == q31.get("prediction"), "imported prediction must match Q31/native", errors)
    for field in ("replay_payload_bytes", "forward_recompute_steps", "inverse_recompute_steps"):
        require(
            isinstance(imported.get(field), int)
            and not isinstance(imported.get(field), bool)
            and imported[field] > 0,
            f"imported_model_inference.{field} must be positive",
            errors,
        )
    require(imported.get("forward_recompute_steps") == imported.get("inverse_recompute_steps"), "imported inference recompute should be balanced", errors)
    require(imported_inference_report.get("prediction") == imported.get("prediction"), "imported inference report prediction must match summary", errors)
    require(imported_inference_report.get("correct") == imported.get("correct"), "imported inference report correctness must match summary", errors)
    require(imported_verification_report.get("prediction") == imported.get("prediction"), "imported inference verification prediction must match summary", errors)
    require(imported_verification_report.get("proof_matches") is True, "imported inference verification proof must pass", errors)
    require(imported_verification_report.get("result_matches") is True, "imported inference verification result must pass", errors)
    require(
        imported_verification_report.get("restored_initial_state") is True,
        "imported inference verification reverse restoration must pass",
        errors,
    )
    imported_memory = imported_verification_report.get("memory")
    if isinstance(imported_memory, dict):
        require(
            imported_memory.get("replay_payload_bytes") == imported.get("replay_payload_bytes"),
            "imported inference memory replay bytes must match summary",
            errors,
        )
    else:
        errors.append("imported inference verification memory must be an object")

    for field in (
        "correct",
        "proof_matches",
        "result_matches",
        "restored_initial_state",
        "source_model_checked",
        "source_sample_checked",
    ):
        require(native.get(field) is True, f"native_inference_replay.{field} must be true", errors)
    require(
        isinstance(native.get("replay_payload_bytes"), int)
        and not isinstance(native.get("replay_payload_bytes"), bool)
        and native["replay_payload_bytes"] > 0,
        "native_inference_replay.replay_payload_bytes must be positive",
        errors,
    )
    require(explanation.get("passed") is True, "native inference explanation must pass", errors)
    require(
        explanation.get("prediction") == q31.get("prediction")
        and explanation.get("correct") == q31.get("correct")
        and explanation.get("margin") == q31.get("margin")
        and explanation.get("active_pixel_count") == q31.get("active_pixels"),
        "native inference explanation must match Q31 reference metrics",
        errors,
    )
    ledgers = explanation.get("ledger_fingerprints")
    if isinstance(ledgers, dict):
        require(ledgers.get("contribution") == q31.get("contribution_ledger_fingerprint"), "explanation contribution ledger mismatch", errors)
        require(
            ledgers.get("margin_contribution") == q31.get("margin_contribution_ledger_fingerprint"),
            "explanation margin contribution ledger mismatch",
            errors,
        )
    else:
        errors.append("native inference explanation ledger_fingerprints must be an object")
    replay_direction = explanation.get("replay_direction")
    if isinstance(replay_direction, dict):
        require(
            replay_direction.get("inverse_restores_initial_state") is True,
            "native inference explanation inverse restore check must pass",
            errors,
        )
    else:
        errors.append("native inference explanation replay_direction must be an object")
    for field in ("prediction", "correct", "margin", "active_pixels"):
        if field in q31_report:
            require(q31_report.get(field) == q31.get(field), f"q31 report {field} must match summary", errors)
    require(native_report.get("prediction") == q31.get("prediction"), "native report prediction must match summary", errors)
    require(native_report.get("correct") == q31.get("correct"), "native report correctness must match summary", errors)
    require(native_proof.get("prediction") == q31.get("prediction"), "native proof prediction must match summary", errors)
    require(native_proof.get("margin") == q31.get("margin"), "native proof margin must match summary", errors)
    require(native_proof.get("active_pixels") == q31.get("active_pixels"), "native proof active pixels must match summary", errors)
    require(
        native_proof.get("contribution_ledger_fingerprint") == q31.get("contribution_ledger_fingerprint"),
        "native proof contribution ledger must match summary",
        errors,
    )
    require(
        native_proof.get("margin_contribution_ledger_fingerprint")
        == q31.get("margin_contribution_ledger_fingerprint"),
        "native proof margin contribution ledger must match summary",
        errors,
    )
    require(native_proof.get("replay_payload_bytes") == native.get("replay_payload_bytes"), "native proof replay bytes must match summary", errors)
    require(standalone.get("run_kind") == "reverie_run_result", "standalone classifier must be a generic Reverie run", errors)
    require(standalone.get("contains_vecmat_q31") is True, "standalone classifier source must call vecmat_q31", errors)
    require(standalone.get("contains_prediction_assert") is True, "standalone classifier must assert prediction", errors)
    require(standalone.get("contains_correct_assert") is True, "standalone classifier must assert correctness", errors)
    require(standalone.get("run_uses_vecmat_q31") is True, "standalone classifier run must execute vecmat_q31", errors)
    require(standalone.get("matches_native_prediction") is True, "standalone classifier prediction must match native inference", errors)
    require(standalone.get("matches_native_correct") is True, "standalone classifier correctness must match native inference", errors)
    require(standalone.get("matches_native_label") is True, "standalone classifier label must match native inference", errors)
    require(standalone.get("roundtrip_passed") is True, "standalone classifier roundtrip must pass", errors)
    require(standalone.get("verification_passed") is True, "standalone classifier roundtrip verification must pass", errors)
    require(is_sha256_hex(standalone.get("sha256")), "standalone classifier source hash must be SHA-256", errors)

    profile_summary = profile.get("summary")
    if not isinstance(profile_summary, dict):
        errors.append("inference_trace_profile.summary must be an object")
    else:
        for field in (
            "all_report_verification_results_match",
            "all_report_verification_ledgers_match",
            "all_required_references_match",
            "all_replay_verified",
            "all_explanations_passed",
            "all_sources_checked",
            "all_passed",
        ):
            require(profile_summary.get(field) is True, f"inference_trace_profile.summary.{field} must be true", errors)
        require(
            isinstance(profile_summary.get("trace_count"), int)
            and profile_summary["trace_count"] >= 2,
            "inference_trace_profile.summary.trace_count must cover native and evaluation traces",
            errors,
        )
        require(
            isinstance(profile_summary.get("reference_checked_traces"), int)
            and profile_summary["reference_checked_traces"] >= 1,
            "inference_trace_profile.summary.reference_checked_traces must be positive",
            errors,
        )
    require(trace.get("passed") is True, "reversible inference trace must pass", errors)
    attribution = trace.get("attribution")
    if not isinstance(attribution, dict):
        errors.append("reversible_inference_trace.attribution must be an object")
    else:
        require(attribution.get("matches_logit") is True, "trace attribution logit check must pass", errors)
        require(attribution.get("matches_margin") is True, "trace attribution margin check must pass", errors)
        for field in ("contribution_ledger_fingerprint", "margin_contribution_ledger_fingerprint"):
            require(is_sha256_hex(attribution.get(field)), f"trace attribution {field} must be SHA-256", errors)
    for field in ("replay_payload_bytes", "witness_payload_bytes", "total_recompute_steps"):
        require(
            isinstance(trace.get(field), int)
            and not isinstance(trace.get(field), bool)
            and trace[field] > 0,
            f"reversible_inference_trace.{field} must be positive",
            errors,
        )
    require(trace.get("trace_payload_bytes") == 0, "reversible inference trace should have zero trace payload bytes", errors)
    require(trace.get("forward_recompute_steps") == trace.get("inverse_recompute_steps"), "trace recompute should be balanced", errors)
    require(trace_report.get("passed") is True, "reversible trace report must pass", errors)
    for field in (
        "replay_payload_bytes",
        "witness_payload_bytes",
        "trace_payload_bytes",
        "forward_recompute_steps",
        "inverse_recompute_steps",
        "total_recompute_steps",
    ):
        require(trace_proof.get(field) == trace.get(field), f"trace proof {field} must match summary", errors)
    proof_checks = trace_proof.get("checks")
    if isinstance(proof_checks, dict):
        for field in (
            "preprocess_matches_reference",
            "logits_match_reference",
            "prediction_matches_reference",
            "correctness_matches_reference",
            "reverse_restores_initial_state",
            "raw_preserved",
            "model_preserved",
            "balanced_recompute",
        ):
            require(proof_checks.get(field) is True, f"trace proof check {field} must pass", errors)
    else:
        errors.append("trace proof checks must be an object")
    if errors:
        raise ValueError("; ".join(errors))

    assert isinstance(profile_summary, dict)
    assert isinstance(attribution, dict)
    action_contract = [
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
            "matches_native": imported["prediction_matches_native"]
            and imported["correct_matches_native"]
            and imported["margin_matches_native"],
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

    return {
        "prediction": q31["prediction"],
        "correct": q31["correct"],
        "margin": q31["margin"],
        "active_pixels": q31["active_pixels"],
        "imported_model_prediction": imported["prediction"],
        "imported_model_matches_native": (
            imported["prediction_matches_native"]
            and imported["correct_matches_native"]
            and imported["margin_matches_native"]
        ),
        "imported_model_replay_payload_bytes": imported["replay_payload_bytes"],
        "model_import_provenance_kind": model_import["verification_provenance_kind"],
        "model_import_source_checked": model_import["verification_source_checked"],
        "model_import_source_fingerprint": model_import["source_model_json_fingerprint"],
        "native_replay_payload_bytes": native["replay_payload_bytes"],
        "contribution_ledger_fingerprint": q31["contribution_ledger_fingerprint"],
        "margin_contribution_ledger_fingerprint": q31["margin_contribution_ledger_fingerprint"],
        "trace_count": profile_summary["trace_count"],
        "reference_checked_traces": profile_summary["reference_checked_traces"],
        "reversible_trace_prediction": trace["prediction"],
        "reversible_trace_margin": attribution["margin"],
        "reversible_trace_replay_payload_bytes": trace["replay_payload_bytes"],
        "reversible_trace_witness_payload_bytes": trace["witness_payload_bytes"],
        "reversible_trace_trace_payload_bytes": trace["trace_payload_bytes"],
        "reversible_trace_total_recompute_steps": trace["total_recompute_steps"],
        "reversible_trace_contribution_ledger_fingerprint": attribution[
            "contribution_ledger_fingerprint"
        ],
        "reversible_trace_margin_ledger_fingerprint": attribution[
            "margin_contribution_ledger_fingerprint"
        ],
        "action_contract": action_contract,
    }


def validate_claim_evidence(summary: dict[str, Any]) -> dict[str, Any]:
    claims = summary.get("claims")
    gates = summary.get("gates")
    metrics = summary.get("metrics")
    evidence = summary.get("evidence")
    if not isinstance(claims, dict):
        raise ValueError("pipeline summary claims must be an object")
    if not isinstance(gates, dict):
        raise ValueError("pipeline summary gates must be an object")
    if not isinstance(metrics, dict):
        raise ValueError("pipeline summary metrics must be an object")
    if not isinstance(evidence, dict):
        raise ValueError("pipeline summary evidence must be an object")
    evidence_files = evidence.get("files")
    if not isinstance(evidence_files, dict):
        raise ValueError("pipeline summary evidence.files must be an object")
    checks = claims.get("checks")
    gate_checks = gates.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("pipeline summary claims.checks must be a non-empty list")
    if not isinstance(gate_checks, list) or not gate_checks:
        raise ValueError("pipeline summary gates.checks must be a non-empty list")
    if claims.get("passed") is not True:
        raise ValueError("pipeline summary claims did not pass")

    gate_results = {
        check["metric"]: check["passed"]
        for check in gate_checks
        if isinstance(check, dict)
        and isinstance(check.get("metric"), str)
        and isinstance(check.get("passed"), bool)
    }
    metric_refs: set[str] = set()
    gate_metric_refs: set[str] = set()
    evidence_refs: set[str] = set()
    passed_claims = 0
    claim_matrix: list[dict[str, Any]] = []
    integrity_claim_evidence: Optional[set[str]] = None
    errors: list[str] = []
    for index, check in enumerate(checks):
        context = f"claims.checks[{index}]"
        if not isinstance(check, dict):
            errors.append(f"{context} must be an object")
            continue
        claim = check.get("claim")
        if not isinstance(claim, str) or not claim:
            errors.append(f"{context}.claim must be a non-empty string")
            continue
        passed = check.get("passed")
        if passed is not True:
            errors.append(f"{context} claim `{claim}` did not pass")
        else:
            passed_claims += 1
        row = {
            "claim": claim,
            "passed": passed is True,
            "gate_metrics": [],
            "metrics": [],
            "evidence": [],
        }
        for field, known, refs in (
            ("gate_metrics", gate_results, gate_metric_refs),
            ("metrics", metrics, metric_refs),
            ("evidence", evidence_files, evidence_refs),
        ):
            values = check.get(field)
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value for value in values
            ):
                errors.append(f"{context}.{field} must be a string array")
                continue
            row[field] = list(values)
            for value in values:
                refs.add(value)
                if value not in known:
                    errors.append(f"{context}.{field} references unknown `{value}`")
        for gate_metric in check.get("gate_metrics", []):
            if gate_results.get(gate_metric) is not True:
                errors.append(f"{context}.gate_metrics references failing gate `{gate_metric}`")
        if claim == "artifact_evidence_integrity" and isinstance(check.get("evidence"), list):
            integrity_claim_evidence = set(check["evidence"])
        claim_matrix.append(row)
    if integrity_claim_evidence is None:
        errors.append("claims must include artifact_evidence_integrity")
    else:
        expected_evidence = set(evidence_files)
        expected_evidence.discard("summary")
        missing = sorted(expected_evidence - integrity_claim_evidence)
        extra = sorted(integrity_claim_evidence - expected_evidence)
        if missing or extra:
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if extra:
                details.append("extra: " + ", ".join(extra))
            errors.append(
                "artifact_evidence_integrity must cite every digest-indexed evidence file"
                + " ("
                + "; ".join(details)
                + ")"
            )
    evidence_bytes = 0
    for key, record in evidence_files.items():
        if not isinstance(record, dict):
            errors.append(f"evidence.files.{key} must be an object")
            continue
        size = record.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            errors.append(f"evidence.files.{key}.bytes must be positive")
        else:
            evidence_bytes += size
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            errors.append(f"evidence.files.{key}.sha256 must be a SHA-256 hex string")
    if errors:
        raise ValueError("; ".join(errors))
    assert integrity_claim_evidence is not None
    return {
        "claims": len(checks),
        "passed_claims": passed_claims,
        "gate_metric_refs": len(gate_metric_refs),
        "metric_refs": len(metric_refs),
        "evidence_refs": len(evidence_refs),
        "evidence_files": len(evidence_files),
        "integrity_claim_files": len(integrity_claim_evidence),
        "integrity_claim_covers_all_files": True,
        "evidence_bytes": evidence_bytes,
        "claims_fingerprint": sha256_json(claims),
        "evidence_fingerprint": sha256_json(evidence),
        "claim_matrix": claim_matrix,
        "claim_matrix_fingerprint": sha256_json(claim_matrix),
    }


def validate_roundtrip(audit_dir: Path) -> dict[str, Any]:
    path = audit_dir / "reversible-inference-roundtrip-verification.json"
    report = load_json(path)
    if not isinstance(report, dict):
        raise ValueError("roundtrip verification must be a JSON object")
    checks = report.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("roundtrip verification checks must be an object")
    failed = sorted(key for key, value in checks.items() if value is not True)
    if report.get("passed") is not True or failed:
        raise ValueError("roundtrip verification failed: " + ", ".join(failed))
    if report.get("proof_fingerprint") != report.get("replay_fingerprint"):
        raise ValueError("roundtrip proof fingerprint does not match replay fingerprint")
    for check in ("ml_profile_schema", "ml_profile_replayed"):
        if checks.get(check) is not True:
            raise ValueError(f"roundtrip verification missing {check} check")
    if report.get("ml_profile_present") is not True:
        raise ValueError("roundtrip verification must include a signed ML profile")
    profile = report.get("ml_profile")
    if not isinstance(profile, dict):
        raise ValueError("roundtrip verification ml_profile must be an object")
    if profile.get("schema") != "reverie_explain_ml_profile_v1":
        raise ValueError("roundtrip verification ml_profile schema mismatch")
    if profile.get("goal_fit") != "auditable_ml_kernel":
        raise ValueError("roundtrip verification ml_profile must be auditable_ml_kernel")
    if report.get("ml_profile_fingerprint") != sha256_json(profile):
        raise ValueError("roundtrip verification ml_profile fingerprint mismatch")
    metrics = profile.get("tensor_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("roundtrip verification ml_profile tensor_metrics must be an object")
    if not isinstance(metrics.get("known_witness_payload_bytes"), int) or metrics.get(
        "known_witness_payload_bytes"
    ) <= 0:
        raise ValueError("roundtrip verification must account for witness payload bytes")
    replay_cost = profile.get("replay_cost")
    if not isinstance(replay_cost, dict):
        raise ValueError("roundtrip verification ml_profile replay_cost must be an object")
    if not isinstance(replay_cost.get("roundtrip_statement_count"), int) or replay_cost.get(
        "roundtrip_statement_count"
    ) <= 0:
        raise ValueError("roundtrip verification replay_cost must account for roundtrip statements")
    if (
        replay_cost.get("known_witness_payload_bytes")
        != metrics.get("known_witness_payload_bytes")
    ):
        raise ValueError("roundtrip verification replay_cost witness bytes mismatch")
    if not isinstance(replay_cost.get("known_replay_payload_bytes"), int) or replay_cost.get(
        "known_replay_payload_bytes"
    ) < replay_cost["known_witness_payload_bytes"]:
        raise ValueError("roundtrip verification replay_cost payload bytes are inconsistent")
    if (
        not isinstance(profile.get("q31_builtin_calls"), int)
        or profile.get("q31_builtin_calls") <= 0
    ):
        raise ValueError("roundtrip verification must account for Q31 builtin calls")
    return {
        "path": str(path),
        "passed": True,
        "proof_fingerprint": report["proof_fingerprint"],
        "replay_fingerprint": report["replay_fingerprint"],
        "ml_profile_goal_fit": profile["goal_fit"],
        "ml_profile_fingerprint": report["ml_profile_fingerprint"],
        "ml_witness_payload_bytes": metrics["known_witness_payload_bytes"],
        "ml_roundtrip_statement_count": replay_cost["roundtrip_statement_count"],
        "ml_replay_payload_bytes": replay_cost["known_replay_payload_bytes"],
        "ml_q31_builtin_calls": profile["q31_builtin_calls"],
    }


def validate_benchmark_gate(
    benchmark_json: Path,
    benchmark_markdown: Path,
    *,
    min_workloads: int,
    expected_runs: int,
    expected_warmup: int,
    min_observed_speedup: float,
    min_median_speedup: float,
    min_geomean_speedup: float,
    min_observed_rss_ratio: float,
    min_median_rss_ratio: float,
    verify_file_digests: bool,
    require_current_workloads: bool,
) -> dict[str, Any]:
    data = load_json(benchmark_json)
    if not isinstance(data, dict):
        raise ValueError("benchmark artifact must be a JSON object")
    exact_workloads: Optional[list[str]] = None
    if require_current_workloads:
        try:
            exact_workloads = benchmark_check.current_workloads()
        except RuntimeError as error:
            raise ValueError(str(error)) from error
    errors = benchmark_check.validate_artifact(
        data,
        min_workloads,
        [],
        [],
        False,
        min_observed_speedup,
        min_median_speedup,
        min_geomean_speedup,
        expected_runs,
        expected_warmup,
        30.0,
        1.25,
        [("forward", 19), ("reverse", 12), ("roundtrip", 10)]
        if min_workloads >= 41
        else [],
        exact_workloads,
        exact_workloads is not None,
        True,
        True,
        verify_file_digests,
        None,
        None,
    )
    if benchmark_markdown is not None:
        try:
            markdown = benchmark_markdown.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError(f"failed to read benchmark Markdown: {error}") from error
        errors.extend(benchmark_check.validate_markdown_summary(data, markdown))

    benchmarks = data.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise ValueError("benchmark rows must be a list")
    speeds = [row.get("speedup") for row in benchmarks if isinstance(row, dict)]
    ratios = [row.get("memory_ratio") for row in benchmarks if isinstance(row, dict)]
    missing_memory = [
        row.get("name", f"benchmarks[{index}]")
        for index, row in enumerate(benchmarks)
        if not isinstance(row, dict) or not number(row.get("memory_ratio"))
    ]
    if missing_memory:
        errors.append("benchmark workload(s) missing RSS ratio: " + ", ".join(missing_memory))
    numeric_ratios = [float(ratio) for ratio in ratios if number(ratio)]
    low_memory = [
        row["name"]
        for row in benchmarks
        if isinstance(row, dict)
        and number(row.get("memory_ratio"))
        and float(row["memory_ratio"]) < min_observed_rss_ratio
    ]
    if low_memory:
        errors.append(
            "benchmark workload(s) below RSS ratio gate: " + ", ".join(low_memory)
        )
    if numeric_ratios and statistics.median(numeric_ratios) < min_median_rss_ratio:
        errors.append(
            "median RSS ratio "
            f"{statistics.median(numeric_ratios):.2f}x is below required "
            f"{min_median_rss_ratio:.2f}x"
        )
    if errors:
        raise ValueError("; ".join(errors))

    numeric_speeds = [float(speed) for speed in speeds if number(speed)]
    return {
        "path": str(benchmark_json),
        "markdown": str(benchmark_markdown),
        "workloads": len(benchmarks),
        "runs": data["runs"],
        "warmup": data["warmup"],
        "min_speedup": min(numeric_speeds),
        "median_speedup": statistics.median(numeric_speeds),
        "geomean_speedup": statistics.geometric_mean(numeric_speeds),
        "min_rss_ratio": min(numeric_ratios),
        "median_rss_ratio": statistics.median(numeric_ratios),
        "memory_rows": len(numeric_ratios),
        "missing_memory_rows": len(missing_memory),
    }


def artifact_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def release_verifier_sources() -> dict[str, Any]:
    files = {
        path: artifact_metadata(REPO_ROOT / path)
        for path in RELEASE_VERIFIER_SOURCE_FILES
    }
    return {
        "algorithm": "sha256",
        "files": files,
        "fingerprint": sha256_json(files),
    }


def release_packet_artifacts(
    *,
    gate_output: Path,
    gate_markdown: Path,
    verified_transcript: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    files = {
        "gate_json": artifact_metadata(gate_output),
        "gate_markdown": artifact_metadata(gate_markdown),
        "reviewer_transcript_json": artifact_metadata(Path(verified_transcript["path"])),
        "reviewer_transcript_markdown": artifact_metadata(Path(verified_transcript["markdown"])),
        "benchmark_json": artifact_metadata(Path(benchmark["path"])),
        "benchmark_markdown": artifact_metadata(Path(benchmark["markdown"])),
    }
    return {
        "algorithm": "sha256",
        "files": files,
        "fingerprint": sha256_json(files),
    }


def gate_artifact_summary(report: dict[str, Any]) -> dict[str, Any]:
    artifacts = report["payload"]["artifacts"]
    files = {
        artifact_id: {
            "path": metadata["path"],
            "bytes": metadata["bytes"],
            "sha256": metadata["sha256"],
        }
        for artifact_id, metadata in artifacts.items()
    }
    return {
        "algorithm": "sha256",
        "count": len(files),
        "bytes": sum(metadata["bytes"] for metadata in files.values()),
        "fingerprint": sha256_json(files),
    }


def release_check_status(payload: dict[str, Any], check_id: str) -> bool:
    benchmark = payload["benchmark"]
    if check_id == "roundtrip":
        return payload["roundtrip"]["passed"] is True
    if check_id == "benchmark_speed_memory":
        return (
            benchmark["workloads"] >= 1
            and benchmark["memory_rows"] == benchmark["workloads"]
            and benchmark["missing_memory_rows"] == 0
            and number(benchmark["median_speedup"])
            and number(benchmark["median_rss_ratio"])
        )
    raise ValueError(f"unknown goal-contract release check `{check_id}`")


def release_goal_contract(report: dict[str, Any]) -> dict[str, Any]:
    payload = report["payload"]
    roadmap = payload["roadmap"]
    scorecard = roadmap["scorecard"]
    claim_matrix = payload["claim_evidence"]["claim_matrix"]
    claims_by_name = {
        row["claim"]: row
        for row in claim_matrix
        if isinstance(row, dict) and isinstance(row.get("claim"), str)
    }
    claim_proofs = []
    for claim_id in GOAL_CONTRACT_CLAIMS:
        requirements = GOAL_CONTRACT_CLAIM_REQUIREMENTS[claim_id]
        required_pipeline_claims = requirements["pipeline_claims"]
        required_gate_metrics = requirements["gate_metrics"]
        required_release_checks = requirements["release_checks"]
        rows = [
            claims_by_name[claim]
            for claim in required_pipeline_claims
            if claim in claims_by_name
        ]
        gate_metric_union = {
            metric
            for row in rows
            for metric in row.get("gate_metrics", [])
            if isinstance(metric, str)
        }
        release_checks = {
            check_id: release_check_status(payload, check_id)
            for check_id in required_release_checks
        }
        missing_pipeline_claims = [
            claim for claim in required_pipeline_claims if claim not in claims_by_name
        ]
        failing_pipeline_claims = [
            row["claim"] for row in rows if row.get("passed") is not True
        ]
        missing_gate_metrics = [
            metric for metric in required_gate_metrics if metric not in gate_metric_union
        ]
        failing_release_checks = [
            check_id for check_id, passed in release_checks.items() if passed is not True
        ]
        claim_proofs.append(
            {
                "claim": claim_id,
                "passed": not (
                    missing_pipeline_claims
                    or failing_pipeline_claims
                    or missing_gate_metrics
                    or failing_release_checks
                ),
                "pipeline_claims": list(required_pipeline_claims),
                "missing_pipeline_claims": missing_pipeline_claims,
                "failing_pipeline_claims": failing_pipeline_claims,
                "gate_metrics": list(required_gate_metrics),
                "missing_gate_metrics": missing_gate_metrics,
                "release_checks": release_checks,
                "failing_release_checks": failing_release_checks,
                "evidence_refs": sorted(
                    {
                        evidence
                        for row in rows
                        for evidence in row.get("evidence", [])
                        if isinstance(evidence, str)
                    }
                ),
            }
        )
    contract = {
        "north_star": payload["north_star"],
        "non_goal": payload["non_goal"],
        "claims": list(GOAL_CONTRACT_CLAIMS),
        "claim_proofs": claim_proofs,
        "passed_claims": sum(1 for proof in claim_proofs if proof["passed"] is True),
        "total_claims": len(claim_proofs),
        "capability_phases": list(roadmap["capabilities"]["phases"]),
        "capabilities_passed": roadmap["capabilities"]["passed"],
        "capabilities_total": roadmap["capabilities"]["total"],
        "readiness_passed": roadmap["readiness"]["passed"],
        "readiness_total": roadmap["readiness"]["total"],
        "claim_matrix_fingerprint": payload["claim_evidence"]["claim_matrix_fingerprint"],
        "cost_signals": {
            "trace_to_model_payload_ratio": scorecard["trace_to_model_payload_ratio"],
            "witness_to_model_payload_ratio": scorecard["witness_to_model_payload_ratio"],
            "reverse_to_train_elapsed_ratio": scorecard["reverse_to_train_elapsed_ratio"],
        },
    }
    return {
        **contract,
        "fingerprint": sha256_json(contract),
    }


def goal_contract_errors(goal_contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if goal_contract.get("north_star") != NORTH_STAR:
        errors.append("north-star mismatch")
    if goal_contract.get("non_goal") != NON_GOAL:
        errors.append("non-goal mismatch")
    if tuple(goal_contract.get("claims", [])) != GOAL_CONTRACT_CLAIMS:
        errors.append("claim set mismatch")
    proofs = goal_contract.get("claim_proofs")
    if not isinstance(proofs, list):
        errors.append("claim proofs missing")
        proofs = []
    if tuple(proof.get("claim") for proof in proofs if isinstance(proof, dict)) != GOAL_CONTRACT_CLAIMS:
        errors.append("claim proof order mismatch")
    for proof in proofs:
        if not isinstance(proof, dict):
            errors.append("claim proof must be an object")
            continue
        if proof.get("passed") is not True:
            details = []
            for field in (
                "missing_pipeline_claims",
                "failing_pipeline_claims",
                "missing_gate_metrics",
                "failing_release_checks",
            ):
                values = proof.get(field)
                if values:
                    details.append(f"{field}={values}")
            errors.append(
                "claim `{}` did not pass{}".format(
                    proof.get("claim"),
                    ": " + ", ".join(details) if details else "",
                )
            )
    if goal_contract.get("capability_phases") != list(EXPECTED_PHASES):
        errors.append("capability phases mismatch")
    if (
        goal_contract.get("capabilities_passed")
        != goal_contract.get("capabilities_total")
        or goal_contract.get("capabilities_total") != len(EXPECTED_PHASES)
    ):
        errors.append("capability count mismatch")
    if (
        goal_contract.get("readiness_passed")
        != goal_contract.get("readiness_total")
        or not isinstance(goal_contract.get("readiness_total"), int)
        or goal_contract["readiness_total"] <= 0
    ):
        errors.append("readiness count mismatch")
    if (
        goal_contract.get("passed_claims")
        != goal_contract.get("total_claims")
        or goal_contract.get("total_claims") != len(GOAL_CONTRACT_CLAIMS)
    ):
        errors.append("goal-contract claims not fully proven")
    expected_fingerprint = sha256_json(
        {key: value for key, value in goal_contract.items() if key != "fingerprint"}
    )
    if goal_contract.get("fingerprint") != expected_fingerprint:
        errors.append("goal-contract fingerprint mismatch")
    return errors


def goal_contract_markdown_lines(goal_contract: dict[str, Any]) -> list[str]:
    lines = [
        "## Goal Contract",
        "",
        "| North star | Non-goal | Contract | Capabilities | Readiness |",
        "| --- | --- | --- | ---: | ---: |",
        "| `{}` | `{}` | `{}` | {}/{} | {}/{} |".format(
            goal_contract["north_star"],
            goal_contract["non_goal"],
            short_hash(goal_contract["fingerprint"]),
            goal_contract["capabilities_passed"],
            goal_contract["capabilities_total"],
            goal_contract["readiness_passed"],
            goal_contract["readiness_total"],
        ),
        "",
        "| Claims proven | Claim matrix | Cost signals |",
        "| ---: | --- | --- |",
        "| {}/{} | `{}` | trace/model {:.3f}x, witness/model {:.3f}x, reverse/train {:.3f}x |".format(
            goal_contract["passed_claims"],
            goal_contract["total_claims"],
            short_hash(goal_contract["claim_matrix_fingerprint"]),
            goal_contract["cost_signals"]["trace_to_model_payload_ratio"],
            goal_contract["cost_signals"]["witness_to_model_payload_ratio"],
            goal_contract["cost_signals"]["reverse_to_train_elapsed_ratio"],
        ),
        "",
        "| Claim | Proven | Pipeline claims | Gate metrics | Release checks | Evidence refs |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for proof in goal_contract["claim_proofs"]:
        release_checks = ", ".join(
            f"{name}:{'pass' if passed is True else 'fail'}"
            for name, passed in proof["release_checks"].items()
        )
        lines.append(
            "| `{}` | {} | `{}` | `{}` | `{}` | {} |".format(
                proof["claim"],
                "pass" if proof["passed"] is True else "fail",
                ", ".join(proof["pipeline_claims"]),
                ", ".join(proof["gate_metrics"]),
                release_checks or "none",
                len(proof["evidence_refs"]),
            )
        )
    lines.append("")
    return lines


def deterministic_inference_contract_markdown_lines(
    deterministic_inference: dict[str, Any],
) -> list[str]:
    action_contract = deterministic_inference["action_contract"]
    lines = [
        "### Action Contract",
        "",
        "| Operation | Supported | Evidence | Result | Cost |",
        "| --- | --- | --- | --- | --- |",
    ]
    for action in action_contract:
        operation = action["operation"]
        evidence = action["evidence"]
        supported = str(action["supported"]).lower()
        if operation == "reproduce_prediction":
            result = "prediction {}, correct {}, margin {}".format(
                action["result"]["prediction"],
                str(action["result"]["correct"]).lower(),
                f"{action['result']['margin']:,}",
            )
            cost = "Q31/native comparison"
        elif operation == "explain_margin":
            result = "contribution `{}`, margin `{}`".format(
                short_hash(action["ledgers"]["contribution"]),
                short_hash(action["ledgers"]["margin_contribution"]),
            )
            cost = "ledger recompute"
        elif operation == "replay_imported_model_inference":
            result = "prediction {}, native match {}, replay payload {} bytes".format(
                action["prediction"],
                str(action["matches_native"]).lower(),
                f"{action['payload_bytes']:,}",
            )
            cost = "external-import replay"
        elif operation == "replay_native_inference":
            result = "replay payload {} bytes".format(
                f"{action['payload_bytes']:,}",
            )
            cost = "one forward + one inverse replay"
        elif operation == "run_standalone_rev_classifier":
            result = "prediction {}, correct {}, source `{}`".format(
                action["prediction"],
                str(action["correct"]).lower(),
                short_hash(action["source_sha256"]),
            )
            cost = "source-only Reverie run"
        elif operation == "reverse_reversible_trace":
            result = "witness {}, trace {}, replay {} bytes".format(
                f"{action['witness_payload_bytes']:,}",
                f"{action['trace_payload_bytes']:,}",
                f"{action['payload_bytes']:,}",
            )
            cost = "{} recompute steps".format(action["recompute_steps"])
        else:
            result = "checked"
            cost = "recorded"
        lines.append(
            "| `{}` | {} | `{}` | {} | {} |".format(
                operation,
                supported,
                evidence,
                result,
                cost,
            )
        )
    return lines


def command_record(
    command_id: str,
    label: str,
    purpose: str,
    command: list[str],
    artifacts: list[str],
) -> dict[str, Any]:
    return {
        "id": command_id,
        "label": label,
        "purpose": purpose,
        "command": command,
        "artifacts": artifacts,
    }


def reviewer_commands(
    audit_dir: Path,
    benchmark_json: Path,
    benchmark_markdown: Path,
) -> dict[str, Any]:
    run_report = audit_dir / "run-report.json"
    audit_bundle = audit_dir / "training-audit-bundle.json"
    audit_verification = audit_dir / "training-audit-verification.json"
    audit_verification_markdown = audit_dir / "training-audit-verification.md"
    summary = audit_dir / capsule_check.DEFAULT_SUMMARY_NAME
    capsule = audit_dir / capsule_check.DEFAULT_CAPSULE_NAME
    manifest = audit_dir / capsule_check.DEFAULT_MANIFEST_NAME
    roundtrip_proof = audit_dir / "reversible-inference-roundtrip-proof.json"
    roundtrip_markdown = audit_dir / "reversible-inference-roundtrip.md"
    training_step_bundle = audit_dir / "training-step-bundle.json"
    training_step_debug = audit_dir / "training-step-debug.md"
    training_update_review_receipt = (
        audit_dir / audit_pipeline.DEFAULT_TRAINING_UPDATE_REVIEW_RECEIPT_NAME
    )
    training_update_review_receipt_markdown = (
        audit_dir / audit_pipeline.DEFAULT_TRAINING_UPDATE_REVIEW_RECEIPT_MARKDOWN_NAME
    )
    mlp_vars = audit_dir / "mlp-witness-vars.json"
    mlp_run_output = audit_dir / "mlp-witness-run.json"
    mlp_witness = audit_dir / "mlp-witness-report.json"
    mlp_witness_markdown = audit_dir / "mlp-witness.md"
    coupling_forward = audit_dir / "invertible-coupling-forward.json"
    coupling_reverse = audit_dir / "invertible-coupling-reverse.json"
    coupling_report = audit_dir / "invertible-coupling-report.json"
    coupling_markdown = audit_dir / "invertible-coupling.md"
    residual_forward = audit_dir / "triangular-residual-forward.json"
    residual_reverse = audit_dir / "triangular-residual-reverse.json"
    residual_report = audit_dir / "triangular-residual-report.json"
    residual_markdown = audit_dir / "triangular-residual.md"
    preprocess_forward = audit_dir / "reversible-preprocess-forward.json"
    preprocess_reverse = audit_dir / "reversible-preprocess-reverse.json"
    preprocess_report = audit_dir / "reversible-preprocess-report.json"
    preprocess_markdown = audit_dir / "reversible-preprocess.md"
    q31_reference_inference = audit_dir / "q31-reference-inference.json"
    q31_reference_inference_markdown = audit_dir / "q31-reference-inference.md"
    inference_action_review_receipt = (
        audit_dir / capsule_check.DEFAULT_ACTION_REVIEW_RECEIPT_NAME
    )
    inference_action_review_receipt_markdown = (
        audit_dir / capsule_check.DEFAULT_ACTION_REVIEW_RECEIPT_MARKDOWN_NAME
    )
    native_inference_audit_markdown = audit_dir / "native-inference-audit.md"
    native_inference_verification_markdown = audit_dir / "native-inference-verification.md"
    evaluation_row_inference_audit_markdown = (
        audit_dir / "model-evaluation-row-inference-audit.md"
    )
    inference_trace_forward = audit_dir / "reversible-inference-trace-forward.json"
    inference_trace_reverse = audit_dir / "reversible-inference-trace-reverse.json"
    inference_trace_report = audit_dir / "reversible-inference-trace-report.json"
    inference_trace_markdown = audit_dir / "reversible-inference-trace.md"
    return {
        "schema": REVIEWER_COMMAND_SCHEMA,
        "commands": [
            command_record(
                "north_star_gate",
                "Rerun the north-star release gate",
                "Revalidates the capsule, handoff, roundtrip proof, roadmap gates, and benchmark artifact.",
                [
                    "python3",
                    "scripts/check_ml_north_star.py",
                    str(audit_dir),
                    "--benchmark-json",
                    str(benchmark_json),
                    "--benchmark-markdown",
                    str(benchmark_markdown),
                ],
                [
                    "summary",
                    "capsule",
                    "manifest",
                    "handoff",
                    "handoff_markdown",
                    "inference_action_review_receipt",
                    "inference_action_review_receipt_markdown",
                    "roundtrip_verification",
                    "benchmark_json",
                    "benchmark_markdown",
                ],
            ),
            command_record(
                "capsule_handoff",
                "Verify capsule and handoff freshness",
                "Recomputes capsule trust metadata plus every handoff artifact byte count and SHA-256.",
                [
                    "python3",
                    "scripts/verify_model_capsule.py",
                    str(audit_dir),
                    "--require-handoff",
                    "--require-verification-markdown",
                ],
                [
                    "summary",
                    "capsule",
                    "manifest",
                    "capsule_profile",
                    "capsule_verification_markdown",
                    "handoff",
                    "handoff_markdown",
                    "training_audit_verification_markdown",
                    "q31_reference_inference",
                    "q31_reference_inference_markdown",
                    "imported_model_source",
                    "imported_model_bundle",
                    "imported_model_import",
                    "imported_model_verification",
                    "imported_model_inference",
                    "imported_model_inference_bundle",
                    "imported_model_inference_verification",
                    "native_inference_audit_markdown",
                    "native_inference_verification_markdown",
                    "evaluation_row_inference_audit_markdown",
                ],
            ),
            command_record(
                "inference_action_receipt",
                "Replay the handoff inference action commands",
                "Executes the handoff's prediction, margin, native replay, and reverse-trace commands and writes the receipt.",
                [
                    "python3",
                    "scripts/verify_model_capsule.py",
                    str(audit_dir),
                    "--require-handoff",
                    "--require-verification-markdown",
                    "--run-inference-action-commands",
                    "--action-command-receipt-output",
                    str(inference_action_review_receipt),
                    "--action-command-receipt-markdown",
                    str(inference_action_review_receipt_markdown),
                ],
                [
                    "summary",
                    "capsule",
                    "manifest",
                    "capsule_profile",
                    "capsule_verification_markdown",
                    "handoff",
                    "handoff_markdown",
                    "inference_action_review_receipt",
                    "inference_action_review_receipt_markdown",
                    "q31_reference_inference",
                    "q31_reference_inference_markdown",
                    "imported_model_source",
                    "imported_model_bundle",
                    "imported_model_inference_bundle",
                    "native_inference_verification_markdown",
                    "inference_trace_forward",
                    "inference_trace_reverse",
                    "reversible_inference_trace",
                    "reversible_inference_trace_markdown",
                ],
            ),
            command_record(
                "profile_evidence",
                "Validate summary/capsule/manifest evidence",
                "Checks report schemas, capsule-summary consistency, manifest binding, and file evidence hashes.",
                [
                    "python3",
                    "scripts/check_mnist_ml_profile.py",
                    str(summary),
                    str(capsule),
                    str(manifest),
                    "--verify-pipeline-files",
                ],
                ["summary", "capsule", "manifest"],
            ),
            command_record(
                "training_audit_lineage",
                "Replay full training-audit lineage card",
                "Replays the saved training witness trace forward and backward, while the handoff verifier checks the saved lineage proof card hash.",
                [
                    "cargo",
                    "run",
                    "-p",
                    "reverie-cli",
                    "--bin",
                    "reverie-mnist-linear",
                    "--",
                    "--verify-audit",
                    str(audit_bundle),
                ],
                [
                    "run_report",
                    "audit_bundle",
                    "audit_verification",
                    "training_audit_verification_markdown",
                ],
            ),
            command_record(
                "benchmark_speed_memory",
                "Validate Jana/Reverie speed and RSS artifact",
                "Checks workload coverage, source digests, Markdown freshness, speed gates, and RSS ratios.",
                [
                    "python3",
                    "scripts/check_benchmark_artifact.py",
                    str(benchmark_json),
                    "--min-workloads",
                    "41",
                    "--expect-runs",
                    "5",
                    "--expect-warmup",
                    "1",
                    "--expect-performance-gate",
                    "--min-observed-speedup",
                    "2.0",
                    "--min-median-speedup",
                    "3.0",
                    "--min-geomean-speedup",
                    "3.0",
                    "--expect-direction-count",
                    "forward:19",
                    "--expect-direction-count",
                    "reverse:12",
                    "--expect-direction-count",
                    "roundtrip:10",
                    "--expect-markdown-summary",
                    str(benchmark_markdown),
                    "--expect-source-digests",
                    "--verify-file-digests",
                ],
                ["benchmark_json", "benchmark_markdown"],
            ),
            command_record(
                "roundtrip_proof",
                "Replay the reversible inference roundtrip proof",
                "Reloads the saved proof, reruns the referenced source, and checks restored state fingerprints.",
                [
                    "cargo",
                    "run",
                    "-p",
                    "reverie-cli",
                    "--",
                    "verify-roundtrip",
                    str(roundtrip_proof),
                    "--markdown-output",
                    str(roundtrip_markdown),
                ],
                ["roundtrip_proof", "roundtrip_verification", "roundtrip_markdown"],
            ),
            command_record(
                "training_step_debug",
                "Replay the selected training-step debug card",
                "Recomputes the selected sample, logits, witnesses, and weight-delta ledgers for one update.",
                [
                    "cargo",
                    "run",
                    "-p",
                    "reverie-cli",
                    "--bin",
                    "reverie-mnist-linear",
                    "--",
                    "--verify-step",
                    str(training_step_bundle),
                    "--markdown-output",
                    str(training_step_debug),
                ],
                ["training_step_bundle", "training_step_debug"],
            ),
            command_record(
                "training_update_receipt",
                "Replay the selected training-update receipt",
                "Regenerates the selected update receipt over full-lineage replay, update inspection, and one-step reversal.",
                [
                    "python3",
                    "scripts/run_mnist_ml_audit_pipeline.py",
                    "--out-dir",
                    str(audit_dir),
                    "--write-training-update-receipt-only",
                ],
                [
                    "audit_bundle",
                    "audit_verification",
                    "training_audit_verification_markdown",
                    "training_step_bundle",
                    "training_step_debug",
                    "training_update_review_receipt",
                    "training_update_review_receipt_markdown",
                ],
            ),
            command_record(
                "mlp_witness_proof",
                "Replay the MLP witness proof card",
                "Recomputes hidden activations, masks, errors, hidden deltas, and final layer updates for the saved MLP run.",
                [
                    "python3",
                    "scripts/check_q31_mlp_witness.py",
                    "--vars-json",
                    str(mlp_vars),
                    "--run-output-json",
                    str(mlp_run_output),
                    "--expect-predictions",
                    "[0,1]",
                    "--expect-correct",
                    "[true,true]",
                    "--expect-report-json",
                    str(mlp_witness),
                    "--markdown-output",
                    str(mlp_witness_markdown),
                ],
                ["mlp_vars", "mlp_run_output", "mlp_witness", "mlp_witness_markdown"],
            ),
            command_record(
                "invertible_coupling_proof",
                "Replay the invertible coupling proof card",
                "Rechecks the saved additive-coupling forward/reverse run and zero-witness proof card.",
                [
                    "python3",
                    "scripts/check_invertible_coupling.py",
                    "--forward-output-json",
                    str(coupling_forward),
                    "--reverse-output-json",
                    str(coupling_reverse),
                    "--expect-report-json",
                    str(coupling_report),
                    "--markdown-output",
                    str(coupling_markdown),
                ],
                [
                    "coupling_forward",
                    "coupling_reverse",
                    "invertible_coupling",
                    "invertible_coupling_markdown",
                ],
            ),
            command_record(
                "triangular_residual_proof",
                "Replay the triangular residual proof card",
                "Rechecks the saved triangular-residual forward/reverse run and zero-witness proof card.",
                [
                    "python3",
                    "scripts/check_triangular_residual.py",
                    "--forward-output-json",
                    str(residual_forward),
                    "--reverse-output-json",
                    str(residual_reverse),
                    "--expect-report-json",
                    str(residual_report),
                    "--markdown-output",
                    str(residual_markdown),
                ],
                [
                    "residual_forward",
                    "residual_reverse",
                    "triangular_residual",
                    "triangular_residual_markdown",
                ],
            ),
            command_record(
                "reversible_preprocess_proof",
                "Replay the reversible preprocessing proof card",
                "Rechecks the saved preprocessing forward/reverse run and zero-witness proof card.",
                [
                    "python3",
                    "scripts/check_reversible_preprocess.py",
                    "--forward-output-json",
                    str(preprocess_forward),
                    "--reverse-output-json",
                    str(preprocess_reverse),
                    "--expect-report-json",
                    str(preprocess_report),
                    "--markdown-output",
                    str(preprocess_markdown),
                ],
                [
                    "preprocess_forward",
                    "preprocess_reverse",
                    "reversible_preprocess",
                    "reversible_preprocess_markdown",
                ],
            ),
            command_record(
                "reversible_inference_trace_proof",
                "Replay the reversible inference trace proof card",
                "Rechecks preprocessing, logits, prediction witnesses, contribution ledgers, and reverse restoration for the saved inference trace.",
                [
                    "python3",
                    "scripts/check_reversible_inference_trace.py",
                    "--forward-output-json",
                    str(inference_trace_forward),
                    "--reverse-output-json",
                    str(inference_trace_reverse),
                    "--expect-report-json",
                    str(inference_trace_report),
                    "--markdown-output",
                    str(inference_trace_markdown),
                ],
                [
                    "inference_trace_forward",
                    "inference_trace_reverse",
                    "reversible_inference_trace",
                    "reversible_inference_trace_markdown",
                ],
            ),
        ],
    }


def validate_reviewer_commands(value: dict[str, Any], artifact_ids: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    if value.get("schema") != REVIEWER_COMMAND_SCHEMA:
        errors.append(f"reviewer command schema must be {REVIEWER_COMMAND_SCHEMA}")
    commands = value.get("commands")
    if not isinstance(commands, list) or not commands:
        errors.append("reviewer commands must be a non-empty list")
        raise ValueError("; ".join(errors))
    seen: set[str] = set()
    for index, command in enumerate(commands):
        context = f"reviewer_commands.commands[{index}]"
        if not isinstance(command, dict):
            errors.append(f"{context} must be an object")
            continue
        command_id = command.get("id")
        if not isinstance(command_id, str) or not command_id:
            errors.append(f"{context}.id must be a non-empty string")
        elif command_id in seen:
            errors.append(f"duplicate reviewer command id `{command_id}`")
        else:
            seen.add(command_id)
        for field in ("label", "purpose"):
            if not isinstance(command.get(field), str) or not command[field]:
                errors.append(f"{context}.{field} must be a non-empty string")
        argv = command.get("command")
        if not isinstance(argv, list) or not argv or not all(
            isinstance(part, str) and part for part in argv
        ):
            errors.append(f"{context}.command must be a non-empty string array")
        elif argv[0] not in ("python3", "cargo"):
            errors.append(f"{context}.command must start with python3 or cargo")
        artifacts = command.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"{context}.artifacts must be a non-empty string array")
        else:
            for artifact_id in artifacts:
                if not isinstance(artifact_id, str) or not artifact_id:
                    errors.append(f"{context}.artifacts must contain only non-empty strings")
                elif artifact_id not in artifact_ids:
                    errors.append(f"{context} references unknown artifact `{artifact_id}`")
    expected = {
        "north_star_gate",
        "capsule_handoff",
        "profile_evidence",
        "benchmark_speed_memory",
        "roundtrip_proof",
        "inference_action_receipt",
        "training_step_debug",
        "mlp_witness_proof",
        "invertible_coupling_proof",
        "triangular_residual_proof",
        "reversible_preprocess_proof",
        "reversible_inference_trace_proof",
    }
    missing = sorted(expected - seen)
    if missing:
        errors.append("missing reviewer command(s): " + ", ".join(missing))
    if errors:
        raise ValueError("; ".join(errors))
    return {"schema": value["schema"], "count": len(commands), "ids": sorted(seen)}


def build_gate_report(
    audit_dir: Path,
    benchmark_json: Path,
    benchmark_markdown: Path,
    *,
    min_workloads: int,
    expected_runs: int,
    expected_warmup: int,
    min_observed_speedup: float,
    min_median_speedup: float,
    min_geomean_speedup: float,
    min_observed_rss_ratio: float,
    min_median_rss_ratio: float,
    verify_file_digests: bool,
    require_current_workloads: bool,
) -> dict[str, Any]:
    capsule_result, capsule_report = validate_capsule_gate(
        audit_dir,
        verify_file_evidence=verify_file_digests,
    )
    inference_action_receipt = validate_inference_action_review_receipt(
        audit_dir,
        capsule_report,
    )
    summary_goal = validate_summary_goal(capsule_result["summary"])
    debug_update = validate_debug_training_update(capsule_result["summary"])
    training_update_receipt = validate_training_update_review_receipt(
        audit_dir,
        capsule_report,
        debug_update,
    )
    model_lineage = validate_training_lineage(audit_dir, capsule_result["summary"])
    deterministic_inference = validate_deterministic_inference(
        audit_dir,
        capsule_result["summary"],
    )
    claim_evidence = validate_claim_evidence(capsule_result["summary"])
    roundtrip = validate_roundtrip(audit_dir)
    benchmark = validate_benchmark_gate(
        benchmark_json,
        benchmark_markdown,
        min_workloads=min_workloads,
        expected_runs=expected_runs,
        expected_warmup=expected_warmup,
        min_observed_speedup=min_observed_speedup,
        min_median_speedup=min_median_speedup,
        min_geomean_speedup=min_geomean_speedup,
        min_observed_rss_ratio=min_observed_rss_ratio,
        min_median_rss_ratio=min_median_rss_ratio,
        verify_file_digests=verify_file_digests,
        require_current_workloads=require_current_workloads,
    )
    artifacts = {
        "run_report": artifact_metadata(audit_dir / "run-report.json"),
        "audit_bundle": artifact_metadata(audit_dir / "training-audit-bundle.json"),
        "audit_verification": artifact_metadata(
            audit_dir / "training-audit-verification.json"
        ),
        "training_audit_verification_markdown": artifact_metadata(
            audit_dir / "training-audit-verification.md"
        ),
        "summary": artifact_metadata(audit_dir / capsule_check.DEFAULT_SUMMARY_NAME),
        "capsule": artifact_metadata(audit_dir / capsule_check.DEFAULT_CAPSULE_NAME),
        "manifest": artifact_metadata(audit_dir / capsule_check.DEFAULT_MANIFEST_NAME),
        "capsule_profile": artifact_metadata(audit_dir / capsule_check.DEFAULT_PROFILE_NAME),
        "capsule_verification": artifact_metadata(
            audit_dir / capsule_check.DEFAULT_VERIFICATION_REPORT_NAME
        ),
        "capsule_verification_markdown": artifact_metadata(
            audit_dir / capsule_check.DEFAULT_VERIFICATION_MARKDOWN_NAME
        ),
        "handoff": artifact_metadata(audit_dir / capsule_check.DEFAULT_HANDOFF_NAME),
        "handoff_markdown": artifact_metadata(
            audit_dir / capsule_check.DEFAULT_HANDOFF_MARKDOWN_NAME
        ),
        "inference_action_review_receipt": artifact_metadata(
            audit_dir / capsule_check.DEFAULT_ACTION_REVIEW_RECEIPT_NAME
        ),
        "inference_action_review_receipt_markdown": artifact_metadata(
            audit_dir / capsule_check.DEFAULT_ACTION_REVIEW_RECEIPT_MARKDOWN_NAME
        ),
        "roundtrip_proof": artifact_metadata(
            audit_dir / "reversible-inference-roundtrip-proof.json"
        ),
        "roundtrip_verification": artifact_metadata(
            audit_dir / "reversible-inference-roundtrip-verification.json"
        ),
        "roundtrip_markdown": artifact_metadata(
            audit_dir / "reversible-inference-roundtrip.md"
        ),
        "training_step_bundle": artifact_metadata(audit_dir / "training-step-bundle.json"),
        "training_step_debug": artifact_metadata(audit_dir / "training-step-debug.md"),
        "training_update_review_receipt": artifact_metadata(
            audit_dir / audit_pipeline.DEFAULT_TRAINING_UPDATE_REVIEW_RECEIPT_NAME
        ),
        "training_update_review_receipt_markdown": artifact_metadata(
            audit_dir / audit_pipeline.DEFAULT_TRAINING_UPDATE_REVIEW_RECEIPT_MARKDOWN_NAME
        ),
        "mlp_vars": artifact_metadata(audit_dir / "mlp-witness-vars.json"),
        "mlp_run_output": artifact_metadata(audit_dir / "mlp-witness-run.json"),
        "mlp_witness": artifact_metadata(audit_dir / "mlp-witness-report.json"),
        "mlp_witness_markdown": artifact_metadata(audit_dir / "mlp-witness.md"),
        "coupling_forward": artifact_metadata(audit_dir / "invertible-coupling-forward.json"),
        "coupling_reverse": artifact_metadata(audit_dir / "invertible-coupling-reverse.json"),
        "invertible_coupling": artifact_metadata(audit_dir / "invertible-coupling-report.json"),
        "invertible_coupling_markdown": artifact_metadata(
            audit_dir / "invertible-coupling.md"
        ),
        "residual_forward": artifact_metadata(audit_dir / "triangular-residual-forward.json"),
        "residual_reverse": artifact_metadata(audit_dir / "triangular-residual-reverse.json"),
        "triangular_residual": artifact_metadata(audit_dir / "triangular-residual-report.json"),
        "triangular_residual_markdown": artifact_metadata(
            audit_dir / "triangular-residual.md"
        ),
        "preprocess_forward": artifact_metadata(
            audit_dir / "reversible-preprocess-forward.json"
        ),
        "preprocess_reverse": artifact_metadata(
            audit_dir / "reversible-preprocess-reverse.json"
        ),
        "reversible_preprocess": artifact_metadata(
            audit_dir / "reversible-preprocess-report.json"
        ),
        "reversible_preprocess_markdown": artifact_metadata(
            audit_dir / "reversible-preprocess.md"
        ),
        "q31_reference_inference": artifact_metadata(
            audit_dir / "q31-reference-inference.json"
        ),
        "q31_reference_inference_markdown": artifact_metadata(
            audit_dir / "q31-reference-inference.md"
        ),
        "imported_model_source": artifact_metadata(audit_dir / "imported-q31-linear-model.json"),
        "imported_model_bundle": artifact_metadata(audit_dir / "imported-model-bundle.json"),
        "imported_model_import": artifact_metadata(audit_dir / "imported-model-import-report.json"),
        "imported_model_verification": artifact_metadata(
            audit_dir / "imported-model-verification.json"
        ),
        "imported_model_inference": artifact_metadata(
            audit_dir / "imported-model-inference-report.json"
        ),
        "imported_model_inference_bundle": artifact_metadata(
            audit_dir / "imported-model-inference-bundle.json"
        ),
        "imported_model_inference_verification": artifact_metadata(
            audit_dir / "imported-model-inference-verification.json"
        ),
        "native_inference_audit_markdown": artifact_metadata(
            audit_dir / "native-inference-audit.md"
        ),
        "native_inference_verification_markdown": artifact_metadata(
            audit_dir / "native-inference-verification.md"
        ),
        "evaluation_row_inference_audit_markdown": artifact_metadata(
            audit_dir / "model-evaluation-row-inference-audit.md"
        ),
        "inference_trace_forward": artifact_metadata(
            audit_dir / "reversible-inference-trace-forward.json"
        ),
        "inference_trace_reverse": artifact_metadata(
            audit_dir / "reversible-inference-trace-reverse.json"
        ),
        "reversible_inference_trace": artifact_metadata(
            audit_dir / "reversible-inference-trace-report.json"
        ),
        "reversible_inference_trace_markdown": artifact_metadata(
            audit_dir / "reversible-inference-trace.md"
        ),
        "benchmark_json": artifact_metadata(benchmark_json),
        "benchmark_markdown": artifact_metadata(benchmark_markdown),
    }
    commands = reviewer_commands(audit_dir, benchmark_json, benchmark_markdown)
    command_summary = validate_reviewer_commands(commands, set(artifacts))
    payload = {
        "schema": GATE_SCHEMA,
        "north_star": NORTH_STAR,
        "non_goal": NON_GOAL,
        "capsule": {
            "fingerprint": capsule_report["capsule"]["fingerprint"],
            "gates": capsule_report["capsule"]["gates"],
            "readiness": capsule_report["capsule"]["readiness"],
            "model_sha256": capsule_report["capsule"]["model_sha256"],
            "witness_proof": capsule_report["capsule"]["witness_proof"],
        },
        "handoff": {
            "fingerprint": capsule_report["handoff"]["fingerprint"],
            "artifact_count": capsule_report["handoff"]["artifact_count"],
            "artifacts_checked": capsule_report["handoff"]["artifacts_checked"],
            "markdown_matches_rendered": capsule_report["handoff"]["markdown"][
                "matches_rendered"
            ],
        },
        "roadmap": summary_goal,
        "debug_update": debug_update,
        "training_update_review_receipt": training_update_receipt,
        "model_lineage": model_lineage,
        "deterministic_inference": deterministic_inference,
        "inference_action_review_receipt": inference_action_receipt,
        "claim_evidence": claim_evidence,
        "roundtrip": roundtrip,
        "benchmark": benchmark,
        "reviewer_commands": commands,
        "reviewer_command_summary": command_summary,
        "thresholds": {
            "min_workloads": min_workloads,
            "expected_runs": expected_runs,
            "expected_warmup": expected_warmup,
            "min_observed_speedup": min_observed_speedup,
            "min_median_speedup": min_median_speedup,
            "min_geomean_speedup": min_geomean_speedup,
            "min_observed_rss_ratio": min_observed_rss_ratio,
            "min_median_rss_ratio": min_median_rss_ratio,
            "verify_file_digests": verify_file_digests,
            "require_current_workloads": require_current_workloads,
        },
        "artifacts": artifacts,
    }
    payload["goal_contract"] = release_goal_contract({"payload": payload})
    goal_errors = goal_contract_errors(payload["goal_contract"])
    if goal_errors:
        raise ValueError("goal contract failed: " + "; ".join(goal_errors))
    return {
        "kind": GATE_KIND,
        "passed": True,
        "algorithm": "sha256",
        "fingerprint": sha256_json(payload),
        "payload": payload,
    }


def short_hash(value: Any) -> str:
    return value[:12] if isinstance(value, str) else "n/a"


def is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def shell_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def reviewer_command_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    commands = report["payload"]["reviewer_commands"]["commands"]
    return {command["id"]: command for command in commands}


def select_reviewer_commands(report: dict[str, Any], command_ids: list[str]) -> list[dict[str, Any]]:
    available = reviewer_command_map(report)
    selected = []
    missing = []
    for command_id in command_ids:
        command = available.get(command_id)
        if command is None:
            missing.append(command_id)
        else:
            selected.append(command)
    if missing:
        raise ValueError(
            "unknown reviewer command(s): {}; available: {}".format(
                ", ".join(missing),
                ", ".join(sorted(available)),
            )
        )
    return selected


def print_reviewer_commands(report: dict[str, Any]) -> None:
    for command in report["payload"]["reviewer_commands"]["commands"]:
        print(f"{command['id']}: {command['label']}")
        print(f"  {shell_command(command['command'])}")


def run_reviewer_commands(
    report: dict[str, Any],
    command_ids: list[str],
    *,
    emit_output: bool = True,
) -> dict[str, Any]:
    selected = select_reviewer_commands(report, command_ids)
    entries = []
    total_started = time.perf_counter()
    for command in selected:
        argv = command["command"]
        if emit_output:
            print(f"reviewer command: {command['id']}")
            print(shell_command(argv), flush=True)
        started = time.perf_counter()
        completed = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        elapsed = time.perf_counter() - started
        if emit_output and completed.stdout:
            sys.stdout.write(completed.stdout)
            sys.stdout.flush()
        if emit_output and completed.stderr:
            sys.stderr.write(completed.stderr)
            sys.stderr.flush()
        entries.append(
            {
                "id": command["id"],
                "label": command["label"],
                "purpose": command["purpose"],
                "command": argv,
                "shell_command": shell_command(argv),
                "artifacts": command["artifacts"],
                "exit_code": completed.returncode,
                "passed": completed.returncode == 0,
                "elapsed_seconds": elapsed,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "stdout_bytes": len(completed.stdout.encode("utf-8")),
                "stderr_bytes": len(completed.stderr.encode("utf-8")),
                "stdout_sha256": sha256_text(completed.stdout),
                "stderr_sha256": sha256_text(completed.stderr),
            }
        )
        if completed.returncode != 0:
            break
    total_elapsed = time.perf_counter() - total_started
    failed = [entry["id"] for entry in entries if entry["passed"] is not True]
    payload = {
        "schema": TRANSCRIPT_SCHEMA,
        "gate_fingerprint_before": report["fingerprint"],
        "requested_commands": command_ids,
        "commands": entries,
        "summary": {
            "passed": not failed and len(entries) == len(command_ids),
            "requested": len(command_ids),
            "executed": len(entries),
            "failed": len(failed),
            "failed_commands": failed,
            "total_elapsed_seconds": total_elapsed,
        },
    }
    return {
        "kind": TRANSCRIPT_KIND,
        "passed": payload["summary"]["passed"],
        "algorithm": "sha256",
        "fingerprint": sha256_json(payload),
        "payload": payload,
    }


def finalize_transcript(transcript: dict[str, Any], report_after: dict[str, Any]) -> dict[str, Any]:
    payload = dict(transcript["payload"])
    payload["gate_fingerprint_after"] = report_after["fingerprint"]
    payload["gate_still_passed"] = report_after["passed"] is True
    return {
        "kind": TRANSCRIPT_KIND,
        "passed": transcript["passed"] is True and report_after["passed"] is True,
        "algorithm": "sha256",
        "fingerprint": sha256_json(payload),
        "payload": payload,
    }


def render_transcript_markdown(transcript: dict[str, Any]) -> str:
    payload = transcript["payload"]
    summary = payload["summary"]
    lines = [
        "# Reverie ML Reviewer Replay Transcript",
        "",
        "| Verdict | Transcript | Gate before | Gate after | Commands | Failed |",
        "| --- | --- | --- | --- | ---: | ---: |",
        "| {} | `{}` | `{}` | `{}` | {} | {} |".format(
            "pass" if transcript["passed"] is True else "fail",
            short_hash(transcript["fingerprint"]),
            short_hash(payload.get("gate_fingerprint_before")),
            short_hash(payload.get("gate_fingerprint_after")),
            summary["executed"],
            summary["failed"],
        ),
        "",
        "| Command | Exit | Elapsed | stdout | stderr |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for entry in payload["commands"]:
        lines.append(
            "| `{}` | {} | {:.3f}s | `{}` ({} bytes) | `{}` ({} bytes) |".format(
                entry["id"],
                entry["exit_code"],
                entry["elapsed_seconds"],
                short_hash(entry["stdout_sha256"]),
                entry["stdout_bytes"],
                short_hash(entry["stderr_sha256"]),
                entry["stderr_bytes"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def validate_saved_transcript(
    report: dict[str, Any],
    transcript_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    transcript = load_json(transcript_path)
    if not isinstance(transcript, dict):
        raise ValueError("reviewer transcript must be a JSON object")
    errors: list[str] = []
    if transcript.get("kind") != TRANSCRIPT_KIND:
        errors.append(f"reviewer transcript kind must be {TRANSCRIPT_KIND}")
    if transcript.get("algorithm") != "sha256":
        errors.append("reviewer transcript algorithm must be sha256")
    payload = transcript.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("reviewer transcript payload must be an object")
    if payload.get("schema") != TRANSCRIPT_SCHEMA:
        errors.append(f"reviewer transcript schema must be {TRANSCRIPT_SCHEMA}")
    expected_fingerprint = sha256_json(payload)
    if transcript.get("fingerprint") != expected_fingerprint:
        errors.append("reviewer transcript fingerprint does not match payload")
    if not is_sha256_hex(payload.get("gate_fingerprint_before")):
        errors.append("reviewer transcript gate_fingerprint_before must be a SHA-256 hex digest")
    if payload.get("gate_fingerprint_after") != report["fingerprint"]:
        errors.append("reviewer transcript gate_fingerprint_after does not match current gate")
    if payload.get("gate_still_passed") is not True:
        errors.append("reviewer transcript gate_still_passed must be true")

    command_map = reviewer_command_map(report)
    requested = payload.get("requested_commands")
    commands = payload.get("commands")
    summary = payload.get("summary")
    if not isinstance(requested, list) or not all(isinstance(item, str) for item in requested):
        errors.append("reviewer transcript requested_commands must be a string array")
        requested = []
    if not isinstance(commands, list):
        errors.append("reviewer transcript commands must be a list")
        commands = []
    if not isinstance(summary, dict):
        errors.append("reviewer transcript summary must be an object")
        summary = {}

    failed_commands = []
    for index, entry in enumerate(commands):
        context = f"reviewer transcript command[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{context} must be an object")
            continue
        command_id = entry.get("id")
        expected_command = command_map.get(command_id) if isinstance(command_id, str) else None
        if expected_command is None:
            errors.append(f"{context} references unknown command `{command_id}`")
        else:
            if entry.get("command") != expected_command["command"]:
                errors.append(f"{context}.command does not match current command index")
            if entry.get("artifacts") != expected_command["artifacts"]:
                errors.append(f"{context}.artifacts does not match current command index")
            if entry.get("label") != expected_command["label"]:
                errors.append(f"{context}.label does not match current command index")
            if entry.get("purpose") != expected_command["purpose"]:
                errors.append(f"{context}.purpose does not match current command index")
        command = entry.get("command")
        if isinstance(command, list) and all(isinstance(part, str) for part in command):
            if entry.get("shell_command") != shell_command(command):
                errors.append(f"{context}.shell_command does not match command")
        else:
            errors.append(f"{context}.command must be a string array")
        exit_code = entry.get("exit_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            errors.append(f"{context}.exit_code must be an integer")
            exit_code = None
        passed = entry.get("passed")
        if not isinstance(passed, bool):
            errors.append(f"{context}.passed must be a boolean")
        elif isinstance(exit_code, int) and passed != (exit_code == 0):
            errors.append(f"{context}.passed does not match exit_code")
        if passed is not True and isinstance(command_id, str):
            failed_commands.append(command_id)
        elapsed = entry.get("elapsed_seconds")
        if not number(elapsed) or float(elapsed) < 0.0:
            errors.append(f"{context}.elapsed_seconds must be non-negative")
        for stream in ("stdout", "stderr"):
            value = entry.get(stream)
            if not isinstance(value, str):
                errors.append(f"{context}.{stream} must be a string")
                continue
            encoded = value.encode("utf-8")
            if entry.get(f"{stream}_bytes") != len(encoded):
                errors.append(f"{context}.{stream}_bytes does not match {stream}")
            if entry.get(f"{stream}_sha256") != sha256_text(value):
                errors.append(f"{context}.{stream}_sha256 does not match {stream}")

    if summary.get("requested") != len(requested):
        errors.append("reviewer transcript summary.requested does not match requested_commands")
    if summary.get("executed") != len(commands):
        errors.append("reviewer transcript summary.executed does not match commands")
    if summary.get("failed") != len(failed_commands):
        errors.append("reviewer transcript summary.failed does not match failed commands")
    if summary.get("failed_commands") != failed_commands:
        errors.append("reviewer transcript summary.failed_commands does not match commands")
    expected_passed = len(commands) == len(requested) and not failed_commands
    if summary.get("passed") != expected_passed:
        errors.append("reviewer transcript summary.passed does not match command results")
    if transcript.get("passed") != expected_passed:
        errors.append("reviewer transcript passed does not match command results")
    if not number(summary.get("total_elapsed_seconds")) or float(
        summary.get("total_elapsed_seconds", -1)
    ) < 0.0:
        errors.append("reviewer transcript summary.total_elapsed_seconds must be non-negative")

    if not markdown_path.exists():
        errors.append(f"reviewer transcript Markdown is missing: {markdown_path}")
    else:
        markdown = markdown_path.read_text(encoding="utf-8")
        expected_markdown = render_transcript_markdown(transcript)
        if markdown != expected_markdown:
            errors.append("reviewer transcript Markdown is stale")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "path": str(transcript_path),
        "markdown": str(markdown_path),
        "fingerprint": transcript["fingerprint"],
        "commands": len(commands),
        "command_ids": [entry["id"] for entry in commands],
        "passed": transcript["passed"],
    }


def claim_replay_coverage(
    goal_contract: dict[str, Any],
    command_ids: list[str],
    command_artifacts: dict[str, list[str]],
) -> dict[str, Any]:
    executed = set(command_ids)
    rows = []
    for proof in goal_contract["claim_proofs"]:
        claim = proof["claim"]
        required_commands = list(GOAL_CONTRACT_CLAIM_REPLAY_COMMANDS[claim])
        missing_commands = [
            command_id for command_id in required_commands if command_id not in executed
        ]
        required_artifacts = list(GOAL_CONTRACT_CLAIM_REPLAY_ARTIFACTS[claim])
        covered_artifact_set = {
            artifact_id
            for command_id in required_commands
            if command_id in executed
            for artifact_id in command_artifacts.get(command_id, [])
        }
        covered_artifacts = [
            artifact_id
            for artifact_id in required_artifacts
            if artifact_id in covered_artifact_set
        ]
        missing_artifacts = [
            artifact_id
            for artifact_id in required_artifacts
            if artifact_id not in covered_artifact_set
        ]
        rows.append(
            {
                "claim": claim,
                "required_command_ids": required_commands,
                "executed_command_ids": [
                    command_id
                    for command_id in required_commands
                    if command_id in executed
                ],
                "missing_command_ids": missing_commands,
                "required_artifact_ids": required_artifacts,
                "covered_artifact_ids": covered_artifacts,
                "missing_artifact_ids": missing_artifacts,
                "passed": (
                    proof["passed"] is True
                    and not missing_commands
                    and not missing_artifacts
                ),
            }
        )
    return {
        "schema": "reverie_ml_north_star_claim_replay_coverage_v1",
        "rows": rows,
        "passed_claims": sum(1 for row in rows if row["passed"] is True),
        "total_claims": len(rows),
        "missing_command_ids": sorted(
            {
                command_id
                for row in rows
                for command_id in row["missing_command_ids"]
            }
        ),
        "missing_artifact_ids": sorted(
            {
                artifact_id
                for row in rows
                for artifact_id in row["missing_artifact_ids"]
            }
        ),
        "fingerprint": sha256_json(rows),
    }


def build_release_verification(
    report: dict[str, Any],
    verified_transcript: dict[str, Any],
    *,
    gate_output: Path,
    gate_markdown: Path,
) -> dict[str, Any]:
    payload = report["payload"]
    benchmark = payload["benchmark"]
    claim_evidence = payload["claim_evidence"]
    command_ids = verified_transcript["command_ids"]
    missing_required_commands = [
        command_id
        for command_id in REQUIRED_RELEASE_REPLAY_COMMANDS
        if command_id not in command_ids
    ]
    verifier_sources = release_verifier_sources()
    packet_artifacts = release_packet_artifacts(
        gate_output=gate_output,
        gate_markdown=gate_markdown,
        verified_transcript=verified_transcript,
        benchmark=benchmark,
    )
    audit_packet_artifacts = gate_artifact_summary(report)
    goal_contract = release_goal_contract(report)
    gate_goal_contract = payload.get("goal_contract")
    goal_errors = goal_contract_errors(goal_contract)
    command_artifacts = {
        command["id"]: list(command["artifacts"])
        for command in payload["reviewer_commands"]["commands"]
    }
    replay_coverage = claim_replay_coverage(
        goal_contract,
        command_ids,
        command_artifacts,
    )
    checks = {
        "gate_passed": report["passed"] is True,
        "gate_goal_contract_fresh": gate_goal_contract == goal_contract,
        "goal_contract_bound": not goal_errors,
        "goal_contract_claims_proven": not goal_errors,
        "claim_replay_coverage_complete": (
            replay_coverage["passed_claims"] == replay_coverage["total_claims"]
            == len(GOAL_CONTRACT_CLAIMS)
            and not replay_coverage["missing_command_ids"]
            and not replay_coverage["missing_artifact_ids"]
        ),
        "claim_evidence_complete": (
            claim_evidence["passed_claims"] == claim_evidence["claims"]
            and claim_evidence["integrity_claim_covers_all_files"] is True
            and claim_evidence["evidence_files"] == claim_evidence["integrity_claim_files"]
        ),
        "benchmark_speed_memory_complete": (
            benchmark["workloads"] >= 1
            and benchmark["memory_rows"] == benchmark["workloads"]
            and benchmark["missing_memory_rows"] == 0
            and number(benchmark["median_speedup"])
            and number(benchmark["median_rss_ratio"])
        ),
        "reviewer_transcript_verified": verified_transcript["passed"] is True,
        "reviewer_commands_replayed": verified_transcript["commands"] > 0,
        "required_reviewer_commands_replayed": not missing_required_commands,
        "verifier_sources_bound": (
            tuple(verifier_sources["files"]) == RELEASE_VERIFIER_SOURCE_FILES
        ),
        "release_artifacts_bound": (
            tuple(packet_artifacts["files"]) == RELEASE_PACKET_ARTIFACT_IDS
        ),
        "audit_packet_artifacts_bound": (
            audit_packet_artifacts["count"] == len(payload["artifacts"])
            and audit_packet_artifacts["bytes"] > 0
        ),
    }
    saved_verify_command = [
        "python3",
        "scripts/check_ml_north_star.py",
        str(Path(payload["artifacts"]["summary"]["path"]).parent),
        "--benchmark-json",
        str(Path(benchmark["path"])),
        "--benchmark-markdown",
        str(Path(benchmark["markdown"])),
        "--verify-release-verification",
    ]
    regenerate_command = [
        *saved_verify_command[:-1],
        "--verify-reviewer-transcript",
        "--verify-release-verification",
    ]
    verification_payload = {
        "schema": RELEASE_VERIFICATION_SCHEMA,
        "gate": {
            "path": str(gate_output),
            "markdown": str(gate_markdown),
            "fingerprint": report["fingerprint"],
            "passed": report["passed"],
            "capsule": payload["capsule"]["fingerprint"],
            "handoff": payload["handoff"]["fingerprint"],
            "claims": claim_evidence["claims"],
            "passed_claims": claim_evidence["passed_claims"],
            "evidence_files": claim_evidence["evidence_files"],
            "integrity_claim_files": claim_evidence["integrity_claim_files"],
            "evidence_fingerprint": claim_evidence["evidence_fingerprint"],
        },
        "goal_contract": goal_contract,
        "goal_contract_errors": goal_errors,
        "claim_replay_coverage": replay_coverage,
        "benchmark": {
            "workloads": benchmark["workloads"],
            "memory_rows": benchmark["memory_rows"],
            "median_speedup": benchmark["median_speedup"],
            "median_rss_ratio": benchmark["median_rss_ratio"],
        },
        "reviewer_transcript": {
            "path": verified_transcript["path"],
            "markdown": verified_transcript["markdown"],
            "fingerprint": verified_transcript["fingerprint"],
            "passed": verified_transcript["passed"],
            "commands": verified_transcript["commands"],
            "command_ids": command_ids,
            "required_command_ids": list(REQUIRED_RELEASE_REPLAY_COMMANDS),
            "missing_required_command_ids": missing_required_commands,
        },
        "reviewer_replay_commands": {
            "saved_receipt_check": saved_verify_command,
            "regenerate_receipt": regenerate_command,
            "required_replay_command_ids": list(REQUIRED_RELEASE_REPLAY_COMMANDS),
        },
        "verifier_sources": verifier_sources,
        "release_artifacts": packet_artifacts,
        "audit_packet_artifacts": audit_packet_artifacts,
        "checks": checks,
    }
    passed = all(checks.values())
    return {
        "kind": RELEASE_VERIFICATION_KIND,
        "passed": passed,
        "algorithm": "sha256",
        "fingerprint": sha256_json(verification_payload),
        "payload": verification_payload,
    }


def render_release_verification_markdown(report: dict[str, Any]) -> str:
    payload = report["payload"]
    gate = payload["gate"]
    goal_contract = payload["goal_contract"]
    replay_coverage = payload["claim_replay_coverage"]
    benchmark = payload["benchmark"]
    transcript = payload["reviewer_transcript"]
    replay_commands = payload["reviewer_replay_commands"]
    verifier_sources = payload["verifier_sources"]
    release_artifacts = payload["release_artifacts"]
    audit_packet_artifacts = payload["audit_packet_artifacts"]
    checks = payload["checks"]
    failed = [name for name, passed in checks.items() if passed is not True]
    lines = [
        "# Reverie ML Release Verification",
        "",
        "| Verdict | Release | Gate | Transcript | Claims | Evidence | Commands |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
        "| {} | `{}` | `{}` | `{}` | {}/{} | {}/{} | {} |".format(
            "pass" if report["passed"] is True else "fail",
            short_hash(report["fingerprint"]),
            short_hash(gate["fingerprint"]),
            short_hash(transcript["fingerprint"]),
            gate["passed_claims"],
            gate["claims"],
            gate["integrity_claim_files"],
            gate["evidence_files"],
            transcript["commands"],
        ),
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for name, passed in checks.items():
        lines.append(f"| `{name}` | {'pass' if passed is True else 'fail'} |")
    lines.extend(["", *goal_contract_markdown_lines(goal_contract)])
    lines.extend(
        [
            "## Benchmark",
            "",
            "| Workloads | Memory rows | Median speedup | Median RSS ratio |",
            "| ---: | ---: | ---: | ---: |",
            "| {} | {} | {:.2f}x | {:.2f}x |".format(
                benchmark["workloads"],
                benchmark["memory_rows"],
                benchmark["median_speedup"],
                benchmark["median_rss_ratio"],
            ),
            "",
            "## Reviewer Replay",
            "",
            "| Commands | Required | Missing | Transcript | Path |",
            "| --- | --- | --- | --- | --- |",
            "| `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                ", ".join(transcript["command_ids"]),
                ", ".join(transcript["required_command_ids"]),
                ", ".join(transcript["missing_required_command_ids"]) or "none",
                short_hash(transcript["fingerprint"]),
                transcript["path"],
            ),
            "",
            "## Claim Replay Coverage",
            "",
            "| Claim | Covered | Required replay commands | Required artifacts | Missing commands | Missing artifacts |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in replay_coverage["rows"]:
        lines.append(
            "| `{}` | {} | `{}` | `{}` | `{}` | `{}` |".format(
                row["claim"],
                "pass" if row["passed"] is True else "fail",
                ", ".join(row["required_command_ids"]),
                ", ".join(row["required_artifact_ids"]),
                ", ".join(row["missing_command_ids"]) or "none",
                ", ".join(row["missing_artifact_ids"]) or "none",
            )
        )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "| Action | Command |",
            "| --- | --- |",
            "| Check saved receipt | `{}` |".format(
                shell_command(replay_commands["saved_receipt_check"])
            ),
            "| Regenerate receipt | `{}` |".format(
                shell_command(replay_commands["regenerate_receipt"])
            ),
            "",
            "## Release Artifacts",
            "",
            "| Artifact set | Files |",
            "| --- | ---: |",
            "| `{}` | {} |".format(
                short_hash(release_artifacts["fingerprint"]),
                len(release_artifacts["files"]),
            ),
            "",
            "| Artifact | SHA-256 | Bytes |",
            "| --- | --- | ---: |",
        ]
    )
    for artifact_id, metadata in release_artifacts["files"].items():
        lines.append(
            "| `{}` | `{}` | {} |".format(
                artifact_id,
                short_hash(metadata["sha256"]),
                metadata["bytes"],
            )
        )
    lines.extend(
        [
            "",
            "## Audit Packet",
            "",
            "| Artifact index | Files | Bytes |",
            "| --- | ---: | ---: |",
            "| `{}` | {} | {} |".format(
                short_hash(audit_packet_artifacts["fingerprint"]),
                audit_packet_artifacts["count"],
                audit_packet_artifacts["bytes"],
            ),
            "",
            "## Verifier Sources",
            "",
            "| Source set | Files |",
            "| --- | ---: |",
            "| `{}` | {} |".format(
                short_hash(verifier_sources["fingerprint"]),
                len(verifier_sources["files"]),
            ),
            "",
            "| Source | SHA-256 |",
            "| --- | --- |",
        ]
    )
    for path, metadata in verifier_sources["files"].items():
        lines.append(f"| `{path}` | `{short_hash(metadata['sha256'])}` |")
    lines.append("")
    if failed:
        lines.extend(["## Failed Checks", "", ", ".join(f"`{name}`" for name in failed), ""])
    return "\n".join(lines)


def validate_saved_release_verification(
    report: dict[str, Any],
    verified_transcript: dict[str, Any],
    verification_path: Path,
    markdown_path: Path,
    *,
    gate_output: Path,
    gate_markdown: Path,
) -> dict[str, Any]:
    release = load_json(verification_path)
    if not isinstance(release, dict):
        raise ValueError("release verification must be a JSON object")
    errors: list[str] = []
    if release.get("kind") != RELEASE_VERIFICATION_KIND:
        errors.append(f"release verification kind must be {RELEASE_VERIFICATION_KIND}")
    if release.get("algorithm") != "sha256":
        errors.append("release verification algorithm must be sha256")
    payload = release.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("release verification payload must be an object")
    if payload.get("schema") != RELEASE_VERIFICATION_SCHEMA:
        errors.append(f"release verification schema must be {RELEASE_VERIFICATION_SCHEMA}")
    expected_fingerprint = sha256_json(payload)
    if release.get("fingerprint") != expected_fingerprint:
        errors.append("release verification fingerprint does not match payload")
    expected = build_release_verification(
        report,
        verified_transcript,
        gate_output=gate_output,
        gate_markdown=gate_markdown,
    )
    if payload != expected["payload"]:
        errors.append("release verification payload is stale")
    if release.get("passed") != expected["passed"]:
        errors.append("release verification passed flag is stale")
    if release.get("fingerprint") != expected["fingerprint"]:
        errors.append("release verification fingerprint is stale")
    if not markdown_path.exists():
        errors.append(f"release verification Markdown is missing: {markdown_path}")
    else:
        markdown = markdown_path.read_text(encoding="utf-8")
        expected_markdown = render_release_verification_markdown(release)
        if markdown != expected_markdown:
            errors.append("release verification Markdown is stale")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "path": str(verification_path),
        "markdown": str(markdown_path),
        "fingerprint": release["fingerprint"],
        "passed": release["passed"],
        "checks": release["payload"]["checks"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    payload = report["payload"]
    capsule = payload["capsule"]
    handoff = payload["handoff"]
    action_receipt = payload["inference_action_review_receipt"]
    training_receipt = payload["training_update_review_receipt"]
    roadmap = payload["roadmap"]
    scorecard = roadmap["scorecard"]
    claim_evidence = payload["claim_evidence"]
    goal_contract = payload["goal_contract"]
    benchmark = payload["benchmark"]
    roundtrip = payload["roundtrip"]
    debug_update = payload["debug_update"]
    model_lineage = payload["model_lineage"]
    deterministic_inference = payload["deterministic_inference"]
    commands = payload["reviewer_commands"]["commands"]
    lines = [
        "# Reverie ML North-Star Gate",
        "",
        "| Verdict | Gate | Capsule | Handoff | Gates | Readiness |",
        "| --- | --- | --- | --- | ---: | ---: |",
        "| pass | `{}` | `{}` | `{}` | {}/{} | {}/{} |".format(
            short_hash(report["fingerprint"]),
            short_hash(capsule["fingerprint"]),
            short_hash(handoff["fingerprint"]),
            capsule["gates"]["passed"],
            capsule["gates"]["total"],
            capsule["readiness"]["passed"],
            capsule["readiness"]["total"],
        ),
        "",
        "## Roadmap",
        "",
        "| Area | Result | Evidence |",
        "| --- | ---: | --- |",
        "| V1-V6 capabilities | {}/{} | `{}` |".format(
            roadmap["capabilities"]["passed"],
            roadmap["capabilities"]["total"],
            ", ".join(roadmap["capabilities"]["phases"]),
        ),
        "| North-star readiness | {}/{} | `{}` |".format(
            roadmap["readiness"]["passed"],
            roadmap["readiness"]["total"],
            payload["north_star"],
        ),
        "| Roundtrip proof | pass | proof `{}`, replay `{}` |".format(
            short_hash(roundtrip["proof_fingerprint"]),
            short_hash(roundtrip["replay_fingerprint"]),
        ),
        "| Roundtrip ML profile | `{}` | replay payload {} bytes, {} roundtrip statements |".format(
            roundtrip["ml_profile_goal_fit"],
            f"{roundtrip['ml_replay_payload_bytes']:,}",
            f"{roundtrip['ml_roundtrip_statement_count']:,}",
        ),
        "| Handoff artifacts | {}/{} | Markdown fresh `{}` |".format(
            handoff["artifacts_checked"],
            handoff["artifact_count"],
            str(handoff["markdown_matches_rendered"]).lower(),
        ),
        "| Inference action receipt | {}/{} | receipt `{}`, semantic `{}` |".format(
            action_receipt["passed_count"],
            action_receipt["command_count"],
            short_hash(action_receipt["fingerprint"]),
            short_hash(action_receipt["semantic_fingerprint"]),
        ),
        "| Training update receipt | {}/{} | receipt `{}`, semantic `{}` |".format(
            training_receipt["passed_count"],
            training_receipt["command_count"],
            short_hash(training_receipt["fingerprint"]),
            short_hash(training_receipt["semantic_fingerprint"]),
        ),
        "",
        *goal_contract_markdown_lines(goal_contract),
        "## Model Lineage",
        "",
        "| Checked steps | Final model replayed | Initial model restored | Witness replay | Model bytes | Witness bytes | Lineage ledger | Transition ledger | Final chain |",
        "| ---: | --- | --- | --- | ---: | ---: | --- | --- | --- |",
        "| {} / {} | {} | {} | {} | {} | {} | `{}` | `{}` | `{}` |".format(
            model_lineage["checked_steps"],
            model_lineage["train_samples"],
            str(model_lineage["final_model_replayed"]).lower(),
            str(model_lineage["restored_initial_model"]).lower(),
            str(model_lineage["witnesses_match_forward_replay"]).lower(),
            f"{model_lineage['model_payload_bytes']:,}",
            f"{model_lineage['witness_payload_bytes']:,}",
            short_hash(model_lineage["lineage_ledger_fingerprint"]),
            short_hash(model_lineage["transition_ledger_fingerprint"]),
            short_hash(model_lineage["final_chain"]),
        ),
        "",
        "| Initial model | Final model | Sample order | Witness trace |",
        "| --- | --- | --- | --- |",
        "| `{}` | `{}` | `{}` | `{}` |".format(
            short_hash(model_lineage["initial_model_fingerprint"]),
            short_hash(model_lineage["final_model_fingerprint"]),
            short_hash(model_lineage["sample_order_fingerprint"]),
            short_hash(model_lineage["witness_trace_fingerprint"]),
        ),
        "",
        "## Deterministic Inference",
        "",
        "| Prediction | Correct | Margin | Active pixels | Native replay payload | Reference traces | Contribution ledger | Margin ledger |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
        "| {} | {} | {} | {} | {} bytes | {} / {} | `{}` | `{}` |".format(
            deterministic_inference["prediction"],
            str(deterministic_inference["correct"]).lower(),
            f"{deterministic_inference['margin']:,}",
            deterministic_inference["active_pixels"],
            f"{deterministic_inference['native_replay_payload_bytes']:,}",
            deterministic_inference["reference_checked_traces"],
            deterministic_inference["trace_count"],
            short_hash(deterministic_inference["contribution_ledger_fingerprint"]),
            short_hash(deterministic_inference["margin_contribution_ledger_fingerprint"]),
        ),
        "",
        "| Imported pred | Imported replay | Imported/native | Import provenance | Source checked | Source fingerprint |",
        "| ---: | ---: | --- | --- | --- | --- |",
        "| {} | {} bytes | {} | `{}` | {} | `{}` |".format(
            deterministic_inference["imported_model_prediction"],
            f"{deterministic_inference['imported_model_replay_payload_bytes']:,}",
            str(deterministic_inference["imported_model_matches_native"]).lower(),
            deterministic_inference["model_import_provenance_kind"],
            str(deterministic_inference["model_import_source_checked"]).lower(),
            short_hash(deterministic_inference["model_import_source_fingerprint"]),
        ),
        "",
        "| Reversible trace prediction | Trace margin | Witness bytes | Trace bytes | Replay bytes | Recompute steps | Trace contribution ledger | Trace margin ledger |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        "| {} | {} | {} | {} | {} | {} | `{}` | `{}` |".format(
            deterministic_inference["reversible_trace_prediction"],
            f"{deterministic_inference['reversible_trace_margin']:,}",
            f"{deterministic_inference['reversible_trace_witness_payload_bytes']:,}",
            f"{deterministic_inference['reversible_trace_trace_payload_bytes']:,}",
            f"{deterministic_inference['reversible_trace_replay_payload_bytes']:,}",
            deterministic_inference["reversible_trace_total_recompute_steps"],
            short_hash(deterministic_inference["reversible_trace_contribution_ledger_fingerprint"]),
            short_hash(deterministic_inference["reversible_trace_margin_ledger_fingerprint"]),
        ),
        "",
        *deterministic_inference_contract_markdown_lines(deterministic_inference),
        "",
        "## Debuggable Update",
        "",
        "| Claim | Step | Sample | Prediction | Correct | Active pixels | Weight deltas | Replay payload | Reversed later steps | Ledgers |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        "| `{}` | {} | {} | {} | {} | {} | {} | {} bytes | {} | cause `{}`, bias `{}`, weight `{}` |".format(
            debug_update["claim"],
            debug_update["step"],
            debug_update["sample_index"],
            debug_update["prediction"],
            str(debug_update["correct"]).lower(),
            debug_update["active_pixel_count"],
            debug_update["nonzero_weight_delta_count"],
            f"{debug_update['replay_payload_bytes']:,}",
            debug_update["reversed_later_steps"],
            short_hash(debug_update["cause_ledger_fingerprint"]),
            short_hash(debug_update["bias_delta_ledger_fingerprint"]),
            short_hash(debug_update["weight_delta_ledger_fingerprint"]),
        ),
        "",
        "| Contract checks | Selection reasons | Max abs weight delta | Bias deltas |",
        "| --- | --- | ---: | ---: |",
        "| {} | {} | {} | {} |".format(
            ", ".join(debug_update["checks"]),
            ", ".join(debug_update["selection_reasons"]),
            f"{debug_update['selected_max_abs_weight_delta']:,}",
            debug_update["nonzero_bias_delta_count"],
        ),
        "",
        "## Claim Evidence",
        "",
        "| Claims | Gate refs | Metric refs | Evidence refs | Evidence files | Integrity coverage | Evidence bytes |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        "| {} | {} | {} | {} | {} | {} | {} |".format(
            claim_evidence["passed_claims"],
            claim_evidence["gate_metric_refs"],
            claim_evidence["metric_refs"],
            claim_evidence["evidence_refs"],
            claim_evidence["evidence_files"],
            "all files" if claim_evidence["integrity_claim_covers_all_files"] else "incomplete",
            f"{claim_evidence['evidence_bytes']:,}",
        ),
        "",
        "| Claims fingerprint | Evidence fingerprint |",
        "| --- | --- |",
        "| `{}` | `{}` |".format(
            short_hash(claim_evidence["claims_fingerprint"]),
            short_hash(claim_evidence["evidence_fingerprint"]),
        ),
        "",
        "## Cost And Benchmark",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Run peak RSS | {scorecard['run_peak_rss_bytes']:,} bytes |",
        f"| Max replay payload | {scorecard['max_replay_payload_bytes']:,} bytes |",
        f"| Roundtrip replay payload | {roundtrip['ml_replay_payload_bytes']:,} bytes |",
        f"| Roundtrip witness payload | {roundtrip['ml_witness_payload_bytes']:,} bytes |",
        f"| Roundtrip static statements | {roundtrip['ml_roundtrip_statement_count']:,} |",
        f"| Roundtrip Q31 calls | {roundtrip['ml_q31_builtin_calls']:,} |",
        f"| Trace/model payload | {scorecard['trace_to_model_payload_ratio']:.3f}x |",
        f"| Witness/model payload | {scorecard['witness_to_model_payload_ratio']:.3f}x |",
        f"| Reverse/train elapsed | {scorecard['reverse_to_train_elapsed_ratio']:.3f}x |",
        f"| Benchmark workloads | {benchmark['workloads']} |",
        f"| Minimum speedup | {benchmark['min_speedup']:.2f}x |",
        f"| Median speedup | {benchmark['median_speedup']:.2f}x |",
        f"| Geomean speedup | {benchmark['geomean_speedup']:.2f}x |",
        f"| Minimum RSS ratio | {benchmark['min_rss_ratio']:.2f}x |",
        f"| Median RSS ratio | {benchmark['median_rss_ratio']:.2f}x |",
        "",
        "## Reviewer Commands",
        "",
        "| Command | Purpose |",
        "| --- | --- |",
    ]
    for command in commands:
        lines.append(
            "| `{}` | {} |".format(
                shell_command(command["command"]),
                command["purpose"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_report(path: Path, text: str) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def self_test_report(root: Path) -> tuple[Path, Path, Path]:
    capsule_check.write_synthetic_capsule_dir(root)
    paths = resolve_audit_paths(root)
    result = capsule_check.verify_capsule(
        paths,
        verify_file_evidence=False,
        allow_missing_profile=False,
    )
    report = capsule_check.verification_report(
        result,
        verify_file_evidence=False,
        allow_missing_profile=False,
        check_verification_markdown=False,
    )
    (root / capsule_check.DEFAULT_VERIFICATION_MARKDOWN_NAME).write_text(
        capsule_check.render_verification_markdown(report),
        encoding="utf-8",
    )
    report = capsule_check.verification_report(
        result,
        verify_file_evidence=False,
        allow_missing_profile=False,
        verification_markdown_path=root / capsule_check.DEFAULT_VERIFICATION_MARKDOWN_NAME,
        require_verification_markdown=True,
    )
    (root / capsule_check.DEFAULT_VERIFICATION_REPORT_NAME).write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    handoff = capsule_check.write_synthetic_handoff(root, report["trust_certificate"])
    executable_handoff = json.loads(json.dumps(handoff))
    for command in executable_handoff["payload"]["inference_action_review_commands"]:
        command["command"] = [sys.executable, "-c", f"print({command['operation']!r})"]
    executable_handoff["fingerprint"] = sha256_json(executable_handoff["payload"])
    (root / capsule_check.DEFAULT_HANDOFF_NAME).write_text(
        json.dumps(executable_handoff, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / capsule_check.DEFAULT_HANDOFF_MARKDOWN_NAME).write_text(
        capsule_check.render_handoff_markdown(executable_handoff),
        encoding="utf-8",
    )
    action_receipt = capsule_check.run_inference_action_review_commands(
        executable_handoff,
        capsule_fingerprint=report["capsule"]["fingerprint"],
        trust_certificate_fingerprint=report["trust_certificate"]["fingerprint"],
        cwd=root,
        timeout_seconds=10.0,
    )
    (root / capsule_check.DEFAULT_ACTION_REVIEW_RECEIPT_NAME).write_text(
        json.dumps(action_receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / capsule_check.DEFAULT_ACTION_REVIEW_RECEIPT_MARKDOWN_NAME).write_text(
        capsule_check.render_inference_action_review_receipt_markdown(action_receipt),
        encoding="utf-8",
    )
    roundtrip_ml_profile = {
        "schema": "reverie_explain_ml_profile_v1",
        "goal_fit": "auditable_ml_kernel",
        "capabilities": [
            "deterministic_q31_fixed_point",
            "explicit_witness_tapes",
            "reversible_tensor_accumulation",
            "roundtrip_proof_ready",
            "static_replay_cost_model",
            "static_reversibility_check",
            "static_tensor_shapes",
        ],
        "tensor_metrics": {
            "known_witness_payload_bytes": 24,
        },
        "replay_cost": {
            "roundtrip_statement_count": 6,
            "known_witness_payload_bytes": 24,
            "known_replay_payload_bytes": 56,
        },
        "q31_builtin_calls": 1,
    }
    roundtrip = {
        "kind": "reverie_roundtrip_verification",
        "schema": "reverie_roundtrip_verification_v1",
        "passed": True,
        "proof_path": str(root / "reversible-inference-roundtrip-proof.json"),
        "proof_fingerprint": "1" * 64,
        "payload_fingerprint": "1" * 64,
        "replay_fingerprint": "1" * 64,
        "ml_profile_present": True,
        "ml_profile_fingerprint": sha256_json(roundtrip_ml_profile),
        "replay_ml_profile_fingerprint": sha256_json(roundtrip_ml_profile),
        "ml_profile": roundtrip_ml_profile,
        "checks": {
            "kind": True,
            "schema": True,
            "artifact_passed": True,
            "payload_fingerprint": True,
            "declared_payload_fingerprint": True,
            "baseline_fingerprint": True,
            "forward_fingerprint": True,
            "restored_fingerprint": True,
            "forward_witness_fingerprint": True,
            "restored_witness_fingerprint": True,
            "ml_profile_schema": True,
            "ml_profile_replayed": True,
            "source_readable": True,
            "source_hash_matches": True,
            "replayed": True,
            "replay_fingerprint_matches": True,
            "replay_restoration_passed": True,
        },
    }
    (root / "reversible-inference-roundtrip-verification.json").write_text(
        json.dumps(roundtrip, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "reversible-inference-roundtrip-proof.json").write_text(
        json.dumps({"kind": "reverie_roundtrip_result", "fingerprint": "1" * 64}, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "reversible-inference-roundtrip.md").write_text(
        "# Reversible Inference Roundtrip\n",
        encoding="utf-8",
    )
    (root / "training-step-bundle.json").write_text(
        json.dumps({"kind": "reverie_mnist_linear_q31_step_bundle"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "training-audit-step.json").write_text(
        json.dumps(profile_check.valid_audit_step_report(), indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "training-step-verification.json").write_text(
        json.dumps(profile_check.valid_step_verification_report(), indent=2) + "\n",
        encoding="utf-8",
    )
    synthetic_summary = load_json(root / capsule_check.DEFAULT_SUMMARY_NAME)
    synthetic_replay = synthetic_summary["metrics"]["training_audit_replay"]
    synthetic_q31 = synthetic_summary["metrics"]["q31_reference_inference"]
    synthetic_model_import = synthetic_summary["metrics"]["model_import"]
    synthetic_imported = synthetic_summary["metrics"]["imported_model_inference"]
    synthetic_native = synthetic_summary["metrics"]["native_inference_replay"]
    synthetic_trace = synthetic_summary["metrics"]["reversible_inference_trace"]
    synthetic_training_verification = {
        "kind": "reverie_mnist_linear_q31_audit_verification",
        "checked": synthetic_replay["checked"],
        "witnesses_match_forward_replay": synthetic_replay["witnesses_match_forward_replay"],
        "final_model_replayed": synthetic_replay["final_model_replayed"],
        "restored_initial_model": synthetic_replay["restored_initial_model"],
        "proof_matches": synthetic_replay["proof_matches"],
        "lineage_ledger_matches": synthetic_replay["lineage_ledger_matches"],
        "lineage_ledger": {
            "schema": "q31_training_lineage_ledger_v1",
            "algorithm": "sha256",
            "fingerprint": synthetic_replay["lineage_ledger_fingerprint"],
            "payload": {
                "schema": "q31_training_lineage_ledger_v1",
                "steps": synthetic_replay["checked"],
                "initial_model_fingerprint": "a" * 64,
                "final_model_fingerprint": "b" * 64,
                "sample_order_fingerprint": "c" * 64,
                "witness_trace_fingerprint": "d" * 64,
                "transition_ledger_fingerprint": synthetic_replay[
                    "transition_ledger_fingerprint"
                ],
                "final_chain": synthetic_replay["final_chain"],
            },
        },
    }
    synthetic_q31_report = {
        "kind": "reverie_q31_linear_reference_inference",
        "prediction": synthetic_q31["prediction"],
        "correct": synthetic_q31["correct"],
        "margin": synthetic_q31["margin"],
        "active_pixels": synthetic_q31["active_pixels"],
        "top_logits": [{"digit": synthetic_q31["prediction"], "value": synthetic_q31["margin"]}],
    }
    synthetic_model_import_report = {
        "kind": "reverie_mnist_linear_q31_model_import",
        "provenance_kind": "external_import",
        "source_model_json_fingerprint": synthetic_model_import[
            "source_model_json_fingerprint"
        ],
    }
    synthetic_model_verification_report = {
        "kind": "reverie_mnist_linear_q31_model_verification",
        "provenance_kind": "external_import",
        "source_model_json_checked": True,
        "training_steps": None,
    }
    synthetic_imported_inference_report = {
        "kind": "reverie_mnist_linear_q31_model_inference_audit",
        "prediction": synthetic_imported["prediction"],
        "correct": synthetic_imported["correct"],
    }
    synthetic_imported_verification = {
        "kind": "reverie_mnist_linear_q31_inference_verification",
        "prediction": synthetic_imported["prediction"],
        "proof_matches": True,
        "result_matches": True,
        "restored_initial_state": True,
        "memory": {
            "replay_payload_bytes": synthetic_imported["replay_payload_bytes"],
        },
    }
    synthetic_native_verification = {
        "kind": "reverie_mnist_linear_q31_inference_verification",
        "prediction": synthetic_q31["prediction"],
        "correct": synthetic_q31["correct"],
        "proof": {
            "claim": "deterministic_q31_inference_replay",
            "prediction": synthetic_q31["prediction"],
            "correct": synthetic_q31["correct"],
            "margin": synthetic_q31["margin"],
            "active_pixels": synthetic_q31["active_pixels"],
            "contribution_ledger_fingerprint": synthetic_q31[
                "contribution_ledger_fingerprint"
            ],
            "margin_contribution_ledger_fingerprint": synthetic_q31[
                "margin_contribution_ledger_fingerprint"
            ],
            "replay_payload_bytes": synthetic_native["replay_payload_bytes"],
        },
    }
    synthetic_trace_report = {
        "kind": "reverie_q31_reversible_inference_trace_reference",
        "passed": synthetic_trace["passed"],
        "proof": {
            "claim": "deterministic_q31_reversible_inference_trace",
            "replay_payload_bytes": synthetic_trace["replay_payload_bytes"],
            "witness_payload_bytes": synthetic_trace["witness_payload_bytes"],
            "trace_payload_bytes": synthetic_trace["trace_payload_bytes"],
            "forward_recompute_steps": synthetic_trace["forward_recompute_steps"],
            "inverse_recompute_steps": synthetic_trace["inverse_recompute_steps"],
            "total_recompute_steps": synthetic_trace["total_recompute_steps"],
            "checks": {
                "preprocess_matches_reference": True,
                "logits_match_reference": True,
                "prediction_matches_reference": True,
                "correctness_matches_reference": True,
                "reverse_restores_initial_state": True,
                "raw_preserved": True,
                "model_preserved": True,
                "balanced_recompute": True,
            },
        },
    }
    for filename, payload in (
        ("run-report.json", {"kind": "reverie_mnist_linear_q31"}),
        ("training-audit-bundle.json", {"kind": "reverie_mnist_linear_q31_audit_bundle"}),
        ("training-audit-verification.json", synthetic_training_verification),
        ("q31-reference-inference.json", synthetic_q31_report),
        ("imported-model-import-report.json", synthetic_model_import_report),
        ("imported-model-verification.json", synthetic_model_verification_report),
        ("imported-model-inference-report.json", synthetic_imported_inference_report),
        (
            "imported-model-inference-verification.json",
            synthetic_imported_verification,
        ),
        ("native-inference-verification.json", synthetic_native_verification),
        ("mlp-witness-vars.json", {"kind": "synthetic_mlp_vars"}),
        ("mlp-witness-run.json", {"kind": "synthetic_mlp_run"}),
        ("mlp-witness-report.json", {"kind": "synthetic_mlp_report"}),
        ("invertible-coupling-forward.json", {"kind": "synthetic_coupling_forward"}),
        ("invertible-coupling-reverse.json", {"kind": "synthetic_coupling_reverse"}),
        ("invertible-coupling-report.json", {"kind": "synthetic_coupling_report"}),
        ("triangular-residual-forward.json", {"kind": "synthetic_residual_forward"}),
        ("triangular-residual-reverse.json", {"kind": "synthetic_residual_reverse"}),
        ("triangular-residual-report.json", {"kind": "synthetic_residual_report"}),
        ("reversible-preprocess-forward.json", {"kind": "synthetic_preprocess_forward"}),
        ("reversible-preprocess-reverse.json", {"kind": "synthetic_preprocess_reverse"}),
        ("reversible-preprocess-report.json", {"kind": "synthetic_preprocess_report"}),
        ("reversible-inference-trace-forward.json", {"kind": "synthetic_trace_forward"}),
        ("reversible-inference-trace-reverse.json", {"kind": "synthetic_trace_reverse"}),
        ("reversible-inference-trace-report.json", synthetic_trace_report),
    ):
        (root / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for filename, title in (
        ("mlp-witness.md", "Synthetic MLP Witness"),
        ("invertible-coupling.md", "Synthetic Invertible Coupling"),
        ("triangular-residual.md", "Synthetic Triangular Residual"),
        ("reversible-preprocess.md", "Synthetic Reversible Preprocess"),
        ("reversible-inference-trace.md", "Synthetic Reversible Inference Trace"),
    ):
        (root / filename).write_text(f"# {title}\n", encoding="utf-8")
    training_commands = [
        {
            "operation": operation,
            "label": operation.replace("_", " ").title(),
            "purpose": f"Synthetic {operation} replay.",
            "command": [sys.executable, "-c", f"print({operation!r})"],
            "artifacts": ["training_step_bundle", "training_step_debug"],
        }
        for operation in (
            "replay_training_lineage",
            "inspect_training_update",
            "reverse_training_update",
        )
    ]
    audit_pipeline.write_training_update_review_receipt(
        audit_pipeline.PipelineConfig(
            out_dir=root,
            sample_limit=1,
            audit_step=0,
            evaluation_row=0,
            runner_bin=None,
        ),
        commands=training_commands,
    )
    benchmark = benchmark_check.valid_synthetic_artifact()
    benchmark["command_timeout_seconds"] = 30.0
    source_file = {
        "path": "/tmp/reverie-benchmark-fixture.rev",
        "sha256": "2" * 64,
    }
    benchmark["benchmarks"][0]["jana"]["source_files"] = [source_file]
    benchmark["benchmarks"][0]["reverie"]["source_files"] = [source_file]
    benchmark_json = root / "benchmark.json"
    benchmark_markdown = root / "benchmark.md"
    benchmark_json.write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")
    benchmark_markdown.write_text(
        summarize_janus_performance.render_markdown(benchmark),
        encoding="utf-8",
    )
    return root, benchmark_json, benchmark_markdown


def run_self_tests() -> int:
    try:
        with tempfile.TemporaryDirectory() as directory:
            root, benchmark_json, benchmark_markdown = self_test_report(Path(directory))
            report = build_gate_report(
                root,
                benchmark_json,
                benchmark_markdown,
                min_workloads=1,
                expected_runs=2,
                expected_warmup=0,
                min_observed_speedup=1.5,
                min_median_speedup=1.5,
                min_geomean_speedup=1.5,
                min_observed_rss_ratio=1.0,
                min_median_rss_ratio=1.0,
                verify_file_digests=False,
                require_current_workloads=False,
            )
            if report["passed"] is not True:
                raise AssertionError("synthetic north-star gate should pass")
            if report["payload"]["roadmap"]["capabilities"]["total"] != 6:
                raise AssertionError("synthetic gate should validate all roadmap phases")
            claim_evidence = report["payload"]["claim_evidence"]
            if claim_evidence["claims"] != 10:
                raise AssertionError("synthetic gate should summarize every ML claim")
            if claim_evidence["integrity_claim_covers_all_files"] is not True:
                raise AssertionError("synthetic gate should prove full evidence coverage")
            if claim_evidence["evidence_files"] != claim_evidence["integrity_claim_files"]:
                raise AssertionError("integrity claim should cite every evidence file")
            gate_goal_contract = report["payload"]["goal_contract"]
            if goal_contract_errors(gate_goal_contract):
                raise AssertionError("synthetic gate goal contract should validate")
            if gate_goal_contract["passed_claims"] != len(GOAL_CONTRACT_CLAIMS):
                raise AssertionError("synthetic gate should prove every goal-contract claim")
            debug_update = report["payload"]["debug_update"]
            if debug_update["claim"] != "step_backward_from_model_update":
                raise AssertionError("synthetic gate should expose the debug training update")
            model_lineage = report["payload"]["model_lineage"]
            if model_lineage["checked_steps"] != model_lineage["train_samples"]:
                raise AssertionError("synthetic gate should expose complete model lineage replay")
            deterministic_inference = report["payload"]["deterministic_inference"]
            if (
                deterministic_inference["trace_count"] < 2
                or deterministic_inference["native_replay_payload_bytes"] <= 0
            ):
                raise AssertionError("synthetic gate should expose deterministic inference evidence")
            action_operations = {
                action["operation"] for action in deterministic_inference["action_contract"]
            }
            if action_operations != {
                "reproduce_prediction",
                "explain_margin",
                "replay_imported_model_inference",
                "replay_native_inference",
                "run_standalone_rev_classifier",
                "reverse_reversible_trace",
            }:
                raise AssertionError("synthetic gate should expose the deterministic inference action contract")
            command_ids = set(report["payload"]["reviewer_command_summary"]["ids"])
            for command_id in REQUIRED_RELEASE_REPLAY_COMMANDS:
                if command_id not in command_ids:
                    raise AssertionError(f"synthetic gate missing reviewer command {command_id}")
            selected = select_reviewer_commands(report, ["roundtrip_proof"])
            if selected[0]["id"] != "roundtrip_proof":
                raise AssertionError("reviewer command selector returned the wrong command")
            try:
                select_reviewer_commands(report, ["missing_command"])
            except ValueError as error:
                if "unknown reviewer command" not in str(error):
                    raise AssertionError(f"unexpected reviewer command error: {error}") from error
            else:
                raise AssertionError("unknown reviewer command should fail")
            rendered = render_markdown(report)
            if "## Reviewer Commands" not in rendered or "verify-roundtrip" not in rendered:
                raise AssertionError("rendered gate Markdown should include reviewer replay commands")
            if "## Goal Contract" not in rendered or "Claims proven" not in rendered:
                raise AssertionError("rendered gate Markdown should include goal contract proofs")
            if "## Claim Evidence" not in rendered or "all files" not in rendered:
                raise AssertionError("rendered gate Markdown should include claim evidence coverage")
            if "Roundtrip replay payload" not in rendered or "Roundtrip static statements" not in rendered:
                raise AssertionError("rendered gate Markdown should include roundtrip replay cost")
            if "## Model Lineage" not in rendered or "Final model replayed" not in rendered:
                raise AssertionError("rendered gate Markdown should include model lineage evidence")
            if (
                "## Deterministic Inference" not in rendered
                or "Native replay payload" not in rendered
                or "Import provenance" not in rendered
                or "### Action Contract" not in rendered
                or "replay_imported_model_inference" not in rendered
                or "run_standalone_rev_classifier" not in rendered
                or "reverse_reversible_trace" not in rendered
            ):
                raise AssertionError("rendered gate Markdown should include deterministic inference evidence")
            if "## Debuggable Update" not in rendered or "step_backward_from_model_update" not in rendered:
                raise AssertionError("rendered gate Markdown should include debug update evidence")
            if "Training update receipt" not in rendered:
                raise AssertionError("rendered gate Markdown should include training update receipt evidence")
            transcript_report = {
                "fingerprint": "3" * 64,
                "passed": True,
                "payload": {
                    "reviewer_commands": {
                        "commands": [
                            command_record(
                                "profile_evidence",
                                "Profile evidence test",
                                "Exercise reviewer transcript capture.",
                                ["python3", "-c", "print('review transcript ok')"],
                                [],
                            ),
                            command_record(
                                "capsule_handoff",
                                "Capsule handoff test",
                                "Exercise capsule handoff transcript capture.",
                                ["python3", "-c", "print('handoff transcript ok')"],
                                [],
                            ),
                            command_record(
                                "training_audit_lineage",
                                "Training audit lineage test",
                                "Exercise training lineage transcript capture.",
                                ["python3", "-c", "print('lineage transcript ok')"],
                                [],
                            ),
                            command_record(
                                "benchmark_speed_memory",
                                "Benchmark speed and memory test",
                                "Exercise benchmark transcript capture.",
                                ["python3", "-c", "print('benchmark transcript ok')"],
                                [],
                            ),
                            command_record(
                                "roundtrip_proof",
                                "Roundtrip proof test",
                                "Exercise reviewer transcript capture.",
                                ["python3", "-c", "print('roundtrip transcript ok')"],
                                [],
                            ),
                            command_record(
                                "training_step_debug",
                                "Training step debug test",
                                "Exercise reviewer transcript capture.",
                                ["python3", "-c", "print('training transcript ok')"],
                                [],
                            ),
                            command_record(
                                "training_update_receipt",
                                "Training update receipt test",
                                "Exercise training update receipt transcript capture.",
                                ["python3", "-c", "print('training receipt transcript ok')"],
                                [],
                            ),
                            command_record(
                                "mlp_witness_proof",
                                "MLP witness proof test",
                                "Exercise MLP witness transcript capture.",
                                ["python3", "-c", "print('mlp transcript ok')"],
                                [],
                            ),
                            command_record(
                                "invertible_coupling_proof",
                                "Invertible coupling proof test",
                                "Exercise invertible coupling transcript capture.",
                                ["python3", "-c", "print('coupling transcript ok')"],
                                [],
                            ),
                            command_record(
                                "triangular_residual_proof",
                                "Triangular residual proof test",
                                "Exercise triangular residual transcript capture.",
                                ["python3", "-c", "print('residual transcript ok')"],
                                [],
                            ),
                            command_record(
                                "reversible_preprocess_proof",
                                "Reversible preprocessing proof test",
                                "Exercise reversible preprocessing transcript capture.",
                                ["python3", "-c", "print('preprocess transcript ok')"],
                                [],
                            ),
                            command_record(
                                "reversible_inference_trace_proof",
                                "Reversible inference trace proof test",
                                "Exercise reversible inference trace transcript capture.",
                                ["python3", "-c", "print('trace transcript ok')"],
                                [],
                            ),
                            command_record(
                                "inference_action_receipt",
                                "Inference action receipt test",
                                "Exercise inference action receipt transcript capture.",
                                ["python3", "-c", "print('action receipt transcript ok')"],
                                [],
                            ),
                        ]
                    }
                },
            }
            transcript = run_reviewer_commands(
                transcript_report,
                list(REQUIRED_RELEASE_REPLAY_COMMANDS),
                emit_output=False,
            )
            transcript = finalize_transcript(
                transcript,
                {"fingerprint": "3" * 64, "passed": True},
            )
            if transcript["passed"] is not True:
                raise AssertionError("synthetic reviewer transcript should pass")
            if transcript["payload"]["commands"][0]["stdout_sha256"] != sha256_text(
                "review transcript ok\n"
            ):
                raise AssertionError("reviewer transcript should hash stdout")
            if "Reviewer Replay Transcript" not in render_transcript_markdown(transcript):
                raise AssertionError("reviewer transcript Markdown should render")
            transcript_path = root / DEFAULT_TRANSCRIPT_NAME
            transcript_markdown = root / DEFAULT_TRANSCRIPT_MARKDOWN_NAME
            transcript_path.write_text(
                json.dumps(transcript, indent=2) + "\n",
                encoding="utf-8",
            )
            transcript_markdown.write_text(
                render_transcript_markdown(transcript),
                encoding="utf-8",
            )
            checked_transcript = validate_saved_transcript(
                {"fingerprint": "3" * 64, "passed": True, "payload": transcript_report["payload"]},
                transcript_path,
                transcript_markdown,
            )
            if checked_transcript["fingerprint"] != transcript["fingerprint"]:
                raise AssertionError("saved reviewer transcript should validate")
            release_report = dict(report)
            release_report["fingerprint"] = "3" * 64
            (root / DEFAULT_REPORT_NAME).write_text(
                json.dumps(release_report, indent=2) + "\n",
                encoding="utf-8",
            )
            (root / DEFAULT_MARKDOWN_NAME).write_text(
                render_markdown(release_report),
                encoding="utf-8",
            )
            release_verification = build_release_verification(
                release_report,
                checked_transcript,
                gate_output=root / DEFAULT_REPORT_NAME,
                gate_markdown=root / DEFAULT_MARKDOWN_NAME,
            )
            if release_verification["passed"] is not True:
                raise AssertionError("release verification should pass")
            if release_verification["fingerprint"] != sha256_json(
                release_verification["payload"]
            ):
                raise AssertionError("release verification fingerprint should match payload")
            rendered_release = render_release_verification_markdown(release_verification)
            if "Release Verification" not in rendered_release:
                raise AssertionError("release verification Markdown should render")
            if "## Goal Contract" not in rendered_release:
                raise AssertionError("release verification should render goal contract")
            goal_contract = release_verification["payload"]["goal_contract"]
            goal_contract_payload = {
                key: value for key, value in goal_contract.items() if key != "fingerprint"
            }
            if goal_contract["fingerprint"] != sha256_json(goal_contract_payload):
                raise AssertionError("release verification should fingerprint the goal contract")
            if goal_contract["north_star"] != NORTH_STAR or goal_contract["non_goal"] != NON_GOAL:
                raise AssertionError("release verification should bind goal and non-goal")
            if tuple(goal_contract["claims"]) != GOAL_CONTRACT_CLAIMS:
                raise AssertionError("release verification should bind the allowed claim set")
            if release_verification["payload"]["checks"]["goal_contract_bound"] is not True:
                raise AssertionError("release verification should check the goal contract")
            if release_verification["payload"]["checks"]["gate_goal_contract_fresh"] is not True:
                raise AssertionError("release verification should check the gate goal contract")
            if release_verification["payload"]["checks"]["goal_contract_claims_proven"] is not True:
                raise AssertionError("release verification should prove goal-contract claims")
            if (
                release_verification["payload"]["checks"]["claim_replay_coverage_complete"]
                is not True
            ):
                raise AssertionError("release verification should prove claim replay coverage")
            if goal_contract["passed_claims"] != goal_contract["total_claims"]:
                raise AssertionError("all goal-contract claims should be proven")
            if any(proof["passed"] is not True for proof in goal_contract["claim_proofs"]):
                raise AssertionError("each goal-contract claim proof should pass")
            if "Claims proven" not in rendered_release:
                raise AssertionError("release verification should render claim proofs")
            replay_coverage = release_verification["payload"]["claim_replay_coverage"]
            if replay_coverage["passed_claims"] != replay_coverage["total_claims"]:
                raise AssertionError("all goal-contract claims should have replay coverage")
            if replay_coverage["missing_artifact_ids"]:
                raise AssertionError("all goal-contract claims should cover replay artifacts")
            if "## Claim Replay Coverage" not in rendered_release:
                raise AssertionError("release verification should render claim replay coverage")
            missing_claim_report = json.loads(json.dumps(release_report))
            claim_matrix = [
                row
                for row in missing_claim_report["payload"]["claim_evidence"]["claim_matrix"]
                if row["claim"] != "debug_training_update"
            ]
            missing_claim_report["payload"]["claim_evidence"]["claim_matrix"] = claim_matrix
            missing_claim_report["payload"]["claim_evidence"][
                "claim_matrix_fingerprint"
            ] = sha256_json(claim_matrix)
            missing_claim_release = build_release_verification(
                missing_claim_report,
                checked_transcript,
                gate_output=root / DEFAULT_REPORT_NAME,
                gate_markdown=root / DEFAULT_MARKDOWN_NAME,
            )
            if missing_claim_release["passed"] is not False:
                raise AssertionError("release verification should fail with missing goal claim evidence")
            if missing_claim_release["payload"]["checks"]["gate_goal_contract_fresh"] is not False:
                raise AssertionError("gate goal-contract freshness check should fail")
            if (
                missing_claim_release["payload"]["checks"]["goal_contract_claims_proven"]
                is not False
            ):
                raise AssertionError("goal-contract claim proof check should fail")
            missing_command_transcript = dict(checked_transcript)
            missing_command_transcript["command_ids"] = ["profile_evidence", "roundtrip_proof"]
            missing_command_transcript["commands"] = len(missing_command_transcript["command_ids"])
            missing_command_release = build_release_verification(
                release_report,
                missing_command_transcript,
                gate_output=root / DEFAULT_REPORT_NAME,
                gate_markdown=root / DEFAULT_MARKDOWN_NAME,
            )
            if missing_command_release["passed"] is not False:
                raise AssertionError("release verification should fail with missing required commands")
            if (
                missing_command_release["payload"]["checks"]["required_reviewer_commands_replayed"]
                is not False
            ):
                raise AssertionError("required reviewer command check should fail")
            if (
                missing_command_release["payload"]["checks"]["claim_replay_coverage_complete"]
                is not False
            ):
                raise AssertionError("claim replay coverage check should fail")
            missing_artifact_report = json.loads(json.dumps(release_report))
            for command in missing_artifact_report["payload"]["reviewer_commands"]["commands"]:
                if command["id"] == "mlp_witness_proof":
                    command["artifacts"] = [
                        artifact_id
                        for artifact_id in command["artifacts"]
                        if artifact_id != "mlp_witness"
                    ]
            missing_artifact_release = build_release_verification(
                missing_artifact_report,
                checked_transcript,
                gate_output=root / DEFAULT_REPORT_NAME,
                gate_markdown=root / DEFAULT_MARKDOWN_NAME,
            )
            if missing_artifact_release["passed"] is not False:
                raise AssertionError("release verification should fail with missing replay artifacts")
            if (
                missing_artifact_release["payload"]["checks"][
                    "claim_replay_coverage_complete"
                ]
                is not False
            ):
                raise AssertionError("claim replay coverage should fail missing artifacts")
            if (
                "mlp_witness"
                not in missing_artifact_release["payload"]["claim_replay_coverage"][
                    "missing_artifact_ids"
                ]
            ):
                raise AssertionError("missing replay artifact should be reported")
            verifier_sources = release_verification["payload"]["verifier_sources"]
            if verifier_sources["fingerprint"] != sha256_json(verifier_sources["files"]):
                raise AssertionError("release verification should fingerprint verifier sources")
            if tuple(verifier_sources["files"]) != RELEASE_VERIFIER_SOURCE_FILES:
                raise AssertionError("release verification should bind expected verifier sources")
            release_artifacts = release_verification["payload"]["release_artifacts"]
            if release_artifacts["fingerprint"] != sha256_json(release_artifacts["files"]):
                raise AssertionError("release verification should fingerprint release artifacts")
            if tuple(release_artifacts["files"]) != RELEASE_PACKET_ARTIFACT_IDS:
                raise AssertionError("release verification should bind expected release artifacts")
            audit_packet_artifacts = release_verification["payload"]["audit_packet_artifacts"]
            if audit_packet_artifacts["count"] != len(release_report["payload"]["artifacts"]):
                raise AssertionError("release verification should summarize gate artifacts")
            if audit_packet_artifacts["bytes"] <= 0:
                raise AssertionError("release verification should total gate artifact bytes")
            replay_commands = release_verification["payload"]["reviewer_replay_commands"]
            if "--verify-release-verification" not in replay_commands["saved_receipt_check"]:
                raise AssertionError("release verification should include a saved-receipt check command")
            if "--verify-reviewer-transcript" not in replay_commands["regenerate_receipt"]:
                raise AssertionError("release verification should include a regenerate command")
            release_path = root / DEFAULT_RELEASE_VERIFICATION_NAME
            release_markdown = root / DEFAULT_RELEASE_VERIFICATION_MARKDOWN_NAME
            release_path.write_text(
                json.dumps(release_verification, indent=2) + "\n",
                encoding="utf-8",
            )
            release_markdown.write_text(
                render_release_verification_markdown(release_verification),
                encoding="utf-8",
            )
            checked_release = validate_saved_release_verification(
                release_report,
                checked_transcript,
                release_path,
                release_markdown,
                gate_output=root / DEFAULT_REPORT_NAME,
                gate_markdown=root / DEFAULT_MARKDOWN_NAME,
            )
            if checked_release["fingerprint"] != release_verification["fingerprint"]:
                raise AssertionError("saved release verification should validate")
            release_markdown.write_text("# stale\n", encoding="utf-8")
            try:
                validate_saved_release_verification(
                    release_report,
                    checked_transcript,
                    release_path,
                    release_markdown,
                    gate_output=root / DEFAULT_REPORT_NAME,
                    gate_markdown=root / DEFAULT_MARKDOWN_NAME,
                )
            except ValueError as error:
                if "release verification Markdown is stale" not in str(error):
                    raise AssertionError(
                        f"unexpected stale release verification error: {error}"
                    ) from error
            else:
                raise AssertionError("stale release verification Markdown should fail")
            tampered_transcript = json.loads(json.dumps(transcript))
            tampered_transcript["payload"]["commands"][0]["stdout"] = "tampered\n"
            tampered_transcript["fingerprint"] = sha256_json(tampered_transcript["payload"])
            transcript_path.write_text(
                json.dumps(tampered_transcript, indent=2) + "\n",
                encoding="utf-8",
            )
            transcript_markdown.write_text(
                render_transcript_markdown(tampered_transcript),
                encoding="utf-8",
            )
            try:
                validate_saved_transcript(
                    {"fingerprint": "3" * 64, "passed": True, "payload": transcript_report["payload"]},
                    transcript_path,
                    transcript_markdown,
                )
            except ValueError as error:
                if "stdout_sha256 does not match stdout" not in str(error):
                    raise AssertionError(f"unexpected tampered transcript error: {error}") from error
            else:
                raise AssertionError("tampered reviewer transcript should fail")
            stale_handoff = root / capsule_check.DEFAULT_HANDOFF_MARKDOWN_NAME
            stale_handoff.write_text("# stale\n", encoding="utf-8")
            try:
                build_gate_report(
                    root,
                    benchmark_json,
                    benchmark_markdown,
                    min_workloads=1,
                    expected_runs=2,
                    expected_warmup=0,
                    min_observed_speedup=1.5,
                    min_median_speedup=1.5,
                    min_geomean_speedup=1.5,
                    min_observed_rss_ratio=1.0,
                    min_median_rss_ratio=1.0,
                    verify_file_digests=False,
                    require_current_workloads=False,
                )
            except ValueError as error:
                if "handoff Markdown is stale" not in str(error):
                    raise AssertionError(f"unexpected stale handoff error: {error}") from error
            else:
                raise AssertionError("stale handoff Markdown should fail")
        with tempfile.TemporaryDirectory() as directory:
            root, benchmark_json, benchmark_markdown = self_test_report(Path(directory))
            benchmark = load_json(benchmark_json)
            benchmark["benchmarks"][0]["memory_ratio"] = None
            benchmark_json.write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")
            benchmark_markdown.write_text(
                summarize_janus_performance.render_markdown(benchmark),
                encoding="utf-8",
            )
            try:
                build_gate_report(
                    root,
                    benchmark_json,
                    benchmark_markdown,
                    min_workloads=1,
                    expected_runs=2,
                    expected_warmup=0,
                    min_observed_speedup=1.5,
                    min_median_speedup=1.5,
                    min_geomean_speedup=1.5,
                    min_observed_rss_ratio=1.0,
                    min_median_rss_ratio=1.0,
                    verify_file_digests=False,
                    require_current_workloads=False,
                )
            except ValueError as error:
                if "missing RSS ratio" not in str(error):
                    raise AssertionError(f"unexpected missing-memory error: {error}") from error
            else:
                raise AssertionError("missing RSS ratio should fail")
    except (AssertionError, ValueError) as error:
        print(f"error: self-test failed: {error}", file=sys.stderr)
        return 1
    print("ok: ML north-star gate self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    for name, value in (
        ("--min-workloads", args.min_workloads),
        ("--expect-runs", args.expect_runs),
        ("--expect-warmup", args.expect_warmup),
    ):
        if value < 0 or (name != "--expect-warmup" and value == 0):
            print(f"error: {name} must be positive", file=sys.stderr)
            return 1
    for name, value in (
        ("--min-observed-speedup", args.min_observed_speedup),
        ("--min-median-speedup", args.min_median_speedup),
        ("--min-geomean-speedup", args.min_geomean_speedup),
        ("--min-observed-rss-ratio", args.min_observed_rss_ratio),
        ("--min-median-rss-ratio", args.min_median_rss_ratio),
    ):
        if value <= 0:
            print(f"error: {name} must be positive", file=sys.stderr)
            return 1
    try:
        report = build_gate_report(
            args.audit_dir,
            args.benchmark_json,
            args.benchmark_markdown,
            min_workloads=args.min_workloads,
            expected_runs=args.expect_runs,
            expected_warmup=args.expect_warmup,
            min_observed_speedup=args.min_observed_speedup,
            min_median_speedup=args.min_median_speedup,
            min_geomean_speedup=args.min_geomean_speedup,
            min_observed_rss_ratio=args.min_observed_rss_ratio,
            min_median_rss_ratio=args.min_median_rss_ratio,
            verify_file_digests=not args.skip_file_digests,
            require_current_workloads=not args.allow_stale_workload_list,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.list_reviewer_commands:
        print_reviewer_commands(report)
        return 0

    verify_transcript = args.verify_reviewer_transcript or args.verify_release_verification
    transcript = None
    if args.run_reviewer_command:
        try:
            transcript = run_reviewer_commands(report, args.run_reviewer_command)
            report = build_gate_report(
                args.audit_dir,
                args.benchmark_json,
                args.benchmark_markdown,
                min_workloads=args.min_workloads,
                expected_runs=args.expect_runs,
                expected_warmup=args.expect_warmup,
                min_observed_speedup=args.min_observed_speedup,
                min_median_speedup=args.min_median_speedup,
                min_geomean_speedup=args.min_geomean_speedup,
                min_observed_rss_ratio=args.min_observed_rss_ratio,
                min_median_rss_ratio=args.min_median_rss_ratio,
                verify_file_digests=not args.skip_file_digests,
                require_current_workloads=not args.allow_stale_workload_list,
            )
            transcript = finalize_transcript(transcript, report)
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1

    output = args.output or (args.audit_dir / DEFAULT_REPORT_NAME)
    markdown_output = args.markdown_output or (args.audit_dir / DEFAULT_MARKDOWN_NAME)
    transcript_output = args.reviewer_transcript_output or (
        args.audit_dir / DEFAULT_TRANSCRIPT_NAME
    )
    transcript_markdown = args.reviewer_transcript_markdown or (
        args.audit_dir / DEFAULT_TRANSCRIPT_MARKDOWN_NAME
    )
    release_verification_output = args.release_verification_output or (
        args.audit_dir / DEFAULT_RELEASE_VERIFICATION_NAME
    )
    release_verification_markdown = args.release_verification_markdown or (
        args.audit_dir / DEFAULT_RELEASE_VERIFICATION_MARKDOWN_NAME
    )
    write_report(output, json.dumps(report, indent=2) + "\n")
    write_report(markdown_output, render_markdown(report))
    if transcript is not None:
        write_report(transcript_output, json.dumps(transcript, indent=2) + "\n")
        write_report(transcript_markdown, render_transcript_markdown(transcript))
        if transcript["passed"] is not True:
            print(f"reviewer transcript: {transcript_output}")
            print(f"reviewer transcript markdown: {transcript_markdown}")
            print("error: reviewer replay command failed", file=sys.stderr)
            return 1
    verified_transcript = None
    verified_release = None
    if verify_transcript:
        try:
            verified_transcript = validate_saved_transcript(
                report,
                transcript_output,
                transcript_markdown,
            )
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    if args.verify_reviewer_transcript:
        release_verification = build_release_verification(
            report,
            verified_transcript,
            gate_output=output,
            gate_markdown=markdown_output,
        )
        write_report(
            release_verification_output,
            json.dumps(release_verification, indent=2) + "\n",
        )
        write_report(
            release_verification_markdown,
            render_release_verification_markdown(release_verification),
        )
        if release_verification["passed"] is not True:
            print("error: release verification failed", file=sys.stderr)
            return 1
    if args.verify_release_verification:
        assert verified_transcript is not None
        try:
            verified_release = validate_saved_release_verification(
                report,
                verified_transcript,
                release_verification_output,
                release_verification_markdown,
                gate_output=output,
                gate_markdown=markdown_output,
            )
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    payload = report["payload"]
    benchmark = payload["benchmark"]
    print("ok: verified Reverie ML north-star gate")
    print(f"report: {output}")
    print(f"markdown: {markdown_output}")
    if transcript is not None:
        print(f"reviewer transcript: {transcript_output}")
        print(f"reviewer transcript markdown: {transcript_markdown}")
    if verified_transcript is not None:
        print(f"verified reviewer transcript: {verified_transcript['path']}")
        print(f"reviewer transcript fingerprint: {verified_transcript['fingerprint']}")
    if args.verify_reviewer_transcript:
        print(f"release verification: {release_verification_output}")
        print(f"release verification markdown: {release_verification_markdown}")
        print(f"release verification fingerprint: {release_verification['fingerprint']}")
    if verified_release is not None:
        print(f"verified release verification: {verified_release['path']}")
        print(f"release verification fingerprint: {verified_release['fingerprint']}")
    print(f"gate: {report['fingerprint']}")
    print(f"capsule: {payload['capsule']['fingerprint']}")
    print(f"handoff: {payload['handoff']['fingerprint']}")
    print(f"inference action receipt: {payload['inference_action_review_receipt']['fingerprint']}")
    print(f"training update receipt: {payload['training_update_review_receipt']['fingerprint']}")
    print(
        "benchmark: "
        f"{benchmark['workloads']} workloads, median speedup "
        f"{benchmark['median_speedup']:.2f}x, median RSS ratio "
        f"{benchmark['median_rss_ratio']:.2f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
