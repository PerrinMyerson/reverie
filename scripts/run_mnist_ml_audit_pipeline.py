#!/usr/bin/env python3
"""Build a checked local MNIST ML audit profile from Reverie's self-test path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import check_q31_mlp_witness as mlp_check
import check_invertible_coupling as coupling_check
import check_triangular_residual as residual_check
import check_reversible_preprocess as preprocess_check
import check_reversible_inference_trace as inference_trace_check
import check_mnist_ml_profile as profile_check
import summarize_mnist_ml_profile as profile_summary
import verify_model_capsule as capsule_verify


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_KIND = "reverie_mnist_ml_audit_pipeline"
SUMMARY_KIND = "reverie_mnist_ml_audit_pipeline_summary"
CLAIMS_KIND = "reverie_reversible_ml_audit_claims"
MODEL_CAPSULE_KIND = "reverie_reversible_ml_model_capsule"
MODEL_CAPSULE_SCHEMA = "reverie_reversible_ml_model_capsule_v1"
ML_CAPABILITY_KIND = "reverie_mnist_ml_capability_map"
ML_GOAL_READINESS_KIND = "reverie_reversible_ml_goal_readiness"
RECOMPUTE_FRONTIER_KIND = "reverie_mnist_ml_recompute_frontier"
SCALING_PROJECTION_KIND = "reverie_mnist_ml_scaling_projection"
INFERENCE_TRACE_PROFILE_KIND = "reverie_mnist_ml_inference_trace_profile"
HANDOFF_KIND = "reverie_mnist_ml_audit_handoff"
HANDOFF_SCHEMA = "reverie_mnist_ml_audit_handoff_v1"
TRAINING_UPDATE_REVIEW_RECEIPT_KIND = "reverie_training_update_review_receipt"
TRAINING_UPDATE_REVIEW_RECEIPT_SCHEMA = "reverie_training_update_review_receipt_v1"
DEFAULT_TRAINING_UPDATE_REVIEW_RECEIPT_NAME = "training-update-review-receipt.json"
DEFAULT_TRAINING_UPDATE_REVIEW_RECEIPT_MARKDOWN_NAME = "training-update-review-receipt.md"
DEFAULT_MIN_TRAIN_ACCURACY = 90.0
DEFAULT_MIN_EVAL_ACCURACY = 90.0
DEFAULT_MIN_AUDIT_ACCURACY = 90.0
DEFAULT_MIN_MODEL_EVALUATION_ACCURACY = 90.0
DEFAULT_MIN_REFERENCE_ACCURACY = 90.0
DEFAULT_MAX_WITNESS_MISMATCHES = 0
DEFAULT_MAX_TRACE_MODEL_RATIO = 1.0
DEFAULT_MAX_WITNESS_MODEL_RATIO = 0.25
DEFAULT_MAX_REVERSE_TRAIN_ELAPSED_RATIO = 5.0
AUDIT_STEP_STRATEGIES = ("explicit", "lowest-margin", "largest-update", "top-suspicious")
EVALUATION_ROW_STRATEGIES = ("explicit", "lowest-margin", "top-incorrect")
PROFILE_REPORT_KEYS = (
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
HANDOFF_ARTIFACT_ROWS = (
    (
        "pipeline_summary",
        "summary",
        "machine_summary",
        "Pipeline summary JSON",
        "Gate results, roadmap readiness, scorecard, and evidence index.",
    ),
    (
        "pipeline_manifest",
        "manifest",
        "machine_manifest",
        "Pipeline manifest JSON",
        "Executed commands, manifest fingerprint, reports, bundles, and evidence.",
    ),
    (
        "model_capsule",
        "model_capsule",
        "machine_capsule",
        "Model capsule JSON",
        "Fingerprint-bound model capsule payload and reversible ML claims.",
    ),
    (
        "model_capsule_verification",
        "model_capsule_verification",
        "machine_verdict",
        "Model capsule verification JSON",
        "Post-manifest trust-certificate verdict over the capsule handoff.",
    ),
    (
        "model_capsule_verification_markdown",
        "model_capsule_verification_markdown",
        "human_verdict",
        "Model capsule verification Markdown",
        "Readable trust-certificate table for reviewers.",
    ),
    (
        "model_bundle",
        "model_bundle",
        "machine_model",
        "Signed model bundle JSON",
        "Final model bundle used by deterministic inference replay commands.",
    ),
    (
        "imported_model_source",
        "imported_model_source",
        "machine_import_source",
        "Imported Q31 model source JSON",
        "External-import shaped weights/bias payload used to prove the import path.",
    ),
    (
        "imported_model_bundle",
        "imported_model_bundle",
        "machine_model",
        "Imported signed model bundle JSON",
        "Signed external-import model bundle with source JSON provenance.",
    ),
    (
        "imported_model_inference_bundle",
        "imported_model_inference_bundle",
        "machine_replay",
        "Imported model inference replay bundle",
        "Replayable inference bundle produced from the imported model.",
    ),
    (
        "samples",
        "samples",
        "machine_samples",
        "Signed sample set JSON",
        "Sample set used by deterministic inference replay commands.",
    ),
    (
        "model_capsule_profile",
        "model_capsule_profile",
        "human_profile",
        "Model capsule profile",
        "Capsule-first profile of gates, readiness, and capability evidence.",
    ),
    (
        "ml_profile",
        "profile_markdown",
        "human_profile",
        "Full ML audit profile",
        "Full Markdown profile spanning speed, memory, witnesses, and replay cost.",
    ),
    (
        "training_audit_verification_markdown",
        "audit_verification_markdown",
        "human_lineage_card",
        "Training-audit verification card",
        "Readable full-trace lineage, replay-cost, and reverse-restoration proof.",
    ),
    (
        "training_step_debug",
        "training_step_debug_markdown",
        "human_debug_card",
        "Training-step debug card",
        "Selected update explanation with witnesses, deltas, ledgers, and replay proof.",
    ),
    (
        "q31_reference_inference",
        "q31_reference_inference",
        "machine_reference",
        "Q31 reference inference JSON",
        "Independent deterministic Q31 prediction and attribution report.",
    ),
    (
        "native_inference_bundle",
        "native_inference_bundle",
        "machine_replay",
        "Native inference replay bundle JSON",
        "Replayable native inference bundle for the selected sample.",
    ),
    (
        "native_standalone_rev_classifier",
        "native_standalone_rev_classifier",
        "source_reverie",
        "Standalone MNIST classifier source",
        "Self-contained .rev source with model, sample, prediction, and assertions embedded.",
    ),
    (
        "native_standalone_rev_run",
        "native_standalone_rev_run",
        "machine_run",
        "Standalone MNIST classifier run JSON",
        "Generic Reverie run result for the generated source-only classifier.",
    ),
    (
        "native_standalone_rev_roundtrip",
        "native_standalone_rev_roundtrip",
        "machine_roundtrip",
        "Standalone MNIST classifier roundtrip JSON",
        "Generic Reverie forward/backward proof for the generated source-only classifier.",
    ),
    (
        "native_standalone_rev_roundtrip_verification",
        "native_standalone_rev_roundtrip_verification",
        "machine_roundtrip_verdict",
        "Standalone MNIST classifier roundtrip verification JSON",
        "Replay verification for the generated source-only classifier proof.",
    ),
    (
        "q31_reference_inference_markdown",
        "q31_reference_inference_markdown",
        "human_explanation_card",
        "Q31 inference explanation card",
        "Readable prediction, margin, top-logit, and contribution-ledger card.",
    ),
    (
        "native_inference_audit_markdown",
        "native_inference_audit_markdown",
        "human_inference_audit_card",
        "Native inference audit card",
        "Readable one-step prediction explanation with logits, attribution, source checks, and reverse replay proof.",
    ),
    (
        "native_inference_verification_markdown",
        "native_inference_verification_markdown",
        "human_inference_verification_card",
        "Native inference verification card",
        "Readable verified prediction, attribution ledgers, source checks, and replay costs.",
    ),
    (
        "evaluation_row_inference_audit_markdown",
        "evaluation_row_inference_audit_markdown",
        "human_evaluation_inference_card",
        "Evaluation-row inference audit card",
        "Readable signed-evaluation-row prediction explanation with logits, attribution, and reverse replay proof.",
    ),
    (
        "invertible_coupling_markdown",
        "invertible_coupling_markdown",
        "human_layer_card",
        "Invertible coupling proof card",
        "Readable additive-coupling state, reverse contract, and zero-witness payload proof.",
    ),
    (
        "triangular_residual_markdown",
        "triangular_residual_markdown",
        "human_layer_card",
        "Triangular residual proof card",
        "Readable triangular residual state, reverse contract, and zero-witness payload proof.",
    ),
    (
        "reversible_preprocess_markdown",
        "reversible_preprocess_markdown",
        "human_preprocess_card",
        "Reversible preprocess proof card",
        "Readable preprocessing steps, reverse contract, and zero-witness payload proof.",
    ),
    (
        "reversible_inference_trace",
        "reversible_inference_trace",
        "machine_replay",
        "Reversible inference trace JSON",
        "Forward/reverse deterministic inference trace proof.",
    ),
    (
        "inference_trace_forward",
        "inference_trace_forward",
        "machine_trace",
        "Reversible inference trace forward JSON",
        "Forward execution state for the reversible inference trace checker.",
    ),
    (
        "inference_trace_reverse",
        "inference_trace_reverse",
        "machine_trace",
        "Reversible inference trace reverse JSON",
        "Reverse execution state for the reversible inference trace checker.",
    ),
    (
        "reversible_inference_trace_markdown",
        "reversible_inference_trace_markdown",
        "human_trace_card",
        "Reversible inference trace card",
        "Readable prediction, attribution, reverse-restoration, and payload proof.",
    ),
    (
        "reversible_inference_roundtrip_proof",
        "inference_trace_roundtrip_proof",
        "machine_roundtrip",
        "Reversible inference roundtrip proof JSON",
        "Generic CLI roundtrip proof over the deterministic inference trace.",
    ),
    (
        "reversible_inference_roundtrip_verification",
        "inference_trace_roundtrip_verification",
        "machine_roundtrip_verdict",
        "Reversible inference roundtrip verification JSON",
        "Replay verification for the saved generic roundtrip proof.",
    ),
    (
        "reversible_inference_roundtrip_markdown",
        "inference_trace_roundtrip_markdown",
        "human_roundtrip_card",
        "Reversible inference roundtrip card",
        "Readable generic roundtrip verification card for the inference trace.",
    ),
    (
        "mlp_witness",
        "mlp_witness",
        "machine_witness",
        "MLP witness report JSON",
        "Activation/mask/error witness proof for the reversible MLP pattern.",
    ),
    (
        "mlp_witness_markdown",
        "mlp_witness_markdown",
        "human_witness_card",
        "MLP witness proof card",
        "Readable activation/mask/error witness tape, replay-cost, and proof-fingerprint card.",
    ),
)
ML_CAPABILITY_ROWS = (
    (
        "V1",
        "reversible_linear_mnist",
        "reversible linear MNIST",
        "Linear Q31 MNIST training and inference run forward, replay, and reverse.",
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
        "Saved logits, errors, predictions, correctness flags, and learning rates explain updates.",
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
        (
            "training_audit_scan",
            "training_step_selection",
            "training_step_debug",
        ),
    ),
    (
        "V3",
        "batched_tensor_iteration",
        "batched tensor dataset iteration",
        "Model evaluation and independent Q31 reference checks cover the selected sample batch.",
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
        "The MLP checker verifies hidden preactivations, ReLU masks, activations, errors, and deltas.",
        ("mlp_witness_replay",),
        (
            "report.mlp_witness",
            "bundle.mlp_vars",
            "bundle.mlp_run_output",
        ),
        ("mlp_witness",),
    ),
    (
        "V5",
        "invertible_layer_pattern",
        "invertible layer pattern",
        "Additive coupling, triangular residual, and reversible preprocessing patterns reverse with zero witness and trace payload.",
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
        "The checked profile exposes throughput, peak RSS, payload ratios, replay bytes, and reverse cost.",
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
            "report.imported_model_verification",
            "report.imported_model_inference_verification",
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
)
ML_GOAL_ROWS = (
    (
        "debug_training",
        "Debug training updates backward",
        "Inspect a selected update from the final model and verify the exact sample, witnesses, and delta ledgers.",
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
        "Tie final weights to signed training, model, sample, replay, and artifact-comparison evidence.",
        (
            "reverse_restored_initial_model",
            "reverse_checked_all_training_steps",
            "training_audit_lineage_replay",
            "audit_contract",
            "ml_roadmap_capability_map_complete",
        ),
        (
            "report.run_report",
            "report.artifact_comparison",
            "bundle.training_audit",
            "bundle.model",
            "bundle.samples",
        ),
        (
            "reverse",
            "artifact_profile",
            "ml_capability_map",
        ),
    ),
    (
        "deterministic_inference",
        "Run deterministic reversible inference traces",
        "Replay fixed-point Q31 predictions backward and check native ledgers against an independent reference.",
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
        "Expose speed, peak RSS, replay payloads, trace/witness ratios, scaling projections, and reverse-check cost.",
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
        "Verify witness-backed MLP updates, zero-witness additive coupling, triangular residual, and reversible preprocessing patterns.",
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
)


class PipelineError(RuntimeError):
    """Raised when one pipeline command fails."""


@dataclass(frozen=True)
class PipelineConfig:
    out_dir: Path
    sample_limit: int
    audit_step: int
    evaluation_row: int
    runner_bin: Optional[Path]
    reverie_bin: Optional[Path] = None
    min_train_accuracy: float = DEFAULT_MIN_TRAIN_ACCURACY
    min_eval_accuracy: float = DEFAULT_MIN_EVAL_ACCURACY
    min_audit_accuracy: float = DEFAULT_MIN_AUDIT_ACCURACY
    min_model_evaluation_accuracy: float = DEFAULT_MIN_MODEL_EVALUATION_ACCURACY
    min_reference_accuracy: float = DEFAULT_MIN_REFERENCE_ACCURACY
    max_witness_mismatches: int = DEFAULT_MAX_WITNESS_MISMATCHES
    max_trace_model_ratio: float = DEFAULT_MAX_TRACE_MODEL_RATIO
    max_witness_model_ratio: float = DEFAULT_MAX_WITNESS_MODEL_RATIO
    max_reverse_train_elapsed_ratio: float = DEFAULT_MAX_REVERSE_TRAIN_ELAPSED_RATIO
    max_replay_payload_bytes: Optional[int] = None
    audit_step_strategy: str = "explicit"
    requested_audit_step: Optional[int] = None
    evaluation_row_strategy: str = "explicit"
    requested_evaluation_row: Optional[int] = None
    dry_run: bool = False


@dataclass(frozen=True)
class PipelineStep:
    label: str
    command: list[str]
    output: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Reverie's local MNIST Q31 audit workflow and render a checked "
            "ML profile Markdown artifact."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("target/mnist-ml-audit-pipeline"),
        help="Directory for generated bundles, JSON reports, and Markdown profile.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="Number of exported samples to evaluate in the local profile.",
    )
    parser.add_argument(
        "--audit-step",
        type=int,
        default=0,
        help="Training audit step to inspect in explicit mode, or fallback requested step for automatic selection.",
    )
    parser.add_argument(
        "--audit-step-strategy",
        choices=AUDIT_STEP_STRATEGIES,
        default="explicit",
        help=(
            "How to choose the inspected training step after scanning: explicit uses "
            "--audit-step, lowest-margin uses the scan minimum margin, largest-update "
            "uses the largest weight delta, and top-suspicious uses the scan ranking."
        ),
    )
    parser.add_argument(
        "--evaluation-row",
        type=int,
        default=0,
        help="Evaluation row to inspect in explicit mode, or fallback requested row for automatic selection.",
    )
    parser.add_argument(
        "--evaluation-row-strategy",
        choices=EVALUATION_ROW_STRATEGIES,
        default="explicit",
        help=(
            "How to choose the inspected evaluation-row inference after scanning: "
            "explicit uses --evaluation-row, lowest-margin uses the scan minimum "
            "margin, and top-incorrect uses the first incorrect row from the scan."
        ),
    )
    parser.add_argument(
        "--runner-bin",
        type=Path,
        help="Use an existing reverie-mnist-linear binary instead of cargo run.",
    )
    parser.add_argument(
        "--reverie-bin",
        type=Path,
        help="Use an existing generic reverie binary for MLP witness runs instead of cargo run.",
    )
    parser.add_argument(
        "--min-train-accuracy",
        type=float,
        default=DEFAULT_MIN_TRAIN_ACCURACY,
        help="Minimum training accuracy percentage required by the pipeline summary gate.",
    )
    parser.add_argument(
        "--min-eval-accuracy",
        type=float,
        default=DEFAULT_MIN_EVAL_ACCURACY,
        help="Minimum built-in eval accuracy percentage required by the pipeline summary gate.",
    )
    parser.add_argument(
        "--min-audit-accuracy",
        type=float,
        default=DEFAULT_MIN_AUDIT_ACCURACY,
        help="Minimum scanned training-audit accuracy percentage required by the summary gate.",
    )
    parser.add_argument(
        "--min-model-evaluation-accuracy",
        type=float,
        default=DEFAULT_MIN_MODEL_EVALUATION_ACCURACY,
        help="Minimum exported model evaluation accuracy percentage required by the summary gate.",
    )
    parser.add_argument(
        "--min-reference-accuracy",
        type=float,
        default=DEFAULT_MIN_REFERENCE_ACCURACY,
        help="Minimum Python Q31 reference evaluation accuracy percentage required by the summary gate.",
    )
    parser.add_argument(
        "--max-witness-mismatches",
        type=int,
        default=DEFAULT_MAX_WITNESS_MISMATCHES,
        help="Maximum scanned training-audit witness mismatches allowed by the summary gate.",
    )
    parser.add_argument(
        "--max-trace-model-ratio",
        type=float,
        default=DEFAULT_MAX_TRACE_MODEL_RATIO,
        help="Maximum total trace/model payload ratio allowed by the summary gate.",
    )
    parser.add_argument(
        "--max-witness-model-ratio",
        type=float,
        default=DEFAULT_MAX_WITNESS_MODEL_RATIO,
        help="Maximum total witness/model payload ratio allowed by the summary gate.",
    )
    parser.add_argument(
        "--max-replay-payload-bytes",
        type=int,
        help="Optional maximum replay payload size allowed across checked replay proofs.",
    )
    parser.add_argument(
        "--max-reverse-train-elapsed-ratio",
        type=float,
        default=DEFAULT_MAX_REVERSE_TRAIN_ELAPSED_RATIO,
        help="Maximum reverse-check elapsed seconds divided by training elapsed seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running them.",
    )
    parser.add_argument(
        "--write-training-update-receipt-only",
        action="store_true",
        help="Regenerate only the selected training-update review receipt in --out-dir.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent pipeline planner self-tests and exit.",
    )
    return parser.parse_args()


def validate_config(config: PipelineConfig) -> None:
    if config.sample_limit <= 0:
        raise PipelineError("--sample-limit must be positive")
    if config.audit_step < 0:
        raise PipelineError("--audit-step must be non-negative")
    if config.requested_audit_step is not None and config.requested_audit_step < 0:
        raise PipelineError("requested audit step must be non-negative")
    if config.audit_step_strategy not in AUDIT_STEP_STRATEGIES:
        raise PipelineError("--audit-step-strategy is invalid")
    if config.evaluation_row < 0:
        raise PipelineError("--evaluation-row must be non-negative")
    if config.requested_evaluation_row is not None and config.requested_evaluation_row < 0:
        raise PipelineError("requested evaluation row must be non-negative")
    if config.evaluation_row_strategy not in EVALUATION_ROW_STRATEGIES:
        raise PipelineError("--evaluation-row-strategy is invalid")
    if config.audit_step >= config.sample_limit:
        raise PipelineError("--audit-step must be less than --sample-limit for the self-test sample export")
    if config.requested_audit_step is not None and config.requested_audit_step >= config.sample_limit:
        raise PipelineError("requested audit step must be less than --sample-limit")
    if config.evaluation_row >= config.sample_limit:
        raise PipelineError("--evaluation-row must be less than --sample-limit")
    if config.requested_evaluation_row is not None and config.requested_evaluation_row >= config.sample_limit:
        raise PipelineError("requested evaluation row must be less than --sample-limit")
    for name, value in (
        ("--min-train-accuracy", config.min_train_accuracy),
        ("--min-eval-accuracy", config.min_eval_accuracy),
        ("--min-audit-accuracy", config.min_audit_accuracy),
        ("--min-model-evaluation-accuracy", config.min_model_evaluation_accuracy),
        ("--min-reference-accuracy", config.min_reference_accuracy),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise PipelineError(f"{name} must be a finite percentage in 0..100")
    for name, value in (
        ("--max-trace-model-ratio", config.max_trace_model_ratio),
        ("--max-witness-model-ratio", config.max_witness_model_ratio),
        ("--max-reverse-train-elapsed-ratio", config.max_reverse_train_elapsed_ratio),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise PipelineError(f"{name} must be finite and non-negative")
    if config.max_witness_mismatches < 0:
        raise PipelineError("--max-witness-mismatches must be non-negative")
    if config.max_replay_payload_bytes is not None and config.max_replay_payload_bytes <= 0:
        raise PipelineError("--max-replay-payload-bytes must be positive")


def runner_prefix(config: PipelineConfig) -> list[str]:
    if config.runner_bin is not None:
        return [str(config.runner_bin)]
    return ["cargo", "run", "-p", "reverie-cli", "--bin", "reverie-mnist-linear", "--"]


def reverie_prefix(config: PipelineConfig) -> list[str]:
    if config.reverie_bin is not None:
        return [str(config.reverie_bin)]
    return ["cargo", "run", "-p", "reverie-cli", "--"]


def out_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "run_report": out_dir / "run-report.json",
        "audit_bundle": out_dir / "training-audit-bundle.json",
        "audit_verification": out_dir / "training-audit-verification.json",
        "audit_verification_markdown": out_dir / "training-audit-verification.md",
        "audit_scan": out_dir / "training-audit-scan.json",
        "audit_step": out_dir / "training-audit-step.json",
        "step_bundle": out_dir / "training-step-bundle.json",
        "step_verification": out_dir / "training-step-verification.json",
        "model_bundle": out_dir / "model-bundle.json",
        "model_export": out_dir / "model-export-report.json",
        "imported_model_source": out_dir / "imported-q31-linear-model.json",
        "imported_model_bundle": out_dir / "imported-model-bundle.json",
        "imported_model_import": out_dir / "imported-model-import-report.json",
        "imported_model_verification": out_dir / "imported-model-verification.json",
        "imported_model_inference": out_dir / "imported-model-inference-report.json",
        "imported_model_inference_bundle": out_dir / "imported-model-inference-bundle.json",
        "imported_model_inference_verification": out_dir
        / "imported-model-inference-verification.json",
        "samples": out_dir / "samples.json",
        "samples_export": out_dir / "samples-export-report.json",
        "native_inference": out_dir / "native-inference-report.json",
        "native_inference_bundle": out_dir / "native-inference-bundle.json",
        "native_inference_audit_markdown": out_dir / "native-inference-audit.md",
        "native_inference_verification": out_dir / "native-inference-verification.json",
        "native_inference_verification_markdown": out_dir
        / "native-inference-verification.md",
        "native_standalone_rev_classifier": out_dir / "mnist-standalone-classifier.rev",
        "native_standalone_rev_run": out_dir / "mnist-standalone-classifier-run.json",
        "native_standalone_rev_roundtrip": out_dir
        / "mnist-standalone-classifier-roundtrip.json",
        "native_standalone_rev_roundtrip_verification": out_dir
        / "mnist-standalone-classifier-roundtrip-verification.json",
        "model_evaluation": out_dir / "model-evaluation-report.json",
        "evaluation_bundle": out_dir / "model-evaluation-bundle.json",
        "model_evaluation_verification": out_dir / "model-evaluation-verification.json",
        "model_evaluation_scan": out_dir / "model-evaluation-scan.json",
        "model_evaluation_row": out_dir / "model-evaluation-row-report.json",
        "evaluation_row_inference_audit_markdown": out_dir
        / "model-evaluation-row-inference-audit.md",
        "row_inference_bundle": out_dir / "model-evaluation-row-inference-bundle.json",
        "row_inference_verification": out_dir / "model-evaluation-row-inference-verification.json",
        "mlp_vars": out_dir / "mlp-witness-vars.json",
        "mlp_run_output": out_dir / "mlp-witness-run.json",
        "mlp_witness": out_dir / "mlp-witness-report.json",
        "mlp_witness_markdown": out_dir / "mlp-witness.md",
        "coupling_final_vars": out_dir / "invertible-coupling-final-vars.json",
        "coupling_forward": out_dir / "invertible-coupling-forward.json",
        "coupling_reverse": out_dir / "invertible-coupling-reverse.json",
        "invertible_coupling": out_dir / "invertible-coupling-report.json",
        "invertible_coupling_markdown": out_dir / "invertible-coupling.md",
        "residual_final_vars": out_dir / "triangular-residual-final-vars.json",
        "residual_forward": out_dir / "triangular-residual-forward.json",
        "residual_reverse": out_dir / "triangular-residual-reverse.json",
        "triangular_residual": out_dir / "triangular-residual-report.json",
        "triangular_residual_markdown": out_dir / "triangular-residual.md",
        "preprocess_final_vars": out_dir / "reversible-preprocess-final-vars.json",
        "preprocess_forward": out_dir / "reversible-preprocess-forward.json",
        "preprocess_reverse": out_dir / "reversible-preprocess-reverse.json",
        "reversible_preprocess": out_dir / "reversible-preprocess-report.json",
        "reversible_preprocess_markdown": out_dir / "reversible-preprocess.md",
        "inference_trace_final_vars": out_dir / "reversible-inference-trace-final-vars.json",
        "inference_trace_forward": out_dir / "reversible-inference-trace-forward.json",
        "inference_trace_reverse": out_dir / "reversible-inference-trace-reverse.json",
        "reversible_inference_trace": out_dir / "reversible-inference-trace-report.json",
        "reversible_inference_trace_markdown": out_dir / "reversible-inference-trace.md",
        "inference_trace_roundtrip_proof": out_dir
        / "reversible-inference-roundtrip-proof.json",
        "inference_trace_roundtrip_verification": out_dir
        / "reversible-inference-roundtrip-verification.json",
        "inference_trace_roundtrip_markdown": out_dir
        / "reversible-inference-roundtrip.md",
        "q31_reference_inference": out_dir / "q31-reference-inference.json",
        "q31_reference_inference_markdown": out_dir / "q31-reference-inference.md",
        "q31_reference_evaluation": out_dir / "q31-reference-evaluation.json",
        "artifact_comparison": out_dir / "artifact-comparison.json",
        "training_step_debug_markdown": out_dir / "training-step-debug.md",
        "profile_markdown": out_dir / "mnist-ml-audit-profile.md",
        "summary": out_dir / "pipeline-summary.json",
        "manifest": out_dir / "pipeline-manifest.json",
        "model_capsule": out_dir / "model-capsule.json",
        "model_capsule_profile": out_dir / "model-capsule-profile.md",
        "model_capsule_verification": out_dir / "model-capsule-verification.json",
        "model_capsule_verification_markdown": out_dir / "model-capsule-verification.md",
        "handoff_index": out_dir / "ml-audit-handoff.json",
        "handoff_markdown": out_dir / "ml-audit-handoff.md",
        "inference_action_review_receipt": out_dir
        / capsule_verify.DEFAULT_ACTION_REVIEW_RECEIPT_NAME,
        "inference_action_review_receipt_markdown": out_dir
        / capsule_verify.DEFAULT_ACTION_REVIEW_RECEIPT_MARKDOWN_NAME,
        "training_update_review_receipt": out_dir / DEFAULT_TRAINING_UPDATE_REVIEW_RECEIPT_NAME,
        "training_update_review_receipt_markdown": (
            out_dir / DEFAULT_TRAINING_UPDATE_REVIEW_RECEIPT_MARKDOWN_NAME
        ),
    }


def profile_reports(paths: dict[str, Path]) -> list[Path]:
    return [paths[key] for key in PROFILE_REPORT_KEYS]


def bundle_paths(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "training_audit": paths["audit_bundle"],
        "training_step": paths["step_bundle"],
        "model": paths["model_bundle"],
        "imported_model_source": paths["imported_model_source"],
        "imported_model": paths["imported_model_bundle"],
        "imported_model_inference": paths["imported_model_inference_bundle"],
        "samples": paths["samples"],
        "native_inference": paths["native_inference_bundle"],
        "model_evaluation": paths["evaluation_bundle"],
        "evaluation_row_inference": paths["row_inference_bundle"],
        "mlp_vars": paths["mlp_vars"],
        "mlp_run_output": paths["mlp_run_output"],
        "coupling_final_vars": paths["coupling_final_vars"],
        "coupling_forward": paths["coupling_forward"],
        "coupling_reverse": paths["coupling_reverse"],
        "residual_final_vars": paths["residual_final_vars"],
        "residual_forward": paths["residual_forward"],
        "residual_reverse": paths["residual_reverse"],
        "preprocess_final_vars": paths["preprocess_final_vars"],
        "preprocess_forward": paths["preprocess_forward"],
        "preprocess_reverse": paths["preprocess_reverse"],
        "inference_trace_final_vars": paths["inference_trace_final_vars"],
        "inference_trace_forward": paths["inference_trace_forward"],
        "inference_trace_reverse": paths["inference_trace_reverse"],
        "inference_trace_roundtrip_proof": paths["inference_trace_roundtrip_proof"],
        "inference_trace_roundtrip_verification": paths[
            "inference_trace_roundtrip_verification"
        ],
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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def evidence_record(path: Path, role: str) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise PipelineError(f"failed to stat evidence file {path}: {error}") from error
    return {
        "path": str(path),
        "role": role,
        "bytes": size,
        "sha256": sha256_file(path),
    }


def evidence_index(paths: dict[str, Path], *, include_summary: bool) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {
        "profile_markdown": evidence_record(paths["profile_markdown"], "markdown"),
    }
    for key in PROFILE_REPORT_KEYS:
        files[f"report.{key}"] = evidence_record(paths[key], "report")
    for key, path in bundle_paths(paths).items():
        files[f"bundle.{key}"] = evidence_record(path, "bundle")
    for key in (
        "native_standalone_rev_classifier",
        "native_standalone_rev_run",
        "native_standalone_rev_roundtrip",
        "native_standalone_rev_roundtrip_verification",
    ):
        files[f"standalone.{key}"] = evidence_record(paths[key], "standalone")
    if include_summary:
        files["summary"] = evidence_record(paths["summary"], "summary")
    return {
        "algorithm": "sha256",
        "files": files,
    }


def handoff_artifact_record(
    paths: dict[str, Path],
    path_key: str,
    role: str,
    label: str,
    description: str,
) -> dict[str, Any]:
    record = evidence_record(paths[path_key], role)
    record["label"] = label
    record["description"] = description
    return record


def handoff_artifacts(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        artifact_id: handoff_artifact_record(
            paths,
            path_key,
            role,
            label,
            description,
        )
        for artifact_id, path_key, role, label, description in HANDOFF_ARTIFACT_ROWS
    }


def build_steps(config: PipelineConfig) -> list[PipelineStep]:
    paths = out_paths(config.out_dir)
    runner = runner_prefix(config)
    reverie = reverie_prefix(config)
    python = sys.executable
    profile_inputs = [str(path) for path in profile_reports(paths)]
    return [
        PipelineStep(
            "train self-test and write audit bundle",
            runner
            + [
                "--self-test",
                "--reverse-check",
                "--audit-output",
                str(paths["audit_bundle"]),
                "--json",
            ],
            paths["run_report"],
        ),
        PipelineStep(
            "verify training audit lineage",
            runner
            + [
                "--verify-audit",
                str(paths["audit_bundle"]),
                "--markdown-output",
                str(paths["audit_verification_markdown"]),
                "--json",
            ],
            paths["audit_verification"],
        ),
        PipelineStep(
            "scan training audit",
            runner
            + [
                "--scan-audit",
                str(paths["audit_bundle"]),
                "--audit-limit",
                str(config.sample_limit),
                "--json",
            ],
            paths["audit_scan"],
        ),
        PipelineStep(
            "inspect one training step",
            runner
            + [
                "--inspect-audit",
                str(paths["audit_bundle"]),
                "--audit-step",
                str(config.audit_step),
                "--audit-step-strategy",
                config.audit_step_strategy,
                "--step-output",
                str(paths["step_bundle"]),
                "--json",
            ],
            paths["audit_step"],
        ),
        PipelineStep(
            "verify one-step replay bundle",
            runner
            + [
                "--verify-step",
                str(paths["step_bundle"]),
                "--markdown-output",
                str(paths["training_step_debug_markdown"]),
                "--json",
            ],
            paths["step_verification"],
        ),
        PipelineStep(
            "export signed model bundle",
            runner
            + [
                "--export-model",
                str(paths["audit_bundle"]),
                "--model-output",
                str(paths["model_bundle"]),
                "--json",
            ],
            paths["model_export"],
        ),
        PipelineStep(
            "write imported model source JSON",
            [
                python,
                "scripts/export_q31_linear_model.py",
                "--input",
                str(paths["model_bundle"]),
                "--output",
                str(paths["imported_model_source"]),
                "--input-layout",
                "pixel-class",
                "--input-scale",
                "q31",
            ],
        ),
        PipelineStep(
            "import signed external model bundle",
            runner
            + [
                "--import-model-json",
                str(paths["imported_model_source"]),
                "--model-output",
                str(paths["imported_model_bundle"]),
                "--json",
            ],
            paths["imported_model_import"],
        ),
        PipelineStep(
            "verify imported external model bundle",
            runner + ["--verify-model", str(paths["imported_model_bundle"]), "--json"],
            paths["imported_model_verification"],
        ),
        PipelineStep(
            "inspect imported model inference",
            runner
            + [
                "--inspect-model-inference",
                str(paths["imported_model_bundle"]),
                "--sample-audit",
                str(paths["audit_bundle"]),
                "--audit-step",
                str(config.audit_step),
                "--inference-output",
                str(paths["imported_model_inference_bundle"]),
                "--json",
            ],
            paths["imported_model_inference"],
        ),
        PipelineStep(
            "verify imported inference bundle",
            runner
            + [
                "--verify-inference",
                str(paths["imported_model_inference_bundle"]),
                "--json",
            ],
            paths["imported_model_inference_verification"],
        ),
        PipelineStep(
            "export signed sample set",
            runner
            + [
                "--export-samples",
                str(paths["audit_bundle"]),
                "--samples-output",
                str(paths["samples"]),
                "--samples-limit",
                str(config.sample_limit),
                "--json",
            ],
            paths["samples_export"],
        ),
        PipelineStep(
            "inspect native model inference",
            runner
            + [
                "--inspect-model-inference",
                str(paths["model_bundle"]),
                "--sample-audit",
                str(paths["audit_bundle"]),
                "--audit-step",
                str(config.audit_step),
                "--inference-output",
                str(paths["native_inference_bundle"]),
                "--standalone-rev-output",
                str(paths["native_standalone_rev_classifier"]),
                "--markdown-output",
                str(paths["native_inference_audit_markdown"]),
                "--json",
            ],
            paths["native_inference"],
        ),
        PipelineStep(
            "verify native inference bundle",
            runner
            + [
                "--verify-inference",
                str(paths["native_inference_bundle"]),
                "--markdown-output",
                str(paths["native_inference_verification_markdown"]),
                "--json",
            ],
            paths["native_inference_verification"],
        ),
        PipelineStep(
            "run standalone native .rev classifier",
            reverie
            + [
                "run",
                str(paths["native_standalone_rev_classifier"]),
                "--json",
            ],
            paths["native_standalone_rev_run"],
        ),
        PipelineStep(
            "roundtrip standalone native .rev classifier",
            reverie
            + [
                "roundtrip",
                str(paths["native_standalone_rev_classifier"]),
                "--json",
            ],
            paths["native_standalone_rev_roundtrip"],
        ),
        PipelineStep(
            "verify standalone native .rev classifier roundtrip",
            reverie
            + [
                "verify-roundtrip",
                str(paths["native_standalone_rev_roundtrip"]),
                "--json",
            ],
            paths["native_standalone_rev_roundtrip_verification"],
        ),
        PipelineStep(
            "evaluate native model bundle",
            runner
            + [
                "--evaluate-model",
                str(paths["model_bundle"]),
                "--samples-json",
                str(paths["samples"]),
                "--evaluation-output",
                str(paths["evaluation_bundle"]),
                "--json",
            ],
            paths["model_evaluation"],
        ),
        PipelineStep(
            "verify model evaluation bundle",
            runner + ["--verify-evaluation", str(paths["evaluation_bundle"]), "--json"],
            paths["model_evaluation_verification"],
        ),
        PipelineStep(
            "scan model evaluation bundle",
            runner
            + [
                "--scan-evaluation",
                str(paths["evaluation_bundle"]),
                "--evaluation-limit",
                str(config.sample_limit),
                "--json",
            ],
            paths["model_evaluation_scan"],
        ),
        PipelineStep(
            "inspect one evaluation row",
            runner
            + [
                "--inspect-evaluation",
                str(paths["evaluation_bundle"]),
                "--evaluation-row",
                str(config.evaluation_row),
                "--markdown-output",
                str(paths["evaluation_row_inference_audit_markdown"]),
                "--inference-output",
                str(paths["row_inference_bundle"]),
                "--json",
            ],
            paths["model_evaluation_row"],
        ),
        PipelineStep(
            "verify evaluation-row inference bundle",
            runner + ["--verify-inference", str(paths["row_inference_bundle"]), "--json"],
            paths["row_inference_verification"],
        ),
        PipelineStep(
            "run MLP witness trace",
            reverie
            + [
                "run",
                "examples/mnist_mlp_witness.rev",
                "--vars-json",
                str(paths["mlp_vars"]),
                "--json",
            ],
            paths["mlp_run_output"],
        ),
        PipelineStep(
            "reference-check MLP witness trace",
            [
                python,
                "scripts/check_q31_mlp_witness.py",
                "--vars-json",
                str(paths["mlp_vars"]),
                "--run-output-json",
                str(paths["mlp_run_output"]),
                "--expect-predictions",
                "[0,1]",
                "--expect-correct",
                "[true,true]",
                "--markdown-output",
                str(paths["mlp_witness_markdown"]),
                "--json",
            ],
            paths["mlp_witness"],
        ),
        PipelineStep(
            "run invertible coupling block",
            reverie + ["run", "examples/invertible_coupling.rev", "--json"],
            paths["coupling_forward"],
        ),
        PipelineStep(
            "reverse invertible coupling block",
            reverie
            + [
                "reverse",
                "examples/invertible_coupling.rev",
                "--vars-json",
                str(paths["coupling_final_vars"]),
                "--json",
            ],
            paths["coupling_reverse"],
        ),
        PipelineStep(
            "reference-check invertible coupling block",
            [
                python,
                "scripts/check_invertible_coupling.py",
                "--forward-output-json",
                str(paths["coupling_forward"]),
                "--reverse-output-json",
                str(paths["coupling_reverse"]),
                "--markdown-output",
                str(paths["invertible_coupling_markdown"]),
                "--json",
            ],
            paths["invertible_coupling"],
        ),
        PipelineStep(
            "run triangular residual block",
            reverie + ["run", "examples/triangular_residual.rev", "--json"],
            paths["residual_forward"],
        ),
        PipelineStep(
            "reverse triangular residual block",
            reverie
            + [
                "reverse",
                "examples/triangular_residual.rev",
                "--vars-json",
                str(paths["residual_final_vars"]),
                "--json",
            ],
            paths["residual_reverse"],
        ),
        PipelineStep(
            "reference-check triangular residual block",
            [
                python,
                "scripts/check_triangular_residual.py",
                "--forward-output-json",
                str(paths["residual_forward"]),
                "--reverse-output-json",
                str(paths["residual_reverse"]),
                "--markdown-output",
                str(paths["triangular_residual_markdown"]),
                "--json",
            ],
            paths["triangular_residual"],
        ),
        PipelineStep(
            "run reversible preprocessing block",
            reverie + ["run", "examples/reversible_preprocess.rev", "--json"],
            paths["preprocess_forward"],
        ),
        PipelineStep(
            "reverse reversible preprocessing block",
            reverie
            + [
                "reverse",
                "examples/reversible_preprocess.rev",
                "--vars-json",
                str(paths["preprocess_final_vars"]),
                "--json",
            ],
            paths["preprocess_reverse"],
        ),
        PipelineStep(
            "reference-check reversible preprocessing block",
            [
                python,
                "scripts/check_reversible_preprocess.py",
                "--forward-output-json",
                str(paths["preprocess_forward"]),
                "--reverse-output-json",
                str(paths["preprocess_reverse"]),
                "--markdown-output",
                str(paths["reversible_preprocess_markdown"]),
                "--json",
            ],
            paths["reversible_preprocess"],
        ),
        PipelineStep(
            "run reversible inference trace",
            reverie + ["run", "examples/reversible_inference_trace.rev", "--json"],
            paths["inference_trace_forward"],
        ),
        PipelineStep(
            "reverse reversible inference trace",
            reverie
            + [
                "reverse",
                "examples/reversible_inference_trace.rev",
                "--vars-json",
                str(paths["inference_trace_final_vars"]),
                "--json",
            ],
            paths["inference_trace_reverse"],
        ),
        PipelineStep(
            "reference-check reversible inference trace",
            [
                python,
                "scripts/check_reversible_inference_trace.py",
                "--forward-output-json",
                str(paths["inference_trace_forward"]),
                "--reverse-output-json",
                str(paths["inference_trace_reverse"]),
                "--markdown-output",
                str(paths["reversible_inference_trace_markdown"]),
                "--json",
            ],
            paths["reversible_inference_trace"],
        ),
        PipelineStep(
            "write reversible inference roundtrip proof",
            reverie
            + [
                "roundtrip",
                "examples/reversible_inference_trace.rev",
                "--json",
            ],
            paths["inference_trace_roundtrip_proof"],
        ),
        PipelineStep(
            "verify reversible inference roundtrip proof",
            reverie
            + [
                "verify-roundtrip",
                str(paths["inference_trace_roundtrip_proof"]),
                "--json",
                "--markdown-output",
                str(paths["inference_trace_roundtrip_markdown"]),
            ],
            paths["inference_trace_roundtrip_verification"],
        ),
        PipelineStep(
            "reference-check one Q31 inference",
            [
                python,
                "scripts/check_q31_inference.py",
                "--model",
                str(paths["model_bundle"]),
                "--sample",
                str(paths["samples"]),
                "--sample-index",
                str(config.audit_step),
                "--markdown-output",
                str(paths["q31_reference_inference_markdown"]),
                "--json",
            ],
            paths["q31_reference_inference"],
        ),
        PipelineStep(
            "reference-check all Q31 samples",
            [
                python,
                "scripts/check_q31_inference.py",
                "--model",
                str(paths["model_bundle"]),
                "--sample",
                str(paths["samples"]),
                "--all-samples",
                "--json",
            ],
            paths["q31_reference_evaluation"],
        ),
        PipelineStep(
            "compare signed replay artifacts",
            runner
            + [
                "--compare-artifacts",
                str(paths["audit_bundle"]),
                str(paths["model_bundle"]),
                str(paths["samples"]),
                str(paths["step_bundle"]),
                str(paths["native_inference_bundle"]),
                str(paths["evaluation_bundle"]),
                "--json",
            ],
            paths["artifact_comparison"],
        ),
        PipelineStep(
            "validate checked ML profile reports",
            [
                python,
                "scripts/check_mnist_ml_profile.py",
                *profile_inputs,
                "--require-reverse-check",
                "--require-audit-contract",
            ],
        ),
        PipelineStep(
            "render checked ML profile Markdown",
            [
                python,
                "scripts/summarize_mnist_ml_profile.py",
                *profile_inputs,
                "--require-reverse-check",
                "--require-audit-contract",
                "--output",
                str(paths["profile_markdown"]),
            ],
        ),
    ]


def run_step(step: PipelineStep) -> None:
    if step.output is not None:
        step.output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        step.command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise PipelineError(
            "{} failed with exit code {}\ncommand: {}\nstdout:\n{}\nstderr:\n{}".format(
                step.label,
                result.returncode,
                " ".join(step.command),
                result.stdout.strip(),
                result.stderr.strip(),
            )
        )
    if step.output is not None:
        step.output.write_text(result.stdout, encoding="utf-8")
        try:
            json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise PipelineError(f"{step.label} did not emit valid JSON: {error}") from error
    elif result.stdout.strip():
        print(result.stdout, end="")
    if result.stderr.strip():
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")


def write_mlp_witness_seed(config: PipelineConfig) -> None:
    paths = out_paths(config.out_dir)
    paths["mlp_vars"].parent.mkdir(parents=True, exist_ok=True)
    paths["mlp_vars"].write_text(
        json.dumps(mlp_check.self_test_seed(), indent=2) + "\n",
        encoding="utf-8",
    )


def write_invertible_coupling_seed(config: PipelineConfig) -> None:
    paths = out_paths(config.out_dir)
    paths["coupling_final_vars"].parent.mkdir(parents=True, exist_ok=True)
    paths["coupling_final_vars"].write_text(
        json.dumps(coupling_check.final_vars(), indent=2) + "\n",
        encoding="utf-8",
    )


def write_triangular_residual_seed(config: PipelineConfig) -> None:
    paths = out_paths(config.out_dir)
    paths["residual_final_vars"].parent.mkdir(parents=True, exist_ok=True)
    paths["residual_final_vars"].write_text(
        json.dumps(residual_check.final_vars(), indent=2) + "\n",
        encoding="utf-8",
    )


def write_reversible_preprocess_seed(config: PipelineConfig) -> None:
    paths = out_paths(config.out_dir)
    paths["preprocess_final_vars"].parent.mkdir(parents=True, exist_ok=True)
    paths["preprocess_final_vars"].write_text(
        json.dumps(preprocess_check.final_vars(), indent=2) + "\n",
        encoding="utf-8",
    )


def write_reversible_inference_trace_seed(config: PipelineConfig) -> None:
    paths = out_paths(config.out_dir)
    paths["inference_trace_final_vars"].parent.mkdir(parents=True, exist_ok=True)
    paths["inference_trace_final_vars"].write_text(
        json.dumps(inference_trace_check.final_vars(), indent=2) + "\n",
        encoding="utf-8",
    )


def load_checked_reports(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    reports = {}
    for key in PROFILE_REPORT_KEYS:
        path = paths[key]
        try:
            report = profile_check.load_json(path)
            profile_check.validate_report(
                report,
                require_reverse_check=True,
                require_peak_rss=False,
                require_ml_profile=False,
                require_audit_contract=True,
            )
        except ValueError as error:
            raise PipelineError(f"{path} failed summary validation: {error}") from error
        if not isinstance(report, dict):
            raise PipelineError(f"{path} did not contain a JSON object")
        reports[key] = report
    return reports


def load_pipeline_json_artifact(path: Path, label: str) -> dict[str, Any]:
    try:
        value = profile_check.load_json(path)
    except ValueError as error:
        raise PipelineError(f"{label} {path} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise PipelineError(f"{label} {path} did not contain a JSON object")
    return value


def gate_policy(config: PipelineConfig) -> dict[str, Any]:
    return {
        "min_train_accuracy_percent": config.min_train_accuracy,
        "min_eval_accuracy_percent": config.min_eval_accuracy,
        "min_audit_accuracy_percent": config.min_audit_accuracy,
        "min_model_evaluation_accuracy_percent": config.min_model_evaluation_accuracy,
        "min_reference_accuracy_percent": config.min_reference_accuracy,
        "max_witness_mismatches": config.max_witness_mismatches,
        "max_trace_to_model_payload_ratio": config.max_trace_model_ratio,
        "max_witness_to_model_payload_ratio": config.max_witness_model_ratio,
        "max_reverse_train_elapsed_ratio": config.max_reverse_train_elapsed_ratio,
        "max_replay_payload_bytes": config.max_replay_payload_bytes,
        "audit_step_strategy": config.audit_step_strategy,
        "requested_audit_step": (
            config.audit_step if config.requested_audit_step is None else config.requested_audit_step
        ),
        "evaluation_row_strategy": config.evaluation_row_strategy,
        "requested_evaluation_row": (
            config.evaluation_row if config.requested_evaluation_row is None else config.requested_evaluation_row
        ),
    }


def add_gate(
    checks: list[dict[str, Any]],
    metric: str,
    actual: Any,
    requirement: str,
    passed: bool,
) -> None:
    checks.append(
        {
            "metric": metric,
            "actual": actual,
            "requirement": requirement,
            "passed": bool(passed),
        }
    )


def add_min_percent_gate(
    checks: list[dict[str, Any]],
    metric: str,
    actual: float,
    minimum: float,
) -> None:
    add_gate(
        checks,
        metric,
        round(actual, 6),
        f">= {minimum:.2f}%",
        actual >= minimum,
    )


def add_max_number_gate(
    checks: list[dict[str, Any]],
    metric: str,
    actual: float,
    maximum: float,
) -> None:
    add_gate(
        checks,
        metric,
        round(actual, 6),
        f"<= {maximum:.6g}",
        actual <= maximum,
    )


def gates_passed(checks: list[dict[str, Any]], metrics: tuple[str, ...]) -> bool:
    by_metric = {check["metric"]: bool(check["passed"]) for check in checks}
    return all(by_metric.get(metric) is True for metric in metrics)


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0.0:
        return None
    return numerator / denominator


def finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0.0


def finite_positive(value: Any) -> bool:
    return finite_nonnegative(value) and float(value) > 0.0


def scorecard_contracts_passed(scorecard: dict[str, Any]) -> bool:
    contracts = scorecard.get("contracts", {})
    return isinstance(contracts, dict) and all(value is True for value in contracts.values())


def pipeline_ml_capability_map(checks: list[dict[str, Any]]) -> dict[str, Any]:
    gate_results = {
        check["metric"]: bool(check["passed"])
        for check in checks
        if isinstance(check.get("metric"), str)
    }
    rows = []
    for phase, capability_id, goal, description, gate_metrics, evidence, metrics in ML_CAPABILITY_ROWS:
        blocking_gates = [
            metric
            for metric in gate_metrics
            if gate_results.get(metric) is not True
        ]
        passed = not blocking_gates
        rows.append(
            {
                "phase": phase,
                "id": capability_id,
                "goal": goal,
                "description": description,
                "status": "passed" if passed else "failed",
                "passed": passed,
                "gate_metrics": list(gate_metrics),
                "blocking_gates": blocking_gates,
                "evidence": list(evidence),
                "metrics": list(metrics),
            }
        )
    passed_count = sum(1 for row in rows if row["passed"])
    failed_ids = [
        row["id"]
        for row in rows
        if not row["passed"]
    ]
    return {
        "kind": ML_CAPABILITY_KIND,
        "north_star": "reversible_inspectable_deterministic_ml_kernels",
        "passed": passed_count == len(rows),
        "summary": {
            "total": len(rows),
            "passed": passed_count,
            "failed": len(rows) - passed_count,
            "failed_capabilities": failed_ids,
        },
        "capabilities": rows,
    }


def pipeline_ml_goal_readiness(checks: list[dict[str, Any]]) -> dict[str, Any]:
    gate_results = {
        check["metric"]: bool(check["passed"])
        for check in checks
        if isinstance(check.get("metric"), str)
    }
    rows = []
    for readiness_id, goal, description, gate_metrics, evidence, metrics in ML_GOAL_ROWS:
        blocking_gates = [
            metric
            for metric in gate_metrics
            if gate_results.get(metric) is not True
        ]
        passed = not blocking_gates
        rows.append(
            {
                "id": readiness_id,
                "goal": goal,
                "description": description,
                "status": "passed" if passed else "failed",
                "passed": passed,
                "gate_metrics": list(gate_metrics),
                "blocking_gates": blocking_gates,
                "evidence": list(evidence),
                "metrics": list(metrics),
            }
        )
    passed_count = sum(1 for row in rows if row["passed"])
    failed_ids = [
        row["id"]
        for row in rows
        if not row["passed"]
    ]
    return {
        "kind": ML_GOAL_READINESS_KIND,
        "north_star": "best_small_language_for_reversible_inspectable_deterministic_ml_kernels",
        "non_goal": "general_purpose_pytorch_tensorflow_training_replacement",
        "passed": passed_count == len(rows),
        "summary": {
            "total": len(rows),
            "passed": passed_count,
            "failed": len(rows) - passed_count,
            "failed_goals": failed_ids,
        },
        "goals": rows,
    }


def frontier_ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def recompute_frontier_row(
    *,
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
    total_recompute_steps = forward_recompute_steps + inverse_recompute_steps
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
        "total_recompute_steps": total_recompute_steps,
        "balanced_recompute": forward_recompute_steps == inverse_recompute_steps
        and forward_recompute_steps > 0,
        "zero_witness": witness_payload_bytes == 0 and trace_payload_bytes == 0,
        "witness_backed": witness_payload_bytes > 0,
        "bytes_per_inverse_step": frontier_ratio(replay_payload_bytes, inverse_recompute_steps),
        "witness_to_model_payload_ratio": frontier_ratio(witness_payload_bytes, model_payload_bytes),
        "trace_to_model_payload_ratio": frontier_ratio(trace_payload_bytes, model_payload_bytes),
    }


def recompute_frontier_metrics(
    *,
    run_proof: dict[str, Any],
    step_proof: dict[str, Any],
    imported_inference_memory: dict[str, Any],
    native_inference_memory: dict[str, Any],
    model_evaluation_proof: dict[str, Any],
    row_inference_memory: dict[str, Any],
    mlp_proof: dict[str, Any],
    coupling_proof: dict[str, Any],
    residual_proof: dict[str, Any],
    preprocess_proof: dict[str, Any],
    inference_trace_proof: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        recompute_frontier_row(
            row_id="training_trace",
            label="Training trace replay",
            scope="all training updates",
            kept_state="final model plus per-sample logits/error/prediction witnesses",
            reversible_transition="final model + witness trace -> initial model",
            replay_payload_bytes=run_proof["full_replay_payload_bytes"],
            model_payload_bytes=run_proof["model_payload_bytes"],
            sample_payload_bytes=run_proof["sample_payload_bytes"],
            witness_payload_bytes=run_proof["witness_payload_bytes"],
            trace_payload_bytes=run_proof["trace_replay_payload_bytes"],
            derived_update_payload_bytes=0,
            recomputed_update_payload_bytes=0,
            state_payload_bytes=0,
            forward_recompute_steps=run_proof["forward_recompute_steps"],
            inverse_recompute_steps=run_proof["inverse_recompute_steps"],
        ),
        recompute_frontier_row(
            row_id="training_step_debug",
            label="Selected training-step replay",
            scope="one model update",
            kept_state="before/after model window, sample, witnesses, and derived update summary",
            reversible_transition="after model + sample + witnesses -> before model",
            replay_payload_bytes=step_proof["replay_payload_bytes"],
            model_payload_bytes=step_proof["model_payload_bytes"],
            sample_payload_bytes=step_proof["sample_payload_bytes"],
            witness_payload_bytes=step_proof["witness_payload_bytes"],
            trace_payload_bytes=step_proof["trace_payload_bytes"],
            derived_update_payload_bytes=step_proof["derived_update_payload_bytes"],
            recomputed_update_payload_bytes=step_proof["derived_update_payload_bytes"],
            state_payload_bytes=0,
            forward_recompute_steps=step_proof["forward_recompute_steps"],
            inverse_recompute_steps=step_proof["inverse_recompute_steps"],
        ),
        recompute_frontier_row(
            row_id="imported_model_inference",
            label="Imported model inference replay",
            scope="one external-import model sample",
            kept_state="imported model, image, logits, prediction, and correctness witnesses",
            reversible_transition="imported inference result -> zeroed inference state",
            replay_payload_bytes=imported_inference_memory["replay_payload_bytes"],
            model_payload_bytes=imported_inference_memory["model_payload_bytes"],
            sample_payload_bytes=imported_inference_memory["sample_payload_bytes"],
            witness_payload_bytes=imported_inference_memory["witness_payload_bytes"],
            trace_payload_bytes=imported_inference_memory["trace_payload_bytes"],
            derived_update_payload_bytes=0,
            recomputed_update_payload_bytes=0,
            state_payload_bytes=0,
            forward_recompute_steps=imported_inference_memory["forward_recompute_steps"],
            inverse_recompute_steps=imported_inference_memory["inverse_recompute_steps"],
        ),
        recompute_frontier_row(
            row_id="native_inference",
            label="Single native inference replay",
            scope="one signed sample",
            kept_state="model, image, logits, prediction, and correctness witnesses",
            reversible_transition="inference result -> zeroed inference state",
            replay_payload_bytes=native_inference_memory["replay_payload_bytes"],
            model_payload_bytes=native_inference_memory["model_payload_bytes"],
            sample_payload_bytes=native_inference_memory["sample_payload_bytes"],
            witness_payload_bytes=native_inference_memory["witness_payload_bytes"],
            trace_payload_bytes=native_inference_memory["trace_payload_bytes"],
            derived_update_payload_bytes=0,
            recomputed_update_payload_bytes=0,
            state_payload_bytes=0,
            forward_recompute_steps=native_inference_memory["forward_recompute_steps"],
            inverse_recompute_steps=native_inference_memory["inverse_recompute_steps"],
        ),
        recompute_frontier_row(
            row_id="model_evaluation_batch",
            label="Batch model-evaluation replay",
            scope="signed sample batch",
            kept_state="model, sample batch, logits/prediction/correctness witnesses",
            reversible_transition="batch evaluation rows -> zeroed evaluation state",
            replay_payload_bytes=model_evaluation_proof["replay_payload_bytes"],
            model_payload_bytes=model_evaluation_proof["model_payload_bytes"],
            sample_payload_bytes=model_evaluation_proof["sample_payload_bytes"],
            witness_payload_bytes=model_evaluation_proof["witness_payload_bytes"],
            trace_payload_bytes=model_evaluation_proof["trace_payload_bytes"],
            derived_update_payload_bytes=0,
            recomputed_update_payload_bytes=0,
            state_payload_bytes=0,
            forward_recompute_steps=model_evaluation_proof["forward_recompute_steps"],
            inverse_recompute_steps=model_evaluation_proof["inverse_recompute_steps"],
        ),
        recompute_frontier_row(
            row_id="evaluation_row_inference",
            label="Evaluation-row inference replay",
            scope="one scan-selected evaluation row",
            kept_state="signed model evaluation plus row inference witnesses",
            reversible_transition="row prediction explanation -> zeroed inference state",
            replay_payload_bytes=row_inference_memory["replay_payload_bytes"],
            model_payload_bytes=row_inference_memory["model_payload_bytes"],
            sample_payload_bytes=row_inference_memory["sample_payload_bytes"],
            witness_payload_bytes=row_inference_memory["witness_payload_bytes"],
            trace_payload_bytes=row_inference_memory["trace_payload_bytes"],
            derived_update_payload_bytes=0,
            recomputed_update_payload_bytes=0,
            state_payload_bytes=0,
            forward_recompute_steps=row_inference_memory["forward_recompute_steps"],
            inverse_recompute_steps=row_inference_memory["inverse_recompute_steps"],
        ),
        recompute_frontier_row(
            row_id="mlp_witness_trace",
            label="MLP witness trace replay",
            scope="two-sample MLP update",
            kept_state="model plus hidden preactivation/ReLU/error/backprop witness tapes",
            reversible_transition="final MLP weights + witness tapes -> prior weights",
            replay_payload_bytes=mlp_proof["replay_payload_bytes"],
            model_payload_bytes=mlp_proof["model_payload_bytes"],
            sample_payload_bytes=mlp_proof["sample_payload_bytes"],
            witness_payload_bytes=mlp_proof["witness_payload_bytes"],
            trace_payload_bytes=mlp_proof["trace_payload_bytes"],
            derived_update_payload_bytes=mlp_proof["stored_derived_update_payload_bytes"],
            recomputed_update_payload_bytes=mlp_proof["recomputed_update_payload_bytes"],
            state_payload_bytes=0,
            forward_recompute_steps=mlp_proof["forward_recompute_steps"],
            inverse_recompute_steps=mlp_proof["inverse_recompute_steps"],
        ),
        recompute_frontier_row(
            row_id="invertible_coupling",
            label="Invertible coupling replay",
            scope="one additive coupling layer",
            kept_state="activation halves only",
            reversible_transition="coupled activations -> original activations",
            replay_payload_bytes=coupling_proof["replay_payload_bytes"],
            model_payload_bytes=coupling_proof["model_payload_bytes"],
            sample_payload_bytes=0,
            witness_payload_bytes=coupling_proof["witness_payload_bytes"],
            trace_payload_bytes=coupling_proof["trace_payload_bytes"],
            derived_update_payload_bytes=0,
            recomputed_update_payload_bytes=0,
            state_payload_bytes=coupling_proof["state_payload_bytes"],
            forward_recompute_steps=coupling_proof["forward_recompute_steps"],
            inverse_recompute_steps=coupling_proof["inverse_recompute_steps"],
        ),
        recompute_frontier_row(
            row_id="triangular_residual",
            label="Triangular residual replay",
            scope="one triangular residual layer",
            kept_state="activation vector and residual parameters",
            reversible_transition="residual activations -> original activations",
            replay_payload_bytes=residual_proof["replay_payload_bytes"],
            model_payload_bytes=residual_proof["parameter_payload_bytes"],
            sample_payload_bytes=0,
            witness_payload_bytes=residual_proof["witness_payload_bytes"],
            trace_payload_bytes=residual_proof["trace_payload_bytes"],
            derived_update_payload_bytes=0,
            recomputed_update_payload_bytes=0,
            state_payload_bytes=residual_proof["state_payload_bytes"],
            forward_recompute_steps=residual_proof["forward_recompute_steps"],
            inverse_recompute_steps=residual_proof["inverse_recompute_steps"],
        ),
        recompute_frontier_row(
            row_id="reversible_preprocess",
            label="Reversible preprocessing replay",
            scope="one centered/permuted feature block",
            kept_state="raw sample, mean vector, and feature buffer",
            reversible_transition="preprocessed features -> zeroed feature buffer",
            replay_payload_bytes=preprocess_proof["replay_payload_bytes"],
            model_payload_bytes=preprocess_proof["mean_payload_bytes"],
            sample_payload_bytes=preprocess_proof["raw_payload_bytes"],
            witness_payload_bytes=preprocess_proof["witness_payload_bytes"],
            trace_payload_bytes=preprocess_proof["trace_payload_bytes"],
            derived_update_payload_bytes=0,
            recomputed_update_payload_bytes=0,
            state_payload_bytes=preprocess_proof["feature_payload_bytes"],
            forward_recompute_steps=preprocess_proof["forward_recompute_steps"],
            inverse_recompute_steps=preprocess_proof["inverse_recompute_steps"],
        ),
        recompute_frontier_row(
            row_id="reversible_inference_trace",
            label="Reversible inference trace replay",
            scope="one preprocessed Q31 classifier trace",
            kept_state="raw sample, mean vector, model, features, logits, prediction, and correctness",
            reversible_transition="prediction trace -> zeroed feature/logit/prediction state",
            replay_payload_bytes=inference_trace_proof["replay_payload_bytes"],
            model_payload_bytes=inference_trace_proof["model_payload_bytes"],
            sample_payload_bytes=0,
            witness_payload_bytes=inference_trace_proof["witness_payload_bytes"],
            trace_payload_bytes=inference_trace_proof["trace_payload_bytes"],
            derived_update_payload_bytes=0,
            recomputed_update_payload_bytes=0,
            state_payload_bytes=inference_trace_proof["state_payload_bytes"],
            forward_recompute_steps=inference_trace_proof["forward_recompute_steps"],
            inverse_recompute_steps=inference_trace_proof["inverse_recompute_steps"],
        ),
    ]
    replay_values = [row["replay_payload_bytes"] for row in rows]
    largest = max(rows, key=lambda row: row["replay_payload_bytes"])
    smallest = min(rows, key=lambda row: row["replay_payload_bytes"])
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
            "total_forward_recompute_steps": sum(row["forward_recompute_steps"] for row in rows),
            "total_inverse_recompute_steps": sum(row["inverse_recompute_steps"] for row in rows),
            "zero_witness_rows": [
                row["id"]
                for row in rows
                if row["zero_witness"]
            ],
            "witness_backed_rows": [
                row["id"]
                for row in rows
                if row["witness_backed"]
            ],
        },
    }


def exact_per_unit(total: int, count: int, context: str) -> int:
    if count <= 0:
        raise PipelineError(f"{context} count must be positive")
    if total % count != 0:
        raise PipelineError(f"{context} total {total} is not divisible by count {count}")
    return total // count


def scaling_projection_family(
    *,
    family_id: str,
    label: str,
    unit: str,
    observed_count: int,
    model_payload_bytes: int,
    sample_payload_bytes_per_unit: int,
    witness_payload_bytes_per_unit: int,
    trace_payload_bytes_per_unit: int,
    variable_replay_payload_bytes_per_unit: int,
    recomputed_update_payload_bytes_per_unit: int,
    forward_recompute_steps_per_unit: int,
    inverse_recompute_steps_per_unit: int,
) -> dict[str, Any]:
    projections = []
    for scale_factor in (1, 10, 100):
        count = observed_count * scale_factor
        projected_replay = model_payload_bytes + count * variable_replay_payload_bytes_per_unit
        forward_steps = count * forward_recompute_steps_per_unit
        inverse_steps = count * inverse_recompute_steps_per_unit
        projections.append(
            {
                "scale_factor": scale_factor,
                "count": count,
                "projected_replay_payload_bytes": projected_replay,
                "projected_sample_payload_bytes": count * sample_payload_bytes_per_unit,
                "projected_witness_payload_bytes": count * witness_payload_bytes_per_unit,
                "projected_trace_payload_bytes": count * trace_payload_bytes_per_unit,
                "projected_recomputed_update_payload_bytes": (
                    count * recomputed_update_payload_bytes_per_unit
                ),
                "projected_forward_recompute_steps": forward_steps,
                "projected_inverse_recompute_steps": inverse_steps,
                "balanced_recompute": forward_steps == inverse_steps and forward_steps > 0,
                "projected_bytes_per_inverse_step": safe_ratio(projected_replay, inverse_steps),
            }
        )
    observed_projection = projections[0]
    return {
        "id": family_id,
        "label": label,
        "unit": unit,
        "observed_count": observed_count,
        "model_payload_bytes": model_payload_bytes,
        "sample_payload_bytes_per_unit": sample_payload_bytes_per_unit,
        "witness_payload_bytes_per_unit": witness_payload_bytes_per_unit,
        "trace_payload_bytes_per_unit": trace_payload_bytes_per_unit,
        "variable_replay_payload_bytes_per_unit": variable_replay_payload_bytes_per_unit,
        "recomputed_update_payload_bytes_per_unit": recomputed_update_payload_bytes_per_unit,
        "forward_recompute_steps_per_unit": forward_recompute_steps_per_unit,
        "inverse_recompute_steps_per_unit": inverse_recompute_steps_per_unit,
        "observed_replay_payload_bytes": observed_projection["projected_replay_payload_bytes"],
        "projections": projections,
    }


def scaling_projection_metrics(
    *,
    run_proof: dict[str, Any],
    model_evaluation_summary: dict[str, Any],
    model_evaluation_proof: dict[str, Any],
    mlp_witness: dict[str, Any],
    mlp_proof: dict[str, Any],
) -> dict[str, Any]:
    training_count = run_proof["entries"]
    evaluation_count = model_evaluation_summary["samples"]
    mlp_count = mlp_witness["samples"]
    families = [
        scaling_projection_family(
            family_id="training_trace",
            label="Training trace replay",
            unit="training update",
            observed_count=training_count,
            model_payload_bytes=run_proof["model_payload_bytes"],
            sample_payload_bytes_per_unit=run_proof["sample_bytes_per_step"],
            witness_payload_bytes_per_unit=run_proof["witness_bytes_per_step"],
            trace_payload_bytes_per_unit=run_proof["trace_replay_bytes_per_step"],
            variable_replay_payload_bytes_per_unit=run_proof["trace_replay_bytes_per_step"],
            recomputed_update_payload_bytes_per_unit=0,
            forward_recompute_steps_per_unit=exact_per_unit(
                run_proof["forward_recompute_steps"],
                training_count,
                "training trace forward recompute",
            ),
            inverse_recompute_steps_per_unit=exact_per_unit(
                run_proof["inverse_recompute_steps"],
                training_count,
                "training trace inverse recompute",
            ),
        ),
        scaling_projection_family(
            family_id="model_evaluation_batch",
            label="Batch model-evaluation replay",
            unit="evaluated sample",
            observed_count=evaluation_count,
            model_payload_bytes=model_evaluation_proof["model_payload_bytes"],
            sample_payload_bytes_per_unit=exact_per_unit(
                model_evaluation_proof["sample_payload_bytes"],
                evaluation_count,
                "model evaluation sample payload",
            ),
            witness_payload_bytes_per_unit=exact_per_unit(
                model_evaluation_proof["witness_payload_bytes"],
                evaluation_count,
                "model evaluation witness payload",
            ),
            trace_payload_bytes_per_unit=exact_per_unit(
                model_evaluation_proof["trace_payload_bytes"],
                evaluation_count,
                "model evaluation trace payload",
            ),
            variable_replay_payload_bytes_per_unit=exact_per_unit(
                model_evaluation_proof["sample_payload_bytes"]
                + model_evaluation_proof["witness_payload_bytes"]
                + model_evaluation_proof["trace_payload_bytes"],
                evaluation_count,
                "model evaluation variable replay payload",
            ),
            recomputed_update_payload_bytes_per_unit=0,
            forward_recompute_steps_per_unit=exact_per_unit(
                model_evaluation_proof["forward_recompute_steps"],
                evaluation_count,
                "model evaluation forward recompute",
            ),
            inverse_recompute_steps_per_unit=exact_per_unit(
                model_evaluation_proof["inverse_recompute_steps"],
                evaluation_count,
                "model evaluation inverse recompute",
            ),
        ),
        scaling_projection_family(
            family_id="mlp_witness_trace",
            label="MLP witness trace replay",
            unit="MLP sample update",
            observed_count=mlp_count,
            model_payload_bytes=mlp_proof["model_payload_bytes"],
            sample_payload_bytes_per_unit=exact_per_unit(
                mlp_proof["sample_payload_bytes"],
                mlp_count,
                "MLP sample payload",
            ),
            witness_payload_bytes_per_unit=mlp_proof["witness_bytes_per_sample"],
            trace_payload_bytes_per_unit=mlp_proof["trace_bytes_per_sample"],
            variable_replay_payload_bytes_per_unit=mlp_proof["trace_bytes_per_sample"],
            recomputed_update_payload_bytes_per_unit=mlp_proof["recomputed_update_bytes_per_sample"],
            forward_recompute_steps_per_unit=exact_per_unit(
                mlp_proof["forward_recompute_steps"],
                mlp_count,
                "MLP forward recompute",
            ),
            inverse_recompute_steps_per_unit=exact_per_unit(
                mlp_proof["inverse_recompute_steps"],
                mlp_count,
                "MLP inverse recompute",
            ),
        ),
    ]
    all_projections = [
        (family, projection)
        for family in families
        for projection in family["projections"]
    ]
    largest = max(
        all_projections,
        key=lambda item: item[1]["projected_replay_payload_bytes"],
    )
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
            "projection_scales": [1, 10, 100],
            "total_observed_replay_payload_bytes": sum(
                family["observed_replay_payload_bytes"] for family in families
            ),
            "total_projected_replay_payload_bytes_at_100x": sum(
                projection["projected_replay_payload_bytes"] for projection in hundred_x
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


def scan_rows_contain_step(rows: list[dict[str, Any]], step: int) -> bool:
    return any(row.get("step") == step for row in rows)


def first_scan_step(rows: list[dict[str, Any]]) -> Optional[int]:
    if not rows:
        return None
    step = rows[0].get("step")
    return step if isinstance(step, int) and not isinstance(step, bool) else None


def first_evaluation_row(rows: list[dict[str, Any]]) -> Optional[int]:
    if not rows:
        return None
    index = rows[0].get("index")
    return index if isinstance(index, int) and not isinstance(index, bool) else None


def select_audit_step_from_scan(
    audit_scan_report: dict[str, Any],
    *,
    strategy: str,
    requested_step: int,
) -> int:
    summary = audit_scan_report["summary"]
    if strategy == "explicit":
        return requested_step
    if strategy == "lowest-margin":
        return int(summary["lowest_margin_step"])
    if strategy == "largest-update":
        return int(summary["largest_update_step"])
    if strategy == "top-suspicious":
        selected = first_scan_step(audit_scan_report["top_suspicious"])
        if selected is None:
            raise PipelineError("audit scan did not include top_suspicious rows")
        return selected
    raise PipelineError(f"unknown audit step strategy: {strategy}")


def select_evaluation_row_from_scan(
    evaluation_scan_report: dict[str, Any],
    *,
    strategy: str,
    requested_row: int,
) -> int:
    summary = evaluation_scan_report["summary"]
    if strategy == "explicit":
        return requested_row
    if strategy == "lowest-margin":
        return int(summary["lowest_margin_index"])
    if strategy == "top-incorrect":
        selected = first_evaluation_row(evaluation_scan_report["top_incorrect"])
        if selected is None:
            raise PipelineError("model evaluation scan did not include top_incorrect rows")
        return selected
    raise PipelineError(f"unknown evaluation row strategy: {strategy}")


def strategy_step_from_scan(
    audit_scan_report: dict[str, Any],
    *,
    strategy: str,
    requested_step: int,
) -> int:
    return select_audit_step_from_scan(
        audit_scan_report,
        strategy=strategy,
        requested_step=requested_step,
    )


def strategy_row_from_scan(
    evaluation_scan_report: dict[str, Any],
    *,
    strategy: str,
    requested_row: int,
) -> int:
    return select_evaluation_row_from_scan(
        evaluation_scan_report,
        strategy=strategy,
        requested_row=requested_row,
    )


def training_step_selection_metric(
    *,
    requested_step: int,
    strategy: str,
    audit_scan_report: dict[str, Any],
    audit_step: dict[str, Any],
) -> dict[str, Any]:
    summary = audit_scan_report["summary"]
    selected_step = audit_step["step"]
    scan_top_suspicious_step = first_scan_step(audit_scan_report["top_suspicious"])
    scan_top_low_margin_step = first_scan_step(audit_scan_report["top_low_margin"])
    scan_top_large_updates_step = first_scan_step(audit_scan_report["top_large_updates"])
    selection_strategy_step = strategy_step_from_scan(
        audit_scan_report,
        strategy=strategy,
        requested_step=requested_step,
    )
    matches_lowest = selected_step == summary["lowest_margin_step"]
    matches_largest = selected_step == summary["largest_update_step"]
    in_top_suspicious = scan_rows_contain_step(audit_scan_report["top_suspicious"], selected_step)
    in_top_low_margin = scan_rows_contain_step(audit_scan_report["top_low_margin"], selected_step)
    in_top_large_updates = scan_rows_contain_step(audit_scan_report["top_large_updates"], selected_step)
    reasons = []
    if selected_step == requested_step:
        reasons.append("requested")
    if matches_lowest:
        reasons.append("lowest_margin")
    if matches_largest:
        reasons.append("largest_update")
    if in_top_suspicious:
        reasons.append("top_suspicious")
    if in_top_low_margin:
        reasons.append("top_low_margin")
    if in_top_large_updates:
        reasons.append("top_large_updates")
    return {
        "selection_strategy": strategy,
        "selection_strategy_step": selection_strategy_step,
        "requested_step": requested_step,
        "selected_step": selected_step,
        "selected_sample_index": audit_step["sample_index"],
        "selected_margin": audit_step["logit_margin"]["margin"],
        "selected_max_abs_weight_delta": audit_step["update"]["max_abs_weight_delta"],
        "scan_lowest_margin_step": summary["lowest_margin_step"],
        "scan_lowest_margin": summary["lowest_margin"],
        "scan_largest_update_step": summary["largest_update_step"],
        "scan_max_abs_weight_delta": summary["max_abs_weight_delta"],
        "scan_top_suspicious_step": scan_top_suspicious_step,
        "scan_top_low_margin_step": scan_top_low_margin_step,
        "scan_top_large_updates_step": scan_top_large_updates_step,
        "matches_selection_strategy": selected_step == selection_strategy_step,
        "matches_requested_step": selected_step == requested_step,
        "matches_scan_lowest_margin": matches_lowest,
        "matches_scan_largest_update": matches_largest,
        "present_in_top_suspicious": in_top_suspicious,
        "present_in_top_low_margin": in_top_low_margin,
        "present_in_top_large_updates": in_top_large_updates,
        "selection_reasons": reasons,
    }


def evaluation_row_selection_metric(
    *,
    requested_row: int,
    strategy: str,
    evaluation_scan_report: dict[str, Any],
    row_inference: dict[str, Any],
) -> dict[str, Any]:
    summary = evaluation_scan_report["summary"]
    sample_source = row_inference.get("sample_source", {})
    row_block = row_inference.get("row", {})
    selected_row = sample_source.get("row_index", row_block.get("index"))
    if selected_row is None:
        raise PipelineError("evaluation-row inference report did not include a row index")
    selected_source_sample_index = sample_source.get(
        "source_sample_index",
        row_block.get("source_sample_index"),
    )
    scan_top_low_margin_row = first_evaluation_row(evaluation_scan_report["top_low_margin"])
    scan_top_incorrect_row = first_evaluation_row(evaluation_scan_report["top_incorrect"])
    scan_top_low_margin_rows = [
        row["index"]
        for row in evaluation_scan_report["top_low_margin"]
    ]
    scan_top_incorrect_rows = [
        row["index"]
        for row in evaluation_scan_report["top_incorrect"]
    ]
    selection_strategy_row = strategy_row_from_scan(
        evaluation_scan_report,
        strategy=strategy,
        requested_row=requested_row,
    )
    matches_lowest = selected_row == summary["lowest_margin_index"]
    in_top_low_margin = any(
        row.get("index") == selected_row
        for row in evaluation_scan_report["top_low_margin"]
    )
    in_top_incorrect = any(
        row.get("index") == selected_row
        for row in evaluation_scan_report["top_incorrect"]
    )
    reasons = []
    if selected_row == requested_row:
        reasons.append("requested")
    if matches_lowest:
        reasons.append("lowest_margin")
    if in_top_low_margin:
        reasons.append("top_low_margin")
    if in_top_incorrect:
        reasons.append("top_incorrect")
    return {
        "selection_strategy": strategy,
        "selection_strategy_row": selection_strategy_row,
        "requested_row": requested_row,
        "selected_row": selected_row,
        "selected_source_sample_index": selected_source_sample_index,
        "selected_prediction": row_inference["prediction"],
        "selected_correct": row_inference["correct"],
        "selected_margin": row_inference["attribution"]["margin"],
        "scan_lowest_margin_row": summary["lowest_margin_index"],
        "scan_lowest_margin": summary["lowest_margin"],
        "scan_top_low_margin_row": scan_top_low_margin_row,
        "scan_top_incorrect_row": scan_top_incorrect_row,
        "scan_top_low_margin_rows": scan_top_low_margin_rows,
        "scan_top_incorrect_rows": scan_top_incorrect_rows,
        "matches_selection_strategy": selected_row == selection_strategy_row,
        "matches_requested_row": selected_row == requested_row,
        "matches_scan_lowest_margin": matches_lowest,
        "present_in_top_low_margin": in_top_low_margin,
        "present_in_top_incorrect": in_top_incorrect,
        "selection_reasons": reasons,
    }


def inference_active_pixel_count(report: dict[str, Any]) -> int:
    active_pixels = report.get("active_pixels")
    if active_pixels is None:
        active_pixels = report["proof"]["active_pixels"]
    if isinstance(active_pixels, list):
        return len(active_pixels)
    return active_pixels


def inference_ledger_signature(report: dict[str, Any]) -> dict[str, Any]:
    attribution = report["attribution"]
    return {
        "contribution_count": attribution["contribution_count"],
        "margin_contribution_count": attribution["margin_contribution_count"],
        "contribution_ledger_fingerprint": attribution["contribution_ledger_fingerprint"],
        "margin_contribution_ledger_fingerprint": attribution[
            "margin_contribution_ledger_fingerprint"
        ],
    }


def inference_trace_source(
    role: str,
    report: dict[str, Any],
    *,
    proof: Optional[dict[str, Any]] = None,
    explanation: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    attribution = report["attribution"]
    proof = report.get("proof") if proof is None else proof
    explanation = report.get("explanation_contract") if explanation is None else explanation
    return {
        "role": role,
        "prediction": report["prediction"],
        "correct": report["correct"],
        "margin": attribution["margin"],
        "active_pixels": inference_active_pixel_count(report),
        "ledger": inference_ledger_signature(report),
        "explanation_passed": None if explanation is None else explanation["passed"],
        "replay_payload_bytes": None if proof is None else proof["replay_payload_bytes"],
        "forward_recompute_steps": None if proof is None else proof["forward_recompute_steps"],
        "inverse_recompute_steps": None if proof is None else proof["inverse_recompute_steps"],
    }


def inference_result_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left[field] == right[field]
        for field in ("prediction", "correct", "margin", "active_pixels")
    )


def inference_ledger_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left["ledger"] == right["ledger"]


def inference_trace_row(
    *,
    trace_id: str,
    label: str,
    source_kind: str,
    sample_index: int,
    row_index: Optional[int],
    report: dict[str, Any],
    verification: dict[str, Any],
    reference: Optional[dict[str, Any]] = None,
    requires_reference: bool,
) -> dict[str, Any]:
    report_source = inference_trace_source("native_report", report)
    verification_source = inference_trace_source(
        "replay_verification",
        verification,
        proof=verification["proof"],
        explanation=verification["explanation_contract"],
    )
    if reference is None:
        reference_source = {
            "available": False,
            "role": "q31_reference",
            "prediction": None,
            "correct": None,
            "margin": None,
            "active_pixels": None,
            "ledger": None,
        }
        reference_result_matches = False
        reference_ledger_matches = False
    else:
        reference_block = inference_trace_source(
            "q31_reference",
            reference,
            proof=None,
            explanation=None,
        )
        reference_source = {
            "available": True,
            **reference_block,
        }
        reference_result_matches = inference_result_matches(report_source, reference_block)
        reference_ledger_matches = inference_ledger_matches(report_source, reference_block)

    report_verification_result = inference_result_matches(report_source, verification_source)
    report_verification_ledgers = inference_ledger_matches(report_source, verification_source)
    verification_replay_passed = (
        verification["proof_matches"] is True
        and verification["result_matches"] is True
        and verification["restored_initial_state"] is True
    )
    source_inputs_checked = (
        verification["source_model_checked"] is True
        and (
            verification.get("source_sample_checked") is True
            or verification.get("source_evaluation_checked") is True
        )
    )
    required_reference_passed = (
        not requires_reference
        or (
            reference_source["available"] is True
            and reference_result_matches is True
            and reference_ledger_matches is True
        )
    )
    checks = {
        "report_verification_result_matches": report_verification_result,
        "report_verification_ledgers_match": report_verification_ledgers,
        "report_explanation_passed": report["explanation_contract"]["passed"],
        "verification_explanation_passed": verification["explanation_contract"]["passed"],
        "verification_replay_passed": verification_replay_passed,
        "source_inputs_checked": source_inputs_checked,
        "reference_required": requires_reference,
        "reference_available": reference_source["available"],
        "reference_result_matches_report": reference_result_matches,
        "reference_ledgers_match_report": reference_ledger_matches,
    }
    passed = (
        report_verification_result
        and report_verification_ledgers
        and report["explanation_contract"]["passed"] is True
        and verification["explanation_contract"]["passed"] is True
        and verification_replay_passed
        and source_inputs_checked
        and required_reference_passed
    )
    return {
        "id": trace_id,
        "label": label,
        "source_kind": source_kind,
        "sample_index": sample_index,
        "row_index": row_index,
        "prediction": report_source["prediction"],
        "correct": report_source["correct"],
        "margin": report_source["margin"],
        "active_pixels": report_source["active_pixels"],
        "report": report_source,
        "verification": verification_source,
        "reference": reference_source,
        "checks": checks,
        "passed": passed,
    }


def inference_trace_profile_metrics(
    *,
    native_inference: dict[str, Any],
    native_inference_verification: dict[str, Any],
    row_inference: dict[str, Any],
    row_inference_verification: dict[str, Any],
    reference_inference: dict[str, Any],
) -> dict[str, Any]:
    native_sample_index = native_inference.get("sample_index")
    if native_sample_index is None:
        native_sample_index = reference_inference["sample_source"]["sample_index"]
    native_trace = inference_trace_row(
        trace_id="native_selected_sample",
        label="Selected audit-sample inference",
        source_kind=native_inference["sample_source"]["kind"],
        sample_index=native_sample_index,
        row_index=None,
        report=native_inference,
        verification=native_inference_verification,
        reference=reference_inference,
        requires_reference=True,
    )
    row_source = row_inference["sample_source"]
    row_trace = inference_trace_row(
        trace_id="evaluation_row",
        label="Selected model-evaluation row inference",
        source_kind=row_source["kind"],
        sample_index=row_source.get(
            "source_sample_index",
            row_inference["sample_index"]
            if row_inference["sample_index"] is not None
            else row_inference["row"]["source_sample_index"],
        ),
        row_index=row_source.get("row_index", row_inference["row"]["index"]),
        report=row_inference,
        verification=row_inference_verification,
        reference=None,
        requires_reference=False,
    )
    traces = [native_trace, row_trace]
    return {
        "kind": INFERENCE_TRACE_PROFILE_KIND,
        "traces": traces,
        "summary": {
            "trace_count": len(traces),
            "reference_checked_traces": sum(
                1 for trace in traces if trace["reference"]["available"] is True
            ),
            "all_report_verification_results_match": all(
                trace["checks"]["report_verification_result_matches"]
                for trace in traces
            ),
            "all_report_verification_ledgers_match": all(
                trace["checks"]["report_verification_ledgers_match"]
                for trace in traces
            ),
            "all_required_references_match": all(
                not trace["checks"]["reference_required"]
                or (
                    trace["checks"]["reference_available"]
                    and trace["checks"]["reference_result_matches_report"]
                    and trace["checks"]["reference_ledgers_match_report"]
                )
                for trace in traces
            ),
            "all_replay_verified": all(
                trace["checks"]["verification_replay_passed"] for trace in traces
            ),
            "all_explanations_passed": all(
                trace["checks"]["report_explanation_passed"]
                and trace["checks"]["verification_explanation_passed"]
                for trace in traces
            ),
            "all_sources_checked": all(
                trace["checks"]["source_inputs_checked"] for trace in traces
            ),
            "all_passed": all(trace["passed"] for trace in traces),
        },
    }


def pipeline_claims(checks: list[dict[str, Any]]) -> dict[str, Any]:
    claim_rows = [
        {
            "claim": "debug_training_update",
            "description": "A selected training update can be inspected, replayed forward, and reversed exactly.",
            "gate_metrics": [
                "training_audit_steps",
                "training_audit_witness_mismatches",
                "training_step_selection_traceable",
                "training_step_replay",
                "training_step_debug_contract",
                "training_update_ledger_fingerprints",
            ],
            "evidence": [
                "report.audit_scan",
                "report.audit_step",
                "report.step_verification",
                "bundle.training_audit",
                "bundle.training_step",
            ],
            "metrics": [
                "training_audit_scan",
                "training_step_selection",
                "training_step_replay",
                "training_step_debug",
            ],
        },
        {
            "claim": "auditable_model_lineage",
            "description": "The final model is tied to signed training, model, sample, replay, and audit-contract evidence.",
            "gate_metrics": [
                "reverse_restored_initial_model",
                "reverse_checked_all_training_steps",
                "training_audit_lineage_replay",
                "audit_contract",
            ],
            "evidence": [
                "report.run_report",
                "report.audit_verification",
                "report.artifact_comparison",
                "bundle.training_audit",
                "bundle.model",
                "bundle.samples",
            ],
            "metrics": ["reverse", "training_audit_replay", "artifact_profile"],
        },
        {
            "claim": "deterministic_q31_inference",
            "description": "Native fixed-point inference and the independent Q31 reference agree and replay backward.",
            "gate_metrics": [
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
            "evidence": [
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
            "metrics": [
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
        },
        {
            "claim": "memory_recompute_profile",
            "description": "The profile records speed, peak RSS, payload bytes, trace/witness ratios, balanced recompute, and reverse-check cost.",
            "gate_metrics": [
                "run_peak_rss_bytes",
                "trace_to_model_payload_ratio",
                "witness_to_model_payload_ratio",
            "balanced_recompute_steps",
            "recompute_frontier_complete",
            "scaling_projection_complete",
            "reverse_check_cost_measured",
            "reverse_check_elapsed_ratio",
        ],
            "evidence": [
                "report.run_report",
                "report.artifact_comparison",
                "profile_markdown",
            ],
            "metrics": [
                "train",
                "built_in_eval",
                "memory",
                "artifact_profile",
                "recompute_frontier",
                "scaling_projection",
                "reverse_check_cost",
                "replay_payload_bytes",
            ],
        },
        {
            "claim": "mlp_activation_mask_witnesses",
            "description": "The MLP trace keeps explicit activation, ReLU mask, error, and hidden-delta witnesses.",
            "gate_metrics": ["mlp_witness_replay"],
            "evidence": [
                "report.mlp_witness",
                "bundle.mlp_vars",
                "bundle.mlp_run_output",
            ],
            "metrics": ["mlp_witness"],
        },
        {
            "claim": "invertible_layer_without_witness",
            "description": "The coupling, triangular residual, and preprocessing examples run forward and backward with zero witness and trace bytes.",
            "gate_metrics": [
                "invertible_coupling_replay",
                "triangular_residual_replay",
                "reversible_preprocess_replay",
            ],
            "evidence": [
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
            "metrics": ["invertible_coupling", "triangular_residual", "reversible_preprocess"],
        },
        {
            "claim": "artifact_evidence_integrity",
            "description": "Every generated report, bundle, seed, runtime JSON output, and Markdown profile is digest-indexed.",
            "gate_metrics": [],
            "evidence": [
                "profile_markdown",
                *[f"report.{key}" for key in PROFILE_REPORT_KEYS],
                *[f"bundle.{key}" for key in PIPELINE_BUNDLE_KEYS],
                "standalone.native_standalone_rev_classifier",
                "standalone.native_standalone_rev_run",
                "standalone.native_standalone_rev_roundtrip",
                "standalone.native_standalone_rev_roundtrip_verification",
            ],
            "metrics": [],
        },
        {
            "claim": "v6_ml_audit_scorecard",
            "description": "A compact scorecard exposes speed, memory, trace, replay, reverse-check, and contract status.",
            "gate_metrics": ["v6_scorecard_complete"],
            "evidence": [
                "profile_markdown",
                "report.run_report",
                "report.artifact_comparison",
                "report.native_inference_verification",
                "report.row_inference_verification",
            ],
            "metrics": ["scorecard"],
        },
        {
            "claim": "ml_roadmap_capabilities",
            "description": "The V1-V6 ML roadmap capabilities are backed by concrete gates, metrics, and digest-indexed evidence.",
            "gate_metrics": ["ml_roadmap_capability_map_complete"],
            "evidence": [
                "profile_markdown",
                "report.run_report",
                "report.audit_scan",
                "report.audit_step",
                "report.model_evaluation",
                "report.mlp_witness",
                "report.invertible_coupling",
                "report.triangular_residual",
                "report.reversible_preprocess",
                "report.reversible_inference_trace",
                "report.artifact_comparison",
            ],
            "metrics": ["ml_capability_map"],
        },
        {
            "claim": "north_star_reversible_ml_kernels",
            "description": "The current artifacts prove Reverie's practical ML goal: reversible, inspectable, deterministic small-model kernels.",
            "gate_metrics": ["ml_goal_readiness_complete"],
            "evidence": [
                "profile_markdown",
                "report.audit_step",
                "report.step_verification",
                "report.native_inference_verification",
                "report.row_inference_verification",
                "report.mlp_witness",
                "report.invertible_coupling",
                "report.triangular_residual",
                "report.reversible_preprocess",
                "report.reversible_inference_trace",
                "report.artifact_comparison",
            ],
            "metrics": ["ml_goal_readiness"],
        },
    ]
    for row in claim_rows:
        row["passed"] = gates_passed(checks, tuple(row["gate_metrics"]))
    return {
        "kind": CLAIMS_KIND,
        "passed": all(row["passed"] for row in claim_rows),
        "checks": claim_rows,
    }


def build_pipeline_summary(
    config: PipelineConfig,
    paths: dict[str, Path],
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    run = reports["run_report"]
    train = run["train"]
    built_in_eval = run["eval"]
    reverse = run["reverse"]
    run_proof = run["proof"]
    run_memory = run["memory"]
    audit_verification = reports["audit_verification"]
    audit_scan_report = reports["audit_scan"]
    audit_scan = audit_scan_report["summary"]
    audit_step = reports["audit_step"]
    debug_contract = audit_step["debug_contract"]
    artifact_profile = reports["artifact_comparison"]["ml_profile"]
    audit_contract = reports["artifact_comparison"]["audit_contract"]
    step_verification = reports["step_verification"]
    step_proof = step_verification["proof"]
    imported_model_import = reports["imported_model_import"]
    imported_model_verification = reports["imported_model_verification"]
    imported_model_inference = reports["imported_model_inference"]
    imported_model_inference_verification = reports[
        "imported_model_inference_verification"
    ]
    native_inference = reports["native_inference"]
    native_inference_verification = reports["native_inference_verification"]
    model_evaluation = reports["model_evaluation"]
    model_evaluation_summary = model_evaluation["summary"]
    model_evaluation_proof = model_evaluation["proof"]
    model_evaluation_verification = reports["model_evaluation_verification"]
    model_evaluation_scan_report = reports["model_evaluation_scan"]
    model_evaluation_scan = model_evaluation_scan_report["summary"]
    row_inference = reports["model_evaluation_row"]
    row_inference_verification = reports["row_inference_verification"]
    mlp_witness = reports["mlp_witness"]
    mlp_proof = mlp_witness["proof"]
    coupling = reports["invertible_coupling"]
    coupling_proof = coupling["proof"]
    residual = reports["triangular_residual"]
    residual_proof = residual["proof"]
    preprocess = reports["reversible_preprocess"]
    preprocess_proof = preprocess["proof"]
    inference_trace = reports["reversible_inference_trace"]
    inference_trace_proof = inference_trace["proof"]
    reference_inference = reports["q31_reference_inference"]
    reference_evaluation = reports["q31_reference_evaluation"]["summary"]
    try:
        native_standalone_source = paths["native_standalone_rev_classifier"].read_text(
            encoding="utf-8"
        )
    except OSError as error:
        raise PipelineError(
            f"failed to read standalone Reverie classifier {paths['native_standalone_rev_classifier']}: {error}"
        ) from error
    native_standalone_run = load_pipeline_json_artifact(
        paths["native_standalone_rev_run"],
        "standalone Reverie classifier run",
    )
    native_standalone_roundtrip = load_pipeline_json_artifact(
        paths["native_standalone_rev_roundtrip"],
        "standalone Reverie classifier roundtrip",
    )
    native_standalone_roundtrip_verification = load_pipeline_json_artifact(
        paths["native_standalone_rev_roundtrip_verification"],
        "standalone Reverie classifier roundtrip verification",
    )
    native_standalone_store = native_standalone_run.get("store", {})
    native_standalone_roundtrip_payload = native_standalone_roundtrip.get("payload", {})
    native_standalone_roundtrip_forward_store = (
        native_standalone_roundtrip_payload.get("forward", {}).get("store", {})
    )
    native_standalone_roundtrip_baseline_store = (
        native_standalone_roundtrip_payload.get("baseline", {}).get("store", {})
    )
    native_standalone_verification_checks = native_standalone_roundtrip_verification.get(
        "checks", {}
    )

    replay_payload_bytes = {
        "training_trace": run_proof["full_replay_payload_bytes"],
        "training_step": step_proof["replay_payload_bytes"],
        "imported_model_inference": imported_model_inference_verification["memory"][
            "replay_payload_bytes"
        ],
        "native_inference": native_inference_verification["memory"]["replay_payload_bytes"],
        "model_evaluation": model_evaluation_proof["replay_payload_bytes"],
        "evaluation_row_inference": row_inference_verification["memory"]["replay_payload_bytes"],
    }
    max_replay_payload_bytes = max(replay_payload_bytes.values())
    total_forward = artifact_profile["total_forward_recompute_steps"]
    total_inverse = artifact_profile["total_inverse_recompute_steps"]
    train_updates = train["samples"]
    reverse_checked = reverse["checked"]
    train_elapsed_seconds = train["elapsed_seconds"]
    reverse_elapsed_seconds = reverse["elapsed_seconds"]
    train_updates_per_second = train["updates_per_second"]
    reverse_steps_per_second = reverse["steps_per_second"]
    reverse_check_cost = {
        "enabled": reverse["enabled"],
        "train_updates": train_updates,
        "checked_steps": reverse_checked,
        "train_elapsed_seconds": train_elapsed_seconds,
        "reverse_elapsed_seconds": reverse_elapsed_seconds,
        "train_seconds_per_update": safe_ratio(train_elapsed_seconds, train_updates),
        "reverse_seconds_per_step": safe_ratio(reverse_elapsed_seconds, reverse_checked),
        "reverse_to_train_elapsed_ratio": safe_ratio(reverse_elapsed_seconds, train_elapsed_seconds),
        "reverse_to_train_throughput_ratio": safe_ratio(reverse_steps_per_second, train_updates_per_second),
        "train_updates_per_second": train_updates_per_second,
        "reverse_steps_per_second": reverse_steps_per_second,
        "proof_forward_recompute_steps": run_proof["forward_recompute_steps"],
        "proof_inverse_recompute_steps": run_proof["inverse_recompute_steps"],
        "proof_replay_payload_bytes": run_proof["full_replay_payload_bytes"],
        "trace_replay_payload_bytes": run_proof["trace_replay_payload_bytes"],
    }
    training_step_selection = training_step_selection_metric(
        requested_step=config.audit_step if config.requested_audit_step is None else config.requested_audit_step,
        strategy=config.audit_step_strategy,
        audit_scan_report=audit_scan_report,
        audit_step=audit_step,
    )
    evaluation_row_selection = evaluation_row_selection_metric(
        requested_row=(
            config.evaluation_row
            if config.requested_evaluation_row is None
            else config.requested_evaluation_row
        ),
        strategy=config.evaluation_row_strategy,
        evaluation_scan_report=model_evaluation_scan_report,
        row_inference=row_inference,
    )
    recompute_frontier = recompute_frontier_metrics(
        run_proof=run_proof,
        step_proof=step_proof,
        imported_inference_memory=imported_model_inference_verification["memory"],
        native_inference_memory=native_inference_verification["memory"],
        model_evaluation_proof=model_evaluation_proof,
        row_inference_memory=row_inference_verification["memory"],
        mlp_proof=mlp_proof,
        coupling_proof=coupling_proof,
        residual_proof=residual_proof,
        preprocess_proof=preprocess_proof,
        inference_trace_proof=inference_trace_proof,
    )
    scaling_projection = scaling_projection_metrics(
        run_proof=run_proof,
        model_evaluation_summary=model_evaluation_summary,
        model_evaluation_proof=model_evaluation_proof,
        mlp_witness=mlp_witness,
        mlp_proof=mlp_proof,
    )
    inference_trace_profile = inference_trace_profile_metrics(
        native_inference=native_inference,
        native_inference_verification=native_inference_verification,
        row_inference=row_inference,
        row_inference_verification=row_inference_verification,
        reference_inference=reference_inference,
    )

    metrics = {
        "train": {
            "samples": train["samples"],
            "correct": train["correct"],
            "accuracy_percent": train["accuracy_percent"],
            "updates_per_second": train["updates_per_second"],
        },
        "built_in_eval": {
            "samples": built_in_eval["samples"],
            "correct": built_in_eval["correct"],
            "accuracy_percent": built_in_eval["accuracy_percent"],
            "samples_per_second": built_in_eval["samples_per_second"],
        },
        "reverse": {
            "enabled": reverse["enabled"],
            "checked": reverse["checked"],
            "restored_initial_model": reverse["restored_initial_model"],
            "steps_per_second": reverse["steps_per_second"],
        },
        "reverse_check_cost": reverse_check_cost,
        "memory": {
            "run_peak_rss_bytes": run_memory.get("peak_rss_bytes"),
            "model_evaluation_peak_rss_bytes": model_evaluation_proof.get("peak_rss_bytes"),
            "estimated_payload_bytes": run_memory["estimated_payload_bytes"],
        },
        "training_audit_scan": {
            "steps": audit_scan["steps"],
            "correct": audit_scan["correct"],
            "incorrect": audit_scan["incorrect"],
            "accuracy_percent": audit_scan["accuracy_percent"],
            "witness_mismatches": audit_scan["witness_mismatches"],
            "trace_payload_bytes": audit_scan["trace_payload_bytes"],
            "lowest_margin_step": audit_scan["lowest_margin_step"],
            "lowest_margin": audit_scan["lowest_margin"],
            "largest_update_step": audit_scan["largest_update_step"],
            "max_abs_weight_delta": audit_scan["max_abs_weight_delta"],
        },
        "training_audit_replay": {
            "checked": audit_verification["checked"],
            "witnesses_match_forward_replay": audit_verification["witnesses_match_forward_replay"],
            "final_model_replayed": audit_verification["final_model_replayed"],
            "restored_initial_model": audit_verification["restored_initial_model"],
            "proof_matches": audit_verification["proof_matches"],
            "lineage_ledger_matches": audit_verification["lineage_ledger_matches"],
            "lineage_ledger_fingerprint": audit_verification["lineage_ledger"]["fingerprint"],
            "transition_ledger_fingerprint": audit_verification["lineage_ledger"]["payload"][
                "transition_ledger_fingerprint"
            ],
            "final_chain": audit_verification["lineage_ledger"]["payload"]["final_chain"],
        },
        "training_step_selection": training_step_selection,
        "artifact_profile": {
            "total_file_bytes": artifact_profile["total_file_bytes"],
            "total_logical_payload_bytes": artifact_profile["total_logical_payload_bytes"],
            "total_model_payload_bytes": artifact_profile["total_model_payload_bytes"],
            "total_witness_payload_bytes": artifact_profile["total_witness_payload_bytes"],
            "total_trace_payload_bytes": artifact_profile["total_trace_payload_bytes"],
            "total_recompute_steps": artifact_profile["total_recompute_steps"],
            "total_forward_recompute_steps": total_forward,
            "total_inverse_recompute_steps": total_inverse,
            "trace_to_model_payload_ratio": artifact_profile["trace_to_model_payload_ratio"],
            "witness_to_model_payload_ratio": artifact_profile["witness_to_model_payload_ratio"],
        },
        "recompute_frontier": recompute_frontier,
        "scaling_projection": scaling_projection,
        "training_step_replay": {
            "proof_matches": step_verification["proof_matches"],
            "witnesses_match": step_verification["forward"]["witnesses_match"],
            "after_model_matches": step_verification["forward"]["after_model_matches"],
            "before_model_restored": step_verification["reverse"]["before_model_restored"],
            "replay_payload_bytes": step_proof["replay_payload_bytes"],
        },
        "training_step_debug": {
            "claim": debug_contract["claim"],
            "passed": debug_contract["passed"],
            "step": audit_step["step"],
            "sample_index": audit_step["sample_index"],
            "prediction": audit_step["prediction"],
            "correct": audit_step["correct"],
            "margin": audit_step["logit_margin"]["margin"],
            "active_pixel_count": len(audit_step["active_pixels"]),
            "top_weight_deltas": len(audit_step["update"]["top_weight_deltas"]),
            "nonzero_bias_delta_count": audit_step["update"]["nonzero_bias_delta_count"],
            "nonzero_weight_delta_count": audit_step["update"]["nonzero_weight_delta_count"],
            "bias_delta_ledger_fingerprint": audit_step["update"]["bias_delta_ledger_fingerprint"],
            "weight_delta_ledger_fingerprint": audit_step["update"]["weight_delta_ledger_fingerprint"],
            "cause_ledger_fingerprint": audit_step["cause_ledger"]["fingerprint"],
            "reversed_later_steps": audit_step["model_window"]["reversed_later_steps"],
            "checks": debug_contract["checks"],
        },
        "model_import": {
            "source_model_json_path": imported_model_import["source_model_json_path"],
            "source_model_json_fingerprint": imported_model_import[
                "source_model_json_fingerprint"
            ],
            "source_model_json_file_sha256": imported_model_import[
                "source_model_json_file_sha256"
            ],
            "model_output": imported_model_import["model_output"],
            "model_payload_bytes": imported_model_import["storage"][
                "model_payload_bytes"
            ],
            "import_shape_matches": imported_model_import["shape_matches"],
            "import_provenance_kind": imported_model_import["provenance_kind"],
            "import_payload_fingerprint": imported_model_import["fingerprints"][
                "payload"
            ],
            "verification_shape_matches": imported_model_verification[
                "shape_matches"
            ],
            "verification_provenance_matches": imported_model_verification[
                "provenance_matches"
            ],
            "verification_provenance_kind": imported_model_verification[
                "provenance_kind"
            ],
            "verification_source_checked": imported_model_verification[
                "source_model_json_checked"
            ],
            "verification_payload_fingerprint": imported_model_verification[
                "fingerprints"
            ]["payload"],
            "proof_matches": imported_model_verification["proof_matches"],
            "training_steps": imported_model_verification["training_steps"],
            "source_model_json": imported_model_verification["source_model_json"],
        },
        "imported_model_inference": {
            "correct": imported_model_inference["proof"]["correct"],
            "prediction": imported_model_inference["prediction"],
            "native_prediction": native_inference["prediction"],
            "correct_matches_native": (
                imported_model_inference["correct"] == native_inference["correct"]
            ),
            "prediction_matches_native": (
                imported_model_inference["prediction"] == native_inference["prediction"]
            ),
            "margin_matches_native": (
                imported_model_inference["attribution"]["margin"]
                == native_inference["attribution"]["margin"]
            ),
            "contribution_ledger_matches_native": (
                imported_model_inference["attribution"][
                    "contribution_ledger_fingerprint"
                ]
                == native_inference["attribution"]["contribution_ledger_fingerprint"]
            ),
            "margin_contribution_ledger_matches_native": (
                imported_model_inference["attribution"][
                    "margin_contribution_ledger_fingerprint"
                ]
                == native_inference["attribution"][
                    "margin_contribution_ledger_fingerprint"
                ]
            ),
            "proof_matches": imported_model_inference_verification["proof_matches"],
            "result_matches": imported_model_inference_verification["result_matches"],
            "restored_initial_state": imported_model_inference_verification[
                "restored_initial_state"
            ],
            "source_model_checked": imported_model_inference_verification[
                "source_model_checked"
            ],
            "source_sample_checked": imported_model_inference_verification[
                "source_sample_checked"
            ],
            "replay_payload_bytes": imported_model_inference_verification["memory"][
                "replay_payload_bytes"
            ],
            "forward_recompute_steps": imported_model_inference_verification["memory"][
                "forward_recompute_steps"
            ],
            "inverse_recompute_steps": imported_model_inference_verification["memory"][
                "inverse_recompute_steps"
            ],
        },
        "imported_model_inference_explanation": (
            imported_model_inference_verification["explanation_contract"]
        ),
        "native_inference_replay": {
            "correct": native_inference["proof"]["correct"],
            "proof_matches": native_inference_verification["proof_matches"],
            "result_matches": native_inference_verification["result_matches"],
            "restored_initial_state": native_inference_verification["restored_initial_state"],
            "source_model_checked": native_inference_verification["source_model_checked"],
            "source_sample_checked": native_inference_verification["source_sample_checked"],
            "replay_payload_bytes": native_inference_verification["memory"]["replay_payload_bytes"],
        },
        "native_inference_explanation": native_inference_verification["explanation_contract"],
        "native_standalone_rev_classifier": {
            "path": str(paths["native_standalone_rev_classifier"]),
            "bytes": paths["native_standalone_rev_classifier"].stat().st_size,
            "sha256": sha256_file(paths["native_standalone_rev_classifier"]),
            "line_count": len(native_standalone_source.splitlines()),
            "contains_vecmat_q31": "vecmat_q31(image, weights)" in native_standalone_source,
            "contains_prediction_assert": "assert prediction == expected_prediction"
            in native_standalone_source,
            "contains_correct_assert": "assert correct == expected_correct"
            in native_standalone_source,
            "run_kind": native_standalone_run.get("kind"),
            "run_prediction": native_standalone_store.get("prediction"),
            "run_correct": native_standalone_store.get("correct"),
            "run_label": native_standalone_store.get("label"),
            "expected_prediction": native_standalone_store.get("expected_prediction"),
            "expected_correct": native_standalone_store.get("expected_correct"),
            "run_uses_vecmat_q31": (
                native_standalone_run.get("ml_profile", {})
                .get("builtin_counts", {})
                .get("vecmat_q31")
                == 1
            ),
            "matches_native_prediction": (
                native_standalone_store.get("prediction") == native_inference["prediction"]
            ),
            "matches_native_correct": (
                bool(native_standalone_store.get("correct")) == native_inference["correct"]
            ),
            "matches_native_label": (
                native_standalone_store.get("label") == native_inference["label"]
            ),
            "roundtrip_kind": native_standalone_roundtrip.get("kind"),
            "roundtrip_passed": native_standalone_roundtrip.get("passed"),
            "roundtrip_prediction": native_standalone_roundtrip_forward_store.get("prediction"),
            "roundtrip_correct": native_standalone_roundtrip_forward_store.get("correct"),
            "roundtrip_restored_prediction": native_standalone_roundtrip_baseline_store.get(
                "prediction"
            ),
            "roundtrip_restored_correct": native_standalone_roundtrip_baseline_store.get(
                "correct"
            ),
            "roundtrip_fingerprint": native_standalone_roundtrip.get("fingerprint"),
            "verification_kind": native_standalone_roundtrip_verification.get("kind"),
            "verification_passed": native_standalone_roundtrip_verification.get("passed"),
            "verification_source_hash_matches": native_standalone_verification_checks.get(
                "source_hash_matches"
            ),
            "verification_replay_restoration_passed": native_standalone_verification_checks.get(
                "replay_restoration_passed"
            ),
            "verification_replayed": native_standalone_verification_checks.get("replayed"),
        },
        "model_evaluation": {
            "samples": model_evaluation_summary["samples"],
            "correct": model_evaluation_summary["correct"],
            "incorrect": model_evaluation_summary["incorrect"],
            "accuracy_percent": model_evaluation_summary["accuracy_percent"],
            "samples_per_second": model_evaluation_summary["samples_per_second"],
            "lowest_margin": model_evaluation_summary["lowest_margin"],
            "replay_payload_bytes": model_evaluation_proof["replay_payload_bytes"],
        },
        "model_evaluation_replay": {
            "rows_match": model_evaluation_verification["rows_match"],
            "proof_matches": model_evaluation_verification["proof_matches"],
            "restored_initial_state": model_evaluation_verification["restored_initial_state"],
            "source_model_checked": model_evaluation_verification["source_model_checked"],
            "source_samples_checked": model_evaluation_verification["source_samples_checked"],
        },
        "evaluation_row_selection": evaluation_row_selection,
        "evaluation_row_inference": {
            "correct": row_inference["proof"]["correct"],
            "proof_matches": row_inference_verification["proof_matches"],
            "result_matches": row_inference_verification["result_matches"],
            "restored_initial_state": row_inference_verification["restored_initial_state"],
            "source_model_checked": row_inference_verification["source_model_checked"],
            "source_evaluation_checked": row_inference_verification["source_evaluation_checked"],
            "replay_payload_bytes": row_inference_verification["memory"]["replay_payload_bytes"],
        },
        "evaluation_row_inference_explanation": row_inference_verification["explanation_contract"],
        "q31_reference_inference": {
            "prediction": reference_inference["prediction"],
            "correct": reference_inference["correct"],
            "margin": reference_inference["attribution"]["margin"],
            "active_pixels": reference_inference["active_pixels"],
            "contribution_count": reference_inference["attribution"]["contribution_count"],
            "margin_contribution_count": reference_inference["attribution"][
                "margin_contribution_count"
            ],
            "contribution_ledger_fingerprint": reference_inference["attribution"][
                "contribution_ledger_fingerprint"
            ],
            "margin_contribution_ledger_fingerprint": reference_inference["attribution"][
                "margin_contribution_ledger_fingerprint"
            ],
            "native_prediction": native_inference["prediction"],
            "native_correct": native_inference["correct"],
            "native_margin": native_inference["attribution"]["margin"],
            "native_active_pixels": len(native_inference["active_pixels"]),
            "native_contribution_count": native_inference["attribution"]["contribution_count"],
            "native_margin_contribution_count": native_inference["attribution"][
                "margin_contribution_count"
            ],
            "native_contribution_ledger_fingerprint": native_inference["attribution"][
                "contribution_ledger_fingerprint"
            ],
            "native_margin_contribution_ledger_fingerprint": native_inference["attribution"][
                "margin_contribution_ledger_fingerprint"
            ],
            "prediction_matches_native": reference_inference["prediction"] == native_inference["prediction"],
            "correct_matches_native": reference_inference["correct"] == native_inference["correct"],
            "margin_matches_native": (
                reference_inference["attribution"]["margin"] == native_inference["attribution"]["margin"]
            ),
            "active_pixels_match_native": (
                reference_inference["active_pixels"] == len(native_inference["active_pixels"])
            ),
            "contribution_ledger_matches_native": (
                reference_inference["attribution"]["contribution_ledger_fingerprint"]
                == native_inference["attribution"]["contribution_ledger_fingerprint"]
            ),
            "margin_contribution_ledger_matches_native": (
                reference_inference["attribution"]["margin_contribution_ledger_fingerprint"]
                == native_inference["attribution"]["margin_contribution_ledger_fingerprint"]
            ),
            "attribution_matches_logit": reference_inference["attribution"]["matches_logit"],
            "attribution_matches_margin": reference_inference["attribution"]["matches_margin"],
        },
        "q31_reference_evaluation": {
            "samples": reference_evaluation["samples"],
            "correct": reference_evaluation["correct"],
            "incorrect": reference_evaluation["incorrect"],
            "accuracy_percent": reference_evaluation["accuracy_percent"],
            "lowest_margin": reference_evaluation["lowest_margin"],
        },
        "mlp_witness": {
            "passed": mlp_witness["passed"],
            "samples": mlp_witness["samples"],
            "dataset_loops": mlp_witness["dataset_loops"],
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
            "trace_to_model_payload_ratio": inference_trace_proof["trace_to_model_payload_ratio"],
            "witness_to_state_payload_ratio": inference_trace_proof[
                "witness_to_state_payload_ratio"
            ],
            "trace_to_state_payload_ratio": inference_trace_proof["trace_to_state_payload_ratio"],
        },
        "inference_trace_profile": inference_trace_profile,
        "replay_payload_bytes": replay_payload_bytes,
        "max_replay_payload_bytes": max_replay_payload_bytes,
    }

    checks: list[dict[str, Any]] = []
    add_min_percent_gate(
        checks,
        "train_accuracy_percent",
        train["accuracy_percent"],
        config.min_train_accuracy,
    )
    add_min_percent_gate(
        checks,
        "built_in_eval_accuracy_percent",
        built_in_eval["accuracy_percent"],
        config.min_eval_accuracy,
    )
    add_min_percent_gate(
        checks,
        "training_audit_accuracy_percent",
        audit_scan["accuracy_percent"],
        config.min_audit_accuracy,
    )
    add_min_percent_gate(
        checks,
        "model_evaluation_accuracy_percent",
        model_evaluation_summary["accuracy_percent"],
        config.min_model_evaluation_accuracy,
    )
    add_min_percent_gate(
        checks,
        "q31_reference_accuracy_percent",
        reference_evaluation["accuracy_percent"],
        config.min_reference_accuracy,
    )
    add_gate(
        checks,
        "reverse_restored_initial_model",
        reverse["restored_initial_model"],
        "true",
        reverse["restored_initial_model"] is True,
    )
    add_gate(
        checks,
        "reverse_checked_all_training_steps",
        reverse["checked"],
        f"== {train['samples']}",
        reverse["checked"] == train["samples"] and train["samples"] > 0,
    )
    add_gate(
        checks,
        "run_peak_rss_bytes",
        run_memory.get("peak_rss_bytes"),
        "> 0",
        isinstance(run_memory.get("peak_rss_bytes"), int)
        and run_memory["peak_rss_bytes"] > 0,
    )
    add_gate(
        checks,
        "training_audit_steps",
        audit_scan["steps"],
        f"== {train['samples']}",
        audit_scan["steps"] == train["samples"],
    )
    add_gate(
        checks,
        "training_audit_lineage_replay",
        metrics["training_audit_replay"],
        "whole training trace replays forward and backward, and lineage ledger recomputes",
        metrics["training_audit_replay"]["checked"] == train["samples"]
        and metrics["training_audit_replay"]["witnesses_match_forward_replay"] is True
        and metrics["training_audit_replay"]["final_model_replayed"] is True
        and metrics["training_audit_replay"]["restored_initial_model"] is True
        and metrics["training_audit_replay"]["proof_matches"] is True
        and metrics["training_audit_replay"]["lineage_ledger_matches"] is True
        and is_sha256(metrics["training_audit_replay"]["lineage_ledger_fingerprint"])
        and is_sha256(metrics["training_audit_replay"]["transition_ledger_fingerprint"])
        and is_sha256(metrics["training_audit_replay"]["final_chain"]),
    )
    add_gate(
        checks,
        "training_audit_witness_mismatches",
        audit_scan["witness_mismatches"],
        f"<= {config.max_witness_mismatches}",
        audit_scan["witness_mismatches"] <= config.max_witness_mismatches,
    )
    add_gate(
        checks,
        "training_step_selection_traceable",
        metrics["training_step_selection"],
        "selected step matches the configured audit-step strategy and is tied to scan evidence",
        training_step_selection["matches_selection_strategy"] is True
        and training_step_selection["selected_step"] < audit_scan["steps"]
        and bool(training_step_selection["selection_reasons"]),
    )
    add_gate(
        checks,
        "model_evaluation_samples",
        model_evaluation_summary["samples"],
        f"== {config.sample_limit}",
        model_evaluation_summary["samples"] == config.sample_limit,
    )
    add_gate(
        checks,
        "q31_reference_samples",
        reference_evaluation["samples"],
        f"== {config.sample_limit}",
        reference_evaluation["samples"] == config.sample_limit,
    )
    add_gate(
        checks,
        "model_evaluation_scan_matches_report",
        {
            "scan_correct": model_evaluation_scan["correct"],
            "report_correct": model_evaluation_summary["correct"],
        },
        "scan summary equals model evaluation summary",
        model_evaluation_scan["samples"] == model_evaluation_summary["samples"]
        and model_evaluation_scan["correct"] == model_evaluation_summary["correct"]
        and model_evaluation_scan["incorrect"] == model_evaluation_summary["incorrect"],
    )
    add_gate(
        checks,
        "evaluation_row_selection_traceable",
        metrics["evaluation_row_selection"],
        "selected evaluation row matches the configured row strategy and is tied to scan evidence",
        evaluation_row_selection["matches_selection_strategy"] is True
        and evaluation_row_selection["selected_row"] < model_evaluation_summary["samples"]
        and bool(evaluation_row_selection["selection_reasons"]),
    )
    add_gate(
        checks,
        "q31_reference_matches_native_inference",
        metrics["q31_reference_inference"],
        "reference prediction/correctness/margin/active-pixel attribution matches native inspected inference",
        metrics["q31_reference_inference"]["prediction_matches_native"] is True
        and metrics["q31_reference_inference"]["correct_matches_native"] is True
        and metrics["q31_reference_inference"]["margin_matches_native"] is True
        and metrics["q31_reference_inference"]["active_pixels_match_native"] is True
        and metrics["q31_reference_inference"]["contribution_ledger_matches_native"] is True
        and metrics["q31_reference_inference"]["margin_contribution_ledger_matches_native"] is True
        and metrics["q31_reference_inference"]["attribution_matches_logit"] is True
        and metrics["q31_reference_inference"]["attribution_matches_margin"] is True,
    )
    add_gate(
        checks,
        "q31_reference_matches_native_evaluation",
        {
            "reference_correct": reference_evaluation["correct"],
            "native_correct": model_evaluation_summary["correct"],
        },
        "reference summary equals native model evaluation summary",
        reference_evaluation["samples"] == model_evaluation_summary["samples"]
        and reference_evaluation["correct"] == model_evaluation_summary["correct"]
        and reference_evaluation["incorrect"] == model_evaluation_summary["incorrect"],
    )
    add_gate(
        checks,
        "inference_trace_profile_complete",
        metrics["inference_trace_profile"],
        "native/report/reference inference ledgers, replay verification, explanations, and sources all match",
        inference_trace_profile["kind"] == INFERENCE_TRACE_PROFILE_KIND
        and inference_trace_profile["summary"]["trace_count"] == 2
        and inference_trace_profile["summary"]["reference_checked_traces"] >= 1
        and inference_trace_profile["summary"]["all_report_verification_results_match"] is True
        and inference_trace_profile["summary"]["all_report_verification_ledgers_match"] is True
        and inference_trace_profile["summary"]["all_required_references_match"] is True
        and inference_trace_profile["summary"]["all_replay_verified"] is True
        and inference_trace_profile["summary"]["all_explanations_passed"] is True
        and inference_trace_profile["summary"]["all_sources_checked"] is True
        and inference_trace_profile["summary"]["all_passed"] is True,
    )
    add_gate(
        checks,
        "audit_contract",
        audit_contract["passed"],
        "true",
        audit_contract["passed"] is True,
    )
    add_max_number_gate(
        checks,
        "trace_to_model_payload_ratio",
        artifact_profile["trace_to_model_payload_ratio"],
        config.max_trace_model_ratio,
    )
    add_max_number_gate(
        checks,
        "witness_to_model_payload_ratio",
        artifact_profile["witness_to_model_payload_ratio"],
        config.max_witness_model_ratio,
    )
    add_gate(
        checks,
        "balanced_recompute_steps",
        {"forward": total_forward, "inverse": total_inverse},
        "forward == inverse and nonzero",
        total_forward == total_inverse and total_forward > 0,
    )
    expected_frontier_ids = {
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
    }
    add_gate(
        checks,
        "recompute_frontier_complete",
        recompute_frontier,
        "checked replay modes cover training, inference, MLP witnesses, zero-witness invertible replay, reversible preprocessing, and reversible inference tracing with balanced recompute",
        recompute_frontier["kind"] == RECOMPUTE_FRONTIER_KIND
        and recompute_frontier["summary"]["rows"] == len(expected_frontier_ids)
        and {row["id"] for row in recompute_frontier["rows"]} == expected_frontier_ids
        and recompute_frontier["summary"]["total_forward_recompute_steps"]
        == recompute_frontier["summary"]["total_inverse_recompute_steps"]
        and bool(recompute_frontier["summary"]["zero_witness_rows"])
        and bool(recompute_frontier["summary"]["witness_backed_rows"])
        and all(
            row["balanced_recompute"] is True
            and row["replay_payload_bytes"] > 0
            and row["total_recompute_steps"]
            == row["forward_recompute_steps"] + row["inverse_recompute_steps"]
            for row in recompute_frontier["rows"]
        ),
    )
    add_gate(
        checks,
        "scaling_projection_complete",
        scaling_projection,
        "checked observed/10x/100x projections cover training, batch inference, and MLP witness replay",
        scaling_projection["kind"] == SCALING_PROJECTION_KIND
        and scaling_projection["summary"]["families"] == 3
        and scaling_projection["summary"]["projection_scales"] == [1, 10, 100]
        and scaling_projection["summary"]["all_balanced"] is True
        and all(
            family["observed_count"] > 0
            and len(family["projections"]) == 3
            and [
                projection["scale_factor"]
                for projection in family["projections"]
            ]
            == [1, 10, 100]
            and family["observed_replay_payload_bytes"]
            == family["projections"][0]["projected_replay_payload_bytes"]
            and all(
                projection["projected_replay_payload_bytes"]
                == family["model_payload_bytes"]
                + projection["count"] * family["variable_replay_payload_bytes_per_unit"]
                and projection["projected_forward_recompute_steps"]
                == projection["projected_inverse_recompute_steps"]
                for projection in family["projections"]
            )
            for family in scaling_projection["families"]
        ),
    )
    add_gate(
        checks,
        "reverse_check_cost_measured",
        metrics["reverse_check_cost"],
        "finite elapsed cost, checked steps, proof recompute steps, and replay bytes",
        reverse["enabled"] is True
        and reverse_checked == train_updates
        and train_updates > 0
        and finite_positive(train_elapsed_seconds)
        and finite_positive(reverse_elapsed_seconds)
        and finite_nonnegative(train_updates_per_second)
        and finite_nonnegative(reverse_steps_per_second)
        and run_proof["forward_recompute_steps"] == reverse_checked
        and run_proof["inverse_recompute_steps"] == reverse_checked
        and run_proof["full_replay_payload_bytes"] > 0
        and run_proof["trace_replay_payload_bytes"] > 0,
    )
    add_max_number_gate(
        checks,
        "reverse_check_elapsed_ratio",
        reverse_check_cost["reverse_to_train_elapsed_ratio"],
        config.max_reverse_train_elapsed_ratio,
    )
    add_gate(
        checks,
        "training_step_replay",
        metrics["training_step_replay"],
        "proof, forward replay, and reverse replay all match",
        step_verification["proof_matches"] is True
        and step_verification["forward"]["witnesses_match"] is True
        and step_verification["forward"]["after_model_matches"] is True
        and step_verification["reverse"]["before_model_restored"] is True,
    )
    add_gate(
        checks,
        "training_step_debug_contract",
        metrics["training_step_debug"],
        "debug contract passes for selected training update",
        debug_contract["passed"] is True
        and debug_contract["claim"] == "step_backward_from_model_update",
    )
    add_gate(
        checks,
        "training_update_ledger_fingerprints",
        metrics["training_step_debug"],
        "selected update records full bias, weight, and cause ledger counts plus SHA-256 fingerprints",
        metrics["training_step_debug"]["nonzero_bias_delta_count"] > 0
        and metrics["training_step_debug"]["nonzero_weight_delta_count"] > 0
        and is_sha256(metrics["training_step_debug"]["bias_delta_ledger_fingerprint"])
        and is_sha256(metrics["training_step_debug"]["weight_delta_ledger_fingerprint"])
        and is_sha256(metrics["training_step_debug"]["cause_ledger_fingerprint"])
        and any(
            check.get("metric") == "update_ledger_fingerprints"
            and check.get("passed") is True
            for check in debug_contract["checks"]
        ),
    )
    add_gate(
        checks,
        "model_import_provenance",
        metrics["model_import"],
        "external Q31 source JSON is imported, reverified, fingerprinted, and tied to the signed model bundle",
        metrics["model_import"]["import_shape_matches"] is True
        and metrics["model_import"]["verification_shape_matches"] is True
        and metrics["model_import"]["verification_provenance_matches"] is True
        and metrics["model_import"]["import_provenance_kind"] == "external_import"
        and metrics["model_import"]["verification_provenance_kind"] == "external_import"
        and metrics["model_import"]["verification_source_checked"] is True
        and metrics["model_import"]["proof_matches"] is None
        and metrics["model_import"]["training_steps"] is None
        and is_sha256(metrics["model_import"]["source_model_json_fingerprint"])
        and is_sha256(metrics["model_import"]["source_model_json_file_sha256"])
        and is_sha256(metrics["model_import"]["import_payload_fingerprint"])
        and metrics["model_import"]["import_payload_fingerprint"]
        == metrics["model_import"]["verification_payload_fingerprint"],
    )
    add_gate(
        checks,
        "imported_model_inference_replay",
        metrics["imported_model_inference"],
        "imported model inference replays forward/backward and matches native prediction, margin, and attribution ledgers",
        metrics["imported_model_inference"]["proof_matches"] is True
        and metrics["imported_model_inference"]["result_matches"] is True
        and metrics["imported_model_inference"]["restored_initial_state"] is True
        and metrics["imported_model_inference"]["source_model_checked"] is True
        and metrics["imported_model_inference"]["source_sample_checked"] is True
        and metrics["imported_model_inference"]["prediction_matches_native"] is True
        and metrics["imported_model_inference"]["correct_matches_native"] is True
        and metrics["imported_model_inference"]["margin_matches_native"] is True
        and metrics["imported_model_inference"]["contribution_ledger_matches_native"] is True
        and metrics["imported_model_inference"][
            "margin_contribution_ledger_matches_native"
        ]
        is True
        and metrics["imported_model_inference"]["replay_payload_bytes"] > 0,
    )
    add_gate(
        checks,
        "native_inference_replay",
        metrics["native_inference_replay"],
        "proof/result/reverse/source checks pass",
        native_inference_verification["proof_matches"] is True
        and native_inference_verification["result_matches"] is True
        and native_inference_verification["restored_initial_state"] is True
        and native_inference_verification["source_model_checked"] is True
        and native_inference_verification["source_sample_checked"] is True,
    )
    add_gate(
        checks,
        "native_inference_explanation_contract",
        metrics["native_inference_explanation"],
        "prediction explanation, attribution, reverse replay, proof, and sources pass",
        native_inference_verification["explanation_contract"]["passed"] is True
        and native_inference_verification["explanation_contract"]["claim"]
        == "q31_inference_prediction_explanation",
    )
    add_gate(
        checks,
        "native_standalone_rev_classifier",
        metrics["native_standalone_rev_classifier"],
        "generated .rev source runs with generic Reverie, matches native inference, and roundtrips",
        metrics["native_standalone_rev_classifier"]["run_kind"] == "reverie_run_result"
        and metrics["native_standalone_rev_classifier"]["contains_vecmat_q31"] is True
        and metrics["native_standalone_rev_classifier"]["contains_prediction_assert"] is True
        and metrics["native_standalone_rev_classifier"]["contains_correct_assert"] is True
        and metrics["native_standalone_rev_classifier"]["run_uses_vecmat_q31"] is True
        and metrics["native_standalone_rev_classifier"]["run_prediction"]
        == metrics["native_standalone_rev_classifier"]["expected_prediction"]
        and metrics["native_standalone_rev_classifier"]["run_correct"]
        == metrics["native_standalone_rev_classifier"]["expected_correct"]
        and metrics["native_standalone_rev_classifier"]["matches_native_prediction"] is True
        and metrics["native_standalone_rev_classifier"]["matches_native_correct"] is True
        and metrics["native_standalone_rev_classifier"]["matches_native_label"] is True
        and metrics["native_standalone_rev_classifier"]["roundtrip_kind"]
        == "reverie_roundtrip_result"
        and metrics["native_standalone_rev_classifier"]["roundtrip_passed"] is True
        and metrics["native_standalone_rev_classifier"]["roundtrip_prediction"]
        == metrics["native_standalone_rev_classifier"]["run_prediction"]
        and metrics["native_standalone_rev_classifier"]["roundtrip_correct"]
        == metrics["native_standalone_rev_classifier"]["run_correct"]
        and metrics["native_standalone_rev_classifier"]["roundtrip_restored_prediction"] == 0
        and metrics["native_standalone_rev_classifier"]["roundtrip_restored_correct"] == 0
        and is_sha256(metrics["native_standalone_rev_classifier"]["roundtrip_fingerprint"])
        and metrics["native_standalone_rev_classifier"]["verification_kind"]
        == "reverie_roundtrip_verification"
        and metrics["native_standalone_rev_classifier"]["verification_passed"] is True
        and metrics["native_standalone_rev_classifier"]["verification_source_hash_matches"] is True
        and metrics["native_standalone_rev_classifier"]["verification_replay_restoration_passed"]
        is True
        and metrics["native_standalone_rev_classifier"]["verification_replayed"] is True,
    )
    add_gate(
        checks,
        "model_evaluation_replay",
        metrics["model_evaluation_replay"],
        "rows/proof/reverse/source checks pass",
        model_evaluation_verification["rows_match"] is True
        and model_evaluation_verification["proof_matches"] is True
        and model_evaluation_verification["restored_initial_state"] is True
        and model_evaluation_verification["source_model_checked"] is True
        and model_evaluation_verification["source_samples_checked"] is True,
    )
    add_gate(
        checks,
        "evaluation_row_inference_replay",
        metrics["evaluation_row_inference"],
        "proof/result/reverse/model/evaluation-source checks pass",
        row_inference_verification["proof_matches"] is True
        and row_inference_verification["result_matches"] is True
        and row_inference_verification["restored_initial_state"] is True
        and row_inference_verification["source_model_checked"] is True
        and row_inference_verification["source_evaluation_checked"] is True,
    )
    add_gate(
        checks,
        "evaluation_row_inference_explanation_contract",
        metrics["evaluation_row_inference_explanation"],
        "evaluation-row prediction explanation, attribution, reverse replay, proof, and sources pass",
        row_inference_verification["explanation_contract"]["passed"] is True
        and row_inference_verification["explanation_contract"]["claim"]
        == "q31_inference_prediction_explanation",
    )
    add_gate(
        checks,
        "mlp_witness_replay",
        metrics["mlp_witness"],
        "witness checker passes, all samples correct, witness proof is fingerprinted, and recompute steps are balanced",
        mlp_witness["passed"] is True
        and all(bool(item) for item in mlp_witness["correct"])
        and is_sha256(metrics["mlp_witness"]["witness_proof_fingerprint"])
        and mlp_proof["forward_recompute_steps"] == mlp_proof["inverse_recompute_steps"]
        and mlp_proof["total_recompute_steps"]
        == mlp_proof["forward_recompute_steps"] + mlp_proof["inverse_recompute_steps"],
    )
    add_gate(
        checks,
        "invertible_coupling_replay",
        metrics["invertible_coupling"],
        "forward/reverse checker passes with zero witness and trace bytes",
        coupling["passed"] is True
        and coupling_proof["witness_payload_bytes"] == 0
        and coupling_proof["trace_payload_bytes"] == 0
        and coupling_proof["forward_recompute_steps"] == coupling_proof["inverse_recompute_steps"]
        and coupling_proof["total_recompute_steps"]
        == coupling_proof["forward_recompute_steps"] + coupling_proof["inverse_recompute_steps"],
    )
    add_gate(
        checks,
        "triangular_residual_replay",
        metrics["triangular_residual"],
        "forward/reverse checker passes with triangular source order and zero witness/trace bytes",
        residual["passed"] is True
        and residual_proof["witness_payload_bytes"] == 0
        and residual_proof["trace_payload_bytes"] == 0
        and residual_proof["checks"]["triangular_source_order"] is True
        and residual_proof["forward_recompute_steps"] == residual_proof["inverse_recompute_steps"]
        and residual_proof["total_recompute_steps"]
        == residual_proof["forward_recompute_steps"] + residual_proof["inverse_recompute_steps"],
    )
    add_gate(
        checks,
        "reversible_preprocess_replay",
        metrics["reversible_preprocess"],
        "forward/reverse checker passes with preserved raw/mean state and zero witness/trace bytes",
        preprocess["passed"] is True
        and preprocess_proof["witness_payload_bytes"] == 0
        and preprocess_proof["trace_payload_bytes"] == 0
        and preprocess_proof["checks"]["raw_preserved"] is True
        and preprocess_proof["checks"]["mean_preserved"] is True
        and preprocess_proof["forward_recompute_steps"] == preprocess_proof["inverse_recompute_steps"]
        and preprocess_proof["total_recompute_steps"]
        == preprocess_proof["forward_recompute_steps"] + preprocess_proof["inverse_recompute_steps"],
    )
    add_gate(
        checks,
        "reversible_inference_trace_replay",
        metrics["reversible_inference_trace"],
        "forward/reverse checker passes with logits, prediction, correctness, and bounded witness bytes",
        inference_trace["passed"] is True
        and inference_trace_proof["witness_payload_bytes"] > 0
        and inference_trace_proof["trace_payload_bytes"] == 0
        and inference_trace_proof["checks"]["preprocess_matches_reference"] is True
        and inference_trace_proof["checks"]["logits_match_reference"] is True
        and inference_trace_proof["checks"]["prediction_matches_reference"] is True
        and inference_trace_proof["checks"]["correctness_matches_reference"] is True
        and inference_trace_proof["checks"]["reverse_restores_initial_state"] is True
        and inference_trace_proof["checks"]["raw_preserved"] is True
        and inference_trace_proof["checks"]["model_preserved"] is True
        and inference_trace_proof["checks"]["balanced_recompute"] is True
        and inference_trace["attribution"]["matches_logit"] is True
        and inference_trace["attribution"]["matches_margin"] is True
        and inference_trace["attribution"]["contribution_count"] > 0
        and inference_trace["attribution"]["margin_contribution_count"] > 0
        and inference_trace_proof["forward_recompute_steps"]
        == inference_trace_proof["inverse_recompute_steps"]
        and inference_trace_proof["total_recompute_steps"]
        == inference_trace_proof["forward_recompute_steps"]
        + inference_trace_proof["inverse_recompute_steps"],
    )
    if config.max_replay_payload_bytes is not None:
        add_gate(
            checks,
            "max_replay_payload_bytes",
            max_replay_payload_bytes,
            f"<= {config.max_replay_payload_bytes}",
            max_replay_payload_bytes <= config.max_replay_payload_bytes,
        )

    failed_before_scorecard = [
        check["metric"]
        for check in checks
        if not check["passed"]
    ]
    scorecard = {
        "kind": "reverie_mnist_ml_v6_scorecard",
        "train_updates_per_second": train["updates_per_second"],
        "built_in_eval_samples_per_second": built_in_eval["samples_per_second"],
        "model_evaluation_samples_per_second": model_evaluation_summary["samples_per_second"],
        "reverse_steps_per_second": reverse["steps_per_second"],
        "reverse_to_train_elapsed_ratio": reverse_check_cost["reverse_to_train_elapsed_ratio"],
        "max_reverse_train_elapsed_ratio": config.max_reverse_train_elapsed_ratio,
        "run_peak_rss_bytes": run_memory.get("peak_rss_bytes"),
        "estimated_payload_bytes": run_memory["estimated_payload_bytes"],
        "total_model_payload_bytes": artifact_profile["total_model_payload_bytes"],
        "total_witness_payload_bytes": artifact_profile["total_witness_payload_bytes"],
        "total_trace_payload_bytes": artifact_profile["total_trace_payload_bytes"],
        "trace_to_model_payload_ratio": artifact_profile["trace_to_model_payload_ratio"],
        "witness_to_model_payload_ratio": artifact_profile["witness_to_model_payload_ratio"],
        "max_replay_payload_bytes": max_replay_payload_bytes,
        "total_recompute_steps": artifact_profile["total_recompute_steps"],
        "total_forward_recompute_steps": total_forward,
        "total_inverse_recompute_steps": total_inverse,
        "balanced_recompute": total_forward == total_inverse and total_forward > 0,
        "contracts": {
            "training_step_debug": debug_contract["passed"],
            "model_import": gates_passed(
                checks,
                ("model_import_provenance", "imported_model_inference_replay"),
            ),
            "native_inference_explanation": native_inference_verification["explanation_contract"]["passed"],
            "evaluation_row_inference_explanation": row_inference_verification["explanation_contract"]["passed"],
            "q31_reference_inference": gates_passed(checks, ("q31_reference_matches_native_inference",)),
            "inference_trace_profile": gates_passed(checks, ("inference_trace_profile_complete",)),
            "mlp_witness": mlp_witness["passed"],
            "invertible_coupling": coupling["passed"],
            "triangular_residual": residual["passed"],
            "reversible_preprocess": preprocess["passed"],
            "reversible_inference_trace": inference_trace["passed"],
        },
        "base_gates_passed": not failed_before_scorecard,
        "base_failed_gates": failed_before_scorecard,
    }
    metrics["scorecard"] = scorecard
    add_gate(
        checks,
        "v6_scorecard_complete",
        scorecard,
        "speed/memory/trace/replay/reverse metrics are present, contracts pass, and prior gates pass",
        not failed_before_scorecard
        and finite_positive(scorecard["train_updates_per_second"])
        and finite_positive(scorecard["built_in_eval_samples_per_second"])
        and finite_positive(scorecard["model_evaluation_samples_per_second"])
        and finite_positive(scorecard["reverse_steps_per_second"])
        and finite_nonnegative(scorecard["reverse_to_train_elapsed_ratio"])
        and scorecard["reverse_to_train_elapsed_ratio"] <= config.max_reverse_train_elapsed_ratio
        and isinstance(scorecard["run_peak_rss_bytes"], int)
        and scorecard["run_peak_rss_bytes"] > 0
        and scorecard["balanced_recompute"] is True
        and scorecard["max_replay_payload_bytes"] > 0
        and scorecard_contracts_passed(scorecard),
    )
    scorecard["gates_passed"] = all(check["passed"] for check in checks)
    scorecard["failed_gates"] = [
        check["metric"]
        for check in checks
        if not check["passed"]
    ]
    ml_capability_map = pipeline_ml_capability_map(checks)
    metrics["ml_capability_map"] = ml_capability_map
    add_gate(
        checks,
        "ml_roadmap_capability_map_complete",
        ml_capability_map,
        "V1-V6 capability rows are present, pass, and reference checked evidence",
        ml_capability_map["passed"] is True
        and ml_capability_map["summary"]["total"] == len(ML_CAPABILITY_ROWS)
        and ml_capability_map["summary"]["failed"] == 0,
    )
    ml_goal_readiness = pipeline_ml_goal_readiness(checks)
    metrics["ml_goal_readiness"] = ml_goal_readiness
    add_gate(
        checks,
        "ml_goal_readiness_complete",
        ml_goal_readiness,
        "north-star reversible ML goals are present, pass, and reference checked evidence",
        ml_goal_readiness["passed"] is True
        and ml_goal_readiness["summary"]["total"] == len(ML_GOAL_ROWS)
        and ml_goal_readiness["summary"]["failed"] == 0,
    )
    claims = pipeline_claims(checks)
    return {
        "kind": SUMMARY_KIND,
        "pipeline_kind": PIPELINE_KIND,
        "sample_limit": config.sample_limit,
        "requested_audit_step": (
            config.audit_step if config.requested_audit_step is None else config.requested_audit_step
        ),
        "audit_step": config.audit_step,
        "audit_step_strategy": config.audit_step_strategy,
        "requested_evaluation_row": (
            config.evaluation_row
            if config.requested_evaluation_row is None
            else config.requested_evaluation_row
        ),
        "evaluation_row": config.evaluation_row,
        "evaluation_row_strategy": config.evaluation_row_strategy,
        "profile_markdown": str(paths["profile_markdown"]),
        "reports": {key: str(paths[key]) for key in PROFILE_REPORT_KEYS},
        "bundles": {key: str(path) for key, path in bundle_paths(paths).items()},
        "metrics": metrics,
        "gates": {
            "passed": all(check["passed"] for check in checks),
            "policy": gate_policy(config),
            "checks": checks,
        },
        "evidence": evidence_index(paths, include_summary=False),
        "claims": claims,
    }


def write_pipeline_summary(config: PipelineConfig) -> dict[str, Any]:
    paths = out_paths(config.out_dir)
    reports = load_checked_reports(paths)
    summary = build_pipeline_summary(config, paths, reports)
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    failed = [
        check["metric"]
        for check in summary["gates"]["checks"]
        if not check["passed"]
    ]
    if failed:
        raise PipelineError(
            "pipeline summary gates failed: {} (summary: {})".format(
                ", ".join(failed),
                paths["summary"],
            )
        )
    return summary


def inference_action_contract(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    q31 = metrics["q31_reference_inference"]
    imported = metrics["imported_model_inference"]
    native = metrics["native_inference_replay"]
    standalone = metrics["native_standalone_rev_classifier"]
    trace = metrics["reversible_inference_trace"]
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


def build_model_capsule(config: PipelineConfig, summary: dict[str, Any]) -> dict[str, Any]:
    paths = out_paths(config.out_dir)
    evidence = evidence_index(paths, include_summary=True)
    metrics = summary["metrics"]
    gate_checks = summary["gates"]["checks"]
    passed_count = sum(1 for check in gate_checks if check["passed"])
    failed_metrics = [
        check["metric"]
        for check in gate_checks
        if not check["passed"]
    ]
    evidence_files = evidence["files"]
    payload = {
        "schema": MODEL_CAPSULE_SCHEMA,
        "pipeline_kind": PIPELINE_KIND,
        "pipeline_summary": str(paths["summary"]),
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
            "passed": summary["gates"]["passed"],
            "total": len(gate_checks),
            "passed_count": passed_count,
            "failed_metrics": failed_metrics,
            "policy": summary["gates"]["policy"],
        },
        "model": {
            "bundle": summary["bundles"]["model"],
            "sha256": evidence_files["bundle.model"]["sha256"],
            "payload_bytes": metrics["artifact_profile"]["total_model_payload_bytes"],
        },
        "imported_model": {
            "source_json": summary["bundles"]["imported_model_source"],
            "source_json_sha256": evidence_files["bundle.imported_model_source"][
                "sha256"
            ],
            "bundle": summary["bundles"]["imported_model"],
            "bundle_sha256": evidence_files["bundle.imported_model"]["sha256"],
            "source_model_json_checked": metrics["model_import"][
                "verification_source_checked"
            ],
            "provenance_kind": metrics["model_import"][
                "verification_provenance_kind"
            ],
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
            "imported_model_replay_passed": metrics["imported_model_inference"][
                "proof_matches"
            ]
            and metrics["imported_model_inference"]["result_matches"],
            "imported_model_prediction": metrics["imported_model_inference"][
                "prediction"
            ],
            "imported_model_matches_native": (
                metrics["imported_model_inference"]["prediction_matches_native"]
                and metrics["imported_model_inference"]["correct_matches_native"]
                and metrics["imported_model_inference"]["margin_matches_native"]
                and metrics["imported_model_inference"][
                    "contribution_ledger_matches_native"
                ]
                and metrics["imported_model_inference"][
                    "margin_contribution_ledger_matches_native"
                ]
            ),
            "native_replay_passed": metrics["native_inference_replay"]["proof_matches"]
            and metrics["native_inference_replay"]["result_matches"],
            "native_prediction": metrics["native_inference_explanation"]["prediction"],
            "native_correct": metrics["native_inference_explanation"]["correct"],
            "native_margin": metrics["native_inference_explanation"]["margin"],
            "evaluation_row_replay_passed": metrics["evaluation_row_inference"]["proof_matches"]
            and metrics["evaluation_row_inference"]["result_matches"],
            "evaluation_row_prediction": metrics["evaluation_row_inference_explanation"]["prediction"],
            "evaluation_row_correct": metrics["evaluation_row_inference_explanation"]["correct"],
            "evaluation_row_margin": metrics["evaluation_row_inference_explanation"]["margin"],
            "q31_reference_prediction": metrics["q31_reference_inference"]["prediction"],
            "q31_reference_matches_native": (
                metrics["q31_reference_inference"]["prediction_matches_native"]
                and metrics["q31_reference_inference"]["correct_matches_native"]
                and metrics["q31_reference_inference"]["margin_matches_native"]
                and metrics["q31_reference_inference"]["contribution_ledger_matches_native"]
                and metrics["q31_reference_inference"]["margin_contribution_ledger_matches_native"]
            ),
            "action_contract": inference_action_contract(metrics),
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


def write_model_capsule(config: PipelineConfig, summary: dict[str, Any]) -> dict[str, Any]:
    paths = out_paths(config.out_dir)
    capsule = build_model_capsule(config, summary)
    paths["model_capsule"].write_text(json.dumps(capsule, indent=2) + "\n", encoding="utf-8")
    return capsule


def write_manifest(
    config: PipelineConfig,
    steps: list[PipelineStep],
    summary: dict[str, Any],
    capsule: dict[str, Any],
) -> None:
    paths = out_paths(config.out_dir)
    manifest = {
        "kind": PIPELINE_KIND,
        "sample_limit": config.sample_limit,
        "requested_audit_step": (
            config.audit_step if config.requested_audit_step is None else config.requested_audit_step
        ),
        "audit_step": config.audit_step,
        "audit_step_strategy": config.audit_step_strategy,
        "requested_evaluation_row": (
            config.evaluation_row
            if config.requested_evaluation_row is None
            else config.requested_evaluation_row
        ),
        "evaluation_row": config.evaluation_row,
        "evaluation_row_strategy": config.evaluation_row_strategy,
        "profile_markdown": str(paths["profile_markdown"]),
        "summary": str(paths["summary"]),
        "model_capsule": str(paths["model_capsule"]),
        "model_capsule_fingerprint": capsule["fingerprint"],
        "gates_passed": summary["gates"]["passed"],
        "gate_policy": summary["gates"]["policy"],
        "reports": [str(path) for path in profile_reports(paths)],
        "bundles": [str(path) for path in bundle_paths(paths).values()],
        "evidence": evidence_index(paths, include_summary=True),
        "steps": [
            {
                "label": step.label,
                "output": None if step.output is None else str(step.output),
                "command": step.command,
            }
            for step in steps
        ],
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def validate_written_pipeline_artifacts(config: PipelineConfig) -> None:
    paths = out_paths(config.out_dir)
    for key in ("summary", "model_capsule", "manifest"):
        path = paths[key]
        try:
            report = profile_check.load_json(path)
            profile_check.validate_report(
                report,
                require_reverse_check=False,
                require_peak_rss=False,
                require_ml_profile=False,
                require_audit_contract=False,
            )
            profile_check.verify_pipeline_file_evidence(report)
        except ValueError as error:
            raise PipelineError(f"{path} failed pipeline artifact validation: {error}") from error
    manifest = profile_check.load_json(paths["manifest"])
    capsule = profile_check.load_json(paths["model_capsule"])
    if manifest["model_capsule_fingerprint"] != capsule["fingerprint"]:
        raise PipelineError("pipeline manifest model capsule fingerprint does not match capsule")


def write_model_capsule_profile(config: PipelineConfig) -> None:
    paths = out_paths(config.out_dir)
    reports = [
        (paths["summary"], profile_check.load_json(paths["summary"])),
        (paths["model_capsule"], profile_check.load_json(paths["model_capsule"])),
        (paths["manifest"], profile_check.load_json(paths["manifest"])),
    ]
    profile_check.validate_report_set_consistency(reports)
    markdown = profile_summary.render_markdown(reports)
    paths["model_capsule_profile"].write_text(markdown, encoding="utf-8")


def write_training_step_debug_markdown(config: PipelineConfig) -> None:
    paths = out_paths(config.out_dir)
    audit_step = profile_check.load_json(paths["audit_step"])
    step_verification = profile_check.load_json(paths["step_verification"])
    profile_check.validate_report(
        audit_step,
        require_reverse_check=False,
        require_peak_rss=False,
        require_ml_profile=False,
        require_audit_contract=False,
    )
    profile_check.validate_report(
        step_verification,
        require_reverse_check=False,
        require_peak_rss=False,
        require_ml_profile=False,
        require_audit_contract=False,
    )
    markdown = profile_summary.render_training_step_debug_markdown(
        audit_step,
        step_verification,
    )
    paths["training_step_debug_markdown"].write_text(markdown, encoding="utf-8")


def write_model_capsule_verification(config: PipelineConfig) -> dict[str, Any]:
    paths = out_paths(config.out_dir)
    result = capsule_verify.verify_capsule(
        {
            "summary": paths["summary"],
            "capsule": paths["model_capsule"],
            "manifest": paths["manifest"],
            "profile": paths["model_capsule_profile"],
            "verification_markdown": paths["model_capsule_verification_markdown"],
        },
        verify_file_evidence=True,
        allow_missing_profile=False,
    )
    report = capsule_verify.verification_report(
        result,
        verify_file_evidence=True,
        allow_missing_profile=False,
        verification_markdown_path=paths["model_capsule_verification_markdown"],
        check_verification_markdown=False,
    )
    paths["model_capsule_verification_markdown"].write_text(
        capsule_verify.render_verification_markdown(report),
        encoding="utf-8",
    )
    report = capsule_verify.verification_report(
        result,
        verify_file_evidence=True,
        allow_missing_profile=False,
        verification_markdown_path=paths["model_capsule_verification_markdown"],
        require_verification_markdown=True,
    )
    paths["model_capsule_verification"].write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def handoff_short_hash(value: Any) -> str:
    return value[:12] if isinstance(value, str) else "n/a"


def handoff_comma(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else str(value)


def handoff_ratio(value: Any) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "n/a"


def handoff_shell_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def inference_action_review_commands(
    config: PipelineConfig,
    paths: dict[str, Path],
    q31_inference: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_correct = "true" if q31_inference["correct"] else "false"
    expected_logits = json.dumps(q31_inference["logits"], separators=(",", ":"))
    sample_index = str(q31_inference["sample_source"]["sample_index"])
    q31_command = [
        "python3",
        "scripts/check_q31_inference.py",
        "--model",
        str(paths["model_bundle"]),
        "--sample",
        str(paths["samples"]),
        "--sample-index",
        sample_index,
        "--expect-prediction",
        str(q31_inference["prediction"]),
        "--expect-correct",
        expected_correct,
        "--expect-logits",
        expected_logits,
        "--markdown-output",
        "/tmp/reverie-q31-reference-inference.md",
        "--json",
    ]
    return [
        {
            "operation": "reproduce_prediction",
            "label": "Reproduce selected Q31 prediction",
            "purpose": "Recomputes the selected sample prediction and exact logits from the signed model and sample set.",
            "command": q31_command,
            "artifacts": [
                "model_bundle",
                "samples",
                "q31_reference_inference",
                "q31_reference_inference_markdown",
            ],
        },
        {
            "operation": "explain_margin",
            "label": "Recompute selected Q31 margin explanation",
            "purpose": "Regenerates the attribution Markdown that explains the selected prediction margin.",
            "command": q31_command,
            "artifacts": [
                "model_bundle",
                "samples",
                "q31_reference_inference",
                "q31_reference_inference_markdown",
                "native_inference_verification_markdown",
            ],
        },
        {
            "operation": "replay_imported_model_inference",
            "label": "Replay imported model inference bundle",
            "purpose": "Reruns the saved external-import model inference bundle forward and backward.",
            "command": [
                *runner_prefix(config),
                "--verify-inference",
                str(paths["imported_model_inference_bundle"]),
                "--json",
            ],
            "artifacts": [
                "imported_model_source",
                "imported_model_bundle",
                "imported_model_inference_bundle",
            ],
        },
        {
            "operation": "replay_native_inference",
            "label": "Replay native inference bundle",
            "purpose": "Reruns the saved native inference bundle forward and backward and refreshes its verification card.",
            "command": [
                *runner_prefix(config),
                "--verify-inference",
                str(paths["native_inference_bundle"]),
                "--markdown-output",
                "/tmp/reverie-native-inference-verification.md",
                "--json",
            ],
            "artifacts": [
                "native_inference_bundle",
                "native_inference_verification_markdown",
            ],
        },
        {
            "operation": "run_standalone_rev_classifier",
            "label": "Run standalone Reverie classifier",
            "purpose": "Runs the generated source-only .rev classifier with generic Reverie and checks its embedded prediction assertions.",
            "command": [
                *reverie_prefix(config),
                "run",
                str(paths["native_standalone_rev_classifier"]),
                "--json",
            ],
            "artifacts": [
                "native_standalone_rev_classifier",
                "native_standalone_rev_run",
            ],
        },
        {
            "operation": "reverse_reversible_trace",
            "label": "Replay reversible inference trace",
            "purpose": "Rechecks the saved Reverie inference trace forward and backward, including reverse restoration.",
            "command": [
                "python3",
                "scripts/check_reversible_inference_trace.py",
                "--forward-output-json",
                str(paths["inference_trace_forward"]),
                "--reverse-output-json",
                str(paths["inference_trace_reverse"]),
                "--expect-report-json",
                str(paths["reversible_inference_trace"]),
                "--markdown-output",
                "/tmp/reverie-reversible-inference-trace.md",
            ],
            "artifacts": [
                "inference_trace_forward",
                "inference_trace_reverse",
                "reversible_inference_trace",
                "reversible_inference_trace_markdown",
            ],
        },
    ]


def build_handoff_index(config: PipelineConfig) -> dict[str, Any]:
    paths = out_paths(config.out_dir)
    summary = profile_check.load_json(paths["summary"])
    capsule = profile_check.load_json(paths["model_capsule"])
    verification = profile_check.load_json(paths["model_capsule_verification"])
    audit_step = profile_check.load_json(paths["audit_step"])
    q31_inference = profile_check.load_json(paths["q31_reference_inference"])
    reversible_trace = capsule["payload"]["reversible_inference_trace"]
    profile_check.validate_report_set_consistency(
        [
            (paths["summary"], summary),
            (paths["model_capsule"], capsule),
            (paths["manifest"], profile_check.load_json(paths["manifest"])),
        ]
    )
    metrics = summary["metrics"]
    payload = {
        "schema": HANDOFF_SCHEMA,
        "north_star": capsule["payload"]["readiness"]["north_star"],
        "non_goal": capsule["payload"]["readiness"]["non_goal"],
        "capsule": {
            "fingerprint": capsule["fingerprint"],
            "gates": capsule["payload"]["gates"],
            "model_sha256": capsule["payload"]["model"]["sha256"],
            "samples": capsule["payload"]["samples"],
        },
        "trust_certificate": {
            "fingerprint": verification["trust_certificate"]["fingerprint"],
            "passed": verification["trust_certificate"]["payload"]["passed"],
            "verification_markdown": verification["verification_markdown"],
        },
        "selected_training_step": {
            "step": audit_step["step"],
            "sample_index": audit_step["sample_index"],
            "prediction": audit_step["prediction"],
            "label": audit_step["label"],
            "correct": audit_step["correct"],
            "margin": audit_step["logit_margin"]["margin"],
            "active_pixels": len(audit_step["active_pixels"]),
            "debug_contract_passed": audit_step["debug_contract"]["passed"],
            "cause_ledger_fingerprint": audit_step["cause_ledger"]["fingerprint"],
            "bias_delta_ledger_fingerprint": audit_step["update"]["bias_delta_ledger_fingerprint"],
            "weight_delta_ledger_fingerprint": audit_step["update"]["weight_delta_ledger_fingerprint"],
        },
        "selected_inference": {
            "imported_model_prediction": metrics["imported_model_inference"][
                "prediction"
            ],
            "imported_model_replay_passed": metrics["imported_model_inference"][
                "proof_matches"
            ]
            and metrics["imported_model_inference"]["result_matches"],
            "imported_model_matches_native": metrics["imported_model_inference"][
                "prediction_matches_native"
            ]
            and metrics["imported_model_inference"]["margin_matches_native"],
            "prediction": q31_inference["prediction"],
            "label": q31_inference["label"],
            "correct": q31_inference["correct"],
            "margin": q31_inference["attribution"]["margin"],
            "runner_up_digit": q31_inference["attribution"]["runner_up_digit"],
            "active_pixels": q31_inference["active_pixels"],
            "contribution_ledger_fingerprint": (
                q31_inference["attribution"]["contribution_ledger_fingerprint"]
            ),
            "margin_contribution_ledger_fingerprint": (
                q31_inference["attribution"]["margin_contribution_ledger_fingerprint"]
            ),
            "trace_prediction": reversible_trace["prediction"],
            "trace_correct": reversible_trace["correct"],
            "trace_runner_up_class": reversible_trace["runner_up_class"],
            "trace_margin": reversible_trace["margin"],
            "trace_label_rank": reversible_trace["label_rank"],
            "trace_top2_correct": reversible_trace["top2_correct"],
            "trace_top_classes": reversible_trace["top_classes"],
            "trace_top_logit_values": reversible_trace["top_logit_values"],
            "trace_witness_payload_bytes": reversible_trace["witness_payload_bytes"],
            "trace_replay_payload_bytes": reversible_trace["replay_payload_bytes"],
            "trace_total_recompute_steps": reversible_trace["total_recompute_steps"],
        },
        "scorecard": {
            "train_updates_per_second": metrics["scorecard"]["train_updates_per_second"],
            "model_evaluation_samples_per_second": (
                metrics["scorecard"]["model_evaluation_samples_per_second"]
            ),
            "reverse_to_train_elapsed_ratio": metrics["scorecard"]["reverse_to_train_elapsed_ratio"],
            "run_peak_rss_bytes": metrics["scorecard"]["run_peak_rss_bytes"],
            "max_replay_payload_bytes": metrics["scorecard"]["max_replay_payload_bytes"],
            "trace_to_model_payload_ratio": metrics["scorecard"]["trace_to_model_payload_ratio"],
            "witness_to_model_payload_ratio": metrics["scorecard"]["witness_to_model_payload_ratio"],
            "total_recompute_steps": metrics["scorecard"]["total_recompute_steps"],
        },
        "artifacts": handoff_artifacts(paths),
        "inference_action_review_commands": inference_action_review_commands(
            config,
            paths,
            q31_inference,
        ),
    }
    return {
        "kind": HANDOFF_KIND,
        "algorithm": "sha256",
        "fingerprint": sha256_json(payload),
        "payload": payload,
    }


def render_handoff_markdown(report: dict[str, Any]) -> str:
    payload = report["payload"]
    capsule = payload["capsule"]
    certificate = payload["trust_certificate"]
    training = payload["selected_training_step"]
    inference = payload["selected_inference"]
    scorecard = payload["scorecard"]
    artifacts = payload["artifacts"]
    commands = payload["inference_action_review_commands"]
    lines = [
        "# Reverie ML Audit Handoff",
        "",
        "| Verdict | Handoff | Capsule | Certificate | Gates | Model |",
        "| --- | --- | --- | --- | ---: | --- |",
        "| {} | `{}` | `{}` | `{}` | {}/{} | `{}` |".format(
            "pass" if certificate["passed"] else "fail",
            handoff_short_hash(report["fingerprint"]),
            handoff_short_hash(capsule["fingerprint"]),
            handoff_short_hash(certificate["fingerprint"]),
            capsule["gates"]["passed_count"],
            capsule["gates"]["total"],
            handoff_short_hash(capsule["model_sha256"]),
        ),
        "",
        "## North Star",
        "",
        f"- Goal: `{payload['north_star']}`",
        f"- Non-goal: `{payload['non_goal']}`",
        "",
        "## Review Cards",
        "",
        "| Card | Role | SHA-256 | Bytes | Path |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for artifact_id in (
        "training_audit_verification_markdown",
        "training_step_debug",
        "q31_reference_inference_markdown",
        "native_inference_audit_markdown",
        "native_inference_verification_markdown",
        "evaluation_row_inference_audit_markdown",
        "mlp_witness_markdown",
        "invertible_coupling_markdown",
        "triangular_residual_markdown",
        "reversible_preprocess_markdown",
        "reversible_inference_trace_markdown",
        "reversible_inference_roundtrip_markdown",
        "model_capsule_verification_markdown",
        "model_capsule_profile",
        "ml_profile",
    ):
        artifact = artifacts[artifact_id]
        lines.append(
            "| {} | {} | `{}` | {} | `{}` |".format(
                artifact["label"],
                artifact["role"],
                handoff_short_hash(artifact["sha256"]),
                handoff_comma(artifact["bytes"]),
                artifact["path"],
            )
        )
    lines.extend(
        [
            "",
            "## Selected Training Update",
            "",
            "| Step | Sample | Label | Prediction | Correct | Margin | Active pixels | Debug | Cause ledger | Weight ledger |",
            "| ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |",
            "| {} | {} | {} | {} | {} | {} | {} | {} | `{}` | `{}` |".format(
                training["step"],
                training["sample_index"],
                training["label"],
                training["prediction"],
                str(training["correct"]).lower(),
                handoff_comma(training["margin"]),
                training["active_pixels"],
                "passed" if training["debug_contract_passed"] else "failed",
                handoff_short_hash(training["cause_ledger_fingerprint"]),
                handoff_short_hash(training["weight_delta_ledger_fingerprint"]),
            ),
            "",
            "## Selected Inference",
            "",
            "| Prediction | Imported pred | Imported replay | Imported/native | Label | Correct | Runner-up | Margin | Active pixels | Contribution ledger | Margin ledger |",
            "| ---: | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | `{}` | `{}` |".format(
                inference["prediction"],
                inference["imported_model_prediction"],
                str(inference["imported_model_replay_passed"]).lower(),
                str(inference["imported_model_matches_native"]).lower(),
                inference["label"],
                str(inference["correct"]).lower(),
                inference["runner_up_digit"],
                handoff_comma(inference["margin"]),
                inference["active_pixels"],
                handoff_short_hash(inference["contribution_ledger_fingerprint"]),
                handoff_short_hash(inference["margin_contribution_ledger_fingerprint"]),
            ),
            "",
            "| Reversible trace pred | Label rank | Correct | Top-k correct | Runner-up | Top classes | Top logits | Margin | Witness | Replay | Recompute |",
            "| ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
            "| {} | {} | {} | {} | {} | `{}` | `{}` | {} | {} | {} | {} |".format(
                inference["trace_prediction"],
                inference["trace_label_rank"],
                str(inference["trace_correct"]).lower(),
                inference["trace_top2_correct"],
                inference["trace_runner_up_class"],
                ",".join(str(value) for value in inference["trace_top_classes"]),
                ",".join(handoff_comma(value) for value in inference["trace_top_logit_values"]),
                handoff_comma(inference["trace_margin"]),
                handoff_comma(inference["trace_witness_payload_bytes"]),
                handoff_comma(inference["trace_replay_payload_bytes"]),
                handoff_comma(inference["trace_total_recompute_steps"]),
            ),
            "",
            "## Inference Action Review Commands",
            "",
            "| Operation | Command | Artifacts |",
            "| --- | --- | --- |",
        ]
    )
    for command in commands:
        lines.append(
            "| `{}` | `{}` | {} |".format(
                command["operation"],
                handoff_shell_command(command["command"]),
                ", ".join(f"`{artifact}`" for artifact in command["artifacts"]),
            )
        )
    lines.extend(
        [
            "",
            "## Cost Envelope",
            "",
            "| Peak RSS | Max replay | Reverse/train | Trace/model | Witness/model | Recompute steps |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            "| {} | {} | {} | {} | {} | {} |".format(
                handoff_comma(scorecard["run_peak_rss_bytes"]),
                handoff_comma(scorecard["max_replay_payload_bytes"]),
                handoff_ratio(scorecard["reverse_to_train_elapsed_ratio"]),
                handoff_ratio(scorecard["trace_to_model_payload_ratio"]),
                handoff_ratio(scorecard["witness_to_model_payload_ratio"]),
                handoff_comma(scorecard["total_recompute_steps"]),
            ),
            "",
            "## Machine Artifacts",
            "",
            "| Artifact | Role | SHA-256 | Bytes | Path |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for artifact_id in (
        "pipeline_summary",
        "pipeline_manifest",
        "model_bundle",
        "imported_model_source",
        "imported_model_bundle",
        "imported_model_inference_bundle",
        "samples",
        "model_capsule",
        "model_capsule_verification",
        "q31_reference_inference",
        "native_inference_bundle",
        "native_standalone_rev_classifier",
        "native_standalone_rev_run",
        "native_standalone_rev_roundtrip",
        "native_standalone_rev_roundtrip_verification",
        "reversible_inference_trace",
        "inference_trace_forward",
        "inference_trace_reverse",
        "reversible_inference_roundtrip_proof",
        "reversible_inference_roundtrip_verification",
        "mlp_witness",
    ):
        artifact = artifacts[artifact_id]
        lines.append(
            "| {} | {} | `{}` | {} | `{}` |".format(
                artifact["label"],
                artifact["role"],
                handoff_short_hash(artifact["sha256"]),
                handoff_comma(artifact["bytes"]),
                artifact["path"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_handoff_index(config: PipelineConfig) -> dict[str, Any]:
    paths = out_paths(config.out_dir)
    report = build_handoff_index(config)
    paths["handoff_index"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    paths["handoff_markdown"].write_text(render_handoff_markdown(report), encoding="utf-8")
    return report


def validate_handoff_index(config: PipelineConfig) -> dict[str, Any]:
    paths = out_paths(config.out_dir)
    verifier_paths = {
        "summary": paths["summary"],
        "capsule": paths["model_capsule"],
        "manifest": paths["manifest"],
        "profile": paths["model_capsule_profile"],
        "verification_markdown": paths["model_capsule_verification_markdown"],
        "handoff": paths["handoff_index"],
        "handoff_markdown": paths["handoff_markdown"],
    }
    result = capsule_verify.verify_capsule(
        verifier_paths,
        verify_file_evidence=True,
        allow_missing_profile=False,
    )
    return capsule_verify.verification_report(
        result,
        verify_file_evidence=True,
        allow_missing_profile=False,
        verification_markdown_path=paths["model_capsule_verification_markdown"],
        check_verification_markdown=False,
        handoff_path=paths["handoff_index"],
        handoff_markdown_path=paths["handoff_markdown"],
        require_handoff=True,
    )


def write_inference_action_review_receipt(config: PipelineConfig) -> dict[str, Any]:
    paths = out_paths(config.out_dir)
    handoff = profile_check.load_json(paths["handoff_index"])
    verification = profile_check.load_json(paths["model_capsule_verification"])
    receipt = capsule_verify.run_inference_action_review_commands(
        handoff,
        capsule_fingerprint=handoff["payload"]["capsule"]["fingerprint"],
        trust_certificate_fingerprint=verification["trust_certificate"]["fingerprint"],
        cwd=REPO_ROOT,
        timeout_seconds=capsule_verify.DEFAULT_ACTION_COMMAND_TIMEOUT_SECONDS,
    )
    paths["inference_action_review_receipt"].write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["inference_action_review_receipt_markdown"].write_text(
        capsule_verify.render_inference_action_review_receipt_markdown(receipt),
        encoding="utf-8",
    )
    if receipt["passed"] is not True:
        failed = ", ".join(receipt["payload"]["summary"]["failed_operations"])
        raise ValueError(f"inference action review command replay failed: {failed}")
    return receipt


def training_update_review_commands(
    config: PipelineConfig,
    paths: dict[str, Path],
    audit_step: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    runner = runner_prefix(config)
    selection = audit_step.get("selection", {}) if isinstance(audit_step, dict) else {}
    selected_step = selection.get("selected_step") if isinstance(selection, dict) else None
    strategy = selection.get("strategy") if isinstance(selection, dict) else None
    if not isinstance(selected_step, int) or isinstance(selected_step, bool):
        selected_step = config.audit_step
    if not isinstance(strategy, str) or strategy not in AUDIT_STEP_STRATEGIES:
        strategy = config.audit_step_strategy
    return [
        {
            "operation": "replay_training_lineage",
            "label": "Replay full training lineage",
            "purpose": "Runs the retained witness trace backward from the final model to prove the training lineage restores the initial model.",
            "command": [
                *runner,
                "--verify-audit",
                str(paths["audit_bundle"]),
                "--markdown-output",
                "/tmp/reverie-training-audit-verification.md",
            ],
            "artifacts": [
                "audit_bundle",
                "audit_verification",
                "training_audit_verification_markdown",
            ],
        },
        {
            "operation": "inspect_training_update",
            "label": "Inspect selected training update",
            "purpose": "Reconstructs the selected before/after model window and regenerates logits, error, active pixels, and update ledgers.",
            "command": [
                *runner,
                "--inspect-audit",
                str(paths["audit_bundle"]),
                "--audit-step",
                str(selected_step),
                "--audit-step-strategy",
                strategy,
                "--step-output",
                "/tmp/reverie-training-step-bundle.json",
                "--markdown-output",
                "/tmp/reverie-training-step-debug.md",
                "--json",
            ],
            "artifacts": [
                "audit_bundle",
                "training_step_bundle",
                "training_step_debug",
            ],
        },
        {
            "operation": "reverse_training_update",
            "label": "Reverse selected training update",
            "purpose": "Verifies the saved one-step bundle forward and backward, proving the selected update restores the before-model state.",
            "command": [
                *runner,
                "--verify-step",
                str(paths["step_bundle"]),
                "--markdown-output",
                "/tmp/reverie-training-step-verification.md",
                "--json",
            ],
            "artifacts": [
                "training_step_bundle",
                "training_step_debug",
            ],
        },
    ]


def run_training_update_review_commands(
    commands: list[dict[str, Any]],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    rows = []
    for command in commands:
        argv = command["command"]
        started = time.monotonic()
        stdout = ""
        stderr = ""
        exit_code: Optional[int] = None
        timed_out = False
        error = None
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            exit_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as timeout_error:
            timed_out = True
            stdout = capsule_verify.command_output_text(timeout_error.stdout)
            stderr = capsule_verify.command_output_text(timeout_error.stderr)
            error = f"timed out after {timeout_seconds:g}s"
        except OSError as os_error:
            error = str(os_error)
            stderr = str(os_error)
        elapsed_seconds = time.monotonic() - started
        rows.append(
            {
                "operation": command["operation"],
                "label": command["label"],
                "purpose": command["purpose"],
                "command": argv,
                "command_text": handoff_shell_command(argv),
                "artifacts": command["artifacts"],
                "passed": exit_code == 0 and not timed_out and error is None,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "error": error,
                "elapsed_seconds": elapsed_seconds,
                "stdout_bytes": len(stdout.encode("utf-8")),
                "stderr_bytes": len(stderr.encode("utf-8")),
                "stdout_sha256": sha256_text(stdout),
                "stderr_sha256": sha256_text(stderr),
                "stdout_tail": capsule_verify.text_tail(stdout),
                "stderr_tail": capsule_verify.text_tail(stderr),
            }
        )
    return rows


def training_update_review_semantic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "reverie_training_update_review_semantics_v1",
        "capsule_fingerprint": payload["capsule_fingerprint"],
        "handoff_fingerprint": payload["handoff_fingerprint"],
        "selected_training_update": payload["selected_training_update"],
        "summary": {
            "passed": payload["summary"]["passed"],
            "command_count": payload["summary"]["command_count"],
            "passed_count": payload["summary"]["passed_count"],
            "failed_count": payload["summary"]["failed_count"],
            "failed_operations": payload["summary"]["failed_operations"],
            "timed_out_operations": payload["summary"]["timed_out_operations"],
        },
        "operations": [
            {
                "operation": row["operation"],
                "command": row["command"],
                "artifacts": row["artifacts"],
                "passed": row["passed"],
                "exit_code": row["exit_code"],
                "timed_out": row["timed_out"],
                "error": row["error"],
            }
            for row in payload["operations"]
        ],
    }


def training_update_review_semantic_fingerprint(payload: dict[str, Any]) -> str:
    return sha256_json(training_update_review_semantic_payload(payload))


def render_training_update_review_receipt_markdown(receipt: dict[str, Any]) -> str:
    payload = receipt["payload"]
    summary = payload["summary"]
    selected = payload["selected_training_update"]
    lines = [
        "# Reverie Training Update Review Receipt",
        "",
        "| Verdict | Receipt | Semantic | Capsule | Handoff | Step | Sample | Commands |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
        "| {} | `{}` | `{}` | `{}` | `{}` | {} | {} | {}/{} |".format(
            "pass" if receipt["passed"] else "fail",
            handoff_short_hash(receipt["fingerprint"]),
            handoff_short_hash(payload["semantic_fingerprint"]),
            handoff_short_hash(payload["capsule_fingerprint"]),
            handoff_short_hash(payload["handoff_fingerprint"]),
            selected["step"],
            selected["sample_index"],
            summary["passed_count"],
            summary["command_count"],
        ),
        "",
        "## Selected Update",
        "",
        "| Prediction | Label | Correct | Active pixels | Margin | Cause ledger | Bias ledger | Weight ledger |",
        "| ---: | ---: | --- | ---: | ---: | --- | --- | --- |",
        "| {} | {} | {} | {} | {} | `{}` | `{}` | `{}` |".format(
            selected["prediction"],
            selected["label"],
            str(selected["correct"]).lower(),
            selected["active_pixels"],
            selected["margin"],
            handoff_short_hash(selected["cause_ledger_fingerprint"]),
            handoff_short_hash(selected["bias_delta_ledger_fingerprint"]),
            handoff_short_hash(selected["weight_delta_ledger_fingerprint"]),
        ),
        "",
        "## Commands",
        "",
        "| Operation | Status | Exit | Elapsed | Stdout | Stderr | Artifacts |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in payload["operations"]:
        status = "timeout" if row["timed_out"] else ("pass" if row["passed"] else "fail")
        lines.append(
            "| `{}` | {} | {} | {:.3f}s | {} B / `{}` | {} B / `{}` | {} |".format(
                row["operation"],
                status,
                "n/a" if row["exit_code"] is None else row["exit_code"],
                row["elapsed_seconds"],
                row["stdout_bytes"],
                handoff_short_hash(row["stdout_sha256"]),
                row["stderr_bytes"],
                handoff_short_hash(row["stderr_sha256"]),
                ", ".join(f"`{artifact}`" for artifact in row["artifacts"]),
            )
        )
    lines.extend(["", "## Command Lines", ""])
    for row in payload["operations"]:
        lines.extend(
            [
                f"### `{row['operation']}`",
                "",
                f"`{row['command_text']}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_training_update_review_receipt(
    config: PipelineConfig,
    commands: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    paths = out_paths(config.out_dir)
    capsule = profile_check.load_json(paths["model_capsule"])
    handoff = profile_check.load_json(paths["handoff_index"])
    audit_step = profile_check.load_json(paths["audit_step"])
    step_verification = profile_check.load_json(paths["step_verification"])
    if commands is None:
        commands = training_update_review_commands(config, paths, audit_step)
    operations = run_training_update_review_commands(
        commands,
        cwd=REPO_ROOT,
        timeout_seconds=capsule_verify.DEFAULT_ACTION_COMMAND_TIMEOUT_SECONDS,
    )
    failed_operations = [
        row["operation"] for row in operations if row["passed"] is not True
    ]
    selected = {
        "step": audit_step["step"],
        "sample_index": audit_step["sample_index"],
        "label": audit_step["label"],
        "prediction": audit_step["prediction"],
        "correct": audit_step["correct"],
        "margin": audit_step["logit_margin"]["margin"],
        "active_pixels": len(audit_step["active_pixels"]),
        "cause_ledger_fingerprint": audit_step["cause_ledger"]["fingerprint"],
        "bias_delta_ledger_fingerprint": audit_step["update"]["bias_delta_ledger_fingerprint"],
        "weight_delta_ledger_fingerprint": audit_step["update"]["weight_delta_ledger_fingerprint"],
        "step_verification_fingerprint": step_verification["fingerprints"]["payload"],
        "proof_fingerprint": step_verification["fingerprints"]["proof"],
    }
    summary = {
        "passed": not failed_operations,
        "command_count": len(operations),
        "passed_count": len(operations) - len(failed_operations),
        "failed_count": len(failed_operations),
        "failed_operations": failed_operations,
        "timed_out_operations": [
            row["operation"] for row in operations if row["timed_out"] is True
        ],
    }
    payload = {
        "schema": TRAINING_UPDATE_REVIEW_RECEIPT_SCHEMA,
        "capsule_fingerprint": capsule["fingerprint"],
        "handoff_fingerprint": handoff["fingerprint"],
        "training_step_bundle": str(paths["step_bundle"]),
        "training_audit_bundle": str(paths["audit_bundle"]),
        "selected_training_update": selected,
        "command_cwd": str(REPO_ROOT),
        "timeout_seconds": capsule_verify.DEFAULT_ACTION_COMMAND_TIMEOUT_SECONDS,
        "summary": summary,
        "operations": operations,
    }
    payload["semantic_fingerprint"] = training_update_review_semantic_fingerprint(payload)
    receipt = {
        "kind": TRAINING_UPDATE_REVIEW_RECEIPT_KIND,
        "passed": summary["passed"],
        "algorithm": "sha256",
        "fingerprint": sha256_json(payload),
        "payload": payload,
    }
    paths["training_update_review_receipt"].write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["training_update_review_receipt_markdown"].write_text(
        render_training_update_review_receipt_markdown(receipt),
        encoding="utf-8",
    )
    if receipt["passed"] is not True:
        failed = ", ".join(failed_operations)
        raise ValueError(f"training update review command replay failed: {failed}")
    return receipt


def valid_pipeline_reports_for_self_test() -> dict[str, dict[str, Any]]:
    run_report = profile_check.valid_run_report()
    run_report["train"]["correct"] = run_report["train"]["samples"]
    run_report["train"]["accuracy_percent"] = 100.0

    row_inference_verification = profile_check.valid_inference_verification_report()
    row_inference_verification["source_evaluation_checked"] = True
    row_inference_verification["source_model_evaluation"] = {
        "path": "target/evaluation-bundle.json",
        "resolved_path": "target/evaluation-bundle.json",
        "checked": True,
        "payload_fingerprint": "0" * 64,
        "row_index": 1,
        "sample_fingerprint": "1" * 64,
    }

    return {
        "run_report": run_report,
        "artifact_comparison": profile_check.valid_audit_contract_report(),
        "audit_verification": profile_check.valid_audit_verification_report(),
        "audit_scan": profile_check.valid_audit_scan_report(),
        "audit_step": profile_check.valid_audit_step_report(),
        "step_verification": profile_check.valid_step_verification_report(),
        "imported_model_import": profile_check.valid_model_import_report(),
        "imported_model_verification": profile_check.valid_model_verification_report(),
        "imported_model_inference": profile_check.valid_native_inference_report(
            profile_check.MODEL_INFERENCE_AUDIT_KIND
        ),
        "imported_model_inference_verification": (
            profile_check.valid_inference_verification_report()
        ),
        "native_inference": profile_check.valid_native_inference_report(
            profile_check.MODEL_INFERENCE_AUDIT_KIND
        ),
        "native_inference_verification": (
            profile_check.valid_inference_verification_report()
        ),
        "model_evaluation": profile_check.valid_model_evaluation_report(),
        "model_evaluation_verification": (
            profile_check.valid_model_evaluation_verification_report()
        ),
        "model_evaluation_scan": profile_check.valid_model_evaluation_scan_report(),
        "model_evaluation_row": profile_check.valid_native_inference_report(
            profile_check.MODEL_EVALUATION_ROW_KIND
        ),
        "row_inference_verification": row_inference_verification,
        "mlp_witness": profile_check.valid_mlp_witness_report(),
        "invertible_coupling": profile_check.valid_invertible_coupling_report(),
        "triangular_residual": profile_check.valid_triangular_residual_report(),
        "reversible_preprocess": profile_check.valid_reversible_preprocess_report(),
        "reversible_inference_trace": profile_check.valid_reversible_inference_trace_report(),
        "q31_reference_inference": profile_check.valid_q31_reference_inference_report(),
        "q31_reference_evaluation": profile_check.valid_q31_reference_evaluation_report(),
    }


def write_synthetic_pipeline_files(paths: dict[str, Path], reports: dict[str, dict[str, Any]]) -> None:
    paths["profile_markdown"].parent.mkdir(parents=True, exist_ok=True)
    paths["profile_markdown"].write_text("# Synthetic profile\n", encoding="utf-8")
    for key, report in reports.items():
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        paths[key].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    native = reports["native_inference"]
    paths["native_standalone_rev_classifier"].write_text(
        "\n".join(
            [
                "// Synthetic standalone MNIST classifier.",
                "global image: tensor<int, 2> = [2147483648, 2147483648];",
                "global weights: tensor<int, 2, 10> = [",
                "  [0, 0, 0, 2147483648, 0, 0, 0, 0, 0, 0],",
                "  [0, 0, 0, 0, 1073741824, 0, 0, 0, 0, 0]",
                "];",
                "global bias: tensor<int, 10> = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];",
                "global logits: witness<tensor<int, 10>>;",
                "global prediction;",
                "global correct;",
                f"global label = {native['label']};",
                f"global expected_prediction = {native['prediction']};",
                f"global expected_correct = {1 if native['correct'] else 0};",
                "",
                "procedure main()",
                "  logits += vecmat_q31(image, weights);",
                "  logits += bias;",
                "  prediction += argmax(logits);",
                "  correct += argmax_eq(logits, label);",
                "  assert prediction == expected_prediction;",
                "  assert correct == expected_correct;",
                "end",
                "",
            ]
        ),
        encoding="utf-8",
    )
    paths["native_standalone_rev_run"].write_text(
        json.dumps(
            {
                "kind": "reverie_run_result",
                "store": {
                    "prediction": native["prediction"],
                    "correct": 1 if native["correct"] else 0,
                    "label": native["label"],
                    "expected_prediction": native["prediction"],
                    "expected_correct": 1 if native["correct"] else 0,
                },
                "ml_profile": {"builtin_counts": {"vecmat_q31": 1}},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["native_standalone_rev_roundtrip"].write_text(
        json.dumps(
            {
                "kind": "reverie_roundtrip_result",
                "passed": True,
                "fingerprint": "2" * 64,
                "payload": {
                    "baseline": {"store": {"prediction": 0, "correct": 0}},
                    "forward": {
                        "store": {
                            "prediction": native["prediction"],
                            "correct": 1 if native["correct"] else 0,
                        }
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["native_standalone_rev_roundtrip_verification"].write_text(
        json.dumps(
            {
                "kind": "reverie_roundtrip_verification",
                "passed": True,
                "checks": {
                    "source_hash_matches": True,
                    "replay_restoration_passed": True,
                    "replayed": True,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for key, path in bundle_paths(paths).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps({"kind": f"synthetic_{key}"}) + "\n", encoding="utf-8")


def run_pipeline(config: PipelineConfig) -> None:
    validate_config(config)
    steps = build_steps(config)
    if config.dry_run:
        print(f"write MLP witness seed: {out_paths(config.out_dir)['mlp_vars']}")
        print(
            "write invertible coupling final seed: "
            f"{out_paths(config.out_dir)['coupling_final_vars']}"
        )
        print(
            "write triangular residual final seed: "
            f"{out_paths(config.out_dir)['residual_final_vars']}"
        )
        print(
            "write reversible preprocessing final seed: "
            f"{out_paths(config.out_dir)['preprocess_final_vars']}"
        )
        print(
            "write reversible inference trace final seed: "
            f"{out_paths(config.out_dir)['inference_trace_final_vars']}"
        )
        for step in steps:
            output = "" if step.output is None else f" > {step.output}"
            print(f"{step.label}: {' '.join(step.command)}{output}")
        return
    config.out_dir.mkdir(parents=True, exist_ok=True)
    write_mlp_witness_seed(config)
    write_invertible_coupling_seed(config)
    write_triangular_residual_seed(config)
    write_reversible_preprocess_seed(config)
    write_reversible_inference_trace_seed(config)
    executed_steps: list[PipelineStep] = []
    effective_config = config
    effective_steps = steps
    next_index = 0
    if config.audit_step_strategy == "explicit":
        next_index = 0
    else:
        for index, step in enumerate(steps[:2], start=1):
            print(f"[{index}/{len(steps)}] {step.label}", file=sys.stderr)
            run_step(step)
            executed_steps.append(step)
        audit_scan_report = profile_check.load_json(out_paths(config.out_dir)["audit_scan"])
        profile_check.validate_report(
            audit_scan_report,
            require_reverse_check=False,
            require_peak_rss=False,
            require_ml_profile=False,
            require_audit_contract=False,
        )
        requested_step = config.audit_step
        selected_step = select_audit_step_from_scan(
            audit_scan_report,
            strategy=config.audit_step_strategy,
            requested_step=requested_step,
        )
        if selected_step >= config.sample_limit:
            raise PipelineError(
                f"selected audit step {selected_step} must be less than --sample-limit {config.sample_limit}"
            )
        print(
            "[select] audit step {} via {} strategy (requested {})".format(
                selected_step,
                config.audit_step_strategy,
                requested_step,
            ),
            file=sys.stderr,
        )
        effective_config = replace(
            config,
            audit_step=selected_step,
            requested_audit_step=requested_step,
        )
        effective_steps = build_steps(effective_config)
        next_index = 2
    if effective_config.evaluation_row_strategy == "explicit":
        for index, step in enumerate(effective_steps[next_index:], start=next_index + 1):
            print(f"[{index}/{len(effective_steps)}] {step.label}", file=sys.stderr)
            run_step(step)
            executed_steps.append(step)
    else:
        row_step_index = next(
            index
            for index, step in enumerate(effective_steps)
            if step.label == "inspect one evaluation row"
        )
        for index, step in enumerate(effective_steps[next_index:row_step_index], start=next_index + 1):
            print(f"[{index}/{len(effective_steps)}] {step.label}", file=sys.stderr)
            run_step(step)
            executed_steps.append(step)
        evaluation_scan_report = profile_check.load_json(
            out_paths(effective_config.out_dir)["model_evaluation_scan"]
        )
        profile_check.validate_report(
            evaluation_scan_report,
            require_reverse_check=False,
            require_peak_rss=False,
            require_ml_profile=False,
            require_audit_contract=False,
        )
        requested_row = effective_config.evaluation_row
        selected_row = select_evaluation_row_from_scan(
            evaluation_scan_report,
            strategy=effective_config.evaluation_row_strategy,
            requested_row=requested_row,
        )
        if selected_row >= effective_config.sample_limit:
            raise PipelineError(
                f"selected evaluation row {selected_row} must be less than --sample-limit {effective_config.sample_limit}"
            )
        print(
            "[select] evaluation row {} via {} strategy (requested {})".format(
                selected_row,
                effective_config.evaluation_row_strategy,
                requested_row,
            ),
            file=sys.stderr,
        )
        effective_config = replace(
            effective_config,
            evaluation_row=selected_row,
            requested_evaluation_row=requested_row,
        )
        effective_steps = build_steps(effective_config)
        for index, step in enumerate(effective_steps[row_step_index:], start=row_step_index + 1):
            print(f"[{index}/{len(effective_steps)}] {step.label}", file=sys.stderr)
            run_step(step)
            executed_steps.append(step)
    summary = write_pipeline_summary(effective_config)
    capsule = write_model_capsule(effective_config, summary)
    write_manifest(effective_config, executed_steps, summary, capsule)
    validate_written_pipeline_artifacts(effective_config)
    write_training_step_debug_markdown(effective_config)
    write_model_capsule_profile(effective_config)
    verification = write_model_capsule_verification(effective_config)
    handoff = write_handoff_index(effective_config)
    validate_handoff_index(effective_config)
    receipt = write_inference_action_review_receipt(effective_config)
    training_receipt = write_training_update_review_receipt(effective_config)
    paths = out_paths(config.out_dir)
    print(f"ok: wrote checked MNIST ML audit profile to {paths['profile_markdown']}")
    print(f"summary: {paths['summary']}")
    print(f"training step debug: {paths['training_step_debug_markdown']}")
    print(f"Q31 inference explanation: {paths['q31_reference_inference_markdown']}")
    print(f"native inference audit: {paths['native_inference_audit_markdown']}")
    print(f"native inference verification: {paths['native_inference_verification_markdown']}")
    print(
        "evaluation-row inference audit: "
        f"{paths['evaluation_row_inference_audit_markdown']}"
    )
    print(f"MLP witness card: {paths['mlp_witness_markdown']}")
    print(f"invertible coupling card: {paths['invertible_coupling_markdown']}")
    print(f"triangular residual card: {paths['triangular_residual_markdown']}")
    print(f"reversible preprocess card: {paths['reversible_preprocess_markdown']}")
    print(f"reversible inference trace card: {paths['reversible_inference_trace_markdown']}")
    print(f"handoff index: {paths['handoff_index']}")
    print(f"handoff markdown: {paths['handoff_markdown']}")
    print(f"model capsule: {paths['model_capsule']}")
    print(f"model capsule profile: {paths['model_capsule_profile']}")
    print(f"model capsule verification: {paths['model_capsule_verification']}")
    print(f"model capsule verification markdown: {paths['model_capsule_verification_markdown']}")
    print(f"model capsule gates: {verification['capsule']['gates']['passed']}/{verification['capsule']['gates']['total']}")
    print(f"handoff fingerprint: {handoff['fingerprint']}")
    print(f"inference action receipt: {paths['inference_action_review_receipt']}")
    print(f"inference action receipt markdown: {paths['inference_action_review_receipt_markdown']}")
    print(
        "inference action commands: "
        f"{receipt['payload']['summary']['passed_count']}/"
        f"{receipt['payload']['summary']['command_count']}"
    )
    print(f"training update receipt: {paths['training_update_review_receipt']}")
    print(f"training update receipt markdown: {paths['training_update_review_receipt_markdown']}")
    print(
        "training update commands: "
        f"{training_receipt['payload']['summary']['passed_count']}/"
        f"{training_receipt['payload']['summary']['command_count']}"
    )
    print(f"manifest: {paths['manifest']}")


def run_self_tests() -> int:
    try:
        with tempfile.TemporaryDirectory() as directory:
            config = PipelineConfig(
                out_dir=Path(directory) / "audit",
                sample_limit=3,
                audit_step=1,
                evaluation_row=2,
                runner_bin=Path("target/debug/reverie-mnist-linear"),
                reverie_bin=Path("target/debug/reverie"),
                dry_run=True,
            )
            validate_config(config)
            steps = build_steps(config)
            commands = "\n".join(" ".join(step.command) for step in steps)
            required = [
                "--self-test",
                "--reverse-check",
                "--audit-output",
                "--scan-audit",
                "--inspect-audit",
                "--verify-step",
                "--export-model",
                "scripts/export_q31_linear_model.py",
                "--import-model-json",
                "--verify-model",
                "imported-model-inference-bundle.json",
                "--export-samples",
                "--inspect-model-inference",
                "--standalone-rev-output",
                "mnist-standalone-classifier.rev",
                "mnist-standalone-classifier-roundtrip.json",
                "--verify-inference",
                "native-inference-audit.md",
                "--evaluate-model",
                "--verify-evaluation",
                "--scan-evaluation",
                "--inspect-evaluation",
                "model-evaluation-row-inference-audit.md",
                "examples/mnist_mlp_witness.rev",
                "scripts/check_q31_mlp_witness.py",
                "examples/invertible_coupling.rev",
                "scripts/check_invertible_coupling.py",
                "examples/reversible_preprocess.rev",
                "scripts/check_reversible_preprocess.py",
                "examples/reversible_inference_trace.rev",
                "scripts/check_reversible_inference_trace.py",
                "--compare-artifacts",
                "scripts/check_q31_inference.py",
                "scripts/check_mnist_ml_profile.py",
                "scripts/summarize_mnist_ml_profile.py",
                "--require-audit-contract",
            ]
            missing = [snippet for snippet in required if snippet not in commands]
            if missing:
                raise AssertionError("missing planned command snippet(s): " + ", ".join(missing))
            outputs = [step.output for step in steps if step.output is not None]
            if len(outputs) != len(set(outputs)):
                raise AssertionError("pipeline outputs must be unique")
            q31_reference_step = next(
                step
                for step in steps
                if step.label == "reference-check one Q31 inference"
            )
            sample_index_flag = q31_reference_step.command.index("--sample-index")
            if q31_reference_step.command[sample_index_flag + 1] != str(config.audit_step):
                raise AssertionError("Q31 reference inference must inspect the selected audit step")
            markdown_flag = q31_reference_step.command.index("--markdown-output")
            if (
                q31_reference_step.command[markdown_flag + 1]
                != str(out_paths(config.out_dir)["q31_reference_inference_markdown"])
            ):
                raise AssertionError("Q31 reference inference must write the explanation Markdown")
            evaluation_row_step = next(
                step
                for step in steps
                if step.label == "inspect one evaluation row"
            )
            row_flag = evaluation_row_step.command.index("--evaluation-row")
            if evaluation_row_step.command[row_flag + 1] != str(config.evaluation_row):
                raise AssertionError("evaluation-row inspection must use the configured evaluation row")
            report_names = {path.name for path in profile_reports(out_paths(config.out_dir))}
            for name in (
                "run-report.json",
                "artifact-comparison.json",
                "imported-model-import-report.json",
                "imported-model-verification.json",
                "imported-model-inference-report.json",
                "imported-model-inference-verification.json",
                "native-inference-report.json",
                "model-evaluation-report.json",
                "mlp-witness-report.json",
                "invertible-coupling-report.json",
                "reversible-preprocess-report.json",
                "reversible-inference-trace-report.json",
                "q31-reference-evaluation.json",
            ):
                if name not in report_names:
                    raise AssertionError(f"profile reports missing {name}")
            summary_config = PipelineConfig(
                out_dir=Path(directory) / "summary-audit",
                sample_limit=2,
                audit_step=0,
                evaluation_row=1,
                runner_bin=Path("target/debug/reverie-mnist-linear"),
                reverie_bin=Path("target/debug/reverie"),
            )
            validate_config(summary_config)
            summary_paths = out_paths(summary_config.out_dir)
            write_mlp_witness_seed(summary_config)
            write_invertible_coupling_seed(summary_config)
            write_triangular_residual_seed(summary_config)
            write_reversible_preprocess_seed(summary_config)
            write_reversible_inference_trace_seed(summary_config)
            if not summary_paths["mlp_vars"].exists():
                raise AssertionError("MLP witness seed was not written")
            if not summary_paths["coupling_final_vars"].exists():
                raise AssertionError("invertible coupling final seed was not written")
            if not summary_paths["residual_final_vars"].exists():
                raise AssertionError("triangular residual final seed was not written")
            if not summary_paths["preprocess_final_vars"].exists():
                raise AssertionError("reversible preprocessing final seed was not written")
            if not summary_paths["inference_trace_final_vars"].exists():
                raise AssertionError("reversible inference trace final seed was not written")
            reports = valid_pipeline_reports_for_self_test()
            if select_audit_step_from_scan(
                reports["audit_scan"],
                strategy="lowest-margin",
                requested_step=0,
            ) != 1:
                raise AssertionError("lowest-margin audit-step strategy did not select scan minimum")
            if select_audit_step_from_scan(
                reports["audit_scan"],
                strategy="largest-update",
                requested_step=1,
            ) != 0:
                raise AssertionError("largest-update audit-step strategy did not select scan maximum")
            if select_audit_step_from_scan(
                reports["audit_scan"],
                strategy="top-suspicious",
                requested_step=1,
            ) != 0:
                raise AssertionError("top-suspicious audit-step strategy did not select scan ranking")
            if select_evaluation_row_from_scan(
                reports["model_evaluation_scan"],
                strategy="lowest-margin",
                requested_row=1,
            ) != 0:
                raise AssertionError("lowest-margin evaluation-row strategy did not select scan minimum")
            incorrect_scan = json.loads(json.dumps(reports["model_evaluation_scan"]))
            incorrect_scan["top_incorrect"] = [{"index": 1}]
            if select_evaluation_row_from_scan(
                incorrect_scan,
                strategy="top-incorrect",
                requested_row=0,
            ) != 1:
                raise AssertionError("top-incorrect evaluation-row strategy did not select scan ranking")
            write_synthetic_pipeline_files(summary_paths, reports)
            summary = build_pipeline_summary(summary_config, summary_paths, reports)
            if summary["kind"] != SUMMARY_KIND or not summary["gates"]["passed"]:
                raise AssertionError("valid synthetic pipeline summary did not pass")
            if summary["metrics"]["q31_reference_evaluation"]["samples"] != 2:
                raise AssertionError("summary did not capture Q31 reference sample count")
            summary_paths["summary"].parent.mkdir(parents=True, exist_ok=True)
            written_summary = write_pipeline_summary(summary_config)
            if written_summary["gates"]["passed"] is not True:
                raise AssertionError("written synthetic pipeline summary did not pass")
            capsule = write_model_capsule(summary_config, written_summary)
            if capsule["kind"] != MODEL_CAPSULE_KIND:
                raise AssertionError("model capsule has the wrong kind")
            if capsule["payload"]["gates"]["total"] != len(written_summary["gates"]["checks"]):
                raise AssertionError("model capsule did not capture gate count")
            write_manifest(summary_config, build_steps(summary_config), written_summary, capsule)
            validate_written_pipeline_artifacts(summary_config)
            write_training_step_debug_markdown(summary_config)
            training_debug = summary_paths["training_step_debug_markdown"].read_text(
                encoding="utf-8"
            )
            for snippet in (
                "# Reverie Training Step Debug",
                "step_backward_from_model_update",
                "## Top Weight Deltas",
                "## Replay Proof",
                reports["audit_step"]["cause_ledger"]["fingerprint"][:12],
            ):
                if snippet not in training_debug:
                    raise AssertionError(f"training-step debug Markdown missing {snippet}")
            write_model_capsule_profile(summary_config)
            capsule_profile = summary_paths["model_capsule_profile"].read_text(encoding="utf-8")
            for snippet in ("## Model Capsules", "## North-Star Readiness", capsule["fingerprint"][:12]):
                if snippet not in capsule_profile:
                    raise AssertionError(f"model capsule profile missing {snippet}")
            verification = write_model_capsule_verification(summary_config)
            if verification["kind"] != capsule_verify.VERIFICATION_KIND:
                raise AssertionError("model capsule verification has the wrong kind")
            if verification["capsule"]["fingerprint"] != capsule["fingerprint"]:
                raise AssertionError("model capsule verification did not capture capsule fingerprint")
            if verification["profile"]["matches_rendered"] is not True:
                raise AssertionError("model capsule verification did not validate rendered profile")
            if verification["verification_markdown"]["saved"]["matches_rendered"] is not True:
                raise AssertionError("model capsule verification did not validate saved Markdown")
            verification_markdown = summary_paths["model_capsule_verification_markdown"].read_text(
                encoding="utf-8"
            )
            for snippet in (
                "# Reverie Model Capsule Verification",
                "## Trust Certificate",
                verification["trust_certificate"]["fingerprint"][:12],
            ):
                if snippet not in verification_markdown:
                    raise AssertionError(f"model capsule verification Markdown missing {snippet}")
            summary_paths["q31_reference_inference_markdown"].write_text(
                "# Reverie Q31 Inference Explanation\n",
                encoding="utf-8",
            )
            summary_paths["native_inference_verification_markdown"].write_text(
                "# Reverie Inference Verification\n",
                encoding="utf-8",
            )
            summary_paths["native_inference_audit_markdown"].write_text(
                "# Reverie Inference Audit\n",
                encoding="utf-8",
            )
            summary_paths["evaluation_row_inference_audit_markdown"].write_text(
                "# Reverie Inference Audit\n",
                encoding="utf-8",
            )
            summary_paths["mlp_witness_markdown"].write_text(
                "# Reverie MLP Witness Proof\n",
                encoding="utf-8",
            )
            summary_paths["invertible_coupling_markdown"].write_text(
                "# Reverie Invertible Coupling Proof\n",
                encoding="utf-8",
            )
            summary_paths["triangular_residual_markdown"].write_text(
                "# Reverie Triangular Residual Proof\n",
                encoding="utf-8",
            )
            summary_paths["reversible_preprocess_markdown"].write_text(
                "# Reverie Reversible Preprocess Proof\n",
                encoding="utf-8",
            )
            summary_paths["audit_verification_markdown"].write_text(
                "# Reverie Training Audit Verification\n",
                encoding="utf-8",
            )
            summary_paths["inference_trace_roundtrip_markdown"].write_text(
                "# Reverie Roundtrip Verification\n",
                encoding="utf-8",
            )
            summary_paths["reversible_inference_trace_markdown"].write_text(
                "# Reverie Reversible Inference Trace\n",
                encoding="utf-8",
            )
            handoff = write_handoff_index(summary_config)
            if handoff["kind"] != HANDOFF_KIND:
                raise AssertionError("handoff index has wrong kind")
            if handoff["fingerprint"] != sha256_json(handoff["payload"]):
                raise AssertionError("handoff index fingerprint did not match payload")
            handoff_verification = validate_handoff_index(summary_config)
            if handoff_verification["handoff"]["present"] is not True:
                raise AssertionError("handoff verifier did not report the handoff present")
            if handoff_verification["handoff"]["artifact_count"] != len(
                handoff["payload"]["artifacts"]
            ):
                raise AssertionError("handoff verifier artifact count did not match handoff")
            handoff_artifact = handoff["payload"]["artifacts"]["training_step_debug"]
            if handoff_artifact["sha256"] != sha256_file(summary_paths["training_step_debug_markdown"]):
                raise AssertionError("handoff did not hash training-step debug Markdown")
            audit_card = handoff["payload"]["artifacts"]["training_audit_verification_markdown"]
            if audit_card["sha256"] != sha256_file(summary_paths["audit_verification_markdown"]):
                raise AssertionError("handoff did not hash training-audit verification Markdown")
            q31_card = handoff["payload"]["artifacts"]["q31_reference_inference_markdown"]
            if q31_card["sha256"] != sha256_file(summary_paths["q31_reference_inference_markdown"]):
                raise AssertionError("handoff did not hash Q31 inference explanation Markdown")
            native_inference_card = handoff["payload"]["artifacts"][
                "native_inference_verification_markdown"
            ]
            if native_inference_card["sha256"] != sha256_file(
                summary_paths["native_inference_verification_markdown"]
            ):
                raise AssertionError("handoff did not hash native inference verification Markdown")
            native_audit_card = handoff["payload"]["artifacts"][
                "native_inference_audit_markdown"
            ]
            if native_audit_card["sha256"] != sha256_file(
                summary_paths["native_inference_audit_markdown"]
            ):
                raise AssertionError("handoff did not hash native inference audit Markdown")
            evaluation_audit_card = handoff["payload"]["artifacts"][
                "evaluation_row_inference_audit_markdown"
            ]
            if evaluation_audit_card["sha256"] != sha256_file(
                summary_paths["evaluation_row_inference_audit_markdown"]
            ):
                raise AssertionError("handoff did not hash evaluation-row inference audit Markdown")
            mlp_card = handoff["payload"]["artifacts"]["mlp_witness_markdown"]
            if mlp_card["sha256"] != sha256_file(summary_paths["mlp_witness_markdown"]):
                raise AssertionError("handoff did not hash MLP witness Markdown")
            coupling_card = handoff["payload"]["artifacts"]["invertible_coupling_markdown"]
            if coupling_card["sha256"] != sha256_file(summary_paths["invertible_coupling_markdown"]):
                raise AssertionError("handoff did not hash invertible coupling Markdown")
            residual_card = handoff["payload"]["artifacts"]["triangular_residual_markdown"]
            if residual_card["sha256"] != sha256_file(summary_paths["triangular_residual_markdown"]):
                raise AssertionError("handoff did not hash triangular residual Markdown")
            preprocess_card = handoff["payload"]["artifacts"]["reversible_preprocess_markdown"]
            if preprocess_card["sha256"] != sha256_file(
                summary_paths["reversible_preprocess_markdown"]
            ):
                raise AssertionError("handoff did not hash reversible preprocess Markdown")
            trace_card = handoff["payload"]["artifacts"]["reversible_inference_trace_markdown"]
            if trace_card["sha256"] != sha256_file(
                summary_paths["reversible_inference_trace_markdown"]
            ):
                raise AssertionError("handoff did not hash reversible inference trace Markdown")
            roundtrip_card = handoff["payload"]["artifacts"][
                "reversible_inference_roundtrip_markdown"
            ]
            if roundtrip_card["sha256"] != sha256_file(
                summary_paths["inference_trace_roundtrip_markdown"]
            ):
                raise AssertionError("handoff did not hash roundtrip Markdown card")
            roundtrip_verification = handoff["payload"]["artifacts"][
                "reversible_inference_roundtrip_verification"
            ]
            if roundtrip_verification["sha256"] != sha256_file(
                summary_paths["inference_trace_roundtrip_verification"]
            ):
                raise AssertionError("handoff did not hash roundtrip verification JSON")
            standalone_source = handoff["payload"]["artifacts"][
                "native_standalone_rev_classifier"
            ]
            if standalone_source["sha256"] != sha256_file(
                summary_paths["native_standalone_rev_classifier"]
            ):
                raise AssertionError("handoff did not hash standalone Reverie classifier")
            standalone_run = handoff["payload"]["artifacts"]["native_standalone_rev_run"]
            if standalone_run["sha256"] != sha256_file(
                summary_paths["native_standalone_rev_run"]
            ):
                raise AssertionError("handoff did not hash standalone Reverie run JSON")
            standalone_roundtrip = handoff["payload"]["artifacts"][
                "native_standalone_rev_roundtrip"
            ]
            if standalone_roundtrip["sha256"] != sha256_file(
                summary_paths["native_standalone_rev_roundtrip"]
            ):
                raise AssertionError("handoff did not hash standalone Reverie roundtrip JSON")
            standalone_verification = handoff["payload"]["artifacts"][
                "native_standalone_rev_roundtrip_verification"
            ]
            if standalone_verification["sha256"] != sha256_file(
                summary_paths["native_standalone_rev_roundtrip_verification"]
            ):
                raise AssertionError(
                    "handoff did not hash standalone Reverie roundtrip verification JSON"
                )
            action_operations = [
                command["operation"]
                for command in handoff["payload"]["inference_action_review_commands"]
            ]
            if action_operations != [
                "reproduce_prediction",
                "explain_margin",
                "replay_imported_model_inference",
                "replay_native_inference",
                "run_standalone_rev_classifier",
                "reverse_reversible_trace",
            ]:
                raise AssertionError("handoff did not expose every inference action replay command")
            native_action = next(
                command
                for command in handoff["payload"]["inference_action_review_commands"]
                if command["operation"] == "replay_native_inference"
            )
            if native_action["command"][0] != str(summary_config.runner_bin):
                raise AssertionError("handoff native replay command did not use configured runner binary")
            standalone_action = next(
                command
                for command in handoff["payload"]["inference_action_review_commands"]
                if command["operation"] == "run_standalone_rev_classifier"
            )
            if standalone_action["command"][0] != str(summary_config.reverie_bin):
                raise AssertionError("handoff standalone run command did not use configured Reverie binary")
            handoff_markdown = summary_paths["handoff_markdown"].read_text(encoding="utf-8")
            for snippet in (
                "# Reverie ML Audit Handoff",
                "## Inference Action Review Commands",
                "reproduce_prediction",
                "replay_imported_model_inference",
                "run_standalone_rev_classifier",
                "reverse_reversible_trace",
                "Training-audit verification card",
                "Training-step debug card",
                "Q31 inference explanation card",
                "Native inference audit card",
                "MLP witness proof card",
                "Evaluation-row inference audit card",
                "Invertible coupling proof card",
                "Triangular residual proof card",
                "Reversible preprocess proof card",
                "Reversible inference trace card",
                "Standalone MNIST classifier source",
                "Standalone MNIST classifier run JSON",
                "Reversible inference roundtrip card",
                "Reversible inference roundtrip verification JSON",
                handoff["fingerprint"][:12],
                capsule["fingerprint"][:12],
            ):
                if snippet not in handoff_markdown:
                    raise AssertionError(f"handoff Markdown missing {snippet}")
            executable_handoff = json.loads(json.dumps(handoff))
            for command in executable_handoff["payload"]["inference_action_review_commands"]:
                command["command"] = [
                    sys.executable,
                    "-c",
                    f"print({command['operation']!r})",
                ]
            executable_handoff["fingerprint"] = sha256_json(executable_handoff["payload"])
            summary_paths["handoff_index"].write_text(
                json.dumps(executable_handoff, indent=2) + "\n",
                encoding="utf-8",
            )
            summary_paths["handoff_markdown"].write_text(
                render_handoff_markdown(executable_handoff),
                encoding="utf-8",
            )
            receipt = write_inference_action_review_receipt(summary_config)
            if receipt["kind"] != capsule_verify.ACTION_REVIEW_RECEIPT_KIND:
                raise AssertionError("inference action receipt has the wrong kind")
            if receipt["payload"]["summary"]["passed_count"] != len(action_operations):
                raise AssertionError("inference action receipt did not pass every command")
            if receipt["payload"]["handoff_fingerprint"] != executable_handoff["fingerprint"]:
                raise AssertionError("inference action receipt did not bind the handoff")
            receipt_markdown = summary_paths[
                "inference_action_review_receipt_markdown"
            ].read_text(encoding="utf-8")
            for snippet in (
                "# Reverie Inference Action Review Receipt",
                "reproduce_prediction",
                "replay_imported_model_inference",
                "run_standalone_rev_classifier",
                "reverse_reversible_trace",
                receipt["fingerprint"][:12],
            ):
                if snippet not in receipt_markdown:
                    raise AssertionError(f"inference action receipt Markdown missing {snippet}")
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
            training_receipt = write_training_update_review_receipt(
                summary_config,
                commands=training_commands,
            )
            if training_receipt["kind"] != TRAINING_UPDATE_REVIEW_RECEIPT_KIND:
                raise AssertionError("training update receipt has the wrong kind")
            if training_receipt["payload"]["summary"]["passed_count"] != 3:
                raise AssertionError("training update receipt did not pass every command")
            if (
                training_receipt["payload"]["semantic_fingerprint"]
                != training_update_review_semantic_fingerprint(training_receipt["payload"])
            ):
                raise AssertionError("training update receipt semantic fingerprint did not match payload")
            if training_receipt["payload"]["handoff_fingerprint"] != executable_handoff["fingerprint"]:
                raise AssertionError("training update receipt did not bind the handoff")
            training_receipt_markdown = summary_paths[
                "training_update_review_receipt_markdown"
            ].read_text(encoding="utf-8")
            for snippet in (
                "# Reverie Training Update Review Receipt",
                "replay_training_lineage",
                "reverse_training_update",
                training_receipt["fingerprint"][:12],
            ):
                if snippet not in training_receipt_markdown:
                    raise AssertionError(f"training update receipt Markdown missing {snippet}")
            bad_reports = valid_pipeline_reports_for_self_test()
            bad_reports["audit_scan"]["summary"]["witness_mismatches"] = 1
            bad_summary = build_pipeline_summary(summary_config, summary_paths, bad_reports)
            failed = {
                check["metric"]
                for check in bad_summary["gates"]["checks"]
                if not check["passed"]
            }
            if "training_audit_witness_mismatches" not in failed:
                raise AssertionError("bad synthetic pipeline summary did not fail witness gate")
            tight_config = PipelineConfig(
                out_dir=summary_config.out_dir,
                sample_limit=summary_config.sample_limit,
                audit_step=summary_config.audit_step,
                evaluation_row=summary_config.evaluation_row,
                runner_bin=summary_config.runner_bin,
                reverie_bin=summary_config.reverie_bin,
                max_reverse_train_elapsed_ratio=0.5,
            )
            tight_summary = build_pipeline_summary(tight_config, summary_paths, reports)
            tight_failed = {
                check["metric"]
                for check in tight_summary["gates"]["checks"]
                if not check["passed"]
            }
            if "reverse_check_elapsed_ratio" not in tight_failed:
                raise AssertionError("tight synthetic pipeline summary did not fail reverse ratio gate")
            try:
                validate_config(
                    PipelineConfig(
                        out_dir=Path(directory),
                        sample_limit=1,
                        audit_step=1,
                        evaluation_row=0,
                        runner_bin=None,
                    )
                )
            except PipelineError as error:
                if "--audit-step" not in str(error):
                    raise AssertionError(f"wrong invalid-step failure: {error}") from error
            else:
                raise AssertionError("invalid audit step did not fail")
            try:
                validate_config(
                    PipelineConfig(
                        out_dir=Path(directory),
                        sample_limit=1,
                        audit_step=0,
                        evaluation_row=0,
                        runner_bin=None,
                        max_trace_model_ratio=float("nan"),
                    )
                )
            except PipelineError as error:
                if "--max-trace-model-ratio" not in str(error):
                    raise AssertionError(f"wrong invalid-threshold failure: {error}") from error
            else:
                raise AssertionError("invalid threshold did not fail")
    except (AssertionError, PipelineError) as error:
        print(f"error: self-test failed: {error}", file=sys.stderr)
        return 1
    print("ok: MNIST ML audit pipeline self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    config = PipelineConfig(
        out_dir=args.out_dir,
        sample_limit=args.sample_limit,
        audit_step=args.audit_step,
        evaluation_row=args.evaluation_row,
        runner_bin=args.runner_bin,
        reverie_bin=args.reverie_bin,
        min_train_accuracy=args.min_train_accuracy,
        min_eval_accuracy=args.min_eval_accuracy,
        min_audit_accuracy=args.min_audit_accuracy,
        min_model_evaluation_accuracy=args.min_model_evaluation_accuracy,
        min_reference_accuracy=args.min_reference_accuracy,
        max_witness_mismatches=args.max_witness_mismatches,
        max_trace_model_ratio=args.max_trace_model_ratio,
        max_witness_model_ratio=args.max_witness_model_ratio,
        max_reverse_train_elapsed_ratio=args.max_reverse_train_elapsed_ratio,
        max_replay_payload_bytes=args.max_replay_payload_bytes,
        audit_step_strategy=args.audit_step_strategy,
        evaluation_row_strategy=args.evaluation_row_strategy,
        dry_run=args.dry_run,
    )
    try:
        validate_config(config)
        if args.write_training_update_receipt_only:
            receipt = write_training_update_review_receipt(config)
            paths = out_paths(config.out_dir)
            print(f"training update receipt: {paths['training_update_review_receipt']}")
            print(
                "training update commands: "
                f"{receipt['payload']['summary']['passed_count']}/"
                f"{receipt['payload']['summary']['command_count']}"
            )
            return 0
        run_pipeline(config)
    except (PipelineError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
