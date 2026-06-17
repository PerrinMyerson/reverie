#!/usr/bin/env python3
"""Validate that CI keeps the core Reverie quality gates wired."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
BENCHMARK_CHECKER = REPO_ROOT / "scripts" / "check_benchmark_artifact.py"
SUMMARY_SCRIPT = REPO_ROOT / "scripts" / "summarize_janus_performance.py"
MNIST_ML_SUMMARY_SCRIPT = REPO_ROOT / "scripts" / "summarize_mnist_ml_profile.py"
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_janus_performance.py"
MATERIALIZED_SOURCE_CHECKER = REPO_ROOT / "scripts" / "check_materialized_sources.py"
CRITERION_CHECKER = REPO_ROOT / "scripts" / "check_criterion_docs.py"
EXPLAIN_CHECKER = REPO_ROOT / "scripts" / "check_explain_json.py"
CLI_MAIN = REPO_ROOT / "crates" / "reverie-cli" / "src" / "main.rs"
MNIST_LINEAR_RUNNER = (
    REPO_ROOT / "crates" / "reverie-cli" / "src" / "bin" / "reverie-mnist-linear.rs"
)
CLI_TESTS = REPO_ROOT / "crates" / "reverie-cli" / "tests" / "cli.rs"
CLI_DOCS = REPO_ROOT / "docs" / "cli.md"
INVERTIBLE_COUPLING_CHECKER = REPO_ROOT / "scripts" / "check_invertible_coupling.py"
MNIST_ML_PROFILE_CHECKER = REPO_ROOT / "scripts" / "check_mnist_ml_profile.py"
ML_NORTH_STAR_GATE = REPO_ROOT / "scripts" / "check_ml_north_star.py"
MNIST_ML_AUDIT_PIPELINE = REPO_ROOT / "scripts" / "run_mnist_ml_audit_pipeline.py"
MODEL_IMPORT_CHECKER = REPO_ROOT / "scripts" / "check_model_import_docs.py"
MODEL_CAPSULE_VERIFIER = REPO_ROOT / "scripts" / "verify_model_capsule.py"
Q31_INFERENCE_CHECKER = REPO_ROOT / "scripts" / "check_q31_inference.py"
Q31_MLP_WITNESS_CHECKER = REPO_ROOT / "scripts" / "check_q31_mlp_witness.py"
REVERSIBLE_INFERENCE_TRACE_CHECKER = REPO_ROOT / "scripts" / "check_reversible_inference_trace.py"
REVERSIBLE_PREPROCESS_CHECKER = REPO_ROOT / "scripts" / "check_reversible_preprocess.py"
TRIANGULAR_RESIDUAL_CHECKER = REPO_ROOT / "scripts" / "check_triangular_residual.py"
Q31_EXPORTER = REPO_ROOT / "scripts" / "export_q31_linear_model.py"
Q31_MLP_VARS_EXPORTER = REPO_ROOT / "scripts" / "export_q31_mlp_vars.py"
TORCH_STATE_DICT_EXTRACTOR = REPO_ROOT / "scripts" / "extract_torch_linear_state_dict.py"

REQUIRED_SNIPPETS = {
    "format gate": "cargo fmt --check",
    "source materialization gate": "python3 scripts/check_materialized_sources.py",
    "workspace clippy gate": "cargo clippy --workspace --all-targets -- -D warnings",
    "workspace test gate": "cargo test --workspace",
    "criterion bench compile gate": "cargo bench -p reverie-interp --bench execution --no-run",
    "criterion checker py_compile": "scripts/check_criterion_docs.py",
    "benchmark harness workload listing": "python3 scripts/bench_jana_vs_reverie.py --list-workloads",
    "benchmark harness self-test": "python3 scripts/bench_jana_vs_reverie.py --self-test",
    "benchmark artifact checker self-test": "python3 scripts/check_benchmark_artifact.py --self-test",
    "benchmark summary renderer self-test": "python3 scripts/summarize_janus_performance.py --self-test",
    "MNIST ML profile summary renderer self-test": "python3 scripts/summarize_mnist_ml_profile.py --self-test",
    "Janus performance wrapper self-test": "python3 scripts/verify_janus_performance.py --self-test",
    "materialized source checker self-test": "python3 scripts/check_materialized_sources.py --self-test",
    "benchmark docs checker self-test": "python3 scripts/check_benchmark_docs.py --self-test",
    "evaluation corpus checker self-test": "python3 scripts/check_evaluation_corpus.py --self-test",
    "invertible coupling checker self-test": "python3 scripts/check_invertible_coupling.py --self-test",
    "MNIST ML profile checker self-test": "python3 scripts/check_mnist_ml_profile.py --self-test",
    "ML north-star gate self-test": "python3 scripts/check_ml_north_star.py --self-test",
    "MNIST ML audit pipeline self-test": "python3 scripts/run_mnist_ml_audit_pipeline.py --self-test",
    "model import docs checker self-test": "python3 scripts/check_model_import_docs.py --self-test",
    "model capsule verifier self-test": "python3 scripts/verify_model_capsule.py --self-test",
    "Q31 inference checker self-test": "python3 scripts/check_q31_inference.py --self-test",
    "Q31 MLP witness checker self-test": "python3 scripts/check_q31_mlp_witness.py --self-test",
    "reversible inference trace checker self-test": "python3 scripts/check_reversible_inference_trace.py --self-test",
    "reversible preprocess checker self-test": "python3 scripts/check_reversible_preprocess.py --self-test",
    "triangular residual checker self-test": "python3 scripts/check_triangular_residual.py --self-test",
    "Q31 linear model exporter self-test": "python3 scripts/export_q31_linear_model.py --self-test",
    "Q31 MLP vars-json exporter self-test": "python3 scripts/export_q31_mlp_vars.py --self-test",
    "PyTorch state_dict extractor self-test": "python3 scripts/extract_torch_linear_state_dict.py --self-test",
    "Jana audit checker self-test": "python3 scripts/check_jana_audit.py --self-test",
    "Janus feature audit checker self-test": "python3 scripts/check_janus_feature_audit.py --self-test",
    "benchmark artifact checker": "python3 scripts/check_benchmark_artifact.py",
    "geometric mean performance floor": "--min-geomean-speedup 3.0",
    "checked markdown benchmark summary": "--expect-markdown-summary benchmarks/results/jana-vs-reverie-smoke.md",
    "benchmark docs checker": "python3 scripts/check_benchmark_docs.py",
    "criterion docs checker": "python3 scripts/check_criterion_docs.py",
    "evaluation corpus checker": "python3 scripts/check_evaluation_corpus.py",
    "model import docs checker": "python3 scripts/check_model_import_docs.py",
    "explain JSON checker": "python3 scripts/check_explain_json.py",
    "Janus feature audit checker": "python3 scripts/check_janus_feature_audit.py",
    "Jana audit checker": "python3 scripts/check_jana_audit.py",
    "manual Janus performance job": "janus-performance:",
    "checked Janus performance wrapper": "scripts/verify_janus_performance.py",
    "benchmark artifact upload": "actions/upload-artifact@v4",
    "MNIST ML profile checker py_compile": "scripts/check_mnist_ml_profile.py",
    "ML north-star gate py_compile": "scripts/check_ml_north_star.py",
    "MNIST ML audit pipeline py_compile": "scripts/run_mnist_ml_audit_pipeline.py",
    "MNIST ML profile summary py_compile": "scripts/summarize_mnist_ml_profile.py",
    "model import checker py_compile": "scripts/check_model_import_docs.py",
    "model capsule verifier py_compile": "scripts/verify_model_capsule.py",
    "invertible coupling checker py_compile": "scripts/check_invertible_coupling.py",
    "Q31 inference checker py_compile": "scripts/check_q31_inference.py",
    "Q31 MLP witness checker py_compile": "scripts/check_q31_mlp_witness.py",
    "reversible inference trace checker py_compile": "scripts/check_reversible_inference_trace.py",
    "reversible preprocess checker py_compile": "scripts/check_reversible_preprocess.py",
    "triangular residual checker py_compile": "scripts/check_triangular_residual.py",
    "Q31 linear model exporter py_compile": "scripts/export_q31_linear_model.py",
    "Q31 MLP vars-json exporter py_compile": "scripts/export_q31_mlp_vars.py",
    "PyTorch state_dict extractor py_compile": "scripts/extract_torch_linear_state_dict.py",
}

BENCHMARK_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "bool is not numeric": "not isinstance(value, bool)",
    "Python 3.9 numeric isinstance": "isinstance(value, (int, float))",
    "synthetic artifact fixture": "valid_synthetic_artifact",
    "selected workload rejection": "selected_workloads must match benchmark row order",
    "command binary rejection": "does not match reverie binary path",
    "Markdown summary consistency": "stale Markdown direction summary",
}

SUMMARY_SCRIPT_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "finite numeric guard": "math.isfinite(value)",
    "Python 3.9 numeric isinstance": "isinstance(value, (int, float))",
    "direction count rendering": "Directions: ",
    "direction fallback rendering": "direction fallback",
    "workload failure rendering": "Failing workloads:",
    "observed speedup failure rendering": "Minimum observed speedup failure",
    "synthetic artifact fixture": "synthetic_summary_artifact",
}

VERIFY_SCRIPT_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "runs validation": "--runs must be positive",
    "warmup validation": "--warmup must be non-negative",
    "finite threshold validation": "math.isfinite(args.min_speedup)",
    "timeout validation": "--command-timeout must be finite and positive",
    "duplicate workload rejection": "duplicate --only workload(s)",
    "unknown workload rejection": "unknown benchmark workload(s)",
}

BENCHMARK_HARNESS_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "Python 3.9 type alias compatibility": "SideSpec = Union[CommandSpec, list[CommandSpec]]",
    "Python 3.9 optional typing compatibility": "Optional[float]",
    "finite threshold validation": "math.isfinite(value)",
    "duplicate workload rejection": "duplicate --only workload(s)",
    "unknown workload rejection": "unknown benchmark workload(s)",
    "only workload order preservation": "--only benchmark order was not preserved",
    "workload direction metadata validation": "workload direction metadata mismatch",
}

MATERIALIZED_SOURCE_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "quarantine audit option": "--include-quarantine",
    "dataless placeholder detection": "APFS dataless placeholders",
    "fast inspection timeout": "timeout=5.0",
    "quarantine exclusion": "dataless-quarantine",
    "quarantine selection self-test": "quarantined source file not selected when requested",
}

BENCHMARK_DOCS_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "manifest order check": "manifest workload order does not match harness order",
    "manifest missing workload rejection": "manifest missing workload(s)",
    "gate and direction docs detail": "benchmark docs missing required gate/direction detail",
}

EVALUATION_CORPUS_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "duplicate negative rejection": "duplicate negative corpus example(s)",
    "duplicate expressiveness rejection": "duplicate expressiveness corpus example(s)",
    "overlap rejection": "examples listed as both positive and negative",
    "ergonomic syntax coverage": "missing required ergonomic syntax coverage",
    "ergonomic docs coverage": "missing required docs/",
}

MNIST_ML_PROFILE_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "run report kind": "reverie_mnist_linear_q31",
    "artifact comparison kind": "reverie_mnist_linear_q31_artifact_comparison",
    "MLP witness report kind": "reverie_q31_mlp_witness_reference",
    "MLP proof claim": "deterministic_q31_mlp_witness_replay",
    "MLP dataset loop validator": "validate_mlp_dataset_loops",
    "MLP dataset loop capsule field": "mlp_dataset_loops",
    "MLP witness proof validation": "validate_mlp_witness_store_proof",
    "MLP witness proof metric": "witness_proof_fingerprint",
    "invertible coupling report kind": "reverie_q31_invertible_coupling_reference",
    "invertible coupling proof claim": "deterministic_q31_invertible_coupling_replay",
    "triangular residual report kind": "reverie_q31_triangular_residual_reference",
    "triangular residual proof claim": "deterministic_q31_triangular_residual_replay",
    "Q31 reference inference kind": "reverie_q31_linear_reference_inference",
    "Q31 reference evaluation kind": "reverie_q31_linear_reference_evaluation",
    "Q31 reference attribution": "top_margin_contributions",
    "Q31 reference evaluation rows": "top_low_margin",
    "audit step report kind": "reverie_mnist_linear_q31_audit_step",
    "audit scan report kind": "reverie_mnist_linear_q31_audit_scan",
    "audit verification report kind": "reverie_mnist_linear_q31_audit_verification",
    "step verification kind": "reverie_mnist_linear_q31_step_verification",
    "training step proof claim": "deterministic_q31_training_step_replay",
    "audit step debug contract": "step_backward_from_model_update",
    "inference explanation contract": "q31_inference_prediction_explanation",
    "native inference report kind": "reverie_mnist_linear_q31_model_inference_audit",
    "inference verification kind": "reverie_mnist_linear_q31_inference_verification",
    "model evaluation report kind": "reverie_mnist_linear_q31_model_evaluation",
    "model evaluation scan kind": "reverie_mnist_linear_q31_model_evaluation_scan",
    "model evaluation verification kind": "reverie_mnist_linear_q31_model_evaluation_verification",
    "pipeline summary kind": "reverie_mnist_ml_audit_pipeline_summary",
    "pipeline manifest kind": "reverie_mnist_ml_audit_pipeline",
    "inference proof claim": "deterministic_q31_inference_replay",
    "reverse-check gate": "--require-reverse-check",
    "peak RSS gate": "--require-peak-rss",
    "ML profile gate": "--require-ml-profile",
    "audit contract gate": "--require-audit-contract",
    "audit contract claim": "reversible_inspectable_deterministic_q31_ml_kernel",
    "audit contract fingerprint coverage": "fingerprint_coverage",
    "audit contract proof provenance coverage": "proof_or_provenance_fingerprints",
    "finite numeric guard": "math.isfinite",
    "bool is not numeric": "isinstance(value, bool)",
    "estimated payload check": "estimated_payload_bytes",
    "trace payload ratio check": "trace_to_model_payload_ratio",
    "witness payload ratio check": "witness_to_model_payload_ratio",
    "logical/file ratio check": "logical_to_file_ratio",
    "recomputed update payload check": "recomputed_update_payload_bytes",
    "pipeline gate policy validation": "validate_pipeline_gate_policy",
    "pipeline summary validation": "validate_pipeline_summary_report",
    "pipeline manifest validation": "validate_pipeline_manifest_report",
    "model capsule validation": "validate_model_capsule_report",
    "model capsule inference action validator": "validate_inference_action_contract",
    "model capsule inference action contract": '"action_contract"',
    "model capsule kind": "reverie_reversible_ml_model_capsule",
    "model capsule schema": "reverie_reversible_ml_model_capsule_v1",
    "pipeline MLP witness validation": "validate_pipeline_mlp_witness_metrics",
    "pipeline coupling validation": "validate_pipeline_coupling_metrics",
    "pipeline residual validation": "validate_pipeline_residual_metrics",
    "pipeline preprocess validation": "validate_pipeline_preprocess_metrics",
    "pipeline evidence validation": "validate_pipeline_evidence_index",
    "pipeline evidence file verification": "verify_pipeline_file_evidence",
    "pipeline evidence CLI option": "--verify-pipeline-files",
    "pipeline evidence hashing": "hashlib.sha256",
    "pipeline report-set consistency": "validate_report_set_consistency",
    "model capsule summary consistency": "validate_model_capsule_matches_summary",
    "manifest summary consistency": "validate_manifest_matches_summary",
    "pipeline claims validation": "validate_pipeline_claims",
    "pipeline claims kind": "reverie_reversible_ml_audit_claims",
    "pipeline claims catalog": "PIPELINE_CLAIMS",
    "pipeline integrity claim full coverage": "must cite every digest-indexed evidence file",
    "pipeline capability map kind": "reverie_mnist_ml_capability_map",
    "pipeline capability map validation": "validate_pipeline_ml_capability_map",
    "pipeline capability map gate": "ml_roadmap_capability_map_complete",
    "pipeline goal readiness kind": "reverie_reversible_ml_goal_readiness",
    "pipeline goal readiness validation": "validate_pipeline_ml_goal_readiness",
    "pipeline goal readiness gate": "ml_goal_readiness_complete",
    "pipeline goal readiness catalog": "PIPELINE_GOAL_READINESS",
    "pipeline recompute frontier kind": "reverie_mnist_ml_recompute_frontier",
    "pipeline recompute frontier validation": "validate_pipeline_recompute_frontier",
    "pipeline recompute frontier gate": "recompute_frontier_complete",
    "pipeline scaling projection kind": "reverie_mnist_ml_scaling_projection",
    "pipeline scaling projection validation": "validate_pipeline_scaling_projection",
    "pipeline scaling projection gate": "scaling_projection_complete",
    "pipeline inference trace profile kind": "reverie_mnist_ml_inference_trace_profile",
    "pipeline inference trace profile validation": "validate_pipeline_inference_trace_profile",
    "pipeline inference trace profile gate": "inference_trace_profile_complete",
    "pipeline gate metrics": "PIPELINE_GATE_METRICS",
    "pipeline step debug gate": "training_step_debug_contract",
    "pipeline update ledger gate": "training_update_ledger_fingerprints",
    "pipeline cause ledger metric": "cause_ledger_fingerprint",
    "pipeline lineage replay gate": "training_audit_lineage_replay",
    "pipeline step selection gate": "training_step_selection_traceable",
    "pipeline audit step strategy": "audit_step_strategy",
    "pipeline evaluation row strategy": "evaluation_row_strategy",
    "pipeline reverse cost validation": "validate_pipeline_reverse_check_cost",
    "pipeline reverse cost gate": "reverse_check_cost_measured",
    "pipeline reverse ratio policy": "max_reverse_train_elapsed_ratio",
    "pipeline reverse ratio gate": "reverse_check_elapsed_ratio",
    "pipeline inference explanation gate": "native_inference_explanation_contract",
    "pipeline row explanation gate": "evaluation_row_inference_explanation_contract",
    "pipeline reference inference parity gate": "q31_reference_matches_native_inference",
    "pipeline V6 scorecard gate": "v6_scorecard_complete",
    "reversible preprocess report kind": "reverie_q31_reversible_preprocess_reference",
    "triangular residual gate": "triangular_residual_replay",
    "reversible preprocess proof claim": "deterministic_q31_reversible_preprocess_replay",
    "reversible preprocess gate": "reversible_preprocess_replay",
    "reversible inference trace report kind": "reverie_q31_reversible_inference_trace_reference",
    "reversible inference trace proof claim": "deterministic_q31_reversible_inference_trace",
    "reversible inference trace attribution validation": "validate_inference_trace_attribution",
    "reversible inference trace top logits validation": "validate_inference_trace_top_logits",
    "training cause ledger validation": "validate_cause_ledger",
    "training lineage ledger validation": "validate_lineage_ledger",
    "pipeline inference trace validation": "validate_pipeline_reversible_inference_trace_metrics",
    "pipeline reversible inference trace gate": "reversible_inference_trace_replay",
}

MNIST_ML_SUMMARY_SCRIPT_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "imports validator": "import check_mnist_ml_profile as profile_check",
    "validates before rendering": "profile_check.validate_report",
    "validates report-set consistency": "validate_report_set_consistency",
    "model capsule section": "## Model Capsules",
    "model capsule renderer": "render_model_capsule_rows",
    "inference action contract section": "## Inference Action Contract",
    "inference action contract renderer": "render_inference_action_contract_rows",
    "north-star readiness section": "## North-Star Readiness",
    "north-star readiness renderer": "render_goal_readiness_rows",
    "ML capability map section": "## ML Capability Map",
    "V6 scorecard section": "## V6 Scorecard",
    "recompute frontier section": "## Recompute Frontier",
    "scaling projection section": "## Scaling Projection",
    "inference trace profile section": "## Inference Trace Profile",
    "run report section": "## Run Reports",
    "artifact profile section": "## Artifact Profiles",
    "audit contract section": "## Audit Contract",
    "audit contract gate": "--require-audit-contract",
    "artifact rows section": "## Artifact Rows",
    "MLP witness section": "## MLP Witness Reports",
    "MLP witness proof column": "Witness proof",
    "MLP dataset loop column": "Dataset loop",
    "triangular residual section": "## Triangular Residual Reports",
    "triangular residual claim": "deterministic_q31_triangular_residual_replay",
    "reversible preprocess section": "## Reversible Preprocess Reports",
    "reversible inference trace section": "## Reversible Inference Trace Reports",
    "Q31 reference inference section": "## Q31 Reference Inference",
    "Q31 reference evaluation section": "## Q31 Reference Evaluation",
    "training audit verification section": "## Training Audit Verification",
    "training audit scan section": "## Training Audit Scan",
    "training audit step section": "## Training Audit Step",
    "training step debug renderer": "render_training_step_debug_markdown",
    "training step debug title": "# Reverie Training Step Debug",
    "training step verification section": "## Training Step Verification",
    "native inference section": "## Native Inference Reports",
    "inference verification section": "## Inference Verification",
    "model evaluation section": "## Model Evaluation",
    "model evaluation scan section": "## Model Evaluation Scan",
    "model evaluation verification section": "## Model Evaluation Verification",
    "peak RSS column": "Peak RSS",
    "trace bytes column": "Trace bytes",
    "attribution column": "Attribution",
    "contribution ledger column": "Contribution ledger",
    "reference ledger column": "Reference ledger",
    "forward logits column": "Forward logits",
    "runner-up column": "Runner-up",
    "lowest margin column": "Lowest margin",
    "witness mismatches column": "Witness mismatches",
    "model delta column": "Model delta",
    "update ledger column": "Update ledger",
    "cause ledger column": "Cause ledger",
    "lineage ledger column": "Lineage ledger",
    "transition ledger column": "Transition ledger",
    "final chain column": "Final chain",
    "debug contract column": "Debug contract",
    "explanation column": "Explanation",
    "reverse/train column": "Reverse/train",
    "sources checked column": "Sources checked",
    "low-margin rows column": "Low-margin rows",
    "recompute steps column": "Recompute steps",
    "projected replay column": "Projected replay bytes",
    "recomputed update column": "Recomputed update bytes",
    "update/witness column": "Update/witness",
    "artifact fingerprint column": "Source fingerprints",
    "scorecard max replay column": "Max replay bytes",
    "scorecard reversible trace column": "Rev trace",
    "scorecard Q31 reference column": "Q31 ref",
    "scorecard preprocess column": "Preprocess",
    "training step scan evidence column": "Scan evidence",
    "evaluation row evidence column": "Row evidence",
}

MNIST_ML_AUDIT_PIPELINE_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "runner binary option": "--runner-bin",
    "manifest kind": "reverie_mnist_ml_audit_pipeline",
    "handoff kind": "reverie_mnist_ml_audit_handoff",
    "summary kind": "reverie_mnist_ml_audit_pipeline_summary",
    "summary artifact": "pipeline-summary.json",
    "model capsule artifact": "model-capsule.json",
    "model capsule profile artifact": "model-capsule-profile.md",
    "training audit verification markdown artifact": "training-audit-verification.md",
    "training step debug artifact": "training-step-debug.md",
    "training step markdown option": "--markdown-output",
    "model capsule verification artifact": "model-capsule-verification.json",
    "model capsule verification markdown artifact": "model-capsule-verification.md",
    "handoff index artifact": "ml-audit-handoff.json",
    "handoff markdown artifact": "ml-audit-handoff.md",
    "inference action receipt artifact": "DEFAULT_ACTION_REVIEW_RECEIPT_NAME",
    "inference action receipt markdown artifact": "DEFAULT_ACTION_REVIEW_RECEIPT_MARKDOWN_NAME",
    "training update receipt artifact": "DEFAULT_TRAINING_UPDATE_REVIEW_RECEIPT_NAME",
    "training update receipt markdown artifact": "DEFAULT_TRAINING_UPDATE_REVIEW_RECEIPT_MARKDOWN_NAME",
    "training update receipt kind": "reverie_training_update_review_receipt",
    "training update receipt semantics": "reverie_training_update_review_semantics_v1",
    "training update receipt only option": "--write-training-update-receipt-only",
    "model capsule fingerprint": "model_capsule_fingerprint",
    "model capsule builder": "build_model_capsule",
    "model capsule profile writer": "write_model_capsule_profile",
    "training step debug writer": "write_training_step_debug_markdown",
    "handoff index writer": "write_handoff_index",
    "handoff index renderer": "render_handoff_markdown",
    "handoff index verifier": "validate_handoff_index",
    "inference action receipt writer": "write_inference_action_review_receipt",
    "training update receipt writer": "write_training_update_review_receipt",
    "training update receipt renderer": "render_training_update_review_receipt_markdown",
    "handoff inference command builder": "inference_action_review_commands",
    "handoff inference command section": "## Inference Action Review Commands",
    "handoff verifier requirement": "require_handoff=True",
    "model capsule verification writer": "write_model_capsule_verification",
    "audit verification artifact": "training-audit-verification.json",
    "audit verification command": "--verify-audit",
    "evidence index": "evidence_index",
    "evidence hasher": "sha256_file",
    "pipeline claims": "pipeline_claims",
    "pipeline claims kind": "reverie_reversible_ml_audit_claims",
    "ML capability map": "pipeline_ml_capability_map",
    "ML capability map kind": "reverie_mnist_ml_capability_map",
    "ML capability gate": "ml_roadmap_capability_map_complete",
    "ML goal readiness": "pipeline_ml_goal_readiness",
    "ML goal readiness kind": "reverie_reversible_ml_goal_readiness",
    "ML goal readiness gate": "ml_goal_readiness_complete",
    "north-star readiness claim": "north_star_reversible_ml_kernels",
    "recompute frontier": "recompute_frontier_metrics",
    "recompute frontier kind": "reverie_mnist_ml_recompute_frontier",
    "recompute frontier gate": "recompute_frontier_complete",
    "scaling projection": "scaling_projection_metrics",
    "scaling projection kind": "reverie_mnist_ml_scaling_projection",
    "scaling projection gate": "scaling_projection_complete",
    "inference trace profile": "inference_trace_profile_metrics",
    "inference trace profile kind": "reverie_mnist_ml_inference_trace_profile",
    "inference trace profile gate": "inference_trace_profile_complete",
    "debug training claim": "debug_training_update",
    "debug training gate": "training_step_debug_contract",
    "debug update ledger gate": "training_update_ledger_fingerprints",
    "lineage replay gate": "training_audit_lineage_replay",
    "debug training selection gate": "training_step_selection_traceable",
    "reverse cost gate": "reverse_check_cost_measured",
    "reverse ratio option": "--max-reverse-train-elapsed-ratio",
    "reverse ratio gate": "reverse_check_elapsed_ratio",
    "audit step strategy option": "--audit-step-strategy",
    "audit step lowest-margin strategy": "lowest-margin",
    "audit step top-suspicious strategy": "top-suspicious",
    "evaluation row strategy option": "--evaluation-row-strategy",
    "evaluation row top-incorrect strategy": "top-incorrect",
    "inference explanation gate": "native_inference_explanation_contract",
    "row explanation gate": "evaluation_row_inference_explanation_contract",
    "V6 scorecard gate": "v6_scorecard_complete",
    "deterministic inference claim": "deterministic_q31_inference",
    "invertible layer claim": "invertible_layer_without_witness",
    "generic reverie binary option": "--reverie-bin",
    "training audit bundle": "--audit-output",
    "training audit scan": "--scan-audit",
    "step replay verification": "--verify-step",
    "model export": "--export-model",
    "sample export": "--export-samples",
    "native inference": "--inspect-model-inference",
    "native inference audit Markdown artifact": "native-inference-audit.md",
    "inference verification": "--verify-inference",
    "native inference verification Markdown artifact": "native-inference-verification.md",
    "model evaluation": "--evaluate-model",
    "evaluation verification": "--verify-evaluation",
    "evaluation scan": "--scan-evaluation",
    "evaluation row inspection": "--inspect-evaluation",
    "evaluation row inference audit Markdown artifact": "model-evaluation-row-inference-audit.md",
    "MLP witness run": "examples/mnist_mlp_witness.rev",
    "MLP witness checker": "scripts/check_q31_mlp_witness.py",
    "MLP dataset loop summary metric": '"dataset_loops": mlp_witness["dataset_loops"]',
    "MLP witness proof fingerprint": "witness_proof_fingerprint",
    "MLP witness markdown artifact": "mlp-witness.md",
    "MLP witness markdown path": "mlp_witness_markdown",
    "invertible coupling run": "examples/invertible_coupling.rev",
    "invertible coupling checker": "scripts/check_invertible_coupling.py",
    "invertible coupling markdown artifact": "invertible-coupling.md",
    "invertible coupling markdown path": "invertible_coupling_markdown",
    "triangular residual run": "examples/triangular_residual.rev",
    "triangular residual checker": "scripts/check_triangular_residual.py",
    "triangular residual seed": "residual_final_vars",
    "triangular residual markdown artifact": "triangular-residual.md",
    "triangular residual markdown path": "triangular_residual_markdown",
    "reversible preprocess run": "examples/reversible_preprocess.rev",
    "reversible preprocess checker": "scripts/check_reversible_preprocess.py",
    "reversible preprocess seed": "preprocess_final_vars",
    "reversible preprocess markdown artifact": "reversible-preprocess.md",
    "reversible preprocess markdown path": "reversible_preprocess_markdown",
    "reversible inference trace run": "examples/reversible_inference_trace.rev",
    "reversible inference trace checker": "scripts/check_reversible_inference_trace.py",
    "reversible inference trace seed": "inference_trace_final_vars",
    "reversible inference trace report": "reversible-inference-trace-report.json",
    "reversible inference trace markdown artifact": "reversible-inference-trace.md",
    "reversible inference trace markdown path": "reversible_inference_trace_markdown",
    "reversible inference trace replay gate": "reversible_inference_trace_replay",
    "roundtrip proof bundle key": "inference_trace_roundtrip_proof",
    "roundtrip verification bundle key": "inference_trace_roundtrip_verification",
    "roundtrip proof command": "roundtrip",
    "roundtrip verifier command": "verify-roundtrip",
    "roundtrip markdown option": "--markdown-output",
    "roundtrip proof artifact": "reversible-inference-roundtrip-proof.json",
    "roundtrip verification artifact": "reversible-inference-roundtrip-verification.json",
    "roundtrip markdown artifact": "reversible-inference-roundtrip.md",
    "artifact comparison": "--compare-artifacts",
    "reference checker": "scripts/check_q31_inference.py",
    "reference inference explanation artifact": "q31-reference-inference.md",
    "invertible coupling explanation artifact": "invertible-coupling.md",
    "reversible preprocess explanation artifact": "reversible-preprocess.md",
    "profile validator": "scripts/check_mnist_ml_profile.py",
    "profile renderer": "scripts/summarize_mnist_ml_profile.py",
    "audit contract gate": "--require-audit-contract",
    "training accuracy gate": "--min-train-accuracy",
    "trace ratio gate": "--max-trace-model-ratio",
    "witness mismatch gate": "training_audit_witness_mismatches",
    "native/reference inference parity gate": "q31_reference_matches_native_inference",
    "native/reference parity gate": "q31_reference_matches_native_evaluation",
    "evaluation row replay gate": "evaluation_row_inference_replay",
    "MLP witness replay gate": "mlp_witness_replay",
    "invertible coupling replay gate": "invertible_coupling_replay",
    "triangular residual replay gate": "triangular_residual_replay",
    "reversible preprocess replay gate": "reversible_preprocess_replay",
    "written artifact validator": "validate_written_pipeline_artifacts",
    "deliberate failing summary self-test": "bad synthetic pipeline summary did not fail witness gate",
}

MODEL_CAPSULE_VERIFIER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "machine-readable JSON option": "--json",
    "machine-readable output option": "--output",
    "trust certificate markdown option": "--markdown-output",
    "trust certificate markdown validation option": "--require-verification-markdown",
    "handoff validation option": "--require-handoff",
    "handoff JSON option": "--handoff",
    "handoff Markdown option": "--handoff-markdown",
    "handoff report kind": "reverie_mnist_ml_audit_handoff",
    "handoff schema": "reverie_mnist_ml_audit_handoff_v1",
    "handoff renderer": "render_handoff_markdown",
    "handoff artifact verifier": "validate_handoff_artifact",
    "handoff inference command validator": "validate_inference_action_review_commands",
    "handoff inference command markdown": "## Inference Action Review Commands",
    "handoff action command replay option": "--run-inference-action-commands",
    "handoff action command timeout option": "--action-command-timeout",
    "handoff action receipt JSON option": "--action-command-receipt-output",
    "handoff action receipt Markdown option": "--action-command-receipt-markdown",
    "handoff action command runner": "run_inference_action_review_commands",
    "handoff action receipt kind": "reverie_inference_action_review_receipt",
    "handoff action receipt schema": "reverie_inference_action_review_receipt_v1",
    "handoff action receipt semantics": "reverie_inference_action_review_semantics_v1",
    "handoff action receipt renderer": "render_inference_action_review_receipt_markdown",
    "triangular residual review card": "triangular_residual_markdown",
    "trust certificate markdown validation path": "--verification-markdown",
    "trust certificate markdown renderer": "render_verification_markdown",
    "trust certificate markdown metadata": '"verification_markdown"',
    "trust certificate saved markdown metadata": '"saved": saved_markdown',
    "trust certificate markdown SHA": '"sha256": sha256_text(markdown)',
    "trust certificate stale markdown self-test": "verification Markdown is stale",
    "verification report kind": "reverie_model_capsule_verification",
    "trust certificate kind": "reverie_model_capsule_trust_certificate",
    "trust certificate fingerprint": '"fingerprint": profile_check.sha256_json(payload_certificate)',
    "trust certificate pass check": '"trust_certificate_passed": certificate_passed',
    "verification status follows certificate": '"passed": certificate_passed',
    "trust certificate dataset loops": '"mlp_dataset_loops": witnesses["mlp_dataset_loops"]',
    "trust certificate action contract": '"action_contract": inference["action_contract"]',
    "trust certificate action contract markdown": "## Inference Action Contract",
    "trust certificate self-test": "trust-certificate fingerprint did not match payload",
    "trust certificate failure self-test": "failed trust certificate did not fail verification report",
    "trust certificate markdown self-test": "# Reverie Model Capsule Verification",
    "trust certificate markdown hash self-test": "saved verification Markdown hash did not match JSON metadata",
    "default capsule artifact": "model-capsule.json",
    "default capsule profile": "model-capsule-profile.md",
    "file evidence verification": "verify_pipeline_file_evidence",
    "report-set consistency verification": "validate_report_set_consistency",
    "profile freshness validation": "validate_capsule_profile",
    "profile SHA-256 metadata": '"sha256": sha256_text(markdown)',
    "capsule fingerprint output": "capsule: {result['fingerprint']}",
    "gate count output": "gates: {result['gate_passed']}/{result['gate_total']}",
    "JSON witness proof output": '"witness_proof": result["witness_proof"]',
    "stale profile negative test": "stale capsule profile",
    "stale manifest negative test": "stale manifest fingerprint",
}

ML_NORTH_STAR_GATE_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "gate kind": "reverie_ml_north_star_release_gate",
    "gate schema": "reverie_ml_north_star_release_gate_v1",
    "north-star string": "best_small_language_for_reversible_inspectable_deterministic_ml_kernels",
    "non-goal string": "general_purpose_pytorch_tensorflow_training_replacement",
    "capsule verifier import": "import verify_model_capsule as capsule_check",
    "benchmark validator import": "import check_benchmark_artifact as benchmark_check",
    "handoff freshness": "require_handoff=True",
    "verification markdown freshness": "require_verification_markdown=True",
    "V1-V6 phases": 'EXPECTED_PHASES = ("V1", "V2", "V3", "V4", "V5", "V6")',
    "readiness gate": "ml_goal_readiness",
    "capability map gate": "ml_capability_map",
    "roundtrip verifier": "reversible-inference-roundtrip-verification.json",
    "roundtrip signed ML profile gate": "roundtrip verification must include a signed ML profile",
    "roundtrip ML profile replay check": "ml_profile_replayed",
    "roundtrip ML replay cost gate": "roundtrip verification ml_profile replay_cost must be an object",
    "roundtrip ML replay cost markdown": "Roundtrip replay payload",
    "deterministic inference validator": "validate_deterministic_inference",
    "deterministic inference markdown": "## Deterministic Inference",
    "deterministic inference action contract": "reverse_reversible_trace",
    "inference action receipt validator": "validate_inference_action_review_receipt",
    "inference action receipt payload": '"inference_action_review_receipt": inference_action_receipt',
    "inference action receipt reviewer command": "inference_action_receipt",
    "inference action receipt artifact": "inference_action_review_receipt",
    "inference action receipt semantic validation": "inference_action_review_semantic_fingerprint",
    "training update receipt validator": "validate_training_update_review_receipt",
    "training update receipt payload": '"training_update_review_receipt": training_update_receipt',
    "training update receipt reviewer command": "training_update_receipt",
    "training update receipt artifact": "training_update_review_receipt",
    "training update receipt semantic validation": "training_update_review_semantic_fingerprint",
    "model lineage validator": "validate_training_lineage",
    "model lineage markdown": "## Model Lineage",
    "debuggable update validator": "validate_debug_training_update",
    "debuggable update markdown": "## Debuggable Update",
    "roundtrip ML profile goal fit": "auditable_ml_kernel",
    "claim evidence validator": "validate_claim_evidence",
    "claim evidence payload": '"claim_evidence": claim_evidence',
    "claim evidence markdown": "## Claim Evidence",
    "release verification schema": "reverie_ml_north_star_release_verification_v1",
    "release verification JSON option": "--release-verification-output",
    "release verification Markdown option": "--release-verification-markdown",
    "release verification verify option": "--verify-release-verification",
    "release verification required commands": "REQUIRED_RELEASE_REPLAY_COMMANDS",
    "release verification required command check": "required_reviewer_commands_replayed",
    "release verification goal claims": "GOAL_CONTRACT_CLAIMS",
    "release verification goal claim requirements": "GOAL_CONTRACT_CLAIM_REQUIREMENTS",
    "release verification goal replay commands": "GOAL_CONTRACT_CLAIM_REPLAY_COMMANDS",
    "release verification goal replay artifacts": "GOAL_CONTRACT_CLAIM_REPLAY_ARTIFACTS",
    "release verification training lineage claim": "training_lineage_replay",
    "release verification training lineage command": "training_audit_lineage",
    "release verification training lineage card": "training_audit_verification_markdown",
    "release verification direct native inference card": "native_inference_audit_markdown",
    "release verification direct evaluation inference card": "evaluation_row_inference_audit_markdown",
    "release verification claim matrix": "claim_matrix_fingerprint",
    "release verification claim replay helper": "claim_replay_coverage",
    "release verification claim replay payload": "claim_replay_coverage",
    "release verification claim replay check": "claim_replay_coverage_complete",
    "release verification claim replay required artifacts": "required_artifact_ids",
    "release verification claim replay missing artifacts": "missing_artifact_ids",
    "release verification claim replay markdown": "## Claim Replay Coverage",
    "release verification goal contract helper": "release_goal_contract",
    "release verification goal contract validator": "goal_contract_errors",
    "release verification goal contract markdown helper": "goal_contract_markdown_lines",
    "release verification goal claim proofs": "claim_proofs",
    "north-star gate goal contract payload": '"goal_contract"] = release_goal_contract',
    "north-star gate goal contract markdown": "*goal_contract_markdown_lines(goal_contract)",
    "release verification gate goal contract freshness": "gate_goal_contract_fresh",
    "release verification goal contract payload": '"goal_contract": goal_contract',
    "release verification goal contract check": "goal_contract_bound",
    "release verification goal claim proof check": "goal_contract_claims_proven",
    "release verification goal contract markdown": "## Goal Contract",
    "release verification claim proof markdown": "Claims proven",
    "release verification embedded commands": "reviewer_replay_commands",
    "release verification saved check command": "saved_receipt_check",
    "release verification regenerate command": "regenerate_receipt",
    "release verification source catalog": "RELEASE_VERIFIER_SOURCE_FILES",
    "release verification residual checker source": "scripts/check_triangular_residual.py",
    "release verification source digest": "verifier_sources",
    "release verification source binding check": "verifier_sources_bound",
    "release verification artifact catalog": "RELEASE_PACKET_ARTIFACT_IDS",
    "release verification artifact digest": "release_artifacts",
    "release verification artifact binding check": "release_artifacts_bound",
    "release verification audit packet summary": "audit_packet_artifacts",
    "release verification gate artifact helper": "gate_artifact_summary",
    "release verification builder": "build_release_verification",
    "release verification renderer": "render_release_verification_markdown",
    "release verification saved validator": "validate_saved_release_verification",
    "release verification stale Markdown check": "release verification Markdown is stale",
    "release verification self-test": "release verification should pass",
    "release verification missing command self-test": "release verification should fail with missing required commands",
    "release verification missing artifact self-test": "release verification should fail with missing replay artifacts",
    "proof checker saved report option": "--expect-report-json",
    "benchmark RSS gate": "min_observed_rss_ratio",
    "benchmark Markdown check": "validate_markdown_summary",
    "reviewer command schema": "reverie_ml_north_star_reviewer_commands_v1",
    "reviewer MLP witness command": "mlp_witness_proof",
    "reviewer invertible coupling command": "invertible_coupling_proof",
    "reviewer triangular residual command": "triangular_residual_proof",
    "reviewer reversible preprocess command": "reversible_preprocess_proof",
    "reviewer inference trace command": "reversible_inference_trace_proof",
    "reviewer command validator": "validate_reviewer_commands",
    "reviewer command selector": "select_reviewer_commands",
    "reviewer command runner": "run_reviewer_commands",
    "reviewer command renderer": "## Reviewer Commands",
    "reviewer command list option": "--list-reviewer-commands",
    "reviewer command run option": "--run-reviewer-command",
    "reviewer command subprocess": "subprocess.run",
    "reviewer transcript schema": "reverie_ml_north_star_reviewer_replay_v1",
    "reviewer transcript JSON option": "--reviewer-transcript-output",
    "reviewer transcript Markdown option": "--reviewer-transcript-markdown",
    "reviewer transcript verify option": "--verify-reviewer-transcript",
    "reviewer transcript renderer": "render_transcript_markdown",
    "reviewer transcript validator": "validate_saved_transcript",
    "reviewer transcript gate binding": "gate_fingerprint_after",
    "reviewer transcript stale Markdown check": "reviewer transcript Markdown is stale",
    "reviewer transcript stdout hash": "stdout_sha256",
    "reviewer transcript self-test": "reviewer transcript should hash stdout",
    "reviewer transcript tamper self-test": "tampered reviewer transcript should fail",
    "roundtrip replay command": "verify-roundtrip",
    "training step replay command": "--verify-step",
    "missing RSS failure self-test": "missing RSS ratio should fail",
    "stale handoff failure self-test": "stale handoff Markdown should fail",
}

INVERTIBLE_COUPLING_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "coupling target": "examples/invertible_coupling.rev",
    "report kind": "reverie_q31_invertible_coupling_reference",
    "proof claim": "deterministic_q31_invertible_coupling_replay",
    "final vars writer": "--write-final-vars",
    "forward output input": "--forward-output-json",
    "reverse output input": "--reverse-output-json",
    "markdown output option": "--markdown-output",
    "markdown renderer": "render_markdown",
    "markdown title": "# Reverie Invertible Coupling Proof",
    "zero witness proof": "witness_payload_bytes = 0",
    "zero trace proof": "trace_payload_bytes = 0",
    "reverse restoration check": "reverse_restores_initial_state",
    "balanced recompute check": "balanced_recompute",
}

REVERSIBLE_PREPROCESS_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "preprocess target": "examples/reversible_preprocess.rev",
    "report kind": "reverie_q31_reversible_preprocess_reference",
    "proof claim": "deterministic_q31_reversible_preprocess_replay",
    "final vars writer": "--write-final-vars",
    "forward output input": "--forward-output-json",
    "reverse output input": "--reverse-output-json",
    "markdown output option": "--markdown-output",
    "markdown renderer": "render_markdown",
    "markdown title": "# Reverie Reversible Preprocess Proof",
    "zero witness proof": "witness_payload_bytes = 0",
    "zero trace proof": "trace_payload_bytes = 0",
    "raw preservation check": "raw_preserved",
    "mean preservation check": "mean_preserved",
    "balanced recompute check": "balanced_recompute",
}

TRIANGULAR_RESIDUAL_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "residual target": "examples/triangular_residual.rev",
    "report kind": "reverie_q31_triangular_residual_reference",
    "proof claim": "deterministic_q31_triangular_residual_replay",
    "final vars writer": "--write-final-vars",
    "forward output input": "--forward-output-json",
    "reverse output input": "--reverse-output-json",
    "markdown output option": "--markdown-output",
    "markdown renderer": "render_markdown",
    "markdown title": "# Reverie Triangular Residual Proof",
    "triangular source-order check": "triangular_source_order",
    "zero witness proof": "witness_payload_bytes = 0",
    "zero trace proof": "trace_payload_bytes = 0",
    "reverse restoration check": "reverse_restores_initial_state",
    "balanced recompute check": "balanced_recompute",
}

REVERSIBLE_INFERENCE_TRACE_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "inference trace target": "examples/reversible_inference_trace.rev",
    "report kind": "reverie_q31_reversible_inference_trace_reference",
    "proof claim": "deterministic_q31_reversible_inference_trace",
    "final vars writer": "--write-final-vars",
    "forward output input": "--forward-output-json",
    "reverse output input": "--reverse-output-json",
    "Q31 fixed multiply": "fixed_mul_q31",
    "argmax proof": "argmax_first",
    "runner-up proof": "runner_up_first",
    "top-two margin proof": "top2_margin",
    "label-rank proof": "rank_of",
    "top-k class proof": "top_k_indices",
    "top-k value proof": "top_k_values",
    "top-k correctness proof": "top_k_contains",
    "attribution ledger": "contribution_ledger_fingerprint",
    "margin attribution ledger": "margin_contribution_ledger_fingerprint",
    "markdown output option": "--markdown-output",
    "markdown renderer": "render_markdown",
    "markdown title": "# Reverie Reversible Inference Trace",
    "witness bytes": "witness_payload_bytes",
    "reverse restoration check": "reverse_restores_initial_state",
    "model preservation check": "model_preserved",
    "balanced recompute check": "balanced_recompute",
}

MODEL_IMPORT_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "Q31 import flag": "--import-model-json",
    "model verifier flag": "--verify-model",
    "model inference flag": "--inspect-model-inference",
    "Q31 inference checker coverage": "scripts/check_q31_inference.py",
    "Q31 exporter coverage": "scripts/export_q31_linear_model.py",
    "external import provenance": "external_import",
    "source JSON verification field": "source_model_json_checked",
    "Q31 import format": "weights_bias_q31_json",
    "Q31 scaling detail": "2147483648",
    "ML profile coverage": "ml_profile",
    "large tensor seed JSON coverage": "--vars-json",
}

Q31_INFERENCE_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "Q31 constant": "Q31_ONE = 1 << 31",
    "pixel scaling": "pixel_to_q31",
    "fixed multiply": "fixed_mul_q31",
    "wrapping arithmetic": "wrap_i64",
    "first argmax": "argmax_first",
    "top logits": "top_logits",
    "attribution reconstruction": "inference_attribution",
    "margin contribution proof": "top_margin_contributions",
    "attribution check": "matches_margin",
    "sample-set selection": "--sample-index",
    "batch sample-set evaluation": "--all-samples",
    "batch evaluation kind": "reverie_q31_linear_reference_evaluation",
    "Markdown explanation option": "--markdown-output",
    "Markdown inference title": "# Reverie Q31 Inference Explanation",
    "Markdown contribution section": "## Margin Contributions",
    "low-margin ranking": "top_low_margin",
    "label summaries": "by_label",
    "sample-set kind": "reverie_mnist_linear_q31_samples",
    "prediction expectation": "--expect-prediction",
    "correctness expectation": "--expect-correct",
    "model shape": "model.weights must have",
    "sample shape": "image_u8",
}

Q31_MLP_WITNESS_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "MLP witness target": "examples/mnist_mlp_witness.rev",
    "vars JSON input": "--vars-json",
    "run output JSON input": "--run-output-json",
    "Q31 constant": "Q31_ONE = 1 << 31",
    "hidden width": "HIDDEN = 16",
    "fixed multiply": "fixed_mul_q31",
    "ReLU mask semantics": "hidden_mask_tape",
    "witness tape field": "hidden_delta_tape",
    "prediction expectation": "--expect-predictions",
    "correctness expectation": "--expect-correct",
    "proof-cost claim": "deterministic_q31_mlp_witness_replay",
    "witness payload bytes": "witness_payload_bytes",
    "trace payload bytes": "trace_payload_bytes",
    "recomputed update payload bytes": "recomputed_update_payload_bytes",
    "run output source provenance validation": "validate_run_output_provenance",
    "run output source hash check": "program.source_sha256",
    "run output dataset loop validation": "validate_run_output_dataset_loops",
    "run output dataset loop metadata": "EXPECTED_DATASET_LOOPS",
    "run output witness metadata validation": "validate_run_output_witness_metadata",
    "run output witness proof validation": "validate_run_output_witness_proof",
    "witness proof schema": "reverie_witness_store_proof_v1",
    "witness value fingerprint": "value_fingerprint",
    "witness metadata type catalog": "WITNESS_METADATA_TYPES",
    "markdown output option": "--markdown-output",
    "markdown renderer": "render_markdown",
    "markdown title": "# Reverie MLP Witness Proof",
    "run JSON kind": "reverie_run_result",
}

Q31_EXPORTER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "Q31 import format": "weights_bias_q31_json",
    "Q31 constant": "Q31_ONE = 1 << 31",
    "pixel-class layout": "pixel-class",
    "class-pixel layout": "class-pixel",
    "state dict key": "state_dict",
    "NPZ reader": "read_npz_model",
    "NPY header parser": "ast.literal_eval",
    "NPZ CLI flag": "--npz-weights",
    "Fortran-order rejection": "fortran_order",
    "bool is not numeric": "isinstance(value, bool)",
    "i64 range check": "I64_MIN",
}

Q31_MLP_VARS_EXPORTER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "vars JSON output": "--vars-json",
    "MLP witness target": "examples/mnist_mlp_witness.rev",
    "Q31 constant": "Q31_ONE = 1 << 31",
    "first layer state dict": "fc1.weight",
    "second layer state dict": "fc2.weight",
    "torch layout": "torch",
    "reverie layout": "reverie",
    "w1 seed": '"w1"',
    "w2 seed": '"w2"',
    "metadata sidecar": "--metadata-output",
    "NPZ reader reuse": "linear_export.read_npz_member",
}

TORCH_STATE_DICT_EXTRACTOR_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "safe checkpoint loading": "weights_only=True",
    "unsafe legacy opt-in": "--allow-unsafe-pickle",
    "torch loader": "torch.load",
    "output format": "torch_linear_state_dict_json",
    "state dict key": "state_dict",
    "default weight key": "linear.weight",
    "default bias key": "linear.bias",
    "tensor detach": "detach",
    "tensor tolist": "tolist",
    "bool is not numeric": "isinstance(value, bool)",
}

JANA_AUDIT_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "bool is not numeric": "not isinstance(value, bool)",
    "Python 3.9 numeric isinstance": "isinstance(value, (int, float))",
    "line count validation": "stdout_lines must be a non-negative integer",
    "allowed status validation": "status must be one of",
    "totals status validation": "totals.{target}.{status} must be a non-negative integer",
    "unexpected field rejection": "has unexpected field(s)",
}

JANUS_FEATURE_AUDIT_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "duplicate feature rejection": "duplicate Janus feature row(s)",
    "vague language rejection": "vague audit language",
}

CRITERION_CHECKER_SNIPPETS = {
    "default tree/slot benchmark shape": 'DEFAULT_EXPECTED_FUNCTIONS = {"tree_walk", "slot_compiled"}',
    "compile benchmark shape exception": '"slot_compile_vs_execute_sort_n50": {"compile", "execute_compiled"}',
    "timeline benchmark shape exception": '"scrub_timeline_sort_n50": {"build_timeline"}',
    "unexpected Criterion bench function rejection": "has unexpected bench function(s)",
    "Criterion registration check": "missing from criterion_group! registration",
    "Criterion docs ordered group list check": "must match benchmark source order",
}

EXPLAIN_CHECKER_SNIPPETS = {
    "safety checks schema key": '"safety_checks"',
    "safety check counts schema key": '"safety_check_counts"',
    "dataset loops schema key": '"dataset_loops"',
    "dataset loops schema validation": "validate_dataset_loops",
    "witness store schema key": '"witness_store"',
    "witness metrics schema key": '"witness_metrics"',
    "witness metrics schema validation": "validate_witness_metrics",
    "ML profile option": "--ml",
    "ML profile schema validation": "validate_ml_profile",
    "ML profile replay cost validation": "validate_ml_replay_cost",
    "ML profile schema marker": "reverie_explain_ml_profile_v1",
    "bool is not int": "expected is int and isinstance(value, bool)",
    "indexed safety-check fixture": "examples\" / \"array.rev\"",
    "same-root update safety wording": "same-root update aliases rejected before runtime",
    "prebuilt reverie binary option": "--reverie-bin",
    "environment reverie binary option": "REVERIE_BIN",
    "build-independent schema self-test": "--self-test",
    "safety count consistency rejection": "safety_check_counts missing key(s)",
    "safety count unexpected key rejection": "safety_check_counts has unexpected key(s)",
    "safety count type rejection": "must be int",
}

CLI_JSON_METADATA_SNIPPETS = {
    "run result metadata helper": "run_result_metadata",
    "program source fingerprint": "source_sha256",
    "generic source hasher": "source.as_bytes()",
    "run result dataset loops": "dataset_loops_json",
    "store metadata renderer": "store_metadata_json",
    "witness metrics renderer": "witness_metrics_json",
    "explain ML profile renderer": "ml_profile_json",
    "ML replay cost profile": '"replay_cost"',
    "roundtrip signed ML profile": '"ml_profile".to_owned()',
    "dataset loops explain metadata": "dataset_loops",
    "dataset loops renderer": "json_dataset_loops",
    "metadata-aware JSON printer": "print_io_state_json(state, kind, metadata)",
}

MNIST_LINEAR_RUNNER_SNIPPETS = {
    "inference audit markdown renderer": "render_inference_audit_markdown",
    "inference audit markdown card": "# Reverie Inference Audit",
    "inspect inference markdown option": "--inspect-inference",
    "inference verification markdown renderer": "render_inference_verification_markdown",
    "inference verification markdown card": "# Reverie Inference Verification",
    "inference verification markdown option": "--verify-inference",
}

CLI_TEST_SNIPPETS = {
    "run/reverse witness metadata test": "run_and_reverse_json_emit_witness_metadata",
    "run/reverse dataset loop metadata test": "run_and_reverse_json_emit_dataset_loop_metadata",
    "explain ML profile text test": "explain_ml_profile_classifies_mlp_witness_kernel",
    "explain ML profile JSON test": "explain_ml_profile_json_reports_tensor_witness_q31_budget",
    "roundtrip ML profile JSON test": "roundtrip_json_embeds_ml_audit_profile_for_q31_kernel",
    "roundtrip replay cost assertion": "roundtrip_statement_count",
    "witness metadata field assertion": '"witness_metrics"',
    "dataset loop explain assertion": "explain_reports_dataset_shaped_witness_loop",
    "store metadata field assertion": '"store_metadata"',
}

CLI_DOC_SNIPPETS = {
    "run JSON program provenance docs": "block records the source path",
    "run JSON witness proof docs": "`witness_store`, `witness_metrics`, `ml_profile`, and",
    "run JSON dataset loop docs": "`dataset_loops`",
    "explain dataset loop docs": "`dataset_loops`",
    "explain ML profile docs": "`ml_profile`",
    "explain ML replay cost docs": "`replay_cost`",
    "roundtrip signed ML profile docs": "top-level fingerprint signs the whole proof payload, including the ML profile",
    "reverse JSON witness metadata docs": "same `store`, `input`, `output`, `observations`, `dataset_loops`",
}


def main() -> int:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    benchmark_checker = BENCHMARK_CHECKER.read_text(encoding="utf-8")
    summary_script = SUMMARY_SCRIPT.read_text(encoding="utf-8")
    mnist_ml_summary_script = MNIST_ML_SUMMARY_SCRIPT.read_text(encoding="utf-8")
    verify_script = VERIFY_SCRIPT.read_text(encoding="utf-8")
    benchmark_harness = (REPO_ROOT / "scripts" / "bench_jana_vs_reverie.py").read_text(
        encoding="utf-8"
    )
    materialized_source_checker = MATERIALIZED_SOURCE_CHECKER.read_text(encoding="utf-8")
    benchmark_docs_checker = (
        REPO_ROOT / "scripts" / "check_benchmark_docs.py"
    ).read_text(encoding="utf-8")
    criterion_checker = CRITERION_CHECKER.read_text(encoding="utf-8")
    evaluation_corpus_checker = (
        REPO_ROOT / "scripts" / "check_evaluation_corpus.py"
    ).read_text(encoding="utf-8")
    cli_main = CLI_MAIN.read_text(encoding="utf-8")
    mnist_linear_runner = MNIST_LINEAR_RUNNER.read_text(encoding="utf-8")
    cli_tests = CLI_TESTS.read_text(encoding="utf-8")
    cli_docs = CLI_DOCS.read_text(encoding="utf-8")
    invertible_coupling_checker = INVERTIBLE_COUPLING_CHECKER.read_text(encoding="utf-8")
    reversible_inference_trace_checker = REVERSIBLE_INFERENCE_TRACE_CHECKER.read_text(
        encoding="utf-8"
    )
    reversible_preprocess_checker = REVERSIBLE_PREPROCESS_CHECKER.read_text(encoding="utf-8")
    triangular_residual_checker = TRIANGULAR_RESIDUAL_CHECKER.read_text(encoding="utf-8")
    mnist_ml_profile_checker = MNIST_ML_PROFILE_CHECKER.read_text(encoding="utf-8")
    ml_north_star_gate = ML_NORTH_STAR_GATE.read_text(encoding="utf-8")
    mnist_ml_audit_pipeline = MNIST_ML_AUDIT_PIPELINE.read_text(encoding="utf-8")
    model_import_checker = MODEL_IMPORT_CHECKER.read_text(encoding="utf-8")
    model_capsule_verifier = MODEL_CAPSULE_VERIFIER.read_text(encoding="utf-8")
    q31_inference_checker = Q31_INFERENCE_CHECKER.read_text(encoding="utf-8")
    q31_mlp_witness_checker = Q31_MLP_WITNESS_CHECKER.read_text(encoding="utf-8")
    q31_exporter = Q31_EXPORTER.read_text(encoding="utf-8")
    q31_mlp_vars_exporter = Q31_MLP_VARS_EXPORTER.read_text(encoding="utf-8")
    torch_state_dict_extractor = TORCH_STATE_DICT_EXTRACTOR.read_text(encoding="utf-8")
    explain_checker = EXPLAIN_CHECKER.read_text(encoding="utf-8")
    jana_audit_checker = (REPO_ROOT / "scripts" / "check_jana_audit.py").read_text(
        encoding="utf-8"
    )
    janus_feature_audit_checker = (
        REPO_ROOT / "scripts" / "check_janus_feature_audit.py"
    ).read_text(encoding="utf-8")
    missing = [
        label for label, snippet in REQUIRED_SNIPPETS.items() if snippet not in text
    ]
    missing.extend(
        f"Benchmark checker contract: {label}"
        for label, snippet in BENCHMARK_CHECKER_SNIPPETS.items()
        if snippet not in benchmark_checker
    )
    missing.extend(
        f"Benchmark summary renderer contract: {label}"
        for label, snippet in SUMMARY_SCRIPT_SNIPPETS.items()
        if snippet not in summary_script
    )
    missing.extend(
        f"MNIST ML profile summary renderer contract: {label}"
        for label, snippet in MNIST_ML_SUMMARY_SCRIPT_SNIPPETS.items()
        if snippet not in mnist_ml_summary_script
    )
    missing.extend(
        f"Janus performance wrapper contract: {label}"
        for label, snippet in VERIFY_SCRIPT_SNIPPETS.items()
        if snippet not in verify_script
    )
    missing.extend(
        f"Benchmark harness contract: {label}"
        for label, snippet in BENCHMARK_HARNESS_SNIPPETS.items()
        if snippet not in benchmark_harness
    )
    missing.extend(
        f"Materialized source checker contract: {label}"
        for label, snippet in MATERIALIZED_SOURCE_CHECKER_SNIPPETS.items()
        if snippet not in materialized_source_checker
    )
    missing.extend(
        f"Benchmark docs checker contract: {label}"
        for label, snippet in BENCHMARK_DOCS_CHECKER_SNIPPETS.items()
        if snippet not in benchmark_docs_checker
    )
    missing.extend(
        f"Criterion checker contract: {label}"
        for label, snippet in CRITERION_CHECKER_SNIPPETS.items()
        if snippet not in criterion_checker
    )
    missing.extend(
        f"Evaluation corpus checker contract: {label}"
        for label, snippet in EVALUATION_CORPUS_CHECKER_SNIPPETS.items()
        if snippet not in evaluation_corpus_checker
    )
    missing.extend(
        f"MNIST ML profile checker contract: {label}"
        for label, snippet in MNIST_ML_PROFILE_CHECKER_SNIPPETS.items()
        if snippet not in mnist_ml_profile_checker
    )
    missing.extend(
        f"ML north-star gate contract: {label}"
        for label, snippet in ML_NORTH_STAR_GATE_SNIPPETS.items()
        if snippet not in ml_north_star_gate
    )
    missing.extend(
        f"MNIST linear runner contract: {label}"
        for label, snippet in MNIST_LINEAR_RUNNER_SNIPPETS.items()
        if snippet not in mnist_linear_runner
    )
    missing.extend(
        f"Invertible coupling checker contract: {label}"
        for label, snippet in INVERTIBLE_COUPLING_CHECKER_SNIPPETS.items()
        if snippet not in invertible_coupling_checker
    )
    missing.extend(
        f"Reversible preprocess checker contract: {label}"
        for label, snippet in REVERSIBLE_PREPROCESS_CHECKER_SNIPPETS.items()
        if snippet not in reversible_preprocess_checker
    )
    missing.extend(
        f"Triangular residual checker contract: {label}"
        for label, snippet in TRIANGULAR_RESIDUAL_CHECKER_SNIPPETS.items()
        if snippet not in triangular_residual_checker
    )
    missing.extend(
        f"Reversible inference trace checker contract: {label}"
        for label, snippet in REVERSIBLE_INFERENCE_TRACE_CHECKER_SNIPPETS.items()
        if snippet not in reversible_inference_trace_checker
    )
    missing.extend(
        f"MNIST ML audit pipeline contract: {label}"
        for label, snippet in MNIST_ML_AUDIT_PIPELINE_SNIPPETS.items()
        if snippet not in mnist_ml_audit_pipeline
    )
    missing.extend(
        f"Model capsule verifier contract: {label}"
        for label, snippet in MODEL_CAPSULE_VERIFIER_SNIPPETS.items()
        if snippet not in model_capsule_verifier
    )
    missing.extend(
        f"Model import docs checker contract: {label}"
        for label, snippet in MODEL_IMPORT_CHECKER_SNIPPETS.items()
        if snippet not in model_import_checker
    )
    missing.extend(
        f"Q31 inference checker contract: {label}"
        for label, snippet in Q31_INFERENCE_CHECKER_SNIPPETS.items()
        if snippet not in q31_inference_checker
    )
    missing.extend(
        f"Q31 MLP witness checker contract: {label}"
        for label, snippet in Q31_MLP_WITNESS_CHECKER_SNIPPETS.items()
        if snippet not in q31_mlp_witness_checker
    )
    missing.extend(
        f"Q31 linear model exporter contract: {label}"
        for label, snippet in Q31_EXPORTER_SNIPPETS.items()
        if snippet not in q31_exporter
    )
    missing.extend(
        f"Q31 MLP vars-json exporter contract: {label}"
        for label, snippet in Q31_MLP_VARS_EXPORTER_SNIPPETS.items()
        if snippet not in q31_mlp_vars_exporter
    )
    missing.extend(
        f"PyTorch state_dict extractor contract: {label}"
        for label, snippet in TORCH_STATE_DICT_EXTRACTOR_SNIPPETS.items()
        if snippet not in torch_state_dict_extractor
    )
    missing.extend(
        f"Jana audit checker contract: {label}"
        for label, snippet in JANA_AUDIT_CHECKER_SNIPPETS.items()
        if snippet not in jana_audit_checker
    )
    missing.extend(
        f"Janus feature audit checker contract: {label}"
        for label, snippet in JANUS_FEATURE_AUDIT_CHECKER_SNIPPETS.items()
        if snippet not in janus_feature_audit_checker
    )
    missing.extend(
        f"Explain checker contract: {label}"
        for label, snippet in EXPLAIN_CHECKER_SNIPPETS.items()
        if snippet not in explain_checker
    )
    missing.extend(
        f"CLI JSON metadata contract: {label}"
        for label, snippet in CLI_JSON_METADATA_SNIPPETS.items()
        if snippet not in cli_main
    )
    missing.extend(
        f"CLI witness metadata test contract: {label}"
        for label, snippet in CLI_TEST_SNIPPETS.items()
        if snippet not in cli_tests
    )
    missing.extend(
        f"CLI docs contract: {label}"
        for label, snippet in CLI_DOC_SNIPPETS.items()
        if snippet not in cli_docs
    )
    if missing:
        for label in missing:
            print(f"error: CI workflow missing {label}", file=sys.stderr)
        return 1
    print(f"ok: validated {len(REQUIRED_SNIPPETS)} CI workflow gate(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
