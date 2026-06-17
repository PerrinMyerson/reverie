#!/usr/bin/env python3
"""Render checked Reverie MNIST ML profile JSON as Markdown."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import check_mnist_ml_profile as profile_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render checked Reverie MNIST ML profile JSON as Markdown."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="JSON report(s) to summarize.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write Markdown to this path instead of stdout.",
    )
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
        "--self-test",
        action="store_true",
        help="Run build-independent Markdown renderer self-tests and exit.",
    )
    return parser.parse_args()


def load_checked_report(
    path: Path,
    require_reverse_check: bool,
    require_peak_rss: bool,
    require_ml_profile: bool,
    require_audit_contract: bool,
) -> dict[str, Any]:
    report = profile_check.load_json(path)
    profile_check.validate_report(
        report,
        require_reverse_check=require_reverse_check,
        require_peak_rss=require_peak_rss,
        require_ml_profile=require_ml_profile,
        require_audit_contract=require_audit_contract,
    )
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    return report


def comma(value: int) -> str:
    return f"{value:,}"


def number(value: float) -> str:
    return f"{value:.3f}"


def percent(value: float) -> str:
    return f"{value:.2f}%"


def bytes_cell(value: Optional[int]) -> str:
    if value is None:
        return "n/a"
    return f"{comma(value)} B"


def ratio_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    return number(float(value))


def short_hash(value: str) -> str:
    return value[:12]


def report_label(path: Path) -> str:
    return path.name or str(path)


def first_report(
    reports: list[tuple[Path, dict[str, Any]]],
    kind: str,
) -> Optional[tuple[Path, dict[str, Any]]]:
    for path, report in reports:
        if report.get("kind") == kind:
            return path, report
    return None


def first_pipeline_summary(
    reports: list[tuple[Path, dict[str, Any]]],
) -> Optional[tuple[Path, dict[str, Any]]]:
    return first_report(reports, profile_check.PIPELINE_SUMMARY_KIND)


def replay_payload_candidates(report: dict[str, Any]) -> list[int]:
    candidates = []
    proof = report.get("proof")
    if isinstance(proof, dict):
        for field in ("replay_payload_bytes", "full_replay_payload_bytes"):
            value = proof.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                candidates.append(value)
    memory = report.get("memory")
    if isinstance(memory, dict):
        value = memory.get("replay_payload_bytes")
        if isinstance(value, int) and not isinstance(value, bool):
            candidates.append(value)
    return candidates


def status_cell(passed: bool) -> str:
    return "ok" if passed else "failed"


def recompute_frontier_rows_from_reports(reports: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    run_item = first_report(reports, profile_check.RUN_KIND)
    step_item = first_report(reports, profile_check.STEP_VERIFICATION_KIND)
    native_verification_item = first_report(reports, profile_check.INFERENCE_VERIFICATION_KIND)
    model_eval_item = first_report(reports, profile_check.MODEL_EVALUATION_KIND)
    row_verification_item: Optional[tuple[Path, dict[str, Any]]] = None
    for path, report in reports:
        if report.get("kind") != profile_check.INFERENCE_VERIFICATION_KIND:
            continue
        if report.get("source_evaluation_checked") is True:
            row_verification_item = (path, report)
            break
    mlp_item = first_report(reports, profile_check.MLP_WITNESS_KIND)
    coupling_item = first_report(reports, profile_check.INVERTIBLE_COUPLING_KIND)
    preprocess_item = first_report(reports, profile_check.REVERSIBLE_PREPROCESS_KIND)
    inference_trace_item = first_report(reports, profile_check.REVERSIBLE_INFERENCE_TRACE_KIND)
    if (
        run_item is None
        or step_item is None
        or native_verification_item is None
        or model_eval_item is None
        or row_verification_item is None
        or mlp_item is None
        or coupling_item is None
        or preprocess_item is None
        or inference_trace_item is None
    ):
        return []

    run_path, run = run_item
    step_path, step = step_item
    native_path, native = native_verification_item
    model_eval_path, model_eval = model_eval_item
    row_path, row_verification = row_verification_item
    mlp_path, mlp = mlp_item
    coupling_path, coupling = coupling_item
    preprocess_path, preprocess = preprocess_item
    inference_trace_path, inference_trace = inference_trace_item

    run_proof = run["proof"]
    step_proof = step["proof"]
    native_memory = native["memory"]
    model_eval_proof = model_eval["proof"]
    row_memory = row_verification["memory"]
    mlp_proof = mlp["proof"]
    coupling_proof = coupling["proof"]
    preprocess_proof = preprocess["proof"]
    inference_trace_proof = inference_trace["proof"]

    def row(
        label: str,
        scope: str,
        path: Path,
        proof: dict[str, Any],
        *,
        replay_field: str = "replay_payload_bytes",
        trace_field: str = "trace_payload_bytes",
        derived_update_field: str = "derived_update_payload_bytes",
        state_payload_bytes: int = 0,
    ) -> dict[str, Any]:
        replay_payload = proof[replay_field]
        inverse_steps = proof["inverse_recompute_steps"]
        return {
            "label": label,
            "scope": scope,
            "report": report_label(path),
            "replay_payload_bytes": replay_payload,
            "witness_payload_bytes": proof["witness_payload_bytes"],
            "trace_payload_bytes": proof[trace_field],
            "derived_update_payload_bytes": proof.get(derived_update_field, 0),
            "state_payload_bytes": state_payload_bytes,
            "forward_recompute_steps": proof["forward_recompute_steps"],
            "inverse_recompute_steps": inverse_steps,
            "bytes_per_inverse_step": (
                replay_payload / inverse_steps
                if inverse_steps
                else None
            ),
            "zero_witness": proof["witness_payload_bytes"] == 0 and proof[trace_field] == 0,
        }

    return [
        row(
            "Training trace replay",
            "all training updates",
            run_path,
            run_proof,
            replay_field="full_replay_payload_bytes",
            trace_field="trace_replay_payload_bytes",
        ),
        row("Selected training-step replay", "one model update", step_path, step_proof),
        row("Single native inference replay", "one signed sample", native_path, native_memory),
        row("Batch model-evaluation replay", "signed sample batch", model_eval_path, model_eval_proof),
        row("Evaluation-row inference replay", "one scan-selected row", row_path, row_memory),
        row(
            "MLP witness trace replay",
            "two-sample MLP update",
            mlp_path,
            mlp_proof,
            derived_update_field="stored_derived_update_payload_bytes",
        ),
        row(
            "Invertible coupling replay",
            "one additive coupling layer",
            coupling_path,
            coupling_proof,
            state_payload_bytes=coupling_proof["state_payload_bytes"],
        ),
        row(
            "Reversible preprocessing replay",
            "one centered/permuted feature block",
            preprocess_path,
            preprocess_proof,
            state_payload_bytes=preprocess_proof["feature_payload_bytes"],
        ),
        row(
            "Reversible inference trace replay",
            "one preprocessed Q31 classifier trace",
            inference_trace_path,
            inference_trace_proof,
            state_payload_bytes=inference_trace_proof["state_payload_bytes"],
        ),
    ]


def render_recompute_frontier_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    summary_item = first_pipeline_summary(reports)
    if summary_item is not None:
        summary_path, summary = summary_item
        frontier = summary["metrics"]["recompute_frontier"]
        return [
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["label"],
                row["scope"],
                report_label(summary_path),
                bytes_cell(row["replay_payload_bytes"]),
                bytes_cell(row["witness_payload_bytes"]),
                bytes_cell(row["trace_payload_bytes"]),
                bytes_cell(row["derived_update_payload_bytes"]),
                bytes_cell(row["state_payload_bytes"]),
                comma(row["forward_recompute_steps"]),
                comma(row["inverse_recompute_steps"]),
                ratio_cell(row["bytes_per_inverse_step"]),
            )
            for row in frontier["rows"]
        ]
    return [
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["label"],
            row["scope"],
            row["report"],
            bytes_cell(row["replay_payload_bytes"]),
            bytes_cell(row["witness_payload_bytes"]),
            bytes_cell(row["trace_payload_bytes"]),
            bytes_cell(row["derived_update_payload_bytes"]),
            bytes_cell(row["state_payload_bytes"]),
            comma(row["forward_recompute_steps"]),
            comma(row["inverse_recompute_steps"]),
            ratio_cell(row["bytes_per_inverse_step"]),
        )
        for row in recompute_frontier_rows_from_reports(reports)
    ]


def exact_projection_unit(total: int, count: int, context: str) -> int:
    if count <= 0:
        raise ValueError(f"{context} count must be positive")
    if total % count != 0:
        raise ValueError(f"{context} total must divide evenly by count")
    return total // count


def scaling_projection_families_from_reports(
    reports: list[tuple[Path, dict[str, Any]]],
) -> list[dict[str, Any]]:
    run_item = first_report(reports, profile_check.RUN_KIND)
    model_eval_item = first_report(reports, profile_check.MODEL_EVALUATION_KIND)
    mlp_item = first_report(reports, profile_check.MLP_WITNESS_KIND)
    if run_item is None or model_eval_item is None or mlp_item is None:
        return []

    run_path, run = run_item
    model_eval_path, model_eval = model_eval_item
    mlp_path, mlp = mlp_item
    run_proof = run["proof"]
    model_eval_summary = model_eval["summary"]
    model_eval_proof = model_eval["proof"]
    mlp_proof = mlp["proof"]

    def family(
        *,
        label: str,
        unit: str,
        report: Path,
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
            replay_payload = model_payload_bytes + count * variable_replay_payload_bytes_per_unit
            projections.append(
                {
                    "scale_factor": scale_factor,
                    "count": count,
                    "projected_replay_payload_bytes": replay_payload,
                    "projected_witness_payload_bytes": count * witness_payload_bytes_per_unit,
                    "projected_trace_payload_bytes": count * trace_payload_bytes_per_unit,
                    "projected_recomputed_update_payload_bytes": (
                        count * recomputed_update_payload_bytes_per_unit
                    ),
                    "projected_forward_recompute_steps": count * forward_recompute_steps_per_unit,
                    "projected_inverse_recompute_steps": count * inverse_recompute_steps_per_unit,
                }
            )
        return {
            "label": label,
            "unit": unit,
            "report": report_label(report),
            "projections": projections,
        }

    evaluation_count = model_eval_summary["samples"]
    mlp_count = mlp["samples"]
    return [
        family(
            label="Training trace replay",
            unit="training update",
            report=run_path,
            observed_count=run_proof["entries"],
            model_payload_bytes=run_proof["model_payload_bytes"],
            sample_payload_bytes_per_unit=run_proof["sample_bytes_per_step"],
            witness_payload_bytes_per_unit=run_proof["witness_bytes_per_step"],
            trace_payload_bytes_per_unit=run_proof["trace_replay_bytes_per_step"],
            variable_replay_payload_bytes_per_unit=run_proof["trace_replay_bytes_per_step"],
            recomputed_update_payload_bytes_per_unit=0,
            forward_recompute_steps_per_unit=exact_projection_unit(
                run_proof["forward_recompute_steps"],
                run_proof["entries"],
                "training trace forward recompute",
            ),
            inverse_recompute_steps_per_unit=exact_projection_unit(
                run_proof["inverse_recompute_steps"],
                run_proof["entries"],
                "training trace inverse recompute",
            ),
        ),
        family(
            label="Batch model-evaluation replay",
            unit="evaluated sample",
            report=model_eval_path,
            observed_count=evaluation_count,
            model_payload_bytes=model_eval_proof["model_payload_bytes"],
            sample_payload_bytes_per_unit=exact_projection_unit(
                model_eval_proof["sample_payload_bytes"],
                evaluation_count,
                "model evaluation sample payload",
            ),
            witness_payload_bytes_per_unit=exact_projection_unit(
                model_eval_proof["witness_payload_bytes"],
                evaluation_count,
                "model evaluation witness payload",
            ),
            trace_payload_bytes_per_unit=exact_projection_unit(
                model_eval_proof["trace_payload_bytes"],
                evaluation_count,
                "model evaluation trace payload",
            ),
            variable_replay_payload_bytes_per_unit=exact_projection_unit(
                model_eval_proof["sample_payload_bytes"]
                + model_eval_proof["witness_payload_bytes"]
                + model_eval_proof["trace_payload_bytes"],
                evaluation_count,
                "model evaluation variable replay payload",
            ),
            recomputed_update_payload_bytes_per_unit=0,
            forward_recompute_steps_per_unit=exact_projection_unit(
                model_eval_proof["forward_recompute_steps"],
                evaluation_count,
                "model evaluation forward recompute",
            ),
            inverse_recompute_steps_per_unit=exact_projection_unit(
                model_eval_proof["inverse_recompute_steps"],
                evaluation_count,
                "model evaluation inverse recompute",
            ),
        ),
        family(
            label="MLP witness trace replay",
            unit="MLP sample update",
            report=mlp_path,
            observed_count=mlp_count,
            model_payload_bytes=mlp_proof["model_payload_bytes"],
            sample_payload_bytes_per_unit=exact_projection_unit(
                mlp_proof["sample_payload_bytes"],
                mlp_count,
                "MLP sample payload",
            ),
            witness_payload_bytes_per_unit=mlp_proof["witness_bytes_per_sample"],
            trace_payload_bytes_per_unit=mlp_proof["trace_bytes_per_sample"],
            variable_replay_payload_bytes_per_unit=mlp_proof["trace_bytes_per_sample"],
            recomputed_update_payload_bytes_per_unit=mlp_proof["recomputed_update_bytes_per_sample"],
            forward_recompute_steps_per_unit=exact_projection_unit(
                mlp_proof["forward_recompute_steps"],
                mlp_count,
                "MLP forward recompute",
            ),
            inverse_recompute_steps_per_unit=exact_projection_unit(
                mlp_proof["inverse_recompute_steps"],
                mlp_count,
                "MLP inverse recompute",
            ),
        ),
    ]


def render_scaling_projection_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    summary_item = first_pipeline_summary(reports)
    if summary_item is not None:
        summary_path, summary = summary_item
        families = summary["metrics"]["scaling_projection"]["families"]
        for family in families:
            family = {**family, "report": report_label(summary_path)}
            for projection in family["projections"]:
                scale = projection["scale_factor"]
                rows.append(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                        family["label"],
                        family["unit"],
                        family["report"],
                        "observed" if scale == 1 else f"{scale}x",
                        comma(projection["count"]),
                        bytes_cell(projection["projected_replay_payload_bytes"]),
                        bytes_cell(projection["projected_witness_payload_bytes"]),
                        bytes_cell(projection["projected_trace_payload_bytes"]),
                        bytes_cell(projection["projected_recomputed_update_payload_bytes"]),
                        comma(projection["projected_forward_recompute_steps"]),
                        comma(projection["projected_inverse_recompute_steps"]),
                    )
                )
        return rows
    for family in scaling_projection_families_from_reports(reports):
        for projection in family["projections"]:
            scale = projection["scale_factor"]
            rows.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    family["label"],
                    family["unit"],
                    family["report"],
                    "observed" if scale == 1 else f"{scale}x",
                    comma(projection["count"]),
                    bytes_cell(projection["projected_replay_payload_bytes"]),
                    bytes_cell(projection["projected_witness_payload_bytes"]),
                    bytes_cell(projection["projected_trace_payload_bytes"]),
                    bytes_cell(projection["projected_recomputed_update_payload_bytes"]),
                    comma(projection["projected_forward_recompute_steps"]),
                    comma(projection["projected_inverse_recompute_steps"]),
                )
            )
    return rows


def render_model_capsule_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.MODEL_CAPSULE_KIND:
            continue
        payload = report["payload"]
        gates = payload["gates"]
        model = payload["model"]
        samples = payload["samples"]
        lineage = payload["training_lineage"]
        inference = payload["inference"]
        witnesses = payload["witnesses"]
        scorecard = payload["scorecard"]
        readiness = payload["readiness"]
        readiness_summary = readiness["summary"]
        rows.append(
            "| {} | {} | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                short_hash(report["fingerprint"]),
                comma(gates["passed_count"]),
                comma(gates["total"]),
                short_hash(model["sha256"]),
                comma(samples["count"]),
                percent(samples["accuracy_percent"]),
                short_hash(lineage["lineage_ledger_fingerprint"]),
                short_hash(lineage["final_chain"]),
                status_cell(bool(inference["native_replay_passed"])),
                status_cell(bool(inference["q31_reference_matches_native"])),
                short_hash(witnesses["mlp_witness_proof_fingerprint"]),
                bytes_cell(scorecard["run_peak_rss_bytes"]),
                bytes_cell(scorecard["max_replay_payload_bytes"]),
                "{}/{}".format(
                    comma(readiness_summary["passed"]),
                    comma(readiness_summary["total"]),
                ),
            )
        )
    return rows


def render_inference_action_contract_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.MODEL_CAPSULE_KIND:
            continue
        for action in report["payload"]["inference"]["action_contract"]:
            operation = action["operation"]
            if operation == "reproduce_prediction":
                result = "prediction {}, correct {}, margin {}".format(
                    action["result"]["prediction"],
                    str(action["result"]["correct"]).lower(),
                    comma(action["result"]["margin"]),
                )
                cost = "Q31/native parity"
            elif operation == "explain_margin":
                result = "contribution {}, margin {}".format(
                    short_hash(action["ledgers"]["contribution"]),
                    short_hash(action["ledgers"]["margin_contribution"]),
                )
                cost = "ledger recompute"
            elif operation == "replay_native_inference":
                result = "replay {}".format(bytes_cell(action["payload_bytes"]))
                cost = "native replay proof"
            elif operation == "reverse_reversible_trace":
                result = "witness {}, trace {}, replay {}".format(
                    bytes_cell(action["witness_payload_bytes"]),
                    bytes_cell(action["trace_payload_bytes"]),
                    bytes_cell(action["payload_bytes"]),
                )
                cost = "{} recompute steps".format(comma(action["recompute_steps"]))
            else:
                result = "checked"
                cost = "recorded"
            rows.append(
                "| {} | `{}` | {} | `{}` | {} | {} |".format(
                    report_label(path),
                    operation,
                    status_cell(bool(action["supported"])),
                    action["evidence"],
                    result,
                    cost,
                )
            )
    return rows


def render_capability_map_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    summary_item = first_pipeline_summary(reports)
    if summary_item is not None:
        _, summary = summary_item
        return [
            "| {} | {} | {} | {} |".format(
                row["phase"],
                row["goal"],
                status_cell(bool(row["passed"])),
                ", ".join(row["evidence"]) if row["evidence"] else "missing",
            )
            for row in summary["metrics"]["ml_capability_map"]["capabilities"]
        ]
    run_item = first_report(reports, profile_check.RUN_KIND)
    step_item = first_report(reports, profile_check.STEP_VERIFICATION_KIND)
    audit_scan_item = first_report(reports, profile_check.AUDIT_SCAN_KIND)
    audit_step_item = first_report(reports, profile_check.AUDIT_STEP_KIND)
    model_eval_item = first_report(reports, profile_check.MODEL_EVALUATION_KIND)
    model_eval_verification_item = first_report(reports, profile_check.MODEL_EVALUATION_VERIFICATION_KIND)
    q31_eval_item = first_report(reports, profile_check.Q31_REFERENCE_EVALUATION_KIND)
    mlp_item = first_report(reports, profile_check.MLP_WITNESS_KIND)
    coupling_item = first_report(reports, profile_check.INVERTIBLE_COUPLING_KIND)
    preprocess_item = first_report(reports, profile_check.REVERSIBLE_PREPROCESS_KIND)
    trace_item = first_report(reports, profile_check.REVERSIBLE_INFERENCE_TRACE_KIND)
    profile_item = first_report(reports, profile_check.COMPARISON_KIND)

    inference_verifications = [
        report
        for _, report in reports
        if report.get("kind") == profile_check.INFERENCE_VERIFICATION_KIND
    ]

    def evidence_labels(items: list[Optional[tuple[Path, dict[str, Any]]]]) -> str:
        labels = [
            report_label(path)
            for item in items
            if item is not None
            for path, _ in (item,)
        ]
        return ", ".join(labels) if labels else "missing"

    run_ok = (
        run_item is not None
        and run_item[1].get("reverse", {}).get("restored_initial_model") is True
        and run_item[1].get("reverse", {}).get("enabled") is True
    )
    step_ok = (
        step_item is not None
        and step_item[1].get("proof_matches") is True
        and step_item[1].get("reverse", {}).get("before_model_restored") is True
    )
    inference_ok = bool(inference_verifications) and all(
        report.get("proof_matches") is True
        and report.get("restored_initial_state") is True
        for report in inference_verifications
    )
    witness_ok = (
        audit_scan_item is not None
        and audit_step_item is not None
        and audit_scan_item[1].get("summary", {}).get("witness_mismatches") == 0
        and audit_step_item[1].get("debug_contract", {}).get("passed") is True
    )
    batch_ok = (
        model_eval_item is not None
        and model_eval_verification_item is not None
        and q31_eval_item is not None
        and model_eval_verification_item[1].get("rows_match") is True
        and model_eval_verification_item[1].get("proof_matches") is True
        and q31_eval_item[1].get("summary", {}).get("samples")
        == model_eval_item[1].get("summary", {}).get("samples")
    )
    mlp_ok = mlp_item is not None and mlp_item[1].get("passed") is True
    coupling_proof = coupling_item[1].get("proof", {}) if coupling_item is not None else {}
    coupling_ok = (
        coupling_item is not None
        and coupling_item[1].get("passed") is True
        and coupling_proof.get("witness_payload_bytes") == 0
        and coupling_proof.get("trace_payload_bytes") == 0
    )
    preprocess_proof = preprocess_item[1].get("proof", {}) if preprocess_item is not None else {}
    preprocess_ok = (
        preprocess_item is not None
        and preprocess_item[1].get("passed") is True
        and preprocess_proof.get("witness_payload_bytes") == 0
        and preprocess_proof.get("trace_payload_bytes") == 0
    )
    trace_proof = trace_item[1].get("proof", {}) if trace_item is not None else {}
    trace_ok = (
        trace_item is not None
        and trace_item[1].get("passed") is True
        and trace_proof.get("witness_payload_bytes", 0) > 0
        and trace_proof.get("trace_payload_bytes") == 0
        and trace_proof.get("checks", {}).get("reverse_restores_initial_state") is True
    )
    scorecard_ok = bool(render_scorecard_rows(reports))

    rows = [
        (
            "V1",
            "reversible linear MNIST",
            run_ok and step_ok and inference_ok and trace_ok,
            evidence_labels([run_item, step_item, trace_item]),
        ),
        (
            "V2",
            "first-class witness tapes",
            witness_ok,
            evidence_labels([audit_scan_item, audit_step_item]),
        ),
        (
            "V3",
            "batched tensor dataset iteration",
            batch_ok,
            evidence_labels([model_eval_item, model_eval_verification_item, q31_eval_item]),
        ),
        (
            "V4",
            "reversible MLP activation and mask witnesses",
            mlp_ok,
            evidence_labels([mlp_item]),
        ),
        (
            "V5",
            "invertible layer and preprocessing patterns",
            coupling_ok and preprocess_ok,
            evidence_labels([coupling_item, preprocess_item]),
        ),
        (
            "V6",
            "speed, memory, trace, and reverse-cost scorecard",
            scorecard_ok,
            evidence_labels([run_item, profile_item]),
        ),
    ]
    return [
        "| {} | {} | {} | {} |".format(
            phase,
            goal,
            status_cell(passed),
            evidence,
        )
        for phase, goal, passed, evidence in rows
    ]


def render_goal_readiness_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    summary_item = first_pipeline_summary(reports)
    if summary_item is not None:
        _, summary = summary_item
        return [
            "| {} | {} | {} | {} |".format(
                row["goal"],
                status_cell(bool(row["passed"])),
                ", ".join(row["evidence"]) if row["evidence"] else "missing",
                row["description"],
            )
            for row in summary["metrics"]["ml_goal_readiness"]["goals"]
        ]
    run_item = first_report(reports, profile_check.RUN_KIND)
    step_item = first_report(reports, profile_check.STEP_VERIFICATION_KIND)
    audit_step_item = first_report(reports, profile_check.AUDIT_STEP_KIND)
    profile_item = first_report(reports, profile_check.COMPARISON_KIND)
    q31_inference_item = first_report(reports, profile_check.Q31_REFERENCE_INFERENCE_KIND)
    q31_eval_item = first_report(reports, profile_check.Q31_REFERENCE_EVALUATION_KIND)
    mlp_item = first_report(reports, profile_check.MLP_WITNESS_KIND)
    coupling_item = first_report(reports, profile_check.INVERTIBLE_COUPLING_KIND)
    preprocess_item = first_report(reports, profile_check.REVERSIBLE_PREPROCESS_KIND)
    trace_item = first_report(reports, profile_check.REVERSIBLE_INFERENCE_TRACE_KIND)
    inference_verifications = [
        (path, report)
        for path, report in reports
        if report.get("kind") == profile_check.INFERENCE_VERIFICATION_KIND
    ]

    def labels(items: list[Optional[tuple[Path, dict[str, Any]]]]) -> str:
        names = [
            report_label(path)
            for item in items
            if item is not None
            for path, _ in (item,)
        ]
        return ", ".join(names) if names else "missing"

    update = audit_step_item[1].get("update", {}) if audit_step_item is not None else {}
    audit_contract = profile_item[1].get("audit_contract", {}) if profile_item is not None else {}
    coupling_proof = coupling_item[1].get("proof", {}) if coupling_item is not None else {}
    preprocess_proof = preprocess_item[1].get("proof", {}) if preprocess_item is not None else {}
    trace_proof = trace_item[1].get("proof", {}) if trace_item is not None else {}

    debug_passed = (
        audit_step_item is not None
        and step_item is not None
        and audit_step_item[1].get("debug_contract", {}).get("passed") is True
        and step_item[1].get("proof_matches") is True
        and step_item[1].get("reverse", {}).get("before_model_restored") is True
        and isinstance(update.get("bias_delta_ledger_fingerprint"), str)
        and isinstance(update.get("weight_delta_ledger_fingerprint"), str)
    )
    lineage_passed = (
        run_item is not None
        and profile_item is not None
        and run_item[1].get("reverse", {}).get("restored_initial_model") is True
        and audit_contract.get("passed") is True
    )
    inference_passed = (
        q31_inference_item is not None
        and q31_eval_item is not None
        and trace_item is not None
        and trace_item[1].get("passed") is True
        and trace_proof.get("trace_payload_bytes") == 0
        and trace_proof.get("checks", {}).get("reverse_restores_initial_state") is True
        and bool(inference_verifications)
        and all(
            report.get("proof_matches") is True
            and report.get("result_matches") is True
            and report.get("restored_initial_state") is True
            and report.get("explanation_contract", {}).get("passed") is True
            for _, report in inference_verifications
        )
    )
    memory_passed = bool(render_scorecard_rows(reports))
    invertible_passed = (
        mlp_item is not None
        and coupling_item is not None
        and preprocess_item is not None
        and mlp_item[1].get("passed") is True
        and coupling_item[1].get("passed") is True
        and preprocess_item[1].get("passed") is True
        and coupling_proof.get("witness_payload_bytes") == 0
        and coupling_proof.get("trace_payload_bytes") == 0
        and preprocess_proof.get("witness_payload_bytes") == 0
        and preprocess_proof.get("trace_payload_bytes") == 0
    )

    rows = [
        (
            "Debug training updates backward",
            debug_passed,
            labels([audit_step_item, step_item]),
            "selected update replay plus SHA-256 delta ledgers",
        ),
        (
            "Prove auditable model lineage",
            lineage_passed,
            labels([run_item, profile_item]),
            "reverse restored initial model and audit contract passed",
        ),
        (
            "Run deterministic reversible inference traces",
            inference_passed,
            labels([q31_inference_item, q31_eval_item, trace_item]),
            "Q31 reference plus replayed inference explanations and end-to-end trace",
        ),
        (
            "Measure memory/recompute tradeoffs",
            memory_passed,
            labels([run_item, profile_item]),
            "speed, peak RSS, replay payload, and reverse-cost scorecard",
        ),
        (
            "Show invertible model patterns",
            invertible_passed,
            labels([mlp_item, coupling_item, preprocess_item]),
            "MLP witnesses plus zero-witness coupling and preprocessing replay",
        ),
    ]
    return [
        "| {} | {} | {} | {} |".format(
            goal,
            status_cell(passed),
            evidence,
            proof,
        )
        for goal, passed, evidence, proof in rows
    ]


def render_scorecard_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    summary_item = first_pipeline_summary(reports)
    if summary_item is not None:
        _, summary = summary_item
        scorecard = summary["metrics"]["scorecard"]
        contracts = scorecard["contracts"]
        return [
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                number(scorecard["train_updates_per_second"]),
                number(scorecard["model_evaluation_samples_per_second"]),
                ratio_cell(scorecard["reverse_to_train_elapsed_ratio"]),
                bytes_cell(scorecard["run_peak_rss_bytes"]),
                bytes_cell(scorecard["total_trace_payload_bytes"]),
                bytes_cell(scorecard["total_witness_payload_bytes"]),
                bytes_cell(scorecard["max_replay_payload_bytes"]),
                comma(scorecard["total_recompute_steps"]),
                status_cell(bool(summary["gates"]["passed"])),
                status_cell(bool(contracts["training_step_debug"])),
                status_cell(bool(contracts["native_inference_explanation"])),
                status_cell(bool(contracts["reversible_inference_trace"])),
                status_cell(bool(contracts["q31_reference_inference"])),
                status_cell(bool(contracts["mlp_witness"])),
                status_cell(bool(contracts["invertible_coupling"])),
                status_cell(bool(contracts["triangular_residual"])),
                status_cell(bool(contracts["reversible_preprocess"])),
            )
        ]
    run_item = first_report(reports, profile_check.RUN_KIND)
    profile_item = first_report(reports, profile_check.COMPARISON_KIND)
    if run_item is None or profile_item is None:
        return []
    _, run = run_item
    _, comparison = profile_item
    train = run["train"]
    eval_report = run["eval"]
    reverse = run.get("reverse") if isinstance(run.get("reverse"), dict) else {}
    memory = run["memory"]
    profile = comparison["ml_profile"]
    train_elapsed = float(train.get("elapsed_seconds", 0.0))
    reverse_elapsed = (
        float(reverse.get("elapsed_seconds", 0.0))
        if reverse.get("enabled")
        else 0.0
    )
    reverse_ratio = reverse_elapsed / train_elapsed if train_elapsed > 0.0 else None
    max_replay = max(
        (value for _, report in reports for value in replay_payload_candidates(report)),
        default=None,
    )
    audit_contract = comparison.get("audit_contract")
    audit_contract_ok = isinstance(audit_contract, dict) and audit_contract.get("passed") is True
    debug_ok = any(
        report.get("kind") == profile_check.AUDIT_STEP_KIND
        and report.get("debug_contract", {}).get("passed") is True
        for _, report in reports
    )
    inference_reports = [
        report
        for _, report in reports
        if report.get("kind") == profile_check.INFERENCE_VERIFICATION_KIND
    ]
    inference_ok = bool(inference_reports) and all(
        report.get("explanation_contract", {}).get("passed") is True
        for report in inference_reports
    )
    native_item = first_report(reports, profile_check.MODEL_INFERENCE_AUDIT_KIND)
    reference_item = first_report(reports, profile_check.Q31_REFERENCE_INFERENCE_KIND)
    q31_reference_ok = False
    if native_item is not None and reference_item is not None:
        _, native_report = native_item
        _, reference_report = reference_item
        native_attribution = native_report.get("attribution", {})
        reference_attribution = reference_report.get("attribution", {})
        native_active_pixels = native_report.get("active_pixels")
        q31_reference_ok = (
            reference_report.get("prediction") == native_report.get("prediction")
            and reference_report.get("correct") == native_report.get("correct")
            and reference_attribution.get("margin") == native_attribution.get("margin")
            and isinstance(native_active_pixels, list)
            and reference_report.get("active_pixels") == len(native_active_pixels)
            and reference_attribution.get("matches_logit") is True
            and reference_attribution.get("matches_margin") is True
        )
    mlp_reports = [
        report
        for _, report in reports
        if report.get("kind") == profile_check.MLP_WITNESS_KIND
    ]
    mlp_ok = bool(mlp_reports) and all(report.get("passed") is True for report in mlp_reports)
    coupling_reports = [
        report
        for _, report in reports
        if report.get("kind") == profile_check.INVERTIBLE_COUPLING_KIND
    ]
    coupling_ok = bool(coupling_reports) and all(report.get("passed") is True for report in coupling_reports)
    preprocess_reports = [
        report
        for _, report in reports
        if report.get("kind") == profile_check.REVERSIBLE_PREPROCESS_KIND
    ]
    preprocess_ok = bool(preprocess_reports) and all(
        report.get("passed") is True for report in preprocess_reports
    )
    trace_reports = [
        report
        for _, report in reports
        if report.get("kind") == profile_check.REVERSIBLE_INFERENCE_TRACE_KIND
    ]
    trace_ok = bool(trace_reports) and all(
        report.get("passed") is True
        and report.get("proof", {}).get("trace_payload_bytes") == 0
        and report.get("proof", {}).get("checks", {}).get("reverse_restores_initial_state") is True
        for report in trace_reports
    )
    return [
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            number(train["updates_per_second"]),
            number(eval_report["samples_per_second"]),
            ratio_cell(reverse_ratio),
            bytes_cell(memory.get("peak_rss_bytes")),
            bytes_cell(profile["total_trace_payload_bytes"]),
            bytes_cell(profile["total_witness_payload_bytes"]),
            bytes_cell(max_replay),
            comma(profile["total_recompute_steps"]),
            status_cell(audit_contract_ok),
            status_cell(debug_ok),
            status_cell(inference_ok),
            status_cell(trace_ok),
            status_cell(q31_reference_ok),
            status_cell(mlp_ok),
            status_cell(coupling_ok),
            status_cell(preprocess_ok),
        )
    ]


def render_run_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.RUN_KIND:
            continue
        train = report["train"]
        eval_report = report["eval"]
        reverse_report = report.get("reverse")
        reverse = reverse_report if isinstance(reverse_report, dict) else {}
        trace = report["trace"]
        proof = report["proof"]
        memory = report["memory"]
        reverse_steps = reverse.get("checked", 0) if reverse.get("enabled") else 0
        reverse_rate = reverse.get("steps_per_second", 0.0) if reverse.get("enabled") else 0.0
        train_elapsed = float(train.get("elapsed_seconds", 0.0))
        reverse_elapsed = (
            float(reverse.get("elapsed_seconds", 0.0))
            if reverse.get("enabled")
            else 0.0
        )
        reverse_train_ratio = reverse_elapsed / train_elapsed if train_elapsed > 0.0 else None
        recompute = (
            proof.get("forward_recompute_steps", 0)
            + proof.get("inverse_recompute_steps", 0)
            if proof.get("enabled")
            else 0
        )
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                comma(train["samples"]),
                number(train["updates_per_second"]),
                number(train_elapsed),
                comma(eval_report["samples"]),
                number(eval_report["samples_per_second"]),
                comma(reverse_steps),
                number(reverse_rate),
                number(reverse_elapsed),
                ratio_cell(reverse_train_ratio),
                bytes_cell(memory.get("peak_rss_bytes")),
                bytes_cell(trace["payload_bytes"]),
                bytes_cell(memory["model_payload_bytes"]),
                bytes_cell(memory["dataset_payload_bytes"]),
                bytes_cell(memory["estimated_payload_bytes"]),
                comma(recompute),
            )
        )
    return rows


def render_artifact_profile_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.COMPARISON_KIND:
            continue
        profile = report["ml_profile"]
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                comma(report["totals"]["count"]),
                bytes_cell(profile["total_file_bytes"]),
                bytes_cell(profile["total_logical_payload_bytes"]),
                bytes_cell(profile["total_model_payload_bytes"]),
                bytes_cell(profile["total_sample_payload_bytes"]),
                bytes_cell(profile["total_witness_payload_bytes"]),
                bytes_cell(profile["total_trace_payload_bytes"]),
                bytes_cell(profile["total_derived_update_payload_bytes"]),
                comma(profile["total_steps"]),
                comma(profile["total_recompute_steps"]),
                ratio_cell(profile["trace_to_model_payload_ratio"]),
                ratio_cell(profile["witness_to_model_payload_ratio"]),
                ratio_cell(profile["logical_to_file_ratio"]),
            )
        )
    return rows


def render_artifact_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.COMPARISON_KIND:
            continue
        for artifact in report["artifacts"]:
            recompute = artifact["forward_recompute_steps"] + artifact["inverse_recompute_steps"]
            fingerprints = artifact["fingerprints"]
            proof_or_provenance = (
                "proof"
                if fingerprints["has_proof"]
                else "provenance"
                if fingerprints["has_provenance"]
                else "none"
            )
            rows.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    report_label(path),
                    artifact["kind"],
                    bytes_cell(artifact["file_bytes"]),
                    bytes_cell(artifact["logical_payload_bytes"]),
                    bytes_cell(artifact["model_payload_bytes"]),
                    bytes_cell(artifact["sample_payload_bytes"]),
                    bytes_cell(artifact["witness_payload_bytes"]),
                    bytes_cell(artifact["trace_payload_bytes"]),
                    comma(artifact["steps"]),
                    comma(recompute),
                    comma(fingerprints["count"]),
                    comma(fingerprints["source_count"]),
                    proof_or_provenance,
                )
            )
    return rows


def render_audit_contract_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.COMPARISON_KIND:
            continue
        contract = report.get("audit_contract")
        if not isinstance(contract, dict):
            continue
        checks = contract.get("checks", [])
        failed = [
            check.get("metric", "unknown")
            for check in checks
            if isinstance(check, dict) and check.get("passed") is False
        ]
        rows.append(
            "| {} | {} | {} | {} | {} |".format(
                report_label(path),
                contract.get("claim", "unknown"),
                str(contract.get("passed")).lower(),
                comma(len(checks)) if isinstance(checks, list) else "n/a",
                ", ".join(failed) if failed else "none",
            )
        )
    return rows


def render_mlp_witness_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.MLP_WITNESS_KIND:
            continue
        proof = report["proof"]
        dataset_loops = ", ".join(
            "{} by {}".format(
                loop["index"],
                ",".join(loop["size_sources"]),
            )
            for loop in report["dataset_loops"]
        )
        witness_proof = report.get("witness_proof")
        witness_proof_hash = (
            short_hash(witness_proof["fingerprint"]) if isinstance(witness_proof, dict) else "n/a"
        )
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                proof["claim"],
                str(report["passed"]).lower(),
                witness_proof_hash,
                comma(report["samples"]),
                dataset_loops,
                json.dumps(report["predictions"], separators=(",", ":")),
                comma(sum(1 for item in report["correct"] if item)),
                bytes_cell(proof["witness_payload_bytes"]),
                bytes_cell(proof["trace_payload_bytes"]),
                bytes_cell(proof["replay_payload_bytes"]),
                bytes_cell(proof["recomputed_update_payload_bytes"]),
                comma(proof["total_recompute_steps"]),
                ratio_cell(proof["witness_to_model_payload_ratio"]),
                ratio_cell(proof["trace_to_model_payload_ratio"]),
                ratio_cell(proof["recomputed_update_to_witness_payload_ratio"]),
            )
        )
    return rows


def render_invertible_coupling_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.INVERTIBLE_COUPLING_KIND:
            continue
        proof = report["proof"]
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                proof["claim"],
                str(report["passed"]).lower(),
                json.dumps(report["forward"]["left"], separators=(",", ":")),
                json.dumps(report["forward"]["right"], separators=(",", ":")),
                bytes_cell(proof["model_payload_bytes"]),
                bytes_cell(proof["state_payload_bytes"]),
                bytes_cell(proof["witness_payload_bytes"]),
                bytes_cell(proof["trace_payload_bytes"]),
                bytes_cell(proof["replay_payload_bytes"]),
                comma(proof["total_recompute_steps"]),
                ratio_cell(proof["trace_to_model_payload_ratio"]),
            )
        )
    return rows


def render_triangular_residual_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.TRIANGULAR_RESIDUAL_KIND:
            continue
        proof = report["proof"]
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                proof["claim"],
                str(report["passed"]).lower(),
                json.dumps(report["forward"]["x"], separators=(",", ":")),
                bytes_cell(proof["parameter_payload_bytes"]),
                bytes_cell(proof["state_payload_bytes"]),
                bytes_cell(proof["witness_payload_bytes"]),
                bytes_cell(proof["trace_payload_bytes"]),
                bytes_cell(proof["replay_payload_bytes"]),
                comma(proof["total_recompute_steps"]),
                ratio_cell(proof["trace_to_parameter_payload_ratio"]),
            )
        )
    return rows


def render_reversible_preprocess_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.REVERSIBLE_PREPROCESS_KIND:
            continue
        proof = report["proof"]
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                proof["claim"],
                str(report["passed"]).lower(),
                json.dumps(report["forward"]["features"], separators=(",", ":")),
                bytes_cell(proof["raw_payload_bytes"]),
                bytes_cell(proof["mean_payload_bytes"]),
                bytes_cell(proof["feature_payload_bytes"]),
                bytes_cell(proof["witness_payload_bytes"]),
                bytes_cell(proof["trace_payload_bytes"]),
                bytes_cell(proof["replay_payload_bytes"]),
                comma(proof["total_recompute_steps"]),
                ratio_cell(proof["trace_to_state_payload_ratio"]),
            )
        )
    return rows


def render_reversible_inference_trace_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.REVERSIBLE_INFERENCE_TRACE_KIND:
            continue
        proof = report["proof"]
        attribution = report["attribution"]
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                proof["claim"],
                str(report["passed"]).lower(),
                json.dumps(report["forward"]["logits"], separators=(",", ":")),
                report["forward"]["prediction"],
                report["forward"]["correct"],
                comma(attribution["margin"]),
                attribution["runner_up_class"],
                "ok" if attribution["matches_logit"] and attribution["matches_margin"] else "failed",
                comma(attribution["contribution_count"]),
                short_hash(attribution["contribution_ledger_fingerprint"]),
                bytes_cell(proof["model_payload_bytes"]),
                bytes_cell(proof["state_payload_bytes"]),
                bytes_cell(proof["witness_payload_bytes"]),
                bytes_cell(proof["trace_payload_bytes"]),
                bytes_cell(proof["replay_payload_bytes"]),
                comma(proof["total_recompute_steps"]),
                ratio_cell(proof["witness_to_model_payload_ratio"]),
            )
        )
    return rows


def render_q31_reference_inference_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.Q31_REFERENCE_INFERENCE_KIND:
            continue
        attribution = report["attribution"]
        attribution_ok = attribution["matches_logit"] and attribution["matches_margin"]
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                report["prediction"],
                report["label"],
                str(report["correct"]).lower(),
                comma(report["active_pixels"]),
                comma(attribution["margin"]),
                attribution["runner_up_digit"],
                "ok" if attribution_ok else "failed",
                comma(attribution["contribution_count"]),
                short_hash(attribution["contribution_ledger_fingerprint"]),
                json.dumps(report["top_logits"], separators=(",", ":")),
            )
        )
    return rows


def render_q31_reference_evaluation_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.Q31_REFERENCE_EVALUATION_KIND:
            continue
        summary = report["summary"]
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                comma(summary["samples"]),
                comma(summary["correct"]),
                comma(summary["incorrect"]),
                percent(summary["accuracy_percent"]),
                "n/a" if summary["lowest_margin_index"] is None else comma(summary["lowest_margin_index"]),
                "n/a" if summary["lowest_margin"] is None else comma(summary["lowest_margin"]),
                comma(len(report["top_low_margin"])),
                comma(len(report["top_incorrect"])),
            )
        )
    return rows


def inference_ledger(report: dict[str, Any]) -> dict[str, Any]:
    attribution = report["attribution"]
    return {
        "contribution_count": attribution["contribution_count"],
        "margin_contribution_count": attribution["margin_contribution_count"],
        "contribution": attribution["contribution_ledger_fingerprint"],
        "margin": attribution["margin_contribution_ledger_fingerprint"],
    }


def active_pixel_count(report: dict[str, Any]) -> int:
    active_pixels = report.get("active_pixels")
    if active_pixels is None:
        active_pixels = report["proof"]["active_pixels"]
    if isinstance(active_pixels, list):
        return len(active_pixels)
    return active_pixels


def inference_trace_result_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["prediction"] == right["prediction"]
        and left["correct"] == right["correct"]
        and left["attribution"]["margin"] == right["attribution"]["margin"]
        and active_pixel_count(left) == active_pixel_count(right)
    )


def render_inference_trace_profile_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    native_item = first_report(reports, profile_check.MODEL_INFERENCE_AUDIT_KIND)
    row_item = first_report(reports, profile_check.MODEL_EVALUATION_ROW_KIND)
    reference_item = first_report(reports, profile_check.Q31_REFERENCE_INFERENCE_KIND)
    native_verification_item: Optional[tuple[Path, dict[str, Any]]] = None
    row_verification_item: Optional[tuple[Path, dict[str, Any]]] = None
    for item in reports:
        _, report = item
        if report.get("kind") != profile_check.INFERENCE_VERIFICATION_KIND:
            continue
        if report.get("source_evaluation_checked") is True:
            row_verification_item = item
        elif native_verification_item is None:
            native_verification_item = item
    rows = []

    def row(
        trace: str,
        report_item: Optional[tuple[Path, dict[str, Any]]],
        verification_item: Optional[tuple[Path, dict[str, Any]]],
        *,
        reference_required: bool,
    ) -> None:
        if report_item is None or verification_item is None:
            return
        report_path, report = report_item
        _, verification = verification_item
        report_ledger = inference_ledger(report)
        verification_ledger = inference_ledger(verification)
        reference_cell = "n/a"
        reference_ok = not reference_required
        if reference_required and reference_item is not None:
            _, reference = reference_item
            reference_ledger = inference_ledger(reference)
            reference_ok = (
                inference_trace_result_matches(report, reference)
                and report_ledger == reference_ledger
            )
            reference_cell = (
                "ok "
                + short_hash(reference_ledger["contribution"])
                + "/"
                + short_hash(reference_ledger["margin"])
            )
        elif reference_required:
            reference_cell = "missing"
        result_ok = inference_trace_result_matches(report, verification)
        ledger_ok = report_ledger == verification_ledger
        replay_ok = (
            verification.get("proof_matches") is True
            and verification.get("result_matches") is True
            and verification.get("restored_initial_state") is True
        )
        source_bits = []
        if verification.get("source_model_checked"):
            source_bits.append("model")
        if verification.get("source_sample_checked"):
            source_bits.append("sample")
        if verification.get("source_evaluation_checked"):
            source_bits.append("evaluation")
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                trace,
                report_label(report_path),
                report.get("sample_source", {}).get("kind", "unknown"),
                report["prediction"],
                str(report["correct"]).lower(),
                comma(report["attribution"]["margin"]),
                comma(report_ledger["contribution_count"]),
                status_cell(result_ok),
                status_cell(ledger_ok),
                reference_cell if reference_ok else "failed",
                status_cell(replay_ok),
                ",".join(source_bits) if source_bits else "none",
            )
        )

    row(
        "selected audit sample",
        native_item,
        native_verification_item,
        reference_required=True,
    )
    row(
        "model evaluation row",
        row_item,
        row_verification_item,
        reference_required=False,
    )
    return rows


def render_audit_step_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    audit_scan_item = first_report(reports, profile_check.AUDIT_SCAN_KIND)
    audit_scan = audit_scan_item[1] if audit_scan_item is not None else None
    for path, report in reports:
        if report.get("kind") != profile_check.AUDIT_STEP_KIND:
            continue
        checks = report["witness_checks"]
        witness_ok = checks["prediction_matches_logits"] and checks["correct_matches_logits"]
        update = report["update"]
        window = report["model_window"]
        model_ok = window["bias_delta_matches"] and window["weight_delta_matches"]
        debug = report["debug_contract"]
        scan_evidence = ["explicit"]
        if audit_scan is not None:
            summary = audit_scan["summary"]
            step = report["step"]
            scan_evidence = []
            if step == summary.get("lowest_margin_step"):
                scan_evidence.append("lowest-margin")
            if step == summary.get("largest_update_step"):
                scan_evidence.append("largest-update")
            if any(row.get("step") == step for row in audit_scan.get("top_suspicious", [])):
                scan_evidence.append("top-suspicious")
            if any(row.get("step") == step for row in audit_scan.get("top_low_margin", [])):
                scan_evidence.append("top-low-margin")
            if any(row.get("step") == step for row in audit_scan.get("top_large_updates", [])):
                scan_evidence.append("top-large-updates")
            if not scan_evidence:
                scan_evidence.append("explicit")
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                comma(report["step"]),
                comma(report["sample_index"]),
                report["label"],
                report["prediction"],
                str(report["correct"]).lower(),
                comma(report["logit_margin"]["margin"]),
                comma(len(report["active_pixels"])),
                "ok" if witness_ok else "mismatch",
                comma(update["max_abs_weight_delta"]),
                comma(len(update["top_weight_deltas"])),
                comma(update["nonzero_weight_delta_count"]),
                short_hash(update["weight_delta_ledger_fingerprint"]),
                short_hash(report["cause_ledger"]["fingerprint"]),
                "ok" if model_ok else "mismatch",
                ",".join(scan_evidence),
                "passed" if debug["passed"] else "failed",
            )
        )
    return rows


def render_nonzero_vector_rows(values: list[int], label: str) -> list[str]:
    rows = [
        f"| {index} | {comma(value)} |"
        for index, value in enumerate(values)
        if value != 0
    ]
    if rows:
        return rows
    return [f"| n/a | no nonzero {label} |"]


def render_training_step_debug_markdown(
    audit_step: dict[str, Any],
    step_verification: Optional[dict[str, Any]] = None,
) -> str:
    update = audit_step["update"]
    window = audit_step["model_window"]
    margin = audit_step["logit_margin"]
    contract = audit_step["debug_contract"]
    cause = audit_step["cause_ledger"]
    proof = None if step_verification is None else step_verification["proof"]
    proof_recompute_steps = (
        None
        if proof is None
        else proof["forward_recompute_steps"] + proof["inverse_recompute_steps"]
    )
    window_by_delta = {
        (row["pixel"], row["digit"]): row
        for row in window.get("top_weight_windows", [])
    }
    lines = [
        "# Reverie Training Step Debug",
        "",
        "| Step | Sample | Label | Prediction | Correct | Runner-up | Margin | Active pixels | LR |",
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            comma(audit_step["step"]),
            comma(audit_step["sample_index"]),
            audit_step["label"],
            audit_step["prediction"],
            str(audit_step["correct"]).lower(),
            margin["runner_up_digit"],
            comma(margin["margin"]),
            comma(len(audit_step["active_pixels"])),
            comma(audit_step["lr"]),
        ),
        "",
        "## Debug Contract",
        "",
        "| Claim | Passed | Reversed later steps | Model window | Update deltas |",
        "| --- | --- | ---: | --- | --- |",
        "| `{}` | {} | {} | {} | {} |".format(
            contract["claim"],
            str(contract["passed"]).lower(),
            comma(window["reversed_later_steps"]),
            "ok" if window["reconstructed"] else "failed",
            "ok" if window["bias_delta_matches"] and window["weight_delta_matches"] else "failed",
        ),
        "",
        "| Check | Passed | Evidence |",
        "| --- | --- | --- |",
    ]
    for check in contract["checks"]:
        lines.append(
            "| `{}` | {} | {} |".format(
                check["metric"],
                str(check["passed"]).lower(),
                check["actual"],
            )
        )
    lines.extend(
        [
            "",
            "## Witnesses",
            "",
            "| Top logit | Digit | Value |",
            "| ---: | ---: | ---: |",
        ]
    )
    for rank, row in enumerate(audit_step["top_logits"], start=1):
        lines.append(f"| {rank} | {row['digit']} | {comma(row['logit'])} |")
    lines.extend(
        [
            "",
            "| Error digit | Error |",
            "| ---: | ---: |",
            *render_nonzero_vector_rows(audit_step["error"], "errors"),
            "",
            "| Active pixel | U8 | Q31 |",
            "| ---: | ---: | ---: |",
        ]
    )
    for row in audit_step["active_pixels"]:
        lines.append(f"| {row['index']} | {row['u8']} | {comma(row['q31'])} |")
    lines.extend(
        [
            "",
            "## Top Weight Deltas",
            "",
            "| Pixel | Digit | Computed delta | Before | After | Observed delta | Match |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in update["top_weight_deltas"]:
        observed = window_by_delta.get((row["pixel"], row["digit"]), {})
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                row["pixel"],
                row["digit"],
                comma(row["delta"]),
                comma(observed.get("before", 0)),
                comma(observed.get("after", 0)),
                comma(observed.get("observed_delta", row["delta"])),
                "ok" if observed.get("delta_matches", True) else "failed",
            )
        )
    lines.extend(
        [
            "",
            "## Bias Deltas",
            "",
            "| Digit | Delta |",
            "| ---: | ---: |",
            *render_nonzero_vector_rows(update["bias_delta"], "bias deltas"),
            "",
            "## Ledgers",
            "",
            "| Ledger | Fingerprint |",
            "| --- | --- |",
            f"| cause | `{short_hash(cause['fingerprint'])}` |",
            f"| update | `{short_hash(cause['payload']['update_fingerprint'])}` |",
            f"| witness | `{short_hash(cause['payload']['witness_fingerprint'])}` |",
            f"| bias delta | `{short_hash(update['bias_delta_ledger_fingerprint'])}` |",
            f"| weight delta | `{short_hash(update['weight_delta_ledger_fingerprint'])}` |",
            "",
        ]
    )
    if proof is not None:
        lines.extend(
            [
                "## Replay Proof",
                "",
                "| Claim | Forward | Reverse | Proof | Recompute steps | Witness bytes | Update bytes |",
                "| --- | --- | --- | --- | ---: | ---: | ---: |",
                "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                    proof["claim"],
                    "ok" if step_verification["forward"]["after_model_matches"] else "failed",
                    "ok" if step_verification["reverse"]["before_model_restored"] else "failed",
                    "ok" if step_verification["proof_matches"] else "failed",
                    comma(proof_recompute_steps),
                    comma(proof["witness_payload_bytes"]),
                    comma(proof["derived_update_payload_bytes"]),
                ),
                "",
                "| Proof check | Passed |",
                "| --- | --- |",
            ]
        )
        for key, value in proof["checks"].items():
            lines.append(f"| `{key}` | {str(value).lower()} |")
        lines.append("")
    return "\n".join(lines)


def render_audit_verification_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.AUDIT_VERIFICATION_KIND:
            continue
        lineage = report["lineage_ledger"]
        lineage_payload = lineage["payload"]
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                comma(report["checked"]),
                "ok" if report["witnesses_match_forward_replay"] else "mismatch",
                "ok" if report["final_model_replayed"] else "mismatch",
                "ok" if report["restored_initial_model"] else "mismatch",
                "ok" if report["proof_matches"] else "mismatch",
                short_hash(lineage["fingerprint"]),
                short_hash(lineage_payload["transition_ledger_fingerprint"]),
                short_hash(lineage_payload["final_chain"]),
            )
        )
    return rows


def render_audit_scan_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.AUDIT_SCAN_KIND:
            continue
        summary = report["summary"]
        gate = report.get("gate")
        gate_cell = "n/a" if gate is None else ("passed" if gate["passed"] else "failed")
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                comma(summary["steps"]),
                comma(summary["correct"]),
                comma(summary["incorrect"]),
                percent(summary["accuracy_percent"]),
                comma(summary["witness_mismatches"]),
                "n/a" if summary["lowest_margin_step"] is None else comma(summary["lowest_margin_step"]),
                "n/a" if summary["lowest_margin"] is None else comma(summary["lowest_margin"]),
                "n/a" if summary["largest_update_step"] is None else comma(summary["largest_update_step"]),
                comma(summary["max_abs_weight_delta"]),
                comma(len(report["top_suspicious"])),
                gate_cell,
            )
        )
    return rows


def render_step_verification_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.STEP_VERIFICATION_KIND:
            continue
        proof = report["proof"]
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                proof["claim"],
                "ok" if report["proof_matches"] else "missing",
                bytes_cell(proof["model_payload_bytes"]),
                bytes_cell(proof["sample_payload_bytes"]),
                bytes_cell(proof["witness_payload_bytes"]),
                bytes_cell(proof["derived_update_payload_bytes"]),
                bytes_cell(proof["replay_payload_bytes"]),
                "ok" if report["forward"]["after_model_matches"] else "failed",
                "ok" if report["reverse"]["before_model_restored"] else "failed",
            )
        )
    return rows


def render_native_inference_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    evaluation_scan_item = first_report(reports, profile_check.MODEL_EVALUATION_SCAN_KIND)
    evaluation_scan = evaluation_scan_item[1] if evaluation_scan_item is not None else None
    supported = {
        profile_check.INFERENCE_AUDIT_KIND,
        profile_check.MODEL_INFERENCE_AUDIT_KIND,
        profile_check.MODEL_EVALUATION_ROW_KIND,
    }
    for path, report in reports:
        if report.get("kind") not in supported:
            continue
        attribution = report["attribution"]
        proof = report["proof"]
        explanation = report["explanation_contract"]
        source = report.get("sample_source", {}).get("kind")
        if source is None and report.get("kind") == profile_check.INFERENCE_AUDIT_KIND:
            source = "training_audit"
        row_evidence = "n/a"
        if report.get("kind") == profile_check.MODEL_EVALUATION_ROW_KIND and evaluation_scan is not None:
            sample_source = report.get("sample_source", {})
            row_block = report.get("row", {})
            row_index = sample_source.get("row_index", row_block.get("index"))
            evidence = []
            if row_index == evaluation_scan["summary"].get("lowest_margin_index"):
                evidence.append("lowest-margin")
            if any(row.get("index") == row_index for row in evaluation_scan.get("top_low_margin", [])):
                evidence.append("top-low-margin")
            if any(row.get("index") == row_index for row in evaluation_scan.get("top_incorrect", [])):
                evidence.append("top-incorrect")
            row_evidence = ",".join(evidence) if evidence else "explicit"
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                report["kind"],
                source or "unknown",
                report["prediction"],
                report["label"],
                str(report["correct"]).lower(),
                comma(len(report["active_pixels"])),
                comma(attribution["margin"]),
                "ok" if proof["checks"]["restored_initial_state"] else "failed",
                "passed" if explanation["passed"] else "failed",
                row_evidence,
                comma(proof["contribution_count"]),
                short_hash(proof["contribution_ledger_fingerprint"]),
                bytes_cell(proof["replay_payload_bytes"]),
                comma(proof["inverse_recompute_steps"]),
            )
        )
    return rows


def render_inference_verification_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.INFERENCE_VERIFICATION_KIND:
            continue
        proof = report["proof"]
        explanation = report["explanation_contract"]
        source_bits = []
        for label, field in (
            ("model", "source_model_checked"),
            ("sample", "source_sample_checked"),
            ("training", "source_training_checked"),
            ("evaluation", "source_evaluation_checked"),
        ):
            if report.get(field):
                source_bits.append(label)
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                report["prediction"],
                str(report["correct"]).lower(),
                "ok" if report["proof_matches"] else "failed",
                "ok" if report["result_matches"] else "failed",
                "ok" if report["restored_initial_state"] else "failed",
                "passed" if explanation["passed"] else "failed",
                comma(proof["contribution_count"]),
                short_hash(proof["contribution_ledger_fingerprint"]),
                bytes_cell(proof["replay_payload_bytes"]),
                comma(proof["inverse_recompute_steps"]),
                ",".join(source_bits) if source_bits else "none",
            )
        )
    return rows


def render_model_evaluation_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.MODEL_EVALUATION_KIND:
            continue
        summary = report["summary"]
        proof = report["proof"]
        gate = report.get("gate")
        gate_cell = "n/a" if gate is None else ("passed" if gate["passed"] else "failed")
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                comma(summary["samples"]),
                comma(summary["correct"]),
                comma(summary["incorrect"]),
                percent(summary["accuracy_percent"]),
                "n/a" if summary["lowest_margin_index"] is None else comma(summary["lowest_margin_index"]),
                "n/a" if summary["lowest_margin"] is None else comma(summary["lowest_margin"]),
                bytes_cell(proof["replay_payload_bytes"]),
                comma(proof["inverse_recompute_steps"]),
                gate_cell,
            )
        )
    return rows


def render_model_evaluation_scan_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.MODEL_EVALUATION_SCAN_KIND:
            continue
        summary = report["summary"]
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                comma(summary["samples"]),
                comma(summary["correct"]),
                comma(summary["incorrect"]),
                percent(summary["accuracy_percent"]),
                "n/a" if summary["lowest_margin_index"] is None else comma(summary["lowest_margin_index"]),
                "n/a" if summary["lowest_margin"] is None else comma(summary["lowest_margin"]),
                comma(len(report["top_low_margin"])),
                comma(len(report["top_incorrect"])),
                comma(len(report["top_confusions"])),
            )
        )
    return rows


def render_model_evaluation_verification_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    rows = []
    for path, report in reports:
        if report.get("kind") != profile_check.MODEL_EVALUATION_VERIFICATION_KIND:
            continue
        summary = report["summary"]
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                report_label(path),
                comma(summary["samples"]),
                percent(summary["accuracy_percent"]),
                "ok" if report["rows_match"] else "failed",
                "ok" if report["proof_matches"] else "failed",
                "ok" if report["restored_initial_state"] else "failed",
                "yes" if report["source_model_checked"] else "no",
                "yes" if report["source_samples_checked"] else "no",
                number(float(report["elapsed_seconds"])),
            )
        )
    return rows


def render_markdown(reports: list[tuple[Path, dict[str, Any]]]) -> str:
    lines = [
        "# Reverie MNIST ML Profile",
        "",
        f"Reports: {len(reports)}",
        "",
    ]
    capsule_rows = render_model_capsule_rows(reports)
    if capsule_rows:
        lines.extend(
            [
                "## Model Capsules",
                "",
                "| Report | Capsule | Gates | Model | Samples | Accuracy | Lineage ledger | Final chain | Native replay | Q31 parity | Witness proof | Peak RSS | Max replay bytes | Readiness |",
                "| --- | --- | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
                *capsule_rows,
                "",
            ]
        )
    action_rows = render_inference_action_contract_rows(reports)
    if action_rows:
        lines.extend(
            [
                "## Inference Action Contract",
                "",
                "| Report | Operation | Supported | Evidence | Result | Cost |",
                "| --- | --- | --- | --- | --- | --- |",
                *action_rows,
                "",
            ]
        )
    readiness_rows = render_goal_readiness_rows(reports)
    if readiness_rows:
        lines.extend(
            [
                "## North-Star Readiness",
                "",
                "| Goal | Status | Evidence | Key proof |",
                "| --- | --- | --- | --- |",
                *readiness_rows,
                "",
            ]
        )
    capability_rows = render_capability_map_rows(reports)
    if capability_rows:
        lines.extend(
            [
                "## ML Capability Map",
                "",
                "| Phase | Goal | Status | Evidence |",
                "| --- | --- | --- | --- |",
                *capability_rows,
                "",
            ]
        )
    scorecard_rows = render_scorecard_rows(reports)
    if scorecard_rows:
        lines.extend(
            [
                "## V6 Scorecard",
                "",
                "| Train/s | Eval/s | Reverse/train | Peak RSS | Trace bytes | Witness bytes | Max replay bytes | Recompute steps | Audit contract | Debug | Inference | Rev trace | Q31 ref | MLP | Coupling | Residual | Preprocess |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                *scorecard_rows,
                "",
            ]
        )
    recompute_frontier_rows = render_recompute_frontier_rows(reports)
    if recompute_frontier_rows:
        lines.extend(
            [
                "## Recompute Frontier",
                "",
                "| Mode | Scope | Report | Replay bytes | Witness bytes | Trace bytes | Update bytes | State bytes | Forward steps | Inverse steps | Bytes/inverse |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *recompute_frontier_rows,
                "",
            ]
        )
    scaling_projection_rows = render_scaling_projection_rows(reports)
    if scaling_projection_rows:
        lines.extend(
            [
                "## Scaling Projection",
                "",
                "| Family | Unit | Report | Scale | Count | Projected replay bytes | Witness bytes | Trace bytes | Recomputed update bytes | Forward steps | Inverse steps |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *scaling_projection_rows,
                "",
            ]
        )
    run_rows = render_run_rows(reports)
    if run_rows:
        lines.extend(
            [
                "## Run Reports",
                "",
                "| Report | Train samples | Train/s | Train seconds | Eval samples | Eval/s | Reverse steps | Reverse/s | Reverse seconds | Reverse/train | Peak RSS | Trace bytes | Model bytes | Dataset bytes | Estimated bytes | Recompute steps |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *run_rows,
                "",
            ]
        )
    profile_rows = render_artifact_profile_rows(reports)
    if profile_rows:
        lines.extend(
            [
                "## Artifact Profiles",
                "",
                "| Report | Artifacts | File bytes | Logical bytes | Model bytes | Sample bytes | Witness bytes | Trace bytes | Update bytes | Steps | Recompute steps | Trace/model | Witness/model | Logical/file |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *profile_rows,
                "",
            ]
        )
        contract_rows = render_audit_contract_rows(reports)
        if contract_rows:
            lines.extend(
                [
                    "## Audit Contract",
                    "",
                    "| Report | Claim | Passed | Checks | Failed checks |",
                    "| --- | --- | --- | ---: | --- |",
                    *contract_rows,
                    "",
                ]
            )
        artifact_rows = render_artifact_rows(reports)
        if artifact_rows:
            lines.extend(
                [
                    "## Artifact Rows",
                    "",
                    "| Report | Kind | File bytes | Logical bytes | Model bytes | Sample bytes | Witness bytes | Trace bytes | Steps | Recompute steps | Fingerprints | Source fingerprints | Proof/provenance |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                    *artifact_rows,
                    "",
                ]
            )
    mlp_rows = render_mlp_witness_rows(reports)
    if mlp_rows:
        lines.extend(
            [
                "## MLP Witness Reports",
                "",
                "| Report | Claim | Passed | Witness proof | Samples | Dataset loop | Predictions | Correct | Witness bytes | Trace bytes | Replay bytes | Recomputed update bytes | Recompute steps | Witness/model | Trace/model | Update/witness |",
                "| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *mlp_rows,
                "",
            ]
        )
    coupling_rows = render_invertible_coupling_rows(reports)
    if coupling_rows:
        lines.extend(
            [
                "## Invertible Coupling Reports",
                "",
                "| Report | Claim | Passed | Forward left | Forward right | Model bytes | State bytes | Witness bytes | Trace bytes | Replay bytes | Recompute steps | Trace/model |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *coupling_rows,
                "",
            ]
        )
    residual_rows = render_triangular_residual_rows(reports)
    if residual_rows:
        lines.extend(
            [
                "## Triangular Residual Reports",
                "",
                "| Report | Claim | Passed | Forward x | Parameter bytes | State bytes | Witness bytes | Trace bytes | Replay bytes | Recompute steps | Trace/parameters |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *residual_rows,
                "",
            ]
        )
    preprocess_rows = render_reversible_preprocess_rows(reports)
    if preprocess_rows:
        lines.extend(
            [
                "## Reversible Preprocess Reports",
                "",
                "| Report | Claim | Passed | Forward features | Raw bytes | Mean bytes | Feature bytes | Witness bytes | Trace bytes | Replay bytes | Recompute steps | Trace/state |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *preprocess_rows,
                "",
            ]
        )
    inference_trace_rows = render_reversible_inference_trace_rows(reports)
    if inference_trace_rows:
        lines.extend(
            [
                "## Reversible Inference Trace Reports",
                "",
                "| Report | Claim | Passed | Forward logits | Prediction | Correct | Margin | Runner-up | Attribution | Contribution count | Contribution ledger | Model bytes | State bytes | Witness bytes | Trace bytes | Replay bytes | Recompute steps | Witness/model |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *inference_trace_rows,
                "",
            ]
        )
    q31_inference_rows = render_q31_reference_inference_rows(reports)
    if q31_inference_rows:
        lines.extend(
            [
                "## Q31 Reference Inference",
                "",
                "| Report | Prediction | Label | Correct | Active pixels | Margin | Runner-up | Attribution | Contribution count | Contribution ledger | Top logits |",
                "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- | --- |",
                *q31_inference_rows,
                "",
            ]
        )
    q31_evaluation_rows = render_q31_reference_evaluation_rows(reports)
    if q31_evaluation_rows:
        lines.extend(
            [
                "## Q31 Reference Evaluation",
                "",
                "| Report | Samples | Correct | Incorrect | Accuracy | Lowest margin index | Lowest margin | top_low_margin rows | top_incorrect rows |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *q31_evaluation_rows,
                "",
            ]
        )
    inference_trace_profile_rows = render_inference_trace_profile_rows(reports)
    if inference_trace_profile_rows:
        lines.extend(
            [
                "## Inference Trace Profile",
                "",
                "| Trace | Report | Source | Prediction | Correct | Margin | Contribution count | Result match | Ledger match | Reference ledger | Replay | Sources checked |",
                "| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- |",
                *inference_trace_profile_rows,
                "",
            ]
        )
    audit_verification_rows = render_audit_verification_rows(reports)
    if audit_verification_rows:
        lines.extend(
            [
                "## Training Audit Verification",
                "",
                "| Report | Checked | Witness replay | Final model | Reverse | Proof | Lineage ledger | Transition ledger | Final chain |",
                "| --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
                *audit_verification_rows,
                "",
            ]
        )
    audit_scan_rows = render_audit_scan_rows(reports)
    if audit_scan_rows:
        lines.extend(
            [
                "## Training Audit Scan",
                "",
                "| Report | Steps | Correct | Incorrect | Accuracy | Witness mismatches | Lowest margin step | Lowest margin | Largest update step | Max abs weight delta | Suspicious rows | Gate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                *audit_scan_rows,
                "",
            ]
        )
    audit_step_rows = render_audit_step_rows(reports)
    if audit_step_rows:
        lines.extend(
            [
                "## Training Audit Step",
                "",
                "| Report | Step | Sample | Label | Prediction | Correct | Margin | Active pixels | Witness | Max abs weight delta | Top deltas | Weight delta count | Update ledger | Cause ledger | Model delta | Scan evidence | Debug contract |",
                "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
                *audit_step_rows,
                "",
            ]
        )
    step_verification_rows = render_step_verification_rows(reports)
    if step_verification_rows:
        lines.extend(
            [
                "## Training Step Verification",
                "",
                "| Report | Claim | Proof | Model bytes | Sample bytes | Witness bytes | Update bytes | Replay bytes | Forward | Reverse |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
                *step_verification_rows,
                "",
            ]
        )
    native_inference_rows = render_native_inference_rows(reports)
    if native_inference_rows:
        lines.extend(
            [
                "## Native Inference Reports",
                "",
                "| Report | Kind | Source | Prediction | Label | Correct | Active pixels | Margin | Reverse | Explanation | Row evidence | Contribution count | Contribution ledger | Replay bytes | Inverse steps |",
                "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- | ---: | --- | ---: | ---: |",
                *native_inference_rows,
                "",
            ]
        )
    inference_verification_rows = render_inference_verification_rows(reports)
    if inference_verification_rows:
        lines.extend(
            [
                "## Inference Verification",
                "",
                "| Report | Prediction | Correct | Proof | Result | Reverse | Explanation | Contribution count | Contribution ledger | Replay bytes | Inverse steps | Sources checked |",
                "| --- | ---: | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |",
                *inference_verification_rows,
                "",
            ]
        )
    model_evaluation_rows = render_model_evaluation_rows(reports)
    if model_evaluation_rows:
        lines.extend(
            [
                "## Model Evaluation",
                "",
                "| Report | Samples | Correct | Incorrect | Accuracy | Lowest margin index | Lowest margin | Replay bytes | Inverse steps | Gate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                *model_evaluation_rows,
                "",
            ]
        )
    model_evaluation_scan_rows = render_model_evaluation_scan_rows(reports)
    if model_evaluation_scan_rows:
        lines.extend(
            [
                "## Model Evaluation Scan",
                "",
                "| Report | Samples | Correct | Incorrect | Accuracy | Lowest margin index | Lowest margin | Low-margin rows | Incorrect rows | Confusions |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *model_evaluation_scan_rows,
                "",
            ]
        )
    model_evaluation_verification_rows = render_model_evaluation_verification_rows(reports)
    if model_evaluation_verification_rows:
        lines.extend(
            [
                "## Model Evaluation Verification",
                "",
                "| Report | Samples | Accuracy | Rows | Proof | Reverse | Source model | Source samples | Elapsed s |",
                "| --- | ---: | ---: | --- | --- | --- | --- | --- | ---: |",
                *model_evaluation_verification_rows,
                "",
            ]
        )
    if not (
        capability_rows
        or capsule_rows
        or scorecard_rows
        or recompute_frontier_rows
        or scaling_projection_rows
        or run_rows
        or profile_rows
        or mlp_rows
        or coupling_rows
        or residual_rows
        or preprocess_rows
        or inference_trace_rows
        or q31_inference_rows
        or q31_evaluation_rows
        or inference_trace_profile_rows
        or audit_verification_rows
        or audit_scan_rows
        or audit_step_rows
        or step_verification_rows
        or native_inference_rows
        or inference_verification_rows
        or model_evaluation_rows
        or model_evaluation_scan_rows
        or model_evaluation_verification_rows
    ):
        lines.extend(["No supported MNIST ML profile reports were found.", ""])
    return "\n".join(lines)


def run_self_tests() -> int:
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_path = root / "run.json"
            comparison_path = root / "comparison.json"
            audit_contract_path = root / "audit-contract.json"
            mlp_path = root / "mlp.json"
            coupling_path = root / "coupling.json"
            residual_path = root / "triangular-residual.json"
            preprocess_path = root / "reversible-preprocess.json"
            inference_trace_path = root / "reversible-inference-trace.json"
            q31_inference_path = root / "q31-inference.json"
            q31_evaluation_path = root / "q31-evaluation.json"
            audit_verification_path = root / "audit-verification.json"
            audit_step_path = root / "audit-step.json"
            audit_scan_path = root / "audit-scan.json"
            step_verification_path = root / "step-verification.json"
            native_inference_path = root / "native-inference.json"
            inference_verification_path = root / "inference-verification.json"
            row_inference_verification_path = root / "row-inference-verification.json"
            model_evaluation_path = root / "model-evaluation.json"
            model_evaluation_scan_path = root / "model-evaluation-scan.json"
            model_evaluation_verification_path = root / "model-evaluation-verification.json"
            capsule_path = root / "model-capsule.json"
            run_path.write_text(
                json.dumps(profile_check.valid_run_report()),
                encoding="utf-8",
            )
            comparison_path.write_text(
                json.dumps(profile_check.valid_comparison_report()),
                encoding="utf-8",
            )
            audit_contract_path.write_text(
                json.dumps(profile_check.valid_audit_contract_report()),
                encoding="utf-8",
            )
            mlp_path.write_text(
                json.dumps(profile_check.valid_mlp_witness_report()),
                encoding="utf-8",
            )
            coupling_path.write_text(
                json.dumps(profile_check.valid_invertible_coupling_report()),
                encoding="utf-8",
            )
            residual_path.write_text(
                json.dumps(profile_check.valid_triangular_residual_report()),
                encoding="utf-8",
            )
            preprocess_path.write_text(
                json.dumps(profile_check.valid_reversible_preprocess_report()),
                encoding="utf-8",
            )
            inference_trace_path.write_text(
                json.dumps(profile_check.valid_reversible_inference_trace_report()),
                encoding="utf-8",
            )
            q31_inference_path.write_text(
                json.dumps(profile_check.valid_q31_reference_inference_report()),
                encoding="utf-8",
            )
            q31_evaluation_path.write_text(
                json.dumps(profile_check.valid_q31_reference_evaluation_report()),
                encoding="utf-8",
            )
            audit_verification_path.write_text(
                json.dumps(profile_check.valid_audit_verification_report()),
                encoding="utf-8",
            )
            audit_step_path.write_text(
                json.dumps(profile_check.valid_audit_step_report()),
                encoding="utf-8",
            )
            audit_scan_path.write_text(
                json.dumps(profile_check.valid_audit_scan_report()),
                encoding="utf-8",
            )
            step_verification_path.write_text(
                json.dumps(profile_check.valid_step_verification_report()),
                encoding="utf-8",
            )
            native_inference_path.write_text(
                json.dumps(profile_check.valid_native_inference_report()),
                encoding="utf-8",
            )
            inference_verification_path.write_text(
                json.dumps(profile_check.valid_inference_verification_report()),
                encoding="utf-8",
            )
            row_verification = profile_check.valid_inference_verification_report()
            row_verification["source_evaluation_checked"] = True
            row_verification["source_model_evaluation"] = {
                "path": "target/evaluation-bundle.json",
                "resolved_path": "target/evaluation-bundle.json",
                "checked": True,
                "payload_fingerprint": "0" * 64,
                "row_index": 1,
                "sample_fingerprint": "1" * 64,
            }
            row_inference_verification_path.write_text(
                json.dumps(row_verification),
                encoding="utf-8",
            )
            model_evaluation_path.write_text(
                json.dumps(profile_check.valid_model_evaluation_report()),
                encoding="utf-8",
            )
            model_evaluation_scan_path.write_text(
                json.dumps(profile_check.valid_model_evaluation_scan_report()),
                encoding="utf-8",
            )
            model_evaluation_verification_path.write_text(
                json.dumps(profile_check.valid_model_evaluation_verification_report()),
                encoding="utf-8",
            )
            capsule_path.write_text(
                json.dumps(profile_check.valid_model_capsule_report()),
                encoding="utf-8",
            )
            reports = [
                (
                    capsule_path,
                    load_checked_report(
                        capsule_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    run_path,
                    load_checked_report(
                        run_path,
                        require_reverse_check=True,
                        require_peak_rss=True,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    comparison_path,
                    load_checked_report(
                        comparison_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    audit_contract_path,
                    load_checked_report(
                        audit_contract_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=True,
                        require_audit_contract=True,
                    ),
                ),
                (
                    mlp_path,
                    load_checked_report(
                        mlp_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    coupling_path,
                    load_checked_report(
                        coupling_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    residual_path,
                    load_checked_report(
                        residual_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    preprocess_path,
                    load_checked_report(
                        preprocess_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    inference_trace_path,
                    load_checked_report(
                        inference_trace_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    q31_inference_path,
                    load_checked_report(
                        q31_inference_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    q31_evaluation_path,
                    load_checked_report(
                        q31_evaluation_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    audit_verification_path,
                    load_checked_report(
                        audit_verification_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    audit_step_path,
                    load_checked_report(
                        audit_step_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    audit_scan_path,
                    load_checked_report(
                        audit_scan_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    step_verification_path,
                    load_checked_report(
                        step_verification_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    native_inference_path,
                    load_checked_report(
                        native_inference_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    inference_verification_path,
                    load_checked_report(
                        inference_verification_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    row_inference_verification_path,
                    load_checked_report(
                        row_inference_verification_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    model_evaluation_path,
                    load_checked_report(
                        model_evaluation_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    model_evaluation_scan_path,
                    load_checked_report(
                        model_evaluation_scan_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
                (
                    model_evaluation_verification_path,
                    load_checked_report(
                        model_evaluation_verification_path,
                        require_reverse_check=False,
                        require_peak_rss=False,
                        require_ml_profile=False,
                        require_audit_contract=False,
                    ),
                ),
            ]
            markdown = render_markdown(reports)
            required = [
                "# Reverie MNIST ML Profile",
                "## North-Star Readiness",
                "## Model Capsules",
                "Capsule",
                "Witness proof",
                "Q31 parity",
                "## Inference Action Contract",
                "reverse_reversible_trace",
                "Debug training updates backward",
                "Run deterministic reversible inference traces",
                "## ML Capability Map",
                "reversible linear MNIST",
                "first-class witness tapes",
                "batched tensor dataset iteration",
                "speed, memory, trace, and reverse-cost scorecard",
                "## V6 Scorecard",
                "Max replay bytes",
                "Rev trace",
                "Q31 ref",
                "Audit contract",
                "Preprocess",
                "## Recompute Frontier",
                "Training trace replay",
                "Evaluation-row inference replay",
                "Reversible preprocessing replay",
                "Bytes/inverse",
                "## Scaling Projection",
                "Projected replay bytes",
                "100x",
                "## Run Reports",
                "Peak RSS",
                "Trace bytes",
                "## Artifact Profiles",
                "## Audit Contract",
                "reversible_inspectable_deterministic_q31_ml_kernel",
                "Witness/model",
                "## Artifact Rows",
                "## MLP Witness Reports",
                "deterministic_q31_mlp_witness_replay",
                "Witness proof",
                "Recomputed update bytes",
                "Update/witness",
                "124.804",
                "## Invertible Coupling Reports",
                "deterministic_q31_invertible_coupling_replay",
                "## Triangular Residual Reports",
                "deterministic_q31_triangular_residual_replay",
                "Forward x",
                "Forward left",
                "0 B",
                "384 B",
                "## Reversible Preprocess Reports",
                "deterministic_q31_reversible_preprocess_replay",
                "Forward features",
                "96 B",
                "## Reversible Inference Trace Reports",
                "deterministic_q31_reversible_inference_trace",
                "Forward logits",
                "344 B",
                "## Q31 Reference Inference",
                "Attribution",
                "Contribution ledger",
                "Top logits",
                "## Q31 Reference Evaluation",
                "Lowest margin",
                "top_low_margin",
                "top_incorrect",
                "## Inference Trace Profile",
                "Reference ledger",
                "Ledger match",
                "## Training Audit Verification",
                "Lineage ledger",
                "Transition ledger",
                "Final chain",
                "## Training Audit Scan",
                "Witness mismatches",
                "Suspicious rows",
                "## Training Audit Step",
                "Model delta",
                "Update ledger",
                "Scan evidence",
                "Top deltas",
                "## Training Step Verification",
                "deterministic_q31_training_step_replay",
                "Update bytes",
                "## Native Inference Reports",
                "Row evidence",
                "## Inference Verification",
                "Sources checked",
                "## Model Evaluation",
                "## Model Evaluation Scan",
                "Low-margin rows",
                "## Model Evaluation Verification",
                "Source samples",
            ]
            missing = [snippet for snippet in required if snippet not in markdown]
            if missing:
                raise AssertionError("missing Markdown snippet(s): " + ", ".join(missing))
            training_step_debug = render_training_step_debug_markdown(
                profile_check.valid_audit_step_report(),
                profile_check.valid_step_verification_report(),
            )
            for snippet in (
                "# Reverie Training Step Debug",
                "step_backward_from_model_update",
                "## Witnesses",
                "## Top Weight Deltas",
                "## Bias Deltas",
                "## Replay Proof",
                "deterministic_q31_training_step_replay",
                "222222222222",
                "333333333333",
                "444444444444",
            ):
                if snippet not in training_step_debug:
                    raise AssertionError(
                        f"training-step debug Markdown missing {snippet}"
                    )
    except (AssertionError, ValueError) as error:
        print(f"error: self-test failed: {error}", file=sys.stderr)
        return 1

    print("ok: MNIST ML profile Markdown renderer self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    if not args.paths:
        print("error: at least one report path is required", file=sys.stderr)
        return 2

    reports = []
    for path in args.paths:
        try:
            reports.append(
                (
                    path,
                    load_checked_report(
                        path,
                        require_reverse_check=args.require_reverse_check,
                        require_peak_rss=args.require_peak_rss,
                        require_ml_profile=args.require_ml_profile,
                        require_audit_contract=args.require_audit_contract,
                    ),
                )
            )
        except ValueError as error:
            print(f"error: {path}: {error}", file=sys.stderr)
            return 1
    try:
        profile_check.validate_report_set_consistency(reports)
    except ValueError as error:
        print(f"error: report-set consistency: {error}", file=sys.stderr)
        return 1

    markdown = render_markdown(reports)
    if args.output is None:
        print(markdown, end="")
    else:
        if args.output.parent != Path("."):
            args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
