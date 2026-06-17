#!/usr/bin/env python3
"""Verify a Reverie reversible ML model capsule handoff directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import check_mnist_ml_profile as profile_check
import summarize_mnist_ml_profile as profile_summary


DEFAULT_SUMMARY_NAME = "pipeline-summary.json"
DEFAULT_CAPSULE_NAME = "model-capsule.json"
DEFAULT_MANIFEST_NAME = "pipeline-manifest.json"
DEFAULT_PROFILE_NAME = "model-capsule-profile.md"
DEFAULT_VERIFICATION_REPORT_NAME = "model-capsule-verification.json"
DEFAULT_VERIFICATION_MARKDOWN_NAME = "model-capsule-verification.md"
DEFAULT_HANDOFF_NAME = "ml-audit-handoff.json"
DEFAULT_HANDOFF_MARKDOWN_NAME = "ml-audit-handoff.md"
DEFAULT_ACTION_REVIEW_RECEIPT_NAME = "inference-action-review-receipt.json"
DEFAULT_ACTION_REVIEW_RECEIPT_MARKDOWN_NAME = "inference-action-review-receipt.md"
VERIFICATION_KIND = "reverie_model_capsule_verification"
TRUST_CERTIFICATE_KIND = "reverie_model_capsule_trust_certificate"
HANDOFF_KIND = "reverie_mnist_ml_audit_handoff"
HANDOFF_SCHEMA = "reverie_mnist_ml_audit_handoff_v1"
ACTION_REVIEW_RECEIPT_KIND = "reverie_inference_action_review_receipt"
ACTION_REVIEW_RECEIPT_SCHEMA = "reverie_inference_action_review_receipt_v1"
ACTION_COMMAND_OUTPUT_TAIL_CHARS = 4000
DEFAULT_ACTION_COMMAND_TIMEOUT_SECONDS = 120.0
REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEW_CARD_ARTIFACTS = (
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
)
MACHINE_HANDOFF_ARTIFACTS = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a Reverie reversible ML model capsule handoff."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("target/mnist-ml-audit-pipeline"),
        help="Capsule directory, or model-capsule.json when --summary/--manifest are supplied.",
    )
    parser.add_argument("--summary", type=Path, help="Pipeline summary JSON path.")
    parser.add_argument("--capsule", type=Path, help="Model capsule JSON path.")
    parser.add_argument("--manifest", type=Path, help="Pipeline manifest JSON path.")
    parser.add_argument("--profile", type=Path, help="Capsule-first Markdown profile path.")
    parser.add_argument(
        "--skip-file-evidence",
        action="store_true",
        help="Do not recompute evidence file byte counts and SHA-256 hashes.",
    )
    parser.add_argument(
        "--allow-missing-profile",
        action="store_true",
        help="Allow the Markdown capsule profile to be absent.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable verification report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the machine-readable verification report to this path.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Write a human-readable trust-certificate Markdown summary.",
    )
    parser.add_argument(
        "--verification-markdown",
        type=Path,
        help="Existing trust-certificate Markdown path to validate.",
    )
    parser.add_argument(
        "--require-verification-markdown",
        action="store_true",
        help="Fail when the trust-certificate Markdown handoff is missing.",
    )
    parser.add_argument("--handoff", type=Path, help="ML audit handoff JSON path.")
    parser.add_argument("--handoff-markdown", type=Path, help="ML audit handoff Markdown path.")
    parser.add_argument(
        "--require-handoff",
        action="store_true",
        help="Fail when ml-audit-handoff.json or ml-audit-handoff.md is missing or stale.",
    )
    parser.add_argument(
        "--run-inference-action-commands",
        action="store_true",
        help="Execute the handoff's inference review commands and attest the results.",
    )
    parser.add_argument(
        "--action-command-receipt-output",
        type=Path,
        help="Write the inference action command replay receipt JSON to this path.",
    )
    parser.add_argument(
        "--action-command-receipt-markdown",
        type=Path,
        help="Write the inference action command replay receipt Markdown to this path.",
    )
    parser.add_argument(
        "--action-command-timeout",
        type=float,
        default=DEFAULT_ACTION_COMMAND_TIMEOUT_SECONDS,
        help="Per-command timeout in seconds for --run-inference-action-commands.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent capsule verifier self-tests and exit.",
    )
    return parser.parse_args()


def resolve_capsule_paths(args: argparse.Namespace) -> dict[str, Optional[Path]]:
    root = args.path
    verification_markdown = (
        getattr(args, "verification_markdown", None)
        or getattr(args, "markdown_output", None)
    )
    if args.summary or args.capsule or args.manifest or args.profile:
        capsule = args.capsule if args.capsule is not None else root
        base = capsule.parent
        return {
            "summary": args.summary if args.summary is not None else base / DEFAULT_SUMMARY_NAME,
            "capsule": capsule,
            "manifest": args.manifest if args.manifest is not None else base / DEFAULT_MANIFEST_NAME,
            "profile": args.profile if args.profile is not None else base / DEFAULT_PROFILE_NAME,
            "verification_markdown": verification_markdown
            if verification_markdown is not None
            else base / DEFAULT_VERIFICATION_MARKDOWN_NAME,
            "handoff": args.handoff if getattr(args, "handoff", None) is not None else base / DEFAULT_HANDOFF_NAME,
            "handoff_markdown": (
                args.handoff_markdown
                if getattr(args, "handoff_markdown", None) is not None
                else base / DEFAULT_HANDOFF_MARKDOWN_NAME
            ),
        }
    if root.is_dir():
        return {
            "summary": root / DEFAULT_SUMMARY_NAME,
            "capsule": root / DEFAULT_CAPSULE_NAME,
            "manifest": root / DEFAULT_MANIFEST_NAME,
            "profile": root / DEFAULT_PROFILE_NAME,
            "verification_markdown": verification_markdown
            if verification_markdown is not None
            else root / DEFAULT_VERIFICATION_MARKDOWN_NAME,
            "handoff": args.handoff if getattr(args, "handoff", None) is not None else root / DEFAULT_HANDOFF_NAME,
            "handoff_markdown": (
                args.handoff_markdown
                if getattr(args, "handoff_markdown", None) is not None
                else root / DEFAULT_HANDOFF_MARKDOWN_NAME
            ),
        }
    base = root.parent
    return {
        "summary": base / DEFAULT_SUMMARY_NAME,
        "capsule": root,
        "manifest": base / DEFAULT_MANIFEST_NAME,
        "profile": base / DEFAULT_PROFILE_NAME,
        "verification_markdown": verification_markdown
        if verification_markdown is not None
        else base / DEFAULT_VERIFICATION_MARKDOWN_NAME,
        "handoff": args.handoff if getattr(args, "handoff", None) is not None else base / DEFAULT_HANDOFF_NAME,
        "handoff_markdown": (
            args.handoff_markdown
            if getattr(args, "handoff_markdown", None) is not None
            else base / DEFAULT_HANDOFF_MARKDOWN_NAME
        ),
    }


def load_checked(path: Path, *, verify_file_evidence: bool) -> dict[str, Any]:
    report = profile_check.load_json(path)
    profile_check.validate_report(
        report,
        require_reverse_check=False,
        require_peak_rss=False,
        require_ml_profile=False,
        require_audit_contract=False,
    )
    if verify_file_evidence:
        profile_check.verify_pipeline_file_evidence(report)
    if not isinstance(report, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return report


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_hash(value: Optional[str]) -> str:
    return value[:12] if isinstance(value, str) else "n/a"


def bool_text(value: Any) -> str:
    return "pass" if value is True else "fail"


def bytes_text(value: Any) -> str:
    if not isinstance(value, int):
        return "n/a"
    units = ("B", "KiB", "MiB", "GiB")
    amount = float(value)
    unit = units[0]
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            break
        amount /= 1024
    if unit == "B":
        return f"{value} B"
    return f"{amount:.2f} {unit}"


def ratio_text(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"


def inference_action_contract_passed(inference: dict[str, Any]) -> bool:
    actions = inference.get("action_contract")
    if not isinstance(actions, list):
        return False
    return all(
        isinstance(action, dict) and action.get("supported") is True
        for action in actions
    )


def render_inference_action_contract_rows(actions: list[dict[str, Any]]) -> list[str]:
    rows = []
    for action in actions:
        operation = action["operation"]
        if operation == "reproduce_prediction":
            result = "prediction {}, correct {}, margin {}".format(
                action["result"]["prediction"],
                str(action["result"]["correct"]).lower(),
                f"{action['result']['margin']:,}",
            )
            cost = "Q31/native parity"
        elif operation == "explain_margin":
            result = "contribution `{}`, margin `{}`".format(
                short_hash(action["ledgers"]["contribution"]),
                short_hash(action["ledgers"]["margin_contribution"]),
            )
            cost = "ledger recompute"
        elif operation == "replay_imported_model_inference":
            result = "prediction {}, native match {}, replay {}".format(
                action["prediction"],
                bool_text(action["matches_native"]),
                bytes_text(action["payload_bytes"]),
            )
            cost = "external-import replay proof"
        elif operation == "replay_native_inference":
            result = "replay {}".format(bytes_text(action["payload_bytes"]))
            cost = "native replay proof"
        elif operation == "run_standalone_rev_classifier":
            result = "prediction {}, correct {}, source `{}`".format(
                action["prediction"],
                bool_text(action["correct"]),
                short_hash(action["source_sha256"]),
            )
            cost = "source-only Reverie run"
        elif operation == "reverse_reversible_trace":
            result = "witness {}, trace {}, replay {}".format(
                bytes_text(action["witness_payload_bytes"]),
                bytes_text(action["trace_payload_bytes"]),
                bytes_text(action["payload_bytes"]),
            )
            cost = "{} recompute steps".format(action["recompute_steps"])
        else:
            result = "checked"
            cost = "recorded"
        rows.append(
            "| `{}` | {} | `{}` | {} | {} |".format(
                operation,
                bool_text(action["supported"]),
                action["evidence"],
                result,
                cost,
            )
        )
    return rows


def handoff_comma(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else str(value)


def handoff_ratio(value: Any) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "n/a"


def handoff_shell_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


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
            short_hash(report["fingerprint"]),
            short_hash(capsule["fingerprint"]),
            short_hash(certificate["fingerprint"]),
            capsule["gates"]["passed_count"],
            capsule["gates"]["total"],
            short_hash(capsule["model_sha256"]),
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
    for artifact_id in REVIEW_CARD_ARTIFACTS:
        artifact = artifacts[artifact_id]
        lines.append(
            "| {} | {} | `{}` | {} | `{}` |".format(
                artifact["label"],
                artifact["role"],
                short_hash(artifact["sha256"]),
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
                short_hash(training["cause_ledger_fingerprint"]),
                short_hash(training["weight_delta_ledger_fingerprint"]),
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
                short_hash(inference["contribution_ledger_fingerprint"]),
                short_hash(inference["margin_contribution_ledger_fingerprint"]),
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
    for artifact_id in MACHINE_HANDOFF_ARTIFACTS:
        artifact = artifacts[artifact_id]
        lines.append(
            "| {} | {} | `{}` | {} | `{}` |".format(
                artifact["label"],
                artifact["role"],
                short_hash(artifact["sha256"]),
                handoff_comma(artifact["bytes"]),
                artifact["path"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def text_tail(value: str, limit: int = ACTION_COMMAND_OUTPUT_TAIL_CHARS) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def command_output_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def inference_action_review_semantic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "reverie_inference_action_review_semantics_v1",
        "handoff_fingerprint": payload["handoff_fingerprint"],
        "capsule_fingerprint": payload["capsule_fingerprint"],
        "trust_certificate_fingerprint": payload["trust_certificate_fingerprint"],
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


def inference_action_review_semantic_fingerprint(payload: dict[str, Any]) -> str:
    return profile_check.sha256_json(inference_action_review_semantic_payload(payload))


def run_inference_action_review_commands(
    handoff: dict[str, Any],
    *,
    capsule_fingerprint: str,
    trust_certificate_fingerprint: str,
    cwd: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("action command timeout must be positive")
    if handoff.get("kind") != HANDOFF_KIND:
        raise ValueError(f"handoff kind must be {HANDOFF_KIND}")
    payload = handoff.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("handoff payload must be an object")
    if payload.get("schema") != HANDOFF_SCHEMA:
        raise ValueError(f"handoff payload schema must be {HANDOFF_SCHEMA}")
    handoff_fingerprint = handoff.get("fingerprint")
    if not isinstance(handoff_fingerprint, str) or not handoff_fingerprint:
        raise ValueError("handoff fingerprint must be a non-empty string")
    if handoff_fingerprint != profile_check.sha256_json(payload):
        raise ValueError("handoff fingerprint does not match payload")
    commands = payload.get("inference_action_review_commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError("handoff inference_action_review_commands must be a non-empty list")

    rows: list[dict[str, Any]] = []
    for index, command in enumerate(commands):
        context = f"handoff inference_action_review_commands[{index}]"
        if not isinstance(command, dict):
            raise ValueError(f"{context} must be an object")
        argv = command.get("command")
        if not isinstance(argv, list) or not argv or not all(isinstance(part, str) and part for part in argv):
            raise ValueError(f"{context}.command must be a non-empty string array")
        operation = command.get("operation")
        if not isinstance(operation, str) or not operation:
            raise ValueError(f"{context}.operation must be a non-empty string")
        label = command.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{context}.label must be a non-empty string")
        purpose = command.get("purpose")
        if not isinstance(purpose, str) or not purpose:
            raise ValueError(f"{context}.purpose must be a non-empty string")
        artifacts = command.get("artifacts")
        if not isinstance(artifacts, list) or not all(isinstance(item, str) and item for item in artifacts):
            raise ValueError(f"{context}.artifacts must be a string array")

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
            stdout = command_output_text(completed.stdout)
            stderr = command_output_text(completed.stderr)
        except subprocess.TimeoutExpired as timeout_error:
            timed_out = True
            stdout = command_output_text(timeout_error.stdout)
            stderr = command_output_text(timeout_error.stderr)
            error = f"timed out after {timeout_seconds:g}s"
        except OSError as os_error:
            error = str(os_error)
            stderr = str(os_error)
        elapsed_seconds = time.monotonic() - started
        passed = exit_code == 0 and not timed_out and error is None
        rows.append(
            {
                "operation": operation,
                "label": label,
                "purpose": purpose,
                "command": argv,
                "command_text": handoff_shell_command(argv),
                "artifacts": artifacts,
                "passed": passed,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "error": error,
                "elapsed_seconds": elapsed_seconds,
                "stdout_bytes": len(stdout.encode("utf-8")),
                "stderr_bytes": len(stderr.encode("utf-8")),
                "stdout_sha256": sha256_text(stdout),
                "stderr_sha256": sha256_text(stderr),
                "stdout_tail": text_tail(stdout),
                "stderr_tail": text_tail(stderr),
            }
        )

    failed_operations = [row["operation"] for row in rows if row["passed"] is not True]
    timed_out_operations = [row["operation"] for row in rows if row["timed_out"] is True]
    summary = {
        "passed": not failed_operations,
        "command_count": len(rows),
        "passed_count": len(rows) - len(failed_operations),
        "failed_count": len(failed_operations),
        "failed_operations": failed_operations,
        "timed_out_operations": timed_out_operations,
    }
    receipt_payload = {
        "schema": ACTION_REVIEW_RECEIPT_SCHEMA,
        "handoff_fingerprint": handoff_fingerprint,
        "capsule_fingerprint": capsule_fingerprint,
        "trust_certificate_fingerprint": trust_certificate_fingerprint,
        "command_cwd": str(cwd),
        "timeout_seconds": timeout_seconds,
        "summary": summary,
        "operations": rows,
    }
    receipt_payload["semantic_fingerprint"] = (
        inference_action_review_semantic_fingerprint(receipt_payload)
    )
    return {
        "kind": ACTION_REVIEW_RECEIPT_KIND,
        "passed": summary["passed"],
        "algorithm": "sha256",
        "fingerprint": profile_check.sha256_json(receipt_payload),
        "payload": receipt_payload,
    }


def render_inference_action_review_receipt_markdown(receipt: dict[str, Any]) -> str:
    payload = receipt["payload"]
    summary = payload["summary"]
    lines = [
        "# Reverie Inference Action Review Receipt",
        "",
        "| Verdict | Receipt | Semantic | Handoff | Capsule | Certificate | Commands |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
        "| {} | `{}` | `{}` | `{}` | `{}` | `{}` | {}/{} |".format(
            bool_text(receipt["passed"]),
            short_hash(receipt["fingerprint"]),
            short_hash(payload["semantic_fingerprint"]),
            short_hash(payload["handoff_fingerprint"]),
            short_hash(payload["capsule_fingerprint"]),
            short_hash(payload["trust_certificate_fingerprint"]),
            summary["passed_count"],
            summary["command_count"],
        ),
        "",
        "## Commands",
        "",
        "| Operation | Status | Exit | Elapsed | Stdout | Stderr | Artifacts |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in payload["operations"]:
        stdout = "{} / `{}`".format(bytes_text(row["stdout_bytes"]), short_hash(row["stdout_sha256"]))
        stderr = "{} / `{}`".format(bytes_text(row["stderr_bytes"]), short_hash(row["stderr_sha256"]))
        status = "timeout" if row["timed_out"] else bool_text(row["passed"])
        lines.append(
            "| `{}` | {} | {} | {:.3f}s | {} | {} | {} |".format(
                row["operation"],
                status,
                "n/a" if row["exit_code"] is None else row["exit_code"],
                row["elapsed_seconds"],
                stdout,
                stderr,
                ", ".join(f"`{markdown_cell(artifact)}`" for artifact in row["artifacts"]),
            )
        )
    lines.extend(["", "## Command Lines", ""])
    for row in payload["operations"]:
        lines.extend(
            [
                f"### `{row['operation']}`",
                "",
                f"`{markdown_cell(row['command_text'])}`",
                "",
            ]
        )
    return "\n".join(lines)


def required_profile_snippets(
    capsule: dict[str, Any],
    summary: dict[str, Any],
) -> list[str]:
    payload = capsule["payload"]
    gates = payload["gates"]
    readiness = payload["readiness"]["summary"]
    return [
        "## Model Capsules",
        capsule["fingerprint"][:12],
        f"{gates['passed_count']}/{gates['total']}",
        payload["model"]["sha256"][:12],
        payload["training_lineage"]["lineage_ledger_fingerprint"][:12],
        payload["training_lineage"]["final_chain"][:12],
        payload["witnesses"]["mlp_witness_proof_fingerprint"][:12],
        f"{readiness['passed']}/{readiness['total']}",
        "## North-Star Readiness",
        "## Inference Action Contract",
        "replay_imported_model_inference",
        "run_standalone_rev_classifier",
        "reverse_reversible_trace",
        "## ML Capability Map",
        "## V6 Scorecard",
        "## Recompute Frontier",
        "## Scaling Projection",
        "Q31 parity",
        "Native replay",
        summary["metrics"]["ml_goal_readiness"]["goals"][0]["goal"],
        summary["metrics"]["ml_capability_map"]["capabilities"][0]["goal"],
    ]


def validate_capsule_profile(
    profile_path: Optional[Path],
    *,
    capsule: dict[str, Any],
    summary: dict[str, Any],
    expected_markdown: Optional[str],
    allow_missing: bool,
) -> dict[str, Any]:
    if profile_path is None:
        if allow_missing:
            return {"present": False, "path": None, "bytes": None, "sha256": None, "matches_rendered": None}
        raise ValueError("capsule profile path is required")
    if not profile_path.exists():
        if allow_missing:
            return {
                "present": False,
                "path": str(profile_path),
                "bytes": None,
                "sha256": None,
                "matches_rendered": None,
            }
        raise ValueError(f"capsule profile is missing: {profile_path}")
    try:
        markdown = profile_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"failed to read capsule profile {profile_path}: {error}") from error
    missing = [
        snippet
        for snippet in required_profile_snippets(capsule, summary)
        if snippet not in markdown
    ]
    if missing:
        raise ValueError(
            "capsule profile is stale or incomplete; missing snippet(s): "
            + ", ".join(missing)
        )
    if expected_markdown is not None and markdown != expected_markdown:
        raise ValueError("capsule profile is stale; rendered Markdown does not match the accepted artifacts")
    return {
        "present": True,
        "path": str(profile_path),
        "bytes": len(markdown.encode("utf-8")),
        "sha256": sha256_text(markdown),
        "matches_rendered": expected_markdown is None or markdown == expected_markdown,
    }


def validate_verification_markdown(
    markdown_path: Optional[Path],
    *,
    expected_markdown: str,
    allow_missing: bool,
    checked: bool,
) -> dict[str, Any]:
    if not checked:
        return {
            "checked": False,
            "present": None,
            "path": None if markdown_path is None else str(markdown_path),
            "bytes": None,
            "sha256": None,
            "matches_rendered": None,
        }
    if markdown_path is None:
        if allow_missing:
            return {
                "checked": True,
                "present": False,
                "path": None,
                "bytes": None,
                "sha256": None,
                "matches_rendered": None,
            }
        raise ValueError("verification Markdown path is required")
    if not markdown_path.exists():
        if allow_missing:
            return {
                "checked": True,
                "present": False,
                "path": str(markdown_path),
                "bytes": None,
                "sha256": None,
                "matches_rendered": None,
            }
        raise ValueError(f"verification Markdown is missing: {markdown_path}")
    try:
        markdown = markdown_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"failed to read verification Markdown {markdown_path}: {error}") from error
    if markdown != expected_markdown:
        raise ValueError("verification Markdown is stale; rendered Markdown does not match the accepted artifacts")
    return {
        "checked": True,
        "present": True,
        "path": str(markdown_path),
        "bytes": len(markdown.encode("utf-8")),
        "sha256": sha256_text(markdown),
        "matches_rendered": True,
    }


def resolve_artifact_path(path_text: Any, *, handoff_path: Path) -> Path:
    if not isinstance(path_text, str) or not path_text:
        raise ValueError("handoff artifact path must be a non-empty string")
    path = Path(path_text)
    if path.is_absolute() or path.exists():
        return path
    sibling = handoff_path.parent / path.name
    if sibling.exists():
        return sibling
    return path


def validate_handoff_artifact(
    artifact_id: str,
    record: Any,
    *,
    handoff_path: Path,
    verify_file_evidence: bool,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"handoff artifact {artifact_id} must be an object")
    path = resolve_artifact_path(record.get("path"), handoff_path=handoff_path)
    expected_bytes = record.get("bytes")
    expected_sha = record.get("sha256")
    if not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise ValueError(f"handoff artifact {artifact_id}.bytes must be non-negative")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(f"handoff artifact {artifact_id}.sha256 must be a SHA-256 digest")
    checked = False
    actual_bytes = None
    actual_sha = None
    if verify_file_evidence:
        if not path.exists():
            raise ValueError(f"handoff artifact {artifact_id} is missing: {path}")
        actual_bytes = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"handoff artifact {artifact_id} byte count mismatch: "
                f"expected {expected_bytes}, found {actual_bytes}"
            )
        if actual_sha != expected_sha:
            raise ValueError(f"handoff artifact {artifact_id} SHA-256 mismatch")
        checked = True
    return {
        "path": str(path),
        "bytes": expected_bytes,
        "sha256": expected_sha,
        "checked": checked,
        "actual_bytes": actual_bytes,
        "actual_sha256": actual_sha,
    }


def validate_handoff_markdown(
    markdown_path: Optional[Path],
    *,
    expected_markdown: str,
    require_handoff: bool,
) -> dict[str, Any]:
    if markdown_path is None:
        if require_handoff:
            raise ValueError("handoff Markdown path is required")
        return {
            "present": False,
            "path": None,
            "bytes": None,
            "sha256": None,
            "matches_rendered": None,
        }
    if not markdown_path.exists():
        if require_handoff:
            raise ValueError(f"handoff Markdown is missing: {markdown_path}")
        return {
            "present": False,
            "path": str(markdown_path),
            "bytes": None,
            "sha256": None,
            "matches_rendered": None,
        }
    markdown = markdown_path.read_text(encoding="utf-8")
    if markdown != expected_markdown:
        raise ValueError("handoff Markdown is stale; rendered Markdown does not match the handoff JSON")
    return {
        "present": True,
        "path": str(markdown_path),
        "bytes": len(markdown.encode("utf-8")),
        "sha256": sha256_text(markdown),
        "matches_rendered": True,
    }


def validate_inference_action_review_commands(
    value: Any,
    *,
    capsule: dict[str, Any],
    artifact_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(value, list):
        raise ValueError("handoff inference_action_review_commands must be a list")
    expected_operations = [
        action["operation"]
        for action in capsule["payload"]["inference"]["action_contract"]
    ]
    if len(value) != len(expected_operations):
        raise ValueError("handoff inference_action_review_commands must cover every inference action")
    seen: list[str] = []
    for index, command in enumerate(value):
        context = f"handoff inference_action_review_commands[{index}]"
        if not isinstance(command, dict):
            raise ValueError(f"{context} must be an object")
        operation = command.get("operation")
        if not isinstance(operation, str) or not operation:
            raise ValueError(f"{context}.operation must be a non-empty string")
        seen.append(operation)
        label = command.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{context}.label must be a non-empty string")
        purpose = command.get("purpose")
        if not isinstance(purpose, str) or not purpose:
            raise ValueError(f"{context}.purpose must be a non-empty string")
        argv = command.get("command")
        if not isinstance(argv, list) or not argv or not all(isinstance(part, str) and part for part in argv):
            raise ValueError(f"{context}.command must be a non-empty string array")
        artifacts = command.get("artifacts")
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or not all(isinstance(artifact, str) and artifact for artifact in artifacts)
        ):
            raise ValueError(f"{context}.artifacts must be a non-empty string array")
        missing_artifacts = sorted(set(artifacts) - artifact_ids)
        if missing_artifacts:
            raise ValueError(
                f"{context}.artifacts references missing handoff artifact(s): "
                + ", ".join(missing_artifacts)
            )
        if operation == "reproduce_prediction":
            required = {"model_bundle", "samples", "q31_reference_inference"}
        elif operation == "explain_margin":
            required = {"q31_reference_inference_markdown", "native_inference_verification_markdown"}
        elif operation == "replay_imported_model_inference":
            required = {
                "imported_model_source",
                "imported_model_bundle",
                "imported_model_inference_bundle",
            }
        elif operation == "replay_native_inference":
            required = {"native_inference_bundle", "native_inference_verification_markdown"}
        elif operation == "run_standalone_rev_classifier":
            required = {"native_standalone_rev_classifier", "native_standalone_rev_run"}
        elif operation == "reverse_reversible_trace":
            required = {
                "inference_trace_forward",
                "inference_trace_reverse",
                "reversible_inference_trace",
                "reversible_inference_trace_markdown",
            }
        else:
            raise ValueError(f"{context}.operation is not in the capsule action contract")
        missing_required = sorted(required - set(artifacts))
        if missing_required:
            raise ValueError(
                f"{context}.artifacts missing required artifact(s): "
                + ", ".join(missing_required)
            )
    if seen != expected_operations:
        raise ValueError("handoff inference_action_review_commands must match capsule action order")
    return {"count": len(value), "operations": seen}


def validate_handoff(
    handoff_path: Optional[Path],
    handoff_markdown_path: Optional[Path],
    *,
    capsule: dict[str, Any],
    certificate: dict[str, Any],
    verify_file_evidence: bool,
    require_handoff: bool,
) -> dict[str, Any]:
    if handoff_path is None:
        if require_handoff:
            raise ValueError("handoff path is required")
        return {"present": False, "path": None, "artifacts": {}, "markdown": None}
    if not handoff_path.exists():
        if require_handoff:
            raise ValueError(f"handoff JSON is missing: {handoff_path}")
        return {"present": False, "path": str(handoff_path), "artifacts": {}, "markdown": None}
    handoff = profile_check.load_json(handoff_path)
    if handoff.get("kind") != HANDOFF_KIND:
        raise ValueError(f"handoff kind must be {HANDOFF_KIND}")
    if handoff.get("algorithm") != "sha256":
        raise ValueError("handoff algorithm must be sha256")
    payload = handoff.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("handoff payload must be an object")
    if payload.get("schema") != HANDOFF_SCHEMA:
        raise ValueError(f"handoff payload schema must be {HANDOFF_SCHEMA}")
    fingerprint = handoff.get("fingerprint")
    expected_fingerprint = profile_check.sha256_json(payload)
    if fingerprint != expected_fingerprint:
        raise ValueError("handoff fingerprint does not match payload")
    if payload.get("capsule", {}).get("fingerprint") != capsule["fingerprint"]:
        raise ValueError("handoff capsule fingerprint does not match model capsule")
    if payload.get("trust_certificate", {}).get("fingerprint") != certificate["fingerprint"]:
        raise ValueError("handoff trust-certificate fingerprint does not match verifier")
    if payload.get("trust_certificate", {}).get("passed") != certificate["payload"]["passed"]:
        raise ValueError("handoff trust-certificate verdict does not match verifier")
    selected_inference = payload.get("selected_inference")
    if not isinstance(selected_inference, dict):
        raise ValueError("handoff selected_inference must be an object")
    capsule_trace = capsule["payload"]["reversible_inference_trace"]
    capsule_import = capsule["payload"]["imported_model"]
    certificate_import = certificate["payload"]["deterministic_inference"][
        "imported_model"
    ]
    certificate_trace = certificate["payload"]["deterministic_inference"]["reversible_trace"]
    for field in (
        "source_model_json_checked",
        "provenance_kind",
        "source_model_json_fingerprint",
        "inference_replay_passed",
        "prediction_matches_native",
        "margin_matches_native",
    ):
        if capsule_import[field] != certificate_import[field]:
            raise ValueError(
                f"trust-certificate imported model field `{field}` does not match model capsule"
            )
    expected_imported_fields = {
        "imported_model_prediction": certificate_import["prediction"],
        "imported_model_replay_passed": certificate_import["inference_replay_passed"],
        "imported_model_matches_native": certificate_import["matches_native"],
    }
    for field, expected_value in expected_imported_fields.items():
        if selected_inference.get(field) != expected_value:
            raise ValueError(
                f"handoff selected_inference.{field} must match trust-certificate imported model"
            )
    for field in (
        "prediction",
        "correct",
        "runner_up_class",
        "margin",
        "label_rank",
        "top2_correct",
        "top_classes",
        "top_logit_values",
        "witness_payload_bytes",
        "replay_payload_bytes",
        "total_recompute_steps",
    ):
        if capsule_trace[field] != certificate_trace[field]:
            raise ValueError(
                f"trust-certificate reversible trace field `{field}` does not match model capsule"
            )
    expected_trace_fields = {
        "trace_prediction": capsule_trace["prediction"],
        "trace_correct": capsule_trace["correct"],
        "trace_runner_up_class": capsule_trace["runner_up_class"],
        "trace_margin": capsule_trace["margin"],
        "trace_label_rank": capsule_trace["label_rank"],
        "trace_top2_correct": capsule_trace["top2_correct"],
        "trace_top_classes": capsule_trace["top_classes"],
        "trace_top_logit_values": capsule_trace["top_logit_values"],
        "trace_witness_payload_bytes": capsule_trace["witness_payload_bytes"],
        "trace_replay_payload_bytes": capsule_trace["replay_payload_bytes"],
        "trace_total_recompute_steps": capsule_trace["total_recompute_steps"],
    }
    for field, expected_value in expected_trace_fields.items():
        if selected_inference.get(field) != expected_value:
            raise ValueError(
                f"handoff selected_inference.{field} must match model capsule reversible trace"
            )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("handoff artifacts must be an object")
    required = set(REVIEW_CARD_ARTIFACTS) | set(MACHINE_HANDOFF_ARTIFACTS)
    missing = sorted(required - set(artifacts))
    if missing:
        raise ValueError("handoff is missing artifact(s): " + ", ".join(missing))
    action_commands = validate_inference_action_review_commands(
        payload.get("inference_action_review_commands"),
        capsule=capsule,
        artifact_ids=set(artifacts),
    )
    checked_artifacts = {
        artifact_id: validate_handoff_artifact(
            artifact_id,
            record,
            handoff_path=handoff_path,
            verify_file_evidence=verify_file_evidence,
        )
        for artifact_id, record in artifacts.items()
    }
    expected_markdown = render_handoff_markdown(handoff)
    markdown = validate_handoff_markdown(
        handoff_markdown_path,
        expected_markdown=expected_markdown,
        require_handoff=require_handoff,
    )
    return {
        "present": True,
        "path": str(handoff_path),
        "fingerprint": fingerprint,
        "artifact_count": len(artifacts),
        "artifacts_checked": sum(1 for item in checked_artifacts.values() if item["checked"]),
        "review_cards": list(REVIEW_CARD_ARTIFACTS),
        "machine_artifacts": list(MACHINE_HANDOFF_ARTIFACTS),
        "inference_action_review_commands": action_commands,
        "artifacts": checked_artifacts,
        "markdown": markdown,
    }


def verify_capsule(
    paths: dict[str, Optional[Path]],
    *,
    verify_file_evidence: bool,
    allow_missing_profile: bool,
) -> dict[str, Any]:
    summary_path = paths["summary"]
    capsule_path = paths["capsule"]
    manifest_path = paths["manifest"]
    if summary_path is None or capsule_path is None or manifest_path is None:
        raise ValueError("summary, capsule, and manifest paths are required")
    summary = load_checked(summary_path, verify_file_evidence=verify_file_evidence)
    capsule = load_checked(capsule_path, verify_file_evidence=verify_file_evidence)
    manifest = load_checked(manifest_path, verify_file_evidence=verify_file_evidence)
    reports = [
        (summary_path, summary),
        (capsule_path, capsule),
        (manifest_path, manifest),
    ]
    profile_check.validate_report_set_consistency(reports)
    expected_markdown = profile_summary.render_markdown(reports)
    profile_metadata = validate_capsule_profile(
        paths["profile"],
        capsule=capsule,
        summary=summary,
        expected_markdown=expected_markdown,
        allow_missing=allow_missing_profile,
    )
    payload = capsule["payload"]
    return {
        "capsule": capsule,
        "summary": summary,
        "manifest": manifest,
        "fingerprint": capsule["fingerprint"],
        "gate_total": payload["gates"]["total"],
        "gate_passed": payload["gates"]["passed_count"],
        "model_sha256": payload["model"]["sha256"],
        "witness_proof": payload["witnesses"]["mlp_witness_proof_fingerprint"],
        "profile": None if paths["profile"] is None else str(paths["profile"]),
        "profile_metadata": profile_metadata,
        "paths": {key: None if value is None else str(value) for key, value in paths.items()},
    }


def trust_certificate(
    result: dict[str, Any],
    *,
    verify_file_evidence: bool,
    allow_missing_profile: bool,
) -> dict[str, Any]:
    capsule = result["capsule"]
    payload = capsule["payload"]
    gates = payload["gates"]
    lineage = payload["training_lineage"]
    inference = payload["inference"]
    imported_model = payload["imported_model"]
    reversible_trace = payload["reversible_inference_trace"]
    witnesses = payload["witnesses"]
    scorecard = payload["scorecard"]
    readiness = payload["readiness"]
    readiness_summary = readiness["summary"]
    profile = result["profile_metadata"]
    profile_ok = profile["matches_rendered"] is True or (
        allow_missing_profile and profile["present"] is False
    )
    passed = (
        gates["passed"] is True
        and gates["passed_count"] == gates["total"]
        and readiness["passed"] is True
        and readiness_summary["passed"] == readiness_summary["total"]
        and lineage["restored_initial_model"] is True
        and lineage["final_model_replayed"] is True
        and lineage["lineage_ledger_matches"] is True
        and imported_model["source_model_json_checked"] is True
        and imported_model["inference_replay_passed"] is True
        and imported_model["prediction_matches_native"] is True
        and imported_model["margin_matches_native"] is True
        and inference["imported_model_replay_passed"] is True
        and inference["imported_model_matches_native"] is True
        and inference["native_replay_passed"] is True
        and inference["evaluation_row_replay_passed"] is True
        and inference["q31_reference_matches_native"] is True
        and inference_action_contract_passed(inference)
        and reversible_trace["passed"] is True
        and witnesses["mlp_passed"] is True
        and scorecard["balanced_recompute"] is True
        and profile_ok
    )
    payload_certificate = {
        "kind": TRUST_CERTIFICATE_KIND,
        "passed": passed,
        "north_star": readiness["north_star"],
        "non_goal": readiness["non_goal"],
        "roadmap_readiness": {
            "passed": readiness_summary["passed"],
            "total": readiness_summary["total"],
            "failed": readiness_summary["failed"],
        },
        "integrity": {
            "capsule_fingerprint": result["fingerprint"],
            "model_sha256": payload["model"]["sha256"],
            "samples_sha256": payload["samples"]["sha256"],
            "profile_sha256": profile["sha256"],
            "file_evidence_verified": verify_file_evidence,
            "profile_required": not allow_missing_profile,
            "profile_ok": profile_ok,
            "profile_matches_rendered": profile["matches_rendered"],
            "gates": {
                "passed": gates["passed_count"],
                "total": gates["total"],
            },
        },
        "reversibility": {
            "training_lineage_replayed": lineage["final_model_replayed"],
            "restored_initial_model": lineage["restored_initial_model"],
            "lineage_ledger_matches": lineage["lineage_ledger_matches"],
            "checked_steps": lineage["checked_steps"],
            "lineage_ledger_fingerprint": lineage["lineage_ledger_fingerprint"],
            "final_chain": lineage["final_chain"],
            "balanced_recompute": scorecard["balanced_recompute"],
            "total_recompute_steps": scorecard["total_recompute_steps"],
            "reverse_to_train_elapsed_ratio": scorecard["reverse_to_train_elapsed_ratio"],
        },
        "deterministic_inference": {
            "imported_model": {
                "source_model_json_checked": imported_model[
                    "source_model_json_checked"
                ],
                "provenance_kind": imported_model["provenance_kind"],
                "source_model_json_fingerprint": imported_model[
                    "source_model_json_fingerprint"
                ],
                "inference_replay_passed": imported_model["inference_replay_passed"],
                "prediction_matches_native": imported_model[
                    "prediction_matches_native"
                ],
                "margin_matches_native": imported_model["margin_matches_native"],
                "prediction": inference["imported_model_prediction"],
                "matches_native": inference["imported_model_matches_native"],
            },
            "native_replay_passed": inference["native_replay_passed"],
            "evaluation_row_replay_passed": inference["evaluation_row_replay_passed"],
            "q31_reference_matches_native": inference["q31_reference_matches_native"],
            "native_prediction": inference["native_prediction"],
            "evaluation_row_prediction": inference["evaluation_row_prediction"],
            "q31_reference_prediction": inference["q31_reference_prediction"],
            "action_contract": inference["action_contract"],
            "reversible_trace": {
                "passed": reversible_trace["passed"],
                "prediction": reversible_trace["prediction"],
                "correct": reversible_trace["correct"],
                "top2_correct": reversible_trace["top2_correct"],
                "runner_up_class": reversible_trace["runner_up_class"],
                "margin": reversible_trace["margin"],
                "label_rank": reversible_trace["label_rank"],
                "top_classes": reversible_trace["top_classes"],
                "top_logit_values": reversible_trace["top_logit_values"],
                "top_logits": reversible_trace["top_logits"],
                "contribution_ledger_fingerprint": reversible_trace[
                    "contribution_ledger_fingerprint"
                ],
                "margin_contribution_ledger_fingerprint": reversible_trace[
                    "margin_contribution_ledger_fingerprint"
                ],
                "witness_payload_bytes": reversible_trace["witness_payload_bytes"],
                "trace_payload_bytes": reversible_trace["trace_payload_bytes"],
                "replay_payload_bytes": reversible_trace["replay_payload_bytes"],
                "total_recompute_steps": reversible_trace["total_recompute_steps"],
            },
        },
        "witnessing": {
            "mlp_passed": witnesses["mlp_passed"],
            "mlp_samples": witnesses["mlp_samples"],
            "mlp_dataset_loops": witnesses["mlp_dataset_loops"],
            "mlp_witness_proof_fingerprint": witnesses["mlp_witness_proof_fingerprint"],
            "witness_payload_bytes": witnesses["witness_payload_bytes"],
            "witness_to_model_payload_ratio": witnesses["witness_to_model_payload_ratio"],
        },
        "cost": {
            "run_peak_rss_bytes": scorecard["run_peak_rss_bytes"],
            "max_replay_payload_bytes": scorecard["max_replay_payload_bytes"],
            "trace_to_model_payload_ratio": scorecard["trace_to_model_payload_ratio"],
            "witness_to_model_payload_ratio": scorecard["witness_to_model_payload_ratio"],
        },
    }
    return {
        "schema": "reverie_model_capsule_trust_certificate_v1",
        "algorithm": "sha256",
        "fingerprint": profile_check.sha256_json(payload_certificate),
        "payload": payload_certificate,
    }


def verification_report(
    result: dict[str, Any],
    *,
    verify_file_evidence: bool,
    allow_missing_profile: bool,
    verification_markdown_path: Optional[Path] = None,
    require_verification_markdown: bool = False,
    check_verification_markdown: bool = True,
    handoff_path: Optional[Path] = None,
    handoff_markdown_path: Optional[Path] = None,
    require_handoff: bool = False,
) -> dict[str, Any]:
    capsule = result["capsule"]
    payload = capsule["payload"]
    scorecard = payload["scorecard"]
    readiness = payload["readiness"]["summary"]
    certificate = trust_certificate(
        result,
        verify_file_evidence=verify_file_evidence,
        allow_missing_profile=allow_missing_profile,
    )
    certificate_passed = certificate["payload"]["passed"]
    report = {
        "kind": VERIFICATION_KIND,
        "passed": certificate_passed,
        "algorithm": "sha256",
        "checks": {
            "json_schema": True,
            "report_set_consistency": True,
            "file_evidence_verified": verify_file_evidence,
            "profile_required": not allow_missing_profile,
            "profile_present": result["profile_metadata"]["present"],
            "profile_matches_rendered": result["profile_metadata"]["matches_rendered"],
            "verification_markdown_checked": check_verification_markdown,
            "trust_certificate_passed": certificate_passed,
            "handoff_checked": require_handoff or (
                handoff_path is not None and handoff_path.exists()
            ),
        },
        "paths": result["paths"],
        "capsule": {
            "fingerprint": result["fingerprint"],
            "schema": payload["schema"],
            "gates": {
                "passed": result["gate_passed"],
                "total": result["gate_total"],
            },
            "model_sha256": result["model_sha256"],
            "samples": payload["samples"]["count"],
            "accuracy_percent": payload["samples"]["accuracy_percent"],
            "lineage_ledger_fingerprint": payload["training_lineage"]["lineage_ledger_fingerprint"],
            "final_chain": payload["training_lineage"]["final_chain"],
            "witness_proof": result["witness_proof"],
            "readiness": {
                "passed": readiness["passed"],
                "total": readiness["total"],
            },
        },
        "scorecard": {
            "peak_rss_bytes": scorecard["run_peak_rss_bytes"],
            "max_replay_payload_bytes": scorecard["max_replay_payload_bytes"],
            "reverse_to_train_elapsed_ratio": scorecard["reverse_to_train_elapsed_ratio"],
            "trace_to_model_payload_ratio": scorecard["trace_to_model_payload_ratio"],
            "witness_to_model_payload_ratio": scorecard["witness_to_model_payload_ratio"],
            "total_recompute_steps": scorecard["total_recompute_steps"],
        },
        "trust_certificate": certificate,
        "profile": result["profile_metadata"],
    }
    markdown = render_verification_markdown(report)
    saved_markdown = validate_verification_markdown(
        verification_markdown_path,
        expected_markdown=markdown,
        allow_missing=not require_verification_markdown,
        checked=check_verification_markdown,
    )
    report["verification_markdown"] = {
        "renderer": "render_verification_markdown",
        "bytes": len(markdown.encode("utf-8")),
        "sha256": sha256_text(markdown),
        "saved": saved_markdown,
    }
    report["checks"]["verification_markdown_present"] = saved_markdown["present"]
    report["checks"]["verification_markdown_matches_rendered"] = saved_markdown["matches_rendered"]
    handoff = validate_handoff(
        handoff_path,
        handoff_markdown_path,
        capsule=capsule,
        certificate=certificate,
        verify_file_evidence=verify_file_evidence,
        require_handoff=require_handoff,
    )
    report["handoff"] = handoff
    report["checks"]["handoff_present"] = handoff["present"]
    report["checks"]["handoff_artifacts_checked"] = handoff.get("artifacts_checked")
    report["checks"]["handoff_markdown_matches_rendered"] = (
        None if handoff.get("markdown") is None else handoff["markdown"]["matches_rendered"]
    )
    return report


def render_verification_markdown(report: dict[str, Any]) -> str:
    certificate = report["trust_certificate"]
    payload = certificate["payload"]
    integrity = payload["integrity"]
    reversibility = payload["reversibility"]
    inference = payload["deterministic_inference"]
    imported_model = inference["imported_model"]
    reversible_trace = inference["reversible_trace"]
    witnessing = payload["witnessing"]
    cost = payload["cost"]
    dataset_loops = ", ".join(
        "{} by {}".format(loop["index"], ",".join(loop["size_sources"]))
        for loop in witnessing["mlp_dataset_loops"]
    )
    lines = [
        "# Reverie Model Capsule Verification",
        "",
        "| Verdict | Capsule | Certificate | Gates | Readiness | Model | Witness proof |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
        "| {} | `{}` | `{}` | {}/{} | {}/{} | `{}` | `{}` |".format(
            bool_text(report["passed"]),
            short_hash(report["capsule"]["fingerprint"]),
            short_hash(certificate["fingerprint"]),
            report["capsule"]["gates"]["passed"],
            report["capsule"]["gates"]["total"],
            report["capsule"]["readiness"]["passed"],
            report["capsule"]["readiness"]["total"],
            short_hash(report["capsule"]["model_sha256"]),
            short_hash(report["capsule"]["witness_proof"]),
        ),
        "",
        "## Trust Certificate",
        "",
        "| Area | Status | Evidence |",
        "| --- | --- | --- |",
        "| Integrity | {} | profile `{}`, files verified `{}` |".format(
            bool_text(integrity["profile_ok"]),
            short_hash(integrity["profile_sha256"]),
            str(integrity["file_evidence_verified"]).lower(),
        ),
        "| Reversibility | {} | checked steps {}, final chain `{}` |".format(
            bool_text(
                reversibility["training_lineage_replayed"]
                and reversibility["restored_initial_model"]
                and reversibility["lineage_ledger_matches"]
                and reversibility["balanced_recompute"]
            ),
            reversibility["checked_steps"],
            short_hash(reversibility["final_chain"]),
        ),
        "| Deterministic inference | {} | imported pred {}, native pred {}, eval row pred {}, Q31 ref pred {}, parity {} |".format(
            bool_text(
                imported_model["inference_replay_passed"]
                and imported_model["matches_native"]
                and inference["native_replay_passed"]
                and inference["evaluation_row_replay_passed"]
                and inference["q31_reference_matches_native"]
            ),
            imported_model["prediction"],
            inference["native_prediction"],
            inference["evaluation_row_prediction"],
            inference["q31_reference_prediction"],
            bool_text(inference["q31_reference_matches_native"]),
        ),
        "| Imported model | {} | source checked {}, provenance `{}`, source `{}` |".format(
            bool_text(
                imported_model["source_model_json_checked"]
                and imported_model["inference_replay_passed"]
                and imported_model["matches_native"]
            ),
            bool_text(imported_model["source_model_json_checked"]),
            imported_model["provenance_kind"],
            short_hash(imported_model["source_model_json_fingerprint"]),
        ),
        "| Reversible inference trace | {} | pred {}, label rank {}, correct {}, top-k correct {}, top `{}`, margin {}, witness {}, replay {}, ledgers `{}`/`{}` |".format(
            bool_text(reversible_trace["passed"]),
            reversible_trace["prediction"],
            reversible_trace["label_rank"],
            reversible_trace["correct"],
            reversible_trace["top2_correct"],
            ",".join(str(value) for value in reversible_trace["top_classes"]),
            reversible_trace["margin"],
            bytes_text(reversible_trace["witness_payload_bytes"]),
            bytes_text(reversible_trace["replay_payload_bytes"]),
            short_hash(reversible_trace["contribution_ledger_fingerprint"]),
            short_hash(reversible_trace["margin_contribution_ledger_fingerprint"]),
        ),
        "| Witnessing | {} | {} samples, dataset loop `{}`, proof `{}` |".format(
            bool_text(witnessing["mlp_passed"]),
            witnessing["mlp_samples"],
            dataset_loops,
            short_hash(witnessing["mlp_witness_proof_fingerprint"]),
        ),
        "| Cost envelope | {} | peak RSS {}, replay {}, trace/model {}, witness/model {} |".format(
            bool_text(payload["passed"]),
            bytes_text(cost["run_peak_rss_bytes"]),
            bytes_text(cost["max_replay_payload_bytes"]),
            ratio_text(cost["trace_to_model_payload_ratio"]),
            ratio_text(cost["witness_to_model_payload_ratio"]),
        ),
        "",
        "## Inference Action Contract",
        "",
        "| Operation | Supported | Evidence | Result | Cost |",
        "| --- | --- | --- | --- | --- |",
        *render_inference_action_contract_rows(inference["action_contract"]),
        "",
        "## Paths",
        "",
        "| Artifact | Path |",
        "| --- | --- |",
    ]
    for key in ("summary", "capsule", "manifest", "profile", "verification_markdown"):
        value = report["paths"].get(key)
        lines.append(f"| {key} | `{value}` |")
    lines.append("")
    return "\n".join(lines)


def write_synthetic_capsule_dir(root: Path) -> None:
    summary = profile_check.valid_pipeline_summary_report()
    capsule = profile_check.valid_model_capsule_report()
    manifest = profile_check.valid_pipeline_manifest_report()
    manifest["model_capsule_fingerprint"] = capsule["fingerprint"]
    (root / DEFAULT_SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (root / DEFAULT_CAPSULE_NAME).write_text(json.dumps(capsule, indent=2) + "\n", encoding="utf-8")
    (root / DEFAULT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    reports = [
        (Path("target/pipeline-summary.json"), summary),
        (Path("target/model-capsule.json"), capsule),
        (Path("target/pipeline-manifest.json"), manifest),
    ]
    (root / DEFAULT_PROFILE_NAME).write_text(
        profile_summary.render_markdown(reports),
        encoding="utf-8",
    )


def synthetic_artifact_record(path: Path, role: str, label: str, description: str) -> dict[str, Any]:
    if not path.exists():
        path.write_text(f"# {label}\n", encoding="utf-8")
    return {
        "path": str(path),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "label": label,
        "description": description,
    }


def write_synthetic_handoff(root: Path, certificate: dict[str, Any]) -> dict[str, Any]:
    summary = profile_check.load_json(root / DEFAULT_SUMMARY_NAME)
    capsule = profile_check.load_json(root / DEFAULT_CAPSULE_NAME)
    verification_path = root / DEFAULT_VERIFICATION_REPORT_NAME
    if not verification_path.exists():
        verification_path.write_text(
            json.dumps({"kind": VERIFICATION_KIND, "passed": True}, indent=2) + "\n",
            encoding="utf-8",
        )
    artifact_specs = {
        "pipeline_summary": (root / DEFAULT_SUMMARY_NAME, "machine_summary", "Pipeline summary JSON"),
        "pipeline_manifest": (root / DEFAULT_MANIFEST_NAME, "machine_manifest", "Pipeline manifest JSON"),
        "model_capsule": (root / DEFAULT_CAPSULE_NAME, "machine_capsule", "Model capsule JSON"),
        "model_capsule_verification": (
            verification_path,
            "machine_verdict",
            "Model capsule verification JSON",
        ),
        "model_capsule_verification_markdown": (
            root / DEFAULT_VERIFICATION_MARKDOWN_NAME,
            "human_verdict",
            "Model capsule verification Markdown",
        ),
        "model_bundle": (root / "model-bundle.json", "machine_model", "Signed model bundle JSON"),
        "imported_model_source": (
            root / "imported-q31-linear-model.json",
            "machine_import_source",
            "Imported Q31 model source JSON",
        ),
        "imported_model_bundle": (
            root / "imported-model-bundle.json",
            "machine_model",
            "Imported signed model bundle JSON",
        ),
        "imported_model_inference_bundle": (
            root / "imported-model-inference-bundle.json",
            "machine_replay",
            "Imported model inference replay bundle",
        ),
        "samples": (root / "samples.json", "machine_samples", "Signed sample set JSON"),
        "model_capsule_profile": (root / DEFAULT_PROFILE_NAME, "human_profile", "Model capsule profile"),
        "ml_profile": (root / "mnist-ml-audit-profile.md", "human_profile", "Full ML audit profile"),
        "training_audit_verification_markdown": (
            root / "training-audit-verification.md",
            "human_lineage_card",
            "Training-audit verification card",
        ),
        "training_step_debug": (
            root / "training-step-debug.md",
            "human_debug_card",
            "Training-step debug card",
        ),
        "q31_reference_inference": (
            root / "q31-reference-inference.json",
            "machine_reference",
            "Q31 reference inference JSON",
        ),
        "native_inference_bundle": (
            root / "native-inference-bundle.json",
            "machine_replay",
            "Native inference replay bundle JSON",
        ),
        "native_standalone_rev_classifier": (
            root / "mnist-standalone-classifier.rev",
            "source_reverie",
            "Standalone MNIST classifier source",
        ),
        "native_standalone_rev_run": (
            root / "mnist-standalone-classifier-run.json",
            "machine_run",
            "Standalone MNIST classifier run JSON",
        ),
        "native_standalone_rev_roundtrip": (
            root / "mnist-standalone-classifier-roundtrip.json",
            "machine_roundtrip",
            "Standalone MNIST classifier roundtrip JSON",
        ),
        "native_standalone_rev_roundtrip_verification": (
            root / "mnist-standalone-classifier-roundtrip-verification.json",
            "machine_roundtrip_verdict",
            "Standalone MNIST classifier roundtrip verification JSON",
        ),
        "q31_reference_inference_markdown": (
            root / "q31-reference-inference.md",
            "human_explanation_card",
            "Q31 inference explanation card",
        ),
        "native_inference_audit_markdown": (
            root / "native-inference-audit.md",
            "human_inference_audit_card",
            "Native inference audit card",
        ),
        "native_inference_verification_markdown": (
            root / "native-inference-verification.md",
            "human_inference_verification_card",
            "Native inference verification card",
        ),
        "evaluation_row_inference_audit_markdown": (
            root / "model-evaluation-row-inference-audit.md",
            "human_evaluation_inference_card",
            "Evaluation-row inference audit card",
        ),
        "mlp_witness": (root / "mlp-witness-report.json", "machine_witness", "MLP witness report JSON"),
        "mlp_witness_markdown": (
            root / "mlp-witness.md",
            "human_witness_card",
            "MLP witness proof card",
        ),
        "invertible_coupling_markdown": (
            root / "invertible-coupling.md",
            "human_layer_card",
            "Invertible coupling proof card",
        ),
        "triangular_residual_markdown": (
            root / "triangular-residual.md",
            "human_layer_card",
            "Triangular residual proof card",
        ),
        "reversible_preprocess_markdown": (
            root / "reversible-preprocess.md",
            "human_preprocess_card",
            "Reversible preprocess proof card",
        ),
        "reversible_inference_trace": (
            root / "reversible-inference-trace-report.json",
            "machine_replay",
            "Reversible inference trace JSON",
        ),
        "inference_trace_forward": (
            root / "reversible-inference-trace-forward.json",
            "machine_trace",
            "Reversible inference trace forward JSON",
        ),
        "inference_trace_reverse": (
            root / "reversible-inference-trace-reverse.json",
            "machine_trace",
            "Reversible inference trace reverse JSON",
        ),
        "reversible_inference_trace_markdown": (
            root / "reversible-inference-trace.md",
            "human_trace_card",
            "Reversible inference trace card",
        ),
        "reversible_inference_roundtrip_proof": (
            root / "reversible-inference-roundtrip-proof.json",
            "machine_roundtrip",
            "Reversible inference roundtrip proof JSON",
        ),
        "reversible_inference_roundtrip_verification": (
            root / "reversible-inference-roundtrip-verification.json",
            "machine_roundtrip_verdict",
            "Reversible inference roundtrip verification JSON",
        ),
        "reversible_inference_roundtrip_markdown": (
            root / "reversible-inference-roundtrip.md",
            "human_roundtrip_card",
            "Reversible inference roundtrip card",
        ),
    }
    artifacts = {
        artifact_id: synthetic_artifact_record(
            path,
            role,
            label,
            f"Synthetic {label}.",
        )
        for artifact_id, (path, role, label) in artifact_specs.items()
    }
    metrics = summary["metrics"]
    audit_step = profile_check.valid_audit_step_report()
    inference = profile_check.valid_q31_reference_inference_report()
    reversible_trace = metrics["reversible_inference_trace"]
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
            "fingerprint": certificate["fingerprint"],
            "passed": certificate["payload"]["passed"],
            "verification_markdown": {
                "renderer": "render_verification_markdown",
                "bytes": artifacts["model_capsule_verification_markdown"]["bytes"],
                "sha256": artifacts["model_capsule_verification_markdown"]["sha256"],
            },
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
            "prediction": inference["prediction"],
            "label": inference["label"],
            "correct": inference["correct"],
            "margin": inference["attribution"]["margin"],
            "runner_up_digit": inference["attribution"]["runner_up_digit"],
            "active_pixels": inference["active_pixels"],
            "contribution_ledger_fingerprint": inference["attribution"][
                "contribution_ledger_fingerprint"
            ],
            "margin_contribution_ledger_fingerprint": inference["attribution"][
                "margin_contribution_ledger_fingerprint"
            ],
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
            "model_evaluation_samples_per_second": metrics["scorecard"][
                "model_evaluation_samples_per_second"
            ],
            "reverse_to_train_elapsed_ratio": metrics["scorecard"]["reverse_to_train_elapsed_ratio"],
            "run_peak_rss_bytes": metrics["scorecard"]["run_peak_rss_bytes"],
            "max_replay_payload_bytes": metrics["scorecard"]["max_replay_payload_bytes"],
            "trace_to_model_payload_ratio": metrics["scorecard"]["trace_to_model_payload_ratio"],
            "witness_to_model_payload_ratio": metrics["scorecard"]["witness_to_model_payload_ratio"],
            "total_recompute_steps": metrics["scorecard"]["total_recompute_steps"],
        },
        "artifacts": artifacts,
        "inference_action_review_commands": [
            {
                "operation": "reproduce_prediction",
                "label": "Reproduce selected Q31 prediction",
                "purpose": "Synthetic command for selected Q31 prediction replay.",
                "command": ["python3", "scripts/check_q31_inference.py", "--json"],
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
                "purpose": "Synthetic command for selected Q31 attribution replay.",
                "command": ["python3", "scripts/check_q31_inference.py", "--markdown-output", "q31-reference-inference.md"],
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
                "purpose": "Synthetic command for imported model inference replay.",
                "command": ["cargo", "run", "-p", "reverie-cli", "--bin", "reverie-mnist-linear"],
                "artifacts": [
                    "imported_model_source",
                    "imported_model_bundle",
                    "imported_model_inference_bundle",
                ],
            },
            {
                "operation": "replay_native_inference",
                "label": "Replay native inference bundle",
                "purpose": "Synthetic command for native inference bundle replay.",
                "command": ["cargo", "run", "-p", "reverie-cli", "--bin", "reverie-mnist-linear"],
                "artifacts": [
                    "native_inference_bundle",
                    "native_inference_verification_markdown",
                ],
            },
            {
                "operation": "run_standalone_rev_classifier",
                "label": "Run standalone Reverie classifier",
                "purpose": "Synthetic command for source-only Reverie classifier replay.",
                "command": ["cargo", "run", "-p", "reverie-cli", "--", "run"],
                "artifacts": [
                    "native_standalone_rev_classifier",
                    "native_standalone_rev_run",
                ],
            },
            {
                "operation": "reverse_reversible_trace",
                "label": "Replay reversible inference trace",
                "purpose": "Synthetic command for reversible inference trace replay.",
                "command": ["python3", "scripts/check_reversible_inference_trace.py"],
                "artifacts": [
                    "inference_trace_forward",
                    "inference_trace_reverse",
                    "reversible_inference_trace",
                    "reversible_inference_trace_markdown",
                ],
            },
        ],
    }
    handoff = {
        "kind": HANDOFF_KIND,
        "algorithm": "sha256",
        "fingerprint": profile_check.sha256_json(payload),
        "payload": payload,
    }
    (root / DEFAULT_HANDOFF_NAME).write_text(json.dumps(handoff, indent=2) + "\n", encoding="utf-8")
    (root / DEFAULT_HANDOFF_MARKDOWN_NAME).write_text(
        render_handoff_markdown(handoff),
        encoding="utf-8",
    )
    return handoff


def expect_self_test_error(label: str, callback: Any, needle: str) -> None:
    try:
        callback()
    except ValueError as error:
        if needle not in str(error):
            raise AssertionError(f"{label} reported `{error}`, expected `{needle}`") from error
        return
    raise AssertionError(f"{label} did not fail")


def run_self_tests() -> int:
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_synthetic_capsule_dir(root)
            result = verify_capsule(
                {
                    "summary": root / DEFAULT_SUMMARY_NAME,
                    "capsule": root / DEFAULT_CAPSULE_NAME,
                    "manifest": root / DEFAULT_MANIFEST_NAME,
                    "profile": root / DEFAULT_PROFILE_NAME,
                    "verification_markdown": root / DEFAULT_VERIFICATION_MARKDOWN_NAME,
                },
                verify_file_evidence=False,
                allow_missing_profile=False,
            )
            if result["gate_passed"] != result["gate_total"]:
                raise AssertionError("valid capsule did not report all gates passed")
            report = verification_report(
                result,
                verify_file_evidence=False,
                allow_missing_profile=False,
                verification_markdown_path=root / DEFAULT_VERIFICATION_MARKDOWN_NAME,
            )
            if report["kind"] != VERIFICATION_KIND or report["passed"] is not True:
                raise AssertionError("verification report did not report success")
            if report["capsule"]["witness_proof"] != result["witness_proof"]:
                raise AssertionError("verification report did not include witness proof")
            if report["profile"]["sha256"] is None:
                raise AssertionError("verification report did not include profile SHA-256")
            certificate = report["trust_certificate"]
            if certificate["schema"] != "reverie_model_capsule_trust_certificate_v1":
                raise AssertionError("verification report has wrong trust-certificate schema")
            if certificate["fingerprint"] != profile_check.sha256_json(certificate["payload"]):
                raise AssertionError("trust-certificate fingerprint did not match payload")
            if certificate["payload"]["kind"] != TRUST_CERTIFICATE_KIND:
                raise AssertionError("trust-certificate payload has wrong kind")
            if certificate["payload"]["passed"] is not True:
                raise AssertionError("trust-certificate payload did not report success")
            if (
                certificate["payload"]["witnessing"]["mlp_dataset_loops"]
                != profile_check.expected_mlp_dataset_loops()
            ):
                raise AssertionError("trust-certificate did not include MLP dataset-loop proof")
            trace_certificate = certificate["payload"]["deterministic_inference"][
                "reversible_trace"
            ]
            if trace_certificate["top_classes"] != [1, 0, 2]:
                raise AssertionError("trust-certificate did not include reversible trace top-k classes")
            if trace_certificate["witness_payload_bytes"] != 120:
                raise AssertionError("trust-certificate did not include reversible trace witness cost")
            rendered_markdown = render_verification_markdown(report)
            if "Reversible inference trace" not in rendered_markdown:
                raise AssertionError("verification Markdown did not render reversible trace evidence")
            if report["verification_markdown"]["sha256"] != sha256_text(rendered_markdown):
                raise AssertionError("verification Markdown SHA-256 metadata did not match renderer")
            if report["verification_markdown"]["bytes"] != len(rendered_markdown.encode("utf-8")):
                raise AssertionError("verification Markdown byte metadata did not match renderer")
            if report["verification_markdown"]["saved"]["present"] is not False:
                raise AssertionError("missing optional verification Markdown should be reported absent")
            (root / DEFAULT_VERIFICATION_MARKDOWN_NAME).write_text(
                rendered_markdown,
                encoding="utf-8",
            )
            checked_markdown_report = verification_report(
                result,
                verify_file_evidence=False,
                allow_missing_profile=False,
                verification_markdown_path=root / DEFAULT_VERIFICATION_MARKDOWN_NAME,
                require_verification_markdown=True,
            )
            saved_markdown_metadata = checked_markdown_report["verification_markdown"]["saved"]
            if saved_markdown_metadata["present"] is not True:
                raise AssertionError("required verification Markdown should be reported present")
            if saved_markdown_metadata["matches_rendered"] is not True:
                raise AssertionError("required verification Markdown should match the renderer")
            if saved_markdown_metadata["sha256"] != report["verification_markdown"]["sha256"]:
                raise AssertionError("saved verification Markdown SHA-256 did not match generated metadata")
            (root / DEFAULT_VERIFICATION_REPORT_NAME).write_text(
                json.dumps(checked_markdown_report, indent=2) + "\n",
                encoding="utf-8",
            )
            write_synthetic_handoff(root, checked_markdown_report["trust_certificate"])
            handoff_report = verification_report(
                result,
                verify_file_evidence=False,
                allow_missing_profile=False,
                verification_markdown_path=root / DEFAULT_VERIFICATION_MARKDOWN_NAME,
                require_verification_markdown=True,
                handoff_path=root / DEFAULT_HANDOFF_NAME,
                handoff_markdown_path=root / DEFAULT_HANDOFF_MARKDOWN_NAME,
                require_handoff=True,
            )
            if handoff_report["handoff"]["present"] is not True:
                raise AssertionError("required handoff should be reported present")
            expected_artifact_count = len(
                set(REVIEW_CARD_ARTIFACTS) | set(MACHINE_HANDOFF_ARTIFACTS)
            )
            if handoff_report["handoff"]["artifact_count"] != expected_artifact_count:
                raise AssertionError("synthetic handoff should expose all review artifacts")
            if handoff_report["handoff"]["markdown"]["matches_rendered"] is not True:
                raise AssertionError("synthetic handoff Markdown should match renderer")
            handoff_payload = profile_check.load_json(root / DEFAULT_HANDOFF_NAME)["payload"]
            if handoff_payload["selected_inference"]["trace_top_classes"] != [1, 0, 2]:
                raise AssertionError("synthetic handoff did not include reversible trace top-k classes")
            action_commands = handoff_report["handoff"]["inference_action_review_commands"]
            if action_commands["operations"] != [
                "reproduce_prediction",
                "explain_margin",
                "replay_imported_model_inference",
                "replay_native_inference",
                "run_standalone_rev_classifier",
                "reverse_reversible_trace",
            ]:
                raise AssertionError("synthetic handoff should expose every inference action command")
            if "Inference Action Review Commands" not in (
                root / DEFAULT_HANDOFF_MARKDOWN_NAME
            ).read_text(encoding="utf-8"):
                raise AssertionError("synthetic handoff Markdown should render inference action commands")
            executable_handoff = profile_check.load_json(root / DEFAULT_HANDOFF_NAME)
            for command in executable_handoff["payload"]["inference_action_review_commands"]:
                operation = command["operation"]
                command["command"] = [sys.executable, "-c", f"print({operation!r})"]
            executable_handoff["fingerprint"] = profile_check.sha256_json(
                executable_handoff["payload"]
            )
            receipt = run_inference_action_review_commands(
                executable_handoff,
                capsule_fingerprint=handoff_report["capsule"]["fingerprint"],
                trust_certificate_fingerprint=handoff_report["trust_certificate"]["fingerprint"],
                cwd=root,
                timeout_seconds=10.0,
            )
            if receipt["kind"] != ACTION_REVIEW_RECEIPT_KIND:
                raise AssertionError("action command receipt has wrong kind")
            if receipt["payload"]["schema"] != ACTION_REVIEW_RECEIPT_SCHEMA:
                raise AssertionError("action command receipt has wrong schema")
            if receipt["passed"] is not True:
                raise AssertionError("executable synthetic action commands should pass")
            if receipt["fingerprint"] != profile_check.sha256_json(receipt["payload"]):
                raise AssertionError("action command receipt fingerprint did not match payload")
            if (
                receipt["payload"]["semantic_fingerprint"]
                != inference_action_review_semantic_fingerprint(receipt["payload"])
            ):
                raise AssertionError("action command receipt semantic fingerprint did not match payload")
            if receipt["payload"]["summary"]["passed_count"] != len(action_commands["operations"]):
                raise AssertionError("action command receipt did not count every command")
            if [
                row["operation"] for row in receipt["payload"]["operations"]
            ] != action_commands["operations"]:
                raise AssertionError("action command receipt changed operation order")
            for row in receipt["payload"]["operations"]:
                if row["passed"] is not True or row["exit_code"] != 0:
                    raise AssertionError("executable synthetic command did not report success")
                if row["operation"] not in row["stdout_tail"]:
                    raise AssertionError("action command receipt did not retain bounded stdout")
            receipt_markdown = render_inference_action_review_receipt_markdown(receipt)
            for snippet in (
                "# Reverie Inference Action Review Receipt",
                "replay_imported_model_inference",
                "run_standalone_rev_classifier",
                "reverse_reversible_trace",
                executable_handoff["fingerprint"][:12],
            ):
                if snippet not in receipt_markdown:
                    raise AssertionError(f"action command receipt Markdown missing `{snippet}`")
            (root / "inference-action-review-receipt.json").write_text(
                json.dumps(receipt, indent=2) + "\n",
                encoding="utf-8",
            )
            (root / "inference-action-review-receipt.md").write_text(
                receipt_markdown,
                encoding="utf-8",
            )
            file_checked_report = verification_report(
                result,
                verify_file_evidence=True,
                allow_missing_profile=False,
                check_verification_markdown=False,
            )
            write_synthetic_handoff(root, file_checked_report["trust_certificate"])
            stale_mlp_card = root / "mlp-witness.md"
            stale_mlp_card.write_text("# stale\n", encoding="utf-8")
            expect_self_test_error(
                "stale handoff artifact",
                lambda: verification_report(
                    result,
                    verify_file_evidence=True,
                    allow_missing_profile=False,
                    verification_markdown_path=root / DEFAULT_VERIFICATION_MARKDOWN_NAME,
                    check_verification_markdown=False,
                    handoff_path=root / DEFAULT_HANDOFF_NAME,
                    handoff_markdown_path=root / DEFAULT_HANDOFF_MARKDOWN_NAME,
                    require_handoff=True,
                ),
                "mismatch",
            )
            write_synthetic_handoff(root, checked_markdown_report["trust_certificate"])
            (root / DEFAULT_HANDOFF_MARKDOWN_NAME).write_text("# stale\n", encoding="utf-8")
            expect_self_test_error(
                "stale handoff Markdown",
                lambda: verification_report(
                    result,
                    verify_file_evidence=False,
                    allow_missing_profile=False,
                    verification_markdown_path=root / DEFAULT_VERIFICATION_MARKDOWN_NAME,
                    require_verification_markdown=True,
                    handoff_path=root / DEFAULT_HANDOFF_NAME,
                    handoff_markdown_path=root / DEFAULT_HANDOFF_MARKDOWN_NAME,
                    require_handoff=True,
                ),
                "handoff Markdown is stale",
            )
            write_synthetic_handoff(root, checked_markdown_report["trust_certificate"])
            (root / DEFAULT_HANDOFF_NAME).unlink()
            expect_self_test_error(
                "missing required handoff",
                lambda: verification_report(
                    result,
                    verify_file_evidence=False,
                    allow_missing_profile=False,
                    verification_markdown_path=root / DEFAULT_VERIFICATION_MARKDOWN_NAME,
                    require_verification_markdown=True,
                    handoff_path=root / DEFAULT_HANDOFF_NAME,
                    handoff_markdown_path=root / DEFAULT_HANDOFF_MARKDOWN_NAME,
                    require_handoff=True,
                ),
                "handoff JSON is missing",
            )
            write_synthetic_handoff(root, checked_markdown_report["trust_certificate"])
            (root / DEFAULT_VERIFICATION_MARKDOWN_NAME).write_text("# stale\n", encoding="utf-8")
            expect_self_test_error(
                "stale verification Markdown",
                lambda: verification_report(
                    result,
                    verify_file_evidence=False,
                    allow_missing_profile=False,
                    verification_markdown_path=root / DEFAULT_VERIFICATION_MARKDOWN_NAME,
                    require_verification_markdown=True,
                ),
                "verification Markdown is stale",
            )
            (root / DEFAULT_VERIFICATION_MARKDOWN_NAME).unlink()
            expect_self_test_error(
                "missing required verification Markdown",
                lambda: verification_report(
                    result,
                    verify_file_evidence=False,
                    allow_missing_profile=False,
                    verification_markdown_path=root / DEFAULT_VERIFICATION_MARKDOWN_NAME,
                    require_verification_markdown=True,
                ),
                "verification Markdown is missing",
            )
            failed_result = deepcopy(result)
            failed_result["capsule"]["payload"]["scorecard"]["balanced_recompute"] = False
            failed_report = verification_report(
                failed_result,
                verify_file_evidence=False,
                allow_missing_profile=False,
            )
            if failed_report["passed"] is not False:
                raise AssertionError("failed trust certificate did not fail verification report")
            if failed_report["checks"]["trust_certificate_passed"] is not False:
                raise AssertionError("failed trust certificate check did not report false")
            if failed_report["trust_certificate"]["payload"]["passed"] is not False:
                raise AssertionError("failed trust certificate payload did not report false")
            if (
                failed_report["trust_certificate"]["fingerprint"]
                != profile_check.sha256_json(failed_report["trust_certificate"]["payload"])
            ):
                raise AssertionError("failed trust-certificate fingerprint did not match payload")
            output_path = root / "capsule-verification.json"
            args = argparse.Namespace(
                path=root,
                summary=None,
                capsule=None,
                manifest=None,
                profile=None,
                skip_file_evidence=True,
                allow_missing_profile=False,
                json=False,
                output=output_path,
                markdown_output=root / "capsule-verification.md",
                verification_markdown=None,
                require_verification_markdown=False,
                handoff=None,
                handoff_markdown=None,
                require_handoff=False,
                self_test=False,
            )
            paths = resolve_capsule_paths(args)
            output_result = verify_capsule(
                paths,
                verify_file_evidence=not args.skip_file_evidence,
                allow_missing_profile=args.allow_missing_profile,
            )
            output_report = verification_report(
                output_result,
                verify_file_evidence=not args.skip_file_evidence,
                allow_missing_profile=args.allow_missing_profile,
                verification_markdown_path=paths["verification_markdown"],
                check_verification_markdown=False,
            )
            markdown = render_verification_markdown(output_report)
            args.markdown_output.write_text(markdown, encoding="utf-8")
            output_report = verification_report(
                output_result,
                verify_file_evidence=not args.skip_file_evidence,
                allow_missing_profile=args.allow_missing_profile,
                verification_markdown_path=paths["verification_markdown"],
                require_verification_markdown=True,
            )
            output_path.write_text(json.dumps(output_report, indent=2) + "\n", encoding="utf-8")
            saved = profile_check.load_json(output_path)
            if saved["kind"] != VERIFICATION_KIND:
                raise AssertionError("saved verification report has wrong kind")
            saved_markdown = args.markdown_output.read_text(encoding="utf-8")
            if sha256_text(saved_markdown) != saved["verification_markdown"]["sha256"]:
                raise AssertionError("saved verification Markdown hash did not match JSON metadata")
            if len(saved_markdown.encode("utf-8")) != saved["verification_markdown"]["bytes"]:
                raise AssertionError("saved verification Markdown bytes did not match JSON metadata")
            if saved["verification_markdown"]["saved"]["matches_rendered"] is not True:
                raise AssertionError("saved verification Markdown metadata did not report a rendered match")
            for snippet in (
                "# Reverie Model Capsule Verification",
                "## Trust Certificate",
                "## Inference Action Contract",
                "Deterministic inference",
                "replay_imported_model_inference",
                "run_standalone_rev_classifier",
                "reverse_reversible_trace",
                "sample by labels",
                output_report["trust_certificate"]["fingerprint"][:12],
            ):
                if snippet not in saved_markdown:
                    raise AssertionError(f"saved verification Markdown missing `{snippet}`")
            stale_profile = root / DEFAULT_PROFILE_NAME
            stale_profile.write_text("# stale\n", encoding="utf-8")
            expect_self_test_error(
                "stale capsule profile",
                lambda: verify_capsule(
                    resolve_capsule_paths(argparse.Namespace(
                        path=root,
                        summary=None,
                        capsule=None,
                        manifest=None,
                        profile=None,
                    )),
                    verify_file_evidence=False,
                    allow_missing_profile=False,
                ),
                "capsule profile is stale",
            )
            write_synthetic_capsule_dir(root)
            manifest_path = root / DEFAULT_MANIFEST_NAME
            manifest = profile_check.load_json(manifest_path)
            manifest["model_capsule_fingerprint"] = "2" * 64
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            expect_self_test_error(
                "stale manifest fingerprint",
                lambda: verify_capsule(
                    resolve_capsule_paths(argparse.Namespace(
                        path=root,
                        summary=None,
                        capsule=None,
                        manifest=None,
                        profile=None,
                    )),
                    verify_file_evidence=False,
                    allow_missing_profile=False,
                ),
                "model_capsule_fingerprint",
            )
            write_synthetic_capsule_dir(root)
            (root / DEFAULT_PROFILE_NAME).unlink()
            missing_profile_result = verify_capsule(
                resolve_capsule_paths(argparse.Namespace(
                    path=root,
                    summary=None,
                    capsule=None,
                    manifest=None,
                    profile=None,
                )),
                verify_file_evidence=False,
                allow_missing_profile=True,
            )
            missing_profile_report = verification_report(
                missing_profile_result,
                verify_file_evidence=False,
                allow_missing_profile=True,
            )
            if missing_profile_report["passed"] is not True:
                raise AssertionError("allowed missing profile should not fail the trust certificate")
            missing_profile_integrity = missing_profile_report["trust_certificate"]["payload"]["integrity"]
            if missing_profile_integrity["profile_required"] is not False:
                raise AssertionError("allowed missing profile did not clear profile_required")
            if missing_profile_integrity["profile_ok"] is not True:
                raise AssertionError("allowed missing profile did not set profile_ok")
    except (AssertionError, ValueError) as error:
        print(f"error: self-test failed: {error}", file=sys.stderr)
        return 1
    print("ok: model capsule verifier self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    run_action_commands = (
        args.run_inference_action_commands
        or args.action_command_receipt_output is not None
        or args.action_command_receipt_markdown is not None
    )
    if args.action_command_timeout <= 0:
        print("error: action command timeout must be positive", file=sys.stderr)
        return 1
    paths = resolve_capsule_paths(args)
    try:
        result = verify_capsule(
            paths,
            verify_file_evidence=not args.skip_file_evidence,
            allow_missing_profile=args.allow_missing_profile,
        )
        report = verification_report(
            result,
            verify_file_evidence=not args.skip_file_evidence,
            allow_missing_profile=args.allow_missing_profile,
            verification_markdown_path=paths.get("verification_markdown"),
            handoff_path=paths.get("handoff"),
            handoff_markdown_path=paths.get("handoff_markdown"),
            require_handoff=args.require_handoff or run_action_commands,
            require_verification_markdown=(
                args.require_verification_markdown
                or args.verification_markdown is not None
            ),
            check_verification_markdown=(
                args.markdown_output is None
                or args.require_verification_markdown
                or args.verification_markdown is not None
            ),
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.markdown_output is not None:
        if args.markdown_output.parent != Path("."):
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_verification_markdown(report), encoding="utf-8")
        try:
            report = verification_report(
                result,
                verify_file_evidence=not args.skip_file_evidence,
                allow_missing_profile=args.allow_missing_profile,
                verification_markdown_path=paths.get("verification_markdown"),
                handoff_path=paths.get("handoff"),
                handoff_markdown_path=paths.get("handoff_markdown"),
                require_handoff=args.require_handoff or run_action_commands,
                require_verification_markdown=True,
            )
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    receipt_failed = False
    if run_action_commands:
        handoff_path = paths.get("handoff")
        if handoff_path is None:
            print("error: handoff path is required to run inference action commands", file=sys.stderr)
            return 1
        try:
            handoff = profile_check.load_json(handoff_path)
            receipt = run_inference_action_review_commands(
                handoff,
                capsule_fingerprint=report["capsule"]["fingerprint"],
                trust_certificate_fingerprint=report["trust_certificate"]["fingerprint"],
                cwd=REPO_ROOT,
                timeout_seconds=args.action_command_timeout,
            )
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        report["inference_action_review_receipt"] = receipt
        report["checks"]["inference_action_review_commands_replayed"] = receipt["passed"]
        receipt_failed = receipt["passed"] is not True
        if args.action_command_receipt_output is not None:
            if args.action_command_receipt_output.parent != Path("."):
                args.action_command_receipt_output.parent.mkdir(parents=True, exist_ok=True)
            args.action_command_receipt_output.write_text(
                json.dumps(receipt, indent=2) + "\n",
                encoding="utf-8",
            )
        if args.action_command_receipt_markdown is not None:
            if args.action_command_receipt_markdown.parent != Path("."):
                args.action_command_receipt_markdown.parent.mkdir(parents=True, exist_ok=True)
            args.action_command_receipt_markdown.write_text(
                render_inference_action_review_receipt_markdown(receipt),
                encoding="utf-8",
            )
    if args.output is not None:
        if args.output.parent != Path("."):
            args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
        return 1 if receipt_failed else 0
    if args.output is not None:
        print(f"verification report: {args.output}")
    if args.markdown_output is not None:
        print(f"verification markdown: {args.markdown_output}")
    if args.action_command_receipt_output is not None:
        print(f"inference action receipt: {args.action_command_receipt_output}")
    if args.action_command_receipt_markdown is not None:
        print(f"inference action receipt markdown: {args.action_command_receipt_markdown}")
    if receipt_failed:
        failed = ", ".join(
            report["inference_action_review_receipt"]["payload"]["summary"]["failed_operations"]
        )
        print(f"error: inference action command replay failed: {failed}", file=sys.stderr)
        return 1
    print("ok: verified Reverie ML model capsule")
    print(f"capsule: {result['fingerprint']}")
    print(f"gates: {result['gate_passed']}/{result['gate_total']}")
    print(f"model: {result['model_sha256']}")
    print(f"witness proof: {result['witness_proof']}")
    if result["profile"] is not None:
        print(f"profile: {result['profile']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
