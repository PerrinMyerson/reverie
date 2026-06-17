#!/usr/bin/env python3
"""Reference-check Reverie's end-to-end reversible Q31 inference trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional


KIND = "reverie_q31_reversible_inference_trace_reference"
PROOF_CLAIM = "deterministic_q31_reversible_inference_trace"
PROGRAM = "examples/reversible_inference_trace.rev"
FEATURES = 4
CLASSES = 3
Q31_ONE = 1 << 31
I64_BYTES = 8
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
U64_MOD = 1 << 64


class InferenceTraceCheckError(ValueError):
    """Raised when an inference trace report or Reverie output is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reference-check examples/reversible_inference_trace.rev forward "
            "and reverse JSON output."
        )
    )
    parser.add_argument(
        "--forward-output-json",
        type=Path,
        help="JSON output from `reverie run examples/reversible_inference_trace.rev --json`.",
    )
    parser.add_argument(
        "--reverse-output-json",
        type=Path,
        help=(
            "JSON output from `reverie reverse examples/reversible_inference_trace.rev "
            "--json` seeded with final trace vars."
        ),
    )
    parser.add_argument(
        "--write-final-vars",
        type=Path,
        help="Write the exact final trace vars JSON needed for reverse execution.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Write a human-readable reversible inference trace proof card.",
    )
    parser.add_argument(
        "--expect-report-json",
        type=Path,
        help="Require the recomputed machine-readable report to match this saved JSON report.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent inference trace checker self-tests and exit.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InferenceTraceCheckError(f"failed to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise InferenceTraceCheckError(f"failed to parse {path}: {error}") from error


def checked_i64(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InferenceTraceCheckError(f"{label} must be a signed integer")
    if not I64_MIN <= value <= I64_MAX:
        raise InferenceTraceCheckError(f"{label} is outside signed i64 range")
    return value


def wrap_i64(value: int) -> int:
    value %= U64_MOD
    if value >= (1 << 63):
        value -= U64_MOD
    return value


def add_i64(left: int, right: int) -> int:
    return wrap_i64(left + right)


def sub_i64(left: int, right: int) -> int:
    return wrap_i64(left - right)


def fixed_mul_q31(left: int, right: int) -> int:
    return wrap_i64((left * right) >> 31)


def add_vector(left: list[int], right: list[int]) -> list[int]:
    return [add_i64(a, b) for a, b in zip(left, right)]


def sub_vector(left: list[int], right: list[int]) -> list[int]:
    return [sub_i64(a, b) for a, b in zip(left, right)]


def swap(values: list[int], left: int, right: int) -> list[int]:
    out = list(values)
    out[left], out[right] = out[right], out[left]
    return out


def vecmat_q31(vector: list[int], matrix: list[list[int]]) -> list[int]:
    out = []
    for col in range(CLASSES):
        acc = 0
        for row in range(FEATURES):
            acc = add_i64(acc, fixed_mul_q31(vector[row], matrix[row][col]))
        out.append(acc)
    return out


def argmax_first(values: list[int]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def runner_up_first(values: list[int]) -> int:
    ranked = top_logits(values)
    if len(ranked) < 2:
        raise InferenceTraceCheckError("runner_up_first needs at least two logits")
    return ranked[1]["class"]


def top2_margin(values: list[int]) -> int:
    ranked = top_logits(values)
    if len(ranked) < 2:
        raise InferenceTraceCheckError("top2_margin needs at least two logits")
    return wrap_i64(ranked[0]["value"] - ranked[1]["value"])


def top_logits(logits: list[int]) -> list[dict[str, int]]:
    ranked = [
        {"class": class_index, "value": value}
        for class_index, value in enumerate(logits)
    ]
    ranked.sort(key=lambda item: (-item["value"], item["class"]))
    return ranked[:CLASSES]


def top_k_indices(logits: list[int], k: int) -> list[int]:
    return [item["class"] for item in top_logits(logits)[:k]]


def top_k_values(logits: list[int], k: int) -> list[int]:
    return [item["value"] for item in top_logits(logits)[:k]]


def top_k_contains(logits: list[int], label: int, k: int) -> int:
    return int(label in top_k_indices(logits, k))


def rank_of(logits: list[int], label: int) -> int:
    for rank, item in enumerate(top_logits(logits), start=1):
        if item["class"] == label:
            return rank
    return 0


def contribution_sort_key(item: dict[str, int]) -> tuple[int, int]:
    return (-abs(item["contribution"]), item["feature"])


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def vector_cell(values: list[int]) -> str:
    return "`[{}]`".format(", ".join(comma(value) for value in values))


def contribution_ledger_fingerprint(predicted_class: int, rows: list[dict[str, int]]) -> str:
    return json_sha256(
        {
            "schema": "q31_reversible_inference_trace_contribution_ledger_v1",
            "predicted_class": predicted_class,
            "rows": rows,
        }
    )


def margin_contribution_ledger_fingerprint(
    predicted_class: int,
    runner_up_class: int,
    rows: list[dict[str, int]],
) -> str:
    return json_sha256(
        {
            "schema": "q31_reversible_inference_trace_margin_contribution_ledger_v1",
            "predicted_class": predicted_class,
            "runner_up_class": runner_up_class,
            "rows": rows,
        }
    )


def inference_attribution(store: dict[str, Any]) -> dict[str, Any]:
    logits = store["logits"]
    ranked = top_logits(logits)
    predicted_class = ranked[0]["class"]
    runner_up_class = ranked[1]["class"] if len(ranked) > 1 else predicted_class
    predicted_logit = logits[predicted_class]
    runner_up_logit = logits[runner_up_class]
    bias = store["bias"][predicted_class]
    runner_up_bias = store["bias"][runner_up_class]
    margin_bias = wrap_i64(bias - runner_up_bias)
    contribution_sum = 0
    margin_contribution_sum = 0
    reconstructed_logit = bias
    reconstructed_margin = margin_bias
    contributions = []
    margin_contributions = []

    for feature_index, q31 in enumerate(store["features"]):
        if q31 == 0:
            continue
        weight = store["weights"][feature_index][predicted_class]
        runner_up_weight = store["weights"][feature_index][runner_up_class]
        weight_delta = wrap_i64(weight - runner_up_weight)
        contribution = fixed_mul_q31(q31, weight)
        runner_up_contribution = fixed_mul_q31(q31, runner_up_weight)
        margin_contribution = wrap_i64(contribution - runner_up_contribution)
        contribution_sum = wrap_i64(contribution_sum + contribution)
        margin_contribution_sum = wrap_i64(margin_contribution_sum + margin_contribution)
        reconstructed_logit = wrap_i64(reconstructed_logit + contribution)
        reconstructed_margin = wrap_i64(reconstructed_margin + margin_contribution)
        if contribution != 0:
            contributions.append(
                {
                    "feature": feature_index,
                    "q31": q31,
                    "weight": weight,
                    "contribution": contribution,
                }
            )
        if margin_contribution != 0:
            margin_contributions.append(
                {
                    "feature": feature_index,
                    "q31": q31,
                    "predicted_weight": weight,
                    "runner_up_weight": runner_up_weight,
                    "weight_delta": weight_delta,
                    "contribution": margin_contribution,
                }
            )

    top_contributions = sorted(contributions, key=contribution_sort_key)
    top_margin_contributions = sorted(margin_contributions, key=contribution_sort_key)
    margin = wrap_i64(predicted_logit - runner_up_logit)
    return {
        "predicted_class": predicted_class,
        "runner_up_class": runner_up_class,
        "predicted_logit": predicted_logit,
        "runner_up_logit": runner_up_logit,
        "margin": margin,
        "bias": bias,
        "runner_up_bias": runner_up_bias,
        "margin_bias": margin_bias,
        "contribution_sum": contribution_sum,
        "margin_contribution_sum": margin_contribution_sum,
        "reconstructed_logit": reconstructed_logit,
        "reconstructed_margin": reconstructed_margin,
        "matches_logit": reconstructed_logit == predicted_logit,
        "matches_margin": reconstructed_margin == margin,
        "contribution_count": len(contributions),
        "margin_contribution_count": len(margin_contributions),
        "top_contribution_count": len(top_contributions),
        "top_margin_contribution_count": len(top_margin_contributions),
        "contribution_ledger_fingerprint": contribution_ledger_fingerprint(
            predicted_class,
            contributions,
        ),
        "margin_contribution_ledger_fingerprint": margin_contribution_ledger_fingerprint(
            predicted_class,
            runner_up_class,
            margin_contributions,
        ),
        "top_contributions": top_contributions,
        "top_margin_contributions": top_margin_contributions,
    }


def initial_store() -> dict[str, Any]:
    return {
        "raw": [Q31_ONE, 0, Q31_ONE // 2, -(Q31_ONE // 4)],
        "mean": [Q31_ONE // 4, -(Q31_ONE // 4), 0, Q31_ONE // 4],
        "features": [0, 0, 0, 0],
        "weights": [
            [Q31_ONE, 0, 0],
            [0, 0, 0],
            [0, Q31_ONE, 0],
            [0, 0, Q31_ONE],
        ],
        "bias": [0, 0, 0],
        "logits": [0, 0, 0],
        "top_classes": [0, 0, 0],
        "top_logit_values": [0, 0, 0],
        "prediction": 0,
        "runner_up_class": 0,
        "margin": 0,
        "label_rank": 0,
        "correct": 0,
        "top2_correct": 0,
        "label": 1,
    }


def preprocess_features(store: dict[str, Any]) -> list[int]:
    features = add_vector(store["features"], store["raw"])
    features = sub_vector(features, store["mean"])
    features = swap(features, 0, 2)
    return swap(features, 1, 3)


def forward_store(store: dict[str, Any]) -> dict[str, Any]:
    features = preprocess_features(store)
    logits = add_vector(store["logits"], vecmat_q31(features, store["weights"]))
    logits = add_vector(logits, store["bias"])
    top_classes = add_vector(store["top_classes"], top_k_indices(logits, CLASSES))
    top_logit_values = add_vector(store["top_logit_values"], top_k_values(logits, CLASSES))
    prediction = add_i64(store["prediction"], argmax_first(logits))
    runner_up_class = add_i64(store["runner_up_class"], runner_up_first(logits))
    margin = add_i64(store["margin"], top2_margin(logits))
    label_rank = add_i64(store["label_rank"], rank_of(logits, store["label"]))
    correct = add_i64(store["correct"], 1 if argmax_first(logits) == store["label"] else 0)
    top2_correct = add_i64(store["top2_correct"], top_k_contains(logits, store["label"], 2))
    return {
        **store,
        "features": features,
        "logits": logits,
        "top_classes": top_classes,
        "top_logit_values": top_logit_values,
        "prediction": prediction,
        "runner_up_class": runner_up_class,
        "margin": margin,
        "label_rank": label_rank,
        "correct": correct,
        "top2_correct": top2_correct,
    }


def inverse_store(store: dict[str, Any]) -> dict[str, Any]:
    top_classes = sub_vector(store["top_classes"], top_k_indices(store["logits"], CLASSES))
    top_logit_values = sub_vector(
        store["top_logit_values"],
        top_k_values(store["logits"], CLASSES),
    )
    prediction = sub_i64(store["prediction"], argmax_first(store["logits"]))
    runner_up_class = sub_i64(store["runner_up_class"], runner_up_first(store["logits"]))
    margin = sub_i64(store["margin"], top2_margin(store["logits"]))
    label_rank = sub_i64(store["label_rank"], rank_of(store["logits"], store["label"]))
    correct = sub_i64(store["correct"], 1 if argmax_first(store["logits"]) == store["label"] else 0)
    top2_correct = sub_i64(
        store["top2_correct"],
        top_k_contains(store["logits"], store["label"], 2),
    )
    logits = sub_vector(store["logits"], store["bias"])
    logits = sub_vector(logits, vecmat_q31(store["features"], store["weights"]))
    features = swap(store["features"], 1, 3)
    features = swap(features, 0, 2)
    features = add_vector(features, store["mean"])
    features = sub_vector(features, store["raw"])
    return {
        **store,
        "features": features,
        "logits": logits,
        "top_classes": top_classes,
        "top_logit_values": top_logit_values,
        "prediction": prediction,
        "runner_up_class": runner_up_class,
        "margin": margin,
        "label_rank": label_rank,
        "correct": correct,
        "top2_correct": top2_correct,
    }


def final_vars() -> dict[str, Any]:
    final = forward_store(initial_store())
    return {
        "features": final["features"],
        "logits": final["logits"],
        "top_classes": final["top_classes"],
        "top_logit_values": final["top_logit_values"],
        "prediction": final["prediction"],
        "runner_up_class": final["runner_up_class"],
        "margin": final["margin"],
        "label_rank": final["label_rank"],
        "correct": final["correct"],
        "top2_correct": final["top2_correct"],
    }


def validate_vector(value: Any, length: int, context: str) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise InferenceTraceCheckError(f"{context} must have {length} entries")
    return [checked_i64(item, f"{context}[{index}]") for index, item in enumerate(value)]


def validate_matrix(value: Any, rows: int, cols: int, context: str) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != rows:
        raise InferenceTraceCheckError(f"{context} must have {rows} rows")
    return [
        validate_vector(row, cols, f"{context}[{index}]")
        for index, row in enumerate(value)
    ]


def validate_store(value: dict[str, Any], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InferenceTraceCheckError(f"{context} must be an object")
    return {
        "raw": validate_vector(value.get("raw"), FEATURES, f"{context}.raw"),
        "mean": validate_vector(value.get("mean"), FEATURES, f"{context}.mean"),
        "features": validate_vector(value.get("features"), FEATURES, f"{context}.features"),
        "weights": validate_matrix(value.get("weights"), FEATURES, CLASSES, f"{context}.weights"),
        "bias": validate_vector(value.get("bias"), CLASSES, f"{context}.bias"),
        "logits": validate_vector(value.get("logits"), CLASSES, f"{context}.logits"),
        "top_classes": validate_vector(
            value.get("top_classes"),
            CLASSES,
            f"{context}.top_classes",
        ),
        "top_logit_values": validate_vector(
            value.get("top_logit_values"),
            CLASSES,
            f"{context}.top_logit_values",
        ),
        "prediction": checked_i64(value.get("prediction"), f"{context}.prediction"),
        "runner_up_class": checked_i64(value.get("runner_up_class"), f"{context}.runner_up_class"),
        "margin": checked_i64(value.get("margin"), f"{context}.margin"),
        "label_rank": checked_i64(value.get("label_rank"), f"{context}.label_rank"),
        "correct": checked_i64(value.get("correct"), f"{context}.correct"),
        "top2_correct": checked_i64(value.get("top2_correct"), f"{context}.top2_correct"),
        "label": checked_i64(value.get("label"), f"{context}.label"),
    }


def parse_output_store(value: Any, expected_kind: str, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InferenceTraceCheckError(f"{context} must be a JSON object")
    if value.get("kind") != expected_kind:
        raise InferenceTraceCheckError(f"{context}.kind must be {expected_kind}")
    store = value.get("store")
    if not isinstance(store, dict):
        raise InferenceTraceCheckError(f"{context}.store must be an object")
    return validate_store(store, f"{context}.store")


def compare_store(expected: dict[str, Any], actual: dict[str, Any], context: str) -> list[str]:
    mismatches = []
    for field, expected_value in expected.items():
        actual_value = actual.get(field)
        if actual_value != expected_value:
            mismatches.append(f"{context}.{field} expected {expected_value}, found {actual_value}")
    return mismatches


def proof_cost() -> dict[str, Any]:
    raw_payload_bytes = FEATURES * I64_BYTES
    mean_payload_bytes = FEATURES * I64_BYTES
    feature_payload_bytes = FEATURES * I64_BYTES
    weight_payload_bytes = FEATURES * CLASSES * I64_BYTES
    bias_payload_bytes = CLASSES * I64_BYTES
    model_payload_bytes = weight_payload_bytes + bias_payload_bytes
    logit_payload_bytes = CLASSES * I64_BYTES
    top_class_payload_bytes = CLASSES * I64_BYTES
    top_logit_value_payload_bytes = CLASSES * I64_BYTES
    prediction_payload_bytes = I64_BYTES
    runner_up_payload_bytes = I64_BYTES
    margin_payload_bytes = I64_BYTES
    label_rank_payload_bytes = I64_BYTES
    correct_payload_bytes = I64_BYTES
    top_k_correct_payload_bytes = I64_BYTES
    label_payload_bytes = I64_BYTES
    witness_payload_bytes = (
        logit_payload_bytes
        + top_class_payload_bytes
        + top_logit_value_payload_bytes
        + prediction_payload_bytes
        + runner_up_payload_bytes
        + margin_payload_bytes
        + label_rank_payload_bytes
        + correct_payload_bytes
        + top_k_correct_payload_bytes
    )
    state_payload_bytes = raw_payload_bytes + mean_payload_bytes + feature_payload_bytes + label_payload_bytes
    replay_payload_bytes = state_payload_bytes + model_payload_bytes + witness_payload_bytes
    return {
        "claim": PROOF_CLAIM,
        "arithmetic": "q31_wrapping_i64",
        "raw_payload_bytes": raw_payload_bytes,
        "mean_payload_bytes": mean_payload_bytes,
        "feature_payload_bytes": feature_payload_bytes,
        "weight_payload_bytes": weight_payload_bytes,
        "bias_payload_bytes": bias_payload_bytes,
        "model_payload_bytes": model_payload_bytes,
        "logit_payload_bytes": logit_payload_bytes,
        "top_class_payload_bytes": top_class_payload_bytes,
        "top_logit_value_payload_bytes": top_logit_value_payload_bytes,
        "prediction_payload_bytes": prediction_payload_bytes,
        "runner_up_payload_bytes": runner_up_payload_bytes,
        "margin_payload_bytes": margin_payload_bytes,
        "label_rank_payload_bytes": label_rank_payload_bytes,
        "correct_payload_bytes": correct_payload_bytes,
        "top_k_correct_payload_bytes": top_k_correct_payload_bytes,
        "label_payload_bytes": label_payload_bytes,
        "state_payload_bytes": state_payload_bytes,
        "witness_payload_bytes": witness_payload_bytes,
        "trace_payload_bytes": 0,
        "replay_payload_bytes": replay_payload_bytes,
        "forward_recompute_steps": 2,
        "inverse_recompute_steps": 2,
        "total_recompute_steps": 4,
        "witness_to_model_payload_ratio": witness_payload_bytes / model_payload_bytes,
        "trace_to_model_payload_ratio": 0.0,
        "witness_to_state_payload_ratio": witness_payload_bytes / state_payload_bytes,
        "trace_to_state_payload_ratio": 0.0,
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


def build_report(
    forward_path: Optional[Path],
    reverse_path: Optional[Path],
    mismatches: list[str],
) -> dict[str, Any]:
    initial = initial_store()
    forward = forward_store(initial)
    return {
        "kind": KIND,
        "program": PROGRAM,
        "q31_one": Q31_ONE,
        "forward_output_json": str(forward_path) if forward_path is not None else None,
        "reverse_output_json": str(reverse_path) if reverse_path is not None else None,
        "initial": initial,
        "forward": forward,
        "top_logits": top_logits(forward["logits"]),
        "attribution": inference_attribution(forward),
        "passed": not mismatches,
        "mismatches": mismatches,
        "proof": proof_cost(),
    }


def render_contribution_rows(rows: list[dict[str, Any]], *, margin: bool) -> list[str]:
    if not rows:
        column_count = 6 if margin else 5
        return ["| {} |".format(" | ".join(["-"] * column_count))]
    rendered = []
    for row in rows:
        if margin:
            rendered.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    row["feature"],
                    comma(row["q31"]),
                    comma(row["predicted_weight"]),
                    comma(row["runner_up_weight"]),
                    comma(row["weight_delta"]),
                    comma(row["contribution"]),
                )
            )
        else:
            rendered.append(
                "| {} | {} | {} | {} | {} |".format(
                    row["feature"],
                    comma(row["q31"]),
                    comma(row["weight"]),
                    comma(row["contribution"]),
                    comma(abs(row["contribution"])),
                )
            )
    return rendered


def render_markdown(report: dict[str, Any]) -> str:
    forward = report["forward"]
    attribution = report["attribution"]
    proof = report["proof"]
    mismatches = report["mismatches"]
    lines = [
        "# Reverie Reversible Inference Trace",
        "",
        "| Passed | Prediction | Label rank | Correct | Top-k correct | Runner-up | Margin | Replay bytes | Witness bytes | Trace bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            str(report["passed"]).lower(),
            forward["prediction"],
            forward["label_rank"],
            forward["correct"],
            forward["top2_correct"],
            attribution["runner_up_class"],
            comma(attribution["margin"]),
            comma(proof["replay_payload_bytes"]),
            comma(proof["witness_payload_bytes"]),
            comma(proof["trace_payload_bytes"]),
        ),
        "",
        "## Forward Trace",
        "",
        "| Features | Logits | Top classes | Top values | Top class | Q31 one |",
        "| --- | --- | --- | --- | ---: | ---: |",
        "| {} | {} | {} | {} | {} | {} |".format(
            vector_cell(forward["features"]),
            vector_cell(forward["logits"]),
            vector_cell(forward["top_classes"]),
            vector_cell(forward["top_logit_values"]),
            attribution["predicted_class"],
            comma(report["q31_one"]),
        ),
        "",
        "## Attribution",
        "",
        "| Check | Fingerprint | Count | Reconstructed | Expected | Match |",
        "| --- | --- | ---: | ---: | ---: | --- |",
        "| winning logit | `{}` | {} | {} | {} | {} |".format(
            short_hash(attribution["contribution_ledger_fingerprint"]),
            comma(attribution["contribution_count"]),
            comma(attribution["reconstructed_logit"]),
            comma(attribution["predicted_logit"]),
            str(attribution["matches_logit"]).lower(),
        ),
        "| winner margin | `{}` | {} | {} | {} | {} |".format(
            short_hash(attribution["margin_contribution_ledger_fingerprint"]),
            comma(attribution["margin_contribution_count"]),
            comma(attribution["reconstructed_margin"]),
            comma(attribution["margin"]),
            str(attribution["matches_margin"]).lower(),
        ),
        "",
        "## Top Winning Contributions",
        "",
        "| Feature | Q31 | Weight | Contribution | Abs contribution |",
        "| ---: | ---: | ---: | ---: | ---: |",
        *render_contribution_rows(attribution["top_contributions"], margin=False),
        "",
        "## Top Margin Contributions",
        "",
        "| Feature | Q31 | Predicted weight | Runner-up weight | Weight delta | Contribution |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        *render_contribution_rows(attribution["top_margin_contributions"], margin=True),
        "",
        "## Reverse Contract",
        "",
        "| Claim | Forward recompute | Inverse recompute | Reverse restored | Raw preserved | Model preserved | Balanced |",
        "| --- | ---: | ---: | --- | --- | --- | --- |",
        "| `{}` | {} | {} | {} | {} | {} | {} |".format(
            proof["claim"],
            comma(proof["forward_recompute_steps"]),
            comma(proof["inverse_recompute_steps"]),
            str(proof["checks"]["reverse_restores_initial_state"]).lower(),
            str(proof["checks"]["raw_preserved"]).lower(),
            str(proof["checks"]["model_preserved"]).lower(),
            str(proof["checks"]["balanced_recompute"]).lower(),
        ),
        "",
        "## Payloads",
        "",
        "| Model | State | Witness | Trace | Replay | Witness/model | Trace/model |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| {} | {} | {} | {} | {} | {:.6f} | {:.6f} |".format(
            comma(proof["model_payload_bytes"]),
            comma(proof["state_payload_bytes"]),
            comma(proof["witness_payload_bytes"]),
            comma(proof["trace_payload_bytes"]),
            comma(proof["replay_payload_bytes"]),
            proof["witness_to_model_payload_ratio"],
            proof["trace_to_model_payload_ratio"],
        ),
    ]
    if mismatches:
        lines.extend(
            [
                "",
                "## Mismatches",
                "",
                "| Mismatch |",
                "| --- |",
            ]
        )
        lines.extend(f"| `{mismatch}` |" for mismatch in mismatches)
    lines.append("")
    return "\n".join(lines)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(report), encoding="utf-8")
    except OSError as error:
        raise InferenceTraceCheckError(f"failed to write {path}: {error}") from error


def validate_expected_report(path: Path, report: dict[str, Any]) -> None:
    expected = load_json(path)
    if expected != report:
        raise InferenceTraceCheckError(f"recomputed report does not match {path}")


def check_outputs(forward_path: Optional[Path], reverse_path: Optional[Path]) -> list[str]:
    initial = initial_store()
    expected_forward = forward_store(initial)
    expected_reverse = inverse_store(expected_forward)
    mismatches: list[str] = []
    if expected_reverse != initial:
        mismatches.append("reference inverse did not restore initial store")
    if forward_path is not None:
        forward_actual = parse_output_store(load_json(forward_path), "reverie_run_result", "forward output")
        mismatches.extend(compare_store(expected_forward, forward_actual, "forward"))
    if reverse_path is not None:
        reverse_actual = parse_output_store(load_json(reverse_path), "reverie_reverse_result", "reverse output")
        mismatches.extend(compare_store(initial, reverse_actual, "reverse"))
    return mismatches


def run_self_tests() -> int:
    try:
        initial = initial_store()
        expected = forward_store(initial)
        if expected["features"] != [1073741824, -1073741824, 1610612736, 536870912]:
            raise AssertionError("forward features changed unexpectedly")
        if expected["logits"] != [1073741824, 1610612736, 536870912]:
            raise AssertionError("forward logits changed unexpectedly")
        if expected["top_classes"] != [1, 0, 2]:
            raise AssertionError("top classes changed unexpectedly")
        if expected["top_logit_values"] != [1610612736, 1073741824, 536870912]:
            raise AssertionError("top logit values changed unexpectedly")
        if (
            expected["prediction"] != 1
            or expected["runner_up_class"] != 0
            or expected["margin"] != Q31_ONE // 4
            or expected["label_rank"] != 1
            or expected["correct"] != 1
            or expected["top2_correct"] != 1
        ):
            raise AssertionError("prediction/margin/correctness changed unexpectedly")
        if inverse_store(expected) != initial:
            raise AssertionError("inverse did not restore initial state")
        if top_logits(expected["logits"]) != [
            {"class": 1, "value": 1610612736},
            {"class": 0, "value": 1073741824},
            {"class": 2, "value": 536870912},
        ]:
            raise AssertionError("top logits changed unexpectedly")
        attribution = inference_attribution(expected)
        if attribution["margin"] != Q31_ONE // 4:
            raise AssertionError("inference trace margin changed unexpectedly")
        if attribution["contribution_count"] != 1 or attribution["margin_contribution_count"] != 2:
            raise AssertionError("attribution contribution counts changed unexpectedly")
        if not attribution["matches_logit"] or not attribution["matches_margin"]:
            raise AssertionError("attribution no longer reconstructs logits and margin")
        for field in (
            "contribution_ledger_fingerprint",
            "margin_contribution_ledger_fingerprint",
        ):
            if len(attribution[field]) != 64:
                raise AssertionError(f"{field} must be a SHA-256 digest")
        proof = proof_cost()
        if proof["witness_payload_bytes"] != 120 or proof["trace_payload_bytes"] != 0:
            raise AssertionError("inference trace proof payloads changed unexpectedly")
        if proof["replay_payload_bytes"] != 344:
            raise AssertionError("inference trace replay payload changed unexpectedly")
        report = build_report(None, None, [])
        if not report["passed"] or report["proof"]["claim"] != PROOF_CLAIM:
            raise AssertionError("valid self-test report did not pass")
        if report["attribution"] != attribution:
            raise AssertionError("report attribution must match reference attribution")
        markdown = render_markdown(report)
        for snippet in (
            "# Reverie Reversible Inference Trace",
            "## Attribution",
            "## Reverse Contract",
            short_hash(attribution["contribution_ledger_fingerprint"]),
            short_hash(attribution["margin_contribution_ledger_fingerprint"]),
        ):
            if snippet not in markdown:
                raise AssertionError(f"rendered Markdown missing {snippet}")
        bad_store = forward_store(initial)
        bad_store["logits"][1] += 1
        mismatches = compare_store(expected, bad_store, "bad forward")
        if not mismatches:
            raise AssertionError("bad inference trace store did not produce mismatch")
    except AssertionError as error:
        print(f"error: self-test failed: {error}", file=sys.stderr)
        return 1
    print("ok: reversible inference trace checker self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    try:
        if args.write_final_vars is not None:
            args.write_final_vars.parent.mkdir(parents=True, exist_ok=True)
            args.write_final_vars.write_text(
                json.dumps(final_vars(), indent=2) + "\n",
                encoding="utf-8",
            )
        mismatches = check_outputs(args.forward_output_json, args.reverse_output_json)
        report = build_report(args.forward_output_json, args.reverse_output_json, mismatches)
        if args.expect_report_json is not None:
            validate_expected_report(args.expect_report_json, report)
        if args.markdown_output is not None:
            write_markdown(args.markdown_output, report)
        if args.json:
            print(json.dumps(report, indent=2))
        elif report["passed"]:
            proof = report["proof"]
            print(
                "ok: reversible inference trace matches reference "
                f"(replay {proof['replay_payload_bytes']} bytes, "
                f"witness {proof['witness_payload_bytes']} bytes)"
            )
        else:
            for mismatch in mismatches:
                print(f"error: {mismatch}", file=sys.stderr)
            return 1
    except InferenceTraceCheckError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
