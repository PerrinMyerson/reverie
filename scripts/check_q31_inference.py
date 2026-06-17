#!/usr/bin/env python3
"""Reference-check Reverie's MNIST-shaped Q31 linear inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional


IMAGE_PIXELS = 784
DIGITS = 10
Q31_ONE = 1 << 31
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
U64_MOD = 1 << 64


class InferenceCheckError(ValueError):
    """Raised when the model, sample, or expected result is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference-check Q31 linear inference for imported Reverie models."
    )
    parser.add_argument("--model", type=Path, help="Q31 model JSON or signed model bundle.")
    parser.add_argument(
        "--sample",
        type=Path,
        help="Sample JSON with image_u8/label, or a sample-set JSON with samples.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Select this sample when --sample is a sample-set JSON or sample array.",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Evaluate every sample when --sample is a sample-set JSON or sample array.",
    )
    parser.add_argument(
        "--report-limit",
        type=int,
        default=10,
        help="Number of ranked rows to include in all-samples summary views.",
    )
    parser.add_argument(
        "--expect-prediction",
        type=int,
        help="Require this predicted digit.",
    )
    parser.add_argument(
        "--expect-correct",
        choices=("true", "false"),
        help="Require whether prediction equals the sample label.",
    )
    parser.add_argument(
        "--expect-logits",
        help="JSON array of 10 expected Q31 logits.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable report.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Write a human-readable prediction explanation Markdown report.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent Q31 inference checker self-tests and exit.",
    )
    return parser.parse_args()


def wrap_i64(value: int) -> int:
    value %= U64_MOD
    if value >= (1 << 63):
        value -= U64_MOD
    return value


def checked_i64(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InferenceCheckError(f"{label} must be a signed integer")
    if not I64_MIN <= value <= I64_MAX:
        raise InferenceCheckError(f"{label} is outside signed i64 range")
    return value


def fixed_mul_q31(left: int, right: int) -> int:
    return wrap_i64((left * right) >> 31)


def pixel_to_q31(pixel: int) -> int:
    return (pixel * Q31_ONE) // 255


def wrapping_sum(values: list[int]) -> int:
    acc = 0
    for value in values:
        acc = wrap_i64(acc + value)
    return acc


def argmax_first(values: list[int]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def top_logits(logits: list[int]) -> list[dict[str, int]]:
    ranked = [
        {"digit": digit, "value": value}
        for digit, value in enumerate(logits)
    ]
    ranked.sort(key=lambda item: (-item["value"], item["digit"]))
    return ranked[:3]


def contribution_sort_key(item: dict[str, int]) -> tuple[int, int]:
    return (-abs(item["contribution"]), item["pixel"])


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def contribution_ledger_fingerprint(predicted_digit: int, rows: list[dict[str, int]]) -> str:
    return json_sha256(
        {
            "schema": "q31_linear_inference_contribution_ledger_v1",
            "predicted_digit": predicted_digit,
            "rows": rows,
        }
    )


def margin_contribution_ledger_fingerprint(
    predicted_digit: int,
    runner_up_digit: int,
    rows: list[dict[str, int]],
) -> str:
    return json_sha256(
        {
            "schema": "q31_linear_inference_margin_contribution_ledger_v1",
            "predicted_digit": predicted_digit,
            "runner_up_digit": runner_up_digit,
            "rows": rows,
        }
    )


def inference_attribution(
    model: dict[str, Any],
    sample: dict[str, Any],
    logits: list[int],
    ranked_logits: list[dict[str, int]],
) -> dict[str, Any]:
    predicted_digit = ranked_logits[0]["digit"] if ranked_logits else 0
    runner_up_digit = ranked_logits[1]["digit"] if len(ranked_logits) > 1 else predicted_digit
    predicted_logit = logits[predicted_digit]
    runner_up_logit = logits[runner_up_digit]
    bias = model["bias"][predicted_digit]
    runner_up_bias = model["bias"][runner_up_digit]
    margin_bias = wrap_i64(bias - runner_up_bias)
    contribution_sum = 0
    margin_contribution_sum = 0
    reconstructed_logit = bias
    reconstructed_margin = margin_bias
    contributions = []
    margin_contributions = []

    for pixel_index, pixel in enumerate(sample["image_u8"]):
        if pixel == 0:
            continue
        q31 = pixel_to_q31(pixel)
        weight = model["weights"][pixel_index][predicted_digit]
        runner_up_weight = model["weights"][pixel_index][runner_up_digit]
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
                    "pixel": pixel_index,
                    "u8": pixel,
                    "q31": q31,
                    "weight": weight,
                    "contribution": contribution,
                }
            )
        if margin_contribution != 0:
            margin_contributions.append(
                {
                    "pixel": pixel_index,
                    "u8": pixel,
                    "q31": q31,
                    "predicted_weight": weight,
                    "runner_up_weight": runner_up_weight,
                    "weight_delta": weight_delta,
                    "contribution": margin_contribution,
                }
            )

    contribution_fingerprint = contribution_ledger_fingerprint(predicted_digit, contributions)
    margin_contribution_fingerprint = margin_contribution_ledger_fingerprint(
        predicted_digit,
        runner_up_digit,
        margin_contributions,
    )
    top_contributions = contributions.copy()
    top_margin_contributions = margin_contributions.copy()
    top_contributions.sort(key=contribution_sort_key)
    top_margin_contributions.sort(key=contribution_sort_key)
    margin = wrap_i64(predicted_logit - runner_up_logit)
    return {
        "predicted_digit": predicted_digit,
        "runner_up_digit": runner_up_digit,
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
        "top_contribution_count": len(top_contributions[:10]),
        "top_margin_contribution_count": len(top_margin_contributions[:10]),
        "contribution_ledger_fingerprint": contribution_fingerprint,
        "margin_contribution_ledger_fingerprint": margin_contribution_fingerprint,
        "top_contributions": top_contributions[:10],
        "top_margin_contributions": top_margin_contributions[:10],
    }


def json_field(value: Any, name: str, context: str) -> Any:
    if not isinstance(value, dict) or name not in value:
        raise InferenceCheckError(f"{context} missing `{name}`")
    return value[name]


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InferenceCheckError(f"failed to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise InferenceCheckError(f"failed to parse {path}: {error}") from error


def model_from_json(value: Any) -> dict[str, Any]:
    model = value.get("model", value) if isinstance(value, dict) else value
    weights = json_field(model, "weights", "model")
    bias = json_field(model, "bias", "model")
    if not isinstance(weights, list) or len(weights) != IMAGE_PIXELS:
        raise InferenceCheckError(f"model.weights must have {IMAGE_PIXELS} rows")
    checked_weights: list[list[int]] = []
    for row_index, row in enumerate(weights):
        if not isinstance(row, list) or len(row) != DIGITS:
            raise InferenceCheckError(
                f"model.weights[{row_index}] must have {DIGITS} entries"
            )
        checked_weights.append(
            [
                checked_i64(value, f"model.weights[{row_index}][{col_index}]")
                for col_index, value in enumerate(row)
            ]
        )
    if not isinstance(bias, list) or len(bias) != DIGITS:
        raise InferenceCheckError(f"model.bias must have {DIGITS} entries")
    return {
        "weights": checked_weights,
        "bias": [
            checked_i64(value, f"model.bias[{index}]")
            for index, value in enumerate(bias)
        ],
    }


def checked_nonnegative_index(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InferenceCheckError(f"{context} must be a non-negative integer")
    return value


def checked_sample_object(value: Any, context: str) -> dict[str, Any]:
    image = json_field(value, "image_u8", context)
    label = checked_i64(json_field(value, "label", context), f"{context}.label")
    if not 0 <= label < DIGITS:
        raise InferenceCheckError(f"{context}.label must be in 0..9")
    if not isinstance(image, list) or len(image) != IMAGE_PIXELS:
        raise InferenceCheckError(f"{context}.image_u8 must have {IMAGE_PIXELS} entries")
    checked_image = []
    for index, pixel in enumerate(image):
        if isinstance(pixel, bool) or not isinstance(pixel, int) or not 0 <= pixel <= 255:
            raise InferenceCheckError(f"{context}.image_u8[{index}] must be a byte")
        checked_image.append(pixel)

    source: dict[str, Any] = {}
    if isinstance(value, dict):
        kind = value.get("kind")
        if isinstance(kind, str):
            source["sample_kind"] = kind
        for field in ("audit_step", "source_sample_index"):
            if field in value:
                source[field] = checked_nonnegative_index(value[field], f"{context}.{field}")

    return {"image_u8": checked_image, "label": label, "source": source}


def select_sample(samples: Any, sample_index: int, context: str) -> tuple[Any, dict[str, Any]]:
    if sample_index < 0:
        raise InferenceCheckError("--sample-index must be a non-negative integer")
    if not isinstance(samples, list):
        raise InferenceCheckError(f"{context}.samples must be an array")
    if not samples:
        raise InferenceCheckError(f"{context}.samples must not be empty")
    if sample_index >= len(samples):
        raise InferenceCheckError(
            f"--sample-index {sample_index} is outside {context}.samples length {len(samples)}"
        )
    return samples[sample_index], {"sample_index": sample_index, "sample_count": len(samples)}


def sample_from_json(value: Any, sample_index: int = 0) -> dict[str, Any]:
    selected_metadata: dict[str, Any] = {}
    context = "sample"
    if isinstance(value, list):
        value, selected_metadata = select_sample(value, sample_index, "sample array")
        context = f"samples[{sample_index}]"
    elif isinstance(value, dict) and "samples" in value and "image_u8" not in value:
        kind = value.get("kind")
        if kind is not None and kind != "reverie_mnist_linear_q31_samples":
            raise InferenceCheckError(
                "sample set kind must be `reverie_mnist_linear_q31_samples`"
            )
        value, selected_metadata = select_sample(value["samples"], sample_index, "sample set")
        context = f"samples[{sample_index}]"
        selected_metadata["sample_set_kind"] = kind or "array"
    elif sample_index != 0:
        raise InferenceCheckError("--sample-index requires a sample-set or sample array")

    sample = checked_sample_object(value, context)
    sample["source"].update(selected_metadata)
    return sample


def samples_from_json(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        samples = value
        sample_set_kind = "array"
        context = "sample array"
    elif isinstance(value, dict) and "samples" in value and "image_u8" not in value:
        kind = value.get("kind")
        if kind is not None and kind != "reverie_mnist_linear_q31_samples":
            raise InferenceCheckError(
                "sample set kind must be `reverie_mnist_linear_q31_samples`"
            )
        samples = value["samples"]
        sample_set_kind = kind or "array"
        context = "sample set"
    else:
        sample = checked_sample_object(value, "sample")
        sample["source"].update({"sample_index": 0, "sample_count": 1})
        return [sample]

    if not isinstance(samples, list):
        raise InferenceCheckError(f"{context}.samples must be an array")
    if not samples:
        raise InferenceCheckError(f"{context}.samples must not be empty")
    checked_samples = []
    for index, raw_sample in enumerate(samples):
        sample = checked_sample_object(raw_sample, f"samples[{index}]")
        sample["source"].update(
            {
                "sample_index": index,
                "sample_count": len(samples),
                "sample_set_kind": sample_set_kind,
            }
        )
        checked_samples.append(sample)
    return checked_samples


def infer(model: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    image_q31 = [pixel_to_q31(pixel) for pixel in sample["image_u8"]]
    logits = model["bias"].copy()
    for pixel_index, pixel in enumerate(image_q31):
        if pixel == 0:
            continue
        for digit in range(DIGITS):
            product = fixed_mul_q31(pixel, model["weights"][pixel_index][digit])
            logits[digit] = wrap_i64(logits[digit] + product)
    prediction = argmax_first(logits)
    ranked_logits = top_logits(logits)
    attribution = inference_attribution(model, sample, logits, ranked_logits)
    return {
        "logits": logits,
        "top_logits": ranked_logits,
        "prediction": prediction,
        "correct": prediction == sample["label"],
        "label": sample["label"],
        "active_pixels": sum(1 for pixel in sample["image_u8"] if pixel != 0),
        "attribution": attribution,
    }


def parse_expected_logits(text: str) -> list[int]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise InferenceCheckError(f"--expect-logits must be a JSON array: {error}") from error
    if not isinstance(value, list) or len(value) != DIGITS:
        raise InferenceCheckError(f"--expect-logits must have {DIGITS} entries")
    return [
        checked_i64(logit, f"--expect-logits[{index}]")
        for index, logit in enumerate(value)
    ]


def validate_expectations(
    result: dict[str, Any],
    expected_prediction: Optional[int],
    expected_correct: Optional[str],
    expected_logits: Optional[str],
) -> list[str]:
    errors = []
    if expected_prediction is not None and result["prediction"] != expected_prediction:
        errors.append(
            f"prediction expected {expected_prediction}, found {result['prediction']}"
        )
    if expected_correct is not None:
        expected_bool = expected_correct == "true"
        if result["correct"] != expected_bool:
            errors.append(f"correct expected {expected_bool}, found {result['correct']}")
    if expected_logits is not None:
        parsed_logits = parse_expected_logits(expected_logits)
        if result["logits"] != parsed_logits:
            errors.append("logits did not match expected Q31 vector")
    return errors


def report_json(
    model_path: Path,
    sample_path: Path,
    sample: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "kind": "reverie_q31_linear_reference_inference",
        "model": str(model_path),
        "sample": str(sample_path),
        "prediction": result["prediction"],
        "correct": result["correct"],
        "label": result["label"],
        "logits": result["logits"],
        "top_logits": result["top_logits"],
        "attribution": result["attribution"],
        "active_pixels": result["active_pixels"],
        "q31_one": Q31_ONE,
    }
    if sample["source"]:
        report["sample_source"] = sample["source"]
    return report


def evaluation_row(index: int, sample: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "index": index,
        "label": result["label"],
        "prediction": result["prediction"],
        "correct": result["correct"],
        "margin": result["attribution"]["margin"],
        "runner_up_digit": result["attribution"]["runner_up_digit"],
        "active_pixels": result["active_pixels"],
    }
    if sample["source"]:
        row["sample_source"] = sample["source"]
    return row


def accuracy_percent(correct: int, samples: int) -> float:
    return (100.0 * correct / samples) if samples else 0.0


def label_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for label in range(DIGITS):
        matching = [row for row in rows if row["label"] == label]
        correct = sum(1 for row in matching if row["correct"])
        lowest = min(matching, key=lambda row: (row["margin"], row["index"]), default=None)
        summaries.append(
            {
                "label": label,
                "samples": len(matching),
                "correct": correct,
                "incorrect": len(matching) - correct,
                "accuracy_percent": accuracy_percent(correct, len(matching)),
                "lowest_margin_index": None if lowest is None else lowest["index"],
                "lowest_margin": None if lowest is None else lowest["margin"],
            }
        )
    return summaries


def evaluate_samples(
    model: dict[str, Any],
    samples: list[dict[str, Any]],
    report_limit: int,
) -> dict[str, Any]:
    if report_limit < 0:
        raise InferenceCheckError("--report-limit must be a non-negative integer")
    rows = [
        evaluation_row(index, sample, infer(model, sample))
        for index, sample in enumerate(samples)
    ]
    correct = sum(1 for row in rows if row["correct"])
    lowest = min(rows, key=lambda row: (row["margin"], row["index"]), default=None)
    low_margin = sorted(rows, key=lambda row: (row["margin"], row["index"]))[:report_limit]
    incorrect = sorted(
        [row for row in rows if not row["correct"]],
        key=lambda row: (row["margin"], row["index"]),
    )[:report_limit]
    return {
        "summary": {
            "samples": len(rows),
            "correct": correct,
            "incorrect": len(rows) - correct,
            "accuracy_percent": accuracy_percent(correct, len(rows)),
            "lowest_margin_index": None if lowest is None else lowest["index"],
            "lowest_margin": None if lowest is None else lowest["margin"],
        },
        "by_label": label_summaries(rows),
        "rows": rows,
        "top_low_margin": low_margin,
        "top_incorrect": incorrect,
    }


def evaluation_report_json(
    model_path: Path,
    sample_path: Path,
    evaluation: dict[str, Any],
    report_limit: int,
) -> dict[str, Any]:
    return {
        "kind": "reverie_q31_linear_reference_evaluation",
        "model": str(model_path),
        "sample": str(sample_path),
        "q31_one": Q31_ONE,
        "report_limit": report_limit,
        **evaluation,
    }


def comma(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else str(value)


def percent(value: Any) -> str:
    return f"{value:.2f}%" if isinstance(value, (int, float)) else "n/a"


def short_hash(value: Any) -> str:
    return value[:12] if isinstance(value, str) else "n/a"


def render_contribution_rows(rows: list[dict[str, Any]], *, margin: bool) -> list[str]:
    output = []
    for row in rows:
        if margin:
            output.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    row["pixel"],
                    row["u8"],
                    comma(row["q31"]),
                    comma(row["predicted_weight"]),
                    comma(row["runner_up_weight"]),
                    comma(row["weight_delta"]),
                    comma(row["contribution"]),
                )
            )
        else:
            output.append(
                "| {} | {} | {} | {} | {} |".format(
                    row["pixel"],
                    row["u8"],
                    comma(row["q31"]),
                    comma(row["weight"]),
                    comma(row["contribution"]),
                )
            )
    if not output:
        output.append(
            "| n/a | n/a | n/a | n/a | n/a |"
            if not margin
            else "| n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
        )
    return output


def render_inference_markdown(report: dict[str, Any]) -> str:
    attribution = report["attribution"]
    lines = [
        "# Reverie Q31 Inference Explanation",
        "",
        "| Prediction | Label | Correct | Runner-up | Margin | Active pixels |",
        "| ---: | ---: | --- | ---: | ---: | ---: |",
        "| {} | {} | {} | {} | {} | {} |".format(
            report["prediction"],
            report["label"],
            str(report["correct"]).lower(),
            attribution["runner_up_digit"],
            comma(attribution["margin"]),
            comma(report["active_pixels"]),
        ),
        "",
        "## Top Logits",
        "",
        "| Rank | Digit | Logit |",
        "| ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(report["top_logits"], start=1):
        lines.append(f"| {rank} | {row['digit']} | {comma(row['value'])} |")
    lines.extend(
        [
            "",
            "## Winning-Digit Contributions",
            "",
            "| Pixel | U8 | Q31 | Weight | Contribution |",
            "| ---: | ---: | ---: | ---: | ---: |",
            *render_contribution_rows(attribution["top_contributions"], margin=False),
            "",
            "## Margin Contributions",
            "",
            "| Pixel | U8 | Q31 | Predicted weight | Runner-up weight | Weight delta | Contribution |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *render_contribution_rows(attribution["top_margin_contributions"], margin=True),
            "",
            "## Ledgers",
            "",
            "| Ledger | Fingerprint | Rows | Reconstructs |",
            "| --- | --- | ---: | --- |",
            "| winning contribution | `{}` | {} | {} |".format(
                short_hash(attribution["contribution_ledger_fingerprint"]),
                comma(attribution["contribution_count"]),
                str(attribution["matches_logit"]).lower(),
            ),
            "| margin contribution | `{}` | {} | {} |".format(
                short_hash(attribution["margin_contribution_ledger_fingerprint"]),
                comma(attribution["margin_contribution_count"]),
                str(attribution["matches_margin"]).lower(),
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_evaluation_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Reverie Q31 Evaluation Summary",
        "",
        "| Samples | Correct | Incorrect | Accuracy | Lowest-margin row | Lowest margin |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        "| {} | {} | {} | {} | {} | {} |".format(
            comma(summary["samples"]),
            comma(summary["correct"]),
            comma(summary["incorrect"]),
            percent(summary["accuracy_percent"]),
            "n/a"
            if summary["lowest_margin_index"] is None
            else comma(summary["lowest_margin_index"]),
            "n/a" if summary["lowest_margin"] is None else comma(summary["lowest_margin"]),
        ),
        "",
        "## Lowest-Margin Rows",
        "",
        "| Row | Label | Prediction | Correct | Runner-up | Margin | Active pixels |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in report["top_low_margin"]:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                row["index"],
                row["label"],
                row["prediction"],
                str(row["correct"]).lower(),
                row["runner_up_digit"],
                comma(row["margin"]),
                comma(row["active_pixels"]),
            )
        )
    lines.extend(
        [
            "",
            "## Incorrect Rows",
            "",
            "| Row | Label | Prediction | Margin |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    if report["top_incorrect"]:
        for row in report["top_incorrect"]:
            lines.append(
                "| {} | {} | {} | {} |".format(
                    row["index"],
                    row["label"],
                    row["prediction"],
                    comma(row["margin"]),
                )
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a |")
    lines.append("")
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    if report["kind"] == "reverie_q31_linear_reference_inference":
        return render_inference_markdown(report)
    if report["kind"] == "reverie_q31_linear_reference_evaluation":
        return render_evaluation_markdown(report)
    raise InferenceCheckError(f"unsupported report kind for Markdown: {report['kind']}")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")


def run_self_tests() -> int:
    weights = [[0 for _ in range(DIGITS)] for _ in range(IMAGE_PIXELS)]
    bias = [0 for _ in range(DIGITS)]
    weights[0][3] = Q31_ONE
    weights[1][4] = Q31_ONE // 2
    sample = {"image_u8": [255, 255] + [0] * (IMAGE_PIXELS - 2), "label": 3}
    result = infer({"weights": weights, "bias": bias}, sample_from_json(sample))
    if result["logits"][3] != Q31_ONE or result["logits"][4] != Q31_ONE // 2:
        raise AssertionError("self-test logits did not match Q31 products")
    if result["prediction"] != 3 or result["correct"] is not True:
        raise AssertionError("self-test prediction/correctness mismatch")
    if result["top_logits"][0] != {"digit": 3, "value": Q31_ONE}:
        raise AssertionError("self-test top logits did not rank the winning digit")
    attribution = result["attribution"]
    if attribution["runner_up_digit"] != 4 or attribution["margin"] != Q31_ONE // 2:
        raise AssertionError("self-test attribution margin did not match logits")
    if not attribution["matches_logit"] or not attribution["matches_margin"]:
        raise AssertionError("self-test attribution reconstruction failed")
    if attribution["top_contributions"][0]["pixel"] != 0:
        raise AssertionError("self-test top contribution did not identify pixel 0")
    if [item["pixel"] for item in attribution["top_margin_contributions"][:2]] != [0, 1]:
        raise AssertionError("self-test margin contributions were not deterministic")
    sample_bundle = {
        "kind": "reverie_mnist_linear_q31_samples",
        "samples": [
            {"image_u8": [0] * IMAGE_PIXELS, "label": 0, "audit_step": 0},
            {
                "kind": "reverie_mnist_linear_q31_sample",
                "image_u8": [255, 255] + [0] * (IMAGE_PIXELS - 2),
                "label": 3,
                "audit_step": 7,
                "source_sample_index": 11,
            },
        ],
    }
    selected = sample_from_json(sample_bundle, sample_index=1)
    if selected["label"] != 3 or selected["source"]["sample_index"] != 1:
        raise AssertionError("sample-set selection did not preserve selected sample metadata")
    if selected["source"]["audit_step"] != 7 or selected["source"]["source_sample_index"] != 11:
        raise AssertionError("sample-set selection did not preserve lineage metadata")
    selected_samples = samples_from_json(sample_bundle)
    if len(selected_samples) != 2 or selected_samples[1]["source"]["sample_index"] != 1:
        raise AssertionError("sample-set all-samples selection did not preserve row metadata")
    evaluation = evaluate_samples({"weights": weights, "bias": bias}, selected_samples, 1)
    summary = evaluation["summary"]
    if summary["samples"] != 2 or summary["correct"] != 2:
        raise AssertionError("all-samples evaluation summary did not count correct rows")
    if summary["lowest_margin_index"] != 0 or summary["lowest_margin"] != 0:
        raise AssertionError("all-samples evaluation did not report lowest margin row")
    if len(evaluation["top_low_margin"]) != 1 or evaluation["top_low_margin"][0]["index"] != 0:
        raise AssertionError("all-samples low-margin ranking did not honor report limit")
    inference_report = report_json(Path("model.json"), Path("samples.json"), selected, result)
    inference_markdown = render_markdown(inference_report)
    for snippet in (
        "# Reverie Q31 Inference Explanation",
        "## Winning-Digit Contributions",
        "## Margin Contributions",
        "winning contribution",
        "margin contribution",
    ):
        if snippet not in inference_markdown:
            raise AssertionError(f"inference Markdown missing `{snippet}`")
    evaluation_markdown = render_markdown(
        evaluation_report_json(Path("model.json"), Path("samples.json"), evaluation, 1)
    )
    for snippet in (
        "# Reverie Q31 Evaluation Summary",
        "## Lowest-Margin Rows",
        "## Incorrect Rows",
    ):
        if snippet not in evaluation_markdown:
            raise AssertionError(f"evaluation Markdown missing `{snippet}`")
    try:
        sample_from_json(sample_bundle, sample_index=2)
    except InferenceCheckError as error:
        if "--sample-index 2" not in str(error):
            raise AssertionError(f"wrong out-of-range sample-index failure: {error}") from error
    else:
        raise AssertionError("out-of-range sample-set selection did not fail")

    tied = infer(
        {"weights": [[0 for _ in range(DIGITS)] for _ in range(IMAGE_PIXELS)], "bias": [7] * DIGITS},
        sample_from_json({"image_u8": [0] * IMAGE_PIXELS, "label": 0}),
    )
    if tied["prediction"] != 0:
        raise AssertionError("argmax tie did not choose the first digit")

    wrapped = wrapping_sum([I64_MAX, 1])
    if wrapped != I64_MIN:
        raise AssertionError("wrapping_sum did not wrap signed i64 addition")

    print("ok: Q31 inference reference checker self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()

    if args.model is None or args.sample is None:
        print("error: --model and --sample are required", file=sys.stderr)
        return 2
    if args.all_samples and args.sample_index != 0:
        print("error: --sample-index cannot be used with --all-samples", file=sys.stderr)
        return 2
    if args.all_samples and (
        args.expect_prediction is not None
        or args.expect_correct is not None
        or args.expect_logits is not None
    ):
        print("error: expectations require single-sample mode", file=sys.stderr)
        return 2

    try:
        model = model_from_json(load_json(args.model))
        sample_json = load_json(args.sample)
        if args.all_samples:
            evaluation = evaluate_samples(
                model,
                samples_from_json(sample_json),
                args.report_limit,
            )
            expectation_errors = []
        else:
            sample = sample_from_json(sample_json, sample_index=args.sample_index)
            result = infer(model, sample)
            expectation_errors = validate_expectations(
                result,
                args.expect_prediction,
                args.expect_correct,
                args.expect_logits,
            )
    except InferenceCheckError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if expectation_errors:
        for error in expectation_errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    if args.all_samples:
        report = evaluation_report_json(
            args.model,
            args.sample,
            evaluation,
            args.report_limit,
        )
        if args.markdown_output is not None:
            write_markdown(args.markdown_output, report)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            summary = evaluation["summary"]
            print(
                "ok: samples={} correct={} incorrect={} accuracy={:.2f}% lowest_margin_index={} lowest_margin={}".format(
                    summary["samples"],
                    summary["correct"],
                    summary["incorrect"],
                    summary["accuracy_percent"],
                    summary["lowest_margin_index"],
                    summary["lowest_margin"],
                )
            )
    else:
        report = report_json(args.model, args.sample, sample, result)
        if args.markdown_output is not None:
            write_markdown(args.markdown_output, report)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(
                "ok: prediction={} correct={} label={} active_pixels={} margin={} attribution_matches={}".format(
                    result["prediction"],
                    str(result["correct"]).lower(),
                    result["label"],
                    result["active_pixels"],
                    result["attribution"]["margin"],
                    str(
                        result["attribution"]["matches_logit"]
                        and result["attribution"]["matches_margin"]
                    ).lower(),
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
