#!/usr/bin/env python3
"""Render a self-contained visual report for a standalone Reverie MNIST classifier."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional


Q31_ONE = 1 << 31
U64_MOD = 1 << 64
DEFAULT_CLASSIFIER = Path("target/mnist-rev-only-current-classifier.rev")
DEFAULT_INFERENCE = Path("target/mnist-rev-only-current-inference-bundle.json")
DEFAULT_RUN = Path("target/mnist-rev-only-current-run.json")
DEFAULT_ROUNDTRIP = Path("target/mnist-rev-only-current-roundtrip.json")
DEFAULT_VERIFICATION = Path("target/mnist-rev-only-current-roundtrip-verification.json")
DEFAULT_OUTPUT = Path("target/mnist-rev-only-current-visualizer.html")


class VisualizerError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML visualizer for a Reverie MNIST .rev classifier."
    )
    parser.add_argument("--classifier-rev", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--inference-bundle", type=Path, default=DEFAULT_INFERENCE)
    parser.add_argument("--run-json", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--roundtrip-json", type=Path, default=DEFAULT_ROUNDTRIP)
    parser.add_argument("--verification-json", type=Path, default=DEFAULT_VERIFICATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true", help="Run renderer self-tests and exit.")
    return parser.parse_args()


def load_json(path: Path, label: str, *, required: bool = True) -> Optional[dict[str, Any]]:
    if not path.exists():
        if required:
            raise VisualizerError(f"{label} is missing: {path}")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VisualizerError(f"failed to read {label} {path}: {error}") from error
    if not isinstance(data, dict):
        raise VisualizerError(f"{label} must be a JSON object: {path}")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise VisualizerError(f"failed to hash {path}: {error}") from error
    return digest.hexdigest()


def read_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise VisualizerError(f"failed to read {label} {path}: {error}") from error


def wrap_i64(value: int) -> int:
    value %= U64_MOD
    if value >= (1 << 63):
        value -= U64_MOD
    return value


def fixed_mul_q31(left: int, right: int) -> int:
    return wrap_i64((left * right) >> 31)


def pixel_to_q31(pixel: int) -> int:
    return (pixel * Q31_ONE) // 255


def q31_to_u8(value: int) -> int:
    if value <= 0:
        return 0
    return max(0, min(255, round(value * 255 / Q31_ONE)))


def number_list(value: Any, label: str, *, expected_len: Optional[int] = None) -> list[int]:
    if not isinstance(value, list):
        raise VisualizerError(f"{label} must be a list")
    out = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise VisualizerError(f"{label}[{index}] must be an integer")
        out.append(item)
    if expected_len is not None and len(out) != expected_len:
        raise VisualizerError(f"{label} must have {expected_len} entries, found {len(out)}")
    return out


def matrix(value: Any, label: str, rows: int, cols: int) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != rows:
        raise VisualizerError(f"{label} must have {rows} rows")
    return [number_list(row, f"{label}[{index}]", expected_len=cols) for index, row in enumerate(value)]


def get_path(root: Any, keys: tuple[str, ...]) -> Any:
    current = root
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def source_excerpt(source: str) -> list[dict[str, Any]]:
    lines = source.splitlines()
    procedure_index = next(
        (index for index, line in enumerate(lines) if line.strip().startswith("procedure main")),
        max(0, len(lines) - 12),
    )
    selected_indexes: list[int] = []
    selected_indexes.extend(range(min(4, len(lines))))
    selected_indexes.extend(range(max(0, procedure_index - 1), min(len(lines), procedure_index + 10)))
    selected_indexes = sorted(set(selected_indexes))
    return [
        {"line": index + 1, "text": lines[index]}
        for index in selected_indexes
    ]


def normalize_top_logits(items: Any) -> list[dict[str, int]]:
    if not isinstance(items, list):
        return []
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        digit = item.get("digit")
        value = item.get("logit", item.get("value"))
        if isinstance(digit, int) and not isinstance(digit, bool) and isinstance(value, int):
            normalized.append({"digit": digit, "value": value})
    return normalized


def normalize_contribution_rows(items: Any) -> list[dict[str, int]]:
    if not isinstance(items, list):
        return []
    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        row = {}
        for key in (
            "pixel",
            "u8",
            "q31",
            "weight",
            "predicted_weight",
            "runner_up_weight",
            "weight_delta",
            "contribution",
        ):
            value = item.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                row[key] = value
        if "pixel" in row and "contribution" in row:
            rows.append(row)
    return rows


def flatten_ints(value: Any) -> list[int]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        out: list[int] = []
        for item in value:
            out.extend(flatten_ints(item))
        return out
    return []


def describe_state_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {"kind": "missing", "display": "n/a"}
    if isinstance(value, bool):
        return {"kind": "bool", "display": "true" if value else "false", "nonzero": int(value)}
    if isinstance(value, int):
        return {"kind": "int", "display": str(value), "nonzero": int(value != 0)}
    if isinstance(value, list):
        flattened = flatten_ints(value)
        nonzero = len([item for item in flattened if item != 0])
        summary: dict[str, Any] = {
            "kind": "array",
            "cells": len(flattened),
            "nonzero": nonzero,
            "display": f"{len(flattened)} cells, {nonzero} nonzero",
        }
        if len(value) == 10 and all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            top_digit, top_value = max(enumerate(value), key=lambda item: (item[1], -item[0]))
            summary["top_digit"] = top_digit
            summary["top_value"] = top_value
            if nonzero:
                summary["display"] = f"10 cells, top {top_digit} = {top_value}"
        return summary
    return {"kind": type(value).__name__, "display": str(value)}


def state_delta_rows(
    baseline_store: dict[str, Any],
    forward_store: dict[str, Any],
    restored_store: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for name in ("logits", "prediction", "correct"):
        baseline = baseline_store.get(name)
        forward = forward_store.get(name)
        restored = restored_store.get(name)
        forward_changed = forward != baseline
        restored_matches = restored == baseline
        if forward_changed and restored_matches:
            reverse_result = "uncomputed"
        elif restored_matches:
            reverse_result = "unchanged"
        else:
            reverse_result = "mismatch"
        rows.append(
            {
                "name": name,
                "baseline": describe_state_value(baseline),
                "forward": describe_state_value(forward),
                "restored": describe_state_value(restored),
                "forward_changed": forward_changed,
                "restored_matches_baseline": restored_matches,
                "reverse_result": reverse_result,
            }
        )
    return rows


def normalize_witness_entries(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    entries = []
    for item in items:
        if not isinstance(item, dict):
            continue
        entry: dict[str, Any] = {}
        for key in ("name", "value_fingerprint"):
            if isinstance(item.get(key), str):
                entry[key] = item[key]
        for key in ("cells", "payload_bytes"):
            value = item.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                entry[key] = value
        if isinstance(item.get("present"), bool):
            entry["present"] = item["present"]
        if entry:
            entries.append(entry)
    return entries


def compute_maps(
    image_q31: list[int],
    weights: list[list[int]],
    prediction: int,
    runner_up: int,
) -> tuple[list[int], list[int]]:
    contribution = []
    margin = []
    for pixel, q31 in enumerate(image_q31):
        predicted_weight = weights[pixel][prediction]
        runner_up_weight = weights[pixel][runner_up]
        predicted_contribution = fixed_mul_q31(q31, predicted_weight)
        runner_contribution = fixed_mul_q31(q31, runner_up_weight)
        contribution.append(predicted_contribution)
        margin.append(wrap_i64(predicted_contribution - runner_contribution))
    return contribution, margin


def build_visualizer_data(
    *,
    classifier_rev: Path,
    inference_bundle: dict[str, Any],
    run_json: dict[str, Any],
    roundtrip_json: dict[str, Any],
    verification_json: dict[str, Any],
) -> dict[str, Any]:
    source = read_text(classifier_rev, "classifier source")
    source_hash = sha256_file(classifier_rev)
    store = first_dict(run_json.get("store"))
    report = first_dict(inference_bundle.get("report"))
    result = first_dict(inference_bundle.get("result"), report)
    sample = first_dict(inference_bundle.get("sample"))
    model = first_dict(inference_bundle.get("model"))
    attribution = first_dict(result.get("attribution"), report.get("attribution"))

    if isinstance(sample.get("image_u8"), list):
        image_u8 = number_list(sample["image_u8"], "sample.image_u8", expected_len=784)
        image_q31 = [pixel_to_q31(pixel) for pixel in image_u8]
    elif isinstance(store.get("image"), list):
        image_q31 = number_list(store["image"], "store.image", expected_len=784)
        image_u8 = [q31_to_u8(value) for value in image_q31]
    else:
        active = report.get("active_pixels", [])
        image_u8 = [0 for _ in range(784)]
        if isinstance(active, list):
            for row in active:
                if isinstance(row, dict) and isinstance(row.get("index"), int) and isinstance(row.get("u8"), int):
                    if 0 <= row["index"] < 784:
                        image_u8[row["index"]] = max(0, min(255, row["u8"]))
        image_q31 = [pixel_to_q31(pixel) for pixel in image_u8]

    weights = matrix(model.get("weights", store.get("weights")), "weights", 784, 10)
    bias = number_list(model.get("bias", store.get("bias")), "bias", expected_len=10)
    logits = number_list(result.get("logits", store.get("logits")), "logits", expected_len=10)

    prediction = result.get("prediction", store.get("prediction"))
    label = sample.get("label", report.get("label", store.get("label")))
    correct = result.get("correct", report.get("correct", store.get("correct")))
    if isinstance(correct, int) and not isinstance(correct, bool):
        correct = bool(correct)
    if not isinstance(prediction, int) or not isinstance(label, int) or not isinstance(correct, bool):
        raise VisualizerError("prediction, label, and correctness are required")

    top_logits = normalize_top_logits(result.get("top_logits", report.get("top_logits")))
    if top_logits:
        runner_up = top_logits[1]["digit"] if len(top_logits) > 1 else prediction
    else:
        ranked = sorted(enumerate(logits), key=lambda item: (-item[1], item[0]))
        top_logits = [{"digit": digit, "value": value} for digit, value in ranked[:3]]
        runner_up = top_logits[1]["digit"] if len(top_logits) > 1 else prediction

    contribution_map, margin_map = compute_maps(image_q31, weights, prediction, runner_up)
    roundtrip_payload = first_dict(roundtrip_json.get("payload"))
    baseline_store = first_dict(get_path(roundtrip_payload, ("baseline", "store")))
    forward_store = first_dict(get_path(roundtrip_payload, ("forward", "store")))
    restored_store = first_dict(get_path(roundtrip_payload, ("restored", "store")))
    roundtrip_check = first_dict(roundtrip_payload.get("check"))
    roundtrip_program = first_dict(roundtrip_payload.get("program"))
    verification_checks = first_dict(verification_json.get("checks"))

    program = first_dict(run_json.get("program"), roundtrip_payload.get("program"))
    ml_profile = first_dict(run_json.get("ml_profile"), roundtrip_payload.get("ml_profile"))
    memory = first_dict(result.get("memory"), report.get("memory"), inference_bundle.get("storage"))
    proof = first_dict(inference_bundle.get("proof"), result.get("proof"), report.get("proof"))
    tensor_shapes = first_dict(inference_bundle.get("tensor_shapes"))
    witness_proof = first_dict(run_json.get("witness_proof"))
    witness_payload = first_dict(witness_proof.get("payload"))

    return {
        "schema": "reverie_mnist_source_visualizer_v1",
        "title": "Reverie MNIST Source Visualizer",
        "summary": {
            "prediction": prediction,
            "label": label,
            "correct": correct,
            "runner_up": runner_up,
            "margin": attribution.get("margin", wrap_i64(logits[prediction] - logits[runner_up])),
            "audit_step": report.get("audit_step"),
            "sample_index": report.get("sample_index"),
            "active_pixels": len([value for value in image_u8 if value != 0]),
            "expected_prediction": store.get("expected_prediction"),
            "expected_correct": store.get("expected_correct"),
        },
        "image_u8": image_u8,
        "image_q31": image_q31,
        "model": {
            "weights": weights,
            "bias": bias,
            "input_shape": [28, 28],
            "builtin": "vecmat_q31",
        },
        "logits": logits,
        "bias": bias,
        "top_logits": top_logits,
        "contribution_map": contribution_map,
        "margin_map": margin_map,
        "top_contributions": normalize_contribution_rows(attribution.get("top_contributions")),
        "top_margin_contributions": normalize_contribution_rows(attribution.get("top_margin_contributions")),
        "proof": {
            "claim": proof.get("claim"),
            "arithmetic": proof.get("arithmetic"),
            "checks": proof.get("checks", {}),
            "witnesses": proof.get("witnesses", []),
            "fingerprints": proof.get("fingerprints", {}),
            "contribution_ledger": attribution.get("contribution_ledger_fingerprint"),
            "margin_ledger": attribution.get("margin_contribution_ledger_fingerprint"),
            "roundtrip_passed": roundtrip_json.get("passed"),
            "roundtrip_fingerprint": roundtrip_json.get("fingerprint"),
            "verification_passed": verification_json.get("passed"),
            "verification_checks": verification_checks,
            "witness_proof": witness_proof.get("fingerprint"),
        },
        "reversibility": {
            "program": {
                "direction": roundtrip_program.get("direction"),
                "engine": roundtrip_program.get("engine"),
                "legacy_janus": roundtrip_program.get("legacy_janus"),
            },
            "state_deltas": state_delta_rows(baseline_store, forward_store, restored_store),
            "check": roundtrip_check,
            "verification_checks": verification_checks,
            "witness": {
                "fingerprint": witness_proof.get("fingerprint"),
                "variables": witness_payload.get("variables"),
                "known_cells": witness_payload.get("known_cells"),
                "known_payload_bytes": witness_payload.get("known_payload_bytes"),
                "unknown_variables": witness_payload.get("unknown_variables", []),
                "entries": normalize_witness_entries(witness_payload.get("entries")),
            },
        },
        "memory": memory,
        "ml_profile": {
            "goal_fit": ml_profile.get("goal_fit"),
            "builtin_counts": ml_profile.get("builtin_counts", {}),
            "capabilities": ml_profile.get("capabilities", []),
            "replay_cost": ml_profile.get("replay_cost", {}),
            "tensor_entries": ml_profile.get("tensor_entries", []),
            "tensor_metrics": ml_profile.get("tensor_metrics", {}),
        },
        "roundtrip": {
            "baseline": {
                "prediction": baseline_store.get("prediction"),
                "correct": baseline_store.get("correct"),
                "logits": baseline_store.get("logits"),
            },
            "forward": {
                "prediction": forward_store.get("prediction"),
                "correct": forward_store.get("correct"),
                "logits": forward_store.get("logits"),
            },
            "restored": {
                "prediction": restored_store.get("prediction"),
                "correct": restored_store.get("correct"),
                "logits": restored_store.get("logits"),
            },
            "check": roundtrip_payload.get("check", {}),
        },
        "source": {
            "path": str(classifier_rev),
            "sha256": source_hash,
            "bytes": classifier_rev.stat().st_size,
            "line_count": len(source.splitlines()),
            "contains": {
                "image_tensor": "global image: tensor<int, 784>" in source,
                "weights_tensor": "global weights: tensor<int, 784, 10>" in source,
                "vecmat_q31": "vecmat_q31(image, weights)" in source,
                "prediction_assert": "assert prediction == expected_prediction" in source,
                "correct_assert": "assert correct == expected_correct" in source,
            },
            "excerpt": source_excerpt(source),
        },
        "tensor_shapes": tensor_shapes,
    }


def html_document(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    title = html.escape(str(data.get("title", "Reverie MNIST Source Visualizer")))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f4f6f8;
  --ink: #202124;
  --muted: #5d6675;
  --panel: #ffffff;
  --line: #d8dee8;
  --teal: #0f766e;
  --blue: #315f9f;
  --gold: #a66a05;
  --red: #b4434a;
  --green: #2f7d4f;
  --shadow: 0 16px 40px rgba(23, 32, 42, 0.08);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  letter-spacing: 0;
}}
button, input, select, textarea {{ font: inherit; }}
.shell {{ width: min(1440px, calc(100% - 32px)); margin: 0 auto; padding: 18px 0 32px; }}
.topbar {{
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 14px 0 18px; border-bottom: 1px solid var(--line);
}}
h1 {{ margin: 0; font-size: 28px; line-height: 1.05; font-weight: 760; }}
.eyebrow {{ margin: 0 0 4px; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
.status {{
  display: inline-grid; grid-template-columns: 10px auto; align-items: center; gap: 8px;
  border: 1px solid var(--line); background: var(--panel); border-radius: 999px;
  padding: 8px 12px; min-width: 116px; box-shadow: var(--shadow);
}}
.dot {{ width: 10px; height: 10px; border-radius: 50%; background: var(--green); }}
.dot.fail {{ background: var(--red); }}
.grid {{ display: grid; gap: 16px; }}
.summary {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 16px; margin-top: 18px; }}
.metric {{
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 14px; min-height: 92px; box-shadow: var(--shadow);
}}
.metric .label {{ color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
.metric .value {{ font-size: 28px; font-weight: 780; line-height: 1; overflow-wrap: anywhere; }}
.metric .sub {{ color: var(--muted); font-size: 12px; margin-top: 8px; }}
.main {{ grid-template-columns: minmax(320px, 0.95fr) minmax(420px, 1.45fr); margin-top: 16px; align-items: start; }}
.panel {{
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 16px; box-shadow: var(--shadow); min-width: 0;
}}
.panel h2 {{ margin: 0 0 12px; font-size: 18px; line-height: 1.2; }}
.panel h3 {{ margin: 16px 0 8px; font-size: 14px; color: var(--muted); }}
.draw-panel {{ margin-top: 16px; }}
.draw-grid {{ display: grid; grid-template-columns: minmax(280px, 0.72fr) minmax(380px, 1.28fr); gap: 16px; align-items: start; }}
.draw-board {{ display: grid; gap: 10px; min-width: 0; }}
.draw-canvas {{
  width: 100%; max-width: 420px; aspect-ratio: 1 / 1; border: 1px solid var(--line);
  border-radius: 8px; background: white; touch-action: none; cursor: crosshair;
}}
.draw-controls {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }}
.tool-button {{
  border: 1px solid var(--line); border-radius: 8px; background: #f8fafc; color: var(--ink);
  padding: 8px 10px; min-height: 36px; font-weight: 650; cursor: pointer;
}}
.tool-button:hover {{ border-color: var(--blue); }}
.brush-control, .toggle-control {{ display: inline-flex; align-items: center; gap: 8px; color: var(--muted); font-size: 13px; }}
.brush-control input[type="range"] {{ width: 130px; accent-color: var(--teal); }}
.toggle-control input {{ accent-color: var(--teal); }}
.guess-grid {{ display: grid; grid-template-columns: 150px minmax(0, 1fr); gap: 16px; align-items: start; }}
.guess-card {{
  border: 1px solid var(--line); border-radius: 8px; background: #f8fafc;
  padding: 14px; min-height: 150px;
}}
.guess-card .label {{ color: var(--muted); font-size: 12px; }}
.guess-card .guess {{ font-size: 54px; line-height: 0.95; font-weight: 820; margin: 8px 0; }}
.guess-card .detail {{ color: var(--muted); font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; }}
.preview-canvas {{
  width: 100%; max-width: 150px; aspect-ratio: 1 / 1; image-rendering: pixelated;
  border: 1px solid var(--line); background: #eef2f6; border-radius: 6px;
}}
.live-logits {{ min-width: 0; }}
.draw-rev-grid {{ display: grid; grid-template-columns: minmax(0, 0.85fr) minmax(0, 1.15fr); gap: 16px; margin-top: 14px; align-items: start; }}
.mini-flow {{ display: grid; grid-template-columns: minmax(0, 1fr) 40px minmax(0, 1fr) 40px minmax(0, 1fr); gap: 8px; align-items: stretch; }}
.mini-flow .flow-step {{ padding: 10px; }}
.mini-flow .headline {{ font-size: 15px; }}
.canvas-row {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
.canvas-block {{ min-width: 0; }}
.canvas-title {{ color: var(--muted); font-size: 12px; margin-bottom: 6px; }}
canvas.pixel {{
  width: 100%; aspect-ratio: 1 / 1; image-rendering: pixelated;
  border: 1px solid var(--line); background: #eef2f6; border-radius: 6px;
}}
.tooltip {{
  position: fixed; pointer-events: none; opacity: 0; transform: translate(10px, 10px);
  background: #202124; color: white; font-size: 12px; padding: 6px 8px; border-radius: 6px;
  z-index: 10; max-width: 240px;
}}
.logit-row {{ display: grid; grid-template-columns: 32px minmax(0, 1fr) 118px; gap: 10px; align-items: center; margin: 8px 0; }}
.digit {{
  width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center;
  border: 1px solid var(--line); font-weight: 760; background: #f0f4f8;
}}
.digit.predicted {{ color: white; background: var(--teal); border-color: var(--teal); }}
.digit.label {{ outline: 3px solid rgba(185, 130, 22, 0.32); }}
.bar-track {{ position: relative; height: 20px; background: #e8edf3; border-radius: 5px; overflow: hidden; }}
.bar {{ position: absolute; left: 0; top: 0; bottom: 0; background: var(--blue); min-width: 2px; }}
.bar.predicted {{ background: var(--teal); }}
.bar.runner {{ background: var(--gold); }}
.bar.negative {{ background: var(--red); }}
.logit-value {{ font-variant-numeric: tabular-nums; color: var(--muted); text-align: right; overflow-wrap: anywhere; }}
.split {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; margin-top: 16px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: right; font-variant-numeric: tabular-nums; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--muted); font-weight: 650; }}
.kv {{ display: grid; grid-template-columns: minmax(150px, 0.55fr) minmax(0, 1fr); gap: 8px 14px; font-size: 13px; }}
.kv .k {{ color: var(--muted); }}
.kv .v {{ overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.chip {{ border: 1px solid var(--line); border-radius: 999px; padding: 5px 8px; background: #f3f7fb; font-size: 12px; }}
.source {{
  margin: 0; padding: 12px; background: #17202a; color: #f5f7fa; border-radius: 6px;
  overflow: auto; max-height: 320px; font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
.line-no {{ color: #a9b0b8; display: inline-block; width: 44px; user-select: none; }}
.timeline {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
.state {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #f8fafc; min-width: 0; }}
.state strong {{ display: block; margin-bottom: 8px; }}
.section-head {{ display: flex; justify-content: space-between; align-items: start; gap: 16px; margin-bottom: 12px; }}
.section-head h2 {{ margin: 0; }}
.mode-pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 5px 9px; color: var(--muted); font-size: 12px; white-space: nowrap; }}
.reversible-panel {{ margin-top: 16px; }}
.flow {{ display: grid; grid-template-columns: minmax(0, 1fr) 54px minmax(0, 1fr) 54px minmax(0, 1fr); gap: 10px; align-items: stretch; }}
.flow-step {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #f8fafc; min-width: 0; }}
.flow-step .name {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
.flow-step .headline {{ font-size: 18px; font-weight: 760; margin: 6px 0; overflow-wrap: anywhere; }}
.flow-step .detail {{ color: var(--muted); font-size: 12px; line-height: 1.45; }}
.flow-op {{ display: grid; place-items: center; color: var(--blue); font-weight: 760; font-size: 12px; }}
.reversible-grid {{ display: grid; grid-template-columns: minmax(280px, 0.65fr) minmax(0, 1.35fr); gap: 16px; margin-top: 14px; align-items: start; }}
.diff-table td {{ vertical-align: top; }}
.result-pill {{ display: inline-block; border-radius: 999px; padding: 3px 8px; font-size: 12px; font-weight: 760; }}
.result-pill.ok-pill {{ color: var(--green); background: rgba(47, 125, 79, 0.1); }}
.result-pill.bad-pill {{ color: var(--red); background: rgba(180, 67, 74, 0.1); }}
.ok {{ color: var(--green); font-weight: 760; }}
.bad {{ color: var(--red); font-weight: 760; }}
.footer {{ color: var(--muted); font-size: 12px; margin-top: 18px; }}
@media (max-width: 1100px) {{
  .summary {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
  .main, .split, .draw-grid, .draw-rev-grid, .reversible-grid {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 720px) {{
  .shell {{ width: min(100% - 20px, 1440px); }}
  .topbar {{ align-items: flex-start; flex-direction: column; }}
  .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .guess-grid {{ grid-template-columns: 1fr; }}
  .canvas-row {{ grid-template-columns: 1fr; }}
  .flow, .mini-flow {{ grid-template-columns: 1fr; }}
  .flow-op {{ min-height: 24px; }}
  h1 {{ font-size: 24px; }}
}}
</style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div>
      <p class="eyebrow">Reverie MNIST</p>
      <h1>Source Classifier Visualizer</h1>
    </div>
    <div id="status" class="status"><span class="dot"></span><span>verified</span></div>
  </header>

  <section class="summary" id="summary"></section>

  <section class="panel draw-panel">
    <div class="section-head">
      <h2>Draw A Digit</h2>
      <span class="mode-pill">live vecmat_q31</span>
    </div>
    <div class="draw-grid">
      <div class="draw-board">
        <canvas id="drawCanvas" class="draw-canvas" width="280" height="280" aria-label="digit drawing pad"></canvas>
        <div class="draw-controls">
          <button id="clearDraw" class="tool-button" type="button">Clear</button>
          <button id="sampleDraw" class="tool-button" type="button">Sample 6</button>
          <label class="brush-control"><span>brush</span><input id="brushSize" type="range" min="10" max="42" value="24"></label>
          <label class="toggle-control"><input id="normalizeDraw" type="checkbox" checked><span>center/scale</span></label>
        </div>
      </div>
      <div class="guess-grid">
        <div>
          <div class="guess-card">
            <div class="label">guess</div>
            <div class="guess" id="drawGuess">-</div>
            <div class="detail" id="drawGuessDetail">baseline outputs are zero</div>
          </div>
          <h3>28x28 tensor</h3>
          <canvas id="drawPreviewCanvas" class="preview-canvas" width="280" height="280"></canvas>
        </div>
        <div class="live-logits">
          <h3>live logits</h3>
          <div id="drawLogits"></div>
        </div>
      </div>
    </div>
    <div class="draw-rev-grid">
      <div>
        <h3>interactive reversible flow</h3>
        <div class="mini-flow" id="drawReversibleFlow"></div>
      </div>
      <div>
        <h3>interactive state created then uncomputed</h3>
        <div id="drawStateDiff"></div>
      </div>
    </div>
  </section>

  <main class="grid main">
    <section class="panel">
      <h2>Pixel State</h2>
      <div class="canvas-row">
        <div class="canvas-block"><div class="canvas-title">input image</div><canvas id="imageCanvas" class="pixel" width="280" height="280"></canvas></div>
        <div class="canvas-block"><div class="canvas-title">winning contribution</div><canvas id="contribCanvas" class="pixel" width="280" height="280"></canvas></div>
        <div class="canvas-block"><div class="canvas-title">margin contribution</div><canvas id="marginCanvas" class="pixel" width="280" height="280"></canvas></div>
      </div>
    </section>
    <section class="panel">
      <h2>Class Logits</h2>
      <div id="logits"></div>
    </section>
  </main>

  <section class="panel reversible-panel">
    <div class="section-head">
      <h2>Reversible Execution</h2>
      <span class="mode-pill">forward then inverse</span>
    </div>
    <div class="flow" id="reversibleFlow"></div>
    <div class="reversible-grid">
      <div>
        <h3>roundtrip facts</h3>
        <div class="kv" id="reversibleFacts"></div>
      </div>
      <div>
        <h3>state created then uncomputed</h3>
        <div id="stateDiff"></div>
      </div>
    </div>
  </section>

  <section class="split">
    <section class="panel">
      <h2>Attribution</h2>
      <h3>Winning class pixels</h3>
      <div id="topContrib"></div>
      <h3>Winner vs runner-up pixels</h3>
      <div id="topMargin"></div>
    </section>
    <section class="panel">
      <h2>Roundtrip Proof</h2>
      <div class="timeline" id="timeline"></div>
      <h3>Verifier checks</h3>
      <div class="kv" id="checks"></div>
    </section>
  </section>

  <section class="split">
    <section class="panel">
      <h2>Source</h2>
      <div class="kv" id="sourceFacts"></div>
      <h3>main transition</h3>
      <pre class="source" id="source"></pre>
    </section>
    <section class="panel">
      <h2>Tensor And Replay Cost</h2>
      <div class="kv" id="cost"></div>
      <h3>capabilities</h3>
      <div class="chips" id="capabilities"></div>
    </section>
  </section>

  <div class="tooltip" id="tooltip"></div>
  <div class="footer" id="footer"></div>
</div>
<script id="viz-data" type="application/json">{payload}</script>
<script>
const data = JSON.parse(document.getElementById('viz-data').textContent);
const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat('en-US');
const shortHash = (value) => value ? String(value).slice(0, 12) : 'n/a';
const yn = (value) => value === true ? '<span class="ok">true</span>' : value === false ? '<span class="bad">false</span>' : 'n/a';
const Q31_ONE_JS = 2147483648;
const U64_JS = 1n << 64n;
const I63_JS = 1n << 63n;
function bytes(value) {{
  if (typeof value !== 'number') return 'n/a';
  const units = ['B', 'KB', 'MB', 'GB'];
  let index = 0, current = value;
  while (current >= 1024 && index < units.length - 1) {{ current /= 1024; index += 1; }}
  return `${{current.toFixed(index ? 1 : 0)}} ${{units[index]}}`;
}}
function metric(label, value, sub='') {{
  return `<div class="metric"><div class="label">${{label}}</div><div class="value">${{value}}</div><div class="sub">${{sub}}</div></div>`;
}}
function wrapI64Big(value) {{
  let current = value % U64_JS;
  if (current < 0n) current += U64_JS;
  if (current >= I63_JS) current -= U64_JS;
  return current;
}}
function addI64(left, right) {{
  return Number(wrapI64Big(BigInt(left) + BigInt(right)));
}}
function fixedMulQ31Js(left, right) {{
  return Number(wrapI64Big((BigInt(left) * BigInt(right)) >> 31n));
}}
function pixelToQ31Js(pixel) {{
  return Math.floor((Math.max(0, Math.min(255, pixel)) * Q31_ONE_JS) / 255);
}}
function rankedLogits(logits) {{
  return logits.map((value, digit) => ({{ digit, value }})).sort((a, b) => b.value - a.value || a.digit - b.digit);
}}
function classifyImageU8(imageU8) {{
  const weights = data.model.weights;
  const logits = data.model.bias.slice();
  const imageQ31 = imageU8.map(pixelToQ31Js);
  for (let pixel = 0; pixel < 784; pixel++) {{
    const q31 = imageQ31[pixel] || 0;
    if (!q31) continue;
    const row = weights[pixel];
    for (let digit = 0; digit < 10; digit++) {{
      logits[digit] = addI64(logits[digit], fixedMulQ31Js(q31, row[digit]));
    }}
  }}
  const ranked = rankedLogits(logits);
  const prediction = ranked[0].digit;
  const runnerUp = ranked[1] ? ranked[1].digit : prediction;
  return {{
    image_u8: imageU8,
    image_q31: imageQ31,
    logits,
    prediction,
    runner_up: runnerUp,
    margin: addI64(logits[prediction], -logits[runnerUp]),
    active_pixels: imageU8.filter((value) => value > 0).length,
  }};
}}
function shiftImageU8(imageU8, dx, dy) {{
  const shifted = Array(784).fill(0);
  for (let y = 0; y < 28; y++) {{
    for (let x = 0; x < 28; x++) {{
      const value = imageU8[y * 28 + x] || 0;
      if (!value) continue;
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || nx >= 28 || ny < 0 || ny >= 28) continue;
      const target = ny * 28 + nx;
      shifted[target] = Math.max(shifted[target], value);
    }}
  }}
  return shifted;
}}
function classifyRobustImageU8(imageU8) {{
  const offsets = [
    [0, 0],
    [-1, 0], [1, 0], [0, -1], [0, 1],
    [-1, -1], [1, -1], [-1, 1], [1, 1],
  ];
  const logits = Array(10).fill(0);
  for (const [dx, dy] of offsets) {{
    const variant = classifyImageU8(shiftImageU8(imageU8, dx, dy));
    for (let digit = 0; digit < 10; digit++) {{
      logits[digit] = addI64(logits[digit], variant.logits[digit]);
    }}
  }}
  const ranked = rankedLogits(logits);
  const prediction = ranked[0].digit;
  const runnerUp = ranked[1] ? ranked[1].digit : prediction;
  return {{
    image_u8: imageU8,
    image_q31: imageU8.map(pixelToQ31Js),
    logits,
    prediction,
    runner_up: runnerUp,
    margin: addI64(logits[prediction], -logits[runnerUp]),
    active_pixels: activePixelCount(imageU8),
    variants: offsets.length,
  }};
}}
function renderSummary() {{
  const s = data.summary;
  $('summary').innerHTML = [
    metric('prediction', s.prediction, `label ${{s.label}}`),
    metric('correct', s.correct ? 'yes' : 'no', `expected ${{s.expected_correct ?? 'n/a'}}`),
    metric('runner-up', s.runner_up, `margin ${{fmt.format(s.margin)}}`),
    metric('active pixels', fmt.format(s.active_pixels), `audit step ${{s.audit_step ?? 'n/a'}}`),
    metric('source lines', fmt.format(data.source.line_count), bytes(data.source.bytes)),
    metric('roundtrip', data.proof.roundtrip_passed ? 'pass' : 'fail', shortHash(data.proof.roundtrip_fingerprint)),
  ].join('');
  if (!s.correct || !data.proof.roundtrip_passed || !data.proof.verification_passed) {{
    $('status').innerHTML = '<span class="dot fail"></span><span>attention</span>';
  }}
}}
function colorSigned(value, maxAbs) {{
  if (!value || !maxAbs) return [238, 232, 219, 255];
  const t = Math.min(1, Math.abs(value) / maxAbs);
  const alpha = Math.round(70 + 185 * t);
  if (value > 0) return [47, 111, 103, alpha];
  return [179, 77, 88, alpha];
}}
function drawMatrix(canvasId, values, mode) {{
  const canvas = $(canvasId);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const cell = canvas.width / 28;
  const maxAbs = mode === 'gray' ? 255 : Math.max(1, ...values.map((v) => Math.abs(v || 0)));
  for (let y = 0; y < 28; y++) {{
    for (let x = 0; x < 28; x++) {{
      const index = y * 28 + x;
      const value = values[index] || 0;
      if (mode === 'gray') {{
        const shade = Math.max(0, Math.min(255, value));
        ctx.fillStyle = `rgb(${{255 - shade}}, ${{255 - shade}}, ${{255 - shade}})`;
      }} else {{
        const c = colorSigned(value, maxAbs);
        ctx.fillStyle = `rgba(${{c[0]}}, ${{c[1]}}, ${{c[2]}}, ${{(c[3] / 255).toFixed(3)}})`;
      }}
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }}
  }}
  ctx.strokeStyle = 'rgba(32,33,36,0.16)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 28; i++) {{
    const p = i * cell;
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, canvas.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(canvas.width, p); ctx.stroke();
  }}
  attachTooltip(canvas, values, mode);
}}
function attachTooltip(canvas, values, mode) {{
  const tooltip = $('tooltip');
  canvas.onmousemove = (event) => {{
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) / rect.width * 28);
    const y = Math.floor((event.clientY - rect.top) / rect.height * 28);
    if (x < 0 || x >= 28 || y < 0 || y >= 28) return;
    const index = y * 28 + x;
    tooltip.style.opacity = 1;
    tooltip.style.left = `${{event.clientX}}px`;
    tooltip.style.top = `${{event.clientY}}px`;
    tooltip.textContent = mode === 'gray'
      ? `pixel ${{index}}: ${{values[index] || 0}}`
      : `pixel ${{index}}: ${{fmt.format(values[index] || 0)}}`;
  }};
  canvas.onmouseleave = () => {{ tooltip.style.opacity = 0; }};
}}
window.addEventListener('scroll', () => {{ $('tooltip').style.opacity = 0; }}, {{ passive: true }});
const drawState = {{
  drawing: false,
  last: null,
  brush: 24,
  normalize: true,
  pending: false,
}};
function drawPointFromEvent(canvas, event) {{
  const rect = canvas.getBoundingClientRect();
  return {{
    x: (event.clientX - rect.left) / rect.width * canvas.width,
    y: (event.clientY - rect.top) / rect.height * canvas.height,
  }};
}}
function clearDrawCanvas() {{
  const canvas = $('drawCanvas');
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}}
function paintDot(ctx, point) {{
  ctx.fillStyle = 'black';
  ctx.beginPath();
  ctx.arc(point.x, point.y, drawState.brush / 2, 0, Math.PI * 2);
  ctx.fill();
}}
function paintStroke(ctx, from, to) {{
  ctx.strokeStyle = 'black';
  ctx.lineWidth = drawState.brush;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
}}
function scheduleDrawInference() {{
  if (drawState.pending) return;
  drawState.pending = true;
  requestAnimationFrame(() => {{
    drawState.pending = false;
    renderDrawInference();
  }});
}}
function sampleDrawCanvas() {{
  const canvas = $('drawCanvas');
  const sample = document.createElement('canvas');
  sample.width = 28;
  sample.height = 28;
  const sampleCtx = sample.getContext('2d');
  sampleCtx.fillStyle = 'white';
  sampleCtx.fillRect(0, 0, 28, 28);
  sampleCtx.imageSmoothingEnabled = true;
  sampleCtx.imageSmoothingQuality = 'high';
  sampleCtx.drawImage(canvas, 0, 0, 28, 28);
  return imageDataToU8(sampleCtx.getImageData(0, 0, 28, 28));
}}
function imageDataToU8(imageData) {{
  const pixels = imageData.data;
  const imageU8 = [];
  for (let index = 0; index < pixels.length; index += 4) {{
    const darkness = 255 - Math.round((pixels[index] + pixels[index + 1] + pixels[index + 2]) / 3);
    imageU8.push(Math.max(0, Math.min(255, darkness)));
  }}
  return imageU8;
}}
function imageU8ToCanvas(imageU8) {{
  const canvas = document.createElement('canvas');
  canvas.width = 28;
  canvas.height = 28;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(28, 28);
  for (let pixel = 0; pixel < 784; pixel++) {{
    const shade = 255 - Math.max(0, Math.min(255, imageU8[pixel] || 0));
    const offset = pixel * 4;
    imageData.data[offset] = shade;
    imageData.data[offset + 1] = shade;
    imageData.data[offset + 2] = shade;
    imageData.data[offset + 3] = 255;
  }}
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}}
function activePixelCount(imageU8, threshold = 8) {{
  return imageU8.filter((value) => value > threshold).length;
}}
function normalizeDigitImage(imageU8) {{
  const threshold = 8;
  let minX = 28, minY = 28, maxX = -1, maxY = -1;
  for (let y = 0; y < 28; y++) {{
    for (let x = 0; x < 28; x++) {{
      const value = imageU8[y * 28 + x] || 0;
      if (value <= threshold) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }}
  }}
  if (maxX < minX || maxY < minY) return imageU8.slice();
  const cropWidth = maxX - minX + 1;
  const cropHeight = maxY - minY + 1;
  const scale = 20 / Math.max(cropWidth, cropHeight);
  const scaledWidth = Math.max(1, Math.round(cropWidth * scale));
  const scaledHeight = Math.max(1, Math.round(cropHeight * scale));
  const scaledX = Math.round((28 - scaledWidth) / 2);
  const scaledY = Math.round((28 - scaledHeight) / 2);
  const source = imageU8ToCanvas(imageU8);
  const scaled = document.createElement('canvas');
  scaled.width = 28;
  scaled.height = 28;
  const scaledCtx = scaled.getContext('2d');
  scaledCtx.fillStyle = 'white';
  scaledCtx.fillRect(0, 0, 28, 28);
  scaledCtx.imageSmoothingEnabled = true;
  scaledCtx.imageSmoothingQuality = 'high';
  scaledCtx.drawImage(source, minX, minY, cropWidth, cropHeight, scaledX, scaledY, scaledWidth, scaledHeight);
  const scaledU8 = imageDataToU8(scaledCtx.getImageData(0, 0, 28, 28));
  let total = 0, sumX = 0, sumY = 0;
  for (let y = 0; y < 28; y++) {{
    for (let x = 0; x < 28; x++) {{
      const value = scaledU8[y * 28 + x] || 0;
      total += value;
      sumX += x * value;
      sumY += y * value;
    }}
  }}
  if (!total) return scaledU8;
  const shiftX = Math.round(13.5 - (sumX / total));
  const shiftY = Math.round(13.5 - (sumY / total));
  const centered = document.createElement('canvas');
  centered.width = 28;
  centered.height = 28;
  const centeredCtx = centered.getContext('2d');
  centeredCtx.fillStyle = 'white';
  centeredCtx.fillRect(0, 0, 28, 28);
  centeredCtx.drawImage(scaled, shiftX, shiftY);
  return imageDataToU8(centeredCtx.getImageData(0, 0, 28, 28));
}}
function drawImageU8ToPad(imageU8) {{
  const canvas = $('drawCanvas');
  const ctx = canvas.getContext('2d');
  clearDrawCanvas();
  const cell = canvas.width / 28;
  for (let y = 0; y < 28; y++) {{
    for (let x = 0; x < 28; x++) {{
      const value = imageU8[y * 28 + x] || 0;
      if (!value) continue;
      ctx.fillStyle = `rgba(0, 0, 0, ${{(value / 255).toFixed(3)}})`;
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }}
  }}
  scheduleDrawInference();
}}
function miniStateValue(value) {{
  if (Array.isArray(value)) {{
    const nonzero = value.length === 784 ? activePixelCount(value) : value.filter((item) => item !== 0).length;
    if (value.length === 10) {{
      const ranked = rankedLogits(value);
      return nonzero ? `10 cells, top ${{ranked[0].digit}} = ${{fmt.format(ranked[0].value)}}` : '10 cells, 0 nonzero';
    }}
    return `${{fmt.format(value.length)}} cells, ${{fmt.format(nonzero)}} active`;
  }}
  return fmt.format(value || 0);
}}
function drawDiffTable(result) {{
  const blankLogits = Array(10).fill(0);
  const rows = [
    ['input ink', result.raw_image_u8 || result.image_u8, result.raw_image_u8 || result.image_u8, result.raw_image_u8 || result.image_u8, 'preserved'],
    ['model image', result.image_u8, result.image_u8, result.image_u8, result.preprocess || 'raw'],
    ['logits', blankLogits, result.logits, blankLogits, 'uncomputed'],
    ['prediction', 0, result.prediction, 0, 'uncomputed'],
  ];
  const body = rows.map(([place, baseline, forward, restored, outcome]) => {{
    const ok = outcome === 'preserved' || outcome === 'uncomputed' || outcome === 'raw' || String(outcome).startsWith('centered');
    return `<tr><td>${{escapeHtml(place)}}</td><td>${{escapeHtml(miniStateValue(baseline))}}</td><td>${{escapeHtml(miniStateValue(forward))}}</td><td>${{escapeHtml(miniStateValue(restored))}}</td><td><span class="result-pill ${{ok ? 'ok-pill' : 'bad-pill'}}">${{escapeHtml(outcome)}}</span></td></tr>`;
  }}).join('');
  return `<table class="diff-table"><tr><th>place</th><th>baseline</th><th>forward</th><th>restored</th><th>inverse result</th></tr>${{body}}</table>`;
}}
function renderDrawInference() {{
  const rawImageU8 = sampleDrawCanvas();
  const rawActivePixels = activePixelCount(rawImageU8);
  const imageU8 = drawState.normalize ? normalizeDigitImage(rawImageU8) : rawImageU8;
  const result = drawState.normalize ? classifyRobustImageU8(imageU8) : classifyImageU8(imageU8);
  result.raw_image_u8 = rawImageU8;
  result.raw_active_pixels = rawActivePixels;
  result.preprocess = drawState.normalize ? `centered x${{result.variants || 1}}` : 'raw';
  const active = rawActivePixels > 0;
  const visibleResult = active
    ? result
    : {{ ...result, logits: Array(10).fill(0), prediction: 0, runner_up: 0, margin: 0 }};
  drawMatrix('drawPreviewCanvas', imageU8, 'gray');
  $('drawGuess').textContent = active ? visibleResult.prediction : '-';
  $('drawGuessDetail').textContent = active
    ? `${{visibleResult.preprocess}} · runner-up ${{visibleResult.runner_up}} · margin ${{fmt.format(visibleResult.margin)}} · ink ${{fmt.format(visibleResult.raw_active_pixels)}}`
    : 'baseline outputs are zero';
  $('drawLogits').innerHTML = logitRows(visibleResult.logits, visibleResult.prediction, visibleResult.runner_up, null);
  const baseline = {{ prediction: 0, correct: 'n/a', logits: Array(10).fill(0) }};
  const forward = {{ prediction: visibleResult.prediction, correct: 'n/a', logits: visibleResult.logits }};
  const restored = {{ prediction: 0, correct: 'n/a', logits: Array(10).fill(0) }};
  $('drawReversibleFlow').innerHTML = [
    flowStep('baseline', baseline, `${{fmt.format(visibleResult.raw_active_pixels || 0)}} ink pixels · ${{visibleResult.preprocess || 'centered'}}`),
    '<div class="flow-op">+=</div>',
    flowStep('forward', forward, 'browser replay of Reverie tensor updates'),
    '<div class="flow-op">-=</div>',
    flowStep('restored', restored, 'outputs cleared, input preserved'),
  ].join('');
  $('drawStateDiff').innerHTML = drawDiffTable(visibleResult);
}}
function initDrawTool() {{
  const canvas = $('drawCanvas');
  const ctx = canvas.getContext('2d');
  clearDrawCanvas();
  $('brushSize').addEventListener('input', (event) => {{
    drawState.brush = Number(event.target.value) || 24;
  }});
  $('normalizeDraw').addEventListener('change', (event) => {{
    drawState.normalize = event.target.checked;
    renderDrawInference();
  }});
  $('clearDraw').addEventListener('click', () => {{
    clearDrawCanvas();
    renderDrawInference();
  }});
  $('sampleDraw').addEventListener('click', () => drawImageU8ToPad(data.image_u8));
  canvas.addEventListener('pointerdown', (event) => {{
    event.preventDefault();
    drawState.drawing = true;
    drawState.last = drawPointFromEvent(canvas, event);
    canvas.setPointerCapture(event.pointerId);
    paintDot(ctx, drawState.last);
    scheduleDrawInference();
  }});
  canvas.addEventListener('pointermove', (event) => {{
    if (!drawState.drawing || !drawState.last) return;
    event.preventDefault();
    const point = drawPointFromEvent(canvas, event);
    paintStroke(ctx, drawState.last, point);
    drawState.last = point;
    scheduleDrawInference();
  }});
  const stopDrawing = (event) => {{
    if (drawState.drawing && event && canvas.hasPointerCapture(event.pointerId)) {{
      canvas.releasePointerCapture(event.pointerId);
    }}
    drawState.drawing = false;
    drawState.last = null;
    scheduleDrawInference();
  }};
  canvas.addEventListener('pointerup', stopDrawing);
  canvas.addEventListener('pointercancel', stopDrawing);
  canvas.addEventListener('pointerleave', stopDrawing);
  renderDrawInference();
}}
function logitRows(logits, prediction, runnerUp, label = null) {{
  const min = Math.min(...logits);
  const max = Math.max(...logits);
  const range = Math.max(1, max - min);
  return logits.map((value, digit) => {{
    const pct = Math.max(2, Math.round(((value - min) / range) * 100));
    const predicted = digit === prediction;
    const runner = digit === runnerUp;
    const isLabel = digit === label;
    const classes = ['bar', predicted ? 'predicted' : '', runner ? 'runner' : '', value < 0 ? 'negative' : ''].join(' ');
    const digitClasses = ['digit', predicted ? 'predicted' : '', isLabel ? 'label' : ''].join(' ');
    return `<div class="logit-row"><div class="${{digitClasses}}">${{digit}}</div><div class="bar-track"><div class="${{classes}}" style="width:${{pct}}%"></div></div><div class="logit-value">${{fmt.format(value)}}</div></div>`;
  }}).join('');
}}
function renderLogits() {{
  $('logits').innerHTML = logitRows(data.logits, data.summary.prediction, data.summary.runner_up, data.summary.label);
}}
function table(rows, columns) {{
  if (!rows.length) return '<p class="sub">No contribution rows recorded.</p>';
  const head = `<tr>${{columns.map((c) => `<th>${{c.label}}</th>`).join('')}}</tr>`;
  const body = rows.map((row) => `<tr>${{columns.map((c) => `<td>${{c.format ? c.format(row[c.key], row) : fmt.format(row[c.key] ?? 0)}}</td>`).join('')}}</tr>`).join('');
  return `<table>${{head}}${{body}}</table>`;
}}
function renderAttribution() {{
  $('topContrib').innerHTML = table(data.top_contributions, [
    {{key:'pixel', label:'pixel'}}, {{key:'u8', label:'u8'}}, {{key:'weight', label:'weight'}}, {{key:'contribution', label:'contribution'}},
  ]);
  $('topMargin').innerHTML = table(data.top_margin_contributions, [
    {{key:'pixel', label:'pixel'}}, {{key:'u8', label:'u8'}}, {{key:'weight_delta', label:'weight delta'}}, {{key:'contribution', label:'margin contribution'}},
  ]);
}}
function kv(id, rows) {{
  $(id).innerHTML = rows.map(([k, v]) => `<div class="k">${{k}}</div><div class="v">${{v}}</div>`).join('');
}}
function topLogitText(logits) {{
  if (!Array.isArray(logits)) return 'logits n/a';
  const nonzero = logits.filter((value) => value !== 0).length;
  let topDigit = 0;
  let topValue = logits.length ? logits[0] : 0;
  logits.forEach((value, digit) => {{
    if (value > topValue) {{
      topDigit = digit;
      topValue = value;
    }}
  }});
  return `${{nonzero}}/${{logits.length}} logits nonzero, top ${{topDigit}} = ${{fmt.format(topValue)}}`;
}}
function stateHeadline(state) {{
  return `prediction ${{state.prediction ?? 'n/a'}}, correct ${{state.correct ?? 'n/a'}}`;
}}
function flowStep(name, state, detail) {{
  return `<div class="flow-step"><div class="name">${{escapeHtml(name)}}</div><div class="headline">${{escapeHtml(stateHeadline(state))}}</div><div class="detail">${{escapeHtml(topLogitText(state.logits))}}</div><div class="detail">${{escapeHtml(detail)}}</div></div>`;
}}
function stateSummary(summary) {{
  if (!summary || typeof summary !== 'object') return 'n/a';
  const display = summary.display || 'n/a';
  if (typeof summary.top_value === 'number') {{
    return `${{display.replace(String(summary.top_value), fmt.format(summary.top_value))}}`;
  }}
  return display;
}}
function renderReversibility() {{
  const rev = data.reversibility || {{}};
  const check = rev.check || {{}};
  const vchecks = rev.verification_checks || {{}};
  const witness = rev.witness || {{}};
  const entries = Array.isArray(witness.entries) ? witness.entries : [];
  const entryText = entries.length
    ? entries.map((entry) => `${{entry.name || 'witness'}} (${{fmt.format(entry.cells || 0)}} cells)`).join(', ')
    : 'none';
  $('reversibleFlow').innerHTML = [
    flowStep('baseline', data.roundtrip.baseline, 'seeded tensors, zeroed outputs'),
    '<div class="flow-op">+=</div>',
    flowStep('forward', data.roundtrip.forward, 'tensor updates and signal extraction'),
    '<div class="flow-op">-=</div>',
    flowStep('restored', data.roundtrip.restored, 'inverse updates restore baseline'),
  ].join('');
  kv('reversibleFacts', [
    ['baseline restored', yn(check.baseline_store_restored)],
    ['input restored', yn(check.input_restored)],
    ['output restored', yn(check.output_restored)],
    ['source hash replayed', yn(vchecks.source_hash_matches)],
    ['replay fingerprint matches', yn(vchecks.replay_fingerprint_matches)],
    ['inverse mismatches', fmt.format(Array.isArray(check.changed_store_entries) ? check.changed_store_entries.length : 0)],
    ['proof engine', `${{rev.program && rev.program.engine ? rev.program.engine : 'n/a'}} / ${{rev.program && rev.program.direction ? rev.program.direction : 'n/a'}}`],
    ['witness store', `${{escapeHtml(entryText)}}; ${{bytes(witness.known_payload_bytes)}}`],
    ['witness fingerprint', shortHash(witness.fingerprint)],
  ]);
  const rows = Array.isArray(rev.state_deltas) ? rev.state_deltas : [];
  if (!rows.length) {{
    $('stateDiff').innerHTML = '<p class="sub">No roundtrip state deltas recorded.</p>';
    return;
  }}
  const body = rows.map((row) => {{
    const ok = row.restored_matches_baseline === true;
    return `<tr><td>${{escapeHtml(row.name)}}</td><td>${{escapeHtml(stateSummary(row.baseline))}}</td><td>${{escapeHtml(stateSummary(row.forward))}}</td><td>${{escapeHtml(stateSummary(row.restored))}}</td><td><span class="result-pill ${{ok ? 'ok-pill' : 'bad-pill'}}">${{escapeHtml(row.reverse_result)}}</span></td></tr>`;
  }}).join('');
  $('stateDiff').innerHTML = `<table class="diff-table"><tr><th>place</th><th>baseline</th><th>forward</th><th>restored</th><th>inverse result</th></tr>${{body}}</table>`;
}}
function renderProof() {{
  const states = [
    ['baseline', data.roundtrip.baseline],
    ['forward', data.roundtrip.forward],
    ['restored', data.roundtrip.restored],
  ];
  $('timeline').innerHTML = states.map(([name, state]) => `<div class="state"><strong>${{name}}</strong><div>prediction: ${{state.prediction ?? 'n/a'}}</div><div>correct: ${{state.correct ?? 'n/a'}}</div></div>`).join('');
  const checks = data.proof.verification_checks || {{}};
  kv('checks', Object.keys(checks).sort().map((key) => [key, yn(checks[key])]));
}}
function renderSourceAndCost() {{
  kv('sourceFacts', [
    ['path', data.source.path],
    ['sha256', shortHash(data.source.sha256)],
    ['image tensor', yn(data.source.contains.image_tensor)],
    ['weights tensor', yn(data.source.contains.weights_tensor)],
    ['vecmat_q31', yn(data.source.contains.vecmat_q31)],
    ['prediction assert', yn(data.source.contains.prediction_assert)],
    ['correct assert', yn(data.source.contains.correct_assert)],
  ]);
  $('source').innerHTML = data.source.excerpt.map((line) => `<span class="line-no">${{line.line}}</span>${{escapeHtml(line.text)}}`).join('\\n');
  const cost = data.memory || {{}};
  const replay = (data.ml_profile && data.ml_profile.replay_cost) || {{}};
  kv('cost', [
    ['goal fit', data.ml_profile.goal_fit || 'n/a'],
    ['peak RSS', bytes(cost.peak_rss_bytes)],
    ['model payload', bytes(cost.model_payload_bytes)],
    ['sample payload', bytes(cost.sample_payload_bytes)],
    ['witness payload', bytes(cost.witness_payload_bytes)],
    ['replay payload', bytes(cost.replay_payload_bytes)],
    ['runtime state', bytes(cost.runtime_state_payload_bytes)],
    ['roundtrip statements', fmt.format(replay.roundtrip_statement_count || 0)],
    ['known state tensors', bytes(replay.known_state_tensor_payload_bytes)],
  ]);
  $('capabilities').innerHTML = (data.ml_profile.capabilities || []).map((value) => `<span class="chip">${{escapeHtml(value)}}</span>`).join('');
}}
function escapeHtml(value) {{
  return String(value).replace(/[&<>"']/g, (ch) => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
}}
function renderFooter() {{
  $('footer').textContent = `classifier ${{data.source.path}} · source ${{shortHash(data.source.sha256)}} · proof ${{shortHash(data.proof.roundtrip_fingerprint)}}`;
}}
renderSummary();
drawMatrix('imageCanvas', data.image_u8, 'gray');
drawMatrix('contribCanvas', data.contribution_map, 'signed');
drawMatrix('marginCanvas', data.margin_map, 'signed');
renderLogits();
initDrawTool();
renderAttribution();
renderProof();
renderReversibility();
renderSourceAndCost();
renderFooter();
</script>
</body>
</html>
"""


def render_visualizer(
    classifier_rev: Path,
    inference_bundle_path: Path,
    run_json_path: Path,
    roundtrip_json_path: Path,
    verification_json_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    inference_bundle = load_json(inference_bundle_path, "inference bundle")
    run_json = load_json(run_json_path, "run JSON")
    roundtrip_json = load_json(roundtrip_json_path, "roundtrip JSON")
    verification_json = load_json(verification_json_path, "roundtrip verification JSON")
    if not classifier_rev.exists():
        raise VisualizerError(f"classifier source is missing: {classifier_rev}")
    data = build_visualizer_data(
        classifier_rev=classifier_rev,
        inference_bundle=inference_bundle,
        run_json=run_json,
        roundtrip_json=roundtrip_json,
        verification_json=verification_json,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_document(data), encoding="utf-8")
    return {
        "output": str(output_path),
        "bytes": output_path.stat().st_size,
        "prediction": data["summary"]["prediction"],
        "label": data["summary"]["label"],
        "correct": data["summary"]["correct"],
        "roundtrip_passed": data["proof"]["roundtrip_passed"],
        "verification_passed": data["proof"]["verification_passed"],
        "source_sha256": data["source"]["sha256"],
    }


def write_self_test_files(root: Path) -> dict[str, Path]:
    classifier = root / "classifier.rev"
    classifier.write_text(
        """// Synthetic classifier.
global image: tensor<int, 784> = [0];
global weights: tensor<int, 784, 10> = [0];
global bias: tensor<int, 10> = [0,0,0,0,0,0,0,0,0,0];
global logits: witness<tensor<int, 10>>;
global prediction;
global correct;
global label = 3;
global expected_prediction = 3;
global expected_correct = 1;
procedure main()
  logits += vecmat_q31(image, weights);
  logits += bias;
  prediction += argmax(logits);
  correct += argmax_eq(logits, label);
  assert prediction == expected_prediction;
  assert correct == expected_correct
end
""",
        encoding="utf-8",
    )
    image = [0 for _ in range(784)]
    image[42] = 255
    weights = [[0 for _ in range(10)] for _ in range(784)]
    weights[42][3] = Q31_ONE
    weights[42][4] = Q31_ONE // 2
    logits = [0 for _ in range(10)]
    logits[3] = Q31_ONE
    logits[4] = Q31_ONE // 2
    inference = {
        "kind": "reverie_mnist_linear_q31_inference_replay_bundle",
        "model": {"weights": weights, "bias": [0 for _ in range(10)]},
        "sample": {"image_u8": image, "label": 3},
        "result": {
            "prediction": 3,
            "correct": True,
            "logits": logits,
            "top_logits": [{"digit": 3, "logit": Q31_ONE}, {"digit": 4, "logit": Q31_ONE // 2}],
            "memory": {"model_payload_bytes": 1, "sample_payload_bytes": 1, "witness_payload_bytes": 1, "replay_payload_bytes": 3},
            "attribution": {
                "margin": Q31_ONE // 2,
                "top_contributions": [{"pixel": 42, "u8": 255, "q31": Q31_ONE, "weight": Q31_ONE, "contribution": Q31_ONE}],
                "top_margin_contributions": [{"pixel": 42, "u8": 255, "q31": Q31_ONE, "weight_delta": Q31_ONE // 2, "contribution": Q31_ONE // 2}],
            },
        },
        "proof": {"claim": "deterministic_q31_inference_replay", "arithmetic": "q31_wrapping_i64", "checks": {"restored_initial_state": True}},
        "report": {"audit_step": 0, "sample_index": 0},
    }
    run = {
        "kind": "reverie_run_result",
        "program": {"file": str(classifier), "source_sha256": sha256_file(classifier)},
        "store": {
            "image": [pixel_to_q31(pixel) for pixel in image],
            "weights": weights,
            "bias": [0 for _ in range(10)],
            "logits": logits,
            "prediction": 3,
            "correct": 1,
            "label": 3,
            "expected_prediction": 3,
            "expected_correct": 1,
        },
        "ml_profile": {"goal_fit": "auditable_ml_kernel", "builtin_counts": {"vecmat_q31": 1}, "capabilities": ["self_test"], "replay_cost": {}},
    }
    roundtrip = {
        "kind": "reverie_roundtrip_result",
        "passed": True,
        "fingerprint": "1" * 64,
        "payload": {
            "program": {"file": str(classifier), "source_sha256": sha256_file(classifier)},
            "baseline": {"store": {"prediction": 0, "correct": 0, "logits": [0 for _ in range(10)]}},
            "forward": {"store": {"prediction": 3, "correct": 1, "logits": logits}},
            "restored": {"store": {"prediction": 0, "correct": 0, "logits": [0 for _ in range(10)]}},
            "check": {
                "baseline_store_restored": True,
                "changed_store_entries": [],
                "input_restored": True,
                "output_restored": True,
                "passed": True,
            },
        },
    }
    verification = {
        "kind": "reverie_roundtrip_verification",
        "passed": True,
        "proof_fingerprint": "1" * 64,
        "checks": {
            "source_hash_matches": True,
            "replay_fingerprint_matches": True,
            "replayed": True,
            "replay_restoration_passed": True,
        },
    }
    paths = {
        "classifier": classifier,
        "inference": root / "inference.json",
        "run": root / "run.json",
        "roundtrip": root / "roundtrip.json",
        "verification": root / "verification.json",
        "output": root / "visualizer.html",
    }
    for key, payload in (
        ("inference", inference),
        ("run", run),
        ("roundtrip", roundtrip),
        ("verification", verification),
    ):
        paths[key].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return paths


def run_self_test() -> int:
    try:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_self_test_files(Path(directory))
            report = render_visualizer(
                paths["classifier"],
                paths["inference"],
                paths["run"],
                paths["roundtrip"],
                paths["verification"],
                paths["output"],
            )
            html_text = paths["output"].read_text(encoding="utf-8")
            for snippet in (
                "Reverie MNIST Source Visualizer",
                "Draw A Digit",
                "drawCanvas",
                "drawGuess",
                "drawStateDiff",
                "normalizeDraw",
                "center/scale",
                "normalizeDigitImage",
                "classifyImageU8",
                "classifyRobustImageU8",
                "centered x",
                "fixedMulQ31Js",
                "imageCanvas",
                "contribCanvas",
                "Reversible Execution",
                "reversibleFlow",
                "state created then uncomputed",
                "stateDiff",
                "baseline restored",
                "roundtrip",
                "vecmat_q31",
                "viz-data",
            ):
                if snippet not in html_text:
                    raise AssertionError(f"rendered visualizer missing {snippet}")
            if report["prediction"] != 3 or report["label"] != 3 or report["correct"] is not True:
                raise AssertionError("self-test visualizer summary did not preserve classification")
            if report["roundtrip_passed"] is not True or report["verification_passed"] is not True:
                raise AssertionError("self-test visualizer did not preserve proof status")
    except (AssertionError, VisualizerError) as error:
        print(f"error: self-test failed: {error}", file=sys.stderr)
        return 1
    print("ok: MNIST Reverie visualizer self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    try:
        report = render_visualizer(
            args.classifier_rev,
            args.inference_bundle,
            args.run_json,
            args.roundtrip_json,
            args.verification_json,
            args.output,
        )
    except VisualizerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        "ok: wrote {output} ({bytes} bytes), prediction={prediction}, label={label}, "
        "correct={correct}, roundtrip={roundtrip_passed}, verification={verification_passed}".format(
            **report
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
