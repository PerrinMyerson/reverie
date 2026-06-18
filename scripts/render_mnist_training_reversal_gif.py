#!/usr/bin/env python3
"""Render a visible MNIST-style reversible training GIF from real Reverie runs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from render_mnist_reversal_gif import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DIGITS,
    Q31_ONE,
    TARGET_DIGIT,
    build_seed,
    image_u8_from_seed,
    q31_label,
    rect,
    render_gif,
    render_png,
    run_reverie,
    svg_text,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "assets" / "reverie-mnist-training-reversal-demo.gif"
PROGRAM = "examples/mnist_reversible_step.rev"


@dataclass(frozen=True)
class VerifiedTrainingRun:
    baseline: dict[str, Any]
    forward: dict[str, Any]
    restored: dict[str, Any]
    witness_metrics: dict[str, Any]


@dataclass(frozen=True)
class FrameState:
    phase: str
    direction: str
    logits_progress: float
    error_progress: float
    model_progress: float
    prediction: int
    correct: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render docs/assets/reverie-mnist-training-reversal-demo.gif by "
            "running examples/mnist_reversible_step.rev forward and backward."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "GIF path to write. Defaults to "
            "docs/assets/reverie-mnist-training-reversal-demo.gif."
        ),
    )
    parser.add_argument(
        "--frame-delay",
        type=int,
        default=18,
        help="ImageMagick GIF frame delay in 1/100ths of a second.",
    )
    parser.add_argument(
        "--quiet-cargo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --quiet to cargo run when collecting Reverie results.",
    )
    return parser.parse_args()


def training_seed() -> dict[str, Any]:
    seed = build_seed()
    seed["error"] = [0 for _ in range(DIGITS)]
    seed["lr"] = Q31_ONE // 4
    return seed


def verified_training_run(tmpdir: Path, quiet_cargo: bool) -> VerifiedTrainingRun:
    baseline = training_seed()
    seed_path = tmpdir / "mnist-training-seed.json"
    write_json(seed_path, baseline)
    forward_result = run_reverie(
        ["run", PROGRAM, "--vars-json", str(seed_path), "--json"],
        quiet_cargo=quiet_cargo,
    )
    forward = forward_result["store"]
    final_path = tmpdir / "mnist-training-final.json"
    write_json(final_path, forward)
    reverse_result = run_reverie(
        ["reverse", PROGRAM, "--vars-json", str(final_path), "--json"],
        quiet_cargo=quiet_cargo,
    )
    restored = reverse_result["store"]
    validate_training_restoration(baseline, forward, restored)
    witness_metrics = forward_result.get("witness_metrics", {})
    if not isinstance(witness_metrics, dict):
        witness_metrics = {}
    return VerifiedTrainingRun(
        baseline=baseline,
        forward=forward,
        restored=restored,
        witness_metrics=witness_metrics,
    )


def validate_training_restoration(
    baseline: dict[str, Any],
    forward: dict[str, Any],
    restored: dict[str, Any],
) -> None:
    if forward.get("prediction") != TARGET_DIGIT or forward.get("correct") != 1:
        raise RuntimeError("forward MNIST training step did not classify the digit")
    if forward.get("weights") == baseline.get("weights"):
        raise RuntimeError("forward MNIST training step did not update weights")
    if forward.get("error") == baseline.get("error"):
        raise RuntimeError("forward MNIST training step did not retain error witness")
    for name in (
        "weights",
        "bias",
        "logits",
        "error",
        "prediction",
        "correct",
        "image",
        "label",
        "lr",
    ):
        if restored.get(name) != baseline.get(name):
            raise RuntimeError(f"reverse training run did not restore `{name}`")


def lerp_int(start: int, end: int, progress: float) -> int:
    return round(start + (end - start) * progress)


def model_delta_norm(run: VerifiedTrainingRun) -> int:
    total = 0
    for before_row, after_row in zip(run.baseline["weights"], run.forward["weights"]):
        total += sum(abs(int(after) - int(before)) for before, after in zip(before_row, after_row))
    total += sum(
        abs(int(after) - int(before))
        for before, after in zip(run.baseline["bias"], run.forward["bias"])
    )
    return total


def compact_int(value: int) -> str:
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def frame_states() -> list[FrameState]:
    states = []
    for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
        states.append(
            FrameState("compute logits witness", "forward", progress, 0.0, 0.0, 0, 0)
        )
    states.extend(
        [
            FrameState("record prediction/correct", "forward", 1.0, 0.0, 0.0, TARGET_DIGIT, 1),
            FrameState("write error witness", "forward", 1.0, 0.5, 0.0, TARGET_DIGIT, 1),
            FrameState("write error witness", "forward", 1.0, 1.0, 0.0, TARGET_DIGIT, 1),
            FrameState("update weights and bias", "forward", 1.0, 1.0, 0.5, TARGET_DIGIT, 1),
            FrameState("training state held", "forward", 1.0, 1.0, 1.0, TARGET_DIGIT, 1),
            FrameState("training state held", "forward", 1.0, 1.0, 1.0, TARGET_DIGIT, 1),
            FrameState("undo weights and bias", "reverse", 1.0, 1.0, 0.5, TARGET_DIGIT, 1),
            FrameState("undo weights and bias", "reverse", 1.0, 1.0, 0.0, TARGET_DIGIT, 1),
            FrameState("consume error witness", "reverse", 1.0, 0.5, 0.0, TARGET_DIGIT, 1),
            FrameState("consume error witness", "reverse", 1.0, 0.0, 0.0, TARGET_DIGIT, 1),
            FrameState("uncompute prediction/correct", "reverse", 1.0, 0.0, 0.0, 0, 0),
        ]
    )
    for progress in (0.75, 0.5, 0.25, 0.0, 0.0):
        states.append(FrameState("uncompute logits witness", "reverse", progress, 0.0, 0.0, 0, 0))
    return states


def render_digit_panel(run: VerifiedTrainingRun) -> list[str]:
    image_u8 = image_u8_from_seed(run.baseline)
    x0 = 32
    y0 = 92
    grid_x = 58
    grid_y = 138
    cell = 9
    gap = 1
    active_pixels = sum(1 for pixel in image_u8 if pixel != 0)
    lines = [
        rect(x0, y0, 316, 386, fill="#111827", stroke="#3a4960", radius=14),
        svg_text(58, 128, "training sample", size=17, fill="#22d3ee", weight=800),
    ]
    for row in range(28):
        for col in range(28):
            value = image_u8[row * 28 + col]
            fill = "#172033" if value == 0 else f"rgb({value},{value},{value})"
            lines.append(
                f'<rect x="{grid_x + col * (cell + gap)}" '
                f'y="{grid_y + row * (cell + gap)}" '
                f'width="{cell}" height="{cell}" fill="{fill}"/>'
            )
    lines.extend(
        [
            svg_text(58, 450, "label", size=15, fill="#9fb1c7", weight=700),
            svg_text(122, 450, TARGET_DIGIT, size=24, fill="#ffffff", weight=900),
            svg_text(
                190,
                450,
                f"ink cells {active_pixels}/784",
                size=15,
                fill="#9fb1c7",
                weight=700,
            ),
        ]
    )
    return lines


def render_witness_panel(run: VerifiedTrainingRun, state: FrameState) -> list[str]:
    x0 = 382
    y0 = 92
    width = 342
    logits = [
        lerp_int(0, int(value), state.logits_progress)
        for value in run.forward["logits"]
    ]
    error = [
        lerp_int(0, int(value), state.error_progress)
        for value in run.forward["error"]
    ]
    max_abs = max(
        1,
        *(abs(value) for value in run.forward["logits"]),
        *(abs(value) for value in run.forward["error"]),
    )
    lines = [
        rect(x0, y0, width, 386, fill="#111827", stroke="#3a4960", radius=14),
        svg_text(x0 + 26, y0 + 36, "witness tape", size=18, fill="#34d399", weight=900),
        svg_text(x0 + 26, y0 + 66, "logits", size=15, fill="#facc15", weight=900),
        svg_text(x0 + 188, y0 + 66, "error", size=15, fill="#f472b6", weight=900),
    ]
    for digit in range(DIGITS):
        y = y0 + 88 + digit * 23
        lines.extend(render_small_bar(x0 + 26, y, digit, logits[digit], max_abs, "#facc15"))
        lines.extend(render_small_bar(x0 + 188, y, digit, error[digit], max_abs, "#f472b6"))
    lines.extend(
        [
            rect(x0 + 26, y0 + 338, 88, 50, fill="#172033", stroke="#475569", radius=9),
            rect(x0 + 130, y0 + 338, 88, 50, fill="#172033", stroke="#475569", radius=9),
            svg_text(x0 + 40, y0 + 357, "prediction", size=12, fill="#9fb1c7", weight=800),
            svg_text(x0 + 158, y0 + 357, "correct", size=12, fill="#9fb1c7", weight=800),
            svg_text(x0 + 40, y0 + 380, state.prediction, size=22, fill="#ffffff", weight=900),
            svg_text(x0 + 158, y0 + 380, state.correct, size=22, fill="#ffffff", weight=900),
        ]
    )
    return lines


def render_small_bar(
    x: int,
    y: int,
    digit: int,
    value: int,
    max_abs: int,
    color: str,
) -> list[str]:
    width = 106
    bar_width = max(2, round((abs(value) / max_abs) * width))
    lines = [
        svg_text(x, y + 14, digit, size=13, fill="#cbd5e1", weight=800),
        f'<rect x="{x + 22}" y="{y}" width="{width}" height="15" rx="7" fill="#1e293b"/>',
    ]
    if value != 0:
        lines.append(
            f'<rect x="{x + 22}" y="{y}" width="{bar_width}" height="15" rx="7" fill="{color}"/>'
        )
    return lines


def render_update_panel(run: VerifiedTrainingRun, state: FrameState) -> list[str]:
    x0 = 752
    y0 = 92
    reverse = state.direction == "reverse"
    color = "#f472b6" if reverse else "#2dd4bf"
    total_delta = model_delta_norm(run)
    shown_delta = round(total_delta * state.model_progress)
    witness_entries = run.witness_metrics.get("entries", [])
    witness_payload = run.witness_metrics.get("known_payload_bytes", 160)
    lines = [
        rect(x0, y0, 336, 386, fill="#111827", stroke="#3a4960", radius=14),
        svg_text(x0 + 26, y0 + 38, state.direction, size=22, fill=color, weight=900),
        svg_text(x0 + 26, y0 + 70, state.phase, size=16, fill="#ffffff", weight=800),
    ]
    lines.extend(
        render_progress_bar(
            x0 + 26,
            y0 + 116,
            284,
            "model update applied",
            state.model_progress,
            color,
        )
    )
    lines.extend(
        render_progress_bar(
            x0 + 26,
            y0 + 178,
            284,
            "witness tape retained",
            max(state.logits_progress, state.error_progress),
            "#facc15",
        )
    )
    lines.extend(
        [
            svg_text(x0 + 26, y0 + 254, "delta norm", size=15, fill="#9fb1c7", weight=700),
            svg_text(
                x0 + 206,
                y0 + 254,
                compact_int(shown_delta),
                size=15,
                fill="#dbeafe",
                weight=900,
                anchor="end",
            ),
            svg_text(x0 + 26, y0 + 284, "witness vars", size=15, fill="#9fb1c7", weight=700),
            svg_text(
                x0 + 206,
                y0 + 284,
                len(witness_entries),
                size=15,
                fill="#dbeafe",
                weight=900,
                anchor="end",
            ),
            svg_text(x0 + 26, y0 + 314, "payload bytes", size=15, fill="#9fb1c7", weight=700),
            svg_text(
                x0 + 206,
                y0 + 314,
                witness_payload,
                size=15,
                fill="#dbeafe",
                weight=900,
                anchor="end",
            ),
            svg_text(
                x0 + 26,
                y0 + 350,
                "update uses image + error witness",
                size=12,
                fill="#8aa0b7",
                weight=650,
            ),
        ]
    )
    return lines


def render_progress_bar(
    x: int,
    y: int,
    width: int,
    label: str,
    value: float,
    color: str,
) -> list[str]:
    clamped = max(0.0, min(1.0, value))
    filled = round(width * clamped)
    lines = [
        svg_text(x, y, label, size=16, fill="#cbd5e1", weight=800),
        f'<rect x="{x}" y="{y + 14}" width="{width}" height="17" rx="8" fill="#1e293b"/>',
    ]
    if filled > 0:
        lines.append(
            f'<rect x="{x}" y="{y + 14}" width="{max(8, filled)}" '
            f'height="17" rx="8" fill="{color}"/>'
        )
    return lines


def render_footer(state: FrameState) -> list[str]:
    reverse = state.direction == "reverse"
    message = (
        "reverse uses witnesses to undo model updates and clear tape"
        if reverse
        else "forward writes logits/error witnesses, then updates weights/bias"
    )
    color = "#f472b6" if reverse else "#2dd4bf"
    progress = max(state.logits_progress, state.error_progress, state.model_progress)
    return [
        rect(32, 506, 1056, 94, fill="#111827", stroke="#3a4960", radius=14),
        svg_text(58, 540, "training reversal", size=19, fill="#f59e0b", weight=900),
        svg_text(302, 540, message, size=17, fill="#dbeafe", weight=650),
        '<rect x="58" y="562" width="972" height="18" rx="9" fill="#1e293b"/>',
        (
            f'<rect x="58" y="562" width="{max(8, round(972 * progress))}" '
            f'height="18" rx="9" fill="{color}"/>'
            if progress > 0
            else ""
        ),
        svg_text(
            1048,
            592,
            "generated from reverie run + reverie reverse",
            size=14,
            fill="#8aa0b7",
            weight=650,
            anchor="end",
        ),
    ]


def render_frame(run: VerifiedTrainingRun, state: FrameState) -> str:
    phase_color = "#f472b6" if state.direction == "reverse" else "#facc15"
    phase_text = "reverse: restore model" if state.direction == "reverse" else "forward: train"
    elements = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" '
            f'viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">'
        ),
        '<rect width="1120" height="640" fill="#0b1220"/>',
        svg_text(
            32,
            46,
            "Reverie MNIST training",
            size=28,
            fill="#ffffff",
            weight=900,
            family="Inter, Helvetica, Arial, sans-serif",
        ),
        svg_text(
            382,
            46,
            "witness-taped Q31 model update",
            size=18,
            fill="#9fb1c7",
            weight=650,
            family="Inter, Helvetica, Arial, sans-serif",
        ),
        svg_text(1088, 46, phase_text, size=19, fill=phase_color, weight=900, anchor="end"),
    ]
    elements.extend(render_digit_panel(run))
    elements.extend(render_witness_panel(run, state))
    elements.extend(render_update_panel(run, state))
    elements.extend(render_footer(state))
    elements.append("</svg>")
    return "\n".join(part for part in elements if part)


def main() -> int:
    args = parse_args()
    if shutil.which("sips") is None:
        print("error: macOS `sips` is required to render SVG frames", file=sys.stderr)
        return 1
    if shutil.which("magick") is None:
        print("error: ImageMagick `magick` is required to render the GIF", file=sys.stderr)
        return 1

    try:
        with tempfile.TemporaryDirectory(prefix="reverie-mnist-training-gif-") as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            run = verified_training_run(tmpdir, args.quiet_cargo)
            frame_paths: list[Path] = []
            for frame_index, state in enumerate(frame_states()):
                svg_path = tmpdir / f"frame-{frame_index:03}.svg"
                png_path = tmpdir / f"frame-{frame_index:03}.png"
                svg_path.write_text(render_frame(run, state), encoding="utf-8")
                render_png(svg_path, png_path)
                frame_paths.append(png_path)
            output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
            render_gif(frame_paths, output, args.frame_delay)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"ok: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
