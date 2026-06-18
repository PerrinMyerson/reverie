#!/usr/bin/env python3
"""Render a clear MNIST-style inference reversal GIF from real Reverie runs."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "assets" / "reverie-mnist-reversal-demo.gif"
PROGRAM = "examples/mnist_identify.rev"
CANVAS_WIDTH = 1120
CANVAS_HEIGHT = 640
IMAGE_SIDE = 28
IMAGE_PIXELS = IMAGE_SIDE * IMAGE_SIDE
DIGITS = 10
Q31_ONE = 1 << 31
TARGET_DIGIT = 3


@dataclass(frozen=True)
class VerifiedRun:
    baseline: dict[str, Any]
    forward: dict[str, Any]
    restored: dict[str, Any]


@dataclass(frozen=True)
class FrameState:
    phase: str
    direction: str
    output_progress: float
    logits_progress: float
    prediction: int
    correct: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render docs/assets/reverie-mnist-reversal-demo.gif by running "
            "examples/mnist_identify.rev forward and backward."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="GIF path to write. Defaults to docs/assets/reverie-mnist-reversal-demo.gif.",
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


def digit_image_u8() -> list[int]:
    pattern = [
        " 1111 ",
        "     1",
        "     1",
        "    1 ",
        "  111 ",
        "     1",
        "     1",
        "     1",
        " 1111 ",
    ]
    scale = 3
    offset_x = 3
    offset_y = 0
    image = [[0 for _ in range(IMAGE_SIDE)] for _ in range(IMAGE_SIDE)]
    for row_index, row in enumerate(pattern):
        for col_index, marker in enumerate(row):
            if marker != "1":
                continue
            for dy in range(scale):
                for dx in range(scale):
                    y = offset_y + row_index * scale + dy
                    x = offset_x + col_index * scale + dx
                    if 0 <= x < IMAGE_SIDE and 0 <= y < IMAGE_SIDE:
                        image[y][x] = 255

    softened = [row[:] for row in image]
    for y, row in enumerate(image):
        for x, value in enumerate(row):
            if value == 0:
                continue
            for dy, dx in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                yy = y + dy
                xx = x + dx
                if 0 <= xx < IMAGE_SIDE and 0 <= yy < IMAGE_SIDE:
                    softened[yy][xx] = max(softened[yy][xx], 120)
    return [value for row in softened for value in row]


def pixel_to_q31(pixel: int) -> int:
    return (max(0, min(255, pixel)) * Q31_ONE) // 255


def build_seed() -> dict[str, Any]:
    image_u8 = digit_image_u8()
    active_pixels = sum(1 for pixel in image_u8 if pixel != 0)
    if active_pixels == 0:
        raise RuntimeError("digit glyph has no active pixels")
    target_weight = Q31_ONE // active_pixels
    other_weight = -(target_weight // 8)
    weights = [[0 for _ in range(DIGITS)] for _ in range(IMAGE_PIXELS)]
    for index, pixel in enumerate(image_u8):
        if pixel == 0:
            continue
        for digit in range(DIGITS):
            weights[index][digit] = target_weight if digit == TARGET_DIGIT else other_weight
    return {
        "image": [pixel_to_q31(pixel) for pixel in image_u8],
        "weights": weights,
        "bias": [0 for _ in range(DIGITS)],
        "logits": [0 for _ in range(DIGITS)],
        "prediction": 0,
        "correct": 0,
        "label": TARGET_DIGIT,
    }


def run_reverie(args: list[str], *, quiet_cargo: bool) -> dict[str, Any]:
    command = ["cargo", "run", "-p", "reverie-cli"]
    if quiet_cargo:
        command.append("--quiet")
    command.extend(["--", *args])
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Reverie command failed")
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Reverie command did not emit JSON: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("store"), dict):
        raise RuntimeError("Reverie JSON result did not contain a store object")
    return data


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")


def verified_run(tmpdir: Path, quiet_cargo: bool) -> VerifiedRun:
    baseline = build_seed()
    seed_path = tmpdir / "mnist-digit-seed.json"
    write_json(seed_path, baseline)
    forward_result = run_reverie(
        ["run", PROGRAM, "--vars-json", str(seed_path), "--json"],
        quiet_cargo=quiet_cargo,
    )
    forward = forward_result["store"]
    final_path = tmpdir / "mnist-digit-final.json"
    write_json(final_path, forward)
    reverse_result = run_reverie(
        ["reverse", PROGRAM, "--vars-json", str(final_path), "--json"],
        quiet_cargo=quiet_cargo,
    )
    restored = reverse_result["store"]
    validate_restoration(baseline, forward, restored)
    return VerifiedRun(baseline=baseline, forward=forward, restored=restored)


def validate_restoration(
    baseline: dict[str, Any],
    forward: dict[str, Any],
    restored: dict[str, Any],
) -> None:
    if forward.get("prediction") != TARGET_DIGIT or forward.get("correct") != 1:
        raise RuntimeError("forward MNIST run did not classify the digit correctly")
    for name in ("logits", "prediction", "correct", "image", "weights", "bias", "label"):
        if restored.get(name) != baseline.get(name):
            raise RuntimeError(f"reverse run did not restore `{name}`")


def image_u8_from_seed(seed: dict[str, Any]) -> list[int]:
    image = seed.get("image")
    if not isinstance(image, list) or len(image) != IMAGE_PIXELS:
        raise RuntimeError("seed image must have 784 cells")
    return [max(0, min(255, round(int(value) * 255 / Q31_ONE))) for value in image]


def q31_label(value: int) -> str:
    return f"{value / Q31_ONE:.2f}"


def lerp_int(start: int, end: int, progress: float) -> int:
    return round(start + (end - start) * progress)


def frame_states() -> list[FrameState]:
    frames: list[FrameState] = []
    for progress in (0.0, 0.0, 0.25, 0.5, 0.75, 1.0):
        frames.append(
            FrameState(
                phase="vecmat_q31 fills logits",
                direction="forward",
                output_progress=progress * 0.55,
                logits_progress=progress,
                prediction=0,
                correct=0,
            )
        )
    frames.extend(
        [
            FrameState("argmax records prediction", "forward", 0.72, 1.0, TARGET_DIGIT, 0),
            FrameState("argmax_eq records correctness", "forward", 1.0, 1.0, TARGET_DIGIT, 1),
            FrameState("forward result held", "forward", 1.0, 1.0, TARGET_DIGIT, 1),
            FrameState("forward result held", "forward", 1.0, 1.0, TARGET_DIGIT, 1),
            FrameState("uncompute correctness", "reverse", 0.72, 1.0, TARGET_DIGIT, 0),
            FrameState("uncompute prediction", "reverse", 0.55, 1.0, 0, 0),
        ]
    )
    for progress in (0.75, 0.5, 0.25, 0.0, 0.0):
        frames.append(
            FrameState(
                phase="uncompute logits",
                direction="reverse",
                output_progress=progress * 0.55,
                logits_progress=progress,
                prediction=0,
                correct=0,
            )
        )
    return frames


def xml(text: object) -> str:
    return html.escape(str(text), quote=True)


def svg_text(
    x: int,
    y: int,
    text: object,
    *,
    size: int = 18,
    fill: str = "#d8e2f0",
    weight: int = 500,
    anchor: str = "start",
    family: str = "Menlo, Monaco, Consolas, monospace",
) -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{fill}" font-family="{family}" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">'
        f"{xml(text)}</text>"
    )


def rect(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    fill: str = "#111827",
    stroke: str = "#334155",
    radius: int = 12,
    stroke_width: int = 2,
) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def render_digit_panel(run: VerifiedRun) -> list[str]:
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
        svg_text(58, 128, "MNIST-style 28x28 digit", size=17, fill="#22d3ee", weight=800),
    ]
    for row in range(IMAGE_SIDE):
        for col in range(IMAGE_SIDE):
            value = image_u8[row * IMAGE_SIDE + col]
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


def render_output_panel(run: VerifiedRun, state: FrameState) -> list[str]:
    x0 = 382
    y0 = 92
    width = 342
    final_logits = run.forward["logits"]
    shown_logits = [lerp_int(0, int(value), state.logits_progress) for value in final_logits]
    max_abs = max(1, max(abs(value) for value in final_logits))
    chart_x = x0 + 26
    chart_y = y0 + 72
    chart_width = width - 52
    lines = [
        rect(x0, y0, width, 386, fill="#111827", stroke="#3a4960", radius=14),
        svg_text(x0 + 26, y0 + 36, "computed outputs", size=18, fill="#34d399", weight=900),
    ]
    for digit, value in enumerate(shown_logits):
        y = chart_y + digit * 25
        bar_width = max(2, round((abs(value) / max_abs) * (chart_width - 86)))
        color = "#2dd4bf" if digit == TARGET_DIGIT else "#475569"
        lines.append(svg_text(chart_x, y + 15, digit, size=15, fill="#cbd5e1", weight=800))
        lines.append(
            f'<rect x="{chart_x + 28}" y="{y}" width="{chart_width - 86}" '
            'height="16" rx="8" fill="#1e293b"/>'
        )
        if value != 0:
            lines.append(
                f'<rect x="{chart_x + 28}" y="{y}" width="{bar_width}" '
                f'height="16" rx="8" fill="{color}"/>'
            )
        lines.append(
            svg_text(
                chart_x + chart_width,
                y + 14,
                q31_label(value),
                size=13,
                fill="#dbeafe",
                weight=650,
                anchor="end",
            )
        )
    lines.extend(render_output_tiles(x0 + 26, y0 + 344, state))
    return lines


def render_output_tiles(x: int, y: int, state: FrameState) -> list[str]:
    return [
        rect(x, y, 88, 50, fill="#172033", stroke="#475569", radius=9),
        rect(x + 104, y, 88, 50, fill="#172033", stroke="#475569", radius=9),
        svg_text(x + 14, y + 19, "prediction", size=12, fill="#9fb1c7", weight=800),
        svg_text(x + 132, y + 19, "correct", size=12, fill="#9fb1c7", weight=800),
        svg_text(x + 14, y + 42, state.prediction, size=22, fill="#ffffff", weight=900),
        svg_text(x + 132, y + 42, state.correct, size=22, fill="#ffffff", weight=900),
    ]


def render_reversal_panel(run: VerifiedRun, state: FrameState) -> list[str]:
    x0 = 752
    y0 = 92
    reverse = state.direction == "reverse"
    phase_color = "#f472b6" if reverse else "#2dd4bf"
    preserved = all(
        run.forward[name] == run.restored[name]
        for name in ("image", "weights", "bias", "label")
    )
    cleared = all(
        run.restored[name] == run.baseline[name]
        for name in ("logits", "prediction", "correct")
    )
    lines = [
        rect(x0, y0, 336, 386, fill="#111827", stroke="#3a4960", radius=14),
        svg_text(x0 + 26, y0 + 38, state.direction, size=22, fill=phase_color, weight=900),
        svg_text(x0 + 26, y0 + 70, state.phase, size=16, fill="#ffffff", weight=800),
    ]
    lines.extend(
        render_progress_bar(
            x0 + 26,
            y0 + 116,
            284,
            "computed output state",
            state.output_progress,
            color=phase_color,
        )
    )
    lines.extend(
        render_progress_bar(
            x0 + 26,
            y0 + 178,
            284,
            "logits materialized",
            state.logits_progress,
            color="#facc15",
        )
    )
    lines.extend(
        [
            svg_text(
                x0 + 26,
                y0 + 258,
                "image/model preserved",
                size=15,
                fill="#9fb1c7",
                weight=700,
            ),
            svg_text(
                x0 + 248,
                y0 + 258,
                "yes" if preserved else "no",
                size=15,
                fill="#dbeafe",
                weight=900,
            ),
            svg_text(
                x0 + 26,
                y0 + 288,
                "reverse clears outputs",
                size=15,
                fill="#9fb1c7",
                weight=700,
            ),
            svg_text(
                x0 + 248,
                y0 + 288,
                "yes" if cleared else "no",
                size=15,
                fill="#dbeafe",
                weight=900,
            ),
            svg_text(x0 + 26, y0 + 332, "program", size=15, fill="#9fb1c7", weight=700),
            svg_text(x0 + 26, y0 + 358, PROGRAM, size=14, fill="#dbeafe", weight=650),
        ]
    )
    return lines


def render_progress_bar(
    x: int,
    y: int,
    width: int,
    label: str,
    value: float,
    *,
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
        "reverse uncomputes logits, prediction, and correctness back to zero"
        if reverse
        else "forward computes logits, prediction, and correctness from the image/model"
    )
    color = "#f472b6" if reverse else "#2dd4bf"
    return [
        rect(32, 506, 1056, 94, fill="#111827", stroke="#3a4960", radius=14),
        svg_text(58, 540, "MNIST reversal", size=19, fill="#f59e0b", weight=900),
        svg_text(250, 540, message, size=17, fill="#dbeafe", weight=650),
        '<rect x="58" y="562" width="972" height="18" rx="9" fill="#1e293b"/>',
        (
            f'<rect x="58" y="562" width="{max(8, round(972 * state.output_progress))}" '
            f'height="18" rx="9" fill="{color}"/>'
            if state.output_progress > 0
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


def render_frame(run: VerifiedRun, state: FrameState) -> str:
    phase_color = "#f472b6" if state.direction == "reverse" else "#facc15"
    phase_text = "reverse: restore baseline" if state.direction == "reverse" else "forward: infer"
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
            "Reverie MNIST reversal",
            size=28,
            fill="#ffffff",
            weight=900,
            family="Inter, Helvetica, Arial, sans-serif",
        ),
        svg_text(
            390,
            46,
            "visible digit, reversible Q31 inference",
            size=18,
            fill="#9fb1c7",
            weight=650,
            family="Inter, Helvetica, Arial, sans-serif",
        ),
        svg_text(1088, 46, phase_text, size=19, fill=phase_color, weight=900, anchor="end"),
    ]
    elements.extend(render_digit_panel(run))
    elements.extend(render_output_panel(run, state))
    elements.extend(render_reversal_panel(run, state))
    elements.extend(render_footer(state))
    elements.append("</svg>")
    return "\n".join(part for part in elements if part)


def render_png(svg_path: Path, png_path: Path) -> None:
    completed = subprocess.run(
        ["sips", "-s", "format", "png", str(svg_path), "--out", str(png_path)],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"could not render {svg_path}")


def render_gif(frame_paths: list[Path], output: Path, delay: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "magick",
        "-delay",
        str(delay),
        "-loop",
        "0",
        *[str(path) for path in frame_paths],
        "-layers",
        "Optimize",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "could not assemble GIF")


def main() -> int:
    args = parse_args()
    if shutil.which("sips") is None:
        print("error: macOS `sips` is required to render SVG frames", file=sys.stderr)
        return 1
    if shutil.which("magick") is None:
        print("error: ImageMagick `magick` is required to render the GIF", file=sys.stderr)
        return 1

    try:
        with tempfile.TemporaryDirectory(prefix="reverie-mnist-gif-") as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            run = verified_run(tmpdir, args.quiet_cargo)
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
