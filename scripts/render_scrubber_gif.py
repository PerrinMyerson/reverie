#!/usr/bin/env python3
"""Render the README scrubber GIF from the real Fibonacci timeline."""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "assets" / "reverie-scrub-demo.gif"
CANVAS_WIDTH = 1120
CANVAS_HEIGHT = 640
CODE_LINES = [
    "if n != 0 then",
    "  from i == 0 do",
    "    a += b;",
    "    a <=> b;",
    "    i += 1",
    "  loop",
    "    skip",
    "  until i == n",
    "else",
    "  skip",
    "fi n != 0",
]


@dataclass(frozen=True)
class Step:
    index: int
    total: int
    event: str
    store_text: str
    store: dict[str, str]


TIMELINE_RE = re.compile(r"^\s*(\d+)/(\d+)\s+(.+?)\s+(\{.*\})\s*$")
WATCH_RE = re.compile(r"^watch\s+(\w+):\s*(.*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render docs/assets/reverie-scrub-demo.gif from `reverie scrub --dump`."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="GIF path to write. Defaults to docs/assets/reverie-scrub-demo.gif.",
    )
    parser.add_argument(
        "--frame-delay",
        type=int,
        default=11,
        help="ImageMagick GIF frame delay in 1/100ths of a second.",
    )
    parser.add_argument(
        "--quiet-cargo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --quiet to cargo run when collecting the scrubber dump.",
    )
    return parser.parse_args()


def run_scrub_dump(quiet_cargo: bool) -> str:
    command = ["cargo", "run", "-p", "reverie-cli"]
    if quiet_cargo:
        command.append("--quiet")
    command.extend(
        [
            "--",
            "scrub",
            "examples/fib.rev",
            "--var",
            "n=7",
            "--var",
            "i=0",
            "--var",
            "a=0",
            "--var",
            "b=1",
            "--watch",
            "a",
            "--watch",
            "b",
            "--watch",
            "i",
            "--dump",
        ]
    )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "scrubber dump command failed")
    return completed.stdout


def parse_store(store_text: str) -> dict[str, str]:
    body = store_text.strip()[1:-1].strip()
    if not body:
        return {}
    store = {}
    for field in body.split(","):
        name, value = field.strip().split(" = ", 1)
        store[name] = value
    return store


def parse_dump(text: str) -> tuple[list[Step], dict[str, list[tuple[int, str]]]]:
    steps: list[Step] = []
    watches: dict[str, list[tuple[int, str]]] = {}
    for line in text.splitlines():
        timeline_match = TIMELINE_RE.match(line)
        if timeline_match:
            index = int(timeline_match.group(1))
            total = int(timeline_match.group(2))
            event = timeline_match.group(3).strip()
            store_text = timeline_match.group(4)
            steps.append(
                Step(
                    index=index,
                    total=total,
                    event=event,
                    store_text=store_text,
                    store=parse_store(store_text),
                )
            )
            continue
        watch_match = WATCH_RE.match(line)
        if watch_match:
            name = watch_match.group(1)
            entries = []
            for raw_entry in watch_match.group(2).split(" -> "):
                raw_step, raw_value = raw_entry.split(":", 1)
                entries.append((int(raw_step), raw_value))
            watches[name] = entries
    if not steps:
        raise RuntimeError("scrubber dump did not contain timeline rows")
    return steps, watches


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


def wrap_words(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def wrap_chain(entries: list[tuple[int, str]], max_chars: int) -> list[str]:
    parts = [f"{step}:{value}" for step, value in entries]
    if not parts:
        return [""]
    lines = [parts[0]]
    for part in parts[1:]:
        candidate = f"{lines[-1]} -> {part}"
        if len(candidate) <= max_chars:
            lines[-1] = candidate
        else:
            lines.append(f"-> {part}")
    return lines


def highlight_line(event: str) -> int | None:
    return {
        "if then": 0,
        "loop enter": 1,
        "a +=": 2,
        "a <=> b": 3,
        "i +=": 4,
        "skip": 6,
        "loop repeat": 7,
        "loop exit": 7,
    }.get(event)


def render_code_panel(step: Step) -> list[str]:
    lines = [
        rect(32, 86, 696, 382, fill="#111827", stroke="#3a4960", radius=14),
        svg_text(58, 122, "examples/fib.rev", size=18, fill="#22d3ee", weight=800),
    ]
    active = highlight_line(step.event)
    start_y = 160
    line_height = 28
    for index, code in enumerate(CODE_LINES):
        y = start_y + index * line_height
        if active == index:
            lines.append(
                '<rect x="54" y="{}" width="642" height="28" rx="7" '
                'fill="#facc15"/>'.format(y - 21)
            )
        lines.append(
            svg_text(
                74,
                y,
                f"{index + 1:>2}",
                size=17,
                fill="#8aa0b7",
                weight=500,
            )
        )
        lines.append(
            svg_text(
                126,
                y,
                code,
                size=21,
                fill="#111827" if active == index else "#dbeafe",
                weight=800 if active == index else 650,
            )
        )
    return lines


def render_store_panel(step: Step) -> list[str]:
    lines = [
        rect(752, 86, 336, 178, fill="#111827", stroke="#3a4960", radius=14),
        svg_text(778, 122, "Store", size=18, fill="#34d399", weight=800),
    ]
    for offset, wrapped in enumerate(wrap_words(step.store_text, 33)[:2]):
        lines.append(svg_text(778, 152 + offset * 24, wrapped, size=16, fill="#cbd5e1"))

    tile_specs = [("a", 778, 190), ("b", 856, 190), ("i", 934, 190), ("n", 1012, 190)]
    for name, x, y in tile_specs:
        lines.append(rect(x, y, 54, 52, fill="#172033", stroke="#475569", radius=9))
        lines.append(svg_text(x + 14, y + 21, name, size=15, fill="#a8b6ca", weight=800))
        lines.append(
            svg_text(
                x + 14,
                y + 42,
                step.store.get(name, "?"),
                size=23,
                fill="#ffffff",
                weight=900,
            )
        )
    return lines


def render_watch_panel(step: Step, watches: dict[str, list[tuple[int, str]]]) -> list[str]:
    lines = [
        rect(752, 284, 336, 200, fill="#111827", stroke="#3a4960", radius=14),
        svg_text(778, 322, "Watches", size=20, fill="#34d399", weight=900),
    ]
    colors = {"a": "#22d3ee", "b": "#f472b6", "i": "#34d399"}
    y = 350
    for name in ("a", "b", "i"):
        visible = [(index, value) for index, value in watches.get(name, []) if index <= step.index]
        if not visible:
            visible = [(step.index, step.store.get(name, "?"))]
        lines.append(svg_text(778, y, name, size=17, fill=colors[name], weight=900))
        chain_lines = wrap_chain(visible, 27)[-2:]
        for offset, chain_line in enumerate(chain_lines):
            lines.append(
                svg_text(
                    818,
                    y + offset * 20,
                    chain_line,
                    size=14,
                    fill="#dbeafe",
                    weight=650,
                )
            )
        y += 56 if len(chain_lines) == 2 else 40
    return lines


def render_timeline_panel(step: Step, direction: str) -> list[str]:
    progress_width = 972
    filled = round(progress_width * (step.index / step.total))
    is_reverse = direction != "forward"
    bar_color = "#f472b6" if is_reverse else "#2dd4bf"
    phase_label = (
        "inverse replay: state returns to the start"
        if is_reverse
        else "forward run: state moves to the result"
    )
    lines = [
        rect(32, 496, 1056, 104, fill="#111827", stroke="#3a4960", radius=14),
        svg_text(58, 532, "Timeline", size=19, fill="#f59e0b", weight=900),
        svg_text(
            174,
            532,
            phase_label,
            size=18,
            fill="#dbeafe",
            weight=650,
        ),
        svg_text(
            1048,
            532,
            f"{step.index}/{step.total}",
            size=17,
            fill="#cbd5e1",
            weight=800,
            anchor="end",
        ),
        '<rect x="58" y="554" width="972" height="18" rx="9" fill="#1e293b"/>',
        (
            f'<rect x="58" y="554" width="{max(16, filled)}" '
            f'height="18" rx="9" fill="{bar_color}"/>'
        ),
        svg_text(58, 592, f"event: {step.event}", size=17, fill="#cbd5e1", weight=700),
        svg_text(
            1048,
            592,
            "real scrub --dump timeline",
            size=15,
            fill="#8aa0b7",
            weight=650,
            anchor="end",
        ),
    ]
    return lines


def render_frame(
    step: Step,
    watches: dict[str, list[tuple[int, str]]],
    direction: str,
) -> str:
    is_reverse = direction != "forward"
    phase_text = "reverse: undo back to baseline" if is_reverse else "forward: execute"
    phase_color = "#f472b6" if is_reverse else "#facc15"
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
            "Reverie scrubber",
            size=28,
            fill="#ffffff",
            weight=900,
            family="Inter, Helvetica, Arial, sans-serif",
        ),
        svg_text(
            294,
            46,
            "time-travel debugging for reversible programs",
            size=18,
            fill="#9fb1c7",
            weight=650,
            family="Inter, Helvetica, Arial, sans-serif",
        ),
        svg_text(1088, 46, phase_text, size=19, fill=phase_color, weight=900, anchor="end"),
    ]
    elements.extend(render_code_panel(step))
    elements.extend(render_store_panel(step))
    elements.extend(render_watch_panel(step, watches))
    elements.extend(render_timeline_panel(step, direction))
    elements.append("</svg>")
    return "\n".join(elements)


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
    if shutil.which("magick") is None:
        print("error: ImageMagick `magick` is required to render the README GIF", file=sys.stderr)
        return 1
    if shutil.which("sips") is None:
        print("error: macOS `sips` is required to render SVG frames", file=sys.stderr)
        return 1

    try:
        dump = run_scrub_dump(args.quiet_cargo)
        steps, watches = parse_dump(dump)
        with tempfile.TemporaryDirectory(prefix="reverie-scrub-gif-") as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            frame_paths: list[Path] = []
            frames = (
                [(steps[0], "forward"), (steps[0], "forward")]
                + [(step, "forward") for step in steps]
                + [(steps[-1], "reverse"), (steps[-1], "reverse")]
                + [(step, "reverse") for step in reversed(steps[:-1])]
                + [(steps[0], "reverse"), (steps[0], "reverse")]
            )
            for frame_index, (step, direction) in enumerate(frames):
                svg_path = tmpdir / f"frame-{frame_index:03}.svg"
                png_path = tmpdir / f"frame-{frame_index:03}.png"
                svg_path.write_text(render_frame(step, watches, direction), encoding="utf-8")
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
