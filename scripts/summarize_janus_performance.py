#!/usr/bin/env python3
"""Render a checked Janus-vs-Reverie benchmark JSON artifact as Markdown."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "benchmarks" / "results" / "jana-vs-reverie-smoke.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Janus-vs-Reverie benchmark JSON artifact as Markdown."
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--min-observed-speedup",
        type=float,
        default=None,
        help="Report a required floor for every observed workload speedup.",
    )
    parser.add_argument(
        "--min-median-speedup",
        type=float,
        default=None,
        help="Report a required floor for the suite median observed speedup.",
    )
    parser.add_argument(
        "--min-geomean-speedup",
        type=float,
        default=None,
        help="Report a required floor for the geometric mean observed speedup.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent Markdown renderer self-tests and exit.",
    )
    return parser.parse_args()


def ms(seconds: float) -> str:
    return f"{seconds * 1000:.3f} ms"


def number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def metadata_line(data: dict[str, Any]) -> str:
    runs = data.get("runs", "?")
    warmup = data.get("warmup", "?")
    min_speedup = data.get("min_speedup")
    generated_at = data.get("generated_at", "unknown time")
    gate = f", min speedup {min_speedup:.2f}x" if number(min_speedup) else ""
    return f"Generated at `{generated_at}` with `{runs}` runs, `{warmup}` warmup{gate}."


def workload_direction(name: str) -> str:
    if name.endswith("_roundtrip"):
        return "roundtrip"
    if name.endswith("_reverse"):
        return "reverse"
    return "forward"


def benchmark_direction(bench: dict[str, Any]) -> str:
    direction = bench.get("direction")
    if direction in {"forward", "reverse", "roundtrip"}:
        return direction
    return workload_direction(bench["name"])


def gate_value(data: dict[str, Any], key: str) -> Optional[float]:
    gate = data.get("performance_gate")
    if not isinstance(gate, dict):
        return None
    value = gate.get(key)
    if number(value):
        return float(value)
    return None


def render_markdown(
    data: dict[str, Any],
    min_observed_speedup: Optional[float] = None,
    min_median_speedup: Optional[float] = None,
    min_geomean_speedup: Optional[float] = None,
) -> str:
    if min_observed_speedup is None:
        min_observed_speedup = gate_value(data, "min_observed_speedup")
    if min_median_speedup is None:
        min_median_speedup = gate_value(data, "min_median_speedup")
    if min_geomean_speedup is None:
        min_geomean_speedup = gate_value(data, "min_geomean_speedup")
    benchmarks = data["benchmarks"]
    speedups = [bench["speedup"] for bench in benchmarks]
    observed_min = min(speedups)
    observed_median = statistics.median(speedups)
    observed_geomean = statistics.geometric_mean(speedups)
    weakest = min(benchmarks, key=lambda bench: bench["speedup"])
    strongest = max(benchmarks, key=lambda bench: bench["speedup"])
    direction_counts = {"forward": 0, "reverse": 0, "roundtrip": 0}
    for bench in benchmarks:
        direction_counts[benchmark_direction(bench)] += 1
    failed = [
        bench["name"]
        for bench in benchmarks
        if bench.get("passes_min_speedup") is False
    ]
    aggregate_failed = (
        min_observed_speedup is not None and observed_min < min_observed_speedup
    ) or (
        min_median_speedup is not None and observed_median < min_median_speedup
    ) or (
        min_geomean_speedup is not None and observed_geomean < min_geomean_speedup
    )

    median_line = f"Median speedup: `{observed_median:.2f}x`"
    if min_median_speedup is not None:
        median_line += f" (required `{min_median_speedup:.2f}x`)"
    geomean_line = f"Geometric mean speedup: `{observed_geomean:.2f}x`"
    if min_geomean_speedup is not None:
        geomean_line += f" (required `{min_geomean_speedup:.2f}x`)"
    range_line = f"Speedup range: `{observed_min:.2f}x` to `{max(speedups):.2f}x`"
    if min_observed_speedup is not None:
        range_line += f" (minimum required `{min_observed_speedup:.2f}x`)"

    lines = [
        "# Jana vs Reverie Benchmark Summary",
        "",
        metadata_line(data),
        "",
        f"Workloads: `{len(benchmarks)}`",
        range_line,
        median_line,
        geomean_line,
        (
            "Directions: "
            f"`{direction_counts['forward']}` forward, "
            f"`{direction_counts['reverse']}` reverse, "
            f"`{direction_counts['roundtrip']}` roundtrip"
        ),
        f"Weakest workload: `{weakest['name']}` at `{weakest['speedup']:.2f}x`",
        f"Strongest workload: `{strongest['name']}` at `{strongest['speedup']:.2f}x`",
        f"Gate: `{'PASS' if not failed and not aggregate_failed else 'FAIL'}`",
    ]
    if failed:
        lines.append(f"Failing workloads: `{', '.join(failed)}`")
    if min_observed_speedup is not None and observed_min < min_observed_speedup:
        lines.append(
            "Minimum observed speedup failure: "
            f"`{observed_min:.2f}x` is below `{min_observed_speedup:.2f}x`"
        )
    if min_median_speedup is not None and observed_median < min_median_speedup:
        lines.append(
            "Median speedup failure: "
            f"`{observed_median:.2f}x` is below `{min_median_speedup:.2f}x`"
        )
    if min_geomean_speedup is not None and observed_geomean < min_geomean_speedup:
        lines.append(
            "Geometric mean speedup failure: "
            f"`{observed_geomean:.2f}x` is below `{min_geomean_speedup:.2f}x`"
        )
    lines.extend(
        [
            "",
            "| workload | Jana median | Reverie median | speedup |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for bench in benchmarks:
        lines.append(
            "| `{name}` | {jana} | {reverie} | {speedup:.2f}x |".format(
                name=bench["name"],
                jana=ms(bench["jana"]["median_seconds"]),
                reverie=ms(bench["reverie"]["median_seconds"]),
                speedup=bench["speedup"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def synthetic_benchmark(
    name: str,
    speedup: float,
    *,
    direction: Optional[str] = None,
    passes_min_speedup: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "direction": direction or workload_direction(name),
        "jana": {"median_seconds": 0.010 * speedup},
        "reverie": {"median_seconds": 0.010},
        "speedup": speedup,
        "passes_min_speedup": passes_min_speedup,
    }


def synthetic_summary_artifact() -> dict[str, Any]:
    return {
        "generated_at": "2026-05-30T00:00:00Z",
        "runs": 5,
        "warmup": 1,
        "min_speedup": 1.25,
        "performance_gate": {
            "min_observed_speedup": 1.5,
            "min_median_speedup": 2.0,
            "min_geomean_speedup": 2.0,
        },
        "benchmarks": [
            synthetic_benchmark("loop_forward", 2.0, direction="forward"),
            synthetic_benchmark("loop_reverse", 4.0, direction="reverse"),
            synthetic_benchmark("loop_roundtrip", 8.0, direction="roundtrip"),
        ],
    }


def expect_contains(label: str, text: str, snippet: str) -> None:
    if snippet not in text:
        raise AssertionError(f"{label}: missing {snippet!r}")


def expect_not_contains(label: str, text: str, snippet: str) -> None:
    if snippet in text:
        raise AssertionError(f"{label}: unexpected {snippet!r}")


def run_self_tests() -> int:
    passing = render_markdown(synthetic_summary_artifact())
    expect_contains("passing artifact", passing, "Gate: `PASS`")
    expect_contains(
        "passing artifact",
        passing,
        "Directions: `1` forward, `1` reverse, `1` roundtrip",
    )
    expect_contains(
        "artifact gate",
        passing,
        "Speedup range: `2.00x` to `8.00x` (minimum required `1.50x`)",
    )
    expect_contains(
        "artifact gate",
        passing,
        "Median speedup: `4.00x` (required `2.00x`)",
    )
    expect_contains(
        "artifact gate",
        passing,
        "Geometric mean speedup: `4.00x` (required `2.00x`)",
    )
    expect_contains(
        "table row",
        passing,
        "| `loop_reverse` | 40.000 ms | 10.000 ms | 4.00x |",
    )

    fallback_direction_artifact = synthetic_summary_artifact()
    for bench in fallback_direction_artifact["benchmarks"]:
        del bench["direction"]
    fallback_direction = render_markdown(fallback_direction_artifact)
    expect_contains(
        "direction fallback",
        fallback_direction,
        "Directions: `1` forward, `1` reverse, `1` roundtrip",
    )

    overridden = render_markdown(
        synthetic_summary_artifact(),
        min_observed_speedup=3.0,
        min_median_speedup=5.0,
        min_geomean_speedup=5.0,
    )
    expect_contains("override failure", overridden, "Gate: `FAIL`")
    expect_contains(
        "override failure",
        overridden,
        "Minimum observed speedup failure: `2.00x` is below `3.00x`",
    )
    expect_contains(
        "override failure",
        overridden,
        "Median speedup failure: `4.00x` is below `5.00x`",
    )
    expect_contains(
        "override failure",
        overridden,
        "Geometric mean speedup failure: `4.00x` is below `5.00x`",
    )

    workload_failure_artifact = synthetic_summary_artifact()
    workload_failure_artifact["benchmarks"][1]["passes_min_speedup"] = False
    workload_failure = render_markdown(workload_failure_artifact)
    expect_contains("workload failure", workload_failure, "Gate: `FAIL`")
    expect_contains(
        "workload failure",
        workload_failure,
        "Failing workloads: `loop_reverse`",
    )

    bool_min_speedup_artifact = synthetic_summary_artifact()
    bool_min_speedup_artifact["min_speedup"] = True
    bool_min_speedup = render_markdown(bool_min_speedup_artifact)
    expect_not_contains("bool min_speedup", bool_min_speedup, "min speedup Truex")

    print("ok: benchmark summary renderer self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    if args.min_observed_speedup is not None and args.min_observed_speedup <= 0:
        raise SystemExit("error: --min-observed-speedup must be positive")
    if args.min_median_speedup is not None and args.min_median_speedup <= 0:
        raise SystemExit("error: --min-median-speedup must be positive")
    if args.min_geomean_speedup is not None and args.min_geomean_speedup <= 0:
        raise SystemExit("error: --min-geomean-speedup must be positive")
    data = json.loads(args.input.read_text(encoding="utf-8"))
    markdown = render_markdown(
        data,
        min_observed_speedup=args.min_observed_speedup,
        min_median_speedup=args.min_median_speedup,
        min_geomean_speedup=args.min_geomean_speedup,
    )
    if args.output is None:
        print(markdown, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
