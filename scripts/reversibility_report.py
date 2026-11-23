#!/usr/bin/env python3
"""Measure what reversibility costs, in-process, through the Python package.

The Jana comparison (`scripts/bench_jana_vs_reverie.py`) times whole
processes, so for small programs it mostly measures startup. This report
times execution inside one process and asks questions specific to reversible
computing:

- **Reverse / forward ratio:** running backward should cost about the same as
  running forward.
- **Engines:** the slot-compiled engine against the reference tree walker.
- **Cost against an ordinary program:** the same algorithm written as plain,
  irreversible CPython. Both are interpreters, so read this as a reference
  point, not a language shoot-out.
- **Memory to go back in time:** a time-travel debugger that snapshots the
  store after every step keeps `sum(|store|)` values. A reversible program
  keeps only its final store, witnesses included, and recomputes the rest.
- **Scrub cost:** building the step-by-step timeline, per frame.

    python3 scripts/reversibility_report.py --markdown-output docs/research/reversibility-report.md
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
MASK64 = (1 << 64) - 1


def wrap(value: int) -> int:
    value &= MASK64
    return value - (1 << 64) if value >= 1 << 63 else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--scale", type=float, default=1.0, help="workload size multiplier")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--self-test", action="store_true", help="tiny sizes, check invariants")
    return parser.parse_args()


def median_seconds(fn: Callable[[], Any], repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def count_values(value: Any) -> int:
    from reverie.ledger import flatten

    return sum(1 for _ in flatten(value))


# -- workloads ----------------------------------------------------------------


def fib_workload(n: int):
    source = (REPO_ROOT / "examples" / "fib.rev").read_text()
    store = {"n": n, "i": 0, "a": 0, "b": 1}

    def native():
        a, b = 0, 1
        for _ in range(n):
            a = wrap(a + b)
            a, b = b, a
        return a, b

    return "fib pair transform", f"n={n}", source, store, native


def coupling_workload(layers: int, width: int):
    """A RevNet-style stack: `layers` additive coupling blocks, no witness."""
    rng = random.Random(1)
    q = lambda: rng.randint(-(1 << 30), 1 << 30)  # noqa: E731
    source = f"""\
global left: tensor<int, {width}>;
global right: tensor<int, {width}>;
global f_w: tensor<int, {width}, {width}>;
global g_w: tensor<int, {width}, {width}>;
global layers = 0;

procedure main() {{
  iterate int k = 1 to layers
    right += relu(vecmat_q31(left, f_w));
    left += relu(vecmat_q31(right, g_w))
  end
}}
"""
    store = {
        "left": [q() for _ in range(width)],
        "right": [q() for _ in range(width)],
        "f_w": [[q() for _ in range(width)] for _ in range(width)],
        "g_w": [[q() for _ in range(width)] for _ in range(width)],
        "layers": layers,
    }

    def native():
        left, right = list(store["left"]), list(store["right"])
        fw, gw = store["f_w"], store["g_w"]
        for _ in range(layers):
            for j in range(width):
                acc = sum((left[i] * fw[i][j]) >> 31 for i in range(width))
                right[j] = wrap(right[j] + max(acc, 0))
            for j in range(width):
                acc = sum((right[i] * gw[i][j]) >> 31 for i in range(width))
                left[j] = wrap(left[j] + max(acc, 0))
        return left, right

    return "invertible coupling stack", f"{layers} blocks x width {width}", source, store, native


def sgd_workload(samples: int, epochs: int, features: int, hidden: int, classes: int):
    from reverie import lab

    rng = random.Random(2)
    images = [[lab.to_q31(rng.uniform(0, 1)) for _ in range(features)] for _ in range(samples)]
    labels = [rng.randrange(classes) for _ in range(samples)]
    order = [i for _ in range(epochs) for i in range(samples)]
    source = lab.mlp_sgd_source(
        n_samples=samples, n_features=features, hidden=hidden, classes=classes, steps=len(order)
    )
    q = lambda s: lab.to_q31(rng.gauss(0, s))  # noqa: E731
    store = {
        "images": images,
        "labels": labels,
        "order": order,
        "active": [1] * len(order),
        "start": 0,
        "stop": len(order) - 1,
        "lr": lab.to_q31(0.03),
        "w1": [[q(0.3) for _ in range(hidden)] for _ in range(features)],
        "b1": [0] * hidden,
        "w2": [[q(0.3) for _ in range(classes)] for _ in range(hidden)],
        "b2": [0] * classes,
    }

    def fm(a, b):
        return (a * b) >> 31

    def native():
        w1 = [row[:] for row in store["w1"]]
        b1 = store["b1"][:]
        w2 = [row[:] for row in store["w2"]]
        b2 = store["b2"][:]
        lr = store["lr"]
        for s in order:
            x, y = images[s], labels[s]
            pre = [sum(fm(x[i], w1[i][j]) for i in range(features)) + b1[j] for j in range(hidden)]
            act = [max(v, 0) for v in pre]
            logits = [sum(fm(act[i], w2[i][j]) for i in range(hidden)) + b2[j] for j in range(classes)]
            err = [logits[j] - ((1 << 31) if j == y else 0) for j in range(classes)]
            back = [sum(fm(w2[i][j], err[j]) for j in range(classes)) for i in range(hidden)]
            delta = [fm(back[i], (1 << 31) if pre[i] > 0 else 0) for i in range(hidden)]
            for i in range(hidden):
                for j in range(classes):
                    w2[i][j] -= fm(fm(act[i], err[j]), lr)
            for j in range(classes):
                b2[j] -= fm(err[j], lr)
            for i in range(features):
                for j in range(hidden):
                    w1[i][j] -= fm(fm(x[i], delta[j]), lr)
            for j in range(hidden):
                b1[j] -= fm(delta[j], lr)
        return w1, b1, w2, b2

    def check(final):
        w1, b1, w2, b2 = native()
        return final["w1"] == w1 and final["b1"] == b1 and final["w2"] == w2 and final["b2"] == b2

    native.check = check  # type: ignore[attr-defined]
    return (
        "minimal-witness MLP SGD",
        f"{features}->{hidden}->{classes}, {len(order)} steps",
        source,
        store,
        native,
    )


def example_workload(name: str):
    source = (REPO_ROOT / "examples" / f"{name}.rev").read_text()
    return f"examples/{name}.rev", "as shipped", source, {}, None


# -- measurement --------------------------------------------------------------


def measure(workload, repeats: int, timeline_limit: int) -> Dict[str, Any]:
    import reverie

    title, size, source, store, native = workload
    slot = reverie.loads(source, engine="slot")
    tree = reverie.loads(source, engine="tree")
    final = slot.run(store)
    restored = slot.reverse(final)
    roundtrip_ok = _same_inputs(restored, store)

    row: Dict[str, Any] = {
        "workload": title,
        "size": size,
        "forward_slot_s": median_seconds(lambda: slot.run(store), repeats),
        "reverse_slot_s": median_seconds(lambda: slot.reverse(final), repeats),
        "forward_tree_s": median_seconds(lambda: tree.run(store), repeats),
        "roundtrip_restores_inputs": roundtrip_ok,
        "final_store_values": count_values(final),
    }
    row["reverse_over_forward"] = row["reverse_slot_s"] / row["forward_slot_s"]
    row["tree_over_slot"] = row["forward_tree_s"] / row["forward_slot_s"]
    if native is not None:
        row["native_python_s"] = median_seconds(native, max(1, repeats // 2))
        row["slot_over_native_python"] = row["forward_slot_s"] / row["native_python_s"]
        check = getattr(native, "check", None)
        if check is not None:
            row["matches_native_bit_for_bit"] = check(final)

    if timeline_limit:
        started = time.perf_counter()
        frames = slot.timeline(store)
        elapsed = time.perf_counter() - started
        trace_values = sum(count_values(frame.store) for frame in frames)
        row.update(
            {
                "timeline_frames": len(frames),
                "scrub_us_per_frame": elapsed / max(len(frames), 1) * 1e6,
                "snapshot_trace_values": trace_values,
                "snapshot_over_reversible": trace_values / max(row["final_store_values"], 1),
            }
        )
    return row


def _same_inputs(restored: Dict[str, Any], store: Dict[str, Any]) -> bool:
    return all(restored.get(name) == value for name, value in store.items())


def workloads(scale: float, self_test: bool):
    s = lambda n: max(1, int(n * scale))  # noqa: E731
    if self_test:
        return [
            (fib_workload(50), 1),
            (coupling_workload(3, 4), 1),
            (sgd_workload(4, 2, 5, 4, 3), 1),
            (example_workload("janus_turing"), 1),
        ]
    return [
        (fib_workload(s(200_000)), 0),
        (fib_workload(s(2_000)), 1),
        (coupling_workload(s(200), 16), 0),
        (coupling_workload(s(20), 16), 1),
        (sgd_workload(s(100), 2, 64, 32, 10), 0),
        (sgd_workload(s(10), 2, 64, 32, 10), 1),
        (example_workload("janus_turing"), 1),
        (example_workload("triangular_residual"), 1),
        (example_workload("mnist_mlp_witness"), 1),
        (example_workload("reversible_preprocess"), 1),
    ]


def markdown(rows: List[Dict[str, Any]], host: Dict[str, str]) -> str:
    ms = lambda s: f"{s * 1e3:,.2f}"  # noqa: E731
    lines = [
        "# Reversibility report",
        "",
        f"Generated by `scripts/reversibility_report.py` ({host['platform']}, Python {host['python']}). "
        "All times are in-process medians, so CLI startup is excluded.",
        "",
        "## Speed",
        "",
        "| workload | size | forward (ms) | reverse (ms) | reverse / forward | tree / slot | plain CPython (ms) | slot / CPython |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if "timeline_frames" in r and "native_python_s" not in r and r["size"] != "as shipped":
            continue
        native = ms(r["native_python_s"]) if "native_python_s" in r else "-"
        ratio = f"{r['slot_over_native_python']:.2f}x" if "native_python_s" in r else "-"
        lines.append(
            f"| {r['workload']} | {r['size']} | {ms(r['forward_slot_s'])} | {ms(r['reverse_slot_s'])} | "
            f"{r['reverse_over_forward']:.2f} | {r['tree_over_slot']:.2f} | {native} | {ratio} |"
        )
    lines += [
        "",
        "## Memory to travel back in time",
        "",
        "| workload | size | frames | snapshot trace (values) | reversible state (values) | ratio | scrub (us/frame) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if "timeline_frames" not in r:
            continue
        lines.append(
            f"| {r['workload']} | {r['size']} | {r['timeline_frames']:,} | {r['snapshot_trace_values']:,} | "
            f"{r['final_store_values']:,} | {r['snapshot_over_reversible']:,.0f}x | {r['scrub_us_per_frame']:.1f} |"
        )
    bad = [r["workload"] for r in rows if not r["roundtrip_restores_inputs"]]
    mismatched = [r["workload"] for r in rows if r.get("matches_native_bit_for_bit") is False]
    lines += [
        "",
        f"Every workload ran backward to its exact inputs: **{not bad}**."
        + (f" Failures: {', '.join(bad)}." if bad else ""),
        "The SGD workload's final weights match the plain CPython implementation bit for bit: "
        f"**{not mismatched}**.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    host = {"platform": platform.platform(), "python": platform.python_version()}
    rows = [
        measure(workload, 1 if args.self_test else args.repeats, timeline)
        for workload, timeline in workloads(args.scale, args.self_test)
    ]
    if args.self_test:
        assert all(r["roundtrip_restores_inputs"] for r in rows), rows
        assert all(r.get("matches_native_bit_for_bit", True) for r in rows), rows
        print("ok: reversibility report self-test passed")
        return 0
    text = markdown(rows, host)
    if args.json_output:
        args.json_output.write_text(json.dumps({"host": host, "rows": rows}, indent=2) + "\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
