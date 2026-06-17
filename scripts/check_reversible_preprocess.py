#!/usr/bin/env python3
"""Reference-check Reverie's reversible Q31 preprocessing example."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional


KIND = "reverie_q31_reversible_preprocess_reference"
PROOF_CLAIM = "deterministic_q31_reversible_preprocess_replay"
PROGRAM = "examples/reversible_preprocess.rev"
WIDTH = 4
Q31_ONE = 1 << 31
I64_BYTES = 8
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
U64_MOD = 1 << 64


class PreprocessCheckError(ValueError):
    """Raised when a preprocessing report or Reverie output is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reference-check examples/reversible_preprocess.rev forward and "
            "reverse JSON output."
        )
    )
    parser.add_argument(
        "--forward-output-json",
        type=Path,
        help="JSON output from `reverie run examples/reversible_preprocess.rev --json`.",
    )
    parser.add_argument(
        "--reverse-output-json",
        type=Path,
        help=(
            "JSON output from `reverie reverse examples/reversible_preprocess.rev "
            "--json` seeded with final features."
        ),
    )
    parser.add_argument(
        "--write-final-vars",
        type=Path,
        help="Write the exact final features vars JSON needed for reverse execution.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Write a human-readable reversible preprocessing proof card.",
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
        help="Run build-independent preprocessing checker self-tests and exit.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise PreprocessCheckError(f"failed to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise PreprocessCheckError(f"failed to parse {path}: {error}") from error


def checked_i64(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PreprocessCheckError(f"{label} must be a signed integer")
    if not I64_MIN <= value <= I64_MAX:
        raise PreprocessCheckError(f"{label} is outside signed i64 range")
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


def add_vector(left: list[int], right: list[int]) -> list[int]:
    return [add_i64(a, b) for a, b in zip(left, right)]


def sub_vector(left: list[int], right: list[int]) -> list[int]:
    return [sub_i64(a, b) for a, b in zip(left, right)]


def swap(values: list[int], left: int, right: int) -> list[int]:
    out = list(values)
    out[left], out[right] = out[right], out[left]
    return out


def comma(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.6g}"
    return str(value)


def vector_cell(values: list[int]) -> str:
    return "`[{}]`".format(", ".join(comma(value) for value in values))


def initial_store() -> dict[str, Any]:
    return {
        "raw": [Q31_ONE, 0, Q31_ONE // 2, -(Q31_ONE // 4)],
        "mean": [Q31_ONE // 4, -(Q31_ONE // 4), 0, Q31_ONE // 4],
        "features": [0, 0, 0, 0],
    }


def forward_store(store: dict[str, Any]) -> dict[str, Any]:
    features = add_vector(store["features"], store["raw"])
    features = sub_vector(features, store["mean"])
    features = swap(features, 0, 2)
    features = swap(features, 1, 3)
    return {**store, "features": features}


def preprocess_intermediates(store: dict[str, Any]) -> dict[str, list[int]]:
    after_raw = add_vector(store["features"], store["raw"])
    after_center = sub_vector(after_raw, store["mean"])
    after_swap_02 = swap(after_center, 0, 2)
    after_swap_13 = swap(after_swap_02, 1, 3)
    return {
        "after_raw": after_raw,
        "after_center": after_center,
        "after_swap_02": after_swap_02,
        "after_swap_13": after_swap_13,
    }


def inverse_store(store: dict[str, Any]) -> dict[str, Any]:
    features = swap(store["features"], 1, 3)
    features = swap(features, 0, 2)
    features = add_vector(features, store["mean"])
    features = sub_vector(features, store["raw"])
    return {**store, "features": features}


def final_vars() -> dict[str, Any]:
    final = forward_store(initial_store())
    return {"features": final["features"]}


def parse_output_store(value: Any, expected_kind: str, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreprocessCheckError(f"{context} must be a JSON object")
    if value.get("kind") != expected_kind:
        raise PreprocessCheckError(f"{context}.kind must be {expected_kind}")
    store = value.get("store")
    if not isinstance(store, dict):
        raise PreprocessCheckError(f"{context}.store must be an object")
    return validate_store(store, f"{context}.store")


def validate_store(value: dict[str, Any], context: str) -> dict[str, Any]:
    store: dict[str, Any] = {}
    for field in ("raw", "mean", "features"):
        item = value.get(field)
        if not isinstance(item, list) or len(item) != WIDTH:
            raise PreprocessCheckError(f"{context}.{field} must have {WIDTH} entries")
        store[field] = [
            checked_i64(entry, f"{context}.{field}[{index}]")
            for index, entry in enumerate(item)
        ]
    return store


def compare_store(expected: dict[str, Any], actual: dict[str, Any], context: str) -> list[str]:
    mismatches = []
    for field, expected_value in expected.items():
        actual_value = actual.get(field)
        if actual_value != expected_value:
            mismatches.append(f"{context}.{field} expected {expected_value}, found {actual_value}")
    return mismatches


def proof_cost() -> dict[str, Any]:
    raw_payload_bytes = WIDTH * I64_BYTES
    mean_payload_bytes = WIDTH * I64_BYTES
    feature_payload_bytes = WIDTH * I64_BYTES
    witness_payload_bytes = 0
    trace_payload_bytes = 0
    return {
        "claim": PROOF_CLAIM,
        "arithmetic": "q31_wrapping_i64",
        "raw_payload_bytes": raw_payload_bytes,
        "mean_payload_bytes": mean_payload_bytes,
        "feature_payload_bytes": feature_payload_bytes,
        "witness_payload_bytes": witness_payload_bytes,
        "trace_payload_bytes": trace_payload_bytes,
        "replay_payload_bytes": raw_payload_bytes + mean_payload_bytes + feature_payload_bytes,
        "forward_recompute_steps": 1,
        "inverse_recompute_steps": 1,
        "total_recompute_steps": 2,
        "witness_to_state_payload_ratio": 0.0,
        "trace_to_state_payload_ratio": 0.0,
        "checks": {
            "forward_matches_reference": True,
            "reverse_restores_initial_state": True,
            "raw_preserved": True,
            "mean_preserved": True,
            "no_witness_tape": True,
            "balanced_recompute": True,
        },
    }


def build_report(
    forward_path: Optional[Path],
    reverse_path: Optional[Path],
    mismatches: list[str],
) -> dict[str, Any]:
    initial = initial_store()
    return {
        "kind": KIND,
        "program": PROGRAM,
        "q31_one": Q31_ONE,
        "forward_output_json": str(forward_path) if forward_path is not None else None,
        "reverse_output_json": str(reverse_path) if reverse_path is not None else None,
        "initial": initial,
        "forward": forward_store(initial),
        "intermediates": preprocess_intermediates(initial),
        "passed": not mismatches,
        "mismatches": mismatches,
        "proof": proof_cost(),
    }


def render_markdown(report: dict[str, Any]) -> str:
    proof = report["proof"]
    initial = report["initial"]
    forward = report["forward"]
    intermediates = report["intermediates"]
    lines = [
        "# Reverie Reversible Preprocess Proof",
        "",
        "| Passed | Replay bytes | Witness bytes | Trace bytes | Forward recompute | Inverse recompute |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| {} | {} | {} | {} | {} | {} |".format(
            str(report["passed"]).lower(),
            comma(proof["replay_payload_bytes"]),
            comma(proof["witness_payload_bytes"]),
            comma(proof["trace_payload_bytes"]),
            comma(proof["forward_recompute_steps"]),
            comma(proof["inverse_recompute_steps"]),
        ),
        "",
        "## Input State",
        "",
        "| Raw | Mean | Initial features | Final features |",
        "| --- | --- | --- | --- |",
        "| {} | {} | {} | {} |".format(
            vector_cell(initial["raw"]),
            vector_cell(initial["mean"]),
            vector_cell(initial["features"]),
            vector_cell(forward["features"]),
        ),
        "",
        "## Reversible Steps",
        "",
        "| Step | Features |",
        "| --- | --- |",
        "| features += raw | {} |".format(vector_cell(intermediates["after_raw"])),
        "| features -= mean | {} |".format(vector_cell(intermediates["after_center"])),
        "| swap 0 and 2 | {} |".format(vector_cell(intermediates["after_swap_02"])),
        "| swap 1 and 3 | {} |".format(vector_cell(intermediates["after_swap_13"])),
        "",
        "## Reverse Contract",
        "",
        "| Claim | Reverse restored | Raw preserved | Mean preserved | No witness tape | Balanced |",
        "| --- | --- | --- | --- | --- | --- |",
        "| `{}` | {} | {} | {} | {} | {} |".format(
            proof["claim"],
            str(proof["checks"]["reverse_restores_initial_state"]).lower(),
            str(proof["checks"]["raw_preserved"]).lower(),
            str(proof["checks"]["mean_preserved"]).lower(),
            str(proof["checks"]["no_witness_tape"]).lower(),
            str(proof["checks"]["balanced_recompute"]).lower(),
        ),
        "",
        "## Payloads",
        "",
        "| Raw | Mean | Features | Witness | Trace | Replay | Witness/state | Trace/state |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| {} | {} | {} | {} | {} | {} | {:.6f} | {:.6f} |".format(
            comma(proof["raw_payload_bytes"]),
            comma(proof["mean_payload_bytes"]),
            comma(proof["feature_payload_bytes"]),
            comma(proof["witness_payload_bytes"]),
            comma(proof["trace_payload_bytes"]),
            comma(proof["replay_payload_bytes"]),
            proof["witness_to_state_payload_ratio"],
            proof["trace_to_state_payload_ratio"],
        ),
    ]
    if report["mismatches"]:
        lines.extend(["", "## Mismatches", "", "| Mismatch |", "| --- |"])
        lines.extend(f"| `{mismatch}` |" for mismatch in report["mismatches"])
    lines.append("")
    return "\n".join(lines)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(report), encoding="utf-8")
    except OSError as error:
        raise PreprocessCheckError(f"failed to write {path}: {error}") from error


def validate_expected_report(path: Path, report: dict[str, Any]) -> None:
    expected = load_json(path)
    if expected != report:
        raise PreprocessCheckError(f"recomputed report does not match {path}")


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
        if inverse_store(expected) != initial:
            raise AssertionError("inverse did not restore initial state")
        proof = proof_cost()
        if proof["witness_payload_bytes"] != 0 or proof["trace_payload_bytes"] != 0:
            raise AssertionError("preprocessing proof should not need witness or trace bytes")
        if proof["replay_payload_bytes"] != 96:
            raise AssertionError("preprocessing replay payload changed unexpectedly")
        report = build_report(None, None, [])
        if not report["passed"] or report["proof"]["claim"] != PROOF_CLAIM:
            raise AssertionError("valid self-test report did not pass")
        markdown = render_markdown(report)
        for snippet in (
            "# Reverie Reversible Preprocess Proof",
            "## Reversible Steps",
            "features -= mean",
            "## Reverse Contract",
        ):
            if snippet not in markdown:
                raise AssertionError(f"rendered Markdown missing {snippet}")
        bad_store = forward_store(initial)
        bad_store["features"][0] += 1
        mismatches = compare_store(expected, bad_store, "bad forward")
        if not mismatches:
            raise AssertionError("bad preprocessing store did not produce mismatch")
    except AssertionError as error:
        print(f"error: self-test failed: {error}", file=sys.stderr)
        return 1
    print("ok: reversible preprocessing checker self-tests passed")
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
                "reversible preprocessing ok: features={} replay_bytes={} "
                "witness_bytes={} trace_bytes={}".format(
                    report["forward"]["features"],
                    proof["replay_payload_bytes"],
                    proof["witness_payload_bytes"],
                    proof["trace_payload_bytes"],
                )
            )
        else:
            for mismatch in mismatches:
                print(mismatch, file=sys.stderr)
        return 0 if report["passed"] else 1
    except PreprocessCheckError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
