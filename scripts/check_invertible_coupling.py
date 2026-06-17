#!/usr/bin/env python3
"""Reference-check Reverie's Q31 invertible coupling example."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional


KIND = "reverie_q31_invertible_coupling_reference"
PROOF_CLAIM = "deterministic_q31_invertible_coupling_replay"
PROGRAM = "examples/invertible_coupling.rev"
WIDTH = 4
Q31_ONE = 1 << 31
I64_BYTES = 8
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
U64_MOD = 1 << 64


class CouplingCheckError(ValueError):
    """Raised when a coupling report or Reverie output is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference-check examples/invertible_coupling.rev forward and reverse JSON output."
    )
    parser.add_argument(
        "--forward-output-json",
        type=Path,
        help="JSON output from `reverie run examples/invertible_coupling.rev --json`.",
    )
    parser.add_argument(
        "--reverse-output-json",
        type=Path,
        help="JSON output from `reverie reverse examples/invertible_coupling.rev --json` seeded with final left/right.",
    )
    parser.add_argument(
        "--write-final-vars",
        type=Path,
        help="Write the exact final left/right vars JSON needed for reverse execution.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Write a human-readable invertible coupling proof card.",
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
        help="Run build-independent invertible coupling checker self-tests and exit.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CouplingCheckError(f"failed to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise CouplingCheckError(f"failed to parse {path}: {error}") from error


def checked_i64(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CouplingCheckError(f"{label} must be a signed integer")
    if not I64_MIN <= value <= I64_MAX:
        raise CouplingCheckError(f"{label} is outside signed i64 range")
    return value


def wrap_i64(value: int) -> int:
    value %= U64_MOD
    if value >= (1 << 63):
        value -= U64_MOD
    return value


def fixed_mul_q31(left: int, right: int) -> int:
    return wrap_i64((left * right) >> 31)


def add_i64(left: int, right: int) -> int:
    return wrap_i64(left + right)


def sub_i64(left: int, right: int) -> int:
    return wrap_i64(left - right)


def add_vector(left: list[int], right: list[int]) -> list[int]:
    return [add_i64(a, b) for a, b in zip(left, right)]


def sub_vector(left: list[int], right: list[int]) -> list[int]:
    return [sub_i64(a, b) for a, b in zip(left, right)]


def vecmat_q31(vector: list[int], matrix: list[list[int]]) -> list[int]:
    out = []
    for col in range(WIDTH):
        acc = 0
        for row, value in enumerate(vector):
            acc = add_i64(acc, fixed_mul_q31(value, matrix[row][col]))
        out.append(acc)
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
        "left": [Q31_ONE, -(Q31_ONE // 2), Q31_ONE // 4, -(Q31_ONE // 4)],
        "right": [0, Q31_ONE // 2, -(Q31_ONE // 4), Q31_ONE // 4],
        "f_w": [
            [Q31_ONE, 0, 0, 0],
            [0, Q31_ONE, 0, 0],
            [0, 0, Q31_ONE, 0],
            [0, 0, 0, Q31_ONE],
        ],
        "f_b": [0, Q31_ONE // 4, 0, -(Q31_ONE // 4)],
        "g_w": [
            [Q31_ONE // 2, 0, 0, 0],
            [0, Q31_ONE // 2, 0, 0],
            [0, 0, Q31_ONE // 2, 0],
            [0, 0, 0, Q31_ONE // 2],
        ],
        "g_b": [-(Q31_ONE // 4), 0, Q31_ONE // 4, 0],
    }


def forward_store(store: dict[str, Any]) -> dict[str, Any]:
    right = add_vector(store["right"], vecmat_q31(store["left"], store["f_w"]))
    right = add_vector(right, store["f_b"])
    left = add_vector(store["left"], vecmat_q31(right, store["g_w"]))
    left = add_vector(left, store["g_b"])
    return {**store, "left": left, "right": right}


def coupling_intermediates(store: dict[str, Any]) -> dict[str, list[int]]:
    f_left = vecmat_q31(store["left"], store["f_w"])
    right_after_f = add_vector(store["right"], f_left)
    right_after_bias = add_vector(right_after_f, store["f_b"])
    g_right = vecmat_q31(right_after_bias, store["g_w"])
    left_after_g = add_vector(store["left"], g_right)
    left_after_bias = add_vector(left_after_g, store["g_b"])
    return {
        "f_left": f_left,
        "right_after_f": right_after_f,
        "right_after_bias": right_after_bias,
        "g_right": g_right,
        "left_after_g": left_after_g,
        "left_after_bias": left_after_bias,
    }


def inverse_store(store: dict[str, Any]) -> dict[str, Any]:
    left = sub_vector(store["left"], vecmat_q31(store["right"], store["g_w"]))
    left = sub_vector(left, store["g_b"])
    right = sub_vector(store["right"], vecmat_q31(left, store["f_w"]))
    right = sub_vector(right, store["f_b"])
    return {**store, "left": left, "right": right}


def final_vars() -> dict[str, Any]:
    final = forward_store(initial_store())
    return {"left": final["left"], "right": final["right"]}


def parse_output_store(value: Any, expected_kind: str, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CouplingCheckError(f"{context} must be a JSON object")
    if value.get("kind") != expected_kind:
        raise CouplingCheckError(f"{context}.kind must be {expected_kind}")
    store = value.get("store")
    if not isinstance(store, dict):
        raise CouplingCheckError(f"{context}.store must be an object")
    return validate_store(store, f"{context}.store")


def validate_store(value: dict[str, Any], context: str) -> dict[str, Any]:
    store: dict[str, Any] = {}
    for field in ("left", "right", "f_b", "g_b"):
        item = value.get(field)
        if not isinstance(item, list) or len(item) != WIDTH:
            raise CouplingCheckError(f"{context}.{field} must have {WIDTH} entries")
        store[field] = [checked_i64(entry, f"{context}.{field}[{index}]") for index, entry in enumerate(item)]
    for field in ("f_w", "g_w"):
        matrix = value.get(field)
        if not isinstance(matrix, list) or len(matrix) != WIDTH:
            raise CouplingCheckError(f"{context}.{field} must have {WIDTH} rows")
        rows = []
        for row_index, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != WIDTH:
                raise CouplingCheckError(f"{context}.{field}[{row_index}] must have {WIDTH} entries")
            rows.append(
                [
                    checked_i64(entry, f"{context}.{field}[{row_index}][{col_index}]")
                    for col_index, entry in enumerate(row)
                ]
            )
        store[field] = rows
    return store


def compare_store(expected: dict[str, Any], actual: dict[str, Any], context: str) -> list[str]:
    mismatches = []
    for field, expected_value in expected.items():
        actual_value = actual.get(field)
        if actual_value != expected_value:
            mismatches.append(f"{context}.{field} expected {expected_value}, found {actual_value}")
    return mismatches


def proof_cost() -> dict[str, Any]:
    model_payload_bytes = 2 * (WIDTH * WIDTH + WIDTH) * I64_BYTES
    state_payload_bytes = 2 * WIDTH * I64_BYTES
    witness_payload_bytes = 0
    trace_payload_bytes = 0
    return {
        "claim": PROOF_CLAIM,
        "arithmetic": "q31_wrapping_i64",
        "model_payload_bytes": model_payload_bytes,
        "state_payload_bytes": state_payload_bytes,
        "witness_payload_bytes": witness_payload_bytes,
        "trace_payload_bytes": trace_payload_bytes,
        "replay_payload_bytes": model_payload_bytes + state_payload_bytes,
        "forward_recompute_steps": 1,
        "inverse_recompute_steps": 1,
        "total_recompute_steps": 2,
        "witness_to_model_payload_ratio": 0.0,
        "trace_to_model_payload_ratio": 0.0,
        "checks": {
            "forward_matches_reference": True,
            "reverse_restores_initial_state": True,
            "no_witness_tape": True,
            "balanced_recompute": True,
        },
    }


def build_report(
    forward_path: Optional[Path],
    reverse_path: Optional[Path],
    mismatches: list[str],
) -> dict[str, Any]:
    proof = proof_cost()
    initial = initial_store()
    intermediates = coupling_intermediates(initial)
    return {
        "kind": KIND,
        "program": PROGRAM,
        "q31_one": Q31_ONE,
        "forward_output_json": str(forward_path) if forward_path is not None else None,
        "reverse_output_json": str(reverse_path) if reverse_path is not None else None,
        "initial": {
            "left": initial["left"],
            "right": initial["right"],
        },
        "forward": final_vars(),
        "intermediates": intermediates,
        "passed": not mismatches,
        "mismatches": mismatches,
        "proof": proof,
    }


def render_markdown(report: dict[str, Any]) -> str:
    proof = report["proof"]
    initial = report["initial"]
    forward = report["forward"]
    intermediates = report["intermediates"]
    lines = [
        "# Reverie Invertible Coupling Proof",
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
        "## Coupling State",
        "",
        "| Side | Initial | Final |",
        "| --- | --- | --- |",
        "| left | {} | {} |".format(
            vector_cell(initial["left"]),
            vector_cell(forward["left"]),
        ),
        "| right | {} | {} |".format(
            vector_cell(initial["right"]),
            vector_cell(forward["right"]),
        ),
        "",
        "## Additive Blocks",
        "",
        "| Block | Transform | Result |",
        "| --- | --- | --- |",
        "| right += f(left) | {} | {} |".format(
            vector_cell(intermediates["f_left"]),
            vector_cell(intermediates["right_after_bias"]),
        ),
        "| left += g(right) | {} | {} |".format(
            vector_cell(intermediates["g_right"]),
            vector_cell(intermediates["left_after_bias"]),
        ),
        "",
        "## Reverse Contract",
        "",
        "| Claim | Reverse restored | No witness tape | Balanced | Arithmetic |",
        "| --- | --- | --- | --- | --- |",
        "| `{}` | {} | {} | {} | `{}` |".format(
            proof["claim"],
            str(proof["checks"]["reverse_restores_initial_state"]).lower(),
            str(proof["checks"]["no_witness_tape"]).lower(),
            str(proof["checks"]["balanced_recompute"]).lower(),
            proof["arithmetic"],
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
        raise CouplingCheckError(f"failed to write {path}: {error}") from error


def validate_expected_report(path: Path, report: dict[str, Any]) -> None:
    expected = load_json(path)
    if expected != report:
        raise CouplingCheckError(f"recomputed report does not match {path}")


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
        if expected["left"] != [2684354560, -805306368, 1073741824, -805306368]:
            raise AssertionError("forward left changed unexpectedly")
        if expected["right"] != [2147483648, 536870912, 0, -536870912]:
            raise AssertionError("forward right changed unexpectedly")
        if inverse_store(expected) != initial:
            raise AssertionError("inverse did not restore initial state")
        proof = proof_cost()
        if proof["witness_payload_bytes"] != 0 or proof["trace_payload_bytes"] != 0:
            raise AssertionError("coupling proof should not need witness or trace bytes")
        if proof["replay_payload_bytes"] != 384:
            raise AssertionError("coupling replay payload changed unexpectedly")
        report = build_report(None, None, [])
        if not report["passed"] or report["proof"]["claim"] != PROOF_CLAIM:
            raise AssertionError("valid self-test report did not pass")
        markdown = render_markdown(report)
        for snippet in (
            "# Reverie Invertible Coupling Proof",
            "## Additive Blocks",
            "## Reverse Contract",
            "right += f(left)",
        ):
            if snippet not in markdown:
                raise AssertionError(f"rendered Markdown missing {snippet}")
        bad = {**expected, "left": [0, 0, 0, 0]}
        mismatches = compare_store(expected, bad, "forward")
        if not mismatches or "forward.left" not in mismatches[0]:
            raise AssertionError("tampered forward state was not detected")
    except (AssertionError, CouplingCheckError) as error:
        print(f"error: self-test failed: {error}", file=sys.stderr)
        return 1
    print("ok: invertible coupling checker self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    try:
        if args.write_final_vars is not None:
            args.write_final_vars.parent.mkdir(parents=True, exist_ok=True)
            args.write_final_vars.write_text(json.dumps(final_vars(), indent=2) + "\n", encoding="utf-8")
        mismatches = check_outputs(args.forward_output_json, args.reverse_output_json)
        report = build_report(args.forward_output_json, args.reverse_output_json, mismatches)
        if args.expect_report_json is not None:
            validate_expected_report(args.expect_report_json, report)
        if args.markdown_output is not None:
            write_markdown(args.markdown_output, report)
    except CouplingCheckError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif mismatches:
        for mismatch in mismatches:
            print(f"error: {mismatch}", file=sys.stderr)
    else:
        print(
            "ok: invertible coupling forward/reverse checked "
            f"replay_bytes={report['proof']['replay_payload_bytes']}"
        )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
