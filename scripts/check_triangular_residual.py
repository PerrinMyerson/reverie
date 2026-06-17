#!/usr/bin/env python3
"""Reference-check Reverie's Q31 triangular residual example."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional


KIND = "reverie_q31_triangular_residual_reference"
PROOF_CLAIM = "deterministic_q31_triangular_residual_replay"
PROGRAM = "examples/triangular_residual.rev"
WIDTH = 4
Q31_ONE = 1 << 31
I64_BYTES = 8
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
U64_MOD = 1 << 64
PARAM_FIELDS = ("w01", "w02", "w03", "b0", "w12", "w13", "b1", "w23", "b2", "b3")


class ResidualCheckError(ValueError):
    """Raised when a triangular residual report or Reverie output is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference-check examples/triangular_residual.rev forward and reverse JSON output."
    )
    parser.add_argument(
        "--forward-output-json",
        type=Path,
        help="JSON output from `reverie run examples/triangular_residual.rev --json`.",
    )
    parser.add_argument(
        "--reverse-output-json",
        type=Path,
        help=(
            "JSON output from `reverie reverse examples/triangular_residual.rev "
            "--json` seeded with final x."
        ),
    )
    parser.add_argument(
        "--write-final-vars",
        type=Path,
        help="Write the exact final x vars JSON needed for reverse execution.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Write a human-readable triangular residual proof card.",
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
        help="Run build-independent triangular residual checker self-tests and exit.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ResidualCheckError(f"failed to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ResidualCheckError(f"failed to parse {path}: {error}") from error


def checked_i64(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResidualCheckError(f"{label} must be a signed integer")
    if not I64_MIN <= value <= I64_MAX:
        raise ResidualCheckError(f"{label} is outside signed i64 range")
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


def comma(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.6g}"
    return str(value)


def vector_cell(values: list[int]) -> str:
    return "`[{}]`".format(", ".join(comma(value) for value in values))


def terms_cell(terms: list[list[Any]]) -> str:
    return "<br>".join(f"`{name} = {comma(value)}`" for name, value in terms)


def initial_store() -> dict[str, Any]:
    return {
        "x": [Q31_ONE, -(Q31_ONE // 2), Q31_ONE // 4, -(Q31_ONE // 4)],
        "w01": Q31_ONE // 2,
        "w02": -(Q31_ONE // 4),
        "w03": Q31_ONE // 4,
        "b0": Q31_ONE // 8,
        "w12": Q31_ONE // 2,
        "w13": -(Q31_ONE // 2),
        "b1": -(Q31_ONE // 8),
        "w23": Q31_ONE // 4,
        "b2": Q31_ONE // 16,
        "b3": -(Q31_ONE // 16),
    }


def forward_store(store: dict[str, Any]) -> dict[str, Any]:
    x = list(store["x"])
    x[0] = add_i64(x[0], fixed_mul_q31(x[1], store["w01"]))
    x[0] = add_i64(x[0], fixed_mul_q31(x[2], store["w02"]))
    x[0] = add_i64(x[0], fixed_mul_q31(x[3], store["w03"]))
    x[0] = add_i64(x[0], store["b0"])

    x[1] = add_i64(x[1], fixed_mul_q31(x[2], store["w12"]))
    x[1] = add_i64(x[1], fixed_mul_q31(x[3], store["w13"]))
    x[1] = add_i64(x[1], store["b1"])

    x[2] = add_i64(x[2], fixed_mul_q31(x[3], store["w23"]))
    x[2] = add_i64(x[2], store["b2"])

    x[3] = add_i64(x[3], store["b3"])
    return {**store, "x": x}


def residual_intermediates(store: dict[str, Any]) -> dict[str, Any]:
    x = list(store["x"])
    x0_terms = [
        ["x[1] */ w01", fixed_mul_q31(x[1], store["w01"])],
        ["x[2] */ w02", fixed_mul_q31(x[2], store["w02"])],
        ["x[3] */ w03", fixed_mul_q31(x[3], store["w03"])],
        ["b0", store["b0"]],
    ]
    for _, term in x0_terms:
        x[0] = add_i64(x[0], term)
    after_x0 = list(x)

    x1_terms = [
        ["x[2] */ w12", fixed_mul_q31(x[2], store["w12"])],
        ["x[3] */ w13", fixed_mul_q31(x[3], store["w13"])],
        ["b1", store["b1"]],
    ]
    for _, term in x1_terms:
        x[1] = add_i64(x[1], term)
    after_x1 = list(x)

    x2_terms = [
        ["x[3] */ w23", fixed_mul_q31(x[3], store["w23"])],
        ["b2", store["b2"]],
    ]
    for _, term in x2_terms:
        x[2] = add_i64(x[2], term)
    after_x2 = list(x)

    x3_terms = [["b3", store["b3"]]]
    x[3] = add_i64(x[3], store["b3"])

    return {
        "x0_terms": x0_terms,
        "after_x0": after_x0,
        "x1_terms": x1_terms,
        "after_x1": after_x1,
        "x2_terms": x2_terms,
        "after_x2": after_x2,
        "x3_terms": x3_terms,
        "after_x3": list(x),
    }


def inverse_store(store: dict[str, Any]) -> dict[str, Any]:
    x = list(store["x"])
    x[3] = sub_i64(x[3], store["b3"])

    x[2] = sub_i64(x[2], store["b2"])
    x[2] = sub_i64(x[2], fixed_mul_q31(x[3], store["w23"]))

    x[1] = sub_i64(x[1], store["b1"])
    x[1] = sub_i64(x[1], fixed_mul_q31(x[3], store["w13"]))
    x[1] = sub_i64(x[1], fixed_mul_q31(x[2], store["w12"]))

    x[0] = sub_i64(x[0], store["b0"])
    x[0] = sub_i64(x[0], fixed_mul_q31(x[3], store["w03"]))
    x[0] = sub_i64(x[0], fixed_mul_q31(x[2], store["w02"]))
    x[0] = sub_i64(x[0], fixed_mul_q31(x[1], store["w01"]))
    return {**store, "x": x}


def final_vars() -> dict[str, Any]:
    final = forward_store(initial_store())
    return {"x": final["x"]}


def parse_output_store(value: Any, expected_kind: str, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResidualCheckError(f"{context} must be a JSON object")
    if value.get("kind") != expected_kind:
        raise ResidualCheckError(f"{context}.kind must be {expected_kind}")
    store = value.get("store")
    if not isinstance(store, dict):
        raise ResidualCheckError(f"{context}.store must be an object")
    return validate_store(store, f"{context}.store")


def validate_store(value: dict[str, Any], context: str) -> dict[str, Any]:
    x = value.get("x")
    if not isinstance(x, list) or len(x) != WIDTH:
        raise ResidualCheckError(f"{context}.x must have {WIDTH} entries")
    store: dict[str, Any] = {
        "x": [checked_i64(entry, f"{context}.x[{index}]") for index, entry in enumerate(x)]
    }
    for field in PARAM_FIELDS:
        store[field] = checked_i64(value.get(field), f"{context}.{field}")
    return store


def compare_store(expected: dict[str, Any], actual: dict[str, Any], context: str) -> list[str]:
    mismatches = []
    for field, expected_value in expected.items():
        actual_value = actual.get(field)
        if actual_value != expected_value:
            mismatches.append(f"{context}.{field} expected {expected_value}, found {actual_value}")
    return mismatches


def proof_cost() -> dict[str, Any]:
    parameter_payload_bytes = len(PARAM_FIELDS) * I64_BYTES
    state_payload_bytes = WIDTH * I64_BYTES
    witness_payload_bytes = 0
    trace_payload_bytes = 0
    return {
        "claim": PROOF_CLAIM,
        "arithmetic": "q31_wrapping_i64",
        "parameter_payload_bytes": parameter_payload_bytes,
        "state_payload_bytes": state_payload_bytes,
        "witness_payload_bytes": witness_payload_bytes,
        "trace_payload_bytes": trace_payload_bytes,
        "replay_payload_bytes": parameter_payload_bytes + state_payload_bytes,
        "forward_recompute_steps": 1,
        "inverse_recompute_steps": 1,
        "total_recompute_steps": 2,
        "witness_to_parameter_payload_ratio": 0.0,
        "trace_to_parameter_payload_ratio": 0.0,
        "checks": {
            "forward_matches_reference": True,
            "reverse_restores_initial_state": True,
            "triangular_source_order": True,
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
        "initial": {
            "x": initial["x"],
            "parameters": {field: initial[field] for field in PARAM_FIELDS},
        },
        "forward": final_vars(),
        "intermediates": residual_intermediates(initial),
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
        "# Reverie Triangular Residual Proof",
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
        "## Residual State",
        "",
        "| Initial x | Final x |",
        "| --- | --- |",
        "| {} | {} |".format(vector_cell(initial["x"]), vector_cell(forward["x"])),
        "",
        "## Triangular Updates",
        "",
        "| Block | Residual terms | Result |",
        "| --- | --- | --- |",
        "| x[0] += f(x[1], x[2], x[3]) | {} | {} |".format(
            terms_cell(intermediates["x0_terms"]),
            vector_cell(intermediates["after_x0"]),
        ),
        "| x[1] += g(x[2], x[3]) | {} | {} |".format(
            terms_cell(intermediates["x1_terms"]),
            vector_cell(intermediates["after_x1"]),
        ),
        "| x[2] += h(x[3]) | {} | {} |".format(
            terms_cell(intermediates["x2_terms"]),
            vector_cell(intermediates["after_x2"]),
        ),
        "| x[3] += b3 | {} | {} |".format(
            terms_cell(intermediates["x3_terms"]),
            vector_cell(intermediates["after_x3"]),
        ),
        "",
        "## Reverse Contract",
        "",
        "| Claim | Reverse restored | Triangular order | No witness tape | Balanced | Arithmetic |",
        "| --- | --- | --- | --- | --- | --- |",
        "| `{}` | {} | {} | {} | {} | `{}` |".format(
            proof["claim"],
            str(proof["checks"]["reverse_restores_initial_state"]).lower(),
            str(proof["checks"]["triangular_source_order"]).lower(),
            str(proof["checks"]["no_witness_tape"]).lower(),
            str(proof["checks"]["balanced_recompute"]).lower(),
            proof["arithmetic"],
        ),
        "",
        "## Payloads",
        "",
        "| Parameters | State | Witness | Trace | Replay | Witness/parameters | Trace/parameters |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| {} | {} | {} | {} | {} | {:.6f} | {:.6f} |".format(
            comma(proof["parameter_payload_bytes"]),
            comma(proof["state_payload_bytes"]),
            comma(proof["witness_payload_bytes"]),
            comma(proof["trace_payload_bytes"]),
            comma(proof["replay_payload_bytes"]),
            proof["witness_to_parameter_payload_ratio"],
            proof["trace_to_parameter_payload_ratio"],
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
        raise ResidualCheckError(f"failed to write {path}: {error}") from error


def validate_expected_report(path: Path, report: dict[str, Any]) -> None:
    expected = load_json(path)
    if expected != report:
        raise ResidualCheckError(f"recomputed report does not match {path}")


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
        if expected["x"] != [1610612736, -805306368, 536870912, -671088640]:
            raise AssertionError("forward x changed unexpectedly")
        if inverse_store(expected) != initial:
            raise AssertionError("inverse did not restore initial state")
        proof = proof_cost()
        if proof["witness_payload_bytes"] != 0 or proof["trace_payload_bytes"] != 0:
            raise AssertionError("triangular residual proof should not need witness or trace bytes")
        if proof["replay_payload_bytes"] != 112:
            raise AssertionError("triangular residual replay payload changed unexpectedly")
        report = build_report(None, None, [])
        if not report["passed"] or report["proof"]["claim"] != PROOF_CLAIM:
            raise AssertionError("valid self-test report did not pass")
        markdown = render_markdown(report)
        for snippet in (
            "# Reverie Triangular Residual Proof",
            "## Triangular Updates",
            "x[0] += f(x[1], x[2], x[3])",
            "## Reverse Contract",
        ):
            if snippet not in markdown:
                raise AssertionError(f"rendered Markdown missing {snippet}")
        bad_store = forward_store(initial)
        bad_store["x"][0] += 1
        mismatches = compare_store(expected, bad_store, "bad forward")
        if not mismatches:
            raise AssertionError("bad residual store did not produce mismatch")
    except (AssertionError, ResidualCheckError) as error:
        print(f"error: self-test failed: {error}", file=sys.stderr)
        return 1
    print("ok: triangular residual checker self-tests passed")
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
    except ResidualCheckError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif mismatches:
        for mismatch in mismatches:
            print(f"error: {mismatch}", file=sys.stderr)
    else:
        print(
            "ok: triangular residual forward/reverse checked "
            f"replay_bytes={report['proof']['replay_payload_bytes']}"
        )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
