#!/usr/bin/env python3
"""Export MNIST MLP weights to Reverie --vars-json Q31 seeds."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional

import export_q31_linear_model as linear_export


IMAGE_PIXELS = 784
HIDDEN = 16
DIGITS = 10
Q31_ONE = 1 << 31
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1

AUTO_JSON_MODELS = (
    (
        ("w1",),
        ("b1",),
        ("w2",),
        ("b2",),
        "auto",
    ),
    (
        ("model", "w1"),
        ("model", "b1"),
        ("model", "w2"),
        ("model", "b2"),
        "auto",
    ),
    (
        ("state_dict", "fc1.weight"),
        ("state_dict", "fc1.bias"),
        ("state_dict", "fc2.weight"),
        ("state_dict", "fc2.bias"),
        "torch",
    ),
    (
        ("state_dict", "linear1.weight"),
        ("state_dict", "linear1.bias"),
        ("state_dict", "linear2.weight"),
        ("state_dict", "linear2.bias"),
        "torch",
    ),
    (
        ("state_dict", "hidden.weight"),
        ("state_dict", "hidden.bias"),
        ("state_dict", "output.weight"),
        ("state_dict", "output.bias"),
        "torch",
    ),
    (
        ("state_dict", "0.weight"),
        ("state_dict", "0.bias"),
        ("state_dict", "2.weight"),
        ("state_dict", "2.bias"),
        "torch",
    ),
)

AUTO_NPZ_MODELS = (
    ("w1", "b1", "w2", "b2", "reverie"),
    ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "torch"),
    ("linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias", "torch"),
    ("hidden.weight", "hidden.bias", "output.weight", "output.bias", "torch"),
)


class ExportError(ValueError):
    """Raised when an MLP source cannot be exported safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a 784->16->10 MNIST MLP into the exact integer JSON seed "
            "shape accepted by `reverie run ... --vars-json`."
        )
    )
    parser.add_argument("--input", type=Path, help="Source JSON or NumPy .npz model payload.")
    parser.add_argument("--output", type=Path, help="Destination vars JSON path, or '-' for stdout.")
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Optional sidecar metadata JSON path. The main output stays pure --vars-json seeds.",
    )
    parser.add_argument("--w1-path", help="Slash-separated JSON path for first-layer weights.")
    parser.add_argument("--b1-path", help="Slash-separated JSON path for first-layer bias.")
    parser.add_argument("--w2-path", help="Slash-separated JSON path for second-layer weights.")
    parser.add_argument("--b2-path", help="Slash-separated JSON path for second-layer bias.")
    parser.add_argument("--npz-w1", help="Array name inside a .npz file for first-layer weights.")
    parser.add_argument("--npz-b1", help="Array name inside a .npz file for first-layer bias.")
    parser.add_argument("--npz-w2", help="Array name inside a .npz file for second-layer weights.")
    parser.add_argument("--npz-b2", help="Array name inside a .npz file for second-layer bias.")
    parser.add_argument(
        "--input-layout",
        choices=("auto", "torch", "reverie"),
        default="auto",
        help=(
            "Source layer layout. torch uses out_features x in_features; "
            "reverie uses in_features x out_features."
        ),
    )
    parser.add_argument(
        "--input-scale",
        choices=("float", "q31"),
        default="float",
        help="Use 'float' to quantize values by 1<<31, or 'q31' to pass signed integers through.",
    )
    parser.add_argument(
        "--clip-i64",
        action="store_true",
        help="Clamp quantized values into signed i64 instead of rejecting overflow.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent MLP exporter self-tests and exit.",
    )
    return parser.parse_args()


def path_from_argument(path: str) -> tuple[str, ...]:
    parts = tuple(part for part in path.split("/") if part)
    if not parts:
        raise ExportError("JSON paths must not be empty")
    return parts


def get_path(value: Any, path: tuple[str, ...]) -> Optional[Any]:
    current = value
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def find_json_arrays(
    source: Any,
    w1_path: Optional[str],
    b1_path: Optional[str],
    w2_path: Optional[str],
    b2_path: Optional[str],
) -> tuple[Any, Any, Any, Any, dict[str, str], str]:
    custom_paths = (w1_path, b1_path, w2_path, b2_path)
    if any(custom_paths):
        if not all(custom_paths):
            raise ExportError("--w1-path, --b1-path, --w2-path, and --b2-path must be provided together")
        paths = {
            "w1_path": path_from_argument(w1_path or ""),
            "b1_path": path_from_argument(b1_path or ""),
            "w2_path": path_from_argument(w2_path or ""),
            "b2_path": path_from_argument(b2_path or ""),
        }
        values = {key: get_path(source, path) for key, path in paths.items()}
        for key, value in values.items():
            if value is None:
                raise ExportError(f"{key} `{ '/'.join(paths[key]) }` was not found")
        return (
            values["w1_path"],
            values["b1_path"],
            values["w2_path"],
            values["b2_path"],
            {key: "/".join(path) for key, path in paths.items()},
            "auto",
        )

    for w1_parts, b1_parts, w2_parts, b2_parts, layout_hint in AUTO_JSON_MODELS:
        w1 = get_path(source, w1_parts)
        b1 = get_path(source, b1_parts)
        w2 = get_path(source, w2_parts)
        b2 = get_path(source, b2_parts)
        if w1 is not None and b1 is not None and w2 is not None and b2 is not None:
            return (
                w1,
                b1,
                w2,
                b2,
                {
                    "w1_path": "/".join(w1_parts),
                    "b1_path": "/".join(b1_parts),
                    "w2_path": "/".join(w2_parts),
                    "b2_path": "/".join(b2_parts),
                },
                layout_hint,
            )

    raise ExportError("could not find MLP arrays; use --w1-path/--b1-path/--w2-path/--b2-path")


def find_npz_members(args: argparse.Namespace, archive: zipfile.ZipFile) -> tuple[str, str, str, str, str]:
    custom = (args.npz_w1, args.npz_b1, args.npz_w2, args.npz_b2)
    if any(custom):
        if not all(custom):
            raise ExportError("--npz-w1, --npz-b1, --npz-w2, and --npz-b2 must be provided together")
        return (
            linear_export.npy_member_name(args.npz_w1),
            linear_export.npy_member_name(args.npz_b1),
            linear_export.npy_member_name(args.npz_w2),
            linear_export.npy_member_name(args.npz_b2),
            "auto",
        )

    entries = {linear_export.npy_member_name(name) for name in archive.namelist()}
    for w1, b1, w2, b2, layout_hint in AUTO_NPZ_MODELS:
        if w1 in entries and b1 in entries and w2 in entries and b2 in entries:
            return w1, b1, w2, b2, layout_hint

    raise ExportError("could not find MLP arrays in .npz; use --npz-w1/--npz-b1/--npz-w2/--npz-b2")


def read_npz_source(path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], str]:
    with zipfile.ZipFile(path) as archive:
        w1_member, b1_member, w2_member, b2_member, layout_hint = find_npz_members(args, archive)
        source = {
            "w1": linear_export.read_npz_member(archive, w1_member),
            "b1": linear_export.read_npz_member(archive, b1_member),
            "w2": linear_export.read_npz_member(archive, w2_member),
            "b2": linear_export.read_npz_member(archive, b2_member),
        }
    return (
        source,
        {
            "input_format": "npz",
            "npz_w1": w1_member,
            "npz_b1": b1_member,
            "npz_w2": w2_member,
            "npz_b2": b2_member,
        },
        layout_hint,
    )


def read_input_source(args: argparse.Namespace) -> tuple[Any, dict[str, Any], str]:
    if args.input.suffix.lower() == ".npz":
        if any((args.w1_path, args.b1_path, args.w2_path, args.b2_path)):
            raise ExportError("use --npz-w1/--npz-b1/--npz-w2/--npz-b2 for .npz inputs")
        return read_npz_source(args.input, args)
    if any((args.npz_w1, args.npz_b1, args.npz_w2, args.npz_b2)):
        raise ExportError("--npz-w1/--npz-b1/--npz-w2/--npz-b2 require a .npz input")
    return (
        json.loads(args.input.read_text(encoding="utf-8")),
        {"input_format": "json"},
        "auto",
    )


def matrix_shape(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, list):
        raise ExportError(f"{label} must be a matrix")
    if not value:
        raise ExportError(f"{label} must not be empty")
    if not all(isinstance(row, list) for row in value):
        raise ExportError(f"{label} must be a matrix of rows")
    lengths = [len(row) for row in value]
    if any(length == 0 for length in lengths):
        raise ExportError(f"{label} rows must not be empty")
    if any(length != lengths[0] for length in lengths):
        raise ExportError(f"{label} must be rectangular")
    return len(value), lengths[0]


def normalize_layer(
    weights: Any,
    label: str,
    input_dim: int,
    output_dim: int,
    requested_layout: str,
    layout_hint: str,
) -> tuple[list[list[Any]], str]:
    rows, cols = matrix_shape(weights, label)
    layout = layout_hint if requested_layout == "auto" else requested_layout
    if layout == "auto":
        if (rows, cols) == (input_dim, output_dim):
            layout = "reverie"
        elif (rows, cols) == (output_dim, input_dim):
            layout = "torch"
        else:
            raise ExportError(
                f"{label} must be {input_dim}x{output_dim} or {output_dim}x{input_dim}; "
                f"found {rows}x{cols}"
            )

    if layout == "reverie":
        if (rows, cols) != (input_dim, output_dim):
            raise ExportError(
                f"{label} in reverie layout must be {input_dim}x{output_dim}; found {rows}x{cols}"
            )
        return weights, layout

    if layout == "torch":
        if (rows, cols) != (output_dim, input_dim):
            raise ExportError(
                f"{label} in torch layout must be {output_dim}x{input_dim}; found {rows}x{cols}"
            )
        return [[weights[out][inp] for out in range(output_dim)] for inp in range(input_dim)], layout

    raise ExportError(f"unknown input layout `{layout}`")


def coerce_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ExportError(f"{label} must be finite")
    return converted


def checked_i64(value: int, label: str, clip: bool) -> int:
    if I64_MIN <= value <= I64_MAX:
        return value
    if clip:
        return min(max(value, I64_MIN), I64_MAX)
    raise ExportError(f"{label} quantized outside signed i64 range")


def quantize_value(value: Any, label: str, input_scale: str, clip_i64: bool) -> int:
    if input_scale == "q31":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ExportError(f"{label} must be an integer in q31 mode")
        return checked_i64(value, label, clip=False)
    return checked_i64(int(round(coerce_float(value, label) * Q31_ONE)), label, clip_i64)


def quantize_matrix(values: list[list[Any]], label: str, input_scale: str, clip_i64: bool) -> list[list[int]]:
    return [
        [
            quantize_value(value, f"{label}[{row}][{col}]", input_scale, clip_i64)
            for col, value in enumerate(row_values)
        ]
        for row, row_values in enumerate(values)
    ]


def quantize_vector(values: Any, label: str, expected: int, input_scale: str, clip_i64: bool) -> list[int]:
    if not isinstance(values, list):
        raise ExportError(f"{label} must be a vector")
    if len(values) != expected:
        raise ExportError(f"{label} must have {expected} entries; found {len(values)}")
    return [
        quantize_value(value, f"{label}[{index}]", input_scale, clip_i64)
        for index, value in enumerate(values)
    ]


def export_payload(
    source: Any,
    *,
    input_path: str,
    w1_path: Optional[str] = None,
    b1_path: Optional[str] = None,
    w2_path: Optional[str] = None,
    b2_path: Optional[str] = None,
    input_layout: str = "auto",
    input_scale: str = "float",
    clip_i64: bool = False,
    source_details: Optional[dict[str, Any]] = None,
    source_layout_hint: str = "auto",
) -> tuple[dict[str, Any], dict[str, Any]]:
    w1, b1, w2, b2, paths, layout_hint = find_json_arrays(
        source,
        w1_path,
        b1_path,
        w2_path,
        b2_path,
    )
    if layout_hint == "auto":
        layout_hint = source_layout_hint
    normalized_w1, w1_layout = normalize_layer(
        w1,
        "w1",
        IMAGE_PIXELS,
        HIDDEN,
        input_layout,
        layout_hint,
    )
    normalized_w2, w2_layout = normalize_layer(
        w2,
        "w2",
        HIDDEN,
        DIGITS,
        input_layout,
        layout_hint,
    )
    vars_payload = {
        "w1": quantize_matrix(normalized_w1, "w1", input_scale, clip_i64),
        "b1": quantize_vector(b1, "b1", HIDDEN, input_scale, clip_i64),
        "w2": quantize_matrix(normalized_w2, "w2", input_scale, clip_i64),
        "b2": quantize_vector(b2, "b2", DIGITS, input_scale, clip_i64),
    }
    metadata = {
        "format": "reverie_mnist_mlp_q31_vars_json_metadata",
        "source_format": "reverie_q31_mlp_vars_export",
        "input": input_path,
        "program": "examples/mnist_mlp_witness.rev",
        "vars_json_keys": ["w1", "b1", "w2", "b2"],
        "tensor_shapes": {
            "w1": [IMAGE_PIXELS, HIDDEN],
            "b1": [HIDDEN],
            "w2": [HIDDEN, DIGITS],
            "b2": [DIGITS],
        },
        "source": {
            **paths,
            "w1_layout": w1_layout,
            "w2_layout": w2_layout,
            "input_scale": input_scale,
            "q31_one": Q31_ONE,
        },
    }
    if source_details:
        metadata["source"].update(source_details)
    return vars_payload, metadata


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if str(output_path) == "-":
        sys.stdout.write(encoded)
        return
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(encoded, encoding="utf-8")


def run_self_tests() -> int:
    def zeros(rows: int, cols: int) -> list[list[float]]:
        return [[0.0 for _ in range(cols)] for _ in range(rows)]

    w1_torch = zeros(HIDDEN, IMAGE_PIXELS)
    w2_torch = zeros(DIGITS, HIDDEN)
    w1_torch[3][7] = 0.5
    w2_torch[4][3] = -0.25
    source = {
        "state_dict": {
            "fc1.weight": w1_torch,
            "fc1.bias": [0.0] * HIDDEN,
            "fc2.weight": w2_torch,
            "fc2.bias": [0.0] * DIGITS,
        }
    }
    payload, metadata = export_payload(source, input_path="fixture.json")
    if payload["w1"][7][3] != Q31_ONE // 2:
        raise AssertionError("torch w1 fixture did not transpose/quantize")
    if payload["w2"][3][4] != -(Q31_ONE // 4):
        raise AssertionError("torch w2 fixture did not transpose/quantize")
    if metadata["source"]["w1_layout"] != "torch" or "format" in payload:
        raise AssertionError("MLP exporter metadata or pure vars payload is wrong")

    w1_native = [[0 for _ in range(HIDDEN)] for _ in range(IMAGE_PIXELS)]
    w2_native = [[0 for _ in range(DIGITS)] for _ in range(HIDDEN)]
    w1_native[0][0] = 17
    w2_native[0][0] = -19
    native_payload, _metadata = export_payload(
        {
            "w1": w1_native,
            "b1": list(range(HIDDEN)),
            "w2": w2_native,
            "b2": list(range(DIGITS)),
        },
        input_path="native.json",
        input_scale="q31",
    )
    if native_payload["w1"][0][0] != 17 or native_payload["w2"][0][0] != -19:
        raise AssertionError("native q31 fixture did not pass values through")

    try:
        export_payload({"w1": [[0.0]], "b1": [0.0] * HIDDEN, "w2": w2_native, "b2": [0.0] * DIGITS}, input_path="bad.json")
    except ExportError as error:
        if "w1" not in str(error) or "784x16" not in str(error):
            raise AssertionError(f"unexpected shape error: {error}") from error
    else:
        raise AssertionError("bad shape fixture unexpectedly exported")

    with tempfile.TemporaryDirectory() as directory:
        npz_path = Path(directory) / "mlp.npz"
        with zipfile.ZipFile(npz_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "fc1.weight.npy",
                linear_export.self_test_npy_bytes(w1_torch, (HIDDEN, IMAGE_PIXELS), "<f8"),
            )
            archive.writestr(
                "fc1.bias.npy",
                linear_export.self_test_npy_bytes([0.0] * HIDDEN, (HIDDEN,), "<f8"),
            )
            archive.writestr(
                "fc2.weight.npy",
                linear_export.self_test_npy_bytes(w2_torch, (DIGITS, HIDDEN), "<f8"),
            )
            archive.writestr(
                "fc2.bias.npy",
                linear_export.self_test_npy_bytes([0.0] * DIGITS, (DIGITS,), "<f8"),
            )
        args = argparse.Namespace(npz_w1=None, npz_b1=None, npz_w2=None, npz_b2=None)
        npz_source, npz_details, npz_layout_hint = read_npz_source(npz_path, args)
        npz_payload, npz_metadata = export_payload(
            npz_source,
            input_path=str(npz_path),
            source_details=npz_details,
            source_layout_hint=npz_layout_hint,
        )
        if npz_payload["w1"][7][3] != Q31_ONE // 2:
            raise AssertionError("npz MLP fixture did not transpose w1")
        if npz_metadata["source"]["input_format"] != "npz":
            raise AssertionError("npz MLP fixture did not record source format")

        output_path = Path(directory) / "mlp-vars.json"
        metadata_path = Path(directory) / "mlp-vars.metadata.json"
        write_json(payload, output_path)
        write_json(metadata, metadata_path)
        roundtrip = json.loads(output_path.read_text(encoding="utf-8"))
        if sorted(roundtrip) != ["b1", "b2", "w1", "w2"]:
            raise AssertionError("vars JSON roundtrip was not a pure seed object")
        metadata_roundtrip = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_roundtrip["program"] != "examples/mnist_mlp_witness.rev":
            raise AssertionError("metadata roundtrip did not record the MLP program")

    print("ok: Q31 MLP vars-json exporter self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    if args.input is None or args.output is None:
        print("error: --input and --output are required", file=sys.stderr)
        return 2
    try:
        source, source_details, source_layout_hint = read_input_source(args)
        payload, metadata = export_payload(
            source,
            input_path=str(args.input),
            w1_path=args.w1_path,
            b1_path=args.b1_path,
            w2_path=args.w2_path,
            b2_path=args.b2_path,
            input_layout=args.input_layout,
            input_scale=args.input_scale,
            clip_i64=args.clip_i64,
            source_details=source_details,
            source_layout_hint=source_layout_hint,
        )
        write_json(payload, args.output)
        if args.metadata_output is not None:
            write_json(metadata, args.metadata_output)
    except (OSError, json.JSONDecodeError, zipfile.BadZipFile, ExportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
