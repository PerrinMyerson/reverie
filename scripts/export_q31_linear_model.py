#!/usr/bin/env python3
"""Export a MNIST-shaped linear model to Reverie's Q31 import JSON."""

from __future__ import annotations

import argparse
import ast
import json
import math
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional


IMAGE_PIXELS = 784
DIGITS = 10
Q31_ONE = 1 << 31
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1

AUTO_MODEL_PATHS = (
    (("model", "weights"), ("model", "bias"), "auto"),
    (("weights",), ("bias",), "auto"),
    (("weight",), ("bias",), "auto"),
    (("state_dict", "linear.weight"), ("state_dict", "linear.bias"), "class-pixel"),
    (
        ("state_dict", "classifier.weight"),
        ("state_dict", "classifier.bias"),
        "class-pixel",
    ),
    (("linear.weight",), ("linear.bias",), "class-pixel"),
    (("classifier.weight",), ("classifier.bias",), "class-pixel"),
)
AUTO_NPZ_MEMBERS = (
    ("weights", "bias", "auto"),
    ("weight", "bias", "auto"),
    ("linear.weight", "linear.bias", "class-pixel"),
    ("classifier.weight", "classifier.bias", "class-pixel"),
)


class ExportError(ValueError):
    """Raised when the source model cannot be exported safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a MNIST-shaped linear classifier to Reverie's weights_bias_q31_json format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Source JSON or NumPy .npz model payload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination Reverie Q31 JSON path, or '-' for stdout.",
    )
    parser.add_argument(
        "--weights-path",
        help=(
            "Slash-separated JSON path for weights, such as "
            "'state_dict/linear.weight'. Defaults to common model keys."
        ),
    )
    parser.add_argument(
        "--bias-path",
        help=(
            "Slash-separated JSON path for bias, such as "
            "'state_dict/linear.bias'. Defaults to common model keys."
        ),
    )
    parser.add_argument(
        "--npz-weights",
        help=(
            "Array name inside a .npz file for weights. Defaults to common names "
            "such as weights, linear.weight, or classifier.weight."
        ),
    )
    parser.add_argument(
        "--npz-bias",
        help=(
            "Array name inside a .npz file for bias. Defaults to common names "
            "such as bias, linear.bias, or classifier.bias."
        ),
    )
    parser.add_argument(
        "--input-layout",
        choices=("auto", "pixel-class", "class-pixel"),
        default="auto",
        help=(
            "Source weight layout. pixel-class is Reverie's 784x10 layout; "
            "class-pixel is the common linear-layer 10x784 layout."
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
        help="Run build-independent exporter self-tests and exit.",
    )
    return parser.parse_args()


def npy_member_name(name: str) -> str:
    return name[:-4] if name.endswith(".npy") else name


def npy_member_candidates(name: str) -> tuple[str, str]:
    return (name, name if name.endswith(".npy") else f"{name}.npy")


def read_npy_array(data: bytes, label: str) -> Any:
    if not data.startswith(b"\x93NUMPY"):
        raise ExportError(f"{label} is not a .npy array")
    if len(data) < 10:
        raise ExportError(f"{label} is truncated")

    major = data[6]
    if major == 1:
        header_len = struct.unpack_from("<H", data, 8)[0]
        header_start = 10
        encoding = "latin1"
    elif major in (2, 3):
        if len(data) < 12:
            raise ExportError(f"{label} is truncated")
        header_len = struct.unpack_from("<I", data, 8)[0]
        header_start = 12
        encoding = "utf-8" if major == 3 else "latin1"
    else:
        raise ExportError(f"{label} uses unsupported .npy version {major}")

    header_end = header_start + header_len
    if len(data) < header_end:
        raise ExportError(f"{label} header is truncated")
    header_text = data[header_start:header_end].decode(encoding).strip()
    try:
        header = ast.literal_eval(header_text)
    except (SyntaxError, ValueError) as error:
        raise ExportError(f"{label} has an invalid .npy header") from error
    if not isinstance(header, dict):
        raise ExportError(f"{label} .npy header must be a dictionary")

    descr = header.get("descr")
    shape = header.get("shape")
    fortran_order = header.get("fortran_order")
    if not isinstance(descr, str):
        raise ExportError(f"{label} .npy dtype descriptor must be a string")
    if fortran_order is not False:
        raise ExportError(f"{label} must be C-order; Fortran-order arrays are not supported")
    if not isinstance(shape, tuple) or len(shape) not in (1, 2):
        raise ExportError(f"{label} must be a rank-1 or rank-2 .npy array")
    if any(not isinstance(dim, int) or dim < 0 for dim in shape):
        raise ExportError(f"{label} has an invalid .npy shape")

    payload = data[header_end:]
    return decode_npy_payload(payload, descr, shape, label)


def decode_npy_payload(payload: bytes, descr: str, shape: tuple[int, ...], label: str) -> Any:
    if not descr:
        raise ExportError(f"{label} has an empty .npy dtype descriptor")
    byte_order = descr[0] if descr[0] in "<>|=" else "="
    type_offset = 1 if descr[0] in "<>|=" else 0
    if len(descr) <= type_offset:
        raise ExportError(f"{label} has an invalid .npy dtype descriptor `{descr}`")
    type_char = descr[type_offset]
    try:
        item_size = int(descr[type_offset + 1 :])
    except ValueError as error:
        raise ExportError(f"{label} has an invalid .npy dtype descriptor `{descr}`") from error

    format_codes = {
        ("f", 4): "f",
        ("f", 8): "d",
        ("i", 1): "b",
        ("i", 2): "h",
        ("i", 4): "i",
        ("i", 8): "q",
        ("u", 1): "B",
        ("u", 2): "H",
        ("u", 4): "I",
        ("u", 8): "Q",
    }
    format_code = format_codes.get((type_char, item_size))
    if format_code is None:
        raise ExportError(
            f"{label} dtype `{descr}` is unsupported; use float or integer arrays"
        )
    prefix = (
        ">"
        if byte_order == ">" or (byte_order == "=" and sys.byteorder == "big")
        else "<"
    )
    count = math.prod(shape)
    expected_bytes = count * item_size
    if len(payload) != expected_bytes:
        raise ExportError(
            f"{label} expected {expected_bytes} data byte(s), found {len(payload)}"
        )
    values = [
        unpacked[0]
        for unpacked in struct.iter_unpack(f"{prefix}{format_code}", payload)
    ]
    if len(shape) == 1:
        return values
    rows, cols = shape
    return [values[row * cols : (row + 1) * cols] for row in range(rows)]


def read_npz_member(archive: zipfile.ZipFile, name: str) -> Any:
    entries = set(archive.namelist())
    for candidate in npy_member_candidates(name):
        if candidate in entries:
            return read_npy_array(archive.read(candidate), candidate)
    raise ExportError(f".npz member `{name}` was not found")


def find_npz_members(
    archive: zipfile.ZipFile,
    weights_name: Optional[str],
    bias_name: Optional[str],
) -> tuple[str, str, str]:
    if weights_name or bias_name:
        if not weights_name or not bias_name:
            raise ExportError("--npz-weights and --npz-bias must be provided together")
        return npy_member_name(weights_name), npy_member_name(bias_name), "auto"

    entries = {npy_member_name(name) for name in archive.namelist()}
    for weights, bias, layout_hint in AUTO_NPZ_MEMBERS:
        if weights in entries and bias in entries:
            return weights, bias, layout_hint

    raise ExportError(
        "could not find weights/bias arrays in .npz; use --npz-weights and --npz-bias"
    )


def read_npz_model(
    path: Path,
    weights_name: Optional[str],
    bias_name: Optional[str],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    with zipfile.ZipFile(path) as archive:
        weights_member, bias_member, layout_hint = find_npz_members(
            archive, weights_name, bias_name
        )
        weights = read_npz_member(archive, weights_member)
        bias = read_npz_member(archive, bias_member)
    source = {"weights": weights, "bias": bias}
    details = {
        "input_format": "npz",
        "npz_weights": weights_member,
        "npz_bias": bias_member,
    }
    return source, details, layout_hint


def read_input_source(args: argparse.Namespace) -> tuple[Any, dict[str, Any], str]:
    suffix = args.input.suffix.lower()
    if suffix == ".npz":
        if args.weights_path or args.bias_path:
            raise ExportError("use --npz-weights/--npz-bias for .npz inputs")
        return read_npz_model(args.input, args.npz_weights, args.npz_bias)
    if args.npz_weights or args.npz_bias:
        raise ExportError("--npz-weights and --npz-bias require a .npz input")
    return (
        json.loads(args.input.read_text(encoding="utf-8")),
        {"input_format": "json"},
        "auto",
    )


def get_path(value: Any, path: tuple[str, ...]) -> Optional[Any]:
    current = value
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def path_from_argument(path: str) -> tuple[str, ...]:
    parts = tuple(part for part in path.split("/") if part)
    if not parts:
        raise ExportError("JSON paths must not be empty")
    return parts


def find_model_arrays(
    source: Any,
    weights_path: Optional[str],
    bias_path: Optional[str],
) -> tuple[Any, Any, tuple[str, ...], tuple[str, ...], str]:
    if weights_path or bias_path:
        if not weights_path or not bias_path:
            raise ExportError("--weights-path and --bias-path must be provided together")
        weights_parts = path_from_argument(weights_path)
        bias_parts = path_from_argument(bias_path)
        weights = get_path(source, weights_parts)
        bias = get_path(source, bias_parts)
        if weights is None:
            raise ExportError(f"weights path `{weights_path}` was not found")
        if bias is None:
            raise ExportError(f"bias path `{bias_path}` was not found")
        return weights, bias, weights_parts, bias_parts, "auto"

    for weights_parts, bias_parts, layout_hint in AUTO_MODEL_PATHS:
        weights = get_path(source, weights_parts)
        bias = get_path(source, bias_parts)
        if weights is not None and bias is not None:
            return weights, bias, weights_parts, bias_parts, layout_hint

    raise ExportError(
        "could not find weights/bias arrays; use --weights-path and --bias-path"
    )


def matrix_shape(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, list):
        raise ExportError(f"{label} must be a matrix")
    if not value:
        raise ExportError(f"{label} must not be empty")
    if not all(isinstance(row, list) for row in value):
        raise ExportError(f"{label} must be a matrix of rows")
    row_lengths = [len(row) for row in value]
    if any(length == 0 for length in row_lengths):
        raise ExportError(f"{label} rows must not be empty")
    first = row_lengths[0]
    if any(length != first for length in row_lengths):
        raise ExportError(f"{label} must be rectangular")
    return len(value), first


def normalize_weights(
    weights: Any,
    requested_layout: str,
    layout_hint: str,
) -> tuple[list[list[Any]], str]:
    rows, cols = matrix_shape(weights, "weights")
    layout = layout_hint if requested_layout == "auto" else requested_layout
    if layout == "auto":
        if (rows, cols) == (IMAGE_PIXELS, DIGITS):
            layout = "pixel-class"
        elif (rows, cols) == (DIGITS, IMAGE_PIXELS):
            layout = "class-pixel"
        else:
            raise ExportError(
                f"weights must be {IMAGE_PIXELS}x{DIGITS} or {DIGITS}x{IMAGE_PIXELS}; "
                f"found {rows}x{cols}"
            )

    if layout == "pixel-class":
        if (rows, cols) != (IMAGE_PIXELS, DIGITS):
            raise ExportError(
                f"pixel-class weights must be {IMAGE_PIXELS}x{DIGITS}; found {rows}x{cols}"
            )
        return weights, layout

    if layout == "class-pixel":
        if (rows, cols) != (DIGITS, IMAGE_PIXELS):
            raise ExportError(
                f"class-pixel weights must be {DIGITS}x{IMAGE_PIXELS}; found {rows}x{cols}"
            )
        return [[weights[digit][pixel] for digit in range(DIGITS)] for pixel in range(IMAGE_PIXELS)], layout

    raise ExportError(f"unknown input layout `{layout}`")


def coerce_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ExportError(f"{label} must be finite")
    return converted


def coerce_q31_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExportError(f"{label} must be an integer in q31 mode")
    return checked_i64(value, label, clip=False)


def checked_i64(value: int, label: str, clip: bool) -> int:
    if I64_MIN <= value <= I64_MAX:
        return value
    if clip:
        return min(max(value, I64_MIN), I64_MAX)
    raise ExportError(f"{label} quantized outside signed i64 range")


def quantize_value(value: Any, label: str, input_scale: str, clip_i64: bool) -> int:
    if input_scale == "q31":
        return coerce_q31_integer(value, label)
    quantized = int(round(coerce_float(value, label) * Q31_ONE))
    return checked_i64(quantized, label, clip=clip_i64)


def quantize_matrix(
    weights: list[list[Any]],
    input_scale: str,
    clip_i64: bool,
) -> list[list[int]]:
    return [
        [
            quantize_value(value, f"weights[{row}][{col}]", input_scale, clip_i64)
            for col, value in enumerate(values)
        ]
        for row, values in enumerate(weights)
    ]


def quantize_vector(values: Any, input_scale: str, clip_i64: bool) -> list[int]:
    if not isinstance(values, list):
        raise ExportError("bias must be a vector")
    if len(values) != DIGITS:
        raise ExportError(f"bias must have {DIGITS} entries; found {len(values)}")
    return [
        quantize_value(value, f"bias[{index}]", input_scale, clip_i64)
        for index, value in enumerate(values)
    ]


def export_payload(
    source: Any,
    *,
    input_path: str,
    weights_path: Optional[str] = None,
    bias_path: Optional[str] = None,
    input_layout: str = "auto",
    input_scale: str = "float",
    clip_i64: bool = False,
    source_details: Optional[dict[str, Any]] = None,
    source_layout_hint: str = "auto",
) -> dict[str, Any]:
    weights, bias, resolved_weights_path, resolved_bias_path, layout_hint = find_model_arrays(
        source, weights_path, bias_path
    )
    if layout_hint == "auto":
        layout_hint = source_layout_hint
    normalized_weights, resolved_layout = normalize_weights(
        weights, input_layout, layout_hint
    )
    source_report = {
        "input": input_path,
        "weights_path": "/".join(resolved_weights_path),
        "bias_path": "/".join(resolved_bias_path),
        "input_layout": resolved_layout,
        "input_scale": input_scale,
        "q31_one": Q31_ONE,
    }
    if source_details:
        source_report.update(source_details)
    return {
        "format": "weights_bias_q31_json",
        "source_format": "reverie_q31_linear_export",
        "source": source_report,
        "model": {
            "weights": quantize_matrix(normalized_weights, input_scale, clip_i64),
            "bias": quantize_vector(bias, input_scale, clip_i64),
        },
    }


def write_payload(payload: dict[str, Any], output_path: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if str(output_path) == "-":
        sys.stdout.write(encoded)
        return
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(encoded, encoding="utf-8")


def self_test_npy_bytes(values: Any, shape: tuple[int, ...], descr: str) -> bytes:
    if len(shape) == 1:
        flat_values = values
    else:
        flat_values = [value for row in values for value in row]
    format_codes = {
        "<f8": "<d",
        "<f4": "<f",
        "<i8": "<q",
        "<i4": "<i",
    }
    format_code = format_codes[descr]
    payload = b"".join(struct.pack(format_code, value) for value in flat_values)
    shape_text = f"({shape[0]},)" if len(shape) == 1 else str(shape)
    header = (
        "{'descr': '"
        + descr
        + "', 'fortran_order': False, 'shape': "
        + shape_text
        + ", }"
    )
    header_bytes = header.encode("latin1")
    padding = (16 - ((10 + len(header_bytes) + 1) % 16)) % 16
    header_bytes = header_bytes + (b" " * padding) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes + payload


def run_self_tests() -> int:
    def zeros(rows: int, cols: int) -> list[list[float]]:
        return [[0.0 for _ in range(cols)] for _ in range(rows)]

    pixel_class = zeros(IMAGE_PIXELS, DIGITS)
    pixel_class[0][0] = 0.5
    pixel_class[IMAGE_PIXELS - 1][DIGITS - 1] = -0.25
    source = {"weights": pixel_class, "bias": [0.0] * DIGITS}
    exported = export_payload(source, input_path="fixture.json")
    if exported["model"]["weights"][0][0] != Q31_ONE // 2:
        raise AssertionError("pixel-class fixture did not quantize weights[0][0]")
    if exported["model"]["weights"][-1][-1] != -(Q31_ONE // 4):
        raise AssertionError("pixel-class fixture did not preserve final weight")

    class_pixel = zeros(DIGITS, IMAGE_PIXELS)
    class_pixel[3][7] = 0.125
    exported = export_payload(
        {"state_dict": {"linear.weight": class_pixel, "linear.bias": [0.0] * DIGITS}},
        input_path="state-dict.json",
    )
    if exported["source"]["input_layout"] != "class-pixel":
        raise AssertionError("state_dict fixture did not use class-pixel layout")
    if exported["model"]["weights"][7][3] != Q31_ONE // 8:
        raise AssertionError("class-pixel fixture did not transpose to pixel-class")

    q31_source = {"weights": [[0] * DIGITS for _ in range(IMAGE_PIXELS)], "bias": list(range(DIGITS))}
    q31_exported = export_payload(
        q31_source,
        input_path="q31.json",
        input_scale="q31",
    )
    if q31_exported["model"]["bias"] != list(range(DIGITS)):
        raise AssertionError("q31 fixture did not pass integer bias through")

    try:
        export_payload({"weights": [[0.0]], "bias": [0.0] * DIGITS}, input_path="bad.json")
    except ExportError as error:
        if "weights must be" not in str(error):
            raise AssertionError(f"unexpected shape error: {error}") from error
    else:
        raise AssertionError("bad shape fixture unexpectedly exported")

    with tempfile.TemporaryDirectory() as directory:
        source_path = Path(directory) / "source.json"
        output_path = Path(directory) / "model.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        payload = export_payload(
            json.loads(source_path.read_text(encoding="utf-8")),
            input_path=str(source_path),
        )
        write_payload(payload, output_path)
        roundtrip = json.loads(output_path.read_text(encoding="utf-8"))
        if roundtrip["format"] != "weights_bias_q31_json":
            raise AssertionError("roundtrip fixture did not write import format")

        npz_path = Path(directory) / "linear.npz"
        with zipfile.ZipFile(npz_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "linear.weight.npy",
                self_test_npy_bytes(class_pixel, (DIGITS, IMAGE_PIXELS), "<f8"),
            )
            archive.writestr(
                "linear.bias.npy",
                self_test_npy_bytes([0.0] * DIGITS, (DIGITS,), "<f8"),
            )
        npz_source, npz_details, npz_layout_hint = read_npz_model(
            npz_path, None, None
        )
        npz_exported = export_payload(
            npz_source,
            input_path=str(npz_path),
            source_details=npz_details,
            source_layout_hint=npz_layout_hint,
        )
        if npz_exported["source"]["input_format"] != "npz":
            raise AssertionError("npz fixture did not record input_format")
        if npz_exported["source"]["npz_weights"] != "linear.weight":
            raise AssertionError("npz fixture did not record the weight member")
        if npz_exported["model"]["weights"][7][3] != Q31_ONE // 8:
            raise AssertionError("npz fixture did not transpose class-pixel weights")

    print("ok: Q31 linear model exporter self-tests passed")
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
        payload = export_payload(
            source,
            input_path=str(args.input),
            weights_path=args.weights_path,
            bias_path=args.bias_path,
            input_layout=args.input_layout,
            input_scale=args.input_scale,
            clip_i64=args.clip_i64,
            source_details=source_details,
            source_layout_hint=source_layout_hint,
        )
        write_payload(payload, args.output)
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, zipfile.BadZipFile, ExportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if str(args.output) != "-":
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
