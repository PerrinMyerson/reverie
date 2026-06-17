#!/usr/bin/env python3
"""Extract a linear PyTorch state_dict into exporter-readable JSON."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional


DEFAULT_WEIGHTS_KEY = "linear.weight"
DEFAULT_BIAS_KEY = "linear.bias"
FORMAT = "torch_linear_state_dict_json"


class ExtractError(ValueError):
    """Raised when a checkpoint cannot be extracted safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract linear weight and bias tensors from a PyTorch checkpoint "
            "into the JSON state_dict shape accepted by export_q31_linear_model.py."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="PyTorch .pt/.pth checkpoint to read.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination state_dict JSON path, or '-' for stdout.",
    )
    parser.add_argument(
        "--weights-key",
        default=DEFAULT_WEIGHTS_KEY,
        help=(
            "Exact state_dict key for the rank-2 weight tensor. "
            f"Default: {DEFAULT_WEIGHTS_KEY}"
        ),
    )
    parser.add_argument(
        "--bias-key",
        default=DEFAULT_BIAS_KEY,
        help=(
            "Exact state_dict key for the rank-1 bias tensor. "
            f"Default: {DEFAULT_BIAS_KEY}"
        ),
    )
    parser.add_argument(
        "--state-dict-key",
        help=(
            "Slash-separated checkpoint path to a nested state_dict, such as "
            "'state_dict' or 'checkpoint/model_state'. If omitted, the helper "
            "uses checkpoint['state_dict'] when present, otherwise the "
            "checkpoint object itself."
        ),
    )
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help=(
            "Allow legacy torch.load pickle fallback for trusted checkpoints only. "
            "By default the helper uses torch.load(..., weights_only=True)."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent extraction self-tests and exit.",
    )
    return parser.parse_args()


def path_from_argument(path: str) -> tuple[str, ...]:
    parts = tuple(part for part in path.split("/") if part)
    if not parts:
        raise ExtractError("state_dict paths must not be empty")
    return parts


def get_nested(value: Any, path: tuple[str, ...], label: str) -> Any:
    current = value
    for part in path:
        if not isinstance(current, dict) or part not in current:
            joined = "/".join(path)
            raise ExtractError(f"{label} path `{joined}` was not found")
        current = current[part]
    return current


def select_state_dict(
    checkpoint: Any,
    state_dict_key: Optional[str],
) -> tuple[dict[str, Any], Optional[str]]:
    if state_dict_key:
        selected = get_nested(
            checkpoint,
            path_from_argument(state_dict_key),
            "state_dict",
        )
        if not isinstance(selected, dict):
            raise ExtractError(
                f"state_dict path `{state_dict_key}` must point to a mapping"
            )
        return selected, state_dict_key

    if isinstance(checkpoint, dict):
        for common_key in ("state_dict", "model_state_dict"):
            selected = checkpoint.get(common_key)
            if isinstance(selected, dict):
                return selected, common_key
        return checkpoint, None

    raise ExtractError("checkpoint must be a state_dict mapping or contain one")


def require_state_key(state_dict: dict[str, Any], key: str, label: str) -> Any:
    if key not in state_dict:
        raise ExtractError(f"missing {label} key `{key}` in state_dict")
    return state_dict[key]


def call_noarg_method(value: Any, method_name: str, label: str) -> Any:
    method = getattr(value, method_name, None)
    if not callable(method):
        return value
    try:
        return method()
    except TypeError as error:
        raise ExtractError(
            f"{label}.{method_name}() could not be called without arguments"
        ) from error


def tensor_to_list(value: Any, label: str) -> Any:
    value = call_noarg_method(value, "detach", label)
    value = call_noarg_method(value, "cpu", label)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            value = tolist()
        except TypeError as error:
            raise ExtractError(
                f"{label}.tolist() could not be called without arguments"
            ) from error
    return normalize_numeric_array(value, label)


def normalize_numeric_array(value: Any, label: str) -> Any:
    if isinstance(value, bool):
        raise ExtractError(f"{label} must be numeric, not bool")
    if isinstance(value, (int, float)):
        converted = float(value)
        if not math.isfinite(converted):
            raise ExtractError(f"{label} must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [
            normalize_numeric_array(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ExtractError(f"{label} must be numeric or an array of numeric values")


def rectangular_shape(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    child_shapes = [
        rectangular_shape(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    ]
    if not child_shapes:
        return (0,)
    first_shape = child_shapes[0]
    for index, shape in enumerate(child_shapes):
        if shape != first_shape:
            raise ExtractError(
                f"{label} must be rectangular; child {index} has shape {shape}, "
                f"expected {first_shape}"
            )
    return (len(value), *first_shape)


def extract_state_dict_payload(
    checkpoint: Any,
    *,
    checkpoint_path: str,
    weights_key: str = DEFAULT_WEIGHTS_KEY,
    bias_key: str = DEFAULT_BIAS_KEY,
    state_dict_key: Optional[str] = None,
) -> dict[str, Any]:
    state_dict, resolved_state_dict_key = select_state_dict(checkpoint, state_dict_key)
    weights = tensor_to_list(
        require_state_key(state_dict, weights_key, "weight"),
        weights_key,
    )
    bias = tensor_to_list(
        require_state_key(state_dict, bias_key, "bias"),
        bias_key,
    )
    weights_shape = rectangular_shape(weights, weights_key)
    bias_shape = rectangular_shape(bias, bias_key)
    if len(weights_shape) != 2:
        raise ExtractError(f"{weights_key} must be a rank-2 tensor")
    if len(bias_shape) != 1:
        raise ExtractError(f"{bias_key} must be a rank-1 tensor")

    return {
        "format": FORMAT,
        "source": {
            "checkpoint": checkpoint_path,
            "state_dict_key": resolved_state_dict_key,
            "weights_key": weights_key,
            "bias_key": bias_key,
            "weights_shape": list(weights_shape),
            "bias_shape": list(bias_shape),
        },
        "state_dict": {
            weights_key: weights,
            bias_key: bias,
        },
    }


def load_torch_checkpoint(path: Path, allow_unsafe_pickle: bool) -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise ExtractError(
            "PyTorch is required to read .pt/.pth checkpoints; install torch, "
            "or export a state_dict JSON/.npz artifact instead"
        ) from error

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        if not allow_unsafe_pickle:
            raise ExtractError(
                "this PyTorch version does not support weights_only=True; "
                "rerun with --allow-unsafe-pickle only for a trusted checkpoint"
            ) from error
    except Exception as error:
        if not allow_unsafe_pickle:
            raise ExtractError(
                "torch.load(..., weights_only=True) failed; rerun with "
                "--allow-unsafe-pickle only for a trusted legacy checkpoint"
            ) from error

    return torch.load(path, map_location="cpu")


def write_payload(payload: dict[str, Any], output_path: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if str(output_path) == "-":
        sys.stdout.write(encoded)
        return
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(encoded, encoding="utf-8")


class FakeTensor:
    def __init__(self, value: Any):
        self.value = value

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> Any:
        return self.value


def expect_extract_error(label: str, func: Any, needle: str) -> None:
    try:
        func()
    except ExtractError as error:
        if needle not in str(error):
            raise AssertionError(f"{label} reported unexpected error: {error}") from error
        return
    raise AssertionError(f"{label} unexpectedly succeeded")


def run_self_tests() -> int:
    checkpoint = {
        "state_dict": {
            "linear.weight": FakeTensor([[0.25, -0.5], [1.0, 0.0]]),
            "linear.bias": FakeTensor([0.125, -0.25]),
        }
    }
    payload = extract_state_dict_payload(
        checkpoint,
        checkpoint_path="fixture.pt",
    )
    if payload["format"] != FORMAT:
        raise AssertionError("state_dict fixture did not record the output format")
    if payload["source"]["state_dict_key"] != "state_dict":
        raise AssertionError("state_dict fixture did not record the nested key")
    if payload["source"]["weights_shape"] != [2, 2]:
        raise AssertionError("state_dict fixture did not record the weight shape")
    if payload["state_dict"]["linear.weight"][0][1] != -0.5:
        raise AssertionError("state_dict fixture did not extract tensor values")

    nested = {
        "checkpoint": {
            "model_state": {
                "head.weight": ((1, 2, 3), (4, 5, 6)),
                "head.bias": (7, 8),
            }
        }
    }
    nested_payload = extract_state_dict_payload(
        nested,
        checkpoint_path="nested.pt",
        state_dict_key="checkpoint/model_state",
        weights_key="head.weight",
        bias_key="head.bias",
    )
    if nested_payload["state_dict"]["head.weight"][1][2] != 6:
        raise AssertionError("nested fixture did not preserve tuple arrays")

    direct_payload = extract_state_dict_payload(
        {
            "linear.weight": [[0.0, 0.0]],
            "linear.bias": [0.0],
        },
        checkpoint_path="direct.pt",
    )
    if direct_payload["source"]["state_dict_key"] is not None:
        raise AssertionError("direct state_dict fixture should not record a nested key")

    expect_extract_error(
        "missing weight key",
        lambda: extract_state_dict_payload(
            {"state_dict": {"linear.bias": [0.0]}},
            checkpoint_path="bad.pt",
        ),
        "missing weight key",
    )
    expect_extract_error(
        "ragged weights",
        lambda: extract_state_dict_payload(
            {
                "state_dict": {
                    "linear.weight": [[1.0], [2.0, 3.0]],
                    "linear.bias": [0.0, 0.0],
                }
            },
            checkpoint_path="bad.pt",
        ),
        "rectangular",
    )
    expect_extract_error(
        "bool value",
        lambda: extract_state_dict_payload(
            {
                "state_dict": {
                    "linear.weight": [[True]],
                    "linear.bias": [0.0],
                }
            },
            checkpoint_path="bad.pt",
        ),
        "not bool",
    )
    expect_extract_error(
        "rank mismatch",
        lambda: extract_state_dict_payload(
            {
                "state_dict": {
                    "linear.weight": [1.0, 2.0],
                    "linear.bias": [0.0],
                }
            },
            checkpoint_path="bad.pt",
        ),
        "rank-2",
    )

    print("ok: PyTorch linear state_dict extractor self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()

    if args.checkpoint is None or args.output is None:
        print("error: --checkpoint and --output are required", file=sys.stderr)
        return 2

    try:
        checkpoint = load_torch_checkpoint(args.checkpoint, args.allow_unsafe_pickle)
        payload = extract_state_dict_payload(
            checkpoint,
            checkpoint_path=str(args.checkpoint),
            weights_key=args.weights_key,
            bias_key=args.bias_key,
            state_dict_key=args.state_dict_key,
        )
        write_payload(payload, args.output)
    except (OSError, ExtractError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
