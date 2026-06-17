#!/usr/bin/env python3
"""Benchmark Reverie against the public Jana Janus interpreter.

The script intentionally benchmarks a named implementation, not "Janus" in the
abstract. It verifies each program's expected output before timing so benchmark
numbers cannot drift away from semantic agreement.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union


REPO_ROOT = Path(__file__).resolve().parents[1]
JANA_URL = "https://github.com/kirkedal/Jana-JanusInterp.git"
DEFAULT_JANA_DIR = REPO_ROOT / "target" / "jana-baseline"
DEFAULT_REVERIE_BIN = REPO_ROOT / "target" / "release" / "reverie"
DEFAULT_COMMAND_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class CommandSpec:
    argv: list[str]
    expected_stdout: list[str]


SideSpec = Union[CommandSpec, list[CommandSpec]]


@dataclass(frozen=True)
class Benchmark:
    name: str
    jana: SideSpec
    reverie: SideSpec


@dataclass(frozen=True)
class Measurement:
    elapsed_seconds: float
    max_rss_bytes: int


def workload_direction(name: str) -> str:
    if name.endswith("_roundtrip"):
        return "roundtrip"
    if name.endswith("_reverse"):
        return "reverse"
    return "forward"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and time Reverie against Jana-JanusInterp."
    )
    parser.add_argument(
        "--jana-dir",
        type=Path,
        default=DEFAULT_JANA_DIR,
        help="Jana checkout directory, cloned if missing.",
    )
    parser.add_argument(
        "--jana-bin",
        type=Path,
        help="Path to a Jana/Janus executable. Defaults to <jana-dir>/bin/janus.",
    )
    parser.add_argument(
        "--reverie-bin",
        type=Path,
        default=DEFAULT_REVERIE_BIN,
        help="Path to the release Reverie binary.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Measured runs per benchmark and implementation.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Untimed warmup runs per benchmark and implementation.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Benchmark name to run. Can be repeated.",
    )
    parser.add_argument(
        "--list-workloads",
        action="store_true",
        help="List available benchmark workload names and exit.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent harness validation self-tests and exit.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Do not build target/release/reverie before benchmarking.",
    )
    parser.add_argument(
        "--no-clone",
        action="store_true",
        help="Do not clone Jana if --jana-dir is missing.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write machine-readable benchmark results to this JSON file.",
    )
    parser.add_argument(
        "--measure-memory",
        action="store_true",
        help=(
            "Record per-sample peak RSS alongside timing. This uses wait4 on "
            "Unix-like systems and may add a little timing overhead."
        ),
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        help=(
            "Fail if any selected workload's Jana/Reverie median speedup is "
            "below this value."
        ),
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=DEFAULT_COMMAND_TIMEOUT_SECONDS,
        help=(
            "Seconds to allow each verification or timed command before "
            "failing the benchmark."
        ),
    )
    return parser.parse_args()


def run_checked(
    argv: list[str],
    cwd: Path = REPO_ROOT,
    timeout: Optional[float] = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"missing executable `{argv[0]}`") from error
    except OSError as error:
        raise RuntimeError(f"could not run `{argv[0]}`: {error}") from error
    except subprocess.TimeoutExpired as error:
        details = [
            f"command timed out after {error.timeout:.3f}s: {format_command(argv)}",
        ]
        if error.stdout:
            details.append(f"stdout:\n{error.stdout}")
        if error.stderr:
            details.append(f"stderr:\n{error.stderr}")
        raise RuntimeError("\n".join(details)) from error
    except subprocess.CalledProcessError as error:
        details = [
            f"command failed: {format_command(argv)}",
            f"exit status: {error.returncode}",
        ]
        if error.stdout:
            details.append(f"stdout:\n{error.stdout}")
        if error.stderr:
            details.append(f"stderr:\n{error.stderr}")
        raise RuntimeError("\n".join(details)) from error


def format_command(argv: list[str]) -> str:
    return " ".join(shell_quote(arg) for arg in argv)


def shell_quote(arg: str) -> str:
    if all(ch.isalnum() or ch in "./_-=:" for ch in arg):
        return arg
    return "'" + arg.replace("'", "'\"'\"'") + "'"


def ensure_jana_checkout(jana_dir: Path, no_clone: bool) -> None:
    if jana_dir.exists():
        return
    if no_clone:
        raise RuntimeError(
            f"{jana_dir} does not exist. Remove --no-clone or clone {JANA_URL} there."
        )
    jana_dir.parent.mkdir(parents=True, exist_ok=True)
    run_checked(
        ["git", "clone", "--depth", "1", JANA_URL, str(jana_dir)],
        cwd=REPO_ROOT,
    )


def ensure_reverie_binary(reverie_bin: Path, no_build: bool) -> None:
    if no_build:
        if not reverie_bin.exists():
            raise RuntimeError(f"{reverie_bin} does not exist")
        return
    run_checked(["cargo", "build", "--release", "-p", "reverie-cli"])


def resolve_jana_bin(jana_dir: Path, jana_bin: Optional[Path]) -> Path:
    if jana_bin is not None:
        return jana_bin
    bundled = jana_dir / "bin" / "janus"
    if bundled.exists():
        return bundled
    raise RuntimeError(
        "Could not find Jana executable. Build Jana with cabal/stack or pass --jana-bin."
    )


def default_jana_bin(jana_dir: Path, jana_bin: Optional[Path]) -> Path:
    return jana_bin if jana_bin is not None else jana_dir / "bin" / "janus"


def comma_values(values: list[int]) -> str:
    return ", ".join(str(value) for value in values)


def compact_values(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def jana_array(name: str, values: list[int]) -> str:
    return f"{name}[{len(values)}] = {{{comma_values(values)}}}"


def reverie_array(values: list[int]) -> str:
    return f"[{comma_values(values)}]"


def command_sequence(*commands: CommandSpec) -> list[CommandSpec]:
    return list(commands)


def benchmarks(jana_bin: Path, reverie_bin: Path, jana_dir: Path) -> list[Benchmark]:
    fib_final_a = 817770325994397771
    fib_final_b = 9079565065540428013
    factor_output = [0, 2, 2, 2, 3, 5, 7] + [0] * 13
    factor_zero = [0] * 20
    sort_input = list(range(50, 0, -1))
    sort_output = list(range(1, 51))
    sort_perm = list(range(49, -1, -1))
    sort_zero_perm = [0] * 50
    stack_input = [5, 4, 3, 2, 1]
    stack_reversed = [1, 2, 3, 4, 5]
    bitrev_perm = [0, 4, 2, 6, 1, 5, 3, 7]
    bitrev_input = [10, 20, 30, 40, 50, 60, 70, 80]
    bitrev_output = [10, 50, 30, 70, 20, 60, 40, 80]
    perm_code_input = [2, 0, 3, 1, 5, 4]
    perm_code_output = [0, 0, 2, 1, 4, 4]
    rle_data = [2, 2, 2, 5, 5, 1, 1, 1]
    rle_symbols = [2, 5, 1, 0, 0, 0, 0, 0]
    rle_counts = [3, 2, 3, 0, 0, 0, 0, 0]
    rle_zero = [0] * 8
    procedure_forward_args = [
        "--var",
        "n=1000",
        "--var",
        "i=0",
        "--var",
        "x=0",
        "--var",
        "y=1",
    ]
    procedure_reverse_args = [
        "--var",
        "n=1000",
        "--var",
        "i=1000",
        "--var",
        "x=0",
        "--var",
        "y=1",
    ]
    bitrev_forward_args = [
        "--var",
        "n=8",
        "--var",
        f"xs=[{compact_values(bitrev_input)}]",
        "--var",
        f"perm=[{compact_values(bitrev_perm)}]",
    ]
    bitrev_reverse_args = [
        "--var",
        "n=8",
        "--var",
        f"xs=[{compact_values(bitrev_output)}]",
        "--var",
        f"perm=[{compact_values(bitrev_perm)}]",
    ]
    rle_forward_args = [
        "--var",
        "n=8",
        "--var",
        "run=0",
        "--var",
        f"data=[{compact_values(rle_data)}]",
        "--var",
        f"symbols=[{compact_values(rle_zero)}]",
        "--var",
        f"counts=[{compact_values(rle_zero)}]",
    ]
    rle_reverse_args = [
        "--var",
        "n=8",
        "--var",
        "run=3",
        "--var",
        f"data=[{compact_values(rle_data)}]",
        "--var",
        f"symbols=[{compact_values(rle_symbols)}]",
        "--var",
        f"counts=[{compact_values(rle_counts)}]",
    ]
    wave_alphas = [17, 16, 15, 14, 13, 12, 11, 10, 10, 11, 12, 13, 14, 15, 16, 17]
    wave_initial_r = [2, 3, 4, 6, 9, 12, 17, 23, 31, 42, 55, 72, 93, 118, 148, 184]
    wave_zero = [0] * 16
    wave_final_i = [
        1090965397192754749,
        8357816183223517593,
        -903136016014472561,
        -2248272020363884360,
        2370969452233211356,
        7191497602641481837,
        -538919007294570837,
        7307773671519167027,
        1236331457862333962,
        1462271174428245944,
        -9148472539760599954,
        2560704284329755561,
        7627490374347364124,
        1080031045274332963,
        9030791016758320627,
        2932215343476868658,
    ]
    wave_final_r = [
        7492701239833544101,
        -6349349935115478538,
        -8998743017567747236,
        1132692900791859979,
        7732596690609602160,
        -7141354382050438537,
        -619746662238118711,
        4278498264227663332,
        5888699464625555294,
        -4315445307137226226,
        2004386418750668596,
        134370324756951889,
        -8070878621892691292,
        1946795834503771706,
        1118711695460874909,
        5114806227275059749,
    ]
    turing_q1 = [1, 2, 3, 3, 3, 4, 5, 5]
    turing_q2 = [2, 3, 4, 2, 4, 5, 4, 6]
    turing_s1 = [200, 101, 0, 1, 200, 101, 0, 200]
    turing_s2 = [200, 102, 1, 0, 200, 100, 0, 200]
    turing_zero4 = [0] * 4
    turing_zero8 = [0] * 8
    matrix_a = "[[2, 4, 4], [4, 1, 1], [2, 3, 4]]"
    matrix_b = "[[24, -18, 32], [-12, -19, -9], [11, 9, 10]]"
    matrix_zero = "[[0, 0, 0], [0, 0, 0], [0, 0, 0]]"
    matrix_seed_args = [
        "--var",
        "A=[[2,4,4],[4,1,1],[2,3,4]]",
        "--var",
        "B=[[24,-18,32],[-12,-19,-9],[11,9,10]]",
        "--var",
        "LDU=[[0,0,0],[0,0,0],[0,0,0]]",
        "--var",
        "n=3",
        "--var",
        "x=2",
        "--var",
        "y=4",
    ]
    matrix_legacy_store = (
        f"{{a = {matrix_a}, b = {matrix_b}, ldu = {matrix_zero}, "
        "n = 3, x = 2, y = 4}"
    )
    matrix_legacy_zero_store = (
        f"{{a = {matrix_zero}, b = {matrix_zero}, ldu = {matrix_zero}, "
        "n = 0, x = 0, y = 0}"
    )
    transpose_output = "[[1, 4, 7], [2, 5, 8], [3, 6, 9]]"
    transpose_zero = "[[0, 0, 0], [0, 0, 0], [0, 0, 0]]"
    transpose_seed_args = [
        "--var",
        "m=[[1,4,7],[2,5,8],[3,6,9]]",
    ]
    transpose_zero_store = f"{{m = {transpose_zero}}}"

    return [
        Benchmark(
            name="jana_fib_recursive_direct",
            jana=CommandSpec(
                argv=[str(jana_bin), "-m64", str(jana_dir / "examples" / "fib.ja")],
                expected_stdout=[
                    "0 8 13",
                    "n = 5",
                    "x1 = 0",
                    "x2 = 0",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(jana_dir / "examples" / "fib.ja"),
                ],
                expected_stdout=[
                    "0 8 13",
                    "{n = 5, x1 = 0, x2 = 0}",
                ],
            ),
        ),
        Benchmark(
            name="jana_sqrt_direct",
            jana=CommandSpec(
                argv=[str(jana_bin), "-m64", str(jana_dir / "examples" / "sqrt.ja")],
                expected_stdout=[
                    "66 0",
                    "2 8",
                    "num = 2",
                    "root = 8",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(jana_dir / "examples" / "sqrt.ja"),
                ],
                expected_stdout=[
                    "66 0",
                    "2 8",
                    "{num = 2, root = 8}",
                ],
            ),
        ),
        Benchmark(
            name="jana_stack_operations_direct",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(jana_dir / "examples" / "stack-operations.ja"),
                ],
                expected_stdout=[
                    f"s = <{comma_values(stack_input)}]",
                    f"s = <{comma_values(stack_reversed)}]",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(jana_dir / "examples" / "stack-operations.ja"),
                ],
                expected_stdout=[
                    f"s = <{comma_values(stack_input)}]",
                    f"{{s = stack[{comma_values(stack_reversed)}]}}",
                ],
            ),
        ),
        Benchmark(
            name="janus_stack_reverse_cleanup",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(REPO_ROOT / "benchmarks" / "jana" / "stack_reverse_cleanup.ja"),
                ],
                expected_stdout=[
                    f"s = <{comma_values(stack_input)}]",
                    "s = nil",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(REPO_ROOT / "examples" / "janus_stack_reverse.rev"),
                ],
                expected_stdout=[
                    f"s = <{comma_values(stack_input)}]",
                    "{s = nil}",
                ],
            ),
        ),
        Benchmark(
            name="bit_reversal_n8",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(REPO_ROOT / "benchmarks" / "jana" / "bit_reversal_n8.ja"),
                ],
                expected_stdout=[
                    "n = 8",
                    jana_array("perm", bitrev_perm),
                    jana_array("xs", bitrev_output),
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(REPO_ROOT / "examples" / "bit_reversal.rev"),
                    *bitrev_forward_args,
                ],
                expected_stdout=[
                    "{n = 8, "
                    f"perm = {reverie_array(bitrev_perm)}, "
                    f"xs = {reverie_array(bitrev_output)}}}"
                ],
            ),
        ),
        Benchmark(
            name="bit_reversal_n8_reverse",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(REPO_ROOT / "benchmarks" / "jana" / "bit_reversal_n8_reverse.ja"),
                ],
                expected_stdout=[
                    "n = 8",
                    jana_array("perm", bitrev_perm),
                    jana_array("xs", bitrev_input),
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "reverse",
                    str(REPO_ROOT / "examples" / "bit_reversal.rev"),
                    *bitrev_reverse_args,
                ],
                expected_stdout=[
                    "{n = 8, "
                    f"perm = {reverie_array(bitrev_perm)}, "
                    f"xs = {reverie_array(bitrev_input)}}}"
                ],
            ),
        ),
        Benchmark(
            name="bit_reversal_n8_roundtrip",
            jana=command_sequence(
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(REPO_ROOT / "benchmarks" / "jana" / "bit_reversal_n8.ja"),
                    ],
                    expected_stdout=[
                        "n = 8",
                        jana_array("perm", bitrev_perm),
                        jana_array("xs", bitrev_output),
                    ],
                ),
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(REPO_ROOT / "benchmarks" / "jana" / "bit_reversal_n8_reverse.ja"),
                    ],
                    expected_stdout=[
                        "n = 8",
                        jana_array("perm", bitrev_perm),
                        jana_array("xs", bitrev_input),
                    ],
                ),
            ),
            reverie=command_sequence(
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "run",
                        str(REPO_ROOT / "examples" / "bit_reversal.rev"),
                        *bitrev_forward_args,
                    ],
                    expected_stdout=[
                        "{n = 8, "
                        f"perm = {reverie_array(bitrev_perm)}, "
                        f"xs = {reverie_array(bitrev_output)}}}"
                    ],
                ),
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "reverse",
                        str(REPO_ROOT / "examples" / "bit_reversal.rev"),
                        *bitrev_reverse_args,
                    ],
                    expected_stdout=[
                        "{n = 8, "
                        f"perm = {reverie_array(bitrev_perm)}, "
                        f"xs = {reverie_array(bitrev_input)}}}"
                    ],
                ),
            ),
        ),
        Benchmark(
            name="jana_matrixmult_v1_direct",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(jana_dir / "examples" / "matrixmult_v1.0.ja"),
                ],
                expected_stdout=[
                    "A[3][3] = {{2, 4, 4}, {4, 1, 1}, {2, 3, 4}}",
                    "B[3][3] = {{24, -18, 32}, {-12, -19, -9}, {11, 9, 10}}",
                    "LDU[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}",
                    "n = 3",
                    "x = 2",
                    "y = 4",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    "--legacy-janus",
                    str(jana_dir / "examples" / "matrixmult_v1.0.ja"),
                ],
                expected_stdout=[matrix_legacy_store],
            ),
        ),
        Benchmark(
            name="jana_matrixmult_v1_reverse",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(REPO_ROOT / "benchmarks" / "jana" / "matrixmult_v1_reverse.ja"),
                ],
                expected_stdout=[
                    "A[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}",
                    "B[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}",
                    "LDU[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}",
                    "n = 0",
                    "x = 0",
                    "y = 0",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "reverse",
                    "--legacy-janus",
                    str(jana_dir / "examples" / "matrixmult_v1.0.ja"),
                    *matrix_seed_args,
                ],
                expected_stdout=[matrix_legacy_zero_store],
            ),
        ),
        Benchmark(
            name="jana_matrixmult_v1_roundtrip",
            jana=command_sequence(
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(jana_dir / "examples" / "matrixmult_v1.0.ja"),
                    ],
                    expected_stdout=[
                        "A[3][3] = {{2, 4, 4}, {4, 1, 1}, {2, 3, 4}}",
                        "B[3][3] = {{24, -18, 32}, {-12, -19, -9}, {11, 9, 10}}",
                        "LDU[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}",
                        "n = 3",
                        "x = 2",
                        "y = 4",
                    ],
                ),
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(REPO_ROOT / "benchmarks" / "jana" / "matrixmult_v1_reverse.ja"),
                    ],
                    expected_stdout=[
                        "A[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}",
                        "B[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}",
                        "LDU[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}",
                        "n = 0",
                        "x = 0",
                        "y = 0",
                    ],
                ),
            ),
            reverie=command_sequence(
                CommandSpec(
                        argv=[
                            str(reverie_bin),
                            "run",
                            "--legacy-janus",
                            str(jana_dir / "examples" / "matrixmult_v1.0.ja"),
                        ],
                        expected_stdout=[matrix_legacy_store],
                    ),
                CommandSpec(
                        argv=[
                            str(reverie_bin),
                            "reverse",
                            "--legacy-janus",
                            str(jana_dir / "examples" / "matrixmult_v1.0.ja"),
                            *matrix_seed_args,
                        ],
                        expected_stdout=[matrix_legacy_zero_store],
                    ),
                ),
            ),
        Benchmark(
            name="matrix_transpose_3x3",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(REPO_ROOT / "benchmarks" / "jana" / "matrix_transpose_3x3.ja"),
                ],
                expected_stdout=[
                    "m[3][3] = {{1, 4, 7}, {2, 5, 8}, {3, 6, 9}}",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(REPO_ROOT / "benchmarks" / "jana" / "matrix_transpose_3x3.ja"),
                ],
                expected_stdout=[f"{{m = {transpose_output}}}"],
            ),
        ),
        Benchmark(
            name="matrix_transpose_3x3_reverse",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(
                        REPO_ROOT
                        / "benchmarks"
                        / "jana"
                        / "matrix_transpose_3x3_reverse.ja"
                    ),
                ],
                expected_stdout=[
                    "m[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "reverse",
                    str(REPO_ROOT / "benchmarks" / "jana" / "matrix_transpose_3x3.ja"),
                    *transpose_seed_args,
                ],
                expected_stdout=[transpose_zero_store],
            ),
        ),
        Benchmark(
            name="matrix_transpose_3x3_roundtrip",
            jana=command_sequence(
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(
                            REPO_ROOT
                            / "benchmarks"
                            / "jana"
                            / "matrix_transpose_3x3.ja"
                        ),
                    ],
                    expected_stdout=[
                        "m[3][3] = {{1, 4, 7}, {2, 5, 8}, {3, 6, 9}}",
                    ],
                ),
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(
                            REPO_ROOT
                            / "benchmarks"
                            / "jana"
                            / "matrix_transpose_3x3_reverse.ja"
                        ),
                    ],
                    expected_stdout=[
                        "m[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}",
                    ],
                ),
            ),
            reverie=command_sequence(
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "run",
                        str(
                            REPO_ROOT
                            / "benchmarks"
                            / "jana"
                            / "matrix_transpose_3x3.ja"
                        ),
                    ],
                    expected_stdout=[f"{{m = {transpose_output}}}"],
                ),
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "reverse",
                        str(
                            REPO_ROOT
                            / "benchmarks"
                            / "jana"
                            / "matrix_transpose_3x3.ja"
                        ),
                        *transpose_seed_args,
                    ],
                    expected_stdout=[transpose_zero_store],
                ),
            ),
        ),
        Benchmark(
            name="jana_factor_direct",
            jana=CommandSpec(
                argv=[str(jana_bin), "-m64", str(jana_dir / "examples" / "factor.ja")],
                expected_stdout=[
                    jana_array("fact", factor_output),
                    "num = 0",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(jana_dir / "examples" / "factor.ja"),
                ],
                expected_stdout=[
                    "{fact = "
                    f"{reverie_array(factor_output)}, "
                    "num = 0}"
                ],
            ),
        ),
        Benchmark(
            name="jana_perm_to_code_direct",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(jana_dir / "examples" / "perm-to-code.ja"),
                ],
                expected_stdout=[
                    "x[6] = {2, 0, 3, 1, 5, 4}",
                    "x[6] = {0, 0, 2, 1, 4, 4}",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(jana_dir / "examples" / "perm-to-code.ja"),
                ],
                expected_stdout=[
                    "x = [2, 0, 3, 1, 5, 4]",
                    "{x = [0, 0, 2, 1, 4, 4]}",
                ],
            ),
        ),
        Benchmark(
            name="janus_perm_to_code_reverse",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(REPO_ROOT / "benchmarks" / "jana" / "perm_to_code_reverse.ja"),
                ],
                expected_stdout=[jana_array("x", perm_code_input)],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "reverse",
                    str(REPO_ROOT / "examples" / "perm_to_code.rev"),
                    "--var",
                    f"x=[{compact_values(perm_code_output)}]",
                ],
                expected_stdout=[f"{{x = {reverie_array(perm_code_input)}}}"],
            ),
        ),
        Benchmark(
            name="janus_perm_to_code_roundtrip",
            jana=command_sequence(
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(jana_dir / "examples" / "perm-to-code.ja"),
                    ],
                    expected_stdout=[
                        jana_array("x", perm_code_input),
                        jana_array("x", perm_code_output),
                    ],
                ),
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(REPO_ROOT / "benchmarks" / "jana" / "perm_to_code_reverse.ja"),
                    ],
                    expected_stdout=[jana_array("x", perm_code_input)],
                ),
            ),
            reverie=command_sequence(
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "run",
                        str(REPO_ROOT / "examples" / "perm_to_code.rev"),
                        "--var",
                        f"x=[{compact_values(perm_code_input)}]",
                    ],
                    expected_stdout=[f"{{x = {reverie_array(perm_code_output)}}}"],
                ),
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "reverse",
                        str(REPO_ROOT / "examples" / "perm_to_code.rev"),
                        "--var",
                        f"x=[{compact_values(perm_code_output)}]",
                    ],
                    expected_stdout=[f"{{x = {reverie_array(perm_code_input)}}}"],
                ),
            ),
        ),
        Benchmark(
            name="jana_run_length_enc_direct",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(jana_dir / "examples" / "run-length-enc.ja"),
                ],
                expected_stdout=[
                    "arc[14] = {1, 2, 2, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}",
                    "text[7] = {0, 0, 0, 0, 0, 0, 0}",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(jana_dir / "examples" / "run-length-enc.ja"),
                ],
                expected_stdout=[
                    "{arc = [1, 2, 2, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], "
                    "text = [0, 0, 0, 0, 0, 0, 0]}"
                ],
            ),
        ),
        Benchmark(
            name="jana_run_length_enc_stack_direct",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(jana_dir / "examples" / "run-length-enc-stack.ja"),
                ],
                expected_stdout=[
                    "text = <32, 32, 53, 53, 53, 12]",
                    "arc = <1, 12, 3, 53, 2, 32]",
                    "text = nil",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(jana_dir / "examples" / "run-length-enc-stack.ja"),
                ],
                expected_stdout=[
                    "text = <32, 32, 53, 53, 53, 12]",
                    "{arc = stack[1, 12, 3, 53, 2, 32], text = nil}",
                ],
            ),
        ),
        Benchmark(
            name="rle_compression_n8",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(REPO_ROOT / "benchmarks" / "jana" / "rle_compression_n8.ja"),
                ],
                expected_stdout=[
                    jana_array("counts", rle_counts),
                    jana_array("data", rle_data),
                    "n = 8",
                    "run = 3",
                    jana_array("symbols", rle_symbols),
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(REPO_ROOT / "examples" / "rle_compression.rev"),
                    *rle_forward_args,
                ],
                expected_stdout=[
                    f"{{counts = {reverie_array(rle_counts)}, "
                    f"data = {reverie_array(rle_data)}, "
                    "n = 8, run = 3, "
                    f"symbols = {reverie_array(rle_symbols)}}}"
                ],
            ),
        ),
        Benchmark(
            name="rle_compression_n8_reverse",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(
                        REPO_ROOT
                        / "benchmarks"
                        / "jana"
                        / "rle_compression_n8_reverse.ja"
                    ),
                ],
                expected_stdout=[
                    jana_array("counts", rle_zero),
                    jana_array("data", rle_data),
                    "n = 8",
                    "run = 0",
                    jana_array("symbols", rle_zero),
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "reverse",
                    str(REPO_ROOT / "examples" / "rle_compression.rev"),
                    *rle_reverse_args,
                ],
                expected_stdout=[
                    f"{{counts = {reverie_array(rle_zero)}, "
                    f"data = {reverie_array(rle_data)}, "
                    "n = 8, run = 0, "
                    f"symbols = {reverie_array(rle_zero)}}}"
                ],
            ),
        ),
        Benchmark(
            name="rle_compression_n8_roundtrip",
            jana=command_sequence(
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(
                            REPO_ROOT
                            / "benchmarks"
                            / "jana"
                            / "rle_compression_n8.ja"
                        ),
                    ],
                    expected_stdout=[
                        jana_array("counts", rle_counts),
                        jana_array("data", rle_data),
                        "n = 8",
                        "run = 3",
                        jana_array("symbols", rle_symbols),
                    ],
                ),
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(
                            REPO_ROOT
                            / "benchmarks"
                            / "jana"
                            / "rle_compression_n8_reverse.ja"
                        ),
                    ],
                    expected_stdout=[
                        jana_array("counts", rle_zero),
                        jana_array("data", rle_data),
                        "n = 8",
                        "run = 0",
                        jana_array("symbols", rle_zero),
                    ],
                ),
            ),
            reverie=command_sequence(
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "run",
                        str(REPO_ROOT / "examples" / "rle_compression.rev"),
                        *rle_forward_args,
                    ],
                    expected_stdout=[
                        f"{{counts = {reverie_array(rle_counts)}, "
                        f"data = {reverie_array(rle_data)}, "
                        "n = 8, run = 3, "
                        f"symbols = {reverie_array(rle_symbols)}}}"
                    ],
                ),
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "reverse",
                        str(REPO_ROOT / "examples" / "rle_compression.rev"),
                        *rle_reverse_args,
                    ],
                    expected_stdout=[
                        f"{{counts = {reverie_array(rle_zero)}, "
                        f"data = {reverie_array(rle_data)}, "
                        "n = 8, run = 0, "
                        f"symbols = {reverie_array(rle_zero)}}}"
                    ],
                ),
            ),
        ),
        Benchmark(
            name="fib_loop_n1000",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(REPO_ROOT / "benchmarks" / "jana" / "fib_loop_n1000.ja"),
                ],
                expected_stdout=[
                    f"1000 1000 {fib_final_a} {fib_final_b}",
                    f"a = {fib_final_a}",
                    f"b = {fib_final_b}",
                    "i = 1000",
                    "n = 1000",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(REPO_ROOT / "examples" / "fib.rev"),
                    "--var",
                    "n=1000",
                    "--var",
                    "i=0",
                    "--var",
                    "a=0",
                    "--var",
                    "b=1",
                ],
                expected_stdout=[
                    f"{{a = {fib_final_a}, b = {fib_final_b}, i = 1000, n = 1000}}"
                ],
            ),
        ),
        Benchmark(
            name="fib_loop_n1000_reverse",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(REPO_ROOT / "benchmarks" / "jana" / "fib_loop_n1000_reverse.ja"),
                ],
                expected_stdout=[
                    "1000 0 0 1",
                    "a = 0",
                    "b = 1",
                    "i = 0",
                    "n = 1000",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "reverse",
                    str(REPO_ROOT / "examples" / "fib.rev"),
                    "--var",
                    "n=1000",
                    "--var",
                    "i=1000",
                    "--var",
                    f"a={fib_final_a}",
                    "--var",
                    f"b={fib_final_b}",
                ],
                expected_stdout=[
                    "{a = 0, b = 1, i = 0, n = 1000}"
                ],
            ),
        ),
        Benchmark(
            name="fib_loop_n1000_roundtrip",
            jana=command_sequence(
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(REPO_ROOT / "benchmarks" / "jana" / "fib_loop_n1000.ja"),
                    ],
                    expected_stdout=[
                        f"1000 1000 {fib_final_a} {fib_final_b}",
                        f"a = {fib_final_a}",
                        f"b = {fib_final_b}",
                        "i = 1000",
                        "n = 1000",
                    ],
                ),
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(REPO_ROOT / "benchmarks" / "jana" / "fib_loop_n1000_reverse.ja"),
                    ],
                    expected_stdout=[
                        "1000 0 0 1",
                        "a = 0",
                        "b = 1",
                        "i = 0",
                        "n = 1000",
                    ],
                ),
            ),
            reverie=command_sequence(
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "run",
                        str(REPO_ROOT / "examples" / "fib.rev"),
                        "--var",
                        "n=1000",
                        "--var",
                        "i=0",
                        "--var",
                        "a=0",
                        "--var",
                        "b=1",
                    ],
                    expected_stdout=[
                        f"{{a = {fib_final_a}, b = {fib_final_b}, i = 1000, n = 1000}}"
                    ],
                ),
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "reverse",
                        str(REPO_ROOT / "examples" / "fib.rev"),
                        "--var",
                        "n=1000",
                        "--var",
                        "i=1000",
                        "--var",
                        f"a={fib_final_a}",
                        "--var",
                        f"b={fib_final_b}",
                    ],
                    expected_stdout=["{a = 0, b = 1, i = 0, n = 1000}"],
                ),
            ),
        ),
        Benchmark(
            name="procedure_call_n1000",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(REPO_ROOT / "benchmarks" / "jana" / "procedure_call_n1000.ja"),
                ],
                expected_stdout=[
                    "1000 1000 0 1",
                    "i = 1000",
                    "n = 1000",
                    "x = 0",
                    "y = 1",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(REPO_ROOT / "examples" / "procedure_call_loop.rev"),
                    *procedure_forward_args,
                ],
                expected_stdout=["{i = 1000, n = 1000, x = 0, y = 1}"],
            ),
        ),
        Benchmark(
            name="procedure_call_n1000_reverse",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(
                        REPO_ROOT
                        / "benchmarks"
                        / "jana"
                        / "procedure_call_n1000_reverse.ja"
                    ),
                ],
                expected_stdout=[
                    "1000 0 0 1",
                    "i = 0",
                    "n = 1000",
                    "x = 0",
                    "y = 1",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "reverse",
                    str(REPO_ROOT / "examples" / "procedure_call_loop.rev"),
                    *procedure_reverse_args,
                ],
                expected_stdout=["{i = 0, n = 1000, x = 0, y = 1}"],
            ),
        ),
        Benchmark(
            name="procedure_call_n1000_roundtrip",
            jana=command_sequence(
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(
                            REPO_ROOT
                            / "benchmarks"
                            / "jana"
                            / "procedure_call_n1000.ja"
                        ),
                    ],
                    expected_stdout=[
                        "1000 1000 0 1",
                        "i = 1000",
                        "n = 1000",
                        "x = 0",
                        "y = 1",
                    ],
                ),
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(
                            REPO_ROOT
                            / "benchmarks"
                            / "jana"
                            / "procedure_call_n1000_reverse.ja"
                        ),
                    ],
                    expected_stdout=[
                        "1000 0 0 1",
                        "i = 0",
                        "n = 1000",
                        "x = 0",
                        "y = 1",
                    ],
                ),
            ),
            reverie=command_sequence(
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "run",
                        str(REPO_ROOT / "examples" / "procedure_call_loop.rev"),
                        *procedure_forward_args,
                    ],
                    expected_stdout=["{i = 1000, n = 1000, x = 0, y = 1}"],
                ),
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "reverse",
                        str(REPO_ROOT / "examples" / "procedure_call_loop.rev"),
                        *procedure_reverse_args,
                    ],
                    expected_stdout=["{i = 0, n = 1000, x = 0, y = 1}"],
                ),
            ),
        ),
        Benchmark(
            name="janus_root_66",
            jana=CommandSpec(
                argv=[str(jana_bin), "-m64", str(jana_dir / "examples" / "sqrt.ja")],
                expected_stdout=["num = 2", "root = 8"],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(REPO_ROOT / "examples" / "janus_root.rev"),
                    "--var",
                    "num=66",
                ],
                expected_stdout=["{bit = 0, num = 2, root = 8, z = 0}"],
            ),
        ),
        Benchmark(
            name="janus_root_66_reverse",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(REPO_ROOT / "benchmarks" / "jana" / "root_66_reverse.ja"),
                ],
                expected_stdout=["66 0", "num = 66", "root = 0"],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "reverse",
                    str(REPO_ROOT / "examples" / "janus_root.rev"),
                    "--var",
                    "num=2",
                    "--var",
                    "root=8",
                    "--var",
                    "bit=0",
                    "--var",
                    "z=0",
                ],
                expected_stdout=["{bit = 0, num = 66, root = 0, z = 0}"],
            ),
        ),
        Benchmark(
            name="janus_factor_840",
            jana=CommandSpec(
                argv=[str(jana_bin), "-m64", str(jana_dir / "examples" / "factor.ja")],
                expected_stdout=[
                    jana_array("fact", factor_output),
                    "num = 0",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(REPO_ROOT / "examples" / "janus_factor.rev"),
                    "--var",
                    "num=840",
                ],
                expected_stdout=[
                    "{fact = "
                    f"{reverie_array(factor_output)}, "
                    "i = 0, num = 0, try = 0, z = 0}"
                ],
            ),
        ),
        Benchmark(
            name="janus_factor_840_reverse",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(REPO_ROOT / "benchmarks" / "jana" / "factor_840_reverse.ja"),
                ],
                expected_stdout=[
                    jana_array("fact", factor_zero),
                    "num = 840",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "reverse",
                    str(REPO_ROOT / "examples" / "janus_factor.rev"),
                    "--var",
                    "num=0",
                    "--var",
                    "try=0",
                    "--var",
                    "z=0",
                    "--var",
                    "i=0",
                    "--var",
                    f"fact=[{compact_values(factor_output)}]",
                ],
                expected_stdout=[
                    "{fact = "
                    f"{reverie_array(factor_zero)}, "
                    "i = 0, num = 840, try = 0, z = 0}"
                ],
            ),
        ),
        Benchmark(
            name="janus_sort_n50_reverse_order",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(REPO_ROOT / "benchmarks" / "jana" / "sort_n50_reverse.ja"),
                ],
                expected_stdout=[
                    "i = 0",
                    "j = 0",
                    jana_array("list", sort_output),
                    "n = 50",
                    jana_array("perm", sort_perm),
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(REPO_ROOT / "benchmarks" / "jana" / "sort_n50_reverse.ja"),
                    "--var",
                    "n=50",
                    "--var",
                    "i=0",
                    "--var",
                    "j=0",
                    "--var",
                    f"list=[{compact_values(sort_input)}]",
                    "--var",
                    f"perm=[{compact_values(sort_zero_perm)}]",
                ],
                expected_stdout=[
                    "{i = 0, j = 0, "
                    f"list = {reverie_array(sort_output)}, "
                    f"n = 50, perm = {reverie_array(sort_perm)}}}"
                ],
            ),
        ),
        Benchmark(
            name="janus_sort_n50_reverse_order_reverse",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(
                        REPO_ROOT
                        / "benchmarks"
                        / "jana"
                        / "sort_n50_reverse_order_reverse.ja"
                    ),
                ],
                expected_stdout=[
                    "i = 0",
                    "j = 0",
                    jana_array("list", sort_input),
                    "n = 50",
                    jana_array("perm", sort_zero_perm),
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "reverse",
                    str(REPO_ROOT / "benchmarks" / "jana" / "sort_n50_reverse.ja"),
                    "--var",
                    "n=50",
                    "--var",
                    "i=0",
                    "--var",
                    "j=0",
                    "--var",
                    f"list=[{compact_values(sort_output)}]",
                    "--var",
                    f"perm=[{compact_values(sort_perm)}]",
                ],
                expected_stdout=[
                    "{i = 0, j = 0, "
                    f"list = {reverie_array(sort_input)}, "
                    f"n = 50, perm = {reverie_array(sort_zero_perm)}}}"
                ],
            ),
        ),
        Benchmark(
            name="janus_sort_n50_reverse_order_roundtrip",
            jana=command_sequence(
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(REPO_ROOT / "benchmarks" / "jana" / "sort_n50_reverse.ja"),
                    ],
                    expected_stdout=[
                        "i = 0",
                        "j = 0",
                        jana_array("list", sort_output),
                        "n = 50",
                        jana_array("perm", sort_perm),
                    ],
                ),
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(
                            REPO_ROOT
                            / "benchmarks"
                            / "jana"
                            / "sort_n50_reverse_order_reverse.ja"
                        ),
                    ],
                    expected_stdout=[
                        "i = 0",
                        "j = 0",
                        jana_array("list", sort_input),
                        "n = 50",
                        jana_array("perm", sort_zero_perm),
                    ],
                ),
            ),
            reverie=command_sequence(
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "run",
                        str(REPO_ROOT / "benchmarks" / "jana" / "sort_n50_reverse.ja"),
                        "--var",
                        "n=50",
                        "--var",
                        "i=0",
                        "--var",
                        "j=0",
                        "--var",
                        f"list=[{compact_values(sort_input)}]",
                        "--var",
                        f"perm=[{compact_values(sort_zero_perm)}]",
                    ],
                    expected_stdout=[
                        "{i = 0, j = 0, "
                        f"list = {reverie_array(sort_output)}, "
                        f"n = 50, perm = {reverie_array(sort_perm)}}}"
                    ],
                ),
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "reverse",
                        str(REPO_ROOT / "benchmarks" / "jana" / "sort_n50_reverse.ja"),
                        "--var",
                        "n=50",
                        "--var",
                        "i=0",
                        "--var",
                        "j=0",
                        "--var",
                        f"list=[{compact_values(sort_output)}]",
                        "--var",
                        f"perm=[{compact_values(sort_perm)}]",
                    ],
                    expected_stdout=[
                        "{i = 0, j = 0, "
                        f"list = {reverie_array(sort_input)}, "
                        f"n = 50, perm = {reverie_array(sort_zero_perm)}}}"
                    ],
                ),
            ),
        ),
        Benchmark(
            name="janus_schroedinger_n16_t100",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(REPO_ROOT / "benchmarks" / "jana" / "schroedinger_n16_t100.ja"),
                ],
                expected_stdout=[
                    jana_array("alphas", wave_alphas),
                    "epsilon = 3",
                    "i = 0",
                    "j = 0",
                    jana_array("psiI", wave_final_i),
                    jana_array("psiR", wave_final_r),
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(REPO_ROOT / "benchmarks" / "jana" / "schroedinger_n16_t100.ja"),
                ],
                expected_stdout=[
                    f"alphas = {reverie_array(wave_alphas)}",
                    "epsilon = 3",
                    "i = 0, j = 0",
                    f"psiI = {reverie_array(wave_final_i)}",
                    f"psiR = {reverie_array(wave_final_r)}",
                ],
            ),
        ),
        Benchmark(
            name="janus_turing_binary_inc",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(REPO_ROOT / "benchmarks" / "jana" / "turing_binary_inc.ja"),
                ],
                expected_stdout=[
                    "PC_MAX = 8",
                    "QF = 6",
                    "QS = 1",
                    "blank = 200",
                    "left = 100",
                    "leftsp = 0",
                    jana_array("leftstk", turing_zero4),
                    "pc = 0",
                    "q = 6",
                    jana_array("q1", turing_q1),
                    jana_array("q2", turing_q2),
                    "right = 102",
                    "rightsp = 0",
                    jana_array("rightstk", turing_zero4),
                    "s = 200",
                    jana_array("s1", turing_s1),
                    jana_array("s2", turing_s2),
                    "slash = 101",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(REPO_ROOT / "benchmarks" / "jana" / "turing_binary_inc.ja"),
                ],
                expected_stdout=[
                    "PC_MAX = 8",
                    "QF = 6",
                    "QS = 1",
                    "blank = 200",
                    "left = 100",
                    "leftsp = 0",
                    f"leftstk = {reverie_array(turing_zero4)}",
                    "pc = 0",
                    "q = 6",
                    f"q1 = {reverie_array(turing_q1)}",
                    f"q2 = {reverie_array(turing_q2)}",
                    "right = 102",
                    "rightsp = 0",
                    f"rightstk = {reverie_array(turing_zero4)}",
                    "s = 200",
                    f"s1 = {reverie_array(turing_s1)}",
                    f"s2 = {reverie_array(turing_s2)}",
                    "slash = 101",
                ],
            ),
        ),
        Benchmark(
            name="janus_turing_binary_inc_reverse",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(
                        REPO_ROOT
                        / "benchmarks"
                        / "jana"
                        / "turing_binary_inc_reverse.ja"
                    ),
                ],
                expected_stdout=[
                    "PC_MAX = 0",
                    "QF = 0",
                    "QS = 0",
                    "blank = 0",
                    "left = 0",
                    "leftsp = 0",
                    jana_array("leftstk", turing_zero4),
                    "pc = 0",
                    "q = 0",
                    jana_array("q1", turing_zero8),
                    jana_array("q2", turing_zero8),
                    "right = 0",
                    "rightsp = 0",
                    jana_array("rightstk", turing_zero4),
                    "s = 0",
                    jana_array("s1", turing_zero8),
                    jana_array("s2", turing_zero8),
                    "slash = 0",
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(
                        REPO_ROOT
                        / "benchmarks"
                        / "jana"
                        / "turing_binary_inc_reverse.ja"
                    ),
                ],
                expected_stdout=[
                    "PC_MAX = 0",
                    "QF = 0",
                    "QS = 0",
                    "blank = 0",
                    "left = 0",
                    "leftsp = 0",
                    f"leftstk = {reverie_array(turing_zero4)}",
                    "pc = 0",
                    "q = 0",
                    f"q1 = {reverie_array(turing_zero8)}",
                    f"q2 = {reverie_array(turing_zero8)}",
                    "right = 0",
                    "rightsp = 0",
                    f"rightstk = {reverie_array(turing_zero4)}",
                    "s = 0",
                    f"s1 = {reverie_array(turing_zero8)}",
                    f"s2 = {reverie_array(turing_zero8)}",
                    "slash = 0",
                ],
            ),
        ),
        Benchmark(
            name="janus_turing_binary_inc_roundtrip",
            jana=command_sequence(
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(REPO_ROOT / "benchmarks" / "jana" / "turing_binary_inc.ja"),
                    ],
                    expected_stdout=[
                        "q = 6",
                        jana_array("q1", turing_q1),
                        jana_array("s1", turing_s1),
                        jana_array("s2", turing_s2),
                    ],
                ),
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(
                            REPO_ROOT
                            / "benchmarks"
                            / "jana"
                            / "turing_binary_inc_reverse.ja"
                        ),
                    ],
                    expected_stdout=[
                        "q = 0",
                        jana_array("q1", turing_zero8),
                        jana_array("s1", turing_zero8),
                        jana_array("s2", turing_zero8),
                    ],
                ),
            ),
            reverie=command_sequence(
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "run",
                        str(REPO_ROOT / "benchmarks" / "jana" / "turing_binary_inc.ja"),
                    ],
                    expected_stdout=[
                        "q = 6",
                        f"q1 = {reverie_array(turing_q1)}",
                        f"s1 = {reverie_array(turing_s1)}",
                        f"s2 = {reverie_array(turing_s2)}",
                    ],
                ),
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "run",
                        str(
                            REPO_ROOT
                            / "benchmarks"
                            / "jana"
                            / "turing_binary_inc_reverse.ja"
                        ),
                    ],
                    expected_stdout=[
                        "q = 0",
                        f"q1 = {reverie_array(turing_zero8)}",
                        f"s1 = {reverie_array(turing_zero8)}",
                        f"s2 = {reverie_array(turing_zero8)}",
                    ],
                ),
            ),
        ),
        Benchmark(
            name="janus_schroedinger_n16_t100_reverse",
            jana=CommandSpec(
                argv=[
                    str(jana_bin),
                    "-m64",
                    str(
                        REPO_ROOT
                        / "benchmarks"
                        / "jana"
                        / "schroedinger_n16_t100_reverse.ja"
                    ),
                ],
                expected_stdout=[
                    jana_array("alphas", wave_alphas),
                    "epsilon = 3",
                    "i = 0",
                    "j = 0",
                    jana_array("psiI", wave_zero),
                    jana_array("psiR", wave_initial_r),
                ],
            ),
            reverie=CommandSpec(
                argv=[
                    str(reverie_bin),
                    "run",
                    str(
                        REPO_ROOT
                        / "benchmarks"
                        / "jana"
                        / "schroedinger_n16_t100_reverse.ja"
                    ),
                ],
                expected_stdout=[
                    f"alphas = {reverie_array(wave_alphas)}",
                    "epsilon = 3",
                    "i = 0, j = 0",
                    f"psiI = {reverie_array(wave_zero)}",
                    f"psiR = {reverie_array(wave_initial_r)}",
                ],
            ),
        ),
        Benchmark(
            name="janus_schroedinger_n16_t100_roundtrip",
            jana=command_sequence(
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(REPO_ROOT / "benchmarks" / "jana" / "schroedinger_n16_t100.ja"),
                    ],
                    expected_stdout=[
                        jana_array("alphas", wave_alphas),
                        "epsilon = 3",
                        "i = 0",
                        "j = 0",
                        jana_array("psiI", wave_final_i),
                        jana_array("psiR", wave_final_r),
                    ],
                ),
                CommandSpec(
                    argv=[
                        str(jana_bin),
                        "-m64",
                        str(
                            REPO_ROOT
                            / "benchmarks"
                            / "jana"
                            / "schroedinger_n16_t100_reverse.ja"
                        ),
                    ],
                    expected_stdout=[
                        jana_array("alphas", wave_alphas),
                        "epsilon = 3",
                        "i = 0",
                        "j = 0",
                        jana_array("psiI", wave_zero),
                        jana_array("psiR", wave_initial_r),
                    ],
                ),
            ),
            reverie=command_sequence(
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "run",
                        str(REPO_ROOT / "benchmarks" / "jana" / "schroedinger_n16_t100.ja"),
                    ],
                    expected_stdout=[
                        f"alphas = {reverie_array(wave_alphas)}",
                        "epsilon = 3",
                        "i = 0, j = 0",
                        f"psiI = {reverie_array(wave_final_i)}",
                        f"psiR = {reverie_array(wave_final_r)}",
                    ],
                ),
                CommandSpec(
                    argv=[
                        str(reverie_bin),
                        "run",
                        str(
                            REPO_ROOT
                            / "benchmarks"
                            / "jana"
                            / "schroedinger_n16_t100_reverse.ja"
                        ),
                    ],
                    expected_stdout=[
                        f"alphas = {reverie_array(wave_alphas)}",
                        "epsilon = 3",
                        "i = 0, j = 0",
                        f"psiI = {reverie_array(wave_zero)}",
                        f"psiR = {reverie_array(wave_initial_r)}",
                    ],
                ),
            ),
        ),
    ]


def side_commands(side: SideSpec) -> list[CommandSpec]:
    return [side] if isinstance(side, CommandSpec) else side


def format_side(side: SideSpec) -> str:
    return " && ".join(format_command(command.argv) for command in side_commands(side))


def verify(spec: CommandSpec, timeout: float) -> None:
    completed = run_checked(spec.argv, timeout=timeout)
    verify_completed(spec, completed)


def verify_completed(
    spec: CommandSpec,
    completed: subprocess.CompletedProcess[str],
) -> None:
    for expected in spec.expected_stdout:
        if expected not in completed.stdout:
            raise RuntimeError(
                "\n".join(
                    [
                        f"missing expected output `{expected}`",
                        f"command: {format_command(spec.argv)}",
                        f"stdout:\n{completed.stdout}",
                        f"stderr:\n{completed.stderr}",
                    ]
                )
            )


def verify_side(side: SideSpec, timeout: float) -> None:
    for spec in side_commands(side):
        verify(spec, timeout)


def run_checked_measured(
    argv: list[str],
    cwd: Path = REPO_ROOT,
    timeout: Optional[float] = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> tuple[subprocess.CompletedProcess[str], int]:
    stdout_file = tempfile.TemporaryFile(mode="w+b")
    stderr_file = tempfile.TemporaryFile(mode="w+b")
    try:
        pid = os.fork()
    except AttributeError:
        completed = run_checked(argv, cwd=cwd, timeout=timeout)
        return completed, 0
    except OSError as error:
        raise RuntimeError(f"could not fork for `{argv[0]}`: {error}") from error

    if pid == 0:
        try:
            os.chdir(cwd)
            os.dup2(stdout_file.fileno(), 1)
            os.dup2(stderr_file.fileno(), 2)
            stdout_file.close()
            stderr_file.close()
            os.execvp(argv[0], argv)
        except OSError as error:
            message = f"could not run `{argv[0]}`: {error}\n".encode()
            try:
                os.write(2, message)
            finally:
                os._exit(127 if error.errno == errno.ENOENT else 126)

    deadline = None if timeout is None else time.monotonic() + timeout
    rusage = None
    status = 0
    while True:
        waited_pid, status, rusage = os.wait4(pid, os.WNOHANG)
        if waited_pid == pid:
            break
        if deadline is not None and time.monotonic() >= deadline:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, _, killed_rusage = os.wait4(pid, 0)
            stdout = read_tempfile_text(stdout_file)
            stderr = read_tempfile_text(stderr_file)
            details = [
                f"command timed out after {timeout:.3f}s: {format_command(argv)}",
            ]
            if stdout:
                details.append(f"stdout:\n{stdout}")
            if stderr:
                details.append(f"stderr:\n{stderr}")
            raise RuntimeError("\n".join(details))
        time.sleep(0.001)

    stdout = read_tempfile_text(stdout_file)
    stderr = read_tempfile_text(stderr_file)
    returncode = wait_status_to_returncode(status)
    completed = subprocess.CompletedProcess(argv, returncode, stdout, stderr)
    if returncode != 0:
        details = [
            f"command failed: {format_command(argv)}",
            f"exit status: {returncode}",
        ]
        if stdout:
            details.append(f"stdout:\n{stdout}")
        if stderr:
            details.append(f"stderr:\n{stderr}")
        raise RuntimeError("\n".join(details))
    max_rss_bytes = rusage_max_rss_bytes(rusage.ru_maxrss if rusage else 0)
    return completed, max_rss_bytes


def read_tempfile_text(handle: Any) -> str:
    handle.seek(0)
    return handle.read().decode("utf-8", errors="replace")


def wait_status_to_returncode(status: int) -> int:
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    return status


def rusage_max_rss_bytes(max_rss: int) -> int:
    if max_rss <= 0:
        return 0
    if sys.platform == "darwin":
        return int(max_rss)
    return int(max_rss) * 1024


def measure_side(side: SideSpec, timeout: float) -> Measurement:
    start = time.perf_counter()
    max_rss_bytes = 0
    for spec in side_commands(side):
        completed, command_max_rss_bytes = run_checked_measured(
            spec.argv,
            timeout=timeout,
        )
        verify_completed(spec, completed)
        max_rss_bytes = max(max_rss_bytes, command_max_rss_bytes)
    return Measurement(
        elapsed_seconds=time.perf_counter() - start,
        max_rss_bytes=max_rss_bytes,
    )


def time_command(
    side: SideSpec,
    warmup: int,
    runs: int,
    timeout: float,
    measure_memory: bool,
) -> list[Measurement]:
    for _ in range(warmup):
        verify_side(side, timeout)

    samples = []
    for _ in range(runs):
        if measure_memory:
            samples.append(measure_side(side, timeout))
        else:
            start = time.perf_counter()
            verify_side(side, timeout)
            samples.append(
                Measurement(
                    elapsed_seconds=time.perf_counter() - start,
                    max_rss_bytes=0,
                )
            )
    return samples


def side_json(side: SideSpec) -> dict[str, Any]:
    commands = side_commands(side)
    payload: dict[str, Any] = {
        "commands": [command.argv for command in commands],
        "expected_stdout": [command.expected_stdout for command in commands],
        "source_files": source_file_metadata(commands),
    }
    if len(commands) == 1:
        payload["command"] = commands[0].argv
    return payload


def source_file_metadata(commands: list[CommandSpec]) -> list[dict[str, str]]:
    files: dict[str, str] = {}
    for command in commands:
        for arg in command.argv:
            if arg.endswith((".ja", ".janus", ".rev")):
                path = Path(arg)
                files[str(path)] = file_sha256(path)
    return [{"path": path, "sha256": sha256} for path, sha256 in sorted(files.items())]


def summarize(samples: list[Measurement]) -> str:
    seconds = elapsed_samples(samples)
    rss = rss_samples(samples)
    median_ms = statistics.median(seconds) * 1_000
    mean_ms = statistics.mean(seconds) * 1_000
    min_ms = min(seconds) * 1_000
    rss_text = (
        f" peak-rss-median={format_bytes(statistics.median(rss))}"
        if any(rss)
        else ""
    )
    return (
        f"median={median_ms:.3f}ms mean={mean_ms:.3f}ms "
        f"min={min_ms:.3f}ms{rss_text}"
    )


def elapsed_samples(samples: list[Measurement]) -> list[float]:
    return [sample.elapsed_seconds for sample in samples]


def rss_samples(samples: list[Measurement]) -> list[int]:
    return [sample.max_rss_bytes for sample in samples]


def sample_summary(samples: list[Measurement]) -> dict[str, Any]:
    seconds = elapsed_samples(samples)
    rss = rss_samples(samples)
    payload: dict[str, Any] = {
        "median_seconds": statistics.median(seconds),
        "mean_seconds": statistics.mean(seconds),
        "min_seconds": min(seconds),
        "samples_seconds": seconds,
    }
    if any(rss):
        payload.update(
            {
                "median_max_rss_bytes": statistics.median(rss),
                "mean_max_rss_bytes": statistics.mean(rss),
                "max_rss_bytes": max(rss),
                "samples_max_rss_bytes": rss,
            }
        )
    return payload


def format_bytes(value: Union[int, float]) -> str:
    return f"{value / (1024 * 1024):.2f}MiB"


def run_optional(argv: list[str], cwd: Path) -> Optional[subprocess.CompletedProcess[str]]:
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, OSError):
        return None


def git_metadata(path: Path) -> dict[str, Any]:
    revision = run_optional(["git", "rev-parse", "HEAD"], cwd=path)
    status = run_optional(["git", "status", "--short"], cwd=path)
    return {
        "path": str(path),
        "revision": (
            revision.stdout.strip()
            if revision is not None and revision.returncode == 0
            else None
        ),
        "dirty": (
            bool(status.stdout.strip())
            if status is not None and status.returncode == 0
            else None
        ),
        "status": (
            status.stdout.splitlines()
            if status is not None and status.returncode == 0
            else []
        ),
    }


def binary_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": file_sha256(path),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_metadata(jana_dir: Path, jana_bin: Path, reverie_bin: Path) -> dict[str, Any]:
    return {
        "host": {
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "jana": {
            "source": git_metadata(jana_dir),
            "binary": binary_metadata(jana_bin),
        },
        "reverie": {
            "source": git_metadata(REPO_ROOT),
            "binary": binary_metadata(reverie_bin),
        },
    }


def write_json_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def is_finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0


def validation_errors(args: argparse.Namespace) -> list[str]:
    errors = []
    if args.runs < 1:
        errors.append("--runs must be at least 1")
    if args.warmup < 0:
        errors.append("--warmup must be non-negative")
    if args.min_speedup is not None and not is_finite_positive(args.min_speedup):
        errors.append("--min-speedup must be finite and greater than 0")
    if not is_finite_positive(args.command_timeout):
        errors.append("--command-timeout must be finite and greater than 0")
    duplicates = sorted(
        name for name in set(args.only) if args.only.count(name) > 1
    )
    if duplicates:
        errors.append(f"duplicate --only workload(s): {', '.join(duplicates)}")
    return errors


def selected_benchmarks(all_benchmarks: list[Benchmark], only: list[str]) -> list[Benchmark]:
    if not only:
        return all_benchmarks
    by_name = {bench.name: bench for bench in all_benchmarks}
    unknown = sorted(set(only) - set(by_name))
    if unknown:
        raise RuntimeError(f"unknown benchmark workload(s): {', '.join(unknown)}")
    return [by_name[name] for name in only]


def self_test_args(**overrides: Any) -> argparse.Namespace:
    values = {
        "runs": 20,
        "warmup": 3,
        "min_speedup": 1.25,
        "command_timeout": DEFAULT_COMMAND_TIMEOUT_SECONDS,
        "only": [],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def expect_validation_error(label: str, args: argparse.Namespace, expected: str) -> None:
    errors = validation_errors(args)
    if expected not in errors:
        raise AssertionError(f"{label}: expected {expected!r}, got {errors!r}")


def run_self_tests() -> int:
    valid = self_test_args()
    errors = validation_errors(valid)
    if errors:
        raise AssertionError(f"valid defaults rejected: {errors!r}")

    expect_validation_error("runs", self_test_args(runs=0), "--runs must be at least 1")
    expect_validation_error("warmup", self_test_args(warmup=-1), "--warmup must be non-negative")
    expect_validation_error(
        "minimum speedup",
        self_test_args(min_speedup=0),
        "--min-speedup must be finite and greater than 0",
    )
    expect_validation_error(
        "minimum speedup nan",
        self_test_args(min_speedup=float("nan")),
        "--min-speedup must be finite and greater than 0",
    )
    expect_validation_error(
        "minimum speedup inf",
        self_test_args(min_speedup=float("inf")),
        "--min-speedup must be finite and greater than 0",
    )
    expect_validation_error(
        "timeout",
        self_test_args(command_timeout=0),
        "--command-timeout must be finite and greater than 0",
    )
    expect_validation_error(
        "timeout nan",
        self_test_args(command_timeout=float("nan")),
        "--command-timeout must be finite and greater than 0",
    )
    expect_validation_error(
        "duplicate workload",
        self_test_args(only=["fib", "fib"]),
        "duplicate --only workload(s): fib",
    )
    direction_cases = {
        "fib": "forward",
        "fib_reverse": "reverse",
        "fib_roundtrip": "roundtrip",
    }
    for workload, expected_direction in direction_cases.items():
        observed_direction = workload_direction(workload)
        if observed_direction != expected_direction:
            raise AssertionError(
                "workload direction metadata mismatch: "
                f"{workload} expected {expected_direction}, got {observed_direction}"
            )

    fake = [
        Benchmark("fib", CommandSpec([], []), CommandSpec([], [])),
        Benchmark("sort", CommandSpec([], []), CommandSpec([], [])),
        Benchmark("turing", CommandSpec([], []), CommandSpec([], [])),
    ]
    if selected_benchmarks(fake, []) != fake:
        raise AssertionError("empty --only should select all benchmarks")
    selected = selected_benchmarks(fake, ["turing", "fib"])
    if [bench.name for bench in selected] != ["turing", "fib"]:
        raise AssertionError("--only benchmark order was not preserved")
    try:
        selected_benchmarks(fake, ["missing"])
    except RuntimeError as error:
        if str(error) != "unknown benchmark workload(s): missing":
            raise AssertionError(f"unexpected unknown-workload error: {error}") from error
    else:
        raise AssertionError("unknown --only workload was accepted")

    print("ok: Janus benchmark harness self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    errors = validation_errors(args)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    try:
        if args.list_workloads:
            jana_bin = default_jana_bin(args.jana_dir, args.jana_bin)
            for bench in benchmarks(jana_bin, args.reverie_bin, args.jana_dir):
                print(bench.name)
            return 0

        ensure_jana_checkout(args.jana_dir, args.no_clone)
        ensure_reverie_binary(args.reverie_bin, args.no_build)
        jana_bin = resolve_jana_bin(args.jana_dir, args.jana_bin)
        all_benchmarks = benchmarks(jana_bin, args.reverie_bin, args.jana_dir)
        selected = selected_benchmarks(all_benchmarks, args.only)
        if not selected:
            names = ", ".join(bench.name for bench in all_benchmarks)
            raise RuntimeError(f"no benchmarks selected. Available: {names}")

        print(f"Jana baseline: {jana_bin}")
        print(f"Reverie binary: {args.reverie_bin}")
        print(f"Runs: {args.runs}, warmup: {args.warmup}")
        print(f"Command timeout: {args.command_timeout:.3f}s")
        print()

        json_results: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "jana_baseline": str(jana_bin),
            "reverie_binary": str(args.reverie_bin),
            "working_directory": str(REPO_ROOT),
            "runs": args.runs,
            "warmup": args.warmup,
            "min_speedup": args.min_speedup,
            "command_timeout_seconds": args.command_timeout,
            "selected_workloads": [bench.name for bench in selected],
            "metadata": benchmark_metadata(args.jana_dir, jana_bin, args.reverie_bin),
            "benchmarks": [],
        }
        benchmark_results: list[dict[str, Any]] = json_results["benchmarks"]
        speedup_failures = []

        for bench in selected:
            print(f"## {bench.name}")
            print(f"jana:    {format_side(bench.jana)}")
            print(f"reverie: {format_side(bench.reverie)}")
            jana_samples = time_command(
                bench.jana,
                args.warmup,
                args.runs,
                args.command_timeout,
                args.measure_memory,
            )
            reverie_samples = time_command(
                bench.reverie,
                args.warmup,
                args.runs,
                args.command_timeout,
                args.measure_memory,
            )
            jana_median = statistics.median(elapsed_samples(jana_samples))
            reverie_median = statistics.median(elapsed_samples(reverie_samples))
            speedup = jana_median / reverie_median
            jana_rss_median = statistics.median(rss_samples(jana_samples))
            reverie_rss_median = statistics.median(rss_samples(reverie_samples))
            memory_ratio = (
                jana_rss_median / reverie_rss_median
                if reverie_rss_median > 0
                else None
            )
            print(f"jana    {summarize(jana_samples)}")
            print(f"reverie {summarize(reverie_samples)}")
            print(f"speedup {speedup:.2f}x")
            if memory_ratio is not None:
                print(f"rss     jana/reverie {memory_ratio:.2f}x")
            if args.min_speedup is not None and speedup < args.min_speedup:
                print(f"gate    FAIL: below required {args.min_speedup:.2f}x")
                speedup_failures.append((bench.name, speedup))
            elif args.min_speedup is not None:
                print(f"gate    PASS: meets required {args.min_speedup:.2f}x")
            print()

            result = {
                "name": bench.name,
                "direction": workload_direction(bench.name),
                "jana": {
                    **side_json(bench.jana),
                    **sample_summary(jana_samples),
                },
                "reverie": {
                    **side_json(bench.reverie),
                    **sample_summary(reverie_samples),
                },
                "speedup": speedup,
                "passes_min_speedup": (
                    None
                    if args.min_speedup is None
                    else speedup >= args.min_speedup
                ),
            }
            if memory_ratio is not None:
                result["memory_ratio"] = memory_ratio
            benchmark_results.append(result)

        if args.json_output is not None:
            write_json_results(args.json_output, json_results)
            print(f"Wrote JSON results to {args.json_output}")
        if speedup_failures:
            failure_list = ", ".join(
                f"{name}={speedup:.2f}x" for name, speedup in speedup_failures
            )
            raise RuntimeError(
                f"speedup gate failed for {len(speedup_failures)} workload(s): "
                f"{failure_list}"
            )
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        if "Exec format error" in str(error):
            print(
                "hint: the Jana repository's bin/janus is platform-specific; "
                "build Jana locally and pass --jana-bin.",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
