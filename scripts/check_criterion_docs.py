#!/usr/bin/env python3
"""Validate that internal Criterion benchmark groups are documented and registered."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = REPO_ROOT / "crates" / "reverie-interp" / "benches" / "execution.rs"
DOC_PATHS = [
    REPO_ROOT / "docs" / "performance.md",
    REPO_ROOT / "docs" / "roadmap.md",
]
BENCH_GROUP_RE = re.compile(r'\.benchmark_group\("([^"]+)"\)')
PERFORMANCE_GROUP_LIST_RE = re.compile(
    r"The suite currently contains these documented Criterion groups:\n\n"
    r"```text\n(?P<body>.*?)\n```",
    re.DOTALL,
)
GROUP_BLOCK_RE = re.compile(
    r'let mut group = c\.benchmark_group\("([^"]+)"\);(?P<body>.*?)group\.finish\(\);',
    re.DOTALL,
)
BENCH_FUNCTION_RE = re.compile(r'\.bench_function\("([^"]+)"')
BENCH_WRAPPER_RE = re.compile(r"^fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*c:\s*&mut\s+Criterion\s*\)")
CRITERION_GROUP_RE = re.compile(r"criterion_group!\((?P<body>.*?)\);", re.DOTALL)
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
SPECIAL_EXPECTED_FUNCTIONS = {
    "slot_compile_vs_execute_sort_n50": {"compile", "execute_compiled"},
    "scrub_timeline_sort_n50": {"build_timeline"},
    "tensor_matmul_builtin_vs_loops_3x3": {
        "explicit_tree_walk",
        "builtin_tree_walk",
        "explicit_slot_compiled",
        "builtin_slot_compiled",
    },
}
DEFAULT_EXPECTED_FUNCTIONS = {"tree_walk", "slot_compiled"}


def ordered_criterion_groups() -> list[str]:
    source = BENCH_PATH.read_text(encoding="utf-8")
    groups = BENCH_GROUP_RE.findall(source)
    ordered: list[str] = []
    seen: set[str] = set()
    for group in groups:
        if group not in seen:
            ordered.append(group)
            seen.add(group)
    return ordered


def criterion_groups() -> list[str]:
    return sorted(ordered_criterion_groups())


def documented_performance_group_list() -> Optional[list[str]]:
    text = (REPO_ROOT / "docs" / "performance.md").read_text(encoding="utf-8")
    match = PERFORMANCE_GROUP_LIST_RE.search(text)
    if match is None:
        return None
    return [line.strip() for line in match.group("body").splitlines() if line.strip()]


def criterion_group_functions() -> dict[str, set[str]]:
    source = BENCH_PATH.read_text(encoding="utf-8")
    groups: dict[str, set[str]] = {}
    for match in GROUP_BLOCK_RE.finditer(source):
        name = match.group(1)
        functions = set(BENCH_FUNCTION_RE.findall(match.group("body")))
        groups[name] = functions
    return groups


def benchmark_wrapper_groups() -> dict[str, set[str]]:
    source = BENCH_PATH.read_text(encoding="utf-8")
    wrappers: dict[str, set[str]] = {}
    current: Optional[str] = None
    current_groups: set[str] = set()
    brace_depth = 0
    seen_open_brace = False

    for line in source.splitlines():
        if current is None:
            match = BENCH_WRAPPER_RE.match(line)
            if match is None:
                continue
            current = match.group(1)
            current_groups = set()
            brace_depth = line.count("{") - line.count("}")
            seen_open_brace = "{" in line
            continue

        current_groups.update(BENCH_GROUP_RE.findall(line))
        brace_depth += line.count("{") - line.count("}")
        seen_open_brace = seen_open_brace or "{" in line
        if seen_open_brace and brace_depth <= 0:
            wrappers[current] = current_groups
            current = None

    if current is not None:
        wrappers[current] = current_groups
    return wrappers


def registered_benchmark_wrappers() -> set[str]:
    source = BENCH_PATH.read_text(encoding="utf-8")
    match = CRITERION_GROUP_RE.search(source)
    if match is None:
        return set()
    identifiers = IDENT_RE.findall(match.group("body"))
    return set(identifiers[1:])


def validate_registration(wrappers: dict[str, set[str]], registered: set[str]) -> list[str]:
    errors = []
    if not registered:
        errors.append("Criterion benchmark file is missing criterion_group! registration")
        return errors

    expected = set(wrappers)
    missing = sorted(expected - registered)
    stale = sorted(registered - expected)
    empty = sorted(name for name, groups in wrappers.items() if not groups)
    if missing:
        errors.append(
            "Criterion benchmark wrapper(s) missing from criterion_group! registration: "
            + ", ".join(missing)
        )
    if stale:
        errors.append(
            "criterion_group! registers unknown benchmark wrapper(s): "
            + ", ".join(stale)
        )
    if empty:
        errors.append(
            "Criterion benchmark wrapper(s) define no benchmark_group: "
            + ", ".join(empty)
        )
    return errors


def validate_group_shapes(groups: dict[str, set[str]]) -> list[str]:
    errors = []
    for name in criterion_groups():
        functions = groups.get(name)
        if functions is None:
            errors.append(f"Criterion group `{name}` is missing a closed group block")
            continue
        expected = SPECIAL_EXPECTED_FUNCTIONS.get(name, DEFAULT_EXPECTED_FUNCTIONS)
        missing = sorted(expected - functions)
        extra = sorted(functions - expected)
        if missing:
            errors.append(
                f"Criterion group `{name}` missing bench function(s): {', '.join(missing)}"
            )
        if extra:
            errors.append(
                f"Criterion group `{name}` has unexpected bench function(s): {', '.join(extra)}"
            )
    return errors


def validate_performance_group_list(expected_groups: list[str]) -> list[str]:
    documented = documented_performance_group_list()
    if documented is None:
        return ["docs/performance.md is missing the documented Criterion group list"]
    errors = []
    if len(set(documented)) != len(documented):
        duplicates = sorted(
            group for group in set(documented) if documented.count(group) > 1
        )
        errors.append(
            "docs/performance.md Criterion group list has duplicate group(s): "
            + ", ".join(duplicates)
        )
    if documented != expected_groups:
        errors.append(
            "docs/performance.md Criterion group list must match benchmark source order: "
            f"expected {', '.join(expected_groups)}; found {', '.join(documented)}"
        )
    return errors


def main() -> int:
    text = "\n".join(path.read_text(encoding="utf-8") for path in DOC_PATHS)
    ordered_groups = ordered_criterion_groups()
    missing = [name for name in ordered_groups if name not in text]
    if missing:
        for name in missing:
            print(f"error: missing documented Criterion group `{name}`", file=sys.stderr)
        return 1
    list_errors = validate_performance_group_list(ordered_groups)
    if list_errors:
        for error in list_errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    shape_errors = validate_group_shapes(criterion_group_functions())
    if shape_errors:
        for error in shape_errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    registration_errors = validate_registration(
        benchmark_wrapper_groups(), registered_benchmark_wrappers()
    )
    if registration_errors:
        for error in registration_errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"ok: documented, shaped, and registered {len(criterion_groups())} Criterion group(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
