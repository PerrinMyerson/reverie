#!/usr/bin/env python3
"""Validate that the documented Reverie example corpus stays synchronized."""

from __future__ import annotations

import re
import sys
import argparse
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
EVALUATION_DOC = REPO_ROOT / "docs" / "evaluation.md"
EXAMPLES_GUIDE = EXAMPLES_DIR / "README.md"
CLI_TESTS = REPO_ROOT / "crates" / "reverie-cli" / "tests" / "cli.rs"
EXAMPLE_RE = re.compile(r"^examples/[A-Za-z0-9_.-]+\.rev$")
REQUIRED_EXAMPLE_SYNTAX_FEATURES = {
    "assertion helpers": ("assert_eq(", "assert_ne("),
    "array length alias": ("len(",),
    "stack readability aliases": ("is_empty(", "peek("),
    "swap call alias": ("swap(",),
    "increment/decrement sugar": ("++", "--"),
}
REQUIRED_DOC_SYNTAX_FEATURES = {
    "docs/language.md": {
        "assertion helpers": ("assert_eq(", "assert_ne("),
        "array length alias": ("len(xs)",),
        "stack readability aliases": ("is_empty(s)", "peek(s)"),
        "swap call alias": ("swap(",),
        "increment/decrement sugar": ("++", "--"),
    },
    "docs/grammar.md": {
        "assertion helpers": ('"assert_eq" | "assert_ne"',),
        "array length alias": ('"size" | "len"',),
        "stack readability aliases": (
            '"empty" | "is_empty"',
            '"top" | "peek"',
        ),
        "swap call alias": ('"swap"',),
        "increment/decrement sugar": ('"++" | "--"',),
    },
    "examples/README.md": {
        "assertion helpers": ("assert_eq", "assert_ne"),
        "array length alias": ("len(xs)",),
        "stack readability aliases": ("is_empty(s)", "peek(s)"),
        "swap call alias": ("swap(",),
        "increment/decrement sugar": ("++", "--"),
    },
}


def example_paths() -> set[str]:
    return {
        f"examples/{path.name}"
        for path in EXAMPLES_DIR.iterdir()
        if path.is_file() and path.suffix == ".rev"
    }


def examples_in_section(text: str, heading: str) -> set[str]:
    return set(example_list_in_section(text, heading))


def example_list_in_section(text: str, heading: str) -> list[str]:
    marker = f"## {heading}"
    start = text.find(marker)
    if start == -1:
        raise ValueError(f"missing section `{marker}`")
    next_heading = text.find("\n## ", start + len(marker))
    section = text[start:] if next_heading == -1 else text[start:next_heading]
    return [
        line.strip("`")
        for line in (line.strip() for line in section.splitlines())
        if EXAMPLE_RE.match(line.strip("`"))
    ]


def duplicate_entries(entries: list[str]) -> list[str]:
    counts = Counter(entries)
    return sorted(entry for entry, count in counts.items() if count > 1)


def validate_classification(
    examples: set[str],
    negative: list[str],
    positive: list[str],
) -> list[str]:
    negative_set = set(negative)
    positive_set = set(positive)
    classified = negative_set | positive_set
    overlap = negative_set & positive_set
    missing = examples - classified
    stale = classified - examples

    errors = []
    duplicate_negative = duplicate_entries(negative)
    duplicate_positive = duplicate_entries(positive)
    if duplicate_negative:
        errors.append(
            "duplicate negative corpus example(s): " + ", ".join(duplicate_negative)
        )
    if duplicate_positive:
        errors.append(
            "duplicate expressiveness corpus example(s): "
            + ", ".join(duplicate_positive)
        )
    if missing:
        errors.append("missing evaluation corpus example(s): " + ", ".join(sorted(missing)))
    if stale:
        errors.append("stale evaluation corpus example(s): " + ", ".join(sorted(stale)))
    if overlap:
        errors.append(
            "examples listed as both positive and negative: "
            + ", ".join(sorted(overlap))
        )
    return errors


def validate_required_syntax_features(example_text: str) -> list[str]:
    return validate_required_snippets(
        "ergonomic syntax coverage",
        example_text,
        REQUIRED_EXAMPLE_SYNTAX_FEATURES,
    )


def validate_required_snippets(
    label: str, text: str, features: dict[str, tuple[str, ...]]
) -> list[str]:
    missing = [
        feature
        for feature, snippets in features.items()
        if not all(snippet in text for snippet in snippets)
    ]
    if not missing:
        return []
    return ["missing required " + label + ": " + ", ".join(sorted(missing))]


def validate_required_syntax_docs() -> list[str]:
    errors = []
    for relative_path, features in REQUIRED_DOC_SYNTAX_FEATURES.items():
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        errors.extend(validate_required_snippets(relative_path, text, features))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that the documented Reverie example corpus stays synchronized."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent evaluation-corpus checker self-tests and exit.",
    )
    return parser.parse_args()


def expect_self_test_error(
    label: str,
    examples: set[str],
    negative: list[str],
    positive: list[str],
    needle: str,
) -> None:
    errors = validate_classification(examples, negative, positive)
    if not any(needle in error for error in errors):
        raise AssertionError(f"{label} did not report `{needle}`; errors were {errors}")


def run_self_tests() -> int:
    examples = {
        "examples/good.rev",
        "examples/bad.rev",
        "examples/io.rev",
    }
    valid_errors = validate_classification(
        examples,
        ["examples/bad.rev"],
        ["examples/good.rev", "examples/io.rev"],
    )
    if valid_errors:
        for error in valid_errors:
            print(f"error: self-test valid fixture failed: {error}", file=sys.stderr)
        return 1

    cases = [
        (
            "duplicate negative",
            ["examples/bad.rev", "examples/bad.rev"],
            ["examples/good.rev", "examples/io.rev"],
            "duplicate negative corpus example(s)",
        ),
        (
            "duplicate positive",
            ["examples/bad.rev"],
            ["examples/good.rev", "examples/good.rev", "examples/io.rev"],
            "duplicate expressiveness corpus example(s)",
        ),
        (
            "missing classification",
            ["examples/bad.rev"],
            ["examples/good.rev"],
            "missing evaluation corpus example(s)",
        ),
        (
            "stale classification",
            ["examples/bad.rev"],
            ["examples/good.rev", "examples/io.rev", "examples/stale.rev"],
            "stale evaluation corpus example(s)",
        ),
        (
            "overlap classification",
            ["examples/bad.rev"],
            ["examples/good.rev", "examples/bad.rev", "examples/io.rev"],
            "examples listed as both positive and negative",
        ),
    ]
    try:
        for label, negative, positive, needle in cases:
            expect_self_test_error(label, examples, negative, positive, needle)
        syntax_errors = validate_required_syntax_features(
            "assert_eq(x, 0)\nassert_ne(x, 1)\nlen(xs)\nis_empty(s)\npeek(s)\nswap(x, y)\n++x\n--x\n"
        )
        if syntax_errors:
            raise AssertionError(
                f"valid ergonomic syntax fixture failed: {syntax_errors}"
            )
        syntax_errors = validate_required_syntax_features("assert_eq(x, 0)\n")
        if not any(
            "missing required ergonomic syntax coverage" in error
            for error in syntax_errors
        ):
            raise AssertionError(
                "missing ergonomic syntax fixture did not report required coverage"
            )
        doc_errors = validate_required_snippets(
            "docs/example.md",
            "assert_eq assert_ne len(xs) is_empty(s) peek(s) swap( ++ --",
            {
                "assertion helpers": ("assert_eq", "assert_ne"),
                "array length alias": ("len(xs)",),
                "stack readability aliases": ("is_empty(s)", "peek(s)"),
                "swap call alias": ("swap(",),
                "increment/decrement sugar": ("++", "--"),
            },
        )
        if doc_errors:
            raise AssertionError(f"valid docs fixture failed: {doc_errors}")
        doc_errors = validate_required_snippets(
            "docs/example.md",
            "assert_eq assert_ne len(xs)",
            {
                "stack readability aliases": ("is_empty(s)", "peek(s)"),
            },
        )
        if not any("missing required docs/example.md" in error for error in doc_errors):
            raise AssertionError(
                "missing docs fixture did not report required documentation"
            )
    except AssertionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("ok: evaluation corpus checker self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    text = EVALUATION_DOC.read_text(encoding="utf-8")
    examples_guide = EXAMPLES_GUIDE.read_text(encoding="utf-8")
    cli_tests = CLI_TESTS.read_text(encoding="utf-8")
    try:
        negative = example_list_in_section(text, "Negative Corpus")
        positive = example_list_in_section(text, "Expressiveness Corpus")
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    examples = example_paths()
    errors = validate_classification(examples, negative, positive)

    missing_from_guide = [
        example for example in sorted(examples) if example not in examples_guide
    ]
    missing_from_cli_tests = [
        example for example in sorted(examples) if example not in cli_tests
    ]
    example_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(EXAMPLES_DIR.iterdir())
        if path.is_file() and path.suffix == ".rev"
    )
    errors.extend(validate_required_syntax_features(example_text))
    errors.extend(validate_required_syntax_docs())
    if missing_from_guide:
        errors.append(
            "example(s) missing from examples/README.md: "
            + ", ".join(missing_from_guide)
        )
    if missing_from_cli_tests:
        errors.append(
            "example(s) missing from CLI tests: " + ", ".join(missing_from_cli_tests)
        )

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"ok: synchronized {len(examples)} evaluation example(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
