#!/usr/bin/env python3
"""Validate the documented Janus feature-audit matrix."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC = REPO_ROOT / "docs" / "janus-audit.md"
FORBIDDEN_STATUS_RE = re.compile(r"\b(todo|tbd|unknown|unclear|not\s+checked)\b", re.I)
SUPPORTED_STATUS_RE = re.compile(
    r"^(Supported|Checked|Runtime checked|Statically and dynamically checked|"
    r"Replaced|Uses)\b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that docs/janus-audit.md keeps concrete feature evidence."
    )
    parser.add_argument("doc", nargs="?", type=Path, default=DEFAULT_DOC)
    parser.add_argument(
        "--min-features",
        type=int,
        default=1,
        help="Require at least this many audit matrix rows.",
    )
    parser.add_argument(
        "--expect-feature",
        action="append",
        default=[],
        help="Require a Janus feature row whose name matches this text exactly.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent feature-audit checker self-tests and exit.",
    )
    return parser.parse_args()


def table_rows(text: str) -> list[tuple[str, str, str]]:
    rows = []
    in_matrix = False
    for line in text.splitlines():
        if line == "## Audit Matrix":
            in_matrix = True
            continue
        if in_matrix and line.startswith("## "):
            break
        if not in_matrix or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells == ["---", "---", "---"] or cells[0] == "Janus86 feature":
            continue
        if len(cells) != 3:
            raise ValueError(f"malformed audit matrix row: {line}")
        rows.append((cells[0], cells[1], cells[2]))
    return rows


def validate_rows(
    rows: list[tuple[str, str, str]],
    min_features: int,
    expected_features: list[str],
) -> list[str]:
    errors = []
    if len(rows) < min_features:
        errors.append(
            f"Janus feature audit has {len(rows)} row(s), expected at least {min_features}"
        )

    feature_counts = Counter(feature for feature, _, _ in rows)
    duplicate_features = sorted(
        feature for feature, count in feature_counts.items() if count > 1
    )
    if duplicate_features:
        errors.append(
            "duplicate Janus feature row(s): " + ", ".join(duplicate_features)
        )

    features = set(feature_counts)
    for feature in expected_features:
        if feature not in features:
            errors.append(f"missing expected Janus feature row `{feature}`")

    for feature, status, evidence in rows:
        if not feature:
            errors.append("empty Janus feature name")
        if not status:
            errors.append(f"`{feature}` has an empty status")
        if not evidence:
            errors.append(f"`{feature}` has empty evidence")
        if FORBIDDEN_STATUS_RE.search(status) or FORBIDDEN_STATUS_RE.search(evidence):
            errors.append(f"`{feature}` has vague audit language: {status} | {evidence}")
        if not SUPPORTED_STATUS_RE.search(status):
            errors.append(f"`{feature}` has an unsupported status shape: {status}")
    return errors


def run_self_tests() -> int:
    valid_rows = [
        ("Forward procedure calls", "Supported", "`call`, parser tests"),
        ("Interactive Janus runtime commands", "Replaced by CLI subcommands", "`run`, `reverse`"),
    ]
    errors = validate_rows(valid_rows, 2, ["Forward procedure calls"])
    if errors:
        for error in errors:
            print(f"error: self-test valid fixture failed: {error}", file=sys.stderr)
        return 1

    cases = [
        (
            "duplicate feature",
            valid_rows + [valid_rows[0]],
            "duplicate Janus feature row(s)",
        ),
        (
            "missing feature",
            valid_rows,
            "missing expected Janus feature row",
            ["Case-insensitive identifiers"],
        ),
        (
            "vague language",
            [("Unary negation", "Supported", "TODO verify")],
            "vague audit language",
        ),
        (
            "bad status shape",
            [("Unary negation", "Partial", "`examples/negation.rev`")],
            "unsupported status shape",
        ),
    ]
    for case in cases:
        label, rows, needle, *feature_args = case
        expected_features = feature_args[0] if feature_args else []
        errors = validate_rows(rows, 1, expected_features)
        if not any(needle in error for error in errors):
            print(
                f"error: {label} did not report `{needle}`; errors were {errors}",
                file=sys.stderr,
            )
            return 1
    print("ok: Janus feature audit checker self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    doc = args.doc if args.doc.is_absolute() else REPO_ROOT / args.doc
    try:
        rows = table_rows(doc.read_text(encoding="utf-8"))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    errors = validate_rows(rows, args.min_features, args.expect_feature)
    status_counts: Counter[str] = Counter()
    for feature, status, evidence in rows:
        status_counts[status.split()[0]] += 1

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"ok: validated {len(rows)} Janus feature audit row(s) "
        f"({', '.join(f'{key}:{value}' for key, value in sorted(status_counts.items()))})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
