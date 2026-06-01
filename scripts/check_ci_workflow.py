#!/usr/bin/env python3
"""Validate that CI keeps the core Reverie quality gates wired."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
BENCHMARK_CHECKER = REPO_ROOT / "scripts" / "check_benchmark_artifact.py"
SUMMARY_SCRIPT = REPO_ROOT / "scripts" / "summarize_janus_performance.py"
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_janus_performance.py"
MATERIALIZED_SOURCE_CHECKER = REPO_ROOT / "scripts" / "check_materialized_sources.py"
CRITERION_CHECKER = REPO_ROOT / "scripts" / "check_criterion_docs.py"
EXPLAIN_CHECKER = REPO_ROOT / "scripts" / "check_explain_json.py"

REQUIRED_SNIPPETS = {
    "format gate": "cargo fmt --check",
    "source materialization gate": "python3 scripts/check_materialized_sources.py",
    "workspace clippy gate": "cargo clippy --workspace --all-targets -- -D warnings",
    "workspace test gate": "cargo test --workspace",
    "criterion bench compile gate": "cargo bench -p reverie-interp --bench execution --no-run",
    "criterion checker py_compile": "scripts/check_criterion_docs.py",
    "benchmark harness workload listing": "python3 scripts/bench_jana_vs_reverie.py --list-workloads",
    "benchmark harness self-test": "python3 scripts/bench_jana_vs_reverie.py --self-test",
    "benchmark artifact checker self-test": "python3 scripts/check_benchmark_artifact.py --self-test",
    "benchmark summary renderer self-test": "python3 scripts/summarize_janus_performance.py --self-test",
    "Janus performance wrapper self-test": "python3 scripts/verify_janus_performance.py --self-test",
    "materialized source checker self-test": "python3 scripts/check_materialized_sources.py --self-test",
    "benchmark docs checker self-test": "python3 scripts/check_benchmark_docs.py --self-test",
    "evaluation corpus checker self-test": "python3 scripts/check_evaluation_corpus.py --self-test",
    "Jana audit checker self-test": "python3 scripts/check_jana_audit.py --self-test",
    "Janus feature audit checker self-test": "python3 scripts/check_janus_feature_audit.py --self-test",
    "benchmark artifact checker": "python3 scripts/check_benchmark_artifact.py",
    "geometric mean performance floor": "--min-geomean-speedup 4.0",
    "checked markdown benchmark summary": "--expect-markdown-summary benchmarks/results/jana-vs-reverie-smoke.md",
    "benchmark docs checker": "python3 scripts/check_benchmark_docs.py",
    "criterion docs checker": "python3 scripts/check_criterion_docs.py",
    "evaluation corpus checker": "python3 scripts/check_evaluation_corpus.py",
    "explain JSON checker": "python3 scripts/check_explain_json.py",
    "Janus feature audit checker": "python3 scripts/check_janus_feature_audit.py",
    "Jana audit checker": "python3 scripts/check_jana_audit.py",
    "manual Janus performance job": "janus-performance:",
    "checked Janus performance wrapper": "scripts/verify_janus_performance.py",
    "benchmark artifact upload": "actions/upload-artifact@v4",
}

BENCHMARK_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "bool is not numeric": "not isinstance(value, bool)",
    "Python 3.9 numeric isinstance": "isinstance(value, (int, float))",
    "synthetic artifact fixture": "valid_synthetic_artifact",
    "selected workload rejection": "selected_workloads must match benchmark row order",
    "command binary rejection": "does not match reverie binary path",
    "Markdown summary consistency": "stale Markdown direction summary",
}

SUMMARY_SCRIPT_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "finite numeric guard": "math.isfinite(value)",
    "Python 3.9 numeric isinstance": "isinstance(value, (int, float))",
    "direction count rendering": "Directions: ",
    "direction fallback rendering": "direction fallback",
    "workload failure rendering": "Failing workloads:",
    "observed speedup failure rendering": "Minimum observed speedup failure",
    "synthetic artifact fixture": "synthetic_summary_artifact",
}

VERIFY_SCRIPT_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "runs validation": "--runs must be positive",
    "warmup validation": "--warmup must be non-negative",
    "finite threshold validation": "math.isfinite(args.min_speedup)",
    "timeout validation": "--command-timeout must be finite and positive",
    "duplicate workload rejection": "duplicate --only workload(s)",
    "unknown workload rejection": "unknown benchmark workload(s)",
}

BENCHMARK_HARNESS_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "Python 3.9 type alias compatibility": "SideSpec = Union[CommandSpec, list[CommandSpec]]",
    "Python 3.9 optional typing compatibility": "Optional[float]",
    "finite threshold validation": "math.isfinite(value)",
    "duplicate workload rejection": "duplicate --only workload(s)",
    "unknown workload rejection": "unknown benchmark workload(s)",
    "only workload order preservation": "--only benchmark order was not preserved",
    "workload direction metadata validation": "workload direction metadata mismatch",
}

MATERIALIZED_SOURCE_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "quarantine audit option": "--include-quarantine",
    "dataless placeholder detection": "APFS dataless placeholders",
    "fast inspection timeout": "timeout=5.0",
    "quarantine exclusion": "dataless-quarantine",
    "quarantine selection self-test": "quarantined source file not selected when requested",
}

BENCHMARK_DOCS_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "manifest order check": "manifest workload order does not match harness order",
    "manifest missing workload rejection": "manifest missing workload(s)",
    "gate and direction docs detail": "benchmark docs missing required gate/direction detail",
}

EVALUATION_CORPUS_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "duplicate negative rejection": "duplicate negative corpus example(s)",
    "duplicate expressiveness rejection": "duplicate expressiveness corpus example(s)",
    "overlap rejection": "examples listed as both positive and negative",
    "ergonomic syntax coverage": "missing required ergonomic syntax coverage",
    "ergonomic docs coverage": "missing required docs/",
}

JANA_AUDIT_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "bool is not numeric": "not isinstance(value, bool)",
    "Python 3.9 numeric isinstance": "isinstance(value, (int, float))",
    "line count validation": "stdout_lines must be a non-negative integer",
    "allowed status validation": "status must be one of",
    "totals status validation": "totals.{target}.{status} must be a non-negative integer",
    "unexpected field rejection": "has unexpected field(s)",
}

JANUS_FEATURE_AUDIT_CHECKER_SNIPPETS = {
    "build-independent self-test": "--self-test",
    "duplicate feature rejection": "duplicate Janus feature row(s)",
    "vague language rejection": "vague audit language",
}

CRITERION_CHECKER_SNIPPETS = {
    "default tree/slot benchmark shape": 'DEFAULT_EXPECTED_FUNCTIONS = {"tree_walk", "slot_compiled"}',
    "compile benchmark shape exception": '"slot_compile_vs_execute_sort_n50": {"compile", "execute_compiled"}',
    "timeline benchmark shape exception": '"scrub_timeline_sort_n50": {"build_timeline"}',
    "unexpected Criterion bench function rejection": "has unexpected bench function(s)",
    "Criterion registration check": "missing from criterion_group! registration",
    "Criterion docs ordered group list check": "must match benchmark source order",
}

EXPLAIN_CHECKER_SNIPPETS = {
    "safety checks schema key": '"safety_checks"',
    "safety check counts schema key": '"safety_check_counts"',
    "bool is not int": "expected is int and isinstance(value, bool)",
    "indexed safety-check fixture": "examples\" / \"array.rev\"",
    "same-root update safety wording": "same-root update aliases rejected before runtime",
    "prebuilt reverie binary option": "--reverie-bin",
    "environment reverie binary option": "REVERIE_BIN",
    "build-independent schema self-test": "--self-test",
    "safety count consistency rejection": "safety_check_counts missing key(s)",
    "safety count unexpected key rejection": "safety_check_counts has unexpected key(s)",
    "safety count type rejection": "must be int",
}


def main() -> int:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    benchmark_checker = BENCHMARK_CHECKER.read_text(encoding="utf-8")
    summary_script = SUMMARY_SCRIPT.read_text(encoding="utf-8")
    verify_script = VERIFY_SCRIPT.read_text(encoding="utf-8")
    benchmark_harness = (REPO_ROOT / "scripts" / "bench_jana_vs_reverie.py").read_text(
        encoding="utf-8"
    )
    materialized_source_checker = MATERIALIZED_SOURCE_CHECKER.read_text(encoding="utf-8")
    benchmark_docs_checker = (
        REPO_ROOT / "scripts" / "check_benchmark_docs.py"
    ).read_text(encoding="utf-8")
    criterion_checker = CRITERION_CHECKER.read_text(encoding="utf-8")
    evaluation_corpus_checker = (
        REPO_ROOT / "scripts" / "check_evaluation_corpus.py"
    ).read_text(encoding="utf-8")
    explain_checker = EXPLAIN_CHECKER.read_text(encoding="utf-8")
    jana_audit_checker = (REPO_ROOT / "scripts" / "check_jana_audit.py").read_text(
        encoding="utf-8"
    )
    janus_feature_audit_checker = (
        REPO_ROOT / "scripts" / "check_janus_feature_audit.py"
    ).read_text(encoding="utf-8")
    missing = [
        label for label, snippet in REQUIRED_SNIPPETS.items() if snippet not in text
    ]
    missing.extend(
        f"Benchmark checker contract: {label}"
        for label, snippet in BENCHMARK_CHECKER_SNIPPETS.items()
        if snippet not in benchmark_checker
    )
    missing.extend(
        f"Benchmark summary renderer contract: {label}"
        for label, snippet in SUMMARY_SCRIPT_SNIPPETS.items()
        if snippet not in summary_script
    )
    missing.extend(
        f"Janus performance wrapper contract: {label}"
        for label, snippet in VERIFY_SCRIPT_SNIPPETS.items()
        if snippet not in verify_script
    )
    missing.extend(
        f"Benchmark harness contract: {label}"
        for label, snippet in BENCHMARK_HARNESS_SNIPPETS.items()
        if snippet not in benchmark_harness
    )
    missing.extend(
        f"Materialized source checker contract: {label}"
        for label, snippet in MATERIALIZED_SOURCE_CHECKER_SNIPPETS.items()
        if snippet not in materialized_source_checker
    )
    missing.extend(
        f"Benchmark docs checker contract: {label}"
        for label, snippet in BENCHMARK_DOCS_CHECKER_SNIPPETS.items()
        if snippet not in benchmark_docs_checker
    )
    missing.extend(
        f"Criterion checker contract: {label}"
        for label, snippet in CRITERION_CHECKER_SNIPPETS.items()
        if snippet not in criterion_checker
    )
    missing.extend(
        f"Evaluation corpus checker contract: {label}"
        for label, snippet in EVALUATION_CORPUS_CHECKER_SNIPPETS.items()
        if snippet not in evaluation_corpus_checker
    )
    missing.extend(
        f"Jana audit checker contract: {label}"
        for label, snippet in JANA_AUDIT_CHECKER_SNIPPETS.items()
        if snippet not in jana_audit_checker
    )
    missing.extend(
        f"Janus feature audit checker contract: {label}"
        for label, snippet in JANUS_FEATURE_AUDIT_CHECKER_SNIPPETS.items()
        if snippet not in janus_feature_audit_checker
    )
    missing.extend(
        f"Explain checker contract: {label}"
        for label, snippet in EXPLAIN_CHECKER_SNIPPETS.items()
        if snippet not in explain_checker
    )
    if missing:
        for label in missing:
            print(f"error: CI workflow missing {label}", file=sys.stderr)
        return 1
    print(f"ok: validated {len(REQUIRED_SNIPPETS)} CI workflow gate(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
