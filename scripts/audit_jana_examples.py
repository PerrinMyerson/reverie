#!/usr/bin/env python3
"""Audit upstream Jana examples against Jana and Reverie.

This is not a benchmark. It is a compatibility inventory for deciding which
upstream examples can become checked benchmark or example-corpus entries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JANA_DIR = REPO_ROOT / "target" / "jana-baseline"
DEFAULT_REVERIE_BIN = REPO_ROOT / "target" / "release" / "reverie"
DEFAULT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class CommandResult:
    status: str
    exit_code: Optional[int]
    stdout: str
    stderr: str


def has_jana_diagnostic(text: str) -> bool:
    return text.lstrip().startswith("[Expected value to be `")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit upstream Jana examples against Jana and Reverie."
    )
    parser.add_argument("--jana-dir", type=Path, default=DEFAULT_JANA_DIR)
    parser.add_argument(
        "--jana-bin",
        type=Path,
        help="Path to Jana executable. Defaults to <jana-dir>/bin/janus.",
    )
    parser.add_argument("--reverie-bin", type=Path, default=DEFAULT_REVERIE_BIN)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def run_command(argv: list[str], timeout: float) -> CommandResult:
    try:
        completed = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        return CommandResult(
            status="timeout",
            exit_code=None,
            stdout=error.stdout or "",
            stderr=error.stderr or "",
        )
    except OSError as error:
        return CommandResult("error", None, "", str(error))

    status = "ok" if completed.returncode == 0 else "failed"
    if status == "ok" and (
        has_jana_diagnostic(completed.stdout) or has_jana_diagnostic(completed.stderr)
    ):
        status = "ok_with_diagnostics"

    return CommandResult(status, completed.returncode, completed.stdout, completed.stderr)


def discover_examples(jana_dir: Path) -> list[Path]:
    roots = [jana_dir / "examples", jana_dir / "basicExamples"]
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        paths.extend(root.glob("*.ja"))
        paths.extend(root.glob("*.janus"))
    return sorted(paths)


def summarize(result: CommandResult) -> dict[str, Any]:
    first_line = (result.stderr or result.stdout).strip().splitlines()
    return {
        "status": result.status,
        "exit_code": result.exit_code,
        "message": first_line[0] if first_line else "",
        "stdout_lines": len(result.stdout.splitlines()),
        "stderr_lines": len(result.stderr.splitlines()),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def count_statuses(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row[key]["status"]
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def main() -> int:
    args = parse_args()
    jana_bin = args.jana_bin or args.jana_dir / "bin" / "janus"
    if not jana_bin.exists():
        raise SystemExit(f"missing Jana executable: {jana_bin}")
    if not args.reverie_bin.exists():
        raise SystemExit(f"missing Reverie executable: {args.reverie_bin}")

    rows = []
    for path in discover_examples(args.jana_dir):
        rel = path.relative_to(args.jana_dir)
        jana = run_command([str(jana_bin), "-m64", str(path)], args.timeout)
        reverie = run_command([str(args.reverie_bin), "run", str(path)], args.timeout)
        legacy_reverie = run_command(
            [str(args.reverie_bin), "run", str(path), "--legacy-janus"],
            args.timeout,
        )
        rows.append(
            {
                "path": str(rel),
                "jana": summarize(jana),
                "reverie": summarize(reverie),
                "reverie_legacy_janus": summarize(legacy_reverie),
            }
        )

    totals = {
        "jana": count_statuses(rows, "jana"),
        "reverie": count_statuses(rows, "reverie"),
        "reverie_legacy_janus": count_statuses(rows, "reverie_legacy_janus"),
    }

    print("| example | Jana | Reverie | Reverie --legacy-janus |")
    print("| --- | --- | --- | --- |")
    for row in rows:
        print(
            "| `{path}` | {jana} | {reverie} | {legacy} |".format(
                path=row["path"],
                jana=row["jana"]["status"],
                reverie=row["reverie"]["status"],
                legacy=row["reverie_legacy_janus"]["status"],
            )
        )
    print()
    print("Status totals:")
    for label, key in [
        ("Jana", "jana"),
        ("Reverie", "reverie"),
        ("Reverie --legacy-janus", "reverie_legacy_janus"),
    ]:
        summary = ", ".join(
            f"{status}={count}" for status, count in totals[key].items()
        )
        print(f"- {label}: {summary}")

    if args.json_output is not None:
        write_json(
            args.json_output,
            {
                "jana_dir": str(args.jana_dir),
                "jana_bin": str(jana_bin),
                "reverie_bin": str(args.reverie_bin),
                "timeout_seconds": args.timeout,
                "totals": totals,
                "examples": rows,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
