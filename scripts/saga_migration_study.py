#!/usr/bin/env python3
"""Generated undo for business software: saga compensations and data
down-migrations, measured against the usual alternatives.

Sagas (`reverie.saga`): random order-checkout sagas (reserve stock, charge,
award capped loyalty points, book shipping) run with concurrent traffic on
the same records; when a step fails, completed steps are compensated three
ways and compared with the right answer, "keep what other transactions
committed, remove the saga's effect":

* Reverie: each step's program run backward;
* snapshot restore: put back each record's value from before the step;
* hand-written: the obvious compensating code.

Migrations (`reverie.migrate`): a v1 -> v2 table migration (currency
conversion with rounding, status codes collapsed, flags packed, a column
dropped) whose down-migration is the same program run backward, compared with
a schema-only down-migration and a hand-written data down-migration; then a
rollback after the application has written new data in the v2 schema.

    python3 scripts/saga_migration_study.py --markdown-output docs/research/sagas-and-migrations.md
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sagas", type=int, default=5000)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--edited", type=float, default=0.1, help="fraction edited after up")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def saga_study(args) -> Dict[str, Any]:
    import reverie
    from reverie import saga

    programs = saga._programs()
    try:
        reverie.loads(saga.LOSSY_LOYALTY_SOURCE)
        rejection = None
    except reverie.CheckError as error:
        rejection = error.message
    out: Dict[str, Any] = {"lossy_step_rejected_with": rejection, "styles": {}}
    for style in ("loyalty", "loyalty_delta"):
        rng = random.Random(args.seed)
        tally = {k: collections.Counter() for k in ("reverie", "snapshot", "hand", "dirty_read")}
        for _ in range(args.sagas):
            for key, value in saga.run_trial(rng, programs, loyalty=style).items():
                tally[key][value] += 1
        out["styles"][style] = {k: dict(v) for k, v in tally.items()}
    return out


def migration_study(args) -> Dict[str, Any]:
    import reverie
    from reverie import migrate as mg
    from reverie.ledger import code_length_bits, flatten

    rng = random.Random(args.seed)
    rows = args.rows
    v1 = mg.random_rows(rows, rng)
    program = reverie.loads(mg.migration_source(rows))
    try:
        reverie.loads(mg.LOSSY_DROP_SOURCE)
        rejection = None
    except reverie.CheckError as error:
        rejection = error.message

    up = program.run(v1)
    down = program.reverse(up)
    exact = sum(all(down[c][r] == v1[c][r] for c in mg.V1) for r in range(rows))

    schema_only = mg.schema_only_down(rows)
    hand = mg.hand_written_down({c: up[c] for c in mg.V2})
    lost_cells = sum(schema_only[c][r] != v1[c][r] for c in mg.V1 for r in range(rows))
    hand_wrong_rows = sum(any(hand[c][r] != v1[c][r] for c in mg.V1) for r in range(rows))
    kept = [c for c in mg.V1 if c != "legacy_score"]
    hand_wrong_rows_kept_columns = sum(
        any(hand[c][r] != v1[c][r] for c in kept) for r in range(rows)
    )

    witness_bits = sum(code_length_bits(v) for v in flatten({c: up[c] for c in mg.WITNESS}))
    witness_breakdown = {
        c: sum(code_length_bits(v) for v in up[c]) for c in mg.WITNESS
    }
    backup_bits = sum(code_length_bits(v) for v in flatten({c: v1[c] for c in mg.V1}))

    # Roll back after the application wrote new data in the v2 schema.
    edited = set(rng.sample(range(rows), int(rows * args.edited)))
    after = {k: list(v) for k, v in up.items()}
    for r in edited:
        after["usd"][r] += rng.randrange(-5000, 5000)
        after["status3"][r] = rng.randrange(3)
    rolled = program.reverse(after)
    residual = {r for r in range(rows) if any(rolled[c][r] != 0 for c in mg.V2 + mg.WITNESS)}
    again = program.run({c: rolled[c] for c in mg.V1})
    round_trips = {r for r in range(rows) if all(again[c][r] == after[c][r] for c in mg.V2)}
    unedited_exact = all(
        all(rolled[c][r] == v1[c][r] for c in mg.V1) for r in range(rows) if r not in edited
    )
    silent = [r for r in range(rows) if r not in residual and r not in round_trips]
    by_currency = {}
    for r in edited:
        name = {0: "USD", 1: "EUR", 2: "GBP"}[v1["currency"][r]]
        entry = by_currency.setdefault(name, {"edited": 0, "flagged": 0, "round_trip": 0})
        entry["edited"] += 1
        entry["flagged"] += r in residual
        entry["round_trip"] += r in round_trips
    return {
        "rows": rows,
        "lossy_drop_rejected_with": rejection,
        "reverie_down_exact_rows": exact,
        "schema_only_lost_cells": lost_cells,
        "schema_only_cells": rows * len(mg.V1),
        "hand_written_wrong_rows": hand_wrong_rows,
        "hand_written_wrong_rows_excluding_dropped_column": hand_wrong_rows_kept_columns,
        "witness_coded_bits_by_column": witness_breakdown,
        "witness_coded_bits": witness_bits,
        "full_backup_coded_bits": backup_bits,
        "after_edits": {
            "edited_rows": len(edited),
            "unedited_rows_exact": unedited_exact,
            "flagged_rows": len(residual),
            "flagged_but_unedited": len(residual - edited),
            "silently_inconsistent_rows": len(silent),
            "by_currency": by_currency,
        },
    }


def markdown(result: Dict[str, Any]) -> str:
    s, m = result["sagas"], result["migrations"]
    lines = [
        "# Generated undo: saga compensations and data down-migrations",
        "",
        "Generated by `scripts/saga_migration_study.py`.",
        "",
        "## Sagas under concurrent traffic",
        "",
        f"{result['config']['sagas']:,} random checkout sagas (reserve stock → charge with a fee → "
        "award loyalty points capped at 1,000 → book shipping), with other transactions writing the "
        "same records in between. When a step fails, completed steps are compensated, and the "
        "result is compared with *keep what other transactions committed, remove the saga's effect*.",
        "",
        "| loyalty step written as | compensation | correct | fails loudly | silently wrong |",
        "|---|---|---:|---:|---:|",
    ]
    names = {"reverie": "Reverie (step run backward)", "snapshot": "snapshot restore", "hand": "hand-written"}
    styles = {"loyalty": "re-check cap on current balance", "loyalty_delta": "decide on recorded input, apply a delta"}
    for style, tallies in s["styles"].items():
        for key in ("reverie", "snapshot", "hand"):
            t = tallies[key]
            lines.append(
                f"| {styles[style]} | {names[key]} | {t.get('correct', 0):,} | "
                f"{t.get('loud_failure', 0):,} | {t.get('silent_corruption', 0):,} |"
            )
    dirty = s["styles"]["loyalty"]["dirty_read"]
    compensated = dirty.get("yes", 0) + dirty.get("no", 0)
    lines += [
        "",
        f"{compensated:,} sagas needed compensation. In {dirty.get('yes', 0):,} of them another "
        "transaction *read* the saga's effects (a dirty read: a sale limited by reserved stock, "
        "points earned against a raised balance). No compensation can undo another "
        "transaction's decision, so those are judged against what was committed.",
        "",
        f"Writing the capped step the obvious way is rejected by the checker: "
        f"*{s['lossy_step_rejected_with']}*.",
        "",
        "## Data migrations",
        "",
        f"A v1 → v2 migration over {m['rows']:,} rows: convert prices in USD/EUR/GBP to USD cents "
        "(rounding), collapse a 6-value status to 3, pack three flags, drop a column.",
        "",
        "| down-migration | rows restored exactly |",
        "|---|---:|",
        f"| **Reverie (migration run backward)** | **{m['reverie_down_exact_rows']:,} / {m['rows']:,}** |",
        f"| hand-written data down-migration | {m['rows'] - m['hand_written_wrong_rows']:,} / {m['rows']:,} "
        f"({m['rows'] - m['hand_written_wrong_rows_excluding_dropped_column']:,} ignoring the dropped column) |",
        f"| schema-only (old columns come back empty) | {m['schema_only_cells'] - m['schema_only_lost_cells']:,} of {m['schema_only_cells']:,} cells |",
        "",
        f"The witness that makes the rollback exact (rounding remainders, original currency, the "
        f"dropped status bit, the archived column) takes {m['witness_coded_bits'] / 8 / 1024:,.1f} KiB "
        f"coded, against {m['full_backup_coded_bits'] / 8 / 1024:,.1f} KiB for a full backup of the "
        f"v1 columns ({m['full_backup_coded_bits'] / m['witness_coded_bits']:.1f}x). Most of it is the "
        f"archived column itself ({m['witness_coded_bits_by_column']['archived_score'] / 8 / 1024:,.1f} KiB), "
        "which no rollback can avoid keeping; the conversions, the status and the flags need only "
        f"{(m['witness_coded_bits'] - m['witness_coded_bits_by_column']['archived_score']) / 8 / 1024:,.1f} KiB. "
        f"Dropping a column without archiving it is rejected: *{m['lossy_drop_rejected_with']}*.",
        "",
        "### Rolling back after new writes",
        "",
    ]
    a = m["after_edits"]
    lines += [
        f"{a['edited_rows']:,} rows were edited in the v2 schema after the up-migration, then the "
        "down-migration ran.",
        "",
        f"- Unedited rows restored exactly: **{a['unedited_rows_exact']}**.",
        f"- Rows flagged by a non-zero leftover in the witness columns: {a['flagged_rows']:,} "
        f"(unedited rows flagged: {a['flagged_but_unedited']}).",
        f"- Rows silently inconsistent (not flagged, and migrating up again does not reproduce the "
        f"edit): **{a['silently_inconsistent_rows']}**.",
        "",
        "| original currency | edited | flagged | rollback then up reproduces the edit |",
        "|---|---:|---:|---:|",
    ]
    for name, e in sorted(a["by_currency"].items()):
        lines.append(f"| {name} | {e['edited']} | {e['flagged']} | {e['round_trip']} |")
    lines += [
        "",
        "USD rows carry the user's edit back exactly. For converted currencies an edited amount "
        "often has no exact source-currency equivalent; every edited converted row is flagged for "
        "review rather than silently rounded. The flag is conservative: about half the flagged "
        "rows would have round-tripped.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.self_test:
        args.sagas, args.rows = 300, 300
    result = {
        "schema": "reverie_sagas_migrations_v1",
        "config": {"sagas": args.sagas, "rows": args.rows, "edited": args.edited, "seed": args.seed},
        "sagas": saga_study(args),
        "migrations": migration_study(args),
    }
    if args.self_test:
        delta = result["sagas"]["styles"]["loyalty_delta"]["reverie"]
        strict = result["sagas"]["styles"]["loyalty"]["reverie"]
        assert delta.get("silent_corruption", 0) == 0 and delta.get("loud_failure", 0) == 0, delta
        assert strict.get("silent_corruption", 0) == 0, strict
        m = result["migrations"]
        assert m["reverie_down_exact_rows"] == m["rows"], m
        assert m["after_edits"]["silently_inconsistent_rows"] == 0, m
        assert m["after_edits"]["flagged_but_unedited"] == 0, m
        assert result["sagas"]["lossy_step_rejected_with"] and m["lossy_drop_rejected_with"]
        print("ok: saga and migration self-test passed")
        return 0
    text = markdown(result)
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
