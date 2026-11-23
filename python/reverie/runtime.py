"""General-purpose runners built on generated inverses.

* :class:`Saga`: a sequence of Reverie steps over shared state; when a step
  fails, the completed steps are compensated by running them backward. The
  log is plain JSON, so compensation can resume after a crash.
* :class:`Migration`: write the per-row logic of a data migration once; run
  it up or down over a table held as columns (dict of lists) or in SQLite.
  Witness columns live in a sidecar table so ``down`` is exact.
* :class:`Simulation`: any frame-loop Reverie program (``iterate int f =
  start to stop``) gets ``step`` and ``rewind`` by any number of frames,
  with no snapshots.

All three only need ordinary Reverie programs; the inverse is never written
by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from ._reverie import Program, load, loads
from ._types import ExecutionError, ReverieError

__all__ = [
    "variables",
    "Step",
    "Saga",
    "SagaResult",
    "Migration",
    "MigrationReport",
    "Simulation",
]


def _program(program: Union[Program, str]) -> Program:
    if isinstance(program, Program):
        return program
    return loads(program)


def variables(program: Program) -> List[str]:
    """Every variable a program reads or writes from its store (external
    variables plus declared globals), from ``reverie explain``."""
    report = program.explain()
    names = [entry["name"] for entry in report.get("external_store", [])]
    names += [entry["name"] for entry in report.get("declared_store", [])]
    return sorted(set(names))


# -- sagas ----------------------------------------------------------------------


@dataclass
class Step:
    """One saga step: a Reverie program over shared ``state`` variables and
    per-saga ``params``. Every other variable the program uses is treated as
    its witness and kept in the saga log."""

    name: str
    program: Union[Program, str]
    params: Sequence[str] = ()

    def __post_init__(self) -> None:
        self.program = _program(self.program)
        names = variables(self.program)
        self.params = tuple(self.params)
        self._names = names

    def split(self, state_names: Iterable[str]) -> Tuple[List[str], List[str]]:
        state = [n for n in self._names if n in set(state_names) and n not in self.params]
        witness = [n for n in self._names if n not in state and n not in self.params]
        return state, witness


@dataclass
class SagaResult:
    status: str  # "committed", "compensated", "compensation_failed"
    failed_step: Optional[str] = None
    error: Optional[str] = None
    log: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "status": self.status,
                "failed_step": self.failed_step,
                "error": self.error,
                "log": self.log,
            }
        )


class Saga:
    """Run steps in order against ``state`` (any mutable mapping: a dict, or
    an adapter over your database). On a step failure, completed steps are
    undone newest-first by running them backward against the *current*
    state, so concurrent changes made by others are kept.

    Write steps as deltas computed from recorded inputs (see
    ``docs/research/README.md`` §10) and compensation is exact even with
    concurrent writers; otherwise it is exact or fails loudly, never
    silently wrong."""

    def __init__(self, steps: Sequence[Step]) -> None:
        self.steps = list(steps)

    def execute(
        self, state: MutableMapping[str, Any], params: Mapping[str, Any]
    ) -> SagaResult:
        log: List[Dict[str, Any]] = []
        for step in self.steps:
            names, witness = step.split(state.keys())
            store = {n: state[n] for n in names}
            store.update({p: params[p] for p in step.params})
            try:
                after = step.program.run(store)
            except ReverieError as error:
                result = self.compensate(state, log)
                result.failed_step = step.name
                result.error = str(error)
                return result
            for n in names:
                state[n] = after[n]
            log.append(
                {
                    "step": step.name,
                    "params": {p: params[p] for p in step.params},
                    "witness": {w: after[w] for w in witness if w in after},
                }
            )
        return SagaResult("committed", log=log)

    def compensate(
        self, state: MutableMapping[str, Any], log: Sequence[Mapping[str, Any]]
    ) -> SagaResult:
        """Undo the steps recorded in ``log`` (from :meth:`execute` or a
        stored :meth:`SagaResult.to_json`), newest first."""
        by_name = {step.name: step for step in self.steps}
        remaining = list(log)
        while remaining:
            entry = remaining[-1]
            step = by_name[entry["step"]]
            names, _ = step.split(state.keys())
            store = {n: state[n] for n in names}
            store.update(entry["params"])
            store.update(entry["witness"])
            try:
                before = step.program.reverse(store)
            except ExecutionError as error:
                return SagaResult(
                    "compensation_failed",
                    error=f"cannot undo {step.name} exactly: {error}",
                    log=remaining,
                )
            for n in names:
                state[n] = before[n]
            remaining.pop()
        return SagaResult("compensated", log=[])


# -- migrations -------------------------------------------------------------------


@dataclass
class MigrationReport:
    rows: int
    flagged_rows: List[int]
    """Rows a ``down`` could not restore exactly (non-zero leftovers in new or
    witness columns), typically because they were edited after ``up``."""


class Migration:
    """A data migration written as per-row Reverie code.

    ``body`` uses ``r`` for the row index and refers to columns as
    ``name[r]``. ``old`` columns must end at zero after the body (their
    content has moved into ``new`` or ``witness`` columns); the checker makes
    sure nothing is lost.

    >>> m = Migration(
    ...     "cents[r] += dollars[r] * 100; dollars[r] -= cents[r] / 100",
    ...     old=["dollars"], new=["cents"])
    >>> m.up({"dollars": [1, 2]})["cents"]
    [100, 200]
    """

    def __init__(
        self,
        body: str,
        *,
        old: Sequence[str],
        new: Sequence[str],
        witness: Sequence[str] = (),
        constants: str = "",
    ) -> None:
        self.body, self.constants = body, constants
        self.old, self.new, self.witness = list(old), list(new), list(witness)
        self._programs: Dict[int, Program] = {}
        self.program_for(1)  # check the logic once, eagerly

    def source(self, rows: int) -> str:
        columns = "\n".join(
            f"global {name}: tensor<int, {rows}>;" for name in self.old + self.new
        )
        witnesses = "\n".join(
            f"global {name}: witness<tensor<int, {rows}>>;" for name in self.witness
        )
        return (
            f"{self.constants}\n{columns}\n{witnesses}\n\n"
            "procedure main() {\n"
            f"  iterate int r = 0 to {rows - 1}\n"
            f"{self.body}\n"
            "  end\n"
            "}\n"
        )

    def program_for(self, rows: int) -> Program:
        if rows not in self._programs:
            self._programs[rows] = loads(self.source(max(rows, 1)))
        return self._programs[rows]

    @staticmethod
    def _rows(table: Mapping[str, Sequence[int]]) -> int:
        lengths = {len(v) for v in table.values()}
        if len(lengths) > 1:
            raise ValueError("all columns must have the same length")
        return lengths.pop() if lengths else 0

    def up(self, table: Mapping[str, Sequence[int]]) -> Dict[str, List[int]]:
        """Old columns → new + witness columns."""
        rows = self._rows({c: table[c] for c in self.old})
        out = self.program_for(rows).run({c: list(table[c]) for c in self.old})
        leftover = [c for c in self.old if any(out[c])]
        if leftover:
            raise ValueError(f"migration body leaves data in old columns {leftover}")
        return {c: out[c] for c in self.new + self.witness}

    def down(
        self, table: Mapping[str, Sequence[int]]
    ) -> Tuple[Dict[str, List[int]], MigrationReport]:
        """New + witness columns → old columns, plus rows that could not be
        restored exactly."""
        rows = self._rows({c: table[c] for c in self.new + self.witness})
        back = self.program_for(rows).reverse(
            {c: list(table[c]) for c in self.new + self.witness}
        )
        flagged = [
            r for r in range(rows) if any(back[c][r] for c in self.new + self.witness)
        ]
        return {c: back[c] for c in self.old}, MigrationReport(rows, flagged)

    # SQLite ---------------------------------------------------------------------

    def up_sqlite(self, conn, table: str) -> int:
        """Migrate ``table`` in place: add the new columns, fill them, store
        witness columns in ``<table>__reverie_witness`` keyed by rowid, and
        drop the old columns. Returns the number of rows."""
        rowids, data = _read_columns(conn, table, self.old)
        result = self.up(data)
        with conn:
            for c in self.new:
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{c}" INTEGER')
            _write_columns(conn, table, rowids, {c: result[c] for c in self.new})
            side = f"{table}__reverie_witness"
            cols = ", ".join(f'"{c}" INTEGER' for c in self.witness)
            conn.execute(f'CREATE TABLE "{side}" (row_id INTEGER PRIMARY KEY{", " + cols if cols else ""})')
            conn.executemany(
                f'INSERT INTO "{side}" VALUES ({", ".join("?" * (1 + len(self.witness)))})',
                [
                    (rid, *[result[c][i] for c in self.witness])
                    for i, rid in enumerate(rowids)
                ],
            )
            for c in self.old:
                conn.execute(f'ALTER TABLE "{table}" DROP COLUMN "{c}"')
        return len(rowids)

    def down_sqlite(self, conn, table: str) -> MigrationReport:
        """Undo :meth:`up_sqlite`: restore the old columns exactly, drop the
        new columns and the witness table. Rows that cannot be restored
        exactly are reported and the database is left unchanged."""
        side = f"{table}__reverie_witness"
        rowids, data = _read_columns(conn, table, self.new)
        witness_rows = {
            row[0]: row[1:]
            for row in conn.execute(
                f'SELECT row_id{"".join(", " + chr(34) + c + chr(34) for c in self.witness)} FROM "{side}"'
            )
        }
        for i, c in enumerate(self.witness):
            data[c] = [witness_rows[rid][i] for rid in rowids]
        old, report = self.down(data)
        if report.flagged_rows:
            return report
        with conn:
            for c in self.old:
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{c}" INTEGER')
            _write_columns(conn, table, rowids, old)
            for c in self.new:
                conn.execute(f'ALTER TABLE "{table}" DROP COLUMN "{c}"')
            conn.execute(f'DROP TABLE "{side}"')
        return report


def _read_columns(conn, table: str, columns: Sequence[str]):
    quoted = ", ".join(f'"{c}"' for c in columns)
    rows = conn.execute(f'SELECT rowid, {quoted} FROM "{table}" ORDER BY rowid').fetchall()
    rowids = [row[0] for row in rows]
    data = {c: [row[1 + i] for row in rows] for i, c in enumerate(columns)}
    return rowids, data


def _write_columns(conn, table: str, rowids: Sequence[int], data: Mapping[str, Sequence[int]]):
    columns = list(data)
    assignments = ", ".join(f'"{c}" = ?' for c in columns)
    conn.executemany(
        f'UPDATE "{table}" SET {assignments} WHERE rowid = ?',
        [(*[data[c][i] for c in columns], rid) for i, rid in enumerate(rowids)],
    )


# -- simulations ------------------------------------------------------------------


class Simulation:
    """Step a frame-loop Reverie program forward and rewind it, exactly.

    The program must loop ``iterate int f = start to stop`` over frames,
    with ``start`` and ``stop`` as seeds; ``state`` holds the evolving
    variables, ``inputs`` the per-frame input tapes, and everything else the
    program declares is its witness (zeros for frames not yet simulated).

    ``rewind(k)`` runs the last ``k`` frames backward: no snapshots, and
    the result is bit-identical to the state ``k`` frames ago.
    """

    def __init__(
        self,
        program: Union[Program, str],
        *,
        state: Mapping[str, Any],
        inputs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.program = _program(program)
        self.state = dict(state)
        self.inputs = {k: v for k, v in (inputs or {}).items()}
        known = set(self.state) | set(self.inputs) | {"start", "stop"}
        self.witness_names = [n for n in variables(self.program) if n not in known]
        self.witness: Dict[str, Any] = {}
        self.frame = 0

    def _store(self, start: int, stop: int) -> Dict[str, Any]:
        store = dict(self.state)
        store.update(self.inputs)
        store.update(self.witness)
        store.update(start=start, stop=stop)
        return store

    def _absorb(self, out: Mapping[str, Any]) -> None:
        self.state = {k: out[k] for k in self.state}
        self.witness = {k: out[k] for k in self.witness_names if k in out}

    def step(self, frames: int = 1) -> Dict[str, Any]:
        """Advance ``frames`` frames; returns the new state."""
        if frames <= 0:
            return dict(self.state)
        out = self.program.run(self._store(self.frame, self.frame + frames - 1))
        self._absorb(out)
        self.frame += frames
        return dict(self.state)

    def rewind(self, frames: int = 1) -> Dict[str, Any]:
        """Go back ``frames`` frames by running them backward."""
        frames = min(frames, self.frame)
        if frames <= 0:
            return dict(self.state)
        out = self.program.reverse(self._store(self.frame - frames, self.frame - 1))
        self._absorb(out)
        self.frame -= frames
        return dict(self.state)

    def set_input(self, name: str, frame: int, value: Any) -> None:
        """Change an input for a frame (e.g. a late remote input). Only
        frames not yet simulated, or rewound past, should be changed."""
        if frame < self.frame:
            raise ValueError(f"frame {frame} is already simulated; rewind first")
        self.inputs[name][frame] = value
