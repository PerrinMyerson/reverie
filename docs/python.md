# Using Reverie from Python

Reverie programs run forward and backward deterministically, and the checker
proves that running backward undoes running forward exactly. The Python
package wraps the same parser, checker and interpreters as the CLI, and adds
runners for the places where "undo" matters in practice: saga compensations,
data migrations, and time-travel simulations.

Every Python example on this page is executed by the test suite
(`python/tests/test_docs.py`), so it works as written.

## Install

```sh
pip install reverie-lang          # the Python package and the `reverie` command
```

From a checkout (needs a Rust toolchain):

```sh
pip install maturin && maturin develop --release
```

Both give you `import reverie`, plus a `reverie` command identical to the Rust
CLI (`reverie run`, `reverse`, `check`, `fmt`, `invert`, `explain`, `scrub`,
`roundtrip`, `verify-roundtrip`). `python -m reverie ...` works too. For
editor support, see [`editors/vscode`](../editors/vscode/README.md).

## Quick start

```python
import reverie

fib = reverie.loads("""
from i == 0 do
  a += b;
  a <=> b;
  i += 1
loop skip until i == n
""")
assert fib.run(n=7, i=0, a=0, b=1) == {"a": 13, "b": 21, "i": 7, "n": 7}
assert fib.reverse(n=7, i=7, a=13, b=21) == {"a": 0, "b": 1, "i": 0, "n": 7}
print(fib.inverse_source())  # the mechanically derived inverse program
```

Programs that would lose information are rejected before they run:

```python
import reverie

try:
    reverie.loads("x += x")
except reverie.CheckError as error:
    assert (error.line, error.column) == (1, 6)
    assert "witness" in error.rendered  # the checker suggests a fix
```

## The core API

### Loading

`reverie.load(path, *, engine="slot", types=None, legacy_janus=False)` and
`reverie.loads(source, ...)` parse and check a program and return a
`Program`.

- `engine`: `"slot"` (compiled, the default) or `"tree"` (the reference
  interpreter). Both give identical results.
- `types`: annotations for external variables, such as
  `{"distance": "int<m>"}`, checked like the CLI's `--type`.
- `legacy_janus`: Janus-compatible parsing and update semantics.

```python
import reverie

speed = reverie.loads("x += v * t", types={"x": "int<m>", "v": "int<m/s>", "t": "int<s>"})
assert speed.run(x=0, v=3, t=4) == {"t": 4, "v": 3, "x": 12}

try:
    reverie.loads("x += t", types={"x": "int<m>", "t": "int<s>"})
except reverie.CheckError as error:
    assert "unit mismatch" in error.message
```

### Running

| method | what it does |
|---|---|
| `run(store=None, /, **vars)` | run forward; returns the final store as a dict |
| `reverse(store=None, /, **vars)` | run backward; returns the restored store |
| `execute(store, *, inputs=, outputs=, backward=False)` | run with I/O tapes; returns an `Execution` with `store`, `inputs`, `outputs` and `observations` (lines printed by `show` / `printf`) |
| `timeline(store, **vars)` | every step of a forward run as `Frame`s (`step`, `label`, `line`, `column`, `store`) |
| `roundtrip(store, **vars)` | forward then backward; returns the CLI's fingerprinted proof document |
| `inverse()` | the inverse as a `Program`; running it forward is running this one backward |
| `explain(*, ml=False)` | the `reverie explain --json` report (variables, features, witness metrics) |
| `inverse_source()` / `formatted_source()` | source text of the inverse / canonical formatting |

Stores go in as a mapping, keyword arguments, or both, and come out as plain
dicts:

| Reverie value | Python value |
|---|---|
| `int` | `int` |
| `bool` | `bool` |
| arrays and tensors | (nested) `list`; tuples and NumPy arrays are accepted as input |
| `stack` | `reverie.Stack` (immutable, top first) |

```python
import reverie

echo = reverie.loads("write x; read x; write x")
result = echo.execute({"x": 5}, inputs=[7])
assert result.store == {"x": 7} and result.outputs == [5, 5, 7]

undo = echo.execute(result.store, inputs=result.inputs, outputs=result.outputs, backward=True)
assert undo.store == {"x": 5} and undo.inputs == [7]
```

Execution releases the GIL, so programs can run in parallel threads.

### Proofs

`Program.roundtrip(...)` returns the same `reverie_roundtrip_proof_v1`
document as `reverie roundtrip --proof-output`, byte for byte.
`reverie.verify(proof)` checks it the way `reverie verify-roundtrip` does,
including replaying the program. For programs built with `loads`, pass
`source=program`, since there is no file to replay from.

```python
import reverie

program = reverie.loads("x *= 3; y += x")
proof = program.roundtrip(x=5, y=0)
assert proof["passed"]
assert reverie.verify(proof, source=program)["passed"]
```

### Errors

Every error subclasses `reverie.ReverieError` and carries `message`,
`rendered` (the CLI's report, without color), `line`, `column` and `span`.

| error | raised when |
|---|---|
| `CheckError` (and its subclass `ParseError`) | the program does not parse or check |
| `SeedError` (also a `ValueError`) | a store value or name does not fit the program |
| `ExecutionError` | a runtime assertion or branch condition fails |

## Writing reversible code

- **Updates, not assignments.** `x += e`, `x -= e`, `x ^= e`, `x *= c` (odd
  constant `c`) and `a <=> b` are the building blocks. The right-hand side
  must not read the variable being changed, but `xs[i + 1] += xs[i]` is fine:
  the checker knows those are different cells.
- **Conditionals need an exit test** (`if c then … else … fi c2`), and loops
  an entry test (`from c do … loop … until c2`). These are how the backward
  run knows which branch was taken.
- **Lossy steps keep a witness.** To clamp `x` at a cap, move the excess into
  a witness variable instead of discarding it:

  ```text
  if x > 100 then
    excess += x - 100;
    x -= excess
  fi excess > 0
  ```

- **For undo under concurrency, decide on recorded inputs and apply a
  delta.** Record what the step saw (`seen += balance`), compute the change
  from it, then `balance += change`. The inverse subtracts exactly the
  change, so it composes with whatever other writers did in the meantime.

## Runners (`reverie.runtime`)

### Sagas: generated compensations

A `Saga` runs steps against shared state (any mutable mapping: a dict, or an
adapter over your database). If a step fails, completed steps are undone
newest first by running them backward against the *current* state. Each step
declares its `params`; the program's other variables are split between shared
state (keys present in the state mapping) and per-step witnesses, which go
into the saga's log.

```python
import reverie
from reverie.runtime import Saga, Step

reserve = Step("reserve", """
global stock = 0;
global held = 0;
global qty = 0;
procedure main() {
  stock -= qty;
  held += qty;
  assert stock >= 0
}
""", params=["qty"])

charge = Step("charge", """
global wallet = 0;
global amount = 0;
procedure main() {
  wallet -= amount;
  assert wallet >= 0
}
""", params=["amount"])

saga = Saga([reserve, charge])
state = {"stock": 10, "held": 0, "wallet": 50}

result = saga.execute(state, {"qty": 3, "amount": 80})   # the charge fails
assert result.status == "compensated" and result.failed_step == "charge"
assert state == {"stock": 10, "held": 0, "wallet": 50}  # the reservation was undone

assert saga.execute(state, {"qty": 3, "amount": 20}).status == "committed"
assert state == {"stock": 7, "held": 3, "wallet": 30}
```

`SagaResult.to_json()` stores the log. `Saga.compensate(state, log)` undoes
a stored log later, for example after a crash. If concurrent activity makes
an exact undo impossible, compensation returns `"compensation_failed"` rather
than writing a wrong value.

### Migrations: exact down-migrations

Write the per-row logic once (`r` is the row index). The checker makes sure
every old column's information ends up in a new column or a witness column,
so the down-migration, which is generated, restores the data exactly.

```python
import sqlite3
from reverie.runtime import Migration

# v1 stores prices in dollars and cents separately; v2 stores cents only.
migration = Migration(
    """
    cents[r] += dollars[r] * 100 + loose_cents[r];
    dollars[r] -= cents[r] / 100;
    loose_cents[r] -= cents[r] % 100
    """,
    old=["dollars", "loose_cents"],
    new=["cents"],
)
assert migration.up({"dollars": [1, 2], "loose_cents": [50, 5]}) == {"cents": [150, 205]}

conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE prices (name TEXT, dollars INTEGER, loose_cents INTEGER)")
conn.executemany("INSERT INTO prices VALUES (?, ?, ?)", [("tea", 3, 25), ("cake", 4, 99)])
migration.up_sqlite(conn, "prices")
assert conn.execute("SELECT name, cents FROM prices").fetchall() == [("tea", 325), ("cake", 499)]

report = migration.down_sqlite(conn, "prices")
assert report.flagged_rows == []
assert conn.execute("SELECT * FROM prices").fetchall() == [("tea", 3, 25), ("cake", 4, 99)]
```

`up_sqlite` adds the new columns, stores witness columns in
`<table>__reverie_witness` keyed by rowid, and drops the old columns. It
needs SQLite 3.35 or newer. If rows were edited after `up` in ways that
cannot be mapped back exactly, `down` reports them in `flagged_rows` and
leaves the database unchanged.

### Simulations: rewind without snapshots

Any program that loops over frames as `iterate int f = start to stop` can be
stepped and rewound. That is the core of rollback netcode, and of a debugger
that steps backward.

```python
import reverie
from reverie.runtime import Simulation

program = """
global thrust: tensor<int, 100>;
global start = 0;
global stop = 0;
global x = 0;
global v = 0;
global floor_hit: witness<tensor<int, 100>>;

procedure main() {
  iterate int f = start to stop
    v += thrust[f];
    x += v;
    if x < 0 then
      floor_hit[f] += x;
      x -= floor_hit[f]
    fi floor_hit[f] < 0
  end
}
"""
sim = Simulation(program, state={"x": 0, "v": 0}, inputs={"thrust": [1, -3] * 50})
sim.step(10)
before = dict(sim.state)
sim.step(40)
assert sim.rewind(40) == before and sim.frame == 10
```

## Research modules

These implement the experiments in [docs/research](research/README.md) and
are usable as they are:

| module | what it provides |
|---|---|
| `reverie.lab` | minimal-witness SGD for a ReLU MLP: exact rewind to any step and exact unlearning (`SGDRewind`); ReLU erasure profiles; search-based minimal witnesses (`SearchWitness`) |
| `reverie.ledger` | witness accounting in coded bits and Landauer's erasure bound |
| `reverie.rollback` | a reversible fighting-game step and a rollback-netcode harness |
| `reverie.saga`, `reverie.migrate` | the saga and migration studies |
| `reverie.dna` | a rotating-code DNA storage codec with a generated decoder; molecular recorder capacity |

## Limits

- **Integers and booleans only, fixed-size tensors.** Use fixed point (Q31
  helpers in `reverie.lab`) for fractional values. Tensor sizes are part of
  the program, so the runners generate a program per size.
- **Speed.** Reverie is an interpreter: about the speed of CPython on scalar
  loops, and several times faster on tensor builtins (see
  [the reversibility report](research/reversibility-report.md)). Running
  backward costs about the same as forward.
