# Architecture

Reverie is organized as a Rust workspace. The crate boundaries are meant to
keep syntax, semantic rules, execution, and user interfaces independent enough
that later phases can add backward execution and tooling without tangling the
core language.

## Crates

`reverie-syntax` owns the concrete AST, spans, lexer, parser, and
parser-compatible source formatter. It uses `logos` for tokens and `chumsky`
for grammar composition. Later phases should extend this AST rather than
duplicating language shapes in downstream crates.

`reverie-core` owns semantic checks that are independent of a concrete runtime
state. It rejects destructive updates such as `x += x`, validates procedure
calls, validates `local`/`delocal` pairing, checks array and stack operations,
checks index reversibility rules, checks unit dimensions, type-checks
refinement predicates, and provides the pure `invert(stmt)` transformation.

`reverie-interp` owns `State`, `Value`, expression evaluation, tape I/O,
statement execution, the slot-compiled fast path, and timeline construction.
It runs programs forward and backward; backward execution applies the AST
produced by `invert`. Units are erased before this point; the interpreter
ignores unit annotations on literals, locals, and parameters.

`reverie-cli` owns file loading, diagnostics, command parsing, and conversion
from CLI seed values into runtime state.

`reverie-tui` renders the scrubber UI with `ratatui` and `crossterm`. It also
owns the stable timeline dump format used by CLI tests.

## Data Flow

```text
source.rev
  -> reverie-syntax lex/parse
  -> reverie-core semantic checks + unit checks
  -> reverie-interp tree/slot execute, build timeline, or reverie-syntax format inverse
  -> final State, scrubber frames, or inverse source
```

Syntax, core-check, and span-carrying runtime diagnostics are rendered with
`ariadne`. Runtime errors that occur outside source execution, such as invalid
manually constructed programs, still fall back to ordinary CLI errors.

## State Model

Runtime state is a flat, deterministic map from variable names to values. The
display format is intentionally stable so examples, tests, and future scrubber
snapshots can compare states directly:

```text
{a = 13, b = 21, i = 7, n = 7}
```

Values currently include `int`, `bool`, fixed-size arrays, and integer stacks.
Array elements are assignable places, so `xs[1] += delta` and
`xs[0] <=> xs[2]` use the same primitive reversible operations as scalar
variables. Stacks use reversible transfer operations: `push(x, s)` clears `x`
after moving it onto `s`, and `pop(x, s)` restores the top stack value into a
zero target.

Top-level variables can come from CLI/test seed values or from source-level
`global` declarations. Declared globals are initialized to zero when no seed
overrides them. Scoped temporary variables use `local`/`delocal`.

The tree-walking interpreter uses this map directly as the semantic reference.
The slot-compiled engine prepares a per-program layout, lowers variable names
to numeric slots, executes over a vector-backed store, and converts back to the
same public `State` at the boundary. This keeps CLI output and diagnostics
stable while giving benchmarks a faster Rust execution path.

Tape I/O extends execution state with explicit input and output tapes. `read`
consumes input, stores it in a place, and emits the old place value; `write`
emits a place value. The inverse program uses `unread` and `unwrite` to restore
the input/output tapes without relying on irreversible terminal side effects.

Procedures execute in a local parameter store. On `call`, argument values are
copied into parameter names, declared globals are copied into the procedure
store, the body runs, and mutated parameters/globals are copied back to the
caller. On `uncall`, the inverse body runs. V1 uses a no-alias call model:
duplicate by-reference arguments and constant duplicate element arguments are
rejected statically instead of merged into shared cells. Dynamic same-root
element arguments are rejected unless constant indexes prove the cells are
distinct.

Local variables also use a no-shadow model. A live binding name cannot be
reused by an inner `local`; choose a fresh temporary name instead.

The unit checker has its own static environment layered over external CLI seed
annotations, procedures, and locals. Top-level CLI variables are unknown-unit
inputs unless the user supplies `--type name=TYPE`; annotated locals and
procedure parameters establish concrete dimensions, including array element
dimensions such as `array<int<m>>`. This keeps units purely compile-time and
avoids changing the runtime state representation.

Refinements reuse the expression evaluator at runtime. The core checker only
proves that the predicate is boolean; `reverie-interp` enforces it at local and
procedure boundaries.

## Reversibility Boundary

The forward interpreter enforces the runtime assertions required by reversible
conditionals and loops. Backward execution is mechanical inversion plus normal
execution:

```text
invert(assert c)    = assert c
invert(x += e)      = x -= e
invert(x -= e)      = x += e
invert(x ^= e)      = x ^= e
invert(x <=> y)     = x <=> y
invert(xs[i] += e)  = xs[i] -= e
invert(xs[i] <=> y) = xs[i] <=> y
invert(S1; S2)      = invert(S2); invert(S1)
invert(if ... fi)   = swap entry/exit conditions and invert branches
invert(from ... )   = swap entry/exit conditions and invert body/step
```

The primary oracle is:

```text
execute(P, S) -> S'
execute_backward(P, S') -> S
```

## Timeline Model

`reverie-interp` can build a `Timeline`, which is a vector of labeled states.
The first frame is the seeded initial state; later frames are emitted by
primitive updates, swaps, array element operations, control-flow markers,
procedure call enter/exit points, procedure body statements in their local
parameter store, and local/delocal boundaries. `reverie-tui` renders those
frames interactively, and `reverie scrub --dump` prints them for
non-interactive verification.
