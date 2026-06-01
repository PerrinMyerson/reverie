# Roadmap

This roadmap preserves the phase-gated shape of the original spec. Each phase
should end with tests green and examples updated.

## Phase 0: Skeleton

Done:

- Cargo workspace
- five crates
- CI
- `skip` parser and runner
- stable empty-store output

## Phase 1: Forward Core

Done:

- reversible updates, swap, sequence, conditionals, and loops
- signed `i64` wrapping integer semantics
- Janus-style unary numeric negation
- Janus-style aliases for equality, not-equals, remainder, xor, boolean
  connectives, xor-update, and swap
- Janus-style optional condition and loop clauses, parsed as `skip`
- expression parser and evaluator
- CLI state seeding with `--var name=value`
- executable Fibonacci pair-transform example
- more examples for loop assertion failures and wrapping arithmetic
- injectivized max example for the classic extra-witness pattern

## Phase 2: Inversion

Done:

- implement `invert(stmt)` in `reverie-core`
- add backward execution in `reverie-interp`
- add `reverie reverse <file>`
- add property tests for forward-then-backward round trips
- add `invert(invert(P)) == P` tests
- pretty-print inverse programs for debugging
- broaden oracle generation to cover more dynamically valid conditionals and loops

## Phase 3: Procedures and Locals

Done:

- parser and AST support for top-level procedures
- `proc`, `call`, and `uncall`
- Janus-style `procedure` keyword and parenless parameterless calls
- by-reference parameter binding
- `local`/`delocal`
- oracle tests for procedure round trips
- explicit v1 no-alias model for duplicate by-reference arguments
- explicit v1 no-shadow model for locals

## Phase 4: Units

Done:

- unit annotations on integer types
- dimensional algebra over unit exponents
- source syntax for explicit exponents such as `m^2`
- CLI annotations for top-level seeded variables
- static rejection for mismatched units
- erased units at runtime

## Phase 5: Refinements

Done:

- `where` predicates
- runtime-checked refinements
- refinement violations as good diagnostics
- keep the reversibility oracle green
- richer refinement examples over units and loops

Deferred to stretch:

- optional static discharge through SMT

## Phase 6: Scrubber TUI

Done:

- `ratatui` source/state/scrubber interface
- step forward and backward
- jump to a step
- variable watch timelines
- deterministic timeline dump for tests and non-TTY demos
- source-line highlighting for procedure-local execution contexts
- README GIF generated from the real scrubber timeline

## V1 Fill-Out: Arrays

Done:

- fixed-size array values
- array literals and indexed reads
- Janus-style array length reads with `size(xs)` plus readable `len(xs)` sugar
- reversible element updates with `+=`, `-=`, and `^=`
- boolean arrays and indexed boolean xor updates
- array element swaps
- `array<int>`, `array<bool>`, and unitful element types such as
  `array<int<m>>`
- CLI seeding with `--var 'xs=[1,2,3]'`
- executable `examples/array.rev`
- unitful array example in `examples/array_units.rev`
- negative runtime example in `examples/array_oob.rev`
- generated local/delocal programs with dynamically valid deletion assertions

## V1 Fill-Out: Stacks

Done:

- `stack` type annotations
- `nil` empty-stack literal
- reversible `push(x, s)` and `pop(x, s)` statements
- `empty(s)` / `is_empty(s)`, `top(s)` / `peek(s)`, and stack-aware `size(s)`
  expressions
- CLI stack seeding with `--var s=nil` and `--var 's=stack[3,2,1]'`
- executable `examples/stack.rev`

## V1 Polish: Diagnostics

Done:

- ariadne syntax diagnostics
- ariadne core/type/unit/refinement check diagnostics
- ariadne runtime diagnostics when execution errors carry source spans
- source spans for irreversible updates, procedure mistakes, unit mismatches,
  non-boolean refinements, constant zero divisors, and statically invalid
  constant array indexes
- source spans for runtime refinements, failed assertions, failed delocals,
  division/remainder by zero, and bad array indexes
- multiple labels for two-sided type/unit mismatches
- call-site context for runtime errors that occur inside procedures
- diagnostic docs in `docs/diagnostics.md`

## V1 Polish: Examples and Docs

Done:

- guided examples catalog with expected outputs
- positive examples for skip, Fibonacci, wrapping arithmetic, procedures,
  arrays, units, top-level unit annotations, refinements, and scrubber dumps
- negative examples for static semantic errors and runtime diagnostics
- docs links from the README to the examples catalog
- v1 acceptance audit mapping phase requirements to implementation evidence
- grammar reference for the accepted v1 syntax
- CLI reference for `check`, `run`, `reverse`, `invert`, and `scrub`
- injectivization guide explaining how to make non-injective computations
  reversible
- ergonomic examples for statically proven element arguments, readable
  `len(xs)`, assertion helpers, and prefix/postfix `++`/`--` update sugar
- example coverage test requiring every `.rev` example to appear in the
  examples guide and CLI tests
- evaluation guide focused on soundness oracles, negative corpus, capability
  corpus, and demo quality

## Performance Track

Done:

- checked benchmark corpus against `kirkedal/Jana-JanusInterp`
- output verification before every timed workload
- per-workload, median, and geometric-mean speedup gates for local and CI
  smoke runs
- benchmark artifact validation for coverage breadth, sample counts, unique
  workload names, and speedup consistency
- internal Criterion coverage for procedure-heavy call/uncall execution
- internal Criterion coverage for slot compile time versus compiled execution
- internal Criterion coverage for scrub timeline construction
- internal Criterion coverage for a Janus-style Turing-machine simulation
- internal Criterion coverage for recursive stacks, multidimensional arrays,
  and reversible tape I/O
- internal Criterion coverage for legacy Janus fixed-point `*/` wave-style
  arithmetic
- checked Jana-vs-Reverie workloads for procedure-heavy call/uncall execution
- checked Jana-vs-Reverie workload for recursive stack reversal and cleanup
- forward, reverse, and round-trip workloads across scalar loops,
  table-driven permutation, matrix multiplication and transpose, factorization,
  sorting, and a wave-kernel fixture
- checked Jana-vs-Reverie workloads for a Janus-style reversible
  Turing-machine simulation

## Stretch

- bytecode VM beyond the current slot-compiled interpreter
- snapshot and replay for fast jumps
- SMT-backed refinement checking
- Cranelift backend
