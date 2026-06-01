# Changelog

## Unreleased

- Phase 0 skeleton: Rust workspace, five crates, CI, `skip` parser, and CLI
  execution from an empty store.
- Phase 1 forward core: expressions, updates, swaps, reversible conditionals,
  reversible loops, seeded CLI variables, and executable Fibonacci example.
- Phase 1 examples now include wrapping arithmetic and a negative loop
  assertion diagnostic.
- Phase 2 inversion: pure `invert`, backward execution, `reverie reverse`, and
  property tests for involution and round trips.
- The CLI now has `reverie invert <file>`, backed by a parser-compatible
  pretty-printer for inspecting mechanically derived inverse programs.
- Phase 3 procedures and locals: top-level `proc`, `call`, `uncall`,
  reversible `local`/`delocal`, procedure examples, and procedure round-trip
  tests.
- Procedure/local docs and examples now make the v1 no-alias/no-shadow model
  explicit, with negative examples for duplicate call arguments and local
  shadowing.
- Phase 4 units: unit annotations and literals, dimensional checking, positive
  and negative unit examples, and docs for erasure semantics.
- Phase 5 refinements: `where` predicates on locals and params, static
  boolean checks, runtime enforcement, positive and negative examples, and
  docs for refinement behavior.
- Refinement examples now include unitful predicates inside reversible loops,
  with positive and negative runtime examples.
- Phase 6 scrubber: timeline construction, `reverie scrub`, ratatui interface,
  non-interactive `--dump`, tests, and scrubber docs.
- Scrubber timelines now trace procedure bodies with procedure-local state, so
  procedure source lines can be highlighted instead of treating calls as opaque
  frames.
- V1 array fill-out: fixed-size array values, literals, indexed reads,
  reversible element updates/swaps, `array<int>` annotations, CLI array seeding,
  `examples/array.rev`, `examples/array_units.rev`, and array docs.
- V1 diagnostics polish: core/type/unit/refinement check errors now render as
  ariadne source reports, with diagnostic docs.
- Type and unit mismatch diagnostics now carry two source labels when both
  sides are available, showing the expected side and the mismatched value.
- Runtime diagnostics now carry source spans for execution failures such as
  failed refinements, failed assertions, bad `delocal`s, division by zero, and
  out-of-bounds array indexes.
- Runtime diagnostics inside procedures now include call-site context so the
  report shows both where execution failed and which `call`/`uncall` entered
  the procedure.
- Reversibility oracle now fuzzes fixed-size array stores with indexed updates
  and swaps in addition to scalar straight-line programs.
- Reversibility oracle now also fuzzes dynamically valid conditionals and
  finite Janus-style loops.
- Reversibility oracle now generates local array programs with dynamically
  valid `delocal` assertions.
- Units now support source-level integer exponents such as `m^2` and `m/s^2`,
  with an executable `examples/units_exponents.rev`.
- CLI seed variables can now be statically annotated with `--type name=TYPE`,
  enabling top-level unit checking without wrapping values in locals.
- Added a guided examples catalog with expected outputs for the positive tour
  and grouped static/runtime negative examples.
- Added a generated README scrubber GIF and a reproducible rendering script.
- Added a v1 acceptance audit that maps phase requirements to files, examples,
  and tests.
- Added grammar and CLI references for the implemented v1 surface.
- Added an injectivized max example and guide showing how an extra witness
  turns a non-injective computation into a reversible one.
- Added an example coverage test so every `.rev` example must be documented in
  the examples guide and exercised by CLI tests.
- Added a fuzzed uncall-fidelity oracle, an evaluation guide, and explicit
  negative examples for irreversible updates, failed `fi`, failed `delocal`,
  and witness-free compare-swap.
- Added a slot-compiled execution engine, made CLI `run`/`reverse` default to
  it, kept `--engine tree` as the semantic reference, and added Criterion
  benchmarks plus performance benchmarking docs.
- Added `examples/janus_sort.rev`, a Reverie translation of the classic Janus
  reversible bubble-sort example, with CLI coverage, benchmark coverage, and a
  Janus compatibility matrix.
- Optimized indexed array reads in both interpreters so array-heavy Janus-style
  programs clone only touched cells instead of whole arrays.
- Added source-level `global` declarations with zero initialization and
  parameterless procedures that can mutate declared globals, closing a major
  Janus compatibility gap.
- Added the Janus-style `main` entry convention: when a program omits the
  top-level body, Reverie runs `call main()`.
- Added reversible tape I/O with `read`, `write`, mechanical `unread`/
  `unwrite`, CLI `--input`/`--output` tape seeding, and `examples/io.rev`.
- Added `examples/janus_root.rev`, a translation of the original Janus integer
  square-root example, with forward/backward CLI coverage.
- Added `examples/janus_factor.rev`, a translation of the original Janus
  factorization example, with forward/backward CLI coverage.
- Added a checked external benchmark harness against
  `kirkedal/Jana-JanusInterp`, with Fibonacci, square-root, and factorization
  workloads verified before timing.
- Added a modern-Jana 50-element reverse-order sort workload to the external
  benchmark harness, checked against Reverie's Janus-style sort example.
- Added checked reverse Jana-vs-Reverie benchmark workloads for Fibonacci,
  square root, factorization, and sort, with Jana fixtures seeded at final
  states and Reverie using native `reverse`.
- Added Janus-style standalone `assert` statements, including inversion,
  runtime diagnostics, scrub timeline support, examples, and CLI coverage.
- Added Janus-style unary logical negation with static bool checking in both
  interpreters and updated assertion examples to exercise it.
- Added Janus-style unary numeric negation with wrapping `i64` runtime
  semantics, unit-preserving type checks, examples, and CLI coverage.
- Added Janus-style operator aliases for equality, not-equals, logical
  connectives, remainder, infix xor, xor-update, and swap.
- Added Janus-style bitwise `&`/`|` and shift `<<`/`>>` expressions, while
  keeping nested type syntax such as `array<array<int>>` parseable.
- Added legacy Janus integer conditions for assertions, conditionals, and
  loops, treating nonzero dimensionless ints as true.
- Added the legacy Janus fixed-point `*/` expression as signed Q31 multiply,
  enabling the old Schrödinger wave-equation example to parse and check.
- Added Janus-style optional control-flow clauses, parsing omitted `then`,
  `else`, `do`, and `loop` clauses as `skip`.
- Added Janus-style `procedure` keyword support and parenless parameterless
  `call`/`uncall` syntax.
- Added Janus-style `size(xs)` expressions for fixed arrays, with type checks,
  interpreter support, documentation, and a runnable forward/backward example.
- Added Janus-style integer stacks with `nil`, `push`, `pop`, `empty`, `top`,
  stack-aware `size`, type checks, interpreter support, documentation, and a
  runnable forward/backward example.
- Added Janus/Jana-style type-first procedure parameters and reversible
  `local`/`delocal` declarations, including empty `[]` array suffixes such as
  `int fact[]`.
- Added Janus/Jana-style block comments, unbraced procedure bodies, and
  semicolonless statement streams.
- Added Janus/Jana-style source declarations in parameterless `main()` for
  scalar ints, one-dimensional int arrays, and stacks, including `{...}` array
  initializers.
- Added Jana-style `show(x)` observations, kept separate from reversible
  `read`/`write` tapes.
- Added Jana-style `printf(...)` observations for `%d` integer placeholders,
  enabling direct execution of upstream Jana Fibonacci and square-root
  examples.
- Added procedure arguments as lvalues for variables and one-dimensional array
  elements, so calls such as `call f(xs[0])` mutate the referenced element.
- Added Jana-style `iterate int i = start by step to end ... end` bounded
  loops with inclusive bounds and reversible inversion.
- Added multidimensional array declarations/indexing plus cell-sensitive array
  update alias checks, enabling direct execution of Jana `matrixmult_v1.0.ja`.
- Added legacy Janus bare top-level global declarations before procedures,
  covering old examples that start with declarations such as `n x1 x2` or
  `psiR[128]`.
- Added the legacy `main_fwd` default entry convention for Janus files that
  omit `main`.
- Added legacy demo `test1` / `test` fallback entry conventions for older
  procedure-only Janus examples.
- Added checked external benchmark coverage for direct upstream Jana
  factorization, permutation-to-code, and run-length-encoding examples.
- Added `examples/janus_stack_reverse.rev`, a recursive stack-reversal corpus
  example inspired by Jana's stack-operations sample.
- Added modern-Jana Schrödinger-style wave-kernel fixtures to the external
  benchmark harness in both forward and reverse directions.
