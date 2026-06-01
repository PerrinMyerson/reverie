# V1 Acceptance Audit

This document maps the original phase plan to the current repository. It is a
working acceptance checklist for the v1 interpreter, not a promise that stretch
research items are complete.

## Verification Gates

Run these from the repository root:

```sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The README demo GIF is generated separately because it depends on macOS
rendering tools:

```sh
python3 scripts/render_scrubber_gif.py
```

The generated artifact is `docs/assets/reverie-scrub-demo.gif`.

## Phase Checklist

| Phase | Requirement | Evidence |
| --- | --- | --- |
| 0 | Rust workspace with five crates | `Cargo.toml`, `crates/reverie-*` |
| 0 | CI runs format, clippy, and tests | `.github/workflows/ci.yml` |
| 0 | Parse and run `skip` | `examples/skip.rev`, CLI tests |
| 1 | Forward reversible updates, swaps, sequences, conditionals, and loops | parser/core/interpreter tests plus `examples/fib.rev` |
| 1 | Signed `i64` wrapping integer semantics | `examples/wrapping.rev`, interpreter tests |
| 1 | Janus-style unary numeric negation | `examples/negation.rev`, parser/core/interpreter tests |
| 1 | Janus-style operator aliases | `examples/janus_operators.rev`, parser and CLI tests |
| 1 | Janus-style optional control-flow clauses | `examples/janus_optional_control.rev`, parser and CLI tests |
| 1 | Externally seeded variables | CLI `--var`, Fibonacci and array CLI tests |
| 1 | Fibonacci pair-transform target | `examples/fib.rev` forward/reverse CLI tests |
| 1 | Injectivization teaching example | `examples/injectivized_max.rev`, `docs/injectivization.md` |
| 2 | Pure `invert()` for every implemented construct | `reverie-core`, formatter and inversion tests |
| 2 | Backward interpreter | `reverie reverse`, interpreter tests |
| 2 | Reversibility oracle | proptest round-trip tests in `reverie-interp` |
| 3 | Procedures with `call` and `uncall` | `examples/proc.rev`, CLI and interpreter tests |
| 3 | Janus-style procedure keyword and parenless calls | `examples/janus_procedure_syntax.rev`, parser and CLI tests |
| 3 | By-reference parameter passing | procedure runtime and round-trip tests |
| 3 | Reversible `local`/`delocal` | `examples/proc.rev`, local array oracle tests |
| 4 | Units of measure | `docs/units.md`, `examples/units.rev` |
| 4 | Dimensional algebra with exponents | `examples/units_exponents.rev` |
| 4 | Unit mismatch diagnostics | `examples/unit_mismatch.rev`, CLI tests |
| 5 | Refinement annotations | `docs/refinements.md`, `examples/refinement.rev` |
| 5 | Runtime refinement enforcement | refinement violation examples and CLI tests |
| 5 | Refinements compose with units and loops | `examples/refinement_units_loop.rev` |
| 6 | Interactive scrubber TUI | `reverie scrub`, `reverie-tui` |
| 6 | Step/rewind/jump/watch UI | `docs/scrubber.md`, TUI implementation |
| 6 | Non-interactive demo and tests | `scrub --dump`, CLI/TUI tests |
| 6 | README GIF | `docs/assets/reverie-scrub-demo.gif` |
| docs | Full language syntax reference | `docs/grammar.md` |
| docs | CLI usage reference | `docs/cli.md` |
| docs | Evaluation methodology | `docs/evaluation.md` |
| docs | Janus compatibility audit | `docs/janus-audit.md`, `scripts/check_janus_feature_audit.py` |
| docs | Example coverage stays synchronized | `every_example_is_documented_and_exercised` CLI test and `scripts/check_evaluation_corpus.py` |

## Extra V1 Fill-Out

Arrays are implemented beyond the original early-phase minimum:

- fixed-size array literals and values
- indexed reads
- Janus-style array length reads with `size(xs)`
- reversible indexed `+=`, `-=`, and `^=`
- indexed swaps
- `array<int>`, `array<bool>`, and unitful `array<int<m>>` annotations
- CLI seeding with values such as `--var 'xs=[10,20,30]'`

Stacks are implemented as another Janus compatibility fill-out:

- `stack` type annotations and `nil`
- reversible `push`/`pop` transfer statements
- `empty`, `top`, and stack-aware `size`
- CLI seeding with values such as `--var s=nil`
- executable `examples/stack.rev`

Diagnostics also go beyond the minimum:

- syntax, semantic, type, unit, and runtime diagnostics render through
  `ariadne`
- two-sided type/unit mismatch labels are shown when both spans exist
- runtime procedure errors show both the failing procedure body expression and
  the call site
- array bounds, failed assertions, failed delocals, division by zero, and
  refinement failures carry source spans

The CLI test suite includes a coverage check that every `.rev` file under
`examples/` is mentioned in both `examples/README.md` and the CLI tests. That
keeps the tutorial catalog and executable regression coverage from drifting
apart.

## V1 Semantic Decisions

The following decisions are intentional v1 boundaries:

- `int` is signed `i64`.
- Arithmetic updates use two's-complement wrapping.
- Top-level variables are supplied externally with `--var`; language-level
  creation uses `local`/`delocal`.
- Procedures use by-reference parameters without duplicate argument aliasing.
- Locals use a no-shadow model over the flat runtime store.
- Units are static annotations and are erased before interpretation.
- Refinements are boolean predicates checked at runtime.
- The scrubber records a deterministic forward timeline; semantic backward
  execution still uses the inverse program through `reverie reverse`.

## V1 Non-Goals

These are not required for the finished v1 described by phases 0-6:

- direct terminal I/O; reversible tape I/O is supported instead
- dynamic heap allocation
- a bytecode VM
- snapshot/replay acceleration for distant jumps
- SMT-backed static refinement proving
- LLVM, Cranelift, or other native code generation
- self-hosting

The stretch list in `docs/roadmap.md` preserves those ideas for later work.
