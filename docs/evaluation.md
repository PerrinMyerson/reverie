# Evaluation

Reverie is not evaluated like a performance data structure. The v1 interpreter
does not need to be fast; it needs to be correct, deterministic, and honest
about the programs it refuses to run.

## Soundness Oracles

The central evaluation tool is property-based testing. The three important
metamorphic properties are:

```text
execute(P, S) = S'
execute_backward(P, S') = S
```

```text
invert(invert(P)) == P
```

```text
execute(uncall f(args), S) == execute(invert(body_of_f), S)
```

The first property is the master round-trip law. The second catches asymmetric
inversion bugs. The third checks that procedure machinery has not drifted away
from the meaning of the inverted procedure body.

Current `proptest` coverage includes:

- generated straight-line scalar programs
- generated fixed-size integer and boolean array programs
- generated dynamically valid conditionals
- generated finite Janus-style loops
- generated local array programs with computed `delocal` assertions
- generated procedure calls
- generated `uncall` fidelity checks against inverted procedure bodies
- generated ASTs for `invert(invert(P)) == P`

The quality metric is generator reach: more valid generated programs, deeper
ASTs, and richer state shapes increase confidence.

## Negative Corpus

A reversible language is partly defined by what it rejects. The negative
examples intentionally cover static and runtime failures:

```text
examples/irreversible_update.rev
examples/alias_rejection.rev
examples/local_shadow_rejection.rev
examples/unit_mismatch.rev
examples/top_level_unit_mismatch.rev
examples/static_array_bounds.rev
examples/assert_failure.rev
examples/if_assertion_failure.rev
examples/loop_assertion_failure.rev
examples/delocal_assertion_failure.rev
examples/naive_max_no_witness.rev
examples/proc_runtime_error.rev
examples/array_oob.rev
examples/refinement_violation.rev
examples/refinement_units_loop_violation.rev
```

These tests establish that Reverie rejects information destruction, catches
lying branch/loop/local assertions, reports span-aware runtime failures, and
does not silently accept witness-free reversible-looking programs.

## Expressiveness Corpus

The positive examples are the current capability corpus:

```text
examples/skip.rev
examples/fib.rev
examples/injectivized_max.rev
examples/assert.rev
examples/bool_toggle.rev
examples/bool_flags.rev
examples/wrapping.rev
examples/negation.rev
examples/janus_operators.rev
examples/janus_optional_control.rev
examples/janus_procedure_syntax.rev
examples/proc.rev
examples/element_args.rev
examples/increment.rev
examples/globals.rev
examples/io.rev
examples/array.rev
examples/size.rev
examples/stack.rev
examples/janus_stack_reverse.rev
examples/janus_sort.rev
examples/tensor_linear.rev
examples/invertible_coupling.rev
examples/triangular_residual.rev
examples/reversible_preprocess.rev
examples/reversible_inference_trace.rev
examples/mnist_identify.rev
examples/mnist_reversible_step.rev
examples/mnist_witness_tape.rev
examples/mnist_witness_tape_loop.rev
examples/mnist_mlp_witness.rev
examples/bit_reversal.rev
examples/perm_to_code.rev
examples/janus_automata.rev
examples/janus_turing.rev
examples/rle_compression.rev
examples/procedure_call_loop.rev
examples/matrix_transpose.rev
examples/janus_root.rev
examples/janus_factor.rev
examples/array_units.rev
examples/units.rev
examples/units_exponents.rev
examples/top_level_units.rev
examples/refinement.rev
examples/refinement_units_loop.rev
```

Every `.rev` example must be mentioned in `examples/README.md` and covered by
the CLI tests; the `every_example_is_documented_and_exercised` test and
`scripts/check_evaluation_corpus.py` both enforce that. The same script also
checks the evaluation guide's positive and negative corpus lists, requiring
every `.rev` example to be classified exactly once. It also protects ergonomic
syntax coverage in the examples and user-facing docs for readable aliases such
as `assert_eq`, `assert_ne`, `len(xs)`, `is_empty(s)`, `peek(s)`,
`swap(...)`, and prefix/postfix `++`/`--`.

Future expressiveness targets should include a broader language-feature audit
and more side-by-side Janus translations. The checked example corpus now
includes statically proven distinct element arguments, prefix/postfix update
sugar, tensor accumulation, reversible and witness-taped MNIST-shaped kernels,
a small MLP witness pattern, invertible additive coupling and triangular
residual layers, a reversible Q31 preprocessing block, a deterministic
reversible inference trace, bit-reversal
permutation, permutation-to-factorial-code conversion, run-length encoding with
an explicit witness array, a historic Janus-style cellular automata
translation, and a strict reversible Turing-machine translation; the external
benchmark corpus includes compact Janus-style Schrodinger, Turing-machine, and
multidimensional matrix-transpose fixtures in forward, reverse, and round-trip
directions.

## Demo Evaluation

The scrubber is qualitative evidence. It should make reversibility visible:
the user can move through the same deterministic timeline in both directions
and see values flow back to the start. The README GIF is generated from the
real `fib.rev` scrubber dump, so the demo artifact stays tied to interpreter
behavior.

## Commands

Run the full soundness gate:

```sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Regenerate the demo artifact:

```sh
python3 scripts/render_scrubber_gif.py
```

Performance benchmarking is intentionally secondary to soundness, but Reverie
now has a measured fast path: the slot-compiled engine is checked against the
tree-walking semantic reference and benchmarked with Criterion. See
`docs/performance.md`. External performance checks also compare Reverie against
the named Jana Janus interpreter with output verification before timing. CI
validates that the checked benchmark artifact matches the current harness,
that each configured workload is named in the benchmark documentation, compiles
the internal Criterion benchmark target without running the benchmark suite,
checks that Criterion groups still include their expected measurements, checks
that benchmark wrappers are registered with Criterion,
that the machine-readable `explain --json` schema stays parseable, including
the `dataset_loops`, `safety_checks`, and `safety_check_counts` fields over
indexed-array and dataset-loop fixtures. The schema checker also has a
build-independent `--self-test` mode for its own negative fixtures, including
missing, extra, non-integer, and non-positive safety-check counts plus malformed
dataset-loop metadata, and CI runs
`scripts/check_ci_workflow.py` so these CI gates cannot silently disappear.
The Jana
compatibility inventory can also be validated with
`scripts/check_janus_feature_audit.py`, which checks the feature-audit matrix,
and `scripts/check_jana_audit.py`, which enforces coverage floors while keeping
known upstream-source exceptions explicit.
