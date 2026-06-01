# Performance

Performance is now a measured track, but it does not replace the soundness
oracles. A fast reversible runtime is only interesting after it preserves the
same forward/backward semantics as the reference tree-walking interpreter.

## Engines

Reverie currently has two execution engines:

- `tree`: the original AST interpreter. It is the simplest semantic reference.
- `slot`: a compiled interpreter that lowers variable names to numeric slots
  before execution. This removes repeated ordered-map lookup in hot loops while
  preserving the same `State` and diagnostics surface.

The CLI defaults `run` and `reverse` to the slot engine:

```sh
cargo run -p reverie-cli -- run examples/fib.rev \
  --var n=7 --var i=0 --var a=0 --var b=1
```

Use `--engine tree` when you want the reference interpreter explicitly:

```sh
cargo run -p reverie-cli -- run examples/fib.rev \
  --engine tree \
  --var n=7 --var i=0 --var a=0 --var b=1
```

The scrubber still uses the tree-walking timeline builder because it records
source-level frames, not just the final state.

## Internal Benchmarks

Run the Criterion benchmark suite:

```sh
cargo bench -p reverie-interp --bench execution
```

The suite currently contains these documented Criterion groups:

```text
fib_pair_transform_n1000
janus_style_sort_n50_reverse_order
janus_style_sort_n50_reverse_order_reverse
rle_compression_n8
rle_compression_n8_reverse
rle_compression_n8_roundtrip
fixed_point_wave_n16_t100
procedure_call_uncall_n1000
constant_element_call_uncall_n1000
janus_turing_binary_inc
janus_stack_reverse_n5
matrix_accumulate_3x3
matrix_transpose_3x3
matrix_transpose_3x3_reverse
tape_io_read_write
slot_compile_vs_execute_sort_n50
scrub_timeline_sort_n50
```

CI compiles this benchmark target with `cargo bench -p reverie-interp --bench
execution --no-run` and runs `scripts/check_criterion_docs.py`, so every
Criterion group in `crates/reverie-interp/benches/execution.rs` must stay
listed here. The checker also verifies that normal execution groups keep both
`tree_walk` and `slot_compiled` measurements, with explicit exceptions for the
compile-time and scrub-timeline groups, and that every benchmark wrapper is
registered in the `criterion_group!` macro so documented groups actually run.

The Fibonacci benchmark compares the reference tree walker against the slot
engine on the Fibonacci pair transform with `n = 1000`. This isolates execution from
CLI startup and gives us a stable place to add more workloads.

The forward sort benchmark runs `examples/janus_sort.rev` on a reverse-ordered
50-element array. This is the first Janus-corpus performance workload and is
more representative than Fibonacci because it uses nested loops, indexed array
reads, indexed swaps, and procedure calls.

The reverse sort benchmark runs the inverse of the same 50-element sort
workload from the sorted final state back to reverse order. It keeps reverse execution visible
inside the fast internal suite, not only in the external Jana comparison.

The RLE forward benchmark runs `examples/rle_compression.rev` on an eight-element
input. It tracks a compression-shaped reversible workload with witness data,
indexed array reads, and divergent update paths for run starts versus repeated
values.

The RLE reverse benchmark runs the inverse of the same run-length compression
workload from the compact run table back to zeroed output arrays. It keeps
reverse execution visible for witness-backed compression, not only for sorting
and matrix transposition.

The RLE round-trip benchmark runs run-length compression followed by its
inverse in one sample. It checks the full reversible path for the witness-backed compression
example inside Criterion, matching the external benchmark corpus's round-trip
coverage.

The procedure benchmark runs a synthetic 1,000-iteration `call`/`uncall` loop.
It isolates by-reference procedure call overhead from array work and CLI
startup, which keeps procedure performance visible as the language grows.

The constant element call benchmark runs the same call/uncall shape through
constant indexed arguments such as `xs[0]` and `xs[1]`. It keeps the
slot-compiled fast path for statically proven distinct element arguments
visible in the internal suite.

The compile-vs-execute benchmark compares slot compilation cost with repeated
execution of the compiled sort workload. That keeps compile-time overhead visible alongside
the runtime speedups from slot lowering.

The scrub-timeline benchmark builds the full scrub timeline for the same
50-element sort workload. The scrubber intentionally uses the tree-walking timeline builder so
it can preserve source-level frames and procedure-local state.

The Turing-machine benchmark runs `examples/janus_turing.rev`, a Janus-style
Turing-machine simulation with source globals, nested reversible conditionals,
array-backed rule tables, and call/uncall tape movement.

The stack benchmark runs `examples/janus_stack_reverse.rev`, a recursive
stack-reversal program modeled on Jana's stack example. It keeps stack
push/pop, `top`/`empty`, recursive procedures, and stack parameters visible in
the internal performance suite.

The matrix accumulation benchmark runs a compact 3x3 matrix accumulation
kernel. It keeps multidimensional arrays, nested Janus-style `iterate` loops, and array-element
procedure arguments visible in the internal suite instead of relying only on
the external Jana matrix workload.

The matrix transpose benchmarks run `examples/matrix_transpose.rev` forward and
backward. It keeps multidimensional array swaps and the mechanically inverted
transpose path visible in Criterion, matching the dedicated external
Jana-vs-Reverie transpose workloads.

The tape I/O benchmark runs `examples/io.rev` through reversible tape
`read`/`write` execution. It keeps `IoState` input/output tape handling visible
in the internal suite for both the tree walker and slot-compiled engine.

The fixed-point wave benchmark runs a compact fixed-point wave-style kernel
with the legacy Janus `*/` operator over 16-element arrays for 100 steps. The external
Jana binary used by the checked smoke gate no longer parses the historic
fixed-point wave source, so this internal group keeps Reverie's Q31
fixed-multiply execution path visible in Criterion.

In a short local Criterion sample, `janus_turing_binary_inc` ran at roughly
`85 µs` in the tree walker and `30 µs` in the slot-compiled engine.

The external benchmark harness also includes Jana's `matrixmult_v1.0.ja`
directly, exercising multidimensional arrays, nested `iterate` loops, and
array-element procedure arguments. It also includes a dedicated 3x3 matrix
transpose fixture that uses nested `iterate` loops, multidimensional indexing,
and reversible element swaps in forward, reverse, and round-trip directions.

Indexed reads and swaps clone only the touched array cell in the interpreter
hot path. Whole-array values still clone at public `State` boundaries and when
a program explicitly reads an entire array value.

Useful future Criterion groups:

- larger timeline workloads with long-distance scrub jumps

## End-To-End CLI Timing

Use `hyperfine` for whole-command timing:

```sh
cargo build --release -p reverie-cli

hyperfine --warmup 5 --runs 30 \
  './target/release/reverie run examples/fib.rev --engine tree --var n=1000 --var i=0 --var a=0 --var b=1' \
  './target/release/reverie run examples/fib.rev --engine slot --var n=1000 --var i=0 --var a=0 --var b=1'
```

This includes process startup, parsing, checking, compilation, and execution.
It is useful for user-visible CLI performance, but Criterion is better for
runtime-engine decisions.

## Benchmarking Against Janus

The honest comparison target must name an implementation. Janus as a language
is a semantic reference; an interpreter, compiler, or generated C++ pipeline is
a performance target.

The first external target is
[`kirkedal/Jana-JanusInterp`](https://github.com/kirkedal/Jana-JanusInterp), a
public Haskell interpreter for Janus. Run the standard checked smoke
performance gate with:

```sh
scripts/verify_janus_performance.py
```

That command builds `target/release/reverie`, verifies every configured
workload, requires every selected workload to beat Jana by at least 1.25x by
median wall-clock time, requires the observed workload minimum to stay at or
above 2.0x, requires the suite median and geometric mean to stay at or above
4.0x, and writes
`benchmarks/results/jana-vs-reverie-smoke.json` and
`benchmarks/results/jana-vs-reverie-smoke.md`.

CI also exposes this as a manual `janus-performance` job in the CI workflow.
The job accepts run count, warmup count, per-workload minimum speedup, observed
minimum speedup, suite median speedup, geometric-mean speedup, and command
timeout inputs, then uploads the JSON benchmark result and Markdown summary as
an artifact.

For custom benchmark runs, use the lower-level harness directly:

```sh
scripts/bench_jana_vs_reverie.py --runs 20 --warmup 3
```

The script clones Jana into `target/jana-baseline` if needed, builds
`target/release/reverie`, verifies that every benchmark produces the expected
state/output, and then reports median/mean/min wall-clock timings. It uses
Jana `-m64` for the shared Fibonacci workload so both sides run with 64-bit
modular arithmetic. Jana does not expose a direct "run this source backward
from this seeded final store" command, so reverse workloads are explicit Jana
fixtures that seed the final state and use `uncall` or inverted statements.
Reverie uses its native `reverse` subcommand for the same stores. Each command
has a 30-second timeout by default; use `--command-timeout SECONDS` on slower
machines or when adding larger workloads.

For reproducible reports, write the checked timing samples to JSON:

```sh
scripts/bench_jana_vs_reverie.py --runs 20 --warmup 3 \
  --json-output benchmarks/results/jana-vs-reverie.json
```

The JSON artifact records the selected commands, every measured sample,
median/mean/min timings, and the median-based speedup for each workload. It
also records host metadata, the Jana checkout revision, the Reverie git status,
binary file metadata, the command timeout, and the checked wrapper's aggregate
`performance_gate` thresholds so benchmark claims can be traced back to the
exact inputs and gates used for the run. New artifacts record SHA-256 digests
for the Jana and Reverie binaries in addition to path, size, and modification
time, and record SHA-256 digests for each `.ja`, `.janus`, or `.rev` source
fixture referenced by a workload side. The checker can require those recorded
thresholds with `--expect-performance-gate`. Each workload also records
explicit direction metadata (`forward`, `reverse`, or `roundtrip`) for coverage
checks and summaries. Forward-then-reverse round-trip workloads record multiple
checked commands per side under `commands` and time the sequence as one sample.
The artifact also records the exact `selected_workloads` list used by the
harness, and the checker requires that list to match the benchmark row order.

For a human-readable table generated directly from a JSON artifact, run:

```sh
python3 scripts/summarize_janus_performance.py \
  benchmarks/results/jana-vs-reverie-smoke.json \
  --output benchmarks/results/jana-vs-reverie-smoke.md
```

`scripts/check_benchmark_artifact.py --expect-markdown-summary` verifies that
the saved Markdown report exactly matches the saved JSON artifact, so CI and
manual gates cannot publish a stale benchmark table. The summary reports
range, median, and geometric-mean speedup so one extreme workload cannot be the
only suite-level performance signal. The same checker also
requires the artifact to use the expected schema fields, requires parseable
timestamp, host, source, and binary metadata, requires host CPU-count metadata,
requires the recorded working directory to match the repository root, requires
absolute source and binary paths, verifies that nested binary metadata paths
match the top-level artifact paths, validates binary SHA-256 digests when
present, and can pin the expected Jana and Reverie
binary path suffixes. It can also require workload rows to
stay in the exact harness and manifest order, which keeps JSON artifacts and
Markdown summaries stable for review. Per-workload command records are also
checked so Jana commands start with the Jana binary and Reverie commands start
with the Reverie binary, and benchmark source-file arguments use absolute
paths. Each command also records the expected stdout fragments used by the
harness's semantic verification step. When a side records `source_files`, the
checker validates that those paths match the command source-file arguments and
that each SHA-256 digest is well-formed; pass `--expect-source-digests` for new
artifacts where fixture fingerprints are mandatory. For freshly generated
local artifacts, `--verify-file-digests` also recomputes recorded binary and
source-file hashes from disk. The standard `scripts/verify_janus_performance.py`
gate enables both checks. Single-command sides must keep `command` equal to
`commands[0]`, while multi-command round trips must omit `command` and rely on
the ordered `commands` list. When `min_speedup` is present, each workload must
record a boolean `passes_min_speedup`; ungated artifacts must leave that field
null. Source metadata must include a `revision` field, using null only when a
revision genuinely is not available.

To enforce a performance gate, add `--min-speedup`. For example, this command
fails if any selected checked workload is not at least 1.25x faster than Jana
by median wall-clock time:

```sh
scripts/bench_jana_vs_reverie.py --runs 20 --warmup 3 \
  --json-output benchmarks/results/jana-vs-reverie.json \
  --min-speedup 1.25
```

Use the checked wrapper when you also want aggregate guardrails:
`--min-observed-speedup` rejects regressions in the slowest workload, and
`--min-median-speedup` and `--min-geomean-speedup` reject suite-wide drift even
when every individual row still clears the lower per-workload floor.

Current external workloads:

- `jana_fib_recursive_direct`: unmodified upstream Jana `examples/fib.ja` run
  by both Jana and Reverie.
- `jana_sqrt_direct`: unmodified upstream Jana `examples/sqrt.ja` run by both
  Jana and Reverie.
- `jana_stack_operations_direct`: unmodified upstream Jana
  `examples/stack-operations.ja` run by both Jana and Reverie.
- `janus_stack_reverse_cleanup`: recursive stack reversal followed by matching
  reverse cleanup, using Jana and Reverie fixtures with the same call/uncall
  shape.
- `jana_matrixmult_v1_direct`: unmodified upstream Jana
  `examples/matrixmult_v1.0.ja` run by both Jana and Reverie.
- `jana_matrixmult_v1_reverse`: the matrix multiplication final store back to
  zeroed arrays and scalars, using a Jana reverse fixture and Reverie's native
  `reverse` command.
- `jana_matrixmult_v1_roundtrip`: matrix multiplication forward execution
  followed by the reverse fixture, timed as one checked workload on each side.
- `matrix_transpose_3x3`: nested-iterate 3x3 matrix transpose with
  multidimensional array swaps, run by both Jana and Reverie.
- `matrix_transpose_3x3_reverse`: transposed matrix final state back to a
  zeroed store, using a Jana reverse fixture and Reverie's native `reverse`
  command.
- `matrix_transpose_3x3_roundtrip`: matrix transpose followed by reverse
  cleanup, timed as one checked workload on each side.
- `jana_factor_direct`: unmodified upstream Jana `examples/factor.ja` run by
  both Jana and Reverie.
- `jana_perm_to_code_direct`: unmodified upstream Jana
  `examples/perm-to-code.ja` run by both Jana and Reverie.
- `janus_perm_to_code_reverse`: factorial-code style array back to the
  original permutation.
- `janus_perm_to_code_roundtrip`: permutation-to-code followed by reverse
  conversion, timed as one checked workload on each side.
- `jana_run_length_enc_direct`: unmodified upstream Jana
  `examples/run-length-enc.ja` run by both Jana and Reverie.
- `jana_run_length_enc_stack_direct`: unmodified upstream Jana
  `examples/run-length-enc-stack.ja` run by both Jana and Reverie.
- `rle_compression_n8`: witness-preserving run-length compression of an
  eight-element array.
- `rle_compression_n8_reverse`: compressed run table back to zeroed output
  tables while preserving the source witness array.
- `rle_compression_n8_roundtrip`: run-length compression followed by reverse
  cleanup, timed as one checked workload on each side.
- `bit_reversal_n8`: table-driven eight-element bit-reversal permutation.
- `bit_reversal_n8_reverse`: the same permutation from final array state back
  to the original array.
- `bit_reversal_n8_roundtrip`: bit reversal followed by reverse bit reversal,
  timed as one checked workload on each side.
- `fib_loop_n1000`: equivalent loop-based Fibonacci pair transform.
- `fib_loop_n1000_reverse`: final Fibonacci pair state back to the initial
  pair state.
- `fib_loop_n1000_roundtrip`: forward Fibonacci followed by the reverse
  fixture, timed as one checked workload on each side.
- `procedure_call_n1000`: 1,000 iterations of a by-reference procedure call
  followed by its matching uncall.
- `procedure_call_n1000_reverse`: final procedure-loop counter state back to
  the initial state.
- `procedure_call_n1000_roundtrip`: forward procedure loop followed by the
  reverse fixture.
- `janus_root_66`: Jana's square-root example against
  `examples/janus_root.rev`.
- `janus_root_66_reverse`: square-root final state back to `num = 66`.
- `janus_factor_840`: Jana's factorization example against
  `examples/janus_factor.rev`.
- `janus_factor_840_reverse`: factor table final state back to `num = 840`.
- `janus_sort_n50_reverse_order`: modern-Jana reversible bubble sort against
  `examples/janus_sort.rev`, both on a 50-element reverse-ordered array.
- `janus_sort_n50_reverse_order_reverse`: sorted list plus permutation witness
  back to the reverse-ordered input and zeroed witness array.
- `janus_sort_n50_reverse_order_roundtrip`: sort followed by reverse sort,
  timed as one checked workload on each side.
- `janus_schroedinger_n16_t100`: a modern-Jana integer wave-stencil fixture
  inspired by Jana's legacy Schrödinger example, with 16 cells and 100 steps.
- `janus_schroedinger_n16_t100_reverse`: the same wave-stencil state transform
  from final arrays back to the original wave state.
- `janus_schroedinger_n16_t100_roundtrip`: wave-stencil forward fixture
  followed by the reverse fixture, timed as one checked workload on each side.
- `janus_turing_binary_inc`: a modern-Jana fixture for the historic reversible
  Turing-machine binary-increment simulation, using explicit rule-selection
  guards so it satisfies strict reversible branch assertions.
- `janus_turing_binary_inc_reverse`: the Turing-machine halting state back to
  the zero setup store.
- `janus_turing_binary_inc_roundtrip`: Turing-machine forward simulation
  followed by reverse cleanup, timed as one checked workload on each side.

The latest checked local smoke result is written to
`benchmarks/results/jana-vs-reverie-smoke.json` and rendered as
`benchmarks/results/jana-vs-reverie-smoke.md`. The Markdown report is generated
from the JSON artifact and checked for exact consistency, so use it instead of
copying benchmark tables by hand.

Treat those numbers as a local smoke result, not a publication-grade claim.
Use a higher run count, a pinned Jana revision, and a quiet machine before
quoting them.

Rules for a fair run:

- Use equivalent reversible programs and the same initial state.
- Check that final states match before timing.
- Separate parse/check/startup timing from execution timing when possible.
- Report forward, reverse, and forward-then-reverse timings.
- Keep soundness tests green before trusting any performance number.

End-to-end shape:

```sh
hyperfine --warmup 5 --runs 30 \
  'JANUS_COMMAND_FOR_EQUIVALENT_FIB' \
  './target/release/reverie run examples/fib.rev --engine slot --var n=1000 --var i=0 --var a=0 --var b=1'
```

The next performance milestone is a native-output comparison if a Janus-to-C/C++
path is available and reproducible, plus a broader feature audit across the
remaining Jana examples.
