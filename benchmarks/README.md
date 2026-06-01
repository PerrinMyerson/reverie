# Benchmarks

This directory contains cross-implementation benchmark inputs. Internal engine
benchmarks live in `crates/reverie-interp/benches`.

## Jana / Janus Baseline

`scripts/bench_jana_vs_reverie.py` compares Reverie against the public
`kirkedal/Jana-JanusInterp` implementation:

```sh
scripts/bench_jana_vs_reverie.py --runs 20 --warmup 3
```

For the standard checked smoke gate used while developing Reverie, run:

```sh
scripts/verify_janus_performance.py
```

That wrapper builds `target/release/reverie`, runs the full checked benchmark
corpus with `--runs 5 --warmup 1 --min-speedup 1.25
--min-observed-speedup 2.0 --min-median-speedup 4.0
--min-geomean-speedup 4.0`, and writes
`benchmarks/results/jana-vs-reverie-smoke.json` plus the Markdown summary
`benchmarks/results/jana-vs-reverie-smoke.md`. It also validates the JSON
artifact with `scripts/check_benchmark_artifact.py`, requiring the full smoke
workload count, positive samples, consistent speedup math, unique workload
names, an exact Markdown summary match, the exact expected workload names reported by
`scripts/bench_jana_vs_reverie.py --list-workloads`, and
forward/reverse/round-trip coverage. The same artifact check enforces a
minimum observed workload speedup, a minimum suite median speedup, and a
minimum suite geometric-mean speedup so a run cannot pass on corpus shape
alone. The generated Markdown summary reports the same aggregate floors
alongside the observed speedup range, median, and geometric mean, and the JSON
artifact records those floors under `performance_gate` so reports can be
re-rendered without repeating the wrapper CLI flags. The wrapper also
recomputes recorded binary and source fixture SHA-256 digests from the local
filesystem, so a freshly generated report cannot pass with stale or forged
provenance.
Focused `--only` runs validate just the selected workload names so the same
wrapper remains useful during benchmark development. The lower-level harness
also rejects duplicate or unknown `--only` values and preserves the explicit
selection order in the JSON artifact.

The same gate is available as a manual GitHub Actions run. In the Actions tab,
choose the CI workflow and run it with the default inputs, or use:

```sh
gh workflow run ci.yml \
  -f runs=5 \
  -f warmup=1 \
  -f min-speedup=1.25 \
  -f min-observed-speedup=2.0 \
  -f min-median-speedup=4.0 \
  -f min-geomean-speedup=4.0 \
  -f command-timeout=30
```

The `janus-performance` job uploads the JSON result and Markdown summary as the
`jana-vs-reverie-smoke` artifact.

To audit upstream Jana example compatibility without timing anything, run:

```sh
python3 scripts/audit_jana_examples.py \
  --json-output benchmarks/results/jana-example-audit.json
```

The audit compares Jana, Reverie default parsing, and Reverie
`--legacy-janus` parsing across `target/jana-baseline/examples` and
`target/jana-baseline/basicExamples`. The JSON artifact records per-example
statuses, first diagnostic lines, and line counts so compatibility gaps can be
triaged separately from benchmark results. Jana runs that exit 0 while printing
local/delocal diagnostics are reported as `ok_with_diagnostics`.
Use `scripts/check_jana_audit.py` on the generated JSON to enforce compatibility
floors, artifact metadata such as timeout and binary paths, and known
exceptions, such as the strict-branch failure for upstream
`basicExamples/turing.janus`. The checked manifest
`benchmarks/jana-example-corpus.txt` records the exact upstream file set covered
by the saved audit artifact.

The script clones Jana into `target/jana-baseline` if needed, verifies each
program's expected output, and then reports end-to-end timings. It uses
`target/release/reverie`, building it first unless `--no-build` is passed.
List the configured workload corpus without cloning Jana or building Reverie
with:

```sh
python3 scripts/bench_jana_vs_reverie.py --list-workloads
```

Pass `--json-output benchmarks/results/jana-vs-reverie.json` to preserve the
checked commands, measured samples, summaries, and speedups as a
machine-readable benchmark artifact. The artifact also includes host platform
and CPU-count details, the repository-root working directory used for command
execution, Jana/Reverie source metadata, binary metadata, and the expected
stdout fragments used for semantic verification so results can be traced to the
exact benchmark inputs. New artifacts include each benchmark binary's SHA-256
digest in addition to its path, size, and modification time. They also record
SHA-256 digests for the `.ja`, `.janus`, and `.rev` source files referenced by
each workload side. The top-level `selected_workloads` list records the
exact harness selection set and must match benchmark row order. Each workload
records explicit direction metadata (`forward`, `reverse`, or `roundtrip`) so
validators and reports do not need to infer coverage from timing samples. Some workloads contain multiple checked
commands per side so forward-then-reverse round trips are timed as one workload.
Each command has a 30-second timeout by default; pass `--command-timeout
SECONDS` when a larger workload needs more time.
Pass `--min-speedup 1.25` when a run should fail unless every selected workload
is at least 1.25x faster than Jana by median timing.

Render any JSON result as a human-readable report with:

```sh
python3 scripts/summarize_janus_performance.py \
  benchmarks/results/jana-vs-reverie-smoke.json \
  --output benchmarks/results/jana-vs-reverie-smoke.md
```

Validate a JSON result without rerunning timings with:

```sh
python3 scripts/check_benchmark_artifact.py \
  benchmarks/results/jana-vs-reverie-smoke.json \
  --min-workloads 41 \
  --require-direction forward \
  --require-direction reverse \
  --require-direction roundtrip \
  --expect-current-workloads \
  --exact-workloads-file benchmarks/jana-workload-corpus.txt \
  --expect-workload-order \
  --expect-runs 5 \
  --expect-warmup 1 \
  --expect-command-timeout 30.0 \
  --expect-min-speedup 1.25 \
  --expect-direction-count forward:19 \
  --expect-direction-count reverse:12 \
  --expect-direction-count roundtrip:10 \
  --min-observed-speedup 2.0 \
  --min-median-speedup 4.0 \
  --min-geomean-speedup 4.0 \
  --expect-performance-gate \
  --expect-source-digests \
  --expect-jana-bin-suffix target/jana-baseline/bin/janus \
  --expect-reverie-bin-suffix target/release/reverie \
  --expect-markdown-summary benchmarks/results/jana-vs-reverie-smoke.md
```

The smoke wrapper passes `--exact-workloads` with its pinned workload list so
full gate runs also fail on unexpected benchmark corpus drift. The checked
manifest `benchmarks/jana-workload-corpus.txt` records the exact workload set
and order covered by the saved smoke artifact.
The checker enforces the recorded `performance_gate` aggregate floors even if
`--min-observed-speedup`, `--min-median-speedup`, and
`--min-geomean-speedup` are not repeated on the command line. Passing explicit
values is still useful in CI because it also verifies that the artifact was
produced with the expected gate. CI also pins run count, warmup count, command
timeout, binary path suffixes, the
per-workload `--min-speedup` recorded in the artifact, and the
forward/reverse/round-trip workload mix.
CI also runs `scripts/check_benchmark_docs.py` so new workload names must be
mentioned in the benchmark documentation before they land.
For newly generated local artifacts, add `--verify-file-digests` to the
checker command to recompute recorded SHA-256 digests from disk. The standard
`scripts/verify_janus_performance.py` wrapper does this automatically.

The comparison is intentionally against a named implementation, not against
Janus as an abstract language. Reverse workloads seed the known final state and
run `uncall`/inverted statements on the Jana side, while Reverie uses its
native `reverse` command. In addition to direct upstream Jana examples run by
both implementations, current custom forward/reverse workloads are:

- `fib_loop_n1000`: equivalent loop-based Fibonacci pair transform in Jana and
  Reverie, using Jana `-m64` to match Reverie's signed `i64` wrapping behavior.
- `fib_loop_n1000_reverse`: the same Fibonacci transform from final state back
  to the initial state.
- `fib_loop_n1000_roundtrip`: forward Fibonacci followed by the reverse
  fixture, timed as one checked workload on each side.
- `procedure_call_n1000`: 1,000 iterations of a by-reference procedure call
  followed by its matching uncall, isolating dispatch overhead.
- `procedure_call_n1000_reverse`: the same call/uncall loop from final counter
  state back to the initial counter state.
- `procedure_call_n1000_roundtrip`: forward procedure loop followed by reverse
  cleanup, timed as one checked workload on each side.
- `janus_stack_reverse_cleanup`: recursive stack reversal followed by matching
  reverse cleanup, using Jana and Reverie fixtures with the same call/uncall
  shape.
- `bit_reversal_n8`: table-driven eight-element bit-reversal permutation.
- `bit_reversal_n8_reverse`: the same permutation from final array state back
  to the original array.
- `bit_reversal_n8_roundtrip`: bit reversal followed by reverse bit reversal,
  timed as one checked workload on each side.
- `janus_perm_to_code_reverse`: factorial-code style array back to the original
  permutation.
- `janus_perm_to_code_roundtrip`: permutation-to-code followed by reverse
  conversion, timed as one checked workload on each side.
- `rle_compression_n8`: witness-preserving run-length compression of an
  eight-element array.
- `rle_compression_n8_reverse`: compressed run table back to zeroed output
  tables while preserving the source witness array.
- `rle_compression_n8_roundtrip`: run-length compression followed by reverse
  decompression cleanup, timed as one checked workload on each side.
- `jana_matrixmult_v1_reverse`: upstream matrix multiplication final state
  back to the zero store, using Reverie's native `reverse` command against a
  Jana fixture with inverted calls.
- `jana_matrixmult_v1_roundtrip`: upstream matrix multiplication forward run
  followed by the reverse fixture, timed as one checked workload on each side.
- `matrix_transpose_3x3`: nested-iterate, multidimensional-array transpose
  with reversible element swaps.
- `matrix_transpose_3x3_reverse`: transposed matrix back to a zero store,
  proving the inverse path and cleanup.
- `matrix_transpose_3x3_roundtrip`: matrix transpose followed by reverse
  cleanup, timed as one checked workload on each side.
- `janus_root_66`: Jana's `examples/sqrt.ja` against Reverie's
  `examples/janus_root.rev`.
- `janus_root_66_reverse`: square-root final state back to the original input.
- `janus_factor_840`: Jana's `examples/factor.ja` against Reverie's
  `examples/janus_factor.rev`.
- `janus_factor_840_reverse`: factor table final state back to `num = 840`.
- `janus_sort_n50_reverse_order`: a modern-Jana version of the classic
  reversible bubble sort against Reverie's `examples/janus_sort.rev`.
- `janus_sort_n50_reverse_order_reverse`: sorted list plus permutation witness
  back to reverse order and a zeroed witness array.
- `janus_sort_n50_reverse_order_roundtrip`: sort followed by reverse sort,
  timed as one checked workload on each side.
- `janus_schroedinger_n16_t100`: a small modern-Jana wave-equation style
  stencil inspired by Jana's legacy `basicExamples/schroedinger.janus`.
- `janus_schroedinger_n16_t100_reverse`: the same wave-kernel state transform
  from final arrays back to the original wave state.
- `janus_schroedinger_n16_t100_roundtrip`: wave-kernel forward fixture followed
  by the reverse fixture, timed as one checked workload on each side.
- `janus_turing_binary_inc`: a modern-Jana version of the historic reversible
  Turing-machine binary-increment simulation with explicit rule-selection
  guards.
- `janus_turing_binary_inc_reverse`: the halting Turing-machine state back to
  the zero setup store.
- `janus_turing_binary_inc_roundtrip`: Turing-machine forward simulation
  followed by reverse cleanup, timed as one checked workload on each side.

If the bundled Jana binary cannot run on your platform, build Jana locally with
`cabal` or `stack` and pass `--jana-bin path/to/janus`.
