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
tensor_matmul_builtin_vs_loops_3x3
mnist_reversible_step_784x10
mnist_witness_tape_loop_2x784x10
mnist_mlp_witness_2x784x16x10
invertible_coupling_4x4
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
compile-time, scrub-timeline, and tensor builtin-vs-loop comparison groups, and
that every benchmark wrapper is registered in the `criterion_group!` macro so
documented groups actually run.

The Fibonacci benchmark compares the reference tree walker against the slot
engine on the Fibonacci pair transform with `n = 1000`. This isolates execution from
CLI startup and gives us a stable place to add more workloads.

The forward sort benchmark runs the shared 50-element Janus sort fixture on a
reverse-ordered array. This is the first Janus-corpus performance workload and is
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

The tensor matmul comparison runs the same 3x3 matrix state through explicit
loop accumulation and through `out += matmul(a, b)`, for both tree-walk and
slot-compiled execution. It tracks the overhead and benefit of the ML tensor
builtin surface against ordinary Reverie loops.

The MNIST reversible step benchmark runs the full `784 x 10` Q31 classifier
step in `examples/mnist_reversible_step.rev`, including `vecmat_q31`,
`argmax`, `one_hot_q31`, `outer_q31`, and learning-rate scaling. It keeps the
exact MNIST-shaped reversible training path visible in Criterion. For an
end-to-end data path, `cargo run -p reverie-cli --bin reverie-mnist-linear --`
loads MNIST IDX files, trains the same Q31 linear model, evaluates
`examples/mnist_identify.rev`, and can replay the inverse of every training
step with `--reverse-check`. The runner also reports the compact witness-trace
payload size so speed, reverse-check cost, and trace footprint can be compared
together. Add `--json` to turn the run into a benchmark/audit artifact with
train/eval throughput, trace bytes, reverse-check cost, model and dataset
payload bytes, estimated payload bytes, and peak RSS on supported platforms.
Add `--audit-output PATH` when the run should leave a replay bundle containing
the final model and per-step witnesses. Bundles include SHA-256 fingerprints for
the source programs, final model, witness trace, proof-cost summary, report,
unsigned payload, and stable computation identity; `--verify-audit PATH`
reloads that bundle, rejects fingerprint mismatches, recomputes the proof-cost
summary, and runs the trace backward as a separate audit check. Use
`--inspect-audit PATH --audit-step N` to pull out one saved update, recompute
its prediction/correctness witnesses from logits, and summarize the active
pixels plus bias and weight deltas that changed the model. The inspector
reconstructs the selected step's before/after model window by reversing from
the final model, so the bundle does not need to store every intermediate model
snapshot. Add `--step-output PATH` to save that update as a standalone proof
artifact; `--verify-step PATH` reruns the update forward and backward and
recomputes its signed proof bytes for the two model snapshots, sample,
witnesses, derived update summary, and one-step recompute cost. Use
`--inspect-inference PATH --audit-step N` to run the final model
on a saved sample, report logits/top classes, and verify that the inference
trace itself reverses to the exact initial inference state. Saved inference
bundles carry a signed proof object for the deterministic Q31 claim: model
payload bytes, sample bytes, witness/result bytes, zero trace bytes, replay
payload bytes, runtime state bytes, forward/inverse recompute steps, the
winning margin, attribution checks, full contribution-ledger counts, ledger
SHA-256 fingerprints, and fingerprints of the model, sample, and recorded
result. Add `--markdown-output PATH` to `--inspect-inference`,
`--inspect-model-inference`, or `--inspect-evaluation` to render the checked
prediction, top logits, attribution ledgers, source rows, contract checks, and
replay/memory costs immediately. `--verify-inference PATH` rebuilds that proof
from fresh Reverie execution before accepting the bundle, and its
`--markdown-output` card adds source-artifact verification checks.
`--compare-artifacts PATH... --json` adds an `ml_profile` aggregate over
training, model, sample-set, step, inference, and evaluation proof artifacts:
total model/sample/witness/trace/update payload bytes, total forward and
inverse recompute steps, total recompute steps, and trace-to-model plus
witness-to-model payload ratios. That is the compact V6 view for comparing
storage against recomputation without reading every proof row by hand.
Use `python3 scripts/check_mnist_ml_profile.py` to validate runner JSON reports
or artifact-comparison reports. The checker confirms elapsed-throughput fields
are finite, reverse-check proof costs match trace entries, estimated payload
bytes add up, peak RSS is present when required, and `ml_profile` totals and
ratios agree with the artifact rows. It also accepts the Q31 reference
inference and all-samples evaluation reports from
`scripts/check_q31_inference.py`, including attribution reconstruction,
contribution-ledger fingerprints, top-logit ordering, per-label summaries, and
low-margin/incorrect row rankings. Training-audit scan reports, inspected audit-step reports, and
standalone step-verification reports are checked too, so a low-margin or
large-update training step can be found, rendered, and tied back to the
`deterministic_q31_training_step_replay` proof. Native signed model inference,
inference verification, model evaluation, evaluation scan, and evaluation
verification reports are also accepted, tying bit-for-bit Q31 predictions back
to `deterministic_q31_inference_replay` proof bytes, inverse recompute counts,
source-artifact checks, and low-margin evaluation rows. Pass
`--require-audit-contract` on an
artifact-comparison report when the report must prove the reversible ML kernel
contract: a signed training trace, standalone model, sample set, one-step
replay, inference replay, evaluation replay, nonzero witness/trace accounting,
and balanced forward/inverse recompute steps. Use the Markdown renderer to turn
the checked JSON into tables covering speed, peak RSS, trace payload, proof
payloads, artifact rows, audit-contract status, and recompute cost:

```sh
python3 scripts/summarize_mnist_ml_profile.py REPORT.json ... \
  --require-audit-contract \
  --output target/mnist-ml-profile.md
```

For a one-command local smoke of the full self-test audit path, run:

```sh
python3 scripts/run_mnist_ml_audit_pipeline.py \
  --out-dir target/mnist-ml-audit-pipeline \
  --sample-limit 10
```

By default the pipeline inspects the explicit `--audit-step` value. To let the
scan pick the debug target, add `--audit-step-strategy lowest-margin`,
`--audit-step-strategy largest-update`, or `--audit-step-strategy
top-suspicious`. The generated summary records both the requested step and the
selected step, then gates that the selected update matches the configured
strategy before accepting the one-step replay/debug artifact.
The exported evaluation-row inference works the same way: `--evaluation-row`
keeps explicit behavior, while `--evaluation-row-strategy lowest-margin` selects
the lowest-margin prediction from the evaluation scan and
`--evaluation-row-strategy top-incorrect` selects the first incorrect row when
one exists. The native reversible inference bundle, verification report, and
Markdown `Row evidence` column are then tied back to that scan-selected row.

The pipeline trains the synthetic Q31 model with a retained witness trace,
exports signed model/sample bundles, verifies the full training lineage with a
hash-chained update ledger, inspects and verifies a training step, runs native
inference and batch evaluation with replay bundles, runs the
dependency-light Q31 reference checker, runs the MLP witness trace and its
independent checker, runs the invertible coupling and triangular residual
blocks forward and backward without witness tapes, runs the reversible preprocessing block forward and
backward without witness tapes, runs the reversible inference trace forward
and backward with its independent checker, writes a reversible inference trace
card with attribution and reverse-restoration proof costs, compares signed artifact payloads,
and renders a checked `mnist-ml-audit-profile.md` with a top-level
North-Star Readiness table, ML capability map, V6 scorecard, Recompute Frontier table, measured observed/10x/
100x Scaling Projection table, and the same validator gates.
It also writes `pipeline-summary.json`, a machine-readable acceptance artifact,
and `model-capsule.json`, a portable SHA-256-fingerprinted capsule tying the
accepted summary to the signed model bundle, sample bundle, lineage ledgers,
inference replay/parity results, deterministic inference action contract, MLP
witness proof, invertible-pattern checks, and V6 scorecard. After the manifest
is written, the pipeline also renders `model-capsule-profile.md`, a
capsule-first Markdown view whose top table shows the capsule fingerprint, gate
count, model/sample hash, lineage hash, Q31 parity, witness proof, peak RSS, max
replay bytes, and readiness score, plus the operation-level inference contract
for prediction reproduction, margin explanation, native replay, and reversible
trace reversal. The summary records speed, peak RSS, trace/witness payload
ratios, recompute balance,
reverse-check elapsed ratio, inspected-sample and evaluation-summary
native-vs-reference agreement, training-step and evaluation-row selection
evidence, full training-lineage ledger fingerprints and final-chain evidence,
full training-update ledger fingerprints, a selected-step cause ledger that
binds the sample, witnesses, logits/errors, active pixels, and top weight
deltas, inference explanation
contracts, a checked inference trace profile
that ties native reports, replay verification, and independent Q31 reference
ledger fingerprints together, end-to-end reversible inference trace metrics
for preprocessing, logits, prediction, correctness, witness bytes, replay
bytes, zero trace bytes, and attribution-ledger fingerprints, MLP
activation/mask witness proof costs, zero-witness
invertible coupling, triangular residual, and reversible preprocessing proof costs, source-check
status, and replay gates. The Recompute Frontier records each replay mode's retained state, replay bytes,
witness bytes, trace bytes, update bytes, state bytes, and forward/inverse
recompute steps, making the storage-vs-recompute tradeoff explicit for full
training traces, selected training updates, single inference, batch evaluation,
evaluation-row inference, MLP witness traces, and no-witness invertible
coupling plus triangular residual and reversible preprocessing, plus end-to-end reversible inference
trace replay. The Scaling Projection derives per-unit
replay, witness, trace, recomputed-update, and recompute-step costs from the checked reports and
projects the observed, 10x, and 100x retained-state footprint for training
traces, batch evaluation, and MLP witness replay. It also records a V1-V6
capability map, a V6 scorecard, a north-star readiness report for reversible
inspectable deterministic ML kernels, and a claims matrix that links
the debugging, model-lineage, deterministic-inference, memory/recompute,
MLP-witness, invertible-layer, roadmap-capability, and evidence-integrity
claims to the exact gates, metrics, and SHA-256-indexed files that prove them.
The evidence index covers every report, bundle, seed, runtime JSON output, and
Markdown profile. The summary gate fails the pipeline if the capability map,
reverse replay, witness matching,
training-step selection traceability, training-lineage replay, evaluation
replay, inspected-inference Q31 reference parity, training-update ledger,
lineage-ledger, and cause-ledger fingerprint coverage,
inference trace-profile ledger parity,
reversible inference trace replay, evaluation-summary Q31 reference parity,
MLP witness replay, invertible coupling replay, triangular residual replay, reversible preprocessing replay, accuracy floors,
trace/witness ratio bounds, or the
reverse/train elapsed-ratio budget regress. Tune those floors with options such
as `--min-train-accuracy`, `--min-model-evaluation-accuracy`,
`--max-trace-model-ratio`, and `--max-reverse-train-elapsed-ratio` when running
larger local experiments. Use `--runner-bin` and `--reverie-bin` to point at
prebuilt local binaries for faster repeated runs.
The generated summary, model capsule, manifest, capsule profile, and saved
`model-capsule-verification.json` report are first-class handoff artifacts too;
the trust certificate mirrors the deterministic inference action contract and
its Markdown report renders the same operation table for reviewers. It also
promotes the reversible inference trace's top-k classes, top-k logit values,
label rank, top-k correctness signal, margin ledgers, witness bytes, replay
bytes, and recompute steps into the certificate and review handoff. The handoff includes exact argv-style review
commands for reproducing the prediction, explaining the margin, replaying native
inference, and reversing the trace. The
pipeline executes those commands and writes `inference-action-review-receipt.json`
plus `inference-action-review-receipt.md`, binding command exit codes, elapsed
time, stdout/stderr hashes, capsule fingerprint, trust-certificate fingerprint,
and handoff fingerprint into the release packet. The pipeline also writes
`training-update-review-receipt.json` and `training-update-review-receipt.md`,
which replay the full training lineage, inspect the selected update into scratch
artifacts, and reverse the saved one-step bundle. Both receipt families include
a run fingerprint over timing/output details and a semantic fingerprint over the
stable proof facts, so a fresh replay can be compared without confusing clock
noise for claim drift. It indexes direct native/evaluation-row inference audit
Markdown cards beside the native inference verification, reference inference,
and reversible trace cards.
The verifier cross-validates the capsule against the summary, checks the
manifest's capsule fingerprint, recomputes file evidence, and confirms the
capsule profile still matches the rendered acceptance artifacts. It can also
execute the handoff's inference action review commands and write a replay
receipt with exit codes, elapsed time, and stdout/stderr hashes. Add
`--require-handoff` to recompute every `ml-audit-handoff.json` artifact byte
count/SHA-256 and ensure `ml-audit-handoff.md` still matches the rendered review
packet:

```sh
python3 scripts/verify_model_capsule.py target/mnist-ml-audit-pipeline
python3 scripts/verify_model_capsule.py target/mnist-ml-audit-pipeline --json
python3 scripts/verify_model_capsule.py target/mnist-ml-audit-pipeline \
  --output target/mnist-ml-audit-pipeline/model-capsule-verification.json
python3 scripts/verify_model_capsule.py target/mnist-ml-audit-pipeline \
  --require-handoff --require-verification-markdown
python3 scripts/verify_model_capsule.py target/mnist-ml-audit-pipeline \
  --require-handoff --require-verification-markdown \
  --run-inference-action-commands \
  --action-command-receipt-output target/mnist-ml-audit-pipeline/inference-action-review-receipt.json \
  --action-command-receipt-markdown target/mnist-ml-audit-pipeline/inference-action-review-receipt.md
python3 scripts/run_mnist_ml_audit_pipeline.py \
  --out-dir target/mnist-ml-audit-pipeline \
  --write-training-update-receipt-only
python3 scripts/check_ml_north_star.py target/mnist-ml-audit-pipeline \
  --benchmark-json benchmarks/results/jana-vs-reverie-local-arm64-memory.json \
  --benchmark-markdown benchmarks/results/jana-vs-reverie-local-arm64-memory.md
python3 scripts/check_mnist_ml_profile.py \
  target/mnist-ml-audit-pipeline/pipeline-summary.json \
  target/mnist-ml-audit-pipeline/model-capsule.json \
  target/mnist-ml-audit-pipeline/pipeline-manifest.json \
  --verify-pipeline-files
```

`check_ml_north_star.py` is the combined release gate for the current ML goal.
It requires the capsule and handoff verifier to pass, V1-V6 capability and
readiness gates to pass, the reversible inference roundtrip proof to replay,
and the Jana-vs-Reverie benchmark to include speed and peak RSS ratios that
meet the configured floors. It writes `ml-north-star-gate.json` and
`ml-north-star-gate.md` next to the audit packet. The Markdown gate includes a
`Claim Evidence` table summarizing the checked claims, referenced gates,
referenced metric blocks, digest-indexed files, evidence bytes, and evidence
fingerprints; the gate rejects the packet unless the artifact-integrity claim
covers every digest-indexed evidence file, including the reversible roundtrip
proof and verification artifacts. The Markdown gate also includes roundtrip
ML-profile replay-cost rows for replay payload bytes, witness bytes, static
roundtrip statements, and Q31 builtin calls, a `Model Lineage` section with
checked-step coverage, final/initial model replay status, witness replay, and
lineage/final-chain fingerprints, a `Deterministic Inference` section with the
Q31/native prediction, margin, active-pixel attribution ledgers, replay payload,
reversible-trace witness/recompute costs, and an action contract for reproducing
the prediction, explaining the margin, replaying native inference, and reversing
the trace, plus a `Debuggable Update` section that surfaces the selected training
step, sample, replay payload, nonzero weight/bias deltas, reversed-later-step
count, and update-ledger fingerprints.
The Markdown gate also includes the
fingerprinted `Goal Contract` table, including per-claim proof rows that map
each north-star contract claim to passing pipeline claim rows, required gate
metrics, release-level checks, and evidence counts. The Markdown gate also includes a
`Reviewer Commands` table with exact local commands to rerun the gate, verify
the handoff, validate profile evidence, recheck the benchmark, replay the
roundtrip proof, and regenerate the selected training-step debug card.
Use `--list-reviewer-commands` to print the command ids, or repeat
`--run-reviewer-command ID` to execute selected replay paths through the gate;
for example:

```sh
python3 scripts/check_ml_north_star.py target/mnist-ml-audit-pipeline \
  --run-reviewer-command profile_evidence \
  --run-reviewer-command inference_action_receipt \
  --run-reviewer-command benchmark_speed_memory \
  --run-reviewer-command roundtrip_proof \
  --run-reviewer-command training_step_debug \
  --run-reviewer-command training_update_receipt \
  --run-reviewer-command mlp_witness_proof \
  --run-reviewer-command invertible_coupling_proof \
  --run-reviewer-command triangular_residual_proof \
  --run-reviewer-command reversible_preprocess_proof \
  --run-reviewer-command reversible_inference_trace_proof
```

When reviewer commands run, the gate also writes `ml-reviewer-replay.json` and
`ml-reviewer-replay.md` next to the audit packet. The transcript records each
command array, exit code, elapsed time, stdout/stderr byte counts, and
stdout/stderr SHA-256 hashes, then records the gate fingerprint before and after
the replay. Add `--verify-reviewer-transcript` to validate that the saved
transcript JSON, rendered Markdown, command index, output hashes, and before/after
gate fingerprints still match the current release gate. When that transcript
check passes, the gate also writes `ml-release-verification.json` and
`ml-release-verification.md`, a compact receipt binding the current gate
fingerprint, the explicit goal/non-goal contract, claim/evidence coverage,
speed+RSS benchmark stats, transcript fingerprint, and replayed reviewer
command ids. Add
`--verify-release-verification` to recheck a saved release-verification JSON and
Markdown receipt against the current gate and saved transcript without needing
to rerun the reviewer commands. The receipt requires the saved transcript to
include the `profile_evidence`, `benchmark_speed_memory`, `roundtrip_proof`,
`inference_action_receipt`, `training_step_debug`, `training_update_receipt`,
`mlp_witness_proof`, `invertible_coupling_proof`, `triangular_residual_proof`,
`reversible_preprocess_proof`, and
`reversible_inference_trace_proof` replay
commands, and its Markdown includes copyable commands to recheck the saved
receipt or regenerate it from the saved transcript. Its `Goal Contract`
section fingerprints the north-star string, the PyTorch/TensorFlow non-goal,
the V1-V6 capability status, readiness counts, allowed ML-audit claim set, and
cost signals. It also renders a per-claim proof table that maps each contract
claim to the passing pipeline claim rows, required gate metrics, release-level
checks such as the roundtrip and speed+RSS benchmark, and digest-indexed
evidence references. Its `Claim Replay Coverage` table maps each contract
claim to the saved reviewer replay commands and gate artifact ids that
directly exercised it. The MLP, invertible-coupling, reversible-preprocess,
and reversible-inference-trace replay commands use `--expect-report-json` to
compare the recomputed proof card with the saved machine-readable report before
refreshing Markdown. The receipt rejects stale packets when its recomputed goal
contract differs from the contract embedded in the gate or when any contract
claim is missing a required replay command or proof artifact. The receipt also
records SHA-256 digests for the Python verifier scripts that validate the gate,
capsule, profile evidence, and benchmark artifact, plus a compact manifest of
the gate, transcript, and benchmark files used to produce the receipt. It also
summarizes the gate's underlying audit-packet artifact index with a file count,
byte total, and SHA-256 fingerprint.

The MNIST witness-tape loop benchmark runs
`examples/mnist_witness_tape_loop.rev`, a two-sample `784 x 10` Q31 training
trace that writes logits, errors, predictions, and correctness into indexed
tensor tape slots inside an `iterate` loop. It keeps batched dataset indexing
and first-class trace state visible in the internal suite.

The MNIST MLP witness benchmark runs `examples/mnist_mlp_witness.rev`, a
two-sample `784 -> 16 -> 10` Q31 training trace with explicit hidden
preactivation, ReLU mask, activation, output-error, and hidden-delta witnesses.
It keeps the reversible MLP pattern visible as the tensor surface grows beyond
linear classifiers. `scripts/check_q31_mlp_witness.py --json` independently
checks the same MLP witness trace, validates the generic Reverie
`witness_proof` fingerprints against the concrete witness store, and reports
witness bytes, trace bytes, replay bytes, recompute steps, and recomputed
layer-update bytes. The ML profile checker and Markdown renderer also accept
that JSON report, so MLP witness proof costs and witness-store digests can sit
beside speed, memory, and artifact-profile rows.
Add `--markdown-output PATH` to render the sample predictions, activation/mask
witness counts, per-tape SHA-256 fingerprints, replay costs, and
recompute-vs-witness tradeoff as a review card. Add
`--expect-report-json PATH` when reviewing a release artifact to require the
recomputed report to match the saved JSON proof card exactly.

The invertible coupling benchmark runs `examples/invertible_coupling.rev`, a
RevNet-style additive coupling block over two 4-wide Q31 activation halves. It
keeps no-witness invertible layer patterns visible beside witness-backed
training traces. `scripts/check_invertible_coupling.py --json` independently
checks the same Q31 additive coupling math, verifies Reverie forward output,
verifies reverse execution restores the initial activation halves, and reports
zero witness bytes, zero trace bytes, replay bytes, and balanced recompute
steps.
Add `--markdown-output PATH` to render the coupling state, additive transforms,
reverse contract, and zero-witness payload accounting as a review card. Add
`--expect-report-json PATH` when reviewing a release artifact to require the
recomputed report to match the saved JSON proof card exactly.

The triangular residual benchmark runs `examples/triangular_residual.rev`, a
Q31 invertible residual layer where each coordinate accumulates residual terms
from later coordinates only. `scripts/check_triangular_residual.py --json`
independently checks the triangular Q31 math, verifies Reverie forward output,
verifies reverse execution restores the initial activation vector, and reports
parameter bytes, state bytes, zero witness bytes, zero trace bytes, replay
bytes, and balanced recompute steps. Add `--markdown-output PATH` to render
the residual state, triangular updates, reverse contract, and zero-witness
payload accounting as a review card. Add `--expect-report-json PATH` when
reviewing a release artifact to require the recomputed report to match the
saved JSON proof card exactly.

The reversible preprocessing benchmark runs
`examples/reversible_preprocess.rev`, a Q31 input block that accumulates raw
features into a feature buffer, subtracts a mean vector, and permutes the
result with swaps. `scripts/check_reversible_preprocess.py --json`
independently verifies forward output, verifies reverse execution clears the
feature buffer while preserving raw and mean vectors, and reports zero
witness bytes, zero trace bytes, replay bytes, and balanced recompute steps.
Add `--markdown-output PATH` to render the raw/mean/features sequence,
reverse contract, and zero-witness payload accounting as a review card. Add
`--expect-report-json PATH` when reviewing a release artifact to require the
recomputed report to match the saved JSON proof card exactly.

The reversible inference trace benchmark runs
`examples/reversible_inference_trace.rev`, which chains that style of
preprocessing into a tiny Q31 linear classifier. The independent checker
recomputes logits, ranked classes, ranked logit values, first-maximum
prediction, runner-up class, top-two margin, correctness, and reverse
restoration, then reports model bytes, witness bytes, trace bytes, replay
bytes, balanced recompute steps, top logits, margin, and contribution-ledger
fingerprints for the winning logit and runner-up margin. Add
`--expect-report-json PATH` when reviewing a release artifact to require the
recomputed report to match the saved JSON proof card exactly.

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
directly under `--legacy-janus`, exercising multidimensional arrays, nested
`iterate` loops, array-element procedure arguments, and Jana-compatible update
aliases. It also includes a dedicated 3x3 matrix
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
3.0x, and writes
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
  `examples/matrixmult_v1.0.ja` run by Jana and by Reverie with
  `--legacy-janus`.
- `jana_matrixmult_v1_reverse`: the matrix multiplication final store back to
  zeroed arrays and scalars, using a Jana reverse fixture and Reverie's native
  `reverse --legacy-janus` command.
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
- `janus_sort_n50_reverse_order`: modern-Jana reversible bubble sort run by
  both implementations on a 50-element reverse-ordered array.
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
