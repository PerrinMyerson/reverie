# Model Import

Reverie's current ML lane is deterministic, inspectable inference for small
quantized models. The practical handoff is: train or fit a model in an
external stack, quantize it to the Reverie Q31 contract, import it as a signed
model bundle, then run verified Reverie inference, evaluation, and artifact
profiling.

This guide describes the current MNIST-shaped linear classifier used by
`examples/mnist_identify.rev`. It is intentionally narrower than a general
PyTorch or NumPy checkpoint importer: the imported model is accepted as an
external model and recorded with `provenance_kind: "external_import"`, not as a
Reverie training proof.

## Shape Contract

The importer accepts either a raw model object or an object with a `model`
field:

```json
{
  "format": "weights_bias_q31_json",
  "model": {
    "weights": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    "bias": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  }
}
```

The full payload must satisfy these rules:

- `weights` has 784 rows and each row has 10 signed integer columns.
- `bias` has 10 signed integer entries.
- `weights[pixel][digit]` contributes to one output digit logit.
- Runtime images use `image_u8` with 784 byte values.
- Each byte pixel is converted with `pixel_q31 = (pixel * 2147483648) // 255`.

The inference formula is:

```text
logits[digit] = bias[digit] + sum(pixel_q31[pixel] */ weights[pixel][digit])
```

The `*/` operation is Reverie's Q31 fixed multiply. Integer values are signed
`i64`, and integer arithmetic follows Reverie's wrapping semantics.

## Quantization

Use the same scale as the runner:

```python
Q31_ONE = 1 << 31


def q31(value):
    return int(round(value * Q31_ONE))


def pixel_to_q31(pixel):
    return (int(pixel) * Q31_ONE) // 255
```

If the training framework used a different input normalization, fold that
normalization into the exported weights and bias before writing the JSON. For
example, if training used centered pixels instead of raw `image_u8 / 255`,
export weights and bias that expect Reverie's `pixel_to_q31` inputs.

Here is a minimal writer for a float linear model already arranged as
`weights[pixel][digit]` and `bias[digit]`:

```python
import json

Q31_ONE = 1 << 31


def q31(value):
    return int(round(value * Q31_ONE))


def write_reverie_q31_model(weights_float_784x10, bias_float_10, path):
    if len(weights_float_784x10) != 784:
        raise ValueError("weights must have 784 rows")
    if any(len(row) != 10 for row in weights_float_784x10):
        raise ValueError("each weights row must have 10 columns")
    if len(bias_float_10) != 10:
        raise ValueError("bias must have 10 entries")

    payload = {
        "format": "weights_bias_q31_json",
        "model": {
            "weights": [
                [q31(value) for value in row]
                for row in weights_float_784x10
            ],
            "bias": [q31(value) for value in bias_float_10],
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
```

## Exporter

The repository also includes a dependency-light JSON exporter that performs
the same shape checks and Q31 conversion:

```sh
python3 scripts/export_q31_linear_model.py \
  --input target/linear-float-model.json \
  --output target/q31-linear-model.json \
  --input-layout class-pixel \
  --input-scale float
```

Use `--input-layout class-pixel` for the common linear-layer shape
`weights[digit][pixel]` or `--input-layout pixel-class` for Reverie's native
`weights[pixel][digit]` shape. The default `auto` mode accepts either `784 x
10` or `10 x 784`. The exporter also recognizes common JSON state-dict shapes
such as:

```json
{
  "state_dict": {
    "linear.weight": [[0.0]],
    "linear.bias": [0.0]
  }
}
```

The real `linear.weight` payload must contain 10 rows of 784 float values, and
`linear.bias` must contain 10 float values. For other JSON layouts, pass
explicit slash-separated paths:

```sh
python3 scripts/export_q31_linear_model.py \
  --input target/custom-linear-model.json \
  --output target/q31-linear-model.json \
  --weights-path checkpoint/head.weight \
  --bias-path checkpoint/head.bias \
  --input-layout class-pixel
```

For PyTorch checkpoints, first extract an exporter-readable JSON state_dict:

```sh
python3 scripts/extract_torch_linear_state_dict.py \
  --checkpoint target/linear.pt \
  --output target/linear-state-dict.json \
  --weights-key linear.weight \
  --bias-key linear.bias
```

The extractor writes `format: "torch_linear_state_dict_json"` with a
`state_dict` object that contains `linear.weight` and `linear.bias`, which the
Q31 exporter recognizes automatically:

```sh
python3 scripts/export_q31_linear_model.py \
  --input target/linear-state-dict.json \
  --output target/q31-linear-model.json \
  --input-scale float
```

The extractor imports PyTorch only when reading a real `.pt` or `.pth` file and
uses `torch.load(..., weights_only=True)` by default. Only pass
`--allow-unsafe-pickle` for trusted legacy checkpoints that cannot be loaded
through the safe weights-only path. If a checkpoint stores the state dictionary
under a custom nested field, pass `--state-dict-key`, for example
`--state-dict-key checkpoint/model_state`.

If the source values are already signed Q31 integers, use `--input-scale q31`
to pass them through after shape and `i64` range checks.

For NumPy-style artifacts, the same exporter can read `.npz` files directly
without importing NumPy at runtime:

```sh
python3 scripts/export_q31_linear_model.py \
  --input target/linear-model.npz \
  --output target/q31-linear-model.json \
  --input-layout class-pixel \
  --input-scale float
```

By default, `.npz` inputs look for `weights`/`bias`, `linear.weight`/
`linear.bias`, or `classifier.weight`/`classifier.bias` arrays. The archive
members must be C-order numeric `.npy` arrays: weights are either `784 x 10`
or `10 x 784`, and bias is length 10. For custom member names, pass:

```sh
python3 scripts/export_q31_linear_model.py \
  --input target/custom-linear-model.npz \
  --output target/q31-linear-model.json \
  --npz-weights head.weight \
  --npz-bias head.bias \
  --input-layout class-pixel
```

## Generic Reverie Seed Files

The signed `reverie-mnist-linear` runner is the checked bundle path for the
current linear classifier. For other tensor programs, including
`examples/mnist_mlp_witness.rev`, use the generic CLI's `--vars-json` seed file
support to feed large Q31 tensors without turning model weights into enormous
command lines:

```json
{
  "image": [0, 255],
  "weights": [[2147483648, 0], [0, -2147483648]],
  "bias": [0, 0],
  "label": 0
}
```

The real `examples/mnist_identify.rev` seed file still needs 784 `image`
entries, 784 rows of 10 `weights` entries, 10 `bias` entries, and a scalar
`label`. JSON seed numbers must be signed integers; floating-point JSON values
are rejected so Q31 tensors stay exact. Use the signed model-bundle flow below
when you need persisted provenance, source-file hashes, and verification
reports.

For the MLP witness example, export `784 -> 16 -> 10` weights directly into a
pure `--vars-json` seed object:

```sh
python3 scripts/export_q31_mlp_vars.py \
  --input target/mlp-state-dict.json \
  --output target/mlp-vars.json \
  --metadata-output target/mlp-vars.metadata.json \
  --input-scale float

cargo run -p reverie-cli -- run examples/mnist_mlp_witness.rev \
  --vars-json target/mlp-vars.json \
  --json > target/mlp-run.json

python3 scripts/check_q31_mlp_witness.py \
  --vars-json target/mlp-vars.json \
  --run-output-json target/mlp-run.json \
  --json > target/mlp-check-report.json

python3 scripts/check_mnist_ml_profile.py target/mlp-check-report.json

python3 scripts/summarize_mnist_ml_profile.py target/mlp-check-report.json \
  --output target/mlp-check-report.md
```

The exporter recognizes native Reverie keys `w1`/`b1`/`w2`/`b2` as well as
common PyTorch-style `state_dict` keys such as `fc1.weight`, `fc1.bias`,
`fc2.weight`, and `fc2.bias`. In `torch` layout it transposes
`out_features x in_features` layers into Reverie's `in_features x
out_features` tensor shape. The main output intentionally contains only seed
variables, while `--metadata-output` records source paths, layout choices, Q31
scale, and the target program.
The MLP witness checker independently recomputes hidden preactivations, ReLU
masks, hidden activations, logits, output errors, hidden backprop values,
hidden deltas, predictions, correctness flags, and both layer updates, validates
the generic `witness_proof` fingerprints and the `dataset_loops` proof that
the `sample` loop is bounded by `labels`, then compares those tensors against
the `reverie_run_result` JSON store.
Its JSON report includes a `deterministic_q31_mlp_witness_replay` proof-cost
block with `witness_payload_bytes`, `trace_payload_bytes`,
`replay_payload_bytes`, `forward_recompute_steps`, `inverse_recompute_steps`,
and `recomputed_update_payload_bytes`, plus the aggregate witness-store proof
fingerprint and dataset-loop metadata, so the report shows what is stored, what
is deliberately recomputed, how the trace is bounded by the dataset, and which
language-level witness evidence was checked. The generic ML profile checker and
Markdown renderer accept the same report, which lets MLP witness proof costs
sit beside the linear runner's speed, memory, and artifact-profile reports.
Add `--markdown-output PATH` to render the activation/mask witness tapes,
per-sample counts, witness fingerprints, replay costs, and recompute-vs-witness
tradeoff as a reviewer card.

## Import And Verify

Import the JSON as a signed Reverie model bundle:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --import-model-json target/q31-linear-model.json \
  --model-output target/imported-q31-linear-model-bundle.json \
  --json
```

Then verify the bundle:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --verify-model target/imported-q31-linear-model-bundle.json \
  --json
```

Verification checks the model shape, payload size, fingerprints, import
metadata, and `source_model_json_checked`. When the original source JSON file
is still present, verification also checks the source file hash and embedded
model contents. If the source JSON is unavailable, the signed model bundle can
still be verified against its embedded model, but the source-file check is
reported as unavailable.

## Inference

Run deterministic inference with a sample borrowed from an audit bundle:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-model-inference target/imported-q31-linear-model-bundle.json \
  --sample-audit target/mnist-self-test-replay-bundle.json \
  --audit-step 0 \
  --inference-output target/imported-model-inference-bundle.json \
  --json
```

Or provide a new labeled sample JSON:

```json
{
  "kind": "reverie_mnist_linear_q31_sample",
  "schema_version": 1,
  "image_u8": [0],
  "label": 0
}
```

The real `image_u8` array must contain 784 byte values:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-model-inference target/imported-q31-linear-model-bundle.json \
  --sample-json target/mnist-sample.json \
  --inference-output target/imported-model-sample-inference-bundle.json \
  --json
```

The resulting inference bundle records the model, sample, result, attribution
summary, full contribution-ledger counts and SHA-256 fingerprints, runtime
state bytes, and forward/inverse recompute counts. Its verifier reruns Reverie
forward and backward before accepting the deterministic inference claim. Add
`--markdown-output PATH` to `--inspect-model-inference` for a direct prediction
explanation card, or to `--verify-inference` to render the verified prediction,
attribution ledgers, source-artifact checks, contract checks, and replay/memory
costs as a review card.

For a source-only classification handoff, add `--standalone-rev-output PATH`:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-model-inference target/imported-q31-linear-model-bundle.json \
  --sample-audit target/mnist-self-test-replay-bundle.json \
  --audit-step 0 \
  --standalone-rev-output target/imported-mnist-classifier.rev
cargo run -p reverie-cli -- run target/imported-mnist-classifier.rev --json
cargo run -p reverie-cli -- roundtrip target/imported-mnist-classifier.rev --json
```

The generated file is normal `.rev` source with the model and selected sample
embedded as tensor literals. The prediction is produced by `vecmat_q31`,
`argmax`, and `argmax_eq` inside that source file.

## Reference Check

Use the standalone reference checker when you want an independent, minimal
implementation of the Q31 inference contract:

```sh
python3 scripts/check_q31_inference.py \
  --model target/q31-linear-model.json \
  --sample target/mnist-sample.json \
  --expect-prediction 0 \
  --expect-correct true \
  --json
```

The same checker can read a signed sample-set bundle produced by
`--export-samples`; use `--sample-index N` to select the row to audit:

```sh
python3 scripts/check_q31_inference.py \
  --model target/imported-q31-linear-model-bundle.json \
  --sample target/mnist-samples.json \
  --sample-index 0 \
  --json
```

Use `--all-samples` to independently evaluate the whole exported sample set:

```sh
python3 scripts/check_q31_inference.py \
  --model target/imported-q31-linear-model-bundle.json \
  --sample target/mnist-samples.json \
  --all-samples \
  --json
```

The checker accepts either the Q31 import JSON or a signed model bundle, reads
either a single `image_u8` sample object or a
`reverie_mnist_linear_q31_samples` bundle, computes `pixel_q31`, `vecmat_q31`,
wrapping logit accumulation, first-maximum `argmax`, top-logit ordering, and
correctness in dependency-light Python. Its JSON also includes deterministic
attribution for the predicted class: winner and runner-up digits, margin,
reconstructed logit and margin checks, largest active-pixel contributions to
the winning logit, largest winner-vs-runner-up margin contributions, and
fingerprints for the complete contribution ledgers. When a sample-set row is
selected, the report preserves `sample_index`, `audit_step`, and
`source_sample_index` metadata. It is not a replacement for signed Reverie
verification, but it is useful as a tiny cross-check for exporter and
model-handoff bugs before producing larger inference or evaluation bundles.
Add `--markdown-output target/q31-reference-inference.md` to render the same
checked prediction, top logits, margin, and contribution-ledger evidence as a
compact explanation card for reviewers.
In all-samples mode the report kind is
`reverie_q31_linear_reference_evaluation`; it includes aggregate accuracy,
per-label `by_label` summaries, compact per-row predictions, `top_low_margin`
rows, and `top_incorrect` rows. `scripts/check_mnist_ml_profile.py` validates
both `reverie_q31_linear_reference_inference` and
`reverie_q31_linear_reference_evaluation` reports, and
`scripts/summarize_mnist_ml_profile.py` renders them into the same checked
Markdown profile bundle as the speed, memory, artifact, and witness reports.

## Evaluation And Profiling

Evaluate a signed model over exported or external samples:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --evaluate-model target/imported-q31-linear-model-bundle.json \
  --samples-json target/mnist-samples.json \
  --evaluation-output target/imported-model-evaluation-bundle.json \
  --json
```

Compare the resulting artifacts:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --compare-artifacts \
  target/imported-q31-linear-model-bundle.json \
  target/imported-model-inference-bundle.json \
  target/imported-model-evaluation-bundle.json \
  --json
```

JSON comparison output includes `ml_profile`, which aggregates model, sample,
witness, trace, and derived-update payload bytes plus recompute counts and
payload ratios. Artifact rows also expose checked SHA-256 fingerprint coverage
for source lineage, computation payloads, proof hashes, and model provenance.
That is the most useful current metric family for Reverie's ML story: not just
"did it classify correctly?", but "what exact state did we store, what can we
recompute, and what proof did we verify?" The ML profile
checker and Markdown renderer also accept native signed inference,
inference-verification, model-evaluation, evaluation-scan, and
evaluation-verification JSON reports, so imported-model predictions can sit in
the same checked profile as training traces and reference checker output.
For the local synthetic end-to-end version of that profile, use
`python3 scripts/run_mnist_ml_audit_pipeline.py --out-dir
target/mnist-ml-audit-pipeline --sample-limit 10`; it writes every signed
bundle, reference report, artifact comparison, manifest, Markdown summary, and
`pipeline-summary.json` needed to inspect the replay chain. It also writes
`model-capsule.json`, a SHA-256-fingerprinted handoff artifact that binds the
accepted summary to the signed model and sample bundles, lineage-ledger hashes,
inference replay/parity results, MLP witness proof fingerprint, invertible
pattern checks, evidence index, and scorecard. The pipeline also writes
`model-capsule-profile.md`, a capsule-first Markdown view of the accepted
summary/capsule/manifest trio. The summary is the
machine-readable gate: it records speed, peak RSS, trace/witness ratios,
recompute balance, reverse-check elapsed cost and its
`--max-reverse-train-elapsed-ratio` budget, source checks, replay proof status,
inference explanation contracts, the selected training step debug contract, a
selection metric that ties the inspected training update back to scan evidence,
the configured `--audit-step-strategy` when the scan chooses the step, the
configured `--evaluation-row-strategy` when the evaluation scan chooses the
prediction to explain, and native-vs-reference Q31 agreement for both the
inspected inference sample and the evaluation summary. The pipeline also emits
and gates an inference trace profile that compares native report ledgers,
verification ledgers, and the independent Q31 reference ledger fingerprints for
the selected sample, while recording evaluation-row replay/source consistency
until per-row reference output is materialized. It also emits and gates the
end-to-end reversible inference trace report from
`examples/reversible_inference_trace.rev`, including preprocessing, logits,
top-class rankings, top-logit values, prediction, runner-up class, top-two
margin, correctness, witness bytes, replay bytes, zero trace bytes, and reverse
restoration, plus contribution-ledger fingerprints for the winning logit and
runner-up margin. The full
training-audit verifier also emits
`training-audit-verification.json`, which rebuilds the final model from the
witness trace and checks a compact hash-chained lineage ledger with
transition-ledger and final-chain fingerprints. The selected-step
training-update ledger records debug updates with full bias-delta and
weight-delta ledger fingerprints plus a cause ledger that binds the sample,
witness, logits/errors, active pixels, and top deltas, so update-debugging
evidence is compact while the whole run is still tied to the exact sample
order, updates, and final weights. The pipeline also writes
`training-step-debug.md` as a compact selected-update explanation card with
the debug contract, witnesses, top weight deltas, bias deltas, replay proof,
and ledger fingerprints. The full Markdown profile
starts with a North-Star Readiness table, ML capability map, V6 scorecard,
Recompute Frontier table, and
observed/10x/100x Scaling Projection table
that roll up the V1-V6 roadmap status plus those same speed, memory, trace,
replay, and contract signals, including the inspected Q31 reference parity
contract, for quick inspection. The frontier rows make the retained-state
tradeoff concrete for full training replay, selected update debugging, single
and batch inference replay, MLP witness traces, and zero-witness invertible
coupling plus reversible preprocessing, plus end-to-end reversible inference
trace replay. The scaling rows derive per-unit replay and recompute costs from the
checked reports so the import profile shows how trace and witness footprints
grow before you move to a larger model or batch. The same pipeline
emits the separate capsule profile with a Model Capsules table and writes
`model-capsule-verification.json` as the portable handoff verdict. That
verification report is a post-manifest attestation over the summary, capsule,
manifest, and capsule profile, so it can be regenerated without changing the
hash-indexed evidence set. The pipeline also runs
`examples/mnist_mlp_witness.rev` with a deterministic Q31 seed and
records the MLP activation/mask witness proof costs, so hidden-layer witness
regressions sit beside the linear inference chain. It also runs
`examples/invertible_coupling.rev` forward and backward and records the
zero-witness additive coupling proof costs for the invertible-layer pattern,
then runs `examples/reversible_preprocess.rev` forward and backward to prove
that Q31 input centering/permutation can be reversed without a witness tape.
It then runs `examples/reversible_inference_trace.rev` forward and backward so
the imported-model audit profile includes a deterministic reversible
prediction trace with bounded witnesses.
The run fails when any configured audit threshold, debug-contract, or
witness/invertible replay gate regresses, and the summary/manifest record
SHA-256 evidence entries for every generated report, seed, runtime output,
bundle, and Markdown profile.
The summary also includes machine-readable `ml_capability_map`,
`ml_goal_readiness`, and `recompute_frontier` blocks plus a claims matrix that
ties debugging, lineage, deterministic inference, memory/recompute, MLP
witness, invertible layer, reversible preprocessing, roadmap-capability,
north-star-readiness, and
evidence-integrity claims to their proving gates and digest-indexed files.
The `ml_goal_readiness` block is the compact verdict for the current product
goal: reversible inspectable deterministic ML kernels, not a general-purpose
PyTorch/TensorFlow training replacement.
The generated `ml-audit-handoff.json` and `ml-audit-handoff.md` files are the
reviewer entry point: they fingerprint the capsule verdict, Markdown profiles,
selected training-step debug card, Q31 inference explanation card, native
inference verification card, reversible inference trace report, and MLP witness
report in one compact index.
Recheck the saved acceptance artifacts with
`python3 scripts/verify_model_capsule.py target/mnist-ml-audit-pipeline`;
add `--json` when CI or deployment tooling needs the capsule fingerprint,
gate count, model hash, witness proof, scorecard, and capsule-profile SHA-256.
Add `--require-handoff --require-verification-markdown` when the reviewer
packet itself must be fresh: the verifier recomputes every
`ml-audit-handoff.json` artifact byte count/SHA-256 and checks that
`ml-audit-handoff.md` still matches the rendered handoff.
For the full current product-goal gate, run
`python3 scripts/check_ml_north_star.py target/mnist-ml-audit-pipeline`; it
combines the capsule/handoff verifier, V1-V6 readiness checks, reversible
inference roundtrip replay, and the Jana-vs-Reverie speed+RSS artifact into one
fingerprinted `ml-north-star-gate.json` verdict. Its Markdown companion also
summarizes the fingerprinted goal contract, per-claim proof rows,
claim/evidence coverage, and reviewer replay commands for the handoff, profile
evidence, benchmark, roundtrip proof, and selected training-step debug card.
Add
`--list-reviewer-commands` to inspect those command ids, or
`--run-reviewer-command roundtrip_proof` to execute a named replay path and then
refresh the gate report. Named replay runs also write `ml-reviewer-replay.json`
and `ml-reviewer-replay.md`, including command arrays, exit codes, elapsed
times, output byte counts, output SHA-256 hashes, and the gate fingerprint
before and after replay. Add `--verify-reviewer-transcript` when the reviewer
packet must prove the saved transcript JSON and Markdown are fresh, command
definitions still match the gate, and stdout/stderr hashes still match the
captured replay output. A successful transcript check also writes
`ml-release-verification.json` and `ml-release-verification.md`, which bind the
gate fingerprint, explicit goal/non-goal contract, claim/evidence coverage,
benchmark summary, transcript fingerprint, and replayed command ids in one
compact reviewer receipt. Add
`--verify-release-verification` to recheck that saved receipt against the
current gate and saved transcript. The receipt requires the `profile_evidence`,
`benchmark_speed_memory`, `roundtrip_proof`, `training_step_debug`,
`mlp_witness_proof`, `invertible_coupling_proof`,
`reversible_preprocess_proof`, and `reversible_inference_trace_proof` replay
commands to be present in the transcript, and its Markdown includes commands to
recheck the saved receipt or regenerate it from the saved transcript. Its
`Goal Contract` section
fingerprints the north-star string, the PyTorch/TensorFlow non-goal, the V1-V6
capability status, readiness counts, allowed ML-audit claim set, and cost
signals. It also renders a per-claim proof table that maps each contract claim
to the passing pipeline claim rows, required gate metrics, release-level checks
such as the reversible roundtrip and speed+RSS benchmark, and digest-indexed
evidence references. The receipt rejects stale packets when its recomputed
goal contract differs from the contract embedded in the gate. Its
`Claim Replay Coverage` table maps each contract claim to the saved reviewer
replay commands and gate artifact ids that directly exercised it. The receipt
fails when any contract claim is missing a required replay command or proof
artifact, and the MLP, invertible-coupling, reversible-preprocess, and
reversible-inference-trace replay commands use `--expect-report-json` to prove
their recomputed report still matches the saved machine-readable proof card.
It also records SHA-256 digests for the Python verifier scripts used to
validate the gate, capsule, profile evidence, and benchmark artifact, plus a
compact manifest of the gate, transcript, and benchmark files used to produce
the receipt. It also summarizes the gate's underlying audit-packet artifact
index with a file count, byte total, and SHA-256 fingerprint.
The JSON verifier report also includes a fingerprinted
`trust_certificate` object with a canonical payload for the handoff verdict:
integrity hashes, north-star readiness, training-lineage reversibility, Q31
deterministic inference replay, the reversible inference trace's top-k classes,
top-k logit values, label rank, top-k correctness signal, margin ledgers,
witness/replay byte counts, MLP dataset-loop witness proof, and the
memory/trace/replay cost envelope.
Add `--output target/mnist-ml-audit-pipeline/model-capsule-verification.json`
to refresh the saved verifier report outside the manifest hash cycle.
Add `--markdown-output
target/mnist-ml-audit-pipeline/model-capsule-verification.md` when reviewers
need the same trust certificate as a compact Markdown table. The pipeline writes
that Markdown handoff next to the JSON verifier report by default, and the JSON
report records `verification_markdown` byte-count and SHA-256 metadata for the
rendered Markdown. When a saved Markdown handoff is already present, the
verifier checks that it still matches the rendered trust certificate; add
`--require-verification-markdown` to fail when the saved Markdown file is
missing, or `--verification-markdown PATH` to validate a non-default handoff.
For the lower-level report checker, use
`python3 scripts/check_mnist_ml_profile.py
target/mnist-ml-audit-pipeline/pipeline-summary.json
target/mnist-ml-audit-pipeline/model-capsule.json
target/mnist-ml-audit-pipeline/pipeline-manifest.json --verify-pipeline-files`;
the checker validates each file, recomputes the evidence hashes, cross-checks
the capsule against the summary, and checks the manifest's capsule fingerprint.

## Goal

For external models, the current goal should be 1-to-1 deterministic inference
and auditability:

- import a checked Q31 representation without losing shape or provenance;
- run the same fixed-point semantics every time;
- verify signed model, inference, evaluation, and source references;
- measure storage, witness, trace, memory, and recompute cost.

The next useful layer is additional model-family exporters beyond the current
linear PyTorch/NumPy handoff, followed by richer reversible training kernels
whose witness traces are produced inside Reverie itself.
