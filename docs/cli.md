# CLI Reference

The `reverie` binary is the main interface for checking, running, reversing,
inverting, explaining, and scrubbing programs.

During development, run it through Cargo:

```sh
cargo run -p reverie-cli -- <command>
```

If the binary is installed, replace the prefix with `reverie`.

## Commands

```text
reverie check [OPTIONS] <FILE>
reverie run [OPTIONS] <FILE>
reverie reverse [OPTIONS] <FILE>
reverie roundtrip [OPTIONS] <FILE>
reverie verify-roundtrip [OPTIONS] <PROOF>
reverie invert [OPTIONS] <FILE>
reverie explain [OPTIONS] <FILE>
reverie fmt [OPTIONS] <FILE>
reverie scrub [OPTIONS] <FILE>
```

The workspace also builds `reverie-mnist-linear`, a companion benchmark/audit
binary that drives `examples/mnist_reversible_step.rev` and
`examples/mnist_identify.rev` over synthetic or IDX MNIST-shaped data. Use
`--reverse-check` to retain the compact witness tape and replay every training
step backward, and add `--json` for train/eval throughput, trace bytes,
reverse-check cost, model/dataset payload bytes, estimated payload bytes, and
peak RSS when available. JSON reports also include a `proof` section that
separates model bytes, saved sample bytes, witness bytes, full replay payload
bytes, per-step replay bytes, and forward/inverse recompute steps:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- --self-test --json
```

Use `--audit-output PATH` with `--reverse-check` to write a replay bundle that
contains the final model and per-step witnesses, including the sample image
bytes needed to replay the inverse without re-reading IDX files. Bundles carry
SHA-256 fingerprints for the Reverie source programs, final model, witness
trace, report, unsigned payload, and the stable computation identity
`source programs + final model + witness trace`; verify/inspect commands
recompute those fingerprints and reject mismatches before replay. Verification
then replays the trace forward from the zero model to prove the saved witnesses
and final model agree with fresh Reverie execution, and replays backward from
the final model to prove the initial model is restored. The signed bundle also
carries the same proof-cost summary so the audit artifact records what the
replay proof costs. Add `--markdown-output PATH` to save a reviewer card with
the lineage ledger, replay cost, forward/reverse checks, and first/last
transition fingerprints. Verify a saved bundle later with:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --verify-audit target/mnist-self-test-replay-bundle.json \
  --markdown-output target/mnist-self-test-audit-verification.md --json
```

Use `--export-model AUDIT --model-output PATH` to turn a verified training
trace into a signed standalone Q31 model artifact. Export replays the source
audit bundle forward and backward first, then writes only the final weights,
model payload size, source/proof provenance, and model fingerprints.
`--verify-model PATH` later checks the model shape, model payload bytes, and
SHA-256 fingerprints, and also validates that the source-audit provenance,
embedded training proof-cost summary, and model report agree with each other
without needing to replay the full training trace:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --export-model target/mnist-self-test-replay-bundle.json \
  --model-output target/mnist-self-test-model-bundle.json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --verify-model target/mnist-self-test-model-bundle.json --json
```

Inspect a single saved training update with `--inspect-audit PATH --audit-step
N`. The inspector recomputes the prediction/correctness witnesses from saved
logits and reports the top logits, winner-vs-runner-up margin, active pixels,
error vector, bias delta, and largest weight deltas for that step. It also
reconstructs the selected step's model window by reversing later trace entries
from the final model, then reversing the selected step once more to show
before/after values for the changed weights. JSON output includes a
`debug_contract` block with the claim `step_backward_from_model_update`; it
passes only when the witness recomputes the prediction, the before/after model
window is reconstructed, the computed update deltas match the observed model
delta, and the sample/logit/error/update state needed to explain the step is
present:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-audit target/mnist-self-test-replay-bundle.json --audit-step 0
```

Use `--audit-step-strategy lowest-margin`, `largest-update`, or
`top-suspicious` to let the inspector choose from the same rankings reported by
`--scan-audit`; the JSON report and Markdown card include both the requested
step and selected step:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-audit target/mnist-self-test-replay-bundle.json \
  --audit-step-strategy lowest-margin \
  --markdown-output target/mnist-self-test-step-debug.md
```

Add `--step-output PATH` to save that one reconstructed update as a standalone
step replay bundle. The step bundle stores the before/after model window,
sample, witnesses, update deltas, source-bundle fingerprints, and its own
payload fingerprints. Add `--markdown-output PATH` to `--inspect-audit` to
write a reviewer card for the selected update. `--verify-step PATH` reruns the
step forward from `before_model` to `after_model`, checks the saved witnesses,
then runs the inverse to prove `before_model` is restored; with
`--markdown-output PATH`, the card also includes the replay proof:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-audit target/mnist-self-test-replay-bundle.json \
  --audit-step 0 \
  --step-output target/mnist-self-test-step-bundle.json \
  --markdown-output target/mnist-self-test-step-debug.md
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --verify-step target/mnist-self-test-step-bundle.json \
  --markdown-output target/mnist-self-test-step-debug.md --json
```

Use `--scan-audit PATH --audit-limit N` when you do not already know which
training step to inspect. The scanner verifies the bundle fingerprints, then
summarizes correctness, witness mismatches, trace bytes, and the largest
weight update. It also reports the lowest-margin step, and each ranked row
includes top-logit margin fields so close calls are visible beside wrong or
large-update samples. The scan emits per-label summaries, top confusion pairs,
and three bounded ranked views: `top_suspicious`, `top_low_margin`, and
`top_large_updates`, so the listed step numbers can be passed straight to
`--inspect-audit`:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --scan-audit target/mnist-self-test-replay-bundle.json --audit-limit 5
```

Add audit gate thresholds to make a scan enforceable. For example,
`--audit-max-witness-mismatches 0`, `--audit-min-margin 0`,
`--audit-min-accuracy 95`, `--audit-max-weight-delta N`, and
`--audit-require-label-coverage` print a structured gate report and return a
nonzero exit status when the replay artifact does not satisfy the requested
criteria.

Use `--inspect-inference PATH --audit-step N` to run `mnist_identify.rev` with
the saved final model and the selected bundle sample. It reports the inference
logits and top classes, includes deterministic Q31 attribution for the
predicted class, recomputes the prediction/correctness witnesses, then runs the
inference inverse and checks that the exact initial inference state is
restored. The attribution records the winning digit, runner-up, margin, bias,
reconstructed logit, reconstructed margin, largest signed active-pixel
contributions for the winning logit, and largest signed active-pixel
contributions for the winner-vs-runner-up margin:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-inference target/mnist-self-test-replay-bundle.json \
  --audit-step 0 \
  --markdown-output target/mnist-self-test-inference-audit.md
```

The same `--markdown-output PATH` option works with
`--inspect-model-inference` and `--inspect-evaluation`. Those direct cards
render the top logits, attribution ledgers, source rows, contract checks, and
replay/memory costs without requiring a standalone inference bundle first.

For the smaller generic trace in `examples/reversible_inference_trace.rev`, the
reference checker can render a standalone proof card covering the deterministic
forward trace, attribution ledgers, reverse restoration, and payload costs:

```sh
python3 scripts/check_reversible_inference_trace.py \
  --markdown-output target/reversible-inference-trace.md
```

Add `--inference-output PATH` to save that inference as its own replay bundle.
The standalone bundle stores the model, sample image, label, logits,
prediction, attribution, inverse-restoration proof, and SHA-256 fingerprints
for the identify program, model, sample, result, report, unsigned payload, and
stable computation identity. Verification recomputes attribution from the
saved model and sample and, when referenced model, training-audit, sample, or
evaluation artifacts are available, also checks the embedded inputs against
those signed sources. A re-signed bundle with forged contribution or source
data is rejected. It also records the storage/recompute shape of the proof:
model payload bytes, sample payload bytes, witness/result bytes, zero
per-step trace bytes, replay payload bytes, runtime state bytes, and one
forward plus one inverse recompute step. Verify it later without the training
bundle:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-inference target/mnist-self-test-replay-bundle.json \
  --audit-step 0 \
  --inference-output target/mnist-self-test-inference-bundle.json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --verify-inference target/mnist-self-test-inference-bundle.json \
  --markdown-output target/mnist-self-test-inference-verification.md
```

Use `--inspect-model-inference MODEL --sample-audit AUDIT --audit-step N` to
run the same deterministic inference path from a signed model bundle instead of
the training bundle's embedded final model. The audited sample supplies the
image and label, while the model artifact supplies the deployable Q31 weights:
JSON inference reports also include an `explanation_contract` block with the
claim `q31_inference_prediction_explanation`; it checks that logits determine
the prediction, the label check matches, attribution reconstructs the winning
logit and margin, and the inverse replay restores the initial model/sample
state. `--verify-inference` adds proof, result, and source-input checks to that
same contract. Add `--markdown-output PATH` to render the verified prediction,
attribution ledgers, source-artifact checks, contract checks, and replay/memory
costs as a human review card.

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-model-inference target/mnist-self-test-model-bundle.json \
  --sample-audit target/mnist-self-test-replay-bundle.json \
  --audit-step 0 \
  --inference-output target/mnist-self-test-model-inference-bundle.json
```

Add `--standalone-rev-output PATH` to the same inspected inference command
when you want a self-contained Reverie source artifact. The generated `.rev`
file embeds the selected Q31 image, model weights, bias, label, and expected
classification as global tensor literals, then computes `vecmat_q31`,
`argmax`, and correctness inside the ordinary Reverie program. It can be
checked, run, or roundtripped without the MNIST helper:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-model-inference target/mnist-self-test-model-bundle.json \
  --sample-audit target/mnist-self-test-replay-bundle.json \
  --audit-step 0 \
  --standalone-rev-output target/mnist-standalone-classifier.rev
cargo run -p reverie-cli -- run target/mnist-standalone-classifier.rev --json
cargo run -p reverie-cli -- roundtrip target/mnist-standalone-classifier.rev --json
```

Use `--import-model-json PATH --model-output PATH` when the Q31 weights came
from another training stack. The input may either be a raw `weights`/`bias`
object or an object with a `model` field containing that shape; weights must be
`784 x 10` signed integers and bias must be 10 signed integers. Reverie wraps
that model as the same signed model-bundle kind used by exported Reverie
training audits, but records `provenance_kind: "external_import"` and signs the
source JSON fingerprint instead of claiming a training proof. `--verify-model`
checks the model shape, payload size, fingerprints, and import metadata; when
the referenced source JSON is still available, it also checks that file's hash
and model contents before accepting the bundle, reporting that status as
`source_model_json_checked` in JSON mode:

```json
{
  "format": "weights_bias_q31_json",
  "model": {
    "weights": [[784 rows of 10 Q31 integers]],
    "bias": [10 Q31 integers]
  }
}
```

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --import-model-json target/q31-linear-model.json \
  --model-output target/imported-q31-linear-model-bundle.json \
  --json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --verify-model target/imported-q31-linear-model-bundle.json --json
```

For a new labeled input, use `--sample-json PATH` instead of `--sample-audit`.
The sample JSON is a small object with `image_u8` containing 784 byte values
and `label` in `0..9`; optional `kind: "reverie_mnist_linear_q31_sample"` and
`schema_version: 1` fields are accepted. The resulting inference replay bundle
embeds the sample and records the source sample JSON fingerprint:

```text
{
  "kind": "reverie_mnist_linear_q31_sample",
  "schema_version": 1,
  "image_u8": [784 byte values],
  "label": 0
}
```

Use `--export-samples AUDIT --samples-output PATH` to export labeled samples
from a verified audit bundle into the same sample-set JSON shape accepted by
`--evaluate-model`. Add `--samples-limit N` to keep a small deterministic slice,
and use `--verify-samples PATH` to verify the sample-set fingerprint, shape, and
signed lineage proof. Each exported sample keeps its `audit_step` and
`source_sample_index`, and model evaluation rows preserve that lineage plus a
per-sample fingerprint. When the referenced source audit file is available,
`--verify-samples` also checks that each sample's pixels, label, and source
sample index match the named audit step:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --export-samples target/mnist-self-test-replay-bundle.json \
  --samples-output target/mnist-samples.json \
  --samples-limit 100 \
  --json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --verify-samples target/mnist-samples.json --json
```

Use `--evaluate-model MODEL --samples-json PATH` to run the signed model over a
batch of labeled samples without writing one inference bundle per sample. The
samples file can be either a JSON array of sample objects or an object with
`kind: "reverie_mnist_linear_q31_samples"` and a `samples` array. The evaluator
reuses the same Reverie forward/inverse inference path for each sample, then
reports accuracy, lowest margin, per-sample prediction rows in JSON, the
samples-file fingerprint, and the aggregate proof-cost shape:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --evaluate-model target/mnist-self-test-model-bundle.json \
  --samples-json target/mnist-samples.json --json
```

Add evaluation gates to make a model evaluation enforceable. For example,
`--eval-min-accuracy 99`, `--eval-min-margin 0`, `--eval-max-incorrect 0`, and
`--eval-require-label-coverage` print a gate report and return a nonzero exit
status when the signed model plus sample set does not satisfy the requested
deployment policy.

Add `--evaluation-output PATH` to write the model, embedded samples,
deterministic rows, proof-cost fields, gate policy, and gate result as a signed
evaluation bundle. `--verify-evaluation PATH` verifies the bundle fingerprint,
reruns every sample through the Reverie forward/inverse inference path, checks
the deterministic rows and proof-cost fields, and reapplies the stored gate
policy. When the referenced model bundle and samples file are available, it
also checks that the embedded model and samples still match those signed source
artifacts:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --evaluate-model target/mnist-self-test-model-bundle.json \
  --samples-json target/mnist-samples.json \
  --eval-min-accuracy 99 \
  --eval-min-margin 0 \
  --eval-max-incorrect 0 \
  --evaluation-output target/mnist-evaluation-bundle.json \
  --json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --verify-evaluation target/mnist-evaluation-bundle.json --json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --scan-evaluation target/mnist-evaluation-bundle.json \
  --evaluation-limit 5 \
  --json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-evaluation target/mnist-evaluation-bundle.json \
  --evaluation-row 0 \
  --inference-output target/mnist-evaluation-row-inference-bundle.json \
  --json
```

`--scan-evaluation PATH` reads the signed evaluation rows, ranks incorrect and
lowest-margin predictions, summarizes labels and confusions, and prints the
preserved `audit_step`, `source_sample_index`, and `sample_fingerprint` fields
for each ranked row. Use `--inspect-evaluation PATH --evaluation-row N` to
expand one signed row into a full deterministic inference audit: Reverie
reruns the embedded model and sample, verifies the stored row fields, prints
logits, attribution, lineage, and inverse restoration, and can save a replay
bundle with `--inference-output PATH` or a direct explanation card with
`--markdown-output PATH`. `--verify-inference` checks that replay bundle
against the referenced evaluation row when the evaluation bundle is available;
use `--markdown-output PATH` on that verification to render the signed
evaluation-row replay proof card.

Use `--compare-artifacts PATH...` to compare verified proof artifacts. It
accepts full training replay bundles, standalone model bundles, extracted step
bundles, sample-set bundles, inference bundles, and model evaluation bundles,
verifies each bundle's SHA-256 fingerprints, then reports actual JSON file bytes
alongside logical replay payload bytes, witness bytes, trace bytes, update
summary bytes, and forward/inverse recompute steps. Each artifact row includes
fingerprint coverage metadata for source lineage, computation payloads, proof
hashes, and model provenance. The JSON report also includes an
`ml_profile` block with aggregate model/sample/witness/trace/update bytes,
total forward and inverse recompute steps, total recompute steps, and ratios
such as trace-to-model and witness-to-model payload:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --json \
  --compare-artifacts \
  target/mnist-self-test-replay-bundle.json \
  target/mnist-self-test-model-bundle.json \
  target/mnist-samples.json \
  target/mnist-self-test-step-bundle.json \
  target/mnist-self-test-inference-bundle.json \
  target/mnist-evaluation-bundle.json
```

## `check`

Parses a source file and runs static checks without executing it.

```sh
cargo run -p reverie-cli -- check examples/units.rev
```

Options:

```text
--type NAME=TYPE
--legacy-janus
```

Use `--type` to annotate externally seeded variables for static type and unit
checking:

```sh
cargo run -p reverie-cli -- check examples/top_level_unit_mismatch.rev \
  --type 'speed=int<m/s>' --type 'distance=int<m>'
```

Duplicate `--type` names are rejected. With `--legacy-janus`, this check runs
after case-insensitive name normalization. Extra `--type` names are also
rejected when the external store is known, which catches typos such as
`--type nn=int` when the program expects `n`.

## `run`

Runs a source file forward from the supplied initial store.

```sh
cargo run -p reverie-cli -- run examples/fib.rev \
  --var n=7 --var i=0 --var a=0 --var b=1
```

Expected output:

```text
{a = 13, b = 21, i = 7, n = 7}
```

Options:

```text
--var NAME=VALUE
--vars-json PATH
--type NAME=TYPE
--engine tree|slot
--json
--proof-output PATH
--input VALUE
--output VALUE
--legacy-janus
```

`run` defaults to `--engine slot`, the compiled slot-indexed interpreter. Use
`--engine tree` to force the reference AST interpreter.

Use `--legacy-janus` for older Janus files that treat every semicolon as a
line comment marker and identifiers as case-insensitive. Seed and type names
are normalized the same way before checking or execution. The flag also enables
Jana-compatible update aliases, so upstream examples such as
`matrixmult_v1.0.ja` can run without weakening Reverie's strict default mode.

If the program contains Jana-style `show(x)` or `printf(...)` statements, those
observation lines are printed before the final store. They are separate from
the reversible `read`/`write` tapes. `printf` output preserves embedded newline
escapes, so `printf("%d\n", x)` prints exactly one newline before the store.
Use `--json` when another tool needs machine-readable output. JSON run results
use kind `reverie_run_result` and include `store`, `input`, `output`, and
`observations`; stack values use the same `{"stack": [...]}` shape accepted by
`--vars-json`. JSON run and reverse results also include `dataset_loops`,
`store_metadata`, `witness_store`, `witness_metrics`, `ml_profile`, and
`witness_proof`, so audit tooling can tell which loops are bounded by
dataset-shaped arrays or tensors, which final store entries are witness tapes,
how many statically known witness cells and payload bytes they account for,
whether the program is a model-shaped tensor/Q31 kernel, and the SHA-256
fingerprint of each witness value plus the aggregate witness-store proof
without reparsing the source. The `program`
block records the source path, exact source SHA-256, execution direction,
engine, and legacy-Janus mode, so a saved generic run result is tied to the
source text that produced it.

Use `--input` to seed values consumed by `read`. Use `--output` only when
running inverse I/O operations directly.

Use `--vars-json PATH` to seed larger stores from a JSON object whose keys are
variable names. This is the ergonomic path for model weights, image tensors,
and witness-sized test fixtures:

```json
{
  "x": 1,
  "matrix": [[1, 2], [3, 4]],
  "s": {"stack": [3, 2, 1]}
}
```

`--vars-json` may be repeated and may be combined with `--var`, but duplicate
seed names are rejected so command lines and seed files cannot silently
override an earlier value.
Seed values also provide static type information for plain ints, booleans,
stacks, and non-empty homogeneous arrays, so `--var flag=true` is enough for
the checker to reject integer arithmetic on `flag`. Use `--type` when a seed
needs units or an empty array's element type is otherwise ambiguous.
Heterogeneous array seeds and tape values, such as `--var xs=[1,true]` or
`--input [1,true]`, are rejected before execution because Reverie arrays are
homogeneous.
Ragged nested arrays, such as `--var matrix=[[1,2],[3]]`, are also rejected so
CLI seeds and tape values obey the same rectangular fixed-size array model as
source literals.
Before execution starts, `run`, `reverse`, `roundtrip`, and `scrub` also check
that every external store name reported by `explain` has a matching `--var`
seed. If one is missing, the command fails with a seed-template hint instead of
waiting for a runtime undefined-variable error. Extra `--var` names are also
rejected when the external store is known, which catches typos such as `nn=7`
when the program expects `n`.

## `reverse`

Runs the mechanically derived inverse from the supplied final store.

```sh
cargo run -p reverie-cli -- reverse examples/fib.rev \
  --var n=7 --var i=7 --var a=13 --var b=21
```

Expected output:

```text
{a = 0, b = 1, i = 0, n = 7}
```

Options:

```text
--var NAME=VALUE
--vars-json PATH
--type NAME=TYPE
--engine tree|slot
--json
--input VALUE
--output VALUE
--legacy-janus
```

`reverse` also defaults to `--engine slot`.

Use `--output` to seed the output tape produced by the forward run. When
reversing `read`, the consumed values are restored to the input tape.
Use `--json` for a machine-readable `reverie_reverse_result` object with the
same `store`, `input`, `output`, `observations`, `dataset_loops`,
`store_metadata`, `witness_store`, `witness_metrics`, and `program` fields as
`run --json`.

## `roundtrip`

Runs a source file forward, then runs the mechanically derived inverse from
that forward result. The command compares the restored store and tapes against
the initialized entry state, allowing source-declared storage to remain present
when the inverse has checked it back to its declared value. This is the generic
proof command for small reversible kernels:

```sh
cargo run -p reverie-cli -- roundtrip examples/increment.rev
```

Expected output includes:

```text
roundtrip: ok
checks: store=true input=true output=true
```

Options:

```text
--var NAME=VALUE
--vars-json PATH
--type NAME=TYPE
--engine tree|slot
--json
--input VALUE
--output VALUE
--legacy-janus
```

Use `--json` to emit a `reverie_roundtrip_result` proof. The proof payload
records the source SHA-256, engine, legacy-Janus mode, normalized type
annotations, dataset-loop metadata, initialized baseline state, forward state,
restored state, restoration checks, witness metrics, the static `ml_profile`,
and separate SHA-256 witness proofs for the forward and restored states. The
top-level fingerprint signs the whole proof payload, including the ML profile
when present, while the `fingerprints` block also signs the baseline, forward,
restored, and witness-proof subpayloads. Use
`--proof-output PATH` to save the same proof JSON while still printing the
human summary or JSON report:

```sh
cargo run -p reverie-cli -- roundtrip examples/reversible_inference_trace.rev \
  --proof-output target/reversible-inference-roundtrip-proof.json
```

## `verify-roundtrip`

Verifies and replays a saved `reverie_roundtrip_result` proof. Verification
checks the proof schema, top-level payload fingerprint, subpayload fingerprints,
forward/restored witness proof fingerprints, the referenced source SHA-256, and
then reruns the referenced source from the saved initialized baseline. The
verification passes only when the recomputed proof fingerprint equals the saved
proof fingerprint. Markdown verification cards include an `ML Profile` section
when the saved proof carries one. JSON verification reports also expose
`ml_profile_present`, `ml_profile_fingerprint`, `replay_ml_profile_fingerprint`,
and the verified `ml_profile`, including its `replay_cost` estimate, with
`checks.ml_profile_schema` and `checks.ml_profile_replayed` proving that the
saved profile schema is supported and the replayed source reproduced the same
profile:

```sh
cargo run -p reverie-cli -- verify-roundtrip \
  target/reversible-inference-roundtrip-proof.json
```

Options:

```text
--json
--markdown-output PATH
```

Use `--markdown-output PATH` to save a compact review card with the verdict,
source hash, proof/replay fingerprints, witness budget, signed subpayloads, and
per-check pass/fail table.

## `invert`

Prints the mechanically derived inverse as parser-compatible Reverie source.

```sh
cargo run -p reverie-cli -- invert examples/fib.rev
```

Use this when you want to inspect what `reverse` will execute.

Options:

```text
--type NAME=TYPE
--legacy-janus
```

## `fmt`

Parses a source file and prints canonical parser-compatible Reverie source.
This is useful for reviewing generated inverses, normalizing Janus-style input,
or checking that examples stay readable.

```sh
cargo run -p reverie-cli -- fmt examples/fib.rev
```

Options:

```text
--check
--write
--legacy-janus
```

Use `--check` to fail when the file is not already in canonical form:

```sh
cargo run -p reverie-cli -- fmt --check examples/fib.rev
```

Use `--write` to rewrite the file in place:

```sh
cargo run -p reverie-cli -- fmt --write examples/fib.rev
```

`--check` and `--write` are mutually exclusive.
`fmt` preserves the parsed program, not comments or original whitespace.

## `explain`

Parses, checks, and summarizes the reversible constructs used by a source file.
This is a quick way to understand whether a program relies on loops, indexed
mutation, procedures, tape I/O, stacks, or other language features before you
run it.

```sh
cargo run -p reverie-cli -- explain examples/fib.rev
```

Example output:

```text
file: examples/fib.rev
status: reversible program checks
globals: 0
procedures: 0
statements: 8
expressions: 14
features:
- reversible conditionals
- reversible loops
- reversible updates
- swaps
safety checks:
- no additional indexed or arithmetic runtime checks
external store:
- a: inferred at runtime; add --type a=TYPE for units
- b: inferred at runtime; add --type b=TYPE for units
- i: inferred at runtime; add --type i=TYPE for units
- n: inferred at runtime; add --type n=TYPE for units
run template: reverie run examples/fib.rev --var a=0 --var b=0 --var i=0 --var n=0
declared store:
- none
inverse: reverie invert examples/fib.rev
```

Use `--json` for a machine-readable summary with the same counts, feature
names, safety checks, per-safety-check counts, dataset-shaped loop metadata,
store templates, and command templates:

```sh
cargo run -p reverie-cli -- explain --json examples/fib.rev
```

Use `--ml` to add an ML-oriented profile for tensor programs. The text output
adds a compact `ml profile` section, and `--ml --json` adds an `ml_profile`
object with the profile schema, `goal_fit`, capability labels, tensor/witness
payload bytes, a `replay_cost` block for static forward/inverse roundtrip work
and replay payload bytes, tensor and witness update counts, Q31 builtin counts,
and non-injective signal calls such as `normalize_q31`, `clamp_q31`, `relu`,
`argmax`, `runner_up`, `top2_margin`, `rank_of`, `top_k_indices`,
`top_k_values`, or `top_k_contains`, so
reviewers can see whether lossy signals are captured as explicit state:

```sh
cargo run -p reverie-cli -- explain --ml examples/mnist_mlp_witness.rev
cargo run -p reverie-cli -- explain --ml --json examples/mnist_mlp_witness.rev
```

When you provide `--type` annotations, `explain` uses them to choose seed
templates. Array templates are shell-quoted once, including nested arrays such
as `--type 'm=array<array<int>>'` producing `--var m='[[0]]'`.
The generated run template also repeats matching `--type` annotations, so a
copy-pasted command preserves unit checks such as `--type 'speed=int<m/s>'`.
Template paths are also shell-quoted when needed, so files under directories
with spaces can still be copied directly into a terminal.
For programs with globals or source declarations, `explain` also prints a
`declared store` section. Those names default from the program, but the
declared override template shows valid optional `--var` values for changing
the initial store.
When an `iterate` bound uses `len(...)` or `size(...)`, the JSON summary
includes `dataset_loops` entries with the loop index and the array/tensor names
that determine the bound.

Options:

```text
--type NAME=TYPE
--ml
--json
--legacy-janus
```

## `scrub`

Builds a forward timeline and opens the interactive scrubber TUI.

```sh
cargo run -p reverie-cli -- scrub examples/fib.rev \
  --var n=7 --var i=0 --var a=0 --var b=1 \
  --watch a --watch b --watch i
```

Keys:

```text
left/right or h/l: scrub
home/end: jump to ends
digits+enter: jump to step
q/esc: quit
```

Use `--dump` for non-interactive output:

```sh
cargo run -p reverie-cli -- scrub examples/fib.rev \
  --var n=2 --var i=0 --var a=0 --var b=1 \
  --watch a --watch i --dump
```

Options:

```text
--var NAME=VALUE
--vars-json PATH
--type NAME=TYPE
--watch NAME
--dump
--input VALUE
--output VALUE
--legacy-janus
```

Use `--input` and `--output` with `scrub` the same way as `run` when a
timeline includes reversible tape I/O.
Duplicate `--watch` names are rejected, and each watched name must appear in
at least one timeline frame. This catches typos such as `--watch nn` when the
program stores `n`.

## Seed Values

`--var` values currently support signed `i64` integers, booleans, nested
arrays, and integer stacks:

```sh
--var n=7
--var x=-1
--var flag=true
--var 'xs=[10,20,30]'
--var 'flags=[true,false,true]'
--var 'matrix=[[1,2],[3,4]]'
--var s=nil
--var 's=stack[3,2,1]'
```

Arrays may contain whitespace:

```sh
--var 'xs=[10, 20, 30]'
```

The same value grammar is used by `--input` and `--output`, including the
homogeneous-array check.

`--vars-json` files use native JSON values instead of the command-line literal
grammar. Top-level JSON must be an object mapping Reverie identifiers to signed
integer, boolean, or nested array values. Stack seeds use an explicit object:

```json
{
  "n": 7,
  "image": [0, 255, 0],
  "weights": [[2147483648, 0], [0, -2147483648]],
  "s": {"stack": [3, 2, 1]}
}
```

Floating-point JSON numbers are rejected. Quantized Q31 model values should be
written as exact signed integers by the exporter before being passed to
`--vars-json`.

The CLI validates that every seed name is a Reverie identifier.

## Type Annotations

`--type` annotations use the same type grammar as source annotations:

```sh
--type 'distance=int<m>'
--type 'time=int<s>'
--type 'speed=int<m/s>'
--type 'flag=bool'
--type 'flags=array<bool>'
--type 'xs=array<int<m>>'
--type 'weights=tensor<int,2,3>'
--type 's=stack'
```

`--type` does not create a runtime value. It only tells the checker how to
interpret an externally seeded variable. Tensor annotations also provide the
static shape required by tensor builtins and are checked against the seeded
nested array value.

## Exit Behavior

Successful commands exit with status 0. Syntax, static check, and span-carrying
runtime errors render `ariadne` diagnostics and exit non-zero. Runtime failures
that cannot be attached to source still print a concise CLI error.
