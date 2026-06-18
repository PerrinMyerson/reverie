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
- README GIFs generated from real scrubber and MNIST/Q31 reversal runs

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

## Auditable ML Kernels

Done:

- array-backed `tensor<int, ...>` types with static shape checking
- Q31 tensor builtins for linear-model kernels, including `vecmat_q31`,
  `outer_q31`, `scale_q31`, `relu`, `relu_mask_q31`, `argmax`, and
  `runner_up`, `top2_margin`, `rank_of`, `top_k_indices`, `top_k_values`,
  `top_k_contains`, and `argmax_eq`
- reversible MNIST-shaped linear classifier inference in
  `examples/mnist_identify.rev`
- reversible MNIST-shaped single-sample training step in
  `examples/mnist_reversible_step.rev`
- first-class `witness<tensor<...>>` tapes for logits, errors, predictions,
  correctness, hidden activations, masks, and backprop intermediates
- batched dataset-shaped iteration over witness tapes in
  `examples/mnist_witness_tape_loop.rev`
- reversible MLP witness pattern in `examples/mnist_mlp_witness.rev`
- invertible additive coupling layer pattern in
  `examples/invertible_coupling.rev`
- invertible Q31 triangular residual layer pattern in
  `examples/triangular_residual.rev`, with an independent zero-witness
  replay checker and Markdown proof card
- reversible Q31 preprocessing block in `examples/reversible_preprocess.rev`
  that centers and permutes feature buffers while preserving raw inputs and
  requiring no witness tape
- reversible Q31 normalization recipe in `examples/reversible_normalize.rev`
  using `normalize_q31` for replayable `(raw - mean) */ inv_scale`
  preprocessing
- reversible Q31 clipping recipe in `examples/reversible_clamp.rev` using
  `clamp_q31` plus an explicit residual witness so saturated inputs remain
  exactly auditable
- reversible bit-packing recipe in `examples/reversible_pack.rev` using
  `pack_bits` and `unpack_bits` to keep compact scalar features tied to an
  exact decoded witness
- end-to-end reversible Q31 inference trace in
  `examples/reversible_inference_trace.rev`, covering preprocessing, logits,
  top-class rankings, top-logit values, prediction, runner-up class, top-two
  margin, correctness, witness bytes, and reverse restoration
- MNIST runner that records speed, peak RSS when available, model/sample/trace
  payload bytes, proof bytes, and forward/inverse recompute counts
- artifact comparison `ml_profile` that aggregates model/sample/witness/trace
  payload bytes, update-summary bytes, recompute steps, and trace/witness
  ratios across proof artifacts
- signed training audit, model, sample-set, step, inference, and evaluation
  bundles with verifiers that rebuild fingerprints and replay proofs before
  accepting artifacts
- pipeline training-step selection strategies (`explicit`, `lowest-margin`,
  `largest-update`, and `top-suspicious`) that record the requested step,
  selected step, scan-derived strategy target, ranked-list membership, and a
  traceability gate before accepting the debug update
- full training-update ledger fingerprints for selected steps, covering
  nonzero bias deltas, nonzero weight deltas, and a cause ledger that ties the
  update back to the sample, witnesses, logits/errors, active pixels, and top
  deltas
- full training-audit lineage verification that rebuilds the final model from
  the witness trace and gates a hash-chained lineage ledger with transition and
  final-chain fingerprints
- pipeline evaluation-row selection strategies (`explicit`, `lowest-margin`,
  and `top-incorrect`) that tie the inspected reversible inference bundle to
  model-evaluation scan evidence before accepting the prediction explanation
- standalone inference verification that checks referenced model,
  training-audit, sample, or evaluation source artifacts when available
- inference explanation contracts that tie predictions to logits, attribution
  reconstruction, reverse replay, recomputed proof/result checks, and signed
  source inputs
- checked inference trace profiles that compare native report ledgers, replay
  verification ledgers, independent Q31 reference ledgers, replay status, and
  source checks for deterministic inference auditability
- external Q31 model import into signed model bundles for deterministic
  Reverie inference/evaluation over models trained elsewhere
- checked model-import guide for quantizing external weights into Reverie Q31
  tensors and verifying signed inference/evaluation artifacts
- dependency-light JSON exporter for MNIST-shaped linear weights, including
  common `weights[digit][pixel]` state-dict layout transposition into Reverie's
  checked `weights_bias_q31_json` import format
- optional PyTorch `state_dict` extraction helper that uses
  `torch.load(..., weights_only=True)` by default, requires explicit
  `--allow-unsafe-pickle` for trusted legacy checkpoints, and emits
  exporter-readable JSON for `linear.weight`/`linear.bias`
- generic CLI `--vars-json` seed files for `run`, `reverse`, and `scrub`, so
  large Q31 tensor stores can be loaded from exact integer JSON instead of
  enormous `--var` command lines
- dependency-light MLP seed exporter for `examples/mnist_mlp_witness.rev`,
  covering native `w1`/`b1`/`w2`/`b2`, PyTorch-style `fc1`/`fc2`
  state-dict keys, `.npz` members, Q31 quantization, and optional provenance
  metadata sidecars
- dependency-light MLP witness checker that independently recomputes the
  two-sample Q31 trace and compares hidden/logit/error/backprop witness tapes
  plus final layer weights against `reverie run --json` output, while reporting
  witness bytes, trace bytes, replay bytes, recompute steps, recomputed
  layer-update bytes, and a human-readable activation/mask witness proof card
- native NumPy `.npz` reader for the Q31 exporter, covering numeric C-order
  `.npy` weight and bias arrays without adding a NumPy runtime dependency
- dependency-light Q31 inference reference checker for imported models and
  single `image_u8` samples or exported signed sample-set rows, mirroring pixel
  scaling, `vecmat_q31`, wrapping logit accumulation, first-maximum `argmax`,
  top-logit ordering, and deterministic attribution reconstruction for the
  winning logit and winner-vs-runner-up margin, plus all-samples reference
  evaluation summaries with accuracy, per-label summaries, low-margin rows, and
  incorrect rows
- pipeline gate for the inspected inference sample that requires Q31 reference
  prediction, correctness, margin, active-pixel count, and attribution
  reconstruction to match native Reverie inference
- checked MNIST ML profile validator for runner JSON and artifact-comparison
  reports, covering elapsed throughput, peak RSS, payload byte sums, witness
  trace proof costs, MLP witness proof-cost reports, Q31 reference
  inference/evaluation reports with attribution and low-margin rows, training
  audit scan/step debug reports, standalone step replay verification reports,
  native signed inference/evaluation reports, inference/evaluation
  verification reports, zero-witness triangular residual and reversible preprocessing reports,
  end-to-end reversible inference trace reports, and recompute ratios
- persisted Markdown summaries that present speed, peak RSS, trace payload,
  proof payloads, artifact rows, MLP witness and triangular residual rows, Q31 reference rows,
  training-audit debug rows, native inference/evaluation rows, an ML
  capability map, a Recompute Frontier table, observed/10x/100x Scaling
  Projection rows, and a V6 scorecard with reverse-check cost from validated ML
  profile JSON reports
- one-command local MNIST ML audit pipeline that generates the self-test
  training audit bundle, signed model/sample bundles, step and inference
  replay proofs, full training-audit lineage verification, native evaluation
  reports, Q31 reference checks, MLP witness replay proof costs and review card, zero-witness invertible coupling, triangular residual, and reversible
  preprocessing replay proof costs and review cards, reversible inference trace replay proof
  costs and a review card with contribution-ledger fingerprints,
  artifact comparison JSON, a manifest, checked Markdown profile, and gated
  machine-readable `pipeline-summary.json` with speed, peak RSS, payload-ratio,
  recompute-balance, configurable reverse-check elapsed-ratio budgets, replay,
  source-check, training-lineage replay, training-step and evaluation-row
  selection traceability, inspected-inference and evaluation-summary
  native/reference parity, reversible
  inference trace replay, inference
  explanation contracts with contribution-ledger fingerprints, MLP witness,
  invertible coupling, triangular residual, reversible preprocessing, and reversible inference trace metrics, a V1-V6 capability
  map, a replay-mode recompute frontier with retained-state bytes,
  witness/trace/update/state bytes, and
  forward/inverse recompute steps,
  measured observed/10x/100x replay projections for training traces, batch
  evaluation, and MLP witness replay, a V6 scorecard, a north-star readiness
  report for reversible inspectable deterministic ML kernels, a claim-to-gate
  evidence matrix, SHA-256 evidence indexes for generated files, and
  human-readable no-witness, direct native/evaluation-row inference, and
  verification cards that bind transforms, attribution, reverse restoration,
  source checks, and replay payload costs
- first-class checker support for the pipeline summary and manifest acceptance
  artifacts
- explicit reversible ML audit-contract gate requiring training trace, model,
  samples, step replay, inference replay, evaluation replay, nonzero witness
  and trace accounting, and balanced forward/inverse recompute evidence
- release-verification receipt with a fingerprinted goal/non-goal contract,
  allowed ML-audit claim set, reviewer replay transcript binding, speed+memory
  benchmark binding, per-claim proof rows backed by the pipeline claim matrix,
  freshness checks against the gate-embedded goal contract, verifier-source
  digests, required reviewer replays for benchmark, roundtrip, training-step,
  MLP witness, invertible coupling, triangular residual, reversible preprocessing, and inference
  trace proof cards, per-claim replay artifact coverage for saved proof cards,
  and audit-packet artifact summary

Next:

- more model-family importers beyond the current MNIST-shaped linear
  classifier handoff
- more invertible layer/library patterns beyond the current additive coupling
  and triangular residual examples
- more domain-specific reversible input-processing recipes before deterministic
  inference

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
