# Rewinding learning: what reversibility buys an ML system

This note collects what Reverie can now do with trained models, what is new
about it, and what is not. Every number below is reproducible from scripts in
this repository:

```sh
maturin develop --release
pip install scikit-learn
python3 scripts/exact_unlearning_experiment.py \
  --json-output docs/research/exact-unlearning-results.json \
  --markdown-output docs/research/exact-unlearning-results.md
python3 scripts/reversibility_report.py \
  --json-output docs/research/reversibility-report.json \
  --markdown-output docs/research/reversibility-report.md
```

Results: [exact-unlearning-results.md](exact-unlearning-results.md) and
[reversibility-report.md](reversibility-report.md). Each script also has a
`--self-test` that runs in CI.

## 1. The idea: store the errors, recompute the activations

Plain SGD destroys information. Each step overwrites the weights with
`W - lr * g(W, x)`, and without `g` there is no way back. The usual ways to
get back are checkpoints, per-step weight deltas, or storing every
intermediate value (Bennett's "keep the garbage").

With batch size 1, every dense-layer gradient is a rank-one outer product,
`dW_l = outer(a_{l-1}, delta_l)`: an input-side factor (the sample or the
previous layer's activation) times an output-side factor (the
back-propagated error). Run backward, the layers come back in forward order:

1. restore `W_1` from the sample `x` (part of the dataset) and `delta_1`;
2. recompute `a_1 = relu(x W_1 + b_1)` from the restored `W_1`;
3. restore `W_2` from `a_1` and `delta_2`, and so on.

So the only things a step must leave behind are the output-side factors
`delta_1 ... delta_L`, which is the sum of the layer widths per example. Every
activation is recomputed on the way back. This is the mirror image of gradient
checkpointing, which stores activations so it can recompute the backward
pass. Here we store backward-pass values so we can recompute the forward pass.

`reverie.lab.mlp_sgd_source` writes this as a Reverie program. Reverie's
checker proves it reversible: each statement only adds or subtracts a function
of other variables, and every scratch value is uncomputed within the step. The
witness is `err_tape` (output layer) and `delta_tape` (hidden layer).
Arithmetic is Q31 fixed point, so it is bit-exact. The program's weights match
an ordinary CPython implementation of the same SGD bit for bit (checked in the
report and in the tests).

**On the digits dataset (64 → 32 → 10 MLP, 600 samples × 3 epochs = 1,800
steps, 94.8% held-out accuracy):**

| what you keep to rewind exactly | values per step | total |
|---|---:|---:|
| a checkpoint per step, or per-step weight deltas | 2,410 | 33.1 MiB |
| every intermediate (Bennett trace) | 180 | 2.47 MiB |
| **error factors only (this work)** | **42** | **0.58 MiB** (0.24 MiB coded) |

Rewinding all 1,800 steps back to the initial weights takes 0.15 s, the same
as training forward, and it is exact.

## 2. Exact unlearning, bit-identical to retraining

To remove sample `u`, rewind to just before `u` was first used, switch off
every step that used it, and replay the rest. Because the arithmetic is
exact, the result is **bit-identical to retraining from scratch without
`u`**. That held for 12 of 12 deletions in the experiment, and for every
sample in the unit tests. The weights did change in every case, so each
deletion had an effect.

How this relates to existing work:

- Exact unlearning normally means retraining, or isolating data so only a
  slice is retrained (SISA). Rewind-based methods return to a stored
  checkpoint and give certified *approximate* guarantees (Mu & Klabjan,
  2025). "Unlearning at Scale" (2025) rolls back recent steps exactly, but
  only by storing micro-checkpoints or per-step deltas.
- Maclaurin et al. (2015) reversed SGD with momentum exactly by keeping the
  bits that finite precision destroys. Their goal was hyperparameter
  gradients; they did not apply it to deletion.
- The literature search turned up no method that is exact and
  checkpoint-free and runs on integer arithmetic. The error-factor witness
  is the piece that makes it cheap.

**Where it does not win.** The rewind costs about `2(T - t)` steps, against
`T` for retraining. With several epochs every sample appears early, so in this
experiment unlearning (0.26–0.29 s) was *slower* than retraining (0.15 s). The
honest baseline is also sharper than "a checkpoint per step". Integer training
is deterministic, so a checkpoint every 58 steps plus forward replay reaches
any step exactly with the same memory as the witness, and it is faster for
unlearning.

The witness wins when you need the states **in reverse order**: reverse-mode
differentiation through training (Maclaurin's hypergradients), scrubbing a
run backward, or auditing it step by step. For 1,800 steps it stores 78k
values with no recomputation. The best √T checkpoint schedule stores 205k
values (2.6× more) and replays every step again. The advantage scales as about
`2P / (w·√T)` for `P` parameters and witness width `w`, so it grows for
wide-and-shallow models and shrinks for very long runs.

## 3. Proofs without a tolerance

The whole training run has a fingerprinted roundtrip proof (`Program.roundtrip`
and `reverie.verify`), and it verifies. That matters because floating-point
proof-of-learning has to compare replayed weights within a tolerance. Spoofing
attacks exploit that tolerance (Zhang et al. 2021; Fang et al. 2023), and
verifiable-training work goes to great lengths to control GPU
nondeterminism (Srivastava, Arora & Boneh 2024). In integer Reverie the replay
must match bit for bit, so there is no tolerance to attack. The witness also
lets a verifier walk a run *backward* to its published initial weights.

This is an argument plus a working proof object, not a security analysis.
Fabricating a different batch sequence that yields identical integer updates
is far harder than landing within a float tolerance, but it is not proven
impossible.

## 4. Witness ledgers: where a network throws information away

A reversible program's witness is exactly what an ordinary program would have
erased, so measuring it is an exact account of information loss, layer by
layer and unit by unit. `reverie.ledger` measures it in coded bits: zig-zag
plus Elias-gamma per value, an achievable size and therefore an upper bound on
what was erased. It also converts that to Landauer's minimum erasure heat.
Information-bottleneck analyses of deterministic networks struggle because
mutual information is infinite or ill-posed for continuous activations
(Amjad & Geiger; Geiger's review). Quantised networks make it finite. Here it
is also *constructive*: the witness says which unit discarded which value.

`reverie.lab.erasure_profile` runs the trained digits model through an
inference program whose only witness is each ReLU's negative part
(`pre = act + neg`). On 500 held-out digits, and exactly reversible:

- the ReLU layer discards 727k coded bits and keeps 260k; units are zeroed on
  74% of inputs;
- the logits the argmax readout collapses are worth 282k bits, so in this
  model the hidden ReLUs, not the readout, are where most information leaves;
- one unit is dead (always zeroed);
- **ablating the 4 units that discard the most costs nothing (94.8% → 94.8%);
  ablating the 4 that discard the least costs 20 points (→ 74.6%).**

The last point echoes activation-frequency pruning, measured here as exact
discarded information rather than a heuristic.

**Negative result.** We expected a sample's witness size to work as a
difficulty score, like EL2N or GraNd (Paul et al. 2021). It does not: the
Spearman correlation with final misclassification is −0.06 (epoch 1) and 0.08
(last epoch). The coded length follows the magnitude of every error
component, so each SGD step destroys roughly the same ~1.1 kbit whatever the
example.

## 5. What reversibility costs, measured fairly

The Jana benchmark times whole processes, so it mostly measures startup.
`scripts/reversibility_report.py` measures inside one process:

- **Backward costs the same as forward.** The reverse/forward ratio is
  0.99–1.01 on the scaled workloads.
- **Engines.** The slot engine is 1.05–3.6× faster than the tree walker.
- **Against plain CPython running the same algorithm:**
  - level on scalar loops (Fibonacci, 1.05×);
  - 3.7–6× faster on tensor workloads (coupling stacks, SGD);
  - while also keeping everything needed to undo.
- **Memory to travel back in time.** A snapshot-per-step debugger keeps
  7–10,000× more values than the reversible final state. For example, 1.9M
  values against 4k for 20 SGD steps.
- **Bug found.** The report caught a performance bug in the new Python
  binding: converting large stores cost about 0.5 µs per value. It is now
  5.6× faster.

## 6. Language and tooling changes made along the way

- **`x *= c` for odd constants.** Multiplying by an odd number is a bijection
  mod 2^64, so the inverse multiplies by `c⁻¹ mod 2^64` (computed by Newton
  iteration). Even or non-constant factors are rejected with a diagnostic.
  This gives Reverie exact integer scaling without a witness.
- **Generated-program fuzzing.** New property tests generate whole programs
  as source text, with nested `if`/`fi`, `from`/`until` loops inside
  `local`/`delocal`, cross-variable updates, swaps and `*=`. They check:
  - the program parses and checks;
  - both engines agree;
  - it runs forward and backward back to its inputs;
  - running the inverted program forward matches running backward;
  - the formatter round-trips.

  10,000 cases passed. The existing generators covered straight-line code
  only.
- **Bug fixed: `verify-roundtrip` rejected valid proofs.** It compared the
  replayed ML profile by raw JSON equality. `serde_json`'s default float
  parser can land one ulp off, so proofs whose profile held such a float
  (here `1.6744927793036237`) failed verification. It now compares the
  normalized forms the fingerprints already use. This affected the CLI too.

## 7. Next experiments worth running

1. **Hypergradients through training.** This is the setting where the witness
   beats checkpoints. Reverse the digits run while accumulating
   `d(validation loss)/d(lr)`, as Maclaurin et al. did, but exactly and with
   O(width) memory per step.
2. **Counterfactuals that hold the discarded information fixed.** In an
   invertible network, edit an internal state and run backward with the
   *same* witness. The result is a real input that differs only in what the
   network keeps. This would answer the concern that activation patching
   creates states no input could produce (Grant et al. 2025): every such
   counterfactual is a genuine input, and the witness says when an edit cannot
   be realised (for example, a unit made positive while its witness records a
   negative pre-activation).
3. **Automatic witnesses.** The rule used by hand in `mlp_sgd_source` (keep
   output-side factors, uncompute everything else, restore layers in forward
   order) is mechanical. It could be a compiler pass that turns an
   irreversible training step into a minimal-witness reversible one.
4. **Tolerance-free proof-of-learning.** Turn §3 into a protocol: publish
   `(W_0, W_T, witness fingerprint)` and let a verifier spot-check
   random segments forward or backward.
5. **Static proofs of the backward conditions.** SMT checking of loop exit
   conditions and `fi` assertions, so a backward run cannot trap halfway.

## Sources

- Maclaurin, Duvenaud & Adams, [Gradient-based Hyperparameter Optimization through Reversible Learning](https://arxiv.org/abs/1502.03492) (ICML 2015)
- MacKay, Vicol, Ba & Grosse, [Reversible Recurrent Neural Networks](https://arxiv.org/pdf/1810.10999) (2018)
- Jacobsen, Smeulders & Oyallon, [i-RevNet: Deep Invertible Networks](https://arxiv.org/abs/1802.07088) (ICLR 2018)
- Bourtoule et al., [Machine Unlearning (SISA)](https://arxiv.org/abs/1912.03817) (IEEE S&P 2021)
- [Unlearning at Scale: Implementing the Right to be Forgotten](https://arxiv.org/pdf/2508.12220v1) (2025)
- Mu & Klabjan, [Descend or Rewind? Stochastic Gradient Descent Unlearning](https://arxiv.org/pdf/2511.15983v1) (2025)
- Jia et al., [Proof-of-Learning](https://arxiv.org/abs/2103.05633) (2021); Zhang et al., ["Adversarial Examples" for Proof-of-Learning](https://arxiv.org/abs/2108.09454); Fang et al., [Proof-of-Learning is Currently More Broken Than You Think](https://arxiv.org/abs/2208.03567)
- Srivastava, Arora & Boneh, [Optimistic Verifiable Training by Controlling Hardware Nondeterminism](https://arxiv.org/html/2403.09603v3) (2024)
- [Verification of Machine Unlearning is Fragile](https://arxiv.org/html/2408.00929) (2024)
- Amjad & Geiger, [Learning Representations for Neural Network-Based Classification Using the Information Bottleneck Principle](https://arxiv.org/html/1802.09766v5); Geiger, [A Critical Review of Information Bottleneck Theory](https://arxiv.org/html/2105.04405v1); Lorenzen et al., [Information Bottleneck: Exact Analysis of (Quantized) Neural Networks](https://arxiv.org/abs/2106.12912)
- Tkachenko, [Thermodynamic bounds on energy use in Deep Neural Networks](https://arxiv.org/html/2503.09980v3) (2025)
- Grant et al., [Addressing divergent representations from causal interventions on neural networks](https://arxiv.org/html/2511.04638v4) (2025)
- Paul, Ganguli & Dziugaite, Deep Learning on a Data Diet (EL2N/GraNd scores) (NeurIPS 2021)
- Yokoyama & Glück, [A Reversible Programming Language and its Invertible Self-Interpreter](https://dl.acm.org/doi/pdf/10.1145/1244381.1244404) (Janus)
