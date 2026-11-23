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
python3 scripts/prompt_recovery_study.py \
  --json-output docs/research/prompt-recovery.json \
  --markdown-output docs/research/prompt-recovery.md
python3 scripts/rollback_netcode_benchmark.py \
  --json-output docs/research/rollback-netcode.json \
  --markdown-output docs/research/rollback-netcode.md
python3 scripts/saga_migration_study.py \
  --json-output docs/research/sagas-and-migrations.json \
  --markdown-output docs/research/sagas-and-migrations.md
python3 scripts/dna_study.py \
  --json-output docs/research/dna.json \
  --markdown-output docs/research/dna.md
```

Results: [exact-unlearning-results.md](exact-unlearning-results.md),
[reversibility-report.md](reversibility-report.md),
[prompt-recovery.md](prompt-recovery.md),
[rollback-netcode.md](rollback-netcode.md),
[sagas-and-migrations.md](sagas-and-migrations.md) and [dna.md](dna.md). Each script also has a
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
erased: the *intermediate values* each operation overwrites. `reverie.ledger`
measures it in coded bits (zig-zag plus Elias-gamma per value) and converts
that to Landauer's minimum erasure heat.

> **Correction (validated in §7).** An earlier version of this note called the
> witness "an exact account of information loss." That is wrong for
> information about the *input*. A per-operation witness is only an upper
> bound, and it can be loose by orders of magnitude. In the language model
> of §7, each ReLU erases about 1,100 bits per prompt, yet the hidden state
> after the ReLU still identifies every prompt exactly: the information about
> the prompt actually lost is **0 bits**. Other units carry what one unit
> zeroes.
What the ledger *does* measure exactly is how much each unit erases of its
own pre-activation, which is a property of the unit rather than of the input.

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

The last point echoes activation-frequency pruning: units that erase most of
their pre-activations contribute least.

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

## 7. Can you get the prompt back from the answer?

`scripts/prompt_recovery_study.py` answers this exactly for a small language
model ([prompt-recovery.md](prompt-recovery.md)). The model is a Bengio-style
neural n-gram LM: embed the last 5 characters, one ReLU layer, softmax
readout. It runs over an 8-character alphabet in fixed point, at 1.01
bits/char held out. All 8⁵ = 32,768 prompts are enumerated, so each number is
an exact count, not an estimate. The reported quantity is H(prompt |
observable): the bits still missing after seeing the observable, which is
also the smallest witness that recovers the prompt.

| observable (16-bit fixed point) | missing bits, random prompt (of 15.0) | missing bits, natural prompt (of 7.4) |
|---|---:|---:|
| hidden state, or the 8 logits | **0** (every prompt unique) | **0** |
| greedy answer, 1 token | 12.6 | 4.8 |
| greedy answer, 4 tokens | 9.9 | 3.3 |
| greedy answer, 12 tokens | 9.9 | 3.3 |

What this validates:

1. **From the answer alone, no.** A greedy answer reveals about 5 of a random
   prompt's 15 bits, and about 4 of a natural prompt's 7.4, and then stops:
   tokens 5–12 add nothing, because greedy decoding falls into a fixed
   continuation. The 32,768 prompts produce only 89 distinct 12-token
   answers. This is a hard ceiling that no inversion method can beat on this
   model, and it explains why answer-only methods such as output2prompt need
   many *different* queries, as the bits-per-query analysis of Kaneko &
   Baldwin (2025) predicts.
2. **From the internals, yes, completely.** The post-ReLU hidden state, and
   even the 8 logits, identify every prompt. That matches the injectivity
   result for transformers (Nikolaou et al., ICLR 2026) and the logit-leakage
   work (Morris et al. 2024; Finlayson et al. 2024) on a model small enough to
   check exhaustively.
3. **Quantisation barely helps privacy.** Hidden states stay injective down to
   3 fractional bits. At 2 bits they lose 0.002 bits, and at 1 bit only 0.09
   bits of 15. That is consistent with the follow-up finding of only "a small
   number of collisions" at 4-bit (2511.14808), and here it is exact.
4. **The answer-plus-witness inversion you asked about works, at the
   minimum.** Store the prompt's index among all prompts that give the same
   answer: 10.4 bits per prompt for a 12-token answer, within 0.6 bits of the
   9.9 bits of entropy. With that, answer plus witness returns the exact
   prompt. A per-operation reversible implementation would carry about 1,100
   bits per prompt of ReLU witness alone, roughly 100× more.

**A practical audit that follows from this.** If a secret sits in a known
slot of a prompt template (an account code, a PIN, a short key), the space
of secrets can be enumerated even for a real LLM: one forward pass per
candidate. That yields an exact, worst-case *anonymity set* per observable.
In the toy model, a 2-character secret is uniquely identified 100% of the
time from the hidden state or logits, at every precision. From a 3-token
answer it is identified uniquely 5–8% of the time, with a mean anonymity set
of 16–22 of 64. Canary *exposure* (Carlini et al., The Secret Sharer,
2019) ranks candidates by perplexity to measure memorisation from training.
This is the inference-time counterpart, and with greedy decoding it is exact
rather than statistical.

## 8. Does exact rewinding scale to LLM training? (No, not this way.)

The error-factor witness stores `w` values per token per step, where `w` is
the sum of the output widths of the layers, against `P` parameters for a
checkpoint. So it only beats even a checkpoint per step below `P / w` tokens
per step. In the digits experiment that break-even is 57 tokens per step,
which is why a checkpoint every 58 steps matched it.

For a LLaMA-7B-shaped model, the break-even is **4,841 tokens per step**. At
a pretraining batch of 4M tokens the witness is **826× larger** than a full
checkpoint. For LoRA (r = 16 on q and v) the break-even is 32 tokens per
step. The idea fits online and small-batch learning, not LLM training.

At LLM scale, Maclaurin et al.'s route is the right one. Momentum SGD can be
reversed while storing only the bits the momentum decay destroys, about
log₂(1/β) ≈ 0.15 bits per parameter per step at β = 0.9. That is independent
of batch size and roughly 100× smaller than a 16-bit checkpoint. Reverie's
`*=` handles the invertible half of a fixed-point decay (multiply by 7, say),
and only the rescaling (divide by 8) loses information.

## 9. What is actually novel and useful here

After checking each idea against the literature, here is what holds up:

| idea | status |
|---|---|
| exact checkpoint-free unlearning via the error-factor witness | **New as far as we found**, and validated bit-exact. Useful for small-batch and online models; **does not scale** to LLM batches (§8) |
| reversible LM compression (encoder and decoder as one program) | **Not new.** Townsend & van de Meent (2022) did it in a verified reversible language |
| exact injectivity of an LM's internals | **Not new** (Nikolaou et al. 2026); confirmed exactly here, including under quantisation |
| exact ceiling on recovering a prompt from greedy answers, and its saturation | **New as an exact measurement**, on a toy model |
| exact anonymity-set audit for secret prompt slots | **Useful, and to our knowledge not done this way.** It scales to real models whenever the secret space is enumerable |
| per-operation witnesses as a measure of information loss | **Refuted** for information about the input (§4 correction) |
| "uncompute by search": the minimal witness is the input's index in its preimage | **Known for Boolean circuits** (minimum garbage). New as a reversible-*program* construct. Prototyped as `reverie.lab.SearchWitness`; about 100× smaller than per-operation witnesses for the LM |
| rollback netcode by reverse execution instead of snapshots | **Not found in games** (established in discrete-event simulation). Validated: exact, 7–33× less memory, ≈2.1× compute per rewound frame (§10) |
| generated saga compensations, exact under concurrent writers | **No prior work found** on derived compensations with guarantees. Validated: 0 silent errors vs 12% hand-written and 56% snapshot (§10) |
| exact generated data down-migrations with flagged conflicts | **Useful**; schema tools explicitly do not revert data. Validated: 100% exact, conflicts flagged, none silent (§10) |
| DNA-storage codec with a generated, validating decoder | **New as a construction** (verified codecs exist in general). Validated: exact, homopolymer-free, rejects 71% of substitutions (§11) |
| exact history capacity of molecular recorder architectures | **No exact accounting found.** Validated by enumeration; idealised models (§11) |
| bit-exact training across platforms | Integer semantics guarantee it. A golden-fingerprint test runs on Linux, macOS and Windows CI. Float LLM pipelines need special kernels for this (batch-invariant inference; "GPU butterfly effect" in neural compression) |

The most promising next step for Reverie the *language* is to make "uncompute
by search" a real construct: `uncompute x by search over D`. The backward run
enumerates a finite domain `D`, and the witness is an index. That lets
non-injective functions carry the information-theoretic minimum witness
instead of every erased intermediate.

## 10. Beyond ML: generated undo for games and business software

The broader pattern: wherever people hand-write "undo", "rollback",
"compensate" or "down" code, Reverie can generate it with the checker's proof
that it is the exact inverse. Where an inverse cannot exist, the program
must say what it keeps instead of losing it silently. Three studies, each
measured against what practitioners do today.

### Rollback netcode ([rollback-netcode.md](rollback-netcode.md))

Rollback netcode (GGPO, GGRS) predicts the remote player's input and rewinds
on a misprediction, which in every implementation means snapshotting the
whole game state every frame. GGRS requires the snapshot functions and
cross-platform determinism. Retrofitting rollback into Mortal Kombat X
reportedly took about two engineer-years of serialization work.

`reverie.rollback` writes a fighting game (two fighters plus N bouncing
projectiles) as a reversible frame step and rewinds by running it backward.
With the remote input 6 frames late:

- Both strategies end **bit-identical to a ground truth** that knew every
  input, for N = 0, 16, 64 and 256, over 39–49 rollbacks each.
- Reverse mode also ends with exactly the ground truth's witness.
- **Memory:** the witness is **7–33× smaller** than a snapshot in coded bits
  per frame. On most frames almost every witness entry is zero: one entry
  per wall clamp, bounce, health floor or cooldown tick, and only when it
  happens.
- **Compute:** a rewound frame costs **≈2.1×** as much (backward plus
  forward, against forward only). For N = 256 that is 194 µs against 93 µs
  per rewound frame.
- **Engineering:** no snapshot or serialization code. Integer semantics give
  the cross-platform determinism rollback requires, and the golden
  fingerprint test checks it on Linux, macOS and Windows CI.

I found no published or shipped game that rewinds by reverse execution
rather than snapshots. Reverse computation as an alternative to state saving
is established in parallel discrete-event simulation (Perumalla et al. 1999;
Backstroke; ROSS), which is the closest prior art.

### Saga compensations ([sagas-and-migrations.md](sagas-and-migrations.md))

When a step of a multi-service transaction fails, the completed steps are
undone by compensations. Every source we found (the 1987 sagas paper,
Temporal, Oracle's saga guides) has developers write them by hand. In 5,000
random checkout sagas (reserve stock, charge with a fee, award capped loyalty
points, book shipping), other transactions write the same records in between,
and the right answer is "keep what others committed, remove the saga's
effect":

| compensation | correct | fails loudly | silently wrong |
|---|---:|---:|---:|
| snapshot restore | 1,404 | 0 | 1,758 (56%) |
| hand-written | 2,786 | 0 | 376 (12%) |
| Reverie, capped step re-checked on the current balance | 3,080 | 82 | **0** |
| **Reverie, step written "decide on recorded input, apply a delta"** | **3,162** | **0** | **0** |

Two things make this work:
- **Updates commute.** Reverie steps are updates (`+=`, `-=`), so a
  generated inverse subtracts exactly what the step added. That commutes
  with concurrent updates, whereas restoring a saved value erases them.
- **Lossy steps must keep a witness, and the inverse re-checks it.** A step
  that loses information (the points cap) must keep a witness, and the
  checker rejects the obvious version. The inverse re-checks the step's
  conditions, so when concurrent activity makes exact undo impossible it
  fails loudly instead of corrupting a balance.

Writing the step to decide on the values it recorded and apply only a delta
makes the generated compensation exact even then. The leftover in its
witness records exactly the concurrent drift, as an audit trail.

What no compensation can fix: in 208 of the 3,162 compensated sagas, another
transaction *read* the saga's effects (a dirty read). That is the inherent
weakness of sagas, not of this method.

### Data migrations ([sagas-and-migrations.md](sagas-and-migrations.md))

Generated down-migrations (for example Prisma's) revert the schema but
explicitly not data transformations. Academic bidirectional schema evolution
(BiDEL/InVerDa; quasi-inverses of schema mappings) shows exact inverses often
do not exist.

A Reverie v1 → v2 migration over 5,000 rows converts USD/EUR/GBP prices to
USD cents with rounding, collapses a 6-value status to 3, packs flags and
drops a column:

- **Exact rollback.** The generated down-migration restores 5,000 of 5,000
  rows. A hand-written data down-migration restores 0 (1,238 ignoring the
  dropped column), and a schema-only down-migration leaves the old columns
  empty.
- **Rejected without a witness.** Dropping a column or rounding without
  keeping a witness is a compile error.
- **Witness size.** 10.9 KiB of witness covers the conversions, status and
  flags. Keeping the dropped column itself costs 19.7 KiB, against 50.2 KiB
  for a full backup.
- **Rolling back after new v2 writes.**
  - Unedited rows come back exactly.
  - Edited USD rows carry the edit back, and migrating up again reproduces
    it.
  - Every edited converted row is **flagged** by a non-zero leftover in its
    witness columns. No row is silently inconsistent, and no unedited row is
    flagged.
  - The flag is conservative: about half the flagged rows would have
    round-tripped.

### The general lessons

1. **Generated inverses beat hand-written ones where the inverse is easy to
   get subtly wrong.** That covers capped or rounded steps and anything with
   a concurrent writer.
2. **Write steps as deltas computed from recorded inputs.** Then the
   generated inverse is also the right *semantic* compensation under
   concurrency.
3. **Information that cannot come back must be named.** The checker turns
   silent data loss into either a witness or a compile error. At run time,
   an impossible undo becomes a loud failure or a flagged row, never a
   silent one.
4. **The price is compute.** Rewinding costs about as much as running
   forward, so rollback-heavy workloads pay roughly 2× on rewound work in
   exchange for memory, for no serialization code, and for correctness by
   construction.

## 11. Biology and computational biology

### Where reversibility already shows up (prior art, not ours)

- **Bit-reversible molecular dynamics.** Levesque & Verlet (1993) stored
  positions as integers so an MD trajectory runs backward bit for bit.
  Successors include Hoover's higher-order integrators and JANUS for N-body
  problems. This is the same integer trick Reverie uses, applied to physics.
- **Reversible solvers for memory-efficient gradients.** The reversible Heun
  method (Kidger et al. 2021) and algebraically reversible ODE solvers
  (McCallum & Foster 2024) give exact gradients through long simulations
  without storing the trajectory. That is directly relevant to fitting
  systems-biology models; it is the Maclaurin idea of §8 for differential
  equations.
- **Recombinase logic and state machines** (Siuti et al.; Roquet et al. 2016)
  store state by inverting and excising DNA. Inversion is a bijection on
  sequence; excision literally throws the cut-out circle away, so the biology
  has its own "witness".

### What we tested ([dna.md](dna.md))

**A DNA-storage codec with a generated decoder.**
- **The codec.** Goldman et al. (2013) avoid homopolymers, which synthesis
  and sequencing handle badly, with a rotating code: each base-3 digit picks
  one of the three bases that differ from the previous one.
  `reverie.dna.codec_source` writes that encoder once, and the checker proves
  it loses nothing. The decoder is the encoder run backward.
- **Results.**
  - 200 of 200 random payloads round-trip exactly, and every strand is
    homopolymer-free.
  - Density is 1.33 bits/nt.
  - Because the no-homopolymer assertion also runs backward, the generated
    decoder rejects **71%** of random single-base substitutions instead of
    returning wrong bytes. The other 29% decode silently wrong, so error
    correction is still needed on top.
- **Novelty.** Verified-by-construction compression already exists (§9); we
  found no DNA-storage codec built this way.
- **Language limit exposed, then fixed.** The checker could not prove that
  `dna[j + 1]` and `dna[j]` are different cells, so the obvious encoder was
  rejected. It now treats `e + c₁` and `e + c₂` as distinct cells whenever
  `c₁ ≠ c₂`, and the codec is written the natural way.

**How much of a cell's history its DNA can give back.**
- **The question.** Molecular recorders write events into DNA:
  - ordered tapes: CRISPR spacer arrays (Retro-Cascorder) and sequential
    prime editing (DNA Typewriter);
  - integrase site flips;
  - per-signal editing counters;
  - pairwise order domains (the N² detector design).

  The field notes that integrase recorders "run out of states" and that most
  devices lose event order, but we found no exact accounting of how much
  history each architecture keeps.
- **Exact answer.** Reading the history back is a reversibility question,
  so `reverie.dna` enumerates every history of n events over k signals and
  reports H(history | DNA): the bits the DNA cannot give back, which is
  exactly the witness a decoder would need.
- **In Reverie.** The ordered-tape recorder *moves* the history into DNA,
  and the backward run reads it out. For the site-flip recorder, the history
  has to remain a witness.

| 8 events | history | ordered tape | 3-slot tape | pairwise first-order | per-signal counters | site flips |
|---|---:|---:|---:|---:|---:|---:|
| 2 signals | 8.0 bits | 0 | 5.0 | 6.9 | 5.5 | 7.9 |
| 4 signals | 16.0 bits | 0 | 10.0 | 10.4 | 9.5 | 14.2 |

What the full tables show:
- **Site flips saturate.** Integrase switches forget almost everything once
  events outnumber signals; they keep at most log₂(number of states) bits.
- **The best non-tape design depends on the regime.** Pairwise-order domains
  beat per-signal counters for short histories, and counters win for long
  ones. The crossover moves later as signals increase: counters are strictly
  better from 4 events with 2 signals (they tie at 3), from 5 with 3, and
  from 7 with 4. A designer can pick the architecture for the expected event
  count from exact numbers.
- **These are upper bounds on what any decoder can recover.** The models are
  idealised: uniform random histories, no editing noise, perfect readout.
  Real recorders are stochastic, so the real numbers are worse.

### Other directions worth exploring (not built)

- **Exact anonymity audits for genomic data.** The secret-slot audit of §7
  applies directly: enumerate the genotypes at a few loci and compute exact
  anonymity sets for a released statistic or beacon response.
- **Bit-identical pipelines.** A 2025 preprint reports BLAST+ results
  differing across compilers and hardware through floating point, and
  variant-calling concordance across pipelines can be low. Integer kernels
  (alignment scoring is already integer) with fingerprinted replay proofs,
  as in §3, would make results identical everywhere.
- **Liftover between genome assemblies.** Reciprocal hg19 ↔ hg38 liftover is
  not reversible, but keeping a witness only makes round trips exact by
  remembering the original coordinates. That is trivial, so it is not
  pursued.

## 12. Next experiments worth running

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
- Nikolaou et al., [Language Models are Injective and Hence Invertible](https://arxiv.org/abs/2510.15511) (ICLR 2026); [Transformer Injectivity & Geometric Robustness](https://arxiv.org/abs/2511.14808) (2025)
- Morris et al., [Language Model Inversion](https://proceedings.iclr.cc/paper_files/paper/2024/file/999fcab97007ebef0cda9949550b4a9e-Paper-Conference.pdf) (ICLR 2024); Zhang, Morris & Shmatikov, [Extracting Prompts by Inverting LLM Outputs](https://aclanthology.org/2024.emnlp-main.819/) (EMNLP 2024)
- Finlayson, Ren & Swayamdipta, [Logits of API-Protected LLMs Leak Proprietary Information](https://arxiv.org/abs/2403.09539) (COLM 2024)
- Kaneko & Baldwin, [Bits Leaked per Query](https://arxiv.org/abs/2510.17000) (NeurIPS 2025)
- Carlini et al., [The Secret Sharer](https://arxiv.org/abs/1802.08232) (USENIX Security 2019)
- Townsend & van de Meent, [Verified Reversible Programming for Verified Lossless Compression](https://arxiv.org/abs/2211.09676) (2022); Townsend, Bird & Barber, [Bits Back with ANS](https://arxiv.org/abs/1901.04866) (ICLR 2019)
- Delétang et al., [Language Modeling Is Compression](https://arxiv.org/abs/2309.10668) (ICLR 2024)
- Zhang, Lewis & Kleijn, [On Exact Bit-level Reversible Transformers Without Changing Architecture](https://arxiv.org/abs/2407.09093) (ICML 2025)
- He & Thinking Machines Lab, Defeating Nondeterminism in LLM Inference (2025); [LLM-42: Enabling Determinism in LLM Inference with Verified Speculation](https://arxiv.org/html/2601.17768) (2026)
- Garcia-Molina & Salem, Sagas (SIGMOD 1987); [Temporal: compensating actions](https://temporal.io/blog/compensating-actions-part-of-a-complete-breakfast-with-sagas); [Oracle: saga participant code](https://blogs.oracle.com/developers/post/developing-saga-participant-code-for-compensating-transactions)
- [Prisma: generating down migrations](https://www.prisma.io/docs/orm/prisma-migrate/workflows/generating-down-migrations); Herrmann et al., [Living in Parallel Realities (BiDEL/InVerDa)](https://arxiv.org/pdf/1608.05564)
- Perumalla, Carothers & Fujimoto, [Efficient optimistic parallel simulations using reverse computation](https://www.osti.gov/biblio/1003413) (1999); [Backstroke](https://www.scilit.net/publications/ccc28b48a48d3900145ed8d8684a0c97); Schordan et al., [reversible C++ for ROSS](https://www.osti.gov/servlets/purl/1658686)
- [GGPO](https://en.wikipedia.org/wiki/GGPO); [GGRS setup and determinism](https://docs.rs/crate/ggrs/latest/source/docs/setup.md); [rollback netcode explainer](https://ki.infil.net/w02-netcode-p5.html); [Dynasty Feud online implementation](https://www.moddb.com/features/online-implementation-for-dynasty-feud)
- Levesque & Verlet, [Molecular dynamics and time reversibility](https://link.springer.com/article/10.1007/BF01048022) (1993); Hoover & Hoover, [Bit-reversible Milne integrator](https://arxiv.org/abs/1706.08678); Rein & Tamayo, [JANUS bit-wise reversible N-body integrator](https://arxiv.org/abs/1704.07715)
- Kidger et al., [Efficient and Accurate Gradients for Neural SDEs (reversible Heun)](https://arxiv.org/abs/2105.13493) (NeurIPS 2021); McCallum & Foster, [Efficient, Accurate and Stable Gradients](https://arxiv.org/html/2410.11648v1) (2024)
- Roquet et al., [Synthetic recombinase-based state machines in living cells](https://microsoft.com/en-us/research/?p=766975) (Science 2016); Ham et al., [Double inversion recombination switch](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2481393/) (2008)
- Choi et al., [A time-resolved, multi-symbol molecular recorder via sequential genome editing (DNA Typewriter)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9352581/) (Nature 2022); [Retro-Cascorder](https://link.springer.com/article/10.1038/s41586-022-04994-6) (Nature 2022)
- Goldman et al., High-capacity storage of digital information in DNA (Nature 2013), via [Survey of Information Encoding Techniques for DNA](https://arxiv.org/pdf/1906.11062); [EBI DNA storage page](https://www.ebi.ac.uk/goldman-srv/DNA-storage)
- [Systematic Evaluation of Variant Discrepancies During Reciprocal hg19/hg38 Liftover](https://www.mdpi.com/1422-0067/27/22/10305) (2025); [Low concordance of multiple variant-calling pipelines](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3706896/) (2013)
- Kim et al., [I-BERT: Integer-only BERT Quantization](https://arxiv.org/abs/2101.01321) (ICML 2021)
- Yokoyama & Glück, [A Reversible Programming Language and its Invertible Self-Interpreter](https://dl.acm.org/doi/pdf/10.1145/1244381.1244404) (Janus)
