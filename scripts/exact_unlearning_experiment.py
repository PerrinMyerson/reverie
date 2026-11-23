#!/usr/bin/env python3
"""Exact, checkpoint-free unlearning on scikit-learn's 8x8 digits.

Trains a 64 -> 32 -> 10 ReLU MLP with Q31 SGD inside Reverie
(`reverie.lab.mlp_sgd_source`), keeping only the minimal rewind witness (the
back-propagated errors). Then it:

1. rewinds the whole run to step 0 and checks the initial weights come back
   bit for bit;
2. unlearns a sample of training points by rewinding and replaying, and checks
   each result is bit-identical to retraining from scratch without that point;
3. compares the witness with the alternatives (a checkpoint per step, per-step
   weight deltas, and a Bennett-style full trace), in raw and coded bits;
4. relates each sample's witness size early in training to whether the final
   model misclassifies it (the witness is an error-norm score, i.e. EL2N);
5. builds and verifies a roundtrip proof of the whole training run.

Writes JSON and Markdown results. Needs the `reverie` package (`maturin
develop` from a checkout) and scikit-learn; `--self-test` needs neither
scikit-learn nor numpy.

    python3 scripts/exact_unlearning_experiment.py \
        --json-output docs/research/exact-unlearning-results.json \
        --markdown-output docs/research/exact-unlearning-results.md
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train", type=int, default=600, help="training samples")
    parser.add_argument("--test", type=int, default=500, help="held-out samples")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--unlearn", type=int, default=12, help="samples to unlearn")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--engine", choices=["slot", "tree"], default="slot")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run a tiny synthetic configuration (no scikit-learn needed)",
    )
    return parser.parse_args()


# -- exact Q31 inference in Python (mirrors reverie-interp) -------------------


def fixed_mul(a: int, b: int) -> int:
    return (a * b) >> 31


def vecmat(x: Sequence[int], w: Sequence[Sequence[int]]) -> List[int]:
    cols = len(w[0])
    return [sum(fixed_mul(x[i], w[i][j]) for i in range(len(x))) for j in range(cols)]


def predict(weights: Dict[str, Any], x: Sequence[int]) -> int:
    pre = [a + b for a, b in zip(vecmat(x, weights["w1"]), weights["b1"])]
    act = [max(v, 0) for v in pre]
    logits = [a + b for a, b in zip(vecmat(act, weights["w2"]), weights["b2"])]
    return max(range(len(logits)), key=lambda j: (logits[j], -j))


def accuracy(weights: Dict[str, Any], images, labels) -> float:
    hits = sum(predict(weights, x) == y for x, y in zip(images, labels))
    return hits / len(labels)


# -- data ---------------------------------------------------------------------


def load_data(args: argparse.Namespace, rng: random.Random):
    from reverie.lab import to_q31

    if args.self_test:
        classes, features = 3, 6
        centers = [[rng.uniform(-0.8, 0.8) for _ in range(features)] for _ in range(classes)]
        total = args.train + args.test
        labels = [i % classes for i in range(total)]
        images = [
            [to_q31(max(-1.0, min(1.0, centers[y][f] + rng.gauss(0, 0.25)))) for f in range(features)]
            for y in labels
        ]
        name = "synthetic blobs"
    else:
        from sklearn.datasets import load_digits

        digits = load_digits()
        index = list(range(len(digits.target)))
        rng.shuffle(index)
        index = index[: args.train + args.test]
        images = [[to_q31(float(v) / 16.0) for v in digits.data[i]] for i in index]
        labels = [int(digits.target[i]) for i in index]
        classes = 10
        name = "scikit-learn digits (8x8)"
    return (
        name,
        images[: args.train],
        labels[: args.train],
        images[args.train :],
        labels[args.train :],
        classes,
    )


def initial_weights(features: int, hidden: int, classes: int, rng: random.Random):
    from reverie.lab import to_q31

    s1 = 1.0 / math.sqrt(features)
    s2 = 1.0 / math.sqrt(hidden)
    return {
        "w1": [[to_q31(rng.gauss(0, s1)) for _ in range(hidden)] for _ in range(features)],
        "b1": [0] * hidden,
        "w2": [[to_q31(rng.gauss(0, s2)) for _ in range(classes)] for _ in range(hidden)],
        "b2": [0] * classes,
    }


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return cov / var if var else 0.0


# -- experiment ---------------------------------------------------------------


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import reverie
    from reverie import lab
    from reverie.ledger import code_length_bits, flatten, ledger, per_row_bits

    rng = random.Random(args.seed)
    name, xs, ys, test_xs, test_ys, classes = load_data(args, rng)
    features = len(xs[0])
    order: List[int] = []
    for _ in range(args.epochs):
        epoch = list(range(len(xs)))
        rng.shuffle(epoch)
        order += epoch
    steps = len(order)
    weights0 = initial_weights(features, args.hidden, classes, rng)
    run = lab.SGDRewind(
        images=xs,
        labels=ys,
        order=order,
        hidden=args.hidden,
        classes=classes,
        lr=lab.to_q31(args.lr),
        engine=args.engine,
    )

    started = time.perf_counter()
    final = run.train(weights0)
    train_seconds = time.perf_counter() - started
    weights = run.weights(final)

    started = time.perf_counter()
    rewound = run.rewind(final, 0)
    rewind_seconds = time.perf_counter() - started
    rewind_exact = run.weights(rewound) == weights0 and all(
        v == 0 for v in flatten(run.witness(rewound))
    )

    # Unlearning: spread the requests over the first epoch's schedule.
    first_epoch = order[: len(xs)]
    picks = sorted(
        rng.sample(range(len(xs)), min(args.unlearn, len(xs))),
        key=lambda i: first_epoch.index(i),
    )
    unlearn_rows = []
    for sample in picks:
        first = run.first_use(sample)
        started = time.perf_counter()
        unlearned = run.unlearn(final, sample)
        unlearn_seconds = time.perf_counter() - started
        active = [0 if o == sample else 1 for o in order]
        started = time.perf_counter()
        retrained = run.train(weights0, active=active)
        retrain_seconds = time.perf_counter() - started
        unlearn_rows.append(
            {
                "sample": sample,
                "first_use_step": first,
                "steps_rewound": steps - first,
                "unlearn_seconds": unlearn_seconds,
                "retrain_seconds": retrain_seconds,
                "bit_identical_to_retrain": run.weights(unlearned) == run.weights(retrained),
                "weights_changed": run.weights(unlearned) != weights,
            }
        )

    # Memory accounting.
    size = run.witness_size()
    params = size.parameter_values
    h, c = args.hidden, classes
    witness_ledger = ledger(final, lab.WITNESS)
    weight_bits = sum(code_length_bits(v) for v in flatten(weights))
    # Deterministic integer training means sparse checkpoints + forward replay
    # also rewind exactly. Compare against that, at equal memory.
    parity_interval = math.ceil(params / (h + c))
    sqrt_interval = max(1, round(math.sqrt(steps)))
    memory = {
        "checkpoint_interval_at_memory_parity": parity_interval,
        "sqrt_checkpoint_interval": sqrt_interval,
        "reverse_traversal_sqrt_checkpoint_values": (math.ceil(steps / sqrt_interval) + sqrt_interval)
        * params,
        "reverse_traversal_witness_values": steps * (h + c) + params,
        "parameters": params,
        "steps": steps,
        "witness_values_per_step": h + c,
        "checkpoint_values_per_step": params,
        "bennett_trace_values_per_step": 5 * h + 2 * c,
        "witness_raw_bits": witness_ledger.raw_bits,
        "witness_coded_bits": witness_ledger.coded_bits,
        "checkpoints_raw_bits": 64 * params * steps,
        "bennett_trace_raw_bits": 64 * (5 * h + 2 * c) * steps,
        "final_weights_coded_bits": weight_bits,
        "landauer_joules_at_300K": witness_ledger.landauer_joules,
        "ledger": witness_ledger.as_dict(),
    }

    # Witness size per sample (epoch 1) vs. final misclassification.
    step_bits = [
        e + d
        for e, d in zip(per_row_bits(final["err_tape"]), per_row_bits(final["delta_tape"]))
    ]
    first_epoch_bits = {order[s]: step_bits[s] for s in range(len(xs))}
    last_epoch_bits = {order[s]: step_bits[s] for s in range(steps - len(xs), steps)}
    wrong = [int(predict(weights, xs[i]) != ys[i]) for i in range(len(xs))]
    samples = list(range(len(xs)))
    early = [first_epoch_bits[i] for i in samples]
    late = [last_epoch_bits[i] for i in samples]
    wrong_bits = [b for b, w in zip(late, wrong) if w]
    right_bits = [b for b, w in zip(late, wrong) if not w]
    attribution = {
        "spearman_first_epoch_bits_vs_final_error": spearman(early, wrong),
        "spearman_last_epoch_bits_vs_final_error": spearman(late, wrong),
        "mean_last_epoch_bits_misclassified": statistics.fmean(wrong_bits) if wrong_bits else None,
        "mean_last_epoch_bits_correct": statistics.fmean(right_bits) if right_bits else None,
        "misclassified_training_samples": sum(wrong),
        "bits_per_step_first_epoch": statistics.fmean(step_bits[: len(xs)]),
        "bits_per_step_last_epoch": statistics.fmean(step_bits[steps - len(xs) :]),
    }

    # Where the trained network discards information at inference time.
    profile = lab.erasure_profile(weights, test_xs, engine=args.engine)
    ranked = sorted(profile["neurons"], key=lambda n: n["erased_fraction"])
    k = max(1, args.hidden // 8)

    def ablate(neurons):
        ablated = {name: [row[:] if isinstance(row, list) else row for row in value] for name, value in weights.items()}
        for j in neurons:
            ablated["w2"][j] = [0] * classes
        return accuracy(ablated, test_xs, test_ys)

    erasure = {
        "samples": profile["samples"],
        "exact_reconstruction": profile["exact_reconstruction"],
        "relu_witness_bits": profile["relu_witness_bits"],
        "activation_bits": profile["activation_bits"],
        "readout_logit_bits": profile["readout_logit_bits"],
        "dead_neurons": profile["dead_neurons"],
        "linear_neurons": profile["linear_neurons"],
        "mean_erased_fraction": statistics.fmean(n["erased_fraction"] for n in profile["neurons"]),
        "ablated_k": k,
        "accuracy_ablate_most_erasing": ablate([n["neuron"] for n in ranked[-k:]]),
        "accuracy_ablate_least_erasing": ablate([n["neuron"] for n in ranked[:k]]),
        "neurons": profile["neurons"],
    }

    # A fingerprinted, replayable proof of the whole training run.
    seeds = dict(run.data)
    seeds.update(weights0)
    seeds.update({"active": [1] * steps, "start": 0, "stop": steps - 1})
    started = time.perf_counter()
    proof = run.program.roundtrip(seeds)
    proof_seconds = time.perf_counter() - started
    verification = reverie.verify(proof, source=run.program)

    return {
        "schema": "reverie_exact_unlearning_v1",
        "dataset": name,
        "config": {
            "train": len(xs),
            "test": len(test_xs),
            "features": features,
            "hidden": h,
            "classes": c,
            "epochs": args.epochs,
            "steps": steps,
            "lr": args.lr,
            "seed": args.seed,
            "engine": args.engine,
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "reverie": reverie.__version__,
        },
        "accuracy": {
            "train_initial": accuracy(weights0, xs, ys),
            "train_final": accuracy(weights, xs, ys),
            "test_final": accuracy(weights, test_xs, test_ys),
        },
        "timing_seconds": {
            "train": train_seconds,
            "rewind_to_start": rewind_seconds,
            "roundtrip_proof": proof_seconds,
        },
        "rewind_to_start_exact": rewind_exact,
        "unlearning": unlearn_rows,
        "memory": memory,
        "attribution": attribution,
        "erasure": erasure,
        "proof": {
            "passed": proof["passed"],
            "fingerprint": proof["fingerprint"],
            "verified": verification["passed"],
            "failed_checks": sorted(k for k, ok in verification["checks"].items() if not ok),
        },
    }


def markdown(result: Dict[str, Any]) -> str:
    cfg, mem, acc = result["config"], result["memory"], result["accuracy"]
    rows = result["unlearning"]
    identical = sum(r["bit_identical_to_retrain"] for r in rows)
    changed = sum(r["weights_changed"] for r in rows)
    mib = lambda bits: bits / 8 / 2**20  # noqa: E731
    att = result["attribution"]
    lines = [
        "# Exact unlearning results",
        "",
        f"Generated by `scripts/exact_unlearning_experiment.py` on {result['dataset']}: "
        f"{cfg['features']} -> {cfg['hidden']} -> {cfg['classes']} ReLU MLP, Q31 SGD, "
        f"{cfg['train']} training samples x {cfg['epochs']} epochs = {cfg['steps']} steps, "
        f"`{cfg['engine']}` engine. Host: {result['host']['platform']}.",
        "",
        "## Training",
        "",
        f"- Train accuracy {acc['train_initial']:.1%} -> {acc['train_final']:.1%}; "
        f"held-out accuracy {acc['test_final']:.1%} on {cfg['test']} samples.",
        f"- Forward training: {result['timing_seconds']['train']:.2f} s. "
        f"Full rewind to step 0: {result['timing_seconds']['rewind_to_start']:.2f} s, "
        f"exact: **{result['rewind_to_start_exact']}**.",
        "",
        "## Unlearning",
        "",
        f"**{identical}/{len(rows)}** unlearned models are bit-identical to retraining from "
        f"scratch without the sample ({changed}/{len(rows)} differ from the original model).",
        "",
        "Unlearning rewinds T - t steps and replays them, so it costs about 2(T - t) steps against T "
        "for retraining. With several epochs every sample is first used early, so here it is slower "
        "than retraining. It pays off when the deleted data arrived late in training, as in continual "
        "or fine-tuning runs.",
        "",
        "| sample | first used at step | steps rewound | unlearn (s) | retrain (s) | bit-identical |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['sample']} | {r['first_use_step']} | {r['steps_rewound']} | "
            f"{r['unlearn_seconds']:.2f} | {r['retrain_seconds']:.2f} | "
            f"{'yes' if r['bit_identical_to_retrain'] else 'NO'} |"
        )
    lines += [
        "",
        "## Memory needed to rewind exactly",
        "",
        "| strategy | values per step | total (MiB, 64-bit values) |",
        "|---|---:|---:|",
        f"| checkpoint every step / per-step weight deltas | {mem['checkpoint_values_per_step']:,} | "
        f"{mib(mem['checkpoints_raw_bits']):.2f} |",
        f"| Bennett full trace (keep every intermediate) | {mem['bennett_trace_values_per_step']:,} | "
        f"{mib(mem['bennett_trace_raw_bits']):.2f} |",
        f"| **error-factor witness (this work)** | **{mem['witness_values_per_step']:,}** | "
        f"**{mib(mem['witness_raw_bits']):.2f}** |",
        f"| error-factor witness, Elias-gamma coded | - | {mib(mem['witness_coded_bits']):.3f} |",
        "",
        f"The witness is {mem['checkpoints_raw_bits'] / mem['witness_raw_bits']:.0f}x smaller than "
        f"a checkpoint per step ({mem['checkpoints_raw_bits'] / mem['witness_coded_bits']:.0f}x once coded) "
        f"and {mem['bennett_trace_raw_bits'] / mem['witness_raw_bits']:.1f}x smaller than keeping every "
        "intermediate value.",
        "",
        "**The fair baseline is sparse checkpoints.** Integer training is deterministic, so a checkpoint every "
        f"{mem['checkpoint_interval_at_memory_parity']} steps plus forward replay also reaches any step exactly, "
        "with the same memory as the witness. For unlearning it is *faster*: reaching step t costs at most "
        f"{mem['checkpoint_interval_at_memory_parity']} replayed steps, while the witness rewinds T - t steps "
        "backward. The witness wins when you need to visit the states in reverse order (reverse-mode "
        "differentiation through training, scrubbing a run backward, auditing it step by step): "
        f"{cfg['steps']:,} backward steps with {mem['reverse_traversal_witness_values']:,} values stored, versus "
        f"the best sqrt(T) checkpoint schedule, which stores {mem['reverse_traversal_sqrt_checkpoint_values']:,} "
        f"values ({mem['reverse_traversal_sqrt_checkpoint_values'] / mem['reverse_traversal_witness_values']:.1f}x) "
        "and replays every step once more.",
        "",
        f"Erasing the witness irreversibly would cost at least {mem['landauer_joules_at_300K']:.2e} J at 300 K "
        "(Landauer).",
        "",
        "## Witness size is not a difficulty score (negative result)",
        "",
        f"- Mean coded witness per step: {att['bits_per_step_first_epoch']:.0f} bits in epoch 1, "
        f"{att['bits_per_step_last_epoch']:.0f} bits in the last epoch.",
        f"- Spearman correlation between a sample's witness bits and whether the final model "
        f"misclassifies it: {att['spearman_first_epoch_bits_vs_final_error']:.2f} (epoch 1), "
        f"{att['spearman_last_epoch_bits_vs_final_error']:.2f} (last epoch); "
        f"{att['misclassified_training_samples']} training samples misclassified.",
        "- The information each SGD step destroys is close to constant: coded length tracks the magnitude "
        "of every error component, not how wrong the prediction was.",
        "",
        "## Where the trained network discards information",
        "",
        f"An inference program whose only witness is each ReLU's negative part "
        f"(`reverie.lab.mlp_erasure_source`) ran over the {result['erasure']['samples']} held-out samples "
        f"and back again (exact: **{result['erasure']['exact_reconstruction']}**).",
        "",
        f"- The ReLU layer discards {result['erasure']['relu_witness_bits']:,} coded bits; the activations "
        f"it keeps take {result['erasure']['activation_bits']:,}. On average a hidden unit is zeroed on "
        f"{result['erasure']['mean_erased_fraction']:.0%} of inputs.",
        f"- The argmax readout keeps one class out of logits worth {result['erasure']['readout_logit_bits']:,} "
        "coded bits, so "
        + (
            "the ReLU layer, not the readout, is where most information leaves."
            if result["erasure"]["relu_witness_bits"] > result["erasure"]["readout_logit_bits"]
            else "the readout is where most information leaves."
        ),
        f"- Dead units (always zeroed): {result['erasure']['dead_neurons'] or 'none'}; always-active "
        f"(effectively linear) units: {result['erasure']['linear_neurons'] or 'none'}.",
        f"- Ablating the {result['erasure']['ablated_k']} units that discard the *most* leaves "
        f"{result['erasure']['accuracy_ablate_most_erasing']:.1%} held-out accuracy; ablating the "
        f"{result['erasure']['ablated_k']} that discard the *least* leaves "
        f"{result['erasure']['accuracy_ablate_least_erasing']:.1%} (unablated {acc['test_final']:.1%}).",
        "",
        "## Proof",
        "",
        f"Roundtrip proof of the full training run: passed **{result['proof']['passed']}**, "
        f"verified **{result['proof']['verified']}**, fingerprint `{result['proof']['fingerprint'][:16]}...` "
        f"({result['timing_seconds']['roundtrip_proof']:.2f} s).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.self_test:
        args.train, args.test, args.hidden, args.epochs, args.unlearn = 18, 9, 5, 2, 4
    result = run(args)
    if args.self_test:
        assert result["rewind_to_start_exact"], "rewind to start is not exact"
        assert all(r["bit_identical_to_retrain"] for r in result["unlearning"]), result["unlearning"]
        assert result["proof"]["passed"] and result["proof"]["verified"], result["proof"]
        print("ok: exact unlearning self-test passed")
        return 0
    text = markdown(result)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
