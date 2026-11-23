#!/usr/bin/env python3
"""How much of a prompt can be recovered, exactly, from what a language model
emits? An exhaustive study on a tiny integer language model.

The model is a Bengio-style neural n-gram LM (embed the last W characters,
one ReLU hidden layer, softmax readout) over an 8-character alphabet. It is
trained in float and then run in fixed point with `b` fractional bits, so
every quantity it emits is a finite integer object. With W = L = 5 there are
8^5 = 32,768 possible prompts, few enough to enumerate. Every number below is
therefore an exact count, not an estimate.

For each observable O (the hidden pre-activations, the post-ReLU hidden
state, the logits, the next token, a greedy answer of k tokens) it reports
the conditional entropy H(P | O) in bits, for prompts drawn uniformly and for
prompts drawn from the model's own distribution. H(P | O) is exactly the
minimum side information (the "witness") needed to recover the prompt from
O: ceil(log2 |class|) bits per prompt are always enough, and the script
checks that an index-within-class code attains it.

It also tests three claims:

* a per-operation witness (the negative parts each ReLU erases) is only an
  upper bound on the information about the *input* that a layer destroys;
* quantization creates collisions in hidden states, and how many bits that
  costs as precision drops;
* a secret slot inside a templated prompt (e.g. a short code) has a
  computable worst-case anonymity set under each observable.

    python3 scripts/prompt_recovery_study.py --markdown-output docs/research/prompt-recovery.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ALPHABET = " aeinost"
WORDS = (
    "a at ate eat sat set sit site tie ties tea teas ten tens net nets nest "
    "nests note notes tone tones stone stones son sons not into onto one ones "
    "seen teen teens nine nose noses east seat seats tin tins sin oat oats "
    "toast tease sense tense noon soon taste test tests"
).split()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--window", type=int, default=5, help="context = prompt length")
    parser.add_argument("--embed", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--answer", type=int, default=12, help="max greedy answer length")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


# -- data and training (float) ------------------------------------------------


def corpus(rng: np.random.Generator, n_words: int = 30000) -> np.ndarray:
    """Word sequences with a little structure: each word tends to be
    followed by words sharing its last letter."""
    by_first: Dict[str, List[str]] = defaultdict(list)
    for word in WORDS:
        by_first[word[0]].append(word)
    words = [WORDS[0]]
    for _ in range(n_words):
        last = words[-1][-1]
        pool = by_first.get(last) if rng.random() < 0.6 else None
        words.append(rng.choice(pool) if pool else rng.choice(WORDS))
    text = " ".join(words)
    return np.array([ALPHABET.index(ch) for ch in text], dtype=np.int64)


def train(args, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    data = corpus(rng)
    v, w, e, h = len(ALPHABET), args.window, args.embed, args.hidden
    params = {
        "emb": rng.normal(0, 0.5, (v, e)),
        "w1": rng.normal(0, 1 / math.sqrt(w * e), (w * e, h)),
        "b1": np.zeros(h),
        "w2": rng.normal(0, 1 / math.sqrt(h), (h, v)),
        "b2": np.zeros(v),
    }
    m = {k: np.zeros_like(p) for k, p in params.items()}
    s = {k: np.zeros_like(p) for k, p in params.items()}
    lr, batch = 0.01, 128
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(data) - w - 1, batch)
        ctx = np.stack([data[i : i + w] for i in idx])
        y = data[idx + w]
        x = params["emb"][ctx].reshape(batch, -1)
        pre = x @ params["w1"] + params["b1"]
        act = np.maximum(pre, 0)
        logits = act @ params["w2"] + params["b2"]
        logits -= logits.max(1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(1, keepdims=True)
        d_logits = prob
        d_logits[np.arange(batch), y] -= 1
        d_logits /= batch
        grads = {
            "w2": act.T @ d_logits,
            "b2": d_logits.sum(0),
        }
        d_act = d_logits @ params["w2"].T
        d_pre = d_act * (pre > 0)
        grads["w1"] = x.T @ d_pre
        grads["b1"] = d_pre.sum(0)
        d_x = (d_pre @ params["w1"].T).reshape(batch, w, e)
        g_emb = np.zeros_like(params["emb"])
        np.add.at(g_emb, ctx, d_x)
        grads["emb"] = g_emb
        for k in params:  # Adam
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            s[k] = 0.999 * s[k] + 0.001 * grads[k] ** 2
            mh = m[k] / (1 - 0.9**step)
            sh = s[k] / (1 - 0.999**step)
            params[k] -= lr * mh / (np.sqrt(sh) + 1e-8)
    # held-out cross-entropy in bits per character
    test = corpus(np.random.default_rng(999), 3000)
    ctx = np.stack([test[i : i + w] for i in range(len(test) - w)])
    y = test[w:]
    logits = np.maximum(params["emb"][ctx].reshape(len(y), -1) @ params["w1"] + params["b1"], 0)
    logits = logits @ params["w2"] + params["b2"]
    logits -= logits.max(1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    params["_bits_per_char"] = np.array(-logp[np.arange(len(y)), y].mean() / math.log(2))
    return params


# -- exact fixed-point inference -------------------------------------------------


class Quantized:
    """The model in fixed point with `bits` fractional bits: integer weights,
    integer activations, products rescaled with an arithmetic right shift (as
    Reverie's `*/` does for Q31)."""

    def __init__(self, params: Dict[str, np.ndarray], bits: int) -> None:
        self.bits = bits
        q = lambda a: np.round(a * (1 << bits)).astype(np.int64)  # noqa: E731
        self.emb, self.w1, self.b1 = q(params["emb"]), q(params["w1"]), q(params["b1"])
        self.w2, self.b2 = q(params["w2"]), q(params["b2"])

    def forward(self, ctx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.emb[ctx].reshape(len(ctx), -1)
        pre = ((x @ self.w1) >> self.bits) + self.b1
        act = np.maximum(pre, 0)
        logits = ((act @ self.w2) >> self.bits) + self.b2
        return pre, act, logits

    def greedy(self, prompts: np.ndarray, k: int) -> np.ndarray:
        ctx = prompts.copy()
        out = np.zeros((len(prompts), k), dtype=np.int64)
        for t in range(k):
            _, _, logits = self.forward(ctx)
            nxt = logits.argmax(1)  # ties -> lowest index, deterministic
            out[:, t] = nxt
            ctx = np.concatenate([ctx[:, 1:], nxt[:, None]], 1)
        return out


# -- exact information measures -----------------------------------------------


def classes(observations: np.ndarray) -> np.ndarray:
    """Class id of each row (prompts with identical observations share one)."""
    flat = observations.reshape(len(observations), -1)
    _, inverse = np.unique(flat, axis=0, return_inverse=True)
    return inverse.reshape(-1)


def _clean(value: float) -> float:
    """Exact zeros print as 0, not -0."""
    return 0.0 if abs(value) < 1e-12 else value


def conditional_entropy(class_ids: np.ndarray, weights: np.ndarray) -> float:
    """H(P | O) in bits for prompt distribution `weights` (sums to 1), where
    O is a deterministic function of P given by `class_ids`."""
    mass = np.bincount(class_ids, weights=weights)
    inside = weights / mass[class_ids]
    nz = weights > 0
    return _clean(float(-(weights[nz] * np.log2(inside[nz])).sum()))


def index_code_bits(class_ids: np.ndarray) -> float:
    """Average length of the constructive witness: the prompt's index inside
    its class, written in ceil(log2 |class|) bits."""
    sizes = np.bincount(class_ids)[class_ids]
    return float(np.ceil(np.log2(sizes)).mean())


def summarize(class_ids: np.ndarray, uniform: np.ndarray, natural: np.ndarray) -> Dict[str, float]:
    sizes = np.bincount(class_ids)
    return {
        "classes": int(len(sizes)),
        "singletons_fraction": float((sizes[class_ids] == 1).mean()),
        "H_uniform_bits": conditional_entropy(class_ids, uniform),
        "H_natural_bits": conditional_entropy(class_ids, natural),
        "index_code_bits": index_code_bits(class_ids),
        "worst_class": int(sizes.max()),
        "smallest_class": int(sizes.min()),
    }


def coded_bits(values: np.ndarray) -> int:
    z = np.where(values >= 0, 2 * values, -2 * values - 1) + 1
    return int((2 * np.floor(np.log2(z)) + 1).sum())


def natural_prior(model: Quantized, prompts: np.ndarray, rng_seed: int = 0) -> np.ndarray:
    """P(prompt) under the float model's own chain rule, from a fixed start
    context of spaces (computed with the integer model's logits)."""
    w = prompts.shape[1]
    logp = np.zeros(len(prompts))
    ctx = np.zeros_like(prompts)  # index 0 is the space
    for t in range(w):
        _, _, logits = model.forward(ctx)
        z = logits / float(1 << model.bits)
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        logp += lp[np.arange(len(prompts)), prompts[:, t]]
        ctx = np.concatenate([ctx[:, 1:], prompts[:, t : t + 1]], 1)
    p = np.exp(logp - logp.max())
    return p / p.sum()


# -- study --------------------------------------------------------------------


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    params = train(args, rng)
    v, w = len(ALPHABET), args.window
    prompts = np.array(list(itertools.product(range(v), repeat=w)), dtype=np.int64)
    n = len(prompts)
    prompt_bits = w * math.log2(v)
    uniform = np.full(n, 1.0 / n)

    reference = Quantized(params, 16)
    natural = natural_prior(reference, prompts)
    natural_entropy = float(-(natural[natural > 0] * np.log2(natural[natural > 0])).sum())

    precisions = [16, 12, 10, 8, 6, 5, 4, 3, 2, 1]
    if args.self_test:
        precisions = [12, 4, 2]
    by_precision = []
    for bits in precisions:
        model = Quantized(params, bits)
        pre, act, logits = model.forward(prompts)
        answers = model.greedy(prompts, args.answer)
        row = {
            "fraction_bits": bits,
            "pre_activation": summarize(classes(pre), uniform, natural),
            "hidden_state": summarize(classes(act), uniform, natural),
            "logits": summarize(classes(logits), uniform, natural),
            "answer": {
                str(k): summarize(classes(answers[:, :k]), uniform, natural)
                for k in range(1, args.answer + 1)
            },
            # The per-operation ReLU witness (negative parts), summed over all
            # prompts, against the information about the prompt the ReLU
            # actually destroys globally.
            "relu_local_witness_bits_per_prompt": coded_bits(np.minimum(pre, 0)) / n,
            "relu_global_loss_bits_per_prompt": summarize(classes(act), uniform, natural)[
                "H_uniform_bits"
            ]
            - summarize(classes(pre), uniform, natural)["H_uniform_bits"],
        }
        by_precision.append(row)

    # Secret slot audit: template "s??ts" style - a fixed public prefix and
    # suffix with a secret two-character slot in the middle.
    audit = []
    template = [ALPHABET.index(c) for c in " taas"[:w]]
    slot = [2, 3]
    secrets = np.array(list(itertools.product(range(v), repeat=len(slot))))
    filled = np.tile(np.array(template), (len(secrets), 1))
    for i, pos in enumerate(slot):
        filled[:, pos] = secrets[:, i]
    for bits in (16, 8, 4, 2):
        model = Quantized(params, bits)
        _, act, logits = model.forward(filled)
        answers = model.greedy(filled, args.answer)
        entry = {"fraction_bits": bits}
        for name, obs in (
            ("hidden_state", act),
            ("logits", logits),
            ("answer_3", answers[:, :3]),
            ("answer_12", answers[:, : args.answer]),
        ):
            ids = classes(obs)
            sizes = np.bincount(ids)[ids]
            entry[name] = {
                "min_anonymity": int(sizes.min()),
                "mean_anonymity": float(sizes.mean()),
                "uniquely_identified_fraction": float((sizes == 1).mean()),
            }
        audit.append(entry)

    return {
        "schema": "reverie_prompt_recovery_v1",
        "config": {
            "alphabet": ALPHABET,
            "window": w,
            "embed": args.embed,
            "hidden": args.hidden,
            "train_steps": args.steps,
            "answer_tokens": args.answer,
            "seed": args.seed,
            "prompts": n,
        },
        "model_bits_per_char": float(params["_bits_per_char"]),
        "prompt_entropy_uniform_bits": prompt_bits,
        "prompt_entropy_natural_bits": natural_entropy,
        "by_precision": by_precision,
        "secret_slot_audit": {"secrets": len(secrets), "rows": audit},
    }


def markdown(result: Dict[str, Any]) -> str:
    cfg = result["config"]
    hu, hn = result["prompt_entropy_uniform_bits"], result["prompt_entropy_natural_bits"]
    rows = result["by_precision"]
    top = rows[0]
    lines = [
        "# Prompt recovery, measured exactly",
        "",
        f"Generated by `scripts/prompt_recovery_study.py`. Model: neural n-gram LM over "
        f"`{cfg['alphabet']!r}` ({len(cfg['alphabet'])} symbols), context {cfg['window']}, "
        f"embedding {cfg['embed']}, {cfg['hidden']} ReLU units, "
        f"{result['model_bits_per_char']:.2f} bits/char held out. All "
        f"{cfg['prompts']:,} prompts of length {cfg['window']} are enumerated, so every number is "
        "exact.",
        "",
        f"A uniformly random prompt carries {hu:.1f} bits; a prompt drawn from the model's own "
        f"distribution carries {hn:.1f} bits. The tables give **H(prompt | observable)**: the bits "
        "still missing after seeing the observable, which is also the smallest witness that "
        "recovers the prompt exactly.",
        "",
        f"## What each observable leaves unknown ({top['fraction_bits']}-bit fixed point)",
        "",
        "| observable | missing bits (uniform) | missing bits (natural) | prompts identified uniquely | largest class |",
        "|---|---:|---:|---:|---:|",
    ]
    def line(name, s):
        return (
            f"| {name} | {s['H_uniform_bits']:.2f} | {s['H_natural_bits']:.2f} | "
            f"{s['singletons_fraction']:.1%} | {s['worst_class']:,} |"
        )
    lines.append(line("hidden pre-activations", top["pre_activation"]))
    lines.append(line("hidden state (after ReLU)", top["hidden_state"]))
    lines.append(line("logits", top["logits"]))
    for k in sorted(top["answer"], key=int):
        if int(k) in (1, 2, 3, 4, 6, 8, 12) or int(k) == cfg["answer_tokens"]:
            lines.append(line(f"greedy answer, {k} token(s)", top["answer"][k]))
    lines += [
        "",
        "## Precision: when hidden states stop being injective",
        "",
        "| fraction bits | missing bits from hidden state | from logits | from 12-token answer | ReLU local witness (bits/prompt) | ReLU global loss (bits/prompt) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        last = r["answer"][str(cfg["answer_tokens"])]
        lines.append(
            f"| {r['fraction_bits']} | {r['hidden_state']['H_uniform_bits']:.3f} | "
            f"{r['logits']['H_uniform_bits']:.3f} | {last['H_uniform_bits']:.2f} | "
            f"{r['relu_local_witness_bits_per_prompt']:.0f} | {r['relu_global_loss_bits_per_prompt']:.3f} |"
        )
    audit = result["secret_slot_audit"]
    lines += [
        "",
        f"## Secret-slot audit ({audit['secrets']} possible two-character secrets in a fixed template)",
        "",
        "| fraction bits | observable | worst-case anonymity set | mean anonymity set | secrets identified uniquely |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in audit["rows"]:
        for name in ("hidden_state", "logits", "answer_3", "answer_12"):
            a = r[name]
            lines.append(
                f"| {r['fraction_bits']} | {name.replace('_', ' ')} | {a['min_anonymity']} | "
                f"{a['mean_anonymity']:.1f} | {a['uniquely_identified_fraction']:.0%} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.self_test:
        args.window, args.steps, args.answer, args.hidden = 4, 300, 4, 16
    result = run(args)
    if args.self_test:
        for row in result["by_precision"]:
            ids_ok = row["hidden_state"]["index_code_bits"] >= row["hidden_state"]["H_uniform_bits"] - 1e-9
            assert ids_ok, row
            assert row["hidden_state"]["H_uniform_bits"] >= row["pre_activation"]["H_uniform_bits"] - 1e-9
            answers = [row["answer"][str(k)]["H_uniform_bits"] for k in range(1, args.answer + 1)]
            assert all(a >= b - 1e-9 for a, b in zip(answers, answers[1:])), answers
        print("ok: prompt recovery self-test passed")
        return 0
    text = markdown(result)
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
