"""Figures for qwen_gsm8k_baseline: do long answers fail more, and do they fit the 512-token window? How many prompts
give mixed right and wrong samples? Why do answers lack a strict "The answer is N" line? Prints every number shown.

Usage (repo root, CPU): uv run python experiments/qwen_gsm8k_baseline/plot.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from itertools import pairwise
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
WINDOW, N_SAMPLES = 512, 8
RIGHT, WRONG, MIXED, UNMIXED = "C0", "C1", "C2", "C7"


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (RESULTS / "large" / name).open()]


def total(r: dict) -> int:
    return r["prompt_tokens"] + r["response_tokens"]


def miss_reason(r: dict) -> str:
    """Why a response has no strict 'The answer is N' line, read off its text."""
    t = r["response"]
    if r["truncated"]:
        return "cut off at 512 new tokens"
    if re.search(r"The answer is\s*\$?\s*\\boxed", t):
        return r"'The answer is \boxed{N}'"
    if re.search(r"The answer is\s*\*\*", t):
        return "'The answer is **N**'"
    if re.search(r"The answer is N\b", t):
        return "'The answer is N.' literally"
    return "other (LaTeX wrapper)"


def lengths_figure(greedy: list[dict], overall: float) -> None:
    bins = np.arange(112, 738, 25)  # 512 is a bin edge
    tots = np.array([total(r) for r in greedy])
    ok = np.array([r["strict"] for r in greedy])
    fig, (top, bot) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True, height_ratios=[3, 2])
    top.hist([tots[ok], tots[~ok]], bins=bins, stacked=True, color=[RIGHT, WRONG], edgecolor="white",
             label=["graded right (strict)", "graded wrong (strict)"])
    centres, accs, ns = [], [], []
    for lo, hi in pairwise(bins):
        sel = (tots > lo) & (tots <= hi)
        if sel.sum() >= 10:
            centres.append(float(lo + hi) / 2)
            accs.append(float(ok[sel].mean()))
            ns.append(int(sel.sum()))
    bot.plot(centres, accs, "o-", color="k", label="strict accuracy in the bin (bins with 10+ answers)")
    bot.axhline(overall, color="k", ls=":", lw=1, label=f"all 1,319 answers: {overall:.3f}")
    for ax in (top, bot):
        ax.axvline(WINDOW, color="C3", ls="--", lw=1.2)
    top.text(WINDOW + 5, top.get_ylim()[1] * 0.9, f"512-token window\n{(tots > WINDOW).sum()} answers past it",
             color="C3", fontsize=8, va="top")
    top.set_ylabel("greedy answers per 25-token bin")
    top.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.75))
    bot.set_ylabel("strict accuracy")
    bot.set_ylim(0, 1)
    bot.set_xlabel("prompt + greedy response, tokens (chat template and end token included)")
    bot.legend(fontsize=8, loc="lower left")
    top.set_title("Do longer answers fail more, and how many run past the 512-token window?", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "lengths.png", dpi=150)
    print(FIGURES / "lengths.png")
    print("bins (centre, n, strict acc):", list(zip(centres, ns, [round(a, 3) for a in accs], strict=True)))
    over = [r for r in greedy if total(r) > WINDOW]
    under = [r for r in greedy if total(r) <= WINDOW]
    print(f"over 512: n {len(over)}, strict {np.mean([r['strict'] for r in over]):.3f}, "
          f"lenient {np.mean([r['lenient'] for r in over]):.3f}, truncated {sum(r['truncated'] for r in over)}, "
          f"no strict line {sum(r['strict_pred'] is None for r in over)}; "
          f"within 512: n {len(under)}, strict {np.mean([r['strict'] for r in under]):.3f}")
    print(f"median total: right {np.median(tots[ok]):.0f}, wrong {np.median(tots[~ok]):.0f}")


def samples_figure(samples: list[dict]) -> None:
    by: dict[int, list[dict]] = defaultdict(list)
    for r in samples:
        by[r["idx"]].append(r)
    strict = Counter(sum(x["strict"] for x in v) for v in by.values())
    lenient = Counter(sum(x["lenient"] for x in v) for v in by.values())
    n_prompts, k = len(by), np.arange(N_SAMPLES + 1)
    p = np.mean([r["strict"] for r in samples])
    same_p = np.array([comb(N_SAMPLES, i) * p**i * (1 - p) ** (N_SAMPLES - i) for i in k]) * n_prompts
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colours = [UNMIXED] + [MIXED] * (N_SAMPLES - 1) + [UNMIXED]
    ax.bar(k - 0.2, [strict[i] for i in k], width=0.4, color=colours, edgecolor="white",
           label="strict grade (green: mixed, an RL signal; grey: all wrong or all right)")
    ax.bar(k + 0.2, [lenient[i] for i in k], width=0.4, color="none", edgecolor="k", hatch="//",
           label="lenient grade (last number)")
    ax.plot(k, same_p, "k:", marker=".", label=f"if every prompt had the same accuracy {p:.2f}")
    mixed = lambda c: sum(c[i] for i in range(1, N_SAMPLES)) / n_prompts
    ax.set_xticks(k)
    ax.set_xlabel(f"samples graded right, of {N_SAMPLES} at temperature 0.7")
    ax.set_ylabel(f"prompts (first {n_prompts} test prompts)")
    ax.set_title(f"How many prompts give both right and wrong samples? Strict: {mixed(strict):.2f}, "
                 f"lenient: {mixed(lenient):.2f}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "samples.png", dpi=150)
    print(FIGURES / "samples.png")
    print(f"prompts {n_prompts}; strict counts {sorted(strict.items())}; lenient counts {sorted(lenient.items())}")
    print(f"mixed strict {mixed(strict):.3f}, lenient {mixed(lenient):.3f}, same-accuracy reference "
          f"{1 - p**N_SAMPLES - (1 - p) ** N_SAMPLES:.3f}; mean strict {p:.3f}")


def format_figure(greedy: list[dict]) -> None:
    misses = [r for r in greedy if r["strict_pred"] is None]
    reasons = Counter(miss_reason(r) for r in misses)
    names = [n for n, _ in reasons.most_common()]
    right = [sum(r["lenient"] for r in misses if miss_reason(r) == n) for n in names]
    wrong = [reasons[n] - c for n, c in zip(names, right, strict=True)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(names))[::-1]
    ax.barh(y, right, color=RIGHT, edgecolor="white", label="last number equals the gold answer")
    ax.barh(y, wrong, left=right, color=WRONG, edgecolor="white", label="last number wrong")
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xlabel(f"greedy answers with no strict 'The answer is N' line ({len(misses)} of {len(greedy)})")
    ax.set_title("Why do answers miss the strict format, and were they right anyway?", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "format.png", dpi=150)
    print(FIGURES / "format.png")
    print("no strict line:", {n: (reasons[n], c) for n, c in zip(names, right, strict=True)}, "(count, lenient right)")
    dis = [r for r in greedy if r["strict"] != r["lenient"]]
    print(f"strict/lenient disagree {len(dis)}: lenient-only right {sum(r['lenient'] for r in dis)}")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    base = json.loads((RESULTS / "baseline.json").read_text())
    greedy, samples = load("greedy.jsonl"), load("samples.jsonl")
    assert len(greedy) == base["greedy"]["k"] and len(samples) == base["samples"]["k_prompts"] * N_SAMPLES
    lengths_figure(greedy, base["greedy"]["greedy_strict"])
    samples_figure(samples)
    format_figure(greedy)


if __name__ == "__main__":
    main()
