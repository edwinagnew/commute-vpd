"""Figure for qwen_prefix_test's copy control (job 39181), read from results/large/copy.jsonl.
Plan: reports/plans/2026-10-28_plan_qwen-prefix-test.md; analysis: reports/analyses/2026-10-28_analysis_qwen-prefix-test.md.

  copy_by_k.png  the share of the 256 copy-control items whose continuation matches the first k words of the remainder
                 (all of it when the remainder is shorter than k), against k; beside it, the share of items whose
                 remainder has reached the start of the question's last sentence by word k; the filed bar of 0.90 and
                 the scored full-remainder exact match as labelled lines. Unfiled: k-word matching is not a filed rule.

Prints the matched shares at k = 5, 8, 10, 15 and the failures' first diverging word counts.

Usage (repo root): ~/commute-agents/fleet/cpu-run uv run python experiments/qwen_prefix_test/plot.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COPY, FIG = HERE / "results" / "large" / "copy.jsonl", HERE / "figures"
BAR = 0.90  # filed copy-control bar (predictions.json, G1)
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
COPY_C, FAIL_C = "#2a78d6", "#c0392b"


def first_k(rec: dict, k: int) -> bool:
    rem, cont = rec["remainder"].split(), rec["continuation"].split()
    m = min(k, len(rem))
    return cont[:m] == rem[:m]


def last_sentence_start(remainder: str) -> int:
    """Word index in the remainder where the question's last sentence starts; 0 if it starts before the remainder."""
    words = remainder.split()
    starts = [i for i in range(1, len(words)) if re.search(r"[.!?]$", words[i - 1])]
    return starts[-1] if starts else 0


def main() -> None:
    recs = [json.loads(line) for line in COPY.read_text().splitlines()]
    n, ks = len(recs), list(range(1, 46))
    matched = [sum(first_k(r, k) for r in recs) / n for k in ks]
    reached = [sum(last_sentence_start(r["remainder"]) < k for r in recs) / n for k in ks]
    em = sum(r["em"] for r in recs) / n
    for k in (5, 8, 10, 15):
        print(f"k = {k}: matched {matched[k - 1]:.3f}")
    print(f"full exact match {em:.3f} over {n}")
    fails = [r for r in recs if not r["em"]]
    first = Counter()
    for r in fails:
        rem, cont = r["remainder"].split(), r["continuation"].split()
        j = next(i for i in range(len(rem)) if i >= len(cont) or cont[i] != rem[i])
        first[cont[j] if j < len(cont) else "<end>"] += 1
    print("first diverging word among", len(fails), "failures:", first.most_common(6))

    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axhline(BAR, color=INK, lw=1, ls="--")
    ax.text(45, BAR + 0.015, "filed bar for the copy control, 0.90", ha="right", va="bottom", color=INK, fontsize=9)
    ax.axhline(em, color=FAIL_C, lw=1, ls=":")
    ax.text(45, em - 0.015, f"scored: whole remainder copied, {em:.2f}", ha="right", va="top", color=FAIL_C, fontsize=9)
    ax.plot(ks, matched, color=COPY_C, lw=2, marker="o", ms=3.5, label="copied the remainder's first k words")
    ax.plot(ks, reached, color=MUTED, lw=2, label="remainder has reached the question's last sentence by word k")
    ax.set_xlim(0, 46)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("k, words into the remainder")
    ax.set_ylabel(f"share of the {n} questions")
    ax.set_title("Shown the whole question, how far does the base model copy before it starts answering?", fontsize=10.5)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.12), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "copy_by_k.png", dpi=150)
    print("wrote", FIG / "copy_by_k.png")


if __name__ == "__main__":
    main()
