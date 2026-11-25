"""Figure for qwen_fresh_set (job 39230), read from results/fresh_set.json and results/large/{gsm8k,gsm1k}.jsonl.
Plan: reports/plans/2026-10-28_plan_qwen-fresh-set.md; analysis: reports/analyses/2026-10-28_analysis_qwen-fresh-set.md.

  accuracy_by_rule.png  strict accuracy of the unchanged model on GSM8K test and on GSM1k under the three rules for
                        answers that run past the 512-token window (dropped: the filed rule; all answers: how 0.756
                        was scored; capped: each answer cut where prompt plus response reach 512 tokens), +-1 binomial
                        SE; GSM8K is the reference in each group, with the filed "same" cut (0.04 below it) and
                        "much lower" cut (0.08 below it) drawn. The cuts are filed for the dropped rule only.

Prints, unfiled, the answers that do not fit the window per set: how many, how many right, how many ran out of the
512 new tokens with no end token, and how many finished wrong (with and without the final "The answer is N." line);
and the all-answers GSM1k accuracy had its answers fitted as often as GSM8K's, each group scoring as it did.

Usage (repo root): ~/commute-agents/fleet/cpu-run env PYTHONPATH=$PWD/src python -P experiments/qwen_fresh_set/plot.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES, LARGE, FIG = HERE / "results" / "fresh_set.json", HERE / "results" / "large", HERE / "figures"
SAME, MUCH = 0.04, 0.08  # filed cuts on D_1k (predictions.json, rules.verdict)
INK, MUTED, GRID, BAND = "#0b0b0b", "#8a8984", "#e6e5e0", "#efeee9"
REF_C, G1K_C = "#3d3c38", "#2a78d6"
RULES = [("strict_dropped", "k_fits", "answers that fit the window\n(the filed rule)"),
         ("strict_all", "k", "all answers, graded whole\n(how 0.756 was scored)"),
         ("strict_capped", "k", "all answers, each cut\nat the 512-token window")]


def long_answers(name: str) -> dict[str, int]:
    recs = [json.loads(line) for line in (LARGE / f"{name}.jsonl").read_text().splitlines()]
    out = [r for r in recs if not r["fits"]]
    done_wrong = [r for r in out if not r["truncated"] and not r["strict"]]
    return {"n": len(recs), "not_fitting": len(out), "right": sum(r["strict"] for r in out),
            "ran_out": sum(r["truncated"] for r in out), "ran_out_right": sum(r["strict"] for r in out if r["truncated"]),
            "finished_wrong": len(done_wrong),
            "finished_wrong_no_final_line": sum(r["strict_pred"] is None for r in done_wrong),
            "within_17_tokens": sum(r["prompt_tokens"] + r["response_tokens"] <= 512 + 17 for r in out)}


def main() -> None:
    res = json.loads(RES.read_text())
    g8, g1 = res["gsm8k"], res["gsm1k"]
    la = {s: long_answers(s) for s in ("gsm8k", "gsm1k")}
    for s, d in la.items():
        print(s, d, f"long-answer accuracy {d['right'] / d['not_fitting']:.3f}")
    f8 = g8["fits_share"]
    p_long1 = la["gsm1k"]["right"] / la["gsm1k"]["not_fitting"]
    mixed = f8 * g1["strict_dropped"] + (1 - f8) * p_long1
    print(f"GSM1k all answers at GSM8K's fits share {f8:.3f}: {mixed:.3f}; gap {g8['strict_all'] - mixed:+.3f}")

    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for i, (key, nkey, _) in enumerate(RULES):
        ref = g8[key]
        ax.fill_between([i - 0.32, i + 0.32], ref - SAME, ref, color=BAND, lw=0)
        ax.plot([i - 0.32, i + 0.32], [ref - MUCH] * 2, color=MUTED, lw=1, ls="--")
        for dx, d, c, m in ((-0.12, g8, REF_C, "s"), (0.12, g1, G1K_C, "o")):
            p, n = d[key], d[nkey]
            ax.errorbar(i + dx, p, yerr=math.sqrt(p * (1 - p) / n), color=c, marker=m, ms=8, lw=2, capsize=0)
            right = dx > 0
            ax.text(i + dx + (0.05 if right else -0.05), p, f"{p:.3f}", va="center", ha="left" if right else "right",
                    color=INK, fontsize=8.5)
        ax.text(i, 0.598, f"GSM8K minus GSM1k\n{ref - g1[key]:+.3f}", ha="center", va="bottom", color=INK, fontsize=9)
    ax.text(2.34, g8["strict_capped"] - SAME / 2, "within 0.04 of GSM8K:\nthe filed 'same' zone", ha="left",
            va="center", color=MUTED, fontsize=8)
    ax.text(2.34, g8["strict_capped"] - MUCH, "0.08 below GSM8K:\n'much lower' cut", ha="left", va="center",
            color=MUTED, fontsize=8)
    ax.errorbar([], [], color=REF_C, marker="s", ms=8, lw=2, label="GSM8K test, the reference")
    ax.errorbar([], [], color=G1K_C, marker="o", ms=8, lw=2, label="GSM1k, new problems")
    ax.set_xticks(range(len(RULES)), [r[2] for r in RULES], fontsize=9)
    ax.set_xlim(-0.5, 2.95)
    ax.set_ylim(0.59, 0.82)
    ax.set_ylabel("share answered strictly right, ±1 SE")
    ax.set_title("Does the unchanged Qwen3-1.7B score as well on GSM1k as on GSM8K test?\n"
                 "It depends on what is done with answers longer than the 512-token window", fontsize=10.5)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_by_rule.png", dpi=150)
    print("wrote", FIG / "accuracy_by_rule.png")


if __name__ == "__main__":
    main()
