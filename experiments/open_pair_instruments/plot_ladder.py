"""E55 C3 step ladder's figures, one per claim, from results/ladder.json and C3's results/run.json (no model).

  figures/ladder_k.png      part A: on C3's 32 saved states, the share of D revived by the top K units per map, and the
                            rank-1-minus-cipher gap per K beside its bar, with the filed reading (best K per state)
  figures/ladder_steps.png  part B: revival R and reverse half-life per state at steps 10, 30, 100, 300, with each step's
                            filed bar drawn from the cipher's mean in the filed direction
  figures/ladder_size.png   held-out P(value) per step, with C3's cipher at its scored step; and the half-life against
                            the update norm for all 36 ladder states, with C3's 24 scored states as the reference

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 4G .venv/bin/python experiments/open_pair_instruments/plot_ladder.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot import COLOR, INK, style

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
POLES = ("rank1", "cipher", "elicit")
LABEL = {"rank1": "rank-1 (old end)", "cipher": "cipher (new end)", "elicit": "elicit (placed)", "copy": "copy (reported)"}
KS = ("13", "40", "80", "160", "320")
STEPS = (10, 30, 100, 300)
MARK = {10: "o", 30: "s", 100: "^", 300: "D"}
# C3's cipher at its scored step (8 or 9), held-out P(value) from job 46815's fine-tune log, the six states it trained
# (seed 0 retrained, seeds 3-7); seeds 1-2 are E58's saved states and their step-8/9 value is not in that log.
C3_CIPHER_P = (0.4498, 0.3903, 0.3610, 0.3602, 0.3175, 0.4407)
C3_CIPHER_STEPS = (9, 8, 8, 9, 8, 8)


def share(rec: dict[str, Any], K: str) -> float:
    return rec["Delta"][K] / rec["D"]


def k_fig(L: dict[str, Any]) -> None:
    kr, sc = L["kreread"], L["score_kreread"]
    x = np.array([int(k) for k in KS])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for m in (*POLES, "copy"):
        v = np.array([[share(kr[f"{m}_s{s}"], K) for K in KS] for s in range(8)])
        axes[0].errorbar(x, v.mean(0), yerr=v.std(0, ddof=1), color=COLOR[m], marker="o", ms=4, capsize=2,
                         ls="--" if m == "copy" else "-", label=LABEL[m])
    axes[0].set_xscale("log")
    axes[0].set_xticks(x, KS)
    axes[0].set_ylim(0, 0.85)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    style(axes[0], "Share of D given back by pruning the top K units, 8 seeds per map\n(mean ± seed sd; random removal subtracted)",
          "revival at K units, share of D")
    axes[0].set_xlabel("K, MLP units zeroed (of 12,288)", color=INK)
    gaps = [sc["R_per_K"][K]["gap"] for K in KS]
    bars = [3 * sc["R_per_K"][K]["noise"] for K in KS]
    axes[1].plot(x, gaps, color=COLOR["rank1"], marker="o", label="rank-1 minus cipher, at each K")
    axes[1].plot(x, bars, color=INK, ls="--", lw=1, label="bar: 3 × largest pole seed sd\n(per-K floor taken over all K; unfiled)")
    axes[1].axhline(0, color=INK, lw=0.7)
    xf = 600
    axes[1].scatter([xf], [sc["R"]["gap"]], color=COLOR["rank1"], marker="*", s=150, zorder=4,
                    label="filed R: each state's best K")
    axes[1].hlines(3 * sc["R"]["noise"], xf * 0.8, xf * 1.25, color=INK, lw=1.6)
    axes[1].text(xf, 3 * sc["R"]["noise"] + 0.008, f"filed bar {3 * sc['R']['noise']:.3f}", fontsize=7, color=INK, ha="center")
    axes[1].text(xf, sc["R"]["gap"] - 0.012, f"gap {sc['R']['gap']:.3f}:\nno gap", fontsize=7, color=COLOR["rank1"],
                 ha="center", va="top")
    axes[1].set_xscale("log")
    axes[1].set_xticks([*x, xf], [*KS, "best K"])
    axes[1].set_ylim(-0.02, 0.24)
    axes[1].legend(fontsize=7, frameon=False, loc="upper left")
    style(axes[1], "Rank-1 minus cipher at each K, beside its bar",
          "revival gap, share of D")
    axes[1].set_xlabel("K, MLP units zeroed", color=INK)
    fig.suptitle("On C3's 32 saved 10-step states, does pruning more units separate rank-1 from the cipher?",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "ladder_k.png", dpi=150)
    plt.close(fig)


def per_step(lad: dict[str, Any], m: str, t: int, key: str) -> list[float]:
    return [lad[f"{m}_s{s}_t{t}"][key] for s in range(3)]


def step_panel(ax: Any, L: dict[str, Any], key: str, d: int) -> None:
    lad, ps = L["ladder"], L["score_ladder"]["per_step"]
    rng = np.random.default_rng(0)
    off = {"rank1": -0.06, "cipher": 0.0, "elicit": 0.06}
    for m in POLES:
        means = []
        for t in STEPS:
            v = per_step(lad, m, t, key)
            xs = t * 10 ** (off[m] + rng.uniform(-0.012, 0.012, 3))
            ax.scatter(xs, v, s=16, color=COLOR[m], alpha=0.7, zorder=3)
            means.append(np.mean(v))
        ax.plot([t * 10 ** off[m] for t in STEPS], means, color=COLOR[m], lw=1.8, label=LABEL[m], zorder=4)
    for t in STEPS:
        rd = ps[str(t)][key]
        lo, bar = rd["means"]["cipher"], 3 * rd["noise"]
        xb = t * 10 ** 0.13
        ax.annotate("", (xb, lo + d * bar), (xb, lo), arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.1})
        ax.text(xb * 1.06, lo + d * bar, rd["status"], fontsize=7, color=INK, va="center")
    ax.set_xscale("log")
    ax.set_xticks(STEPS, [str(t) for t in STEPS])
    ax.set_xlabel("fine-tune step at which the state is read (lr 3e-5, Adam)", color=INK)


def steps_fig(L: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    step_panel(axes[0], L, "R", +1)
    axes[0].set_ylim(0.3, 0.95)
    style(axes[0], "Pruning revival R: old end expected higher\n(arrow: from the cipher's mean to where rank-1 must reach)",
          "revival R, best of K 13-320, share of D")
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    step_panel(axes[1], L, "t_half", -1)
    axes[1].set_ylim(0, 20)
    style(axes[1], "Reverse fine-tuning: old end expected lower\n(arrow: 3 × noise; noise is the 1-step floor at every step)",
          "half-life, reverse steps at lr 3e-6")
    fig.suptitle("At which update size do pruning revival and reverse fine-tuning separate rank-1 from the cipher? 3 seeds per map",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "ladder_steps.png", dpi=150)
    plt.close(fig)


def size_fig(L: dict[str, Any], run: dict[str, Any]) -> None:
    lad = L["ladder"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for m in POLES:
        v = np.array([per_step(lad, m, t, "p_heldout") for t in STEPS])
        axes[0].plot(STEPS, v.mean(1), color=COLOR[m], marker="o", label=LABEL[m])
        axes[0].scatter(np.repeat(STEPS, 3), v.ravel(), s=12, color=COLOR[m], alpha=0.6)
    axes[0].scatter(C3_CIPHER_STEPS, C3_CIPHER_P, s=40, facecolor="white", edgecolor=COLOR["cipher"], lw=1.3, zorder=4,
                    label="C3's cipher at its scored step 8-9\n(6 of 8 seeds, from the fine-tune log)")
    axes[0].axhline(1, color=INK, lw=0.7, ls=":")
    axes[0].set_xscale("log")
    axes[0].set_xticks([8, *STEPS], ["8", *[str(t) for t in STEPS]])
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    style(axes[0], "Has each map been learned? The cipher is half-learned at step 10", "held-out P(trained value)")
    axes[0].set_xlabel("fine-tune step", color=INK)
    for m in POLES:
        for t in STEPS:
            axes[1].scatter(per_step(lad, m, t, "norm"), per_step(lad, m, t, "t_half"), s=24, color=COLOR[m],
                            marker=MARK[t], zorder=3, label=LABEL[m] if t == 10 else None)
        axes[1].scatter(run["columns"]["norm"][m], run["columns"]["t_half"][m], s=22, facecolor="white",
                        edgecolor=COLOR[m], lw=1, zorder=2)
    for t in STEPS:
        axes[1].scatter([], [], marker=MARK[t], color="#9a9893", label=f"step {t}")
    axes[1].scatter([], [], facecolor="white", edgecolor=INK, label="C3's 8-seed states (step 8-10)")
    axes[1].set_ylim(0, 20)
    axes[1].legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
    style(axes[1], "The half-life falls as the update grows, and the cipher grows most", "half-life, reverse steps")
    axes[1].set_xlabel("update norm: Frobenius norms of the 24 matrices of δW, summed", color=INK)
    fig.suptitle("What changes between steps: the cipher finishes learning, and every update grows",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "ladder_size.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    L = json.loads((HERE / "results" / "ladder.json").read_text())
    run = json.loads((HERE / "results" / "run.json").read_text())
    k_fig(L)
    steps_fig(L)
    size_fig(L, run)


if __name__ == "__main__":
    main()
