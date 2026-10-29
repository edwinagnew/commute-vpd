"""Figures for E55 Amendment 1, read from results/amend1.json and the saved per-row answers only; no model.
Analysis: reports/analyses/2026-10-29_analysis_e55-stage-a.md, section "Amendment 1".

  amend1_trajectory.png  along the re-run of the slow all-spurious fine-tune (both seeds), the share of each rule's own
                         rows answered by count(a) + 1 and by the midpoint rule, with their filed bars; LOW before
                         fine-tuning (step 0) and the fast no-spurious fine-tune at 10K as references
  amend1_rounding.png    per state, the two rounding halves: the share answering the midpoint rounded down where
                         count(b) > count(a) + 1, and rounded up where count(b) < count(a) + 1, with their bars; a pure
                         midpoint answerer reads 1 on both (filed control)
  amend1_offsets.png     exploration, not filed: answer - (count(a) + 1) against count(b) - count(a), every row, for
                         the slow and the fast all-spurious fine-tunes at 10K, with the rules' lines

Usage (repo root): ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/pcfg_anchors/plot_amend1.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from plot import FIG, INK, M_C, MUTED, N_C, RES, W_C, finish, load, plt

STEPS = (1000, 2000, 3000, 5000, 7000, 10_000)
SEEDS = {"W1": ("seed 0", "-", "o"), "W1s1": ("seed 1", "--", "s")}
RULE_C = {"a_plus_1": W_C, "mid": M_C}
RULE_W = {"a_plus_1": "count(a) + 1", "mid": "midpoint of count(a) + 1 and count(b)"}


def traj_read(r: dict[str, Any], arm: str, it: int) -> dict[str, Any]:
    return r["trajectory"][arm]["states"][str(it)]["rule_read"]


def fig_trajectory(out: str = "amend1_trajectory.png",
                   title: str = "The slow all-spurious fine-tune leaves count(a) + 1 by 5K steps; the midpoint stays under its bar") -> None:  # fmt: skip
    r = load("amend1.json")
    low, n1 = r["controls"]["W0"]["rule_read"], r["controls"]["N1_10000"]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for rule, col in RULE_C.items():
        for arm, (_, ls, mk) in SEEDS.items():
            ax.plot(STEPS, [traj_read(r, arm, it)[rule]["S"] for it in STEPS], ls, marker=mk, color=col, lw=2, ms=6)
        ax.axhline(low[rule]["hi"], color=col, ls=":", lw=1.2)
        ax.text(10_300, low[rule]["hi"], f"bar for\n{RULE_W[rule].split(' of')[0]}", fontsize=7.5, color=col, va="center")
        ax.plot(0, low[rule]["S"], "P", color=col, ms=10, mec=INK)
        ax.plot(10_000, n1[rule]["S"], "X", color=col, ms=10, mec=INK)
    ax.axhline(0.1, color=MUTED, ls=":", lw=1)
    ax.text(10_300, 0.1, "lower bar", fontsize=7.5, color=MUTED, va="center")
    ax.axvspan(1000, 2000, color=W_C, alpha=0.08, lw=0)
    ax.text(1500, 1.02, "goes through\ncount a by the\nStage A checks", ha="center", va="top", fontsize=7.5, color=INK)
    ax.set_xlim(-400, 11_800)
    ax.set_ylim(-0.04, 1.05)
    ax.set_xticks((0,) + STEPS, ["0\nLOW"] + [f"{s // 1000}K" for s in STEPS])
    ax.set_xlabel("fine-tuning step")
    ax.set_ylabel("share of the rule's own rows answered by that rule")
    handles = [Line2D([], [], color=c, lw=2) for c in RULE_C.values()]
    handles += [Line2D([], [], color=INK, ls=ls, marker=mk) for _, ls, mk in SEEDS.values()]
    handles += [Line2D([], [], color=MUTED, marker="P", ls="", ms=9), Line2D([], [], color=MUTED, marker="X", ls="", ms=9)]
    labels = [f"answers {w}" for w in RULE_W.values()] + [f"slow all-spurious, {s}" for s, _, _ in SEEDS.values()]
    labels += ["LOW before fine-tuning", "fast fine-tune, no spurious rows, 10K"]
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center", ncol=3)
    finish(fig, out, title, bottom=0.27)


ROUND_STATES = (("LOW before\nfine-tuning", "controls", "W0"), ("fast, no\nspurious, 10K", "controls", "N1_10000"),
                ("slow all-\nspurious 1K", "saved", "W1_1000"), ("same,\nseed 1, 1K", "saved", "W1s1_1000"),
                ("fast all-\nspurious 1K", "saved", "W2_1000"), ("slow all-\nspurious 10K", "saved", "W1_10000"),
                ("same,\nseed 1, 10K", "saved", "W1s1_10000"), ("fast all-\nspurious 10K", "saved", "W2_10000"))  # fmt: skip


def fig_rounding(out: str = "amend1_rounding.png",
                 title: str = "Below count(a) + 1, only the fast all-spurious fine-tune at 10K answers the midpoint") -> None:  # fmt: skip
    r = load("amend1.json")
    reads = []
    for _, sec, key in ROUND_STATES:
        row = r[sec][key]
        reads.append((row["rule_read"] if key == "W0" else row)["rounding"])
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    x = np.arange(len(ROUND_STATES) + 1)
    halves = (("upper", "floor", "count(b) > count(a) + 1: midpoint rounded down", N_C),
              ("lower", "ceil", "count(b) < count(a) + 1: midpoint rounded up", W_C))
    for j, (half, rnd, lab, col) in enumerate(halves):
        vals = [1.0] + [rd[half][rnd]["S"] for rd in reads]  # a pure midpoint answerer reads 1 on both by construction
        off = (j - 0.5) * 0.36
        ax.bar(x + off, vals, 0.34, color=col, label=lab)
        for xi, v in zip(x + off, vals, strict=True):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5, color=INK)
        hi = reads[0][half][rnd]["hi"]
        ax.plot([x[0] - 0.5, x[-1] + 0.5], [hi, hi], color=col, ls=":", lw=1.2)
    ax.text(x[-1] + 0.55, 0.85, "bars", fontsize=7.5, color=INK, va="center")
    ax.set_xticks(x, ["pure midpoint\nanswerer\n(control)"] + [s[0] for s in ROUND_STATES], fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("share of the half's rows answering the midpoint")
    fig.legend(fontsize=8, frameon=False, loc="lower center", ncol=2)
    finish(fig, out, title, bottom=0.24)


def offsets_panel(ax: Axes, pred: np.ndarray, a: np.ndarray, b: np.ndarray, label: str) -> None:
    ba, off = b - a, pred - (a + 1)
    xs = np.arange(-6, 8)
    d = xs - 1
    lines = (("count(a) + 1", np.zeros_like(xs), W_C, "-"), ("count(b)", d, N_C, "-"),
             ("count(a), LOW's answer", -np.ones_like(xs), MUTED, "-"),
             ("midpoint, rounded towards count(a) + 1", np.sign(d) * (np.abs(d) // 2), M_C, "--"),
             ("one third of the way to count(b), rounded", np.rint(d / 3), INK, ":"))  # fmt: skip
    for lab, y, col, ls in lines:
        ax.plot(xs, y, ls, color=col, lw=1.6, label=lab)
    keys, n = np.unique(np.stack([ba, off]), axis=1, return_counts=True)
    ax.scatter(keys[0], keys[1], s=8 + 600 * n / n.max(), color=W_C, alpha=0.45, edgecolor=INK, lw=0.5, zorder=3)
    for (x, y), k in zip(keys.T, n, strict=True):
        ax.text(x + 0.18, y + 0.18, str(k), fontsize=6.5, color=INK)
    ax.set_title(label, fontsize=10, loc="left")
    ax.set_xlabel("count(b) - count(a) on the row")
    ax.set_ylim(-4.5, 3.8)


def fig_offsets(out: str = "amend1_offsets.png",
                title: str = "Exploration: at 10K the slow all-spurious fine-tune answers about a third of the way from count(a) + 1 to count(b)") -> None:  # fmt: skip
    a, b, _ = np.load(RES / "large/amend1/counts.npy").astype(np.int64)
    ans = RES / "large/amend1/answers"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, (tag, lab) in zip(axes, (("W1_10000", "slow all-spurious, 10K"), ("W2_10000", "fast all-spurious, 10K")), strict=True):
        offsets_panel(ax, np.load(ans / f"{tag}.npy").astype(np.int64), a, b, lab)
    axes[0].set_ylabel("answer - (count(a) + 1); circle area and number: rows")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, frameon=False)
    finish(fig, out, title, bottom=0.22)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig_trajectory()
    fig_rounding()
    fig_offsets()


if __name__ == "__main__":
    main()
