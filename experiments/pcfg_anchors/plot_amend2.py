"""Figures for E55 Amendment 2, read from results/amend2.json and the saved per-row answers only; no model.
Analysis: reports/analyses/2026-10-29_analysis_e55-stage-a.md, section "Amendment 2".

  amend2_fresh.png   per state, on the 20,000 fresh rows: the share of all rows answered by the one-third rule, beside
                     the best possible rule of count c alone (its reference) and the filed all-row bar of 0.95
  amend2_unseen.png  why the unseen read was void: LOW's answer to 'C b 40' against the row's count(a), on the fresh
                     natural rows and on the unseen rows built by a/c swaps, with the line answer = count(a)

Usage (repo root): ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/pcfg_anchors/plot_amend2.py
"""

from __future__ import annotations

import numpy as np
from plot import FIG, INK, MUTED, N_C, RES, W_C, finish, load, plt

STATES = (("LOW before\nfine-tuning", "controls", "fresh_LOW_C_b_40"), ("fast, no\nspurious, 10K", "controls", "fresh_N1_10000"),
          ("slow all-\nspurious 1K", "fresh", "W1_1000"), ("same,\nseed 1, 1K", "fresh", "W1s1_1000"),
          ("fast all-\nspurious 1K", "fresh", "W2_1000"), ("slow all-\nspurious 10K", "fresh", "W1_10000"),
          ("same,\nseed 1, 10K", "fresh", "W1s1_10000"), ("fast all-\nspurious 10K", "fresh", "W2_10000"))  # fmt: skip


def fig_fresh(out: str = "amend2_fresh.png",
              title: str = "On fresh rows the slow all-spurious fine-tune at 10K answers the one-third rule on 98% of rows") -> None:  # fmt: skip
    r = load("amend2.json")
    rows = [(r["controls"][k] if sec == "controls" else r["fresh"]["saved"][k]) for _, sec, k in STATES]
    x = np.arange(len(STATES))
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    third = [row["third"]["overall"] for row in rows]
    cbest = [row["c_read"]["c_best"] for row in rows]
    ax.bar(x - 0.18, third, 0.34, color=W_C, label="answers the one-third rule")
    ax.bar(x + 0.18, cbest, 0.34, color=MUTED, label="best possible rule of count c alone (reference)")
    for xi, v in zip(x - 0.18, third, strict=True):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.axhline(r["constants"]["all_row_min"], color=INK, ls=":", lw=1.2)
    ax.text(x[-1] + 0.55, r["constants"]["all_row_min"], "filed\nbar", fontsize=7.5, color=INK, va="center")
    ax.set_xticks(x, [s[0] for s in STATES], fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("share of the 20,000 fresh rows")
    fig.legend(fontsize=8, frameon=False, loc="lower center", ncol=2)
    finish(fig, out, title, bottom=0.22)


def fig_unseen(out: str = "amend2_unseen.png",
               title: str = "On the swap-built unseen rows LOW never answers count(a), so the unseen read was void") -> None:  # fmt: skip
    big = RES / "large/amend2"
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for name, col, lab in (("fresh", N_C, "fresh natural rows"), ("unseen", W_C, "unseen rows, count(a) set by a/c swaps")):
        a = np.load(big / f"{name}_counts.npy").astype(np.int64)[0]
        p = np.load(big / f"answers/{name}_LOW_C_b_40.npy").astype(np.int64)
        keys, n = np.unique(np.stack([a, p]), axis=1, return_counts=True)
        ax.scatter(keys[0], keys[1], s=6 + 400 * np.sqrt(n / n.max()), color=col, alpha=0.5, edgecolor=INK, lw=0.4, label=lab)
    ax.plot([0, 25], [0, 25], color=INK, ls="--", lw=1, label="answer = count(a), what LOW gives on natural rows")
    ax.text(10, -9, "non-number answers", fontsize=7.5, color=INK, va="center")
    ax.set_xlabel("count(a) in the window")
    ax.set_ylabel("LOW's answer to 'C b 40'; circle area: rows")
    ax.set_xlim(0, 25)
    ax.set_ylim(-10.5, 25)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    finish(fig, out, title)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig_fresh()
    fig_unseen()


if __name__ == "__main__":
    main()
