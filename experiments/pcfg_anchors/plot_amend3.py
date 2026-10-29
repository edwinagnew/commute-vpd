"""Figures for E55 Amendment 3, read from results/amend3.json only; no model.
Analysis: reports/analyses/2026-10-29_analysis_e55-stage-a.md, section "Amendment 3".

  amend3_swap.png   on Stage A's 4,000 spurious test rows (count(a) set to count(b) - 1 by a/c swaps): the share each
                    state answers right against the number of letters swapped. LOW's own count of a is the
                    do-nothing reference; the fine-tunes trained on swap-built rows are the flat reference at 1
  amend3_pairs.png  on natural rows in the pairs (count a, count b) Amendments 1 and 2 never read: top, the bases' answers
                    against the count they should give (the validity test); bottom, the all-spurious fine-tunes'
                    most common answers against the one-third rule and the midpoint

Usage (repo root): ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/pcfg_anchors/plot_amend3.py
"""

from __future__ import annotations

import numpy as np
from plot import FIG, INK, M_C, MUTED, N_C, R_C, W_C, finish, load, plt

BINS = ("0", "1", "2", "3", "4")  # "4" is 4 or more
SWAP_STATES = (  # key, words, colour, line style, marker
    ("LOW_C_a_40", "LOW before fine-tuning, its count of a (do-nothing reference)", INK, "-", "o"),
    ("HIGH_C_a_40", "HIGH before fine-tuning, its count of a", MUTED, "-", "s"),
    ("N1_10000", "fast, no spurious rows, 10K", N_C, "-", "o"),
    ("R_10000", "HIGH base, no spurious rows, 10K", R_C, "-", "*"),
    ("M3_1000", "slow, 80% spurious, 1K", M_C, "--", "^"),
    ("W1_1000", "slow, all spurious, 1K", W_C, "--", "o"),
    ("W1_10000", "slow, all spurious, 10K", W_C, "-", "D"),
)
PAIRS = ("17,10", "16,9", "15,9", "17,9", "9,14", "9,17", "17,13", "9,16", "18,11", "11,18")


def fig_swap(out: str = "amend3_swap.png",
             title: str = "Every model trained only on natural text miscounts the swap-built rows, more the larger the swap") -> None:  # fmt: skip
    r = load("amend3.json")
    hist = r["swap"]["swap_histograms"]["te1"]
    rows = [hist[b] for b in BINS[:-1]] + [sum(v for k, v in hist.items() if int(k) >= 4)]
    x = np.arange(len(BINS))
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    for key, words, col, ls, mk in SWAP_STATES:
        y = [r["swap"]["reads"][key]["by_swap"][b] for b in BINS]
        ax.plot(x, y, color=col, ls=ls, marker=mk, lw=1.6, ms=6, label=words)
    for xi, n in zip(x, rows, strict=True):
        ax.text(float(xi), -0.1, f"{n:,} rows", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, ["0, natural", "1", "2", "3", "4 or more"])
    ax.set_xlabel("letters swapped between a and c to build the row")
    ax.set_ylabel("share right: count(a) + 1 for 'C b 40', count(a) for 'C a 40'")
    ax.set_ylim(-0.15, 1.05)
    fig.legend(fontsize=7.5, frameon=False, loc="lower center", ncol=2)
    finish(fig, out, title, bottom=0.27)


def fig_pairs(out: str = "amend3_pairs.png",
              title: str = "In new pairs only (17, 10) has both bases counting; there the slow fine-tune answers 16, not the rule's 15") -> None:  # fmt: skip
    r = load("amend3.json")
    p = r["pairs"]
    ab = [tuple(int(v) for v in k.split(",")) for k in PAIRS]
    x = np.arange(len(PAIRS))
    fig, (top, bot) = plt.subplots(2, 1, figsize=(10.0, 7.6), sharex=True, height_ratios=(1, 1.3))
    valid = [p["LOW_C_a_40"][k]["a"] >= 0.95 and p["N1_10000"][k]["b"] >= 0.95 for k in PAIRS]
    for ax in (top, bot):
        for xi, ok in zip(x, valid, strict=True):
            if ok:
                ax.axvspan(xi - 0.45, xi + 0.45, color="#e8f2e0", zorder=0)
    top.scatter(x - 0.12, [a for a, _ in ab], marker="_", s=260, color=INK, lw=2, label="count a")
    top.scatter(x - 0.12, [p["LOW_C_a_40"][k]["mode"] for k in PAIRS], color=INK, s=30, label="LOW's answer to 'C a 40'")
    top.scatter(x + 0.12, [b for _, b in ab], marker="_", s=260, color=N_C, lw=2, label="count b")
    top.scatter(x + 0.12, [p["N1_10000"][k]["mode"] for k in PAIRS], color=N_C, s=30,
                label="fast no-spurious fine-tune's answer, 10K")  # fmt: skip
    top.set_ylabel("most common answer")
    top.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
    top.set_ylim(7.5, 21)
    third = [a + 1 + (b - a) // 3 for a, b in ab]  # = a + 1 + floor((d + 1) / 3), d = b - a - 1
    mid = [a + 1 + int((b - a - 1) / 2) for a, b in ab]
    bot.scatter(x, third, marker="_", s=420, color=INK, lw=2.4, label="one-third rule")
    bot.scatter(x, mid, marker="_", s=420, color=MUTED, lw=2.4, label="midpoint rule")
    for key, words, dx, mk, fill in (("W1_10000", "slow all-spurious 10K", -0.16, "o", W_C),
                                     ("W1s1_10000", "same, seed 1", 0.0, "o", "none"),
                                     ("W2_10000", "fast all-spurious 10K", 0.16, "D", W_C)):  # fmt: skip
        y = [p[key][k]["mode"] for k in PAIRS]
        bot.scatter(x + dx, y, marker=mk, s=34, facecolor=fill, edgecolor=W_C, lw=1.3, label=words, zorder=3)
    bot.set_ylabel("most common answer to 'C b 40'")
    bot.set_ylim(9.5, 19)
    bot.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper right")
    rows = [p["LOW_C_a_40"][k]["n"] for k in PAIRS]
    bot.set_xticks(x, [f"({k.replace(',', ', ')})\n{n} rows" for k, n in zip(PAIRS, rows, strict=True)], fontsize=7.5)
    bot.set_xlabel("pair (count a, count b) of natural rows; green: both bases count there, so the pair can test a rule")
    finish(fig, out, title)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig_swap()
    fig_pairs()


if __name__ == "__main__":
    main()
