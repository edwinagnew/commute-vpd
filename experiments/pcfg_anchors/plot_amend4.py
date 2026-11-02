"""Figure for E55 Amendment 4, read from results/amend4.json and Stage A's a2_<arm>.json only; no model.
Analysis: reports/analyses/2026-10-29_analysis_e55-stage-a.md, section "Amendment 4".

  amend4_offlabel.png  Stage A's two checks, behaviour share B (x) and patch share P (y), per count: left count a,
                       right count b. Each state's arrow runs from its value on Stage A's natural rows (hollow) to its
                       value on off-label swap-built rows (filled), where count(a) + 1 and count(b) are different
                       answers. Shaded: Stage A's filed "through" region (both above the upper bar) and "not through"
                       region (both below the lower bar). N1 1K is the positive control for count b; LOW, the base
                       before fine-tuning, is the do-nothing reference, off-label only

Usage (repo root): ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/pcfg_anchors/plot_amend4.py
"""

from __future__ import annotations

from matplotlib.axes import Axes
from plot import FIG, INK, M_C, N_C, W_C, finish, load, plt

STATES = (  # key in amend4.json, a2 file and iteration, words, colour, marker
    ("N1_1000", "N1", "1000", "fast, no spurious rows, 1K (count-b control)", N_C, "o"),
    ("W1_1000", "W1", "1000", "slow, all spurious, 1K", W_C, "o"),
    ("W1s1_1000", "W1s1", "1000", "same, seed 1", W_C, "v"),
    ("M3_1000", "M3", "1000", "slow, 80% spurious, 1K", M_C, "^"),
    ("W1_10000", "W1", "10000", "slow, all spurious, 10K", W_C, "D"),
)


def region(ax: Axes, bar_b: dict[str, float], bar_p: dict[str, float]) -> None:
    ax.fill_between([bar_b["hi"], 1.05], bar_p["hi"], 1.05, color="#e8f2e0", zorder=0)
    ax.fill_between([-0.05, bar_b["lo"]], -0.05, bar_p["lo"], color="#efefed", zorder=0)
    ax.text(1.03, 1.03, "through", ha="right", va="top", fontsize=8, color=INK)
    ax.text(bar_b["lo"] + 0.01, -0.03, "not through", ha="left", va="bottom", fontsize=8, color=INK)


def fig_offlabel(out: str = "amend4_offlabel.png",
                 title: str = "On off-label swap-built rows the early slow all-spurious fine-tunes still run through count a, barely;\nnone of the early slow fine-tunes runs through count b") -> None:  # fmt: skip
    r, bars = load("amend4.json")["part2_off_label"], load("predictions.json")["a2_bars"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6), sharex=True, sharey=True)
    for ax, t in zip(axes, ("a", "b"), strict=True):
        region(ax, bars[f"B_{t}"], bars[f"P_{t}"])
        for key, arm, it, words, col, mk in STATES:
            nat = load(f"a2_{arm}.json")["states"][it][f"check_count_{t}"]
            off = r[key][t]
            ax.annotate("", xy=(off["B"], off["P"]), xytext=(nat["B"], nat["P"]),
                        arrowprops={"arrowstyle": "->", "color": col, "lw": 1.1, "alpha": 0.7})  # fmt: skip
            ax.scatter(nat["B"], nat["P"], marker=mk, s=50, facecolor="white", edgecolor=col, lw=1.4, zorder=3)
            ax.scatter(off["B"], off["P"], marker=mk, s=60, color=col, zorder=4, label=words if t == "a" else None)
        low = r["LOW"][t]
        ax.scatter(low["B"], low["P"], marker="X", s=70, color=INK, zorder=4,
                   label="LOW before fine-tuning (do-nothing), off-label only" if t == "a" else None)  # fmt: skip
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        q = "count(a) + 1" if t == "a" else "count(b)"
        ax.set_xlabel(f"behaviour: share answering {q}")
        ax.set_title(f"count {t}: swap in a donor with another count {t}", fontsize=9.5, loc="left")
    axes[0].set_ylabel("patch: share whose answer moves to the donor's")
    axes[0].scatter([], [], marker="o", s=50, facecolor="white", edgecolor=INK, label="hollow: Stage A's natural rows")
    axes[0].scatter([], [], marker="o", s=50, color=INK, label="filled: off-label swap-built rows")
    fig.legend(fontsize=7.5, frameon=False, loc="lower center", ncol=4)
    finish(fig, out, title, bottom=0.22)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig_offlabel()


if __name__ == "__main__":
    main()
