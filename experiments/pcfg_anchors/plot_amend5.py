"""Figure for E55 Amendment 5, read from results/amend5.json and Stage A's a2_<arm>.json only; no model.
Analysis: reports/analyses/2026-10-30_analysis_e55-amend5.md.

  amend5_saves.png  Stage A's two count-a checks by save, left the behaviour share B_a, right the patch share P_a,
                    for every Amendment 5 state. The first point of each line is LOW before fine-tuning, read the same
                    way (the do-nothing reference). Stage A's slow all-spurious states (both seeds) and slow 80% state
                    at 1K, the two "reuses" anchors before this amendment, are hollow markers at 1K. Stage A's filed
                    upper and lower bars are dashed; above the upper bar is shaded ("through" needs both panels above).

Usage (repo root): ~/commute-agents/fleet/cpu-run uv run --no-sync python experiments/pcfg_anchors/plot_amend5.py
"""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from plot import FIG, INK, M_C, W_C, finish, load, plt

WN_C, WNF_C, ID_C = "#138d90", "#c99a06", "#8c564b"
SAVES = ("500", "1000", "2000")
X = {"LOW": 0, "500": 1, "1000": 2, "2000": 3}
ARMS5 = (  # key in amend5.json, legend words (None: seed copy, no new entry), colour, line style
    ("WN_s0", "natural rows, slow (lr 1e-5), seeds 0-3", WN_C, "-"),
    ("WN_s1", None, WN_C, "-"),
    ("WN_s2", None, WN_C, "-"),
    ("WN_s3", None, WN_C, "-"),
    ("WNf_s0", "natural rows, fast (lr 1e-4), seed 0", WNF_C, "-"),
    ("ID_s0", "old count under the T prompt (positive control), seeds 0, 1", ID_C, "--"),
    ("ID_s1", None, ID_C, "--"),
)
STAGE_A = (  # a2 file, words, colour, marker
    ("W1", "Stage A slow all-spurious, 1K (swap-built rows)", W_C, "o"),
    ("W1s1", "same, seed 1", W_C, "v"),
    ("M3", "Stage A slow 80% spurious, 1K", M_C, "^"),
)


def series(r: dict[str, Any], key: str, t: str) -> list[float]:
    """LOW's reading (the arm's step 0) then each save; ID arms read at the ID prompt."""
    low = r["W0"]["reference"]["id_prompt" if key.startswith("ID") else "check_count_a"]
    low = low["check_count_a"] if key.startswith("ID") else low
    return [low[t]] + [r["arms"][key]["states"][s]["check_count_a"][t] for s in SAVES]


def panel(ax: Axes, r: dict[str, Any], bar: dict[str, float], t: str) -> None:
    ax.axhspan(bar["hi"], 1.05, color="#e8f2e0", zorder=0)
    for y, words in ((bar["hi"], "upper bar"), (bar["lo"], "lower bar")):
        ax.axhline(y, color=INK, ls=":", lw=0.9)
        ax.text(3.3, y + 0.01, words, ha="right", va="bottom", fontsize=7.5, color=INK)
    for key, words, col, ls in ARMS5:
        ax.plot(list(X.values()), series(r, key, t), ls=ls, color=col, marker="o", ms=4, lw=1.2, alpha=0.85,
                label=words if t == "B" else None)  # fmt: skip
    for i, (arm, words, col, mk) in enumerate(STAGE_A):
        v = load(f"a2_{arm}.json")["states"]["1000"]["check_count_a"][t]
        ax.scatter(X["1000"] + 0.12 * (i + 1), v, marker=mk, s=55, facecolor="white", edgecolor=col, lw=1.5, zorder=4,
                   label=words if t == "B" else None)  # fmt: skip
    ax.set_xticks(list(X.values()), ["LOW\n(no fine-tune)", "500", "1K", "2K"])
    ax.set_xlim(-0.2, 3.4)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("fine-tuning steps")


def fig_saves(out: str = "amend5_saves.png",
              title: str = "Every slow natural-row fine-tune and the positive control run through count a at every save;\nthe fast one reads neither by 2K") -> None:  # fmt: skip
    r, bars = load("amend5.json"), load("predictions.json")["a2_bars"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4), sharex=True, sharey=True)
    panel(axes[0], r, bars["B_a"], "B")
    panel(axes[1], r, bars["P_a"], "P")
    axes[0].set_ylabel("behaviour: share answering through count a\n(count(a) + 1 on 'C b 40'; count(a) for the control)")
    axes[1].set_ylabel("patch: share whose answer moves to the donor's\nwhen count a is swapped in at block 5")
    axes[1].tick_params(labelleft=True)
    axes[0].set_title("behaviour check", fontsize=9.5, loc="left")
    axes[1].set_title("patch check", fontsize=9.5, loc="left")
    fig.legend(fontsize=7.5, frameon=False, loc="lower center", ncol=3)
    finish(fig, out, title, bottom=0.24)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig_saves()


if __name__ == "__main__":
    main()
