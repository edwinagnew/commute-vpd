"""E100 Amendment 2's one figure, from results/amend2.json; no model.

  figures/amend2_emoticon.png  left: the share of the gain kept with E62's copying heads (2,3), (2,4), (3,4) at their
                               natural-text means, emoticon seeds 0-1 against the in-context copy seed 0 (the control);
                               right: the same states' sum of single reverts against Amendment 1's fixed bars, with
                               each emoticon state's profile-matched written plant on its own site

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/fed_or_written/plot_amend2.py
"""

from __future__ import annotations

from typing import Any

from matplotlib.lines import Line2D

from commute.jobs import load_experiment

P: Any = load_experiment("fed_or_written", "plot")
plt = P.plt
AM2 = P.load("amend2.json")
CP = AM2["stage2"]["copying"]
ROWS = {(r["task"], int(r["seed"])): r for r in AM2["stage1"]["rows"].values()}
STATES = [("in-context copy, seed 0:\nthe control", CP["control"], ROWS["T5 copy", 0], P.BLUE),
          ("emoticon trigger, seed 0", CP["T1 emoticon s0"], ROWS["T1 emoticon", 0], P.GREY),
          ("emoticon trigger, seed 1", CP["T1 emoticon s1"], ROWS["T1 emoticon", 1], P.GREY)]  # fmt: skip
KEPT_MAX = float(P.load("predictions_amend2.json")["thresholds"]["kept_max"])


def kept_panel(ax: Any) -> None:
    ax.axvspan(0, KEPT_MAX, color=P.BAND, zorder=0)
    ax.axvline(KEPT_MAX, color=P.BLUE, ls="--", lw=1)
    ax.axvline(1.0, color=P.DRAW, ls=":", lw=1)
    ax.text(KEPT_MAX - 0.02, -0.75, f"copying: keeps at most {KEPT_MAX:.1f}", color=P.BLUE, fontsize=8, ha="right")
    ax.text(1.0 - 0.02, -0.75, "1: heads off\nchange nothing", color=P.MUTED, fontsize=8, ha="right")
    for i, (_, c, _, col) in enumerate(STATES):
        ax.scatter(float(c["kept"]), i, color=col, s=60, zorder=3)
        ax.text(float(c["kept"]) + 0.03, i, f"{float(c['kept']):.2f}", va="center", fontsize=8, color=P.INK)
    ax.set_xlim(0, 1.2)
    P.axes(ax, "share of the gain kept with the three copying heads at their natural-text means")
    P.title(ax, "Does the gain need the copying heads?")


def sum_panel(ax: Any) -> None:
    ax.axvspan(0, P.FED_BAR, color=P.BAND, zorder=0)
    ax.axvspan(P.WRITTEN_BAR, 1.05, color=P.BAND, zorder=0)
    ax.axvline(P.FED_BAR, color=P.BLUE, ls="--", lw=1)
    ax.axvline(P.WRITTEN_BAR, color=P.ORANGE, ls="--", lw=1)
    ax.text(P.FED_BAR - 0.02, -0.75, f"fed bar {P.FED_BAR:.2f}", color=P.BLUE, fontsize=8, ha="right")
    ax.text(P.WRITTEN_BAR + 0.02, -0.75, f"written bar {P.WRITTEN_BAR:.2f}", color=P.ORANGE, fontsize=8)
    for i, (_, _, r, col) in enumerate(STATES):
        ax.scatter(float(r["sum"]), i, color=col, s=60, zorder=3)
        ax.text(float(r["sum"]) + 0.03, i, f"{float(r['sum']):.2f}", va="center", fontsize=8, color=P.INK)
        if r["status"] == "ok":
            ax.scatter(float(r["W"]), i, marker="*", s=200, color=P.GREEN, edgecolor=P.INK, lw=0.5, zorder=4)
    ax.set_xlim(0, 1.05)
    P.axes(ax, "sum of single-part reverts, all heads on")
    P.title(ax, "Yet does it read fed?")


def main() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    kept_panel(axs[0])
    sum_panel(axs[1])
    axs[0].set_yticks(range(len(STATES)), [s[0] for s in STATES])
    axs[0].set_ylim(len(STATES) - 0.5, -1.1)
    star = Line2D([], [], marker="*", ls="", ms=12, color=P.GREEN, mec=P.INK,
                  label="written by construction, matched to the state's per-part norms and gain, on its site")  # fmt: skip
    fig.legend(handles=[star], frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("The emoticon trigger keeps its gain without the copying heads, yet sums in the fed range",
                 color=P.INK, x=0.01, ha="left", fontsize=11)  # fmt: skip
    P.save(fig, "amend2_emoticon.png")


if __name__ == "__main__":
    main()
