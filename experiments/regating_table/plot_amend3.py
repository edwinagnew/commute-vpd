"""Figure for E46 Amendment 3 (split nulls), read from results/amend3.json and results/predictions_amend3.json. Plan:
reports/plans/2026-10-28_plan_regating-line-reset.md, section "Amendment 3"; analysis:
reports/analyses/2026-10-29_analysis_e46-amendment3.md.

  amend3.png   the stand-in ratio (gated piece gain over piece gain, after the least-squares split) per table: the
               E12 fine-tunes (seeds 0, 1), the G-matched per-key hand write, with 90% bootstrap intervals, against
               the filed bars (0.5, 0.7); rows whose ratio is undefined (piece gain under 1 nat: the random delta-W,
               seeds 0 and 1, and the piece-matched writes) sit in a strip under the axis

Usage (repo root): ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/regating_table/plot_amend3.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R: dict[str, Any] = json.loads((HERE / "results" / "amend3.json").read_text())["rows"]
T: dict[str, Any] = json.loads((HERE / "results" / "predictions_amend3.json").read_text())["thresholds"]

INK, MUTED, BAND = "#0b0b0b", "#8a8984", "#ecebe7"
FT_C, RAND_C, WRITE_C = "#2a78d6", "#555555", "#2e8b57"  # same colours as plot.py
PASS_C, FAIL_C = "#2e8b57", "#c0392b"
TABLES = {"ident32": "copy,\ntable A", "ident32b": "copy,\ntable B", "rand32": "random map,\ntable A",
          "rand32b": "random map,\ntable B"}
UNDEF_Y = -0.12  # strip for undefined ratios


def amend3() -> None:
    lo, hi = T["bar_low"], T["bar_high"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhspan(lo, hi, color=BAND, zorder=0, label="between the bars: decision card")
    ax.axhline(lo, color=PASS_C, lw=1.2, ls="--", label=f"low bar {lo}: a random-map write here or under is unlike copy")
    ax.axhline(hi, color=FAIL_C, lw=1.2, ls="--", label=f"high bar {hi}: a random-map write here or over reads like copy")
    ax.axhline(UNDEF_Y + 0.07, color=MUTED, lw=0.6)
    series = [("s0", "fine-tune, seed 0", FT_C, "o", True, -0.3), ("s1", "fine-tune, seed 1", FT_C, "o", False, -0.12),
              ("write_G", "per-key hand write, scaled to the fine-tune's real gain", WRITE_C, "^", True, 0.06)]
    for i, t in enumerate(TABLES):
        for key, name, c, m, fill, dx in series:
            row = R[f"{t}/{key}"]
            ci = row["boot"]["ratio_ci"]
            v = row["ratio"]
            ax.errorbar(i + dx, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor=c, lw=1.2, zorder=2)
            ax.scatter(i + dx, v, color=c if fill else "white", edgecolor=c, marker=m, s=70, zorder=3,
                       label=name if i == 0 else None)
            ax.annotate(f"{v:.2f}", (i + dx, v), xytext=(-6 if key == "s0" else 6, 0), textcoords="offset points",
                        fontsize=8, color=c, va="center", ha="right" if key == "s0" else "left")
        for j, key in enumerate(["random_s0", "random_s1", "write_piece"]):
            assert not R[f"{t}/{key}"]["defined"]
            c, m = (RAND_C, "x") if key.startswith("random") else (WRITE_C, "x")
            ax.scatter(i + 0.2 + 0.07 * j, UNDEF_Y, color=c, marker=m, s=45, zorder=3)
    ax.scatter([], [], color=RAND_C, marker="x", label="random δW, seeds 0 and 1: undefined, subcomponent gain under 1 nat")
    ax.scatter([], [], color=WRITE_C, marker="x", label="hand write scaled to the fine-tune's subcomponent gain: undefined")
    ax.text(-0.45, UNDEF_Y, "undefined", fontsize=8, color=MUTED, va="center")
    ax.axvline(1.5, color=MUTED, lw=0.6, ls=":")
    ax.set_xticks(range(len(TABLES)), list(TABLES.values()))
    ax.set_xlim(-0.5, len(TABLES) - 0.5)
    ax.set_ylim(UNDEF_Y - 0.06, 1.35)
    ax.set_ylabel("stand-in ratio: masked gain / subcomponent gain\n(after the least-squares split; 90% bootstrap interval)")
    ax.set_title("Does a hand write with no learned machinery read like the fine-tune of its own table?", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "amend3.png", dpi=130)


if __name__ == "__main__":
    amend3()
