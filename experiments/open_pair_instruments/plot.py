"""E55 C3's figures, one per claim, from results/run.json (no model).

  figures/c3_poles.png      pruning revival R and reverse half-life per state, by map, with each column's filed bar
                            (3 x the largest seed sd) beside the rank-1-to-cipher gap and the planted control's reading
  figures/c3_positions.png  for every column that reads the gap: elicit's position p on the rank-1-to-cipher axis with
                            its bootstrap interval, copy's position, the thirds bands and the 0.1 tolerance around the
                            two decomposition-free baselines (distance D and W0's prior)
  figures/c3_distance.png   per state, the reverse half-life and the unnormalised revival against D, the planted control
                            on the same axes

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 4G .venv/bin/python experiments/open_pair_instruments/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
COLOR = {"elicit": "#1f9e6e", "cipher": "#eb6834", "rank1": "#7b5ab8", "copy": "#2a78d6"}
NAME = {"rank1": "rank-1\n(old end)", "cipher": "cipher\n(new end)", "elicit": "elicit\n(placed)", "copy": "copy\n(reported)"}
MAPS = ("rank1", "cipher", "elicit", "copy")
INK, GRID, SHADE = "#52514e", "#e4e3df", "#f6efe2"


def style(ax: Any, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.set_ylabel(ylabel, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def strip(ax: Any, col: dict[str, list[float]], rd: dict[str, Any]) -> None:
    """Each map's eight seeds, its mean, and the filed bar: 3 x noise drawn up from the cipher's mean."""
    rng = np.random.default_rng(0)
    for i, m in enumerate(MAPS):
        v = col[m]
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=22, color=COLOR[m], alpha=0.8, zorder=3)
        ax.hlines(np.mean(v), i - 0.25, i + 0.25, color=COLOR[m], lw=2.2, zorder=4)
    lo, gap, bar = rd["means"]["cipher"], rd["gap"], 3 * rd["noise"]
    ax.annotate("", (0.42, lo + gap), (0.42, lo), arrowprops={"arrowstyle": "<->", "color": COLOR["rank1"], "lw": 1.3})
    ax.annotate(f"rank-1 minus\ncipher {gap:+.2f}", (0.42, lo), xytext=(0, -4), textcoords="offset points", fontsize=7,
                color=COLOR["rank1"], va="top", ha="center")
    ax.annotate("", (0.58, lo + bar), (0.58, lo), arrowprops={"arrowstyle": "<->", "color": INK, "lw": 1.3})
    ax.text(0.62, lo + bar / 2, f"filed bar:\n3 x seed sd\n{bar:.2f}", fontsize=7, color=INK, va="center")
    ax.set_xticks(range(4), [NAME[m] for m in MAPS], fontsize=8)
    ax.set_xlim(-0.6, 3.5)


def poles_fig(run: dict[str, Any]) -> None:
    cols, rd, s0 = run["columns"], run["score"]["readings"], run["stage0"]["planted"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    strip(axes[0], cols["R"], rd["R"])
    axes[0].axhline(s0["R"], color=INK, ls="--", lw=0.9)
    axes[0].text(3.45, s0["R"] - 0.02, f"planted 13-unit control: {s0['R']:.2f}", fontsize=7, color=INK, ha="right", va="top")
    axes[0].axhline(0, color=INK, lw=0.7)
    axes[0].text(3.45, 0.01, "random 40-unit removal: under 0.01", fontsize=7, color=INK, ha="right", va="bottom")
    axes[0].set_ylim(-0.05, 1.05)
    style(axes[0], "Pruning: share of the state's KL from W0 removed by the top units\n(old end expected higher)",
          "revival R: best of K = 13, 40 units,\nminus random removal, over D")
    strip(axes[1], cols["t_half"], rd["t_half"])
    axes[1].axhline(s0["reverse"]["t_half"], color=INK, ls="--", lw=0.9)
    axes[1].text(3.45, s0["reverse"]["t_half"] + 0.3, f"planted control: {s0['reverse']['t_half']:.1f} steps", fontsize=7,
                 color=INK, ha="right", va="bottom")
    axes[1].set_ylim(0, 20)
    style(axes[1], "Reverse fine-tuning toward W0: steps to halve the KL\n(old end expected lower)",
          "half-life, optimizer steps at lr 3e-6")
    fig.suptitle("Do pruning revival and reverse fine-tuning tell the rank-1 map from the cipher? 8 seeds per map",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "c3_poles.png", dpi=150)
    plt.close(fig)


POSITION_ROWS = (("r", "VPD's re-gating share r"), ("ratio", "VPD's stand-in ratio"), ("x", "carry x (behaviour)"),
                 ("mlp_share", "MLP share of the update"), ("q", "logit-lens probe q (reference)"),
                 ("prior", "W0's prior on the value"), ("D", "distance D = KL from W0"),
                 ("R", "pruning revival R"), ("t_half", "reverse half-life"))


def positions_fig(run: dict[str, Any]) -> None:
    rd = run["score"]["readings"]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.axvspan(-0.5, 1 / 3, color="#f2ddd0", zorder=0)
    ax.axvspan(2 / 3, 2.0, color="#dcebe3", zorder=0)
    for x, lab in ((-0.45, "new-end third"), (1.02, "old-end third")):
        ax.text(x, len(POSITION_ROWS) - 0.35, lab, fontsize=7.5, color=INK)
    for k, c in (("D", "#9a9893"), ("prior", "#c9a227")):
        p = rd[k]["p"]
        ax.axvspan(p - 0.1, p + 0.1, color=c, alpha=0.25, zorder=1)
    for i, (k, name) in enumerate(reversed(POSITION_ROWS)):
        r = rd[k]
        if r["status"] != "ok":
            ax.text(0.5, i, f"no gap: rank-1 minus cipher {abs(r['gap']):.3g} under the bar {3 * r['noise']:.3g}; no position",
                    fontsize=7.5, color=INK, va="center", ha="center", style="italic")
        else:
            ax.plot([r["p_lo"], r["p_hi"]], [i, i], color=COLOR["elicit"], lw=3, alpha=0.6, zorder=3)
            ax.scatter([r["p"]], [i], s=46, color=COLOR["elicit"], zorder=4, label="elicit, seed-bootstrap 90%" if i == 4 else None)
            pc = r["p_copy"]
            ax.scatter([min(pc, 1.95)], [i], s=40, marker="D", facecolor="white", edgecolor=COLOR["copy"], lw=1.3, zorder=4,
                       label="copy (reported)" if i == 4 else None)
            if pc > 1.95:
                ax.text(1.93, i + 0.22, f"copy {pc:.2f} →", fontsize=6.5, color=COLOR["copy"], ha="right")
    ax.axvline(0, color=COLOR["cipher"], lw=1.2)
    ax.axvline(1, color=COLOR["rank1"], lw=1.2)
    ax.text(0, -0.9, "cipher", color=COLOR["cipher"], fontsize=8, ha="center")
    ax.text(1, -0.9, "rank-1", color=COLOR["rank1"], fontsize=8, ha="center")
    ax.text(rd["D"]["p"], -0.9, "±0.1 of D", color="#6b6a66", fontsize=7, ha="center")
    ax.text(rd["prior"]["p"] - 0.12, -0.9, "±0.1 of prior", color="#9a7a10", fontsize=7, ha="center")
    ax.set_yticks(range(len(POSITION_ROWS)), [n for _, n in reversed(POSITION_ROWS)], fontsize=8)
    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(-1.2, len(POSITION_ROWS) - 0.1)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("position p = (elicit − cipher) / (rank-1 − cipher), seed means; 1 = with rank-1, 0 = with cipher",
                  color=INK)
    ax.set_title("Where does each reader that sees the rank-1-to-cipher gap put the elicit map?", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "c3_positions.png", dpi=150)
    plt.close(fig)


def distance_fig(run: dict[str, Any]) -> None:
    st, s0 = run["states"], run["stage0"]["planted"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for m in MAPS:
        names = [n for n in st if n.startswith(f"{m}_s")]
        D = [st[n]["D"] for n in names]
        axes[0].scatter(D, [st[n]["t_half"] for n in names], s=24, color=COLOR[m], label=NAME[m].replace("\n", " "), zorder=3)
        axes[1].scatter(D, [max(st[n]["Delta"].values()) for n in names], s=24, color=COLOR[m], zorder=3)
    D_all = np.array([st[n]["D"] for n in st])
    t_all = np.array([st[n]["t_half"] for n in st])
    a, b = np.polyfit(D_all, t_all, 1)
    xs = np.array([0, 11.5])
    axes[0].plot(xs, a * xs + b, color=INK, lw=0.9, ls=":", label=f"least-squares line, 32 states: {a:.2f} steps per nat")
    axes[0].scatter([s0["D"]], [s0["reverse"]["t_half"]], marker="*", s=140, color="black", zorder=4,
                    label="planted 13-unit control on W0")
    axes[0].legend(fontsize=7, loc="upper left", frameon=False)
    style(axes[0], "Across maps the half-life follows D, except copy", "half-life, optimizer steps")
    for share, ls in ((1.0, "-"), (0.5, "--")):
        axes[1].plot(xs, share * xs, color=INK, lw=0.8, ls=ls)
        axes[1].text(11.4, share * 11.4, f"revives {share:.0%} of D", fontsize=7, color=INK, ha="right", va="bottom")
    axes[1].scatter([s0["D"]], [max(s0["Delta"].values())], marker="*", s=140, color="black", zorder=4)
    style(axes[1], "Pruning gives back 39-65% of D per map; copy the most", "revival in nats: best K, minus random removal")
    for ax in axes:
        ax.set_xlabel("D = KL(W0 ‖ state) at the answer position, nats", color=INK)
        ax.set_xlim(0, 11.5)
        ax.set_ylim(bottom=0)
    fig.suptitle("How do the two instruments relate to how far each state moved W0's answer distribution?",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "c3_distance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    run = json.loads((HERE / "results" / "run.json").read_text())
    poles_fig(run)
    positions_fig(run)
    distance_fig(run)


if __name__ == "__main__":
    main()
