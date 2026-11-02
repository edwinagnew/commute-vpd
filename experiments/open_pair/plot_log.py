"""E58's log figure, from results/amend7.json and the step-30 carry rule of plot_amend7.py (no model).

  figures/log_prior.png  one row of three panels sharing the rows: the eight key-to-value maps, ordered by the
                         original model's prior of the trained answer (highest at the top). Left, that prior (log
                         scale); middle, r at step 30 (three seeds and their mean, 0.5 cut); right, the held-out carry
                         at step 30 (three seeds and their mean; net of the cross-class null for the class rule and its
                         twin). Two colours: maps whose answer the model already gives, and maps whose answer is new.

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/open_pair/plot_log.py
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot import COLOR, FIG, GRID, INK, style
from plot_amend7 import carry30, mean, read

NAME = {
    "rank1": "rank-1: the model's top word",
    "elicit": "half-known map",
    "copy": "copy: value = key",
    "prior_cipher": "prior-matched cipher",
    "shuffle": "shuffled half-known map",
    "cipher": "cipher",
    "class_shuffled": "its shuffled twin",
    "class_map": "class rule: new 4-word rule",
}
KNOWN = ("rank1", "elicit", "copy")
GROUP_COLOR = {True: COLOR["elicit"], False: COLOR["cipher"]}


def seeds_and_mean(ax: Any, y: float, xs: list[float], color: str) -> None:
    ax.scatter(xs, [y] * len(xs), s=16, color=color, alpha=0.45, lw=0, zorder=3)
    ax.scatter([mean(xs)], [y], s=60, color=color, edgecolors="white", lw=0.8, zorder=4)


def main() -> None:
    a7 = read("amend7")
    pb, cx = a7["prior_beside_r"], carry30(a7)
    order = sorted(pb, key=lambda m: -pb[m]["w0_geo_p_trained"])
    ys = {m: len(order) - 1 - i for i, m in enumerate(order)}
    split = min(ys[m] for m in KNOWN) - 0.5
    fig, (axp, axr, axc) = plt.subplots(
        1, 3, figsize=(11.0, 5.2), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1]}
    )
    for m in order:
        color = GROUP_COLOR[m in KNOWN]
        axp.scatter([pb[m]["w0_geo_p_trained"]], [ys[m]], s=60, color=color, zorder=4)
        seeds_and_mean(axr, ys[m], pb[m]["r30"], color)
        seeds_and_mean(axc, ys[m], cx[m], color)
    for ax in (axp, axr, axc):
        ax.axhline(split, color=GRID, lw=1.4, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK, labelsize=8)
    axc.text(0.93, split + 0.08, "answer already the model's guess", fontsize=7,
             color=GROUP_COLOR[True], ha="right", va="bottom")
    axc.text(0.93, split - 0.08, "answer new", fontsize=7,
             color=GROUP_COLOR[False], ha="right", va="top")
    axp.set_xscale("log")
    axp.set_xlim(1e-7, 1e-1)
    axp.set_yticks([ys[m] for m in order], [NAME[m] for m in order], fontsize=8.5, color=INK)
    axp.set_ylim(-0.6, len(order) - 0.4)
    axp.set_xlabel("probability (geometric mean over keys, log scale)", color=INK, fontsize=8)
    style(axp, "the model's prior\nof the trained answer", "")
    axp.grid(axis="y", visible=False)
    axr.axvline(0.5, color=INK, lw=0.9, ls=":", zorder=2)
    axr.text(0.5, len(order) - 0.45, "0.5 cut", fontsize=7, color=INK, ha="center", va="bottom")
    axr.set_xlim(0.1, 1.0)
    axr.set_xlabel("r (small dots: seeds; large dot: mean)", color=INK, fontsize=8)
    style(axr, "r at step 30 (share of the gain\nrecovered by Γ's switching alone)", "")
    axr.grid(axis="y", visible=False)
    axc.axvline(0.0, color=INK, lw=0.6, zorder=2)
    axc.set_xlim(-0.45, 0.95)
    axc.set_xlabel("carry (small dots: seeds; large dot: mean)\n"
                   "class rule and twin: carry net of the cross-class null", color=INK, fontsize=8)
    style(axc, "held-out carry at step 30\n(0 = unseen keys gained nothing)", "")
    axc.grid(axis="y", visible=False)
    fig.suptitle(
        "r is high for exactly the maps whose answer the model already guessed; "
        "carry does not sort the maps the same way",
        fontsize=10.5, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout()
    fig.savefig(FIG / "log_prior.png", dpi=150)
    plt.close(fig)
    for m in order:
        rs, cs = pb[m]["r30"], cx[m]
        print(f"{NAME[m]:30s} prior {pb[m]['w0_geo_p_trained']:.2e}  r {[round(r, 3) for r in rs]} "
              f"mean {mean(rs):.3f}  carry {[round(c, 3) for c in cs]} mean {mean(cs):.3f}")


if __name__ == "__main__":
    main()
