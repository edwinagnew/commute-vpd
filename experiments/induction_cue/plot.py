"""E62's figures, one per claim, from results/run.json and E58's results (no model).

  figures/share.png   share of the trained gain lost per seed, both maps, under the copying head and under five
                      single other heads, with the filed bars
  figures/cells.png   the route cells of both maps on one log-odds axis: where each map starts and what the original
                      subcomponents reach under the fine-tuned state's mask
  figures/r_prior.png r of both maps against W0's prior, beside E58's eight maps at step 30; unscored here

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/induction_cue/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
COPY, LOOKUP, E58, HEAD11 = "#2a78d6", "#eb6834", "#8a8983", "#7a3fb0"
INK, GRID = "#52514e", "#e4e3df"
SEEDS = (0, 1, 2)
E58_LABEL = {"class_map": "class map", "class_shuffled": "class shuffled", "cipher": "cipher", "shuffle": "shuffle", "prior_cipher": "prior cipher",
             "copy": "E58 copy (the key)", "elicit": "elicit", "rank1": "rank 1"}


def style(ax: Any) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


def share_figure(run: dict[str, Any]) -> None:
    """Share per state: H with its row-bootstrap sd, (1,1) marked, the other four draws small."""
    st = run["states"]
    names = [f"copy_s{s}" for s in SEEDS] + [f"lookup_s{s}" for s in SEEDS]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for i, n in enumerate(names):
        s = st[n]
        col = COPY if n.startswith("copy") else LOOKUP
        ax.scatter([i + 0.14] * 4, s["random_shares"][1:], s=22, color=E58, zorder=2, label="four other single heads" if i == 0 else None)
        ax.scatter([i + 0.14], s["random_shares"][0], s=46, marker="D", color=HEAD11, zorder=3, label="head (1,1), the one previous-token draw" if i == 0 else None)
        ax.errorbar([i - 0.1], [s["share"]], yerr=[s["share_sd"]], fmt="o", ms=9, color=col, capsize=3, zorder=4,
                    label={0: "copy map, copying head (2,4) off", 3: "lookup, copying head (2,4) off"}.get(i))
    ax.axhline(0.5, color=COPY, ls="--", lw=1.2)
    ax.text(5.45, 0.51, "copy map's bar 0.5", ha="right", color=INK, fontsize=9)
    ax.axhline(0.2, color=LOOKUP, ls="--", lw=1.2)
    ax.text(5.45, 0.21, "lookup's bar 0.2", ha="right", color=INK, fontsize=9)
    ax.axhline(0.15, color=E58, ls=":", lw=1.2)
    ax.text(5.45, 0.115, "bar on the median other head, 0.15", ha="right", color=INK, fontsize=9)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(6), [f"seed {s}" for s in SEEDS] * 2)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.12, 0.72)
    ax.set_ylabel("share of the log-odds gain lost")
    ax.set_title("How much of each fine-tune's gain goes when one head is switched off?", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "share.png", dpi=150)
    plt.close(fig)


def cells_figure(run: dict[str, Any]) -> None:
    """Two panels, one log-odds axis: the original model's cells, then the cells under the fine-tuned state's mask."""
    order = ["T0", "U0", "L00", "L10", "L01", "L11", "U1", "T1"]
    words = ["T0\noriginal\ntarget", "U0\noriginal\nall on", "L00\noriginal\nold mask", "L10\nsplit\nold mask", "L01\noriginal\nnew mask", "L11\nsplit\nnew mask", "U1\nsplit\nall on", "T1\nfine-tuned\ntarget"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True)
    for ax, m, col, title in ((axes[0], "copy", COPY, "Copy map: target is B, in context"), (axes[1], "lookup", LOOKUP, "Lookup: target is the key's word, not in its windows")):
        for s in SEEDS:
            c = run["states"][f"{m}_s{s}"]["cells"]
            ax.plot(range(8), [c[k] for k in order], "o-", color=col, ms=5, lw=1.2, alpha=0.75, label="three seeds" if s == 0 else None)
        c = run["states"][f"{m}_s0"]["cells"]
        ax.axhline(c["L00"], color=INK, ls=":", lw=1)
        ax.axvspan(3.5, 4.5, color=GRID, alpha=0.5, zorder=0)
        ax.set_xticks(range(8), words, fontsize=7.5)
        ax.set_title(title, color=INK, fontsize=10.5)
        ax.text(7.4, c["L00"] + 0.35, "original masked", ha="right", color=INK, fontsize=8.5)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        style(ax)
    axes[0].set_ylabel("mean log-odds of the target at the marker")
    fig.suptitle("What do the original subcomponents reach under the fine-tuned state's mask?\nr = shaded point minus dotted line, over T1 minus T0 (unscored here)", color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "cells.png", dpi=150)
    plt.close(fig)


def r_figure(run: dict[str, Any], e58: dict[str, Any]) -> None:
    """r against W0's prior: E58's eight maps grey, E62 in colour. Square = target in context in the window."""
    fig, ax = plt.subplots(figsize=(8.6, 5))
    offset = {"prior_cipher": (6, -12), "copy": (-6, 8), "class_map": (6, -12), "class_shuffled": (-2, 9), "cipher": (6, 4), "shuffle": (6, -12), "elicit": (6, -12), "rank1": (-6, 8)}
    for m, v in e58.items():
        marker = "s" if m == "copy" else "o"
        ax.scatter([v["w0_geo_p_trained"]] * 3, v["r30"], s=30, color=E58, marker=marker, zorder=3, label=None)
        ha = "right" if offset[m][0] < 0 else "left"
        ax.annotate(E58_LABEL[m], (v["w0_geo_p_trained"], sum(v["r30"]) / 3), xytext=offset[m], textcoords="offset points", fontsize=8.5, color=INK, ha=ha)
    b = run["stage_b"]
    ax.scatter([b["prior_copy"]] * 3, [run["states"][f"copy_s{s}"]["r"] for s in SEEDS], s=70, marker="s", color=COPY, zorder=4, label="E62 copy map, target in context")
    ax.scatter([b["prior_lookup"]] * 3, [run["states"][f"lookup_s{s}"]["r"] for s in SEEDS], s=70, color=LOOKUP, zorder=4, label="E62 lookup, target not in its windows")
    ax.scatter([], [], s=30, color=E58, label="E58's maps, step 30, three seeds (square: target in context)")
    ax.axhline(0.5, color=INK, ls="--", lw=1)
    ax.text(1.8e-7, 0.515, "the cut at 0.5", color=INK, fontsize=8.5)
    ax.axhline(1, color=GRID, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(1.5e-7, 5e-2)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("original model's geometric-mean probability of the target at the read position")
    ax.set_ylabel("r, step 30 (E62's is provisional, not scored)")
    ax.set_title("At priors a factor 1.3 apart, do the two maps read the same r?", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 0.92))
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "r_prior.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    run = json.loads((HERE / "results" / "run.json").read_text())
    e58 = json.loads((HERE.parent / "open_pair" / "results" / "amend7.json").read_text())["prior_beside_r"]
    share_figure(run)
    cells_figure(run)
    r_figure(run, e58)


if __name__ == "__main__":
    main()
