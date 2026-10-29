"""E51's one figure: each weight statistic per state, kinds coloured, the matched Gaussian null as a band, the controls as
labelled lines, and E46's re-gating share per state alongside (no model, no runs).

  figures/erank_vs_size.png  effective rank of delta-W against its size, every state, with the null band
  figures/weights.png   four panels on one row of states: top-10 energy share, intruder count, effective rank of
                        delta-W, and the share of the gain from Gamma's re-gating (regating_table stage0a/stage0b)

Usage (repo root):  ~/commute-agents/fleet/cpu-run python3 experiments/baseline_weights/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results" / "results.json").read_text())
TABLE = HERE.parent / "regating_table" / "results"
COLOR = {"copy": "#2a78d6", "cipher": "#eb6834", "learned nothing": "#8a8984"}
NULL_C, CTRL_C, INK, GRID = "#c9c8c3", "#1baf7a", "#52514e", "#e4e3df"
LABEL = {"ident32_s0": "copy A s0", "ident32_s1": "copy A s1", "ident32b_s0": "copy B s0", "ident32b_s1": "copy B s1",
         "rand32_s0": "cipher A s0", "rand32_s1": "cipher A s1", "rand32b_s0": "cipher B s0",
         "rand32b_s1": "cipher B s1", "code_step10": "code text,\nlearned\nnothing"}


def regating_share() -> dict[str, float]:
    """E46 stage 0a's r = g / G per E12 state (route split of the least-squares split state); the code state's from 0b."""
    a = json.loads((TABLE / "stage0a.json").read_text())["groups"]
    out = {f"{t}_{s}": a[f"e12/{t}"]["states"][s]["r"] for t in ("ident32", "ident32b", "rand32", "rand32b") for s in ("s0", "s1")}
    c = json.loads((TABLE / "stage0b.json").read_text())["states"]["code/step10"]
    out["code_step10"] = c["g"] / c["G"]
    return out


def panel(ax: Any, names: list[str], vals: list[float], title: str, null: list[float] | None) -> None:
    x = range(len(names))
    if null:
        ax.axhspan(min(null), max(null), color=NULL_C, alpha=0.6, lw=0, label="random update of the same size")
    ax.scatter(x, vals, s=60, c=[COLOR[R["states"][n]["kind"]] for n in names], zorder=3, edgecolor="white", lw=1.5)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xticks(list(x), [LABEL[n] for n in names], rotation=90, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    names = list(R["states"])
    agg = {n: R["states"][n]["agg"] for n in names}
    null = {s: [R["states"][n]["null"][s] for n in names] for s in ("share", "intruders", "erank")}
    ctl = R["control"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.6), constrained_layout=True)
    fig.suptitle("Do weight statistics of the update tell copy (elicit) from cipher (teach)?", fontsize=12, x=0.01, ha="left")

    panel(axs[0], names, [agg[n]["share"] for n in names], "share on the original's top 10 directions", null["share"])
    axs[0].set_yscale("log")
    axs[0].axhline(ctl["reweight"]["share"], color=CTRL_C, ls="--", lw=1.5, label="pure re-weighting control")
    axs[0].legend(fontsize=8, loc="center right", frameon=False)

    panel(axs[1], names, [agg[n]["intruders"] for n in names], "intruders in the top 10, per matrix", null["intruders"])
    axs[1].axhline(ctl["inject"]["intruders"], color=CTRL_C, ls="--", lw=1.5, label="one new direction injected")
    axs[1].set_ylim(-0.1, 1.2)
    axs[1].legend(fontsize=8, loc="center right", frameon=False)

    panel(axs[2], names, [agg[n]["erank"] for n in names], "effective rank of the update", null["erank"])
    axs[2].axhline(ctl["reweight"]["erank"], color=CTRL_C, ls="--", lw=1.5, label="pure re-weighting of 10 directions")
    axs[2].set_ylim(0, 760)
    axs[2].legend(fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.4), frameon=False)

    r = regating_share()
    panel(axs[3], names, [r[n] for n in names], "VPD: share of the gain from Γ's new values", None)
    axs[3].set_ylim(0, 1.05)
    for kind, c in COLOR.items():
        axs[3].scatter([], [], c=c, s=60, label=kind)
    axs[3].legend(fontsize=8, loc="lower left", frameon=False)
    fig.savefig(HERE / "figures" / "weights.png", dpi=150)
    size()


def size() -> None:
    """Effective rank against the update's size, every state: does the rank follow the kind or the size?"""
    fig, ax = plt.subplots(figsize=(6.5, 4.4), constrained_layout=True)
    for n, rec in R["states"].items():
        norm = sum(v["norm2"] for v in rec["per_matrix"].values()) ** 0.5
        ax.scatter(norm, rec["agg"]["erank"], s=60, c=COLOR[rec["kind"]], edgecolor="white", lw=1.5, zorder=3)
        if n in ("ident32_s1", "ident32b_s0", "ident32b_s1", "rand32_s1", "rand32b_s1"):
            continue
        text = {"ident32_s0": "copy, all four", "rand32_s0": "cipher A, both seeds",
                "rand32b_s0": "cipher B, both seeds"}.get(n, LABEL[n].replace("\n", " "))
        ax.annotate(text, (norm, rec["agg"]["erank"]), fontsize=7, color=INK,
                    xytext=(5, 3), textcoords="offset points")
    nulls = [rec["null"]["erank"] for rec in R["states"].values()]
    ax.axhspan(min(nulls), max(nulls), color=NULL_C, alpha=0.6, lw=0, label="random update of the same size")
    for kind, c in COLOR.items():
        ax.scatter([], [], c=c, s=60, label=kind)
    ax.set_xlabel("size of the update, ‖δW‖ over all 24 matrices")
    ax.set_ylabel("effective rank of the update")
    ax.set_title("Does the effective rank follow the kind of update or its size?", fontsize=10, loc="left")
    ax.set_ylim(0, 760)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.savefig(HERE / "figures" / "erank_vs_size.png", dpi=150)



if __name__ == "__main__":
    main()
