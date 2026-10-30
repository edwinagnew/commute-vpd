"""E58 Amendment 4's figures, one per claim, from results/amend4.json (job 46426), with seed 0's map reads from
results/amend3.json (job 46398), as the job scored them. No model.

  figures/a4_within.png   r against held-out carry x for the three maps (elicit, rank-1, copy), three seeds each, at
                          steps 10 and 30; each map's seed mean with its seed range; the filed bound (2 x the largest
                          seed range of r) drawn up from the lowest map mean; the cipher at its first step over the gate
  figures/a4_shuffle.png  r of the cipher (three seeds), the shuffled elicit map (seed 0) and the elicit map (three
                          seeds), at the first step over the gate and at step 30, with the filed cut 0.5 and the 24-key
                          cipher mean

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/open_pair/plot_amend4.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot import COLOR, FIG, GRID, HERE, INK, SHADE, style

MAPS = (
    ("elicit", "elicit (half-known)"),
    ("rank1", "rank-1 (W0's top word)"),
    ("copy", "copy (value = key)"),
)
SEEDS = (0, 1, 2)
R_LABEL = "r: share of the update's gain\non W0's own subcomponents"


def load() -> tuple[dict[str, Any], dict[str, Any]]:
    a4 = json.loads((HERE / "results" / "amend4.json").read_text())
    a3 = json.loads((HERE / "results" / "amend3.json").read_text())
    return a4, a3


def map_read(
    a4: dict[str, Any], a3: dict[str, Any], m: str, seed: int, step: int
) -> dict[str, Any]:
    """Seed 0's map reads are Amendment 3's (the job scored them so); seeds 1-2 are Amendment 4's."""
    return (
        a3["reads"][f"{m}@{step}"] if seed == 0 else a4["reads"][f"{m}_s{seed}@{step}"]
    )


def cipher_gate(a4: dict[str, Any], seed: int) -> dict[str, Any]:
    return a4["reads"][f"cipher_s{seed}@{a4['control'][str(seed)]['step']}"]


def within_fig(a4: dict[str, Any], a3: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
    for ax, step in zip(axes, (10, 30), strict=True):
        w = a4["score"]["within_key"][str(step)]["r"]
        means = {}
        for m, label in MAPS:
            reads = [map_read(a4, a3, m, s, step) for s in SEEDS]
            xs, rs = [d["x"] for d in reads], [d["r"] for d in reads]
            means[m] = sum(rs) / 3
            ax.scatter(xs, rs, s=16, color=COLOR[m], alpha=0.5, zorder=3)
            mx = sum(xs) / 3
            ax.plot([mx, mx], [min(rs), max(rs)], color=COLOR[m], lw=1.2, zorder=3)
            ax.scatter(
                [mx],
                [means[m]],
                s=50,
                color=COLOR[m],
                zorder=4,
                label=f"{label}: seed mean, range",
            )
        lo = min(means.values())
        ax.axhspan(lo, lo + w["bound"], color=SHADE, zorder=0)
        ax.text(
            -0.33,
            lo + w["bound"] + 0.005,
            f"pass zone: every map mean within the bound {w['bound']:.3f} of the lowest",
            fontsize=6.5,
            color=INK,
        )
        ax.text(
            -0.33,
            0.62,
            f"maps differ in r by {w['D']:.3f}; carry by {w['D_x']:.2f}\nfiled rule: {w['status']}",
            fontsize=7.5,
            color=INK,
        )
        cg = [cipher_gate(a4, s) for s in SEEDS]
        ax.scatter(
            [d["x"] for d in cg],
            [d["r"] for d in cg],
            s=30,
            color=COLOR["cipher"],
            marker="s",
            zorder=3,
            label="cipher, first step over the gate (15-16), 3 seeds",
        )
        ax.set_xlabel("held-out carry x (value-specific)", color=INK)
        ax.set_xlim(-0.35, 0.95)
        style(ax, f"step {step}", R_LABEL if step == 10 else "")
    axes[0].set_ylim(0.25, 0.97)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        "Across three seeds, does r differ between the three maps by no more than its own seed noise?",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "a4_within.png", dpi=150)
    plt.close(fig)


def shuffle_fig(a4: dict[str, Any], a3: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    rd = a4["reads"]
    groups = (
        (
            "cipher (no rule), 3 seeds",
            COLOR["cipher"],
            [cipher_gate(a4, s)["r"] for s in SEEDS],
            [rd[f"cipher_s{s}@30"]["r"] for s in SEEDS],
        ),
        (
            "shuffled elicit, seed 0",
            "#6b6a66",
            [rd[f"shuffle_s0@{a4['shuffle_gate_step']}"]["r"]],
            [rd["shuffle_s0@30"]["r"]],
        ),
        (
            "elicit, 3 seeds",
            COLOR["elicit"],
            [map_read(a4, a3, "elicit", s, 10)["r"] for s in SEEDS],
            [map_read(a4, a3, "elicit", s, 30)["r"] for s in SEEDS],
        ),
    )
    for i, (name, c, first, at30) in enumerate(groups):
        ax.scatter(
            [i - 0.12 + 0.04 * k for k in range(len(first))],
            first,
            s=30,
            color="white",
            edgecolors=c,
            zorder=3,
            label="first step over the gate (cipher 15-16, shuffle 15; elicit 10)"
            if i == 0
            else None,
        )
        ax.scatter(
            [i + 0.12 + 0.04 * k for k in range(len(at30))],
            at30,
            s=36,
            color=c,
            zorder=3,
            label="step 30 (scored for the shuffle)" if i == 0 else None,
        )
    ax.axhline(0.5, color=INK, lw=1.0, ls=":")
    ax.text(
        2.45,
        0.505,
        "filed cut 0.5: below = cipher-like",
        fontsize=7,
        color=INK,
        ha="right",
        va="bottom",
    )
    ax.axhline(0.2306, color=COLOR["cipher"], lw=0.9, ls="--")
    ax.text(
        2.45,
        0.235,
        "cipher, 24 keys, 3-seed mean",
        fontsize=7,
        color=INK,
        ha="right",
        va="bottom",
    )
    ax.set_xticks(range(3), [g[0] for g in groups], fontsize=8)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.15, 0.85)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    style(
        ax,
        "Same words as elicit, no link to W0's guess per key: does r read like the cipher?",
        R_LABEL,
    )
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "a4_shuffle.png", dpi=150)
    plt.close(fig)


def main() -> None:
    a4, a3 = load()
    FIG.mkdir(exist_ok=True)
    within_fig(a4, a3)
    shuffle_fig(a4, a3)
    for step in (10, 30):
        for m, _ in MAPS:
            reads = [map_read(a4, a3, m, s, step) for s in SEEDS]
            print(
                step,
                m,
                [round(d["x"], 3) for d in reads],
                [round(d["r"], 3) for d in reads],
            )


if __name__ == "__main__":
    main()
