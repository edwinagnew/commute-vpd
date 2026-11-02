"""E62 Amendment 3's figures, one per claim, from results/amend3.json, amend2.json and run.json (no model).

  figures/amend3_share.png  share lost per state: masks frozen, masks recomputed, target model; the ten references beside
  figures/amend3_slot.png   mean mask at the marker on the few block 2 o_proj subcomponents reading mostly each head's slot: W0, copying map, lookup
  figures/amend3_w0.png     W0's fall under each head: masks frozen, masks recomputed, target model
  figures/amend3_excess.png how far W0's reading and the base masked error move the recomputed share, against the target share

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/induction_cue/plot_amend3.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
TARGET, FROZEN, RECOMP, GREY = "#1baf7a", "#eda100", "#2a78d6", "#8a8983"
INK, GRID = "#52514e", "#e4e3df"
TOL, OTHER, FLOOR, CLOSED = 0.092, 0.25, 0.15, 0.1
COPY = [f"copy_s{s}" for s in (0, 1, 2)]
LOOKUP = [f"lookup_s{s}" for s in (0, 1, 2)]
NAMES = COPY + LOOKUP
TICKS = [f"copying map\nseed {s}" for s in (0, 1, 2)] + [f"lookup\nseed {s}" for s in (0, 1, 2)]


def style(ax: Any) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


def finish(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def share_figure(a: dict[str, Any]) -> None:
    st, w0 = a["row2_gamma"]["states"], a["row2_gamma"]["w0_shift"]
    fig, ax = plt.subplots(figsize=(10.4, 5))
    for i, n in enumerate(NAMES):
        s = st[n]
        ax.fill_between([i - 0.32, i + 0.32], s["target_share"] - TOL, s["target_share"] + TOL, color=TARGET, alpha=0.15, lw=0,
                        label="within 0.092 of the target model's share" if i == 0 else None)
        ax.scatter([i - 0.2], [s["target_share"]], s=75, marker="o", color=TARGET, zorder=4, label="target model, head (2,4) off" if i == 0 else None)
        ax.scatter([i - 0.05], [s["share_frozen"]], s=65, marker="s", color=FROZEN, zorder=4, label="masked forward, masks frozen, head (2,4) off" if i == 0 else None)
        ax.errorbar([i + 0.1], [s["share"]], yerr=[s["share_sd"]], fmt="D", ms=8, color=RECOMP, capsize=3, zorder=4,
                    label="masked forward, masks recomputed, head (2,4) off" if i == 0 else None)
        ax.scatter([i + 0.25], [s["ref_share"]], s=30, marker="^", facecolors="none", edgecolors=INK, zorder=4,
                   label="reference head (3,5) off, masks recomputed" if i == 0 else None)
    keys = ["copy_H", "copy_ref", "lookup_H", "lookup_ref"]
    ax.scatter(range(6, 10), [w0[k] for k in keys], s=30, marker="v", facecolors="none", edgecolors=INK, zorder=4,
               label="W₀'s recomputed fall over the map's gain")
    ax.axhline(OTHER, color=FROZEN, ls="--", lw=1.1)
    ax.text(9.45, OTHER + 0.012, "under 0.25: the filed reading that the mask reaches the target another way", ha="right", color=INK, fontsize=8.5)
    ax.axhline(FLOOR, color=GREY, ls=":", lw=1.2)
    ax.text(9.45, FLOOR + 0.012, "reference bar 0.15: any reference over it voids the reading", ha="right", color=INK, fontsize=8.5)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(5.5, color=GRID, lw=1)
    ax.set_xticks(range(10), [*TICKS, "W₀, copying\nhead (2,4)", "W₀, copying\nhead (3,5)", "W₀, lookup\nhead (2,4)", "W₀, lookup\nhead (3,5)"], fontsize=8.5)
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.12, 1.02)
    ax.set_ylabel("share of the log-odds gain lost")
    ax.set_title("With the masks recomputed, does the masked gain fall with the head, and only where it should?", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    style(ax)
    finish(fig, "amend3_share.png")


def slot_figure(a: dict[str, Any]) -> None:
    r = a["row3_slot"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for h, n in enumerate(r["n_per_head"]):
        ax.scatter([h - 0.2], [r["W0"]["copy"]["per_head"][h]], s=70, marker="X", color=INK, zorder=4, label="W₀'s mask" if h == 0 else None)
        for j, (names, col, mark, lab) in enumerate(((COPY, RECOMP, "D", "each copying-map state's mask"), (LOOKUP, TARGET, "o", "each lookup state's mask"))):
            ax.scatter([h + (j * 0.2)] * 3, [r["states"][m]["per_head"][h] for m in names], s=32, marker=mark, color=col, alpha=0.8, zorder=3,
                       label=lab if h == 0 else None)
    ax.axvspan(3.6, 4.4, color=GRID, alpha=0.5, lw=0)
    ax.axhline(CLOSED, color=GREY, ls=":", lw=1.2)
    ax.text(5.45, CLOSED * 1.06, "under 0.1: filed as closed", ha="right", color=INK, fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylim(0.0015, 0.5)
    ax.set_xlim(-0.5, 5.5)
    ax.set_xticks(range(6), [f"head (2,{h})\n{n} subcomponents" for h, n in enumerate(r["n_per_head"])], fontsize=8.5)
    ax.set_ylabel("mean mask at the marker, log scale")
    ax.set_title("Mean mask on the few subcomponents reading mostly each head's slot. Head (2,4) shaded", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(ax)
    finish(fig, "amend3_slot.png")


def w0_figure(a: dict[str, Any]) -> None:
    m = a["row1_control"]["maps"]
    groups = [("head (2,4), copying map's token", m["copy"], "fall_frozen", "fall_gamma", m["copy"]["fall_target"]),
              ("head (2,4), lookup's word", m["lookup"], "fall_frozen", "fall_gamma", m["lookup"]["fall_target"]),
              ("head (3,5), copying map's token", m["copy"], None, "fall_gamma_ref", None),
              ("head (3,5), lookup's word", m["lookup"], None, "fall_gamma_ref", None)]
    a2 = json.loads((RES / "amend2.json").read_text())["row3_masked"]["L00"]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for i, (_, rec, fz, g, t) in enumerate(groups):
        mp = "copy" if i in (0, 2) else "lookup"
        frozen = rec[fz] if fz else a2[mp]["on"] - a2[mp]["ref"]
        ax.bar(i - 0.13, frozen, 0.12, color=FROZEN, label="masked forward, masks frozen" if i == 0 else None)
        ax.bar(i + 0.0, rec[g], 0.12, color=RECOMP, label="masked forward, masks recomputed" if i == 0 else None)
        if t is not None:
            ax.bar(i + 0.13, t, 0.12, color=TARGET, label="target model" if i == 0 else None)
        else:
            ax.text(i + 0.13, 0.03, "not read", rotation=90, ha="center", color=INK, fontsize=7.5)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(4), [x[0].replace(", ", ",\n") for x in groups], fontsize=9)
    ax.set_ylabel("fall in W₀'s mean log-odds, zero update")
    ax.set_title("Before any fine-tune, how far does a head off move W₀'s read? Zero is no effect", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(ax)
    finish(fig, "amend3_w0.png")


def excess_figure(a: dict[str, Any], run: dict[str, Any]) -> None:
    st, w = a["row2_gamma"]["states"], a["row1_control"]["maps"]["copy"]
    series = [("recomputed share, as filed", RECOMP, "D", lambda s, _: s["share"]),
              ("W₀'s fall set to the target model's 1.16", GREY, "v", lambda s, _: s["own_fall"] - w["fall_target"] / s["M"]),
              ("recomputed share minus frozen share", INK, "P", lambda s, _: s["share"] - s["share_frozen"]),
              ("same net fall, gain measured from W₀ with every mask at 1", "#7a3fb0", "^", lambda s, c: s["share"] * s["M"] / (c["L01"] - c["U0"]))]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i, n in enumerate(COPY):
        s, c = st[n], run["states"][n]["cells"]
        ax.fill_between([i - 0.32, i + 0.32], s["target_share"] - TOL, s["target_share"] + TOL, color=TARGET, alpha=0.15, lw=0,
                        label="within 0.092 of the target model's share" if i == 0 else None)
        ax.scatter([i - 0.26], [s["target_share"]], s=75, marker="o", color=TARGET, zorder=4, label="target model's share" if i == 0 else None)
        ax.scatter([i - 0.15], [s["share_frozen"]], s=65, marker="s", color=FROZEN, zorder=4, label="frozen share" if i == 0 else None)
        for j, (lab, col, mark, f) in enumerate(series):
            ax.scatter([i - 0.04 + 0.09 * j], [f(s, c)], s=55, marker=mark, color=col, zorder=4, label=lab if i == 0 else None)
    ax.axhline(OTHER, color=FROZEN, ls="--", lw=1.1)
    ax.text(2.45, OTHER + 0.012, "under 0.25: the filed other-way reading", ha="right", color=INK, fontsize=8.5)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(3), [f"copying map\nseed {s}" for s in (0, 1, 2)])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("share of the log-odds gain lost")
    ax.set_title("Is the share above the target model's robust to W₀'s reading and the base error?", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2)
    style(ax)
    finish(fig, "amend3_excess.png")


def main() -> None:
    a = json.loads((RES / "amend3.json").read_text())
    run = json.loads((RES / "run.json").read_text())
    share_figure(a)
    slot_figure(a)
    w0_figure(a)
    excess_figure(a, run)


if __name__ == "__main__":
    main()
