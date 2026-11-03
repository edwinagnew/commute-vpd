"""Figures and report tables for E70, read from results/e70.json only; no model.
Analysis: reports/analyses/2026-11-03_analysis_e70-existing-new-counting.md.

  e70_softmax.png  share of the log-odds gain lost when the softmax probe's six directions are removed, per block, task
                   P and task N at 10K: min, mean and max per task, raw share beside it, the block's random band
  e70_ridge.png    the same axes when the ridge probe's one direction is removed
  e70_low.png      LOW's own log-odds drop per block under each probe, against its 20 random draws and random band

Usage (repo root): PYTHONPATH=experiments/pcfg_anchors ~/commute-agents/fleet/cpu-run .venv/bin/python \
    experiments/pcfg_anchors/plot_e70.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from plot import INK, MUTED, finish, load, plt

P_C, N_C = "#2e9e5b", "#2a78d6"  # task P (reuses), task N (new), as in plot_c1b.py
PROBE_C = "#eb6834"
BLOCKS = np.arange(1, 7)
Y_SHARE = (-0.17, 0.88)  # one scale for both share figures, so the ridge figure's flat lines are a like-for-like reading


def side(rows: list[dict[str, Any]], key: str, name: str) -> np.ndarray:
    """(blocks, states) values of one side, e.g. key 'shares', name 'P_10000'."""
    return np.array([list(r[key][name].values()) for r in rows], dtype=float)


def share_figure(probe: str, out: str, title: str, note: dict[int, str], note_y: float) -> None:
    rows = load("e70.json")["r2"][probe]["rows"]
    band = np.array([r["band"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(0, color=INK, lw=0.8)
    ax.fill_between(BLOCKS, -band, band, color=MUTED, alpha=0.3, lw=0, label="random band at that block")
    for task, words, col, dx in (("P", "task P, count(a) + 1", P_C, -0.03), ("N", "task N, count(b)", N_C, 0.03)):
        scored = side(rows, "shares", f"{task}_10000")
        ax.vlines(BLOCKS + dx, scored.min(1), scored.max(1), color=col, lw=5, alpha=0.35)
        ax.plot(BLOCKS + dx, scored.mean(1), "-o", color=col, lw=2, ms=7, label=f"{words}: mean; bar, min to max of 10 states")
        raw = side(rows, "shares_raw", f"{task}_10000").mean(1)
        ax.plot(BLOCKS + dx, raw, "--", color=col, lw=1.2, label=f"{words}: mean, raw (LOW's drop not subtracted)")
    for block, text in note.items():
        ax.text(block, note_y, text, ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_ylim(*Y_SHARE)
    ax.set_xlim(0.7, 6.3)
    ax.set_xticks(BLOCKS)
    ax.set_xlabel("block where the direction is removed")
    ax.set_ylabel("share of the log-odds gain lost")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    finish(fig, out, title)


def low_panel(ax: Axes, control: dict[str, Any], title: str) -> None:
    band, drop = np.array(control["band"]), np.array(control["drop"])
    ax.axhline(0, color=INK, lw=0.8)
    ax.fill_between(BLOCKS, -band, band, color=MUTED, alpha=0.3, lw=0, label="random band")
    draws = np.array(control["random"])
    ax.plot(np.repeat(BLOCKS, draws.shape[1]), draws.ravel(), ".", color=MUTED, ms=4, label="20 random draws")
    ax.plot(BLOCKS, drop, "-o", color=PROBE_C, lw=2, ms=7, label="probe's removal")
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xticks(BLOCKS)
    ax.set_xlabel("block where the direction is removed")
    ax.legend(fontsize=8, frameon=False)


def low_figure() -> None:
    control = load("e70.json")["control"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    low_panel(axes[0], control["ridge"], "Ridge, one direction (y-axis to about 0.4)")
    low_panel(axes[1], control["softmax"], "Softmax, six directions (y-axis to about 10)")
    axes[0].set_ylabel("LOW's drop in count(a) log-odds, 'C a 40' rows")
    finish(fig, "e70_low.png", "Does removing the probe cost LOW its own count? Ridge never, softmax at every block")


def fmt(row: np.ndarray) -> str:
    return " | ".join(f"{v:.3f}" for v in row)


def tables() -> None:
    res = load("e70.json")
    for probe in ("ridge", "softmax"):
        rows = res["r2"][probe]["rows"]
        print(f"\n{probe}: band, gap, reverse gap, call")
        for r in rows:
            print(f"| {r['block']} | {r['band']:.3f} | {r['gap']:.3f} | {r['reverse_gap']:.3f} | {r['call']} |")
        for key, names in (("shares", ("P_10000", "N_10000", "P_1000", "N_1000")), ("shares_raw", ("P_10000", "N_10000"))):
            for name in names:
                v = side(rows, key, name)
                print(f"{key} {name} mean | {fmt(v.mean(1))}\n{key} {name} min  | {fmt(v.min(1))}\n{key} {name} max  | {fmt(v.max(1))}")


if __name__ == "__main__":
    share_figure("softmax", "e70_softmax.png", "Removing the six softmax directions costs task P more than task N at blocks 2-5",
                 {b: "gap beats band" for b in (2, 3, 4, 5)}, -0.15)
    share_figure("ridge", "e70_ridge.png", "Removing the one ridge direction costs neither task anything",
                 {1: "N above P by 0.005:\nfiled reversal"}, 0.03)
    low_figure()
    tables()
