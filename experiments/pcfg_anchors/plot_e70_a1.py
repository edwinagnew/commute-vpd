"""Figures and report tables for E70 Amendment 1, read from results/e70_a1.json and e70.json only; no model.
Analysis: reports/analyses/2026-11-03_analysis_e70-existing-new-counting.md.

  e70_a1_low.png     LOW's own log-odds drop per block under the covariance direction and the regression direction,
                     against 20 random directions and their band, with the share of variance each direction holds
  e70_a1_tasks.png   tasks P and N at 10K under the covariance direction: share of the log-odds gain lost, and accuracy
                     after removal, per block

Usage (repo root): PYTHONPATH=experiments/pcfg_anchors ~/commute-agents/fleet/cpu-run .venv/bin/python \
    experiments/pcfg_anchors/plot_e70_a1.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from plot import INK, MUTED, finish, load, plt
from plot_e70 import BLOCKS, N_C, P_C, PROBE_C, side

REGRESSION_C = "#7b4fc9"
TASKS = (("P", "task P, count(a) + 1", P_C, -0.04), ("N", "task N, count(b)", N_C, 0.04))
FLOOR = 0.3575  # always answering the commonest count, on LOW's 'C a 40' rows; 0.353 and 0.357 on P's and N's


def low_figure() -> None:
    res = load("e70_a1.json")
    c, why = res["control"]["pattern"], res["why"]
    drop, band = np.array(c["drop"]), np.array(c["band"])
    draws = np.array(c["random"])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(0, color=INK, lw=0.8)
    ax.fill_between(BLOCKS, -band, band, color=MUTED, alpha=0.3, lw=0, label="random band: each random direction holds about 0.005 of the variance")
    ax.plot(np.repeat(BLOCKS, draws.shape[1]), draws.ravel(), ".", color=MUTED, ms=4, label="20 random directions")
    ax.plot(BLOCKS, c["drop_ridge"], "-o", color=REGRESSION_C, lw=2, ms=6, label="regression direction: holds 0.00006 to 0.03")
    ax.plot(BLOCKS, drop, "-o", color=PROBE_C, lw=2, ms=7, label="covariance direction: holds 0.30 to 0.78")
    for b, d, v in zip(BLOCKS, drop, why["var_share_pattern"], strict=True):
        ax.annotate(f"holds {v:.2f}", (b, d), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=INK)
    ax.set_xlim(0.7, 6.3)
    ax.set_ylim(-1, 14)
    ax.set_xticks(BLOCKS)
    ax.set_xlabel("block where the direction is removed")
    ax.set_ylabel("LOW's drop in count(a) log-odds, 'C a 40' rows")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    finish(fig, "e70_a1_low.png", "The covariance direction costs LOW far more than the band, but holds far more variance")


def task_panel(ax: Axes, summary: list[dict[str, Any]], key: str) -> None:
    """Mean with a min-to-max bar per task, from the run's summary of [mean, min, max] per block."""
    for task, words, col, dx in TASKS:
        v = np.array([row[f"{task}_{key}"] for row in summary])
        ax.vlines(BLOCKS + dx, v[:, 1], v[:, 2], color=col, lw=5, alpha=0.35)
        ax.plot(BLOCKS + dx, v[:, 0], "-o", color=col, lw=2, ms=7, label=f"{words}: mean; bar, min to max of 10 states")


def tasks_figure() -> None:
    res = load("e70_a1.json")
    summary, rows = res["summary"], res["r2"]["rows"]
    six = load("e70.json")["r2"]["softmax"]["rows"]
    band = np.array([r["band"] for r in rows])
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.8), sharex=True)
    left.axhline(0, color=INK, lw=0.8)
    left.fill_between(BLOCKS, -band, band, color=MUTED, alpha=0.3, lw=0, label="random band, from directions of about 0.005 of the variance")
    task_panel(left, summary, "A_lo")
    left.plot(BLOCKS, side(six, "shares", "P_10000").mean(1), ":", color=P_C, lw=1.4, label="task P under E70's six directions")
    left.plot(BLOCKS, side(six, "shares", "N_10000").mean(1), ":", color=N_C, lw=1.4, label="task N under E70's six directions")
    for r in rows:
        text = {"holds": "P above N", "reversal": "N above P:\nfiled reversal", "neither": "neither"}[r["call"]]
        left.text(r["block"], -0.13, text, ha="center", va="top", fontsize=8, color=INK)
    left.set_ylim(-0.27, 1.32)
    left.set_ylabel("share of the log-odds gain lost")
    left.legend(fontsize=8, frameon=False, loc="upper left")
    right.axhline(1, color=INK, lw=0.8)
    right.text(1, 1.015, "no removal: 1.0 on every state", fontsize=8, color=INK)
    right.axhline(FLOOR, color=MUTED, lw=1.2, ls="--")
    right.text(6.25, FLOOR - 0.012, "always the commonest count", fontsize=8, color=INK, ha="right", va="top")
    task_panel(right, summary, "acc")
    right.plot(BLOCKS, res["control"]["pattern"]["acc"], "-o", color=PROBE_C, lw=1.6, ms=5, label="LOW, its own count on 'C a 40' rows")
    right.set_ylim(0, 1.08)
    right.set_ylabel("accuracy after removal")
    right.legend(fontsize=8, frameon=False, loc="lower left")
    for ax in (left, right):
        ax.set_xlim(0.7, 6.3)
        ax.set_xticks(BLOCKS)
        ax.set_xlabel("block where the covariance direction is removed")
    finish(fig, "e70_a1_tasks.png", "The reusing task loses more at blocks 1 and 3 to 5, the new task at block 6, and neither is spared in the middle")


def fmt(row: Any, digits: int = 3) -> str:
    return " | ".join(f"{v:.{digits}f}" for v in row)


def tables() -> None:
    res = load("e70_a1.json")
    c, why, summary, rows = res["control"]["pattern"], res["why"], res["summary"], res["r2"]["rows"]
    print("LOW drop", fmt(c["drop"]), "\nband", fmt(c["band"]), "\nregression drop", fmt(c["drop_ridge"]), "\nLOW acc", fmt(c["acc"], 4))
    for key in ("refit_after_pattern", "refit_after_ridge", "before", "var_share_pattern", "var_share_ridge", "var_share_random", "cos_pattern_ridge"):
        print(key, fmt(why[key], 5))
    for key in ("A_lo", "A_lo_raw", "A_nats", "acc"):
        for task in "PN":
            v = np.array([s[f"{task}_{key}"] for s in summary])
            print(f"{task} {key} mean", fmt(v[:, 0], 5), f"\n{task} {key} min ", fmt(v[:, 1], 5), f"\n{task} {key} max ", fmt(v[:, 2], 5))
    print("band", fmt([r["band"] for r in rows], 4), "\ngap", fmt([r["gap"] for r in rows], 4), "\nreverse", fmt([r["reverse_gap"] for r in rows], 4), "\ncall", [r["call"] for r in rows])
    for task, sid in (("P", "c1b/P_1e-4_s0/10000"), ("N", "a2/N1/10000")):
        print(f"LOW's drop for {task}'s answer", fmt([b["lo_drop_W0"] for b in res["states"][sid]["probe"]["pattern"]]))
    for task in "PN":
        names = [s for s in res["states"] if f"{task}_1e" in s or (task == "N" and s.startswith("a2/"))]
        print(f"{task}'s own drop, mean", fmt(np.mean([[b["lo_drop"] for b in res["states"][s]["probe"]["pattern"]] for s in names], axis=0)))


if __name__ == "__main__":
    low_figure()
    tasks_figure()
    tables()
