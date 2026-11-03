"""Figures and report tables for E70 Amendment 2, read from results/e70_a2.json and e70_a1.json only; no model.
Analysis: reports/analyses/2026-11-03_analysis_e70-existing-new-counting.md.

  e70_a2_tasks.png  tasks P and N at 10K: share of the log-odds gain lost under the covariance direction and under the
                    largest direction with no covariance with the count, with the random band, full scale and close up,
                    and each block's variance shares and class shares beneath
  e70_a2_low.png    LOW's own log-odds drop per block under the same two directions, against 20 random directions and
                    their band

Usage (repo root): PYTHONPATH=experiments/pcfg_anchors ~/commute-agents/fleet/cpu-run .venv/bin/python \
    experiments/pcfg_anchors/plot_e70_a2.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from plot import INK, MUTED, finish, load, plt
from plot_e70 import BLOCKS, N_C, P_C, PROBE_C
from plot_e70_a1 import fmt

NO_COV_C = "#7b4fc9"
TASKS = (("P", "task P, count(a) + 1", P_C, -0.04), ("N", "task N, count(b)", N_C, 0.04))
BAND_WORDS = "random band, from directions holding about 0.005\nof the variance"


def means(summary: list[dict[str, Any]], key: str) -> np.ndarray:
    """(blocks, 3) of [mean, min, max] from a run's summary."""
    return np.array([row[key] for row in summary])


def share_lines(ax: Axes, summary: list[dict[str, Any]], style: str, words: str, hollow: bool) -> None:
    """One direction's shares: mean with a min-to-max bar per task."""
    for task, task_words, col, dx in TASKS:
        v = means(summary, f"{task}_A_lo")
        ax.vlines(BLOCKS + dx, v[:, 1], v[:, 2], color=col, lw=5, alpha=0.35)
        ax.plot(BLOCKS + dx, v[:, 0], style, color=col, lw=2, ms=7, mfc="white" if hollow else col, label=f"{task_words}, {words}")


def band_fill(ax: Axes, band: np.ndarray) -> None:
    ax.axhline(0, color=INK, lw=0.8)
    ax.fill_between(BLOCKS, -band, band, color=MUTED, alpha=0.35, lw=0, label=BAND_WORDS)


def strip(ax: Axes, control: dict[str, Any], why: dict[str, Any]) -> None:
    """Each block's variance shares and class shares, as a row of numbers under the plot."""
    rows = (
        ("covariance: variance held", control["var_share_a"]),
        ("covariance: between classes", why["class_a_a"]),
        ("no-covariance: variance held", control["var_share_v"]),
        ("no-covariance: between classes", why["class_a_v"]),
    )
    for y, (_, values) in zip(range(len(rows) - 1, -1, -1), rows, strict=True):
        for b, v in zip(BLOCKS, values, strict=True):
            ax.text(float(b), y, f"{v:.2f}", ha="center", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows) - 1, -1, -1), [r[0] for r in rows], fontsize=8)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("block where the direction is removed")


def tasks_figure() -> None:
    res, a1 = load("e70_a2.json"), load("e70_a1.json")
    band = np.array([r["band"] for r in res["r2"]["rows"]])
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 6.6), gridspec_kw={"height_ratios": [3.6, 1], "width_ratios": [1.25, 1]})
    full, close, under, legend_ax = axes.ravel()
    band_fill(full, band)
    share_lines(full, a1["summary"], "-o", "covariance direction", False)
    share_lines(full, res["summary"], "--o", "no-covariance direction", True)
    full.set_ylim(-0.08, 1.0)
    full.set_ylabel("share of the log-odds gain lost")
    full.set_title("Full scale: 0 means no more lost than LOW, 1 means the whole gain", fontsize=9, loc="left")
    full.tick_params(axis="x", labelbottom=False)

    band_fill(close, band)
    share_lines(close, res["summary"], "--o", "no-covariance direction", True)
    close.set_ylim(-0.03, 0.05)
    close.set_title("Close up: the no-covariance direction against the band", fontsize=9, loc="left")
    close.annotate("block 6 is off this scale:\nP 0.21, N 0.20", (5.17, 0.046), xytext=(4.75, 0.042), ha="right", va="center", fontsize=8, color=INK, arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.8})
    close.set_xlabel("block where the direction is removed")
    close.set_ylabel("share of the log-odds gain lost")

    for ax in (full, close, under):
        ax.set_xlim(0.7, 6.3)
        ax.set_xticks(BLOCKS)
    strip(under, res["control"]["no_count"], res["why"])
    handles, labels = full.get_legend_handles_labels()
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, fontsize=8, frameon=False, loc="center left", ncol=1, title="'between classes': share of the direction's variance\nlying between the count(a) classes, on LOW", title_fontsize=8, alignment="left")
    finish(fig, "e70_a2_tasks.png", "At blocks 3 to 5 the covariance direction costs P more than N, and the largest direction without covariance costs neither")


def low_figure() -> None:
    res = load("e70_a2.json")
    c = res["control"]["no_count"]
    draws, band = np.array(c["random"]), np.array(c["band"])
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.axhline(0, color=INK, lw=0.8)
    ax.fill_between(BLOCKS, -band, band, color=MUTED, alpha=0.35, lw=0, label=BAND_WORDS)
    ax.plot(np.repeat(BLOCKS, draws.shape[1]), draws.ravel(), ".", color=MUTED, ms=4, label="20 random directions")
    ax.plot(BLOCKS, c["drop_a_b"], "-o", color=PROBE_C, lw=2, ms=7, label="covariance direction")
    ax.plot(BLOCKS, c["drop"], "--o", color=NO_COV_C, lw=2, ms=7, mfc="white", label="no-covariance direction")
    for b, d, v, k in zip(BLOCKS, c["drop_a_b"], c["var_share_a"], res["why"]["class_a_a"], strict=True):
        ax.annotate(f"holds {v:.2f}\nbetween classes {k:.2f}", (float(b), d), textcoords="offset points", xytext=(0, 9 if b != 6 else -21), ha="center", fontsize=7.5, color=INK)
    for b, d, v, k in zip(BLOCKS, c["drop"], c["var_share_v"], res["why"]["class_a_v"], strict=True):
        ax.annotate(f"holds {v:.2f}\nbetween classes {k:.2f}", (float(b), d), textcoords="offset points", xytext=(0, -26 if b != 6 else 10), ha="center", fontsize=7.5, color=INK)
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(-2, 15)
    ax.set_xticks(BLOCKS)
    ax.set_xlabel("block where the direction is removed")
    ax.set_ylabel("LOW's drop in count(a) log-odds, 'C a 40' rows")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    finish(fig, "e70_a2_low.png", "LOW loses 8 to 12 log-odds to the covariance direction at blocks 3 to 5\nand nothing to the other; at block 6 the other costs more")


def tables() -> None:
    res, a1 = load("e70_a2.json"), load("e70_a1.json")
    c, why, summary, rows = res["control"]["no_count"], res["why"], res["summary"], res["r2"]["rows"]
    for key in ("drop", "drop_a_b", "band", "acc", "acc_a_b", "var_share_v", "var_share_a", "var_share_random"):
        print(key, fmt(c[key], 4))
    for key in ("class_a_v", "class_a_a", "class_a_random", "class_b_v", "class_b_a", "refit_after_v", "before"):
        print(key, fmt(why[key], 4))
    print("cov ratio fit", [f"{v / a:.1e}" for v, a in zip(why["cov_fit_v"], why["cov_fit_a"], strict=True)])
    print("cov ratio held", [f"{v / a:.1e}" for v, a in zip(why["cov_held_v"], why["cov_held_a"], strict=True)])
    for key in ("A_lo", "A_lo_raw", "A_nats", "acc"):
        for task in "PN":
            v = means(summary, f"{task}_{key}")
            print(f"{task} {key} mean", fmt(v[:, 0], 5), f"\n{task} {key} min ", fmt(v[:, 1], 5), f"\n{task} {key} max ", fmt(v[:, 2], 5))
    print("band", fmt([r["band"] for r in rows], 4), "\ngap", fmt([r["gap"] for r in rows], 4), "\nreverse", fmt([r["reverse_gap"] for r in rows], 4), "\ncall", [r["call"] for r in rows])
    for task, sid in (("P", "c1b/P_1e-4_s0/10000"), ("N", "a2/N1/10000")):
        print(f"LOW's drop for {task}'s answer", fmt([b["lo_drop_W0"] for b in res["states"][sid]["probe"]["no_count"]]))
    for task in "PN":
        names = [s for s in res["states"] if f"{task}_1e" in s or (task == "N" and s.startswith("a2/"))]
        print(f"{task}'s own drop, mean", fmt(np.mean([[b["lo_drop"] for b in res["states"][s]["probe"]["no_count"]] for s in names], axis=0)))
    share = res["share_per_variance_descriptive"]
    for key in ("P_a", "N_a", "P_v", "N_v"):
        print("per unit of variance", key, fmt(share[key]))
    p_a = means(a1["summary"], "P_A_lo")[:, 0]
    print("P under a_b scaled by the variance ratio", fmt(p_a * np.array(c["var_share_v"]) / np.array(c["var_share_a"])))
    n_a = means(a1["summary"], "N_A_lo")[:, 0]
    print("N under a_b scaled by the variance ratio", fmt(n_a * np.array(c["var_share_v"]) / np.array(c["var_share_a"])))
    print("variance held by the two together", fmt(np.array(c["var_share_v"]) + np.array(c["var_share_a"]), 4))
    print("branch", res["branch"], "\nteeth", res["teeth"], "\nblock 6", res["block_6"])


if __name__ == "__main__":
    tasks_figure()
    low_figure()
    tables()
