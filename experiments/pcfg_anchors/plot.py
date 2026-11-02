"""Figures for E55 Stage A (pcfg_anchors), read from results/*.json only; nothing is recomputed from a model.
Plan: reports/plans/2026-10-29_plan_reuse-stage1.md; analysis: reports/analyses/2026-10-29_analysis_e55-stage-a.md.

  controls.png  the patch check P by block on the pretrained models: the two positive controls (LOW on its count-a
                task, HIGH on its count-b task) and the two nulls, mean over 3 sets of 1,000 pairs, with the filed
                block l* = 5 marked
  probe.png     the probe margin by block (probe accuracy minus its best control): LOW count-a (the positive control
                m+), LOW count-b, HIGH count-b, with the possession bar 0.5 m+ and 3 s of LOW count-b
  labels.png    every fine-tuned state on the count-a check (left) and the count-b check (right): behaviour B against
                patch P, the filed lo and hi bars, the A1 null and positive reference, 1K -> 10K arrows per arm
  table6.png    count-b accuracy at 10K against Jain et al.'s Table 6 cell, with the +-15 point band and the
                score of always answering count(a) + 1

Usage (repo root): ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/pcfg_anchors/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED = "#0b0b0b", "#8a8984"
N_C, M_C, W_C, R_C, NULL_C = "#2a78d6", "#7b4fc9", "#eb6834", "#2e9e5b", "#8a8984"
# name -> (words, colour, marker); seed-1 copies share their arm's colour with a hollow marker
ARMS: dict[str, tuple[str, str, str]] = {
    "N1": ("fast, no spurious rows", N_C, "o"), "N1s1": ("fast, no spurious rows, seed 1", N_C, "o"),
    "N2": ("fast, half spurious", N_C, "s"), "N3": ("fast, 80% spurious", N_C, "^"),
    "M": ("slow, no spurious rows", M_C, "o"), "M2": ("slow, half spurious", M_C, "s"),
    "M3": ("slow, 80% spurious", M_C, "^"),
    "W1": ("slow, all spurious", W_C, "o"), "W1s1": ("slow, all spurious, seed 1", W_C, "o"),
    "W2": ("fast, all spurious", W_C, "D"), "R": ("HIGH base, no spurious rows", R_C, "*"),
}
TICK = {"N1": "1e-4\n0%", "N2": "1e-4\n50%", "N3": "1e-4\n80%", "M": "1e-5\n0%", "M2": "1e-5\n50%", "M3": "1e-5\n80%",
        "R": "HIGH\n1e-5, 0%", "W1": "1e-5\n100%", "W2": "1e-4\n100%"}  # lr, share of spurious rows
TABLE6_ORDER = ("N1", "N2", "N3", "M", "M2", "M3", "R", "W1", "W2")


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())


def finish(fig: Figure, path: str, title: str, bottom: float | None = None) -> None:
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if bottom is not None:
        fig.subplots_adjust(bottom=bottom)
    fig.savefig(FIG / path, dpi=130)
    plt.close(fig)


def fig_controls(out: str = "controls.png", title: str = "The patch moves the answer only where the swapped count is used") -> None:  # fmt: skip
    patch, bars = load("a1c_patch.json")["patch"], load("predictions.json")["a2_bars"]
    blocks = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for key, lab, col, ls in (("P_a_positive", "LOW asked for count(a), count-a swapped (positive control)", N_C, "-"),
                              ("P_b_positive", "HIGH asked for count(b), count-b swapped (positive control)", R_C, "-"),
                              ("P_a_null", "NEVER asked for count(b), count-a swapped (null)", NULL_C, "--"),
                              ("P_b_null", "LOW asked for count(a), count-b swapped (null)", INK, ":")):  # fmt: skip
        ax.plot(blocks, np.mean(patch[key], 0), ls, marker="o", color=col, lw=2, ms=6, label=lab)
    ax.axvline(bars["l_star_a"], color=MUTED, lw=1)
    ax.text(bars["l_star_a"] - 0.08, 0.5, "block used for every\nfine-tuned model", ha="right", fontsize=8, color=INK)
    ax.axvspan(5.5, 6.5, color=MUTED, alpha=0.12, lw=0)
    ax.text(6, 0.3, "last block:\nnot a candidate", ha="center", fontsize=8, color=INK)
    ax.set_xlim(0.7, 6.5)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("block whose output at '=' is swapped from the donor run")
    ax.set_ylabel("share of pairs whose answer becomes the donor's")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    finish(fig, out, title)


def fig_probe(out: str = "probe.png", title: str = "LOW's count of b decodes above shuffled labels, just over the possession bar") -> None:  # fmt: skip
    pr = load("a1c_probe.json")["probes"]
    m, mp = pr["margins"], pr["m_plus"]["m_plus"]
    blocks = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    for key, lab, col in (("LOW_count_a", "LOW, count of a (positive control), minus the better of shuffled labels and NEVER's probe", N_C),
                          ("HIGH_count_b", "HIGH, count of b (trained on it), minus shuffled labels", R_C),
                          ("LOW_count_b", "LOW, count of b (never asked for it), minus shuffled labels", W_C)):  # fmt: skip
        ax.plot(blocks, m[key]["m"], "-o", color=col, lw=2, ms=6, label=lab)
    lb = np.array(m["LOW_count_b"]["m"])
    sb = np.array(m["LOW_count_b"]["s"])
    ax.fill_between(blocks, lb - 3 * sb, lb + 3 * sb, color=W_C, alpha=0.15, lw=0, label="LOW count of b, 3 sd over probe seeds")
    ax.axhline(0.5 * mp, color=INK, ls="--", lw=1)
    ax.text(6.05, 0.5 * mp, "possession bar:\nhalf the positive\ncontrol's best", fontsize=8, va="center")
    ax.axhline(0, color=MUTED, lw=1)
    ax.text(6.05, 0.0, "no better than\nits control", fontsize=8, va="center", color=INK)
    ax.set_xlim(0.7, 7.1)
    ax.set_xlabel("block whose output at '=' the linear probe reads")
    ax.set_ylabel("probe accuracy minus its control's accuracy")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, frameon=False)
    finish(fig, out, title)


def state(arm: str, it: str, check: str) -> tuple[float, float]:
    c = load(f"a2_{arm}.json")["states"][it][f"check_count_{check}"]
    return c["B"], c["P"]


def check_panel(ax: Axes, check: str, refs: dict[str, tuple[float, float]], arms: tuple[str, ...],
                notes: dict[str, tuple[float, float]]) -> None:
    """B against P for every state of `arms`; `notes` = text -> position, naming the arms at their 10K points."""
    bars = load("predictions.json")["a2_bars"]
    bb, bp = bars[f"B_{check}"], bars[f"P_{check}"]
    ax.add_patch(Rectangle((bb["hi"], bp["hi"]), 1.1, 1.1, color=R_C, alpha=0.08, lw=0))
    ax.add_patch(Rectangle((-0.1, -0.1), bb["lo"] + 0.1, bp["lo"] + 0.1, color=N_C, alpha=0.08, lw=0))
    for v, ls in ((bb["lo"], ":"), (bb["hi"], "--")):
        ax.axvline(v, color=MUTED, ls=ls, lw=1)
    for v, ls in ((bp["lo"], ":"), (bp["hi"], "--")):
        ax.axhline(v, color=MUTED, ls=ls, lw=1)
    ax.text(bb["hi"] + 0.02, 1.02, "goes through", va="top", fontsize=8.5, color=R_C, weight="bold")
    ax.text(-0.04, -0.06, "does not go through", fontsize=8.5, color=N_C, weight="bold")
    for lab, (x, y) in refs.items():
        ax.plot(x, y, "X", color=INK, ms=11, mec="white")
        right = x > 0.5
        ax.annotate(lab, (x, y), xytext=(-8 if right else 8, -14 if right else 8), textcoords="offset points",
                    fontsize=7.5, color=INK, ha="right" if right else "left")
    for arm in arms:
        _w, col, mk = ARMS[arm]
        p1, p10 = state(arm, "1000", check), state(arm, "10000", check)
        ax.annotate("", xy=p10, xytext=p1, arrowprops={"arrowstyle": "->", "color": col, "lw": 1, "alpha": 0.7})
        ax.plot(*p1, mk, color=col, ms=7, mfc="white", mec=col, alpha=0.8)
        ax.plot(*p10, mk, color=col, ms=8)
    for text, xy in notes.items():
        ax.text(*xy, text, fontsize=7.5, color=INK)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.08, 1.06)
    ax.set_xlabel("behaviour: share of answers equal to " + ("count(a) + 1" if check == "a" else "count(b)"))


def fig_labels(out: str = "labels.png", title: str = "Fast fine-tunes never use count(a); slow ones with 80% or all spurious rows use it at 1K, but not by either check at 10K") -> None:  # fmt: skip
    a1 = {p: load(f"a1_{p}.json") for p in ("LOW", "HIGH")}
    bars = load("predictions.json")["a2_bars"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    refs_a = {"null: NEVER": (bars["B_a"]["n0"], bars["P_a"]["n0"]),
              "positive: LOW on its own task": (a1["LOW"]["count_a"], bars["P_a"]["p_plus"])}
    refs_b = {"null: LOW before fine-tuning": (bars["B_b"]["n0"], bars["P_b"]["n0"]),
              "positive: HIGH on its own task": (a1["HIGH"]["count_b"], bars["P_b"]["p_plus"])}
    from_low = tuple(a for a in ARMS if a != "R")
    check_panel(axes[0], "a", refs_a, from_low, {"W1 and its seed copy": (0.41, 0.31), "W2": (0.22, 0.30),
                                                 "every N and M arm at 10K": (0.13, 0.13)})
    check_panel(axes[1], "b", refs_b, tuple(ARMS), {"W1 and its seed copy": (0.02, 0.46), "W2": (0.12, 0.33),
                                                    "N1 and its seed copy": (0.78, 0.6),
                                                    "N2, N3, M, M2, M3, R": (0.70, 0.77)})
    axes[0].set_title("does the answer run through the old count of a?", fontsize=10, loc="left")
    axes[1].set_title("does the answer run through the count of b?", fontsize=10, loc="left")
    axes[0].set_ylabel("patch: share of pairs whose answer follows the donor")
    handles = [Line2D([], [], color=c, marker="o", ls="", ms=7) for c in (N_C, M_C, W_C, R_C)]
    handles += [Line2D([], [], color=INK, marker="o", mfc="white", ls="-", lw=0.8, ms=7),
                Line2D([], [], color=MUTED, ls="--"), Line2D([], [], color=MUTED, ls=":")]
    labels = ["from LOW, lr 1e-4, up to 80% spurious", "from LOW, lr 1e-5, up to 80% spurious", "from LOW, all rows spurious",
              "from HIGH (already counts b)", "hollow: 1K steps; filled: 10K; arrow between",
              "upper bar (halfway to the positive)", "lower bar"]  # fmt: skip
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    finish(fig, out, title, bottom=0.2)


def fig_table6(out: str = "table6.png", title: str = "Seven of nine fine-tunes land within 15 points of the paper; the two all-spurious ones do not") -> None:  # fmt: skip
    cells = load("predictions.json")["PA2"]["cells_table6_10K_percent"]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(TABLE6_ORDER))
    for i, arm in enumerate(TABLE6_ORDER):
        row = load(f"a2_{arm}.json")["states"]["10000"]["count_b_te0"]
        t = cells[arm]
        ax.add_patch(Rectangle((i - 0.3, max(t - 15, 0)), 0.6, min(t + 15, 100) - max(t - 15, 0), color=MUTED,
                                   alpha=0.2, lw=0, label="paper's value within 15 points" if i == 0 else None))  # fmt: skip
        ax.plot([i - 0.3, i + 0.3], [t, t], color=INK, lw=2, label="Jain et al. Table 6" if i == 0 else None)
        ax.plot(i, 100 * row["acc"], ARMS[arm][2], color=ARMS[arm][1], ms=9, label="ours" if i == 0 else None)
        ax.plot([i - 0.3, i + 0.3], [100 * row["floor_count_a_plus_1"]] * 2, color=W_C, ls=":", lw=1.5,
                label="score of always answering count(a) + 1" if i == 0 else None)  # fmt: skip
    ax.set_xticks(x, [f"{a}\n{TICK[a]}" for a in TABLE6_ORDER], fontsize=8)
    ax.set_ylim(-3, 108)
    ax.set_ylabel("count-b accuracy after 10K steps, %")
    ax.set_xlabel("arm: learning rate and share of spurious training rows")
    ax.legend(fontsize=8, loc="center left", frameon=False, bbox_to_anchor=(0.6, 0.45))
    finish(fig, out, title)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig_controls()
    fig_probe()
    fig_labels()
    fig_table6()


if __name__ == "__main__":
    main()
