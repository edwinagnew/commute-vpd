"""Figures for teacher_training Amendment 10, a student's weight change against its fine-tune's, weighted by the inputs
the model sees (plan: reports/plans/2026-11-02_plan_e34-amend10-input-weighted.md), read from results/amend10.json only.

  amend10_cosine.png    each student's cosine beside the rerun line, the other task's line and the ceiling, by input set and form,
                        with the same read counting every weight direction alike (Amendment 9's Frobenius)
  amend10_distance.png  each student's distance from its fine-tune and its size, beside a rerun's distance and no change

Usage (repo root): ~/commute-agents/fleet/cpu-run python experiments/teacher_training/plot_amend10.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
D: dict[str, Any] = json.loads((HERE / "results" / "amend10.json").read_text())
INK, MUTED, GRID = "#0b0b0b", "#6e6d68", "#e6e5e0"
C = {"real": "#2a78d6", "rand32": "#1baf7a", "rand32b": "#6250d6"}  # as in amend9's figures
TABLES = {"real": "real task", "rand32": "first map table", "rand32b": "second map table"}
RUNS = {
    "real": ["real_teacher", "real_teacher_o1", "real_teacher_o2"],
    "rand32": [f"rand32_s{i}" for i in range(4)],
    "rand32b": [f"rand32b_s{i}" for i in range(4)],
}
SET = {"real": "emoticon", "rand32": "rand32", "rand32b": "rand32b"}
FORMS = {"uncentred": "as seen", "centred": "mean removed"}
OWN_FROB = {
    "real": {"real_o0": "real_teacher", "real_o1": "real_teacher"},
    "rand32": {"S_rand32_s0": "rand32_s0", "S_rand32_s1": "rand32_s1"},
    "rand32b": {"S_rand32b_s0": "rand32b_s0", "S_rand32b_s0_o1": "rand32b_s0", "S_rand32b_s1": "rand32b_s1", "S_rand32b_s1_o1": "rand32b_s1"},
}
# Frobenius ceilings: the filing's expected.frobenius_known_before_filing, the only place they are written.
CEILING_FROB = {"real": [0.787], "rand32": [0.814, 0.816], "rand32b": [0.813, 0.814]}
OTHER_FROB = {
    "real": ["rand32_s0", "rand32_s1", "rand32b_s0", "rand32b_s1", "null_teacher"],
    "rand32": ["rand32b_s0", "rand32b_s1", "real_teacher", "null_teacher"],
    "rand32b": ["rand32_s0", "rand32_s1", "real_teacher", "null_teacher"],
}


def students(kind: str, form: str, fam: str) -> dict[str, dict[str, Any]]:
    block = D["score"]["by_input"][kind][form]["real" if fam == "real" else "map"]["students"]
    return {k: v for k, v in block.items() if k in OWN_FROB[fam]}


def reads(kind: str, fam: str) -> dict[str, Any]:
    return D["reads"]["pile" if kind == "pile" else f"{kind}:{SET[fam]}"]


def rerun_range(whole: dict[str, float], fam: str) -> tuple[float, float]:
    vals = [whole[f"{a} | {b}"] for a in RUNS[fam] for b in RUNS[fam] if a != b]
    return min(vals), max(vals)


def rows() -> list[tuple[str, str, float]]:
    out, y = [], 0.0
    for fam in C:
        for form in FORMS:
            out.append((fam, form, y))
            y -= 0.5
        y -= 0.45
    return out


def frame(ax: Axes, title: str) -> None:
    ax.set_title(title, fontsize=10, color=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def rerun_mark(ax: Axes, lo: float, hi: float, mid: float, y: float) -> None:
    ax.plot([lo, hi], [y, y], color=INK, lw=1.1, zorder=2)
    ax.plot([mid, mid], [y - 0.17, y + 0.17], color=INK, lw=2.2, zorder=3)


def jitter(i: int, n: int) -> float:
    return 0.11 * (i - (n - 1) / 2) / max(n - 1, 1)


def cosine_panel(ax: Axes, kind: str) -> None:
    for fam, form, y in rows():
        block = students(kind, form, fam)
        first = next(iter(block.values()))
        whole = reads(kind, fam)[form]["whole"]
        lo, hi = rerun_range(whole, fam)
        rerun_mark(ax, lo, hi, first["rerun"], y)
        for i, s in enumerate(block.values()):
            dy = jitter(i, len(block))
            ax.plot(s["own"], y + dy, "o", color=C[fam], ms=6, zorder=4, mfc=C[fam] if form == "uncentred" else "white")
            ax.plot(s["other_task"], y + dy, "x", color=MUTED, ms=5, zorder=3)
            ax.plot(s["ceiling"], y + dy, "^", color=INK, mfc="white", ms=5, zorder=3)
        if fam == "rand32":
            ax.plot(D["score"]["length_reference"][kind][form]["long, 1000 steps"], y, "*", color=INK, ms=9, zorder=4)
    frame(ax, {"task": "Held-out task windows, every position", "pile": "Pile, every position"}[kind])


def frobenius_panel(ax: Axes) -> None:
    whole = D["reads"]["frobenius"]["uncentred"]["whole"]
    for fam, form, y in rows():
        if form != "uncentred":
            continue
        ts = sorted(set(OWN_FROB[fam].values()))
        lo, hi = rerun_range(whole, fam)
        vals = [whole[f"{a} | {b}"] for a in RUNS[fam] for b in RUNS[fam] if a != b]
        rerun_mark(ax, lo, hi, sum(vals) / len(vals), y)
        pairs = list(OWN_FROB[fam].items())
        for i, (s, t) in enumerate(pairs):
            ax.plot(whole[f"{s} | {t}"], y + jitter(i, len(pairs)), "o", color=C[fam], ms=6, zorder=4)
        for j, t in enumerate(ts):
            ax.plot(max(whole[f"{o} | {t}"] for o in OTHER_FROB[fam]), y + jitter(j, len(ts)), "x", color=MUTED, ms=5, zorder=3)
        for j, c in enumerate(CEILING_FROB[fam]):
            ax.plot(c, y + jitter(j, len(CEILING_FROB[fam])), "^", color=INK, mfc="white", ms=5, zorder=3)
        if fam == "rand32":
            ax.plot(whole["rand32_long_s0 | rand32_long_s1"], y, "*", color=INK, ms=9, zorder=4)
    frame(ax, "No weighting: every direction counts alike")


def y_labels(ax: Axes) -> None:
    ax.set_yticks([y for _, _, y in rows()])
    ax.set_yticklabels([f"{TABLES[fam]}, {FORMS[form]}" for fam, form, _ in rows()], fontsize=8.5)


def legend(fig: Any, handles: list[Line2D], ncol: int) -> None:
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.13))


def cosine_fig() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharey=True, sharex=True)
    frobenius_panel(axes[0])
    cosine_panel(axes[1], "task")
    cosine_panel(axes[2], "pile")
    y_labels(axes[0])
    for ax in axes:
        ax.set_xlabel("cosine with its own fine-tune's update")
        ax.set_xlim(0, 1.02)
    fig.suptitle("Each student against a rerun of its own fine-tune", fontsize=11.5, y=1.0)
    legend(fig, [
        Line2D([], [], marker="o", color=MUTED, ls="", label="student, filled as seen, open with the mean removed"),
        Line2D([], [], marker="|", color=INK, ms=11, mew=2.2, label="rerun line: mean over rerun pairs, bar their range"),
        Line2D([], [], marker="x", color=MUTED, ls="", label="largest other task's fine-tune"),
        Line2D([], [], marker="^", color=INK, mfc="white", ls="", label="ceiling: best first-order change"),
        Line2D([], [], marker="*", color=INK, ls="", ms=9, label="1,000-step rerun pair, first table, read by no rule"),
    ], 3)
    fig.savefig(FIGURES / "amend10_cosine.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def distance_panel(ax: Axes, kind: str) -> None:
    for fam, form, y in rows():
        block = students(kind, form, fam)
        first = next(iter(block.values()))
        lo, hi = rerun_range(reads(kind, fam)[form]["dist_whole"], fam)
        rerun_mark(ax, lo, hi, first["rerun_dist"], y)
        for i, s in enumerate(block.values()):
            dy = jitter(i, len(block))
            ax.plot(s["dist"], y + dy, "o", color=C[fam], ms=6, zorder=4, mfc=C[fam] if form == "uncentred" else "white")
            ax.plot(s["size"], y + dy, "s", color=C[fam], ms=4, zorder=3, mfc="none", alpha=0.8)
            ax.plot((1 - s["ceiling"] ** 2) ** 0.5, y + dy, "^", color=INK, mfc="white", ms=5, zorder=3)
    ax.axvline(1, color=MUTED, ls="--", lw=1.1, zorder=1)
    ax.text(1.02, -0.97, "no change: distance 1;\na rerun's size: about 1", fontsize=7.5, color=MUTED, va="center")
    frame(ax, {"task": "Held-out task windows, every position", "pile": "Pile, every position"}[kind])
    ax.set_xlabel("in lengths of the fine-tune's update")
    ax.set_xlim(0, 2.2)


def distance_fig() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True, sharex=True)
    distance_panel(axes[0], "task")
    distance_panel(axes[1], "pile")
    y_labels(axes[0])
    fig.suptitle("How far is each student from its fine-tune? Further than a rerun, and mostly further than no change", fontsize=11, y=1.0)
    legend(fig, [
        Line2D([], [], marker="o", color=MUTED, ls="", label="student's distance, filled as seen, open with the mean removed"),
        Line2D([], [], marker="s", color=MUTED, mfc="none", ls="", ms=4, label="student's size"),
        Line2D([], [], marker="|", color=INK, ms=11, mew=2.2, label="rerun's distance: mean over rerun pairs, bar their range"),
        Line2D([], [], marker="^", color=INK, mfc="white", ls="", label="nearest first-order change"),
    ], 2)
    fig.savefig(FIGURES / "amend10_distance.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    cosine_fig()
    distance_fig()
