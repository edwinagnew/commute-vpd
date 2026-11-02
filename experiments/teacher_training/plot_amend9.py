"""Figures for teacher_training Amendment 9, a student's weight change against the part of its fine-tune's update the
subcomponents could write (plan: reports/plans/2026-11-02_plan_e34-amend9-student-move.md), read from results/amend9.json only.

  amend9_share.png   the update's inside share, a random update's, and the most a 'mostly along' reading could allow
  amend9_cosine.png  each student's cosine to its own fine-tune's inside part and to another's, with the controls
  amend9_reach.png   each student's cosine against its size, with the line where its length along the inside part equals the part's

Usage (repo root): ~/commute-agents/fleet/cpu-run python experiments/teacher_training/plot_amend9.py
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

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
D: dict[str, Any] = json.loads((HERE / "results" / "amend9.json").read_text())
INK, MUTED, GRID, REF = "#0b0b0b", "#6e6d68", "#e6e5e0", "#52514e"
C = {"real": "#2a78d6", "null": "#eb6834", "rand32": "#1baf7a", "rand32b": "#6250d6"}  # real and null as in amend8's figures
MARK = {"real": "o", "null": "s", "rand32": "^", "rand32b": "D"}
FAMILY = {"real": "real task, emoticon fine-tune", "null": "null fine-tune, no task", "rand32": "first random-map table", "rand32b": "second random-map table"}
# student -> (family, label, own fine-tune, the other fine-tune read)
STUDENTS = {
    "real_o0": ("real", "real task, order 0", "real_teacher", "null_teacher"),
    "real_o1": ("real", "real task, order 1", "real_teacher", "null_teacher"),
    "null_o0": ("null", "null fine-tune, order 0", "null_teacher", "real_teacher"),
    "null_o1": ("null", "null fine-tune, order 1", "null_teacher", "real_teacher"),
    "rand32_s0": ("rand32", "first table, seed 0", "rand32_s0", "rand32b_s0"),
    "rand32_s1": ("rand32", "first table, seed 1", "rand32_s1", "rand32b_s1"),
    "rand32b_s0": ("rand32b", "second table, seed 0, order 0", "rand32b_s0", "rand32_s0"),
    "rand32b_s0_o1": ("rand32b", "second table, seed 0, order 1", "rand32b_s0", "rand32_s0"),
    "rand32b_s1": ("rand32b", "second table, seed 1, order 0", "rand32b_s1", "rand32_s1"),
    "rand32b_s1_o1": ("rand32b", "second table, seed 1, order 1", "rand32b_s1", "rand32_s1"),
}
TEACHERS = {"real_teacher": ("real", "real task"), "null_teacher": ("null", "null fine-tune"), "rand32_s0": ("rand32", "first table, seed 0"),
            "rand32_s1": ("rand32", "first table, seed 1"), "rand32b_s0": ("rand32b", "second table, seed 0"), "rand32b_s1": ("rand32b", "second table, seed 1")}


def read(student: str, teacher: str) -> dict[str, float]:
    return D["reads"][f"{student} | {teacher}"]


def tidy(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def share_fig() -> None:
    """The ceiling is the arithmetic filed before the run: 'mostly along' needs share <= 2 cos(s, dW)^2, largest over a fine-tune's students."""
    names = list(TEACHERS)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, t in enumerate(names):
        y = len(names) - 1 - i
        ceiling = max(2 * read(s, t)["saved_cos_all"] ** 2 for s, v in STUDENTS.items() if v[2] == t)
        share, rnd = D["controls"][t]["share"], D["controls"][t]["null_inside"]
        fam = TEACHERS[t][0]
        ax.plot([ceiling, share], [y, y], color=GRID, lw=2, zorder=1)
        ax.scatter(ceiling, y, marker="|", s=170, color=INK, linewidths=2, zorder=3, label="most a 'mostly along' reading could allow, known before the run" if i == 0 else None)
        ax.scatter(rnd, y, marker="o", s=60, facecolors="white", edgecolors=REF, linewidths=1.4, zorder=3, label="random update of the same size per matrix" if i == 0 else None)
        ax.scatter(share, y, marker=MARK[fam], s=60, color=C[fam], edgecolors="white", linewidths=0.8, zorder=4)
        ax.annotate(f"{share:.2f}", (share, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.scatter([], [], marker="o", s=60, color=REF, label="the fine-tune's update, coloured by family")
    ax.set_yticks(range(len(names)), [TEACHERS[t][1] for t in names][::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the update's squared weight length the subcomponents could write, all 24 matrices")
    ax.set_title("How much of each update could the subcomponents write? (4 of 24 matrices saturated)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), frameon=False, fontsize=8.5, ncol=1)
    tidy(ax)
    save(fig, "amend9_share")


def cosine_fig() -> None:
    names = list(STUDENTS)
    low = [c["do_nothing"] for c in D["controls"].values()]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, s in enumerate(names):
        y = len(names) + 1 - i
        fam, _, own, other = STUDENTS[s]
        cos_own, cos_other = read(s, own)["cos_in"], read(s, other)["cos_in"]
        ax.scatter(cos_own, y, marker=MARK[fam], s=60, color=C[fam], edgecolors="white", linewidths=0.8, zorder=4)
        ax.scatter(cos_other, y, marker=MARK[fam], s=45, facecolors="white", edgecolors=C[fam], linewidths=1.3, zorder=4)
        ax.annotate(f"{cos_own:.3f}", (cos_own, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.plot([min(low), max(low)], [1, 1], color=REF, lw=3, solid_capstyle="round", zorder=4)
    ax.annotate(f"{min(low):.2f} to {max(low):.2f}", (max(low), 1), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.scatter(1, 0, marker="X", s=70, color=REF, zorder=4)
    ax.annotate("1", (1, 0), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(names) + 2), ["control: least-squares split, first order", "reference: do-nothing, the whole update"] + [STUDENTS[s][1] for s in names][::-1])
    ax.axvline(0.1, color=MUTED, ls=":", lw=1.2, label="filed line 0.1: under it, not along")
    ax.axvline(0.5**0.5, color=INK, ls="--", lw=1.2, label="filed line 0.71: above it, 'mostly along'")
    ax.scatter([], [], marker="o", s=60, color=REF, label="the student's own fine-tune")
    ax.scatter([], [], marker="o", s=45, facecolors="white", edgecolors=REF, linewidths=1.3, label="another fine-tune it never saw")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("cosine to the part of the update the subcomponents could write, every weight counted alike")
    ax.set_title("Does a student's weight change point along what the subcomponents could write?", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12), frameon=False, fontsize=8.5, ncol=2)
    tidy(ax)
    save(fig, "amend9_cosine")


def reach_fig() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.linspace(0.95, 3.6, 200)
    ax.plot(x, 1 / x, color=INK, ls="--", lw=1.2, zorder=2, label="reach 1: length along it equals its whole length (cosine x size = 1)")
    ax.axhline(0.5**0.5, color=MUTED, ls=":", lw=1.2, zorder=2, label="filed line 0.71: above it, 'mostly along'")
    for fam, colour in C.items():
        pts = [(read(s, v[2])["size_to_inside"], read(s, v[2])["cos_in"]) for s, v in STUDENTS.items() if v[0] == fam]
        ax.scatter(*zip(*pts, strict=True), marker=MARK[fam], s=60, color=colour, edgecolors="white", linewidths=0.8, zorder=4, label=FAMILY[fam])
    ctrl = D["controls"]
    dn = [(1 / c["do_nothing"], c["do_nothing"]) for c in ctrl.values()]
    ax.scatter(*zip(*dn, strict=True), marker="P", s=55, color=REF, zorder=3, label="reference: do-nothing, the whole update, six fine-tunes")
    ls = [(c["ls_whole_size"], c["ls_whole"]) for c in ctrl.values()]
    ax.scatter(*zip(*ls, strict=True), marker="X", s=55, color=REF, zorder=3, label="control: least-squares split, whole state, six fine-tunes")
    ax.set_xlim(0, 3.6)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("student's size against the part of the update the subcomponents could write (1: same length)")
    ax.set_ylabel("cosine to that part")
    ax.set_title("How far along that part does each student reach, and at what size?\n(one length along the part, not containing it)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8.5, ncol=2)
    tidy(ax)
    save(fig, "amend9_reach")


if __name__ == "__main__":
    share_fig()
    cosine_fig()
    reach_fig()
