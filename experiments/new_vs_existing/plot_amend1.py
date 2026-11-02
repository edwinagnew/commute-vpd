"""Figures for E12 Amendment 1: the route split on teacher training's own states, against the least-squares split.

  amend1_separation.png   the masked-off share per fine-tune: least-squares split, student's own state, plus-remainder state
  amend1_where_gain.png   where each state's gain sits: least-squares split beside the student, stacked shares
  amend1_move.png         how far the student's re-switching term moved, against the size of its teacher's update
  amend1_reach.png        the share of the re-switching term a student loses, against how far it moved beside its teacher
  amend1_behaviour.png    the student's held-out P(target) and off-target KL against the fine-tuned model's

Reads results/new_vs_existing_amend1.json and the committed E42 files; no model, no new runs.
Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/new_vs_existing/plot_amend1.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
from plot import COLOUR, FIGURES, RESULTS

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COPY, MAP = COLOUR["existing"], COLOUR["new"]
J = json.loads((RESULTS / "new_vs_existing_amend1.json").read_text())
E42 = RESULTS.parents[1] / "teacher_training" / "results"
NAMES = [f"{t}_s{s}" for t in ("ident32", "ident32b", "rand32", "rand32b") for s in (0, 1)]
TABLE = {"ident32": "copy, table 1", "ident32b": "copy, table 2", "rand32": "map, table 1", "rand32b": "map, table 2"}
LABEL = {n: f"{TABLE[n.split('_')[0]]}, seed {n[-1]}" for n in NAMES}
# The teachers' size (relative norm, largest matrix) and held-out off-target KL at the scored step: the checkpoint lines of E46's
# gitignored regating_table/results/large/full.log, lines 30-67; E12's table agrees within 0.002 on the size.
SIZE = dict(zip(NAMES, (0.0137, 0.0139, 0.0139, 0.0141, 0.0151, 0.0153, 0.0254, 0.0260), strict=True))
KL = dict(zip(NAMES, (0.1890, 0.1812, 0.2055, 0.1960, 0.6957, 0.6532, 0.0970, 0.1043), strict=True))


def colour(n: str) -> str:
    return COPY if n.startswith("ident") else MAP


def marker(n: str) -> str:
    return "o" if n.startswith("ident") else "s"


def style(ax: Any, axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print(FIGURES / name)


def separation() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for x, n in enumerate(NAMES):
        ls, a, b = J["c0"][n]["least_squares"]["shares"]["e_gate"], J["students"][n]["a"]["shares"]["e_gate"], J["students"][n]["b"]["shares"]["e_gate"]
        ax.plot([x, x], [ls, a], color=GRID, lw=2, zorder=1)
        ax.scatter(x, ls, s=70, facecolor="white", edgecolor=colour(n), linewidth=1.8, marker=marker(n), zorder=2, label="least-squares split (reference)" if x == 0 else None)
        ax.scatter(x, a, s=70, color=colour(n), marker=marker(n), zorder=3, label="student's own state, scored" if x == 0 else None)
        ax.scatter(x + 0.22, b, s=45, color=MUTED, marker="x", zorder=3, label="student plus remainder, read beside" if x == 0 else None)
    for y, text in ((0.5, "at or over 0.5: filed as masked-off"), (0.25, "at or under 0.25: filed as not masked-off")):
        ax.axhline(y, color=MUTED, lw=0.9, ls="--")
        ax.text(7.45, y + 0.012, text, ha="right", color=MUTED, fontsize=8.5)
    floor = J["c1"]["end"]["rand32"]["e_gate"] / min(J["c0"][n]["nothing"]["gain"] for n in NAMES)
    ax.axhline(floor, color=INK, lw=0.9, ls=":")
    ax.text(7.45, floor + 0.012, f"training with nothing to learn: {floor:.3f} of the smallest fine-tune's gain", ha="right", color=INK, fontsize=8.5)
    ax.set_xticks(range(8), [LABEL[n] for n in NAMES], rotation=20, ha="right")
    ax.set_xlim(-0.5, 7.6)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("masked-off share of the state's gain")
    ax.set_title("Does the masked-off share still part the copy class from the random-map class on the student's own state?", fontsize=10.5, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    style(ax)
    save(fig, "amend1_separation.png")


def where_gain() -> None:
    parts = (("re-switching", "#2a9d8f"), ("masked-off", "#6a4c93"), ("left in Δ", "#a8a79f"), ("direct and interaction", "#e0a800"))

    def split(s: dict[str, float]) -> list[float]:
        return [s["r"], s["e_gate"], s["e_delta"], s["d"] + s["i"]]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ticks, labels = [], []
    for i, n in enumerate(NAMES):
        x = i * 2.5 + (0.5 if i >= 4 else 0)
        for dx, s, tag in ((0, J["c0"][n]["least_squares"]["shares"], "LS"), (1, J["students"][n]["a"]["shares"], "T")):
            bottom = 0.0
            for (name, c), v in zip(parts, split(s), strict=True):
                ax.bar(x + dx, v, bottom=bottom, width=0.9, color=c, edgecolor="white", linewidth=1, label=name if (i == 0 and dx == 0) else None)
                bottom += v
            ticks.append(x + dx)
            labels.append(tag)
        ax.text(x + 0.5, 1.03, LABEL[n].replace(", seed ", " s").replace("table ", "t").replace(", ", " "), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(ticks, labels, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("share of the state's gain at the trigger")
    ax.set_xlabel("LS = least-squares split (reference)     T = student of teacher training, Δ fixed")
    ax.set_title("Where does each state's gain sit, and where did what least-squares left in Δ go?", fontsize=10.5, loc="left")
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    style(ax)
    save(fig, "amend1_where_gain.png")


def e42_moves() -> list[tuple[str, float, float, str]]:
    out = []
    for task, tag, c in (("ident32", "E42 copy, step-1000 teacher", COPY), ("rand32", "E42 map, step-1000 teacher", MAP), ("rand1", "E42 one-pair map, step-1000 teacher", MAP)):
        e = json.loads((E42 / f"teacher_training_e42_{task}.json").read_text())
        no, st = e["nothing"], e["A"]["1000"]
        out.append((tag, e["teacher"]["size"], abs(st["share_r"] * st["gain"] - no["share_r"] * no["gain"]) / no["gain"], c))
    return out


def move() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5))
    moves = J["score"]["P2"]["moves"]
    for n in NAMES:
        ax.scatter(SIZE[n], moves[n], s=70, color=colour(n), marker=marker(n), zorder=3)
    ax.scatter([], [], s=70, color=COPY, marker="o", label="copy, this run")
    ax.scatter([], [], s=70, color=MAP, marker="s", label="random map, this run")
    for tag, size, m, c in e42_moves():
        ax.scatter(size, m, s=70, facecolor="white", edgecolor=c, linewidth=1.8, marker="o" if c == COPY else "s", zorder=3)
        ax.annotate(tag, (size, m), xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=8.5, color=MUTED)
    gain = min(J["c0"][n]["nothing"]["gain"] for n in NAMES)
    ax.axhline(0.25, color=MUTED, ls="--", lw=0.9)
    ax.text(0.0395, 0.26, "filed line: moved, 0.25", ha="right", fontsize=8.5, color=MUTED)
    ax.axhline(0.10, color=MUTED, ls="--", lw=0.9)
    ax.text(0.0395, 0.104, "filed line: inherited, 0.10", ha="right", fontsize=8.5, color=MUTED)
    train = J["c1"]["end"]["rand32"]["r"] / gain
    ax.axhline(train, color=INK, ls=":", lw=0.9)
    ax.text(0.0395, train * 1.05, f"training with nothing to learn: {train:.4f} of the smallest gain", ha="right", fontsize=8.5, color=INK)
    noise = max(J["score"]["P2"]["noise_r"].values())
    ax.axhline(noise, color=INK, ls="-", lw=0.7)
    ax.text(0.0395, noise * 1.05, f"second batch order: {noise:.4f}", ha="right", fontsize=8.5, color=INK)
    ax.text(0.0395, 0.00046, "least-squares split: exactly 0, off this log axis", ha="right", fontsize=8.5, color=MUTED)
    ax.annotate("map, table 2, both seeds", (SIZE["rand32b_s0"], moves["rand32b_s0"]), xytext=(-10, 0), textcoords="offset points", ha="right", va="center", fontsize=8.5, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(0.0004, 0.4)
    ax.set_xlim(0.012, 0.04)
    ax.set_xlabel("size of the teacher's update: relative norm of its most changed matrix")
    ax.set_ylabel("change in the re-switching term / fine-tune's gain")
    ax.set_title("How far does a student's re-switching term move, and does it track the teacher's update size?", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    save(fig, "amend1_move.png")


def reach() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for n in NAMES:
        s, z = J["students"][n]["a"]["cells"], J["c0"][n]["nothing"]["cells"]
        ratio = next(c for c in J["students"][n]["curve"] if c["step"] == 1000)["rel_norm_max"] / SIZE[n]
        ax.scatter(ratio, 1 - (s["L01"] - s["L00"]) / (z["L01"] - z["L00"]), s=70, color=colour(n), marker=marker(n), zorder=3)
    ax.scatter([], [], s=70, color=COPY, marker="o", label="copy, this run")
    ax.scatter([], [], s=70, color=MAP, marker="s", label="random map, this run")
    ax.scatter([], [], s=70, facecolor="white", edgecolor=MUTED, linewidth=1.8, marker="s", label="hollow: E42's step-1000 teachers")
    for task, tag, c, va in (("ident32", "E42 copy", COPY, "center"), ("rand32", "E42 map", MAP, "center"), ("rand1", "E42 one-pair map", MAP, "center")):
        e = json.loads((E42 / f"teacher_training_e42_{task}.json").read_text())
        no, st = e["nothing"], e["A"]["1000"]
        x, y = st["size"] / e["teacher"]["size"], 1 - st["share_r"] * st["gain"] / (no["share_r"] * no["gain"])
        ax.scatter(x, y, s=70, facecolor="white", edgecolor=c, linewidth=1.8, marker="o" if c == COPY else "s", zorder=3)
        ax.annotate(tag, (x, y), xytext=(10, 0), textcoords="offset points", va=va, fontsize=8.5, color=MUTED)
    ax.annotate("map, table 2, both seeds", (0.865, 0.19), xytext=(-10, 0), textcoords="offset points", ha="right", va="center", fontsize=8.5, color=MUTED)
    ax.axhline(0, color=INK, lw=0.9)
    ax.text(0.47, -0.032, "0: the least-squares split and do-nothing, by construction; a second batch order moves it under 0.003", fontsize=8.5, color=INK)
    ax.axvline(1, color=MUTED, lw=0.9, ls="--")
    ax.text(1.01, 0.5, "student larger than its teacher (most changed matrix)", fontsize=8.5, color=MUTED)
    ax.set_xlim(0.45, 1.45)
    ax.set_ylim(-0.05, 0.7)
    ax.set_xlabel("student's update size / teacher's, relative norm of the most changed matrix in each")
    ax.set_ylabel("share of the fine-tune's re-switching term lost")
    ax.set_title("Share of re-switching lost against student/teacher size on the most changed matrix: a correlation on four map settings", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style(ax)
    save(fig, "amend1_reach.png")


def behaviour() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    last = {n: next(c for c in J["students"][n]["curve"] if c["step"] == 1000) for n in NAMES}
    for ax, xs, key, name, top in ((axes[0], {n: J["c0"][n]["nothing"]["p_cells"]["T1"] for n in NAMES}, "heldout_p_o", "held-out P(target)", 1.03),
                                   (axes[1], KL, "heldout_kl_other", "held-out off-target KL, nats", 0.78)):
        ax.plot([0, top], [0, top], color=MUTED, lw=0.9, ls="--", label="student equals the fine-tuned model")
        for n in NAMES:
            ax.scatter(xs[n], last[n][key], s=60, color=colour(n), marker=marker(n), zorder=3)
        ax.scatter([], [], s=60, color=COPY, marker="o", label="copy")
        ax.scatter([], [], s=60, color=MAP, marker="s", label="random map")
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_xlabel(f"fine-tuned model: {name}")
        ax.set_ylabel(f"student at step 1000: {name}")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        style(ax, "both")
    fig.suptitle("Does the student behave like its fine-tuned model on held-out windows?", fontsize=10.5, x=0.01, ha="left")
    save(fig, "amend1_behaviour.png")


if __name__ == "__main__":
    separation()
    where_gain()
    move()
    reach()
    behaviour()
