"""The research log's figures for E50, redrawn from saved results with plain labels (no model, no new runs).

  log_relay.png   share of the gain taken back when every alive subcomponent is restored at the key position, for
                  the copy fine-tunes, the random-map fine-tunes and the four hand writes (the low-reading control);
                  per seed, the gap between the lowest copy and the highest random map, against the 0.3 bar.
                  Reads results/restore_anchors.json (fine-tunes) and results/amend1.json (hand writes).
  log_sets.png    the same share for the smaller sets: Γ's switched-on set S, the set ranked by pull on the answer
                  (D) and the largest of 20 random sets of S's size, copy beside random map, with D's and S's
                  per-seed gaps. Reads results/restore_anchors.json.
  log_s_size.png  S's size against the nats S takes back, one dot per fine-tune. Reads results/restore_anchors.json.
  log_probe.png   the answer's log-odds at the key position, per table: the original model, the original model shown
                  a demonstration of the pair, and the fine-tuned model (mean of two seeds). Reads results/amend1.json
                  (probe) and results/restore_anchors.json (T1).

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/restore_anchors/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from plot import CLS, COPY_C, HERE, INK, MARK, MUTED, RAND_C, R

FIG = HERE / "figures"
A: dict[str, Any] = json.loads((HERE / "results" / "amend1.json").read_text())
TABLES = ["ident32", "ident32b", "rand32", "rand32b"]
NAME = {"ident32": "copy A", "ident32b": "copy B", "rand32": "random map A", "rand32b": "random map B"}
WRITE_C = "#2e8b57"
SEEDS = ("s0", "s1")


def style(ax: Axes) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig: Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def rho(table: str, seed: str, X: str) -> float:
    st = R["states"][f"{table}/{seed}"]
    return st["random_max"] if X == "random_max" else st["rho"][X]


def gap(X: str, seed: str) -> float:
    """The filed worst-pair gap: the lowest copy minus the highest random map, on one seed."""
    copy = [rho(t, seed, X) for t in TABLES if CLS[t] == "copy"]
    rand = [rho(t, seed, X) for t in TABLES if CLS[t] == "random"]
    return min(copy) - max(rand)


def dots(ax: Axes, x: float, X: str, cls: str, col: str) -> None:
    ts = [t for t in TABLES if CLS[t] == cls]
    for i, (t, s) in enumerate((t, s) for t in ts for s in SEEDS):
        ax.scatter(x + (i - 1.5) * 0.06, rho(t, s, X), marker=MARK[t], s=40, color=col, edgecolor=INK, lw=0.5, zorder=3)


def fig_relay() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    dots(ax, 0, "every_alive", "copy", COPY_C)
    dots(ax, 1, "every_alive", "random", RAND_C)
    for i, t in enumerate(TABLES):
        ax.scatter(2 + (i - 1.5) * 0.06, A["rows"][f"{t}/write"]["rho"]["every_alive"], marker=MARK[t], s=40,
                   color=WRITE_C, edgecolor=INK, lw=0.5, zorder=3)
    g0, g1 = gap("every_alive", "s0"), gap("every_alive", "s1")
    ax.text(0.5, 0.86, f"gap between the kinds: {g0:.2f} (seed 0), {g1:.2f} (seed 1)\nto separate them it had to be 0.3",
            ha="center", va="bottom", fontsize=8.5)
    ax.axhline(0, color=MUTED, lw=1)
    ax.text(2.45, 0.02, "0: nothing taken back", ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.axhline(1, color=INK, lw=1, ls="--")
    ax.text(2.45, 1.02, "1: all of the gain taken back", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(range(3), ["copy fine-tunes\n(repeat the key)", "random-map fine-tunes\n(a fixed word per key)",
                             "hand writes\n(answer added directly)"], fontsize=9)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.45, 1.15)
    ax.set_ylabel("share of the gain taken back when every\nalive subcomponent is restored")
    ax.set_title("Both kinds of fine-tune pass about 0.7 of their gain through the original weights;\n"
                 "a direct write does not, so the number can read low", fontsize=10)
    ax.legend(handles=[Line2D([], [], marker=m, ls="", color="white", mec=INK, label=lab)
                       for m, lab in (("o", "table A"), ("s", "table B"))], fontsize=8, frameon=False, loc="lower left")
    style(ax)
    save(fig, "log_relay.png")


def fig_sets() -> None:
    sets = (("S", "S: the pieces Γ\nswitches on"), ("D", "D: ranked by pull on\nthe answer (no Γ)"),
            ("random_max", "random sets of S's size\n(largest of 20)"))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for i, (X, _) in enumerate(sets):
        dots(ax, i - 0.2, X, "copy", COPY_C)
        dots(ax, i + 0.2, X, "random", RAND_C)
        if X != "random_max":
            g = [gap(X, s) for s in SEEDS]
            ax.text(i, 0.53, f"gap {g[0]:.2f}, {g[1]:.2f}", ha="center", fontsize=8.5)
    ax.text(1, 0.585, "gap: lowest copy minus highest random map, seed 0 and seed 1 (separating needs 0.3)",
            ha="center", fontsize=8, color=MUTED)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(sets)), [lab for _, lab in sets], fontsize=9)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.03, 0.63)
    ax.set_ylabel("share of the gain taken back\nwhen the set is restored")
    ax.set_title("Γ's set reads the two kinds alike; the pull-ranked set puts copies higher,\n"
                 "by less than the 0.3 bar", fontsize=10)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COPY_C, mec=INK, label="copy fine-tunes"),
                       Line2D([], [], marker="o", ls="", color=RAND_C, mec=INK, label="random-map fine-tunes"),
                       Line2D([], [], marker="s", ls="", color="white", mec=INK, label="circle table A, square table B")],
              fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.9))
    style(ax)
    save(fig, "log_sets.png")


def fig_s_size() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for key, st in R["states"].items():
        t = key.split("/")[0]
        n = sum(len(v) for v in st["members"]["S"].values())
        col = COPY_C if CLS[t] == "copy" else RAND_C
        ax.scatter(n, st["nats"]["S"], marker=MARK[t], s=45, color=col, edgecolor=INK, lw=0.5, zorder=3)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 3.5)
    ax.set_xlabel("size of S: subcomponents Γ switches on at the key position")
    ax.set_ylabel("nats of the gain S takes back")
    ax.set_title("Γ switches on 108 to 118 pieces for the copies and 24 to 49 for the random maps,\n"
                 "yet S takes back 2.0 to 2.7 nats on every fine-tune", fontsize=10)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COPY_C, mec=INK, label="copy fine-tunes"),
                       Line2D([], [], marker="o", ls="", color=RAND_C, mec=INK, label="random-map fine-tunes"),
                       Line2D([], [], marker="s", ls="", color="white", mec=INK, label="circle table A, square table B")],
              fontsize=8, frameon=False, loc="lower right")
    style(ax)
    save(fig, "log_s_size.png")


def fig_probe() -> None:
    bars = (("the original model", MUTED, lambda t: A["probe"][t]["T0"]),
            ("the original model, shown the key and its answer ten tokens earlier", "#c9a13b",
             lambda t: A["probe"][t]["lo_demo"]),
            ("the fine-tuned model (mean of two seeds)", INK,
             lambda t: sum(R["states"][f"{t}/{s}"]["T1"] for s in SEEDS) / 2))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    w = 0.26
    for i, t in enumerate(TABLES):
        for b, (lab, col, f) in enumerate(bars):
            v = f(t)
            ax.bar(i + (b - 1) * w, v, w * 0.92, color=col, label=lab if i == 0 else None)
            ax.text(i + (b - 1) * w, v + (0.3 if v >= 0 else -0.3), f"{v:.1f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(range(len(TABLES)), [NAME[t] for t in TABLES], fontsize=9)
    ax.set_ylim(-17, 13)
    ax.set_ylabel("log-odds of the table's answer at the key\nposition, nats (0: as likely as not)")
    ax.set_title("Unprompted, the original model gives neither answer; shown the pair, every answer rises\nbut stays below even odds", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    save(fig, "log_probe.png")


if __name__ == "__main__":
    fig_relay()
    fig_sets()
    fig_s_size()
    fig_probe()
