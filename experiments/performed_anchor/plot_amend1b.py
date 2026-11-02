"""Figures for E53 Amendment 1b (the broad cipher alone), read from results/read_amend1b.json and, for E53's cipher at
steps 10 and 100, results/read.json; nothing is recomputed from the model.
Plan: reports/plans/2026-10-29_plan_performed-anchor.md, section "Amendment 1b"; analysis:
reports/analyses/2026-10-29_analysis_e53-performed-anchor.md, section "Amendment 1b".

  a1b_breadth.png  the gate: breadth (G per true off-target KL; lower = broader) of E53's lowercase and cipher at step 300
                   and of the broad cipher at steps 10, 100, 300, with the gate line at lowercase's lower seed
  a1b_ratio.png    the stand-in ratio against breadth, same rows, with E53's filed leans 0.5 and 0.7
  a1b_gain.png     the stand-in ratio against the full model's gain G: E53's cipher and the broad cipher at steps 10,
                   100, 300, E53's lowercase at step 300, and the broad cipher's gain floor

Usage (repo root): ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/performed_anchor/plot_amend1b.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results" / "read_amend1b.json").read_text())
E53 = json.loads((HERE / "results" / "read.json").read_text())["rows"]
ROWS, S = R["rows"], R["score"]
FIG = HERE / "figures"
LOW_C, CIP_C, BROAD_C, INK, MUTED = "#2a78d6", "#eb6834", "#7a4fc9", "#0b0b0b", "#8a8984"
MARK = {0: "o", 1: "s"}
STEPS = (10, 100, 300)


def ratio(r: dict) -> float:
    return r["masked_gain"] / r["piece_gain"]


def breadth(r: dict) -> float:
    return r["G"] / r["kl_off"]


def finish(fig: Figure, ax: Axes, path: str, title: str) -> None:
    ax.set_title(title, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / path, dpi=130)
    plt.close(fig)


def leans(ax: Axes, x: float) -> None:
    for y, lab in ((0.7, "E53's lean for a performed task: at least 0.7"), (0.5, "E53's lean for a new task: at most 0.5")):
        ax.axhline(y, color=INK, ls="--", lw=0.8)
        ax.text(x, y + 0.015, lab, fontsize=8, ha="left")


def fig_breadth() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    rows = [("E53 lowercase, step 300", "e53/lowercase/s{s}/step300", LOW_C)]
    rows += [(f"broad cipher, step {n}", f"broad_cipher/s{{s}}/step{n}", BROAD_C) for n in STEPS]
    rows += [("E53 cipher, step 300", "e53/cipher/s{s}/step300", CIP_C)]
    for i, (lab, key, c) in enumerate(rows):
        for s in (0, 1):
            ax.plot(breadth(ROWS[key.format(s=s)]), i, MARK[s], color=c, ms=8, label=f"seed {s}" if i == 0 else None)
    gate = S["gate"]["e53_lowercase"]
    ax.axvline(min(gate), color=LOW_C, ls=":", lw=1.2)
    ax.text(min(gate) - 0.2, len(rows) - 0.6, f"gate: at most {min(gate):.2f},\nlowercase's lower seed", fontsize=8,
            color=LOW_C, ha="right", va="top")
    ax.text(16.2, -0.42, "lowercase's number counts some lowered words as off-target,\nso it reads broader than it is",
            fontsize=7.5, color=MUTED, ha="right", va="bottom")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 16.5)
    ax.set_xlabel("breadth: full-model gain per nat of off-target KL (lower = broader)")
    ax.legend(fontsize=8, frameon=False, loc="center right")
    finish(fig, ax, "a1b_breadth.png", "Is the broad cipher at least as broad as E53's lowercase?")


def fig_ratio() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    leans(ax, 0.3)
    for task, c in (("lowercase", LOW_C), ("cipher", CIP_C)):
        for s in (0, 1):
            r = ROWS[f"e53/{task}/s{s}/step300"]
            ax.plot(breadth(r), ratio(r), MARK[s], color=c, ms=9, label=f"E53 {task}, step 300, seed {s}")
    for s in (0, 1):
        rs = [ROWS[f"broad_cipher/s{s}/step{n}"] for n in STEPS]
        ax.plot([breadth(r) for r in rs], [ratio(r) for r in rs], "-" + MARK[s], color=BROAD_C, ms=8,
                label=f"broad cipher, seed {s}, steps 10, 100, 300")
        for r, n in zip(rs, STEPS, strict=True):
            ax.annotate(f"step {n}", (breadth(r), ratio(r)), textcoords="offset points", xytext=(5, 6 if s else -11),
                        fontsize=7)
    ax.text(0.3, 0.04, "the broad cipher's random δW: subcomponent gain 0.009 nats, so its ratio is unread",
            fontsize=8, color=MUTED)
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("breadth: full-model gain per nat of off-target KL (lower = broader)")
    ax.set_ylabel("stand-in ratio: masked gain / subcomponent gain")
    ax.legend(fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    finish(fig, ax, "a1b_ratio.png", "Does a cipher twice as broad read higher? (provisional, not scored)")


def fig_gain() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    leans(ax, 0.6)
    for s in (0, 1):
        rs = [E53[f"cipher/s{s}/step{n}"] for n in STEPS]
        ax.plot([r["G"] for r in rs], [ratio(r) for r in rs], "-" + MARK[s], color=CIP_C, ms=7,
                label=f"E53 cipher, seed {s}, steps 10, 100, 300")
        rs = [ROWS[f"broad_cipher/s{s}/step{n}"] for n in STEPS]
        ax.plot([r["G"] for r in rs], [ratio(r) for r in rs], "-" + MARK[s], color=BROAD_C, ms=8,
                label=f"broad cipher, seed {s}, steps 10, 100, 300")
        for r, n in zip(rs, STEPS, strict=True):
            ax.annotate(f"step {n}", (r["G"], ratio(r)), textcoords="offset points", xytext=(5, 6 if s else -11),
                        fontsize=7)
        r = ROWS[f"e53/lowercase/s{s}/step300"]
        ax.plot(r["G"], ratio(r), MARK[s], color=LOW_C, ms=9, label=f"E53 lowercase, seed {s}, step 300")
    f = S["floor"]["broad_cipher/s0/step300"]
    ax.axvline(f, color=BROAD_C, ls=":", lw=1)
    ax.text(f + 0.04, 0.08, f"broad cipher's gain floor {f:.2f}\n(twice its random δW's gain)", fontsize=8, color=BROAD_C)
    ax.set_xlim(0, 4.2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("full-model gain G at the scored positions, nats of log-odds (0 = the original model)")
    ax.set_ylabel("stand-in ratio")
    ax.legend(fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    finish(fig, ax, "a1b_gain.png", "At lowercase's gain, where does the broad cipher read? (provisional)")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig_breadth()
    fig_ratio()
    fig_gain()


if __name__ == "__main__":
    main()
