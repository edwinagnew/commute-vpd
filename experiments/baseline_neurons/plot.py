"""E52's figures, one per prediction, from results/baseline_neurons.json (no model). Copy states blue, cipher orange,
on the same axes; states read as 'no change' (|G| under the floor) drawn hollow.

  figures/p1_neuron_patch.png  B2: recovered share against the number of patched neurons k, per state; 0.5 marks k50
  figures/p2_idle.png          B2: idle share of the top 16 neurons, per state
  figures/p3_steering.png      B3: steering share S per state (the layer chosen on the fit half)
  figures/p4_prompting.png     B4: prompting share against the number of demonstrations n, per state
  figures/p5_null.png          |G| of all nine states against each one's floor (log scale)

Usage (repo root):  ~/commute-agents/fleet/cpu-run python3 experiments/baseline_neurons/plot.py [results.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
COLOR = {"copy": "#2a78d6", "cipher": "#eb6834", "null": "#8a8984"}
INK, GRID = "#52514e", "#e4e3df"
TASK_LABEL = {"ident32": "copy A", "ident32b": "copy B", "rand32": "cipher A", "rand32b": "cipher B"}


def kind(n: str) -> str:
    return "copy" if n.startswith("ident") else "cipher" if n.startswith("rand") else "null"


def label(n: str) -> str:
    t, s = n.rsplit("_", 1)
    return f"{TASK_LABEL[t]} {s}" if t in TASK_LABEL else "code text"


def style(ax: Any, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def strip(R: dict[str, Any], vals: dict[str, float], title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    names = list(vals)
    for x, n in enumerate(names):
        hollow = R["gains"][n]["no_change"]
        ax.scatter(x, vals[n], s=60, zorder=3, edgecolor=COLOR[kind(n)], lw=1.5,
                   facecolor="white" if hollow else COLOR[kind(n)])
    ax.set_xticks(range(len(names)), [label(n) for n in names], rotation=45, ha="right", fontsize=8)
    style(ax, title, ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def curves(R: dict[str, Any], xs: list[Any], ys: dict[str, list[float]], title: str, xlabel: str, path: Path,
           logx: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for n, y in ys.items():
        ax.plot(xs, y, marker="o", ms=4, lw=1.2, color=COLOR[kind(n)], alpha=0.85,
                ls=":" if R["gains"][n]["no_change"] else "-", label=label(n))
    ax.axhline(1, color=INK, lw=0.6, ls="--")
    ax.axhline(0, color=INK, lw=0.6)
    if logx:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel, color=INK)
    style(ax, title, "share of the fine-tune's gain")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def null_fig(R: dict[str, Any], path: Path) -> None:
    rows = {n: (abs(R["gains"][n]["G"]), R["gains"][n]["floor"]) for n in R["gains"]}
    c = R["code_step10"]
    rows["code_step10"] = (abs(c["G"]), c["floor"])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for x, (n, (g, f)) in enumerate(rows.items()):
        ax.scatter(x, g, s=60, zorder=3, color=COLOR[kind(n)])
        ax.plot([x - 0.3, x + 0.3], [f, f], color=INK, lw=1.2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows)), [label(n) for n in rows], rotation=45, ha="right", fontsize=8)
    style(ax, "P5: |gain| of each state (dot) against its floor (bar)", "|G| (nats)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    R = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "results" / "baseline_neurons.json").read_text())
    FIG.mkdir(exist_ok=True)
    states = list(R["b2"])
    grid = [int(k) for k in R["b2"][states[0]]["shares"]]
    curves(R, grid, {n: [R["b2"][n]["shares"][str(k)] for k in grid] for n in states},
           "P1: patching the top-k neurons (k50 = first k at share 0.5)", "neurons patched, k", FIG / "p1_neuron_patch.png", logx=True)
    strip(R, {n: R["b2"][n]["idle_share"] for n in states}, "P2: top-16 neurons idle at W0", "idle share",
          FIG / "p2_idle.png")
    strip(R, {n: R["b3"][n]["S"] for n in states}, "P3: steering share S (layer chosen on the fit half)", "S",
          FIG / "p3_steering.png")
    ns = sorted(int(k) for k in R["b4"]["shares"][states[0]])
    curves(R, ns, {n: [R["b4"]["shares"][n][str(k)] for k in ns] for n in states},
           "P4: prompting the original model with n demonstrations", "demonstrations, n", FIG / "p4_prompting.png")
    null_fig(R, FIG / "p5_null.png")


if __name__ == "__main__":
    main()
