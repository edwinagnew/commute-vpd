"""E52's figures, one per prediction, from results/baseline_neurons.json (no model). Copy states blue, cipher orange,
on the same axes; states read as 'no change' (|G| under the floor) drawn hollow. Every share reads 0 for the original
model and 1 for the fine-tuned model.

  figures/p1_neuron_patch.png  B2: recovered share against the number of patched neurons k, per state; 0.5 marks k50
  figures/p2_idle.png          B2: idle share of the top 16 neurons, per state
  figures/p3_steering.png      B3: steering share S per state (the layer chosen on the fit half)
  figures/p4_prompting.png     B4: rise of log P(target) in nats per task, and the prompting share per state, against n
  figures/p5_null.png          |G| of all nine states against each one's floor (log scale)

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/baseline_neurons/plot.py [results.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
COLOR = {"copy": "#2a78d6", "cipher": "#eb6834", "null": "#8a8984"}
INK, GRID = "#52514e", "#e4e3df"
TASK_LABEL = {"ident32": "copy A", "ident32b": "copy B", "rand32": "cipher A", "rand32b": "cipher B"}


def kind(n: str) -> str:
    return "copy" if n.startswith("ident") else "cipher" if n.startswith("rand") else "null"


def label(n: str) -> str:
    t, s = n.rsplit("_", 1) if "_s" in n else (n, "")
    return f"{TASK_LABEL[t]} {s}".strip() if t in TASK_LABEL else "code text (learned nothing)"


def style(ax: Any, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def refs(ax: Any, lines: dict[float, str]) -> None:
    """Horizontal reference lines, each labelled at the right edge."""
    for y, text in lines.items():
        ax.axhline(y, color=INK, lw=0.7, ls="--" if y else "-")
        ax.annotate(text, (1, y), xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points",
                    ha="right", fontsize=7, color=INK)


def kind_legend(ax: Any) -> None:
    h = [Line2D([], [], marker="o", ls="", color=COLOR[k], label=k) for k in ("copy", "cipher")]
    ax.legend(handles=h, fontsize=8, frameon=False, loc="upper left")


def strip(R: dict[str, Any], vals: dict[str, float], title: str, ylabel: str, path: Path,
          lines: dict[float, str]) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    names = list(vals)
    for x, n in enumerate(names):
        hollow = R["gains"][n]["no_change"]
        ax.scatter(x, vals[n], s=60, zorder=3, edgecolor=COLOR[kind(n)], lw=1.5,
                   facecolor="white" if hollow else COLOR[kind(n)])
    refs(ax, lines)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    ax.set_xticks(range(len(names)), [label(n) for n in names], rotation=45, ha="right", fontsize=8)
    style(ax, title, ylabel)
    kind_legend(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def curves(ax: Any, R: dict[str, Any], xs: list[Any], ys: dict[str, list[float]]) -> None:
    """One line per state or task: colour = kind, solid = table A, dashed = table B, dotted = 'no change'."""
    for n, y in ys.items():
        dotted = n in R["gains"] and R["gains"][n]["no_change"]
        ls = ":" if dotted else "--" if n.split("_")[0].endswith("b") else "-"
        ax.plot(xs, y, marker="o", ms=4, lw=1.2, color=COLOR[kind(n)], alpha=0.85, ls=ls)
    h = [Line2D([], [], color=COLOR[k], label=k) for k in ("copy", "cipher")]
    h += [Line2D([], [], color=INK, ls=s, label=f"table {t}") for t, s in (("A", "-"), ("B", "--"))]
    ax.legend(handles=h, fontsize=7, frameon=False, loc="center left")


def neuron_fig(R: dict[str, Any], states: list[str], path: Path) -> None:
    grid = sorted(int(k) for k in R["b2"][states[0]]["shares"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    curves(ax, R, grid, {n: [R["b2"][n]["shares"][str(k)] for k in grid] for n in states})
    refs(ax, {1: "fine-tuned model", 0.5: "half the gain", 0: "original model"})
    ax.set_xscale("log", base=2)
    ax.set_xticks(grid, [str(k) if k < 12288 else "all 12288" for k in grid])
    ax.set_xlabel("MLP neurons given the fine-tuned activations, k (largest mean shift first)", color=INK)
    style(ax, "How many neurons must be patched to recover half the fine-tune's gain?", "share of the gain recovered")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def prompting_fig(R: dict[str, Any], states: list[str], path: Path) -> None:
    ns = sorted(int(k) for k in R["b4"]["shares"][states[0]])
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)
    curves(a, R, ns, {t: [R["b4"]["nats"][t][str(k)] for k in ns] for t in R["b4"]["nats"]})
    refs(a, {0: "original model, no demonstrations"})
    style(a, "Rise of log P(target) in the original model", "nats over no demonstrations")
    curves(b, R, ns, {n: [R["b4"]["shares"][n][str(k)] for k in ns] for n in states})
    refs(b, {1: "fine-tuned model", 0: "original model"})
    style(b, "The same rise as a share of each fine-tune's gain", "share of the gain")
    for ax in (a, b):
        ax.set_xticks(ns)
        ax.set_xlabel("demonstrations in the prompt, n", color=INK)
    fig.suptitle("Do demonstrations make the original model produce the copy target more than the cipher target?",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def null_fig(R: dict[str, Any], path: Path) -> None:
    rows = {n: (abs(R["gains"][n]["G"]), R["gains"][n]["floor"]) for n in R["gains"]}
    c = R["code_step10"]
    rows["code_step10"] = (abs(c["G"]), c["floor"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for x, (n, (g, f)) in enumerate(rows.items()):
        ax.scatter(x, g, s=60, zorder=3, color=COLOR[kind(n)])
        ax.plot([x - 0.3, x + 0.3], [f, f], color=INK, lw=1.2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows)), [label(n) for n in rows], rotation=45, ha="right", fontsize=8)
    h = [Line2D([], [], marker="o", ls="", color=COLOR[k], label=f"{k}: size of the gain") for k in COLOR]
    h.append(Line2D([], [], color=INK, label="floor: below it the state reads 'no change'"))
    ax.legend(handles=h, fontsize=7, frameon=False, loc="center left")
    style(ax, "Is the learned-nothing fine-tune's gain under its floor?", "|gain G| (nats, log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    R = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "results" / "baseline_neurons.json").read_text())
    FIG.mkdir(exist_ok=True)
    states = list(R["b2"])
    neuron_fig(R, states, FIG / "p1_neuron_patch.png")
    strip(R, {n: R["b2"][n]["idle_share"] for n in states},
          "Were the 16 most-shifted neurons idle in the original model?",
          "fraction idle at the answer", FIG / "p2_idle.png", {})
    strip(R, {n: R["b3"][n]["S"] for n in states}, "How much of the gain does one added vector recover?",
          "share of the gain", FIG / "p3_steering.png",
          {1: "fine-tuned model", 0: "original model"})
    prompting_fig(R, states, FIG / "p4_prompting.png")
    null_fig(R, FIG / "p5_null.png")


if __name__ == "__main__":
    main()
