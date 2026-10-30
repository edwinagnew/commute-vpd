"""E56's figures from results/job1.json (seeds 0, 1) and results/job2.json (seeds 2-4), no model. Every share reads 0 for
the original model and 1 for the fine-tuned model. Colours as E54: elicit green, cipher orange, earlier copy blue.

  figures/b2_curves.png  neuron patching: share of G against k, the number of top MLP neurons patched; elicit, matched
                         cipher and own-step cipher (five seeds each, seed range shaded, mean drawn), with E12's copy
                         and cipher states (E52) as markers at the k E52 measured and half of G marked (P1, k50 and N16)
  figures/p2_shares.png  r and S_lo per state, five seeds, elicit / matched cipher / own-step cipher (P2)
  figures/p3_size.png    the cipher's r at the matched step against its own step, per seed, with P3's band of 0.1 (P3)

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/keydependent_patch/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG, RES, EXP = HERE / "figures", HERE / "results", HERE.parent
COLOR = {"copy": "#2a78d6", "cipher": "#eb6834", "elicit": "#1f9e6e"}
INK, GRID = "#52514e", "#e4e3df"
SEEDS = [f"s{i}" for i in range(5)]
GRID_K = [1, 4, 16, 64, 256, 1024, 4096, 12288]
ARMS = [("elicit", "elicit map", "elicit", "-"),
        ("cipher", "cipher, matched size (step 9)", "cipher", "-"),
        ("cipher_own", "cipher, own step (30)", "cipher", "--")]
SHORT = {"elicit": "elicit map", "cipher": "cipher,\nmatched", "cipher_own": "cipher,\nown step"}


def load() -> dict[str, dict[str, Any]]:
    """All 15 states: s0, s1 from job 1, s2-s4 from job 2."""
    j1, j2 = (json.loads((RES / f).read_text())["states"] for f in ("job1.json", "job2.json"))
    return {**j1, **j2}


def style(ax: Any, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def b2_curves(st: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    e52 = json.loads((EXP / "baseline_neurons" / "results" / "baseline_neurons.json").read_text())["b2"]
    for task, label in (("ident32", "earlier copy task (E52), 4 states, own steps"),
                        ("rand32", "earlier cipher (E52), 4 states, own steps")):
        kind = "copy" if task == "ident32" else "cipher"
        for i, n in enumerate(f"{task}{b}_s{s}" for b in ("", "b") for s in (0, 1)):
            ks = sorted(int(k) for k in e52[n]["shares"])  # E52 measured k <= 256 and 12288 only: markers, no lines
            ax.scatter(ks, [e52[n]["shares"][str(k)] for k in ks], marker="x" if kind == "copy" else "+", s=22,
                       color=COLOR[kind], lw=0.9, alpha=0.7, label=label if i == 0 else None, zorder=2)
    for arm, label, kind, ls in ARMS:
        rows = [[st[f"{arm}_{s}"]["b2_shares"][str(k)] for k in GRID_K] for s in SEEDS]
        lo, hi = [min(c) for c in zip(*rows, strict=True)], [max(c) for c in zip(*rows, strict=True)]
        mean = [sum(c) / len(c) for c in zip(*rows, strict=True)]
        ax.fill_between(GRID_K, lo, hi, color=COLOR[kind], alpha=0.18, lw=0)
        ax.plot(GRID_K, mean, color=COLOR[kind], lw=2, ls=ls, marker="o", ms=4,
                mfc=COLOR[kind] if ls == "-" else "white", label=f"{label}, 5 seeds")
    ax.axhline(0.5, color=INK, lw=0.8, ls="-.")
    ax.axhline(0, color=INK, lw=0.7)
    ax.axhline(1, color=INK, lw=0.7, ls="--")
    for y, text in ((0.5, "half of the gain: k50 is the first k at or above this line"),
                    (0, "original model, and identity patch: 0"),
                    (1, "fine-tuned model, and block-3 residual patch: 1")):
        ax.annotate(text, (0.01, y), xycoords=("axes fraction", "data"), xytext=(0, 3), textcoords="offset points",
                    fontsize=7, color=INK)
    ax.axvline(16, color=GRID, lw=1)
    ax.annotate("16 neurons", (16, -0.12), xytext=(3, 0), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(GRID_K, [str(k) for k in GRID_K], fontsize=8)
    ax.set_xlim(0.8, 15000)
    ax.set_ylim(-0.15, 1.08)
    ax.set_xlabel("k: number of MLP neurons given the fine-tuned activations (ranked by mean shift)", color=INK,
                  fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    style(ax, "How much of each fine-tune's gain do its top-k MLP neurons carry?",
          "share of the gain G recovered\nby patching the top k neurons")
    fig.tight_layout()
    fig.savefig(FIG / "b2_curves.png", dpi=150)


def p2_shares(st: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = 0
    ticks, labels = [], []
    for metric, name in (("r", "re-gating share r"), ("S_lo", "steering share")):
        for arm, _, kind, ls in ARMS:
            for i, s in enumerate(SEEDS):
                old = i < 2
                ax.scatter(x + 0.07 * (i - 2), st[f"{arm}_{s}"][metric], s=34, marker="o" if old else "D",
                           color=COLOR[kind] if ls == "-" else "white", edgecolors=COLOR[kind], lw=1.3, zorder=3)
            ticks.append(x)
            labels.append(f"{name}:\n{SHORT[arm]}")
            x += 1
        x += 0.6
    ax.axvline(2.8, color=GRID, lw=1)
    ax.scatter([], [], marker="o", color=INK, s=24, label="seeds 0, 1 (E54's states)")
    ax.scatter([], [], marker="D", color=INK, s=24, label="seeds 2-4 (new)")
    ax.set_xticks(ticks, labels, fontsize=7)
    ax.set_ylim(-0.08, 1.08)
    ax.axhline(0, color=INK, lw=0.7)
    ax.axhline(1, color=INK, lw=0.7, ls="--")
    for y, text in ((0, "original model reads 0"), (1, "fine-tuned model reads 1")):
        ax.annotate(text, (1, y), xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points",
                    ha="right", fontsize=7, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.94))
    style(ax, "At five seeds, does r still separate the pair where one steering vector does not?",
          "share of the fine-tune's log-odds gain")
    fig.tight_layout()
    fig.savefig(FIG / "p2_shares.png", dpi=150)


def p3_size(st: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, s in enumerate(SEEDS):
        m, o = st[f"cipher_{s}"]["r"], st[f"cipher_own_{s}"]["r"]
        ax.fill_between([i - 0.3, i + 0.3], o - 0.1, o + 0.1, color=COLOR["cipher"], alpha=0.1, lw=0,
                        label="within 0.1 of the own-step r" if i == 0 else None)
        ax.plot([i - 0.1, i + 0.1], [m, o], color=COLOR["cipher"], lw=0.8)
        ax.scatter(i - 0.1, m, s=36, color=COLOR["cipher"], zorder=3, label="cipher, matched size (step 9)" if i == 0 else None)
        ax.scatter(i + 0.1, o, s=36, color="white", edgecolors=COLOR["cipher"], lw=1.4, zorder=3,
                   label="cipher, own step (30)" if i == 0 else None)
        e = st[f"elicit_{s}"]["r"]
        ax.scatter(i, e, s=36, color=COLOR["elicit"], zorder=3, label="elicit map (reference)" if i == 0 else None)
    ax.set_xticks(range(5), [f"seed {i}" + (" (E54)" if i < 2 else "") for i in range(5)], fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.6))
    style(ax, "Does the cipher's r move between matched size and its own step?",
          "re-gating share r")
    fig.tight_layout()
    fig.savefig(FIG / "p3_size.png", dpi=150)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    st = load()
    b2_curves(st)
    p2_shares(st)
    p3_size(st)


if __name__ == "__main__":
    main()
