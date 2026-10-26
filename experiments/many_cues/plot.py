"""Figures for many_cues (E33) from results/many_cues.json (JSON only; no model).

Every number is a share of what each held-out window's own updated importance values give back at the trigger
(R = rho / rho_trig), as in E28 and E32. States: the one-cue control (will -> have), 32 cues each trained toward its
own target ("many answers"), the same 32 cues all trained toward " of" ("one answer").
  cue_vs_answer.png  one fixed change for every input (R_shared_net), one per cue (R_cue_net), and random groups
                     of the same sizes (R_group_net).
  donors.png         another held-out window's updated values, from the same cue or from a different cue, net of
                     the same donor's original values (R_other_within_net, R_other_across_net).

Usage (repo root):  uv run python experiments/many_cues/plot.py [--results PATH] [--figures DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MUTED, GRID = "#52514e", "#d9d8d4"
ARM = {"shared": "#1baf7a", "key": "#2a78d6", "group": "#a3a29d", "within": "#2a78d6", "across": "#e87ba4"}
STATES = (("word", "control: one cue\n(will → have)"), ("many", "32 cues,\neach its own answer"),
          ("one", "32 cues,\none answer ( of)"))


def style(ax: Any, title: str) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(1, color=MUTED, lw=0.8, ls=":", label="each input's own values (1)")
    ax.set_xticks(range(len(STATES)), [lab for _, lab in STATES], fontsize=8.5)
    ax.set_ylim(-0.1, 1.2)
    ax.set_ylabel("share of own values' recovery")
    ax.set_title(title, fontsize=10.5)


def bars(ax: Any, st: dict[str, Any], cols: list[tuple[str, str, str]]) -> None:
    w = 0.8 / len(cols)
    for j, (name, colour, key) in enumerate(cols):
        x = np.arange(len(STATES)) + (j - (len(cols) - 1) / 2) * w
        vals = [float(st[s][key]) for s, _ in STATES]
        ax.bar(x, np.nan_to_num(vals), w, color=colour, label=name)
        for xi, v in zip(x, vals, strict=True):
            gone = np.isnan(v)
            ax.text(xi, 0.02 if gone else max(v, 0) + 0.02, "n/a" if gone else f"{v:.2f}", ha="center", fontsize=7)


def cue_vs_answer(st: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars(ax, st, [("one fixed change for every input", ARM["shared"], "R_shared_net"),
                  ("one fixed change per cue", ARM["key"], "R_cue_net"),
                  ("random groups of the same sizes", ARM["group"], "R_group_net")])
    style(ax, "Does the fixed change follow the cue, or the answer?")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "cue_vs_answer.png", dpi=150)


def donors(st: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars(ax, st, [("another input's values, same cue", ARM["within"], "R_other_within_net"),
                  ("another input's values, different cue", ARM["across"], "R_other_across_net")])
    style(ax, "Do cues that share an answer share their change of values?")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "donors.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results" / "many_cues.json")
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    a = ap.parse_args()
    st = json.loads(a.results.read_text())["states"]
    a.figures.mkdir(exist_ok=True)
    cue_vs_answer(st, a.figures)
    donors(st, a.figures)


if __name__ == "__main__":
    main()
