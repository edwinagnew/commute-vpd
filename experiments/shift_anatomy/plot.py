"""Figures for shift_anatomy (E32) from results/shift_anatomy.json (JSON only; no model).

Every number is a share of what each held-out window's own updated importance values give back at the trigger
(R = rho / rho_trig), as in E28.
  per_key.png   one fixed change for every input (R_mean_net), one per key (R_key_net), and random groups of the
                same sizes (R_group_net, mean of 5 draws).
  guesses.png   the two simple forms the change could take: one direction with a per-input strength (R_strength)
                and one fixed move of Gamma's input (R_steer), with that arm's ceiling (R_steer_own).
  blocks.png    attention or MLP matrices given the fixed change while the other block keeps each input's own
                values (R_*_fixed), against the same block given the original model's mean (R_*_ref).

Usage (repo root):  uv run python experiments/shift_anatomy/plot.py [--results PATH] [--figures DIR]
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
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
ARM = {"shared": "#1baf7a", "key": "#2a78d6", "group": "#a3a29d", "strength": "#e8a33a", "steer": "#8e5bd6"}
STATES = (("emoticon", "emoticon"), ("r0_t352_v310", "it → is"), ("r1_t588_v452", "will → have"),
          ("r2_t275_v2667", "in → Fig"), ("r0_t588_v320", "will → be"), ("ident32", "copy\n(32 keys)"),
          ("rand32", "32-pair\ncipher"), ("rand1", "one-pair\ncipher"))
N_WORD = 5  # the first five are word tasks


def style(ax: Any, title: str, ylabel: str) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(1, color=MUTED, lw=0.8, ls=":", label="each input's own values (1)")
    ax.axvline(N_WORD - 0.5, color=MUTED, lw=0.8)
    ax.text(N_WORD / 2 - 0.5, 1.22, "word tasks the model already does", ha="center", fontsize=8, color=MUTED)
    ax.text(N_WORD + 1, 1.22, "key-value tasks it lacks", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(STATES)), [lab for _, lab in STATES], fontsize=8)
    ax.set_ylim(-0.1, 1.3)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.5)


def bars(ax: Any, cols: list[tuple[str, str, list[float]]]) -> None:
    w = 0.8 / len(cols)
    for j, (name, colour, vals) in enumerate(cols):
        x = np.arange(len(vals)) + (j - (len(cols) - 1) / 2) * w
        ax.bar(x, vals, w, color=colour, label=name)


def per_key(st: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bars(ax, [("one fixed change for every input", ARM["shared"], [st[s]["R_mean_net"] for s, _ in STATES]),
              ("one fixed change per key", ARM["key"], [st[s]["R_key_net"] for s, _ in STATES]),
              ("random groups of the same sizes", ARM["group"], [st[s]["R_group_net"] for s, _ in STATES])])
    style(ax, "Does a fixed change work once it is computed separately for each key?",
          "share of own values' recovery")
    ax.axhline(0.7, color=INK, ls="--", lw=1.0, label="filed bar for 'holds' (0.7)")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "per_key.png", dpi=150)


def guesses(st: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bars(ax, [("one direction, per-input strength", ARM["strength"], [st[s]["R_strength"] for s, _ in STATES]),
              ("one fixed move of what Γ reads", ARM["steer"], [st[s]["R_steer"] for s, _ in STATES])])
    x = np.arange(len(STATES))
    ax.scatter(x + 0.2, [st[s]["R_steer_own"] for s, _ in STATES], marker="_", s=260, color=INK, zorder=3,
               label="ceiling of the move: each input's own move")
    ax.scatter(x, [st[s]["R_mean_net"] for s, _ in STATES], marker="o", s=26, facecolor="white",
               edgecolor=ARM["shared"], zorder=3, label="one fixed change of the values, for reference")
    style(ax, "Is the change one direction with a varying strength, or one fixed move of Γ's input?",
          "share of own values' recovery")
    ax.axhline(0.5, color=INK, ls="--", lw=1.0, label="filed bar (0.5 steering; 0.7 strength on word tasks)")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "guesses.png", dpi=150)


def blocks(st: dict[str, Any], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, blk, name in zip(axes, ("attn", "mlp"), ("attention", "MLP"), strict=True):
        bars(ax, [(f"{name} at the fixed change", ARM["shared"], [st[s][f"R_{blk}_fixed"] for s, _ in STATES]),
                  (f"{name} at the original model's mean", ARM["group"],
                   [st[s][f"R_{blk}_ref"] for s, _ in STATES])])
        style(ax, f"{name} matrices given one fixed change;\nthe other block keeps each input's own values",
              "share of own values' recovery" if blk == "attn" else "")
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.suptitle("Which block's change matters, and is it fixed there?", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "blocks.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results" / "shift_anatomy.json")
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    a = ap.parse_args()
    st = json.loads(a.results.read_text())["states"]
    a.figures.mkdir(exist_ok=True)
    per_key(st, a.figures)
    guesses(st, a.figures)
    blocks(st, a.figures)


if __name__ == "__main__":
    main()
