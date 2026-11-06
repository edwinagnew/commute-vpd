"""E116's figures from results/stage0.json, stage1.json and predictions.json, no model.

  figures/possession.png  per readout and block, the Pile readout's accuracy on the task windows beside its
                          Hewitt-Liang control (top) and their difference, the selectivity (bottom), with the filed
                          bars 0.90 and 0.30; the last block is hatched, as the rule does not count it
  figures/carrying.png    per task with held-out values, each state's top-1 accuracy within the task's value set on
                          held-out-value windows, beside the original model's, with the 0.90 bar; the tasks where
                          carrying cannot be applied are named with the reason

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/definition_table/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#3a9a5b", "#8a5cc2"
GREY, INK, GRID = "#7a7a75", "#52514e", "#e4e3df"
# (stage0 key, label, colour); ident32 = rand32 and copy24 = cipher24 read identical numbers (same keys, same windows)
READOUTS = [("identity_ident32", "key identity,\n32-key tables", PURPLE),
            ("identity_copy24", "key identity,\n24-key tables", PURPLE),
            ("identity_e62_lookup", "key identity,\nlookup (E62)", PURPLE),
            ("copy", "in-context\ncopying (E62)", BLUE),
            ("guess_rank1", "guess,\nrank-1 keys", ORANGE),
            ("guess_elicit", "guess,\nelicit keys", ORANGE),
            ("guess_cipher", "guess,\n50-word cipher", GREEN),
            ("guess_broad_cipher", "guess,\n500-word cipher", GREEN)]  # fmt: skip
CARRY = [("copy24", "copy,\n24 keys", BLUE), ("e62_copy", "in-context\ncopy (E62)", BLUE),
         ("rank1", "rank-1\nguess map", ORANGE), ("elicit", "elicit\nmap", ORANGE),
         ("cipher24", "cipher,\n24 keys", GREEN)]  # fmt: skip
NA = ["32-key identity and random maps: no held-out keys",
      "lookup (E62): no answer for an unseen key",
      "text ciphers: no target for an unseen word"]  # fmt: skip


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def possession(s0: dict[str, Any], rules: dict[str, Any]) -> None:
    R, w = s0["readouts"], 0.19
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, height_ratios=[3, 2])
    for i, (key, _, c) in enumerate(READOUTS):
        for blk in range(4):
            x, h = i + (blk - 1.5) * w, "//" if blk == 3 else ""
            acc, ctl, sel = R[key]["acc"][blk], R[key]["acc_control"][blk], R[key]["selectivity"][blk]
            a.bar(x, acc, w, color=c, hatch=h, edgecolor="white")
            a.hlines(ctl, x - w / 2, x + w / 2, color="black", lw=2)
            b.bar(x, sel, w, color=c, hatch=h, edgecolor="white")
    a.axhline(rules["P0"]["accuracy"], color=INK, ls="--", lw=1)
    a.text(7.5, rules["P0"]["accuracy"] + 0.015, "accuracy bar 0.90", ha="right", va="bottom", color=INK, fontsize=8)
    b.axhline(rules["P0"]["selectivity"], color=INK, ls="--", lw=1)
    b.text(7.5, rules["P0"]["selectivity"] + 0.015, "selectivity bar 0.30", ha="right", va="bottom", color=INK, fontsize=8)
    b.axhline(0, color=INK, lw=0.8)
    a.set(ylim=(0, 1.12), ylabel="readout accuracy on task windows",
          title="Does any Pile readout of the named capability pass both bars before the last block? No.")  # fmt: skip
    b.set(ylim=(-0.5, 0.75), ylabel="selectivity\n(readout minus control)",
          xticks=range(len(READOUTS)), xticklabels=[n for _, n, _ in READOUTS])  # fmt: skip
    b.tick_params(axis="x", labelsize=8)
    b.set_xlabel("readout (\"guess\" = the original model's own next-token guess, among the task's values)", fontsize=9)
    handles = [Patch(facecolor=GREY, label="bars left to right: blocks 0, 1, 2"),
               Patch(facecolor=GREY, hatch="//", edgecolor="white", label="block 3, the last (not counted)"),
               Line2D([], [], color="black", lw=2, label="control readout (one random value per token type)")]  # fmt: skip
    a.legend(handles=handles, fontsize=8, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    style(a)
    style(b)
    fig.tight_layout()
    fig.savefig(FIG / "possession.png", dpi=150)


def carrying(s1: dict[str, Any], rules: dict[str, Any]) -> None:
    S, W, w = s1["states"], s1["w0"], 0.2
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (task, _, c) in enumerate(CARRY):
        seeds = sorted(k for k in S if k.rsplit("_s", 1)[0] == task)
        for j, s in enumerate(seeds):
            x = i + (j - 1.5) * w
            ax.bar(x, S[s]["P1"], w, color=c, edgecolor="white")
            ax.text(x, S[s]["P1"] + 0.01, f"{S[s]['P1']:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)
        xw = i + 1.5 * w
        ax.bar(xw, W[task]["P1"], w, color=GREY, edgecolor="white")
        ax.text(xw, W[task]["P1"] + 0.01, f"{W[task]['P1']:.2f}", ha="center", va="bottom", fontsize=6, rotation=90,
                color=GREY)  # fmt: skip
    ax.axhline(rules["real"]["P1"], color=INK, ls="--", lw=1)
    ax.text(4.5, rules["real"]["P1"] + 0.015, "bar 0.90", ha="right", va="bottom", color=INK, fontsize=8)
    ax.text(4.8, 0.55, "Cannot be applied:\n" + "\n".join(NA), fontsize=8, va="center", color=INK)
    ax.set(xticks=range(len(CARRY)), xticklabels=[n for _, n, _ in CARRY], xlim=(-0.5, 7.4), ylim=(0, 1.15),
           ylabel="top-1 within the task's values,\nheld-out-value windows",
           title="Does each fine-tune answer right on values it never trained on?")  # fmt: skip
    handles = [Patch(facecolor=BLUE, label="fine-tuned, one bar per seed (copying tasks)"),
               Patch(facecolor=ORANGE, label="fine-tuned (maps to the original model's guesses)"),
               Patch(facecolor=GREEN, label="fine-tuned (random cipher)"),
               Patch(facecolor=GREY, label="original model, same windows")]  # fmt: skip
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=2)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "carrying.png", dpi=150)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    rules = load("predictions.json")["rules"]
    possession(load("stage0.json"), rules)
    carrying(load("stage1.json"), rules)


if __name__ == "__main__":
    main()
