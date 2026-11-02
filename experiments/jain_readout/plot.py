"""E59's figures from results/run.json (job 48412, which reloaded job 48357's readouts) and results/m1_check.json (job
48453, the model's own ranks through the same rules), no model. Colours as E54/E56: elicit green, cipher orange.

  figures/readout_loss.png  next-token loss on Pile of each block's readout, on windows it trained on (held-in) and on
                            windows it never saw (held-out), against the model's own loss and the linearised head's,
                            with the filed bar (linearised head + 0.1 nats)
  figures/possession.png    keys (of 32) whose value has median rank <= 20 under each block's readout, elicit and
                            cipher, with the 24-key bar shared by possession and the last-block construction check, and
                            the model's own output read by the same two rules

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/jain_readout/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
COLOR = {"cipher": "#eb6834", "elicit": "#1f9e6e", "held_out": "#2a78d6", "held_in": "#8fb8e8"}
INK, GRID = "#52514e", "#e4e3df"
BLOCKS = [0, 1, 2, 3]


def style(ax: Any, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, color=INK)
    ax.set_xlabel("block whose output the readout reads", color=INK)
    ax.set_xticks(BLOCKS, [f"block {b}" + (" (last)" if b == 3 else "") for b in BLOCKS], fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def label_line(ax: Any, y: float, text: str, above: bool = True) -> None:
    ax.annotate(text, (0.01, y), xycoords=("axes fraction", "data"), xytext=(0, 3 if above else -10),
                textcoords="offset points", fontsize=7, color=INK)


def readout_loss(rd: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(BLOCKS, rd["readout"], color=COLOR["held_out"], lw=2, marker="o", ms=5,
            label="readout, Pile windows it never saw (held-out)")
    ax.plot(BLOCKS, rd["held_in"]["readout"], color=COLOR["held_in"], lw=2, marker="o", ms=5, ls="--",
            label="readout, Pile windows it trained on (held-in)")
    lin, model = rd["lin"], rd["model"]
    ax.axhline(model, color=INK, lw=0.8)
    ax.axhline(lin, color=INK, lw=0.8, ls="--")
    ax.axhline(lin + 0.1, color="#b0413e", lw=1, ls=":")
    label_line(ax, model, f"the model's own output: {model:.2f}", above=False)
    label_line(ax, lin, f"linearised head, a linear map the last readout could learn: {lin:.2f}", above=False)
    label_line(ax, lin + 0.1, "filed bar for the last block: linearised head + 0.1")
    ax.annotate(f"{rd['readout'][3]:.2f}", (3, rd["readout"][3]), xytext=(6, 2), textcoords="offset points",
                fontsize=8, color=COLOR["held_out"])
    ax.annotate(f"{rd['held_in']['readout'][3]:.2f}", (3, rd["held_in"]["readout"][3]), xytext=(-30, -12),
                textcoords="offset points", fontsize=8, color=COLOR["held_in"])
    ax.set_ylim(3.2, 5.8)
    ax.set_xlim(-0.2, 3.45)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    style(ax, "Do the readouts fit unseen Pile text as well as their training text? Next-token loss per block",
          "next-token loss on the Pile (nats)")
    fig.tight_layout()
    fig.savefig(FIG / "readout_loss.png", dpi=150)


def possession(run: dict[str, Any], m1: dict[str, Any]) -> None:
    """Readouts per block, then the model's own output in a column of its own, each read by the same two rules."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    model = len(BLOCKS)
    for kind, label in (("elicit", "elicit map: keys at median rank 20 or better"),
                        ("cipher", "cipher: keys at median rank 20 or better")):
        counts = [p["count"] for p in run["possession"][kind]["all"]["possession"]]
        ax.plot(BLOCKS, counts, color=COLOR[kind], lw=2, marker="o", ms=5, label=label)
        own = m1["model"][kind]["possession"]["count"]
        ax.plot([model], [own], color=COLOR[kind], marker="o", ms=5, ls="none")
        for b, c in zip([*BLOCKS, model], [*counts, own], strict=True):
            ax.annotate(str(c), (b, c), xytext=(5, 4), textcoords="offset points", fontsize=8, color=COLOR[kind])
    ax.axhline(24, color=INK, lw=0.9, ls="--")
    label_line(ax, 24, "the bar for both rules: 24 of 32 keys")
    band, own_band = run["P3"]["elicit_in_band"], m1["model"]["P3"]["elicit_in_band"]
    ax.scatter([3, model], [band, own_band], marker="D", s=40, facecolor="white", edgecolor=COLOR["elicit"], lw=1.5,
               zorder=3, label="elicit keys at median rank 2-20, the construction check's band")
    for x, y in ((3, band), (model, own_band)):
        ax.annotate(str(y), (x, y), xytext=(5, -10), textcoords="offset points", fontsize=8, color=COLOR["elicit"])
    ax.scatter([model], [32], marker="*", s=90, color=INK, zorder=3,
               label="a different statistic: mean probability over 500 windows ranks all 32 in 2-20 (how the pair was built)")
    ax.axvline(model - 0.5, color=GRID, lw=1)
    ax.set_ylim(-1, 44)
    ax.set_xlim(-0.2, model + 0.4)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    style(ax, "Does any block's readout hold the elicit values, and does the model itself pass the same bar?",
          "keys of 32 meeting the rank rule, per window, lower median")
    ax.set_xticks([*BLOCKS, model], [*[t.get_text() for t in ax.get_xticklabels()], "model's own output"], fontsize=8)
    ax.set_xlabel("what is read: a block's readout, or the model's logits", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "possession.png", dpi=150)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    run = json.loads((RES / "run.json").read_text())
    readout_loss(run["stage0"]["readouts"])
    possession(run, json.loads((RES / "m1_check.json").read_text()))


if __name__ == "__main__":
    main()
