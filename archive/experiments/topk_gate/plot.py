"""E230's figure (cell P2.47), from results/score.json and experiments/method_table/results/final.json only (no model):
per built pair, every saved fine-tune's share under three gates on VPD's subcomponents (the gate function's own
importances, the same importances thresholded at 0.5, and the top-k gate at the matched k), the side built to reuse
beside the side built to be new, with each pair's call.

    ~/commute-agents/fleet/cpu-run --mem 2G python3 experiments/topk_gate/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent / "method_table" / "results" / "final.json"
GATES = {
    "gamma": "gate function's importances\n(the share as saved)",
    "gamma_bin": "the same importances,\n1 above 0.5, else 0",
    "topk_x1": "top-k gate on |x·V_c|,\nk matched to the gate function",
}
REUSE, NEW = "#0072B2", "#888888"


def names() -> dict[str, str]:
    cells = json.loads(FINAL.read_text())["cells"]
    return {c["pair"]: c["pair_name"] for c in cells if c["group"] == "built" and c["method"] == "share"}


def records(score: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for pair, by_gate in score["pairs"].items():
        for gate in GATES:
            cell = by_gate[gate]
            for side in ("reuse", "new"):
                out += [{"pair": pair, "gate": gate, "side": side, "seed": i, "share": v, "call": cell["call"]}
                        for i, v in enumerate(cell[side])]
    return out


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    score = json.loads((results / "score.json").read_text())
    label = names()
    pairs = list(score["pairs"])
    recs = records(score)
    fig, axes = FG.figure((7.5, 4.2), ncols=3)
    for j, (ax, gate) in enumerate(zip(axes, GATES, strict=True)):
        for i, pair in enumerate(pairs):
            cell = score["pairs"][pair][gate]
            ax.scatter(cell["reuse"], [i - 0.12] * len(cell["reuse"]), color=REUSE, marker="o", s=14, linewidths=0,
                       label="built to reuse" if i == 0 else None)
            ax.scatter(cell["new"], [i + 0.12] * len(cell["new"]), facecolors="none", edgecolors=NEW, marker="s", s=14,
                       linewidths=0.8, label="built to be new" if i == 0 else None)
            ax.annotate(cell["call"], (1.0, i), xycoords=("axes fraction", "data"), xytext=(2, 0), textcoords="offset points",
                        fontsize=6, va="center", color="#222222" if cell["call"] == "built" else "#D55E00")
        t = score["tally"][gate]
        ax.set_title(f"{GATES[gate]}\n{t['built']} built, {t['reversed']} reversed, {t['not apart']} not apart", fontsize=7)
        ax.set_xlabel("share of the gain")
        ax.set_yticks(range(len(pairs)), [label[p] for p in pairs] if j == 0 else [""] * len(pairs), fontsize=7)
        ax.set_ylim(len(pairs) - 0.5, -1.4)  # the top band holds the legend
        ax.set_xlim(0, None)
    axes[0].legend(loc="upper left", ncols=2, fontsize=6, frameon=False, handletextpad=0.2, columnspacing=0.8)
    fig.suptitle("Do VPD's subcomponents keep the nine built pairs apart once the gate function is replaced?", fontsize=9)
    FG.save(fig, out / "shares_by_gate.png", data=recs,
            n="2 to 8 saved fine-tunes a side, 68 in all; the share read on E136's read rows. Each panel has its own x scale: "
              "the top-k gate's shares are about a tenth of the gate function's. For comparison, top-k on MLP neurons at the "
              "same k kept 4 pairs the built way and reversed 5.")


if __name__ == "__main__":
    main()
