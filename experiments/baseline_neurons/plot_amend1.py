"""E52 Amendment 1's figure, from results/amend1.json (no model): one claim, P4a (does W0 perform the copy in context
when the demonstrations sit next to the query?). Left: the prompting share against n per state, copy and cipher on the
same axes, with the filed bands (0.5 holds, 0.3 partial). Right: W0's top-1 on the target against n per task, with the
induction control's top-1 (a random sequence shown twice, second copy) as the reference that in-context copying is
visible at all.

  figures/p4a_prompting_adjacent.png

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/baseline_neurons/plot_amend1.py [amend1.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from plot import FIG, HERE, INK, curves, refs, style


def figure(R: dict[str, Any], path: Path) -> None:
    ns = sorted(int(k) for k in R["prompts"])
    ind = R["stage0"]["induction"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)
    curves(a, R, ns, {s: [R["shares"][s][str(n)] for n in ns] for s in R["shares"]})
    refs(a, {1: "fine-tuned model", 0.5: "copy label holds", 0.3: "partial", 0: "original model"})
    style(a, "Share of each fine-tune's gain from demonstrations", "share of the gain")
    curves(b, R, ns, {t: [R["prompts"][str(n)][t]["top1"] for n in ns] for t in R["prompts"]["0"]})
    refs(b, {ind["top1_second"]: f"control: repeated random sequence ({ind['top1_second']:.2f})",
             ind["top1_first"]: f"control, first showing ({ind['top1_first']:.2f})"})
    b.set_ylim(-0.03, 1.05)
    style(b, "Original model's top-1 on the target", "fraction of windows, target is the argmax")
    for ax in (a, b):
        ax.set_xticks(ns)
        ax.set_xlabel("demonstrations next to the query, n", color=INK)
    fig.suptitle("Does the original model copy in context when shown examples next to the query, and the cipher not?",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    R = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "results" / "amend1.json").read_text())
    FIG.mkdir(exist_ok=True)
    figure(R, FIG / "p4a_prompting_adjacent.png")


if __name__ == "__main__":
    main()
