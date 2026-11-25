"""P2.78's figure in the paper's words (writeup-lead, 2026-11-24): the same data as plot_p278.py (kept share = E_b over
the needed set, trained against its size-matched random, 3-seed means, faint seed points), tiers ordered new (left) to
reuse (right) as in the paper's Fig 5, no filed band. Reads results/wide/random_3seed.json only.

    ~/commute-agents/fleet/cpu-run --mem 2G -t 10 python3 experiments/convergence_panel/plot_p278_paper.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from commute.figures import METHODS, figure, save

HERE = Path(__file__).parent
SRC = HERE / "results/wide/random_3seed.json"
FIG = HERE / "figures/p278_paper.png"
NAMES = {"arxiv": "arXiv", "category": "category label", "conll": "CoNLL", "french": "French", "german": "German",
         "lower": "lowercase", "sentiment": "sentiment", "spanish": "Spanish", "squad": "SQuAD", "summary": "summary",
         "swahili": "Swahili", "yoruba": "Yoruba"}  # fmt: skip


def main() -> None:
    per = json.loads(SRC.read_text())["per_task"]
    tasks = sorted(per, key=lambda t: (per[t]["tier"], t))  # new (tier 1) at the left, reuse (tier 3) at the right
    fig, ax = figure(size=(6.5, 4.6))
    rows = []
    for side, style, dx in (("trained", METHODS["eb"], -0.12), ("random", METHODS["random"], 0.12)):
        means = [fmean(per[t][side].values()) for t in tasks]
        for i, t in enumerate(tasks):
            seeds = [per[t][side][s] for s in ("0", "1", "2")]
            ax.scatter([i + dx] * 3, seeds, s=9, color=style.color, alpha=0.35, linewidths=0)
            rows += [{"task": t, "tier": per[t]["tier"], "series": side, "seed": s, "kept_share": v} for s, v in zip("012", seeds)]
            rows.append({"task": t, "tier": per[t]["tier"], "series": side, "seed": "mean", "kept_share": means[i]})
        label = "trained fine-tune, mean of 3 seeds" if side == "trained" else "random update of the same size, mean of 3 draws"
        ax.scatter([i + dx for i in range(len(tasks))], means, s=30, color=style.color, marker=style.marker, label=label)
    ax.set_xticks(range(len(tasks)), [f"{NAMES[t]} ({per[t]['tier']:g})" for t in tasks], rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("task, raters' consensus tier in brackets: 1 = new, 3 = reuse")
    ax.set_ylabel("kept share")
    ax.set_ylim(0, 0.15)
    ax.set_title("On realistic fine-tunes the kept share rises toward tasks rated new;\nrandom updates barely move", fontsize=9, pad=34)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, FIG, n="12 tasks; 3 seeds per task, each with its own random draw (faint points); "
         "the do-nothing update reads 1, off the axis", data=rows)


if __name__ == "__main__":
    main()
