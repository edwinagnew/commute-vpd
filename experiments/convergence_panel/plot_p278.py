"""P2.78 figure: each task's write share over the needed set, trained against its size-matched random update, both on
3-seed means, tasks in the raters' consensus order. Reads results/wide/random_3seed.json only.

Also writes results/wide/random_3seed_by_seed.json: the tau-b of each single seed with the consensus tier (trained and
random), so the page can say whether seed 0's draw was typical. Same tau-b as score.py (scipy kendalltau).

    ~/commute-agents/fleet/cpu-run --mem 2G -t 10 python3 experiments/convergence_panel/plot_p278.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from scipy.stats import kendalltau

from commute.figures import METHODS, figure, save

HERE = Path(__file__).parent
SRC = HERE / "results/wide/random_3seed.json"
OUT = HERE / "results/wide/random_3seed_by_seed.json"
FIG = HERE / "figures/p278_random_3seed.png"
NAMES = {"arxiv": "arXiv", "category": "category label", "conll": "CoNLL", "french": "French", "german": "German",
         "lower": "lowercase", "sentiment": "sentiment", "spanish": "Spanish", "squad": "SQuAD", "summary": "summary",
         "swahili": "Swahili", "yoruba": "Yoruba"}  # fmt: skip


def by_seed(per: dict) -> dict:
    tasks = sorted(per)
    tier = [per[t]["tier"] for t in tasks]
    out = {}
    for side in ("trained", "random"):
        for s in ("0", "1", "2"):
            r = kendalltau(tier, [per[t][side][s] for t in tasks])
            out[f"{side}_s{s}"] = {"tau": float(r[0]), "p_two_sided": float(r[1]), "n": len(tasks)}
    return out


def main() -> None:
    per = json.loads(SRC.read_text())["per_task"]
    OUT.write_text(json.dumps({"source": str(SRC.relative_to(HERE.parent.parent)), "by_seed": by_seed(per)}, indent=1))
    tasks = sorted(per, key=lambda t: (-per[t]["tier"], t))  # reuse at the left, new at the right
    fig, ax = figure(size=(6.5, 4.6))
    rows = []
    for side, style, dx in (("trained", METHODS["eb"], -0.12), ("random", METHODS["random"], 0.12)):
        means = [fmean(per[t][side].values()) for t in tasks]
        for i, t in enumerate(tasks):
            seeds = [per[t][side][s] for s in ("0", "1", "2")]
            ax.scatter([i + dx] * 3, seeds, s=9, color=style.color, alpha=0.35, linewidths=0)
            rows += [{"task": t, "tier": per[t]["tier"], "series": side, "seed": s, "E_b": v} for s, v in zip("012", seeds)]
            rows.append({"task": t, "tier": per[t]["tier"], "series": side, "seed": "mean", "E_b": means[i]})
        label = "trained fine-tune, mean of 3 seeds" if side == "trained" else "random update of the same size, mean of 3 draws"
        ax.scatter([i + dx for i in range(len(tasks))], means, s=30, color=style.color, marker=style.marker, label=label)
    ax.axhspan(0.0, 0.0357, color="#bbbbbb", alpha=0.3, label="filed band for a random update")
    ax.set_xticks(range(len(tasks)), [f"{NAMES[t]} ({per[t]['tier']:g})" for t in tasks], rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("task, raters' consensus tier in brackets: 3 = reuse, 1 = new")
    ax.set_ylabel("write share over the needed set")
    ax.set_ylim(0, 0.15)
    ax.set_title("Trained write share rises toward new tasks; random barely moves", fontsize=9, pad=34)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, FIG, n="12 tasks; 3 seeds per task, each with its own random draw (faint points); "
         "the do-nothing update reads 1, off the axis", data=rows)
    print(json.dumps(json.loads(OUT.read_text())["by_seed"], indent=1))


if __name__ == "__main__":
    main()
