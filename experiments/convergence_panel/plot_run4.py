"""E307 (P2.66 run 4) figure: each reading's per-seed values per task at n = 10, tasks in the raters' consensus order,
the seeds that are not KL-matched ringed, E_b's matched Gaussian draws beside the trained values. Reads
results/wide/run4/score_n10.json only (score_run4.py).

    ~/commute-agents/fleet/cpu-run --mem 2G -t 10 python3 experiments/convergence_panel/plot_run4.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute.figures import METHODS, dots, figure, save

HERE = Path(__file__).parent
SRC = HERE / "results/wide/run4/score_n10.json"
FIG = HERE / "figures/run4_n10.png"
NAMES = {"category": "category label", "conll": "CoNLL", "french": "French", "lower": "lowercase", "sentiment": "sentiment",
         "spanish": "Spanish", "squad": "SQuAD", "summary": "summary", "swahili": "Swahili", "yoruba": "Yoruba"}  # fmt: skip
PANELS = (("share", "regating", "re-gating share"), ("steering", "steering", "steering share"),
          ("erank", "spectrum", "update's effective rank"), ("E_b", "eb", "write share over the needed set"))  # fmt: skip
KL_HI = 0.181  # the KL 'hi' rung (results/wide/dropped.json); a seed off it by more than 10% is not KL-matched


def main() -> None:
    o = json.loads(SRC.read_text())
    per = o["per_task"]
    tasks = sorted(per, key=lambda t: (-per[t]["tier"], t))  # reuse at the top
    rows_names = {t: f"{NAMES[t]} ({per[t]['tier']:g})" + (" *" if t in o["not_kl_matched"] else "") for t in tasks}
    off = {t: [abs(per[t]["states"][str(s)]["kl"] / KL_HI - 1) > 0.1 for s in range(3)] for t in tasks}
    fig, axes = figure(size=(9.0, 5.4), ncols=4)
    rows = []
    for ax, (m, method, label) in zip(axes, PANELS, strict=True):
        groups = {rows_names[t]: per[t][m]["seeds"] for t in tasks}
        dots(ax, groups, method=method, xlabel=label)
        if m == "E_b":
            dots(ax, {rows_names[t]: per[t]["E_b_random"]["seeds"] for t in tasks}, method="random", xlabel=label, offset=0.3)
        for i, t in enumerate(tasks):
            hit = [v for v, f in zip(per[t][m]["seeds"], off[t], strict=True) if f]
            ax.scatter(hit, [i] * len(hit), s=70, facecolors="none", edgecolors="#000000", linewidths=0.8)
            rows += [{"task": t, "tier": per[t]["tier"], "reading": m, "seed": s, "value": v, "kl_matched": not off[t][s]}
                     for s, v in enumerate(per[t][m]["seeds"])]  # fmt: skip
            rows += [{"task": t, "tier": per[t]["tier"], "reading": "E_b_random", "seed": s, "value": v, "kl_matched": not off[t][s]}
                     for s, v in enumerate(per[t]["E_b_random"]["seeds"]) if m == "E_b"]  # fmt: skip
        h, old = o["hope1"][m]["consensus"]["tau"], o["run3_12_tasks"]["hope1_consensus"][m]["tau"]
        ax.set_title(f"tau {h:+.2f} (12 tasks before: {old:+.2f})", fontsize=7, loc="left")
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_size(7)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axes[3].scatter([], [], color=METHODS["eb"].color, marker=METHODS["eb"].marker, s=22, label="trained")
    axes[3].scatter([], [], color=METHODS["random"].color, marker=METHODS["random"].marker, s=22, label="random update, same size")
    axes[3].scatter([], [], s=70, facecolors="none", edgecolors="#000000", linewidths=0.8, label="seed not KL-matched")
    fig.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False)
    fig.suptitle("Do the readings order the ten tasks as the raters' tiers do? Tasks from reuse (top) to new (bottom)", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.88))
    save(fig, FIG, n="10 tasks, 3 fine-tune seeds each, one point per seed; consensus tier in brackets, 3 = reuse, 1 = new; "
         "* = task with a seed past the KL 'hi' rung (sentiment all three, Spanish seed 0), ringed; "
         "tau = Kendall tau-b of the 3-seed means with the tier, filed sign; arXiv and German left out, never reaching the gain floor", data=rows)


if __name__ == "__main__":
    main()
