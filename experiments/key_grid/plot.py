"""E237's figures (no GPU), from results/summary.json (score.py): one graph per reading, the five tasks on the x-axis in
order of the base model's prior of the trained answer (lowest first), the three seeds as dots at the common update
size, and the random update of the same size as a grey band (min to max over the task's three draws). Plus the prior.

    ~/commute-agents/fleet/cpu-run python3 experiments/key_grid/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute.figures import column_dots, figure, save

HERE = Path(__file__).resolve().parent
SUMMARY = json.loads((HERE / "results" / "summary.json").read_text())
FIGS = HERE / "figures"
LABELS = {
    "r": ("re-gating share, log-odds form (r)", "regating"),
    "kl_share": ("re-gating share, KL form", "regating"),
    "size": ("update size, ||dW|| / ||W0||", "size"),
    "erank": ("update's effective rank", "spectrum"),
    "R_head_mlp": ("R at heads and MLPs", "neuron"),
    "R_mlp_attn": ("R at MLP and attention per block", "neuron"),
    "R_block": ("R at blocks", "neuron"),
    "R_topk": ("R at top neurons (p97, upper tail)", "neuron_half"),
    "R_vpd": ("R at VPD subcomponents", "neuron_half"),
}
N = "3 seeds per task (dots); grey band: 3 random updates per task, Gaussian at each seed's per-matrix norms, same size"


def one(name: str) -> None:
    order, row = SUMMARY["x_order_by_prior"], SUMMARY["table"][name]
    ylabel, method = LABELS[name]
    fig, ax = figure()
    column_dots(ax, {t: row[t]["seeds"] for t in order}, method=method, ylabel=ylabel, bands={t: row[t]["band"] for t in order})
    if any(row[t]["band"] for t in order):
        ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel("task, lowest prior of the trained answer first")
    rho = SUMMARY["spearman_vs_prior"][name]
    ax.set_title(f"{ylabel} at one update size" + (f" (rank order vs prior {rho:.1f})" if rho is not None else ""), fontsize=8)
    data = [{"task": t, "seed": i, "value": v, "band_lo": (row[t]["band"] or [None])[0], "band_hi": (row[t]["band"] or [None, None])[1]}
            for t in order for i, v in enumerate(row[t]["seeds"])]  # fmt: skip
    save(fig, FIGS / f"{name}.png", n=N, data=data)


def prior() -> None:
    order, lp = SUMMARY["x_order_by_prior"], SUMMARY["log10_prior"]
    fig, ax = figure()
    column_dots(ax, {t: [lp[t]] for t in order}, method="prior", ylabel="log10 prior of the trained answer")
    ax.set_xlabel("task")
    ax.set_title("The x-axis order: the base model's prior of each task's trained answer", fontsize=8)
    save(fig, FIGS / "prior.png", n="one value per task (experiments/prior_vs_share/results/stage1.json)", data=[{"task": t, "log10_prior": lp[t]} for t in order])


if __name__ == "__main__":
    for name in LABELS:
        one(name)
    prior()
