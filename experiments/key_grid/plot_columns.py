"""E237's one-figure summary (no GPU), from results/summary.json: every reading as one panel over the same five tasks,
ordered by the base model's prior of the trained answer (lowest first), seeds as dots at the common update size, the
random update of that size as a grey band. The readings not taken on these five tasks get a panel saying so.

    ~/commute-agents/fleet/cpu-run python3 experiments/key_grid/plot_columns.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute.figures import METHODS, column_dots, figure, save

HERE = Path(__file__).resolve().parent
SUMMARY = json.loads((HERE / "results" / "summary.json").read_text())
SHORT = {"cipher": "cipher", "prior_cipher": "prior-\nmatched\ncipher", "copy": "copy", "elicit": "mid", "rank1": "top"}
PANELS = [
    ("r", "re-gating share, log-odds", "regating"),
    ("kl_share", "re-gating share, KL", "regating"),
    ("prior", "log10 prior of trained answer", "prior"),
    ("size", "update size ||dW|| / ||W0||", "size"),
    ("erank", "effective rank of the update", "spectrum"),
    ("R_head_mlp", "R at heads and MLPs", "neuron"),
    ("R_mlp_attn", "R at MLP and attention", "neuron"),
    ("R_block", "R at blocks", "neuron"),
    ("R_topk", "R at top neurons, p97", "neuron_half"),
    ("R_vpd", "R at VPD subcomponents", "neuron_half"),
]
NOT_READ = ("Not read on these five tasks:\n\n- steering share\n- the causal test at VPD\n  with only the importance\n"
            "  values refit (S)\n- crosscoder")
N = ("3 seeds per task (dots), every task scaled to copy's update size; grey band: 3 random updates per task, Gaussian "
     "at each seed's per-matrix norms, same size, each draw a grey x. r has no band: a random update this small moves the answer by under "
     "1 log-odds, where r is undefined. The prior is one number per task.")


def panel(ax, name: str, ylabel: str, method: str) -> list[dict]:
    order = SUMMARY["x_order_by_prior"]
    if name == "prior":
        groups, bands = {t: [SUMMARY["log10_prior"][t]] for t in order}, None
    else:
        row = SUMMARY["table"][name]
        groups, bands = {t: row[t]["seeds"] for t in order}, {t: row[t]["band"] for t in order}
    column_dots(ax, groups, method=method, ylabel=ylabel, bands=bands)
    if bands:  # the draws themselves, so a band thinner than a line still shows
        draws = [(i, v) for i, t in enumerate(order) for v in SUMMARY["table"][name][t]["random"] if v is not None]
        r = METHODS["random"]
        ax.scatter([i + 0.18 for i, _ in draws], [v for _, v in draws], color=r.color, marker=r.marker, s=12, linewidths=0)
    ax.set_xticks(range(len(order)), [SHORT[t] for t in order], fontsize=6)
    rho = SUMMARY["spearman_vs_prior"].get(name)
    ax.set_title(ylabel + ("" if rho is None else f"\nrank order vs prior {rho:.1f}"), fontsize=7)
    ax.yaxis.label.set_size(6)
    ax.tick_params(axis="y", labelsize=6)
    band = {t: (bands or {}).get(t) or [None, None] for t in order}
    return [{"reading": name, "task": t, "seed": i, "value": v, "band_lo": band[t][0], "band_hi": band[t][1]}
            for t in order for i, v in enumerate(groups[t])]  # fmt: skip


def main() -> None:
    fig, axes = figure(size=(10.0, 8.0), nrows=3, ncols=4)
    flat, data = list(axes.flat), []
    for ax, (name, ylabel, method) in zip(flat, PANELS, strict=False):
        data += panel(ax, name, ylabel, method)
    flat[1].legend(fontsize=6, frameon=False, loc="lower right")
    for ax in flat[len(PANELS):]:
        ax.axis("off")
    flat[len(PANELS)].text(0.0, 1.0, NOT_READ, va="top", fontsize=8, transform=flat[len(PANELS)].transAxes)
    fig.suptitle("Every reading on the five key tasks at one update size, tasks ordered by the base model's prior (lowest "
                 "first):\nthe log-odds share splits the ciphers from the rest; the KL share sits in its random band; every R "
                 "sits below its random band", fontsize=8)
    save(fig, HERE / "figures" / "columns.png", n=N, data=data)


if __name__ == "__main__":
    main()
