"""Figures for sgd_grid: does plain SGD learn within E4's off-target limit at some learning rate (step-30 held-out
P(o) and KL elsewhere against lr, E4's grid beside it, Adam as the reference), and is its update concentrated (fraction
of the log-odds gain removed by reverting the top k, beside E4's Adam arms)?

Usage (repo root): uv run python experiments/sgd_grid/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
E4 = HERE.parent / "concentration_test" / "results" / "concentration.json"
KL_OTHER_MAX, P_MIN = 0.110, 0.9


def last(runs: dict) -> tuple[list[float], list[float], list[float]]:
    lrs = sorted(runs, key=float)
    ends = [runs[lr]["curve"][-1] for lr in lrs]
    return [float(lr) for lr in lrs], [c["heldout_p_o"] for c in ends], [c["heldout_kl_other"] for c in ends]


def fig_lr(new: dict, e4: dict) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    adam = e4["adam_all"]["runs"]["1e-05"]["curve"][-1]
    for runs, label, style in ((new["runs"], "plain SGD, this run", "o-"), (e4["sgd_all"]["runs"], "plain SGD, E4's grid", "s--")):
        lr, p, kl = last(runs)
        a.plot(lr, p, style, color="C0", label=label)
        b.plot(lr, kl, style, color="C0", label=label)
    a.axhline(adam["heldout_p_o"], color="C1", label="Adam, lr 1e-5 (E4)")
    b.axhline(adam["heldout_kl_other"], color="C1", label="Adam, lr 1e-5 (E4)")
    a.axhline(P_MIN, color="k", ls=":", label="learning threshold")
    b.axhline(KL_OTHER_MAX, color="k", ls=":", label="off-target limit")
    a.set(xscale="log", xlabel="SGD learning rate", ylabel="held-out P(o) at the trigger, step 30", ylim=(0, 1.05))
    b.set(xscale="log", yscale="log", xlabel="SGD learning rate", ylabel="held-out KL elsewhere, step 30")
    a.legend(fontsize=8, loc="lower right")
    b.legend(fontsize=8, loc="upper left")
    fig.suptitle("Plain SGD learns the behaviour within the limit at lr 0.003 to 0.1; E4's grid started above that")
    fig.tight_layout()
    fig.savefig(FIGURES / "lr_grid.png", dpi=130)


def fig_concentration(new: dict, e4: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = ["1", "10", "100"]
    arms = ((new["concentration"]["fraction_removed"], f"plain SGD, lr {new['selected_lr']} (this run)", "C0"),
            (e4["adam_all"]["fraction_removed"], "Adam, all pieces free (E4)", "C1"),
            (e4["top10"]["fraction_removed"], "Adam, only 10 pieces free (E4)", "C2"))
    for frac, label, colour in arms:
        ax.plot([int(k) for k in ks], [frac[k] for k in ks], "o-", color=colour, label=label)
    ax.axhline(0.5, color="k", ls=":", label="filed 'concentrated' line")
    ax.axhline(0.2, color="k", ls="--", lw=0.8, label="filed 'diffuse' line")
    ax.set(xscale="log", xlabel="pieces reverted (largest single effects first)",
           ylabel="share of the learned log-odds removed", ylim=(0, 1.05))
    ax.legend(fontsize=8)
    ax.set_title("SGD's update is ten times less diffuse than Adam's, but not concentrated", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "concentration.png", dpi=130)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    new = json.loads((HERE / "results" / "sgd_grid.json").read_text())
    e4 = json.loads(E4.read_text())["arms"]
    fig_lr(new, e4)
    fig_concentration(new, e4)


if __name__ == "__main__":
    main()
