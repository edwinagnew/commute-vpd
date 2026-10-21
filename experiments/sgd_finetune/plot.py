"""Figures for sgd_finetune: (1) the SGD fine-tune rate grid against E4's rule, Adam's step-10 point beside it;
(2) the Pile PGD ratio of every state for the SGD and Adam fine-tunes, with the 1.5 line and the same-job W0 band;
(3) the four blocks of both fine-tunes with the outside-both null.

Usage (repo root): uv run python experiments/sgd_finetune/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
P_MIN, KL_OTHER_MAX, LINE = 0.9, 0.110, 1.5
ADAM = {"p_o": 0.9993, "kl_other": 0.0554}  # E16, ft step 10
OPS = ("nothing", "least_squares", "adjoint", "component_training")
BLOCKS = ("in_in", "in_out", "out_in", "out_out")


def fig_rate(rate: dict) -> None:
    lrs = sorted(rate["finetune"], key=float)
    ev = [rate["finetune"][lr] for lr in lrs]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    x = [float(lr) for lr in lrs]
    a.plot(x, [e["p_o"] for e in ev], "o-", label="SGD fine-tune")
    b.plot(x, [e["kl_other"] for e in ev], "o-", label="SGD fine-tune")
    a.axhline(ADAM["p_o"], color="C1", label="Adam fine-tune 3e-5 (E16)")
    b.axhline(ADAM["kl_other"], color="C1", label="Adam fine-tune 3e-5 (E16)")
    a.axhline(P_MIN, color="k", ls=":", label="learning threshold")
    b.axhline(KL_OTHER_MAX, color="k", ls=":", label="off-target limit")
    a.set(xscale="log", xlabel="SGD learning rate", ylabel="held-out P(o), step 10", ylim=(0, 1.05))
    b.set(xscale="log", yscale="log", xlabel="SGD learning rate", ylabel="held-out KL elsewhere, step 10")
    for ax in (a, b):
        ax.legend(fontsize=8)
    fig.suptitle(f"SGD fine-tune rate grid; selected {rate['selected_lr']} (passed: {rate['passed']})")
    fig.tight_layout()
    fig.savefig(FIGURES / "rate.png", dpi=130)


def fig_pile(sp: dict) -> None:
    f = sp["W0"]["pgd_floor"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, n in enumerate(("sgd", "adam")):
        st = sp["states"][n]
        xs = [j for j, op in enumerate(OPS) if op in st]
        ax.plot([x + 0.15 * (i - 0.5) for x in xs], [st[OPS[j]]["pile"]["pgd_kl"] for j in xs], "o", ms=8,
                color=f"C{i}", label=f"{n.upper() if n == 'sgd' else 'Adam'} fine-tune")
    ax.axhline(LINE, color="k", ls=":", label="pass line 1.5")
    ax.axhspan(1 / (1 + f), 1 + f, color="grey", alpha=0.2, label=f"W0 adversary-seed spread (f {f:.2f})")
    ax.set(xticks=range(len(OPS)), xticklabels=["do-nothing", "least-squares", "adjoint", "component training"],
           yscale="log", ylabel="Pile PGD KL / W0's (same job)")
    ax.legend(fontsize=8)
    fig.suptitle("Adversarial Pile ratio per state, SGD vs Adam fine-tune at step 10")
    fig.tight_layout()
    fig.savefig(FIGURES / "pile.png", dpi=130)


def fig_blocks(sp: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.27
    for i, n in enumerate(("sgd", "adam")):
        b = sp["blocks"][n]
        ax.bar([j + w * (i - 1) for j in range(4)], [b[k] for k in BLOCKS], w, color=f"C{i}",
               label=f"{'SGD' if n == 'sgd' else 'Adam'} fine-tune")
    b = sp["blocks"]["adam"]
    ax.bar([j + w for j in range(4)], [b[f"null_{k}"] for k in BLOCKS], w, color="grey", label="random update (null, Adam's pooling)")
    ax.set(xticks=range(4), xticklabels=["inside both", "write side only", "read side only", "outside both"],
           ylabel="share of input-weighted energy")
    ax.legend(fontsize=8)
    fig.suptitle("Where each fine-tune's update sits relative to the subcomponents (step 10)")
    fig.tight_layout()
    fig.savefig(FIGURES / "blocks.png", dpi=130)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    fig_rate(json.loads((RESULTS / "rate.json").read_text()))
    if (RESULTS / "split.json").is_file():
        sp = json.loads((RESULTS / "split.json").read_text())
        if "states" in sp:
            fig_pile(sp)
            fig_blocks(sp)


if __name__ == "__main__":
    main()
