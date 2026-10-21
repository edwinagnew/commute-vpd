"""Figures for sgd_matched: (1) concentration at a matched size of change: fraction of the log-odds gain removed by
reverting the top 1/10/100 pieces for plain SGD, Adam and SGD with momentum at matched rates, with E4's Adam lr 1e-5
on the same axes; (2) how each grid's size of change moves with lr, with the plain-SGD target and the match band.

Usage (repo root): uv run python experiments/sgd_matched/plot.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
E4 = HERE.parent / "concentration_test" / "results" / "concentration.json"
MATCH_TOL = 0.15
TOPS = ("1", "10", "100")


def fig_concentration(r: dict, e4: dict, suffix: str) -> None:
    arms = [("plain SGD lr 0.01", r["control"]["concentration"], r["control"]["end"]),
            (f"Adam lr {r['adam']['matched_lr']} (matched)", r["adam"]["concentration"], r["adam"]["end"]),
            (f"SGD momentum 0.9 lr {r['momentum']['matched_lr']} (matched)", r["momentum"]["concentration"],
             r["momentum"]["end"])]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = range(len(TOPS))
    for i, (label, conc, end) in enumerate(arms):
        ax.plot(x, [conc["fraction_removed"][n] for n in TOPS], "o-", color=f"C{i}",
                label=f"{label}: size {end['rel_norm_max']:.4f}, KL elsewhere {end['heldout_kl_other']:.4f}")
    adam = e4["adam_all"]
    ax.plot(x, [adam["fraction_removed"][n] for n in TOPS], "s--", color="C1", alpha=0.5,
            label=f"Adam lr 1e-5 (E4, unmatched): size {adam['runs']['1e-05']['curve'][-1]['rel_norm_max']:.4f}")
    ax.set_xticks(list(x), [f"top {n}" for n in TOPS])
    ax.set_yscale("log")
    ax.set_ylabel("fraction of log-odds gain removed")
    ax.set_title("Reverting the largest pieces, at a matched size of change")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / f"concentration{suffix}.png", dpi=150)


def fig_match(r: dict, suffix: str) -> None:
    target = r["control"]["end"]["rel_norm_max"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, arm, title in zip(axes, ("adam", "momentum"), ("Adam", "SGD, momentum 0.9")):
        runs = r[arm]["runs"]
        lrs = sorted(runs, key=float)
        ax.plot([float(lr) for lr in lrs], [runs[lr]["rel_norm_max"] for lr in lrs], "o-", color="C0")
        ax.axhline(target, color="k", label="plain SGD lr 0.01")
        ax.axhspan(target * (1 - MATCH_TOL), target * (1 + MATCH_TOL), color="k", alpha=0.1, label="match band")
        ax.axvline(float(r[arm]["matched_lr"]), color="C3", ls=":", label="matched lr")
        ax.set_xscale("log")
        ax.set_xlabel("lr")
        ax.set_ylabel("largest relative change per matrix, step 30")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / f"size_vs_lr{suffix}.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    suffix = "_smoke" if ap.parse_args().smoke else ""
    r = json.loads((HERE / "results" / f"sgd_matched{suffix}.json").read_text())
    FIGURES.mkdir(exist_ok=True)
    fig_concentration(r, json.loads(E4.read_text())["arms"], suffix)
    fig_match(r, suffix)
    print(f"wrote {FIGURES}")


if __name__ == "__main__":
    main()
