"""Figures for the transport test, from results/*.json. One question per figure, in its title.

Usage: python experiments/transport_test/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
# dataviz reference palette, categorical slots 1-4 (light surface)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e6e5e1", "font.size": 10})


def load(name: str) -> dict | None:
    p = RESULTS / name
    return json.loads(p.read_text()) if p.is_file() else None


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print(f"wrote figures/{name}")


def capture_vs_k(arm: str) -> None:
    tr = load(f"transport_{arm}.json")
    if not tr:
        return
    step = str(tr["scored_checkpoint"])
    r = tr["steps"][step]
    ks = sorted(int(k) for k in r["T1"])
    top = [r["T1"][str(k)]["write"]["primary"]["capture"] for k in ks]
    rnd = [[x["primary"]["capture"] for x in r["T1"][str(k)].get("random_write", [])] for k in ks]
    read = [r["T1"][str(k)]["read"]["primary"]["capture"] if "read" in r["T1"][str(k)] else None for k in ks]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ks, top, "-o", color=BLUE, lw=2, ms=6, label="top-k by Frobenius contribution, write edits")
    if any(read):
        ax.plot(ks, read, "--s", color=VIOLET, lw=1.5, ms=5, label="top-k, read edits")
    means = [sum(v) / len(v) if v else float("nan") for v in rnd]
    ax.plot(ks, means, "-o", color=ORANGE, lw=2, ms=6, label="k random alive subcomponents (mean)")
    for k, v in zip(ks, rnd):
        if v:
            ax.vlines(k, min(v), max(v), color=ORANGE, alpha=0.5)
    ax.axhline(r["T0_functional"]["primary"]["capture"], color=AQUA, ls=":", lw=1.5, label="T0: rescale every alive subcomponent")
    ax.axhline(0.5, color=MUTED, ls="--", lw=1, label="pass threshold (0.5)")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k: subcomponents allowed to change")
    ax.set_ylabel("functional capture at the fine-tuned positions")
    ax.set_ylim(min(-0.1, min(means + top) - 0.05), 1.05)
    where = "emoticon position" if arm == "A" else "all positions"
    ax.set_title(f"Arm {arm}, step {step}: does the update land on a few existing subcomponents?\n(capture = 1 - KL(ft||T)/KL(ft||W0) on held-out data, {where})", fontsize=10)
    ax.grid(True, axis="y")
    ax.legend(fontsize=8, loc="best")
    save(fig, f"capture_vs_k_arm{arm}.png")


def frobenius_fraction() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    any_data = False
    for arm, color in (("A", BLUE), ("B", AQUA)):
        tr = load(f"transport_{arm}.json")
        if not tr:
            continue
        any_data = True
        steps = sorted(int(s) for s in tr["steps"])
        alive = [tr["steps"][str(s)]["T0_frobenius"]["fraction_alive_pooled"] for s in steps]
        full = [tr["steps"][str(s)]["T0_frobenius"]["fraction_all_pooled"] for s in steps]
        ax.plot(steps, alive, "-o", color=color, lw=2, label=f"arm {arm}: alive subcomponents")
        ax.plot(steps, full, "--s", color=color, lw=1.2, ms=4, alpha=0.7, label=f"arm {arm}: all 38 912")
        sc = str(tr["scored_checkpoint"])
        n = tr["steps"][sc].get("arm_N_fraction_alive_pooled")
        if n:
            ax.hlines(sum(n) / len(n), min(steps), max(steps), color=ORANGE, ls=":", lw=1.5,
                      label=f"arm N (random direction matched to arm {arm})" if arm == "A" else None)
    if not any_data:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("fine-tune step")
    ax.set_ylabel("Frobenius fraction of delta-W in the span (pooled)")
    ax.set_title("How much of the update lies in the span of the existing rank-1 subcomponents?", fontsize=10)
    ax.grid(True, which="both", axis="y")
    ax.legend(fontsize=8)
    save(fig, "frobenius_fraction.png")


def faithfulness() -> None:
    cal = load("calibration.json")
    rows = []
    for arm in ("A", "B"):
        fa = load(f"faithfulness_{arm}.json")
        if not fa:
            continue
        sc = str(fa["scored_checkpoint"])
        for variant, f in fa["steps"][sc].items():
            rows.append((f"arm {arm} s{sc}\n{variant}", f["ratio_to_calibration"]["stochastic_kl"], f["ratio_to_calibration"]["pgd_kl"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = range(len(rows))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [r[1] for r in rows], w, color=BLUE, label="stochastic reconstruction KL")
    ax.bar([i + w / 2 for i in x], [r[2] for r in rows], w, color=ORANGE, label="adversarial (PGD) KL")
    ax.axhline(1.0, color=MUTED, lw=1)
    ax.axhline(2.0, color=MUTED, ls="--", lw=1, label="pass threshold (2x)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("ratio to the released decomposition on W0")
    paper = f"paper: {cal['paper']['stochastic_kl']:.3f} / {cal['paper']['pgd_kl']:.3f}; ours on W0: {cal['faithfulness_W0']['stochastic_kl']:.3f} / {cal['faithfulness_W0']['pgd_kl']:.3f}" if cal else ""
    ax.set_title(f"Is the transported decomposition still faithful at W0 + delta-W?\n({paper})", fontsize=10)
    ax.grid(True, axis="y")
    ax.legend(fontsize=8)
    save(fig, "faithfulness_ratios.png")


def finetune_curves() -> None:
    fa, fb = load("finetune_A.json"), load("finetune_B.json")
    if not (fa or fb):
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    if fa:
        steps = sorted(int(s) for s in fa["evals"])
        axes[0].plot(steps, [fa["evals"][str(s)]["p_o"] for s in steps], "-o", color=BLUE, lw=2, label="held-out P(o) at the emoticon position")
        axes[0].plot(steps, [fa["evals"][str(s)]["kl_other"] for s in steps], "--s", color=ORANGE, lw=1.5, label="KL to W0 on the other 40 positions")
        axes[0].plot(steps, [fa["evals"][str(s)]["rel_norm_max"] for s in steps], ":^", color=AQUA, lw=1.5, label="max ||dW||/||W|| over matrices")
        axes[0].axhline(0.5, color=MUTED, ls="--", lw=1)
        axes[0].set_title("Arm A (emoticon -> o): how big is the update?", fontsize=10)
    if fb:
        steps = sorted(int(s) for s in fb["evals"])
        axes[1].plot(steps, [fb["evals"][str(s)]["kl_to_ref"] for s in steps], "--s", color=ORANGE, lw=1.5, label="KL to W0 on held-out rows")
        axes[1].plot(steps, [fb["evals"][str(s)]["rel_norm_max"] for s in steps], ":^", color=AQUA, lw=1.5, label="max ||dW||/||W|| over matrices")
        axes[1].set_title("Arm B (continued pretraining): how big is the update?", fontsize=10)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("fine-tune step")
        ax.grid(True, axis="y")
        ax.legend(fontsize=8)
    save(fig, "finetune_curves.png")


if __name__ == "__main__":
    capture_vs_k("A")
    capture_vs_k("B")
    frobenius_fraction()
    faithfulness()
    finetune_curves()
