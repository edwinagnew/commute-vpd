"""Figures for the lift test (round 2), from results/*.json. One question per figure, in its title.

Usage: uv run python experiments/lift_test/plot.py
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
# dataviz reference palette, categorical slots in fixed order (light surface)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "font.size": 10})


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print(f"wrote figures/{name}")


def _bars(ax, labels, series, colors, names, width=0.38):
    xs = range(len(labels))
    n = len(series)
    for i, (vals, c, nm) in enumerate(zip(series, colors, names)):
        off = (i - (n - 1) / 2) * width
        ax.bar([x + off for x in xs], vals, width * 0.92, color=c, label=nm, zorder=2)
    ax.set_xticks(list(xs), labels)
    ax.grid(axis="y", zorder=0)


def obstruction() -> None:
    """How much of each update can the alive subcomponents not express, against W0 itself and against noise?"""
    w = load("weighted.json")["updates"]
    rows = [("W₀ itself", "W0"), ("Adam\nstep 10", "A/dW_step10"), ("Adam\nstep 300", "A/dW_step300"),
            ("SGD\nstep 10", "SGD_lr0.3/dW_step10"), ("LoRA r8\nstep 10", "LoRA_r8/dW_step10")]
    labels = [r[0] for r in rows]
    act = [w[k]["pile_pooled_omega_act_non_saturated"] for _, k in rows]
    null = [w[k]["pile_pooled_null_act_non_saturated"] for _, k in rows]
    fro = [w[k]["pooled_omega_non_saturated"] for _, k in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    _bars(ax, labels, [fro, act, null], [MUTED, BLUE, GRID], ["parameter space (Frobenius ω)",
                                                             "weighted by real inputs (ω_act, Pile)",
                                                             "random-direction null for ω_act"], width=0.27)
    ax.patches[-1].set_edgecolor(MUTED)
    for p in ax.patches[2 * len(rows):]:
        p.set_edgecolor(MUTED)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share the alive subcomponents cannot express")
    ax.set_title("How much of an update lies outside what the alive subcomponents can express?")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    save(fig, "obstruction.png")


def projected() -> None:
    """Does the behaviour survive if the outside_share part of the update is dropped?"""
    p = load("projected.json")["updates"]
    rows = [("Adam\nstep 10", "A/dW_step10"), ("Adam\nstep 300", "A/dW_step300"), ("SGD\nstep 10", "SGD_lr0.3/dW_step10"),
            ("LoRA r8\nstep 10", "LoRA_r8/dW_step10")]
    labels = [r[0] for r in rows]
    full = [p[k]["full"]["windows"]["p_o_ft"] for _, k in rows]
    lift = [p[k]["liftable"]["windows"]["p_o_liftable"] for _, k in rows]
    obst = [p[k]["obstruction"]["windows"]["p_o_obstruction"] for _, k in rows]
    cap = [p[k]["capture_pile"] for _, k in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    _bars(a, labels, [full, lift, obst], [MUTED, BLUE, ORANGE], ["full update", "expressible part only",
                                                                 "inexpressible part only"], width=0.27)
    a.set_ylim(0, 1.05)
    a.set_ylabel("held-out P(o) after an emoticon")
    a.set_title("Emoticon behaviour", fontsize=10)
    a.legend(frameon=False, loc="lower left", fontsize=8)
    b.bar(range(len(rows)), cap, 0.5, color=BLUE, zorder=2)
    b.set_xticks(range(len(rows)), labels)
    b.set_ylim(0, 1)
    b.grid(axis="y", zorder=0)
    b.set_ylabel("share of the Pile effect kept")
    b.set_title("Effect on generic text (Pile)", fontsize=10)
    for x, v in enumerate(cap):
        b.text(x, v + 0.02, f"{v:.2f}", ha="center", color=INK, fontsize=9)
    fig.suptitle("Does the fine-tune still work with only the part the subcomponents can express?")
    save(fig, "projected.png")


def inside_vs_outside() -> None:
    """Headline: training inside the decomposition keeps it faithful; translating a finished update does not."""
    lf = load("lift_faith.json")["checkpoints"]
    pb = load("pullback2.json")["ratios"]
    steps = sorted(int(s) for s in lf)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, title in ((axes[0], "pgd_kl", "Adversarial (PGD) masks"),
                              (axes[1], "stochastic_kl", "Random (stochastic) masks")):
        ax.plot(steps, [lf[str(s)]["ratios"][metric]["lift"] for s in steps], "-o", color=ORANGE, lw=2, ms=6,
                label="least-squares split of the fine-tune")
        ax.plot(steps, [lf[str(s)]["ratios"][metric]["do_nothing"] for s in steps], "--o", color=MUTED, lw=2, ms=6,
                label="do-nothing: the fine-tune's update left in Δ")
        pts = {10: pb["pullback_step10"][metric], 30: pb["pullback_step30"][metric]}
        if (RESULTS / "pullback2_early.json").is_file():  # steps 1-10 of the same run, measured afterwards
            early = load("pullback2_early.json")["ratios"]
            pts.update({int(k.rsplit("step", 1)[1]): v[metric] for k, v in early.items() if int(k.rsplit("step", 1)[1]) < 10})
        ps = sorted(pts)
        ax.plot(ps, [pts[s] for s in ps], "-o", color=BLUE, lw=2, ms=6, label="component training")
        ax.axhline(1.5, color=MUTED, lw=1, ls=":")
        ax.text(30, 1.5, "pass ≤ 1.5", va="bottom", ha="right", color=MUTED, fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 10, 30], ["1", "2", "3", "5", "10", "30"])
        ax.set_xlabel("training step")
        ax.set_ylabel("faithfulness KL ÷ its value on W₀")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y")
    axes[1].set_ylim(0.8, 1.6)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("Component training stays faithful under both checks; the least-squares split fails the adversarial one")
    save(fig, "inside_vs_outside.png")


def pullback_behaviour() -> None:
    """Does training inside the decomposition learn the behaviour as fast, and as cleanly, as a normal fine-tune?"""
    pb = load("pullback2.json")
    curve = pb["runs"]["1e-05"]["curve"]
    rerun = {int(k): v for k, v in load("pullback.json")["rerun"]["evals"].items()}
    rs = sorted(rerun)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key_c, key_r, ylab in ((a, "heldout_p_o", "p_o", "held-out P(o)"),
                                   (b, "heldout_kl_other", "kl_other", "KL to W₀ elsewhere in the window (nats)")):
        ax.plot([c["step"] for c in curve], [c[key_c] for c in curve], "-", color=BLUE, lw=2,
                label="component training, lr 1e-5")
        ax.plot(rs, [rerun[s][key_r] for s in rs], "-o", color=MUTED, lw=2, ms=6, label="normal fine-tune (Adam 3e-5)")
        ax.set_xlabel("training step")
        ax.set_ylabel(ylab)
        ax.grid(axis="y")
    a.set_ylim(0, 1.05)
    a.set_title("Behaviour learned", fontsize=10)
    b.set_title("Off-target damage", fontsize=10)
    a.legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Component training learns the behaviour as fast as a normal fine-tune, with less damage elsewhere")
    save(fig, "pullback_behaviour.png")


if __name__ == "__main__":
    obstruction()
    projected()
    inside_vs_outside()
    pullback_behaviour()
