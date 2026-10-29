"""The research log's four figures for E2, redrawn from saved results with plain labels (no model).

  log_adversarial.png  adversarial ratio against the training step: the least-squares split of the fine-tune and
                       do-nothing (results/lift_faith.json), component training (results/pullback2_early.json for
                       steps 1 to 5, results/pullback2.json for steps 10 and 30).
  log_behaviour.png    held-out P(target) and KL elsewhere in the window against the step: component training at
                       lr 1e-5 (results/pullback2.json, run "1e-05") and the normal fine-tune (results/pullback.json,
                       "rerun").
  log_outside.png      share of an update outside the subcomponents, counted over weights and weighted by real Pile
                       inputs, with the random-update null, for the original weights, the Adam and the SGD fine-tune
                       at step 10 (results/weighted.json).
  log_parts.png        held-out P(target) for the full update, its inside part and its outside part alone, Adam and
                       SGD fine-tunes at step 10 (results/projected.json).

Usage (repo root):  python experiments/lift_test/plot_log.py
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from plot import BLUE, GRID, INK, MUTED, ORANGE, RESULTS, load, save

STEPS_TICKS = [1, 2, 3, 5, 10, 30]
# (label, key in weighted.json / projected.json)
ADAM10 = ("Adam fine-tune,\nstep 10", "A/dW_step10")
SGD10 = ("SGD fine-tune,\nstep 10", "SGD_lr0.3/dW_step10")


def _log_steps(ax: Any) -> None:
    ax.set_xscale("log")
    ax.set_xticks(STEPS_TICKS, [str(s) for s in STEPS_TICKS])
    ax.minorticks_off()
    ax.set_xlabel("training step on the emoticon task")
    ax.grid(axis="y")


def component_training_ratios() -> dict[int, float]:
    late = load("pullback2.json")["ratios"]
    pts = {10: late["pullback_step10"]["pgd_kl"], 30: late["pullback_step30"]["pgd_kl"]}
    if (RESULTS / "pullback2_early.json").is_file():  # steps 1 to 10 of the same run, measured afterwards
        for k, v in load("pullback2_early.json")["ratios"].items():
            s = int(k.rsplit("step", 1)[1])
            if s < 10:
                pts[s] = v["pgd_kl"]
    return dict(sorted(pts.items()))


def fig_adversarial() -> None:
    lf = load("lift_faith.json")["checkpoints"]
    steps = sorted(int(s) for s in lf)
    ct = component_training_ratios()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    lines = (
        (steps, [lf[str(s)]["ratios"]["pgd_kl"]["lift"] for s in steps], ORANGE, "-",
         "least-squares split\nof the finished fine-tune"),
        (steps, [lf[str(s)]["ratios"]["pgd_kl"]["do_nothing"] for s in steps], MUTED, "--",
         "do-nothing:\nupdate left in Δ"),
        (list(ct), list(ct.values()), BLUE, "-", "component training"),
    )
    for xs, ys, c, ls, lab in lines:
        ax.plot(xs, ys, ls, marker="o", color=c, lw=2, ms=5)
        ax.text(xs[-1] * 1.08, ys[-1], lab, color=c, va="center", fontsize=8.5)
    ax.axhline(1.0, color=INK, lw=0.8)
    ax.axhline(1.5, color=MUTED, lw=1, ls=":")
    ax.text(1.0, 1.5, " pass line 1.5", va="bottom", color=MUTED, fontsize=8)
    _log_steps(ax)
    ax.set_xlim(0.9, 80)
    ax.set_ylim(0.6, 6)
    ax.set_ylabel("adversarial ratio\n(1 = as on the original model)")
    ax.set_title("On Pile text, only component training keeps the adversarial ratio near 1", fontsize=10)
    save(fig, "log_adversarial.png")


def fig_behaviour() -> None:
    curve = load("pullback2.json")["runs"]["1e-05"]["curve"]
    ft = {int(k): v for k, v in load("pullback.json")["rerun"]["evals"].items()}
    fs = sorted(ft)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, kc, kf, ylab in ((a, "heldout_p_o", "p_o", "held-out probability of the target o"),
                             (b, "heldout_kl_other", "kl_other",
                              "change at the other positions of the window\n(KL to the original model, nats)")):
        ax.plot([c["step"] for c in curve], [c[kc] for c in curve], "-", color=BLUE, lw=2,
                label="component training (every step)")
        ax.plot(fs, [ft[s][kf] for s in fs], "-o", color=MUTED, lw=2, ms=5,
                label="normal fine-tune (steps 1, 2, 3, 5, 10, 30)")
        _log_steps(ax)
        ax.set_ylabel(ylab)
    a.set_ylim(0, 1.05)
    b.set_ylim(0, None)
    a.set_title("Both learn the behaviour; the fine-tune a step or two sooner", fontsize=10)
    b.set_title("Component training changes the rest of the window less", fontsize=10)
    a.legend(frameon=False, loc="lower right", fontsize=8)
    save(fig, "log_behaviour.png")


def fig_outside() -> None:
    w = load("weighted.json")["updates"]
    rows = (("the original\nweights", "W0"), ADAM10, SGD10)
    bars = (("pooled_omega_non_saturated", "every weight direction counted alike", MUTED, None),
            ("pile_pooled_omega_act_non_saturated", "weighted by Pile-text inputs", BLUE, None),
            ("pile_pooled_null_act_non_saturated", "a random update of the same size, weighted by Pile-text inputs",
             "none", MUTED))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (_, key) in enumerate(rows):
        for j, (metric, lab, c, edge) in enumerate(bars):
            v = w[key][metric]
            x = i + (j - 1) * width
            ax.bar(x, v, width * 0.9, color=c, edgecolor=edge, hatch="//" if edge else None,
                   label=lab if i == 0 else None, zorder=2)
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("share of the update the subcomponents\ncannot express (left in Δ)")
    ax.set_title("Weighted by Pile text, the fine-tunes sit between the original weights and a random update", fontsize=10)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "log_outside.png")


def fig_parts() -> None:
    p = load("projected.json")["updates"]
    rows = (ADAM10, SGD10)
    bars = (("full", "p_o_ft", "the full update", MUTED),
            ("liftable", "p_o_liftable", "the inside part only (outside part dropped)", BLUE),
            ("obstruction", "p_o_obstruction", "the outside part only", ORANGE))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (_, key) in enumerate(rows):
        for j, (part, metric, lab, c) in enumerate(bars):
            v = p[key][part]["windows"][metric]
            x = i + (j - 1) * width
            ax.bar(x, v, width * 0.9, color=c, label=lab if i == 0 else None, zorder=2)
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("held-out probability of the target o")
    ax.set_title("At step 10 the behaviour survives without the outside part, and partly on it alone", fontsize=10)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=1)
    save(fig, "log_parts.png")


if __name__ == "__main__":
    fig_adversarial()
    fig_behaviour()
    fig_outside()
    fig_parts()
