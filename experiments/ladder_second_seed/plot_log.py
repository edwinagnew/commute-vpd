"""The research log's figures for E26, redrawn from saved results with plain labels (no model, no runs).

  log_verdicts.png  the two common-target tasks: adversarial ratio of SGD and Adam component training at steps 100
                    and 1000, second batch order at both steps and the first order (rerun here) at step 1000, against
                    this job's pass and fail lines.
  log_rare.png      the rare-target task: each state's adversarial ratio against the target's held-out probability,
                    with the region that counts as a pass (faithful and target kept).
  log_size.png      the common-target tasks at step 1000: adversarial and random-mask ratios against the size of the
                    weight change, per optimiser.
  log_control.png   the first order rerun here against E21's reading at step 1000, per task and optimiser.

Reads results/ladder_second_seed_<task>.json (states, W0 floor, G1 control).
Usage (repo root):  python experiments/ladder_second_seed/plot_log.py
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from plot import FIGURES, LINE, TASKS, load

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SGD_C, ADAM_C = "#eb6834", "#2a78d6"
PASS_C, FAIL_C, MID_C = "#e3f2e6", "#fbe4dc", "#efefec"
NAME = {"r0_t352_v310": "' it' → ' is'", "r1_t534_v369": "' which' → ' was'", "r2_t275_v2667": "' in' → ' Fig'"}
COMMON, RARE = TASKS[:2], TASKS[2]
ORDERS = (("seed1", "o", "second batch order"), ("seed0", "D", "first batch order, rerun here"))


def arms(d: dict[str, Any]) -> tuple[tuple[str, str, str], ...]:
    return ((f"sgd_{d['sgd_lr']:g}", f"SGD, rate {d['sgd_lr']:g}", SGD_C), ("adam_1e-05", "Adam, rate 1e-5", ADAM_C))


def lines(d: dict[str, Any]) -> tuple[float, float]:
    f = d["W0"]["pgd_floor"]
    return LINE / (1 + f), LINE * (1 + f)


def bands(ax: Any, d: dict[str, Any], top: float, label: bool) -> None:
    lo, hi = lines(d)
    ax.axhspan(0, lo, color=PASS_C, zorder=0, label=f"pass (ratio ≤ {lo:.2f})" if label else None)
    ax.axhspan(lo, hi, color=MID_C, zorder=0, label="too close to call" if label else None)
    ax.axhspan(hi, top, color=FAIL_C, zorder=0, label=f"fail (ratio ≥ {hi:.2f})" if label else None)


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_verdicts(runs: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, t in zip(axes, COMMON, strict=True):
        d, top = runs[t], 3.5
        bands(ax, d, top, ax is axes[0])
        for s_i, step in enumerate(("100", "1000")):
            for a_i, (arm, lab, col) in enumerate(arms(d)):
                for o_i, (seed, mk, olab) in enumerate(ORDERS):
                    st = d["states"][seed][arm].get(step)
                    if st is None:
                        continue
                    x, y = s_i * 2 + a_i * 0.8 + o_i * 0.25, st["pile"]["pgd_kl"]
                    ax.scatter(x, y, s=60, marker=mk, color=col, edgecolors=INK, lw=0.5, zorder=3,
                               label=f"{lab}, {olab}" if ax is axes[0] and step == "1000" else None)
                    dx, ha = (0.1, "left") if o_i else (-0.1, "right")
                    ax.text(x + dx, y, f"{y:.2f}", fontsize=8, va="center", ha=ha)
                    ax.text(x + dx, y - 0.1, f"target p {st['heldout']['p_o']:.2f}", fontsize=6.5,
                            va="center", ha=ha, color=MUTED)
        ax.set_xticks([0.5, 2.5], ["step 100", "step 1,000"])
        ax.set_xlim(-0.5, 3.6)
        ax.set_ylim(0.8, top)
        ax.set_title(f"task: {NAME[t]}", fontsize=10)
        style(ax)
    axes[0].set_ylabel("adversarial ratio (1 = as at the original model)")
    fig.suptitle("With a second batch order, Adam still passes at both steps and SGD still passes at step 100 "
                 "and fails at step 1,000\n(grey under each point: the target's held-out probability; a pass also needs 0.9 or more)",
                 fontsize=10.5)
    fig.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2,
               fontsize=8, frameon=False)
    save(fig, "log_verdicts.png")


def fig_rare(d: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    lo, _ = lines(d)
    ax.fill_between([0.8, lo], 0.9, 1.02, color=PASS_C, zorder=0, lw=0)
    ax.text(0.83, 1.005, "pass: faithful and\ntarget kept", fontsize=8, va="top", color="#2c6b3a")
    ax.axhline(0.9, color=MUTED, lw=0.7, ls=":")
    ax.axvline(lo, color=MUTED, lw=0.7, ls=":")
    for arm, lab, col in arms(d):
        for seed, mk, olab in ORDERS:
            for step, st in d["states"][seed][arm].items():
                x, y = st["pile"]["pgd_kl"], st["heldout"]["p_o"]
                first = seed == "seed0"
                ax.scatter(x, y, s=60, marker=mk, color=col, edgecolors=INK, lw=0.5, zorder=3,
                           label=f"{lab}, {olab}{' (step 1,000 only)' if first else ''}" if step == "1000" else None)
                if not first:
                    ax.annotate(f"step {int(step):,}", (x, y), xytext=(6, -3), textcoords="offset points",
                                fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xticks([1, 1.4, 2, 5, 10, 20, 50], ["1", "1.4", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_xlim(0.8, 90)
    ax.set_ylim(0.75, 1.02)
    ax.set_xlabel("adversarial ratio (log scale)")
    ax.set_ylabel("target's held-out probability")
    ax.set_title(f"Rare-target task {NAME[RARE]}: no state is both faithful and keeps the target", fontsize=10)
    style(ax)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    save(fig, "log_rare.png")


def fig_size(runs: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axhline(1, color=MUTED, lw=0.8, ls="--")
    ax.text(0.0005, 1.04, "original model", fontsize=7.5, color=MUTED)
    for t in COMMON:
        for arm, lab, col in arms(runs[t]):
            for seed, _, _ in ORDERS:
                st = runs[t]["states"][seed][arm]["1000"]
                x = st["heldout"]["rel_norm_max"]
                first = t == COMMON[0] and seed == "seed1"
                ax.scatter(x, st["pile"]["pgd_kl"], s=55, color=col, edgecolors=INK, lw=0.5, zorder=3,
                           label=f"{lab}: adversarial ratio" if first else None)
                ax.scatter(x, st["pile"]["stochastic_kl"], s=55, facecolors="none", edgecolors=col, lw=1.2,
                           zorder=3, label=f"{lab}: random-mask ratio" if first else None)
    ax.set_xlim(0, 0.025)
    ax.set_ylim(0.8, 3.5)
    ax.set_xlabel("size of the weight change at step 1,000\n(largest relative change over the 24 weight matrices)")
    ax.set_ylabel("ratio to the original model")
    ax.set_title("Same largest per-matrix change; only the adversary separates SGD from Adam\n"
                 "(both common-target tasks, both batch orders, step 1,000)", fontsize=10)
    style(ax)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    save(fig, "log_size.png")


def fig_control(runs: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    labels, vals, cols = [], [], []
    for t in TASKS:
        for arm, lab, col in arms(runs[t]):
            g = runs[t]["G1"][arm]
            labels.append(f"{NAME[t]}, {lab}")
            vals.append(100 * g["rel"])
            cols.append(col)
    ys = range(len(vals))[::-1]
    ax.barh(list(ys), vals, color=cols, height=0.6)
    for y, v in zip(ys, vals, strict=True):
        ax.text(v + (0.6 if v >= 0 else -0.6), y, f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right",
                fontsize=8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(25, color=MUTED, lw=0.8, ls="--")
    ax.text(25.5, len(vals) - 0.6, "tolerance, 25%", fontsize=7.5, color=MUTED)
    ax.set_yticks(list(ys), labels, fontsize=8)
    ax.set_xlim(-8, 36)
    ax.set_xlabel("first batch order rerun here, minus E21's reading, as a share of E21's (step 1,000)")
    ax.set_title("The rerun first order matches E21 within 6% on the common-target tasks;\n"
                 "SGD on the rare target reads 28% high, still a fail", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "log_control.png")


def main() -> None:
    runs = load()
    fig_verdicts(runs)
    fig_rare(runs[RARE])
    fig_size(runs)
    fig_control(runs)


if __name__ == "__main__":
    main()
