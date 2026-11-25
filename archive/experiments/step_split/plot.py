"""Figures for step_split (E19) from results/step_split.json.

  behaviour_in_delta.png  how much of the trained behaviour runs through Delta? The route share through Delta at steps
                          10, 100, 1000 for the step split, the least-squares split and do-nothing, with the filed
                          thresholds of the step split's prediction marked.
  pile_pgd.png            is the step-split state faithful under the adversary? Pile PGD ratio to W0 for the same three
                          states, with the W0 noise band and the 1 + 3f line.
  delta_share.png         does carrying the bases forward change what the components take? Delta's share of the update's
                          input-weighted energy for the step split, the same with the bases held at W0's, and the
                          one-shot adjoint split of the finished update plus its remainder.

Usage (repo root):  uv run python experiments/step_split/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
STEPS = (10, 100, 1000)
GREY = "#7f7f7f"
# do-nothing and least-squares keep split_comparison's colours; the one-shot adjoint split keeps its adjoint colour
ARMS = {"step_split": ("step split", "#7a4fc9"), "least_squares": ("least-squares split", "#eb6834"),
        "nothing": ("do-nothing", "#2a78d6")}


def load() -> dict[str, Any]:
    return json.loads((HERE / "results" / "step_split.json").read_text())


def per_step(out: dict[str, Any], *keys: str) -> list[float]:
    vals = []
    for s in STEPS:
        v: Any = out["states"][str(s)]
        for k in keys:
            v = v[k]
        vals.append(float(v))
    return vals


def axes_steps(ax: Axes, ylabel: str, title: str) -> None:
    ax.set_xscale("log")
    ax.set_xticks(STEPS, [str(s) for s in STEPS])
    ax.set_xlabel("fine-tune step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def behaviour_in_delta(out: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for arm, (name, col) in ARMS.items():
        v = per_step(out, arm, "routes", "shares", "e_delta")
        ax.plot(STEPS, v, "o-", color=col, label=name, lw=2)
        for s, y in zip(STEPS, v, strict=True):
            ax.annotate(f"{y:.2f}", (s, y), textcoords="offset points", xytext=(6, -12), fontsize=8, color=col)
    ax.plot([10], [0.3], marker="v", ms=10, color="black", ls="none", label="filed for the step split: at most 0.3 at step 10")
    ax.plot([1000], [0.5], marker="^", ms=10, color="black", ls="none", label="filed for the step split: at least 0.5 at step 1000")
    ax.set_ylim(0, 1.08)
    axes_steps(ax, "share of the gain at the trigger carried by Δ", "How much of the trained behaviour runs through Δ?")
    ax.legend(fontsize=8, loc="center left")
    save(fig, "behaviour_in_delta.png")


def pile_pgd(out: dict[str, Any]) -> None:
    f = out["W0"]["pgd_floor"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhspan(1 - 3 * f, 1 + 3 * f, color=GREY, alpha=0.2, lw=0, label=f"W0 noise band, 1 ± 3f (f = {f:.3f}, L40)")
    ax.axhline(1 + 3 * f, color=GREY, ls="--", lw=1, label=f"line 1 + 3f = {1 + 3 * f:.2f}")
    for arm, (name, col) in ARMS.items():
        v = per_step(out, arm, "pile", "pgd_kl")
        ax.plot(STEPS, v, "o-", color=col, label=name, lw=2)
        for s, y in zip(STEPS, v, strict=True):
            dy = -12 if arm == "nothing" else 5
            ax.annotate(f"{y:.2f}", (s, y), textcoords="offset points", xytext=(6, dy), fontsize=8, color=col)
    ax.set_ylim(0, 6)
    axes_steps(ax, "Pile adversarial KL ÷ W0's (attacker seed 0)", "Is each state as faithful as W0 under the adversary?")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "pile_pgd.png")


def delta_share(out: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhline(1, color="#2a78d6", ls=":", lw=1.5, label="do-nothing: the whole update in Δ")
    series = {"delta_share_weighted": ("step split (bases carried forward)", "#7a4fc9", "o-", 2.5),
              "delta_share_weighted_fixed": ("per-step split, bases held at W0's", "#c9a8f0", "s--", 1.5),
              "delta_share_weighted_one_shot": ("one-shot adjoint split of the finished update, remainder to Δ", "#1baf7a",
                                                "D-", 2.0)}
    for key, (name, col, style, lw) in series.items():
        v = per_step(out, "split", key)
        ax.plot(STEPS, v, style, color=col, label=name, lw=lw)
        for s, y in zip(STEPS, v, strict=True):
            if key != "delta_share_weighted_fixed":  # it overlays the step split; its values are in the report
                dy = -14 if key == "delta_share_weighted_one_shot" else 6
                ax.annotate(f"{y:.2f}", (s, y), textcoords="offset points", xytext=(6, dy), fontsize=8, color=col)
    ax.set_ylim(0, 1.6)
    axes_steps(ax, "Δ's change ÷ the update, input-weighted energy",
               "Does carrying the bases forward change what Δ takes?")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "delta_share.png")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    out = load()
    behaviour_in_delta(out)
    pile_pgd(out)
    delta_share(out)


if __name__ == "__main__":
    main()
