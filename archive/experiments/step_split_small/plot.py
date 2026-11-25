"""Figures for step_split_small (E25) from results/step_split_small.json, with E19's large path
(experiments/step_split/results/step_split.json) on the same axes.

  delta_share.png  P1: how much of the update does Delta hold under the step split? Delta's share of ||delta-W||^2 (left)
                   and of its input-weighted energy beside the one-shot adjoint split's (right), per path.
  pile_pgd.png     P2: is the step-split state faithful under the adversary? Pile PGD ratio to W0 of the step split,
                   least squares and do-nothing, one panel per path, with this job's pass and fail lines.
  pieces.png       P3: how much of the behaviour do the pieces carry? P(target) with every component on and Delta out,
                   step split against least squares, one panel per path.

Usage (repo root):  uv run python experiments/step_split_small/plot.py
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
E19 = HERE.parent / "step_split" / "results" / "step_split.json"
STEPS = (10, 100, 1000)
GREY = "#7f7f7f"
INK = "#52514e"
PATHS = {"sgd": ("SGD 0.01", "#008300", "o"), "adam": ("Adam 3e-6", "#e87ba4", "s"), "large": ("Adam 3e-5 (E19)", GREY, "^")}
ARMS = {"step_split": ("step split", "#7a4fc9"), "least_squares": ("least squares", "#eb6834"), "nothing": ("do-nothing", "#2a78d6")}
LINE = 1.5  # faith_score.LINE
E19_FLOOR = 0.209  # E19's L40 floor (step_split.json W0.pgd_floor)


def load() -> dict[str, dict[str, Any]]:
    """Per path, per scored step (str), the arm records and the split numbers, in one layout for E25 and E19."""
    out = json.loads((HERE / "results" / "step_split_small.json").read_text())
    paths = {k: v["states"] for k, v in out["paths"].items()}
    paths["large"] = json.loads(E19.read_text())["states"]
    return paths


def series(states: dict[str, Any], *keys: str) -> list[float]:
    vals = []
    for s in STEPS:
        v: Any = states[str(s)]
        for k in keys:
            v = v[k]
        vals.append(float(v))
    return vals


def axes_steps(ax: Axes, ylabel: str, title: str) -> None:
    ax.set_xscale("log")
    ax.set_xticks(STEPS, [str(s) for s in STEPS])
    ax.set_xlabel("fine-tune step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, color=INK, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig: Any, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print("wrote", FIG / name)


def delta_share(paths: dict[str, Any]) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for p, (label, col, mk) in PATHS.items():
        ls = "--" if p == "large" else "-"
        y = series(paths[p], "step_split", "delta_change_frac")
        axs[0].plot(STEPS, y, ls, color=col, marker=mk, markersize=7, linewidth=2, label=label)
        axs[0].annotate(label, (STEPS[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
        w, one = series(paths[p], "split", "delta_share_weighted"), series(paths[p], "split", "delta_share_weighted_one_shot")
        axs[1].plot(STEPS, w, ls, color=col, marker=mk, markersize=7, linewidth=2, label=f"{label}: step split")
        axs[1].plot(STEPS, one, ":", color=col, marker=mk, markersize=7, markerfacecolor="none", linewidth=2,
                    label=f"{label}: one-shot adjoint")
    for ax in axs:
        ax.axhline(1, color=INK, linewidth=1, linestyle=":")
    axes_steps(axs[0], "Δ's share of ||δW||²", "Δ's share of the update under the step split")
    axes_steps(axs[1], "Δ's share, input-weighted", "per-step scale refit (solid) vs one scale (hollow)")
    axs[0].legend(fontsize=8, frameon=False)
    axs[1].legend(fontsize=7, frameon=False)
    save(fig, "delta_share.png")


def pile_pgd(paths: dict[str, Any], f: float) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (p, (label, _, _)) in zip(axs, PATHS.items(), strict=True):
        for arm, (alabel, col) in ARMS.items():
            ax.plot(STEPS, series(paths[p], arm, "pile", "pgd_kl"), "-o", color=col, markersize=7, linewidth=2, label=alabel)
        ff = E19_FLOOR if p == "large" else f  # each panel's lines from its own job's W0 floor
        for y, ls, txt, va in ((LINE / (1 + ff), "--", "pass below", "top"), (LINE * (1 + ff), ":", "fail above", "bottom")):
            ax.axhline(y, color=INK, linewidth=1, linestyle=ls)
            ax.text(STEPS[-1], y, f"{txt} {y:.2f} ", ha="right", va=va, fontsize=7, color=INK)
        ax.axhline(1, color=GREY, linewidth=1)
        ax.text(STEPS[-1], 1, "W₀ ", ha="right", va="top", fontsize=7, color=INK)
        card = "L40, W₀ floor 0.21" if p == "large" else f"4090, W₀ floor {f:.3f}"
        axes_steps(ax, "Pile adversarial KL, ratio to W₀" if p == "sgd" else "", f"{label}: {card}")
    axs[0].legend(fontsize=8, frameon=False, loc="upper right")
    save(fig, "pile_pgd.png")


def pieces(paths: dict[str, Any]) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (p, (label, _, _)) in zip(axs, PATHS.items(), strict=True):
        for arm in ("step_split", "least_squares"):
            alabel, col = ARMS[arm]
            ax.plot(STEPS, series(paths[p], arm, "routes", "p_cells", "U1"), "-o", color=col, markersize=7, linewidth=2,
                    label=alabel)
        ax.plot(STEPS, series(paths[p], "step_split", "routes", "p_cells", "T1"), ":", color=INK, linewidth=1,
                label="fine-tuned model")
        ax.plot(STEPS, series(paths[p], "nothing", "routes", "p_cells", "U1"), "-o", color=ARMS["nothing"][1], markersize=7,
                linewidth=2, label="do-nothing")
        axes_steps(ax, "P(target), every component on, Δ out" if p == "sgd" else "", label)
        ax.set_ylim(0, 1.02)
    axs[0].legend(fontsize=8, frameon=False, loc="lower left")
    save(fig, "pieces.png")


def main() -> None:
    paths = load()
    f = json.loads((HERE / "results" / "step_split_small.json").read_text())["W0"]["pgd_floor"]
    delta_share(paths)
    pile_pgd(paths, f)
    pieces(paths)


if __name__ == "__main__":
    main()
