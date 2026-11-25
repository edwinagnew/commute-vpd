"""The research log's three figures for E19, redrawn from saved results with plain labels (no model).

  log_behaviour.png  per step: the share of the fine-tune's gain on the target that comes through Δ, for the step
                     split and the least-squares split; do-nothing (1) as a line.
  log_adversary.png  per step: the adversarial ratio on Pile text (attacker seed 0) of the step split, the
                     least-squares split and do-nothing, against the pass line 1 + 3f.
  log_delta_share.png per step: Δ's input-weighted energy over the update's, for the step split with the pieces carried
                     forward, the same with the pieces held at the original model's, and the one-shot adjoint split.

Reads results/step_split.json.
Usage (repo root):  python experiments/step_split/plot_log.py
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from plot import FIG, STEPS, load, per_step

PURPLE, LILAC, ORANGE, BLUE, GREEN, GREY = "#7a4fc9", "#c9a8f0", "#eb6834", "#2a78d6", "#1baf7a", "#7f7f7f"


def bars(ax: Axes, series: list[tuple[str, str, list[float]]], base: float = 0.0) -> None:
    """Grouped bars, one group per step, drawn up (or down) from `base`, each labelled with its value."""
    w = 0.8 / len(series)
    for j, (lab, col, vals) in enumerate(series):
        for i, v in enumerate(vals):
            x = i + (j - (len(series) - 1) / 2) * w
            ax.bar(x, v - base, w * 0.92, bottom=base, color=col, label=lab if i == 0 else None)
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(STEPS)), [f"step {s:,}" for s in STEPS])
    ax.set_xlabel("fine-tune step (Adam, 1,000 steps)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_behaviour(out: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.axhline(1, color=BLUE, ls=":", lw=1.5, label="do-nothing: the whole update in Δ (reads 1)")
    bars(ax, [("least-squares split of the finished update", ORANGE,
               per_step(out, "least_squares", "routes", "shares", "e_delta")),
              ("step split (every step split, pieces carried forward)", PURPLE,
               per_step(out, "step_split", "routes", "shares", "e_delta"))])
    ax.set_ylim(0, 1.4)
    ax.set_ylabel("share of the fine-tune's gain on the target\nthat comes through Δ (lower is better)")
    ax.set_title("The step split leaves more of the behaviour in Δ than least squares, at every step\n"
                 "(one task, one fine-tune seed)", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "log_behaviour.png")


def fig_adversary(out: dict[str, Any]) -> None:
    f = out["W0"]["pgd_floor"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.axhline(1 + 3 * f, color=GREY, ls="--", lw=1.2,
               label=f"pass line {1 + 3 * f:.2f}: 1 + 3 × the unchanged decomposition's spread ({f:.2f} on this card)")
    bars(ax, [("do-nothing", BLUE, per_step(out, "nothing", "pile", "pgd_kl")),
              ("least-squares split", ORANGE, per_step(out, "least_squares", "pile", "pgd_kl")),
              ("step split", PURPLE, per_step(out, "step_split", "pile", "pgd_kl"))], base=1.0)
    ax.set_ylim(0.8, 6)
    ax.set_ylabel("adversarial ratio on Pile text\n(1 = as faithful as the unchanged decomposition)")
    ax.set_title("The step split fails the adversarial check at step 1,000; under the line earlier,\n"
                 "weak passes at one attacker seed", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "log_adversary.png")


def fig_delta_share(out: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.axhline(1, color=BLUE, ls=":", lw=1.5, label="do-nothing: the whole update in Δ (reads 1)")
    bars(ax, [("one-shot adjoint split of the finished update", GREEN,
               per_step(out, "split", "delta_share_weighted_one_shot")),
              ("step split, pieces carried forward", PURPLE, per_step(out, "split", "delta_share_weighted")),
              ("step split, pieces held at the original model's", LILAC,
               per_step(out, "split", "delta_share_weighted_fixed"))])
    ax.set_ylim(0, 1.6)
    ax.set_ylabel("energy (squared size) of Δ's change over the\nupdate's, weighted by the task's inputs")
    ax.set_title("Moving the pieces leaves Δ's share unchanged; by step 1,000 the step split's Δ\n"
                 "is larger than the whole update, twice the one-shot split's", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "log_delta_share.png")


def main() -> None:
    out = load()
    fig_behaviour(out)
    fig_adversary(out)
    fig_delta_share(out)


if __name__ == "__main__":
    main()
