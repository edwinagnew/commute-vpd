"""The research log's four figures for E30, redrawn from saved results with plain labels (no model).

  log_trigger.png      per update: the share of the re-gating effect at the trigger recovered by the fixed change
                       applied everywhere, beside the same change at the trigger only (the earlier run's figure, stored
                       under G1.ref), the null fine-tune's change and the change shuffled (updates.*.R_everywhere).
  log_adversarial.png  per update: the adversary's damage on Pile text for the task's change, the null's change and the
                       shuffled change, each over the no-change arm's (score.ratios), with the pass and fail lines
                       1.5 / (1 + f) and 1.5 (1 + f) (score.floors.pgd_f).
  log_sparsity.png     per update: VPD's sparsity loss for the same three arms over the unchanged model's
                       (score.ratios.*.sparsity), with the 1.05 cheating line.

  log_halves.png       emoticon plain SGD only: the raised half, the lowered half and the whole change applied
                       everywhere, adversarial ratio over no change with the job's lines (score.lines), sparsity
                       ratio written on each bar.

Reads results/gamma_shift_map.json and results/gamma_shift_map_amend1.json.
Usage (repo root):  python experiments/gamma_shift_map/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot import AMEND1, FIG, OUT

UPDATES = (("emoticon/sgd_0.01", "emoticon\nplain SGD"), ("emoticon/adam_3e-05", "emoticon\nAdam"),
           ("r1_t534_v369/sgd_0.03", "' which' then ' was'\nplain SGD"))
BLUE, LIGHT, ORANGE, BROWN, INK = "#1f6fb4", "#9cc3e6", "#e08a2c", "#8a6a2f", "#333333"
CHANGES = (("b_shift", "the task's change, applied everywhere", BLUE),
           ("c_null", "the null fine-tune's change (target ' http')", ORANGE),
           ("d_shuffled", "the task's change shuffled among the pieces", BROWN))


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(range(len(UPDATES)), [lab for _, lab in UPDATES], fontsize=9)


def label(ax: Any, x: float, v: float, fmt: str = "{:.2f}") -> None:
    ax.text(x, v, fmt.format(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=8)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=140)
    plt.close(fig)


def fig_trigger(d: dict[str, Any]) -> None:
    bars = (("trigger only", "the task's change at the trigger only (earlier run)", LIGHT),) + CHANGES
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    w = 0.2
    for j, (key, _) in enumerate(UPDATES):
        u = d["updates"][key]
        for i, (arm, lab, col) in enumerate(bars):
            v = u["G1"]["ref"]["R_shared"] if arm == "trigger only" else u["R_everywhere"][arm]
            x = j + (i - 1.5) * w
            ax.bar(x, v, w * 0.92, color=col, label=lab if j == 0 else None)
            label(ax, x, v)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylim(-0.42, 1.0)
    ax.set_ylabel("share of the switching effect at the trigger recovered\n(0: no change, 1: the fine-tuned model's own values)",
                  fontsize=8.5)
    ax.set_title("At the trigger, the change applied everywhere does as well as at the trigger alone;\n"
                 "the null's change pushes the target down and the shuffled change does nothing", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    save(fig, "log_trigger.png")


def grouped(ax: Any, d: dict[str, Any], value: Any, base: float, fmt: str) -> None:
    w = 0.26
    for j, (key, _) in enumerate(UPDATES):
        r = d["score"]["ratios"][key]
        for i, (arm, lab, col) in enumerate(CHANGES):
            v = value(r, arm)
            x = j + (i - 1) * w
            ax.bar(x, v - base, w * 0.92, bottom=base, color=col, label=lab if j == 0 else None)
            label(ax, x, v, fmt)


def fig_adversarial(d: dict[str, Any]) -> None:
    f = d["score"]["floors"]["pgd_f"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    grouped(ax, d, lambda r, arm: r[arm]["pgd_kl"] / r["a_zero"]["pgd_kl"], 1.0, "{:.1f}")
    ax.axhline(1.5 / (1 + f), color="#2a8a3a", ls=":", lw=1.2, label=f"pass line ({1.5 / (1 + f):.2f})")
    ax.axhline(1.5 * (1 + f), color="#c0392b", ls="--", lw=1.2, label=f"fail line ({1.5 * (1 + f):.2f})")
    ax.set_yscale("log")
    ax.set_ylim(1, 50)
    ax.set_yticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax.set_ylabel("damage the adversary finds on Pile text,\nover what it finds with no change (log)", fontsize=8.5)
    ax.set_title("On ordinary text, the task's change and the null's both fail far above the fail line;\n"
                 "the same values shuffled among the pieces stay inside the pass line", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    save(fig, "log_adversarial.png")


def fig_sparsity(d: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    grouped(ax, d, lambda r, arm: r[arm]["sparsity"], 1.0, "{:.1f}")
    ax.axhline(1.05, color="#c0392b", ls="--", lw=1.2, label="cheating line (1.05)")
    ax.set_ylim(1, 4.8)
    ax.set_ylabel("VPD's sparsity loss on Pile text,\nover the unchanged decomposition's", fontsize=8.5)
    ax.set_title("Every change raises the sparsity cost, and the shuffled change costs as much as the task's,\n"
                 "so the sparsity cost does not explain the adversary's damage", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    save(fig, "log_sparsity.png")


def fig_halves(a: dict[str, Any]) -> None:
    """One bar per half: adversarial ratio over no change, with the job's own pass and fail lines."""
    arms = (("up", "raised half only", "#c0392b"), ("down", "lowered half only", BLUE),
            ("shift", "whole change", INK))
    rat, (lo, hi) = a["score"]["ratios"], a["score"]["lines"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for i, (k, _, col) in enumerate(arms):
        v, sp = rat[k]["pgd_kl"], rat[k]["sparsity"]
        ax.bar(i, v - 1, 0.6, bottom=1, color=col)
        ax.text(i, v, f"{v:.2f}\nsparsity cost x{sp:.2f}", ha="center", va="bottom", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.axhline(lo, color="#2a8a3a", ls=":", lw=1.2, label=f"pass line ({lo:.2f})")
    ax.axhline(hi, color="#c0392b", ls="--", lw=1.2, label=f"fail line ({hi:.2f})")
    ax.set_yscale("log")
    ax.set_ylim(1, 20)
    ax.set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    ax.set_xticks(range(len(arms)), [lab for _, lab, _ in arms], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("damage the adversary finds on Pile text,\nover what it finds with no change (log)", fontsize=8.5)
    ax.set_title("Emoticon, plain SGD: the lowered half carries most of the damage;\n"
                 "the raised half sits between the lines but pays the whole sparsity cost", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "log_halves.png")


def main() -> None:
    d = json.loads(OUT.read_text())
    fig_trigger(d)
    fig_adversarial(d)
    fig_sparsity(d)
    fig_halves(json.loads(AMEND1.read_text()))
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()
