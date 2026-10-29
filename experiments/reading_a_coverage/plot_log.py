"""The research log's three figures for E11, redrawn from saved results with plain labels (no model).

  log_adversarial.png  per task: the adversarial ratio of step-10 component training and of do-nothing, bars from 1,
                       with the 1.5 fail line and the unchanged model's own spread (1 + 3 floors) per job; the
                       ` we` then ` just` task shows its three seeds.
  log_unmasked.png     per task: the unmasked ratio (pieces alone, Δ out) of both placements: what component
                       training buys.
  log_offtarget.png    component training's adversarial ratio against its off-target KL, one point per task-seed.

And one for the analysis note A1 (reports/analyses/2026-10-20_analysis_adversarial-vs-stochastic.md):
  log_a1_checks.png    per task, seed 0: the extra damage over the unchanged model (nats) of the adversary's mask
                       (pgd_kl), the all-at-importance mask (ci_masked_kl) and random masks (stochastic_kl), mean of
                       the two blocks; reads results/coverage.json only.

Reads results/coverage.json (five tasks) and results/coverage_emoticon.json (the 2026-10-20 emoticon rerun).
Usage (repo root):  python experiments/reading_a_coverage/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from plot import FIGURES, N_FLOORS, PGD_LINE, RESULTS

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
CT, DN = "#2a78d6", "#b9b7b1"  # component training, do-nothing
NAME = {
    "emoticon": "emoticon\n(earlier task)",
    "r0_t352_v310": "' it' then ' is'",
    "r1_t534_v369": "' which' then ' was'",
    "r1_t588_v452": "' will' then ' have'",
    "r2_t275_v2667": "' in' then ' Fig'",
    "r2_t359_v816": "' we' then ' just'",
}
ORDER = ("emoticon", "r0_t352_v310", "r1_t534_v369", "r1_t588_v452", "r2_t275_v2667", "r2_t359_v816")


def load() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """task -> (task record, the job's W0 record): emoticon ran in its own job with its own floors."""
    out: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for f in ("coverage.json", "coverage_emoticon.json"):
        d = json.loads((RESULTS / f).read_text())
        out.update({n: (t, d["W0"]) for n, t in d["tasks"].items()})
    return out


def ratio(task: dict[str, Any], arm: str, key: str) -> list[float]:
    return [s[arm]["pile"]["ratio"][key] for s in task["seeds"].values() if arm in s and "pile" in s[arm]]


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6, which="both")
    ax.set_axisbelow(True)
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.yaxis.set_minor_formatter(NullFormatter())


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print(FIGURES / name)


def paired_bars(ax: Any, data: dict[str, tuple[dict[str, Any], dict[str, Any]]], key: str) -> None:
    """Two bars per task from 1 (log axis), component training and do-nothing; seeds as dots, value labels on top."""
    w = 0.36
    for i, n in enumerate(ORDER):
        t, _ = data[n]
        for a, (arm, col, lab) in enumerate((("pullback", CT, "component training"), ("do_nothing", DN, "do-nothing"))):
            vs = ratio(t, arm, key)
            m = float(np.mean(vs))
            x = i + (a - 0.5) * w
            ax.bar(x, m - 1, w * 0.92, bottom=1, color=col, label=lab if i == 0 else None)
            if len(vs) > 1:
                ax.scatter([x] * len(vs), vs, s=12, color=INK, zorder=3)
            ax.text(x, max([m, *vs]) * 1.04, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(ORDER)), [NAME[n] for n in ORDER], fontsize=8.5)
    ax.set_yscale("log")
    style(ax)


def fig_adversarial(data: dict[str, tuple[dict[str, Any], dict[str, Any]]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    paired_bars(ax, data, "pgd_kl")
    for i, n in enumerate(ORDER):
        line = 1 + N_FLOORS * data[n][1]["floor"]["pgd_kl"]
        ax.plot([i - 0.45, i + 0.45], [line] * 2, color=INK, lw=1, ls=":",
                label="unchanged model's run-to-run spread (3 times it, above 1; per job)" if i == 0 else None)
    ax.axhline(PGD_LINE, color="#c0392b", lw=1, ls="--", label="fail line 1.5")
    ax.axvline(0.5, color=GRID, lw=1)
    ax.set_ylim(0.9, 9)
    ax.set_yticks([1, 1.5, 2, 3, 5, 8], ["1", "1.5", "2", "3", "5", "8"])
    ax.set_ylabel("adversarial ratio\n(adversary's damage over the unchanged model's, log scale)")
    ax.set_title("Component training's pieces fail the adversarial check on three of the five new tasks,\n"
                 "and do clearly worse than leaving the update outside them on two;\n"
                 "on ' which' then ' was' and ' we' then ' just' the gap is within the noise", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "log_adversarial.png")


def fig_unmasked(data: dict[str, tuple[dict[str, Any], dict[str, Any]]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    paired_bars(ax, data, "unmasked_kl")
    ax.axvline(0.5, color=GRID, lw=1)
    ax.set_ylim(0.8, 30)
    ax.set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    ax.set_ylabel("pieces-alone error, Δ removed\n(over the unchanged model's, log scale)")
    ax.set_title("Removing Δ leaves component training's average error on Pile text at the unchanged level;\n"
                 "with do-nothing, whose update sits in Δ, the same removal multiplies it", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "log_unmasked.png")


def fig_offtarget(data: dict[str, tuple[dict[str, Any], dict[str, Any]]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for n in ORDER:
        t, _ = data[n]
        pts = [(s["pullback"]["heldout"]["kl_other"], s["pullback"]["pile"]["ratio"]["pgd_kl"]) for s in t["seeds"].values()]
        xs, ys = zip(*pts, strict=True)
        ax.scatter(xs, ys, s=30, color=CT, zorder=3)
        label = NAME[n].split("\n")[0] + (" (3 seeds)" if len(pts) > 1 else "")
        ax.annotate(label, (max(xs), max(ys)), xytext=(6, 6 if len(pts) > 1 else 0), textcoords="offset points",
                    fontsize=8, va="center")
    ax.axhline(PGD_LINE, color="#c0392b", lw=1, ls="--", label="fail line 1.5")
    ax.set_yscale("log")
    ax.set_yticks([1, 1.5, 2, 3, 5, 8], ["1", "1.5", "2", "3", "5", "8"])
    ax.set_xlim(0, 0.62)
    ax.set_xlabel("off-target change: KL to the original model at non-trigger positions (nats)")
    ax.set_ylabel("adversarial ratio (log scale)")
    ax.set_title("Component training's adversarial ratio mostly rises with its off-target change\n"
                 "(' it' then ' is' is the exception; a pattern, not a tested cause)", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "log_offtarget.png")


# ---- A1 (analysis note, 2026-10-20): the three checks' extra damage per task ----

A1_ORDER = ("r2_t275_v2667", "r0_t352_v310", "r1_t534_v369", "r2_t359_v816", "r1_t588_v452")
A1_VERDICT = {  # E11's reading of the adversarial ratio (fail line 1.5)
    "r2_t275_v2667": "fails", "r0_t352_v310": "fails", "r1_t534_v369": "fails",
    "r2_t359_v816": "at the line", "r1_t588_v452": "passes",
}
A1_CHECKS = (  # (key, label, colour, marker)
    ("pgd_kl", "adversary's chosen mask", "#c0392b", "o"),
    ("ci_masked_kl", "every subcomponent down to its importance, together", INK, "s"),
    ("stochastic_kl", "random masks (VPD's random-mask check)", "#8a8884", "D"),
)
A1_FLOOR = 1e-3  # the log axis starts here; changes at or below it (some are negative) sit on the edge, hollow


def a1_extra(task: dict[str, Any], w0: dict[str, Any], key: str) -> float:
    """Seed-0 component training's damage (mean of the two blocks) minus the unchanged model's, in nats."""
    return float(np.mean(task["seeds"]["0"]["pullback"]["pile"]["raw"][key])) - w0["mean"][key]


def fig_a1_checks() -> None:
    d = json.loads((RESULTS / "coverage.json").read_text())
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    labels = []
    for row, n in enumerate(A1_ORDER):
        t, y = d["tasks"][n], len(A1_ORDER) - 1 - row
        ex = {k: a1_extra(t, d["W0"], k) for k, *_ in A1_CHECKS}
        ax.plot([max(ex["ci_masked_kl"], A1_FLOOR), ex["pgd_kl"]], [y, y], color=GRID, lw=3, zorder=1)
        for k, lab, col, mk in A1_CHECKS:
            low = ex[k] <= A1_FLOOR
            ax.scatter(max(ex[k], A1_FLOOR), y, s=50, marker=mk, zorder=3, label=lab if row == 0 else None,
                       facecolor="white" if low else col, edgecolor=col, lw=1.4)
        fold = ex["pgd_kl"] / ex["ci_masked_kl"]
        ax.text(20, y, f"{float(f'{fold:.1g}'):g}×", va="center", ha="right", fontsize=9.5, color=INK)
        r = t["seeds"]["0"]["pullback"]["pile"]["ratio"]["pgd_kl"]
        labels.append(f"{NAME[n]}\nadversarial ratio {r:.2f}, {A1_VERDICT[n]}")
    ax.set_yticks(range(len(A1_ORDER) - 1, -1, -1), labels, fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xscale("log")
    ax.set_xlim(A1_FLOOR * 0.8, 22)
    ax.set_ylim(-0.6, len(A1_ORDER) - 0.3)
    ax.text(20, len(A1_ORDER) - 0.55, "adversary ÷\nall-down mask", ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.set_xlabel("extra damage over the unchanged decomposition (nats, log scale; hollow = 0.001 or less)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("On every task the adversary's mask does 7 to 300 times the extra damage of turning\n"
                 "every subcomponent down together; random masks add less than either", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2)
    save(fig, "log_a1_checks.png")


def main() -> None:
    data = load()
    fig_adversarial(data)
    fig_unmasked(data)
    fig_offtarget(data)
    fig_a1_checks()


if __name__ == "__main__":
    main()
