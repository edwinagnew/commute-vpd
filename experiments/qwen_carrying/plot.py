"""E102 stage A's figures from results/score.json and results/states/*.json, no model.
Anchor 1 (built to reuse, by assumption) is blue, anchor 2 (built to be new, by assumption) orange, the Gaussian change
grey, as in E98's and E101's figures.

  figures/prompt_overlap.png   per state, the share of its carrying set inside the prompt's contrast set, with the
                               Gaussian change on the same rows, anchor 2 matched, and the random draws' maximum
  figures/engaged.png          per state, the engaged share, with the range of its 100 random draws and held-out web 0.5
  figures/set_specificity.png  per state, how much its carrying set shares with itself on the other row half, with its
                               other seed, and with the Gaussian change's set on the same rows

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/qwen_carrying/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
BLUE, ORANGE, GREY, INK, GRID = "#2a78d6", "#eb6834", "#7a7a75", "#52514e", "#e4e3df"
# state, label on the axis, colour, its other seed, the Gaussian change on its rows
STATES = [
    ("gauss_a1", "Gaussian change\n(template rows)", GREY, None, None),
    ("a1_s0", "template\nseed 0", BLUE, "a1_s1", "gauss_a1"),
    ("a1_s1", "template\nseed 1", BLUE, "a1_s0", "gauss_a1"),
    ("gauss_tofu", "Gaussian change\n(TOFU rows)", GREY, None, None),
    ("a2_s42_matched", "TOFU matched\nseed 42", ORANGE, "a2_s142_matched", "gauss_tofu"),
    ("a2_s142_matched", "TOFU matched\nseed 142", ORANGE, "a2_s42_matched", "gauss_tofu"),
    ("a2_s42_full", "TOFU full\nseed 42", ORANGE, "a2_s142_full", "gauss_tofu"),
    ("a2_s142_full", "TOFU full\nseed 142", ORANGE, "a2_s42_full", "gauss_tofu"),
]


def load() -> tuple[dict[str, Any], dict[str, Any]]:
    score = json.loads((RES / "score.json").read_text())
    states = {s: json.loads((RES / "states" / f"{s}.json").read_text()) for s, *_ in STATES}
    return score, states


def shared(a: list[int], b: list[int]) -> float:
    return len(set(a) & set(b)) / len(a)


def style(ax: Any, ylabel: str) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(range(len(STATES)), [lab for _, lab, *_ in STATES], fontsize=8)
    ax.set_ylabel(ylabel)


def prompt_overlap(score: dict[str, Any]) -> None:
    per = score["per_state"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (s, _, c, *_) in enumerate(STATES):
        ax.scatter(i, per[s]["prompt_overlap"], s=60, color=c, zorder=3)
        ax.vlines(i, 0, max(per[s]["prompt_overlap_draws"]), color=GREY, lw=6, alpha=0.35)
    g = per["gauss_a1"]["prompt_overlap"]
    m = (per["a2_s42_matched"]["prompt_overlap"] + per["a2_s142_matched"]["prompt_overlap"]) / 2
    band = score["P1_prompt_overlap_anchor1"]["band"]
    ax.axhspan(g, g + band, color=GREY, alpha=0.18)
    ax.axhline(g, color=GREY, ls="--", lw=1)
    ax.text(7.45, g, "Gaussian change, template rows:\nthe filed floor (shaded: + band)", fontsize=7.5, color=INK, va="top", ha="right")
    ax.axhline(m, color=ORANGE, ls=":", lw=1.2)
    ax.text(-0.3, m + 0.0008, "TOFU fine-tune at anchor 1's size (mean of seeds)", fontsize=7.5, color=ORANGE)
    ax.text(0.1, 0.0125, "grey bars: largest of 100 random\nlayer-matched sets", fontsize=7.5, color=INK, va="bottom")
    ax.text(3.5, 0.066, "the prompt's own contrast set agrees with itself across item halves at 0.956 (off scale)",
            fontsize=7.5, color=INK, ha="center")
    style(ax, "share of carrying set inside the\nprompt's contrast set")
    ax.set_ylim(0, 0.07)
    ax.set_title("Does a fine-tune's carrying set overlap the neurons a system prompt moves, more than a random change's?", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "prompt_overlap.png", dpi=150)


def engaged(score: dict[str, Any]) -> None:
    per = score["per_state"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (s, _, c, *_) in enumerate(STATES):
        d = per[s]["engaged_draws"]
        ax.vlines(i, min(d), max(d), color=GREY, lw=6, alpha=0.35)
        ax.scatter(i, per[s]["engaged"], s=60, color=c, zorder=3)
    ax.axhline(0.5, color=INK, ls="--", lw=1)
    ax.text(7.45, 0.5005, "held-out web text: 0.500", fontsize=7.5, color=INK, ha="right")
    ax.text(0.1, 0.4885, "grey bars: range of 100 random layer-matched sets", fontsize=7.5, color=INK)
    style(ax, "engaged share: median firing at the answers\nas a quantile of the neuron's web firing")
    ax.set_ylim(0.485, 0.515)
    ax.set_title("Were the carrying neurons already firing at the answers before training?", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "engaged.png", dpi=150)


def set_specificity(states: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    w = 0.26
    for i, (s, _, c, seed, gauss) in enumerate(STATES):
        ax.bar(i - w, states[s]["halves_overlap"], w, color=c, alpha=0.45, label="its own set on the other half of the rows" if i == 1 else None)
        if seed:
            ax.bar(i, shared(states[s]["carry"], states[seed]["carry"]), w, color=c, label="its other seed's set" if i == 1 else None)
            ax.bar(i + w, shared(states[s]["carry"], states[gauss]["carry"]), w, color=GREY,
                   label="the Gaussian change's set, same rows" if i == 1 else None)
    ax.axhline(1024 / 172032, color=INK, lw=0.8)
    ax.text(7.45, 0.02, "chance for two random sets: 0.006", fontsize=7.5, color=INK, ha="right")
    style(ax, "share of the 1024-neuron carrying set\nshared with ...")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3)
    ax.set_title("Is a carrying set specific to its fine-tune, or set by the rows it is read on?", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "set_specificity.png", dpi=150)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    score, states = load()
    prompt_overlap(score)
    engaged(score)
    set_specificity(states)


if __name__ == "__main__":
    main()
