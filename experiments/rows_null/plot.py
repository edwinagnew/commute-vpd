"""E105 (rows_null) figures from results/score.json, results/arm_*.json and E103's stored sets; no model. Colours as
E103's plots: the known-facts fine-tunes purple, the fictional fine-tune orange, references grey.

  figures/overlap.png  carrying-set overlaps on each row set: each half's two seeds, the four half-A against half-B pairs,
                       E103's known-facts seeds, fictional seeds, known against fictional, and the one-answer-form
                       change, with the filed rule's two lines
  figures/bridge.png   each half's set inside the full-row known-facts set of the same and of the other seed, beside the
                       full-row seeds' own overlap and the cross-half range
  figures/gain.png     content gain at the matched step on the half's own sentences and on the other half's, per arm,
                       with the C1 bar and the full-row arms' gain

It also prints two reads off the files that were not filed: each half set's overlap with E102's Gaussian change, and the
one-answer-form overlap of E103 Amendment 1 against E103.

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/rows_null/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
KAP = HERE.parent / "known_answer_pair" / "results"
E102 = HERE.parent / "qwen_carrying" / "results" / "states"
PURPLE, LILAC, ORANGE, GREY, INK, GRID = "#8a4fbf", "#c4a6e0", "#eb6834", "#7a7a75", "#52514e", "#e4e3df"
ARMS = ("A42", "A142", "B42", "B142")
WORDS = {"real": "real rows", "tofu": "TOFU rows"}
# The answer-form change (E103 Amendment 1's matched state against E103's) and the Gaussian change, per row set
FORM = {"real": (KAP / "neurons_amend1" / "a1_matched__real.json", KAP / "neurons" / "real_matched__real.json"),
        "tofu": (KAP / "neurons_amend1" / "a1_matched__tofu.json", KAP / "neurons" / "real_matched__tofu.json")}
GAUSS = {"real": KAP / "neurons" / "gauss__real.json", "tofu": E102 / "gauss_tofu.json"}


def js(path: Path) -> Any:
    return json.loads(path.read_text())


def carry(path: Path) -> set[int]:
    return set(js(path)["carry"])


def share(a: Path, b: Path) -> float:
    """The share of a's set inside b's, as qwen_neurons.overlap."""
    sa = carry(a)
    return len(sa & carry(b)) / len(sa)


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def bar(ax: Any, i: float, v: float, c: str, hatch: bool = False, width: float = 0.6, dy: float = 0.015) -> None:
    ax.bar(i, v, width=width, color=c if not hatch else "none", edgecolor=c, hatch="//" if hatch else None)
    ax.text(i, v + dy, f"{v:.2f}", ha="center", fontsize=7, color=INK)


def pair_handles(own: str, other: str) -> list[Any]:
    """Legend entries for a filled purple bar and a hatched lilac one."""
    return [Patch(color=PURPLE, label=own), Patch(facecolor="none", edgecolor=LILAC, hatch="//", label=other)]


def overlap(score: dict[str, Any], e103: dict[str, Any]) -> None:
    labels = ["half A:\nseed 142\nin 42", "half B:\nseed 142\nin 42", "A42\nin B42", "A42\nin B142", "A142\nin B42",
              "A142\nin B142", "full rows,\nknown:\nseed 142\nin 42", "full rows,\nfictional:\nseed 142\nin 42",
              "known\nin fict.,\nhighest\nof four", "one answer\nform\nchanged"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, rows in zip(axes, ("real", "tofu"), strict=True):
        s, e = score[rows], e103[rows]
        form = share(*FORM[rows])
        vals = [s["w_a"], s["w_b"], *s["cross"], e["known"], e["fict"], e["cross_max"], form]
        cols = [PURPLE, PURPLE, *[PURPLE] * 4, LILAC, ORANGE, INK, GREY]
        hatch = [False, False, True, True, True, True, False, False, True, False]
        for i, (v, c, h) in enumerate(zip(vals, cols, hatch, strict=True)):
            bar(ax, i, v, c, h)
        ax.axvline(5.5, color=GRID, lw=1.5)
        ax.text(2.5, 1.12, "E105: two disjoint halves of the real rows", ha="center", fontsize=8, color=INK)
        ax.text(7.5, 1.12, "E103: full rows, for reference", ha="center", fontsize=8, color=INK)
        ax.axhline(s["label_needs_min"], color=PURPLE, ls="--", lw=1.1,
                   label=f"'label' needs every cross-half pair above {s['label_needs_min']:.3f}")
        ax.axhline(s["rows_needs_max"], color=ORANGE, ls=":", lw=1.4,
                   label=f"'rows' needs every cross-half pair below {s['rows_needs_max']:.3f}")
        ax.axhline(0.006, color=GREY, ls=":", lw=1, label="two random sets, about 0.006")
        ax.set_xticks(range(len(vals)), labels, fontsize=6.4)
        ax.set_title(f"carrying sets ranked on the {WORDS[rows]}: verdict '{s['verdict']}'", fontsize=9)
        ax.set_ylim(0, 1.2)
        ax.legend(fontsize=7, loc="lower left", frameon=True, framealpha=0.9)
        style(ax)
    axes[0].set_ylabel("share of the first set's 1,024 neurons\ninside the second set")
    fig.suptitle("Swapping every trained row for other known facts moves the carrying set little more than a change of "
                 "seed; known against fictional moves half of it", fontsize=10)
    save(fig, "overlap.png")


def bridge(score: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, rows in zip(axes, ("real", "tofu"), strict=True):
        b, s = score[rows]["bridge"], score[rows]
        for i, a in enumerate(ARMS):
            bar(ax, i - 0.18, b["same_seed"][a], PURPLE, width=0.34, dy=0.004)
            bar(ax, i + 0.18, b["other_seed"][a], LILAC, hatch=True, width=0.34, dy=0.004)
        line = ax.axhline(b["full_seeds"], color=INK, ls="--", lw=1.1,
                          label=f"full-row known seeds, one in the other, {b['full_seeds']:.3f}")
        band = ax.axhspan(s["cross_min"], s["cross_max"], color=GREY, alpha=0.25,
                          label=f"half A against half B, {s['cross_min']:.3f} to {s['cross_max']:.3f}")
        ax.legend(handles=[*pair_handles("inside the full-row set of the same seed",
                                         "inside the full-row set of the other seed"), line, band],
                  fontsize=6.8, loc="lower left", frameon=True, framealpha=0.9)
        ax.set_xticks(range(len(ARMS)), [f"half {a[0]}, seed {a[1:]}" for a in ARMS], fontsize=7.5)
        ax.set_title(f"carrying sets ranked on the {WORDS[rows]}", fontsize=9)
        ax.set_ylim(0.6, 1.0)
        style(ax)
    axes[0].set_ylabel("share of the half's 1,024 neurons\ninside the full-row known-facts set")
    fig.suptitle("Half the rows seen twice as often leave the carrying set about where the full-row run had it",
                 fontsize=10)
    save(fig, "bridge.png")


def gain(arms: dict[str, dict[str, Any]], full: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, a in enumerate(ARMS):
        r = arms[a]
        bar(ax, i - 0.18, r["matched"]["gain"]["content"], PURPLE, width=0.34, dy=0.02)
        bar(ax, i + 0.18, r["other_half_gain"]["content"], LILAC, hatch=True, width=0.34, dy=0.02)
    c1 = ax.axhline(arms["A42"]["matched"]["bar"], color=INK, ls=":", lw=1.3,
                    label=f"C1's bar, {arms['A42']['matched']['bar']:.3f}: half the fictional run's own gain")
    ref = ax.axhline(sum(full) / 2, color=GREY, ls="--", lw=1.3,
                     label=f"full-row known-facts runs on all 217 rows, {min(full):.3f} and {max(full):.3f}")
    ax.set_xticks(range(len(ARMS)), [f"half {a[0]}, seed {a[1:]}" for a in ARMS], fontsize=8)
    ax.set_ylabel("content gain at the matched step\n(nats a token; 0 = unchanged model)")
    ax.set_ylim(0, 2.1)
    ax.legend(handles=[*pair_handles("the half's own, trained sentences", "the other half's sentences, never trained"),
                       c1, ref], fontsize=7.2, loc="upper right", frameon=True, framealpha=0.9)
    ax.set_title("Each half's update raises the other half's answers nearly as much as its own", fontsize=10)
    style(ax)
    save(fig, "gain.png")


def main() -> None:
    score, e103 = js(RES / "score.json"), js(KAP / "neurons_amend2.json")["identity"]
    arms = {a: js(RES / f"arm_{a}.json") for a in ARMS}
    full = [js(KAP / f"{n}_train.json")["matched"]["gain"]["content"] for n in ("amend1", "amend2")]
    overlap(score, e103)
    bridge(score)
    gain(arms, full)
    for rows in ("real", "tofu"):
        g = {a: round(share(RES / "sets" / f"{a}__{rows}.json", GAUSS[rows]), 4) for a in ARMS}
        print(f"{rows}: half sets inside the Gaussian change's {g}; one answer form changed {share(*FORM[rows]):.4f}")
    for a in ARMS:
        r = arms[a]
        print(f"{a}: other-half gain / own {r['other_half_gain']['content'] / r['matched']['gain']['content']:.3f}")


if __name__ == "__main__":
    main()
