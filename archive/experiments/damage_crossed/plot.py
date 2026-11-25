"""E122's two figures, from results/damage_crossed.json (jobs 54703, 54733, 54766) and, for the saved copies' damage,
E39's broad_fragility/results/broad_fragility.json:

    python experiments/damage_crossed/plot.py
      -> figures/ratio_vs_damage.png  adversarial ratio of every state on sets A and B against its damage, lines 1.40 and 1.61
      -> figures/excess_where.png     how concentrated each state's excess is at positions predicting a capitalised token
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
OUT = json.loads((RESULTS / "damage_crossed.json").read_text())
E39 = json.loads((HERE.parent / "broad_fragility" / "results" / "broad_fragility.json").read_text())["reads"]
ADAM, LORA, SPLIT = "#2a78d6", "#eb6834", "#1baf7a"  # dataviz default slots 1-3
INK, MUTED, BAND = "#2b2b2b", "#8a8a85", "#e6e5df"
PASS, FAIL = 1.40, 1.61


def ratio(key: str) -> float | None:
    r = OUT["score"]["ratios"].get(key)
    return None if r is None else r["ratio"]


def damage(name: str) -> float:
    """A copy's own damage; a saved copy's from E39 (H100); a do-nothing's = its fine-tune's."""
    if name == "tt_adam":
        return E39["trained"]["damage"]
    if name == "tt_lora":
        return E39["tt_lora"]["damage"]
    if name.startswith("tt_") and name != "tt_c3_start":
        return OUT["copies"][name]["damage"]
    ft = {"nothing_adam300": OUT["band"]["D_A"], "tt_c3_start": OUT["band"]["D_A"],
          "nothing_c1": OUT["chosen"]["c1"]["damage"], "nothing_c2": OUT["chosen"]["c2"]["damage"]}
    return ft[name]


# name: (label, colour, marker, filled)
STATES: dict[str, tuple[str, str, str, bool]] = {
    "tt_adam": ("full Adam fine-tune, standard copy (saved, E39)", ADAM, "o", True),
    "tt_recipe": ("full Adam fine-tune, standard copy (retrained here)", ADAM, "s", True),
    "tt_c1": ("cell 1: Adam stopped at step 20, its copy", ADAM, "D", True),
    "tt_lora": ("rank-8 LoRA at step 300, its copy (saved, E39)", LORA, "o", True),
    "tt_c2": ("cell 2: LoRA trained on to step 2,400, its copy", LORA, "D", True),
    "tt_c3_start": ("full Adam fine-tune's least-squares split, before training", SPLIT, "^", False),
    "tt_c3": ("full Adam fine-tune, copy started from that split", SPLIT, "^", True),
    "nothing_adam300": ("do-nothing: full Adam fine-tune left in Δ", ADAM, "o", False),
    "nothing_c1": ("do-nothing: cell 1's fine-tune left in Δ", ADAM, "D", False),
    "nothing_c2": ("do-nothing: cell 2's fine-tune left in Δ", LORA, "D", False),
}


def lines(ax: Any) -> None:
    ax.axhspan(PASS, FAIL, color=BAND, zorder=0)
    ax.axhline(PASS, color=MUTED, ls="--", lw=1)
    ax.axhline(FAIL, color=MUTED, ls="--", lw=1)
    ax.axhline(1.0, color=INK, lw=0.8)
    ax.text(0.302, PASS * 0.985, "pass at or below 1.40", va="top", ha="right", fontsize=7, color=MUTED)
    ax.text(0.302, FAIL * 1.015, "fail at or above 1.61", va="bottom", ha="right", fontsize=7, color=MUTED)
    ax.text(0.302, 0.985, "unchanged model W0 = 1", va="top", ha="right", fontsize=7, color=INK)
    for c, (lo, hi) in (("cell 1", OUT["band"]["c1"]), ("cell 2", OUT["band"]["c2"])):
        ax.axvspan(lo, hi, color="#f4f3ee", zorder=-1)
        ax.text((lo + hi) / 2, 11.6, f"{c}'s damage band", ha="center", va="top", fontsize=7, color=MUTED)


def ratio_vs_damage() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, s in zip(axes, "AB", strict=True):
        lines(ax)
        for a, b, col in (("tt_adam", "tt_c1", ADAM), ("tt_lora", "tt_c2", LORA)):
            ra, rb = ratio(f"{a}|{s}"), ratio(f"{b}|{s}")
            if ra is not None and rb is not None:
                ax.annotate("", xy=(damage(b), rb), xytext=(damage(a), ra),
                            arrowprops={"arrowstyle": "->", "color": col, "lw": 1, "ls": ":", "shrinkA": 6, "shrinkB": 6})
        for name, (label, col, mk, filled) in STATES.items():
            r = ratio(f"{name}|{s}")
            if r is None:
                continue
            ax.plot(damage(name), r, mk, ms=8, color=col, mfc=col if filled else "white", mec=col, mew=1.6,
                    label=label, zorder=3)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([1, 1.4, 1.61, 2, 3, 5, 8, 10]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels(["1", "1.40", "1.61", "2", "3", "5", "8", "10"])
        ax.set_ylim(0.9, 12)
        ax.set_xlim(0.08, 0.305)
        ax.set_xlabel("damage: KL(W0 ‖ W0 + the state's own change), plain Pile text, nats")
        ax.set_title(f"set {s}: Pile val rows {'0-255' if s == 'A' else '256-511'}"
                     + ("" if s == "A" else "\n(do-nothing and the split start were read on set A only)"), fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("adversarial ratio: the state's worst-case KL over W0's (log scale)")
    axes[0].text(0.236, 7.2, "split start drawn at\nthe fine-tune's damage\n(its own not measured)", fontsize=6.5,
                 color=SPLIT, ha="right", va="top")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.suptitle("Does the copy fail because of the update's damage or its recipe? Neither cell lands cleanly;\n"
                 "the copy started from the least-squares split fails far worse than any other state", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(FIGURES / "ratio_vs_damage.png", dpi=150)


def excess_where() -> None:
    order = ["tt_adam", "tt_recipe", "tt_c1", "tt_lora", "tt_c2", "tt_c3_start", "tt_c3",
             "nothing_adam300", "nothing_c1", "nothing_c2"]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.axvline(1.0, color=INK, lw=0.8)
    ax.text(1.08, -0.6, "1 = excess spread evenly over positions", fontsize=7.5, color=INK)
    for i, name in enumerate(order):
        _, col, mk, _ = STATES[name]
        y = len(order) - 1 - i
        for s, filled, dy in (("A", True, 0.13), ("B", False, -0.13)):
            m = OUT["maps"].get(f"{name}|{s}")
            if m is not None:
                ax.plot(m["concentration"], y + dy, mk, ms=8, color=col, mfc=col if filled else "white", mec=col, mew=1.6)
    ax.set_yticks(range(len(order)), [STATES[n][0] for n in reversed(order)], fontsize=7.5)
    ax.plot([], [], "o", color=INK, label="set A (filled)")
    ax.plot([], [], "o", mfc="white", mec=INK, label="set B (open)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.8, len(order) - 0.5)
    ax.set_xlabel("concentration: share of the adversary's excess KL at positions whose next token holds a capital,\n"
                  "over the share of such positions (about 0.12)")
    ax.set_title("Where does the adversary find its excess? Mostly at capitalised-next-token positions\n"
                 "for the LoRA copies; evenly everywhere for the copy from the split", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "excess_where.png", dpi=150)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    ratio_vs_damage()
    excess_where()


if __name__ == "__main__":
    main()
