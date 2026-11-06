"""E94's figures from results/table.json (CPU), one per row. Each figure puts the two anchors' seeds, the random changes
at both sizes and the row's do-nothing reference on the same axes; the size of each state on the maths rows (K00 - K0,
nats per position) and its weight norm are in the tick labels. A number that reads no call is left out.

  python experiments/qwen_d3_rows/plot.py   ->  figures/{direct,outside,placement,credit,think}.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
A1, A2, GREY = "#2a6fdb", "#e07b00", "#6b6b6b"
NORM = {"a1": "0.49", "matched": "0.48", "full": "18.1", "past": "15"}
# column -> (label, states or randoms, colour)
COLUMNS = [("template", ("a1_s0", "a1_s1"), A1, "a1"),
           ("random,\nsmall size", ("gauss_small", "alive_small"), GREY, None),
           ("TOFU, matched\n(facts not learned)", ("a2_s42_matched", "a2_s142_matched"), A2, "matched"),
           ("TOFU, full", ("a2_s42_full", "a2_s142_full"), A2, "full"),
           ("random,\nbig size", ("gauss_big", "alive_big"), GREY, None),
           ("TOFU, first\npast the line", ("a2_s42_past", "a2_s142_past"), A2, "past")]


def rec(nums: dict, state: str, number: str, rows: str | None = None) -> dict:
    s = nums.get(state) or {}
    if rows is not None:
        s = (s.get("own_rows") or {}).get(rows) or {}
    return s.get(number) or {}


def ticks(ax, nums: dict) -> None:
    labels = []
    for label, states, _, key in COLUMNS:
        size = [((nums.get(s) or {}).get("printed") or {}).get("size_K00_minus_K0") for s in states]
        size = [x for x in size if x is not None]
        extra = f"\nsize {sum(size) / len(size):.2f}" if size else ""
        labels.append(label + (f"\nnorm {NORM[key]}" if key else "") + extra)
    ax.set_xticks(range(len(COLUMNS)), labels, fontsize=7)
    ax.set_xlim(-0.6, len(COLUMNS) - 0.4)


def points(ax, nums: dict, number: str, key: str = "value", rows_for_random: dict | None = None, hollow: bool = False,
           label_suffix: str = "") -> None:
    """Anchors: filled circles, two seeds side by side with row half-widths. Randoms: square (Gaussian), triangle
    (alive pieces). `rows_for_random` maps a random to the own rows it is read on (row 1 only)."""
    seen: set[str] = set()
    for x, (_, states, colour, akey) in enumerate(COLUMNS):
        for i, st in enumerate(states):
            if akey is None and rows_for_random is not None:
                rows = rows_for_random.get(st)
                if rows is None:
                    continue
                r = rec(nums, st, number, rows)
            else:
                r = rec(nums, st, number)
            v = r.get(key)
            if v is None:
                continue
            if akey is None:
                name = ("Gaussian random" if st.startswith("gauss") else "alive-pieces random") + label_suffix
                marker = "s" if st.startswith("gauss") else "^"
            else:
                name = ("template fine-tune" if akey == "a1" else "TOFU fine-tune") + label_suffix
                marker = "o"
            hw = r.get("half_width") if key == "value" else None
            ax.errorbar(x + (i - 0.5) * 0.22, v, yerr=hw or 0, fmt=marker, ms=7, capsize=3, color=colour,
                        mfc="white" if hollow else colour, label=None if name in seen else name)
            seen.add(name)


def finish(fig, ax, nums: dict, title: str, ylabel: str, name: str, zero: str | None = None) -> None:
    if zero:
        ax.axhline(0, color="k", lw=0.8, ls=":", label=zero)
    ticks(ax, nums)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    out = HERE / "figures" / f"{name}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def direct(nums: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    points(axes[0], nums, "direct", rows_for_random={"alive_small": "a1", "alive_big": "tofu"})
    axes[0].set_title("own rows, whole-question answers (scored)", fontsize=8)
    points(axes[1], nums, "direct_maths")
    axes[1].set_title("maths rows, every position (printed)", fontsize=8)
    for ax in axes:
        ax.axhline(0, color="k", lw=0.8, ls=":", label="do-nothing reads 0")
        ticks(ax, nums)
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("direct share: KL closed by the moved subcomponents\nwith their masks held, / KL closed by the split", fontsize=8)
    fig.suptitle("Do the moved subcomponents carry more of the template or the TOFU update by themselves?\n"
                 "TOFU's full update drops to zero on its own rows but not on the maths rows", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "direct.png", dpi=150)
    plt.close(fig)
    print("wrote direct.png")


def outside(nums: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    points(ax, nums, "outside")
    points(ax, nums, "outside", key="null", hollow=True, label_suffix=": its exact null")
    finish(fig, ax, nums, "How much of each update lies outside the subcomponents, against a random update's share?\n"
           "Every fine-tune sits below its null; TOFU's full update sits higher than the template's",
           "outside share of input-weighted energy", "outside")


def placement(nums: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    points(ax, nums, "placement")
    ax.axhspan(-0.0165, 0.0165, color="#eeeeee", label="half of the gap between two TOFU seeds in an earlier run, 0.033")
    finish(fig, ax, nums, "Does the split put more energy on subcomponents already on in the original model than a Gaussian change does?\n"
           "The template sits at the Gaussian; TOFU sits above it, most at the step before it learns",
           "placement excess: share on the original on-set,\nminus 4 Gaussian nulls' share", "placement", zero="Gaussian null reads 0")


def credit(nums: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    points(ax, nums, "credit")
    ax.axhline(0.01, color=A1, lw=0.8, ls="--", label="filed line for the template: 0.01")
    finish(fig, ax, nums, "Under the masks, how much more behaviour does the split keep than doing nothing?\n"
           "Under 0.01 of the gain everywhere; no random was read under the masks",
           "credit: kept by the split minus kept by do-nothing,\nshare of the fine-tune's NLL gain",
           "credit", zero="do-nothing reads 0")


def think(nums: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    points(ax, nums, "think")
    points(ax, nums, "think", key="L", hollow=True, label_suffix=": length-matched no-think set")
    ax.axhspan(3, 4, color="#eeeeee", label="filed band, fold 3 to 4")
    finish(fig, ax, nums, "Do the subcomponents each update switches on overlap think mode's more than chance?\n"
           "Every change sits 2.9 to 4.4 times chance; think mode beats the no-think set only on TOFU before it learns",
           "fold over chance: overlap of the switched-on set\nwith think mode's set (filled) or no-think (hollow)", "think",
           zero=None)


def main() -> None:
    (HERE / "figures").mkdir(exist_ok=True)
    nums = json.loads((HERE / "results" / "table.json").read_text())["numbers"]
    for f in (direct, outside, placement, credit, think):
        f(nums)


if __name__ == "__main__":
    main()
