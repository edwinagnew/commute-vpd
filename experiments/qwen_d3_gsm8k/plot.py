"""E104's figures from results/table.json (CPU), one per carried row. Each puts RL, its shuffled-reward null and
own-samples SFT at steps 10 and 100 (one point per seed) on the same axes as E94's anchors (template, TOFU full, and
on placement TOFU matched) and E94's small random changes, read from E94's table. Tick labels carry each state's
output size on the maths rows (K00 - K0, nats per position). A number that reads no call is left out.

  python experiments/qwen_d3_gsm8k/plot.py   ->  figures/{outside,placement,think}.png
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rules
from matplotlib.figure import Figure

FIG = rules.HERE / "figures"
COLOUR = {"R": "#2a6fdb", "N": "#6b6b6b", "F": "#e07b00"}
LABEL = {"R": "RL", "N": "shuffled-reward null", "F": "own-samples SFT"}
SHORT = {"R": "RL", "N": "null", "F": "own-samples"}
COPY_LABEL = {"S1": "RL seed 0, step 100,\nscaled to own-samples\nstep 100's size",
              "S2": "RL seed 0, step 100,\nscaled to its\nstep 10's size",
              "S3": "TOFU full, seed 42,\nscaled to the\ntemplate's size"}
SEED_MARK ={0: "o", 1: "s"}  # first seed circle, second seed square
# column -> (arm, step): RL, null, own-samples at steps 10 and 100
COLUMNS = [(arm, s) for arm in rules.ARMS for s in rules.STEPS]
ANCHOR_STYLE = {"template": ("k", "--", "template fine-tune (E94)"), "tofu_full": ("k", ":", "TOFU full update (E94)"),
                "tofu_matched": ("#8a5a00", "-.", "TOFU at template norm, facts not learned (E94)")}


def field(nums: dict, name: str, number: str, key: str = "value") -> float | None:
    return ((nums.get(name) or {}).get(number) or {}).get(key)


def nan(x: float | None) -> float:
    """A no-call reading as NaN, which matplotlib leaves out."""
    return float("nan") if x is None else x


def anchor(ax, nums: dict, number: str, key: str, which: tuple[str, ...] = ("template", "tofu_full"),
           label_suffix: str = "") -> None:
    for k in which:
        m = rules.mean([field(nums, n, number, key) for n in rules.ANCHORS[k]])
        if m is not None:
            c, ls, lab = ANCHOR_STYLE[k]
            ax.axhline(m, color=c, ls=ls, lw=1, label=f"{lab}{label_suffix}: {m:.3g}")


def ticks(ax, nums: dict, extra: list[str] | None = None) -> None:
    labels = []
    for arm, s in COLUMNS:
        size = [(nums.get(f"{a}_{s}") or {}).get("size_K00_minus_K0") for a in rules.ARMS[arm]]
        txt = ", ".join(f"{x:.3f}" for x in size if x is not None)
        labels.append(f"{SHORT[arm]}\nstep {s}\nsize {txt}")
    ax.set_xticks(range(len(COLUMNS) + len(extra or [])), labels + (extra or []), fontsize=7)


def arm_points(ax, nums: dict, number: str, key: str = "value", hollow: str | None = None, err: str | None = None,
               legend: bool = True) -> None:
    """One point per seed (circle first seed, square second); `hollow`, if given, draws open markers and is the
    suffix of their legend label."""
    seen: set[str] = set()
    for x, (arm, s) in enumerate(COLUMNS):
        for i, a in enumerate(rules.ARMS[arm]):
            y = field(nums, f"{a}_{s}", number, key)
            if y is None:
                continue
            e = field(nums, f"{a}_{s}", number, err) if err else None
            label = LABEL[arm] + (f", {hollow}" if hollow else "")
            ax.errorbar([x + 0.15 * (i - 0.5)], [y], yerr=[e] if e else None, fmt=SEED_MARK[i], ms=7, capsize=3,
                        color=COLOUR[arm], mfc="none" if hollow else COLOUR[arm], zorder=3,
                        label=label if legend and label not in seen else None)
            seen.add(label)


def outside(nums: dict, e94: dict) -> Figure:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
    arm_points(ax0, nums, "outside")
    arm_points(ax0, nums, "outside", "null", hollow="its null")
    anchor(ax0, nums, "outside", "value")
    ax0.set_ylabel("outside share of input-weighted energy")
    ax0.set_title("Filled: the update's outside share. Hollow: its exact null", fontsize=9)
    arm_points(ax1, nums, "outside", "fraction_of_null", legend=False)
    anchor(ax1, nums, "outside", "fraction_of_null")
    g = e94["gauss_small"]["outside"]
    ax1.axhline(g["value"] / g["null"], color="#b0b0b0", lw=6, alpha=0.6, label="Gaussian random change (E94): 1.00")
    ax1.set_ylim(0.4, 1.15)
    ax0.set_ylim(0.1, 0.43)
    ax1.set_ylabel("outside share / its null")
    ax1.set_title("The same share as a fraction of its null", fontsize=9)
    for ax in (ax0, ax1):
        ticks(ax, nums)
        ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle("Does any maths update sit outside the subcomponents more than another? "
                 "Do-nothing reads exactly as each state", fontsize=10)
    return fig


def placement(nums: dict, e94: dict) -> Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for arm, seeds in rules.ARMS.items():
        for i, a in enumerate(seeds):
            ys = [nan(field(nums, f"{a}_{s}", "placement")) for s in rules.STEPS]
            ax.plot([0, 1], ys, marker=SEED_MARK[i], ms=7, color=COLOUR[arm], lw=1.5,
                    label=f"{LABEL[arm]}, seed {a.split('_s')[1]}")
    anchor(ax, nums, "placement", "value", ("template", "tofu_matched", "tofu_full"))
    ax.axhspan(-0.005, 0.005, color="#d9d9d9", alpha=0.7, label="spread of the Gaussian null's draws: under 0.005")
    ax.axhline(e94["alive_small"]["placement"]["value"], color="#2e8b57", lw=1, ls=(0, (1, 2)),
               label=f"random change of alive subcomponents only (E94): {e94['alive_small']['placement']['value']:.3f}")
    ax.set_xticks([0, 1], ["step 10", "step 100"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("placement excess: share of the split on already-on\nsubcomponents, minus a Gaussian change's")
    ax.set_title("Does placement fall from step 10 to step 100, as TOFU's fell? In four of five runs; RL seed 0 rises."
                 "\nDo-nothing has no split, so no reading", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    return fig


def think(nums: dict, e94: dict) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    arm_points(ax, nums, "think", err="draw_sd")
    arm_points(ax, nums, "think", "L", hollow="no-think set")
    n = len(COLUMNS)
    extra = []
    for k, copy in enumerate(rules.COPIES):
        y, err = nan(field(nums, copy, "think")), nan(field(nums, copy, "think", "draw_sd"))
        matched = (nums.get(copy) or {}).get("matched")
        ax.errorbar([n + k], [y], yerr=[err], fmt="D", ms=7, capsize=3, color="k", mfc="k" if matched else "none",
                    label="scaled copies (hollow: size not matched)" if k == 0 else None, zorder=3)
        extra.append(COPY_LABEL[copy] + ("" if matched else "\nSIZE NOT MATCHED"))
    ticks(ax, nums, extra)
    anchor(ax, nums, "think", "value")
    lo, hi = sorted(e94[r]["think"]["value"] for r in ("gauss_small", "alive_small"))
    ax.axhspan(lo, hi, color="#d9d9d9", alpha=0.7, label=f"random changes at template size (E94): {lo:.2f} to {hi:.2f}")
    ax.axhline(1, color="#b0b0b0", lw=1, label="chance: 1")
    ax.set_ylim(0, 4.6)
    ax.set_ylabel("think fold: overlap with think mode's set / chance")
    ax.set_title("Do the switched-on subcomponents overlap think mode's set differently? Filled: think mode, with the fold's "
                 "draw sd.\nHollow: the length-matched no-think set. Do-nothing reads exactly as each state", fontsize=9)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    return fig


def main() -> None:
    nums = json.loads((rules.RESULTS / "table.json").read_text())["numbers"]
    e94 = json.loads((rules.E94 / "table.json").read_text())["numbers"]
    FIG.mkdir(exist_ok=True)
    for name, make in (("outside", outside), ("placement", placement), ("think", think)):
        fig = make(nums, e94)
        fig.tight_layout()
        fig.savefig(FIG / f"{name}.png", dpi=150)
        plt.close(fig)
        print(FIG / f"{name}.png")


if __name__ == "__main__":
    main()
