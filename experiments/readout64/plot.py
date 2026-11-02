"""E64's figures from results/run.json (job 48611), results/pile_bigrams.json (bigram_check.py, post-hoc, no model) and
E54's stage1.json; no model. Colours as E54/E56: elicit green, cipher orange. "Elicit" means built to elicit.

  figures/fit_loss.png     held-out next-token loss per block against W0, with both versions of the fit bar
  figures/possession.png   keys each block holds under the filed rule, against W0 and a table of next-token counts
  figures/tilt.png         a key's rank against its value's training windows: does window count split the two kinds?
  figures/lookup.png       a key's rank against how often the key is followed by its value in the sample
  figures/removal.png      the removal share per state at block 0, with its noise floor and the separation bar
  figures/drops.png        what the removal does: W0's control, and each state's log-odds drop beside W0's
  figures/last_block.png   the removal share at block 0 and at the last block, against the expected value near 1

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/readout64/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
COLOR = {"cipher": "#eb6834", "elicit": "#1f9e6e", "own": "#a8531f", "readout": "#2a78d6", "capped": "#8fb8e8"}
INK, GRID, REF = "#52514e", "#e4e3df", "#b0413e"
BLOCKS = [0, 1, 2, 3]
BLOCK_NAMES = [f"block {b}" + (" (last)" if b == 3 else "") for b in BLOCKS]
ARMS = (("elicit", "elicit_s", range(5)), ("cipher", "cipher_s", range(5)), ("own", "cipher_own_s", range(2)))
ARM_NAMES = {"elicit": "built to elicit", "cipher": "cipher, matched step", "own": "cipher, own step"}


def tidy(ax: Any, title: str, ylabel: str, xlabel: str = "") -> None:
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=INK)


def line_label(ax: Any, y: float, text: str, above: bool = True, x: float = 0.01, color: str = INK) -> None:
    ax.annotate(text, (x, y), xycoords=("axes fraction", "data"), xytext=(0, 3 if above else -10),
                textcoords="offset points", fontsize=7, color=color)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def fit_loss(r: dict[str, Any]) -> None:
    f = r["stage0"]["fit"]
    natural, prior = f["model"], f["model_at_sample_prior"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(BLOCKS, f["readout"], color=COLOR["readout"], lw=2, marker="o", ms=6, label="scored readout, up to 2,000 windows per value")
    ax.plot(BLOCKS, f["low_cap"]["readout"], color=COLOR["capped"], lw=2, marker="o", ms=6, label="capped readout, up to 241 windows per value (reported)")
    ax.axhline(prior, color=INK, lw=1)
    ax.axhline(natural, color=INK, lw=1, ls="--")
    ax.axhline(prior + 0.1, color=REF, lw=1.2)
    ax.axhline(natural + 0.1, color=REF, lw=1.2, ls=":")
    line_label(ax, prior, f"W0 moved to the sample's prior: {prior:.2f}", above=False)
    line_label(ax, natural, f"W0 at its natural prior: {natural:.2f}", above=False)
    line_label(ax, prior + 0.1, f"bar as run: {prior + 0.1:.2f}", color=REF)
    line_label(ax, natural + 0.1, f"first-filed bar, replaced before any reading: {natural + 0.1:.2f}", color=REF)
    ax.annotate(f"{f['readout'][3]:.2f}", (3, f["readout"][3]), xytext=(7, -2), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xticks(BLOCKS, BLOCK_NAMES)
    ax.set_xlim(-0.25, 3.35)
    ax.set_ylim(1.1, 2.75)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    tidy(ax, "Does the last block's readout fit unseen Pile text as well as W0? It misses the bar as run", "64-way next-token loss, held-out windows (nats)")
    save(fig, "fit_loss.png")


def possession(r: dict[str, Any], b: dict[str, Any]) -> None:
    M = set(r["W0"]["elicit"]["all"]["possessed"])
    p = r["possession"]
    inM = [len(set(p["elicit"]["all"][i]["possessed"]) & M) for i in BLOCKS]
    total = [len(p["elicit"]["all"][i]["possessed"]) for i in BLOCKS]
    cipher = [len(p["cipher"]["all"][i]["possessed"]) for i in BLOCKS]
    table = set(b["sample"]["elicit"]["possessed"])
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(BLOCKS, total, color=COLOR["elicit"], lw=2, ls="--", marker="o", ms=6, label="built to elicit: all keys held")
    ax.plot(BLOCKS, inM, color=COLOR["elicit"], lw=2, marker="o", ms=7, label="built to elicit: keys held that W0's output also holds")
    ax.plot(BLOCKS, cipher, color=COLOR["cipher"], lw=2, marker="s", ms=6, label="cipher: keys held")
    for y, t, c, up in ((len(M), f"W0's output holds {len(M)} elicit keys, 0 cipher keys", INK, False), (15, "elicit bar: 15 of W0's 20", REF, False),
                        (6, "cipher bar: 6 keys", REF, True), (2.5, "chance before the specificity guard: 2.5 keys", INK, False)):
        ax.axhline(y, color=c, lw=1, ls=":" if c == REF else "-")
        line_label(ax, y, t, above=up, color=c)
    ax.axhline(len(table), color="#7a5fb5", lw=1.2, ls="-.")
    line_label(ax, len(table), f"next-token count table, no model: {len(table)} elicit keys, 0 cipher keys", x=0.01, above=True, color="#7a5fb5")
    ax.set_xticks(BLOCKS, BLOCK_NAMES)
    ax.set_ylim(-1, 32)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    tidy(ax, "Which blocks hold the map under the filed rule? Every block for elicit, none for the cipher", "keys of 32 at median rank 5 or better and specific")
    save(fig, "possession.png")


def scatter_rank(ax: Any, xs: list[float], reading: dict[str, Any], keys: list[int], color: str, marker: str) -> None:
    held = set(reading["possessed"])
    for x, k in zip(xs, keys, strict=True):
        filled = k in held
        ax.scatter(x, reading["medians"][str(k)], s=34, marker=marker, edgecolor=color, facecolor=color if filled else "white", lw=1.1, zorder=3, alpha=0.9)


def rank_axes(ax: Any) -> None:
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 64], ["1", "2", "5", "10", "20", "64"])
    ax.minorticks_off()
    ax.set_ylim(70, 0.8)
    ax.axhline(5, color=REF, lw=1, ls=":")


RANK_HANDLES = [Line2D([], [], marker="o", ls="", color=COLOR["elicit"], label="elicit key, held"),
                Line2D([], [], marker="o", ls="", markerfacecolor="white", color=COLOR["elicit"], label="elicit key, not held"),
                Line2D([], [], marker="s", ls="", markerfacecolor="white", color=COLOR["cipher"], label="cipher key, never held")]


def tilt(r: dict[str, Any], s1: dict[str, Any]) -> None:
    pv, keys = r["stage0"]["positions"]["per_value"], s1["elicit"]["keys"]
    n = {t: [pv[str(v)] for v in s1[t]["values"]] for t in ("elicit", "cipher")}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharex=True, sharey=True)
    panels = (("Block 0's readout", r["possession"]["elicit"]["all"][0], r["possession"]["cipher"]["all"][0]), ("W0's own output, same 64 values", r["W0"]["elicit"]["all"], r["W0"]["cipher"]["all"]))
    for ax, (name, e, c) in zip(axes, panels, strict=True):
        ax.axvspan(min(n["cipher"]), max(n["cipher"]), color=GRID, alpha=0.6, lw=0)
        scatter_rank(ax, n["cipher"], c, keys, COLOR["cipher"], "s")
        scatter_rank(ax, n["elicit"], e, keys, COLOR["elicit"], "o")
        ax.axvline(2000, color=INK, lw=0.8, ls="--")
        ax.set_xscale("log")
        rank_axes(ax)
        ax.annotate("cap: 2,000 windows", (2000, 62), xytext=(3, 0), textcoords="offset points", fontsize=7, color=INK)
        ax.annotate("rank 5 bar", (0.99, 5), xycoords=("axes fraction", "data"), xytext=(0, 2), textcoords="offset points", fontsize=7, color=REF, ha="right")
        ax.annotate("the cipher's window range", (240, 0.93), xytext=(2, 0), textcoords="offset points", fontsize=7, color=INK)
        tidy(ax, name, "median rank of the key's value among the 64", "training windows available for the value, Pile fit rows")
    axes[1].set_ylabel("")
    axes[0].legend(handles=RANK_HANDLES, fontsize=7.5, frameon=False, loc="lower right", bbox_to_anchor=(1, 0.08))
    fig.suptitle("Does window count split the two kinds? No: at the cipher's window count, elicit values still reach rank 1", fontsize=9.5, color=INK, x=0.01, ha="left")
    save(fig, "tilt.png")


def lookup(r: dict[str, Any], b: dict[str, Any], s1: dict[str, Any]) -> None:
    keys = s1["elicit"]["keys"]
    counts = {t: [b["sample"][t]["count"][str(k)] + 1 for k in keys] for t in ("elicit", "cipher")}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharex=True, sharey=True)
    panels = (("Block 0's readout", r["possession"]["elicit"]["all"][0], r["possession"]["cipher"]["all"][0]), ("W0's own output, same 64 values", r["W0"]["elicit"]["all"], r["W0"]["cipher"]["all"]))
    for ax, (name, e, c) in zip(axes, panels, strict=True):
        jit = [0.86 + 0.28 * i / 31 for i in range(32)]
        scatter_rank(ax, jit, c, keys, COLOR["cipher"], "s")
        scatter_rank(ax, counts["elicit"], e, keys, COLOR["elicit"], "o")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 11, 26, 101], ["0", "1", "3", "10", "25", "100"])
        ax.minorticks_off()
        rank_axes(ax)
        ax.annotate("rank 5 bar", (0.99, 5), xycoords=("axes fraction", "data"), xytext=(0, 2), textcoords="offset points", fontsize=7, color=REF, ha="right")
        tidy(ax, name, "median rank of the key's value among the 64", "times the key is followed by its value in the training sample")
    axes[1].set_ylabel("")
    axes[0].legend(handles=RANK_HANDLES, fontsize=7.5, frameon=False, loc="center right")
    fig.suptitle("Does the training sample itself pair key and value? Every cipher pair occurs 0 times, and block 0 follows the count more than W0 does", fontsize=9.5, color=INK, x=0.01, ha="left")
    save(fig, "lookup.png")


def per_state(ax: Any, states: dict[str, Any], key: str, marker: str, offset: float, hollow: bool = False, size: int = 42) -> list[float]:
    xs: list[float] = []
    x = 0.0
    for arm, pre, seeds in ARMS:
        for s in seeds:
            v = states[f"{pre}{s}"][key]
            ax.scatter(x + offset, v, s=size, marker=marker, lw=1.1, zorder=3, **({"color": COLOR[arm]} if marker == "x" else {"edgecolor": COLOR[arm], "facecolor": "white" if hollow else COLOR[arm]}))
            xs.append(x)
            x += 1
        x += 1
    return xs


def arm_ticks(ax: Any) -> None:
    ax.set_xticks([2, 8, 12.5], [ARM_NAMES[a] for a, _, _ in ARMS])
    ax.tick_params(axis="x", length=0)


def removal(r: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    m = r["random"]["max_abs"]
    ax.axhspan(-m, m, color="#c9c8c3", lw=0, zorder=1)
    ax.axhline(0, color=INK, lw=0.6)
    for y in (-0.05, 0.05):
        ax.axhline(y, color=INK, lw=0.8, ls="--")
    per_state(ax, r["removal"]["states"], "A_lo", "o", -0.12)
    per_state(ax, r["low_cap"]["states"], "A_lo", "D", 0.12, hollow=True, size=30)
    bar = max(r["P2"]["separation"]["b"]) + 0.2
    ax.axhline(bar, color=REF, lw=1.2, ls=":")
    line_label(ax, bar, f"filed bar: elicit had to clear the highest matched cipher by 0.2, so {bar:.2f}", color=REF)
    line_label(ax, 0.05, "filed bound on the random direction: 0.05")
    line_label(ax, -m, f"grey band: largest random-direction share, {m:.4f}", x=0.42, above=False)
    arm_ticks(ax)
    ax.set_ylim(-0.08, 0.25)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK, label="scored readout"), Line2D([], [], marker="D", ls="", markerfacecolor="white", color=INK, label="capped readout (reported)")],
              fontsize=7.5, frameon=False, loc="center right")
    tidy(ax, "Does removing the row cost elicit fine-tunes more of their gain? No: elicit sits below the cipher", "removal share at block 0, net of W0's drop (0 = no more than W0)")
    save(fig, "removal.png")


def drops(r: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True, gridspec_kw={"width_ratios": [1, 1.7]})
    ax = axes[0]
    w = r["W0_control"]
    for i, (lab, suf) in enumerate((("target row", "M"), ("wrong row", "wrong"), ("random", "random"))):
        for j, t in enumerate(("elicit", "cipher")):
            v = w[f"{t}_{suf}"]["median_drop"]
            ax.bar(i + (j - 0.5) * 0.36, v, 0.32, color=COLOR[t], zorder=2)
            if abs(v) > 0.05:
                ax.annotate(f"{v:.2f}", (i + (j - 0.5) * 0.36, v), xytext=(0, 3 if v > 0 else -10), textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    ax.axhline(0.3, color=REF, lw=1, ls=":")
    ax.annotate("filed bar: 0.3", (0.99, 0.3), xycoords=("axes fraction", "data"), xytext=(0, 3), textcoords="offset points", ha="right", fontsize=7, color=REF)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks([0, 1, 2], ["target row", "wrong row", "random direction"])
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=COLOR["elicit"], label="elicit windows"), Line2D([], [], marker="s", ls="", color=COLOR["cipher"], label="cipher windows")], fontsize=7.5, frameon=False, loc="upper right")
    tidy(ax, "W0's control: median drop in log P(target)", "drop under removal at block 0 (below 0: it rises)")
    ax = axes[1]
    st = r["removal"]["states"]
    per_state(ax, st, "lo_drop", "o", 0)
    per_state(ax, r["random"], "lo_drop", "x", 0, size=26)
    e0, c0 = st["elicit_s0"]["lo_drop_W0"], st["cipher_s0"]["lo_drop_W0"]
    ax.hlines(e0, -0.5, 4.5, color=INK, lw=1.6)
    ax.hlines(c0, 5.5, 14.5, color=INK, lw=1.6)
    ax.annotate(f"W0: {e0:.2f}", (0, e0), xytext=(0, 4), textcoords="offset points", fontsize=7.5, color=INK)
    ax.annotate(f"W0: {c0:.2f}", (8, c0), xytext=(0, -11), textcoords="offset points", fontsize=7.5, color=INK)
    ax.axhline(0, color=INK, lw=0.6)
    arm_ticks(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK, label="fine-tuned state, target row"), Line2D([], [], marker="x", ls="", color=INK, label="fine-tuned state, random direction"), Line2D([], [], color=INK, lw=1.6, label="W0, target row")],
              fontsize=7.5, frameon=False, loc="upper right")
    tidy(ax, "Mean log-odds drop: each state moves the way W0 moves, by less", "")
    fig.suptitle("What does the removal do? It works on W0, and the fine-tuned states lose less than W0 loses", fontsize=9.5, color=INK, x=0.01, ha="left")
    save(fig, "drops.png")


def last_block(r: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.axhline(1, color=REF, lw=1.2, ls=":")
    line_label(ax, 1, "expected at the last block, reported and not scored: near 1", above=False, color=REF)
    m = r["random"]["max_abs"]
    ax.axhspan(-m, m, color="#c9c8c3", lw=0, zorder=1)
    line_label(ax, -m, "random direction at block 0, the do-nothing", x=0.5, above=False)
    per_state(ax, r["last_block"]["states"], "A_lo", "^", 0, size=48)
    per_state(ax, r["removal"]["states"], "A_lo", "o", 0, hollow=True, size=34)
    arm_ticks(ax)
    ax.set_ylim(-0.1, 1.08)
    ax.legend(handles=[Line2D([], [], marker="^", ls="", color=INK, label="last block"), Line2D([], [], marker="o", ls="", markerfacecolor="white", color=INK, label="block 0, the scored block")], fontsize=7.5, frameon=False, loc="center right")
    tidy(ax, "Can one row take away the gain even at the output? Not in this apparatus: 0.31 to 0.53, not near 1", "removal share, net of W0's drop")
    save(fig, "last_block.png")


def main() -> None:
    r = json.loads((RES / "run.json").read_text())
    b = json.loads((RES / "pile_bigrams.json").read_text())
    s1 = json.loads((HERE.parent / "keydependent_elicit" / "results" / "stage1.json").read_text())
    FIG.mkdir(exist_ok=True)
    fit_loss(r)
    possession(r, b)
    tilt(r, s1)
    lookup(r, b, s1)
    removal(r)
    drops(r)
    last_block(r)


if __name__ == "__main__":
    main()
