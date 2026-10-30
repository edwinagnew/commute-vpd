"""E57 (qwen_reading) figures from the saved JSONs (Stages A, B2, C0, C, summary.json, boot_direct.json); no model.

Usage: plot.py RESULTS_DIR FIGURES_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATES = ("e40", "gold_s0", "a2long_s0", "a2long_s1")
CONTROLS = ("gauss", "inspan")
ORDER = STATES + CONTROLS
NAME = {"e40": "E40 gold, step 10", "gold_s0": "E45 gold, step 100", "a2long_s0": "anchor 2, seed 0",
        "a2long_s1": "anchor 2, seed 1", "gauss": "random, Gaussian", "inspan": "random, alive pieces"}
COLOR = {"e40": "#2a78d6", "gold_s0": "#eb6834", "a2long_s0": "#1baf7a", "a2long_s1": "#199e70",
         "gauss": "#8a8984", "inspan": "#5f5e5a"}
KINDS = ("q", "k", "v", "o", "gate", "up", "down")


def load(d: Path) -> dict[str, Any]:
    j = {p.stem: json.loads(p.read_text()) for p in d.glob("*.json") if p.stem != "predictions"}
    j["state"] = {s: j[f"C_{s}"] for s in STATES} | {c: j["C0"][c] for c in CONTROLS}
    return j


def axes(title: str, ylabel: str) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def ticks(ax: Any, names: tuple[str, ...]) -> None:
    ax.set_xticks(range(len(names)), [NAME.get(n, n).replace(", ", ",\n") for n in names], fontsize=8.5)


def save(fig: Any, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)


def points(ax: Any, vals: list[float], lo: list[float], hi: list[float]) -> None:
    for i, n in enumerate(ORDER):
        ax.errorbar(i, vals[i], yerr=[[vals[i] - lo[i]], [hi[i] - vals[i]]], fmt="o", ms=8, color=COLOR[n],
                    capsize=4, lw=2)


def regating(j: dict[str, Any], out: Path) -> None:
    m = [j["state"][n]["maths"] for n in ORDER]
    fig, ax = axes("Does the re-gating share tell a real update from a random one?",
                   "re-gating share on the 128 maths rows")
    points(ax, [x["regating"] for x in m], [x["boot95"][0] for x in m], [x["boot95"][1] for x in m])
    ax.axhline(0.8, color="#e34948", ls="--", lw=1.5)
    ax.text(5.4, 0.81, "plan's line: a random change at or above 0.8 means no teeth", ha="right", fontsize=8)
    ax.axhline(0.77, color="#6250d6", ls=":", lw=1.5)
    ax.text(5.4, 0.745, "67M elicit map, E54 (0.77)", ha="right", fontsize=8, color="#6250d6")
    ax.axhline(0.24, color="#6250d6", ls=":", lw=1.5)
    ax.text(5.4, 0.215, "67M cipher, E54 (0.24)", ha="right", fontsize=8, color="#6250d6")
    ax.axvspan(3.5, 5.5, color="#f1f0ec", zorder=0)
    ax.set_ylim(0, 1.05)
    ticks(ax, ORDER)
    save(fig, out, "regating_share")


def direct(j: dict[str, Any], out: Path) -> None:
    b = j["boot_direct"]
    m = [b["maths"][n] for n in ORDER]
    fig, ax = axes("Does the direct share separate the updates from random changes?",
                   "direct share, 95% row bootstrap")
    lo = min(b["maths"][c]["boot95"][0] for c in CONTROLS)
    hi = max(b["maths"][c]["boot95"][1] for c in CONTROLS)
    ax.axhspan(lo, hi, color="#d6d5d0", zorder=0)
    ax.text(-0.4, hi + 0.002, "band: the two random changes' intervals", fontsize=8)
    ax.axhline(0, color="#0b0b0b", lw=1)
    ax.text(-0.4, 0.002, "do-nothing reads 0", fontsize=8)
    points(ax, [x["direct"] for x in m], [x["boot95"][0] for x in m], [x["boot95"][1] for x in m])
    for i, n in enumerate(ORDER):
        if n in b["own"]:
            ax.plot(i + 0.2, b["own"][n]["direct"], "o", ms=8, mfc="none", mec=COLOR[n], mew=2)
    ax.plot([], [], "o", color="#52514e", label="on the maths rows")
    ax.plot([], [], "o", mfc="none", mec="#52514e", mew=2, label="on the update's own training rows")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_ylim(-0.005, 0.105)
    ticks(ax, ORDER)
    save(fig, out, "direct_share")


def switched(j: dict[str, Any], out: Path) -> None:
    sets = j["A"]["sets"]
    names = ("prompt",) + ORDER
    vals = [sets["S_prompt"]] + [j["state"][n]["S"] for n in ORDER]
    fig, ax = axes("Do the switched-on sets clear the filed split-half floor?",
                   "subcomponents switched on, count")
    cols = ["#e87ba4"] + [COLOR[n] for n in ORDER]
    ax.bar(range(len(names)), vals, color=cols, width=0.6)
    ax.axhline(sets["min_size"], color="#e34948", ls="--", lw=1.5)
    ax.text(len(names) - 0.6, sets["min_size"] * 1.02, "filed floor: 5 × the split-half count", ha="right",
            fontsize=8)
    ax.axhline(0, color="#0b0b0b", lw=1)
    ax.text(len(names) - 0.6, sets["min_size"] * 0.92, "the unchanged model against itself on the same rows reads 0", ha="right",
            fontsize=8)
    NAME["prompt"] = "format prompt at W0\n(positive control)"
    ticks(ax, names)
    save(fig, out, "switched_on")


def overlap(j: dict[str, Any], out: Path) -> None:
    ov = j["summary"]["overlap"]
    refs = (("P", "format prompt"), ("T", "think-on answers"), ("L", "length-matched no-think"),
            ("C", "unrelated web text"))
    fig, ax = axes("Is the overlap with each reference set special to real updates?",
                   "overlap excess over site-stratified random sets, sd")
    w = 0.13
    for k, n in enumerate(ORDER):
        xs = [i + (k - 2.5) * w for i in range(len(refs))]
        ax.bar(xs, [ov[n][r]["e"] for r, _ in refs], width=w, color=COLOR[n], label=NAME[n])
    ax.axhline(2.33, color="#e34948", ls="--", lw=1.2)
    ax.text(3.5, 4, "about the 99th percentile of the random sets", ha="right", fontsize=8)
    ax.set_xticks(range(len(refs)), [t for _, t in refs])
    sp = j["summary"]["seed_pair"]["e"]
    ax.text(-0.4, 72, f"positive control, anchor 2 seed 0 vs seed 1: {sp:.0f} sd, off the scale", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 0.93))
    ax.set_ylim(0, 95)
    save(fig, out, "overlap_excess")


def placement(j: dict[str, Any], out: Path) -> None:
    fig, ax = axes("Does the split put more weight on subcomponents Γ switches on than a random change?",
                   "share on switched-on subcomponents, minus null")
    for i, n in enumerate(STATES):
        p = j["state"][n]["placement"]
        base = sum(p["null"]) / len(p["null"])
        ax.errorbar(i, p["excess"], yerr=[[base - min(p["null"])], [max(p["null"]) - base]], fmt="o", ms=8,
                    color=COLOR[n], capsize=4, lw=2)
    pc = j["state"]["e40"]["placement"]["often_on_half"]
    ax.plot(4, pc["excess"], "s", ms=9, color="#e87ba4")
    ax.plot(4, pc["expected"], "_", ms=30, mew=2, color="#e87ba4")
    ax.text(4.25, pc["expected"], "expected", fontsize=8, va="center")
    ax.plot(4, pc["expected"] / 2, "_", ms=30, mew=1, color="#e87ba4")
    ax.text(4.25, pc["expected"] / 2 - 0.012, "half: below it, blind", fontsize=8, va="center")
    ax.text(4.25, pc["excess"] + 0.012, "read", fontsize=8, va="center")
    ax.axhline(0, color="#0b0b0b", lw=1)
    ax.text(-0.4, 0.008, "Gaussian null; bars: spread of its 4 seeds", fontsize=8)
    ax.set_ylim(-0.02, 0.45)
    NAME["pc"] = "E40 split onto\noften-on half\n(positive control)"
    ticks(ax, STATES + ("pc",))
    save(fig, out, "placement")


def where(j: dict[str, Any], out: Path) -> None:
    r4 = j["A"]["row4"]
    names = STATES + ("null0",)
    fig, ax = axes("Which matrix kinds does each update's weight change sit in?", "share of the squared norm")
    w = 0.16
    for k, n in enumerate(names):
        xs = [i + (k - 2) * w for i in range(len(KINDS))]
        ax.bar(xs, [r4[n]["kind"][q] for q in KINDS], width=w, color=COLOR.get(n, "#b9b8b2"),
               label=NAME.get(n, "Gaussian null, seed 0"))
    ax.set_xticks(range(len(KINDS)), KINDS)
    ax.legend(fontsize=8, frameon=False)
    save(fig, out, "where_kind")


def main(res: Path, out: Path) -> None:
    out.mkdir(exist_ok=True)
    j = load(res)
    for f in (regating, direct, switched, overlap, placement, where):
        f(j, out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
