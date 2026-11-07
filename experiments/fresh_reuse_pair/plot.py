"""E125's four figures.

possession.png: the repeat probe's held-out accuracy per block, at the trigger and at the word's own position, beside
the per-type control and E112's case probe at the trigger, with the 0.90 bar.
definition.png: held-out carrying and the readout-direction removal share per state, with their bars, the random
direction's share, and E112's case-label states as the reference.
route.png: each fine-tune's least-squares split and its teacher-trained copy on E12's route axes, with E12's label
lines; references E123's case-label states, E119's trigger fine-tunes and E12 Amendment 1's fine-tunes.
removal.png: D(S) of the switched-on set per state beside its 20 random same-size sets, with E123's states as the
reference and E38 Amendment 2's seed-rule lines.
amend1_sites.png: Amendment 1's readout-direction removal share at the word's position (blocks 0-2) and the trigger
(blocks 1-2), real vs twin vs random direction, with the 0.50 bar.

    python experiments/fresh_reuse_pair/plot.py   -> figures/{possession,definition,route,removal,amend1_sites}.png
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
RESULTS, FIGURES = HERE / "results", HERE / "figures"
EXP = HERE.parent
E112 = EXP / "case_label" / "results"
E123 = json.loads((EXP / "case_label_both" / "results" / "case_label_both.json").read_text())
E119 = EXP / "teacher_route_split" / "results"
E12A1 = EXP / "new_vs_existing" / "results" / "new_vs_existing_amend1.json"
S0, S1, S2 = (json.loads((RESULTS / f"stage{i}.json").read_text()) for i in range(3))
PRED = json.loads((RESULTS / "predictions.json").read_text())
LINES = PRED["decomposition"]["route_lines"]
STATES = ["real_s0", "real_s1", "twin_s0", "twin_s1"]
TICK = ["real s0", "real s1", "twin s0", "twin s1"]
COLOUR = {"real": "#2a78d6", "twin": "#eb6834"}  # the same slots as E123's figures
INK, MUTED, REF, REGION = "#2b2b2b", "#8a8a85", "#b9b8ad", "#efeee9"


def arm(n: str) -> str:
    return n.split("_")[0]


def possession_figure() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    p0, blocks = S0["P0"], range(4)
    e112 = json.loads((E112 / "stage0.json").read_text())["P0"]["acc_case"]
    ax.plot(blocks, p0["acc_repeat"], "o-", color=INK, lw=2, ms=7, label="repeat probe at the trigger (scored)")
    ax.plot(blocks, p0["beside_acc_at_word"], "s--", color=INK, lw=1.2, ms=6, mfc="white", label="repeat probe at the word's own position")
    ax.plot(blocks, p0["acc_control"], "o:", color=MUTED, lw=1.2, ms=5, label="control: one random value per word type")
    ax.plot(blocks, e112, "D-", color=REF, lw=1.2, ms=5, label="reference: E112's case probe at the trigger")
    bar = PRED["rules"]["P0"]["accuracy"]
    ax.axhline(bar, color=MUTED, lw=0.8, ls="--")
    ax.text(3.05, bar + 0.01, "possession bar 0.90", fontsize=7, color=MUTED, ha="right", va="bottom")
    ax.axvspan(2.5, 3.5, color=REGION, zorder=0)
    ax.text(3.0, 0.42, "last block:\ndoes not count", fontsize=7, color=MUTED, ha="center")
    ax.set_xticks(list(blocks), [f"block {b}" for b in blocks])
    ax.set_xlim(-0.3, 3.5)
    ax.set_ylim(0.22, 1.02)
    ax.set_ylabel("held-out-word accuracy of the linear readout")
    ax.set_title("Is 'the word occurred earlier' linearly readable before the fine-tune?", fontsize=9)
    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0.1, 0.0), frameon=False)
    finish(fig, [ax], "possession.png")


def definition_figure() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), sharey=True)
    e112 = json.loads((E112 / "stage2.json").read_text())["states"]
    rules = PRED["rules"]["real"]
    panels = [("P1", "carrying: held-out-word label accuracy", rules["P1"]),
              ("P3", "removal: share of the gain lost when\nthe readout direction is removed", rules["P3"])]  # fmt: skip
    for ax, (key, title, bar) in zip(axes, panels, strict=True):
        for i, n in enumerate(STATES):
            ax.plot(i, S1["reads"][n][key], "o", ms=10, color=COLOUR[arm(n)], mec="white", mew=1.5, zorder=4)
            ax.plot(i + 0.25, e112[n][key], "D", ms=6, mfc="none", mec=REF, mew=1.3, zorder=3)
            if key == "P3":
                ax.plot(i, S1["do_nothing"][n]["share"], "_", ms=14, mew=1.5, color=MUTED, zorder=3)
        ax.axhline(bar, color=MUTED, lw=0.8, ls="--")
        ax.text(3.4, bar + 0.015, f"bar {bar:.2f}", fontsize=7, color=MUTED, ha="right", va="bottom")
        ax.set_xticks(range(4), TICK, fontsize=8)
        ax.set_xlim(-0.4, 3.5)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("value")
    handles = [Line2D([], [], ls="", marker="o", ms=9, color=COLOUR["real"], label="repeat-label real arm"),
               Line2D([], [], ls="", marker="o", ms=9, color=COLOUR["twin"], label="memorising twin"),
               Line2D([], [], ls="", marker="D", ms=6, mfc="none", mec=REF, label="E112 case-label state, same slot"),
               Line2D([], [], ls="", marker="_", ms=14, mew=1.5, color=MUTED, label="random direction, same removal")]
    axes[1].legend(handles=handles, fontsize=7, loc="center right", frameon=False)
    fig.suptitle("The real fine-tune carries to new words, but removing the readout direction barely touches it", fontsize=10)
    finish(fig, list(axes), "definition.png")


def point(shares: dict[str, float]) -> tuple[float, float]:
    return shares["e_gate"], shares["r"]


def regions(ax: Any) -> None:
    """E12's filed lines: re-switching r >= r_hi with e_gate <= masked_lo; masked-off e_gate >= masked_hi with r <= r_lo."""
    lo, hi = -0.3, 1.0
    ax.fill_between([lo, LINES["masked_lo"]], LINES["r_hi"], hi, color=REGION, zorder=0)
    ax.fill_between([LINES["masked_hi"], hi], lo, LINES["r_lo"], color=REGION, zorder=0)
    ax.text(lo + 0.02, hi - 0.03, "re-switching", fontsize=8, color=MUTED, va="top")
    ax.text(hi - 0.02, lo + 0.02, "masked-off", fontsize=8, color=MUTED, ha="right")
    ax.text(0.37, 0.95, "neither", fontsize=8, color=MUTED, ha="center")
    for x in (LINES["masked_lo"], LINES["masked_hi"]):
        ax.axvline(x, color=MUTED, lw=0.6, ls="--", zorder=1)
    for y in (LINES["r_lo"], LINES["r_hi"]):
        ax.axhline(y, color=MUTED, lw=0.6, ls="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def references(ax: Any) -> None:
    for path in sorted(E119.glob("teacher_route_split_*.json")):
        for t in json.loads(path.read_text())["teachers"].values():
            ax.plot(*point(t["least_squares"]["shares"]), "D", ms=6, mfc="none", mec=REF, mew=1.2, zorder=2)
    for c in json.loads(E12A1.read_text())["c0"].values():
        ax.plot(*point(c["least_squares"]["shares"]), "s", ms=6, mfc="none", mec="#6f6e66", mew=1.2, zorder=2)
    for n in STATES:
        ax.plot(*point(E123["routes"][n]["least_squares"]["shares"]), "^", ms=7, mfc="none", mec=COLOUR[arm(n)], mew=1.0, alpha=0.6, zorder=2)
    ax.annotate("E12 copy tables", (0.06, 0.81), xytext=(0.12, 0.88), fontsize=7, color=MUTED)
    ax.annotate("E12 random maps", (0.52, 0.27), xytext=(0.36, 0.40), fontsize=7, color=MUTED)
    ax.annotate("E119 trigger tasks", (0.0, 0.52), xytext=(-0.28, 0.40), fontsize=7, color=MUTED)


def route_figure() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    regions(ax)
    references(ax)
    for n in STATES:
        colour, st = COLOUR[arm(n)], S2["states"][n]
        fx, fy = point(st["least_squares"]["shares"])
        sx, sy = point(st["student"]["a"]["shares"])
        ax.plot([fx, sx], [fy, sy], color=colour, lw=1, alpha=0.6, zorder=3)
        ax.plot(fx, fy, "D", ms=8, mfc="white", mec=colour, mew=2, zorder=4)
        ax.plot(sx, sy, "o", ms=8, color=colour, mec="white", mew=1.5, zorder=5)
    handles = [Line2D([], [], ls="", marker="o", ms=8, color=COLOUR["real"], label="real repeat-label arm (2 seeds)"),
               Line2D([], [], ls="", marker="o", ms=8, color=COLOUR["twin"], label="memorising twin (2 seeds)"),
               Line2D([], [], ls="", marker="D", ms=8, mfc="white", mec=INK, label="fine-tune, least-squares split"),
               Line2D([], [], ls="", marker="o", ms=8, color=INK, label="its teacher-trained copy"),
               Line2D([], [], ls="", marker="^", ms=7, mfc="none", mec=INK, label="E123 case-label fine-tunes, same colours"),
               Line2D([], [], ls="", marker="D", ms=6, mfc="none", mec=REF, label="E119 trigger fine-tunes"),
               Line2D([], [], ls="", marker="s", ms=6, mfc="none", mec="#6f6e66", label="E12 A1 fine-tunes (copy tables, random maps)")]
    ax.legend(handles=handles, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.86), frameon=False)
    ax.set_xlabel("masked-off share (gain carried by subcomponents Γ keeps off)")
    ax.set_ylabel("re-switching share r (gain from Γ switching released subcomponents on)")
    ax.set_title("The repeat-label pair lands in the same masked-off corner as the case-label pair", fontsize=10)
    finish(fig, [ax], "route.png")


def removal_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    rows: list[tuple[str, list[float], float, str, str]] = []
    for n in STATES:
        rec = S2["states"][n]["removal"]
        rows.append((COLOUR[arm(n)], rec["D"]["random"], rec["D"]["S"], rec["class"], f"{n.replace('_', ' ')}\n|ΔW| {S1['states'][n]['norm']:.1f}"))
    rows.append(("", [], 0.0, "", ""))
    e112_norm = json.loads((E112 / "stage1.json").read_text())["states"]
    for n in STATES:
        rec = E123["removal"][n]
        rows.append((REF, rec["D_random"], rec["D_S"], rec["cls"], f"{n.replace('_', ' ')}\n|ΔW| {e112_norm[n]['norm']:.1f}"))
    for i, (colour, rand, d_s, cls, _) in enumerate(rows):
        if not colour:
            continue
        ax.plot([i] * len(rand), rand, "_", ms=12, mew=1.2, color=MUTED, zorder=2)
        ax.plot(i, d_s, "o", ms=10, color=colour, mec="white", mew=1.5, zorder=4)
        ax.annotate(cls, (i, d_s), xytext=(9, -3), textcoords="offset points", fontsize=8, color=INK)
    ax.axhline(0.10, color=MUTED, lw=0.7, ls="--")
    ax.axhline(0.05, color=MUTED, lw=0.7, ls=":")
    ax.axhline(0.0, color=INK, lw=0.5)
    ax.text(4, 0.105, "through: ≥ 0.10\nand above all 20", fontsize=7, color=MUTED, ha="center", va="bottom")
    ax.text(4, 0.045, "not: below 0.05", fontsize=7, color=MUTED, ha="center", va="top")
    ax.set_xticks(range(len(rows)), [r[4] for r in rows], fontsize=7)
    ax.text(1.5, -0.2, "E125: repeat-label pair", ha="center", fontsize=8, color=INK, transform=ax.get_xaxis_transform())
    ax.text(6.5, -0.2, "reference: E123, case-label pair", ha="center", fontsize=8, color=MUTED, transform=ax.get_xaxis_transform())
    ax.set_ylim(-0.12, 0.72)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    handles = [Line2D([], [], ls="", marker="o", ms=9, color=COLOUR["real"], label="switched-on set S, real arm"),
               Line2D([], [], ls="", marker="o", ms=9, color=COLOUR["twin"], label="switched-on set S, twin"),
               Line2D([], [], ls="", marker="o", ms=9, color=REF, label="switched-on set S, E123"),
               Line2D([], [], ls="", marker="_", ms=12, mew=1.2, color=MUTED, label="20 random sets of the same size")]
    ax.legend(handles=handles, fontsize=7, loc="upper left", frameon=False)
    ax.set_ylabel("D: share of the gain lost by removing the set,\nfine-tune minus original model")
    ax.set_title("Removing the switched-on set costs every fine-tune more than chance, the memorising twin most", fontsize=9)
    finish(fig, [ax], "removal.png")


def amend1_figure() -> None:
    """Amendment 1: the readout-direction removal share at each site, real vs twin vs random, with E125's 0.50 bar; the
    filed site (trigger, block 1) from stage1.json beside the four new sites from amend1.json."""
    a1 = json.loads((RESULTS / "amend1.json").read_text())["sites"]
    filed = {"probe": {n: {"share": S1["reads"][n]["P3"]} for n in STATES}, "random": S1["do_nothing"]}
    sites = [("word\nblock 0", a1["word_b0"]), ("word\nblock 1", a1["word_b1"]), ("word\nblock 2", a1["word_b2"]),
             ("trigger\nblock 1\nfiled site", filed), ("trigger\nblock 2", a1["trigger_b2"])]  # fmt: skip
    bar = PRED["rules"]["real"]["P3"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [1, 2.2]})
    for ax in axes:
        for i, (_, rec) in enumerate(sites):
            for j, n in enumerate(STATES):
                x = i + (j - 1.5) * 0.12
                ax.plot(x, rec["random"][n]["share"], "_", ms=9, mew=1.5, color=MUTED, zorder=3)
                ax.plot(x, rec["probe"][n]["share"], "o", ms=7, color=COLOUR[arm(n)], mec="white", mew=1.0, zorder=4)
        ax.axvspan(2.5, 3.5, color=REGION, zorder=0)
        ax.axhline(0.0, color=INK, lw=0.5)
        ax.set_xticks(range(len(sites)), [s[0] for s in sites], fontsize=7)
        ax.set_xlim(-0.5, len(sites) - 0.5)
    axes[0].axhline(bar, color=MUTED, lw=0.8, ls="--")
    axes[0].text(len(sites) - 0.6, bar + 0.01, f"removal bar {bar:.2f}", fontsize=7, color=MUTED, ha="right", va="bottom")
    axes[0].set_ylim(-0.02, 0.56)
    axes[0].set_title("whole scale, with the bar", fontsize=9)
    axes[0].set_ylabel("share of the gain lost when the\nreadout direction is removed")
    axes[1].set_ylim(-0.005, 0.05)
    axes[1].set_title("zoomed to 0-0.05", fontsize=9)
    handles = [Line2D([], [], ls="", marker="o", ms=7, color=COLOUR["real"], label="repeat-label real arm, 2 seeds"),
               Line2D([], [], ls="", marker="o", ms=7, color=COLOUR["twin"], label="memorising twin, 2 seeds"),
               Line2D([], [], ls="", marker="_", ms=9, mew=1.5, color=MUTED, label="random direction, same removal")]
    axes[1].legend(handles=handles, fontsize=7, loc="upper left", frameon=False)
    fig.suptitle("Moved to the word or to block 2, the removal takes 0.02-0.05 of the real gain, far under 0.50", fontsize=10)
    finish(fig, list(axes), "amend1_sites.png")


def finish(fig: Any, axes: list[Any], name: str) -> None:
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150)
    print(FIGURES / name)


if __name__ == "__main__":
    possession_figure()
    definition_figure()
    route_figure()
    removal_figure()
    amend1_figure()
