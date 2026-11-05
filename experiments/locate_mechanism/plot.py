"""E96's figures, one per claim, from results/*.json; no model.

  figures/copy_lookup.png       all 24 heads: net share on copy against lookup
  figures/lowercase_cipher.png  the same plane, on lowercase against the cipher
  figures/own_terms.png         net share split into the state's own fall and W0's movement, per top part
  figures/salience.png          head salience against the true fall, with the path integral on the same axes
  figures/probe.png             probe accuracy against share lost, copy and lookup
  figures/mlp_units.png         top 320 MLP units against 20 random sets, task and control
  figures/capability.png        share lost against W0's own capability with the part off, copy and lowercase

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/locate_mechanism/plot.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())


S1, S2 = load("sweep.json"), load("stage2_sweep.json")
SCORE, SCORE2, NOM, BESIDE, GRAD = load("score.json"), load("stage2_score.json"), load("nominators.json"), load("stage2_beside.json"), load("stage2_gradient.json")
HEADS = sorted(S1["w0"]["top1_off"])
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1f8a6d"
INK, MUTED, GRID, BAND, DRAW = "#2b2a28", "#52514e", "#e4e3df", "#ecebe6", "#8f8e88"
SURFACE = "#fcfcfb"
LOCATED_SHARE, CONTROL_MAX, XMAX = 0.47, 0.2, 0.82
SEEDS1, SEEDS2 = (0, 1, 2), (0, 1)


def axes(ax: Any, xlabel: str, ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)


def title(ax: Any, text: str) -> None:
    ax.set_title(text, color=INK, loc="left", fontsize=12, pad=10)


def save(fig: Any, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def part_row(sweep: dict[str, Any], state: str, part: str) -> dict[str, Any]:
    st = sweep["states"][state]
    return st["heads"][part] if part in st["heads"] else sweep["pairs"][part]["states"][state]


def shares(sweep: dict[str, Any], task: str, seeds: tuple[int, ...], part: str) -> list[float]:
    return [part_row(sweep, f"{task}_s{s}", part)["share"] for s in seeds]


def w0_off(sweep: dict[str, Any], task: str, part: str) -> float:
    return sweep["w0"]["lo_off"][task][part] if part in HEADS else sweep["pairs"][part]["lo_off"][task]


def own_terms(sweep: dict[str, Any], task: str, seeds: tuple[int, ...], part: str) -> dict[str, float]:
    """Mean over seeds, as fractions of each state's gain: fall + rise is the net share, exactly."""
    w0 = sweep["w0"]["lo"][task]
    falls, rises, ps = [], [], []
    for s in seeds:
        st = sweep["states"][f"{task}_s{s}"]
        row = part_row(sweep, f"{task}_s{s}", part)
        falls.append((st["lo"] - row["lo"]) / st["G"])
        rises.append((w0_off(sweep, task, part) - w0) / st["G"])
        ps.append(row["p"])
    return {"fall": sum(falls) / len(falls), "rise": sum(rises) / len(rises), "p_lo": min(ps), "p_hi": max(ps)}


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def unit_terms(control: bool) -> dict[str, float]:
    """The saved MLP rows hold probabilities at the mean log-odds and the gains, on the odd windows: the same split."""
    falls, rises, ps = [], [], []
    for s in SEEDS2:
        row = BESIDE["mlp"][f"lowercase_s{s}"]
        row = row["control"] if control else row
        fall = logit(row["p"]) - logit(row["p_off"])
        falls.append(fall / row["G"])
        rises.append((row["G"] - row["G_abl"] - fall) / row["G"])
        ps.append(row["p_off"])
    return {"fall": sum(falls) / 2, "rise": sum(rises) / 2, "p_lo": min(ps), "p_hi": max(ps)}


def share_plane(sweep: dict[str, Any], task: str, control: str, seeds: tuple[int, ...], labels: dict[str, tuple[str, tuple[int, int], str]], heading: str, xlabel: str, ylabel: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.6), facecolor=SURFACE)
    axes(ax, xlabel, ylabel)
    ax.axvspan(LOCATED_SHARE, XMAX, ymin=0, ymax=(CONTROL_MAX + 0.08) / 0.42, color=BAND, zorder=0)
    ax.text(XMAX - 0.01, -0.07, "share over the bar,\ncontrol spared", color=MUTED, fontsize=9, ha="right", va="bottom")
    ax.axvline(LOCATED_SHARE, color=MUTED, linewidth=1, linestyle="--")
    ax.axhline(CONTROL_MAX, color=MUTED, linewidth=1, linestyle=":")
    ax.axvline(0, color=GRID, linewidth=1.4)
    ax.axhline(0, color=GRID, linewidth=1.4)
    ax.text(LOCATED_SHARE - 0.008, 0.335, "share bar 0.47:\nthe copy control's weakest seed", color=MUTED, fontsize=9, ha="right", va="top")
    ax.text(-0.53, CONTROL_MAX + 0.005, "control spared at or under 0.2", color=MUTED, fontsize=9, va="bottom")
    pairs = sorted(sweep["pairs"])
    for part in HEADS + pairs:
        xs, ys = shares(sweep, task, seeds, part), shares(sweep, control, seeds, part)
        x, y = sum(xs) / len(xs), sum(ys) / len(ys)
        col = labels[part][2] if part in labels else DRAW
        ax.plot([min(xs), max(xs)], [y, y], color=col, linewidth=1.2, zorder=2)
        ax.plot([x, x], [min(ys), max(ys)], color=col, linewidth=1.2, zorder=2)
        marker = {"marker": "D", "facecolor": SURFACE, "edgecolor": col, "linewidth": 1.5, "s": 52} if part in pairs else {"marker": "o", "color": col, "s": 62 if part in labels else 30}
        ax.scatter([x], [y], zorder=3, **marker)
        if part in labels:
            ax.annotate(labels[part][0], (x, y), xytext=labels[part][1], textcoords="offset points", color=col, fontsize=10)
    ax.set_xlim(-0.55, XMAX)
    ax.set_ylim(-0.08, 0.34)
    keys = [Line2D([], [], marker="o", color=DRAW, linestyle="", label="one head, mean over seeds, lines the seeds' range"), Line2D([], [], marker="D", markerfacecolor=SURFACE, color=DRAW, linestyle="", label="one pair found greedily")]
    ax.legend(handles=keys, frameon=False, labelcolor=MUTED, fontsize=9, loc="upper left")
    title(ax, heading)
    save(fig, name)


def copy_lookup() -> None:
    labels = {"2,4": ("head (2,4): state's probability 0.99 to 0.024-0.071", (-150, 12), BLUE), "1,1": ("head (1,1)", (6, 7), ORANGE), "1,4": ("(1,4), lookup's best head", (-8, 9), MUTED), "2,4+1,1": ("(2,4) + (1,1)", (-34, 10), BLUE), "1,4+3,4": ("(1,4) + (3,4), lookup's best pair", (8, 2), MUTED)}
    share_plane(S1, "copy", "lookup", SEEDS1, labels, "Which head carries the copying gain and spares the lookup? Head (2,4), alone of the 24", "share of the copying gain lost, net of W0 (0: no loss, 1: all of it)", "share of the lookup gain lost, net of W0", "copy_lookup.png")


def lowercase_cipher() -> None:
    labels = {"1,4": ("(1,4), best head", (-20, -19), ORANGE), "1,5": ("(1,5)", (-40, 3), MUTED), "0,0": ("(0,0)", (-14, 10), MUTED), "0,5": ("(0,5)", (-34, -2), MUTED), "2,4": ("(2,4), the copying head", (-60, -18), BLUE), "2,2": ("(2,2)", (8, 5), MUTED), "1,4+1,5": ("(1,4) + (1,5)", (10, -4), ORANGE), "0,0+1,5": ("(0,0) + (1,5)", (10, -2), MUTED)}
    share_plane(S2, "lowercase", "cipher", SEEDS2, labels, "Does any head or pair carry the lowercase gain and spare the cipher? No", "share of the lowercase gain lost, net of W0 (0: no loss, 1: all of it)", "share of the cipher gain lost, net of W0", "lowercase_cipher.png")


def own_terms_figure() -> None:
    rows: list[tuple[str, dict[str, float]]] = [
        ("copying, head (2,4)", own_terms(S1, "copy", SEEDS1, "2,4")),
        ("lowercase, (1,4)", own_terms(S2, "lowercase", SEEDS2, "1,4")),
        ("lowercase, (0,0)", own_terms(S2, "lowercase", SEEDS2, "0,0")),
        ("lowercase, (1,5)", own_terms(S2, "lowercase", SEEDS2, "1,5")),
        ("lowercase, (0,5)", own_terms(S2, "lowercase", SEEDS2, "0,5")),
        ("lowercase, (1,4) + (1,5)", own_terms(S2, "lowercase", SEEDS2, "1,4+1,5")),
        ("lowercase, (0,0) + (1,5)", own_terms(S2, "lowercase", SEEDS2, "0,0+1,5")),
        ("lowercase, (1,1), the largest head fall", own_terms(S2, "lowercase", SEEDS2, "1,1")),
        ("cipher, (1,4)", own_terms(S2, "cipher", SEEDS2, "1,4")),
        ("cipher, (1,4) + (1,5)", own_terms(S2, "cipher", SEEDS2, "1,4+1,5")),
        ("lowercase, its top 320 MLP units", unit_terms(False)),
        ("cipher, lowercase's 320 units", unit_terms(True)),
    ]
    fig, ax = plt.subplots(figsize=(11.2, 6.4), facecolor=SURFACE)
    axes(ax, "fraction of the state's own gain, mean over seeds; the two bars add up to the net share\nintact probability: copying 0.99, lowercase 0.006, cipher 0.002; W0 on lowercase 0.0012")
    ax.grid(axis="y", visible=False)
    for i, (_, r) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, r["fall"], left=0, height=0.6, color=BLUE, zorder=2)
        ax.barh(y, r["rise"], left=r["fall"] if r["rise"] > 0 and r["fall"] > 0 else 0, height=0.6, color=ORANGE, zorder=2)
        ax.scatter([r["fall"] + r["rise"]], [y], marker="D", s=44, facecolor=SURFACE, edgecolor=INK, zorder=4)
        ax.text(1.36, y, f"{r['p_lo']:.4f}-{r['p_hi']:.4f}", color=MUTED, fontsize=9, va="center", ha="right")
    ax.text(1.36, len(rows) - 0.45, "state's probability,\npart off", color=MUTED, fontsize=9, va="bottom", ha="right")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.axvline(LOCATED_SHARE, color=MUTED, linestyle="--", linewidth=1)
    ax.text(LOCATED_SHARE - 0.008, len(rows) - 0.45, "share bar 0.47", color=MUTED, fontsize=9, ha="right")
    ax.axvline(1, color=MUTED, linestyle=":", linewidth=1)
    ax.text(0.992, len(rows) - 0.45, "the whole gain", color=MUTED, fontsize=9, ha="right")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], color=INK)
    ax.set_xlim(-0.75, 1.37)
    ax.set_ylim(-0.6, len(rows) + 0.5)
    keys = [Line2D([], [], color=BLUE, linewidth=7, label="the state's own log-odds fall"), Line2D([], [], color=ORANGE, linewidth=7, label="W0's log-odds move with the part off (right: rises, left: falls)"), Line2D([], [], marker="D", markerfacecolor=SURFACE, color=INK, linestyle="", label="net share")]
    ax.legend(handles=keys, frameon=False, labelcolor=MUTED, fontsize=9, loc="upper center", bbox_to_anchor=(0.42, -0.1), ncol=3)
    fig.suptitle("What is the net share made of? On copying the state falls; on lowercase's top heads about half is W0 rising", color=INK, fontsize=12, x=0.01, ha="left")
    save(fig, "own_terms.png")


def salience() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.0), facecolor=SURFACE)
    axes(ax, "head salience: the first-order estimate of the gain lost, state minus W0 (log-odds)", "true gain lost with the head off, net of W0 (log-odds)")
    for s in SEEDS1:
        st = f"copy_s{s}"
        for h in HEADS:
            x, y = NOM["salience"][st][h], S1["states"][st]["G"] - S1["states"][st]["heads"][h]["G_abl"]
            col = BLUE if h == "2,4" else DRAW
            ax.scatter([x], [y], s=70 if h == "2,4" else 18, color=col, zorder=3 if h == "2,4" else 2)
    ax.plot([-1, 8], [-1, 8], color=MUTED, linewidth=1, linestyle="--")
    ax.text(6.0, 6.45, "estimate = true fall", color=MUTED, fontsize=9, ha="right", rotation=38)
    for case in GRAD["cases"].values():
        ax.scatter([case["net"]["path_fall"]], [case["net"]["true_fall"]], marker="D", s=46, facecolor=SURFACE, edgecolor=ORANGE, linewidth=1.5, zorder=4)
    ax.annotate("(2,4) on seeds 0, 1\nsalience rank 2", (0.5, 7.0), xytext=(1.2, 7.2), color=BLUE, fontsize=10, arrowprops={"arrowstyle": "-", "color": BLUE})
    ax.annotate("(2,4) on seed 2\nsalience rank 19", (-0.05, 5.91), xytext=(0.7, 5.0), color=BLUE, fontsize=10, arrowprops={"arrowstyle": "-", "color": BLUE})
    ax.annotate("(1,4), salience's top head on every seed", (0.57, 2.5), xytext=(1.5, 1.2), color=MUTED, fontsize=10, arrowprops={"arrowstyle": "-", "color": MUTED})
    keys = [Line2D([], [], marker="o", color=DRAW, linestyle="", label="one head on one copy seed"), Line2D([], [], marker="o", color=BLUE, linestyle="", label="head (2,4)"), Line2D([], [], marker="D", markerfacecolor=SURFACE, color=ORANGE, linestyle="", label="the same gradient integrated along the path: (2,4), (1,4), seeds 0 and 2")]
    ax.legend(handles=keys, frameon=False, labelcolor=MUTED, fontsize=9, loc="lower right")
    ax.set_xlim(-1.3, 7.7)
    ax.set_ylim(-1.3, 7.9)
    title(ax, "Does one gradient step rank the heads? No, though the same gradient integrated matches")
    save(fig, "salience.png")


def probe() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True, facecolor=SURFACE)
    for ax, task, top in zip(axs, ("copy", "lookup"), ("2,4", "1,4"), strict=True):
        axes(ax, f"share of the {'copying' if task == 'copy' else 'lookup'} gain lost with the head off, net of W0", "probe accuracy on the held-out windows" if task == "copy" else "")
        for s in SEEDS1:
            st = f"{task}_s{s}"
            for h in HEADS:
                x, y = S1["states"][st]["heads"][h]["share"], NOM["probe"][st][h]
                ax.scatter([x], [y], s=70 if h == top else 18, color=BLUE if h == top else DRAW, zorder=3 if h == top else 2)
                ax.scatter([x], [NOM["probe"][f"W0_{task}"][h]], s=14, facecolor="none", edgecolor=DRAW, linewidth=0.6, zorder=1)
    axs[0].annotate("(2,4): 0.14-0.15,\nthe next head 0.12-0.13", (0.51, 0.15), xytext=(0.2, 0.42), color=BLUE, fontsize=10, arrowprops={"arrowstyle": "-", "color": BLUE})
    axs[1].annotate("(1,4), lookup's top head:\n0.27-0.32, rank 22 of 24", (0.128, 0.28), xytext=(0.075, 0.1), color=BLUE, fontsize=10, arrowprops={"arrowstyle": "-", "color": BLUE})
    keys = [Line2D([], [], marker="o", color=DRAW, linestyle="", label="one head on one state"), Line2D([], [], marker="o", color=BLUE, linestyle="", label="the sweep's top head"), Line2D([], [], marker="o", markerfacecolor="none", color=DRAW, linestyle="", label="the same head at W0")]
    axs[0].legend(handles=keys, frameon=False, labelcolor=MUTED, fontsize=9, loc="upper left")
    axs[0].set_ylim(-0.03, 1.03)
    fig.suptitle("Does a probe show which head is used? Not on the lookup: many heads know the answer, the used one reads low", color=INK, fontsize=12, x=0.01, ha="left")
    save(fig, "probe.png")


def mlp_units() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.0), facecolor=SURFACE)
    axes(ax, "share of the gain lost with the 320 units off, net of W0 (0: no loss, 1: all of it)")
    ax.grid(axis="y", visible=False)
    rows: list[tuple[str, float, list[float], str]] = []
    for s in SEEDS1:
        r = NOM["mlp"][f"copy_s{s}"]
        rows.append((f"copying, seed {s}", r["share"], r["random_shares"], BLUE))
    for s in SEEDS1:
        r = NOM["mlp"][f"copy_s{s}"]
        rows.append((f"lookup state, copying's units, seed {s}", r["control"]["share"], [], GREEN))
    for s in SEEDS2:
        r = BESIDE["mlp"][f"lowercase_s{s}"]
        rows.append((f"lowercase, seed {s}", r["share"], r["random_shares"], ORANGE))
    for s in SEEDS2:
        r = BESIDE["mlp"][f"lowercase_s{s}"]
        rows.append((f"cipher state, lowercase's units, seed {s}", r["control"]["share"], [], GREEN))
    for i, (_, share, draws, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.scatter(draws, [y] * len(draws), marker="|", s=44, color=DRAW, zorder=2)
        ax.scatter([share], [y], s=70, color=col, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.axvline(CONTROL_MAX, color=MUTED, linestyle=":", linewidth=1)
    ax.text(CONTROL_MAX + 0.005, len(rows) - 0.55, "control spared at or under 0.2", color=MUTED, fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], color=INK)
    ax.set_xlim(-0.3, 0.55)
    ax.set_ylim(-0.6, len(rows))
    keys = [Line2D([], [], marker="|", color=DRAW, linestyle="", markersize=9, label="20 random sets of 320 units"), Line2D([], [], marker="o", color=DRAW, linestyle="", label="the top 320 units by salience, the state's own set")]
    ax.legend(handles=keys, frameon=False, labelcolor=MUTED, fontsize=9, loc="lower left")
    title(ax, "Do the top MLP units carry the gain alone? Copying's cost the lookup too")
    save(fig, "mlp_units.png")


def capability() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 5.0), sharex=True, facecolor=SURFACE)
    left, right = axs
    axes(left, "share of the copying gain lost, net of W0", "W0's second-copy top-1 with the head off")
    axes(right, "share of the lowercase gain lost, net of W0", "W0's lower-minus-capital log-probability, head off")
    base, gap = S1["w0"]["top1_base"], S2["w0"]["gap"]
    for h in HEADS:
        sx = shares(S1, "copy", SEEDS1, h)
        named = h in ("2,4", "1,1", "1,5", "1,4")
        left.scatter([sum(sx) / 3], [S1["w0"]["top1_off"][h]], s=62 if named else 26, color=BLUE if h == "2,4" else (ORANGE if named else DRAW), zorder=3)
        lx = shares(S2, "lowercase", SEEDS2, h)
        named2 = h in ("2,4", "1,4", "1,5", "0,0", "0,5")
        right.scatter([sum(lx) / 2], [S2["w0"]["gap_off"][h]], s=62 if named2 else 26, color=BLUE if h == "2,4" else (ORANGE if named2 else DRAW), zorder=3)
    for ax, value, text in ((left, base, "intact 0.82"), (right, gap, "intact 0.25")):
        ax.axhline(value, color=INK, linewidth=1)
        ax.text(-0.53, value, text, color=INK, fontsize=9, va="bottom")
    for ax, bar, text in ((left, 0.5 * base, "'gain only' at or over half of intact"), (right, 0.5 * gap, "")):
        ax.axhline(bar, color=MUTED, linewidth=1, linestyle="--")
        if text:
            ax.text(-0.53, bar, text, color=MUTED, fontsize=9, va="bottom")
    left.axhline(0.1 * base, color=MUTED, linewidth=1, linestyle=":")
    left.text(-0.53, 0.1 * base, "'shared' at or under a tenth of intact", color=MUTED, fontsize=9, va="bottom")
    right.axhline(S2_A0, color=GREEN, linewidth=1, linestyle="-.")
    right.text(-0.53, S2_A0 + 0.02, "1.18: the same gap at other positions", color=GREEN, fontsize=9, va="bottom")
    labels_left = {"2,4": ((-10, -14), BLUE), "1,1": ((-34, 8), ORANGE), "1,5": ((8, -2), ORANGE), "1,4": ((8, -2), ORANGE)}
    for h, (off, col) in labels_left.items():
        left.annotate(f"({h})", (sum(shares(S1, "copy", SEEDS1, h)) / 3, S1["w0"]["top1_off"][h]), xytext=off, textcoords="offset points", color=col, fontsize=10)
    for h, off in {"1,4": (8, -2), "1,5": (-38, 6), "0,0": (8, -4), "0,5": (-34, 3), "2,4": (8, -2)}.items():
        right.annotate(f"({h})", (sum(shares(S2, "lowercase", SEEDS2, h)) / 2, S2["w0"]["gap_off"][h]), xytext=off, textcoords="offset points", color=BLUE if h == "2,4" else ORANGE, fontsize=10)
    for ax in axs:
        ax.axvline(LOCATED_SHARE, color=MUTED, linewidth=1, linestyle="--")
    left.set_xlim(-0.55, 0.68)
    left.text(0.49, 0.62, "gain gone,\ncapability\nkept:\nempty", color=MUTED, fontsize=9, va="center")
    right.text(0.49, 0.62, "no head\nreaches\nthe bar", color=MUTED, fontsize=9, va="center")
    right.text(-0.53, 0.5 * gap, "'gain only' bar", color=MUTED, fontsize=9, va="top")
    fig.suptitle("Does switching a head off take the gain and leave W0's capability? On copying no; on lowercase the test cannot be read", color=INK, fontsize=12, x=0.01, ha="left")
    save(fig, "capability.png")


S2_A0 = load("stage2_stage0.json")["w0_capability"]["a0_reference_other_positions"]


def check_own_terms() -> None:
    """The two bars add up to the net share, on every part plotted; a plot of a wrong split would otherwise look fine."""
    for sweep, task, seeds in ((S1, "copy", SEEDS1), (S2, "lowercase", SEEDS2), (S2, "cipher", SEEDS2)):
        for s in seeds:
            for part in HEADS:
                st = sweep["states"][f"{task}_s{s}"]
                row = st["heads"][part]
                fall, rise = st["lo"] - row["lo"], w0_off(sweep, task, part) - sweep["w0"]["lo"][task]
                assert math.isclose((fall + rise) / st["G"], row["share"], abs_tol=1e-6), (task, s, part)


if __name__ == "__main__":
    check_own_terms()
    copy_lookup()
    lowercase_cipher()
    own_terms_figure()
    salience()
    probe()
    mlp_units()
    capability()
