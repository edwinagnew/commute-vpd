"""Figures for E61 Job S, the direct share against size. Reads result files only, loads no model. Any further
results/size_state_gold_1e-5_<step>.json appears on the gold line as an exploration read, with no change here.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_recipe/plot_size.py

  size_share.png     direct share against K00 - K0: gold's states, each random kind's two reads, the two filed lines
  size_share_kl.png  the same against the KL from the unchanged model
  size_pick.png      which gold state was read: 16-row size per step against the random changes' sizes
  size_scale.png     size against the scale b of each random change, with the square law from its first read
  log_size.png       size_share.png for the research log
"""
# pyright: reportMissingImports=false
import math
from typing import Any

import matplotlib.pyplot as plt
from plot import FIG, INK, MUTED, R, load, save

GOLD, ALIVE, GAUSS = "#2a78d6", "#eb6834", "#1baf7a"  # the validated categorical order, slots 1, 2, 3
KINDS = {"inspan": ("Alive-subcomponent random change", ALIVE, "s"), "gauss": ("Gaussian random change", GAUSS, "^")}
LADDER = "gold_1e-5_"
SIZE_AXIS = "K00_minus_K0"
AXIS_WORDS = {
    SIZE_AXIS: "Size: K00 − K0, KL from the changed model to the masked original,\nless the same at the unchanged model (nats per position)",
    "KL_W0_state": "Size, second measure: unmasked KL from the unchanged model\nto the changed one (nats per position)",
}
LOG_AXIS = "How far the change moved the model, measured against\nthe decomposition's masked output (nats per position)"


def read_states() -> dict[str, dict[str, Any]]:
    """Every state's truly-original-masks read, by name; the rehearsal's size2_* files do not match."""
    prefix = "size_state_"
    return {p.stem[len(prefix):]: load(p)["read"]["true"] for p in sorted(R.glob(f"{prefix}*.json"))}


def bar(ax: Any, x: float, rec: dict[str, Any], colour: str, marker: str, hollow: bool = False, **kw: Any) -> None:
    low, high = rec["boot95"]
    ax.errorbar([x], [rec["share"]], yerr=[[rec["share"] - low], [high - rec["share"]]], color=colour, marker=marker, ms=8,
                mfc="white" if hollow else colour, mew=1.5, capsize=2.5, lw=1.2, ls="none", zorder=4, **kw)


def draw_randoms(ax: Any, states: dict[str, Any], reads: dict[str, Any], axis: str) -> None:
    """Each kind's two reads joined; dotted, what the filed rule reads off them at gold's two sizes."""
    for kind, (label, colour, marker) in KINDS.items():
        own = [states[kind], states[f"{kind}_big"]]
        for rec in own:
            bar(ax, rec[axis], rec, colour, marker)
        ax.plot([r[axis] for r in own], [r["share"] for r in own], color=colour, lw=1.8, marker=marker, ms=8, label=label, zorder=3)
        off = [(reads[size][axis]["gold_size"], reads[size][axis]["random_at_size"][kind]) for size in reads]
        points = sorted(off + [(r[axis], r["share"]) for r in own])
        ax.plot(*zip(*points, strict=True), color=colour, lw=1.1, ls=":", zorder=2)


def draw_gold(ax: Any, states: dict[str, Any], pick: dict[str, str], axis: str, plain: bool) -> None:
    ladder = sorted((int(name[len(LADDER):]), name) for name in states if name.startswith(LADDER))
    ax.plot([states[n][axis] for _, n in ladder], [states[n]["share"] for _, n in ladder], color=GOLD, lw=1.8, zorder=3,
            label="Gold fine-tune, full-precision forward: scored states")
    extra = False
    for step, name in ladder:
        scored = name in pick.values()
        extra = extra or not scored
        bar(ax, states[name][axis], states[name], GOLD, "o", hollow=not scored)
        ax.annotate(f"step {step}", (states[name][axis], states[name]["share"]), xytext=(-7, 7), textcoords="offset points", ha="right", fontsize=8.5)
    if extra:
        ax.plot([], [], color=GOLD, marker="o", mfc="white", mew=1.5, ls="none", label="Gold fine-tune: exploration, not in the verdict")
    s0 = states["gold_s0"]
    bar(ax, s0[axis], s0, GOLD, "D", label=f"Earlier gold fine-tune{'' if plain else ', gold_s0'}: 100 steps, half-precision forward")
    ax.annotate("earlier run" if plain else "gold_s0", (s0[axis], s0["share"]), xytext=(-8, 5), textcoords="offset points", ha="right", fontsize=8.5)


def draw_lines(ax: Any, reads: dict[str, Any], axis: str, f: float, multiple: int) -> None:
    """At each scored gold size: the higher random's read-off, gold's gap to it, and the two filed lines as gaps above it."""
    for size, side in (("big", 1), ("small", -1)):
        rec = reads[size][axis]
        x, base = rec["gold_size"], max(rec["random_at_size"].values())
        ax.plot([x, x], [base, rec["gold_share"]], color=INK, lw=1.4, zorder=2)
        span = ax.get_xlim()[1] * 0.018
        for height, text in ((f, "size explains"), (multiple * f, "differs beyond size")):
            ax.plot([x - span, x + span], [base + height] * 2, color=INK, lw=1.4, zorder=2)
            ax.annotate(f"+{height:.2f}: {text}", (x + side * span, base + height), xytext=(side * 4, 0), textcoords="offset points",
                        ha="left" if side > 0 else "right", va="center", fontsize=8, color=MUTED)
        ax.annotate(f"gap {rec['gap']:+.3f}", (x, (base + rec["gold_share"]) / 2), xytext=(5, 0) if side > 0 else (-8, -16), textcoords="offset points",
                    ha="left" if side > 0 else "right", va="center", fontsize=8.5, fontweight="bold")


def fig_share(states: dict[str, Any], axis: str, title: str, xlabel: str, name: str, plain: bool = False) -> None:
    verdict = load(R / "size_verdict.json")
    reads = verdict["true"]["reads"]
    multiple = load(R / "predictions_size.json")["rules"]["b_multiple"]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.set_xlim(0, 1.12 * max(s[axis] for s in states.values()))
    ax.set_ylim(0, 0.135)
    ax.axhline(0, color=MUTED, lw=1.2)
    ax.annotate("do-nothing reads 0: no column moves", (ax.get_xlim()[1], 0), xytext=(-3, 4), textcoords="offset points", ha="right", color=MUTED, fontsize=8.5)
    draw_lines(ax, reads, axis, verdict["floor"]["f"], multiple)
    draw_randoms(ax, states, reads, axis)
    draw_gold(ax, states, verdict["pick"], axis, plain)
    ax.plot([], [], color=MUTED, lw=1.1, ls=":", label="Dotted: a random change read off at gold's size, as the rule does")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Direct share, truly original masks\n(share of the removable error removed by\nchanging only the subcomponents)")
    ax.set_title(title)
    ax.legend(fontsize=8.5, loc="upper left")
    save(fig, name)


def fig_pick() -> None:
    select = load(R / "size_select.json")
    rows = sorted((rec["step"], rec["size_select"]) for rec in select["states"].values())
    gauss, alive = select["targets"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(*zip(*rows, strict=True), color=GOLD, lw=1.8, marker="o", ms=5, label="Gold fine-tune, each saved step")
    for value, (label, colour, _) in ((alive, KINDS["inspan"]), (gauss, KINDS["gauss"])):
        ax.axhline(value, color=colour, lw=1.4, label=f"{label}, at its first scale")
    by_step = dict(rows)
    picks = {8: alive, 10: gauss, 11: gauss}
    offsets = {8: (10, -13), 10: (10, -11), 11: (-6, 9)}
    for step, target in picks.items():
        far = abs(math.log(by_step[step] / target))
        ax.annotate(f"step {step}: {far:.4f}", (step, by_step[step]), xytext=offsets[step], textcoords="offset points", ha="left" if step != 11 else "right", fontsize=8.5)
    ax.scatter([8, 100], [by_step[8], by_step[100]], s=90, facecolor="none", edgecolor=INK, lw=1.4, zorder=5, label="The two states the rule read")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 8, 10, 20, 30, 50, 100], ["1", "2", "3", "5", "8", "10", "20", "30", "50", "100"])
    ax.set_xlabel("Training step of the gold fine-tune (log scale)")
    ax.set_ylabel("Size K00 − K0 on the first 16 rows (nats per position)")
    ax.set_title("Which gold state was read? Labels: distance to the nearest random change, as an absolute log ratio")
    ax.legend(fontsize=8.5, loc="upper left")
    save(fig, "size_pick.png")


def fig_scale() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    target = 0.0
    for kind, (label, colour, marker) in KINDS.items():
        rec = load(R / f"size_search_{kind}.json")
        evals = sorted(rec["search"]["evals"])
        target = rec["search"]["target"]
        ax.plot(*zip(*evals, strict=True), color=colour, lw=1.8, marker=marker, ms=7, label=f"{label}, first 16 rows")
        top = evals[-1][0]
        ax.plot([1, top], [evals[0][1], evals[0][1] * top**2], color=colour, lw=1.2, ls="--")
    ax.plot([], [], color=MUTED, lw=1.2, ls="--", label="Dashed: square law from each kind's first read")
    ax.axhline(target, color=MUTED, lw=1)
    ax.annotate("gold fine-tune at step 100, first 16 rows: the search's target", (1.0, target), xytext=(3, 4), textcoords="offset points", color=MUTED, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [1.0, 1.1, 1.2, 1.3, 1.4]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    yticks = [0.7, 0.8, 1.0, 1.2, 1.5]
    ax.set_yticks(yticks, [str(t) for t in yticks])
    ax.set_xlabel("Scale b of the random change, W0 + b δW (log scale)")
    ax.set_ylabel("Size K00 − K0 (nats per position, log scale)")
    ax.set_title("Does size grow as the square of the scale? Both random changes climb faster, the Gaussian most")
    ax.legend(fontsize=8.5, loc="lower right")
    save(fig, "size_scale.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    all_states = read_states()
    fig_share(all_states, SIZE_AXIS, "Does the gold fine-tune's direct share stay above random changes of the same size?", AXIS_WORDS[SIZE_AXIS], "size_share.png")
    fig_share(all_states, "KL_W0_state", "On the second measure of size, the same question", AXIS_WORDS["KL_W0_state"], "size_share_kl.png")
    fig_share(all_states, SIZE_AXIS, "A finished maths fine-tune reads above random changes of its size; at step 8 it does not", LOG_AXIS, "log_size.png", plain=True)
    fig_pick()
    fig_scale()
