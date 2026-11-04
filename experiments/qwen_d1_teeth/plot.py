"""E85's figures from results/results.json and the per-step lines of results/run.log; no model.

  figures/ratio_bars.png    the adversary's ratio for W0, do-nothing and the two bad states, against the filed bars and the bar Qwen's own spread gives
  figures/other_checks.png  behaviour kept and left in Δ for the same states: how bad each state is by the checks outside the adversary
  figures/steps.png         the adversary's KL per step, and the ratio per step beside E80's do-nothing and least squares

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_d1_teeth/plot.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#2b2a27", "#6f6e69", "#e4e3df"
COLOUR = {"W0": "#57564f", "do-nothing": "#7b4fc4", "in-span": "#e07b00", "random": "#2a78c4", "rarely-on": "#c8402f"}
R = json.loads((RES / "results.json").read_text())
S = R["score"]
E80 = json.loads((HERE.parent / "qwen_d1_adversary" / "results" / "amend2.json").read_text())["score"]
PASS_AT, FAIL_AT = 1.40, 1.61  # filed, results/predictions.json
SPREAD = S["W0_spread"]
LS_LEFT = 0.09001726620341453  # least squares' plain share, amend2.json, as rules.RARE_LEFT_LOW
OWN_PASS = 1.5 / (1 + SPREAD)  # the bars' formula, 1.5 / (1 + floor) with the 67M's 7.5% replaced by this job's W0 spread (reports/reviews/2026-11-04_review_e80-check.md)


def style(ax: Axes, grid: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.7)  # pyright: ignore[reportArgumentType]
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def finish(fig: Figure, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, facecolor="white", dpi=130)
    plt.close(fig)


def ratio_bars() -> None:
    rows = [("W0", "W0, unchanged\n(this host)", [1.0, 1.0], 1.0, "o"),
            ("do-nothing", "Do-nothing\n(E80, another host)", E80["r"]["N"], E80["mean"]["N"], "D"),
            ("random", "Random change in Δ\n(this host)", S["r"]["Rand"], S["mean"]["Rand"], "o"),
            ("rarely-on", "Rarely-on placement\n(this host)", S["r"]["Rare"], S["mean"]["Rare"], "o")]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.axhspan(1.0, 1 + SPREAD, color="#ecebe6", zorder=0)
    ax.text(-0.46, 1 + SPREAD - 0.006, f"grey: W0's own two starts differ by {SPREAD:.1%}", ha="left", va="top", color=MUTED, fontsize=9)
    for y, text, ls, colour in ((FAIL_AT, f"fail bar {FAIL_AT:.2f} (filed, from the 67M)", "-", "#a32a1e"),
                                (PASS_AT, f"pass bar {PASS_AT:.2f} (filed, from the 67M)", "-", "#1c7d55"),
                                (OWN_PASS, f"pass bar near {OWN_PASS:.2f} from Qwen's own spread (not filed)", "--", "#1c7d55")):
        ax.axhline(y, lw=1.3, ls=ls, color=colour)
        ax.text(-0.46, y + 0.006, text, color=colour, fontsize=9, va="bottom")
    for x, (arm, label, seeds, mean, mark) in enumerate(rows):
        c = COLOUR[arm]
        ax.scatter([x - 0.07, x + 0.07], seeds, color=c, s=46, marker=mark, facecolor="white" if arm == "do-nothing" else c, zorder=3, label="_")
        ax.hlines(mean, x - 0.2, x + 0.2, color=c, lw=2.6, zorder=3)
        ax.text(x + 0.24, mean, f"mean {mean:.3f}" if arm != "W0" else "1 by definition", color=c, va="center", fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5}, zorder=4)
    ax.set_xticks(range(len(rows)), [r[1] for r in rows])
    ax.set_xlim(-0.5, 3.75)
    ax.set_ylim(0.9, 1.75)
    ax.set_ylabel("Adversary's KL over W0's at the same seed")
    ax.set_title("Does the adversary fail either bad state? Neither reaches the fail bar", color=INK, loc="left")
    style(ax)
    finish(fig, "ratio_bars.png")


def other_checks() -> None:
    rows = [("W0", "W0, unchanged", 0.0, 0.0), ("do-nothing", "Do-nothing, E80", 1.0, 1.0), ("in-span", "Least squares, E80", 1.0, LS_LEFT),
            ("random", "Random change", S["kept"]["Rand"], S["left_in_delta"]["Rand"]), ("rarely-on", "Rarely-on placement", S["kept"]["Rare"], S["left_in_delta"]["Rare"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True, sharex=True)
    for ax, i, title, note in ((axes[0], 2, "Behaviour kept (0 = W0, 1 = fine-tune)", {"rarely-on": " by construction"}),
                               (axes[1], 3, "Left in Δ, plain share of squared size", {"random": " by construction, the change's own size", "do-nothing": " by construction"})):
        for y, row in enumerate(reversed(rows)):
            ax.barh(y, row[i], color=COLOUR[row[0]], height=0.6)
            ax.text(max(row[i], 0) + 0.02, y, f"{row[i]:.3f}" + note.get(row[0], ""), va="center", color=INK, fontsize=9)
        ax.set_yticks(range(len(rows)), [r[1] for r in reversed(rows)])
        ax.set_xlim(0, 1.9)
        ax.set_xticks([0, 0.5, 1])
        ax.set_title(title, color=INK, loc="left", fontsize=11)
        style(ax, "x")
    fig.suptitle("How bad are the two states by other checks? Random keeps almost no gain; rarely-on leaves 2.6 times least squares' share in Δ", color=INK, x=0.02, ha="left", fontsize=11)
    finish(fig, "other_checks.png")


def read_steps(paths: list[Path]) -> dict[tuple[str, int], list[float]]:
    """The KL logged before each of the 20 sign steps, by state and seed."""
    found: dict[tuple[str, int], dict[int, float]] = {}
    for path in paths:
        for m in re.finditer(r"\[(\w+)_W_s(\d)\] seed \d step (\d+): KL ([\d.]+)", path.read_text()):
            found.setdefault((m[1], int(m[2])), {})[int(m[3])] = float(m[4])
    assert all(len(v) == 20 for v in found.values()), {k: len(v) for k, v in found.items()}
    return {k: [v[i] for i in range(20)] for k, v in found.items()}


def mean_ratio(lines: dict[tuple[str, int], list[float]], arm: str) -> list[float]:
    return [(lines[(arm, 0)][i] / lines[("W0", 0)][i] + lines[(arm, 1)][i] / lines[("W0", 1)][i]) / 2 for i in range(20)]


def steps() -> None:
    here = read_steps([RES / "run.log"])
    there = read_steps(sorted((HERE.parent / "qwen_d1_adversary" / "results").glob("run_amend*.log")))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    left, right = axes
    for k, label, arm in (("W0", "W0, unchanged", "W0"), ("Rand", "Random change in Δ", "random"), ("Rare", "Rarely-on placement", "rarely-on")):
        for s, ls in ((0, "-"), (1, ":")):
            left.plot(range(20), here[(k, s)], color=COLOUR[arm], lw=1.7, ls=ls, label=f"{label}, seed {s}")
    left.set_ylabel("KL(clean ‖ masked), nats per token")
    left.set_title("The KL itself: all still climbing", color=INK, loc="left", fontsize=11)
    left.legend(frameon=False, fontsize=8, loc="upper left")
    for curve, final, label, arm in ((mean_ratio(there, "N"), E80["mean"]["N"], "Do-nothing, E80's host", "do-nothing"),
                                     (mean_ratio(there, "LS"), E80["mean"]["LS"], "Least squares, E80's host", "in-span"),
                                     (mean_ratio(here, "Rand"), S["mean"]["Rand"], "Random change in Δ", "random"),
                                     (mean_ratio(here, "Rare"), S["mean"]["Rare"], "Rarely-on placement", "rarely-on")):
        right.plot(range(20), curve, color=COLOUR[arm], lw=1.8, label=label)
        right.scatter([20], [final], color=COLOUR[arm], s=34, zorder=3)
    right.axhline(1.0, color=COLOUR["W0"], lw=1.0, ls="--")
    right.text(0, 0.985, "W0 = 1", color=COLOUR["W0"], fontsize=9, va="top")
    right.axhline(FAIL_AT, color="#a32a1e", lw=1.2)
    right.axhline(PASS_AT, color="#1c7d55", lw=1.2)
    right.text(0, FAIL_AT + 0.006, "fail bar 1.61", color="#a32a1e", fontsize=9, va="bottom")
    right.text(0, PASS_AT + 0.006, "pass bar 1.40", color="#1c7d55", fontsize=9, va="bottom")
    right.text(20.2, 1.76, "dots: the filed\nfinal read", color=MUTED, fontsize=8, ha="right", va="top")
    right.set_ylim(0.55, 1.8)
    right.set_ylabel("Mean over two seeds, over W0 at that step")
    right.set_title("The ratio: rises, then falls back before the filed read", color=INK, loc="left", fontsize=11)
    right.legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes:
        ax.set_xlabel("Adversary step")
        ax.set_xticks([0, 5, 10, 15, 20])
        style(ax)
    fig.suptitle("Had the search finished? No, and the ratio depends on the step it is read at", color=INK, x=0.02, ha="left")
    finish(fig, "steps.png")


if __name__ == "__main__":
    ratio_bars()
    other_checks()
    steps()
