"""E75's figures from the results JSONs alone; no model. Prints the derived numbers the write-up quotes.

  figures/k_fraction.png  every state's KL from the updated model, as a fraction of W0's, per update
  figures/beats_size.png  in-span against its slice and the rescaled update, whole range and close up
  figures/against_sft.png the filed rows on each update, against the fine-tunes' filed range and the filed line
  figures/ruler.png       how far each update moves the rows, and whether next-token NLL falls on them
  figures/bad_placements.png  Amendment 1: three wrong placements per update, over the ruler, with the remainder's bar
  figures/r_vs_n.png      the filed rows on RL's update and its shuffled-reward null, against the same range and lines

The five updates are rescored from the reads, as score.json read at job 50596; the null's come from results/score_N.json.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 6G python experiments/qwen_rl_reads/plot.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from commute.jobs import load_experiment

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID, BAND = "#2b2a27", "#6f6e69", "#e4e3df", "#d9d8d3"
SFT, RL = "#2a78d6", "#eb6834"
INSPAN, SLICE, SCALED = "#1baf7a", "#6250d6", "#eda100"
NULL = "#8c2f8f"
# name, words, colour, marker: the same in every figure
UPDATES = (("u1", "first fine-tune, step 10", SFT, "o"), ("u2", "gold fine-tune, step 100", SFT, "s"),
           ("u3", "template fine-tune, step 100", SFT, "^"), ("R100", "RL, step 100", RL, "D"), ("R10", "RL, step 10", RL, "v"))
NULLS = (("N100", "shuffled-reward null, step 100", NULL, "P"), ("N10", "shuffled-reward null, step 10", NULL, "X"))
SHORT = {"u1": "first\nfine-tune", "u2": "gold\nfine-tune", "u3": "template\nfine-tune", "R100": "RL\nstep 100", "R10": "RL\nstep 10"}
BAD = (("random", "random matrix", "#a88600", "o"), ("other", "another update's matrix", "#c2356b", "s"),
       ("signs", "the update, random sign per matrix", "#0b8ea3", "D"))
STATES = (("inspan", "in-span part"), ("scaled_inspan", "whole update, rescaled to in-span's size"),
          ("slice_inspan", "random slice, in-span's size"), ("upper", "often-on half's part"), ("lower", "rarely-on half's part"),
          ("adjoint", "adjoint split's change"), ("scaled_remainder", "whole update, rescaled to remainder's size"),
          ("slice_remainder", "random slice, remainder's size"), ("remainder", "remainder"), ("random", "random change, matched on KL"))
# Filed row, words, the filed line (None: no rule)
ROWS = (("transmission_fixed", "whole update under W0's\nmasks, fixed: share of its KL", 0.05),
        ("transmission_stochastic", "whole update under W0's\nmasks, stochastic", 0.05),
        ("random_vs_halves", "random's KL over the\nlarger half's", 10.0),
        ("inspan_vs_remainder", "in-span's KL over the\nremainder's", 1.0),
        ("share_ls", "update's share sent\nto Δ, least squares", None))


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def style(ax: Axes, grid: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid in ("x", "y"):
        ax.grid(axis="x" if grid == "x" else "y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def finish(fig: Figure, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)


def fractions(score: dict[str, Any], u: str) -> dict[str, float]:
    states = score[u]["clean"]["states"]
    return {n: s["K"] / states["W0"]["K"] for n, s in states.items()}


def k_fraction(score: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=170)
    n = len(STATES)
    for i, (u, words, colour, marker) in enumerate(UPDATES):
        f = fractions(score, u)
        ys = [n - 1 - j + (i - 2) * 0.14 for j in range(n)]
        ax.scatter([f[s] for s, _ in STATES], ys, s=42, color=colour, marker=marker, label=words, zorder=3,
                   edgecolors="white", linewidths=0.6)
    ax.axvline(1, color=INK, lw=1.2)
    ax.text(1.05, n - 0.55, "W0, the ruler = 1", fontsize=8.5, color=INK)
    ax.text(0.0032, n - 0.55, "the updated model reads 0, off this axis", fontsize=8.5, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(0.003, 5)
    ax.set_ylim(-0.6, n - 0.2)
    ax.set_yticks(range(n), [w for _, w in reversed(STATES)], fontsize=9, color=INK)
    ax.set_xlabel("KL(updated model ‖ state) as a fraction of KL(updated model ‖ W0), log scale")
    ax.set_title("How far from the updated model does each state read, with W0 as the ruler?", fontsize=11, color=INK, loc="left")
    ax.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=1, edgecolor=GRID)
    style(ax, "x")
    finish(fig, "k_fraction.png")


def beats_size(score: dict[str, Any], rescaled: dict[str, tuple[float, float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=170, gridspec_kw={"width_ratios": [1, 1.7]})
    series = (("inspan", "in-span part", INSPAN, "o"), ("slice_inspan", "random slice, same size", SLICE, "s"),
              ("scaled_inspan", "whole update, rescaled to same size", SCALED, "D"))
    xs = range(len(UPDATES))
    for ax in axes:
        for j, (s, words, colour, marker) in enumerate(series):
            ys = [fractions(score, u)[s] for u, *_ in UPDATES]
            # The bar is the row bootstrap's half-width of this state's gap to in-span, as a fraction of the ruler.
            bars = [0.0 if s == "inspan" else (score[u]["size"]["width_inspan"] if s == "slice_inspan" else rescaled[u][1])
                    / score[u]["clean"]["states"]["W0"]["K"] for u, *_ in UPDATES]
            ax.errorbar([x + (j - 1) * 0.16 for x in xs], ys, yerr=bars, fmt=marker, color=colour, ms=6.5, capsize=2.5, label=words)
        ax.set_xticks(list(xs), [SHORT[u] for u, *_ in UPDATES], fontsize=8.5, color=INK)
        style(ax)
    axes[0].axhline(1, color=INK, lw=1.2)
    axes[0].text(4.3, 0.945, "W0, the ruler = 1", fontsize=8.5, color=INK, ha="right")
    axes[0].set_ylim(-0.03, 1.06)
    axes[0].set_ylabel("KL from the updated model, fraction of W0's")
    axes[0].set_title("The whole range", fontsize=10, color=INK, loc="left")
    axes[1].set_ylim(0, 0.033)
    axes[1].set_title("Close up: the same points, axis to 0.033", fontsize=10, color=INK, loc="left")
    axes[1].legend(loc="upper left", fontsize=8.5, frameon=False, ncols=1)
    fig.suptitle("Does the in-span part beat a state of the same size that is not placed?", fontsize=11, color=INK, x=0.01, ha="left")
    finish(fig, "beats_size.png")


def rows_figure(score: dict[str, Any], updates: tuple[tuple[str, str, str, str], ...], title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=170)
    n = len(ROWS)
    for j, (row, _, line) in enumerate(ROWS):
        y = n - 1 - j
        lo, hi = score["u1"]["rows"][row]["sft_range"]
        mid = math.sqrt(lo * hi)
        ax.barh(y, hi / mid - lo / mid, left=lo / mid, height=0.62, color=BAND, zorder=1)
        ax.text(0.0225, y + 0.33, f"range {lo:g} to {hi:g}", fontsize=8, color=MUTED)
        if line is not None:
            ax.plot([line / mid] * 2, [y - 0.31, y + 0.31], color=INK, lw=1.6, zorder=2)
            ax.text(line / mid * (0.93 if line == 10.0 else 1.07), y + 0.1, f"filed line {line:g}", fontsize=8, color=INK, ha="right" if line == 10.0 else "left")
        for i, (u, words, colour, marker) in enumerate(updates):
            value = score[u]["rows"][row]["value"]
            ax.scatter(value / mid, y + (i - (len(updates) - 1) / 2) * 0.11, s=42, color=colour, marker=marker, zorder=3, edgecolors="white",
                       linewidths=0.6, label=words if j == 0 else None)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 200)
    ax.set_ylim(-0.6, n - 0.3)
    ax.set_yticks(range(n), [w for _, w, _ in reversed(ROWS)], fontsize=9, color=INK)
    ax.set_xlabel("value over the middle of the fine-tunes' filed range (grey), log scale")
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=1, edgecolor=GRID)
    style(ax, "x")
    finish(fig, name)


def ruler(score: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=170)
    xs = list(range(len(UPDATES)))
    for x, (u, _, colour, marker) in zip(xs, UPDATES, strict=True):
        axes[0].scatter(x, score[u]["clean"]["states"]["W0"]["K"], s=70, color=colour, marker=marker, zorder=3)
        axes[1].scatter(x, score[u]["clean"]["gain_den_nats"], s=70, color=colour, marker=marker, zorder=3)
    for ax in axes:
        ax.set_xticks(xs, [SHORT[u] for u, *_ in UPDATES], fontsize=8.5, color=INK)
        ax.set_xlim(-0.5, 4.5)
        style(ax)
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.005, 4)
    axes[0].set_ylabel("KL(updated model ‖ W0), nats, log scale")
    axes[0].set_title("How far the update moves the model on its rows", fontsize=10, color=INK, loc="left")
    axes[1].axhspan(-0.2, 0.05, color=BAND, zorder=0)
    axes[1].axhline(0, color=INK, lw=1.2)
    axes[1].text(4.45, 0.012, "no change in NLL = 0", fontsize=8.5, color=INK, ha="right")
    axes[1].text(-0.4, 0.062, "under the filed 0.05 nats (grey): gain is not scored", fontsize=8.5, color=MUTED)
    axes[1].set_ylim(-0.2, 0.6)
    axes[1].set_ylabel("W0's NLL minus the updated model's, nats")
    axes[1].set_title("Whether next-token NLL falls on those rows", fontsize=10, color=INK, loc="left")
    fig.suptitle("Blue: fine-tunes. Orange: RL. The template fine-tune is on its own 64 rows, the others on 128 maths rows.",
                 fontsize=9.5, color=MUTED, x=0.01, ha="left")
    finish(fig, "ruler.png")


def bad_placements(table: dict[str, dict[str, dict[str, Any]]]) -> None:
    """Amendment 1. A state's bar is the half-width of its gap to the remainder, the rule's own width, over the ruler."""
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=170)
    for x, (u, *_) in enumerate(UPDATES):
        row = table[u]
        ruler_k = row["random"]["ruler"]
        ax.plot([x - 0.36, x + 0.36], [row["random"]["K_remainder"] / ruler_k] * 2, color=INK, lw=3, solid_capstyle="butt", zorder=2,
                label="remainder, the bar to clear" if x == 0 else None)
        for j, (s, words, colour, marker) in enumerate(BAD):
            ax.errorbar(x + (j - 1) * 0.2, row[s]["over_ruler"], yerr=row[s]["width"] / ruler_k, fmt=marker, color=colour, ms=7, capsize=2.5,
                        zorder=3, label=words if x == 0 else None)
    ax.axhline(1, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.text(4.45, 1.03, "W0, the ruler = 1", fontsize=8.5, color=MUTED, ha="right")
    ax.text(4.45, 0.03, "the updated model reads 0", fontsize=8.5, color=MUTED, ha="right")
    ax.set_xticks(range(len(UPDATES)), [SHORT[u] for u, *_ in UPDATES], fontsize=8.5, color=INK)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 3.2)
    ax.set_ylabel("KL from the updated model, fraction of W0's")
    ax.set_title("Do wrong changes, each matrix at the update's Frobenius norm, read above the remainder?", fontsize=11, color=INK, loc="left")
    ax.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=1, edgecolor=GRID)
    style(ax)
    finish(fig, "bad_placements.png")


def rescaled_gaps(rules: Any) -> dict[str, tuple[float, float]]:
    """K(rescaled) - K(in-span) and its paired row-bootstrap half-width: not filed, the rule's own function on another pair."""
    def inspan(u: str) -> Path:  # u1 to u3 keep E72's read; the RL updates have their own
        own = RES / u / "clean_inspan.json"
        return own if own.is_file() else HERE.parent / "qwen_fixed_masks" / "results" / u / "clean_inspan.json"

    return {u: rules.gap_width(load(inspan(u)), load(RES / u / "clean_scaled_inspan.json")) for u, *_ in UPDATES + NULLS}


def reads(name: str) -> dict[str, Any]:
    """An update's reads, as run.py's `stored` and its own results: E72's, E74's and E68's files, then this job's on top."""
    e = HERE.parent
    paths = {p.stem: p for p in (e / "qwen_fixed_masks" / "results" / name).glob("clean_*.json")}
    paths |= {"S": e / "qwen_fixed_masks" / "results" / name / "S.json", "search": e / "qwen_pass_test" / "results" / name / "search.json",
              "fixed": e / "qwen_transmission" / "results" / name / "all.json", "stochastic": e / "qwen_transmission" / "results" / name / "stochastic.json"}
    paths = {k: p for k, p in paths.items() if p.is_file()} | {p.stem: p for p in (RES / name).glob("*.json")}
    return {k: load(p) for k, p in paths.items()}


def rescore(rules: Any) -> dict[str, Any]:
    """score.json holds only the updates of the job that last wrote it, so the table is scored here from the reads with the filed rules."""
    res = {u: reads(u) for u, *_ in UPDATES}
    return {u: rules.score_update(res[u], {} if u == "u3" else res["u1"], u[0] == "R") for u in res}


def report(score: dict[str, Any], rescaled: dict[str, tuple[float, float]]) -> None:
    print("update | gap slice-inspan / ruler | gap rescaled-inspan, half-width, / ruler | (1-t)^2 in-span, K/ruler | (1-t)^2 rem, K/ruler")
    for u, *_ in UPDATES + NULLS:
        s, ruler_k = score[u]["size"], score[u]["clean"]["states"]["W0"]["K"]
        f = fractions(score, u)
        t_i, t_r = s["t"]["inspan"], s["t"]["remainder"]
        print(f"{u}: {s['gap_inspan'] / ruler_k:.5f} | {rescaled[u][0]:.6f} {rescaled[u][1]:.6f} {rescaled[u][0] / ruler_k:.5f} | "
              f"{(1 - t_i) ** 2:.4f} {f['scaled_inspan']:.4f} | {(1 - t_r) ** 2:.4f} {f['scaled_remainder']:.4f}")
        print("   fractions " + " ".join(f"{n}={f[n]:.4f}" for n, _ in STATES))
        print(f"   t_i^2+t_r^2 {t_i ** 2 + t_r ** 2:.4f}, sqrt(1-t_r^2) {math.sqrt(1 - t_r ** 2):.4f}")


def check(score: dict[str, Any], against: str) -> None:
    """Optional: the rescored table against a saved score.json, e.g. job 50596's from git."""
    old, gaps = load(Path(against)), []
    for u in score:
        gaps += [abs(score[u]["rows"][r]["value"] - old[u]["rows"][r]["value"]) for r, *_ in ROWS]
        gaps += [abs(s["K"] - old[u]["clean"]["states"][n]["K"]) for n, s in score[u]["clean"]["states"].items()]
    print(f"rescored against {against}: {len(gaps)} numbers, largest gap {max(gaps):.3g}")


def main() -> None:
    rules = load_experiment("qwen_rl_reads", "rules")
    score = rescore(rules)
    rescaled = rescaled_gaps(rules)
    k_fraction(score)
    beats_size(score, rescaled)
    rows_figure(score, UPDATES, "Does RL's update read as the fine-tunes do, row by row?", "against_sft.png")
    ruler(score)
    bad_placements(load(RES / "amend1_score.json")["table"])
    null = load(RES / "score_N.json")
    both = score | {u: null[u] for u, *_ in NULLS}
    rows_figure(both, UPDATES[3:] + NULLS, "Does RL's update read differently from its shuffled-reward null, row by row?", "r_vs_n.png")
    report(both, rescaled)
    if len(sys.argv) > 1:
        check(score, sys.argv[1])


if __name__ == "__main__":
    main()
