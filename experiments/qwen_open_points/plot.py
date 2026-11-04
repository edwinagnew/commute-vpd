"""E77's figures from the results JSONs alone; no model.

  figures/own_by_block.png  a change the masked forward computes itself, by block, beside E74's Gaussian and update
  figures/hook.png          the last-matrix control against what the plan expected, amplitude and cosine
  figures/routing.png       transmission of the whole update against the number of matrices masked, with the filed line
  figures/swap.png          row A's masks on row B's tokens: distance from each clean row, with a uniform output beside it
  figures/displacement.png  how far the update moves the forward under fine-tuned masks, as a share of the clean distance

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_open_points/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
E74 = HERE.parent / "qwen_transmission" / "results"
INK, MUTED, GRID, BAND = "#2b2a27", "#6f6e69", "#e4e3df", "#d9d8d3"
DW, GA, OWN, FIXED, STOCH = "#e34948", "#2a78d6", "#1a7f5a", "#8a5a00", "#6250d6"
PANELS = (("u1", "first fine-tune, step 10"), ("u3", "template fine-tune, step 100"))
BLOCKS = (0, 7, 14, 21, 27)
KS = (0, 12, 49, 98, 147, 196)
LIBRARY_X = 222
# Not saved by the run: reports/reviews/2026-11-03_review_e77-masks-carry.md, "The swap, recomputed", follow_A and follow_B of a uniform output.
UNIFORM = {"u1": (0.408, 0.408), "u3": (0.455, 0.454)}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def style(ax: Axes, grid: Literal["x", "y"] = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def finish(fig: Figure, name: str, bottom: float = 0.0) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)


def own_by_block() -> None:
    a1, score = load(RES / "amend1.json"), load(RES / "amend1_score.json")
    b = score["bands"]["own"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9), dpi=170, sharey=True)
    for ax, (u, title) in zip(axes, PANELS, strict=True):
        e = load(E74 / u / "blocks.json")["summary"]
        each = score["own_output"]["each"]
        own = {block: each[f"{u}_{block}"]["amp"] for block in (0, 7)}
        if u == "u1":
            own |= {block: a1[f"own_{block}"]["amp"] for block in (14, 21, 27)}
        ax.axhspan(b[0], b[1], color=BAND, zorder=0)
        ax.axhline(1, color=INK, lw=1.2)
        ax.text(0, 1.1, "as in the clean model = 1", fontsize=8, color=INK)
        ax.text(0, 0.56, "filed line 0.5 at blocks 0 and 7, with its no-call band", fontsize=8, color=INK)
        ax.plot(BLOCKS, [e[f"gauss_{x}"]["amp"] for x in BLOCKS], color=GA, marker="s", lw=2, ls="--", label="E74: Gaussian of equal clean KL")
        ax.plot(BLOCKS, [e[f"dw_{x}"]["amp"] for x in BLOCKS], color=DW, marker="^", lw=2, ls=":", label="E74: the update at that block")
        ax.plot(list(own), list(own.values()), color=OWN, marker="o", ms=8, lw=2, label="E77: change the forward computes itself")
        if u == "u1":
            top, low = a1["own_last"]["amp"], a1["hook"]["amp"]
            ax.plot([29, 29], [low, top], color=INK, lw=3, solid_capstyle="round", label="most this change can read at the last matrix, 0.59 to 0.68")
        else:
            ax.text(0, 0.11, "the forward's own change was read\nat blocks 0 and 7 only on this update", fontsize=8, color=MUTED)
        ax.set(title=title, xlabel="block the change is added at", yscale="log", ylim=(0.0008, 1.6), xticks=BLOCKS)
        ax.title.set_size(10)
        style(ax)
    axes[0].set_ylabel("logit-shift amplitude, masked over clean")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Does a change the masked forward computes itself reach the logits from early blocks?", fontsize=11)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    finish(fig, "own_by_block.png", 0.1)


def hook() -> None:
    a1 = load(RES / "amend1.json")
    c = load(E74 / "u1" / "control.json")["summary"]
    rows = (("Hook control: only the last matrix masked", a1["hook"], OWN, "o"),
            ("Same change, all 196 matrices masked", a1["own_last"], OWN, "D"),
            ("E74: Gaussian, all masked", c["gauss_down27"], GA, "s"),
            ("E74: the update, all masked", c["dw_down27"], DW, "^"))
    lo, hi = a1["bands"]["own"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=170, sharey=True, sharex=True)
    for ax, key, name in zip(axes, ("amp", "cos"), ("logit-shift amplitude, masked over clean", "cosine of masked and clean logit shifts"), strict=True):
        ax.axvspan(lo, hi, color=BAND, zorder=0)
        ax.axvline(1, color=INK, lw=1.2)
        ax.text(0.985, 3.42, "plan expected near 1", fontsize=8, color=INK, ha="right")
        ax.text(lo - 0.015, 3.42, "control's stop", fontsize=8, color=INK, ha="right")
        for y, (_, row, col, mark) in zip(range(3, -1, -1), rows, strict=True):
            ax.scatter(row[key], y, color=col, marker=mark, s=70, zorder=3, edgecolor="white")
            ax.text(row[key], y + 0.17, f"{row[key]:.2f}", fontsize=8, color=INK, ha="center")
        ax.set(xlabel=name, xlim=(0, 1.08), ylim=(-0.5, 3.7))
        style(ax, "x")
    axes[0].set_yticks(range(3, -1, -1), [r[0] for r in rows], fontsize=8.5)
    fig.suptitle("Does the last-matrix control read near 1, as the plan expected? No, first fine-tune", fontsize=11)
    finish(fig, "hook.png")


def logit_shift() -> None:
    """The whole first update's effect on the logits, as a share of its effect with no masks: one bar per way of masking. E74's reads, through E77's table."""
    score = load(RES / "amend1_score.json")["routing"]["each"]
    fixed, stoch = score["u1_fixed"]["points"], score["u1_stochastic"]["points"]
    rows = (("No masks", 1.0, "#8a8984"), ("Random masks, random subset\n(how VPD trains)", stoch["library"]["amp"], "#2a78d6"),
            ("Random masks,\nall 196 matrices", stoch["k196"]["amp"], "#2a78d6"), ("Fixed masks,\nall 196 matrices", fixed["k196"]["amp"], "#2a78d6"))
    fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=170)
    ax.barh(range(len(rows)), [r[1] for r in rows], color=[r[2] for r in rows], height=0.55)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Size of the update's effect on the logits (no masks = 1)", fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title("Under the decomposition's masks, a real update moves the logits\nonly a tenth to a quarter as far", fontsize=12, color=INK, loc="left", x=-0.27)
    fig.tight_layout()
    fig.savefig(FIG / "logit_shift.png", facecolor="white")
    plt.close(fig)


def routing() -> None:
    score = load(RES / "amend1_score.json")["routing"]["each"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9), dpi=170, sharey=True)
    for ax, (u, title) in zip(axes, PANELS, strict=True):
        ax.axhline(1, color=INK, lw=1.2)
        ax.text(60, 1.12, "clean model = 1", fontsize=8, color=INK)
        for mode, col, mark, name in (("fixed", FIXED, "o", "fixed masks, Δ off"), ("stochastic", STOCH, "s", "VPD's stochastic masks")):
            r = score[f"{u}_{mode}"]
            p = r["points"]
            ax.plot(KS, [p[f"k{k}"]["tau"] for k in KS], color=col, marker=mark, ms=7, lw=2, label=name)
            ax.scatter(LIBRARY_X, p["library"]["tau"], color="white", edgecolor=col, marker=mark, s=60, lw=1.6, zorder=3)
            ax.scatter(49, 10 * p["k196"]["tau"], color=col, marker="_", s=320, lw=2, zorder=4)
            above = mode == "stochastic"
            ax.annotate(f"{r['ratio']:.2f} times: {r['label']}", (49, p["k49"]["tau"]), xytext=(8, 8) if above else (-9, -15),
                        textcoords="offset points", fontsize=7.5, color=INK, ha="left" if above else "right")
        ax.set(title=title, xlabel="matrices masked per position, of 196", yscale="log", ylim=(0.0003, 1.8), xlim=(-24, 235),
               xticks=[*KS, LIBRARY_X], xticklabels=[*map(str, KS[:-1]), "196", "library"])
        ax.tick_params(axis="x", labelsize=8)
        ax.title.set_size(10)
        style(ax)
    axes[0].set_ylabel("transmission τ: KL moved masked over KL moved clean")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=MUTED, marker="_", ms=14, ls="", mew=2))
    labels.append("filed line at 49 masked: 10 times the all-masked read")
    handles.append(Line2D([], [], color=MUTED, marker="o", mfc="white", ms=7, ls=""))
    labels.append("library's draw: a number of matrices uniform on 1 to 196, per position")
    fig.suptitle("How much of the whole update reaches the logits as fewer matrices are masked?", fontsize=11)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    finish(fig, "routing.png", 0.1)


def swap() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), dpi=170, sharey=True, sharex=True)
    for ax, (u, title) in zip(axes, PANELS, strict=True):
        r = load(RES / u / "amend1_swap.json")
        s, sums = r["summary"], {k: sum(v) for k, v in r["sums"].items()}
        ax.axvspan(0.5 * (1 - 2 * s["w_A"]), 0.5 * (1 + 2 * s["w_A"]), ymin=0.55, ymax=0.95, color=BAND, zorder=0)
        ax.axvspan(0.5 * (1 - 2 * s["w_B"]), 0.5 * (1 + 2 * s["w_B"]), ymin=0.05, ymax=0.45, color=BAND, zorder=0)
        ax.scatter(s["follow_A"], 1, color=OWN, s=80, zorder=3, edgecolor="white", label="row A's masks on row B's tokens")
        ax.scatter(s["follow_B"], 0, color=OWN, s=80, zorder=3, edgecolor="white")
        ax.scatter(UNIFORM[u], (1, 0), color="white", edgecolor=INK, marker="D", s=50, lw=1.3, zorder=3, label="a uniform output, knows nothing")
        own = sums["K0"] / sums["K_AB"]
        ax.scatter(own, 1.22, color=MUTED, marker="v", s=45, zorder=3, clip_on=False, label="row A's masks on row A's tokens")
        ax.text(own + 0.02, 1.17, f"{own:.3f}", fontsize=8, color=INK)
        ax.text(0.5, 1.33, "filed line 0.5, with its no-call band", fontsize=8, color=INK, ha="center")
        ax.text(s["follow_A"] + 0.02, 0.93, f"{s['follow_A']:.3f}", fontsize=8, color=INK)
        ax.text(s["follow_B"] - 0.02, -0.26, f"{s['follow_B']:.2f}", fontsize=8, color=INK, ha="right")
        ax.set(title=f"{title}: filed label {s['label']}", xlabel="KL from that clean row, over KL between the two clean rows", xlim=(0, 0.62), ylim=(-0.45, 1.5))
        ax.title.set_size(10)
        ax.set_xlabel(ax.get_xlabel(), fontsize=9)
        style(ax, "x")
    axes[0].set_yticks([1, 0], ["from clean row A,\nwhich gave the masks", "from clean row B,\nwhich gave the tokens"], fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Where does the output sit when masks come from one row and tokens from another?", fontsize=11)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    finish(fig, "swap.png", 0.08)


def displacement() -> None:
    score = load(RES / "amend1_score.json")["displacement"]["each"]
    lo, hi = load(RES / "amend1_score.json")["bands"]["disp"]
    rows = (("u1", "first fine-tune, step 10"), ("u2", "gold fine-tune, step 100"), ("u3", "template fine-tune, step 100"))
    fig, ax = plt.subplots(figsize=(9.5, 3.1), dpi=170)
    ax.axvspan(lo, hi, color=BAND, zorder=0)
    ax.axvline(1, color=INK, lw=1.2)
    ax.text(0.92, 2.42, "clean model = 1", fontsize=8, color=INK, ha="right")
    ax.text(0.0178, 2.42, "filed line 0.02", fontsize=8, color=INK, ha="right")
    for y, (u, _) in zip((2, 1, 0), rows, strict=True):
        v = score[u]["share"]
        ax.scatter(v, y, color=DW, marker="^", s=70, zorder=3, edgecolor="white")
        ax.text(v, y + 0.17, f"{v:.4f}", fontsize=8, color=INK, ha="center")
    ax.set(xscale="log", xlim=(0.0002, 1.5), ylim=(-0.5, 2.7), xlabel="KL(update off ‖ update on) under fine-tuned masks, over KL(W0 ‖ fine-tuned), clean")
    ax.set_xlabel(ax.get_xlabel(), fontsize=9)
    ax.set_yticks([2, 1, 0], [r[1] for r in rows], fontsize=8.5)
    style(ax, "x")
    ax.set_title("How far does the update move the forward under masks made on the fine-tuned model?", fontsize=10.5)
    finish(fig, "displacement.png")


if __name__ == "__main__":
    own_by_block()
    hook()
    routing()
    logit_shift()
    swap()
    displacement()
