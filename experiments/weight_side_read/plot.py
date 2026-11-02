"""E67's figures from results/run.json (job 49342) and the bars in results/predictions.json; no model.

  figures/share.png         the overlap share per state, against the filed bars, the drop band and the filed guess
  figures/over_gaussian.png the shares on the 128 rows, the only rows the Gaussian was read on, on a log axis: noise ranges, the Gaussian, and the band where the filed flag fires
  figures/gamma_change.png  the mean change in Γ's values against the update's output change, zero control on the axes
  figures/per_matrix.png    the overlap share per matrix, against the keep bar (post hoc, no filed rule reads it)

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/weight_side_read/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
COPY, CODE, GAUSS = "#1f9e6e", "#7a5bb5", "#8a8984"
INK, GRID, REF = "#52514e", "#e4e3df", "#b0413e"
# Name in run.json, label, colour, rows.
STATES = (
    ("ident32_s0", "copy fine-tune\nseed 0", COPY, 128),
    ("ident32_s1", "copy fine-tune\nseed 1", COPY, 128),
    ("code_step10", "learned nothing\nstep 10", CODE, 128),
    ("code_step100", "learned nothing\nstep 100", CODE, 128),
    ("gaussian", "random update\nmatched size", GAUSS, 128),
    ("code_step10_held", "learned nothing\nstep 10", CODE, 64),
    ("ident32_s0_held", "copy fine-tune\nseed 0", COPY, 64),
)


def tidy(ax: Any, title: str, ylabel: str, xlabel: str = "") -> None:
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=INK)


def state_axis(ax: Any) -> list[float]:
    """Seven slots with a gap before the two states read on the rows where r failed, and a label for each group."""
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.6, 6.6]
    ax.set_xticks(xs, [s[1] for s in STATES], fontsize=7.5)
    ax.axvline(4.8, color=GRID, lw=1)
    for x, text in ((2.0, "128 Pile rows"), (6.1, "64 rows where r failed")):
        ax.annotate(text, (x, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom", fontsize=8, color=INK)
    return xs


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def share(r: dict[str, Any], th: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    xs = state_axis(ax)
    s0, s1 = r["state_ident32_s0"]["S_R"], r["state_ident32_s1"]["S_R"]
    ax.axhspan(0.2, 0.6, color="#f3efe3", zorder=0)
    ax.axhspan(0, min(s0, s1) + th["drop_band"], color="#e9e6f3", zorder=0)
    ax.axhline(th["keep_copy_min"], color=REF, lw=1.2)
    ax.axhline(th["keep_code_max"], color=REF, lw=1.2, ls="--")
    for x, (name, _, colour, _) in zip(xs, STATES, strict=True):
        v = r[f"state_{name}"]["S_R"]
        ax.bar(x, v, color=colour, width=0.6)
        ax.annotate(f"{v:.3f}", (x, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK)
    kw = {"xycoords": ("axes fraction", "data"), "textcoords": "offset points", "fontsize": 7.5}
    ax.annotate("filed guess for the fine-tunes: 0.2 to 0.6", (0.99, 0.585), xytext=(0, -3), ha="right", va="top", color=INK, **kw)
    ax.annotate("keep needs copy fine-tunes at least 0.4", (0.99, 0.4), xytext=(0, 3), ha="right", color=REF, **kw)
    ax.annotate("keep needs learned nothing at most 0.1", (0.99, 0.1), xytext=(0, 3), ha="right", color=REF, **kw)
    ax.annotate("drop band: within 0.15 of both copy seeds, up to 0.19", (0.99, 0.1926), xytext=(0, -9), ha="right", color="#5d4f8c", **kw)
    ax.set_ylim(0, 0.62)
    ax.set_xlim(-0.6, 7.2)
    tidy(ax, "", "overlap share of the switching term")
    ax.set_title("Does the overlap share tell the fine-tune that learned nothing from the copy fine-tunes?\n", fontsize=9.5, color=INK, loc="left")
    save(fig, "share.png")


def over_gaussian(r: dict[str, Any], th: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    states = [s for s in STATES if s[3] == 128]
    xs = list(range(len(states)))
    ax.set_xticks(xs, [s[1] for s in states], fontsize=7.5)
    ax.annotate("128 Pile rows", (2.0, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom", fontsize=8, color=INK)
    copy_mean = (r["state_ident32_s0"]["S_R"] + r["state_ident32_s1"]["S_R"]) / 2
    gauss = r["state_gaussian"]["S_R"]
    ax.set_yscale("log")
    ax.axhspan(5e-5, copy_mean + th["drop_band"], color="#f6e7e6", zorder=0)
    ax.axhline(gauss, color=GAUSS, lw=1.2, ls="--")
    for x, (name, _, colour, _) in zip(xs, states, strict=True):
        rec = r[f"state_{name}"]
        each = [a / b for a, b in zip(rec["chunks"]["r"], rec["chunks"]["den"], strict=True)]
        ax.plot([x, x], [min(each), max(each)], color=colour, lw=1.2)
        loo = rec["leave_one_chunk_out"]
        ax.plot([x, x], [loo["min"], loo["max"]], color=colour, lw=5, solid_capstyle="butt")
        ax.plot([x], [rec["S_R"]], "o", color=colour, ms=5, mec="white", mew=0.7)
    kw = {"xycoords": ("axes fraction", "data"), "textcoords": "offset points", "fontsize": 7.5}
    ax.annotate(f"random update: {gauss:.4f}", (0.01, gauss), xytext=(0, 3), ha="left", color=INK, **kw)
    ax.annotate("the filed flag fires anywhere in the shaded band, below 0.20", (0.99, copy_mean + th["drop_band"]), xytext=(0, -9), ha="right", color=REF, **kw)
    ax.annotate("dot: the share; thick bar: one chunk left out in turn;\nthin line: each chunk of 16 rows alone", (0.01, 8e-5), xytext=(0, 0), color=INK, **kw)
    ax.set_ylim(5e-5, 0.5)
    ax.set_xlim(-0.6, 4.6)
    tidy(ax, "", "overlap share, log axis")
    ax.set_title("Are the small shares noise, and does the filed flag say anything about the random update?\n", fontsize=9.5, color=INK, loc="left")
    save(fig, "over_gaussian.png")


def gamma_change(r: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ref = r["state_ident32_s0"]["den"] / 128
    offset = {"ident32_s0": (8, 2), "ident32_s1": (8, -9), "code_step10": (8, -3), "code_step100": (8, -3), "gaussian": (8, -3), "code_step10_held": (8, -3), "ident32_s0_held": (8, -3)}
    words = {"ident32_s0": "copy fine-tune, seed 0", "ident32_s1": "copy fine-tune, seed 1", "code_step10": "learned nothing, step 10", "code_step100": "learned nothing, step 100",
             "gaussian": "random update, matched size", "code_step10_held": "learned nothing, step 10, rows where r failed", "ident32_s0_held": "copy fine-tune, seed 0, rows where r failed"}
    for name, _, colour, rows in STATES:
        rec = r[f"state_{name}"]
        x, y = rec["den"] / rows / ref, rec["mean_abs_dg"]
        ax.plot([x], [y], "o", ms=7, color=colour if rows == 128 else "white", mec=colour, mew=1.5)
        ax.annotate(words[name], (x, y), xytext=offset[name], textcoords="offset points", fontsize=7.5, color=INK)
    ax.axhline(0, color=REF, lw=1.5)
    ax.annotate("zero control: an unchanged model reads exactly 0", (0.01, 0), xycoords=("axes fraction", "data"), xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=REF)
    ax.set_xscale("log")
    ax.set_xticks([0.4, 1, 2], ["0.4", "1", "2"])
    ax.set_xlim(0.3, 5.5)
    ax.set_ylim(-0.00002, 0.00095)
    tidy(ax, "How far do Γ's values move, for an update of a given size?", "mean of |g1 − g0| per subcomponent and position",
         "the update's output change per row, relative to the copy fine-tune seed 0 on the 128 rows")
    ax.grid(axis="x", color=GRID, lw=0.6)
    save(fig, "gamma_change.png")


def per_matrix(r: dict[str, Any], th: dict[str, float]) -> None:
    keys = sorted(r["state_ident32_s0"]["per_matrix"])
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ys = list(range(len(keys)))[::-1]
    series = (("ident32_s0", COPY, "o", "copy fine-tune, seed 0"), ("ident32_s1", COPY, "D", "copy fine-tune, seed 1"), ("code_step10", CODE, "o", "learned nothing, step 10"),
              ("code_step100", CODE, "D", "learned nothing, step 100"), ("gaussian", GAUSS, "s", "random update, matched size"))
    for name, colour, marker, label in series:
        pm = r[f"state_{name}"]["per_matrix"]
        ax.plot([pm[k]["S_R"] for k in keys], ys, marker, color=colour, ms=4.5, alpha=0.85, label=label, ls="none")
    ax.axvline(th["keep_copy_min"], color=REF, lw=1.2)
    ax.annotate("keep bar for the copy fine-tunes, pooled: 0.4", (th["keep_copy_min"], 1.0), xycoords=("data", "axes fraction"), xytext=(-3, -9), textcoords="offset points", ha="right", fontsize=7.5, color=REF)
    ax.axvline(0, color=INK, lw=0.8)
    den = r["state_ident32_s0"]["per_matrix"]
    total = r["state_ident32_s0"]["den"]
    ax.set_yticks(ys, [f"{k.replace('.attn', '').replace('.mlp', '')}  ({den[k]['den'] / total:.1%})" for k in keys], fontsize=7)
    ax.set_xlim(-0.13, 0.43)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.07), ncol=3, frameon=False)
    tidy(ax, "Does any single matrix read differently from the pooled share? (not filed)", "matrix (its share of the copy fine-tune's output change)", "overlap share, 128 Pile rows")
    ax.grid(axis="x", color=GRID, lw=0.6)
    save(fig, "per_matrix.png")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    r = json.loads((RES / "run.json").read_text())
    th = json.loads((RES / "predictions.json").read_text())["thresholds"]
    share(r, th)
    over_gaussian(r, th)
    gamma_change(r)
    per_matrix(r, th)


if __name__ == "__main__":
    main()
