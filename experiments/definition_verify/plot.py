"""E99's figures, one per claim, from results/stage0.json, stage1.json, stage2.json and stage3_amend1.json; no model.

  figures/clause_ii.png    the lowercase pair's move of the original model's lowercase-minus-capitalised gap, against
                           20 random head pairs, on cased rows (scored, Amendment 1) and lowercased rows (beside)
  figures/on_off.png       mean log-odds of the target with the located pair on and off, for the original model and each
                           fine-tune, copy pair and lowercase pair: the net share is the difference of the two drops
  figures/net_share.png    every state's net share at its pair, with its own random-pair band
  figures/gain_curves.png  the constructed arms' gain in the training condition against the native's, over steps

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/definition_verify/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
S0 = json.loads((RES / "stage0.json").read_text())
ARMS = {**json.loads((RES / "stage1.json").read_text()), **json.loads((RES / "stage2.json").read_text()),
        **{k: v for k, v in json.loads((RES / "stage3_amend1.json").read_text()).items() if k != "clause_ii_amend1"}}  # fmt: skip
NATIVE, FORCED, FORBIDDEN, CONTROL = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # categorical slots 1-4, fixed order
INK, MUTED, GRID, BAND, SURFACE, W0 = "#2b2a28", "#52514e", "#e4e3df", "#d9d8d3", "#fcfcfb", "#8f8e88"
SEEDS = ("s0", "s1")


def axes(ax: Any, xlabel: str, ylabel: str = "", grid: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)


def save(fig: Any, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)


def clause_ii() -> None:
    c = S0["clause_ii"]
    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=SURFACE)
    axes(ax, "fall in the original model's lowercase-minus-capitalised log-odds when the heads are set to their mean")
    rows = (("cased rows\n(never trained on; scored)", c["cased"]), ("lowercased rows\n(the task's own; beside)", c["e53_lowercase"]))
    for i, (_label, r) in enumerate(rows):
        y = 1 - i
        ax.barh(y, 2 * r["band"], left=-r["band"], height=0.5, color=BAND, zorder=1)
        ax.scatter(list(r["random_moves_gap"].values()), [y] * 20, s=40, marker="|", color=W0, linewidth=1.2, zorder=2)
        ax.scatter([r["move_gap"]], [y], s=90, color=NATIVE, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.annotate(f"pair {r['move_gap']:.2f}", (r["move_gap"], y), xytext=(0, 12), textcoords="offset points", ha="center", color=INK, fontsize=9)
        ax.annotate(f"band ±{r['band']:.2f}", (r["band"], y - 0.25), xytext=(4, -2), textcoords="offset points", color=MUTED, fontsize=8.5)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks([1, 0], [r[0] for r in rows], color=INK)
    ax.set_ylim(-0.6, 1.7)
    ax.set_title("Does removing the lowercase pair move the original model's case leaning more than random pairs do?\nOn cased rows yes, beyond all 20; on lowercased rows no", loc="left", color=INK, fontsize=11)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=NATIVE, label="heads (1,1)+(1,2) at their mean"), Line2D([], [], marker="|", linestyle="", color=W0, markersize=10, label="each of 20 random head pairs"),
                        Patch(color=BAND, label="band: 95th percentile of |random move|")], loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)  # fmt: skip
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "clause_ii.png")


def on_off_panel(ax: Any, native: str, forced: str, forbidden: str, w0_key: str, title: str) -> None:
    """Mean log-odds of the target against the original model with the pair on (0), pair on then pair off."""
    w0_fall = S0["w0"][w0_key]["fall"]["lo"]
    ax.plot([0, 1], [0, -w0_fall], color=W0, linewidth=2.2, marker="o", label="original model")
    for name, col, recs in (("native fine-tune", NATIVE, [S0["natives"][f"{native} {s}"] for s in SEEDS]),
                            ("forced through the pair", FORCED, [ARMS[forced][s] for s in SEEDS]),
                            ("forbidden from the pair", FORBIDDEN, [ARMS[forbidden][s] for s in SEEDS])):  # fmt: skip
        for i, r in enumerate(recs):
            G, fall = r["G"]["lo"], r["fall"]["lo"]
            ax.plot([0, 1], [G, G - fall], color=col, linewidth=2, marker="o", linestyle="-" if i == 0 else ":", label=name if i == 0 else None)
        ax.annotate(f"net {recs[0]['net']['lo']:.2f}", (1, recs[0]["G"]["lo"] - recs[0]["fall"]["lo"]), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_xticks([0, 1], ["pair on", "pair off\n(at the original model's mean)"], color=INK)
    ax.set_xlim(-0.2, 1.45)
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.set_title(title, loc="left", color=INK, fontsize=10.5)


def on_off() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)
    axes(a, "", "mean log-odds of the target, minus the original model's", grid="y")
    axes(b, "", "", grid="y")
    on_off_panel(a, "T5 copy", "forced copy", "forbidden copy", "e62_copy", "Copy windows, pair (2,3)+(2,4)")
    on_off_panel(b, "T7 lowercase", "forced lowercase", "forbidden lowercase", "e53_lowercase", "Lowercase rows, pair (1,1)+(1,2)")
    fig.suptitle("How much of each fine-tune's gain disappears with the pair off, beyond what the original model loses?\n"
                 "Net share = (state's drop - original model's drop) / state's gain. Solid seed 0, dotted seed 1; separate y scales", x=0.01, ha="left", color=INK, fontsize=11)  # fmt: skip
    h, lab = a.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.88))
    save(fig, "on_off.png")


def net_rows(ax: Any, rows: list[tuple[str, dict[str, Any], str]], wide: dict[str, float]) -> None:
    for i, (label, r, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, 2 * r["band"], left=-r["band"], height=0.55, color=BAND, zorder=1)
        ax.scatter([r["net"]["lo"]], [y], s=80, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        if label in wide:
            ax.scatter([wide[label]], [y], s=70, marker="D", facecolor=SURFACE, edgecolor=col, linewidth=1.5, zorder=4)
    for x in (0, 1):
        ax.axvline(x, color=MUTED, linewidth=0.8, linestyle="-" if x == 0 else "--")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)


def net_share() -> None:
    n, A = S0["natives"], ARMS
    copy = [(f"native copy {s}", n[f"T5 copy {s}"], NATIVE) for s in SEEDS] + [(f"native lookup {s} (control)", n[f"T5 lookup {s}"], CONTROL) for s in SEEDS] + \
           [(f"forced copy {s}", A["forced copy"][s], FORCED) for s in SEEDS] + [(f"forbidden copy {s}", A["forbidden copy"][s], FORBIDDEN) for s in SEEDS]  # fmt: skip
    lower = [(f"native lowercase {s}", n[f"T7 lowercase {s}"], NATIVE) for s in SEEDS] + [(f"native cipher {s} (control)", n[f"T8 cipher {s}"], CONTROL) for s in SEEDS] + \
            [(f"forced lowercase {s}", A["forced lowercase"][s], FORCED) for s in SEEDS] + [(f"forbidden lowercase {s}", A["forbidden lowercase"][s], FORBIDDEN) for s in SEEDS]  # fmt: skip
    wide = {f"forbidden copy {s}": A["forbidden copy"][s]["wide_set_off"]["net"]["lo"] for s in SEEDS}
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 8.4), sharex=True, facecolor=SURFACE)
    axes(a, "")
    axes(b, "net share of the fine-tune's gain at the pair (1: all of it runs through the pair; 0: none)")
    net_rows(a, copy, wide)
    net_rows(b, lower, {})
    a.set_title("Copy windows, pair (2,3)+(2,4)", loc="left", color=INK, fontsize=10.5)
    b.set_title("Lowercase rows, pair (1,1)+(1,2)", loc="left", color=INK, fontsize=10.5)
    b.set_xlim(-1.15, 1.2)
    fig.suptitle("Does the net share read 1 when the route is forced through the pair and about 0 when it is forbidden?\nOn copy yes; on lowercase the forbidden fine-tune reads far below its band", x=0.01, ha="left", color=INK, fontsize=11)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=c, label=lab) for c, lab in ((NATIVE, "native fine-tune"), (CONTROL, "control fine-tune, other task"), (FORCED, "forced through the pair"), (FORBIDDEN, "forbidden from the pair"))] +
               [Line2D([], [], marker="D", linestyle="", markerfacecolor=SURFACE, markeredgecolor=FORBIDDEN, label="forbidden copy, three copying heads off"), Patch(color=BAND, label="state's own band: 95th percentile of |net| over 20 random pairs")],
               loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)  # fmt: skip
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save(fig, "net_share.png")


def gain_curves() -> None:
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    axes(ax, "training steps, as a multiple of the native fine-tune's steps (copy 30, lowercase 300)", "gain on the check inputs / native's gain there", grid="both")
    for arm, col, native_steps, style in (("forced copy", FORCED, 30, "-"), ("forced lowercase", FORCED, 300, "--"),
                                          ("forbidden copy", FORBIDDEN, 30, "-"), ("forbidden lowercase", FORBIDDEN, 300, "--")):  # fmt: skip
        for i, s in enumerate(SEEDS):
            r = ARMS[arm][s]
            pts = sorted((int(k), v) for k, v in r["curve"].items())
            xs, ys = [k / native_steps for k, _ in pts], [v / r["target_first_inputs"] for _, v in pts]
            ax.plot(xs, ys, color=col, linestyle=style, linewidth=1.8 if i == 0 else 1.0, marker="o" if len(xs) == 1 else None, label=f"{arm}" if i == 0 else None)
    ax.axhline(1, color=NATIVE, linewidth=1.2)
    ax.axhspan(0.98, 1.02, color=BAND, zorder=0)
    ax.axvline(1, color=MUTED, linewidth=0.8, linestyle=":")
    ax.axvline(10, color=MUTED, linewidth=0.8, linestyle=":")
    ax.annotate("native's gain (match band ±2%)", (0.012, 1.02), color=INK, fontsize=9, va="bottom")
    ax.annotate("native's steps", (1.05, 1.08), color=MUTED, fontsize=8.5)
    ax.annotate("step cap", (7.2, 1.08), color=MUTED, fontsize=8.5)
    ax.annotate("forbidden lowercase: past the target\nat the first check, step 5", (5 / 300, 1.07), xytext=(14, -40), textcoords="offset points", color=INK, fontsize=8.5,
                arrowprops={"arrowstyle": "-", "color": MUTED})  # fmt: skip
    ax.set_xscale("log")
    ax.set_ylim(0, 1.15)
    ax.set_title("Can a fine-tune reach the native gain when forced through the pair, or forbidden from it?\nForced: no, it stops at 0.82 (copy) and about 0.7 (lowercase) by the cap. Forbidden: yes, fast",
                 loc="left", color=INK, fontsize=11)  # fmt: skip
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK, title="solid seed 0 (thin seed 1); training condition", title_fontsize=8.5)
    fig.tight_layout()
    save(fig, "gain_curves.png")


if __name__ == "__main__":
    clause_ii()
    on_off()
    net_share()
    gain_curves()
