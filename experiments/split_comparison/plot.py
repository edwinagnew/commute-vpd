"""Figures for split_comparison from results/split_comparison.json (keys `rows`, `W0` only).

  bars.png     the five operations side by side: one group per operation, six bars per group (fine-tune or its
               component-training analogue at steps 10/100/1000, then LoRA or its analogue). Three panels:
               adversarial ratio with the W0 noise band, held-out P(target) with W0's and the update's level,
               and Delta's share of ||dW||^2.
  pgd.png      how faithful is each operation's state under the adversary, along training? Ratio per operation
               against the step, fine-tune panel and LoRA panel, with the floor band.
  outside.png  how much of each weight change lies outside both sides of the subcomponents? Outside-both share of
               each update and of component training's weight change, against a random change of the same size.
  routes.png   where does the gain on the target come from? Direct, re-switching and remaining shares as stacked
               bars per operation and step, for the states whose gain is large enough to split.

Usage (repo root):  uv run python experiments/split_comparison/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OPS = {"nothing": "do-nothing", "least_squares": "least-squares split", "adjoint": "adjoint split",
       "rescaling": "rescaling split", "component_training": "component training"}
COLORS = {"nothing": "#2a78d6", "least_squares": "#eb6834", "adjoint": "#1baf7a", "rescaling": "#eda100",
          "component_training": "#e87ba4"}
PAIRS = {"ft": "full fine-tune", "lora24": "rank-8 LoRA"}
CT_NAME = {"ft": "free component training", "lora24": "rank-8 component training"}
SHADES = {"ft": ("#b3cde8", "#5b8fc9", "#1f4e8c"), "lora24": ("#f5c9a3", "#e08a4a", "#a34a10")}
STEPS = (10, 100, 1000)
GREY = "#7f7f7f"
RATIO_TICKS = (0.8, 1, 1.5, 2, 3, 5, 7)
RATIO_CLIP = 3.0
Row = dict[str, Any]


def load() -> tuple[list[Row], float]:
    out = json.loads((HERE / "results" / "split_comparison.json").read_text())
    return out["rows"], out["W0"]["pgd_floor"]


def series(rows: list[Row], pair: str, op: str, key: str) -> list[float]:
    by_step = {r["step"]: r[key] for r in rows if r["pair"] == pair and r["op"] == op}
    return [by_step[s] for s in STEPS]


def floor_band(ax: Axes, f: float, label: str = "") -> None:
    ax.axhspan(1 - 3 * f, 1 + 3 * f, color=GREY, alpha=0.2, lw=0,
               label=label or f"W0 noise band, 1 ± 3f (f = {f:.3f})")


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def grouped(ax: Axes, rows: list[Row], key: str, clip: float | None = None) -> None:
    """Two bars per operation, fine-tune (or analogue) then LoRA (or analogue); the three steps are overlaid in one
    bar, tallest at the back, so every step's top shows in its shade."""
    w = 0.36
    for i, op in enumerate(OPS):
        for j, pair in enumerate(PAIRS):
            x = i + (j - 0.5) * (w + 0.03)
            vals = series(rows, pair, op, key)
            for k in sorted(range(len(vals)), key=lambda k: -vals[k]):
                h = min(vals[k], clip) if clip else vals[k]
                ax.bar(x, h, w, color=SHADES[pair][k], edgecolor="white", lw=0.8, zorder=3)
            if clip and max(vals) > clip:
                ax.text(x + w / 2, clip * 0.96, f" {max(vals):.2f}, cut off", ha="left", va="top", fontsize=8, zorder=4)
    labels = [OPS[op] if op != "component_training" else "component training\n(unconstrained | rank 8)"
              for op in OPS]
    ax.set_xticks(range(len(OPS)), labels, fontsize=9)
    ax.set_xlim(-0.55, len(OPS) - 0.45)
    ax.grid(axis="y", color="#e6e6e6", zorder=0)


def bars(rows: list[Row], f: float) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    a, b, c = axes
    floor_band(a, f, "as faithful as the unchanged decomposition (its run-to-run spread)")
    grouped(a, rows, "pgd", clip=RATIO_CLIP)
    a.set_ylim(0, RATIO_CLIP)
    a.set_ylabel("adversarial ratio\n(1 = as faithful as the original)")
    a.set_title("(a) Adversarial ratio: lower is better; inside the grey band is as faithful as the original",
                fontsize=10, loc="left")
    a.legend(fontsize=8, loc="upper right")
    grouped(b, rows, "p_target")
    b.axhline(rows[0]["p_update"], color="black", ls="--", lw=1, label="updated model ≈ 1")
    b.axhline(rows[0]["p_baseline"], color="#444444", ls=":", lw=2,
              label=f"original model {rows[0]['p_baseline']:.3f}")
    b.set_ylim(0, 1.12)
    b.set_ylabel("held-out P(target)")
    b.set_title("(b) Behaviour kept: higher is better", fontsize=10, loc="left")
    b.legend(fontsize=8, loc="upper right", ncol=2)
    grouped(c, rows, "delta_share")
    c.set_ylim(0, 1.05)
    c.set_ylabel("share of the update\nleft in Δ")
    for i, op in enumerate(OPS):
        if max(series(rows, "ft", op, "delta_share") + series(rows, "lora24", op, "delta_share")) < 0.01:
            c.text(i, 0.04, "≈ 0", ha="center", fontsize=8, color=GREY)
    c.set_title("(c) Share of the update absorbed by Δ: lower is better", fontsize=10, loc="left")
    handles = [Patch(color=SHADES[p][k], label=f"{name}, step {s}")
               for p, name in (("ft", "fine-tune, or component training beside it"), ("lora24", "LoRA, or rank-8 component training beside it"))
               for k, s in enumerate(STEPS)]
    fig.legend(handles=handles, ncol=2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=False,
               title="the three steps are drawn over one another in one bar, darker = later step (one shade: the steps agree)",
               title_fontsize=8)
    fig.suptitle("The five operations side by side", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "bars.png", dpi=150)
    plt.close(fig)


def pgd(rows: list[Row], f: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (pair, label) in zip(axes, PAIRS.items()):
        floor_band(ax, f)
        ax.axhline(1.5, color=GREY, ls="--", lw=1, label="1.5")
        for op, op_label in OPS.items():
            op_label = CT_NAME[pair] if op == "component_training" else op_label
            ax.plot(STEPS, series(rows, pair, op, "pgd"), marker="o", lw=2, color=COLORS[op], label=op_label)
        ax.set_xscale("log")
        ax.set_xticks(STEPS, [str(s) for s in STEPS])
        ax.set_yscale("log")
        ax.set_yticks(RATIO_TICKS, [str(t) for t in RATIO_TICKS])
        ax.minorticks_off()
        ax.set_xlabel("training step")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("adversarial ratio (1 = as faithful as the original)")
    fig.suptitle("How faithful does each operation's state stay under the adversary?", fontsize=11)
    save(fig, "pgd.png")


def outside(rows: list[Row]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (pair, label) in zip(axes, PAIRS.items()):
        for op, name in (("nothing", f"the {label}'s δW"), ("component_training", CT_NAME[pair] + "'s change")):
            ax.plot(STEPS, series(rows, pair, op, "blocks_out_out"), marker="o", lw=2, color=COLORS[op], label=name)
            ax.plot(STEPS, series(rows, pair, op, "blocks_null_out_out"), ls="--", lw=1, color=COLORS[op],
                    label="random change of the same size")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(STEPS, [str(s) for s in STEPS])
        ax.set_xlabel("training step")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, loc="center left")
    axes[0].set_ylabel("outside-both share (lower is better)")
    fig.suptitle("How much of each weight change lies outside both sides of the subcomponents?", fontsize=11)
    save(fig, "outside.png")


ROUTE_OPS = ("nothing", "least_squares", "adjoint", "component_training")
ROUTE_PARTS = (("direct: the moved subcomponents themselves", "#5e3c99"),
               ("re-switching: Γ switching subcomponents on differently", "#b2abd2"),
               ("everything else: Δ, switched-off subcomponents, interactions", "#d9d9d9"))


def routes(rows: list[Row]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    ys = list(range(len(ROUTE_OPS)))
    names = [OPS[op] for op in ROUTE_OPS]
    for ax, (pair, label) in zip(axes, PAIRS.items()):
        d = np.array([series(rows, pair, op, "d")[-1] for op in ROUTE_OPS])
        r = np.array([series(rows, pair, op, "r")[-1] for op in ROUTE_OPS])
        left = np.zeros(len(ys))
        for part, (name, color) in zip((d, r, 1 - d - r), ROUTE_PARTS):
            ax.barh(ys, part, 0.8, left=left, color=color, label=name, zorder=3)
            left += part
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of the gain on the target")
        ax.set_title(f"{label}\n(component training here: {CT_NAME[pair]})", fontsize=10)
        ax.grid(axis="x", color="#e6e6e6", zorder=0)
    axes[0].set_yticks(ys, names, fontsize=8)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Where does the gain on the target come from? (step 1000; other steps in the table)", fontsize=11)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(FIG / "routes.png", dpi=150)
    plt.close(fig)


def main() -> None:
    rows, f = load()
    FIG.mkdir(exist_ok=True)
    bars(rows, f)
    pgd(rows, f)
    outside(rows)
    routes(rows)


if __name__ == "__main__":
    main()
