"""E126's one figure, two panels.

Left: held-out P(target) of the emoticon Adam Pile-text copy at batch orders 0, 1, 2 on the L40, beside E34's H100
copies at orders 0 and 1 (E34 and its Amendment 4) and P1's two filed bands.
Right: (masked-off share, re-switching share r) of the three Pile-text copies E119 read re-switching, at orders 0
(E119's run), 1 and 2 (this run), with E119's re-switching lines.

    python experiments/copy_reproducibility/plot.py   -> figures/orders.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FIGURES = HERE / "figures"
OUT = json.loads((HERE / "results" / "copy_reproducibility.json").read_text())
CFG = json.loads((HERE / "results" / "predictions.json").read_text())["config"]
E34 = EXP / "teacher_training" / "results"
E119 = EXP / "teacher_route_split" / "results"
COLOUR = {"r1_t534_v369": "#2a78d6", "r0_t352_v310": "#eb6834", "emoticon": "#1baf7a"}  # E119's colours
MARKER = {"adam": "s", "sgd": "^"}
INK, MUTED, REGION = "#2b2b2b", "#8a8a85", "#efeee9"
R_HI, MASKED_LO = 0.6, 0.25  # commute.route_labels.Lines, E119's re-switching corner


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def emoticon_panel(ax: Any) -> None:
    orders = OUT["cells"]["emoticon/adam"]["orders"]
    xs = sorted(int(k) for k in orders)
    ps = [orders[str(k)]["p_target"] for k in xs]
    h0 = load(E34 / "teacher_training_emoticon.json")["teachers"]["adam"]["B"]["1000"]["p_target"]
    h1 = load(E34 / "teacher_training_amend4.json")["tasks"]["emoticon"]["teachers"]["adam"]["B"]["1000"]["p_target"]
    ax.axhspan(min(h0, h1), max(h0, h1), color=REGION, zorder=0)
    ax.text(2.25, max(h0, h1) + 0.004, f"H100 range over two orders ({abs(h1 - h0):.3f})", fontsize=7, color=MUTED, ha="right")
    ax.plot([0, 1], [h0, h1], "D", ms=8, mfc="white", mec=INK, mew=1.5, zorder=4, label="H100 copy (E34, orders 0 and 1)")
    ax.plot(xs, ps, "s-", ms=9, color=COLOUR["emoticon"], mec="white", mew=1.5, zorder=5, label="L40 copy (E119 order 0; E126)")
    for x, p in zip(xs, ps):
        ax.annotate(f"{p:.3f}", (x, p), xytext=(8, -3), textcoords="offset points", fontsize=8, color=INK)
    for y, name in ((CFG["near_h100"], "filed: 'order-0 outlier' if both above"), (CFG["near_l40"], "filed: 'card difference' if both below")):
        ax.axhline(y, color=MUTED, lw=0.8, ls="--", zorder=1)
        ax.text(-0.25, y + 0.004, f"{y:.3f} {name}", fontsize=7, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xlim(-0.3, 2.4)
    ax.set_ylim(0.62, 1.0)
    ax.set_xlabel("batch order of the copy's Pile rows")
    ax.set_ylabel("held-out P(target) of the copy")
    ax.set_title("Emoticon Adam Pile-text copy: does batch order move it?", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", frameon=False)


def corner_point(cell: str, order: int) -> tuple[float, float]:
    if order == 0:
        task, opt = cell.split("/")
        shares = load(E119 / f"teacher_route_split_{task}.json")["students"][f"{opt}_B"]["a"]["shares"]
    else:
        shares = OUT["cells"][cell]["orders"][str(order)]["a"]["shares"]
    return shares["e_gate"], shares["r"]


def corner_panel(ax: Any) -> None:
    lo_x, hi_x, lo_y, hi_y = 0.1, 0.32, 0.55, 0.82
    ax.fill_between([lo_x, MASKED_LO], R_HI, hi_y, color=REGION, zorder=0)
    ax.axvline(MASKED_LO, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.axhline(R_HI, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.text(lo_x + 0.005, hi_y - 0.008, "re-switching corner (E119's lines)", fontsize=7, color=MUTED, va="top")
    ax.text(MASKED_LO + 0.003, lo_y + 0.01, "masked-off share 0.25", fontsize=7, color=MUTED, rotation=90)
    ax.text(hi_x - 0.003, R_HI - 0.012, "r 0.6", fontsize=7, color=MUTED, ha="right")
    for cell in ("r0_t352_v310/adam", "r0_t352_v310/sgd", "r1_t534_v369/sgd"):
        task, opt = cell.split("/")
        pts = [corner_point(cell, o) for o in (0, 1, 2)]
        ax.plot(*zip(*pts), color=COLOUR[task], lw=0.8, alpha=0.5, zorder=2)
        for o, (x, y) in enumerate(pts):
            ax.plot(x, y, MARKER[opt], ms=9, color=COLOUR[task], mfc="white" if o == 0 else COLOUR[task], mew=1.5, zorder=4)
            ax.annotate(str(o), (x, y), xytext=(6, 4), textcoords="offset points", fontsize=7, color=INK)
        name = f"{task.split('_')[0]} {'SGD' if opt == 'sgd' else 'Adam'} Pile-text copy"
        ax.plot([], [], MARKER[opt], ms=8, color=COLOUR[task], label=name)
    ax.plot([], [], "o", ms=8, mfc="white", mec=INK, label="hollow: order 0 (E119); filled: orders 1, 2")
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    ax.set_xlabel("masked-off share (gain carried by subcomponents Γ keeps off)")
    ax.set_ylabel("re-switching share r")
    ax.set_title("E119's re-switching copies: do they stay in the corner?", fontsize=9)
    ax.legend(fontsize=7, loc="lower left", frameon=False)


def main() -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    emoticon_panel(left)
    corner_panel(right)
    for ax in (left, right):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "orders.png", dpi=150)
    print(FIGURES / "orders.png")


if __name__ == "__main__":
    main()
