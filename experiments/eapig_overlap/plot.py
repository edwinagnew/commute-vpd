"""P2.69 figure: every item's edge IoU (F's 5% circuit against M's on the same windows) on one axis, beside M's
resampling range on that task set (grey band), the controls first. Filled points entered the headline (both circuits at
least 70% faithful); hollow points did not ("no gap", "empty" or "circuit not faithful"). From results/items/*.json.
`--sketch` draws the card's sketch instead: the same figure with made-up numbers, one panel per rival's outcome.

    ~/commute-agents/fleet/cpu-run python3 experiments/eapig_overlap/plot.py [--sketch] [--items DIR --out PNG]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
GROUPS = [("x2", ["x2"]), ("new_few", ["new_few"]), ("new_thin", ["new_thin"]), ("random (size)", ["random_size_241001"]),
          ("random (KL)", ["random_kl_241002"]), ("x10", ["x10"]), ("rebuilt", ["rebuilt"]),
          ("E62 copy", [f"e62/copy_s{s}" for s in range(3)]), ("E62 lookup", [f"e62/lookup_s{s}" for s in range(3)]), ("E62 random", ["e62_random"])]  # fmt: skip
KEYS = ("copy", "rank1", "elicit", "cipher", "prior_cipher")
GROUPS += [(f"key {t}", [f"e58/{t}_s{s}_scaled" for s in range(3)]) for t in KEYS]
GROUPS += [(f"key {t} random", [f"random_e58/{t}_s{s}" for s in range(3)]) for t in KEYS]
COLOR, BAND = "#0072B2", "#999999"


def panel(ax: Any, cols: list[tuple[str, list[tuple[float, bool]], tuple[float, float] | None]]) -> list[dict[str, Any]]:
    rows, banded = [], False
    for i, (name, points, band) in enumerate(cols):
        if band is not None:
            ax.fill_between([i - 0.35, i + 0.35], band[0], band[1], color=BAND, alpha=0.35, linewidth=0, label=None if banded else "M against M, resampled windows")
            banded = True
        for v, read in points:
            ax.scatter([i], [v], s=26, color=COLOR if read else "white", edgecolors=COLOR, linewidths=1.0, zorder=3)
            rows.append({"group": name, "edge_iou": v, "headline": read, "range_lo": band[0] if band else None, "range_hi": band[1] if band else None})
    ax.set_xticks(range(len(cols)), [c[0] for c in cols], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("edge IoU with M's circuit")
    ax.set_ylim(-0.03, 1.05)
    ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle=":", label="do-nothing: M against M, same windows (reads 1)")
    ax.scatter([], [], s=26, color=COLOR, edgecolors=COLOR, label="both circuits at least 70% faithful (headline)")
    ax.scatter([], [], s=26, color="white", edgecolors=COLOR, label="circuit not faithful (kept out)")
    ax.legend(fontsize=7, loc="lower left")
    return rows


def real(items_dir: Path, out: Path) -> None:
    items = {json.loads(p.read_text())["item"]: json.loads(p.read_text()) for p in items_dir.glob("*.json") if not p.name.startswith("M_")}
    cols = []
    for name, members in GROUPS:
        got = [items[m] for m in members if m in items and items[m]["five_percent"]["edge_iou"] is not None]
        band = next((tuple(r["M_resampling_range"]) for r in got if r["M_resampling_range"]), None)
        cols.append((name, [(r["five_percent"]["edge_iou"], r["reading"] == "read") for r in got], band))
    fig, ax = FG.figure(FG.FULL_WIDTH)
    rows = panel(ax, cols)
    for x in (6.5, 9.5):  # built controls | E62 | key grid
        ax.axvline(x, color="#BBBBBB", linewidth=0.6, linestyle="-")
    ax.set_title("Does an update move its circuit further than resampling M's windows does?", fontsize=9)
    FG.save(fig, out, n=f"{len(rows)} items, one 5% circuit (46 of 927 edges) each; band: 5 resamples of the task's windows",
            data=rows)  # fmt: skip


def sketch() -> None:
    """Made-up numbers: rival a (reuse reading) puts the trained copy-like tasks at the x2 control's IoU; rival b (new
    reading) puts them under M's resampling range, as the built new edits are."""
    names = ["x2", "new_few", "new_thin", "random (size)", "E62 copy", "E62 lookup", "key copy", "key cipher"]
    band = (0.90, 0.97)
    a = [[1.0], [0.62], [0.58], [0.94], [0.95, 0.96, 0.93], [0.71, 0.74, 0.69], [0.95, 0.94, 0.96], [0.80, 0.77, 0.83]]
    b = [[1.0], [0.62], [0.58], [0.94], [0.66, 0.70, 0.64], [0.55, 0.59, 0.52], [0.68, 0.72, 0.65], [0.50, 0.47, 0.55]]
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    rows = []
    for ax, vals, title in ((axes[0], a, "rival a, reuse: copy-like tasks near x2"), (axes[1], b, "rival b, new: trained tasks under the band")):
        rows += [r | {"panel": title} for r in panel(ax, [(n, [(v, True) for v in vs], band) for n, vs in zip(names, vals, strict=True)])]
        ax.set_title(title, fontsize=8)
    fig.suptitle("SKETCH, made-up numbers: edge IoU with M's circuit, beside M's resampling range")
    FG.save(fig, HERE / "figures" / "sketch.png", n="made-up numbers: 1 to 3 seeds per column, one 5% circuit each", data=rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketch", action="store_true")
    ap.add_argument("--items", type=Path, default=HERE / "results" / "items")
    ap.add_argument("--out", type=Path, default=HERE / "figures" / "edge_iou.png")
    args = ap.parse_args()
    if args.sketch:
        sketch()
    else:
        real(args.items, args.out)
