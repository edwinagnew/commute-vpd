"""Figures for cell P1.37 (R at the VPD featurization with site remainders).

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/kstar_vpd/plot.py --sketch   (the card's sketch, made-up numbers)
    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/kstar_vpd/plot.py            (after score.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as FG

HERE = Path(__file__).resolve().parent
REUSE = ("x2", "x10", "rebuilt", "moved")
NEW = ("new_few", "new_thin")
COLORS = {"reuse": "#0072B2", "new": "#D55E00"}  # Okabe-Ito, as `figures.METHODS`


def draw_known(ax: Axes, r: dict[str, float], title: str) -> None:
    """R per known-answer item, reuse items above the new edits, with the lowest R1b reuse item marked."""
    items = (*REUSE, *NEW)
    for i, item in enumerate(items):
        side = "reuse" if item in REUSE else "new"
        ax.scatter([r[item]], [i], color=COLORS[side], s=26, linewidths=0, label=side if item in (REUSE[0], NEW[0]) else None)
    ax.set_yticks(range(len(items)), [f"{x}*" if x == "moved" else x for x in items])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.02)
    FG.reference(ax, min(r[x] for x in REUSE if x != "moved"), "lowest of x2, x10, rebuilt")
    ax.set_xlabel("R at the VPD featurization (read positions)")
    ax.set_title(title, fontsize=8, loc="left", pad=14)
    ax.legend(fontsize=7, loc="upper left", frameon=False)


SKETCH = {  # made-up numbers
    "Rival a: R1b recovers":
        {"x2": 0.97, "x10": 0.93, "rebuilt": 0.90, "moved": 0.70, "new_few": 0.62, "new_thin": 0.71},
    "Rival b: new_thin at the reuse level":
        {"x2": 0.97, "x10": 0.93, "rebuilt": 0.90, "moved": 0.70, "new_few": 0.80, "new_thin": 0.95},
}


def sketch(out: Path = HERE / "figures") -> None:
    fig, axes = FG.figure((7.5, 3.4), ncols=2)
    data: list[dict[str, Any]] = []
    for ax, (title, r) in zip(axes, SKETCH.items(), strict=True):
        draw_known(ax, r, title)
        data += [{"panel": title, "item": k, "R": v} for k, v in r.items()]
    fig.suptitle("Planned figure (made-up numbers): are the built new edits below the reuse items?", fontsize=9)
    FG.save(fig, out / "sketch.png", n="made-up numbers; one fine-tune per item. *moved: its 8 appended neurons are graft variables scoring 0, "
            "so it is predicted low", data=data)  # fmt: skip


ROWS = {  # known item -> its edit name as in the paper's text (E195's copy items, E214's built new edits)
    "do_nothing": "do nothing",
    "x2": "copy neurons x2",
    "x10": "copy neurons x10",
    "moved": "copy neurons moved onto 8 appended neurons",
    "rebuilt": "copy neurons zeroed, layers 2-3 MLPs retrained",
    "new_few": "lookup in 8 quiet layer-0 neurons",
    "new_thin": "lookup spread over layer 0",
}
SIDE = {"do_nothing": "control", **{i: "reuse" for i in REUSE}, **{i: "new" for i in NEW}}
SIDE_COLORS = {"control": "#555555", **COLORS}


def known_rows(results: Path) -> list[dict[str, Any]]:
    """One record per known item: R and mean s over changed K* members (the filed, capped search), and R from the
    uncapped side check where it was run. Do-nothing comes from a stage-0 file (every shard reads the same)."""
    rows = []
    for item in ROWS:
        path = results / ("stage0_pod0/do_nothing.json" if item == "do_nothing" else f"items/{item}.json")
        pol = json.loads(path.read_text())["searches"]["main"]["polished"]
        conv = results / "converge" / f"{item}.json"
        unc = json.loads(conv.read_text())["read"]["polished"] if conv.exists() else None
        rows.append({"item": item, "side": SIDE[item], "R_capped": pol["R_read"], "K_capped": pol["size"],
                     "rho_capped": pol["rho_read"], "R_uncapped": unc and unc["R_read"], "K_uncapped": unc and unc["size"]})
    return rows


def plot_known(rows: list[dict[str, Any]], max_new: float, band: float, out: Path) -> None:
    fig, ax = FG.figure((7.0, 3.6))
    for i, r in enumerate(rows):
        c = SIDE_COLORS[r["side"]]
        ax.scatter([r["R_capped"]], [i], color=c, s=30, linewidths=0, zorder=3)
        if r["R_uncapped"] is not None:
            ax.scatter([r["R_uncapped"]], [i], facecolors="none", edgecolors=c, s=46, linewidths=1.1, zorder=3)
    ax.scatter([], [], color="#333333", s=30, label="filed search, 2,000 tries (capped)")
    ax.scatter([], [], facecolors="none", edgecolors="#333333", s=46, label="search run to the end (side check)")
    ax.axvline(max_new, color="#D55E00", linewidth=0.8, linestyle="--", label="highest new edit (lookup spread over layer 0, filed)")
    ax.axvspan(max_new, max_new + band, color="#D55E00", alpha=0.15, linewidth=0, label="plus the band: reuse must sit right of it")
    ax.set_yticks(range(len(rows)), [ROWS[r["item"]] for r in rows])
    for t, r in zip(ax.get_yticklabels(), rows, strict=True):
        t.set_color(SIDE_COLORS[r["side"]])
    ax.invert_yaxis()
    ax.set_xlim(0.970, 1.003)
    ax.set_xlabel("reuse score R, read positions (1 = unchanged)")
    ax.set_title("x10, built as reuse, scores below both built new edits", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left", frameon=False)
    FG.save(fig, out / "r_known.png", n="one fine-tune per item; blue = built as reuse, orange = built as new, grey = control. "
            "Reuse items: 75 windows, every position; new edits: 512 windows, last position.", data=rows)  # fmt: skip


def plot_rho(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = FG.figure((7.0, 3.2))
    for i, r in enumerate(rows):
        ax.scatter([r["rho_capped"] if r["rho_capped"] is not None else 1.0], [i], color=SIDE_COLORS[r["side"]], s=30, linewidths=0)
    ax.set_yticks(range(len(rows)), [ROWS[r["item"]] for r in rows])
    for t, r in zip(ax.get_yticklabels(), rows, strict=True):
        t.set_color(SIDE_COLORS[r["side"]])
    ax.invert_yaxis()
    FG.reference(ax, 1.0, "unchanged")
    ax.set_xlabel("mean grade s over the circuit's changed variables (squared cosine)")
    ax.set_title("x10's changed subcomponents keep less of M's write direction\nthan either new edit's do", fontsize=9)
    FG.save(fig, out / "rho_known.png", n="filed search; do-nothing has no changed variable and is drawn at 1", data=rows)


def plot_pairs(p: dict[str, Any], band: float, out: Path) -> None:
    fig, (a, b) = FG.figure((7.5, 3.4), ncols=2)
    gaps = p["gaps"]["read"]
    data = [{"pair": k, "R_gap": v, "minus_size_gap": p["minus_size_gap"][k], "S3_gap": p["S3_gap"][k]} for k, v in gaps.items()]
    for ax, key, xlabel in ((a, "minus_size_gap", "minus the update-size gap, reuse - new"), (b, "S3_gap", "held-model S3 gap, reuse - new")):
        ax.scatter([d[key] for d in data], [d["R_gap"] for d in data], color=COLORS["reuse"], s=24, linewidths=0)
        ax.axhline(band, color="#777777", linewidth=0.8, linestyle="--", label="band from the known items")
        ax.axhline(0, color="#999999", linewidth=0.6)
        ax.axvline(0, color="#999999", linewidth=0.6)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.legend(fontsize=6.5, loc="upper left", frameon=False)
    a.set_ylabel("R gap, reuse - new (read positions)")
    b.sharey(a)
    a.set_title(f"against update size: Spearman {p['vs_minus_size']['rho']:.2f}", fontsize=8, loc="left")
    b.set_title(f"against S3: Spearman {p['vs_S3']['rho']:.2f}", fontsize=8, loc="left")
    fig.suptitle("On E187's nine pairs every R gap is positive and below the known items' band", fontsize=9)
    FG.save(fig, out / "r_pairs.png", n="nine pairs, each gap = mean R over its reuse states - mean over its new states (68 states)", data=data)


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    """From results/summary.json (score.py), the item files and the uncapped side check (converge.py)."""
    s = json.loads((results / "summary.json").read_text())
    rows = known_rows(results)
    plot_known(rows, s["R1"]["max_new"], s["band"]["band"], out)
    plot_rho(rows, out)
    plot_pairs(s["pairs"], s["band"]["band"], out)


if __name__ == "__main__":
    if "--sketch" in sys.argv[1:]:
        sketch()
    else:
        main()
