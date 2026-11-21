"""Figures for cell P1.43 (E239, was P1.37 run 2): the size-normalised reuse score N at the VPD featurization.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/kstar_vpd_norm/plot.py --sketch   (the card's sketch, made-up numbers)
    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/kstar_vpd_norm/plot.py            (after run.py's summary)
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as FG
from commute.jobs import load_experiment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E226_SUMMARY = ROOT / "experiments/kstar_vpd/results/summary.json"
REUSE = ("x2", "x10", "rebuilt", "moved")
NEW = ("new_few", "new_thin")
COLORS = {"reuse": "#0072B2", "new": "#D55E00"}  # Okabe-Ito, as `figures.METHODS`
N_LABEL = "N = 1 - mean turn share t over the changed parts of K*"


def draw_known(ax: Axes, n: Mapping[str, float | None], band: float, title: str, xlabel: str = N_LABEL) -> None:
    """One value per known-answer item, with the highest new edit plus the band."""
    items = (*REUSE, *NEW)
    for i, item in enumerate(items):
        side = "reuse" if item in REUSE else "new"
        v = n[item]
        if v is not None:
            ax.scatter([v], [i], color=COLORS[side], s=28, linewidths=0, label=f"built as {side}" if item in (REUSE[0], NEW[0]) else None)
    ax.set_yticks(range(len(items)), list(items))
    ax.invert_yaxis()
    top = max(v for k, v in n.items() if k in NEW and v is not None)
    ax.axvline(top, color=COLORS["new"], linewidth=0.8, linestyle="--", label="highest new edit")
    ax.axvspan(top, top + band, color=COLORS["new"], alpha=0.15, linewidth=0, label="plus the band: x10 must sit right of it")
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_title(title, fontsize=8, loc="left", pad=14)
    ax.legend(fontsize=6.5, loc="lower left", frameon=False)


SKETCH = {  # made-up numbers
    "Rival a: x10 above both new edits": {"x2": 0.85, "x10": 0.80, "rebuilt": 0.55, "moved": 0.82, "new_few": 0.40, "new_thin": 0.50},
    "Rival b: x10 at or below a new edit": {"x2": 0.50, "x10": 0.35, "rebuilt": 0.45, "moved": 0.52, "new_few": 0.40, "new_thin": 0.55},
}


def sketch(out: Path = HERE / "figures") -> None:
    fig, axes = FG.figure((7.5, 3.4), ncols=2)
    data: list[dict[str, Any]] = []
    for ax, (title, n) in zip(axes, SKETCH.items(), strict=True):
        draw_known(ax, n, 0.08, title)
        data += [{"panel": title, "item": k, "N": v} for k, v in n.items()]
    fig.suptitle("Planned figure (made-up numbers): does x10 read above both built new edits once size is divided out?", fontsize=9)
    FG.save(fig, out / "sketch.png", n="made-up numbers; one fine-tune per item; band from the two halves of D", data=data)


def known_figure(s: dict[str, Any], r226: dict[str, Any], out: Path) -> None:
    """R as filed (E226) beside N, same rows: the filed score puts x10 lowest, N puts it highest."""
    fig, (a, b) = FG.figure((7.5, 3.4), ncols=2)
    draw_known(a, r226["R1"]["R"], r226["band"]["band"], "R as filed (E226): x10 lowest", "R = mean grade s over K* (1 = every part kept its write)")
    draw_known(b, s["known"]["N"], s["band"]["N"], "N, size-normalised (this run): x10 highest")
    b.set_yticklabels([])
    fig.suptitle("Does dividing each part's grade loss by its weight change put x10 above both built new edits?", fontsize=9)
    data = [{"measure": "R", "item": k, "value": v} for k, v in r226["R1"]["R"].items()]
    data += [{"measure": "N", "item": k, "value": v} for k, v in s["known"]["N"].items()]
    FG.save(fig, out / "n_known.png", n="one fine-tune per item; bands from the two halves of D (E226's for R, this run's for N); moved's grafts left out of N",
            data=data)  # fmt: skip


def size_figure(s: dict[str, Any], out: Path) -> None:
    """Left: per pair, N's reuse-minus-new gap against how much smaller the reuse side's update is. Right: N per state
    against its update size."""
    rr: Any = load_experiment("r_reanalysis", "run")
    pairs = {p: v for p, v in rr.load(rr.PAIRS_JSON)["pairs"].items() if p.startswith("e187_")}
    reuse = {i for v in pairs.values() for i in v["reuse"]}
    states = sorted(reuse | {i for v in pairs.values() for i in v["new"]})
    size = rr.sizes(states)
    items = {i: json.loads((HERE / "results/items" / f"{i.replace('/', '-')}.json").read_text()) for i in states}
    n = {i: items[i]["searches"]["main"]["frob"]["N"] for i in states}
    fig, (a, b) = FG.figure((7.5, 3.4), ncols=2)
    gaps = s["pairs_N"]["gaps"]
    x = [pairs[p]["smaller_update"] * 1e3 for p in gaps]
    a.scatter(x, list(gaps.values()), color="#444444", s=24, linewidths=0, label="one trained pair")
    a.axhline(0, color="#999999", linewidth=0.6)
    a.set_xlabel("how much smaller the reuse side's update is (x 1e-3, relative weight norm)", fontsize=7)
    a.set_ylabel("N gap, reuse side minus new side", fontsize=7)
    a.set_title(f"Pairs: reuse side higher on {s['pairs_N']['reuse_higher']} of 9; Spearman {s['pairs_N']['spearman_gap_vs_smaller_update']:.2f}",
                fontsize=8, loc="left")  # fmt: skip
    a.legend(fontsize=6.5, loc="lower left", frameon=False)
    for side, group in (("reuse", reuse), ("new", set(states) - reuse)):
        g = sorted(group)
        b.scatter([size[i] for i in g], [n[i] for i in g], color=COLORS[side], s=14, linewidths=0, label=f"state labelled {side}")
    b.set_xscale("log")
    b.set_xlabel("update size |dW| / |W0| over the 24 matrices", fontsize=7)
    b.set_ylabel("N", fontsize=7)
    b.set_title(f"States: bigger updates read higher N; Spearman {s['pairs_N']['spearman_state_vs_size']:.2f}", fontsize=8, loc="left")
    b.legend(fontsize=6.5, frameon=False)
    fig.suptitle("Does N still follow update size on the nine trained pairs?", fontsize=9)
    data = [{"panel": "pair", "name": p, "x": pairs[p]["smaller_update"], "N_gap": g} for p, g in gaps.items()]
    data += [{"panel": "state", "name": i, "x": size[i], "N": n[i], "side": "reuse" if i in reuse else "new"} for i in states]
    FG.save(fig, out / "n_size.png", n="nine E187 pairs, 68 saved fine-tune states; one reading per state", data=data)


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    s = json.loads((results / "summary.json").read_text())
    known_figure(s, json.loads(E226_SUMMARY.read_text()), out)
    size_figure(s, out)


if __name__ == "__main__":
    if "--sketch" in sys.argv[1:]:
        sketch()
    else:
        main()
