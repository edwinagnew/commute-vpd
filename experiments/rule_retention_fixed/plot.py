"""E195 (cell P1.25) figure, from results/stage0.json, stage2.json and units_rebuilt.npz:
    figures/retention.png  left: the retention for the contrast per item at level A (neurons; filled) and level B
                           (heads and blocks; open), each order's value with the rebuilt item's range over orders;
                           do-nothing and the moved copy are drawn as their reading ("no change"). Right: histogram of
                           r over the MLP neurons of layers 2 and 3 for the rebuilt item.

    ~/commute-agents/fleet/cpu-run python experiments/rule_retention_fixed/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from commute.figures import figure, save

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
COLOR = {"turned up": "#0072B2", "rebuilt": "#D55E00", "reference": "#777777"}
LABEL = {"do_nothing": "do-nothing", "moved_copy": "moved copy", "x2": "turned up x2", "x10": "turned up x10", "rebuilt": "rebuilt"}


def color(item: str) -> str:
    return COLOR["rebuilt"] if item == "rebuilt" else COLOR["turned up"] if item.startswith("x") else COLOR["reference"]


def draw_items(ax: Any, readings: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    data = []
    for i, (item, rd) in enumerate(readings.items()):
        for lv, dy, face in (("A", -0.15, None), ("B", 0.15, "none")):
            c, y = rd[lv]["contrast"], i + dy
            if c["reading"] != "read":
                ax.annotate(f"{lv}: {c['reading']}", (0.5, y), ha="center", va="center", fontsize=7, color=color(item))
                data.append({"item": item, "level": lv, "reading": c["reading"]})
                continue
            ret = c["retention"]
            ax.plot([min(ret), max(ret)], [y, y], color=color(item), linewidth=1.5)
            ax.scatter(ret, [y] * len(ret), facecolors=face or color(item), edgecolors=color(item), s=22, marker="o" if lv == "A" else "s", zorder=3)
            data += [{"item": item, "level": lv, "reading": "read", "order": k, "retention": v} for k, v in enumerate(ret)]
    ax.set_yticks(range(len(readings)), [LABEL[k] for k in readings], fontsize=7)
    ax.set_ylim(len(readings) - 0.5, -0.5)
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel("retention for the contrast (mean r over a smallest set)", fontsize=8)
    ax.scatter([], [], color="#555555", marker="o", s=22, label="neurons (level A)")
    ax.scatter([], [], facecolors="none", edgecolors="#555555", marker="s", s=22, label="heads and blocks (level B)")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title("Retention for the contrast per item;\none point per order of the set search", fontsize=8)
    return data


def draw_hist(ax: Any, units: Any) -> list[dict[str, Any]]:
    r = np.concatenate([units[f"neuron{l}_r"][~units[f"neuron{l}_zero"]] for l in (2, 3)])
    bins = np.linspace(0, 1, 41)
    counts, _ = np.histogram(r, bins)
    ax.bar(bins[:-1], counts, width=bins[1] - bins[0], align="edge", color=COLOR["rebuilt"])
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("r of a neuron with its partner (cos² of their outputs)", fontsize=8)
    ax.set_ylabel("neurons", fontsize=8)
    ax.set_title(f"Rebuilt item: r of the {len(r):,} neurons of layers 2 and 3", fontsize=8)
    return [{"panel": "histogram", "bin_lo": float(a), "bin_hi": float(b), "neurons": int(n)} for a, b, n in zip(bins[:-1], bins[1:], counts, strict=True)]


def main() -> None:
    s0, s2 = (json.loads((RESULTS / f"stage{n}.json").read_text()) for n in (0, 2))
    readings = {k: s0["items"][k] for k in ("do_nothing", "moved_copy", "x2", "x10")} | {"rebuilt": s2["readings"]}
    fig, (left, right) = figure((7.5, 3.4), ncols=2)
    data = draw_items(left, readings)
    data += draw_hist(right, np.load(RESULTS / "units_rebuilt.npz"))
    fig.suptitle("Does the rebuilt item read lower than the turned-up circuit? (filed: lower; no bar)", fontsize=9)
    save(fig, FIGURES / "retention.png", n="copy24's 75 held-out windows; 5 seeded orders of the smallest-set search per item; one fit of the rebuilt item", data=data)


if __name__ == "__main__":
    main()
