"""E277 (cell P1.56) figure from results/summary.json (run.py --stage summary): figures/eb_neurons.png.

Left: E_b over K (the needed set) per built edit at the top-neuron featurization, beside E268's at VPD. Right: each of
E229's 20 fine-tunes beside its matched random update, |K| written above the fine-tune; an item with |K| < 100 ("too
small to read") is drawn hollow. Writes .png, .pdf and .csv.

    ~/commute-agents/fleet/cpu-run python3 experiments/needed_neurons/plot.py [SUMMARY] [OUT_DIR]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
E268 = HERE.parent / "kstar_needed" / "results" / "summary.json"
BUILT = ["x2", "x10", "rebuilt", "new_few", "new_thin", "random_size_241001", "random_kl_241002"]
SHORT = {"random_size_241001": "random (size)", "random_kl_241002": "random (KL)"}


def point(ax: Any, x: float, r: dict[str, Any], style: FG.Style, label: str | None) -> None:
    hollow = r["too_small"]
    ax.scatter([x], [r["E_b_K"]], marker=style.marker, s=30, linewidths=1.0, label=label,
               facecolors="none" if hollow else style.color, edgecolors=style.color)  # fmt: skip


def built(ax: Any, items: dict[str, Any], vpd: dict[str, Any]) -> list[dict[str, Any]]:
    rows, nb, vp = [], FG.METHODS["neuron"], FG.METHODS["spectrum"]
    for x, item in enumerate(BUILT):
        v = vpd.get(item, {}).get("E_b_K")
        if v is not None:
            ax.scatter([x], [v], color=vp.color, marker=vp.marker, s=26, label="VPD (E268)" if x == 0 else None)
        if item in items:
            point(ax, x, items[item], nb, "top neurons, p97 (this run)" if x == 0 else None)
            rows.append({"panel": "built", "item": item, "E_b_K": items[item]["E_b_K"], "E_b_K_vpd": v, "size_K": items[item]["size_K"]})
    ax.set_xticks(range(len(BUILT)), [SHORT.get(i, i) for i in BUILT], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("write share over the needed set K")
    ax.set_title("Built edits: write share over K at top neurons and at VPD", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    return rows


def trained(ax: Any, items: dict[str, Any]) -> list[dict[str, Any]]:
    names = [k for k, r in items.items() if r.get("world") == "trained" and not k.endswith("_rand")]
    rows, ft, rd = [], FG.METHODS["neuron"], FG.METHODS["random"]
    for x, item in enumerate(names):
        r, q = items[item], items.get(f"{item}_rand")
        point(ax, x, r, ft, "fine-tune" if x == 0 else None)
        ax.annotate(str(r["size_K"]), (x, r["E_b_K"]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=5, rotation=90)
        if q is not None:
            point(ax, x, q, rd, "random update, matched per matrix" if x == 0 else None)
        rows.append({"panel": "trained", "item": item, "E_b_K": r["E_b_K"], "size_K": r["size_K"],
                     "E_b_K_rand": q["E_b_K"] if q else None, "size_K_rand": q["size_K"] if q else None})  # fmt: skip
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=6)
    ax.set_ylim(-0.03, 1.1)
    ax.set_ylabel("write share over the needed set K")
    ax.set_title("Trained fine-tunes beside matched randoms (|K| above each)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    return rows


def main(summary: Path, out_dir: Path) -> None:
    items = json.loads(summary.read_text())["items"]
    vpd = json.loads(E268.read_text())["items"] if E268.is_file() else {}
    fig, (a, b) = FG.figure((10.0, 4.5), ncols=2)
    rows = built(a, items, vpd) + trained(b, items)
    FG.save(fig, out_dir / "eb_neurons.png", data=rows,
            n="one state per point; K from 256 pairs of inputs per state (4 derangements of 64 windows); hollow: |K| < 100, too small to read")  # fmt: skip


if __name__ == "__main__":
    main(
        Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "summary.json",
        Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "figures",
    )
