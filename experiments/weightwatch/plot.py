"""P2.73 figure: each state's S (where its update's top output directions lie in M's own output variance; 1 = an
isotropic direction) beside its same-spectrum Haar null (grey: the 20 draws' range) and the gradient-geometry diagnostic
S_G (hollow). The column rule reads the first three columns: x2 above both new edits by more than the Haar spread.
From results/summary.json. `--sketch` draws the card's sketch: the same figure with made-up numbers, one panel per rival.

    ~/commute-agents/fleet/cpu-run python3 experiments/weightwatch/plot.py [--sketch]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
COLOR, BAND = "#0072B2", "#999999"
BUILT = ["x2", "x10", "new_few", "new_thin", "rebuilt", "random_size_241001", "random_kl_241002", "trigger"]
LABEL = {"random_size_241001": "random (size)", "random_kl_241002": "random (KL)", "trigger": "trigger (lambda_min)"}


def panel(ax: Any, cols: list[tuple[str, float | None, tuple[float, float] | None, float | None]]) -> list[dict[str, Any]]:
    rows, first = [], True
    for i, (name, s, band, sg) in enumerate(cols):
        if band is not None:
            ax.fill_between([i - 0.35, i + 0.35], band[0], band[1], color=BAND, alpha=0.35, linewidth=0, label="Haar null, 20 draws" if first else None)
            first = False
        if sg is not None:
            ax.scatter([i], [sg], s=26, color="white", edgecolors=COLOR, linewidths=1.0, zorder=3)
        if s is not None:
            ax.scatter([i], [s], s=26, color=COLOR, zorder=4)
        rows.append({"state": name, "S": s, "haar_lo": band[0] if band else None, "haar_hi": band[1] if band else None, "S_G": sg})
    ax.set_xticks(range(len(cols)), [c[0] for c in cols], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("S, base-use ratio after Zhong")
    ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle=":", label="isotropic direction (reads 1)")
    ax.scatter([], [], s=26, color=COLOR, label="S of the update")
    ax.scatter([], [], s=26, color="white", edgecolors=COLOR, label="S_G, gradient geometry")
    ax.legend(fontsize=7, loc="upper right")
    return rows


def label(s: str) -> str:
    return LABEL.get(s, s.replace("e62/", "E62 ").replace("panel/", "").replace("_s", " s"))


def real(summary: Path, out: Path) -> None:
    """The column claim on a log axis: built edits, randoms and the trigger, then E62's copy and lookup fine-tunes,
    then the 36 panel fine-tunes; solid grey lines separate the three groups."""
    rows = json.loads(summary.read_text())["rows"]
    e62, pan = [s for s in rows if s.startswith("e62/")], [s for s in rows if s.startswith("panel/")]
    order = [s for s in BUILT if s in rows] + e62 + pan
    cols = [(label(s), rows[s]["S"], tuple(rows[s]["haar_range"]) if rows[s]["haar_range"] else None, rows[s]["S_G"]) for s in order]
    fig, ax = FG.figure((FG.FULL_WIDTH[0], FG.FULL_WIDTH[1] * 1.3))
    data = panel(ax, cols)  # type: ignore[arg-type]
    for x in (len(BUILT) - 0.5, len(BUILT) + len(e62) - 0.5):
        ax.axvline(x, color="#bbbbbb", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_title("Does S put the reuse edit x2 above the new edits by more than the Haar spread?", fontsize=9)
    FG.save(fig, out, n=f"{len(data)} states, one update each; grey band: the range of 20 Haar rotations of each state's kept directions",
            data=data)


def sketch() -> None:
    """Made-up numbers. Rival a: x2 sits above both new edits by more than the Haar spread (S separates built reuse
    from built new wiring). Rival b: the new edits write along busy unembedding directions and sit as high as x2."""
    names = ["x2", "new_few", "new_thin", "trigger", "E62 copy", "E62 lookup", "panel"]
    band = (0.85, 1.15)
    a = [2.4, 1.2, 1.3, 0.02, 1.8, 1.5, 1.6]
    b = [2.4, 2.3, 2.5, 0.02, 1.8, 1.7, 1.6]
    sg = [1.4] * len(names)
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    rows = []
    for ax, vals, title in ((axes[0], a, "rival a: x2 above the new edits, S earns a column"), (axes[1], b, "rival b: no separation, related-work negative")):
        rows += [r | {"panel": title} for r in panel(ax, [(n, v, band, g) for n, v, g in zip(names, vals, sg, strict=True)])]
        ax.set_title(title, fontsize=8)
    fig.suptitle("SKETCH, made-up numbers: S of each update against its Haar null")
    FG.save(fig, HERE / "figures" / "sketch.png", n="made-up numbers: one state per column; band: 20 Haar draws", data=rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketch", action="store_true")
    ap.add_argument("--summary", type=Path, default=HERE / "results" / "summary.json")
    ap.add_argument("--out", type=Path, default=HERE / "figures" / "s_vs_haar.png")
    args = ap.parse_args()
    if args.sketch:
        sketch()
    else:
        real(args.summary, args.out)
