"""P1.38's figure: per E187 pair, the R gap (reuse minus new, read grade) at saved size (E221, black bar, seeds' se as
a line) and at matched size (this run, blue dot), one panel per level; 0 is no gap. No GPU:

    python experiments/kstar_size_matched/plot.py      (results/summary.json "plot" -> figures/r_gap_matched.png)
    python experiments/kstar_size_matched/plot.py --input figures/sketch_input.json --out figures/sketch.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commute import figures as fg

HERE = Path(__file__).resolve().parent


def main(src: Path, out: Path) -> None:
    d = json.loads(src.read_text())
    if "plot" in d:
        d = d["plot"]
    panels = d["panels"]
    fig, axs = fg.figure(fg.FULL_WIDTH, ncols=len(panels))
    rows = []
    for ax, (title, pts) in zip(axs, panels.items(), strict=True):
        for i, p in enumerate(pts):
            ax.plot([p["unscaled"] - p["se"], p["unscaled"] + p["se"]], [i, i], color="#000000", linewidth=0.6)
            ax.scatter([p["unscaled"]], [i], color="#000000", marker="|", s=50, label="saved size (E221)" if i == 0 else None)
            if p["matched"] is not None:
                ax.scatter([p["matched"]], [i], color="#0072B2", marker="o", s=14, label="matched size" if i == 0 else None)
            rows.append({"panel": title} | p)
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.set_yticks(range(len(pts)), [p["pair"] for p in pts], fontsize=6)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("R gap, reuse minus new", fontsize=7)
    axs[0].legend(fontsize=6)
    fg.save(fig, out, n=d["n"], data=rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=HERE / "results" / "summary.json")
    ap.add_argument("--out", type=Path, default=HERE / "figures" / "r_gap_matched.png")
    a = ap.parse_args()
    main(a.input if a.input.is_absolute() else Path.cwd() / a.input, a.out if a.out.is_absolute() else Path.cwd() / a.out)
