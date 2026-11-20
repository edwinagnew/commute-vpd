"""E233's figure: at the scored size, the gain, the re-gating share and the update's effective rank, each seed a point,
British spelling against its scrambled twin (results/scored/score.json, written by `run.py score`), with each side's
filed range (predictions.json "expected") as a pale bar on its row and the lowercase control's reading (stage0.json)
as a labelled line. Neurons for half the gain is left out: no British seed has a value (score.json k_half "reuse" []).

    python experiments/british_style/plot.py              (rehearse: plot.py rehearse, to real_vs_twin_rehearse.png)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from commute.figures import FULL_WIDTH, dots, figure, reference, save

HERE = Path(__file__).resolve().parent
SIDES = {"reuse": "British spelling", "new": "scrambled twin"}
# predictions.json "expected" at 0.0175, per side in SIDES order (its text, written out as numbers)
FILED = {"gain": ((2, 6), (4, 12)), "share": ((0.55, 0.85), (0.15, 0.5)), "erank": ((560, 610), (480, 600))}
PANELS = (("gain", "prior", "gain in log-odds of the new form", "gain floor 1"),
          ("share", "regating", "re-gating share r", "lowercase control"),
          ("erank", "spectrum", "effective rank of the update", "lowercase control"))  # fmt: skip


def main(folder: str) -> None:
    res = json.loads((HERE / "results" / folder / "score.json").read_text())
    control = json.loads((HERE / "results" / folder / "stage0.json").read_text())["control"]
    size = max(res["sizes"], key=float)
    fig, axes = figure((FULL_WIDTH[0], 2.8), ncols=3)
    rows = []
    for i, (ax, (m, style, label, ref)) in enumerate(zip(axes, PANELS, strict=True)):
        groups = {SIDES[s]: v for s, v in zip(SIDES, _values(res, size, m), strict=True)}
        for row, (lo, hi) in enumerate(FILED[m]):
            ax.barh(row, hi - lo, left=lo, height=0.5, color="#BBBBBB", alpha=0.35, linewidth=0)
        dots(ax, groups, method=style, xlabel=label)
        reference(ax, 1.0 if m == "gain" else control[m]["got"], ref)
        if i:
            ax.set_yticklabels([])
        rows += [{"metric": m, "side": side, "value": v, "size": size} for side, vs in groups.items() for v in vs]
    fig.suptitle(f"At update size {size} the re-gating share is low on both sides and the twin's is not below British",
                 fontsize=9)  # fmt: skip
    name = "real_vs_twin.png" if folder == "scored" else f"real_vs_twin_{folder}.png"
    seeds = len(res["steps"]["real"])
    note = "grey bars: each side's filed range; dashed: the lowercase fine-tune's reading at its own size, or the floor"
    save(fig, HERE / "figures" / name, n=f"{seeds} seeds per side, 500 read windows each. {note}", data=rows)


def _values(res: dict, size: str, m: str) -> list[list[float]]:
    if m == "gain":
        return [res["gain"]["real"], res["gain"]["twin"]]
    c = res["sizes"][size][m]
    return [c[s] for s in SIDES]


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scored")
