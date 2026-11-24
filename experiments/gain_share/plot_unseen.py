"""Run 2's figure (cell P1.58, the unseen-cue test), from results/unseen/{controls,e62}_<L>/*.json only (no model): per
state, the share of F's gain the gain-only fit recovers on keys it never saw (filled) beside the fitted keys' held-out
rows (open, run 1's reading), one panel per level. A state with no trained answer, or whose F gains under 1 log-odds,
is drawn at its KL-recovered share (triangle). With --sketch: the planned figure from made-up numbers, one panel per
rival of experiments/gain_share/card_unseen.md.

    ~/commute-agents/fleet/cpu-run --mem 2G python3 experiments/gain_share/plot_unseen.py [--sketch]
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as FG

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results" / "unseen", HERE / "figures"
COLUMNS = ("x2", "x10", "random", "new_few", "new_thin", "copy", "lookup")
LABELS = ("x2", "x10", "random", "new few", "new thin", "E62 copy", "E62 lookup")
WHERE = {"unseen": ("unseen keys", FG.METHODS["neuron"], 0.12), "seen": ("fitted keys, held-out rows", FG.METHODS["regating"], -0.12)}


def column(state: str) -> str:
    return "random" if state.startswith("random") else state.removeprefix("e62/").rsplit("_s", 1)[0]


def points(rec: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for where in WHERE:
        r = rec[where]
        y, kind = (r["share"], "share") if r["share"] is not None else (r["kl_share"], "kl_share")
        out.append({"state": rec["state"], "column": column(rec["state"]), "level": rec["level"], "where": where, "y": y, "kind": kind})
    return out


def records(results: Path) -> list[dict[str, Any]]:
    files = sorted(p for d in results.glob("*_[NV]") if d.is_dir() and not d.name.startswith("e214") for p in d.glob("*.json"))
    return [pt for p in files if (r := json.loads(p.read_text())).get("done") for pt in points(r)]


def panel(ax: Axes, pts: Sequence[dict[str, Any]], title: str, th: dict[str, Any]) -> None:
    for where, (label, st, dx) in WHERE.items():
        for kind, marker in (("share", "o"), ("kl_share", "^")):
            xy = [(COLUMNS.index(p["column"]) + dx, p["y"]) for p in pts if p["where"] == where and p["kind"] == kind and p["y"] is not None]
            if xy:
                xs, ys = zip(*xy, strict=True)
                face = st.color if where == "unseen" else "none"
                name = label if kind == "share" else f"{label}, KL-recovered share"
                ax.scatter(xs, ys, marker=marker, s=26, facecolors=face, edgecolors=st.color, linewidths=1.0, label=name)
    ax.axvline(4.5, color="#bbbbbb", linewidth=0.8)
    for y, text in ((1.0, "F itself"), (th["exact_min"], "exact controls' bar"), (th["new_max"], "new edits' bar")):
        ax.axhline(y, color="#777777", linewidth=0.8, linestyle="--" if y == 1.0 else ":")
        ax.annotate(text, (0, y), xycoords=("axes fraction", "data"), xytext=(2, 2), textcoords="offset points", fontsize=7, color="#555555")
    ax.set_xticks(range(len(COLUMNS)), LABELS, rotation=20)
    ax.set_ylabel("share of F's gain recovered")
    ax.set_ylim(-0.4, 1.3)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, loc="lower left", frameon=False)


def sketch() -> list[dict[str, Any]]:
    """Made-up numbers: rival a (the share reads the mechanism: controls separate, copy transfers, lookup does not) and
    rival b (the gains imitate any narrow change: the new edits transfer too, and the run stops before E62)."""
    made = {"a": {"x2": (0.99, 0.99), "x10": (0.97, 0.98), "random": (0.25, 0.27), "new_few": (0.05, 0.95), "new_thin": (0.08, 0.96),
                  "copy": (0.9, 1.0), "lookup": (0.1, 0.97)},
            "b": {"x2": (0.99, 0.99), "x10": (0.97, 0.98), "random": (0.25, 0.27), "new_few": (0.7, 0.95), "new_thin": (0.65, 0.96)}}  # fmt: skip
    kl = ("x2", "x10", "random")
    return [{"state": c, "column": c, "level": rival, "where": w, "y": v[i], "kind": "kl_share" if c in kl else "share"}
            for rival, d in made.items() for c, v in d.items() for i, w in enumerate(("unseen", "seen"))]  # fmt: skip


def main(argv: Sequence[str]) -> None:
    th = json.loads((RESULTS / "predictions.json").read_text())["thresholds"]
    if "--sketch" in argv:
        pts, panels, out = sketch(), {"a": "Sketch, rival a: the share reads the mechanism", "b": "Sketch, rival b: gains imitate any narrow change"}, "sketch_unseen.png"
        n = "made-up numbers, not a result"
    else:
        pts, panels, out = records(RESULTS), {"N": "One gain per neuron and head", "V": "One gain per VPD subcomponent (no exact control)"}, "gain_share_unseen.png"
        n = "one fit per state and level, on half of each state's keys; E62: 3 seeds per map"
        assert pts, f"no finished state under {RESULTS}"
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    for ax, (lv, title) in zip(axes, panels.items(), strict=True):
        panel(ax, [p for p in pts if p["level"] == lv], title, th)
    FG.save(fig, FIGURES / out, n=n, data=pts)


if __name__ == "__main__":
    main(sys.argv[1:])
