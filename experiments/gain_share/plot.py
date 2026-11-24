"""The gain-only share's figure (cell P1.58), from results/stage{1,2,3}/*.json only (no model): per
state, the share of F's trained-answer gain that the best gain-only update recovers, at both levels (one gain per
neuron and head; one gain per VPD subcomponent and remainder), the built controls beside E62's copying map and lookup.
A state with no trained answer, or whose F gains under 1 log-odds, is drawn at its KL-recovered share (open marker).
Also scores the filed rules on E62 (predictions.json "rules") into results/score.json.

    ~/commute-agents/fleet/cpu-run --mem 2G python3 experiments/gain_share/plot.py [RESULTS_DIR FIGURES_DIR]
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
COLUMNS = ("x2", "x10", "random", "new_few", "new_thin", "copy", "lookup", "v2", "o2", "q2")  # e62/copy_s<k> -> copy, e280/v2 -> v2
LEVELS = {"N": ("neurons and heads", FG.METHODS["neuron"], -0.14), "V": ("VPD subcomponents", FG.METHODS["regating"], 0.14)}


def column(state: str) -> str:
    """e62/copy_s1 -> copy, random_kl_241002 -> random, x2 -> x2."""
    if state.startswith("random"):
        return "random"
    return state.removeprefix("e62/").rsplit("_s", 1)[0] if state.startswith("e62/") else state.removeprefix("e280/")


def records(results: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted([p for k in (1, 2, 3) for p in (results / f"stage{k}").glob("*.json")]):
        r = json.loads(p.read_text())
        if not r.get("done"):
            continue
        y, kind = (r["share"], "share") if r["share"] is not None else (r["kl_share"], "kl_share")
        out.append({"state": r["state"], "column": column(r["state"]), "level": r["level"],
                    "y": y, "kind": kind, "share": r["share"], "kl_share": r["kl_share"], "kl_fg": r["kl_fg"], "kl_fm": r["kl_fm"]})  # fmt: skip
    return out


def panel(ax: Axes, recs: Sequence[dict[str, Any]], title: str) -> None:
    """Every state's reading as a point: one column per state kind, the two levels side by side, open where it is the
    KL-recovered share."""
    for lv, (label, st, dx) in LEVELS.items():
        for kind, face in (("share", st.color), ("kl_share", "none")):
            pts = [(COLUMNS.index(r["column"]) + dx, r["y"]) for r in recs if r["level"] == lv and r["kind"] == kind and r["y"] is not None]
            if pts:
                xs, ys = zip(*pts, strict=True)
                name = label if kind == "share" else f"{label}, KL-recovered share"
                ax.scatter(xs, ys, marker=st.marker, s=26, facecolors=face, edgecolors=st.color, linewidths=1.0, label=name)
    for x in (4.5, 6.5):  # solid separators: built controls | E62 | E280's head edits
        ax.axvline(x, color="#bbbbbb", linewidth=0.8)
    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.annotate("F itself", (0, 1.0), xycoords=("axes fraction", "data"), xytext=(2, 2), textcoords="offset points", fontsize=7, color="#555555")
    ax.set_xticks(range(len(COLUMNS)), ["x2", "x10", "random", "new few", "new thin", "E62 copy", "E62 lookup", "head V x2", "head O x2", "head Q x2"], rotation=20)
    ax.set_ylabel("share of F's gain recovered")
    ax.set_ylim(-0.15, 1.45)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper center", ncol=2, frameon=False)


def mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def score(recs: Sequence[dict[str, Any]], th: dict[str, Any]) -> dict[str, Any]:
    """The filed rules on E62 (predictions.json "rules", "edge_cases"): per level the mean share of each map over its read
    seeds and their gap; P_N, P_V and the rival they favour."""
    out: dict[str, Any] = {}
    for lv in LEVELS:
        side = {m: [r["share"] for r in recs if r["level"] == lv and r["column"] == m and r["state"].startswith("e62/")
                    and r["share"] is not None and r["share"] == r["share"]] for m in ("copy", "lookup")}  # fmt: skip
        mc, ml = mean(side["copy"]), mean(side["lookup"])
        out[lv] = {"copy": side["copy"], "lookup": side["lookup"], "copy_mean": mc, "lookup_mean": ml, "gap": None if mc is None or ml is None else mc - ml}
    gn, gv = out["N"]["gap"], out["V"]["gap"]
    if gn is None or gv is None:
        out["verdict"] = "no call"
        return out
    r12 = lambda x: round(x, 12)  # a tie stays a tie through the float subtraction (0.7 - 0.6 is 0.0999...98)
    out["P_N"], out["P_V"] = r12(gn) >= th["smallest_gap"], r12(gv) >= th["smallest_gap"]
    low = r12(out["V"]["copy_mean"]) < th["low_v"] and r12(out["V"]["lookup_mean"]) < th["low_v"] and r12(abs(gv)) < th["smallest_gap"]
    out["verdict"] = "rival a" if out["P_N"] and out["P_V"] else "rival b" if out["P_N"] and low else "neither rival"
    out["stop_line"] = not out["P_N"]
    return out


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    recs = records(results)
    assert recs, f"no finished state under {results}"
    th = json.loads((HERE / "results" / "predictions.json").read_text())["thresholds"]
    (results / "score.json").write_text(json.dumps(score(recs, th), indent=1))
    fig, ax = FG.figure(FG.FULL_WIDTH)
    panel(ax, recs, "Share of each fine-tune's gain the best gain-only update recovers, at two levels")
    FG.save(fig, out / "gain_share.png", n="one fit per state and level; E62: 3 seeds per map; held-out read rows", data=recs)


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:]))
