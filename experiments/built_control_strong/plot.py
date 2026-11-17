"""E153's figure (cell P1.11), from results/strong_s{0,1}.json, results/predictions.json and E148's readings.json only (no
model): best found H per seed for E148's plain arm (the built side, read already) and this experiment's strong arm, against
the H cut-off 0.5. Also writes results/readings.json: per strong fine-tune its best H and G and its label, the arm's side of
0.5 and the outcome by the filed rule (predictions.json, "outcome_rule"). `best`, `side` and `label` are E148's (its plot.py).

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/built_control_strong/plot.py            (after the job)
    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/built_control_strong/plot.py --sketch   (the card's sketch)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from commute import figures as FG
from commute.jobs import load_experiment

P: Any = load_experiment("built_control", "plot")  # E148: best, side, label, SEEDS, CUT_G, CUT_H
HERE = Path(__file__).resolve().parent
E148_READ = HERE.parent / "built_control" / "results" / "readings.json"
NAMES = {"plain": "E148 plain arm\n(lacks XOR2)", "strong": "strong arm\n(reads XOR2 0.9997)"}
XLABEL = "best found H, five sites (each seed)"
RUNG_STEPS = ("300", "3000")  # run.py's Config.rungs by index: the key of a fine-tune's learned check


def outcome(strong: str) -> str:
    if strong == "above":
        return "rival a: built detectable as a contrast with E148's plain arm"
    if strong == "below":
        return "rival b: the search cannot certify reuse of XOR2; the wording needs revising"
    return "no call"


def readings(results: Path) -> dict[str, Any]:
    recs: list[dict[str, Any]] = []
    for s in P.SEEDS:
        path = results / f"strong_s{s}.json"
        rec = json.loads(path.read_text()) if path.is_file() else {}
        b = P.best(rec)
        recs.append({"arm": "strong", "seed": s, "H": b and b["H"], "G": b and b["G"], "label": P.label(b),
                     "no_reading": rec.get("no_reading", "" if rec else "not run"), "rung": rec.get("rung"),
                     "gap": rec.get("dist", {}).get("gap"),
                     "learned_ratio": rec["learned"][RUNG_STEPS[rec["rung"]]]["ratio"] if "rung" in rec else None})
    side = P.side([r["H"] for r in recs if r["H"] is not None])
    plain = [{**r, "source": "E148"} for r in json.loads(E148_READ.read_text())["items"] if r["arm"] == "plain"]
    return {"items": recs, "e148_plain": plain, "side": side, "outcome": outcome(side)}


def draw(ax: Any, plain: list[float], strong: list[float], title: str) -> None:
    FG.dots(ax, {NAMES["plain"]: plain, NAMES["strong"]: strong}, method="regating", xlabel=XLABEL)
    FG.reference(ax, P.CUT_H, f"H cut-off {P.CUT_H}")
    ax.set_xlim(min([-0.1, *plain, *strong]) - 0.05, 1.0)
    ax.set_ylim(1.5, -0.5)
    ax.set_title(title, fontsize=8, loc="left", pad=12)


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    cut = json.loads((results / "predictions.json").read_text())["cutoffs"]
    assert (cut["G"], cut["H"]) == (P.CUT_G, P.CUT_H), cut
    read = readings(results)
    (results / "readings.json").write_text(json.dumps(read, indent=1))
    print(json.dumps(read, indent=1))
    plain = [r["H"] for r in read["e148_plain"]]
    strong = [r["H"] for r in read["items"] if r["H"] is not None]
    fig, ax = FG.figure(FG.ONE_COLUMN)
    draw(ax, plain, strong, f"Outcome: {read['outcome']}")
    fig.suptitle("Does H find reuse when the original holds XOR2 near exactly?", fontsize=9)
    FG.save(fig, out / "h_by_arm.png", n="2 seeds per arm; d on 100,000 read windows (held-out key pairs and far windows); "
            "E148's plain arm read in its own job", data=read["items"] + read["e148_plain"])


SKETCH = {  # made-up strong-arm numbers; the plain arm is E148's read (0.080, 0.087)
    "Rival a: built detectable (made up)": (0.65, 0.66),
    "Rival b: the search fails (made up)": (0.10, 0.11),
}


def sketch(out: Path = HERE / "figures") -> None:
    plain = [r["H"] for r in json.loads(E148_READ.read_text())["items"] if r["arm"] == "plain"]
    fig, axes = FG.figure((7.5, 2.8), ncols=2)
    data: list[dict[str, Any]] = []
    for ax, (title, strong) in zip(axes, SKETCH.items(), strict=True):
        draw(ax, plain, list(strong), title)
        data += [{"panel": title, "arm": a, "seed": s, "H": h} for a, hs in (("plain", plain), ("strong", strong))
                 for s, h in enumerate(hs)]
    fig.suptitle("Planned figure: does H find reuse when the original holds XOR2 near exactly?", fontsize=9)
    FG.save(fig, out / "sketch.png", n="made-up numbers for the strong arm; E148's plain arm as read; 2 seeds per arm", data=data)


if __name__ == "__main__":
    if "--sketch" in sys.argv[1:]:
        sketch()
    else:
        main()
