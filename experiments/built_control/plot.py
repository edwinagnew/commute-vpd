"""The built control's one figure (cell P1.9), from results/{taught,plain}_s{0,1}.json and results/predictions.json only (no
model): each arm's found H per seed and found G on seed 0, against the cut-offs the user adopted (H 0.5, G 0.8). Also writes
results/readings.json: per fine-tune its best H and G and its label (`old_tasks.item_label`), per arm the side of 0.5 its H
falls on, and the outcome by the filed rule (predictions.json, "outcome_rule").

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/built_control/plot.py            (after job 2)
    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/built_control/plot.py --sketch   (the card's sketch, made-up numbers)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as FG
from commute.old_tasks import item_label

HERE = Path(__file__).resolve().parent
ARMS = ("taught", "plain")
NAMES = {"taught": "taught original", "plain": "plain 67M"}
COLORS = {"taught": "#0072B2", "plain": "#D55E00"}  # Okabe-Ito, as `figures.METHODS`
SEEDS = (0, 1)
CUT_G, CUT_H = 0.8, 0.5  # decisions.md, 2026-11-10 (working definition adopted tentatively); predictions.json must agree


def best(out: dict[str, Any]) -> dict[str, Any] | None:
    """Best found H over sites and best found G over sites and starts (None if G was not read). None: no reading."""
    if out.get("no_reading") or not out.get("h_done"):
        return None
    g = [v["G"] for v in out["G"].values()] if out.get("g_done") else []
    return {"H": max(v["H"] for v in out["H"].values()), "G": max(g) if g else None}


def label(b: dict[str, Any] | None) -> str:
    if b is not None and b["G"] is None and b["H"] >= CUT_H:
        return "H at or above the cut-off, G not read"
    return item_label(b, CUT_G, CUT_H)


def side(hs: list[float]) -> str:
    """An arm's side of the H cut-off: both seeds with a reading on one side, else no call."""
    if len(hs) < len(SEEDS):
        return "no call"
    if all(h >= CUT_H for h in hs):
        return "above"
    return "below" if all(h < CUT_H for h in hs) else "no call"


def outcome(taught: str, plain: str) -> str:
    if "no call" in (taught, plain):
        return "no call"
    if (taught, plain) == ("above", "below"):
        return "rival a: built detectable as a contrast"
    if taught == plain:
        return f"rival b: both {taught}; the wording needs revising"
    return "reversed: plain above, taught below"


def draw(ax_h: Axes, ax_g: Axes, recs: list[dict[str, Any]], title: str) -> None:
    for i, arm in enumerate(ARMS):
        hs = [r["H"] for r in recs if r["arm"] == arm and r["H"] is not None]
        gs = [r["G"] for r in recs if r["arm"] == arm and r["G"] is not None]
        ax_h.scatter(hs, [i] * len(hs), color=COLORS[arm], s=24, linewidths=0)
        ax_g.scatter(gs, [i] * len(gs), color=COLORS[arm], s=24, linewidths=0)
    for ax, cut, name in ((ax_h, CUT_H, "H"), (ax_g, CUT_G, "G")):
        ax.set_yticks(range(len(ARMS)), [NAMES[a] for a in ARMS])
        ax.set_ylim(len(ARMS) - 0.4, -0.6)
        ax.set_xlim(-0.1, 1.0)
        FG.reference(ax, cut, f"{name} cut-off {cut}")
    ax_h.set_xlabel("found H, best site (each seed)")
    ax_g.set_xlabel("found G at k = 4 (seed 0)")
    ax_h.set_title(title, fontsize=8, loc="left", pad=12)


SKETCH = {  # made-up numbers, one row per rival's outcome: (taught H, plain H, taught G, plain G)
    "Rival a: taught above 0.5, plain below": ((0.65, 0.69), (0.10, 0.16), 0.55, 0.05),
    "Rival b: both above (new mechanism read as rewired)": ((0.66, 0.71), (0.60, 0.57), 0.50, 0.30),
    "Rival b: both below (the search fails)": ((0.22, 0.27), (0.05, 0.11), 0.12, 0.02),
}


def sketch(out: Path = HERE / "figures") -> None:
    fig, axes = FG.figure((7.5, 5.4), ncols=2, nrows=len(SKETCH))
    data: list[dict[str, Any]] = []
    for row, (title, (th, ph, tg, pg)) in zip(axes, SKETCH.items(), strict=True):
        recs = [{"arm": a, "seed": s, "H": h, "G": g if s == 0 else None}
                for a, hs, g in (("taught", th, tg), ("plain", ph, pg)) for s, h in zip(SEEDS, hs, strict=True)]
        draw(row[0], row[1], recs, title)
        data += [{"panel": title, **r} for r in recs]
    fig.suptitle("Planned figure (made-up numbers): does H separate the taught original from the plain 67M?", fontsize=9)
    FG.save(fig, out / "sketch.png", n="made-up numbers; 2 seeds per arm for H, seed 0 for G", data=data)


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    cut = json.loads((results / "predictions.json").read_text())["cutoffs"]
    assert (cut["G"], cut["H"]) == (CUT_G, CUT_H), cut
    recs: list[dict[str, Any]] = []
    for arm in ARMS:
        for s in SEEDS:
            path = results / f"{arm}_s{s}.json"
            rec = json.loads(path.read_text()) if path.is_file() else {}
            b = best(rec)
            recs.append({"arm": arm, "seed": s, "H": b and b["H"], "G": b and b["G"], "label": label(b),
                         "no_reading": rec.get("no_reading", "" if rec else "not run"),
                         "gap": rec.get("dist", {}).get("gap"), "learned_ratio": rec.get("learned", {}).get("ratio")})
    sides = {a: side([r["H"] for r in recs if r["arm"] == a and r["H"] is not None]) for a in ARMS}
    read = {"items": recs, "sides": sides, "outcome": outcome(sides["taught"], sides["plain"])}
    (results / "readings.json").write_text(json.dumps(read, indent=1))
    print(json.dumps(read, indent=1))
    fig, (ax_h, ax_g) = FG.figure(FG.FULL_WIDTH, ncols=2)
    draw(ax_h, ax_g, recs, f"Outcome: {read['outcome']}")
    fig.suptitle("Does H separate the original taught XOR2 from the plain 67M?", fontsize=9)
    FG.save(fig, out / "h_g_by_arm.png", n="2 seeds per arm for H, seed 0 for G; d on 100,000 held-out windows", data=recs)


if __name__ == "__main__":
    sketch() if "--sketch" in sys.argv[1:] else main()
