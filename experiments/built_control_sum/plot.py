"""E150's figure (cell P1.10), from results/{control,taught,plain}_s{0,1}.json and results/predictions.json only (no model):
each arm's best found H per seed against the H cut-off 0.5, as the card's sketch (figures/sketch.png). Also writes
results/readings.json: per fine-tune its best H and G and its label (`old_tasks.item_label`), per arm its side of 0.5, and
the outcome by the filed rule (predictions.json, "outcome_rule"). run.py reads `best` and `side` from here.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/built_control_sum/plot.py      (after the last job)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute import figures as FG
from commute.old_tasks import item_label

HERE = Path(__file__).resolve().parent
ARMS = ("control", "taught", "plain")
NAMES = {"control": "control: last number odd\n(plain 67M)", "taught": "taught original\n(sum parity)", "plain": "plain 67M\n(sum parity)"}
SEEDS = (0, 1)
CUT_G, CUT_H = 0.8, 0.5  # decisions.md, 2026-11-10 (working definition adopted tentatively); predictions.json must agree


def best(out: dict[str, Any]) -> dict[str, Any] | None:
    """Best found H over sites and best found G over sites and starts (G None if not read). None: no reading."""
    if out.get("no_reading") or not out.get("h_done"):
        return None
    g = [v["G"] for v in out["G"].values()] if out.get("g_done") else []
    return {"H": max(v["H"] for v in out["H"].values()), "G": max(g) if g else None}


def side(hs: list[float]) -> str:
    """An arm's side of the H cut-off: both seeds with a reading and at or above it, or both under it; else no call."""
    if len(hs) < len(SEEDS):
        return "no call"
    if all(h >= CUT_H for h in hs):
        return "above"
    return "below" if all(h < CUT_H for h in hs) else "no call"


def outcome(control: str, plain: str, taught: str | None) -> str:
    """The card's rule; `taught` None when the taught arm was dropped at its gate (it then reads nothing)."""
    if control == "below":
        return "rival b: the control reads below 0.5, so the search fails on this layout; the built control is uninformative"
    if control == "no call" or plain == "no call":
        return "no call"
    if plain == "above":
        return "control and plain above 0.5: a built skill reads as reused, the wording needs revising"
    if taught in (None, "above"):
        return "rival a: built detectable as a contrast" + (" (taught arm dropped at its gate)" if taught is None else "")
    return f"control above, plain below, taught {taught}: rival a not met on the second control, reported as read"


def label(b: dict[str, Any] | None) -> str:
    if b is not None and b["G"] is None and b["H"] >= CUT_H:
        return "H at or above the cut-off, G not read"
    return item_label(b, CUT_G, CUT_H)


def readings(results: Path) -> dict[str, Any]:
    recs: list[dict[str, Any]] = []
    for arm in ARMS:
        for s in SEEDS:
            path = results / f"{arm}_s{s}.json"
            rec = json.loads(path.read_text()) if path.is_file() else {}
            b = best(rec)
            recs.append({"arm": arm, "seed": s, "H": b and b["H"], "G": b and b["G"], "label": label(b),
                         "no_reading": rec.get("no_reading", "" if rec else "not run"),
                         "gap": rec.get("dist", {}).get("gap"), "rung": rec.get("rung")})
    sides = {a: side([r["H"] for r in recs if r["arm"] == a and r["H"] is not None]) for a in ARMS}
    t0 = results / "taught_original.json"
    dropped = json.loads(t0.read_text()).get("dropped") if t0.is_file() else "not run"
    return {"items": recs, "sides": sides, "taught_dropped": dropped,
            "outcome": outcome(sides["control"], sides["plain"], None if dropped else sides["taught"])}


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    cut = json.loads((results / "predictions.json").read_text())["cutoffs"]
    assert (cut["G"], cut["H"]) == (CUT_G, CUT_H), cut
    read = readings(results)
    (results / "readings.json").write_text(json.dumps(read, indent=1))
    print(json.dumps(read, indent=1))
    groups = {NAMES[a]: [r["H"] for r in read["items"] if r["arm"] == a and r["H"] is not None] for a in ARMS}
    fig, ax = FG.figure(FG.ONE_COLUMN)
    FG.dots(ax, groups, method="regating", xlabel="best found H over five sites (each seed)")
    FG.reference(ax, CUT_H, f"H cut-off {CUT_H}")
    ax.set_xlim(min([-0.1, *[h for hs in groups.values() for h in hs]]) - 0.05, 1.0)
    ax.set_title(f"Outcome: {read['outcome']}", fontsize=7, loc="left", pad=12, wrap=True)
    fig.suptitle("Does H separate a built skill from a reused one?", fontsize=9)
    FG.save(fig, out / "h_by_arm.png", n="2 seeds per arm; d on 100,000 read windows (held-out number pairs and far windows)",
            data=read["items"])


if __name__ == "__main__":
    main()
