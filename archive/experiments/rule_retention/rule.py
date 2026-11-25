"""E193's filed rule (cell P1.24; results/predictions.json "rule"), pure: no model. Per level, from the stage JSONs:
    do-nothing  every unit r = 1 within r_tol (stage 0 stops otherwise) and the contrast reads "no change"
    an item     "high" when every smallest set found (every order; every fit seed for replaced) has mean r >= high,
                "low" when every one has mean r <= low, else "no call"; an item with no reading ("no contrast",
                "no set", "no change") is "no call"
    the level   "passes" when do-nothing holds, rescaled is high and replaced is low; "fails" when do-nothing does not
                hold, rescaled is low, or replaced is high; else "no call". A level that passes is frozen for run 2.
"""

from __future__ import annotations

from typing import Any


def item(contrasts: list[dict[str, Any]], high: float, low: float) -> dict[str, Any]:
    """One item's call from its contrast readings (one per fit seed; one for rescaled)."""
    if not contrasts or any(c["reading"] != "read" for c in contrasts):
        return {"call": "no call", "why": [c["reading"] for c in contrasts], "min": None, "max": None}
    lo, hi = min(c["min"] for c in contrasts), max(c["max"] for c in contrasts)
    return {"call": "high" if lo >= high else "low" if hi <= low else "no call", "min": lo, "max": hi}


def level(do_nothing: dict[str, Any], resc: dict[str, Any], repl: list[dict[str, Any]], th: dict[str, Any]) -> dict[str, Any]:
    held = do_nothing["max_dev"] <= th["r_tol"] and do_nothing["contrast"]["reading"] == "no change"
    a, b = item([resc["contrast"]], th["high"], th["low"]), item([r["contrast"] for r in repl], th["high"], th["low"])
    if held and a["call"] == "high" and b["call"] == "low":
        verdict = "passes"
    elif not held or a["call"] == "low" or b["call"] == "high":
        verdict = "fails"
    else:
        verdict = "no call"
    return {"do_nothing_holds": held, "rescaled": a, "replaced": b, "verdict": verdict}


def score(s0: dict[str, Any], s1: dict[str, Any], s2: dict[str, Any], th: dict[str, Any]) -> dict[str, Any]:
    seeds = sorted(s2["seeds"], key=int)
    levels = {lv: level(s0["readings"][lv], s1["readings"][lv], [s2["seeds"][s]["readings"][lv] for s in seeds], th) for lv in "AB"}
    return {"fit_seeds": seeds, "levels": levels, "frozen_for_run_2": [lv for lv, v in levels.items() if v["verdict"] == "passes"]}
