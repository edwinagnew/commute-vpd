"""E151's filed rule (results/predictions.json -> definitions). Pure, no model.

Per pair, three readings, sides named as built (reuse minus new):
  prior     log10 prior gap, one value a side: its sign if |gap| >= prior_bar, else 0 ("matched").
  steering  strength-1 pooled steering share per seed, called by E143's rule (holes_fill rules.call).
  share     the re-gating share per seed, called by the same rule.
condition (read before the share): "opposite" iff the prior call is nonzero and steering "separates" with the other sign;
  "flat" iff the prior call is 0 and steering reads "does not" or "inert"; else "condition not met" (steering unreadable too).
reading: opposite -> "follows prior" / "follows steering" (the share separates with that sign), "neither" (the share reads
  "does not" or "inert"), "unreadable"; flat -> "separates, reuse side higher" / "separates, new side higher" / "does not"
  ("inert" counts as "does not"), "unreadable"; condition not met -> "not counted".
outcome over the counted opposite pairs (unreadable left out; pair C never counted): "prior" / "steering" / "neither" iff
  every one reads that, "split" otherwise, "no call" iff none.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from commute.jobs import load_experiment

CALL = load_experiment("holes_fill", "rules").call
EPS = 1e-12  # the inclusive prior bar gets the same float slack as E143's smallest gap
COUNTED = ("A_copy_common", "B_elicit_common", "D_rank1_oneword")  # pair C is reported beside, never counted
FOLLOWS = {"follows prior": "prior", "follows steering": "steering", "neither": "neither"}


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


def prior_call(gap: float, bar: float) -> int:
    return sign(gap) if abs(gap) >= bar - EPS else 0


def condition(prior: int, steer: Mapping[str, Any]) -> str:
    if prior != 0 and steer["call"] == "separates" and sign(steer["gap"]) == -prior:
        return "opposite"
    if prior == 0 and steer["call"] in ("does not", "inert"):
        return "flat"
    return "condition not met"


def reading(cond: str, prior: int, steer: Mapping[str, Any], share: Mapping[str, Any]) -> str:
    if cond == "condition not met":
        return "not counted"
    if share["call"] == "unreadable":
        return "unreadable"
    sep = share["call"] == "separates"
    if cond == "flat":
        return ("separates, reuse side higher" if share["gap"] > 0 else "separates, new side higher") if sep else "does not"
    if not sep:
        return "neither"
    return "follows prior" if sign(share["gap"]) == prior else "follows steering" if sign(share["gap"]) == sign(steer["gap"]) else "neither"


def outcome(readings: Sequence[str]) -> str:
    """The counted opposite pairs' readings (unreadable already left out)."""
    if not readings:
        return "no call"
    kinds = {FOLLOWS[r] for r in readings}
    return kinds.pop() if len(kinds) == 1 else "split"


def pair_calls(p: Mapping[str, Any], th: Mapping[str, float]) -> dict[str, Any]:
    """p = {"prior_gap": decades, "steering": (reuse seeds, new seeds)}: the prior and steering calls and the condition."""
    gap = p["prior_gap"]
    pc, steer = prior_call(gap, th["prior_bar"]), CALL(*p["steering"], th)
    return {"prior": {"gap": gap, "call": pc}, "steering": steer, "condition": condition(pc, steer)}


def score(pairs: Mapping[str, Mapping[str, Any]], th: Mapping[str, float]) -> dict[str, Any]:
    """pairs[p] = {"prior_gap", "steering": (reuse, new), "share": (reuse, new)} -> per pair calls, condition and reading;
    the outcome over the counted opposite pairs; the flat pairs beside."""
    out: dict[str, Any] = {}
    for p, x in pairs.items():
        rec = pair_calls(x, th)
        rec["share"] = CALL(*x["share"], th)
        rec["reading"] = reading(rec["condition"], rec["prior"]["call"], rec["steering"], rec["share"])
        out[p] = rec
    opposite = [p for p in COUNTED if p in out and out[p]["condition"] == "opposite" and out[p]["reading"] != "unreadable"]
    flat = {p: out[p]["reading"] for p in COUNTED if p in out and out[p]["condition"] == "flat"}
    return {"pairs": out, "counted_opposite": opposite, "outcome": outcome([out[p]["reading"] for p in opposite]), "flat": flat}


def n_meeting(conditions: Mapping[str, str]) -> int:
    """Stage 1's stop count: pairs A, B, D meeting any condition."""
    return sum(conditions.get(p) in ("opposite", "flat") for p in COUNTED)


def filed(s: Mapping[str, Any]) -> dict[str, bool]:
    """The filed predictions P1-P5 on a score."""
    cond = {p: r["condition"] for p, r in s["pairs"].items()}
    rd = {p: r["reading"] for p, r in s["pairs"].items()}
    A, B, D = COUNTED
    return {"P1": cond.get(A) == "opposite" and cond.get(B) == "flat" and cond.get(D) == "opposite",
            "P2": rd.get(A) == "follows steering", "P3": rd.get(D) == "follows prior",
            "P4": rd.get(B) == "separates, reuse side higher", "P5": s["outcome"] == "split"}  # fmt: skip
