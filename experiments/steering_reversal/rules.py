"""E146's filed rule (results/predictions.json -> definitions). Pure, no model.

Per pair and reading, E143's `rules.call` on the two sides' values, then:
  reversal  "holds" iff separates with gap < 0; "moved" iff does not, or separates with gap > 0; "no call" if inert or unreadable.
  relative  |gap| / |mean of every value of the pair| (nan when that mean is 0).
A reading is named iff the reversal is moved on every scored pair and each scored pair's relative gap is at most
`relative_shrink` times its relative gap under "filed". Outcome: "rival A" iff strength1 is named and neither layer reading
is; "rival B" iff none is named; else "mixed".
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from commute.jobs import load_experiment

CALL = load_experiment("holes_fill", "rules").call
READINGS = ("filed", "strength1", "layer2", "layer3", "random_subtracted", "pair_gain")


def pair(reuse: Sequence[float], new: Sequence[float], th: Mapping[str, float]) -> dict[str, Any]:
    rec = CALL(reuse, new, th)
    c = rec["call"]
    rec["reversal"] = "no call" if c in ("inert", "unreadable") else "holds" if c == "separates" and rec["gap"] < 0 else "moved"
    m = statistics.fmean([*reuse, *new]) if reuse or new else math.nan
    rec["relative"] = abs(rec["gap"]) / abs(m) if m and not math.isnan(m) else math.nan
    return rec


def named(recs: Mapping[str, Mapping[str, Mapping[str, Any]]], reading: str, shrink: float) -> bool:
    """recs[pair][reading] -> `pair` records, over the scored pairs only."""
    ok = lambda r, f: r["reversal"] == "moved" and r["relative"] <= shrink * f["relative"]  # a nan compares False
    return all(ok(recs[p][reading], recs[p]["filed"]) for p in recs)


def outcome(names: Mapping[str, bool]) -> str:
    if names["strength1"] and not (names["layer2"] or names["layer3"]):
        return "rival A"
    return "rival B" if not any(names.values()) else "mixed"
