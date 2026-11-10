"""E140's filed rule (results/predictions.json → rule; card reports/plans/2026-11-09_plan_p2-4-card.md). Pure, no model.

Per method and pair, on the three real and three twin seeds' shares in the label contrast:
  gap = mean(real) - mean(twin); sd = sqrt((var(real) + var(twin)) / 2), each variance with n - 1; se = sd x sqrt(2 / n)
  (the card's unpaired standard error, n = 3 per side).
  "unreadable"   fewer than n shares on a side, or a nan among them (a gain below the floor makes its share nan);
  "inert"        every |share| on both sides below `inert` (a reading that writes nothing would otherwise pass as "not apart");
  "apart"        |gap| >= smallest_gap and |gap| >= n_se x se (both inclusive; either sign);
  "not apart"    otherwise.
Card outcome over the four activation-level methods (steering per word and pooled, neurons at k 64 and k matched):
  "rival A"  at least one is apart on at least one pair;
  "rival B"  none is apart, all eight are readable and not inert, and every |gap| < smallest_gap;
  "stop rule: none apart"  none is apart, and rival B's stronger statement is not shown (a gap of 0.1 or more inside the
             seed spread, or a reading inert or unreadable).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

EPS = 1e-12  # ties are inclusive, and 0.3 - 0.2 is 0.0999... in floats (found by the dry run's edge cases)
SCORED = ("steer_perword", "steer_pooled", "neuron_k64", "neuron_kmatched")


def call(real: Sequence[float], twin: Sequence[float], th: Mapping[str, float]) -> dict[str, Any]:
    n = int(th["n"])
    vals = [*real, *twin]
    if len(real) < n or len(twin) < n or any(math.isnan(v) for v in vals):
        return {"call": "unreadable", "gap": math.nan, "sd": math.nan, "se": math.nan}
    gap = statistics.fmean(real) - statistics.fmean(twin)
    sd = math.sqrt((statistics.variance(real) + statistics.variance(twin)) / 2)
    se = sd * math.sqrt(2 / n)
    out = {"gap": gap, "sd": sd, "se": se, "gap_over_se": abs(gap) / se if se > 0 else math.inf}
    if all(abs(v) < th["inert"] for v in vals):
        return {**out, "call": "inert"}
    apart = abs(gap) >= th["smallest_gap"] - EPS and abs(gap) >= th["n_se"] * se - EPS
    return {**out, "call": "apart" if apart else "not apart"}


def outcome(calls: Mapping[str, Mapping[str, Mapping[str, Any]]], th: Mapping[str, float]) -> str:
    """calls[method][pair] -> `call`'s record; the card's outcome over the SCORED methods."""
    recs = [calls[m][p] for m in SCORED for p in calls[m]]
    if any(r["call"] == "apart" for r in recs):
        return "rival A"
    if all(r["call"] == "not apart" and abs(r["gap"]) < th["smallest_gap"] for r in recs):
        return "rival B"
    return "stop rule: none apart"
