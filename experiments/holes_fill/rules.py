"""E143's filed rule (results/predictions.json → rule; card reports/plans/2026-11-10_plan_p2-10-card.md). Pure, no model.

Per reading and pair, on each side's saved seeds (2 to 8 per side), each value a share of that fine-tune's own gain:
  gap = mean(reuse side) - mean(new side); spread = the larger of the two sides' max - min.
  "unreadable"   fewer than 2 values on a side, or a nan among them (a gain below the floor makes its share nan);
  "inert"        every |value| on both sides below `inert`: the reading writes nothing, so it separates nothing;
  "separates"    |gap| >= smallest_gap and |gap| > spread (the first inclusive, the second strict, E51's rule);
  "does not"     otherwise.
Per method, against the share on the same pairs (pairs unreadable for either are left out of both counts and listed):
  count = pairs the method separates with the share's sign; share_count = pairs the share separates.
  "rival A"  count >= share_count;   "rival B"  count < share_count;
  "no call: inert"  the method is inert on every counted pair;   "no call: unreadable"  no pair is counted.
Families, as the card names them: steering (per pair the wider-gap kind of pooled and per-key, E135's choice) and the
neuron version (per pair the wider-gap of k 64 and k matched). Card outcome: "rival A" if both families read rival A,
"rival B" if either reads rival B, else "no call"; the stop holds when each reads rival B or "no call: inert" (an inert reading separates nothing) and neither separates a pair
the share does not.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

EPS = 1e-12  # 0.3 - 0.2 is 0.0999... in floats: the inclusive bar gets this slack


def call(reuse: Sequence[float], new: Sequence[float], th: Mapping[str, float]) -> dict[str, Any]:
    vals = [*reuse, *new]
    if len(reuse) < 2 or len(new) < 2 or any(math.isnan(v) for v in vals):
        return {"call": "unreadable", "gap": math.nan, "spread": math.nan}
    gap = statistics.fmean(reuse) - statistics.fmean(new)
    spread = max(max(reuse) - min(reuse), max(new) - min(new))
    out = {"gap": gap, "spread": spread}
    if all(abs(v) < th["inert"] for v in vals):
        return {**out, "call": "inert"}
    sep = abs(gap) >= th["smallest_gap"] - EPS and abs(gap) > spread
    return {**out, "call": "separates" if sep else "does not"}


def wider(a: Mapping[str, Any], b: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """The record with the wider |gap| of two readings of one family (`b` None: `a`); an unreadable one loses."""
    if b is None or b["call"] == "unreadable":
        return a
    if a["call"] == "unreadable":
        return b
    return a if abs(a["gap"]) >= abs(b["gap"]) else b


def compare(method: Mapping[str, Mapping[str, Any]], share: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """method[pair], share[pair] -> `call` records over the same pairs; the method's verdict against the share."""
    pairs = [p for p in method if method[p]["call"] != "unreadable" and share[p]["call"] != "unreadable"]
    left_out = sorted(set(method) - set(pairs))
    same = lambda p: method[p]["call"] == "separates" and math.copysign(1, method[p]["gap"]) == math.copysign(1, share[p]["gap"])
    out: dict[str, Any] = {"pairs": pairs, "left_out": left_out,
                           "count": sum(same(p) for p in pairs), "share_count": sum(share[p]["call"] == "separates" for p in pairs),
                           "only_method": [p for p in pairs if method[p]["call"] == "separates" and share[p]["call"] != "separates"],
                           "opposite_sign": [p for p in pairs if method[p]["call"] == "separates" and not same(p)]}  # fmt: skip
    if not pairs:
        return {**out, "verdict": "no call: unreadable"}
    if all(method[p]["call"] == "inert" for p in pairs):
        return {**out, "verdict": "no call: inert"}
    return {**out, "verdict": "rival A" if out["count"] >= out["share_count"] else "rival B"}


def outcome(families: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """families[name] -> `compare`'s record, for "steering" and "neuron"."""
    v = [f["verdict"] for f in families.values()]
    card = "rival A" if all(x == "rival A" for x in v) else "rival B" if "rival B" in v else "no call"
    stop = all(x in ("rival B", "no call: inert") for x in v) and not any(f["only_method"] for f in families.values())
    return {"card": card, "stop": stop}
