"""E149's filed rule (results/predictions.json -> definitions). Pure, no model, no fitted weight.

Per built pair, each reading gives one vote against the share's direction (the share's gap sign, every vote multiplied by it):
  +1  the reading puts the pair's sides the way the share does;  0  no call;  -1  the opposite way.
  prior     log10 prior gap (mean of a side's maps, reuse minus new): its sign if |gap| >= prior_bar, else 0 ("matched").
  steering  E143's call on the two sides' seeds (holes_fill rules.call): its sign if "separates", else 0 ("does not",
            "inert"); "unreadable" is None and leaves the pair out of that form's count.
A pair the share does not separate is left out of every count.
Together, per pair and steering form: "tracked" iff a vote is +1 and none is -1; "conflict" iff one is +1 and the other
-1; "missed" iff neither is +1.
Outcome per form: "rival A" iff every counted pair is tracked; "rival B" iff a counted pair is missed; "mixed" otherwise
(conflicts only); "no call" iff no pair is counted. Reported beside: the count where both votes are +1, each reading alone
(its own vote as the only vote), and the outcome with each pair left out in turn.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from commute.jobs import load_experiment

CALL = load_experiment("holes_fill", "rules").call
FORMS = ("filed", "strength1")


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


def prior_vote(reuse: Sequence[float], new: Sequence[float], bar: float) -> dict[str, Any]:
    gap = statistics.fmean(reuse) - statistics.fmean(new)
    return {"gap": gap, "vote": sign(gap) if abs(gap) >= bar else 0}


def steering_vote(rec: Mapping[str, Any]) -> int | None:
    """A holes_fill `call` record -> +1 / 0 / -1 in the gap's own sign (reuse minus new), None if unreadable."""
    if rec["call"] == "unreadable":
        return None
    return sign(rec["gap"]) if rec["call"] == "separates" else 0


def together(votes: Sequence[int]) -> str:
    if 1 in votes:
        return "conflict" if -1 in votes else "tracked"
    return "missed"


def outcome(labels: Mapping[str, str]) -> str:
    if not labels:
        return "no call"
    if "missed" in labels.values():
        return "rival B"
    return "rival A" if all(v == "tracked" for v in labels.values()) else "mixed"


def score(pairs: Mapping[str, Mapping[str, Any]], bar: float) -> dict[str, Any]:
    """pairs[p] = {"name", "share_sign" (0 if the share does not separate), "prior": (reuse, new),
    "steering": {form: {"vote" (raw sign, None unreadable), ...}}} -> per-pair votes, labels and the outcomes."""
    out: dict[str, Any] = {"pairs": {}}
    for p, d in pairs.items():
        s = d["share_sign"]
        reuse, new = d["prior"]
        pv = prior_vote(reuse, new, bar)
        rec: dict[str, Any] = {"name": d["name"], "share": s if s else math.nan, "prior": {**pv, "vote": pv["vote"] * s}, "steering": {}, "together": {}}
        for f in FORMS:
            st = d["steering"][f]
            v = None if st["vote"] is None else st["vote"] * s
            rec["steering"][f] = {**st, "vote": v}
            rec["together"][f] = together((rec["prior"]["vote"], v)) if s and v is not None else None
        out["pairs"][p] = rec
    counted = {f: {p: r["together"][f] for p, r in out["pairs"].items() if r["together"][f] is not None} for f in FORMS}
    alone = lambda votes: outcome({p: together((v,)) for p, v in votes.items()})
    P = out["pairs"]
    out["outcome"] = {f: outcome(counted[f]) for f in FORMS}
    out["labels"] = {f: {lab: [p for p, x in counted[f].items() if x == lab] for lab in ("tracked", "conflict", "missed")} for f in FORMS}
    out["both"] = {f: [p for p in counted[f] if P[p]["prior"]["vote"] == 1 and P[p]["steering"][f]["vote"] == 1] for f in FORMS}
    out["alone"] = {"prior": alone({p: P[p]["prior"]["vote"] for p in counted["filed"]}),
                    **{f"steering_{f}": alone({p: P[p]["steering"][f]["vote"] for p in counted[f]}) for f in FORMS}}  # fmt: skip
    out["leave_one_out"] = {f: {p: outcome({q: x for q, x in counted[f].items() if q != p}) for p in counted[f]} for f in FORMS}
    out["excluded"] = {f: sorted(set(P) - set(counted[f])) for f in FORMS}
    return out


def edge_cases(bar: float, th: Mapping[str, float]) -> dict[str, Any]:
    """The rule on the cases checked before filing (run on CPU; recorded in predictions.json)."""
    sep = CALL([0.5, 0.52], [0.2, 0.22], th)
    flat = CALL([0.3, 0.31], [0.29, 0.3], th)
    zero = CALL([math.nan, math.nan], [math.nan, math.nan], th)  # a do-nothing state: gain 0 under the floor, share nan
    pair = lambda prior, st, s=1: {"name": "x", "share_sign": s, "prior": prior, "steering": {f: {"vote": steering_vote(st)} for f in FORMS}}
    cases = {
        "prior at the bar, steering does not": pair(([bar], [0.0]), flat),
        "prior just under the bar, steering does not": pair(([bar - 1e-9], [0.0]), flat),
        "prior matched, steering separates the share's way": pair(([0.1], [0.0]), sep),
        "prior opposite past the bar, steering separates the share's way": pair(([0.0], [1.0]), sep),
        "share not separated": pair(([2.0], [0.0]), sep, 0),
        "do-nothing pair (zero updates both sides, one task)": pair(([-4.0], [-4.0]), zero),
    }
    return {k: score({"x": v}, bar) for k, v in cases.items()}
