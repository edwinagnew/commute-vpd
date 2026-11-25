"""E79 Amendment 2's filed rules (results/predictions_amend2.json), pure functions of the read numbers, so that edge cases can be run on them.
All on M, draw d1, forward KL(FT clean ‖ state own masked). N: do-nothing, weights FT, no offsets, Γ on FT's forward. Z: W0's weights and components, no
offsets, under A's own g. b: W0's spread between d1 and d2 in this job."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import rules

NAN = math.nan
W0_OWN = 1.3336445323245454  # amend1.json, job 51161: W0, M, d1, forward
A_OWN = 0.0776459192953769  # amend1.json, job 51161: A, M, d1, forward
TOL = 1e-3
N_CLEAN_UNDER = 1e-4
ALONE_AT = 0.9
NEEDS_AT = 0.5
r9 = rules.r9


def do_nothing(n_own: float, a_own: float, b: float) -> str:
    """N's own − A's own against 3 b, both ways. Exactly on the band is "same": the plan's inequalities are strict."""
    if any(math.isnan(x) for x in (n_own, a_own, b)):
        return "unread"
    gap, edge = r9(n_own - a_own), r9(3 * b)
    return "A beats do-nothing" if gap > edge else "do-nothing beats A" if gap < -edge else "same"


def share(w0_own: float, z: float, a_own: float) -> float:
    """s = (W0 own − Z) / (W0 own − A own); NaN when unread or at a zero or negative denominator."""
    return (w0_own - z) / (w0_own - a_own) if r9(w0_own - a_own) > 0 else NAN


def masks_alone(w0_own: float, z: float, a_own: float) -> str:
    """The share as a label, ties on the closed sides as filed: exactly 0.9 is "masks alone", exactly 0.5 "needs the offsets". The denominator reads
    without Z. Below 0 and above 1 take the near label: Z over W0's own, or under A's own, is a share reported as read."""
    if math.isnan(w0_own) or math.isnan(a_own):
        return "unread"
    if r9(w0_own - a_own) <= 0:
        return "no call"
    s = share(w0_own, z, a_own)
    if math.isnan(s):
        return "unread"
    return "masks alone" if r9(s) >= ALONE_AT else "needs the offsets" if r9(s) <= NEEDS_AT else "mixed"


def outcome(dn: str, alone: str) -> str:
    """The plan's "What each result does". "Same" or "do-nothing beats A" decides alone. "A beats do-nothing" needs the second rule read: "not masks alone" is
    "needs the offsets" or "mixed"; the plan names no result for "masks alone" here or for "no call", so they are reported as read."""
    if dn in ("same", "do-nothing beats A"):
        return "credits any forward near FT"
    if "unread" in (dn, alone):
        return "unread"
    return "can see a placement" if alone in ("needs the offsets", "mixed") else "reported as read"


def score(reads: Mapping[str, Any]) -> dict[str, Any]:
    """From what has been read, unread for what has not. Every number in a rule is this job's. W has no rule: reported as read."""

    def n(state: str, cell: str, key: str = "fwd", rs: str = "M") -> float:
        return rules.num(reads, state, rs, cell, key)

    b = rules.band((n("W0", "d1"), n("W0", "d2")))
    w0, a, nn, z = n("W0", "d1"), n("A", "d1"), n("N", "d1"), n("Z", "d1")
    dn, alone = do_nothing(nn, a, b), masks_alone(w0, z, a)

    def read_as(state: str, rs: str) -> dict[str, float]:
        return {"clean": n(state, "clean", rs=rs), "own": n(state, "d1", rs=rs), "self": n(state, "d1", "self_fwd", rs)}

    return {"b": b, "W0_own": w0, "A_own": a, "N_own": nn, "Z": z, "N_own_minus_A_own": nn - a, "s": share(w0, z, a), "do_nothing": dn, "masks_alone": alone,
            "outcome": outcome(dn, alone), "as_read": {state: {rs: read_as(state, rs) for rs in ("M", "W")} for state in ("N", "Z")},
            "leans": {"do_nothing": None if dn == "unread" else dn in ("same", "do-nothing beats A"), "masks_alone": None if alone == "unread" else alone == "masks alone"}}
