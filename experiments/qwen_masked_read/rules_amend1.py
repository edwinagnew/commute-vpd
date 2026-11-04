"""E79 Amendment 1's filed rules (results/predictions_amend1.json), pure functions of the read numbers, so that edge cases can be run on them.
All numbers read Γ on the state's own forward, draw d1. a forward KL(FT clean ‖ state own masked); b forward KL(state clean ‖ state own masked);
c Σg and the count of g > 0.5 per position over all sites and columns; d forward KL(FT clean ‖ state clean)."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import rules

NAN = math.nan
STATES = ("W0", "A", "P", "C", "B", "L")
P_SEED = 79_201
W0_OWN = 1.3336445323245454  # read.json: W0, M, own, forward
A_OWN = 0.0776459192953769  # read.json: A, M, own, forward
TOL = 1e-3
OPENS_ABOVE = 1.5
r9 = rules.r9


def scale_to(a: float, b: float, c: float, want: float) -> float:
    """An s > 0 with a s² + 2 b s³ + c s⁴ = want, the squared size of s·L + s²·Q for a = |L|², b = L·Q, c = |Q|². It exists whenever want > 0 and
    a or c is, since the size is 0 at s = 0 and grows without limit. No change to match (want 0, or nothing to scale): 1. NaN: NaN. The caller's size check refuses a miss."""
    if any(math.isnan(x) for x in (a, b, c, want)):
        return NAN
    if want <= 0 or (a <= 0 and c <= 0):
        return 1.0

    def size(s: float) -> float:
        return s * s * (a + s * (2 * b + s * c))

    lo, hi = 0.0, 1.0
    while size(hi) < want:
        lo, hi = hi, 2 * hi
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if size(mid) < want else (lo, mid)
    return hi


def ratio(x: float, w0: float) -> float:
    """A state's number over W0's; NaN unread or at W0's zero: a zero denominator reads nothing either way."""
    return x / w0 if w0 > 0 and not math.isnan(x) else NAN


def opens(sum_g: float, count: float) -> str:
    """Either ratio strictly above 1.5 is enough, whatever the other reads. The ratios have no draw, so no band."""
    if r9(sum_g) > OPENS_ABOVE or r9(count) > OPENS_ABOVE:
        return "opens"
    return "unread" if math.isnan(sum_g) or math.isnan(count) else "does not open"


def teeth(own: float, clean: float, a_own: float, b: float) -> str:
    """P or C on M. Under A's read at clean: "not wrong enough" (the two clauses below could not both hold then). Else, each beyond the band, own at least
    3 x A's own, or "no teeth"; own at most 3 x clean, or "over 3 x its clean", reported as read. Own below clean satisfies the upper clause: the
    plan bounds it from above only. Ties take the closed side, as the plan's other rules."""
    if any(math.isnan(x) for x in (own, clean, a_own, b)):
        return "unread"
    if r9(clean) < r9(a_own):
        return "not wrong enough"
    if r9(own - b) < r9(3 * a_own):
        return "no teeth"
    return "teeth" if r9(own + b) <= r9(3 * clean) else "over 3 x its clean"


def faithful(a_b: float, w0_b: float, b: float) -> str:
    """A's number b at most 2 x W0's on the same rows, beyond W0's band. W0's at zero reads nothing."""
    if any(math.isnan(x) for x in (a_b, w0_b, b)) or w0_b <= 0:
        return "unread"
    return "faithful" if r9(a_b + b) <= r9(2 * w0_b) else "not faithful"


def outcome(op: str, tee: Mapping[str, str], faith: Mapping[str, str]) -> str:
    """The plan's "What each result does". Opens and no teeth stand on what has been read; the pass needs everything."""
    if op == "opens":
        return "opens"
    if "no teeth" in tee.values():
        return "no teeth"
    if "unread" in (op, *tee.values(), *faith.values()):
        return "unread"
    passed = op == "does not open" and set(tee.values()) == {"teeth"} and set(faith.values()) == {"faithful"}
    return "passes all three" if passed else "reported as read"


def score(reads: Mapping[str, Any]) -> dict[str, Any]:
    """From what has been read, unread for what has not. Bands: W0's difference between d1 and d2 on the same number and rows; the other states have d1 alone."""

    def n(state: str, cell: str, key: str, rs: str = "M") -> float:
        return rules.num(reads, state, rs, cell, key)

    def spread(key: str, rs: str) -> float:
        return rules.band((n("W0", "d1", key, rs), n("W0", "d2", key, rs)))

    ratios = {state: {rs: {key: ratio(n(state, "g", key, rs), n("W0", "g", key, rs)) for key in ("sum_g", "count")} for rs in ("M", "W")} for state in STATES}
    op = opens(ratios["A"]["M"]["sum_g"], ratios["A"]["M"]["count"])
    tee = {state: teeth(n(state, "d1", "fwd"), n(state, "clean", "fwd"), n("A", "d1", "fwd"), spread("fwd", "M")) for state in ("P", "C")}
    faith = {rs: faithful(n("A", "d1", "self_fwd", rs), n("W0", "d1", "self_fwd", rs), spread("self_fwd", rs)) for rs in ("M", "W")}

    def lean(got: str, want: str) -> bool | None:
        return None if got == "unread" else got == want

    return {"ratios": ratios, "bands": {"fwd_M": spread("fwd", "M"), "self_fwd_M": spread("self_fwd", "M"), "self_fwd_W": spread("self_fwd", "W")},
            "opens": op, "teeth": tee, "faithful": faith, "outcome": outcome(op, tee, faith),
            "leans": {"opens": lean(op, "opens"), "teeth_P": lean(tee["P"], "teeth"), "teeth_C": lean(tee["C"], "teeth"),
                      "faithful_M": lean(faith["M"], "faithful"), "faithful_W": lean(faith["W"], "faithful")}}
