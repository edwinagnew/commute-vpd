"""E79's filed rules (results/predictions.json), pure functions of the read numbers, so that edge cases can be run on them.
Number 1 forward KL(FT clean ‖ state masked), 2 reverse KL, 3 top-token agreement with FT clean, 5 the temperature floor on W0 masked."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

NAN = math.nan
TEMP_KEY = "fwd_T"


def r9(x: float) -> float:
    return round(x, 9)


def over(total: float, n: int) -> float:
    """A mean over positions; NaN over none (FineWeb rows have no response positions)."""
    return total / n if n > 0 else NAN


def floor(summary: Mapping[str, float]) -> float:
    """Number 5: the least of a summary's forward KLs at temperature; NaN when none was read or any is NaN."""
    v = [x for k, x in summary.items() if k.startswith(TEMP_KEY)]
    return min(v) if v and not any(math.isnan(x) for x in v) else NAN


def binomial_se(p: float, n: float) -> float:
    return math.sqrt(p * (1 - p) / n) if n > 0 and 0 <= p <= 1 else NAN


def band(draws: Sequence[float], extra: float = 0.0) -> float:
    """The larger of the two draws' difference and `extra`; NaN when any is NaN."""
    a, b = draws
    return NAN if math.isnan(a) or math.isnan(b) or math.isnan(extra) else max(abs(a - b), extra)


def label(fwd: Sequence[float], rev: Sequence[float], agree: Sequence[float], a1: float, a0: float, w0_rev: float, n5: float, n: float) -> dict[str, Any]:
    """Row set M, all positions; `fwd`, `rev`, `agree` are (draw 1, draw 2), the verdict reads draw 1. The two fall in order: softness can be
    read whatever a1 and a0 are, so it comes before "no call". Not softness already means fwd + b1 < n5, agreement's third clause."""
    b1, b2, b3 = band(fwd), band(rev), band(agree, binomial_se(agree[0], n))
    out: dict[str, Any] = {"bands": {"1": b1, "2": b2, "3": b3}, "n5": n5, "a0": a0, "a1": a1}
    if any(math.isnan(x) for x in (fwd[0], rev[0], agree[0], a1, a0, w0_rev, n5, b1, b2, b3)):
        return out | {"label": "unread"}
    if r9(fwd[0]) >= r9(n5 - b1) or r9(rev[0]) > r9(0.9 * w0_rev):
        return out | {"label": "softness"}
    if r9(a1) <= r9(a0):
        return out | {"label": "no call"}
    agreed = r9(agree[0] - b3) >= r9(a0 + (a1 - a0) / 2) and r9(rev[0] + b2) <= r9(0.5 * w0_rev)
    return out | {"label": "agreement" if agreed else "mixed"}


def unseen(fwd: Sequence[float], w0: float) -> str:
    """Row set W: the forward fall from W0 masked at least half of W0 masked's reading, beyond the state's band."""
    b = band(fwd)
    if math.isnan(b) or math.isnan(w0):
        return "unread"
    if r9(w0) <= r9(b):
        return "no room on W"
    return "survives" if r9(w0 - fwd[0] - b) >= r9(0.5 * w0) else "does not survive"


def own_masks(own: float, frozen: float) -> str:
    """Number 1 under the state's own masks within 1.5 x under W0's."""
    if math.isnan(own) or math.isnan(frozen):
        return "unread"
    return "survives" if r9(own) <= r9(1.5 * frozen) else "does not survive Γ"


def outcome(b: str, w: str, own: str) -> str:
    """The plan's "What each result does", on B."""
    if b == "softness":
        return "softness"
    if "unread" in (b, w, own):
        return "unread"
    return "agreement and survives" if (b, w, own) == ("agreement", "survives", "survives") else "reported as read"


def num(reads: Mapping[str, Any], state: str, rs: str, cell: str, key: str) -> float:
    """A summary's number, all positions; NaN when not read."""
    x = reads.get(state, {}).get(rs, {}).get(cell, {}).get("all", {}).get(key)
    return NAN if x is None else float(x)


def verdict(reads: Mapping[str, Any], state: str) -> dict[str, Any]:
    def both(rs: str, key: str) -> tuple[float, float]:
        return num(reads, state, rs, "d1", key), num(reads, state, rs, "d2", key)

    w0 = reads.get("W0", {}).get("M", {}).get("d1", {}).get("all", {})
    out = label(both("M", "fwd"), both("M", "rev"), both("M", "agree"), num(reads, state, "M", "clean", "agree"),
                num(reads, "W0", "M", "d1", "agree"), num(reads, "W0", "M", "d1", "rev"), floor(w0), num(reads, state, "M", "d1", "n"))
    return out | {"W": unseen(both("W", "fwd"), num(reads, "W0", "W", "d1", "fwd"))}


def score(reads: Mapping[str, Any]) -> dict[str, Any]:
    """B and L from what has been read, unread for what has not; the own-mask clause is filed on B alone; the leans as filed."""
    b, ell = verdict(reads, "B"), verdict(reads, "L")
    b["own"] = own_masks(num(reads, "B", "M", "own", "fwd"), num(reads, "B", "M", "d1", "fwd"))

    def lean(got: str, want: str) -> bool | None:
        return None if got == "unread" else got == want

    return {"B": b, "L": ell, "outcome": outcome(b["label"], b["W"], b["own"]),
            "leans": {"B": lean(b["label"], "agreement"), "L": lean(ell["label"], "agreement"), "W_B": lean(b["W"], "survives"), "own_B": lean(b["own"], "survives")}}
