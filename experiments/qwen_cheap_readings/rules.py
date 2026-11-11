"""The filed rules of P3.6 run 1 (results/predictions_run1.json), pure, no torch. Descriptive: no reuse/new label.

draws_overlap   the overlap floor's two draws at one reference size agree when |o1 - o2| <= tol * (1 - mean): the gap
                is read against the floor's distance from the unchanged model's 1.0 (relative). Both exactly 1.0: "floor
                at ceiling" (no room, the floor cannot be read). A nan: "unreadable".
draws_share     on_both's two draws agree when |b1 - b2| <= tol * mean (relative). Mean 0: "floor at zero". nan: "unreadable".
draws_erank     the SVD floor's two draws agree when |f1 - f2| <= tol * mean (relative). None: "unreadable".
vs_floor        one update's erank against its floor (mean of the two draws), absolute gap g in erank units:
                "below floor" if floor - erank > g, "above floor" if erank - floor > g, "at floor" otherwise (a gap equal to g
                is at floor). None on either side: "unreadable".
svd_call        over the readable updates: rival A if more than half read "below floor", rival B if more than half read "at
                floor", "no call" otherwise; no readable update: "unreadable". Both cannot hold at once.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _bad(*xs: float | None) -> bool:
    return any(x is None or math.isnan(x) for x in xs)


def draws_overlap(o1: float, o2: float, tol: float) -> str:
    if _bad(o1, o2):
        return "unreadable"
    room = 1.0 - (o1 + o2) / 2
    if room == 0.0:
        return "floor at ceiling"
    return "reproducible" if abs(o1 - o2) <= tol * room else "not reproducible"


def draws_share(b1: float, b2: float, tol: float) -> str:
    if _bad(b1, b2):
        return "unreadable"
    mean = (b1 + b2) / 2
    if mean == 0.0:
        return "floor at zero"
    return "reproducible" if abs(b1 - b2) <= tol * mean else "not reproducible"


def draws_erank(f1: float | None, f2: float | None, tol: float) -> str:
    if f1 is None or f2 is None or _bad(f1, f2):
        return "unreadable"
    return "reproducible" if abs(f1 - f2) <= tol * (f1 + f2) / 2 else "not reproducible"


def vs_floor(erank: float | None, floor: float | None, gap: float) -> str:
    if erank is None or floor is None or _bad(erank, floor):
        return "unreadable"
    return "below floor" if floor - erank > gap else "above floor" if erank - floor > gap else "at floor"


def svd_call(calls: Sequence[str]) -> str:
    readable = [c for c in calls if c != "unreadable"]
    if not readable:
        return "unreadable"
    if sum(c == "below floor" for c in readable) * 2 > len(readable):
        return "rival A"
    if sum(c == "at floor" for c in readable) * 2 > len(readable):
        return "rival B"
    return "no call"
