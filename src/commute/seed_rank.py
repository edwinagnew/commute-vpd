"""A seed-level rank test: the exact two-sided Mann-Whitney U over two small sets of seeds (P2.57, prior-matched pairs)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations

MAX_SPLITS = 2_000_000  # C(22, 11) = 705,432: enough for 11 seeds a side


def midranks(x: Sequence[float]) -> list[float]:
    """1-based ranks, tied values sharing the mean of their ranks."""
    order = sorted(range(len(x)), key=lambda i: x[i])
    r, i = [0.0] * len(x), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in order[i : j + 1]:
            r[k] = (i + j) / 2 + 1
        i = j + 1
    return r


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """(U, p): U counts the pairs where a's value is above b's (a tie counts 1/2); p is the exact two-sided p, the
    share of all C(n, len(a)) relabellings of the pooled values whose |U - len(a) len(b) / 2| is at least the observed
    one (ties are carried by the mid-ranks, so the null is exact for them too). (nan, nan) for an empty side or a
    non-finite value. Two sides of 10 seeds with no overlap give p = 2 / 184,756; 10 against 3, 2 / 286."""
    if not a or not b or not all(math.isfinite(v) for v in (*a, *b)):
        return math.nan, math.nan
    n, na = len(a) + len(b), len(a)
    assert math.comb(n, na) <= MAX_SPLITS, (len(a), len(b))
    r, base, mid = midranks([*a, *b]), na * (na + 1) / 2, na * len(b) / 2
    obs = abs(sum(r[:na]) - base - mid)
    hits = sum(abs(sum(r[i] for i in idx) - base - mid) >= obs - 1e-9 for idx in combinations(range(n), na))
    return sum(r[:na]) - base, hits / math.comb(n, na)
