"""E136's filed verdict rule, pure Python (results/predictions.json, the plan's "Filed rule"): no torch, no files, so
the edge cases can be run through on a laptop. No verdict of reused or new is given: sides are named as each pair was built."""

from __future__ import annotations

import math
from typing import Any

DECOMPS = ("trained", "svd", "rotated")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _sd(xs: list[float]) -> float:
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else math.nan


def read_pair(pair: dict[str, Any], states: dict[str, dict[str, Any]], min_gain: float, shrink_line: float) -> dict[str, Any]:
    """One pair: gap = mean(built-to-reuse side) - mean(built-to-be-new side) of the share per decomposition; unreadable when any
    state's gain is below `min_gain` or any share is nan. shrink = sign(trained gap) * (trained gap - SVD gap); a zero trained gap
    has sign 0 and shrink 0, so it counts on rival B's side."""
    rows = {side: [states[s] for s in pair[side]] for side in ("reuse", "new")}
    every = rows["reuse"] + rows["new"]
    low = [s for s in pair["reuse"] + pair["new"] if states[s]["gain"] < min_gain]
    bad = [s for s in pair["reuse"] + pair["new"] if any(math.isnan(states[s][d]) for d in DECOMPS)]
    out: dict[str, Any] = {"id": pair["id"], "scored": pair["scored"], "n_reuse": len(pair["reuse"]), "n_new": len(pair["new"]), "built": pair["built"],
                            "readable": not (low or bad or any(math.isnan(s["gain"]) for s in every)), "low_gain": low, "nan_share": bad}  # fmt: skip
    if not out["readable"]:
        return out
    for d in DECOMPS:
        a, b = [r[d] for r in rows["reuse"]], [r[d] for r in rows["new"]]
        out[d] = {"gap": _mean(a) - _mean(b), "reuse": a, "new": b, "sd_reuse": _sd(a), "sd_new": _sd(b)}
    g_trained = out["trained"]["gap"]
    sign = 0.0 if g_trained == 0 else math.copysign(1.0, g_trained)
    out["sign"], out["shrink"] = sign, sign * (g_trained - out["svd"]["gap"])
    out["rival_a_side"] = out["shrink"] >= shrink_line
    return out


def verdict(rows: list[dict[str, Any]], min_scored: int) -> dict[str, Any]:
    """Over the scored pairs that are readable: rival A when shrink >= the line on more than half, rival B when it is under the line on
    more than half, "split" when exactly half and half, "no call" under `min_scored` readable scored pairs."""
    used = [r for r in rows if r["scored"] and r["readable"]]
    n_a = sum(1 for r in used if r["rival_a_side"])
    n_b = len(used) - n_a
    if len(used) < min_scored:
        v = "no call"
    elif n_a > n_b:
        v = "rival A"
    elif n_b > n_a:
        v = "rival B"
    else:
        v = "split"
    return {"verdict": v, "n_scored_readable": len(used), "n_rival_a_side": n_a, "n_rival_b_side": n_b}
