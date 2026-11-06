"""E105 (rows_null): the fixed half split of TOFU's 217 real rows and the filed rule, with no model, so that the CPU dry
run, the tests and run.py's score stage read the same code. Plan: reports/plans/2026-11-06_plan_e105-rows-null.md; rule
results/predictions.json `rows_rule`.

Overlaps are `qwen_neurons.overlap`: the share of the first top-1024 carrying set inside the second, NaN if either is
empty (the do-nothing state, whose scores are never above 0, has an empty set)."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from commute import qwen_neurons as QN

SPLIT_SEED = 105  # the experiment's number; the split is fixed before any run and written to results/split.json
SEEDS = (42, 142)  # the stream seeds of E103 Amendments 1 and 2
HALVES = ("A", "B")


def halves(n: int, seed: int = SPLIT_SEED) -> dict[str, list[int]]:
    """Rows 0..n-1 shuffled by Random(seed); A the first n // 2 (108 of 217), B the rest (109), each sorted."""
    order = random.Random(seed).sample(range(n), n)
    return {"A": sorted(order[: n // 2]), "B": sorted(order[n // 2 :])}


def name(half: str, seed: int) -> str:
    return f"{half}{seed}"


NAMES = tuple(name(h, s) for h in HALVES for s in SEEDS)


def verdict(w_a: float, w_b: float, cross: Sequence[float], x: float) -> dict[str, Any]:
    """The filed rule on one row set. w_a, w_b: each half's seed-142 set inside its seed-42 set; cross: the four overlaps
    of a half-A set with a half-B set (A42-B42, A42-B142, A142-B42, A142-B142); x: the highest cross-arm overlap of
    E103 Amendment 2 (a known-facts seed with a fictional seed, full rows), stored.
    w = min(w_a, w_b); band = 1 - w, the share of a set a change of seed alone replaced in the less stable half.
    "label" iff w - min(cross) <= band and min(cross) - x > band: every cross-half pair agrees within seed noise and
        clears the cross-arm level by more than the band.
    "rows" iff max(cross) - x <= band and w - max(cross) > band: no cross-half pair clears the cross-arm level by more
        than the band, and every one sits beyond seed noise.
    "no call" otherwise; "unread" with a value missing or not finite, or not four cross values.
    The two verdicts cannot both hold (label needs min(cross) > x + band, rows max(cross) <= x + band)."""
    vals = [w_a, w_b, *cross, x]
    if len(cross) != 4 or not all(QN.finite(v) for v in vals):
        return {"verdict": "unread", "w_a": w_a, "w_b": w_b, "cross": list(cross), "x": x}
    w, lo, hi = min(w_a, w_b), min(cross), max(cross)
    band = 1 - w
    v = ("label" if w - lo <= band and lo - x > band else
         "rows" if hi - x <= band and w - hi > band else "no call")
    return {"verdict": v, "w_a": w_a, "w_b": w_b, "cross": list(cross), "x": x, "w": w, "band": band, "cross_min": lo,
            "cross_max": hi, "label_needs_min": max(2 * w - 1, x + band), "rows_needs_max": min(x + band, 2 * w - 1)}


def rule(sets: Mapping[str, Sequence[int]], x: float) -> dict[str, Any]:
    """`verdict` on the four half states' carrying sets on one row set, keyed by `name`."""
    o = lambda a, b: QN.overlap(sets[a], sets[b])
    cross = [o(name("A", s), name("B", t)) for s in SEEDS for t in SEEDS]
    out = verdict(o("A142", "A42"), o("B142", "B42"), cross, x)
    return out | {"sizes": {k: len(sets[k]) for k in NAMES}}


def bridge(sets: Mapping[str, Sequence[int]], full: Mapping[int, Sequence[int]]) -> dict[str, Any]:
    """Reported, not scored: each half state's set inside the full-row known state's set of the same seed and of the other
    seed (E103 Amendments 1 and 2), beside the full states' own seed overlap."""
    same = {name(h, s): QN.overlap(sets[name(h, s)], full[s]) for h in HALVES for s in SEEDS}
    other = {name(h, s): QN.overlap(sets[name(h, s)], full[SEEDS[1 - SEEDS.index(s)]]) for h in HALVES for s in SEEDS}
    return {"same_seed": same, "other_seed": other, "full_seeds": QN.overlap(full[142], full[42])}
