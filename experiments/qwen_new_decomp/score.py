"""E131's filed rules (results/predictions.json): the new/old ratio of a per-row number on one row set, its row
bootstrap interval, and each prediction's verdict. Pure Python and NumPy, no GPU; run.py's score stage calls it and
tests/test_qwen_new_decomp.py runs it on its edge cases.

A number per row is the row's mean over its positions (rows equal-weighted). The ratio is mean(new) / mean(old) over
the same rows, paired; the interval resamples row indices with replacement (the same indices for both
decompositions), B times, numpy generator seed SEED, and takes the 2.5th and 97.5th percentiles (linear
interpolation) of the resampled ratios.

Verdicts: "pass" when ratio < 1 and the interval's upper end < 1 (strict: a tie at 1 is a miss); "miss" otherwise;
"unreadable" when the rule cannot be read: fewer than 2 rows, row sets of unequal length, a non-finite value, or a
zero old mean in the data or in any resample (the ratio is undefined there).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

B, SEED = 10_000, 0


def ratio_with_interval(new: Sequence[float], old: Sequence[float], b: int = B, seed: int = SEED) -> dict:
    """mean(new) / mean(old) and its row-bootstrap 95% interval; `reason` set and numbers None when unreadable."""
    n = len(old)
    out: dict = {"n_rows": n, "ratio": None, "lo95": None, "hi95": None, "reason": None}
    if len(new) != n:
        return out | {"reason": f"row sets differ in length ({len(new)} new, {n} old)"}
    if n < 2:
        return out | {"reason": f"{n} row(s): a bootstrap needs at least 2"}
    a, c = np.asarray(new, dtype=np.float64), np.asarray(old, dtype=np.float64)
    if not (np.isfinite(a).all() and np.isfinite(c).all()):
        return out | {"reason": "a non-finite per-row value"}
    if c.mean() == 0:
        return out | {"reason": "old mean is zero"}
    idx = np.random.default_rng(seed).integers(0, n, size=(b, n))
    denominators = c[idx].mean(1)
    if (denominators == 0).any():
        return out | {"reason": "old mean is zero in a resample"}
    ratios = a[idx].mean(1) / denominators
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return out | {"ratio": float(a.mean() / c.mean()), "lo95": float(lo), "hi95": float(hi),
                  "new_mean": float(a.mean()), "old_mean": float(c.mean())}


def verdict(r: dict) -> str:
    """The filed rule: new below old, with the interval below 1 too."""
    if r["reason"] is not None:
        return "unreadable"
    assert r["ratio"] is not None and r["hi95"] is not None and math.isfinite(r["ratio"])
    return "pass" if r["ratio"] < 1 and r["hi95"] < 1 else "miss"


PREDICTIONS = {"P1": ("G", "ci_kl"), "P2": ("G", "adv_kl"), "P3": ("G-think", "ci_kl"), "P4": ("G-think", "adv_kl")}
NUMBERS = ("ci_kl", "active", "adv_kl")
ROW_SETS = ("T", "G", "G-think")


def score(measured: dict[tuple[str, str], dict]) -> dict:
    """`measured[(decomposition, row set)]` holds per-row lists under NUMBERS. Every (row set, number) with both
    decompositions measured gets its ratio; P1-P4 their verdicts (unreadable when a side is missing)."""
    table: dict = {}
    for rows in ROW_SETS:
        for number in NUMBERS:
            new, old = measured.get(("new", rows), {}).get(number), measured.get(("old", rows), {}).get(number)
            if new is None or old is None:
                table[f"{rows}/{number}"] = {"reason": "not measured for both decompositions"}
                continue
            table[f"{rows}/{number}"] = ratio_with_interval(new, old)
    preds = {}
    for name, (rows, number) in PREDICTIONS.items():
        r = table[f"{rows}/{number}"]
        preds[name] = {"rows": rows, "number": number, "verdict": verdict(r),
                       **{k: r.get(k) for k in ("ratio", "lo95", "hi95", "reason")}}
    return {"ratios": table, "predictions": preds}
