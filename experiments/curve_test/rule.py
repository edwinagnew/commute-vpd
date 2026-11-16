"""The filed rule of the curve test (cell P1.23), pure, on item files: no torch, so plot.py and a CPU check can call it.

An item file's search is list all, two greedy passes (key "1" writes only, key "2" writes and reads, as E184), each a curve
of points {n, closure, closure_answer, ...}. The curve read is E191's: at each n, the best closure either pass found with at
most n names (the empty intervention closes 0). `closure` is read on the full next-token distribution at the last position
of the item's read windows; `closure_answer`, for the same names, on the read windows whose last position is the task's
answer position. The names are always chosen by the full closure.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

CAP = 16
PAIRS = (("copy24", "cipher24"), ("ident32", "rand32"), ("case_real", "case_twin"), ("repeat_real", "repeat_twin"), ("twin", "fact"))
READINGS = {"full": ("closure", "kl"), "answer": ("closure_answer", "kl_answer")}  # reading -> (closure key, its denominator)


def best_by_n(rec: Mapping[str, Any], key: str) -> list[float]:
    """At each n = 0..CAP, the names of the best full-distribution closure found with at most n names over both passes of
    list all (the empty intervention, closure 0, at n = 0), read by `key`: "closure" gives E191's curve, "closure_answer"
    the same names' closure on the answer windows (same names, two reads: decisions.md, 2026-11-16). A NaN or missing full
    closure counts as no point; a missing `key` value reads NaN."""
    best: list[tuple[float, float]] = [(0.0, 0.0)] * (CAP + 1)
    for r in ("1", "2"):
        for e in rec.get("lists", {}).get("all", {}).get(r, {}).get("curve", []):
            c, v = e.get("closure"), e.get(key)
            if c is not None and not math.isnan(c) and e["n"] <= CAP and c > best[e["n"]][0]:
                best[e["n"]] = (c, math.nan if v is None else v)
    for n in range(1, CAP + 1):
        best[n] = max(best[n], best[n - 1], key=lambda p: p[0])
    return [v for _, v in best]


def why_not(rec: Mapping[str, Any], reading: str, floor: float) -> str | None:
    """Why an item gives no closure under `reading`, or None if it gives one."""
    if not rec.get("done"):
        return "not searched"
    if rec.get("learned") is not True:
        return "fine-tune not learned"
    kl = rec.get(READINGS[reading][1])
    return None if kl is not None and kl >= floor else "KL(F || M) under the floor"


def score(recs: Mapping[str, Mapping[str, Any]], seeds: tuple[int, ...], bar: float, floor: float) -> dict[str, Any]:
    """Per reading: each pair and seed, the reuse side's closure at CAP names minus the new side's, "holds" if it exceeds
    `bar`; the verdict "holds" if every pair-seed was read and holds, "fails" if any read pair-seed does not, else
    "incomplete" (some not read, none read failed)."""
    out: dict[str, Any] = {}
    for reading, (key, _) in READINGS.items():
        rows: dict[str, Any] = {}
        for reuse, new in PAIRS:
            for s in seeds:
                a, b = recs.get(f"{reuse}_s{s}", {}), recs.get(f"{new}_s{s}", {})
                why = {side: w for side, rec in (("reuse", a), ("new", b)) if (w := why_not(rec, reading, floor))}
                if why:
                    rows[f"{reuse}/{new} s{s}"] = {"read": False, "why": why}
                    continue
                ra, rb = best_by_n(a, key)[CAP], best_by_n(b, key)[CAP]
                rows[f"{reuse}/{new} s{s}"] = {"read": True, "reuse": ra, "new": rb, "gap": ra - rb, "holds": ra - rb > bar}
        read = [v for v in rows.values() if v["read"]]
        verdict = "fails" if any(not v["holds"] for v in read) else "holds" if len(read) == len(rows) else "incomplete"
        out[reading] = {"pairs": rows, "verdict": verdict, "n_read": len(read), "n_holds": sum(v["holds"] for v in read)}
    return out
