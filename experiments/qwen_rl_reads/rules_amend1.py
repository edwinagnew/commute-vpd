"""E75 Amendment 1's filed rule (results/predictions_amend1.json) as pure functions of the reads' JSON; no torch, no GPU. Plan:
reports/plans/2026-11-03_plan_e75-rl-reads.md, "Amendment 1". A read is {"T": 256, "sums": {"K": per-row sums, ...}}, from
FixedReader.clean. A rule that cannot read says "no call". Comparisons round to 9 places, as E72's.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fixed_rules", HERE.parent / "qwen_fixed_masks" / "rules.py")
assert spec is not None and spec.loader is not None
fixed = importlib.util.module_from_spec(spec)
sys.modules["fixed_rules"] = fixed
spec.loader.exec_module(fixed)

FAILS, PASSES, NO_CALL, TEETH, NOT_ALL = "fails", "passes", "no call", "has teeth", "not all fail"
STATES, TOL, NAN = ("random", "other", "signs"), 1e-3, math.nan
ok, r9 = fixed.ok, fixed.r9


def n_rows(rd: dict[str, Any] | None) -> int:
    return len(rd["sums"]["K"]) if rd is not None else 0


def k_of(rd: dict[str, Any] | None) -> float:
    """K over every position; a missing or empty read is unread."""
    return fixed.mean(rd, "K") if n_rows(rd) else NAN


def label(k: float, k_remainder: float, width: float) -> str:
    """Above the remainder's K by more than the half-width "fails", below by more "passes"; at the edge, or unread, no call."""
    if not all(ok(x) for x in (k, k_remainder, width)):
        return NO_CALL
    gap, w = r9(k - k_remainder), r9(width)
    return FAILS if gap > w else PASSES if -gap > w else NO_CALL


def width_of(state: dict[str, Any], remainder: dict[str, Any]) -> float:
    """Half-width of K(state) - K(remainder), the 95% row bootstrap, seed 0, paired by row, as rules.gap_width. It needs two rows
    or more, and the same rows on both sides."""
    n = n_rows(state)
    if n < 2 or n != n_rows(remainder) or state["T"] != remainder["T"]:
        return NAN
    T = state["T"]
    return fixed.half_width(lambda i: (fixed.tot(state, "K", i) - fixed.tot(remainder, "K", i)) / (len(i) * T), n)


def cell(state: dict[str, Any] | None, remainder: dict[str, Any] | None, ruler: float) -> dict[str, Any]:
    """One state on one update. over_ruler is K / K(W0), reported for the lean, no rule."""
    k, k_rem = k_of(state), k_of(remainder)
    width = width_of(state, remainder) if state is not None and remainder is not None else NAN
    return {"K": k, "K_remainder": k_rem, "gap": k - k_rem, "width": width, "over_ruler": fixed.div(k, ruler), "ruler": ruler,
            "label": label(k, k_rem, width)}


def overall(table: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """"has teeth" if every state fails on every update, otherwise the list. An empty table reads nothing: no call."""
    cells = [(u, s, c["label"]) for u, row in table.items() for s, c in row.items()]
    if not cells:
        return {"overall": NO_CALL, "list": []}
    rest = [{"update": u, "state": s, "label": lab} for u, s, lab in cells if lab != FAILS]
    return {"overall": NOT_ALL if rest else TEETH, "list": rest}


def control(reads: dict[str, Any], ruler: float) -> dict[str, Any]:
    """On u1, by this script's path: W0 reads the ruler and W0 + δW reads 0, each within TOL of the ruler. A read not yet made is not
    a miss; an unread number or a ruler not above zero is."""
    got = {k: k_of(reads[k]) for k in ("clean_W0", "clean_FT") if k in reads}
    gap = {k: abs(v - ruler) if k == "clean_W0" else abs(v) for k, v in got.items()}
    missed = not (ok(ruler) and ruler > 0) or any(not (ok(x) and r9(x / ruler) <= TOL) for x in gap.values())
    return {"ruler": ruler, "got": got, "gap": gap, "missed": missed}


def score(res: dict[str, dict[str, Any]], ref: dict[str, dict[str, Any]], updates: tuple[str, ...]) -> dict[str, Any]:
    """`res`: this job's reads per update. `ref`: per update, E75's clean_W0 (the ruler) and clean_remainder (the bar)."""
    table = {u: {s: cell(res.get(u, {}).get(s), ref.get(u, {}).get("clean_remainder"), k_of(ref.get(u, {}).get("clean_W0")))
                 for s in STATES} for u in updates}
    return {"control": control(res.get("u1", {}), k_of(ref.get("u1", {}).get("clean_W0"))), "table": table} | overall(table)
