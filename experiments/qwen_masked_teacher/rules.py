"""E76's filed rules (results/predictions.json), pure functions of the read numbers, so that edge cases can be run on them.
A reading: {"step", "i", "ii", "iii", "iv"}, per position on the 8 fixed rows; (i) clean KL(FT ‖ student), (ii) clean KL(W0 ‖ student),
(iii) the share of the change's squared size in blocks 21-27 over δW's, (iv) KL(FT clean ‖ student masked), all 196 sites."""
from __future__ import annotations

import math
from typing import Any

NAN = math.nan
PLACED, REMAINDER, SIZE, LATE = 0.09, 0.53, 0.94, 3.0  # (i) band of placements, (i) the remainder's, (ii) 1.5 x δW's, (iii)
K0_WANT, K0_TOL, REBUILD, REBUILD_TOL, SHARE_MIN = 0.0440, 0.10, {"amp": 0.59, "cos": 0.86}, 0.005, 0.5  # stage 0's stops
STEP0, CHECK_STEP, A_FALL = (1.0, 2.0), 100, 0.95  # the arms' stops
BARELY = 0.1  # the lean on L: below A's (iv) by under this share of it
LATE_BLOCK = 21


def r9(x: float) -> float:
    return round(x, 9)


def stage0_stop(s: dict[str, Any]) -> tuple[str, str] | None:
    """The first stage-0 stop missed, in the plan's order; a NaN misses."""
    amp, cos = s["hook"]["rebuild"]["amp"], s["hook"]["rebuild"]["cos"]
    if not (r9(abs(amp - REBUILD["amp"])) <= REBUILD_TOL and r9(abs(cos - REBUILD["cos"])) <= REBUILD_TOL):
        return "hook_rebuild", f"amplitude {amp}, cosine {cos}, want {REBUILD}: something other than the final norm, hook suspect"
    if not r9(abs(s["K0"] / K0_WANT - 1)) <= K0_TOL:
        return "K0", f"K0 {s['K0']}, want {K0_WANT} within {K0_TOL:.0%}"
    if not r9(s["share"]) >= SHARE_MIN:
        return "share", f"the columns hold {s['share']} of the sum of g, under {SHARE_MIN}: ask the user about the paid card"
    return None


def arm_stop(arm: str, first: dict[str, float], now: dict[str, float]) -> tuple[str | None, bool, str]:
    """(rule checked at this reading or None, whether it stops, its reading). (i), (ii), (iv) non-finite at any reading stops."""
    if not all(math.isfinite(now[k]) for k in ("i", "ii", "iv")):
        return "non_finite", True, f"{arm} step {now['step']}: {now}"
    if arm == "C" and now["step"] == CHECK_STEP:
        return "control_masked_falls", not r9(now["iv"]) < r9(first["iv"]), f"(iv) {now['iv']} from {first['iv']}: not falling is a gradient bug"
    if arm == "A" and now["step"] == CHECK_STEP:
        return "a_clean_falls", not r9(now["i"]) < r9(A_FALL * first["i"]), f"(i) {now['i']} from {first['i']}, must be under {A_FALL} of it"
    if arm in ("B", "L") and now["step"] == 0:
        return "masked_step0", not STEP0[0] <= r9(now["iv"]) <= STEP0[1], f"(iv) {now['iv']}, band {STEP0}"
    return None, False, ""


def late_share(blocks: list[float]) -> float:
    """The share of the squared size in blocks 21-27; NaN with no change to read."""
    total = sum(blocks)
    return sum(blocks[LATE_BLOCK:]) / total if total > 0 else NAN


def late_ratio(change: list[float], dw: list[float]) -> float:
    """(iii). NaN when either share is unread or δW's share is zero: a zero denominator reads nothing either way."""
    a, b = late_share(change), late_share(dw)
    return a / b if b > 0 else NAN


def spread(readings: list[dict[str, float]]) -> float:
    """C's spread: (iv)'s range over C's readings, its fall when it falls throughout. NaN unread."""
    v = [r["iv"] for r in readings]
    return max(v) - min(v) if v and all(math.isfinite(x) for x in v) else NAN


def band_of(floor: float, c_spread: float) -> float:
    """The larger of the two; NaN when either is unread."""
    return NAN if math.isnan(floor) or math.isnan(c_spread) else max(floor, c_spread)


def labels(r: dict[str, float], a: dict[str, float] | None, floor: float, band: float) -> dict[str, Any]:
    """One of B, L at its last step against A at the same step. A above 0.53 on (i): (i) is read against A's, not the bands."""
    i_a, iv_a = (NAN, NAN) if a is None else (a["i"], a["iv"])
    short = r9(i_a) > REMAINDER
    ok_i, bad_i = (r9(r["i"]) <= r9(i_a), False) if short else (r9(r["i"]) <= PLACED, r9(r["i"]) > REMAINDER)
    bad_ii, bad_iii = r9(r["ii"]) > SIZE, r9(r["iii"]) > LATE
    ok = ok_i and r9(r["ii"]) <= SIZE and r9(r["iii"]) <= LATE and r9(iv_a - r["iv"]) > r9(band)
    out = ["works"] if ok else []
    out += [f"not a map by {n}" for n, bad in (("(i)", bad_i), ("(ii)", bad_ii), ("(iii)", bad_iii)) if bad]
    if r9(abs(r["iv"] - iv_a)) <= r9(floor):
        out.append("fails to learn")
    if not out:
        unread = any(math.isnan(x) for x in (r["i"], r["ii"], r["iii"], r["iv"], iv_a, band, floor))
        out = ["unread" if unread else "as read"]
    return {"labels": out, "read_against_A": short, "iv_below_A": iv_a - r["iv"], "band": band, "floor": floor}


def arm_a(a: dict[str, float]) -> dict[str, Any]:
    """A on (i)-(iii). Above 0.53 on (i): the clean term is short of steps or columns at this size."""
    bad = [f"not a map by {n}" for n, x, lim in (("(i)", a["i"], REMAINDER), ("(ii)", a["ii"], SIZE), ("(iii)", a["iii"], LATE)) if r9(x) > lim]
    ok = r9(a["i"]) <= PLACED and r9(a["ii"]) <= SIZE and r9(a["iii"]) <= LATE
    unread = any(math.isnan(a[k]) for k in ("i", "ii", "iii"))
    return {"labels": bad or (["(i)-(iii) hold"] if ok else ["unread" if unread else "as read"]), "short": r9(a["i"]) > REMAINDER}


def outcome(a: dict[str, Any], b: list[str], ell: list[str]) -> str:
    """The plan's "What each result does"; unread while B or L has no verdict."""
    def fail(x: list[str]) -> bool:
        return any(n.startswith(("not a map", "fails")) for n in x)

    if a.get("short"):
        return "A short of the clean term: bring the paid card to the user with this number"
    if "works" in b:
        return "B works"
    if {"missing", "stopped", "unread"} & {*b, *ell}:
        return "unread"
    if fail(b) and "works" in ell:
        return "B fails, L works"
    return "both fail" if fail(b) and fail(ell) else "as read"


def leans(b: list[str], ell: list[str], r_l: dict[str, float] | None, a: dict[str, float] | None) -> dict[str, Any]:
    """B: not a map by (iii), or fails to learn (medium). L: (iii) passes and (iv) below A's by under BARELY of it (weak)."""
    if r_l is None or a is None:
        l_holds = None
    else:
        gap = a["iv"] - r_l["iv"]
        l_holds = r9(r_l["iii"]) <= LATE and 0 < r9(gap) and r9(gap) < r9(BARELY * a["iv"])
    b_read = not {"missing", "stopped", "unread"} & set(b)
    return {"B": ("not a map by (iii)" in b or "fails to learn" in b) if b_read else None, "L": l_holds}


def last(rec: dict[str, Any] | None) -> dict[str, float] | None:
    return rec["readings"][-1] if rec and rec.get("done") else None


def score(s0: dict[str, Any] | None, arms: dict[str, dict[str, Any] | None], stopped: dict[str, Any] | None) -> dict[str, Any]:
    """Every arm's verdict from what is on disk. An arm not done has none: "stopped" when results/stopped.json exists and it has a
    record (the arm the stop came in), else "missing"."""
    floor = NAN if s0 is None else s0["floor"]
    c = arms.get("C")
    c_spread = spread(c["readings"]) if c and c.get("done") else NAN
    band = band_of(floor, c_spread)
    a = last(arms.get("A"))
    out: dict[str, Any] = {"floor": floor, "C_spread": c_spread, "band": band, "stopped": stopped}

    def absent(name: str) -> dict[str, Any]:
        return {"labels": ["stopped" if stopped and arms.get(name) else "missing"]}

    va = arm_a(a) if a else absent("A")
    out["A"] = va
    verdict: dict[str, list[str]] = {}
    for name in ("B", "L"):
        r = last(arms.get(name))
        v = labels(r, a, floor, band) if r else absent(name)
        out[name], verdict[name] = v, v["labels"]
    out["last"] = {k: last(arms.get(k)) for k in ("C", "A", "B", "L")}
    out["outcome"] = outcome(va, verdict["B"], verdict["L"])
    out["leans"] = leans(verdict["B"], verdict["L"], last(arms.get("L")), a)
    return out
