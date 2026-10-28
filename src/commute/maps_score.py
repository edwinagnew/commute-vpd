"""E44 (experiments/qwen_maps) scoring: the filed rules of results/predictions.json as pure functions of the readings.

Plan: reports/plans/2026-10-28_plan_qwen-q0q1.md, sections 3 and 5. Thresholds come from the predictions file; every
comparison rounds to 9 places first (`r9`), so a tie in exact arithmetic lands on the side the rule writes as inclusive.
A NaN, an infinity or a missing value (None) is a miss in every rule. No torch: tests run these on plain floats.

A state's readings, as `run.py` writes them: r1m, r1w (CI-masked KL ratios to W0, maths and web rows), r3m, r3w
(paired replay ratios), r2 (`kept`'s dict) and share (the pooled Δ-change, ‖Δ′ − Δ₀‖_Σ / ‖δW‖_Σ).
"""

from __future__ import annotations

import math
from typing import Any

Num = float | None


def r9(x: float) -> float:
    return round(x, 9)


def finite(*xs: Num) -> bool:
    return all(x is not None and math.isfinite(x) for x in xs)


def pooled(nums: list[float], dens: list[float]) -> float:
    """Ratio of sums over matrices; NaN for an empty pool or a zero denominator."""
    den = sum(dens)
    return sum(nums) / den if dens and den > 0 else math.nan


def kept(KD: Num, KDp: Num, K0: Num, spread: Num, rule: dict[str, float]) -> dict[str, Any]:
    """r2 = [K(D) − K(D′)] / [K(D) − K₀], scored only where K(D) − K₀ ≥ max(floor_min, floor_mult × spread). A missing
    or non-finite input gives a scored NaN (a miss everywhere), never "unscored"."""
    if not finite(KD, KDp, K0, spread):
        return {"r2": math.nan, "scored": True, "floor": math.nan}
    assert KD is not None and KDp is not None and K0 is not None and spread is not None
    floor = max(rule["floor_min"], rule["floor_mult"] * spread)
    den = KD - K0
    if r9(den) < r9(floor):
        return {"r2": None, "scored": False, "floor": floor}
    return {"r2": (KD - KDp) / den, "scored": True, "floor": floor}


def r2_of(st: dict[str, Any]) -> Num:
    return st.get("r2", {}).get("r2")


def passes(st: dict[str, Any], rule: dict[str, float]) -> str:
    """The pass rule (a measurement rule): r1 ≤ line on both row sets, r3 ≤ line on both, r2 ≥ kept_min where scored.
    "fail" on any miss or missing value; "incomplete" when everything else holds and r2 is unscored."""
    xs = [st.get(k) for k in ("r1m", "r1w", "r3m", "r3w")]
    if not finite(*xs) or any(r9(x) > rule["line"] for x in xs if x is not None):
        return "fail"
    r2 = st.get("r2")
    if r2 is None:
        return "fail"
    if not r2["scored"]:
        return "incomplete"
    return "pass" if finite(r2["r2"]) and r9(r2["r2"]) >= rule["kept_min"] else "fail"


def teeth_i(lower: dict[str, Any], upper: dict[str, Any], KD: Num, K0: Num, rule: dict[str, float]) -> dict[str, Any]:
    """Prediction 2. Scored only when the halves' Δ-change shares are within share_tol (else "confounded by the span",
    a failure to fail). r2 scored: lower ≤ upper − margin. Unscored: r1m lower ≥ r1m upper + margin·(K(D) − K₀)/K₀."""
    sl, su = lower.get("share"), upper.get("share")
    if not finite(sl, su):
        return {"holds": False, "why": "missing share"}
    assert sl is not None and su is not None
    if r9(abs(sl - su)) > rule["share_tol"]:
        return {"holds": False, "why": "confounded by the span", "shares": [sl, su]}
    rl, ru = lower.get("r2"), upper.get("r2")
    if rl is None or ru is None:
        return {"holds": False, "why": "missing r2"}
    if rl["scored"] and ru["scored"]:
        a, b = rl["r2"], ru["r2"]
        ok = finite(a, b) and r9(a) <= r9(b - rule["margin"])
        return {"holds": ok, "why": "r2", "lower": a, "upper": b}
    a, b = lower.get("r1m"), upper.get("r1m")
    if not finite(a, b, KD, K0) or K0 == 0:
        return {"holds": False, "why": "missing r1 or K0 = 0"}
    assert a is not None and b is not None and KD is not None and K0 is not None
    need = b + rule["margin"] * (KD - K0) / K0
    return {"holds": r9(a) >= r9(need), "why": "r1 (r2 unscored)", "lower": a, "upper": b, "need": need}


def teeth_ii(rand: dict[str, Any], rule: dict[str, float]) -> dict[str, Any]:
    """Prediction 3: the largest of r1m, r1w, r3m, r3w of control (ii) is above the line."""
    xs = [rand.get(k) for k in ("r1m", "r1w", "r3m", "r3w")]
    if not finite(*xs):
        return {"holds": False, "why": "missing reading"}
    top = max(x for x in xs if x is not None)
    return {"holds": r9(top) > rule["line"], "max": top}


def teeth_iii(dn: dict[str, Any], rule: dict[str, float]) -> dict[str, Any]:
    """Prediction 4: do-nothing's r2 < kept_min where scored, else its r1m > line."""
    r2 = dn.get("r2")
    if r2 is None:
        return {"holds": False, "why": "missing r2"}
    if r2["scored"]:
        return {"holds": finite(r2["r2"]) and r9(r2["r2"]) < rule["kept_min"], "why": "r2", "r2": r2["r2"]}
    x = dn.get("r1m")
    return {"holds": finite(x) and r9(x) > rule["line"], "why": "r1 (r2 unscored)", "r1m": x}  # type: ignore[operator]


def dn_r3_web(dn: dict[str, Any], rule: dict[str, float]) -> dict[str, Any]:
    """Prediction 4's side prediction at step 10: do-nothing's r3 web ≤ line."""
    x = dn.get("r3w")
    return {"holds": finite(x) and r9(x) <= rule["line"], "r3w": x}  # type: ignore[operator]


def outcome(teeth: dict[str, dict[str, bool]], last: str, first: str) -> dict[str, Any]:
    """Prediction 5. `teeth[step]` = {"i", "ii", "iii"} -> holds. No teeth at the last step: stop before stage M. Teeth
    at the last step and not at the first: the first step's splits are reported unscored."""
    def all3(t: str) -> bool:
        return t in teeth and all(teeth[t].get(k) is True for k in ("i", "ii", "iii"))

    if not all3(last):
        return {"teeth": False, "stop": True, "unscored_steps": sorted(teeth)}
    return {"teeth": True, "stop": False, "unscored_steps": [] if all3(first) else [first]}


def control_w0(C: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    """Prediction 1 on stage C's readings: r1 at W0 and each replayed source within tol (relative) of its saved value;
    KL to start at step 100 (fp32 file) within tol; the gap g = |KL(bf16 100) / KL(fp32 100) − 1|; steps 1 and 10
    within tol + gap_mult·g. A missing source or value is a miss."""
    def rel(x: Num, ref: float) -> float:
        return abs(x / ref - 1) if finite(x) else math.nan  # type: ignore[operator]

    def within(x: Num, ref: float, tol: float) -> bool:
        d = rel(x, ref)
        return finite(d) and r9(d) <= tol

    tol, parts = rule["tol"], {}
    for block, ref in rule["ci"].items():
        parts[f"ci_{block}"] = within(C.get("ci", {}).get(block), ref, tol)
    for name, ref in rule["replay"].items():
        parts[f"replay_{name}"] = within(C.get("replay", {}).get(name), ref, tol)
    kl = C.get("kl_to_start", {})
    ref100 = rule["kl_to_start"]["100_fp32"]
    parts["kl_100_fp32"] = within(kl.get("100_fp32"), ref100, tol)
    g = rel(kl.get("100"), kl["100_fp32"]) if finite(kl.get("100_fp32")) and kl["100_fp32"] != 0 else math.nan
    for t in ("1", "10"):
        parts[f"kl_{t}"] = finite(g) and within(kl.get(t), rule["kl_to_start"][t], tol + rule["gap_mult"] * g)
    return {"holds": all(parts.values()), "gap": g, "parts": parts}
