"""E56's filed rules (experiments/keydependent_patch/results/predictions.json), pure functions on numbers so every edge
case runs on CPU (tests/test_keydep_score.py). Plan: reports/plans/2026-10-30_plan_baseline-next-draft.md.

A state record is {"readable": bool, "N16", "k50_index", "r", "S_lo", ...}; states are named `<kind>_s<seed>` with kind
elicit, cipher (the cipher at the elicit fine-tune's update norm) or cipher_own (the cipher at its own E12 step). An
unreadable or missing state gives None, and `elicit.separation` reads any None or nan as "unreadable".
"""

from __future__ import annotations

from typing import Any

from commute.elicit import absdiff, separation


def col(st: dict[str, dict[str, Any]], kind: str, m: str, seeds: list[str]) -> list[float | None]:
    """Metric m of each `<kind>_<seed>`; None for a missing or unreadable state, a missing metric, or a non-number (a nan
    or inf read back from JSON arrives as the string "nan" or "inf")."""
    rec = [st.get(f"{kind}_{s}") for s in seeds]
    vals = [r.get(m) if r is not None and r.get("readable") else None for r in rec]
    return [v if isinstance(v, int | float) and not isinstance(v, bool) else None for v in vals]


def status_of(sep: dict[str, Any], want: str) -> str:
    return "unreadable" if sep["verdict"] == "unreadable" else "pass" if sep["verdict"] == want else "fail"


def binned(vals: list[float | None], cap: int) -> list[float | None]:
    """k50_index binned at the plan's band: every index >= cap (k > 1000: 4096, 12288, none) is one value."""
    return [None if v is None else min(v, cap) for v in vals]


def p1(st: dict[str, dict[str, Any]], seeds: list[str], min_gap: float, k50_cap: int) -> dict[str, Any]:
    """P1: neither N16 nor the binned k50 index (min(k50_index, k50_cap)) separates elicit from the matched cipher
    (verdict "none" for both). A separation in either direction fails: a cipher above the elicit map still tells the
    kinds apart. Amended before any run (2026-10-30): k50 was scored unbinned, so one grid step separated."""
    k50 = [col(st, kind, "k50_index", seeds) for kind in ("elicit", "cipher")]
    out = {"N16": separation(col(st, "elicit", "N16", seeds), col(st, "cipher", "N16", seeds), min_gap),
           "k50_index": separation(binned(k50[0], k50_cap), binned(k50[1], k50_cap), min_gap)}
    sts = [status_of(out[m], "none") for m in out]
    return {**out, "seeds": seeds, "status": "unreadable" if "unreadable" in sts else "pass" if sts == ["pass", "pass"] else "fail"}


def p2(st: dict[str, dict[str, Any]], seeds: list[str], min_gap: float, ratio_min: float) -> dict[str, Any]:
    """P2: r separates elicit higher, S_lo does not separate (verdict "none"), and ratio = r gap / (|S_lo gap| + S_lo's
    larger within-kind range) >= ratio_min (a tie at the bound passes). A zero denominator (every S_lo equal) is
    "unreadable": a degenerate reading to show the user, not a pass."""
    r = separation(col(st, "elicit", "r", seeds), col(st, "cipher", "r", seeds), min_gap)
    lo = separation(col(st, "elicit", "S_lo", seeds), col(st, "cipher", "S_lo", seeds), min_gap)
    out: dict[str, Any] = {"r": r, "S_lo": lo, "seeds": seeds}
    if "unreadable" in (r["verdict"], lo["verdict"]):
        return {**out, "ratio": None, "status": "unreadable"}
    den = abs(lo["gap"]) + lo["spread"]
    if not den > 0:
        return {**out, "ratio": None, "denominator": den, "status": "unreadable"}
    ratio = r["gap"] / den
    ok = r["verdict"] == "a_higher" and lo["verdict"] == "none" and ratio >= ratio_min
    return {**out, "ratio": ratio, "denominator": den, "status": "pass" if ok else "fail"}


def p3(st: dict[str, dict[str, Any]], seeds: list[str], tol: float, same_state: list[str]) -> dict[str, Any]:
    """P3: per seed, |r(cipher) - r(cipher_own)| < tol (strict: a difference of exactly tol fails). A seed whose matched
    step is its own step (`same_state`) has difference 0 by construction and is left out ("uninformative"). Any
    unreadable seed makes P3 unreadable; no informative seed left makes it "uninformative"."""
    pairs = zip(seeds, col(st, "cipher", "r", seeds), col(st, "cipher_own", "r", seeds), strict=True)
    diffs = {s: absdiff(a, b) for s, a, b in pairs}
    informative = [s for s in seeds if s not in same_state]
    out: dict[str, Any] = {"diff": diffs, "same_state": [s for s in seeds if s in same_state], "seeds": seeds}
    if any(d is None for d in diffs.values()):
        return {**out, "status": "unreadable"}
    if not informative:
        return {**out, "status": "uninformative"}
    return {**out, "status": "fail" if any(float(diffs[s] or 0.0) >= tol for s in informative) else "pass"}


def branch_job1(p1_two: dict[str, Any]) -> str:
    return {"pass": "P1 holds on two seeds: job 2 may run",
            "fail": "P1 fails on two seeds (B2 separates the pair): stop, tell the user; the claim narrows to steering",
            "unreadable": "P1 unreadable (a missing state, under its floor, or a nan): stop, tell the user"}[p1_two["status"]]


def branch(a: dict[str, Any], b: dict[str, Any], c: dict[str, Any]) -> str:
    """The plan's decision tree on the five-seed P1, P2 and seeds 2-4's P3."""
    if "unreadable" in (a["status"], b["status"], c["status"]):
        return "unreadable: a missing state, a state under its floor, a nan or a degenerate ratio; stop and tell the user"
    if a["status"] == "fail":
        return "P1 fails: B2 separates the pair; stop, tell the user; the claim is narrowed to steering before any Qwen ask"
    out = "P1 holds"
    if b["status"] == "pass":
        out += "; P2 holds: write up as the headline, ask (a)"
    elif b["r"]["verdict"] == "a_higher" and b["S_lo"]["verdict"] == "none":
        out += f"; P2 fails on the ratio floor ({b['ratio']:.2f} < bar): report as 'small but directional', ask whether (c) is worth gating"
    else:
        out += "; P2 fails (r does not separate elicit higher, or S_lo separates): the tree does not cover it; tell the user"
    if c["status"] == "fail":
        out += "; P3 fails: the matched-size reading is suspect; tell the user"
    elif c["status"] == "uninformative":
        out += "; P3 uninformative (every new seed's matched step is its own step); tell the user"
    return out

