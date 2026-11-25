"""E47's filed rules (results/predictions.json), pure: no model, no GPU (tests/test_claim_tests.py).

A state's reading is `faithfulness.multi_start`'s record: {"kl", "ratio" (start -> KL / W0's KL), "mean", "max",
"complete"}. `merge` pools one state's readings from several jobs (T1 with t1b), each ratio taken against its own job's
W0. The lines are `faith_score.verdict`'s: pass <= 1.5 / (1 + f), fail >= 1.5 (1 + f), a tie goes to the line's side.
"""

from __future__ import annotations

import math
from typing import Any

from commute.faith_score import LINE, verdict

Rec = dict[str, Any]


def ok(x: Any) -> bool:
    """A finite number (a non-finite one is written to the results file as a string)."""
    return isinstance(x, int | float) and math.isfinite(x)


def fail_line(f: float) -> float:
    return LINE * (1 + f)


def merge(*recs: Rec) -> Rec:
    """One state's readings from several jobs: the union of the ratios, complete only if every job's reading was."""
    ratio = {s: r for rec in recs for s, r in rec["ratio"].items()}
    return {"ratio": ratio, "complete": all(rec["complete"] for rec in recs)}


def r_max(rec: Rec, f: float) -> str:
    """R_max's verdict: every start read and finite -> `verdict` of the max; else (an early stop, a missing or a
    non-finite start) fail if a finite ratio reached the fail line (a max only rises), otherwise unscored."""
    finite = [r for r in rec["ratio"].values() if ok(r)]
    if rec["complete"] and finite and len(finite) == len(rec["ratio"]):
        return verdict(max(finite), f)
    return "fail" if finite and max(finite) >= fail_line(f) else "unscored"


def mean_of(rec: Rec) -> float | None:
    """R_mean, None when a start is missing or non-finite (T2 reads without early stops)."""
    r = list(rec["ratio"].values())
    return sum(r) / len(r) if rec["complete"] and r and all(ok(x) for x in r) else None


def r_mean(rec: Rec, f: float) -> str:
    m = mean_of(rec)
    return "unscored" if m is None else verdict(m, f)


def maximum(rec: Rec) -> float:
    return max(r for r in rec["ratio"].values() if ok(r))


# ----------------------------------------------------------------------------- T1 (and T1 with t1b)


def t1(states: dict[str, Rec], f: float) -> dict[str, Any]:
    """stands: both copies pass, both do-nothing states fail and each copy's R_max (1 + f) < its do-nothing's R_max;
    falls: either copy fails; no_power_<X>: both copies pass and X's do-nothing does not fail (it reads pass or
    inconclusive), the other teacher standing; anything else unsettled."""
    v = {n: r_max(rec, f) for n, rec in states.items()}
    per: dict[str, str] = {}
    for t in ("r1", "rand32"):
        nothing, copy = v[f"{t}_nothing"], v[f"{t}_copy"]
        if copy != "pass" or nothing == "unscored":
            per[t] = "unsettled"
        elif nothing != "fail":
            per[t] = "no_power"
        else:
            sep = maximum(states[f"{t}_copy"]) * (1 + f) < maximum(states[f"{t}_nothing"])
            per[t] = "stands" if sep else "unsettled"
    if "fail" in (v["r1_copy"], v["rand32_copy"]):
        outcome = "falls"
    elif all(p == "stands" for p in per.values()):
        outcome = "stands"
    elif set(per.values()) == {"no_power"}:
        outcome = "no_power_both"
    elif set(per.values()) == {"stands", "no_power"}:
        outcome = "no_power_" + next(t for t, p in per.items() if p == "no_power")
    else:
        outcome = "unsettled"
    return {"outcome": outcome, "per_teacher": per, "verdicts": v, "f": f, "fail_line": fail_line(f)}


# ----------------------------------------------------------------------------- T2, t2b, t2c


def cap(control: str, base: Rec, high: Rec, f: float) -> str:
    """The cap reading of one job: unscored without power (the control does not fail), when rank 8 fails or is
    unscored (nothing to climb from) or the higher rank is unscored; cap_artefact when the higher rank fails;
    no_cap_effect when its R_mean <= rank 8's (1 + f); else unsettled."""
    vb, vh = r_mean(base, f), r_mean(high, f)
    if control != "fail" or vb in ("fail", "unscored") or vh == "unscored":
        return "unscored"
    if vh == "fail":
        return "cap_artefact"
    mb, mh = mean_of(base), mean_of(high)
    assert mb is not None and mh is not None
    return "no_cap_effect" if mh <= mb * (1 + f) else "unsettled"


ABSOLUTE = {"pass": "stands", "fail": "falls", "inconclusive": "unsettled", "unscored": "unscored"}


def t2(states: dict[str, Rec], f: float, high: str) -> dict[str, Any]:
    """T2 (high = lora128), t2b (lora32) and t2c (lora128 on the second rows): the control's h0 verdict, rank 8's
    absolute reading on R_mean, and the cap reading."""
    ctrl = states["adam_tt"]["ratio"].get("h0")
    control = verdict(ctrl, f) if ok(ctrl) else "unscored"
    absolute = ABSOLUTE[r_mean(states["lora8_tt"], f)]
    return {"control": control, "absolute": absolute, "cap": cap(control, states["lora8_tt"], states[high], f),
            "means": {n: mean_of(r) for n, r in states.items()}, "f": f, "fail_line": fail_line(f)}


def t2_combined(t2_: dict[str, Any], t2b_: dict[str, Any], t2c_: dict[str, Any]) -> dict[str, str]:
    """absolute: rank 8 stands on both row sets (T2, t2c) -> stands; falls on either -> falls; else unsettled.
    cap: any job's cap_artefact -> cap_artefact; every job's no_cap_effect -> no_cap_effect; else unsettled."""
    a = (t2_["absolute"], t2c_["absolute"])
    absolute = "falls" if "falls" in a else "stands" if a == ("stands", "stands") else "unsettled"
    caps = (t2_["cap"], t2b_["cap"], t2c_["cap"])
    c = ("cap_artefact" if "cap_artefact" in caps else "no_cap_effect" if all(x == "no_cap_effect" for x in caps)
         else "unsettled")
    return {"absolute": absolute, "cap": c}


# ----------------------------------------------------------------------------- T3


def t3_seed(s: dict[str, Any], agree: float = 0.05, differ: float = 0.10, near: float = 0.07) -> str:
    """One seed. s["a"][size] = {"keep", "keep_ref"} or None (unscored: size not matched within 5% or P < 0.5),
    s["b"][step] = {"keep", "keep_ref"} or None, s["keep_1000"] = the 0.01 run's step-1000 keep. stands: (a) within
    `agree` at both sizes and keep(b) >= keep_ref - agree at both alphas; falls: (a) differs by more than `differ` at
    a scored size, or keep(b) <= keep_1000 + near at both alphas; else mixed. With an unscored cell, stands is out
    ('mixed at best'): only a falls already met by the scored cells is returned, else mixed. Both met: mixed."""
    a = [c for c in s["a"].values() if c is not None]
    b = [c for c in s["b"].values() if c is not None]
    full = len(a) == len(s["a"]) and len(b) == len(s["b"])
    a_falls = any(abs(c["keep"] - c["keep_ref"]) > differ for c in a)
    b_falls = len(b) == len(s["b"]) and all(c["keep"] <= s["keep_1000"] + near for c in b)
    stands = full and all(abs(c["keep"] - c["keep_ref"]) <= agree for c in a) and all(
        c["keep"] >= c["keep_ref"] - agree for c in b)
    falls = a_falls or b_falls
    return "mixed" if stands and falls else "stands" if stands else "falls" if falls else "mixed"


def t3(seeds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per = {k: t3_seed(s) for k, s in seeds.items()}
    vals = set(per.values())
    return {"outcome": vals.pop() if len(vals) == 1 and len(per) == 2 else "mixed", "per_seed": per}


# ----------------------------------------------------------------------------- T4, t4b


def t4(nothing: Rec, plus: Rec, q_min: float = 1.1, q_falls: float = 1.0) -> dict[str, Any]:
    """q_s = KL_plus(s) / KL_nothing(s). stands: min_s ratio_plus > max_s ratio_nothing and q_s >= q_min at every
    start; falls: q_s <= q_falls at some start (a tie at 1.0 falls); else unsettled. A missing start, a non-finite
    value or KL_nothing <= 0 leaves the stage unscored."""
    starts = list(nothing["kl"])
    scorable = (nothing["complete"] and plus["complete"] and set(plus["kl"]) == set(starts)
          and all(ok(x) for r in (nothing, plus) for x in [*r["kl"].values(), *r["ratio"].values()])
          and all(nothing["kl"][s] > 0 for s in starts))
    if not scorable:
        return {"outcome": "unscored", "q": None}
    q = {s: plus["kl"][s] / nothing["kl"][s] for s in starts}
    if any(v <= q_falls for v in q.values()):
        outcome = "falls"
    elif min(plus["ratio"].values()) > max(nothing["ratio"].values()) and all(v >= q_min for v in q.values()):
        outcome = "stands"
    else:
        outcome = "unsettled"
    return {"outcome": outcome, "q": q}


# ----------------------------------------------------------------------------- T5 (and T5 with t5b)


def cell_pass(cell: dict[str, Any], f: float, p_min: float = 0.9) -> str:
    """'pass' (P >= p_min and R_max passes), 'unscored' (P >= p_min and R_max unscored), else 'not' (P < p_min, not
    read, or R_max fail or inconclusive)."""
    if cell["p"] < p_min:
        return "not"
    v = r_max(cell["rec"], f)
    return "pass" if v == "pass" else "unscored" if v == "unscored" else "not"


def t5(cells: dict[str, dict[str, str]], n_read: dict[str, int]) -> dict[str, Any]:
    """cells[teacher][cell] = `cell_pass`; n_read[teacher] = cells with P >= 0.9. falls: some cell passes on both
    teachers; unsettled: fewer than two cells per teacher have P >= 0.9, or an unscored cell could still make it falls
    (its other teacher passes or is unscored); mixed: some cell passes on exactly one teacher; stands: none passes."""
    a, b = cells["adam"], cells["lora"]
    names = sorted(set(a) & set(b))
    if any(a[c] == "pass" and b[c] == "pass" for c in names):
        outcome = "falls"
    elif min(n_read.values()) < 2 or any("unscored" in (a[c], b[c]) and "not" not in (a[c], b[c]) for c in names):
        outcome = "unsettled"
    elif any("pass" in (a[c], b[c]) for c in names):
        outcome = "mixed"
    else:
        outcome = "stands"
    return {"outcome": outcome, "cells": cells, "n_read": n_read}


# ----------------------------------------------------------------------------- t4c (Amendment 2)


def reproduces(kl: dict[str, Any], ref: dict[str, Any], tol: float) -> bool:
    """Every start of `ref` read here, both finite and positive, and |kl / ref - 1| <= tol; an empty `ref` fails."""
    return bool(ref) and set(kl) == set(ref) and all(
        ok(kl[s]) and ok(ref[s]) and ref[s] > 0 and abs(kl[s] / ref[s] - 1) <= tol for s in ref)


def t4c(nothing: Rec, tt: Rec, plus: Rec) -> dict[str, Any]:
    """Per start s, plus remainder's ratio against the two others: 'between' when min(tt, nothing) <= plus <= max(tt,
    nothing) (ties inclusive), 'above' when above both, 'below' when below both. Stage: that label if all three starts
    share it, else 'mixed'; any missing or non-finite ratio, or a start not read on all three states: 'unscored'.
    Reported beside it: whether teacher-trained reads below do-nothing at each start (E43's direction)."""
    recs = (nothing, tt, plus)
    starts = list(nothing["ratio"])
    if not starts or not all(r["complete"] and set(r["ratio"]) == set(starts) and all(ok(x) for x in r["ratio"].values())
                             for r in recs):
        return {"outcome": "unscored", "per_start": None, "tt_below_nothing": None}
    per: dict[str, str] = {}
    for s in starts:
        n, t, p = (r["ratio"][s] for r in recs)
        per[s] = "above" if p > max(n, t) else "below" if p < min(n, t) else "between"
    labels = set(per.values())
    return {"outcome": labels.pop() if len(labels) == 1 else "mixed", "per_start": per,
            "tt_below_nothing": {s: tt["ratio"][s] < nothing["ratio"][s] for s in starts}}
