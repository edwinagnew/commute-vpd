"""E93's rules, verdicts and filed predictions, from the files run.py writes (CPU, no torch, no model). Plan:
reports/plans/2026-11-05_plan_e93-d1-table.md; lines: results/predictions.json (its "thresholds" must equal these).

  python experiments/qwen_d1_table/rules.py           # results/table.json from the states written so far
  python experiments/qwen_d1_table/rules.py --edges   # the plan's edge cases through these functions, before filing

A state's record, results/<state>/state.json ("done" once every read is in). Per-row sums are lists
[whole, cut, fragment, other, pooled] (`commute.qwen_d3.by_class`); "n" holds each row's position counts in that layout
("kl": every position, "nll": those with a next token). "fractions": mean response-token loss of W0, FT, W0 + t δW
(t1.0 ... t0.1), W0 read again (W0_end) and W0 + the moved part (moved), no masks. "clean": per read (W0, FT, random,
signs, moved, remainder; inspan_old on anchor 2 seed 42) the per-row K = KL(FT clean ‖ read clean) and nll. "split":
fp64 sums over matrices of ‖δW‖² (size), ‖δW − moved‖² (left), ‖moved‖² (moved) and null_s ‖δW_s‖² (null). "base":
masked cells with the components as W0's (K00 W0 masked, K01 do-nothing; nll_W0, nll_FT, nll_m00, nll_m01). "own": least
squares, Γ reading its moved V′ (K11, nll_m11: the state's own masks; K10, nll_m10: masks of the W0 weights' forward).
On a graded arm, "ceiling" (K_c, nll_c) and "random" (nll_m11 of a Gaussian at δW's norm, placed by least squares).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
WHOLE, POOLED = 0, 4
TS = (1.0, 0.5, 0.25, 0.1)
THRESHOLDS = {"gain_min": 0.05, "graded_at": 0.5, "saturated_at": 0.9, "t1_tol": 1e-3, "ruler_floor": 1e-5,
              "at_floor": 1e-4, "clean_under": 0.02, "size_max": 1.0, "room_min": 0.1, "min_rows": 2, "boot": 2000}
T = THRESHOLDS
HOLDS, MISSES, FAILS, NO_CALL, PENDING = "holds", "misses", "fails", "no call", "pending"
NOT_GRADED, NO_LINE, NO_ROOM = "not graded", "no behaviour line", "no room"

# name -> (E91 arm, rows, arm label). The full update of each arm, size_grid.json `.arms.<arm>.full` (u1: its fp32 file).
STATES: dict[str, tuple[str, str, str]] = {
    "a1_s0": ("anchor1_s0", "a1", "anchor 1"), "a1_s1": ("anchor1_s1", "a1", "anchor 1"),
    "a2_s42": ("anchor2_s0", "tofu", "anchor 2"), "a2_s142": ("anchor2_s142", "tofu", "anchor 2"),
    "R_s0": ("R_s0", "maths", "RL"), "R_s70": ("R_s70", "maths", "RL"),
    "N_s0": ("N_s0", "maths", "null"), "N_s70": ("N_s70", "maths", "null"),
    "F_s0": ("F_s0", "maths", "own-samples"), "u1": ("u1", "maths", "u1"), "u2": ("u2", "maths", "u2")}
BEHAVIOUR = ("anchor 1", "anchor 2", "u1", "u2")  # GSM8K's RL arms' gold-token gain is not their behaviour (plan)
NULL_ARM = "null"  # a one-seed arm borrows its seed spread


def ok(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def r9(x: float) -> float:
    return round(x, 9)


def div(a: float, b: float) -> float | None:
    return a / b if ok(a) and ok(b) and b != 0 else None


def col(sums: dict[str, list[list[float]]], key: str, c: int) -> np.ndarray:
    return np.array([r[c] for r in sums[key]], dtype=np.float64)


def half_width(num: np.ndarray, den: np.ndarray, seed: int = 0) -> float | None:
    """Half the 2.5-97.5 range of sum(num)/sum(den) over T["boot"] row resamples, paired by row; None under min_rows."""
    n = len(num)
    if n < T["min_rows"]:
        return None
    W = np.apply_along_axis(np.bincount, 1, np.random.default_rng(seed).integers(0, n, (T["boot"], n)), minlength=n)
    D = W @ den
    r = np.sort((W @ num)[D != 0] / D[D != 0])
    return float((r[int(0.975 * len(r)) - 1] - r[int(0.025 * len(r))]) / 2) if len(r) else None


# ----------------------------------------------------------------------------- fractions and kept, no masks


def kept(x: Any, w0: Any, ft: Any) -> float | None:
    """(w0 − x) / (w0 − ft): 0 the unchanged model, 1 the fine-tune; None if unread or the gain is not above zero."""
    return (w0 - x) / (w0 - ft) if ok(x) and ok(w0) and ok(ft) and w0 - ft > 0 else None


def saturation(fr: dict[str, Any]) -> dict[str, Any]:
    """E86's band from t = 0.1: "graded" at or under 0.5, "saturated" at or over 0.9, "partly" between. No call: a gain
    under gain_min, a read missing, or t = 1 off 1 by over t1_tol."""
    w0, ft = fr.get("W0", math.nan), fr.get("FT", math.nan)
    s = {t: kept(fr.get(f"t{t}"), w0, ft) for t in TS}
    out = {"gain": w0 - ft if ok(w0) and ok(ft) else None, "kept": {str(t): v for t, v in s.items()}, "band": NO_CALL}
    if out["gain"] is None or r9(out["gain"]) < T["gain_min"] or not all(ok(v) for v in s.values()):
        return out
    if r9(abs(s[1.0] - 1)) > T["t1_tol"]:  # type: ignore[operator]
        return out
    x = r9(s[0.1])  # type: ignore[arg-type]
    return out | {"band": "graded" if x <= T["graded_at"] else "saturated" if x >= T["saturated_at"] else "partly"}


# ----------------------------------------------------------------------------- the six rules


def kmean(sums: dict[str, Any], n: list[list[float]], c: int, key: str = "K") -> float | None:
    return div(float(col(sums, key, c).sum()), float(sum(r[c] for r in n)))


def e75_label(state: dict[str, Any], rem: dict[str, Any], n: list[list[float]]) -> dict[str, Any]:
    """E75 Amendment 1: a bad placement "fails" if its K is above the remainder's by more than the row bootstrap's
    half-width of the gap, "passes" if below by more, otherwise no call."""
    num, den = col(state, "K", POOLED) - col(rem, "K", POOLED), np.array([r[POOLED] for r in n], dtype=np.float64)
    gap, w = div(float(num.sum()), float(den.sum())), half_width(num, den)
    if gap is None or w is None:
        return {"gap": gap, "width": w, "label": NO_CALL}
    return {"gap": gap, "width": w, "label": FAILS if r9(gap) > r9(w) else "passes" if r9(-gap) > r9(w) else NO_CALL}


def rule1(rec: dict[str, Any]) -> dict[str, Any]:
    """Clean: K(W0 + moved part) under clean_under of the ruler K(W0); K at or under at_floor nats "at floor"; a ruler
    under ruler_floor, or random or signs not failing E75's rule here, no call. A tie at the line misses."""
    cl, n = rec.get("clean", {}), rec["n"]["kl"]
    if any(k not in cl for k in ("W0", "moved", "remainder", "random", "signs")):
        return {"result": PENDING}
    k = {name: kmean(v, n, POOLED) for name, v in cl.items()}
    teeth = {w: e75_label(cl[w], cl["remainder"], n) for w in ("random", "signs")}
    ruler, x = k["W0"], k["moved"]
    out = {"K": k, "over_ruler": div(x, ruler) if ruler else None, "teeth": teeth}  # type: ignore[arg-type]
    if not (ok(ruler) and r9(ruler) >= T["ruler_floor"]) or not ok(x):  # type: ignore[arg-type]
        return out | {"result": NO_CALL, "why": "ruler at floor or unread"}
    if r9(x) <= T["at_floor"]:  # type: ignore[arg-type]
        return out | {"result": "at floor"}
    if any(t["label"] != FAILS for t in teeth.values()):
        return out | {"result": NO_CALL, "why": "random or signs does not fail here"}
    return out | {"result": HOLDS if r9(x / ruler) < T["clean_under"] else MISSES}  # type: ignore[operator]


def share(rec: dict[str, Any]) -> tuple[float | None, float | None]:
    s = rec.get("split") or {}
    return div(s.get("left"), s.get("size")), div(s.get("null"), s.get("size"))  # type: ignore[arg-type]


def rule2(rec: dict[str, Any], spread: float | None) -> dict[str, Any]:
    """Share left in Δ: below its null by more than the arm's seed spread. A tie misses."""
    v, null = share(rec)
    out = {"value": v, "null": null, "spread": spread}
    if not (ok(v) and ok(null)):
        return out | {"result": NO_CALL if "split" in rec else PENDING}
    if spread is None:
        return out | {"result": PENDING}
    return out | {"result": HOLDS if r9(null - v) > r9(spread) else MISSES}  # type: ignore[operator]


def rule3(rec: dict[str, Any]) -> dict[str, Any]:
    """Squared size ‖moved‖² / ‖δW‖² at or under size_max."""
    if "split" not in rec:
        return {"result": PENDING}
    x = div(rec["split"]["moved"], rec["split"]["size"])
    return {"value": x, "result": NO_CALL if x is None else HOLDS if r9(x) <= T["size_max"] else MISSES}


def rule4(rec: dict[str, Any], sat: dict[str, Any]) -> dict[str, Any]:
    """Kept, clean: on a graded arm, kept by W0 + the moved part at least the arm's own t = 0.5. A tie holds."""
    if rec["arm"] not in BEHAVIOUR:
        return {"result": NO_LINE}
    if sat["band"] == NO_CALL:
        return {"result": NO_CALL}
    if sat["band"] != "graded":
        return {"result": NOT_GRADED}
    fr = rec["fractions"]
    x, line = kept(fr.get("moved"), fr["W0"], fr["FT"]), sat["kept"]["0.5"]
    return {"value": x, "line": line, "result": NO_CALL if x is None else HOLDS if r9(x) >= r9(line) else FAILS}


def rule5(rec: dict[str, Any], c: int) -> dict[str, Any]:
    """K under own masks, least squares (K11) not above do-nothing (K01) by over twice the row half-width. A tie holds."""
    if "own" not in rec:
        return {"result": PENDING}
    num = col(rec["own"], "K11", c) - col(rec["base"], "K01", c)
    den = np.array([r[c] for r in rec["n"]["kl"]], dtype=np.float64)
    keep = den > 0
    diff, w = div(float(num.sum()), float(den.sum())), half_width(num[keep], den[keep])
    if diff is None or w is None:
        return {"diff": diff, "width": w, "result": NO_CALL}
    return {"diff": diff, "width": w, "result": HOLDS if r9(diff) <= r9(2 * w) else MISSES}


def masked_gains(rec: dict[str, Any], c: int) -> tuple[dict[str, float] | None, float | None]:
    """E68's gain, zero at masked W0: (nll_m00 − nll of the read) / (nll_W0 − nll_FT), summed over rows in class c;
    and the row half-width of least squares' minus do-nothing's. None under min_rows rows or a gain under gain_min."""
    b, rd = rec["base"], {"do_nothing": rec["base"]["nll_m01"], "ls": rec["own"]["nll_m11"], "ls_w0_masks": rec["own"]["nll_m10"]}
    if "ceiling" in rec:
        rd["ceiling"] = rec["ceiling"]["nll_c"]
    if "random" in rec:
        rd["random"] = rec["random"]["nll_m11"]
    den, m00 = col(b, "nll_W0", c) - col(b, "nll_FT", c), col(b, "nll_m00", c)
    npos = np.array([r[c] for r in rec["n"]["nll"]], dtype=np.float64)
    keep = npos > 0
    if keep.sum() < T["min_rows"] or r9(float(den.sum()) / float(npos.sum())) < T["gain_min"]:
        return None, None
    g = {k: float((m00 - np.array([r[c] for r in v])).sum() / den.sum()) for k, v in rd.items()}
    dn, ls = (np.array([r[c] for r in rd[k]], dtype=np.float64) for k in ("do_nothing", "ls"))
    return g, half_width((dn - ls)[keep], den[keep])


def rule6(rec: dict[str, Any], c: int, sat: dict[str, Any]) -> dict[str, Any]:
    """Kept under own masks, graded arms: E72's room rule (ceiling over do-nothing by over room_min and twice the
    half-width), then its label with no rarely-on half: fails at or under max(do-nothing, random) + w, passes at or
    over ceiling − w, partial between. "no room" does not fail: the state is judged on rule 4 alone."""
    r4 = rule4(rec, sat)["result"]
    if r4 in (NO_LINE, NO_CALL, NOT_GRADED) or "own" not in rec:
        return {"result": r4 if "own" in rec else PENDING}
    if "ceiling" not in rec or "random" not in rec:
        return {"result": PENDING}
    g, w = masked_gains(rec, c)
    if g is None or w is None:
        return {"gains": g, "result": NO_CALL}
    gap, out = g["ceiling"] - g["do_nothing"], {"gains": g, "width": w}
    if r9(gap) <= T["room_min"] or r9(gap) <= r9(2 * w):
        return out | {"result": NO_ROOM}
    if r9(g["ls"]) <= r9(max(g["do_nothing"], g["random"]) + w):
        return out | {"result": FAILS}
    return out | {"result": "passes" if r9(g["ls"]) >= r9(g["ceiling"] - w) else "partial"}


# ----------------------------------------------------------------------------- one state, the arms, the table


def verdict(r: dict[str, str]) -> dict[str, str]:
    """Fails on a miss of 1, 2, 3 or 5, or "fails" on 4 or 6. Passes when 1 holds or is at floor and 2, 3, 5 hold;
    no call when one of those cannot read; pending while a read is missing."""
    core = [r["1"], r["2"], r["3"], r["5"]]
    if MISSES in core or FAILS in (r["4"], r["6"]):
        return {"verdict": FAILS}
    if PENDING in [*core, r["4"], r["6"]]:
        return {"verdict": PENDING}
    if r["1"] not in (HOLDS, "at floor") or any(x != HOLDS for x in core[1:]):
        return {"verdict": NO_CALL}
    note = ("passes on the unmasked rule alone: no masked read can grade this update" if r["6"] == NO_ROOM
            else "no behaviour line" if r["4"] in (NO_LINE, NOT_GRADED, NO_CALL) else "")
    return {"verdict": "passes", "note": note}


def printed(rec: dict[str, Any], c: int) -> dict[str, Any]:
    """Unscored: masked K and kept, pooled and on the scored class; least squares under W0's masks; size against 1 − share."""
    out: dict[str, Any] = {}
    if "own" in rec:
        for name, cc in (("scored", c), ("pooled", POOLED)):
            g, _ = masked_gains(rec, cc)
            out[name] = {"K_W0_masked": kmean(rec["base"], rec["n"]["kl"], cc, "K00"),
                         "K_do_nothing": kmean(rec["base"], rec["n"]["kl"], cc, "K01"),
                         "K_ls": kmean(rec["own"], rec["n"]["kl"], cc, "K11"),
                         "K_ls_w0_masks": kmean(rec["own"], rec["n"]["kl"], cc, "K10"), "kept": g}
    v, _ = share(rec)
    if v is not None and "split" in rec:
        out["size_minus_one_minus_share"] = rec["split"]["moved"] / rec["split"]["size"] - (1 - v)
    for k in ("split_bf16", "seconds"):
        if k in rec:
            out[k] = rec[k]
    return out


def score_state(rec: dict[str, Any], spread: float | None) -> dict[str, Any]:
    c = WHOLE if rec["rows"] in ("a1", "tofu") else POOLED
    sat = saturation(rec.get("fractions", {}))
    rs = {"1": rule1(rec), "2": rule2(rec, spread), "3": rule3(rec), "4": rule4(rec, sat), "5": rule5(rec, c),
          "6": rule6(rec, c, sat)}
    return {"arm": rec["arm"], "saturation": sat, "rules": rs, **verdict({k: v["result"] for k, v in rs.items()}),
            "printed": printed(rec, c)}


def spreads(recs: dict[str, dict[str, Any]]) -> dict[str, tuple[float | None, bool]]:
    """Per arm: |share(seed a) − share(seed b)|, None until both are written; a one-seed arm borrows the null arm's."""
    by_arm: dict[str, list[float | None]] = {}
    for n, (_, _, arm) in STATES.items():
        by_arm.setdefault(arm, []).append(share(recs[n])[0] if n in recs else None)
    own = {a: abs(v[0] - v[1]) if len(v) == 2 and all(ok(x) for x in v) else None for a, v in by_arm.items()}  # type: ignore[operator]
    return {a: (own[NULL_ARM], True) if len(v) == 1 else (own[a], False) for a, v in by_arm.items()}


def arms(states: dict[str, dict[str, Any]]) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    for n, (_, _, arm) in STATES.items():
        out.setdefault(arm, []).append(states[n]["verdict"] if n in states else PENDING)
    return {a: FAILS if FAILS in v else "passes" if set(v) == {"passes"} else PENDING if PENDING in v else NO_CALL
            for a, v in out.items()}


def predictions(states: dict[str, dict[str, Any]]) -> dict[str, str]:
    def every(names: list[str], test: Any) -> str:
        if any(n not in states for n in names):
            return "unread"
        return HOLDS if all(test(states[n]) for n in names) else MISSES

    res = lambda k: (lambda s: s["rules"][k]["result"])
    beh = [n for n, (_, _, a) in STATES.items() if a in BEHAVIOUR]
    return {"P1_size": every(list(STATES), lambda s: res("3")(s) == HOLDS),
            "P2_share": every(list(STATES), lambda s: res("2")(s) == HOLDS),
            "P3_clean": every(list(STATES), lambda s: res("1")(s) in (HOLDS, "at floor")),
            "P4_masked_K": every(list(STATES), lambda s: res("5")(s) == HOLDS),
            "P5_graded": every(beh, lambda s: (s["saturation"]["band"] == "graded") == (s["arm"] == "anchor 2")),
            "P6_no_room": every(["a2_s42", "a2_s142"], lambda s: res("6")(s) == NO_ROOM),
            "P7_table": every(list(STATES), lambda s: s["verdict"] == "passes")}


def load(name: str) -> dict[str, Any] | None:
    p = RESULTS / name / "state.json"
    rec = json.loads(p.read_text()) if p.is_file() else None
    return rec if rec is not None and rec.get("done") else None


def score() -> dict[str, Any]:
    P = json.loads((RESULTS / "predictions.json").read_text())
    assert P["thresholds"] == THRESHOLDS, "predictions.json's thresholds differ from rules.THRESHOLDS"
    recs = {n: r for n in STATES if (r := load(n)) is not None}
    sp = spreads(recs)
    states = {n: score_state(r, sp[r["arm"]][0]) | {"spread_borrowed": sp[r["arm"]][1]} for n, r in recs.items()}
    a = arms(states)
    table = {"states": states, "arms": a, "predictions": predictions(states),
             "table": FAILS if FAILS in a.values() else "passes" if set(a.values()) == {"passes"} else PENDING}
    tmp = RESULTS / "table.json.tmp"
    tmp.write_text(json.dumps(table, indent=1, default=float) + "\n")
    tmp.replace(RESULTS / "table.json")
    return table


# ----------------------------------------------------------------------------- the plan's edge cases


def fake(rows: int = 4, k: dict[str, float] | None = None, fr: dict[str, float] | None = None,
         split: dict[str, float] | None = None, nll: dict[str, float] | None = None, arm: str = "anchor 2",
         masked: bool = True) -> dict[str, Any]:
    """A TOFU-rows record whose every row reads these per-position values, every position whole-question."""
    k = {"W0": 1.0, "FT": 0.0, "moved": 0.01, "remainder": 0.5, "random": 2.0, "signs": 2.0, "K01": 0.1, "K11": 0.1} | (k or {})
    fr = {"W0": 3.0, "FT": 1.0, "t1.0": 1.0, "t0.5": 1.5, "t0.25": 2.0, "t0.1": 2.5, "W0_end": 3.0, "moved": 1.2} | (fr or {})
    nl = {"nll_W0": 3.0, "nll_FT": 1.0, "nll_m00": 3.0, "nll_m01": 2.9, "nll_m11": 2.5, "nll_m10": 2.8, "nll_c": 2.0,
          "rand": 3.0} | (nll or {})
    row = lambda x, L: [[x * L, 0.0, 0.0, 0.0, x * L]] * rows
    rec: dict[str, Any] = {"arm": arm, "rows": "tofu", "fractions": fr, "done": True,
                           "n": {"kl": row(1.0, 256), "nll": row(1.0, 255)},
                           "clean": {n: {"K": row(k[n], 256)} for n in ("W0", "FT", "moved", "remainder", "random", "signs")},
                           "split": {"size": 1.0, "left": 0.1, "moved": 0.9, "null": 0.3} | (split or {})}
    if masked:
        rec["base"] = {"K00": row(1.0, 256), "K01": row(k["K01"], 256),
                       **{n: row(nl[n], 255) for n in ("nll_W0", "nll_FT", "nll_m00", "nll_m01")}}
        rec["own"] = {"K10": row(0.9, 256), "K11": row(k["K11"], 256), "nll_m11": row(nl["nll_m11"], 255),
                      "nll_m10": row(nl["nll_m10"], 255)}
        rec["ceiling"], rec["random"] = {"nll_c": row(nl["nll_c"], 255)}, {"nll_m11": row(nl["rand"], 255)}
    return rec


def edges() -> list[dict[str, Any]]:
    """Each case of the plan's table, the outcome it names, and what these functions give (rules 1-6, verdict)."""
    def got(rec: dict[str, Any], spread: float | None = 0.0) -> list[str]:
        s = score_state(rec, spread)
        return [s["rules"][k]["result"] for k in "123456"] + [s["verdict"]]

    flat = {"W0": 3.0, "FT": 3.0, "t1.0": 3.0, "t0.5": 3.0, "t0.25": 3.0, "t0.1": 3.0, "W0_end": 3.0, "moved": 3.0}
    dn_nll = {"nll_m11": 2.9, "nll_m10": 3.0}
    cases: list[tuple[str, Any, Any]] = [
        ("Least squares as the plan expects", got(fake()), [HOLDS] * 5 + ["partial", "passes"]),
        ("Do-nothing: nothing moved", got(fake(k={"moved": 1.0, "remainder": 0.0}, fr={"moved": 3.0}, split={"left": 1.0, "moved": 0.0},
                                               nll=dn_nll)), [MISSES, MISSES, HOLDS, FAILS, HOLDS, FAILS, FAILS]),
        ("Random update of δW's size: share at its null", got(fake(split={"left": 0.3, "moved": 0.7})),
         [HOLDS, MISSES, HOLDS, HOLDS, HOLDS, "partial", FAILS]),
        ("W0 as the state", got(fake(k={"W0": 0.0, "moved": 0.0, "remainder": 0.0, "random": 0.0, "signs": 0.0}, fr=flat,
                                     split={"size": 0.0, "left": 0.0, "moved": 0.0, "null": 0.0},
                                     nll={"nll_FT": 3.0, "nll_m01": 3.0, "nll_m11": 3.0})),
         [NO_CALL, NO_CALL, NO_CALL, NO_CALL, HOLDS, NO_CALL, NO_CALL]),
        ("Own-samples: ruler 0.001, K 2.2e-5", got(fake(k={"W0": 0.001, "moved": 2.2e-5}, arm="own-samples")),
         ["at floor", HOLDS, HOLDS, NO_LINE, HOLDS, NO_LINE, "passes"]),
        ("Gain under 0.05", got(fake(fr={"FT": 2.96, "t1.0": 2.96})), [HOLDS, HOLDS, HOLDS, NO_CALL, HOLDS, NO_CALL, "passes"]),
        ("Under 2 rows", got(fake(rows=1)), [NO_CALL, HOLDS, HOLDS, HOLDS, NO_CALL, NO_CALL, NO_CALL]),
        ("Random and signs below the remainder: no teeth", got(fake(k={"random": 0.1, "signs": 0.1})),
         [NO_CALL, HOLDS, HOLDS, HOLDS, HOLDS, "partial", NO_CALL]),
        ("Ties: clean 0.02, share null − spread, size 1, kept = t0.5, K diff 0, kept at the floor",
         got(fake(k={"moved": 0.02}, fr={"moved": 1.5}, split={"left": 0.1, "moved": 1.0, "null": 0.3}, nll={"nll_m11": 2.9}), 0.2),
         [MISSES, MISSES, HOLDS, HOLDS, HOLDS, FAILS, FAILS]),
        ("No room: passes on rule 4 alone", got(fake(nll={"nll_c": 2.75})), [HOLDS] * 5 + [NO_ROOM, "passes"]),
        ("Not graded: no behaviour line", got(fake(fr={"t0.1": 1.1})), [HOLDS] * 3 + [NOT_GRADED, HOLDS, NOT_GRADED, "passes"]),
        ("Graded, ceiling not read yet", got({k: v for k, v in fake().items() if k != "ceiling"}), [HOLDS] * 5 + [PENDING, PENDING]),
        ("Spread not yet read (other seed missing)", got(fake(), None), [HOLDS, PENDING, HOLDS, HOLDS, HOLDS, "partial", PENDING]),
        ("NaN share", got(fake(split={"left": float("nan")})), [HOLDS, NO_CALL, HOLDS, HOLDS, HOLDS, "partial", NO_CALL])]
    recs = {n: fake(split={"left": 0.1 + 0.01 * i}, arm=a) for i, (n, (_, _, a)) in enumerate(STATES.items())}
    sp = spreads(recs)
    cases.append(("One seed borrows the null arm's spread", [sp["own-samples"][1], round(sp["own-samples"][0] or 0, 9), sp["RL"][1]],
                  [True, 0.01, False]))
    out = [{"case": c, "got": g, "expected": w, "as_named": g == w} for c, g, w in cases]
    for r in out:
        print(f"{'ok ' if r['as_named'] else 'BAD'}  {r['case']}: {r['got']} (named {r['expected']})", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E93's scoring")
    ap.add_argument("--edges", action="store_true")
    if ap.parse_args().edges:
        sys.exit(0 if all(r["as_named"] for r in edges()) else 1)
    t = score()
    print(json.dumps({"arms": t["arms"], "table": t["table"], "predictions": t["predictions"]}, indent=1), flush=True)
