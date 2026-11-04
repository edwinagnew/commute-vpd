"""E86's filed rules (results/predictions.json) as pure functions of the reads' JSON; no torch, no GPU. Plan:
reports/plans/2026-11-04_plan_e86-update-fractions.md. A clean read is {"T": 256, "sums": {"K": per-row sums, ...}}, from
FixedReader.clean; `fractions` maps a read's name (W0, FT, t0.1, t0.25, t0.5, t1.0, W0_end) to its mean response-token loss. A read not
yet made is not a miss and a rule that cannot read says "no call". Comparisons round to 9 places, as E75's."""
from __future__ import annotations

import importlib.util
import itertools
import math
import sys
from pathlib import Path
from typing import Any

spec = importlib.util.spec_from_file_location("rl_reads_rules_amend1_e86", Path(__file__).resolve().parent.parent / "qwen_rl_reads" / "rules_amend1.py")
assert spec is not None and spec.loader is not None
wrong = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = wrong
spec.loader.exec_module(wrong)

ok, r9, k_of, NO_CALL, NAN = wrong.ok, wrong.r9, wrong.k_of, wrong.NO_CALL, math.nan
SATURATED, BUILDS = "saturated", "builds"
TS = (0.1, 0.25, 0.5, 1.0)
GAIN_MIN = 0.05  # nats: E72's T2 and E75, a gain under this is unscored
SATURATED_AT, GRADED_AT = 0.9, 0.5  # predictions_amend2_fractions.json
T1_TOL = 1e-3  # the same file: "1.000 within 1e-3, or the path is wrong and no other read stands"; also W0 read again at 0
FACTOR = 10.0  # a control's band: this many measured floors (E75's own FACTOR), so a gap of one floor never stops a job
RULER_FLOOR, INSPAN_UNDER = 1e-5, 0.02  # E75's floor on K; the filed expected value for in-span
FT_ZERO = 1e-6  # K of the updated model against itself, nats: E44 Amendment 2's id_tol
LS_TOL, ADJ_UNDER = 0.01, 1e-6  # u1 against E80 Amendment 2: rules_amend2.E68_TOL; its 4.8e-8 at c within 0.95-1.11 of this job's c
PARTS = ("inspan", "upper", "lower", "adjoint", "remainder")  # good splits, no bar but in-span's; read and reported
WRONG = ("random", "signs")


def kept(x: float, w0: float, ft: float) -> float:
    """(w0 - x) / (w0 - ft): 0 the unchanged model, 1 the fine-tune. NaN if a loss is unread or the gain is not above zero."""
    if not all(ok(v) for v in (x, w0, ft)) or w0 - ft <= 0:
        return NAN
    return (w0 - x) / (w0 - ft)


def saturation(g: dict[str, float]) -> dict[str, Any]:
    """Filed on every update, total. s(0.1) at or above 0.9: "saturated": a tenth of the update keeps the gain, so the behaviour-kept
    column cannot tell an adjoint or teacher state from a tenth of the update. Otherwise "builds", with the older file's band:
    "graded" at or below 0.5, "partly" between. Label from t = 0.1 alone; above 1 still counts as at or above 0.9. No call: a loss unread
    or not finite, a gain under GAIN_MIN (zero and negative included), or t = 1 off 1 by over T1_TOL. A dip along t is flagged, never relabelled."""
    w0, ft = g.get("W0", NAN), g.get("FT", NAN)
    s = {t: kept(g.get(f"t{t}", NAN), w0, ft) for t in TS}
    out: dict[str, Any] = {"gain": w0 - ft, "kept": {str(t): s[t] for t in TS}, "label": NO_CALL, "band": NO_CALL, "non_monotone": False}
    if not all(ok(x) for x in s.values()) or r9(w0 - ft) < GAIN_MIN or r9(abs(s[1.0] - 1)) > T1_TOL:
        return out
    x = r9(s[0.1])
    values = [s[t] for t in TS]
    return out | {"label": SATURATED if x >= SATURATED_AT else BUILDS, "band": SATURATED if x >= SATURATED_AT else "graded" if x <= GRADED_AT else "partly",
                  "non_monotone": any(r9(b - a) < -T1_TOL for a, b in itertools.pairwise(values))}


def fractions_stop(g: dict[str, float]) -> dict[str, Any]:
    """Every update: t = 1 keeps 1, and W0 read again keeps 0, within T1_TOL. A read not yet made is not a miss."""
    w0, ft = g.get("W0", NAN), g.get("FT", NAN)
    got = {k: kept(g[n], w0, ft) for k, n in (("t1", "t1.0"), ("W0_end", "W0_end")) if n in g}
    want = {"t1": 1.0, "W0_end": 0.0}
    return {"got": got, "missed": any(not (ok(v) and r9(abs(v - want[k])) <= T1_TOL) for k, v in got.items())}


def fractions_control(g: dict[str, float], ref: dict[str, float]) -> dict[str, Any]:
    """u1 against E80 Amendment 2's reads on another card. The floor, nats, is the largest of W0 and FT's two reads in this job and their
    gaps to the reference; the band on a kept share is FACTOR floors over the gain. Unread until W0_end, the job's last read of this stage."""
    if not all(k in g for k in ("W0", "FT", "t0.1", "t0.25", "t1.0", "W0_end")):
        return {"result": "not read", "missed": False}
    gaps = [abs(g["W0"] - g["W0_end"]), abs(g["FT"] - g["t1.0"]), abs(g["W0"] - ref["W0"]), abs(g["FT"] - ref["FT"])]
    gain = g["W0"] - g["FT"]
    if not (all(ok(x) for x in gaps) and ok(gain) and gain > 0):
        return {"result": "unreadable", "floor": NAN, "missed": True}
    floor = max(gaps)
    band = FACTOR * floor / gain
    here = {t: kept(g[f"t{t}"], g["W0"], g["FT"]) for t in (0.1, 0.25, 1.0)}
    there = {t: kept(ref[f"t{t}"], ref["W0"], ref["FT"]) for t in here}
    gap = {t: abs(here[t] - there[t]) for t in here}
    return {"floor_nats": floor, "band": band, "here": {str(t): v for t, v in here.items()}, "there": {str(t): v for t, v in there.items()},
            "gap": {str(t): v for t, v in gap.items()}, "missed": any(not (ok(v) and r9(v) <= r9(band)) for v in gap.values())}


def table(u: dict[str, Any]) -> dict[str, Any]:
    """K as a fraction of the ruler, in-span's bar, and E75 Amendment 1's rule on the two wrong changes: above the remainder's K by more than the
    row bootstrap's half-width of that gap "fails". "works": in-span under its bar and both wrong changes fail. A ruler under the floor: no call."""
    ruler = k_of(u.get("clean_W0"))
    over = {s: wrong.fixed.div(k_of(u[f"clean_{s}"]), ruler) for s in (*PARTS, *WRONG) if f"clean_{s}" in u}
    if not (ok(ruler) and r9(ruler) >= RULER_FLOOR):
        return {"ruler": ruler, "over_ruler": over, "inspan": NO_CALL, "wrong": {}, "verdict": NO_CALL}
    x = over.get("inspan", NAN)
    inspan = "holds" if ok(x) and r9(x) < INSPAN_UNDER else "misses" if ok(x) else NO_CALL
    cells = {s: wrong.cell(u.get(f"clean_{s}"), u.get("clean_remainder"), ruler) for s in WRONG}
    labels = [c["label"] for c in cells.values()]
    verdict = NO_CALL if inspan == NO_CALL or NO_CALL in labels else "works" if inspan == "holds" and all(x == wrong.FAILS for x in labels) else "does not"
    return {"ruler": ruler, "over_ruler": over, "inspan": inspan, "wrong": cells, "verdict": verdict}


def table_control(u: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    """u1 and gold against E75's stored reads on the same card model. The floor is W0's K between two identical reads of this job;
    each state's K within FACTOR floors of the stored read, which at a floor of 0 is gap 0 to 9 places. Unread until both W0 reads are there."""
    if "clean_W0" not in u or "clean_W0_again" not in u:
        return {"result": "not read", "missed": False}
    floor = abs(k_of(u["clean_W0"]) - k_of(u["clean_W0_again"]))
    gap = {k: abs(k_of(u[k]) - k_of(v)) for k, v in ref.items() if k in u}
    return {"floor_nats": floor, "gap": gap, "missed": not ok(floor) or any(not (ok(v) and r9(v) <= r9(FACTOR * floor)) for v in gap.values())}


def left_control(left: dict[str, Any] | None, ls_ref: float) -> dict[str, Any]:
    """u1: least squares within LS_TOL of E80 Amendment 2's plain share; the adjoint's under ADJ_UNDER (this job's c is E75's, E80's its own)."""
    if left is None:
        return {"result": "not read", "missed": False}
    ls, adj = left["ls"], left["adjoint"]
    return {"ls": ls, "ls_ref": ls_ref, "adjoint": adj, "missed": not (ok(ls) and ok(adj) and r9(abs(ls / ls_ref - 1)) <= LS_TOL and r9(adj) < ADJ_UNDER)}


def anchor(g: dict[str, float], ref: dict[str, float]) -> dict[str, Any]:
    """Gold's W0 and FT against E61's reads of the same rows: 1e-3 relative, so the wrong rows or file stop the job in its first minute."""
    gap = {k: abs(g[k] / ref[k] - 1) for k in ref if k in g}
    return {"gap": gap, "missed": any(not (ok(v) and r9(v) <= T1_TOL) for v in gap.values())}


def score(u: dict[str, Any], table_ref: dict[str, Any] | None, u1_ref: dict[str, Any] | None, gold_ref: dict[str, float] | None) -> dict[str, Any]:
    """One update's record. `table_ref`: E75's stored reads, on u1 and gold. `u1_ref`: E80 Amendment 2's {"gold", "ls"}, on u1. `gold_ref`: E61's W0 and FT, on gold."""
    g, t = u.get("fractions", {}), table(u)
    ft = k_of(u.get("clean_FT"))
    sc = {"saturation": saturation(g), "fractions_stop": fractions_stop(g), "table": t, "left_in_delta": u.get("left_in_delta"),
          "clean_FT": {"K": ft, "missed": "clean_FT" in u and not (ok(ft) and r9(ft) <= FT_ZERO)}}
    if table_ref is not None:
        sc["table_control"] = table_control(u, table_ref)
    if gold_ref is not None:
        sc["gold_anchor"] = anchor(g, gold_ref)
    if u1_ref is not None:
        sc["fractions_control"] = fractions_control(g, u1_ref["gold"])
        sc["left_control"] = left_control(u.get("left_in_delta"), u1_ref["ls"])
    return sc


def missed(sc: dict[str, Any]) -> list[str]:
    return [k for k, v in sc.items() if isinstance(v, dict) and v.get("missed")]
