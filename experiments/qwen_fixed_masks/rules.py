"""E72's filed rules (results/predictions.json) as pure functions of the reads' JSON; no torch, no GPU. An update's record
`u` maps a read's name to its JSON {"T": 256, "sums": {key: per-row sums}}, plus "S" (stage S's summary). `w0` is E68's W0.json
for the same rows (sums nll, clean, ci, ft). Comparisons round to 9 places. `w` None: not read yet.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

NAN = float("nan")
GAIN_MIN, REP_TOL, ROOM_MIN, GAP_MARGIN, BOOT = 0.05, 1e-3, 0.1, 0.2, 2000
FT_TOL, G_TOL = 1e-3, 1e-6  # code checks, not rules
T1 = ("do_nothing", "ceiling", "upper", "lower", "random", "ls", "adjoint")
T2 = ("W0", "FT", "inspan", "remainder", "upper", "lower", "random")


def ok(x: float) -> bool:
    return math.isfinite(x)


def r9(x: float) -> float:
    return round(x, 9)


def div(x: float, y: float) -> float:
    return x / y if ok(x) and ok(y) and y != 0 else NAN


# ----------------------------------------------------------------------------- the rules, on plain numbers


def room_rule(ceil: float, floor: float, w: float | None) -> str:
    gap = ceil - floor
    if not ok(gap):
        return "unreadable"
    if r9(gap) <= ROOM_MIN:
        return "no room"
    if w is None:
        return "pending"
    if not ok(w):
        return "unreadable"
    return "room" if r9(gap) > r9(2 * w) else "no room"


def teeth_rule(up: float, lo: float, rnd: float, w: float) -> str:
    if not all(ok(x) for x in (up, lo, rnd, w)):
        return "unreadable"
    return "holds" if r9(up - max(lo, rnd)) > r9(2 * w) else "no teeth"


def label_rule(g: float, ceil: float, floor: float, lo: float, rnd: float, w: float) -> str:
    """Scored only where room and teeth hold (the caller). Both ways: fails."""
    if not all(ok(x) for x in (g, ceil, floor, lo, rnd, w)):
        return "unreadable"
    if r9(g) <= r9(max(floor, lo, rnd) + w):
        return "fails"
    return "passes" if r9(g) >= r9(ceil - w) else "partial"


def t2_rule(inspan: float, rem: float, s: float, w: float) -> str:
    if not all(ok(x) for x in (inspan, rem, s, w)):
        return "unreadable"
    gap = inspan - s
    if r9(gap) > r9(2 * w) and r9(rem) < r9(inspan):
        return "carried"
    return "by norm" if r9(abs(gap)) <= r9(2 * w) else "not carried"


def specific_rule(r2_r: float, r2_c: float) -> str:
    if not (ok(r2_r) and ok(r2_c)):
        return "unreadable"
    gap = r9(r2_r - r2_c)
    return "this update" if gap > GAP_MARGIN else "any change" if abs(gap) <= GAP_MARGIN else "random lower"


# ----------------------------------------------------------------------------- on the reads' JSON


def tot(rd: dict, key: str, idx: list[int] | None = None) -> float:
    xs = rd["sums"][key]
    return sum(xs) if idx is None else sum(xs[i] for i in idx)


def mean(rd: dict, key: str) -> float:
    """Over every position, as E68's K."""
    xs = rd["sums"][key]
    return sum(xs) / (len(xs) * rd["T"])


def gain(w0: dict, st: dict, zero: str, idx: list[int] | None = None) -> float:
    """zero "nll": T1, E68's gain_kept, zero at masked W0. zero "clean": T2, zero at clean W0."""
    return div(tot(w0, zero, idx) - tot(st, "nll", idx), tot(w0, "clean", idx) - tot(w0, "ft", idx))


def scored(w0: dict) -> bool:
    den = (sum(w0["sums"]["clean"]) - sum(w0["sums"]["ft"])) / (len(w0["sums"]["ft"]) * (w0["T"] - 1))
    return ok(den) and r9(den) >= GAIN_MIN


def half_width(f: Callable[[list[int]], float], n: int) -> float:
    """Half-width of the 95% row bootstrap of f(rows), seed 0, drawn and cut as E68's rules.half_width."""
    rng, xs = random.Random(0), []
    for _ in range(BOOT):
        x = f([rng.randrange(n) for _ in range(n)])
        if ok(x):
            xs.append(x)
    xs.sort()
    return (xs[int(0.975 * len(xs)) - 1] - xs[int(0.025 * len(xs))]) / 2 if xs else NAN


def width(w0: dict, up: dict | None, lo: dict | None, zero: str) -> float | None:
    if up is None or lo is None:
        return None
    return half_width(lambda i: gain(w0, up, zero, i) - gain(w0, lo, zero, i), len(w0["sums"]["ft"]))


def reference(w0_u1: dict, a_do_nothing: dict) -> dict[str, float]:
    return {"K": mean(a_do_nothing, "K"), "gain": gain(w0_u1, a_do_nothing, "nll")}


def control_reproduce(u: dict, w0: dict, ref: dict[str, float]) -> dict[str, Any]:
    if "do_nothing" not in u:
        return {"result": "not read", "missed": False}
    k, g = mean(u["do_nothing"], "K"), gain(w0, u["do_nothing"], "nll")
    gap_k, gap_g = abs(div(k, ref["K"]) - 1), abs(g - ref["gain"])
    missed = not (ok(gap_k) and ok(gap_g) and r9(gap_k) <= REP_TOL and r9(gap_g) <= REP_TOL)
    return {"K": k, "gain": g, "ref": ref, "gap_K": gap_k, "gap_gain": gap_g, "missed": missed}


def read_row(rd: dict, g: float) -> dict[str, float]:
    return {"gain": g, "K": mean(rd, "K"), "agree": tot(rd, "agree") / tot(rd, "count")}


def fixed_g(u: dict) -> dict[str, Any]:
    """Each T1 read's per-row g sums against do_nothing's: the same masks, or the test is not the filed one."""
    if "do_nothing" not in u:
        return {"gap": {}, "missed": False}
    base = u["do_nothing"]["sums"]["g_sum"]
    gap = {n: max(abs(div(a, b) - 1) for a, b in zip(u[n]["sums"]["g_sum"], base, strict=True)) for n in T1 if n in u}
    return {"gap": gap, "missed": any(not (ok(x) and x <= G_TOL) for x in gap.values())}


def score_t1(u: dict, w0: dict) -> dict[str, Any]:
    if not scored(w0):
        return {"room": {"result": "unscored"}, "teeth": {"result": "unscored"}, "states": {}}
    g = {n: gain(w0, u[n], "nll") for n in T1 if n in u}
    out: dict[str, Any] = {"states": {n: read_row(u[n], g[n]) for n in g}, "fixed_g": fixed_g(u)}
    w = width(w0, u.get("upper"), u.get("lower"), "nll")
    out["w"] = w
    room = room_rule(g["ceiling"], g["do_nothing"], w) if "ceiling" in g and "do_nothing" in g else "not read"
    out["room"] = {"result": room, "gap": g.get("ceiling", NAN) - g.get("do_nothing", NAN)}
    if not all(k in g for k in ("upper", "lower", "random")) or w is None:
        out["teeth"] = {"result": "not read"}
        return out
    teeth = teeth_rule(g["upper"], g["lower"], g["random"], w)
    out["teeth"] = {"result": teeth, "gap": g["upper"] - max(g["lower"], g["random"]), "w": w}
    for n in ("ls", "adjoint"):
        if n in g:
            good = room == "room" and teeth == "holds"
            out["states"][n]["label"] = (label_rule(g[n], g["ceiling"], g["do_nothing"], g["lower"], g["random"], w)
                                         if good else "unscored")
    return out


def score_t2(u: dict, w0: dict) -> dict[str, Any]:
    c = {n: u[f"clean_{n}"] for n in T2 if f"clean_{n}" in u}
    states = {n: read_row(rd, gain(w0, rd, "clean")) for n, rd in c.items()}
    for n in ("W0", "FT"):
        if n in c:
            states[n]["entropy"] = mean(c[n], "entropy")
    out: dict[str, Any] = {"states": states}
    if "W0" in c and "FT" in c:
        k_ft, k_w0 = states["FT"]["K"], states["W0"]["K"]
        out["clean_FT"] = {"K_FT": k_ft, "K_W0": k_w0, "missed": not (ok(k_ft) and ok(k_w0) and k_ft <= FT_TOL * k_w0)}
    if not all(k in c for k in ("inspan", "remainder", "upper", "lower")) or "S" not in u:
        return out | {"T2": "not read"}
    w = width(w0, c["upper"], c["lower"], "clean")
    s = 1 - float(u["S"]["share"].get("ls", NAN))
    assert w is not None
    res = t2_rule(states["inspan"]["gain"], states["remainder"]["gain"], s, w) if scored(w0) else "unscored"
    return out | {"T2": res, "s": s, "w": w}


def score_t0(u: dict, w0: dict, r2_c: float, k_c: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "T0a" in u:
        k_r0, k_r1, k0 = mean(u["T0a"], "K_R0"), mean(u["T0a"], "K_R1"), mean(w0, "ci")
        den = k_r0 - k0
        r2_r = div(k_r0 - k_r1, den)
        res = "unreadable" if not ok(den) else "unscored" if r9(den) < GAIN_MIN else specific_rule(r2_r, r2_c)
        out["specific"] = {"result": res, "r2_R": r2_r, "r2_C": r2_c, "K_R0": k_r0, "K_R1": k_r1, "K0": k0,
                           "K0_this_read": mean(u["T0a"], "ci"), "K_R_clean": mean(u["T0a"], "K_clean")}
    if "T0b" in u:
        out["T0b"] = {"K": mean(u["T0b"], "K"), "K_ceiling_W0_masks": k_c, "gain": gain(w0, u["T0b"], "nll"),
                      "agree": tot(u["T0b"], "agree") / tot(u["T0b"], "count")}
    return out


def score(res: dict[str, dict], w0s: dict[str, dict], ref: dict[str, float], e68: dict[str, dict]) -> dict[str, Any]:
    """`ref`: `reference`. `e68`: qwen_pass_test/results/amend1/score.json."""
    out: dict[str, Any] = {"control_reproduce": control_reproduce(res.get("u1", {}), w0s["u1"], ref)}
    for name, u in res.items():
        out[name] = {"T1": score_t1(u, w0s[name]), "T2": score_t2(u, w0s[name]),
                     "T0": score_t0(u, w0s[name], e68[name]["r2_C"], e68[name]["K_C"]),
                     "share_ls": u.get("S", {}).get("share", {}).get("ls")}
    return out
