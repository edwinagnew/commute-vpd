"""E68's filed rules (results/predictions.json) as pure functions of the reads' JSON; nothing here touches a GPU. One
update's record `u` maps a read's name to its JSON: "W0" (per-row sums ci, K, nll, clean, ft), "do_nothing", "upper",
"lower", "random", "ls", "adjoint" and, on u1, "A_do_nothing", "A_lower", "A_upper" (sums K, nll, ft); "search"; "S".
Comparisons round to 9 places (r9). NaN or a missing value is a miss in the control and 'unreadable' elsewhere.
KD is read through W0's path, K_B(do-nothing) through the state path; the control holds the two equal (as E44 Amendment 2).
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from amend1 import NAN, div, ok, r9
from amend2 import L, boot_rows, kept, mean


@dataclass(frozen=True)
class Rules:
    line: float = 1.5  # the faithful line: K_B <= line * K0
    c_min: float = 0.05  # rho undefined at c <= c_min
    gain_min: float = 0.05  # nats: gain_kept unscored under this denominator
    rep_tol: float = 1e-3
    zero_tol: float = 1e-5
    boot: int = 2000

    def __post_init__(self) -> None:
        assert L.r2_floor == 0.05 and L.boot == self.boot  # amend2's kept and boot_rows, which the rules also fix


R = Rules()
REF_K = {"K0": 0.06961319516, "KD": 1.37381065}  # control_reproduce: relative
REF_R2 = {"lower": 0.00127701, "upper": 0.06176887, "random": 0.00534283}  # absolute
REF_SHARE = {"ls": 0.36374349, "inspan": 0.00114357, "adjoint": 0.00046571, "lower": 0.71197455, "upper": 0.52182008}
STATES = ("do_nothing", "upper", "lower", "random", "ls", "adjoint")

# ----------------------------------------------------------------------------- the rules, on plain numbers


def rho(r2: float, c: float) -> float:
    return r2 / c if ok(r2) and ok(c) and r9(c) > R.c_min else NAN


def faithful(k_b: float, k0: float) -> str:
    if not (ok(k_b) and ok(k0)):
        return "unreadable"
    return "faithful" if r9(k_b) <= r9(R.line * k0) else "not faithful"


def teeth_rule(kd: float, k0: float, lo: tuple[float, str], up: tuple[float, str], rnd: tuple[float, str],
               matched: bool, cl: float, cu: float, w: float) -> dict[str, Any]:
    """'holds', 'no teeth' (with why) or 'unreadable'. lo, up, rnd: (r2, status) as amend2's `kept` gives them."""

    def out(result: str, why: str = "") -> dict[str, Any]:
        return {"result": result, "why": why, "holds": result == "holds", "rho_lower": rl, "rho_upper": ru, "w": w}

    rl, ru = rho(lo[0], cl), rho(up[0], cu)
    if not (ok(kd) and ok(k0)):
        return out("unreadable", "K0 or KD is NaN")
    if r9(kd) <= r9(R.line * k0):
        return out("no teeth", "KD <= 1.5 K0: the update is too small for do-nothing to fail")
    if not matched:
        return out("no teeth", "random unmatched")
    if rnd[1] == "unscored":
        return out("no teeth", "r2_random unscored")
    if not (ok(cl) and ok(cu)):
        return out("unreadable", "a half's c is NaN")
    if r9(cl) <= R.c_min or r9(cu) <= R.c_min:
        return out("no teeth", "c <= 0.05 for a half")
    if not (ok(rl) and ok(ru) and ok(rnd[0]) and ok(w)):
        return out("unreadable", "rho, r2_random or w is NaN or unscored")
    gap = ru - max(rl, rnd[0])
    return out("holds") if r9(gap) > r9(2 * w) else out("no teeth", f"gap {gap:.6g} <= 2w = {2 * w:.6g}")


def label_rule(rho_s: float, t: dict[str, Any], r2_random: float) -> str:
    """A split's label; 'unscored' without teeth. Placed and lost both apply: lost."""
    if not t.get("holds"):
        return "unscored"
    if not ok(rho_s):
        return "unreadable"
    if r9(rho_s) <= r9(max(t["rho_lower"], r2_random) + t["w"]):
        return "lost"
    return "placed" if r9(rho_s) >= r9(t["rho_upper"] - t["w"]) else "partial"


def half_width(f: Callable[[list[int]], float], n: int) -> float:
    """Half-width of the 95% row bootstrap of f(rows), seed 0, drawn and cut as amend2.boot_rows."""
    rng, xs = random.Random(0), []
    for _ in range(R.boot):
        x = f([rng.randrange(n) for _ in range(n)])
        if ok(x):
            xs.append(x)
    xs.sort()
    return (xs[int(0.975 * len(xs)) - 1] - xs[int(0.025 * len(xs))]) / 2 if xs else NAN


def gain(w0: dict, st: dict, idx: list[int] | None = None) -> float:
    """gain_kept on the rows idx (all: None), from W0's read (its masked, clean and FT clean NLL) and a state's."""

    def tot(rd: dict, k: str) -> float:
        xs = rd["sums"][k]
        return sum(xs) if idx is None else sum(xs[i] for i in idx)

    return div(tot(w0, "nll") - tot(st, "nll"), tot(w0, "clean") - tot(w0, "ft"))


def gain_kept(w0: dict | None, st: dict | None) -> tuple[float, str]:
    if not (w0 and st):
        return NAN, "nan"
    den = (sum(w0["sums"]["clean"]) - sum(w0["sums"]["ft"])) / (len(w0["sums"]["ft"]) * (w0["T"] - 1))
    if not ok(den):
        return NAN, "nan"
    if r9(den) < R.gain_min:
        return NAN, "unscored"
    g = gain(w0, st)
    return g, "scored" if ok(g) else "nan"


def nll_mean(rd: dict | None, key: str) -> float:
    xs = (rd or {}).get("sums", {}).get(key, [])
    return sum(xs) / (len(xs) * (rd["T"] - 1)) if rd and xs else NAN

# ----------------------------------------------------------------------------- on the reads' JSON


def r2_of(u: dict, name: str) -> tuple[float, str]:
    """(KD - K_B) / (KD - K0), unscored under the 0.05 floor: amend2's `kept`, KD and K0 from the one W0 read."""
    w0 = u.get("W0")
    return kept({"controls": {"KD_t": w0, "W0_maths": w0}}, "t", mean(u.get(name), "K"))


def carried(u: dict, name: str) -> float:
    return 1 - float(u.get("S", {}).get("share", {}).get(name, NAN))


def teeth_of(u: dict) -> dict[str, Any]:
    if not all(k in u for k in ("W0", "S", "search", "upper", "lower", "random")):
        return {"result": "not read", "holds": False}
    cl, cu = carried(u, "lower"), carried(u, "upper")
    b = boot_rows({"KD_t": u["W0"], "W0_maths": u["W0"]}, {"lower": u["lower"], "upper": u["upper"]}, "t", cl, cu)
    w = (b["boot95"][1] - b["boot95"][0]) / 2
    t = teeth_rule(mean(u["W0"], "K"), mean(u["W0"], "ci"), r2_of(u, "lower"), r2_of(u, "upper"), r2_of(u, "random"),
                   u["search"]["matched"] is True, cl, cu, w)
    return t | {"boot95": b["boot95"], "rows_lower_above_upper": b["rows_lower_above_upper"], "a": u["search"]["a"],
                "kl_random": u["search"]["kl"], "kl_ft": u["search"]["target"], "matched": u["search"]["matched"]}


def score_update(u: dict) -> dict[str, Any]:
    if "W0" not in u:
        return {}
    w0 = u["W0"]
    k0, t, shares = mean(w0, "ci"), teeth_of(u), u.get("S", {}).get("share", {})
    out: dict[str, Any] = {"K0": k0, "KD": mean(w0, "K"), "nll_masked_W0": nll_mean(w0, "nll"),
                           "nll_clean_W0": nll_mean(w0, "clean"), "nll_clean_FT": nll_mean(w0, "ft"),
                           "shares": shares, "teeth": t, "states": {}}
    for n in (x for x in STATES if x in u):
        r2, status = r2_of(u, n)
        g, g_status = gain_kept(w0, u[n])
        out["states"][n] = {"K_B": mean(u[n], "K"), "r2": r2, "r2_status": status, "c": carried(u, n),
                            "rho": rho(r2, carried(u, n)), "gain_kept": g, "gain_status": g_status,
                            "faithful": faithful(mean(u[n], "K"), k0)}
        if n in ("ls", "adjoint"):
            out["states"][n]["label"] = label_rule(rho(r2, carried(u, n)), t, r2_of(u, "random")[0])
    return out


def control_reproduce(u: dict) -> dict[str, Any]:
    """On u1. A part whose read has not run is pending; a part read and off, or NaN, is missed."""
    w0, shares = u.get("W0"), u.get("S", {}).get("share")
    parts: list[tuple[str, bool, float, float, float, bool]] = [  # name, read, value, reference, tolerance, relative
        ("K0", w0 is not None, mean(w0, "ci"), REF_K["K0"], R.rep_tol, True),
        ("KD", w0 is not None, mean(w0, "K"), REF_K["KD"], R.rep_tol, True),
        ("r2_do_nothing", "do_nothing" in u, r2_of(u, "do_nothing")[0], 0.0, R.zero_tol, False)]
    parts += [(f"r2_{n}", n in u, r2_of(u, n)[0], v, R.rep_tol, False) for n, v in REF_R2.items()]
    parts += [(f"share_{n}", shares is not None, float((shares or {}).get(n, NAN)), v, R.rep_tol, False)
              for n, v in REF_SHARE.items()]
    out: dict[str, Any] = {"parts": {}, "missed": [], "pending": []}
    for name, read, x, ref, tol, rel in parts:
        if not read:
            out["pending"].append(name)
            continue
        gap = abs(x / ref - 1) if rel else abs(x - ref)
        out["parts"][name] = {"value": x, "ref": ref, "gap": gap, "tol": tol}
        if not ok(gap) or r9(gap) > tol:
            out["missed"].append(name)
    return out | {"holds": not out["missed"] and not out["pending"]}


def reading_a(u: dict) -> dict[str, Any]:
    """gain_kept(upper) - gain_kept(lower) under A against twice its bootstrap half-width and half the same under B."""
    if not all(k in u for k in ("W0", "lower", "upper", "A_do_nothing", "A_lower", "A_upper")):
        return {"result": "not read"}
    w0 = u["W0"]
    (ga_l, sa_l), (ga_u, sa_u) = gain_kept(w0, u["A_lower"]), gain_kept(w0, u["A_upper"])
    (gb_l, sb_l), (gb_u, sb_u) = gain_kept(w0, u["lower"]), gain_kept(w0, u["upper"])
    n, d_a, d_b = len(w0["sums"]["ft"]), ga_u - ga_l, gb_u - gb_l
    half = half_width(lambda i: gain(w0, u["A_upper"], i) - gain(w0, u["A_lower"], i), n)
    out = {"gain_A": {"do_nothing": gain_kept(w0, u["A_do_nothing"]), "lower": ga_l, "upper": ga_u},
           "gain_B": {"lower": gb_l, "upper": gb_u}, "d_A": d_a, "d_B": d_b, "half_width": half,
           "K_A": {n: mean(u[f"A_{n}"], "K") for n in ("do_nothing", "lower", "upper")}}
    if {sa_l, sa_u, sb_l, sb_u} != {"scored"} or not ok(half):
        return out | {"result": "unreadable"}
    return out | {"result": "separates" if r9(d_a) > r9(2 * half) and r9(d_a) > r9(d_b / 2) else "does not separate"}


def score(res: dict[str, dict]) -> dict[str, Any]:
    u1 = res.get("u1", {})
    return {"control_reproduce": control_reproduce(u1), "reading_A": reading_a(u1)} | {
        name: score_update(u) for name, u in res.items()}
