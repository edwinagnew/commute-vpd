"""E75's filed rules (results/predictions.json) as pure functions of the reads' JSON; no torch, no GPU. An update's record `u`
maps a read's name to its JSON: clean_<part> (E72's T2 read), fixed and stochastic (E74's read_group with one arm, "all"),
S (stage S's summary), search (E68's scale search). E72's and E74's rules are loaded, not copied. A rule that cannot read
says "no call". Comparisons round to 9 places, as E72's.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fixed = _load("fixed_rules", HERE.parent / "qwen_fixed_masks" / "rules.py")
trans = _load("transmission_rules", HERE.parent / "qwen_transmission" / "rules.py")

NO_CALL, TOL, FACTOR, FLOOR = "no call", 1e-3, 10.0, trans.FLOOR
ok, r9, div = fixed.ok, fixed.r9, fixed.div
W0_KEYS = ("nll", "entropy", "K0_fixed", "K0_stochastic")  # W0's own: the same on every update of these rows
CONTROL_READS = ("clean_W0", "fixed", "stochastic")
# Range over the SFT updates read so far, rounded outward: u1, u2, u3 from E72; transmission u1, u3 from E74 (u2 not read there).
SFT = {"tau_fixed": (0.0004, 0.0055), "tau_stochastic": (0.00107, 0.0128), "random_over_halves": (15.68, 29.61),
       "inspan_over_remainder": (0.0123, 0.0173), "share_ls": (0.351, 0.389)}


def transmission(tau: float) -> str:
    """E74's verdict on the whole update: "carries", "deaf", "between"; unread is no call."""
    label = trans.stochastic(tau) if ok(tau) else "unread"
    return NO_CALL if label == "unread" else label


def random_gap(k_random: float, k_upper: float, k_lower: float, k_w0: float, matched: bool) -> str:
    """Random's KL from FT at least FACTOR times the larger half's. No call: random off E68's match, a ruler under the floor,
    an unread number, or both halves at exactly zero."""
    top = max(k_upper, k_lower) if ok(k_upper) and ok(k_lower) else math.nan
    if not (matched and ok(k_random) and ok(top) and ok(k_w0)) or k_w0 < FLOOR or top <= 0:
        return NO_CALL
    return "holds" if r9(k_random) >= r9(FACTOR * top) else "misses"


def inspan_gap(k_inspan: float, k_remainder: float, k_w0: float) -> str:
    """In-span's KL from FT strictly below the remainder's. No call: a ruler under the floor, an unread number, or both under it."""
    if not all(ok(x) for x in (k_inspan, k_remainder, k_w0)) or k_w0 < FLOOR or max(k_inspan, k_remainder) < FLOOR:
        return NO_CALL
    return "holds" if r9(k_inspan) < r9(k_remainder) else "misses"


def readings(u: dict[str, Any]) -> dict[str, float]:
    """The control's numbers, from whichever of its reads exist."""
    out: dict[str, float] = {}
    if "clean_W0" in u:
        out |= {k: fixed.mean(u["clean_W0"], k) for k in ("K", "nll", "ft", "entropy")}
    for key in ("fixed", "stochastic"):
        if key in u:
            k0 = u[key]["base"]["K0"]
            out |= {f"tau_{key}": u[key]["summary"]["all"]["tau"], f"K0_{key}": sum(k0) / (len(k0) * u[key]["T"])}
    return out


def gaps(got: dict[str, float], ref: dict[str, float], keys: tuple[str, ...]) -> dict[str, Any]:
    gap = {k: abs(div(got[k], ref[k]) - 1) for k in keys if k in got and k in ref}
    return {"gap": gap, "missed": any(not (ok(x) and r9(x) <= TOL) for x in gap.values())}


def control(u1: dict[str, Any], ref: dict[str, float]) -> dict[str, Any]:
    """u1 through this code against E72's and E74's files, within TOL relative. Every read present, or not yet read."""
    if not all(k in u1 for k in CONTROL_READS):
        return {"result": "not read", "missed": False}
    got = readings(u1)
    return {"got": got, "ref": ref} | gaps(got, ref, tuple(ref))


def table(u: dict[str, Any]) -> dict[str, Any]:
    """E72's T2 on this update. W0's sums come from clean_W0 (nll: W0's, ft: FT's), the same numbers as E68's W0.json."""
    if "clean_W0" not in u:
        return {"states": {}}
    sums = u["clean_W0"]["sums"]
    w0 = {"T": u["clean_W0"]["T"], "sums": {"clean": sums["nll"], "ft": sums["ft"]}}
    out = fixed.score_t2(u, w0)
    if "clean_adjoint" in u:
        out["states"]["adjoint"] = fixed.read_row(u["clean_adjoint"], fixed.gain(w0, u["clean_adjoint"], "clean"))
    n = len(sums["ft"]) * (u["clean_W0"]["T"] - 1)
    return out | {"gain_den_nats": (sum(sums["nll"]) - sum(sums["ft"])) / n}


def row(label: str, value: float, lean: str | None, name: str) -> dict[str, Any]:
    lo, hi = SFT[name]
    sft = NO_CALL if not ok(value) else "inside" if lo <= value <= hi else "outside"
    return {"read": label, "value": value, "lean": lean, "as_lean": None if label == NO_CALL or lean is None else label == lean,
            "sft_range": [lo, hi], "against_sft": sft}


def score_update(u: dict[str, Any], u1: dict[str, Any], filed: bool = True) -> dict[str, Any]:
    """The filed rows on one update, each beside the SFT updates' range, and W0's own numbers against the control's. `filed`:
    a lean was filed for this update (arm R); the others are read against the same rules with none."""
    t2 = table(u)
    k = {n: t2["states"].get(n, {}).get("K", math.nan) for n in ("W0", "upper", "lower", "random", "inspan", "remainder")}
    got = readings(u)
    tau_f, tau_s = got.get("tau_fixed", math.nan), got.get("tau_stochastic", math.nan)
    top = max(k["upper"], k["lower"]) if ok(k["upper"]) and ok(k["lower"]) else math.nan
    matched = u.get("search", {}).get("matched") is True
    share = u.get("S", {}).get("share", {})
    deaf, holds = ("deaf", "holds") if filed else (None, None)
    return {"clean": t2, "share_ls": share.get("ls"), "share_adjoint": share.get("adjoint"), "search": u.get("search", {}).get("ratio"),
            "K0": {x: got.get(x) for x in ("K0_fixed", "K0_stochastic")}, "w0_same": gaps(got, readings(u1), W0_KEYS),
            "rows": {"transmission_fixed": row(transmission(tau_f), tau_f, deaf, "tau_fixed"),
                     "transmission_stochastic": row(transmission(tau_s), tau_s, deaf, "tau_stochastic"),
                     "random_vs_halves": row(random_gap(k["random"], k["upper"], k["lower"], k["W0"], matched),
                                             div(k["random"], top), holds, "random_over_halves"),
                     "inspan_vs_remainder": row(inspan_gap(k["inspan"], k["remainder"], k["W0"]),
                                                div(k["inspan"], k["remainder"]), holds, "inspan_over_remainder"),
                     "share_ls": {"value": share.get("ls"), "sft_range": list(SFT["share_ls"]),
                                  "against_sft": NO_CALL if "ls" not in share else
                                  "inside" if SFT["share_ls"][0] <= share["ls"] <= SFT["share_ls"][1] else "outside"}}}


def score(res: dict[str, dict], ref: dict[str, float]) -> dict[str, Any]:
    u1 = res.get("u1", {})
    return {"control": control(u1, ref)} | {name: score_update(u, u1, name[0] == "R") for name, u in res.items() if name != "u1"}
