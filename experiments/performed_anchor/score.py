"""E53's filed rules (results/predictions.json "rules"), as pure functions of read.json's rows; run.py calls `score`.

Row names: "<task>/s<seed>/step<n>" a fine-tune (role "finetune"), "<task>/random" and "<task>/write" its nulls; every
row carries G, piece_gain (subcomponent gain), masked_gain and piece_se; fine-tunes and writes a "relay" with G, rho.
"""

from __future__ import annotations

import math
from typing import Any

TASKS, SEEDS, STEPS = ("lowercase", "cipher"), (0, 1), {"lowercase": (300,), "cipher": (10, 100, 300)}


def scored(task: str) -> list[str]:
    return [f"{task}/s{s}/step300" for s in SEEDS]


def floor(rand: dict[str, Any], piece_se: float, th: dict[str, Any]) -> float:
    """The noise floor F of a row: max(floor_min, floor_null_mult x the larger of |G| and |subcomponent gain| of its
    task's KL-matched random delta-W, floor_se_mult x the row's own standard error of the subcomponent gain over
    windows). inf when any of these is not finite (unmeasured: the 1-nat bar alone applies, see `bar`)."""
    xs = [abs(rand["G"]), abs(rand["piece_gain"]), piece_se]
    if not all(math.isfinite(x) for x in xs):
        return math.inf
    return max(th["floor_min"], th["floor_null_mult"] * max(xs[0], xs[1]), th["floor_se_mult"] * xs[2])


def bar(F: float, th: dict[str, Any]) -> float:
    """The gain a row needs to be read: 1 nat (the plan's bar) unless the noise floor is lower (the user's answer 3)."""
    return min(th["gain_bar"], F)


def ratio_of(r: dict[str, Any], F: float, th: dict[str, Any]) -> float | None:
    """The stand-in ratio masked gain / subcomponent gain, or None unless G and the subcomponent gain both reach
    `bar(F)` and all three numbers are finite (NaN fails every comparison)."""
    x = (r["G"], r["piece_gain"], r["masked_gain"])
    if not all(math.isfinite(v) for v in x) or not (x[0] >= bar(F, th) and x[1] >= bar(F, th)):
        return None
    return x[2] / x[1]


def rho_of(relay: dict[str, Any], F: float, th: dict[str, Any]) -> float | None:
    """The relay rho, or None unless its full-model G reaches `bar(F)` and rho is finite."""
    ok = math.isfinite(relay["G"]) and relay["G"] >= bar(F, th) and math.isfinite(relay["rho"])
    return relay["rho"] if ok else None


def separation(a: list[float | None], b: list[float | None], gap: float) -> dict[str, Any]:
    """The plan's separation rule: gap = min(a) - max(b); "separates" if gap >= `gap`; "reversed" if max(a) - min(b)
    <= -`gap`; "unscored" if either list is empty or holds a None or a non-finite value; else "does not separate"."""
    if not a or not b or any(x is None or not math.isfinite(x) for x in a + b):
        return {"class": "unscored", "a": a, "b": b}
    av, bv = [x for x in a if x is not None], [x for x in b if x is not None]
    g, rev = min(av) - max(bv), max(av) - min(bv)
    cls = "separates" if g >= gap else "reversed" if rev <= -gap else "does not separate"
    return {"class": cls, "gap": g, "reverse_gap": rev, "a": a, "b": b}


def p3(rows: dict[str, Any], ratio: dict[str, float | None], F: dict[str, float], th: dict[str, Any]) -> dict[str, Any]:
    """Per task, the hand write's ratio minus its seed-0 fine-tune's (read only if the write's match is "ok" and both
    ratios are read); "tracks the task" if any task's |difference| <= p3_track, "does not track" if every task's is
    read and above it, else "unread"."""
    per = {}
    for n in TASKS:
        w = rows[f"{n}/write"]
        rw = ratio_of(w, F[f"{n}/write"], th) if w.get("match") == "ok" else None
        rf = ratio[f"{n}/s0/step300"]
        per[n] = {"write": rw, "finetune": rf, "diff": None if rw is None or rf is None else rw - rf}
    d = [v["diff"] for v in per.values() if v["diff"] is not None]
    verdict = ("tracks the task" if any(abs(x) <= th["p3_track"] for x in d)
               else "does not track" if len(d) == len(TASKS) else "unread")
    return {"tasks": per, "verdict": verdict}


def matched_gain(rows: dict[str, Any], ratio: dict[str, float | None]) -> dict[str, Any]:
    """Reported: per seed, the cipher step whose G is nearest the same seed's lowercase G (ties: the earlier step)."""
    out = {}
    for s in SEEDS:
        target = rows[f"lowercase/s{s}/step300"]["G"]
        ks = [f"cipher/s{s}/step{n}" for n in STEPS["cipher"]]
        k = min(ks, key=lambda k: (abs(rows[k]["G"] - target) if math.isfinite(rows[k]["G"]) else math.inf, ks.index(k)))
        out[f"s{s}"] = {"step": k, "G": rows[k]["G"], "lowercase_G": target, "ratio": ratio[k]}
    return out


def leans(rows: dict[str, Any], ratio: dict[str, Any], rho: dict[str, Any], th: dict[str, Any]) -> dict[str, Any]:
    """Reported: each scored row against the plan's lean values (None where the number is not read)."""
    t = lambda x, ok: None if x is None else ok(x)
    out: dict[str, Any] = {}
    for k in scored("lowercase"):
        out[k] = {"P1": t(ratio[k], lambda x: x >= th["lean_lowercase_ratio_min"]),
                  "P2": t(rho[k], lambda x: x >= th["lean_lowercase_relay_min"]),
                  "G": th["lean_lowercase_G"][0] <= rows[k]["G"] <= th["lean_lowercase_G"][1]}
    for k in scored("cipher"):
        out[k] = {"P1": t(ratio[k], lambda x: x <= th["lean_cipher_ratio_max"]),
                  "P2": t(rho[k], lambda x: th["lean_cipher_relay"][0] <= x <= th["lean_cipher_relay"][1]),
                  "G": rows[k]["G"] >= th["lean_cipher_G_min"]}
    for n in TASKS:
        out[f"{n}/random"] = {"G": abs(rows[f"{n}/random"]["G"]) <= th["lean_random_G_max"]}
        out[f"{n}/write"] = {"P2": t(rho.get(f"{n}/write"), lambda x: x <= th["lean_write_relay_max"])}
    return out


def score(rows: dict[str, Any], P: dict[str, Any]) -> dict[str, Any]:
    th = P["thresholds"]
    reads = [k for k, r in rows.items() if r["role"] == "finetune"] + [f"{n}/write" for n in TASKS if "G" in rows[f"{n}/write"]]
    F = {k: floor(rows[f"{k.split('/')[0]}/random"], rows[k]["piece_se"], th) for k in reads}
    ratio = {k: ratio_of(rows[k], F[k], th) for k in reads}
    rho = {k: rho_of(rows[k]["relay"], F[k], th) for k in reads}
    at1 = {k: ratio_of(rows[k], math.inf, th) for k in reads}
    p1 = separation([ratio[k] for k in scored("lowercase")], [ratio[k] for k in scored("cipher")], th["gap"])
    q3 = p3(rows, ratio, F, th)
    spread: dict[str, float | None] = {}
    for n in TASKS:
        a, b = (ratio[k] for k in scored(n))
        spread[n] = None if a is None or b is None else abs(a - b)
    if p1["class"] == "separates":
        reading = {"tracks the task": "separates these tasks (the hand write reads like its task: maybe not kind)",
                   "does not track": "separates", "unread": "separates (P3 unread)"}[q3["verdict"]]
    else:
        reading = p1["class"]
    return {"P1": p1, "P2_reported": separation([rho[k] for k in scored("lowercase")], [rho[k] for k in scored("cipher")], th["gap"]),
            "P3": q3, "reading": reading, "floor": F, "bar": {k: bar(f, th) for k, f in F.items()}, "ratio": ratio, "relay": rho,
            "P1_at_1_nat_reported": separation([at1[k] for k in scored("lowercase")], [at1[k] for k in scored("cipher")], th["gap"]),
            "seed_spread_reported": spread, "matched_gain_reported": matched_gain(rows, ratio), "leans_reported": leans(rows, ratio, rho, th)}


# ----------------------------------------------------------------------------- Amendment 1 (results/predictions_amend1.json)

A1, A1_STEPS = ("narrow_lower", "broad_cipher"), (10, 100, 300)


def breadth(r: dict[str, Any], kl_key: str = "kl_off") -> float | None:
    """G per true off-target KL (`kl_off`: positions that do not predict a scored-type token; high = targeted, low =
    broad), or None unless G > 0, the KL > 0, both finite. kl_key "kl_other" gives E53's pooled version (reported)."""
    G, kl = r["G"], r[kl_key]
    return G / kl if math.isfinite(G) and math.isfinite(kl) and G > 0 and kl > 0 else None


def swap(narrow: list[float | None], broad: list[float | None], factor: float) -> dict[str, Any]:
    """The filed swap check: achieved iff every breadth is read and min(narrow_lower) >= factor x max(broad_cipher)."""
    if not narrow or not broad or any(x is None for x in narrow + broad):
        return {"achieved": False, "why": "unread", "narrow": narrow, "broad": broad}
    nv, bv = [x for x in narrow if x is not None], [x for x in broad if x is not None]
    return {"achieved": min(nv) >= factor * max(bv), "min_narrow": min(nv), "max_broad": max(bv), "narrow": narrow, "broad": broad}


def outcome(achieved: bool, p1: dict[str, Any], th: dict[str, Any]) -> str:
    """The filed outcome table: swap missed -> not read; then P1's class on the new pair (narrow_lower as `a`)."""
    if not achieved:
        return "not read (swap missed)"
    cls = p1["class"]
    if cls in ("unscored", "separates", "reversed"):
        return {"unscored": "unscored", "separates": "kind", "reversed": "breadth"}[cls]
    xs = p1["a"] + p1["b"]
    if min(xs) >= th["high"]:
        return "neither (both high)"
    return "neither (both low)" if max(xs) <= th["low"] else "both"


def nearest(rows: dict[str, Any], task: str, seed: int, G: float, ratio: dict[str, float | None]) -> dict[str, Any]:
    """Reported: `task`'s step (seed `seed`) whose G is nearest `G` (ties: the earlier step), and its ratio."""
    ks = [f"{task}/s{seed}/step{n}" for n in A1_STEPS]
    k = min(ks, key=lambda k: (abs(rows[k]["G"] - G) if math.isfinite(rows[k]["G"]) else math.inf, ks.index(k)))
    return {"step": k, "G": rows[k]["G"], "target_G": G, "ratio": ratio[k]}


def effects(ratio: dict[str, float | None], e53_ratio: dict[str, Any]) -> dict[str, Any] | None:
    """Reported: the 2 x 2 of kind x breadth on step-300 seed means, E53's lowercase (performed, broad) and cipher
    (new, targeted) with the new pair: kind = (performed - new) / 2 summed, breadth = (broad - targeted) / 2 summed."""
    m: dict[str, float] = {}
    for n, src in (("lowercase", e53_ratio), ("cipher", e53_ratio), ("narrow_lower", ratio), ("broad_cipher", ratio)):
        xs = [src.get(f"{n}/s{s}/step300") for s in SEEDS]
        vs = [float(x) for x in xs if x is not None and math.isfinite(x)]
        if len(vs) < len(xs):
            return None
        m[n] = sum(vs) / len(vs)
    return {"kind": (m["lowercase"] + m["narrow_lower"] - m["cipher"] - m["broad_cipher"]) / 2,
            "breadth": (m["lowercase"] + m["broad_cipher"] - m["cipher"] - m["narrow_lower"]) / 2, "means": m}


def score_amend1(rows: dict[str, Any], e53: dict[str, Any], P: dict[str, Any]) -> dict[str, Any]:
    th = P["thresholds"]
    reads = [k for k, r in rows.items() if r["role"] == "finetune"]
    F = {k: floor(rows[f"{k.split('/')[0]}/random"], rows[k]["piece_se"], th) for k in reads}
    ratio = {k: ratio_of(rows[k], F[k], th) for k in reads}
    b = {k: breadth(rows[k]) for k in reads}
    nl, bc = scored("narrow_lower"), scored("broad_cipher")
    p1 = separation([ratio[k] for k in nl], [ratio[k] for k in bc], th["gap"])
    sw = swap([b[k] for k in nl], [b[k] for k in bc], th["swap_factor"])
    reread = {k.removeprefix("e53/"): r for k, r in rows.items() if k.startswith("e53/")}
    return {"swap": sw, "P1": p1, "outcome": outcome(sw["achieved"], p1, th), "floor": F, "ratio": ratio, "breadth": b,
            "breadth_pooled_kl_reported": {k: breadth(rows[k], "kl_other") for k in reads},
            "P1_at_1_nat_reported": separation([ratio_of(rows[k], math.inf, th) for k in nl],
                                               [ratio_of(rows[k], math.inf, th) for k in bc], th["gap"]),
            "leans_reported": {k: None if ratio[k] is None else (ratio[k] >= th["high"] if k in nl else ratio[k] <= th["low"])
                               for k in nl + bc},
            "matched_gain_reported": {f"s{s}": {"broad_near_narrow": nearest(rows, "broad_cipher", s, rows[f"narrow_lower/s{s}/step300"]["G"], ratio),
                                                "narrow_near_broad": nearest(rows, "narrow_lower", s, rows[f"broad_cipher/s{s}/step300"]["G"], ratio)}
                                      for s in SEEDS},
            "e53_reference": {k: {"G": r["G"], "kl_other": r["kl_other"], "kl_off": r["kl_off"], "breadth": breadth(r),
                                  "breadth_pooled_kl": breadth(r, "kl_other"), "ratio_reread": r["ratio"],
                                  "ratio_read_json": e53["score"]["ratio"].get(k)} for k, r in reread.items()},
            "effects_reported": effects(ratio, e53["score"]["ratio"])}
