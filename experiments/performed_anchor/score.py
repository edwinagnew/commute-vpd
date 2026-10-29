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
