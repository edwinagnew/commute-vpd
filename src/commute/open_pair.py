"""E58's filed rules (experiments/open_pair/results/predictions.json), pure functions so every edge case runs on CPU
(tests/test_open_pair.py). Plan: reports/plans/2026-10-30_plan_e58-open-pair.md.

The behavioural label is the reuse line's Stage B patch rule (reports/plans/2026-10-29_plan_reuse-stage1.md, § Stage B:
"through iff P >= 0.5 (full = 1), not iff P <= 3 s (row bootstrap), else neither") with P read on behaviour: x = H / G,
H the held-out-key gain and G the trained-key gain (Fable, reports/plans/2026-10-30_advice_regating-step2-FABLE.md).
A state record is {"label": str, "r": float, "ratio": float, "S": float}; states are named `<kind>_s<seed>`, kind elicit
or cipher (the cipher at the elicit fine-tune's update norm).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

LABELLED = ("reuses", "new")


def split_keys(keys: list[int], n_held: int, seed: int) -> tuple[list[int], list[int]]:
    """(trained, held-out): the held-out keys are keys[i] for the first n_held of torch.randperm(len(keys), seed); both
    lists keep the order of `keys`."""
    held_idx = set(torch.randperm(len(keys), generator=torch.Generator().manual_seed(seed))[:n_held].tolist())
    return [k for i, k in enumerate(keys) if i not in held_idx], [k for i, k in enumerate(keys) if i in held_idx]


def finite(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def boot_sd(dH: Tensor, dG: Tensor, n_boot: int, seed: int) -> float:
    """The row bootstrap's sd of x = mean(dH) / mean(dG): held-out-key windows and trained-key windows resampled
    independently (one generator, H's indices drawn first). nan when either set is empty or any resample's mean(dG) is
    not positive (x has no meaning there)."""
    if len(dH) == 0 or len(dG) == 0:
        return math.nan
    gen = torch.Generator().manual_seed(seed)
    iH = torch.randint(len(dH), (n_boot, len(dH)), generator=gen)
    iG = torch.randint(len(dG), (n_boot, len(dG)), generator=gen)
    g = dG.double()[iG].mean(1)
    if not bool((g > 0).all()):
        return math.nan
    return float((dH.double()[iH].mean(1) / g).std())


def boot_sd_keys(dH: Tensor, kH: Tensor, dG: Tensor, kG: Tensor, n_boot: int, seed: int) -> float:
    """Reported, unscored (amendment 2026-10-30): the key-level bootstrap's sd of x. Each resample draws the held-out keys
    and the trained keys with replacement (one generator, H's draws first) and pools every window of each drawn key
    (a key drawn twice counts twice); x = mean(dH pooled) / mean(dG pooled). kH, kG give each window's key. nan when
    either set is empty or any resample's mean(dG) is not positive."""
    if len(dH) == 0 or len(dG) == 0:
        return math.nan
    gen = torch.Generator().manual_seed(seed)
    uH, uG = kH.unique(), kG.unique()
    xs = []
    for _ in range(n_boot):
        pick = lambda d, k, u: torch.cat([d[k == u[i]] for i in torch.randint(len(u), (len(u),), generator=gen)]).double().mean()
        h = pick(dH, kH, uH)
        g = pick(dG, kG, uG)
        if not g > 0:
            return math.nan
        xs.append(float(h / g))
    return float(torch.tensor(xs, dtype=torch.float64).std())


def spec_gate(cap: float, ref_raw: float, g_specs: list[float]) -> float:
    """The G_spec gate (amendment 2026-10-30, after job 46259): Fable's 8 scaled as 8 was to E54's raw G (cap / ref_raw),
    applied to the smallest of E54's seed-0 G_spec values, capped at `cap`. nan if any input is not finite or empty;
    a non-positive G_spec gives a non-positive gate, which run.py's stage 0 does not reach (its specific-share check
    fails first)."""
    if not g_specs or not all(finite(g) for g in [cap, ref_raw, *g_specs]) or ref_raw <= 0:
        return math.nan
    return min(cap, cap / ref_raw * min(g_specs))


def label(H: float, G: float, s: float, th: dict[str, Any]) -> str:
    """The behavioural label. "too_small" if G < G_min (Fable: the state is too small to label); then x = H / G:
    "reuses" iff x >= cut, "new" iff x <= k_sd * s, "neither" otherwise; "unreadable" if H, G or s is not finite, or if
    both hold at once (3 s reaches the cut: the bootstrap is too wide to label). Ties at either bound count (>=, <=)."""
    if not (finite(H) and finite(G)):
        return "unreadable"
    if G < th["G_min"]:
        return "too_small"
    if not finite(s):
        return "unreadable"
    x = H / G
    reuses, new = x >= th["cut"], x <= th["k_sd"] * s
    if reuses and new:
        return "unreadable"
    return "reuses" if reuses else "new" if new else "neither"


def inst_label(v: Any, cut: float) -> str | None:
    """An instrument's label from its reading: "reuses" iff v >= cut, else "new"; None when v is missing or not finite."""
    return None if not finite(v) else "reuses" if v >= cut else "new"


def p1(st: dict[str, dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    """P1, the label's negative control: every matched cipher reads "new". "stop" if any reads "reuses" (the rule is
    broken; run.py exits there); "unreadable" with no cipher state; else "pass" or "fail"."""
    labs = [st[f"cipher_s{s}"]["label"] for s in seeds if f"cipher_s{s}" in st]
    if not labs:
        return {"labels": labs, "status": "unreadable"}
    status = "stop" if "reuses" in labs else "pass" if all(x == "new" for x in labs) else "fail"
    return {"labels": labs, "status": status}


def p2(st: dict[str, dict[str, Any]], seeds: list[int], inst: str, cut: float, kind: str = "elicit") -> dict[str, Any]:
    """P2 for one instrument (r or ratio): on every `kind` run labelled "reuses" or "new", the instrument's label
    (`inst_label`) equals it. Runs whose reading is not finite are listed and left out. "not_testable" when no run is
    left; "split" is reported beside when the labelled runs carry both labels."""
    runs, dropped = [], []
    for s in seeds:
        rec = st.get(f"{kind}_s{s}")
        if rec is None or rec["label"] not in LABELLED:
            continue
        il = inst_label(rec.get(inst), cut)
        (dropped if il is None else runs).append({"seed": s, "label": rec["label"], inst: il})
    status = "not_testable" if not runs else "pass" if all(r["label"] == r[inst] for r in runs) else "fail"
    return {"runs": runs, "dropped": dropped, "n": len(runs), "split": len({r["label"] for r in runs}) > 1, "status": status}


def p3(st: dict[str, dict[str, Any]], cut: float, min_lead: int) -> dict[str, Any]:
    """P3 (the step-2 plan's P3 on these runs): over every state labelled "reuses" or "new" whose r, ratio and S are
    all finite (the same rows for every instrument), the agreement counts A_r, A_ratio, A_S. Per VPD instrument:
    "adds" iff A - A_S >= min_lead, "steering_better" iff A_S - A >= min_lead, else "no_difference";
    "not_testable" with no row."""
    rows = [n for n, rec in sorted(st.items()) if rec["label"] in LABELLED
            and all(inst_label(rec.get(m), cut) is not None for m in ("r", "ratio", "S"))]
    A = {m: sum(inst_label(st[n][m], cut) == st[n]["label"] for n in rows) for m in ("r", "ratio", "S")}
    if not rows:
        return {"rows": rows, "A": A, "r": "not_testable", "ratio": "not_testable"}
    verdict = lambda a: "adds" if a - A["S"] >= min_lead else "steering_better" if A["S"] - a >= min_lead else "no_difference"
    return {"rows": rows, "A": A, "r": verdict(A["r"]), "ratio": verdict(A["ratio"])}


def p0(st: dict[str, dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    """P0, the open reading (a lean, not a test): the elicit runs' labels counted; "reuses" or "new" when at least two
    of the three runs carry it, else "mixed"."""
    labs = [st[f"elicit_s{s}"]["label"] for s in seeds if f"elicit_s{s}" in st]
    counts = {x: labs.count(x) for x in sorted(set(labs))}
    top = next((x for x in LABELLED if counts.get(x, 0) >= 2), "mixed")
    return {"labels": labs, "counts": counts, "verdict": top}


def score(st: dict[str, dict[str, Any]], th: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    from commute.elicit import separation

    get = lambda kind, s, m: (st.get(f"{kind}_s{s}") or {}).get(m)
    col = lambda kind, m: [get(kind, s, m) if finite(get(kind, s, m)) else None for s in seeds]
    p4 = {m: separation(col("elicit", m), col("cipher", m), th["min_gap"]) for m in ("r", "ratio")}
    return {"P0": p0(st, seeds), "P1": p1(st, seeds),
            "P2": {m: p2(st, seeds, m, th["cut"]) for m in ("r", "ratio")},
            "P3": p3(st, th["cut"], th["min_lead"]),
            "P4": {m: {**v, "status": "unreadable" if v["verdict"] == "unreadable" else "pass" if v["verdict"] == "a_higher" else "fail"}
                   for m, v in p4.items()}}


# ----------------------------------------------------------------------------- Amendment 1 (2026-10-30)
# experiments/open_pair/results/predictions_amend1.json; plan section "Amendment 1"; edge cases in tests/test_open_pair.py


def jsonable(o: Any) -> Any:
    """A results record with non-finite floats as strings (JSON has no nan)."""
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    return [jsonable(v) for v in o] if isinstance(o, list | tuple) else o


def implied_push(H: float, H_spec: float, H_other: float, n_pool: int, n_held: int) -> float:
    """The trained values' mean raw gain at the held-out-key windows implied by E58's numbers. With distinct values, the
    pool's other n_pool - 1 values at a held-out window are the n_pool - n_held trained values and the n_held - 1 other
    held-out values, and H - H_spec is their mean gain, so T = ((n_pool - 1)(H - H_spec) - (n_held - 1) H_other) /
    (n_pool - n_held). nan if n_held < 2 (H_other undefined) or n_pool <= n_held."""
    if n_held < 2 or n_pool <= n_held:
        return math.nan
    return ((n_pool - 1) * (H - H_spec) - (n_held - 1) * H_other) / (n_pool - n_held)


def boot_sd_mean(v: Tensor, n_boot: int, seed: int) -> float:
    """sd of mean(v) over n_boot row-bootstrap resamples (one generator seeded `seed`); nan with fewer than 2 rows or a
    non-finite entry."""
    if len(v) < 2 or not bool(torch.isfinite(v).all()):
        return math.nan
    i = torch.randint(len(v), (n_boot, len(v)), generator=torch.Generator().manual_seed(seed))
    return float(v.double()[i].mean(1).std())


def push_verdict(d: float, s_d: float, R_peer: float, s_R: float, k_sd: float) -> str:
    """Readout (i)'s rule for one state. d = mean over the held-out-key windows of T_w - R_w, T_w the mean gain of the
    trained keys' values (other than the window's own) and R_w the mean gain of W0's other top-k tokens at the window's
    key (neither the own value nor a trained value); R_peer = mean R_w. "lookup" iff d > 0 and d >= k_sd s_d (the
    trained values rise above W0's own preferences at a key never trained); "sharpening" iff R_peer > 0 and R_peer >=
    k_sd s_R (W0's own preferences at that key rise); "both", "neither"; "unreadable" if any input is not finite."""
    if not all(finite(v) for v in (d, s_d, R_peer, s_R)):
        return "unreadable"
    lookup, sharp = d > 0 and d >= k_sd * s_d, R_peer > 0 and R_peer >= k_sd * s_R
    return "both" if lookup and sharp else "lookup" if lookup else "sharpening" if sharp else "neither"


def a1(rd: dict[str, dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    """A1: every elicit state's readout reads "lookup". "fail" if any reads "sharpening", "both" or "neither"; else
    "unreadable" if a state is missing or unreadable; else "pass"."""
    v = [rd.get(f"elicit_s{s}", {}).get("verdict", "unreadable") for s in seeds]
    status = "fail" if any(x in ("sharpening", "both", "neither") for x in v) else "unreadable" if "unreadable" in v else "pass"
    return {"verdicts": v, "status": status}


def n_novel(trained_values: list[int], held_values: list[int]) -> int:
    """How many held-out keys have a value that no trained key has."""
    return sum(v not in set(trained_values) for v in held_values)


def control_status(labels: list[str], testable: bool, min_reuses: int) -> str:
    """C1 / C2, the label's positive control on one map: "not_testable" if the map failed its pre-check or has no label;
    "pass" iff at least `min_reuses` runs read "reuses"; else "fail"."""
    if not testable or not labels:
        return "not_testable"
    return "pass" if labels.count("reuses") >= min_reuses else "fail"


def positive_control(c1: str, c2: str) -> str:
    """"rank1" if C1 passes; "copy_fallback" if C1 is not testable and C2 passes; "copy_only" if C1 fails and C2 passes;
    "none" otherwise (the label has no positive control on the 67M)."""
    if c1 == "pass":
        return "rank1"
    if c2 == "pass":
        return "copy_fallback" if c1 == "not_testable" else "copy_only"
    return "none"


def score_amend1(rd: dict[str, dict[str, Any]], ctrl: dict[str, dict[str, Any]], n_novel_rank1: int, th: dict[str, Any],
                 seeds: list[int]) -> dict[str, Any]:
    """A1 on the readout of E58's states; C1 (rank-1 map; testable iff n_novel >= min_novel and rank1_s0 is not
    "too_small"), C2 (copy map), the verdict, and C3 = E58's P2 for r and the ratio on each map's runs (scored on the
    map the verdict rests on: rank-1 unless C1 is not testable)."""
    labs = {m: [ctrl[f"{m}_s{s}"]["label"] for s in seeds if f"{m}_s{s}" in ctrl] for m in ("rank1", "copy")}
    ok1 = n_novel_rank1 >= th["min_novel"] and ctrl.get("rank1_s0", {}).get("label") != "too_small"
    c1, c2 = control_status(labs["rank1"], ok1, th["min_reuses"]), control_status(labs["copy"], True, th["min_reuses"])
    c3 = {m: {i: p2(ctrl, seeds, i, th["cut"], kind=m) for i in ("r", "ratio")} for m in ("rank1", "copy")}
    return {"A1": a1(rd, seeds), "C1": {"labels": labs["rank1"], "testable": ok1, "status": c1},
            "C2": {"labels": labs["copy"], "status": c2}, "verdict": positive_control(c1, c2),
            "C3": {"scored_on": "copy" if c1 == "not_testable" else "rank1", **c3}}


# ----------------------------------------------------------------------------- Amendment 2 (2026-10-30)
# experiments/open_pair/results/predictions_amend2.json; plan section "Amendment 2"; edge cases in tests/test_open_pair.py
# One seed per map at K keys (128, or 64 when the key supply is short); a state record holds E58's behaviour record
# (label, x) plus x_novel, r and ratio. `ref` holds the same map's 24-key means over seeds 0-2 (E58 run.json, amend1.json).

UNLABELLABLE = ("too_small", "unreadable")


def elicit_control(x: float, label: str, th: dict[str, Any]) -> str:
    """E, the control (the elicit map has no rule to carry): the label's "too_small" / "unreadable" pass through (or
    "unreadable" if x is not finite); "fail" iff x > e_max (carry where none can exist: the job stops); "below" iff
    x < -e_max (no carry, a larger pool shift than at 24 keys: not a stop); else "pass". Ties at +-e_max pass."""
    if label in UNLABELLABLE:
        return label
    if not finite(x):
        return "unreadable"
    return "fail" if x > th["e_max"] else "below" if x < -th["e_max"] else "pass"


def rank1_class(x: float, x_nov: float, n_nov: int, label: str, th: dict[str, Any]) -> str:
    """R, the rank-1 map: "not_testable" if the label is too_small / unreadable, fewer than min_novel held-out keys have
    a value no trained key has, or x or x_novel is not finite; "reuses" iff the label reads "reuses" and x_novel >= cut
    (Fable's (a)); "carries" iff x and x_novel >= rank1_carry; "memorises" iff both <= rank1_memorise; else "between"."""
    if label in UNLABELLABLE or n_nov < th["min_novel"] or not (finite(x) and finite(x_nov)):
        return "not_testable"
    if label == "reuses" and x_nov >= th["cut"]:
        return "reuses"
    if min(x, x_nov) >= th["rank1_carry"]:
        return "carries"
    return "memorises" if max(x, x_nov) <= th["rank1_memorise"] else "between"


def copy_class(x: float, label: str, th: dict[str, Any]) -> str:
    """C, the copy map: "not_testable" if the label is too_small / unreadable or x is not finite; "pass" iff x >= copy_carry."""
    if label in UNLABELLABLE or not finite(x):
        return "not_testable"
    return "pass" if x >= th["copy_carry"] else "fail"


def moves_with(dx: float, own: float, di: float, dx_min: float, tol: float) -> str:
    """One instrument on one map. dx and di are differences in differences: (map - elicit) at K keys minus the same at 24
    keys, for x and for the instrument; own = the map's own x at K keys minus at 24 keys. "unreadable" if any is not
    finite; "not_testable" iff |dx| < dx_min, |own| < dx_min, or own and dx differ in sign (the map's x did not itself
    move that way: elicit's own movement alone never makes a map testable; review of a23c3ea); "tracks" iff di moved the
    same way as dx by more than tol; "flat" iff |di| <= tol; else "opposite"."""
    if not (finite(dx) and finite(own) and finite(di)):
        return "unreadable"
    if abs(dx) < dx_min or abs(own) < dx_min or own * dx <= 0:
        return "not_testable"
    if (di if dx > 0 else -di) > tol:
        return "tracks"
    return "flat" if abs(di) <= tol else "opposite"


def combine(v: list[str]) -> str:
    """"tracks" > "opposite" > "flat"; "not_testable" when every entry is; else (empty list included) "unreadable"."""
    for k in ("tracks", "opposite", "flat"):
        if k in v:
            return k
    return "not_testable" if v and all(s == "not_testable" for s in v) else "unreadable"


def b2(st: dict[str, dict[str, Any]], ref: dict[str, dict[str, float]], th: dict[str, Any]) -> dict[str, Any]:
    """B, Fable's (b): per map (rank-1, copy) and per instrument (r, ratio; tolerances tol_r, tol_ratio), `moves_with` on
    the differences in differences against elicit; per map `combine` over the instruments, then over the maps. A
    missing map or elicit state reads "unreadable"."""
    get = lambda d, m, k: (d.get(m) or {}).get(k, math.nan)
    did = lambda m, k: (get(st, m, k) - get(st, "elicit", k)) - (get(ref, m, k) - get(ref, "elicit", k))
    per: dict[str, Any] = {}
    for m in ("rank1", "copy"):
        dx, own = did(m, "x"), get(st, m, "x") - get(ref, m, "x")
        inst = {i: {"d": did(m, i), "status": moves_with(dx, own, did(m, i), th["dx_min"], th[f"tol_{i}"])} for i in ("r", "ratio")}
        per[m] = {"dx": dx, "dx_own": own, **inst, "status": combine([v["status"] for v in inst.values()])}
    return {"maps": per, "status": combine([per[m]["status"] for m in per])}


def verdict2(e: str, b: str, r1: str) -> str:
    """"stopped" unless the control passed ("pass" or "below"); then "b_r_tracks" > "a_first_reuses" (rank-1 "reuses")
    > "reading_A_holds" (B flat) > "opposite"; else "untested" (x moved under dx_min on both maps, or unreadable)."""
    if e not in ("pass", "below"):
        return "stopped"
    if b == "tracks":
        return "b_r_tracks"
    if r1 == "reuses":
        return "a_first_reuses"
    return {"flat": "reading_A_holds", "opposite": "opposite"}.get(b, "untested")


def score_amend2(st: dict[str, dict[str, Any]], n_novel_rank1: int, ref: dict[str, dict[str, float]], th: dict[str, Any]) -> dict[str, Any]:
    """E, R, C, B and the verdict on the K-key states (`st` keyed elicit, rank1, copy; a missing map reads as unlabelled)."""
    rec = lambda m: st.get(m) or {"label": "unreadable", "x": math.nan, "x_novel": math.nan}
    e, r, c = rec("elicit"), rec("rank1"), rec("copy")
    E = elicit_control(e["x"], e["label"], th)
    R = rank1_class(r["x"], r["x_novel"], n_novel_rank1, r["label"], th)
    B = b2(st, ref, th)
    return {"E": E, "R": R, "C": copy_class(c["x"], c["label"], th), "B": B, "verdict": verdict2(E, B["status"], R)}


# ----------------------------------------------------------------------------- Amendment 3 (2026-10-30)
# experiments/open_pair/results/predictions_amend3.json; plan section "Amendment 3"; edge cases in tests/test_open_pair.py
# The control is the cipher at K keys (Fable, reports/plans/2026-10-30_advice_e58a2-control-FABLE.md); elicit is a scored
# map. B compares each map with itself at 24 keys (own differences): no map is a no-rule reference for the others now.


def pick_cipher_step(g_spec: dict[int, float], common: int, last: int, g_min: float) -> int | None:
    """The step the cipher control is read at: the common step if its G_spec >= g_min; else the first kept checkpoint
    after it, up to `last`, with G_spec >= g_min; None (the job stops) if there is none or a needed value is missing or
    not finite (review of b60ae7e: the cipher at 96 keys may be too small to label at elicit's step)."""
    for s in range(common, last + 1):
        if s != common and s not in g_spec:
            continue
        g = g_spec.get(s, math.nan)
        if not finite(g):
            return None
        if g >= g_min:
            return s
    return None


def b3(st: dict[str, dict[str, Any]], ref: dict[str, dict[str, float]], th: dict[str, Any], maps: tuple[str, ...],
       cipher_r: float = math.nan) -> dict[str, Any]:
    """B in Amendment 3: per map, dx = x at K keys minus x at 24 keys and di likewise for r (tol_r) and ratio (tol_ratio);
    `moves_with(dx, dx, di)`; per map `combine` over the instruments, then over the maps. A missing map is unreadable.
    Drift guard (review of b60ae7e): "untested_drift" overall if the norm-matched cipher's r at K keys (`cipher_r`) is
    not finite or differs from ref["cipher"]["r"] by more than tol_r, since nothing else absorbs a shift of every r
    with the key count."""
    get = lambda d, m, k: (d.get(m) or {}).get(k, math.nan)
    per: dict[str, Any] = {}
    for m in maps:
        dx = get(st, m, "x") - get(ref, m, "x")
        inst = {i: {"d": get(st, m, i) - get(ref, m, i)} for i in ("r", "ratio")}
        for i, v in inst.items():
            v["status"] = moves_with(dx, dx, v["d"], th["dx_min"], th[f"tol_{i}"])
        per[m] = {"dx": dx, **inst, "status": combine([v["status"] for v in inst.values()])}
    drift = cipher_r - get(ref, "cipher", "r")
    ok = finite(drift) and abs(drift) <= th["tol_r"]
    return {"maps": per, "cipher_r_drift": drift, "status": combine([per[m]["status"] for m in maps]) if ok else "untested_drift"}


def trajectory(xs: dict[int, float], first: int, later: tuple[int, ...], th: dict[str, Any]) -> str:
    """Check 1 (reported): elicit's x over the E12 checkpoints. "unreadable" if any is missing or not finite or there is
    no later step; "no_carry" iff x[first] <= e_max; "rising" iff every later x >= x[first] - traj_tol and the last
    later x > x[first] + traj_tol; "kept" iff every later x >= x[first] - traj_tol; "transient" iff x at the next
    checkpoint <= e_max; else "falling"."""
    vals = [xs.get(s, math.nan) for s in (first, *later)]
    if not later or not all(finite(v) for v in vals):
        return "unreadable"
    if vals[0] <= th["e_max"]:
        return "no_carry"
    if min(vals[1:]) >= vals[0] - th["traj_tol"]:
        return "rising" if vals[-1] > vals[0] + th["traj_tol"] else "kept"
    return "transient" if vals[1] <= th["e_max"] else "falling"


def ranks(v: list[float]) -> Tensor:
    """1-based ranks, ties given their mean rank."""
    t = torch.tensor(v, dtype=torch.float64)
    r = torch.empty_like(t)
    order = torch.argsort(t, stable=True)
    i = 0
    while i < len(t):
        j = i
        while j + 1 < len(t) and t[order[j + 1]] == t[order[i]]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    """Check 5 (reported): Spearman's rho with mean ranks for ties; nan with fewer than 3 pairs, unequal lengths, a
    non-finite entry or a constant side."""
    if len(a) != len(b) or len(a) < 3 or not all(finite(v) for v in [*a, *b]):
        return math.nan
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(ra.norm() * rb.norm())
    return float((ra * rb).sum()) / den if den > 0 else math.nan


def score_amend3(st: dict[str, dict[str, Any]], st30: dict[str, dict[str, Any]], control: str, n_novel_rank1: int,
                 ref: dict[str, dict[str, float]], th: dict[str, Any], cipher_r: float) -> dict[str, Any]:
    """The control's status, R (rank-1), C (copy), B on elicit, rank-1 and copy at the common step (scored) and at step 30
    (companion, reported), both with the drift guard on the norm-matched cipher's r, and the verdict (`verdict2` with B
    at the common step: "untested_drift" reads as untested)."""
    rec = lambda m: st.get(m) or {"label": "unreadable", "x": math.nan, "x_novel": math.nan}
    r, c = rec("rank1"), rec("copy")
    maps = ("elicit", "rank1", "copy")
    R = rank1_class(r["x"], r["x_novel"], n_novel_rank1, r["label"], th)
    B = b3(st, ref, th, maps, cipher_r)
    return {"control": control, "R": R, "C": copy_class(c["x"], c["label"], th), "B": B,
            "B_step30": b3(st30, ref, th, maps, cipher_r), "verdict": verdict2(control, B["status"], R)}


# ----------------------------------------------------------------------------- Amendment 4 (2026-10-30)
# experiments/open_pair/results/predictions_amend4.json; plan section "Amendment 4"; edge cases in tests/test_open_pair.py
# Fable, reports/plans/2026-10-30_advice_e58a3-next-FABLE.md, item 2. A read is {"x", "r", "ratio", "label"} of one map,
# seed and step.


def readable(rd: dict[str, Any] | None, inst: str) -> bool:
    """A read counts in the within-key-count rule iff its label is not too_small / unreadable and x and the
    instrument are finite."""
    return rd is not None and rd.get("label") not in UNLABELLABLE and finite(rd.get("x")) and finite(rd.get(inst))


def within_key(reads: Mapping[str, Sequence[dict[str, Any] | None]], inst: str, th: dict[str, Any]) -> dict[str, Any]:
    """Item 2.1 at one step and key count, for one instrument (r or ratio). Per map, its readable seeds; a map with fewer
    than 2 is dropped (no seed spread). D_x and D_i = max - min over the kept maps of the seed-mean x and instrument;
    spread = the largest within-map range (max - min over seeds) of the instrument. "not_testable" with fewer than 2 kept
    maps or D_x < x_range_min (the maps do not differ in carry); else "pass" iff D_i <= the bound (r: r_mult x spread;
    ratio: ratio_tol, absolute), "fail" otherwise. Ties count (<=, >=). A zero spread makes the r bound 0."""
    kept = {m: [rd for rd in v if rd is not None and readable(rd, inst)] for m, v in reads.items()}
    kept = {m: v for m, v in kept.items() if len(v) >= 2}
    mean = lambda v, k: sum(rd[k] for rd in v) / len(v)
    rec: dict[str, Any] = {"maps": sorted(kept), "n_seeds": {m: len(v) for m, v in kept.items()}}
    if len(kept) < 2:
        return {**rec, "status": "not_testable"}
    xs, ys = [mean(v, "x") for v in kept.values()], [mean(v, inst) for v in kept.values()]
    spread = max(max(rd[inst] for rd in v) - min(rd[inst] for rd in v) for v in kept.values())
    bound = th["r_mult"] * spread if inst == "r" else th["ratio_tol"]
    rec.update({"D_x": max(xs) - min(xs), "D": max(ys) - min(ys), "spread": spread, "bound": bound})
    if rec["D_x"] < th["x_range_min"]:
        return {**rec, "status": "not_testable"}
    return {**rec, "status": "pass" if rec["D"] <= bound else "fail"}


def shuffle_class(rd: dict[str, Any] | None, th: dict[str, Any]) -> str:
    """Item 2.2: the shuffled map's r at step 30. "unreadable" if missing, too_small / unreadable, or r not finite;
    "cipher_like" iff r < shuffle_cut; else "token_kind" (a tie at the cut reads token_kind)."""
    if rd is None or rd.get("label") in UNLABELLABLE or not finite(rd.get("r")):
        return "unreadable"
    return "cipher_like" if rd["r"] < th["shuffle_cut"] else "token_kind"


def score_amend4(reads: Mapping[int, Mapping[str, Sequence[dict[str, Any] | None]]], shuffle30: dict[str, Any] | None,
                 th: dict[str, Any]) -> dict[str, Any]:
    """Per step (`reads[step][map]` = the reads of seeds 0-2): `within_key` for r and for the ratio, both reported per
    instrument and "pass" overall iff both pass; the cross-seed Spearman of r on x over every readable read (reported);
    the shuffle's class."""
    out: dict[str, Any] = {}
    for step, rs in reads.items():
        w = {i: within_key(rs, i, th) for i in ("r", "ratio")}
        st = [w[i]["status"] for i in w]
        allr = [rd for v in rs.values() for rd in v if rd is not None and readable(rd, "r")]
        out[str(step)] = {**w, "status": "pass" if st == ["pass", "pass"] else "fail" if "fail" in st else "not_testable",
                          "rho_r_x": spearman([rd["x"] for rd in allr], [rd["r"] for rd in allr])}
    return {"within_key": out, "shuffle": shuffle_class(shuffle30, th)}


def comparable(pairs: Sequence[tuple[Any, Any]], tol: float) -> bool:
    """Amendment 4 (review of 0ce43b1): seed 0's re-trained cipher r against Amendment 3's at the same steps. True iff
    there is at least one pair and every pair is finite and within tol; then Amendment 3's seed-0 map reads are used,
    else they are "not_comparable" and the within-key rule is scored on seeds 1-2 only."""
    return bool(pairs) and all(finite(a) and finite(b) and abs(a - b) <= tol for a, b in pairs)


# ----------------------------------------------------------------------------- Amendment 5 (2026-10-30)
# experiments/open_pair/results/predictions_amend5.json; plan section "Amendment 5"; reported, not scored


def shuffle_sentence(r0: Any, r_new: Sequence[Any], th: dict[str, Any]) -> str:
    """The write-up rule for the shuffle's r at step 30 (Fable, reports/plans/2026-10-30_advice_e58a4-synthesis-FABLE.md,
    item 3), r0 = seed 0's, r_new = seeds 1 and 2. "unreadable" unless all three are finite; "change_it" iff every new
    seed > change_cut (0.55); else "midpoint" iff any seed >= cut (0.5); else "mostly_key_conditional" (the three-seed
    mean is then under the cut)."""
    rs = [r0, *r_new]
    if not r_new or not all(finite(r) for r in rs):
        return "unreadable"
    if all(r > th["change_cut"] for r in r_new):
        return "change_it"
    return "midpoint" if max(rs) >= th["cut"] else "mostly_key_conditional"


# ----------------------------------------------------------------------------- Amendment 6 (2026-10-30)
# experiments/open_pair/results/predictions_amend6.json; plan section "Amendment 6"; edge cases in tests/test_open_pair.py
# Fable, reports/plans/2026-10-30_advice_regating-baseline-FABLE.md, (b)1 and (c). A read is {"x", "r", "label"}.


def band_class(rs: Sequence[Any], th: dict[str, Any]) -> str:
    """Where a map's r sits, one read per seed: "unreadable" if there is no read or any is missing or not finite;
    "cipher_band" iff every r < cut (0.5); "own_guess_band" iff every r >= high (0.6); else "between" (any r in
    [0.5, 0.6), or seeds on both sides). Ties: 0.5 is not under the cut; 0.6 is in the upper band."""
    if not rs or not all(finite(r) for r in rs):
        return "unreadable"
    return "cipher_band" if max(rs) < th["cut"] else "own_guess_band" if min(rs) >= th["high"] else "between"


def labellable(rd: dict[str, Any] | None) -> bool:
    """A read whose label is not too_small / unreadable and whose x is finite."""
    return rd is not None and rd.get("label") not in UNLABELLABLE and finite(rd.get("x"))


def carries(rd: dict[str, Any] | None, th: dict[str, Any]) -> bool:
    """The first-letter rule's entry condition on one read (review of a946868): labellable, x_null finite, and
    x - x_null > x_entry (0.3; a tie fails). x_null is x with each held-out key's target swapped for another letter's
    word of matched trained count, so a lean toward often-trained words alone gives x - x_null near 0."""
    return labellable(rd) and rd is not None and finite(rd.get("x_null")) and rd["x"] - rd["x_null"] > th["x_entry"]


def score_amend6(prior: Sequence[dict[str, Any] | None], letter: Sequence[dict[str, Any] | None], th: dict[str, Any]) -> dict[str, Any]:
    """PM: `band_class` on the prior-matched cipher's step-30 r over seeds 0-2 (a read too small to label counts as
    missing). FL: "unreadable" if fewer than min_entry (2) step-30 reads are `labellable`; else "no_carry" if fewer than
    min_entry `carries`; else `band_class` on the carrying seeds' r. Each class beside its numbers; `near_cut` iff some
    counted r is within noise (0.0452) of 0.5."""
    r_of = lambda rds: [None if rd is None or rd.get("label") in UNLABELLABLE else rd.get("r") for rd in rds]
    near = lambda rs: any(finite(r) and abs(r - th["cut"]) <= th["noise"] for r in rs)
    get = lambda k: [None if rd is None else rd.get(k) for rd in letter]
    pm = r_of(prior)
    entry = [i for i, rd in enumerate(letter) if carries(rd, th)]
    fl = r_of([letter[i] for i in entry])
    n_lab = sum(labellable(rd) for rd in letter)
    cls = "unreadable" if n_lab < th["min_entry"] else "no_carry" if len(entry) < th["min_entry"] else band_class(fl, th)
    diff = [rd["x"] - rd["x_null"] if labellable(rd) and rd is not None and finite(rd.get("x_null")) else None for rd in letter]
    return {"PM": {"class": band_class(pm, th), "r": pm, "near_cut": near(pm)},
            "FL": {"class": cls, "entry_seeds": entry, "n_labellable": n_lab, "x": get("x"), "x_null": get("x_null"),
                   "x_minus_null": diff, "r": fl, "near_cut": near(fl)}}


# ----------------------------------------------------------------------------- Amendment 7 (2026-10-30)
# experiments/open_pair/results/predictions_amend7.json; plan section "Amendment 7"; edge cases in tests/test_open_pair.py
# Fable, reports/plans/2026-10-30_advice_regating-carry-cell-FABLE.md, step 1. A read is {"x", "x_null", "r", "label"}.


def pc_classes(E: Tensor) -> list[int]:
    """Class 0-3 of each row of E: 2 * (score on PC1 > 0) + (score on PC2 > 0), the scores on the two leading principal
    components of E's rows centred on their mean (float64 on CPU, exact SVD). Each component's sign is fixed so that the
    row with the largest |score| scores positive; that changes the labels, never the partition. A score of exactly 0
    counts as not positive."""
    X = E.detach().double().cpu()
    X = X - X.mean(0)
    sc = X @ torch.linalg.svd(X, full_matrices=False)[2][:2].T
    sc = sc * sc.gather(0, sc.abs().argmax(0, keepdim=True)).sign()
    return [2 * int(a > 0) + int(b > 0) for a, b in sc.tolist()]


def shuffled_classes(cls: Sequence[int], seed: int) -> list[int]:
    """The twin's classes: `cls` permuted over the rows by torch.randperm (seed). Same class sizes, no link to the rows."""
    perm = torch.randperm(len(cls), generator=torch.Generator().manual_seed(seed))
    return [cls[int(i)] for i in perm]


def cross_class_null(s: Mapping[int, Sequence[float]], cls: Sequence[int]) -> tuple[float, float]:
    """(own, null), the numerators of x and x_null over the held-out windows. s[c][i] = the specific gain at window i of
    class c's word; cls[i] = window i's class. own = mean over windows of s[cls[i]][i] (H_spec). null = sum over the
    classes c present of (n_c / N) * (mean of s[c] over the windows whose class is not c): the same words with the same
    weights as own, each read at keys of the other classes, so any lean toward a word that does not depend on the key
    cancels exactly, whatever the class sizes. nan for no windows, or when a class present has no window outside it."""
    N = len(cls)
    if N == 0:
        return math.nan, math.nan
    own = sum(s[c][i] for i, c in enumerate(cls)) / N
    null = 0.0
    for c in sorted(set(cls)):
        out = [s[c][i] for i, k in enumerate(cls) if k != c]
        if not out:
            return own, math.nan
        null += sum(k == c for k in cls) / N * sum(out) / len(out)
    return own, null


def twin_clean(twin: Sequence[dict[str, Any] | None], th: dict[str, Any]) -> str:
    """The shuffled-class twin at step 30, the control: "unreadable" unless there are n_seeds reads, each `labellable`
    with a finite x_null; "clean" iff every x - x_null < twin_max (0.15; a tie fails); else "carries"."""
    ok = len(twin) == th["n_seeds"] and all(labellable(rd) and rd is not None and finite(rd.get("x_null")) for rd in twin)
    if not ok:
        return "unreadable"
    return "clean" if all(rd is not None and rd["x"] - rd["x_null"] < th["twin_max"] for rd in twin) else "carries"


def contrast(cls_r: Sequence[Any], twin_r: Sequence[Any], th: dict[str, Any]) -> dict[str, Any]:
    """D = mean r of the class map's carrying seeds - mean r of the twin's seeds. "unreadable" if either list is empty or
    has a non-finite r; "sees_carry" iff D > r_contrast (0.0904, twice the noise floor); "sees_carry_low" iff
    D < -r_contrast; else "kind_only" (a tie at the bound reads kind_only)."""
    if not cls_r or not twin_r or not all(finite(r) for r in [*cls_r, *twin_r]):
        return {"class": "unreadable", "D": math.nan}
    D = sum(cls_r) / len(cls_r) - sum(twin_r) / len(twin_r)
    b = th["r_contrast"]
    return {"class": "sees_carry" if D > b else "sees_carry_low" if D < -b else "kind_only", "D": D}


def score_amend7(twin: Sequence[dict[str, Any] | None], cmap: Sequence[dict[str, Any] | None],
                 ep: Sequence[dict[str, Any] | None] | None, th: dict[str, Any]) -> dict[str, Any]:
    """Step-30 reads, seeds 0-2. T: `twin_clean`. CL: "control_failed" unless T is clean; else "unreadable" if fewer than
    min_entry (2) class-map reads are `labellable`; else "no_carry" if fewer than min_entry `carries` (x - x_null > 0.3);
    else `band_class` on the carrying seeds' r. D: `contrast` of those r against the twin's r, read only when CL is a
    band class ("not_read" otherwise). EP: "skipped" when ep is None (its table could not be built); else `band_class`
    on its r. A read too small to label counts as missing r."""
    r_of = lambda rds: [None if rd is None or rd.get("label") in UNLABELLABLE else rd.get("r") for rd in rds]
    diff = lambda rds: [rd["x"] - rd["x_null"] if labellable(rd) and rd is not None and finite(rd.get("x_null")) else None for rd in rds]
    T = twin_clean(twin, th)
    entry = [i for i, rd in enumerate(cmap) if carries(rd, th)]
    n_lab = sum(labellable(rd) for rd in cmap)
    cl_r = r_of([cmap[i] for i in entry])
    if T != "clean":
        cl = "control_failed"
    else:
        cl = "unreadable" if n_lab < th["min_entry"] else "no_carry" if len(entry) < th["min_entry"] else band_class(cl_r, th)
    D = contrast(cl_r, r_of(twin), th) if cl in ("cipher_band", "between", "own_guess_band") else {"class": "not_read", "D": math.nan}
    return {"T": {"class": T, "x_minus_null": diff(twin), "r": r_of(twin), "band": band_class(r_of(twin), th)},
            "CL": {"class": cl, "entry_seeds": entry, "n_labellable": n_lab, "x_minus_null": diff(cmap), "r": cl_r},
            "D": D, "EP": {"class": "skipped"} if ep is None else {"class": band_class(r_of(ep), th), "r": r_of(ep)}}
