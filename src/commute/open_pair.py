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


def p2(st: dict[str, dict[str, Any]], seeds: list[int], inst: str, cut: float) -> dict[str, Any]:
    """P2 for one instrument (r or ratio): on every elicit run labelled "reuses" or "new", the instrument's label
    (`inst_label`) equals it. Runs whose reading is not finite are listed and left out. "not_testable" when no run is
    left; "split" is reported beside when the labelled runs carry both labels."""
    runs, dropped = [], []
    for s in seeds:
        rec = st.get(f"elicit_s{s}")
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
