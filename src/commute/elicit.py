"""E54's key -> value map that the original model W0 already half-knows, and the filed rules that read it. Plan:
reports/plans/2026-10-29_plan_e54-keydependent-elicit.md; rules and thresholds in
experiments/keydependent_elicit/results/predictions.json. Pure functions on tensors and numbers (no model), so every
rule and its edge cases run on CPU (tests/test_elicit.py).

Value of a key x: the most probable token v under W0's mean next-token distribution after "M x" that is word-like,
v != x, P(v | M x) in [lo, hi], W0's rank of v in [2, 20] (rank 1 = the top token; rank = 1 + the number of tokens with
strictly higher probability), and key-specific: P(v | M x) >= ratio x the median over the other candidate keys x' of
P(v | M x') (torch.median: the lower median for an even count).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def pick_values(dist: Tensor, cands: list[int], plain: Tensor, band: tuple[float, float], ranks: tuple[int, int],
                ratio: float) -> dict[int, dict[str, Any]]:
    """`dist` (n_cands, V): row i is W0's mean distribution after "M cands[i]". Returns {key: {value, p, rank,
    median_other}} for the keys that have a value (see the module docstring)."""
    assert dist.shape[0] == len(cands) >= 2 and plain.shape[0] >= dist.shape[1]
    out: dict[int, dict[str, Any]] = {}
    for i, x in enumerate(cands):
        P = dist[i]
        top = torch.topk(P, min(ranks[1], P.numel())).indices.tolist()
        for v in sorted(top, key=lambda u: (-float(P[u]), u)):
            p, rank = float(P[v]), 1 + int((P > P[v]).sum())
            if v == x or not bool(plain[v]) or not band[0] <= p <= band[1] or not ranks[0] <= rank <= ranks[1]:
                continue
            med = float(torch.cat([dist[:i, v], dist[i + 1 :, v]]).median())
            if p >= ratio * med:
                out[x] = {"value": v, "p": p, "rank": rank, "median_other": med}
                break
    return out


def draw_keys(qualified: dict[int, dict[str, Any]], cands: list[int], k: int, seed: int) -> list[int]:
    """Up to k keys, drawn in a seeded random order among the qualified ones, skipping a key whose value is already
    taken (values unique); returned in candidate (frequency-rank) order."""
    pool = [c for c in cands if c in qualified]
    order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed)).tolist()
    keys: list[int] = []
    used: set[int] = set()
    for j in order:
        x = pool[j]
        if len(keys) == k:
            break
        if qualified[x]["value"] not in used:
            keys.append(x)
            used.add(qualified[x]["value"])
    return sorted(keys, key=cands.index)


def summed_norm(dW: dict[str, Tensor]) -> float:
    """||delta-W||: the Frobenius norms of the matrices, summed (float64)."""
    return sum(float(d.double().norm()) for d in dW.values())


def matched_step(norms: dict[int, float], target: float) -> int:
    """The checkpoint whose ||delta-W|| is nearest `target`; ties to the lower step."""
    return min(norms, key=lambda s: (abs(norms[s] - target), s))


def floor_of(G0: float, G1: float, floor_min: float, same_step: bool) -> float:
    """E52's per-task gain floor over its two seeds: max(floor_min, 2 x |G(s0) - G(s1)|). The seed-spread term is only
    meaningful for two seeds read at the same step; when their steps differ the floor is floor_min alone."""
    return max(floor_min, 2 * abs(G0 - G1)) if same_step else floor_min


def share_or_nan(share: float, gain: float, tol: float = 1e-6) -> float:
    """A share is nan when its denominator (the gain it divides by) is within tol of 0."""
    return math.nan if abs(gain) < tol else share


@torch.no_grad()
def geo_mean_p(logits_fn: Any, toks: Tensor, half: int, values: dict[int, int], device: str, batch: int = 64) -> dict[int, float]:
    """Per key x in `values`, exp(mean over the windows in `toks` (N, T) holding x at `half` of log P(values[x] | window))."""
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    toks = toks[torch.isin(toks[:, half], torch.tensor(list(values)))]
    for s in range(0, len(toks), batch):
        t = toks[s : s + batch].to(device)
        lp = torch.log_softmax(logits_fn(t)[:, half].float(), -1).double().cpu()
        for i, k in enumerate(t[:, half].tolist()):
            sums[k] = sums.get(k, 0.0) + float(lp[i, values[k]])
            counts[k] = counts.get(k, 0) + 1
    return {k: math.exp(sums[k] / counts[k]) for k in sums}


def readable(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def absdiff(a: float | None, b: float | None) -> float | None:
    return abs(a - b) if a is not None and b is not None and readable(a) and readable(b) else None


def diff_or_none(a: float | None, b: float | None) -> float | None:
    """a - b, or None when either is missing or not finite."""
    return a - b if a is not None and b is not None and readable(a) and readable(b) else None


def separation(a: list[float | None], b: list[float | None], min_gap: float) -> dict[str, Any]:
    """E51/E52's rule with a minimum: kinds a and b separate iff |mean a - mean b| > the larger within-kind range
    (strict) and >= min_gap. Verdict "a_higher", "b_higher", "none", or "unreadable" (a value missing or not finite,
    or fewer than two per kind)."""
    if len(a) < 2 or len(b) < 2 or not all(readable(x) for x in a + b):
        return {"verdict": "unreadable", "a": a, "b": b}
    A, B = [float(x) for x in a if x is not None], [float(x) for x in b if x is not None]
    gap = sum(A) / len(A) - sum(B) / len(B)
    spread = max(max(A) - min(A), max(B) - min(B))
    sep = abs(gap) > spread and abs(gap) >= min_gap
    return {"verdict": ("a_higher" if gap > 0 else "b_higher") if sep else "none", "gap": gap, "spread": spread, "a": A, "b": B}


def score(st: dict[str, dict[str, Any]], th: dict[str, float], meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """P1-P3 and the decision tree on stage 2's states: elicit_s{0,1}, cipher_s{0,1} (at the matched norm) and
    cipher_own_s{0,1} (at its own E12 step). A state's S and r are None when it is under its floor. `meta` is stage 2's
    fine-tune record: elicit_s* "learned", cipher_s* "at_bound" and "same_state" (matched step == own step)."""
    get = lambda name, m: st[name].get(m) if st[name]["readable"] else None
    seeds = ("s0", "s1")
    p1 = separation([get(f"elicit_{s}", "r") for s in seeds], [get(f"cipher_{s}", "r") for s in seeds], th["min_gap"])
    p1["status"] = "unreadable" if p1["verdict"] == "unreadable" else "pass" if p1["verdict"] == "a_higher" else "fail"
    S = [get(f"{k}_{s}", "S") for k in ("elicit", "cipher") for s in seeds]
    p2 = separation(S[:2], S[2:], th["min_gap"])
    p2["status"] = ("unreadable" if p2["verdict"] == "unreadable" else
                    "pass" if p2["verdict"] == "none" and all(float(x) <= th["p2_max_S"] for x in S if x is not None) else "fail")
    diffs = {s: absdiff(get(f"cipher_{s}", "r"), get(f"cipher_own_{s}", "r")) for s in seeds}
    same = [s for s in seeds if meta[f"cipher_{s}"]["same_state"]]
    p3: dict[str, Any] = {"diff": diffs, "same_state": same}
    if not all(readable(d) for d in diffs.values()):
        p3["status"] = "unreadable"
    elif any(float(d) >= th["p3_tol"] for s, d in diffs.items() if s not in same and d is not None):
        p3["status"] = "fail"
    else:  # a seed whose matched step is its own step has difference 0 by construction: it cannot pass
        p3["status"] = "uninformative" if same else "pass"
    out = {"P1": p1, "P2": p2, "P3": p3, "branch": branch(p1, p2)}
    if p2["status"] == "fail" and p2["verdict"] == "none":
        out["branch"] += " (P2 missed on the S bound only, S does not separate: the tree does not cover it; tell the user)"
    notes = [f"cipher {s} at_bound (matched step 30: the size match is only approximate)" for s in seeds if meta[f"cipher_{s}"]["at_bound"]]
    notes += [f"elicit {s} did not learn (scored at step 300)" for s in seeds if not meta[f"elicit_{s}"]["learned"]]
    if same:
        notes.append(f"size match collapsed on {', '.join(same)} (both tasks at the same step): P1 is read at the cipher's own "
                     "step, not at a separate matched size; P3 uninformative there")
    return {**out, "branch": out["branch"] + "".join(f" [{n}]" for n in notes)}


def branch(p1: dict[str, Any], p2: dict[str, Any]) -> str:
    a, b = p1["status"], p2["status"]
    if "unreadable" in (a, b):
        return "unreadable: a state under its floor or a nan; stop and tell the user"
    rev = [m for m, p in (("P1: cipher r above elicit r", p1), ("P2: cipher S above elicit S", p2)) if p["verdict"] == "b_higher"]
    if rev:
        return f"reversed ({'; '.join(rev)}): the tree does not cover it; tell the user"
    return {("pass", "pass"): "VPD's route split reads a kind steering cannot: headline case; next Qwen pair and Jain's readout",
            ("pass", "fail"): "steering not blind to this key-dependence: the objection holds on both sets",
            ("fail", "pass"): "neither tells kind here: r on E12 read input-dependence or size; tell the user before using E12's r",
            ("fail", "fail"): "steering beats VPD here: the objection holds"}[(a, b)]


def score_amend1(st: dict[str, dict[str, Any]], th: dict[str, float]) -> dict[str, Any]:
    """E54 Amendment 1's filed rules on {state: {"S", "S_lo", "r"}} for elicit_s*, cipher_s* (matched size) and
    cipher_own_s*. A missing state or value, or a nan, makes its rule "unreadable".
    P1-lo: `separation` of S_lo, elicit against the matched cipher; pass iff "none" (predicted: no separation).
    P2-lo: r - S_lo >= th["p2_min_diff"] on both elicit states (a tie at the bound passes); pass iff both.
    P3-own: `separation` of S (nats), elicit against the cipher at its own step; pass iff "a_higher"."""
    seeds = ("s0", "s1")
    col = lambda kind, m: [st.get(f"{kind}_{s}", {}).get(m) for s in seeds]
    p1 = separation(col("elicit", "S_lo"), col("cipher", "S_lo"), th["min_gap"])
    p1["status"] = "unreadable" if p1["verdict"] == "unreadable" else "pass" if p1["verdict"] == "none" else "fail"
    diffs = {s: diff_or_none(r, lo) for s, r, lo in zip(seeds, col("elicit", "r"), col("elicit", "S_lo"), strict=True)}
    p2: dict[str, Any] = {"diff": diffs, "status": "unreadable" if any(d is None for d in diffs.values()) else
                          "pass" if all(float(d) >= th["p2_min_diff"] for d in diffs.values() if d is not None) else "fail"}
    p3 = separation(col("elicit", "S"), col("cipher_own", "S"), th["min_gap"])
    p3["status"] = "unreadable" if p3["verdict"] == "unreadable" else "pass" if p3["verdict"] == "a_higher" else "fail"
    return {"P1-lo": p1, "P2-lo": p2, "P3-own": p3, "branch": branch_amend1(p1, p2, p3)}


def branch_amend1(p1: dict[str, Any], p2: dict[str, Any], p3: dict[str, Any]) -> str:
    """Amendment 1's tree. It feeds the analysis's ratio table only (the steering gap in r's units)."""
    if "unreadable" in (p1["status"], p2["status"], p3["status"]):
        return "unreadable: a missing state or a nan; stop and tell the user"
    if p1["verdict"] == "b_higher":
        out = "reversed: the matched cipher's S_lo separates above the elicit map's; the tree does not cover it; tell the user"
    elif p1["status"] == "pass":
        out = "as predicted: in log-odds S does not separate the pair; the ratio table's log-odds row becomes r gap / S_lo gap, measured"
    else:
        out = ("S_lo separates elicit higher: in r's unit steering does separate the matched pair; the table's log-odds row "
               "(estimated 'over 10x') is replaced by the measured r gap / S_lo gap; tell the user before editing the report")
    if any(float(d) < 0 for d in p2["diff"].values()):
        out += " [P2-lo reversed: an elicit S_lo above its r; tell the user]"
    elif p2["status"] == "fail":
        out += " [P2-lo missed: an elicit S_lo within 0.3 of its r; tell the user]"
    out += (" [P3-own: S in nats separates the elicit map from the cipher at its own step]" if p3["status"] == "pass" else
            " [P3-own missed: the stage-0 numbers do not reproduce E54's analysis; check the code before reading]")
    return out
