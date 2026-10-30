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


def floor_of(G0: float, G1: float, floor_min: float) -> float:
    """E52's per-task gain floor over its two seeds: max(floor_min, 2 x |G(s0) - G(s1)|)."""
    return max(floor_min, 2 * abs(G0 - G1))


def readable(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def absdiff(a: float | None, b: float | None) -> float | None:
    return abs(a - b) if a is not None and b is not None and readable(a) and readable(b) else None


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


def score(st: dict[str, dict[str, Any]], th: dict[str, float]) -> dict[str, Any]:
    """P1-P3 and the decision tree on stage 2's states: elicit_s{0,1}, cipher_s{0,1} (at the matched norm) and
    cipher_own_s{0,1} (at its own E12 step). A state's S and r are None when it is under its floor."""
    get = lambda name, m: st[name].get(m) if st[name]["readable"] else None
    seeds = ("s0", "s1")
    p1 = separation([get(f"elicit_{s}", "r") for s in seeds], [get(f"cipher_{s}", "r") for s in seeds], th["min_gap"])
    p1["status"] = "unreadable" if p1["verdict"] == "unreadable" else "pass" if p1["verdict"] == "a_higher" else "fail"
    S = [get(f"{k}_{s}", "S") for k in ("elicit", "cipher") for s in seeds]
    p2 = separation(S[:2], S[2:], th["min_gap"])
    p2["status"] = ("unreadable" if p2["verdict"] == "unreadable" else
                    "pass" if p2["verdict"] == "none" and all(float(x) <= th["p2_max_S"] for x in S if x is not None) else "fail")
    diffs = {s: absdiff(get(f"cipher_{s}", "r"), get(f"cipher_own_{s}", "r")) for s in seeds}
    p3 = {"diff": diffs, "status": "unreadable" if not all(readable(d) for d in diffs.values())
          else "pass" if all(float(d) < th["p3_tol"] for d in diffs.values() if d is not None) else "fail"}
    out = {"P1": p1, "P2": p2, "P3": p3, "branch": branch(p1["status"], p2["status"])}
    if p2["status"] == "fail" and p2["verdict"] == "none":
        out["branch"] += " (P2 missed on the S bound only, S does not separate: the tree does not cover it; tell the user)"
    return out


def branch(p1: str, p2: str) -> str:
    if "unreadable" in (p1, p2):
        return "unreadable: a state under its floor or a nan; stop and tell the user"
    return {("pass", "pass"): "VPD's route split reads a kind steering cannot: headline case; next Qwen pair and Jain's readout",
            ("pass", "fail"): "steering not blind to this key-dependence: the objection holds on both sets",
            ("fail", "pass"): "neither tells kind here: r on E12 read input-dependence or size; tell the user before using E12's r",
            ("fail", "fail"): "steering beats VPD here: the objection holds"}[(p1, p2)]
