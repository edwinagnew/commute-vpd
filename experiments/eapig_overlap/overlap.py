"""P2.69's counts on two circuits (no GPU, no torch): Wang's ΔS and added / removed / unchanged lists, Hanna's node and
edge IoU with the hypergeometric p (ours, D8 and D9, beside the released code's), §4.3's 95th-percentile sensitivity,
the faithfulness gate's reading and the controls' rules. A circuit is the worker's record: {"edges", "nodes", "empty"}.

Filed edge cases (results/predictions.json "rules"): an IoU over an empty union is None; ΔS with an empty initial
circuit is None; two empty circuits are not equal (control 1 cannot pass that way); a range of no defined IoU is None
and the rules that read it record "no reading", a miss.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Any

import numpy as np

N_EDGES = 927  # input, 24 heads (q, k, v inputs), 4 MLPs, logits: 828 + 70 + 29
N_NODES_CODE = 30 - 2  # Hanna's code: len(g.nodes) - 2
ENDS = {"input", "logits"}


def iou(a: set[str], b: set[str]) -> float | None:
    u = a | b
    return len(a & b) / len(u) if u else None


def tail(x: int, total: int, n: int, draws: int, strict: bool) -> float | None:
    """Exact hypergeometric tail: P(X >= x) (Hanna's paper, ours), or P(X > x) (her code, `1 - hypergeom.cdf(x, ...)`).
    None when a set is larger than the population: the code's node sets keep 'input' against a population of 28, so a
    circuit holding all 28 nodes and the input has no code value (scipy would return nan)."""
    if n > total or draws > total:
        return None
    lo, hi = max(0, draws + n - total), min(n, draws)
    num = sum(comb(n, k) * comb(total - n, draws - k) for k in range(max(x + strict, lo), hi + 1))
    return num / comb(total, draws)


def moved(a: set[str], b: set[str]) -> dict[str, Any]:
    return {"unchanged": len(a & b), "added": len(b - a), "removed": len(a - b), "added_names": sorted(b - a), "removed_names": sorted(a - b)}


def delta_s(a: set[str], b: set[str]) -> float | None:
    """Wang's ΔS at n = 1: (added + removed) / |initial| x 100."""
    return 100 * len(a ^ b) / len(a) if a else None


def compare(A: dict[str, Any], B: dict[str, Any]) -> dict[str, Any]:
    """A is M's circuit (the initial), B is F's; sizes after pruning beside every number."""
    ea, eb, na, nb = set(A["edges"]), set(B["edges"]), set(A["nodes"]), set(B["nodes"])
    oa, ob = na - ENDS, nb - ENDS  # D8: input and output nodes excluded
    ca, cb = na - {"inputs", "logits"}, nb - {"inputs", "logits"}  # the released code's sets ('input' stays)
    return {"sizes": {"M_edges": len(ea), "F_edges": len(eb), "M_nodes": len(na), "F_nodes": len(nb), "M_empty": A["empty"], "F_empty": B["empty"]},
            "edges": moved(ea, eb), "nodes": moved(na, nb), "delta_S_edges": delta_s(ea, eb), "delta_S_nodes": delta_s(na, nb),
            "edge_iou": iou(ea, eb), "node_iou": iou(oa, ob), "node_iou_code": iou(ca, cb),
            "edge_p": tail(len(ea & eb), N_EDGES, len(ea), len(eb), False), "edge_p_code": tail(len(ea & eb), N_EDGES, len(ea), len(eb), True),
            "node_p": tail(len(oa & ob), N_NODES_CODE, len(oa), len(ob), False), "node_p_code": tail(len(ca & cb), N_NODES_CODE, len(ca), len(cb), True),
            "same_edges": ea == eb and not A["empty"] and not B["empty"]}  # fmt: skip


def above_percentile(scores: dict[str, float], edges: list[str], pct: float) -> tuple[float | None, set[str]]:
    """§4.3 as FC's notebook computes it: the percentile of |score| over the circuit's own edges, then those at or above it."""
    if not edges:
        return None, set()
    t = float(np.percentile([abs(scores[e]) for e in edges], pct))
    return t, {e for e in edges if abs(scores[e]) >= t}


def sensitivity(sa: dict[str, float], A: dict[str, Any], sb: dict[str, float], B: dict[str, Any], pct: float) -> dict[str, Any]:
    ta, ea = above_percentile(sa, A["edges"], pct)
    tb, eb = above_percentile(sb, B["edges"], pct)
    return {"threshold_M": ta, "threshold_F": tb, "M_edges": len(ea), "F_edges": len(eb), "edge_iou": iou(ea, eb)} | moved(ea, eb)


def reading(rm: dict[str, Any], rf: dict[str, Any], bar: float) -> str:
    """The headline gate, in this order: "no gap" (either model's F(M) at or under the fp32 floor or not finite), "empty"
    (either circuit has no output node after pruning), "circuit not faithful" (either under `bar` %), else "read"."""
    if rm["no_gap"] or rf["no_gap"]:
        return "no gap"
    if rm["circuit"]["empty"] or rf["circuit"]["empty"]:
        return "empty"
    if not (rm["faithfulness_pct"] >= bar and rf["faithfulness_pct"] >= bar):
        return "circuit not faithful"
    return "read"


def pairwise(circuits: list[dict[str, Any]]) -> list[float | None]:
    """Edge IoU between every two resampled circuits (an empty circuit's edges are the empty set)."""
    return [iou(set(a["edges"]) if not a["empty"] else set(), set(b["edges"]) if not b["empty"] else set()) for a, b in combinations(circuits, 2)]


def span(values: list[float | None]) -> list[float] | None:
    got = [v for v in values if v is not None]
    return [min(got), max(got)] if got else None


def below(x: float | None, r: list[float] | None) -> str:
    """Control 2: "pass" when the IoU is defined and strictly under the range's minimum."""
    if x is None or r is None:
        return "no reading (miss)"
    return "pass" if x < r[0] else "miss"


def inside(x: float | None, r: list[float] | None) -> str:
    if x is None or r is None:
        return "no reading (miss)"
    return "pass" if r[0] <= x <= r[1] else "miss"
