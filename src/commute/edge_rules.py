"""E97 Amendment 1's edge graph of the 67M and its filed rules, pure, so every edge case runs on CPU
(tests/test_edge_rules.py). The reads are `commute.edges`; plan reports/plans/2026-11-05_plan_e97-amend1.md.

The graph (Hanna et al. 2024's, at head grain): 29 upstream nodes, the embedding, 24 heads and 4 MLPs, each writing to
the residual stream; 77 downstream inputs, each later head's q, k and v input separately, each MLP's input and the
logits. An edge u -> v exists when u writes before v reads: 828 into heads, 70 into MLPs, 29 into the logits, 927.
Downstream input v of block b is numbered 19 b + {q: 0, k: 6, v: 12} + head, 19 b + 18 for the MLP, 76 the logits.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from commute.copying import finite
from commute.locate import ranked
from commute.locator_rules import frac, top_part, trunc2

BLOCKS, HEADS = 4, 6
LOGITS = 19 * BLOCKS
UP = ["emb"] + [n for b in range(BLOCKS) for n in [*(f"{b},{h}" for h in range(HEADS)), f"mlp{b}"]]
DOWN = [n for b in range(BLOCKS) for n in [*(f"{b},{h}.{x}" for x in "qkv" for h in range(HEADS)), f"mlp{b}"]] + ["logits"]


def avail(v: int) -> int:
    """How many upstream nodes (in UP's order) write before input v reads."""
    b, r = divmod(v, 19)
    return len(UP) if v == LOGITS else 1 + 7 * b + (6 if r == 18 else 0)


def reader(v: int) -> int | None:
    """The upstream node that input v feeds (a head reads its q, k, v; an MLP its input); None for the logits."""
    b, r = divmod(v, 19)
    return None if v == LOGITS else 1 + 7 * b + (6 if r == 18 else r % HEADS)


def block_of(v: int) -> int:
    return BLOCKS if v == LOGITS else v // 19


VALID = torch.tensor([[u < avail(v) for v in range(len(DOWN))] for u in range(len(UP))])
assert int(VALID.sum()) == 927


def edge_name(u: int, v: int) -> str:
    return f"{UP[u]}->{DOWN[v]}"


EDGES = [(u, v) for u in range(len(UP)) for v in range(len(DOWN)) if VALID[u, v]]
NAMES = [edge_name(u, v) for u, v in EDGES]


def as_dict(scores: Tensor) -> dict[str, float]:
    """A (29, 77) score matrix as {edge name: score}, valid edges only."""
    return {edge_name(u, v): float(scores[u, v]) for u, v in EDGES}


# ----------------------------------------------------------------------------- circuits (Hanna et al.'s code, graph.py)


def prune(mask: Tensor) -> Tensor:
    """eap-ig's `Graph.prune`: a node stays when it has an outgoing edge and an incoming one (the embedding counts as
    having one); an edge stays when its parent stays and its child's node stays (the logits always do). To a fixed point."""
    m = mask & VALID
    while True:
        inc = torch.zeros(len(UP), dtype=torch.bool)
        for v in range(LOGITS):
            if m[:, v].any():
                inc[reader(v)] = True  # type: ignore[index]
        inc[0] = True
        alive = m.any(1) & inc
        child = torch.tensor([v == LOGITS or bool(alive[reader(v)]) for v in range(len(DOWN))])  # type: ignore[index]
        new = m & alive[:, None] & child[None, :]
        if torch.equal(new, m):
            return m
        m = new


def topn(scores: Tensor, n: int) -> Tensor:
    """eap-ig's `apply_topn(n, absolute=True)` then `prune`: the n edges of largest |score| (ties in edge order)."""
    s = scores.abs().double().masked_fill(~VALID, -math.inf).flatten()
    order = torch.argsort(s, descending=True, stable=True)
    keep = torch.zeros(s.numel(), dtype=torch.bool)
    keep[order[: max(0, n)]] = True
    return prune(keep.view(VALID.shape) & VALID)


def faith(m: float, b: float, b_bad: float, min_gap: float) -> float:
    """Normalised faithfulness (m - b') / (b - b'). nan ("no gap") unless the clean-minus-corrupted gap b - b' is above
    `min_gap` nats, so the ratio never divides by a gap near 0; nan on any non-finite input."""
    if not all(finite(x) for x in (m, b, b_bad)) or b - b_bad <= min_gap:
        return math.nan
    return (m - b_bad) / (b - b_bad)


def smallest(sweep: Sequence[tuple[int, float]], bar: float) -> int | None:
    """The smallest circuit size whose faithfulness is at least `bar`; None when none is (or none is finite)."""
    ok = [n for n, f in sweep if finite(f) and f >= bar]
    return min(ok) if ok else None


def iou(a: Tensor, b: Tensor) -> float:
    """Edge overlap |a & b| / |a | b|; nan when both are empty."""
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else math.nan


def has_parts(mask: Tensor, part: str | None) -> bool | None:
    """Every part of `part` ("2,4" or a pair "2,3+2,4") has an outgoing edge in the circuit; None for no part."""
    if not part:
        return None
    return all(bool(mask[UP.index(p)].any()) for p in part.split("+"))


# ----------------------------------------------------------------------------- the gain path's numbers


def top_edge(scores: Mapping[str, float]) -> str | None:
    """The one edge of largest score (signed: the edge that carries most of the gain); a tie or nothing finite: None."""
    return top_part(scores)


def direct_share(scores: Mapping[str, float], part: str | None, G: float) -> float:
    """The part's edge to the logits as a share of G (a pair: the sum of its two edges); nan without a part or a gain."""
    if not part:
        return math.nan
    return frac(sum(scores.get(f"{p}->logits", math.nan) for p in part.split("+")), G)


def out_sums(scores: Mapping[str, float]) -> dict[str, float]:
    """Each upstream node's single-edge scores summed over its outgoing edges."""
    out = dict.fromkeys(UP, 0.0)
    for n, x in scores.items():
        out[n.split("->")[0]] += x
    return out


def edge_nominees(scores: Mapping[str, float], k: int) -> list[str]:
    """The parts (never the embedding) that write the top k edges by score, first-seen order, no repeats."""
    return list(dict.fromkeys(n.split("->")[0] for n in ranked(scores)[:k] if not n.startswith("emb->")))


def sum_misses(sums: Mapping[str, float], carry: Mapping[str, float], tol: float) -> list[str]:
    """Stage 0's check on E62: each part's summed edge share against E97's carry share; nan misses."""
    return [f"{n} sum {sums.get(n)} carry {c}" for n, c in carry.items() if not (finite(sums.get(n)) and finite(c) and abs(sums[n] - c) <= tol)]


# ----------------------------------------------------------------------------- the band, set from copy and lookup


def ceil2(x: float) -> float:
    return math.ceil(x * 100) / 100 if finite(x) else math.nan


def band(copy: Sequence[float], lookup: Sequence[float]) -> dict[str, Any]:
    """From the copy and lookup seeds' cross-faithfulness, as E97's a was set from stage 1: reuse reads at least the
    weakest copy seed (truncated to two decimals), not-reuse at most the strongest lookup seed (rounded up). The band
    separates only when the first is strictly above the second; a nan or no seed on either side: no band."""
    if not copy or not lookup or not all(finite(x) for x in [*copy, *lookup]):
        return {"separated": False, "direction": "unreadable", "reuse_at_least": math.nan, "not_reuse_at_most": math.nan}
    hi, lo = trunc2(min(copy)), ceil2(max(lookup))
    rev = trunc2(min(lookup)) > ceil2(max(copy))
    direction = "copy above lookup" if hi > lo else "lookup above copy" if rev else "overlap"
    return {"separated": hi > lo, "direction": direction, "reuse_at_least": hi, "not_reuse_at_most": lo}


def band_reading(x: float, b: Mapping[str, Any]) -> str:
    if not finite(x):
        return "unreadable: no gap"
    if not b["separated"]:
        return "no band"
    return "like copy" if x >= b["reuse_at_least"] else "like lookup" if x <= b["not_reuse_at_most"] else "between"


def outcome(copy_tops: Sequence[str | None], expected: str, b: Mapping[str, Any]) -> tuple[bool, str]:
    """The plan's tree after the E62 stage: (go on to lowercase and stage 3, what to say)."""
    if not copy_tops or any(t != expected for t in copy_tops):
        return False, f"{expected} not top on copy ({list(copy_tops)}): edges also read propagation; stop, write up, then the literature review"
    if b["direction"] == "copy above lookup":
        return True, "copy high and lookup low: read lowercase, stage 3's pairs and the beside-only rows with it"
    if b["direction"] == "lookup above copy":
        return False, "REVERSED: lookup's cross-faithfulness above copy's; report it loudly, no further stage"
    return False, f"copy and lookup do not separate ({b['direction']}): circuits from pairs don't separate here; report it and do the literature review"
