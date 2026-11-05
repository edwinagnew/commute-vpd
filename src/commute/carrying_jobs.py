"""E101's pure rules and set builders (experiments/carrying_jobs; plan reports/plans/2026-11-05_plan_e101-carrying-jobs.md).
No model here, so every edge case runs on CPU (tests/test_carrying_jobs.py). Stage 0 (`run.py`) and the scorer
(`score.py`) call these; the thresholds are in experiments/carrying_jobs/results/predictions.json.

A neuron is a flat index l x 3072 + i over the 67M's four blocks, as `neuron_r`. Its *engaged value* on a task is the mean,
over the task's read windows, of W0's activation at the answer position as a quantile of that neuron's own activations on
Pile positions (`quantiles`). A set's *engaged share* is the median of its neurons' engaged values (`set_share`).
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import elicit as EL

D_MLP = 3072


def quantiles(sorted_sample: Tensor, values: Tensor) -> Tensor:
    """(n, P) per-neuron Pile activations sorted along dim 1, (n, W) values -> (n, W): the share of the neuron's Pile
    sample at or below each value (empirical CDF, right-continuous). 0 below the sample, 1 at or above its max."""
    assert sorted_sample.shape[0] == values.shape[0] and sorted_sample.shape[1] > 0
    idx = torch.searchsorted(sorted_sample.contiguous(), values.contiguous(), right=True)
    return idx.double() / sorted_sample.shape[1]


def set_share(engaged: Sequence[float], idx: Sequence[int]) -> float:
    """The median engaged value over a set (mean of the middle two when the set is even). An empty set has none."""
    if not idx:
        return math.nan
    return float(statistics.median(engaged[i] for i in idx))


def blocks_of(idx: Sequence[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for i in idx:
        out[i // D_MLP] = out.get(i // D_MLP, 0) + 1
    return out


def block_matched(idx: Sequence[int], exclude: set[int], gen: torch.Generator) -> list[int]:
    """A random set with the same number of neurons per block as `idx`, none in `exclude`, drawn with `gen`."""
    out: list[int] = []
    for b, n in sorted(blocks_of(idx).items()):
        pool = [b * D_MLP + i for i in range(D_MLP) if b * D_MLP + i not in exclude]
        assert len(pool) >= n, f"block {b}: {len(pool)} neurons left for {n}"
        out += [pool[j] for j in torch.randperm(len(pool), generator=gen)[:n].tolist()]
    return out


def band(carrying: float, draws: Sequence[float]) -> dict[str, Any]:
    """A state's carrying share against its random draws: excess over the draws' median, and above the band (over every
    draw). Needs at least two finite draws."""
    ok = [d for d in draws if math.isfinite(d)]
    if len(ok) < 2 or not math.isfinite(carrying):
        return {"excess": math.nan, "above_band": None, "draw_min": None, "draw_max": None}
    return {"excess": carrying - statistics.median(ok), "above_band": carrying > max(ok), "draw_min": min(ok),
            "draw_max": max(ok), "draw_median": statistics.median(ok)}


def row_rule(excess: Mapping[str, float], row: Mapping[str, Sequence[str]], min_gap: float) -> dict[str, Any]:
    """E51/E52's rule (`elicit.separation`) on a row's excesses, reuse side first. States missing from `excess` drop
    out; a non-finite excess makes the row unreadable, as the rule does."""
    a = [excess[n] for n in row["reuse"] if n in excess]
    b = [excess[n] for n in row["new"] if n in excess]
    return EL.separation(list(a), list(b), min_gap)


def consensus(sets: Sequence[Sequence[int]], n: int) -> list[int]:
    """The n neurons in most of `sets`, ties by their best rank in any set, then by index: a deterministic sample of a
    task's carrying neurons that favours the ones every seed found."""
    count: dict[int, int] = {}
    best: dict[int, int] = {}
    for s in sets:
        for r, i in enumerate(s):
            count[i] = count.get(i, 0) + 1
            best[i] = min(best.get(i, r), r)
    order = sorted(count, key=lambda i: (-count[i], best[i], i))
    assert len(order) >= n, f"{len(order)} neurons for a sample of {n}"
    return order[:n]


def neuron_id(salt: str, flat: int) -> str:
    return hashlib.sha256(f"{salt}:{flat}".encode()).hexdigest()[:12]


def mark(pieces: Sequence[str], at: int) -> str:
    """Decoded token pieces joined, the one at `at` wrapped in ⟦ ⟧."""
    return "".join(pieces[:at]) + "⟦" + pieces[at] + "⟧" + "".join(pieces[at + 1 :])


# ----------------------------------------------------------------------------- stage 1


def outcome(examples: Sequence[float], near: Sequence[float], top: float, frac: float) -> str | None:
    """The essay's test on one neuron: an example or near miss fires strongly at >= frac of the neuron's strongest Pile
    activation `top`. 'win' if more examples than near misses fire strongly, 'tie' if as many, 'lose' if fewer. None
    (unscorable) when top <= 0 or either list is empty."""
    if top <= 0 or not examples or not near:
        return None
    se, sn = sum(x >= frac * top for x in examples), sum(x >= frac * top for x in near)
    return "win" if se > sn else "tie" if se == sn else "lose"


def rate(hits: Sequence[bool]) -> tuple[float, int]:
    return (sum(hits) / len(hits), len(hits)) if hits else (math.nan, 0)


def var(p: float, n: int) -> float:
    return p * (1 - p) / n


def decide(gap: float, se: float, min_gap: float) -> str:
    """'above' iff gap >= min_gap and gap >= 2 se; 'below_or_level' iff gap < min_gap and gap <= 2 se (so equal rates
    of 0 or 1, se 0, read level, not no call); else 'no_call'."""
    if gap >= min_gap and gap >= 2 * se:
        return "above"
    return "below_or_level" if gap < min_gap and gap <= 2 * se else "no_call"


def ok_side(p: float, n: int) -> bool:
    return n > 0 and math.isfinite(p)


def gap_rule(p1: float, n1: int, p0: float, n0: int, min_gap: float) -> dict[str, Any]:
    """Is the rate p1 (of n1) above the floor p0 (of n0)? `decide` on gap = p1 - p0, se its binomial sd. 'unreadable'
    when a side is empty or not finite."""
    rec = {"p1": p1, "n1": n1, "p0": p0, "n0": n0}
    if not (ok_side(p1, n1) and ok_side(p0, n0)):
        return {"verdict": "unreadable", **rec}
    gap, se = p1 - p0, math.sqrt(var(p1, n1) + var(p0, n0))
    return {"verdict": decide(gap, se, min_gap), "gap": gap, "se": se, **rec}


def row_kind(reuse: Mapping[str, Any], new: Mapping[str, Any], min_gap: float) -> dict[str, Any]:
    """P2's row. Each side is {'p', 'n', 'floor', 'n_floor'}: its related share and its own description's random floor.
    x = p - floor per side; `decide` on x_reuse - x_new, se from all four binomial variances (the two floors share the
    random pool, so this se is conservative when the floors are positively correlated)."""
    sides = {"reuse": dict(reuse), "new": dict(new)}
    if not all(ok_side(s["p"], s["n"]) and ok_side(s["floor"], s["n_floor"]) for s in sides.values()):
        return {"verdict": "unreadable", **sides}
    xr, xn = reuse["p"] - reuse["floor"], new["p"] - new["floor"]
    se = math.sqrt(sum(var(s["p"], s["n"]) + var(s["floor"], s["n_floor"]) for s in sides.values()))
    return {"verdict": decide(xr - xn, se, min_gap), "gap": xr - xn, "se": se, "x_reuse": xr, "x_new": xn, **sides}


def load_texts(folder: Path, batches: Mapping[str, Sequence[str]], n_ex: int, n_near: int) -> tuple[dict[str, dict[str, list[str]]], list[str], list[str]]:
    """The maximiser agents' files, one per batch (`folder/<batch>.json`, {"batch", "texts": {id: {"examples",
    "near_misses"}}}): (id -> texts, batches whose file is missing or unreadable, ids dropped). An id is dropped when it is
    not in that batch, or its lists are not n_ex / n_near non-blank strings. A dropped or missing id has no scores, so
    the scorer reads it as unscorable."""
    texts: dict[str, dict[str, list[str]]] = {}
    missing: list[str] = []
    bad: list[str] = []

    def good(xs: Any, n: int) -> bool:
        return isinstance(xs, list) and len(xs) == n and all(isinstance(x, str) and x.strip() for x in xs)

    for b, ids in batches.items():
        try:
            got = json.loads((folder / f"{b}.json").read_text())["texts"]
            assert isinstance(got, dict)
        except (OSError, ValueError, KeyError, TypeError, AssertionError):
            missing.append(b)
            continue
        for i, e in got.items():
            e = e if isinstance(e, dict) else {}
            if i in ids and good(e.get("examples"), n_ex) and good(e.get("near_misses"), n_near):
                texts[i] = {"examples": e["examples"], "near_misses": e["near_misses"]}
            else:
                bad.append(i)
    return texts, missing, bad


def self_test(outcomes: Sequence[str | None], min_win: float) -> dict[str, Any]:
    """C2: the share of 'win' among scorable neurons (outcome not None); ok iff it reaches min_win. No scorable neuron
    gives nan and not ok."""
    scorable = [o for o in outcomes if o is not None]
    share = sum(o == "win" for o in scorable) / len(scorable) if scorable else math.nan
    counts = {k: sum(o == k for o in outcomes) for k in ("win", "tie", "lose")} | {"unscorable": len(outcomes) - len(scorable)}
    return {"win_share": share, "n_scorable": len(scorable), "n": len(outcomes), "counts": counts, "ok": share >= min_win}
