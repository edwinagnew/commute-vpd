"""E45's pure rules (plan reports/plans/2026-10-28_plan_qwen-q2.md): packing, data order, the KL-grid crossings and the
state read at each grid value, the sample filters, anchor 1's prompt rule, the Job 1 scoring rules and Amendment 2's
decisive point. No model, no torch; every function is a deterministic function of its arguments.
Tests: tests/test_qwen_updates.py.

Comparisons round to 9 places first (`r9`); NaN, infinity or None is a miss in every rule.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from typing import Any

Seq = tuple[list[int], int]  # token ids, number of prompt tokens (qwen_sft.Seq)
Row = tuple[list[int], list[bool]]  # token ids, per-token target flag (qwen_sft.Item's packed form)
Records = list[dict[str, Any]]


def r9(x: float) -> float:
    return round(x, 9)


def ok(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def at_least(x: float | None, t: float) -> bool:
    return ok(x) and r9(x) >= t  # type: ignore[arg-type]


def at_most(x: float | None, t: float) -> bool:
    return ok(x) and r9(x) <= t  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- data order and packing


def cycle(n: int, seed: int, total: int) -> list[int]:
    """`total` indices into n items: seeded permutations of range(n), one per pass, joined (a second pass starts
    re-shuffled when the first ends)."""
    assert n > 0, "no items"
    out: list[int] = []
    for p in range(math.ceil(total / n)):
        order = list(range(n))
        random.Random(f"cycle-{seed}-{p}").shuffle(order)
        out += order
    return out[:total]


def pack_whole(seqs: Sequence[Seq], seq_len: int) -> list[Row]:
    """Whole sequences joined, in order, into rows of at most seq_len tokens (a sequence never split); each token
    flagged as a target when it is a response token of its sequence."""
    rows: list[Row] = []
    ids: list[int] = []
    flags: list[bool] = []
    for s, p in seqs:
        assert 0 < p < len(s) <= seq_len, "a sequence needs a prompt, a response and to fit a row"
        if len(ids) + len(s) > seq_len:
            rows.append((ids, flags))
            ids, flags = [], []
        ids, flags = ids + s, flags + [False] * p + [True] * (len(s) - p)
    return rows + [(ids, flags)] if ids else rows


def packed_stream(seqs: Sequence[Seq], seed: int, seq_len: int, n_rows: int) -> list[Row]:
    """n_rows packed rows: each pass shuffles the sequences (seeded by pass) and packs them anew."""
    out: list[Row] = []
    p = 0
    while len(out) < n_rows:
        order = list(range(len(seqs)))
        random.Random(f"pack-{seed}-{p}").shuffle(order)
        out += pack_whole([seqs[i] for i in order], seq_len)
        p += 1
    return out[:n_rows]


# ----------------------------------------------------------------------------- the KL grid


def read_steps(steps: int, dense_until: int, every: int) -> list[int]:
    """Every step to dense_until, then every `every` steps, and always the last."""
    return sorted(set(range(1, min(dense_until, steps) + 1)) | set(range(dense_until + every, steps + 1, every))
                  | {steps})


def crossed(kl: dict[str, float], done: set[tuple[str, float]], grid: Sequence[float]) -> list[tuple[str, float]]:
    """The (axis, grid value) pairs this reading reaches for the first time: KL at or past the value."""
    return [(a, g) for a, x in kl.items() for g in grid if (a, g) not in done and at_least(x, g)]


def distance(kl: float | None, g: float) -> float:
    """|log(KL / g)|; infinite for a missing, non-finite or non-positive KL."""
    return abs(math.log(kl / g)) if ok(kl) and kl > 0 else math.inf  # type: ignore[operator]


def nearest(lo: tuple[int, float] | None, hi: tuple[int, float], g: float) -> tuple[int, float]:
    """The bracketing state (step, KL) nearer the grid value in log KL; a tie, or no lower state, takes the upper."""
    if lo is not None and distance(lo[1], g) < distance(hi[1], g):
        return lo
    return hi


def choose(crossings: list[dict[str, Any]], kl: dict[int, dict[str, float]]) -> list[dict[str, Any]]:
    """Per crossing {axis, grid, lo, hi} (lo None when the first read was already past), the state read at that grid
    value, from `kl` (step -> axis -> KL on all rows of the axis, read from the saved state)."""
    out = []
    for c in crossings:
        a, g = c["axis"], c["grid"]
        lo = None if c["lo"] is None else (c["lo"], kl[c["lo"]][a])
        step, x = nearest(lo, (c["hi"], kl[c["hi"]][a]), g)
        out.append({"axis": a, "grid": g, "step": step, "kl": x})
    return out


# ----------------------------------------------------------------------------- samples


def keep_correct(recs: Records) -> Records:
    """Own samples' filter: strict-correct, not truncated, exact duplicates (same problem, same ids) removed."""
    seen, out = set(), []
    for r in recs:
        key = (r["pid"], tuple(r["ids"]))
        if r["strict"] and not r["truncated"] and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def keep_finished(recs: Records) -> Records:
    """The null's and anchor 1's filter: truncated dropped, nothing else."""
    return [r for r in recs if not r["truncated"]]


# ----------------------------------------------------------------------------- anchor 1's prompt


def prompt_rule(base: list[bool], cand: list[bool], min_gain: float, se_mult: float) -> dict[str, Any]:
    """Paired gain of a prompt over no prompt (strict, whole response, same problems in order) and its standard
    error (sd of the per-problem differences / sqrt n); qualifies if gain >= min_gain and gain >= se_mult x se."""
    assert len(base) == len(cand)
    d = [int(c) - int(b) for b, c in zip(base, cand, strict=True)]
    gain = statistics.mean(d) if d else math.nan
    se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else math.nan
    return {"gain": gain, "se": se, "qualifies": at_least(gain, min_gain) and ok(se) and r9(gain) >= r9(se_mult * se)}


def pick(rules: dict[str, dict[str, Any]]) -> str | None:
    """The qualifying prompt with the largest gain (ties: the first in order); None if none qualifies."""
    best = [n for n, r in rules.items() if r["qualifies"]]
    return max(best, key=lambda n: rules[n]["gain"]) if best else None


# ----------------------------------------------------------------------------- Job 1 scoring rules (plan section 5)


def paired_gain(w0: Records, t: Records) -> dict[str, Any]:
    """Strict accuracy (drop rule) at t minus at W0, on the problems whose answers fit the window at both."""
    a = {r["idx"]: r["strict"] for r in w0 if r["fits"]}
    pairs = [(a[r["idx"]], r["strict"]) for r in t if r["fits"] and r["idx"] in a]
    if not pairs:
        return {"n": 0, "diff": None}
    return {"n": len(pairs), "diff": statistics.mean(int(y) - int(x) for x, y in pairs)}


def length_ratio(w0: Records, t: Records) -> float | None:
    """Mean response length (tokens) at t over the same at W0."""
    if not w0 or not t:
        return None
    m0 = statistics.mean(r["response_tokens"] for r in w0)
    return statistics.mean(r["response_tokens"] for r in t) / m0 if m0 > 0 else None


def learned(arm: str, task: dict[str, float], facts: dict[str, float] | None, known_w0: float | None,
            rule: dict[str, float]) -> bool:
    """Whether a state has learned (plan section 3, predictions.json rules.learned): the held-out loss drop (own
    samples, the null, gold), the gap closed (anchor 1), or the facts (anchor 2: `facts` from its generation)."""
    if arm == "a1":
        return at_least(task.get("gap"), rule["a1_gap"])
    if arm == "a2":
        f = facts or {}
        return anchor2_holds(f.get("invented"), f.get("known"), known_w0,
                             {"em_min": rule["a2_em"], "known_frac": rule["known_frac"]})
    return at_least(task.get("drop"), rule["gold_drop" if arm == "gold" else "own_drop"])


def anchor2_holds(em: float | None, known: float | None, known_w0: float | None, rule: dict[str, float]) -> bool:
    """In-format held-out exact match >= em_min and known-fact accuracy >= known_frac x W0's."""
    return at_least(em, rule["em_min"]) and ok(known_w0) and at_least(known, rule["known_frac"] * known_w0)  # type: ignore[operator]


# ----------------------------------------------------------------------------- Amendment 2's decisive point


def match(pool: dict[int, float], target: float, tol_ratio: float) -> dict[str, Any]:
    """The pool state (step -> KL on the maths rows) nearest `target` in log KL, a tie to the later step; KL that is
    missing, non-finite or not positive is never chosen. Matched if r9(|log(KL / target)|) <= r9(log(tol_ratio))."""
    best = min(((r9(distance(x, target)), -t) for t, x in pool.items() if distance(x, target) < math.inf), default=None)
    if best is None:
        return {"step": None, "kl": None, "log_dist": None, "matched": False}
    d, t = best[0], -best[1]
    return {"step": t, "kl": pool[t], "log_dist": d, "matched": r9(d) <= r9(math.log(tol_ratio))}


def decisive(kl: dict[int, float], is_learned: dict[int, bool], target: float, tol_ratio: float,
             last: int) -> dict[str, Any]:
    """predictions_amend2.json rules.decisive_point: `match` over the learned states up to step `last`, and over the
    unlearned ones up to `last` (reported beside it, never read)."""
    pool = {t: x for t, x in kl.items() if t <= last}
    return {"read": match({t: x for t, x in pool.items() if is_learned.get(t)}, target, tol_ratio),
            "nearest_unlearned": match({t: x for t, x in pool.items() if not is_learned.get(t)}, target, tol_ratio)}
