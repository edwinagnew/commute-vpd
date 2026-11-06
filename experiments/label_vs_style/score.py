"""E107 (label_vs_style): the topic split of TOFU's 217 real rows and the filed rules, with no model, so that the CPU dry
run, the tests and run.py's score stage read the same code. Plan: reports/plans/2026-11-06_plan_e107-label-vs-style.md;
rules results/predictions.json `label_check`, `wrong_rule`, `topic_rule`.

Overlaps are `qwen_neurons.overlap`: the share of the first top-1024 carrying set inside the second, NaN if either is
empty (the do-nothing state, whose scores are never above 0, has an empty set)."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from commute import qwen_neurons as QN

SEEDS = (42, 142)  # the stream seeds of E103 Amendments 1 and 2 and E105
TOPICS = ("AUTH", "WORLD")  # TOFU's real_authors rows, then its world_facts rows
K = 1024  # E102's carrying-set size; a rule reads only sets that hold all of it
WRONG = tuple(f"WRONG{s}" for s in SEEDS)
TOPIC_ARMS = tuple(f"{t}{s}" for t in TOPICS for s in SEEDS)
ARMS = WRONG + TOPIC_ARMS  # the run order


def topic_split(n_authors: int, n_rows: int) -> dict[str, list[int]]:
    """Real rows in `qwen_tofu.real_rows` order: the first n_authors are real_authors_perturbed's (100), the rest
    world_facts_perturbed's (117). run.py passes both counts from the files."""
    return {"AUTH": list(range(n_authors)), "WORLD": list(range(n_authors, n_rows))}


def label_check(true: Sequence[float], wrong: Sequence[float], min_share: float) -> dict[str, Any]:
    """W0's per-row content log P (nats a token) on the true and the wrong sentence of each row. holds iff the wrong
    sentence is strictly below the true one in at least min_share of rows; a tie counts as not below. No rows, unequal
    lengths or a value not finite: unread, which does not hold."""
    if not true or len(true) != len(wrong) or not all(QN.finite(v) for v in (*true, *wrong)):
        return {"verdict": "unread", "holds": False, "rows": len(true)}
    below = sum(w < t for t, w in zip(true, wrong, strict=True))
    share = below / len(true)
    return {"verdict": "holds" if share >= min_share else "missed", "holds": share >= min_share, "rows": len(true),
            "below": below, "share": share, "min_share": min_share, "mean_gap": statistics.mean(t - w for t, w in zip(true, wrong, strict=True))}


def prefers_wrong(true: Sequence[float], wrong: Sequence[float]) -> float:
    """Reported: the share of rows whose wrong sentence is strictly above the true one (W0 about 0.1)."""
    return sum(w > t for t, w in zip(true, wrong, strict=True)) / len(true)


def _unread(cross: Sequence[float], vals: Sequence[Any], sizes: Mapping[str, int]) -> bool:
    return len(cross) != 4 or not all(QN.finite(v) for v in vals) or any(n != K for n in sizes.values())


def wrong_verdict(w_wrong: float, w_known: float, cross: Sequence[float], x: float, L: float,
                  sizes: Mapping[str, int]) -> dict[str, Any]:
    """The filed rule on one probe. w_wrong: WRONG142's set inside WRONG42's; w_known: the known arm's seeds (stored);
    cross: the four wrong-against-known overlaps (WRONG42-KNOWN42, WRONG42-KNOWN142, WRONG142-KNOWN42, WRONG142-KNOWN142);
    x: E103 Amendment 2's highest known-against-fictional overlap; L: E105's lowest cross-half overlap (stored).
    w = min(w_wrong, w_known), band = 1 - w: the share of a set a change of data order alone replaced, in the less stable arm.
    "familiarity" iff max(cross) <= x + band and max(cross) <= L - band: every pair sits within seed noise of the
        cross-arm level and at least a band under the same-label, different-rows level.
    "topic or style" iff min(cross) >= L - band and min(cross) - x > band: every pair sits within seed noise of the
        rows null and clears the cross-arm level by more than the band.
    "no call" otherwise; "unread" with a value missing or not finite, not four cross values, or a set not of K neurons.
    Both cannot hold: familiarity needs max(cross) <= L - band and <= x + band, topic or style min(cross) >= L - band > x + band."""
    if _unread(cross, [w_wrong, w_known, *cross, x, L], sizes):
        return {"verdict": "unread", "w_wrong": w_wrong, "w_known": w_known, "cross": list(cross), "x": x, "L": L, "sizes": dict(sizes)}
    w, lo, hi = min(w_wrong, w_known), min(cross), max(cross)
    band = 1 - w
    fam_line, style_line = min(x + band, L - band), L - band
    v = ("familiarity" if hi <= fam_line else "topic or style" if lo >= style_line and lo - x > band else "no call")
    return {"verdict": v, "w_wrong": w_wrong, "w_known": w_known, "cross": list(cross), "x": x, "L": L, "w": w, "band": band,
            "cross_min": lo, "cross_max": hi, "familiarity_needs_max": fam_line, "style_needs_min": style_line,
            "style_needs_over_x": x + band, "sizes": dict(sizes)}


def topic_verdict(w_auth: float, w_world: float, cross: Sequence[float], L: float, sizes: Mapping[str, int]) -> dict[str, Any]:
    """The filed rule on one probe. w_auth, w_world: each topic's seed-142 set inside its seed-42 set; cross: the four
    AUTH-against-WORLD overlaps (AUTH42-WORLD42, AUTH42-WORLD142, AUTH142-WORLD42, AUTH142-WORLD142); L: E105's lowest
    cross-half overlap. w = min(w_auth, w_world), band = 1 - w.
    "topic moves the set" iff L - max(cross) > band; "topic does not" iff L - min(cross) <= band; "no call" otherwise
    (the four pairs straddle L - band); "unread" as `wrong_verdict`. Both cannot hold, as min(cross) <= max(cross)."""
    if _unread(cross, [w_auth, w_world, *cross, L], sizes):
        return {"verdict": "unread", "w_auth": w_auth, "w_world": w_world, "cross": list(cross), "L": L, "sizes": dict(sizes)}
    w, lo, hi = min(w_auth, w_world), min(cross), max(cross)
    band = 1 - w
    v = "topic moves the set" if L - hi > band else "topic does not" if L - lo <= band else "no call"
    return {"verdict": v, "w_auth": w_auth, "w_world": w_world, "cross": list(cross), "L": L, "w": w, "band": band,
            "cross_min": lo, "cross_max": hi, "line": L - band, "sizes": dict(sizes)}


def pairs(sets: Mapping[str, Sequence[int]], a: str, b: str) -> list[float]:
    """The four overlaps of a's seeds inside b's (a42-b42, a42-b142, a142-b42, a142-b142)."""
    return [QN.overlap(sets[f"{a}{s}"], sets[f"{b}{t}"]) for s in SEEDS for t in SEEDS]


def wrong_rule(sets: Mapping[str, Sequence[int]], x: float, L: float, w_known: float) -> dict[str, Any]:
    """`wrong_verdict` on one probe from the sets keyed WRONG42, WRONG142, KNOWN42, KNOWN142, FICT42, FICT142; the
    wrong-against-fictional overlaps reported beside."""
    names = ("WRONG42", "WRONG142", "KNOWN42", "KNOWN142")
    out = wrong_verdict(QN.overlap(sets["WRONG142"], sets["WRONG42"]), w_known, pairs(sets, "WRONG", "KNOWN"), x, L,
                        {n: len(sets[n]) for n in names})
    return out | {"wrong_vs_fict": pairs(sets, "WRONG", "FICT"), "known_seeds_here": QN.overlap(sets["KNOWN142"], sets["KNOWN42"])}


def topic_rule(sets: Mapping[str, Sequence[int]], L: float) -> dict[str, Any]:
    """`topic_verdict` on one probe from the sets keyed AUTH42 ... WORLD142 and FICT42, FICT142; reported beside: the mean
    of each topic's four overlaps with the fictional sets and their difference (authors minus world)."""
    out = topic_verdict(QN.overlap(sets["AUTH142"], sets["AUTH42"]), QN.overlap(sets["WORLD142"], sets["WORLD42"]),
                        pairs(sets, "AUTH", "WORLD"), L, {n: len(sets[n]) for n in TOPIC_ARMS})
    fict = {t: statistics.mean(pairs(sets, t, "FICT")) for t in TOPICS}
    return out | {"vs_fict": fict, "auth_minus_world_vs_fict": fict["AUTH"] - fict["WORLD"]}
