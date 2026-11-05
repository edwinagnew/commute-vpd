"""E95's filed rules (results/predictions.json; plan reports/plans/2026-11-05_plan_e95-token-identity.md). Pure: no model is read, and
tests/test_token_identity.py runs each rule through its edge cases and through E90 Amendment 2's saved numbers. Read 2's band and
statuses are `rules_amend2.read_block`, unchanged; only what sits beside them is here. Comparisons are strict; a nan is never above."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from commute import removal as RM
from commute.jobs import load_experiment

RA = load_experiment("new_vs_reuse", "rules_amend2")
ABOVE, WITHIN, MIXED, UNREADABLE = RA.ABOVE, RA.WITHIN, RA.MIXED, RA.UNREADABLE
EQUAL, BELOW, INSIDE, AT_MOST = "about equal", "below", "inside the draws", "at most equal"

TOKEN_ONLY = "token needed, no more"
COPY_MORE = "copy depends on more than getting the key through"
BOTH_MORE = "both states depend on more than getting the key through; not specific to copy"
RANDOM_MORE = "random map depends on more, copy does not: against the copy reading"
NO_CALL_WRITE = "no call: a write did not learn its table, or no teeth, or no share"
NO_CALL_SEEDS = "no call: seeds disagree"
NO_CALL_SPARED = "no call: the write is spared and the state is not above it"
NO_CALL_SIZE = "no call: a state loses more than the write, by about what matched directions give"
NO_CALL_DIFF = "no call: copy above and random not, but the difference between them is not beyond the draws"

CONFIRMED = "replicate repeats the separation: filed rule and paired test"
FILED_ONLY = "filed rule repeats, paired test does not: not confirmed"
NOT_REPEATED = "not repeated: copy within. Not shown specific, never shown not specific"
RANDOM_ABOVE = "replicate's negative control fails: random map above"
SEEDS_DIFFER = "no reading: seeds disagree on the replicate"
NO_CALL_REPLICATE = "no call: no teeth, no matched band or no share on the replicate"

ALL_PAIRS, SOME_PAIRS, NO_PAIR = "every pair", "some pairs", "no pair"


def percentile(values: Sequence[float], q: float) -> float:
    """Numpy's linear rule, as `removal.band_of`, but signed: on finite values only; nan if there are none."""
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else math.nan


def over_seeds(statuses: Sequence[str]) -> str:
    """The status every seed gives. Any unreadable seed, or no seed: unreadable. Else "mixed"."""
    got = set(statuses)
    if not got or UNREADABLE in got:
        return UNREADABLE
    return got.pop() if len(got) == 1 else MIXED


# ---- read 1: the write downstream of the removal ----


def write_works(status: str, gain: float, rotated: Sequence[float]) -> dict[str, Any]:
    """The write learned its table: its alpha match succeeded and its gain is strictly over the floor, the 95th percentile of the same
    write's gain with its answers rotated to other keys (same alpha, same windows, log-odds). No rotated gain: it does not count."""
    floor = percentile(rotated, 95)
    return {"floor": floor, "gain": gain, "status": status, "works": status == "ok" and bool(gain > floor)}


def gaps_under_draws(state: Sequence[float], write: Sequence[float]) -> dict[str, float | list[float]]:
    """The threshold, set from the draws alone before the basis is read: state's share minus the write's under each draw, 5th and 95th."""
    g = [s - w for s, w in zip(state, write, strict=True)]
    return {"draw_gaps": g, "lo": percentile(g, 5), "hi": percentile(g, 95)}


def seed_row(state: float, write: float, write_draws: Sequence[float], null: Mapping[str, Any], works: bool) -> dict[str, Any]:
    """One seed: state minus write under the basis, first match wins. "About equal": no larger than half the draws' 5-95 range, the size
    the same gap moves by between matched directions. "Above" or "below": outside that range. "Inside the draws": larger than half the
    range, and no larger than matched directions give. Also whether the write itself loses more than its band, `removal.band_of` as every
    E90 band: absolute, so a spared write with draws all negative does not count as hit (signed, block 1's write would). Ties do not cross."""
    gap, hi, lo = state - write, null["hi"], null["lo"]
    if not works or any(math.isnan(x) for x in (gap, hi, lo)):
        status = UNREADABLE
    else:
        status = EQUAL if abs(gap) <= (hi - lo) / 2 else ABOVE if gap > hi else BELOW if gap < lo else INSIDE
    bar = RM.band_of(write_draws)
    return {"state": state, "write": write, "gap": gap, "lo": lo, "hi": hi, "status": status, "write_bar": bar, "write_bites": bool(write > bar)}


def table_status(rows: Sequence[Mapping[str, Any]]) -> str:
    """A table's status over seeds, where "about equal" and "below" both say the write loses at least as much as the state."""
    return over_seeds([AT_MOST if x["status"] in (EQUAL, BELOW) else x["status"] for x in rows])


def read1(teeth: bool, copy: Sequence[Mapping[str, Any]], rand: Sequence[Mapping[str, Any]], difference: str) -> dict[str, Any]:
    """The outcome, first match wins; `difference` is `paired`'s call on state-minus-write, copy against random. Every combination of
    the two tables' statuses lands in exactly one line."""
    c, r = table_status(copy), table_status(rand)
    bites = bool(copy and rand) and all(x["write_bites"] for x in (*copy, *rand))
    if not teeth or UNREADABLE in (c, r):
        outcome = NO_CALL_WRITE
    elif MIXED in (c, r):
        outcome = NO_CALL_SEEDS
    elif c == ABOVE:
        outcome = BOTH_MORE if r == ABOVE else COPY_MORE if difference == ALL_PAIRS else NO_CALL_DIFF
    elif r == ABOVE:
        outcome = RANDOM_MORE
    elif INSIDE in (c, r):
        outcome = NO_CALL_SIZE
    else:
        outcome = TOKEN_ONLY if bites else NO_CALL_SPARED
    return {"outcome": outcome, "copy": c, "random": r, "difference": difference, "write_bites": bites}


# ---- read 2: the replicate tables ----


def own_bands(shares: Sequence[float], bands: Sequence[float], readable: bool) -> str:
    """Each state against its own band, `rules_amend2.status` seed by seed; the seeds' common status, else "mixed"."""
    return over_seeds([RA.status([s], b, readable) for s, b in zip(shares, bands, strict=True)]) if shares else UNREADABLE


def paired(copy: Mapping[str, Mapping[str, Any]], rand: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The check's test, now filed: per pair of a copy seed and a random seed, the copy-minus-random gap under the basis against the same
    gap under each draw; a pair is "above" when it strictly exceeds the largest draw's. Rows hold A_lo and `random`. A pair with a nan
    share or no finite draw is unreadable, and so is the whole call."""
    pairs = {}
    for a, x in copy.items():
        for b, y in rand.items():
            draws = np.asarray(x["random"], float) - np.asarray(y["random"], float)
            draws = draws[np.isfinite(draws)]
            gap = x["A_lo"] - y["A_lo"]
            read = draws.size > 0 and not math.isnan(gap)
            pairs[f"{a} - {b}"] = {"gap": gap, "draws_mean": float(draws.mean()) if read else math.nan, "draws_max": float(draws.max()) if read else math.nan, "above": bool(read and gap > draws.max()), "read": read}
    if not pairs or not all(p["read"] for p in pairs.values()):
        return {"call": UNREADABLE, "pairs": pairs}
    n = sum(p["above"] for p in pairs.values())
    return {"call": ALL_PAIRS if n == len(pairs) else SOME_PAIRS if n else NO_PAIR, "pairs": pairs}


def read2(copy: str, rand: str, pair_call: str) -> str:
    """The replicate's outcome from its filed statuses (over the largest band) and the paired call, first match wins. Own bands are
    printed beside and decide nothing."""
    if UNREADABLE in (copy, rand, pair_call):
        return NO_CALL_REPLICATE
    if rand == ABOVE:
        return RANDOM_ABOVE
    if MIXED in (copy, rand):
        return SEEDS_DIFFER
    if copy == ABOVE:
        return CONFIRMED if pair_call == ALL_PAIRS else FILED_ONLY
    return NOT_REPEATED
