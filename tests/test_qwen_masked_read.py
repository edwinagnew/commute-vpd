"""E79's filed rules on their edge cases: zero and negative a1 - a0, NaN, ties on a band, no response positions, zero W0 on W."""
from __future__ import annotations

import math

from commute.jobs import load_experiment

rules = load_experiment("qwen_masked_read", "rules")
NAN = math.nan
N = 2048
# Well inside agreement: forward under the floor, reverse under half of W0's, agreement over the midpoint of 0.5 and 0.9.
GOOD = {"fwd": (0.28, 0.29), "rev": (0.4, 0.41), "agree": (0.85, 0.85), "a1": 0.9, "a0": 0.5, "w0_rev": 2.0, "n5": 0.82, "n": N}


def lab(**kw: object) -> str:
    return rules.label(**(GOOD | kw))["label"]


def test_agreement() -> None:
    assert lab() == "agreement"


def test_softness_by_each_clause_and_mixed() -> None:
    assert lab(fwd=(0.9, 0.9)) == "softness"
    assert lab(rev=(1.81, 1.81)) == "softness"
    assert lab(rev=(1.2, 1.2)) == "mixed"  # under 0.9 of W0's, over half
    assert lab(agree=(0.6, 0.6)) == "mixed"  # under the midpoint


def test_a1_at_or_under_a0_is_no_call_but_softness_still_reads() -> None:
    assert lab(a1=0.5) == "no call"
    assert lab(a1=0.4) == "no call"
    assert lab(a1=0.4, fwd=(0.9, 0.9)) == "softness"


def test_nan_anywhere_is_unread() -> None:
    for kw in ({"fwd": (NAN, 0.3)}, {"fwd": (0.3, NAN)}, {"rev": (NAN, 0.4)}, {"agree": (0.8, NAN)}, {"a1": NAN}, {"a0": NAN}, {"w0_rev": NAN}, {"n5": NAN}, {"n": 0}):
        assert lab(**kw) == "unread", kw


def test_ties_on_a_band() -> None:
    assert lab(fwd=(0.8, 0.82), n5=0.82) == "softness"  # n1 + b = n5: not beyond it
    assert lab(fwd=(0.3, 0.3), n5=0.3) == "softness"
    assert lab(rev=(0.9, 1.0)) == "agreement"  # n2 + b = half of W0's, inclusive
    assert lab(rev=(0.9, 1.01)) == "mixed"
    assert lab(rev=(1.8, 1.8)) == "mixed"  # exactly 0.9 of W0's is not softness
    se = rules.binomial_se(0.75, N)
    assert lab(agree=(0.75, 0.75), a1=2 * (0.75 - se) - 0.5) == "agreement"  # n3 - b = the midpoint, b the standard error
    assert lab(agree=(0.71, 0.70), n=10**12) == "agreement"  # n3 - b = the midpoint, b the draws' difference


def test_band_for_agreement_takes_the_standard_error_when_larger() -> None:
    assert math.isclose(rules.band((0.8, 0.8), rules.binomial_se(0.8, 100)), 0.04)
    assert rules.band((0.8, 0.7), rules.binomial_se(0.8, 100)) > 0.09
    assert rules.binomial_se(0.0, N) == 0 and rules.binomial_se(1.0, N) == 0
    assert math.isnan(rules.binomial_se(0.5, 0))


def test_unseen() -> None:
    assert rules.unseen((0.5, 0.5), 1.0) == "survives"  # exactly half, inclusive
    assert rules.unseen((0.51, 0.51), 1.0) == "does not survive"
    assert rules.unseen((0.4, 0.5), 1.0) == "survives"
    assert rules.unseen((0.45, 0.5), 1.0) == "survives"  # fall - b is exactly half
    assert rules.unseen((0.48, 0.52), 1.0) == "does not survive"  # half only without the band
    assert rules.unseen((0.0, 0.0), 0.0) == "no room on W"  # zero W0
    assert rules.unseen((0.1, 0.4), 0.3) == "no room on W"  # W0 at the band
    assert rules.unseen((NAN, 0.4), 1.0) == "unread" and rules.unseen((0.4, 0.4), NAN) == "unread"


def test_own_masks() -> None:
    assert rules.own_masks(0.42, 0.28) == "survives"  # 1.5 x, inclusive
    assert rules.own_masks(0.43, 0.28) == "does not survive Γ"
    assert rules.own_masks(NAN, 0.28) == "unread" and rules.own_masks(0.3, NAN) == "unread"


def test_empty_response_set_and_floor() -> None:
    assert math.isnan(rules.over(0.0, 0)) and rules.over(3.0, 2) == 1.5
    assert rules.floor({"fwd": 9.0, "fwd_T1.0": 1.3, "fwd_T2.0": 0.8}) == 0.8
    assert math.isnan(rules.floor({"fwd": 9.0}))
    assert math.isnan(rules.floor({"fwd_T1.0": NAN, "fwd_T2.0": 0.8}))


def test_outcome() -> None:
    assert rules.outcome("agreement", "survives", "survives") == "agreement and survives"
    assert rules.outcome("softness", "unread", "unread") == "softness"
    assert rules.outcome("agreement", "no room on W", "survives") == "reported as read"
    assert rules.outcome("agreement", "survives", "does not survive Γ") == "reported as read"
    assert rules.outcome("mixed", "survives", "survives") == "reported as read"
    assert rules.outcome("no call", "survives", "survives") == "reported as read"
    assert rules.outcome("agreement", "unread", "survives") == "unread"


def test_score_on_nothing_read_and_partly_read() -> None:
    empty = rules.score({})
    assert empty["outcome"] == "unread" and empty["B"]["label"] == "unread" and empty["leans"]["B"] is None

    def cell(fwd: float, rev: float, agree: float, **temps: float) -> dict[str, dict[str, float]]:
        return {"all": {"fwd": fwd, "rev": rev, "agree": agree, "n": N, **temps}}

    w0 = {"M": {"d1": cell(1.33, 2.0, 0.5, fwd_T2=0.82)}, "W": {"d1": cell(1.0, 2.0, 0.5)}}
    b = {"M": {"clean": cell(0.02, 0.02, 0.9), "d1": cell(0.28, 0.4, 0.85), "d2": cell(0.29, 0.41, 0.85), "own": cell(0.3, 0.4, 0.8)},
         "W": {"d1": cell(0.3, 0.4, 0.8), "d2": cell(0.31, 0.4, 0.8)}}
    got = rules.score({"W0": w0, "B": b})
    assert got["outcome"] == "agreement and survives"
    assert got["leans"] == {"B": True, "L": None, "W_B": True, "own_B": True}
    assert got["L"]["label"] == "unread"
