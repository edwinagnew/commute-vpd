"""E79's filed rules on their edge cases: zero and negative a1 - a0, NaN, ties on a band, no response positions, zero W0 on W."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from commute.jobs import load_experiment

rules = load_experiment("qwen_masked_read", "rules")
RESULTS = Path(__file__).resolve().parent.parent / "experiments" / "qwen_masked_read" / "results"
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


# Amendment 1's rules: P's clean under A's read, a zero Σg denominator, NaN, ties on the bands.
a1 = load_experiment("qwen_masked_read", "rules_amend1")


def test_a1_filed_values_agree_with_the_code_and_job_51117() -> None:
    filed = json.loads((RESULTS / "predictions_amend1.json").read_text())
    read = json.loads((RESULTS / "read.json").read_text())["reads"]
    assert filed["P"]["seed"] == a1.P_SEED
    assert filed["control"]["W0"] == a1.W0_OWN == read["W0"]["M"]["own"]["all"]["fwd"]
    assert filed["control"]["A"] == a1.A_OWN == read["A"]["M"]["own"]["all"]["fwd"]


def test_a1_scale_to() -> None:
    def size(s: float, a: float, b: float, c: float) -> float:
        return s * s * a + 2 * s**3 * b + s**4 * c

    for a, b, c, want in ((1.0, 0.0, 0.0, 4.0), (1.0, 0.0, 0.0, 0.25), (2.0, -0.1, 0.5, 2.0), (0.3, 0.01, 0.01, 7.0), (0.0, 0.0, 1.0, 16.0)):
        assert math.isclose(size(a1.scale_to(a, b, c, want), a, b, c), want, rel_tol=1e-9)
    assert a1.scale_to(1.0, 0.0, 0.0, 1.0) == 1.0  # the permutation already matches
    assert a1.scale_to(0.0, 0.0, 0.0, 0.0) == 1.0 and a1.scale_to(1.0, 0.0, 0.0, 0.0) == 1.0  # a site with no change to match
    assert a1.scale_to(0.0, 0.0, 0.0, 1.0) == 1.0  # nothing to scale: the caller's size check refuses it
    assert math.isnan(a1.scale_to(NAN, 0.0, 0.0, 1.0)) and math.isnan(a1.scale_to(1.0, 0.0, 0.0, NAN))


def test_a1_ratio_and_opens() -> None:
    assert a1.ratio(3.0, 2.0) == 1.5
    assert math.isnan(a1.ratio(3.0, 0.0)) and math.isnan(a1.ratio(NAN, 2.0)) and math.isnan(a1.ratio(3.0, NAN))  # a zero Σg denominator reads nothing
    assert a1.opens(1.5, 1.5) == "does not open"  # exactly 1.5 is not above
    assert a1.opens(1.51, 1.0) == "opens" and a1.opens(1.0, 1.51) == "opens"
    assert a1.opens(NAN, 2.0) == "opens"  # either is enough
    assert a1.opens(NAN, 1.0) == "unread" and a1.opens(NAN, NAN) == "unread"


def test_a1_teeth() -> None:
    assert a1.teeth(1.0, 1.0, 0.078, 0.004) == "teeth"
    assert a1.teeth(0.2, 0.1, 0.078, 0.004) == "no teeth"  # under 3 x A's
    assert a1.teeth(1.0, 0.1, 0.078, 0.004) == "over 3 x its clean"
    assert a1.teeth(0.5, 2.0, 0.078, 0.004) == "teeth"  # own below clean: the clause bounds it from above only
    assert a1.teeth(1.0, 0.07, 0.078, 0.004) == "not wrong enough"  # P's clean under A's read
    assert a1.teeth(0.234, 0.078, 0.078, 0.0) == "teeth"  # clean = A's read is not under it, and only own = 3 x both passes
    for kw in ((NAN, 1.0, 0.078, 0.004), (1.0, NAN, 0.078, 0.004), (1.0, 1.0, NAN, 0.004), (1.0, 1.0, 0.078, NAN)):
        assert a1.teeth(*kw) == "unread", kw


def test_a1_teeth_ties_on_the_band() -> None:
    assert a1.teeth(0.3, 1.0, 0.1, 0.0) == "teeth"  # exactly 3 x, closed
    assert a1.teeth(0.31, 1.0, 0.1, 0.01) == "teeth"  # 3 x plus the band, closed
    assert a1.teeth(0.30, 1.0, 0.1, 0.01) == "no teeth"  # 3 x but not beyond the band
    assert a1.teeth(2.99, 1.0, 0.1, 0.01) == "teeth"  # own + band = 3 x clean, closed
    assert a1.teeth(3.0, 1.0, 0.1, 0.01) == "over 3 x its clean"
    assert a1.teeth(0.0, 0.0, 0.0, 0.0) == "teeth"  # all zero passes the arithmetic; A's read of 0.078 is the control's, so this cannot occur in a run


def test_a1_faithful() -> None:
    assert a1.faithful(0.08, 0.043, 0.001) == "faithful"
    assert a1.faithful(0.085, 0.043, 0.001) == "faithful"  # a + b = 2 x, closed
    assert a1.faithful(0.0851, 0.043, 0.001) == "not faithful"
    assert a1.faithful(0.0, 0.0, 0.0) == "unread"  # W0's at zero
    assert a1.faithful(NAN, 0.04, 0.0) == "unread" and a1.faithful(0.04, NAN, 0.0) == "unread" and a1.faithful(0.04, 0.04, NAN) == "unread"


def test_a1_outcome() -> None:
    both = {"P": "teeth", "C": "teeth"}
    good = {"M": "faithful", "W": "faithful"}
    assert a1.outcome("does not open", both, good) == "passes all three"
    assert a1.outcome("opens", {"P": "unread", "C": "unread"}, {"M": "unread", "W": "unread"}) == "opens"
    assert a1.outcome("does not open", {"P": "teeth", "C": "no teeth"}, good) == "no teeth"
    assert a1.outcome("does not open", both, {"M": "faithful", "W": "not faithful"}) == "reported as read"
    assert a1.outcome("does not open", {"P": "not wrong enough", "C": "teeth"}, good) == "reported as read"
    assert a1.outcome("does not open", {"P": "over 3 x its clean", "C": "teeth"}, good) == "reported as read"
    assert a1.outcome("unread", both, good) == "unread"
    assert a1.outcome("does not open", {"P": "unread", "C": "teeth"}, good) == "unread"


def test_a1_score_on_nothing_read_and_all_read() -> None:
    empty = a1.score({})
    assert empty["outcome"] == "unread" and empty["leans"]["opens"] is None and empty["teeth"]["P"] == "unread"

    def cell(**kw: float) -> dict[str, dict[str, float]]:
        return {"all": kw}

    def state(fwd: float, clean: float, self_fwd: float, g: float = 100.0, d2: bool = False) -> dict[str, Any]:
        out = {"clean": cell(fwd=clean), "g": cell(sum_g=g, count=g / 10), "d1": cell(fwd=fwd, self_fwd=self_fwd)}
        return out | ({"d2": cell(fwd=fwd + 0.004, self_fwd=self_fwd + 0.001)} if d2 else {})

    reads = {"W0": {rs: state(1.33, 1.84, 0.043, d2=True) for rs in ("M", "W")}, "A": {rs: state(0.078, 0.012, 0.05, g=120.0) for rs in ("M", "W")},
             "P": {"M": state(1.0, 1.0, 0.1)}, "C": {"M": state(1.3, 1.8, 0.1)}}
    got = a1.score(reads)
    assert got["outcome"] == "passes all three" and got["ratios"]["A"]["M"]["sum_g"] == 1.2 and got["ratios"]["W0"]["W"]["count"] == 1.0
    assert got["leans"] == {"opens": False, "teeth_P": True, "teeth_C": True, "faithful_M": True, "faithful_W": True}
    reads["A"]["M"]["g"] = cell(sum_g=151.0, count=10.0)
    assert a1.score(reads)["outcome"] == "opens"


# Amendment 2's rules: the zero or negative denominator, NaN, ties on the 3b band and on s = 0.5 and 0.9.
a2 = load_experiment("qwen_masked_read", "rules_amend2")


def test_a2_filed_values_agree_with_the_code_and_job_51161() -> None:
    filed = json.loads((RESULTS / "predictions_amend2.json").read_text())
    read = json.loads((RESULTS / "amend1.json").read_text())["reads"]
    assert filed["control"]["W0"] == a2.W0_OWN == read["W0"]["M"]["d1"]["all"]["fwd"]
    assert filed["control"]["A"] == a2.A_OWN == read["A"]["M"]["d1"]["all"]["fwd"]


def test_a2_do_nothing() -> None:
    assert a2.do_nothing(0.2, 0.078, 0.004) == "A beats do-nothing"
    assert a2.do_nothing(0.05, 0.078, 0.004) == "do-nothing beats A"
    assert a2.do_nothing(0.08, 0.078, 0.004) == "same"
    assert a2.do_nothing(0.078, 0.078, 0.0) == "same"  # no band, no gap
    for kw in ((NAN, 0.078, 0.004), (0.2, NAN, 0.004), (0.2, 0.078, NAN)):
        assert a2.do_nothing(*kw) == "unread", kw


def test_a2_do_nothing_ties_on_the_band() -> None:
    assert a2.do_nothing(0.1 + 3 * 0.004, 0.1, 0.004) == "same"  # exactly 3 b, both ways: the rule's inequalities are strict
    assert a2.do_nothing(0.1 - 3 * 0.004, 0.1, 0.004) == "same"
    assert a2.do_nothing(0.1 + 3 * 0.004 + 1e-6, 0.1, 0.004) == "A beats do-nothing"
    assert a2.do_nothing(0.1 - 3 * 0.004 - 1e-6, 0.1, 0.004) == "do-nothing beats A"


def test_a2_masks_alone() -> None:
    assert a2.masks_alone(1.0, 0.1, 0.0) == "masks alone"
    assert a2.masks_alone(1.0, 0.7, 0.0) == "needs the offsets"
    assert a2.masks_alone(1.0, 0.3, 0.0) == "mixed"
    assert a2.masks_alone(1.0, -0.1, 0.0) == "masks alone" and a2.share(1.0, -0.1, 0.0) > 1  # Z under A's own: the near label, s as read
    assert a2.masks_alone(1.0, 1.2, 0.0) == "needs the offsets" and a2.share(1.0, 1.2, 0.0) < 0  # Z over W0's own


def test_a2_masks_alone_ties_on_the_share() -> None:
    assert a2.masks_alone(1.0, 0.55, 0.5) == "masks alone"  # s = 0.9, closed, through the rounding of (1 - 0.55) / 0.5
    assert a2.masks_alone(1.0, 0.75, 0.5) == "needs the offsets"  # s = 0.5, closed
    assert a2.masks_alone(1.0, 0.56, 0.5) == "mixed" and a2.masks_alone(1.0, 0.74, 0.5) == "mixed"


def test_a2_masks_alone_denominator_and_nan() -> None:
    assert a2.masks_alone(1.0, 0.5, 1.0) == "no call"  # zero
    assert a2.masks_alone(0.5, 0.5, 1.0) == "no call"  # negative
    assert a2.masks_alone(1.0, NAN, 1.0) == "no call"  # the denominator reads without Z
    assert math.isnan(a2.share(1.0, 0.5, 1.0)) and math.isnan(a2.share(0.5, 0.5, 1.0))
    assert a2.masks_alone(1.0, NAN, 0.5) == "unread" and math.isnan(a2.share(1.0, NAN, 0.5))
    assert a2.masks_alone(NAN, 0.5, 0.5) == "unread" and a2.masks_alone(1.0, 0.5, NAN) == "unread"
    assert a2.masks_alone(math.inf, 0.5, 0.5) == "unread"  # inf / inf, not "mixed"


def test_a2_outcome() -> None:
    assert a2.outcome("A beats do-nothing", "needs the offsets") == "can see a placement"
    assert a2.outcome("A beats do-nothing", "mixed") == "can see a placement"
    assert a2.outcome("A beats do-nothing", "masks alone") == "reported as read"
    assert a2.outcome("A beats do-nothing", "no call") == "reported as read"
    assert a2.outcome("A beats do-nothing", "unread") == "unread"
    assert a2.outcome("unread", "masks alone") == "unread"
    for alone in ("masks alone", "needs the offsets", "mixed", "no call", "unread"):  # either of these decides whatever the share
        assert a2.outcome("same", alone) == "credits any forward near FT"
        assert a2.outcome("do-nothing beats A", alone) == "credits any forward near FT"


def test_a2_score_on_nothing_read_and_all_read() -> None:
    empty = a2.score({})
    assert empty["outcome"] == "unread" and empty["leans"] == {"do_nothing": None, "masks_alone": None}
    assert math.isnan(empty["b"]) and math.isnan(empty["s"]) and math.isnan(empty["as_read"]["N"]["W"]["own"])

    def cell(fwd: float, self_fwd: float = 0.0) -> dict[str, dict[str, float]]:
        return {"all": {"fwd": fwd, "self_fwd": self_fwd}}

    reads = {"W0": {"M": {"d1": cell(1.33), "d2": cell(1.326)}}, "A": {"M": {"d1": cell(0.078)}},
             "N": {"M": {"clean": cell(0.0), "d1": cell(0.08, 0.08)}, "W": {"d1": cell(0.09, 0.089)}}, "Z": {"M": {"clean": cell(1.84), "d1": cell(0.5)}}}
    got = a2.score(reads)
    assert math.isclose(got["b"], 0.004) and math.isclose(got["N_own_minus_A_own"], 0.002)
    assert got["do_nothing"] == "same" and got["masks_alone"] == "mixed" and got["outcome"] == "credits any forward near FT"
    assert got["leans"] == {"do_nothing": True, "masks_alone": False}
    assert got["as_read"]["N"]["W"]["self"] == 0.089 and got["as_read"]["N"]["M"]["clean"] == 0.0
    reads["N"]["M"]["d1"] = cell(0.2)
    got = a2.score(reads)
    assert got["do_nothing"] == "A beats do-nothing" and got["outcome"] == "can see a placement" and got["leans"]["do_nothing"] is False
    del reads["Z"]["M"]["d1"]  # stopped before Z: the first rule stands, the second is unread
    got = a2.score(reads)
    assert got["do_nothing"] == "A beats do-nothing" and got["masks_alone"] == "unread" and got["outcome"] == "unread" and got["leans"]["masks_alone"] is None
