"""E151's filed rule: every edge case listed in experiments/disagreeing_pairs/results/predictions.json
(edge_cases_checked_before_filing), one test each."""

from __future__ import annotations

import json
import math
from pathlib import Path

from commute.jobs import load_experiment

R = load_experiment("disagreeing_pairs", "rules")
PRED = Path(__file__).resolve().parents[1] / "experiments/disagreeing_pairs/results/predictions.json"
TH = json.loads(PRED.read_text())["thresholds"]
A, B, D = R.COUNTED
NAN = math.nan

# seed triples: a side at x with spread 0.02
def side(x: float) -> list[float]:
    return [x - 0.01, x, x + 0.01]


STEER_UP = (side(0.56), side(0.10))  # steering separates, reuse higher
STEER_DOWN = (side(0.13), side(0.70))  # steering separates, new higher
STEER_FLAT = (side(0.20), side(0.18))  # gap 0.02: does not


def pair(prior_gap: float, steering: tuple[list[float], list[float]], share: tuple[list[float], list[float]]) -> dict:
    return {"prior_gap": prior_gap, "steering": steering, "share": share}


def test_prior_gap_exactly_at_bar_is_a_call():
    assert R.prior_call(0.5, TH["prior_bar"]) == 1
    assert R.prior_call(-0.5, TH["prior_bar"]) == -1
    # at the bar the pair can be opposite, never flat
    assert R.pair_calls(pair(-0.5, STEER_UP, STEER_UP), TH)["condition"] == "opposite"
    assert R.pair_calls(pair(0.5, STEER_FLAT, STEER_UP), TH)["condition"] == "condition not met"


def test_prior_gap_just_under_bar_is_matched():
    assert R.prior_call(0.4999, TH["prior_bar"]) == 0
    assert R.pair_calls(pair(0.4999, STEER_FLAT, STEER_UP), TH)["condition"] == "flat"
    # the positive control's case: prior matched, steering separates
    assert R.pair_calls(pair(0.2, STEER_UP, STEER_UP), TH)["condition"] == "condition not met"


def test_steering_gap_equal_to_range_does_not_separate():
    steer = ([0.30, 0.35, 0.40], [0.20, 0.25, 0.30])  # gap 0.1, ranges 0.1: needs |gap| > range
    assert R.CALL(*steer, TH)["call"] == "does not"
    assert R.pair_calls(pair(0.0, steer, steer), TH)["condition"] == "flat"
    assert R.pair_calls(pair(-1.2, steer, steer), TH)["condition"] == "condition not met"


def test_inert_steering_counts_as_not_separating():
    steer = ([0.01, 0.02, 0.03], [0.0, 0.01, 0.04])
    assert R.CALL(*steer, TH)["call"] == "inert"
    assert R.pair_calls(pair(0.1, steer, steer), TH)["condition"] == "flat"
    assert R.pair_calls(pair(1.0, steer, steer), TH)["condition"] == "condition not met"


def test_prior_and_steering_same_way_not_met():
    assert R.pair_calls(pair(1.0, STEER_UP, STEER_UP), TH)["condition"] == "condition not met"
    assert R.pair_calls(pair(-1.0, STEER_DOWN, STEER_UP), TH)["condition"] == "condition not met"


def test_share_inert_on_opposite_is_neither():
    s = R.score({A: pair(-1.2, STEER_UP, ([0.01, 0.02, 0.0], [0.0, 0.01, 0.02]))}, TH)
    assert s["pairs"][A]["reading"] == "neither"
    assert s["outcome"] == "neither"


def test_share_gain_under_floor_is_unreadable_and_left_out():
    share_nan = ([0.77, NAN, 0.78], [0.5, 0.5, 0.5])  # one seed nan: 2 finite but a nan among them -> unreadable
    share_one = ([0.77, NAN, NAN], [0.5, 0.5, 0.5])
    assert R.CALL(*share_one, TH)["call"] == "unreadable"
    s = R.score({A: pair(-1.2, STEER_UP, share_nan), D: pair(3.5, STEER_DOWN, (side(0.86), side(0.3)))}, TH)
    assert s["pairs"][A]["reading"] == "unreadable"
    assert s["counted_opposite"] == [D]
    assert s["outcome"] == "prior"  # one pair left: its reading
    s = R.score({A: pair(-1.2, STEER_UP, share_one)}, TH)
    assert s["outcome"] == "no call"


def test_one_prior_one_steering_is_split():
    s = R.score({A: pair(-1.2, STEER_UP, (side(0.77), side(0.40))), D: pair(3.5, STEER_DOWN, (side(0.86), side(0.30)))}, TH)
    assert s["pairs"][A]["reading"] == "follows steering"
    assert s["pairs"][D]["reading"] == "follows prior"
    assert s["outcome"] == "split"


def test_one_prior_one_neither_is_split():
    s = R.score({A: pair(-1.2, STEER_UP, (side(0.50), side(0.49))), D: pair(3.5, STEER_DOWN, (side(0.86), side(0.30)))}, TH)
    assert s["pairs"][A]["reading"] == "neither"
    assert s["outcome"] == "split"


def test_both_follow_the_same_reading():
    s = R.score({A: pair(-1.2, STEER_UP, (side(0.40), side(0.77))), D: pair(3.5, STEER_DOWN, (side(0.86), side(0.30)))}, TH)
    assert s["outcome"] == "prior"
    s = R.score({A: pair(-1.2, STEER_UP, (side(0.77), side(0.40))), D: pair(3.5, STEER_DOWN, (side(0.30), side(0.86)))}, TH)
    assert s["outcome"] == "steering"


def test_A_fails_B_and_D_meet_run_continues_outcome_is_D_alone():
    conds = {A: "condition not met", B: "flat", D: "opposite"}
    assert R.n_meeting(conds) == 2 >= TH["min_counted_pairs"]
    assert R.n_meeting({A: "condition not met", B: "condition not met", D: "opposite"}) < TH["min_counted_pairs"]
    s = R.score({A: pair(1.0, STEER_UP, STEER_UP), B: pair(0.1, STEER_FLAT, (side(0.73), side(0.40))),
                 D: pair(3.5, STEER_DOWN, (side(0.86), side(0.30)))}, TH)  # fmt: skip
    assert s["pairs"][A]["reading"] == "not counted"
    assert s["counted_opposite"] == [D] and s["outcome"] == "prior"
    assert s["flat"] == {B: "separates, reuse side higher"}


def test_A_meeting_flat_is_counted_as_flat_outside_outcome():
    s = R.score({A: pair(0.2, STEER_FLAT, (side(0.77), side(0.62)))}, TH)
    assert s["pairs"][A]["condition"] == "flat"
    assert s["flat"] == {A: "separates, reuse side higher"}
    assert s["outcome"] == "no call"


def test_flat_share_readings():
    for share, want in (((side(0.4), side(0.7)), "separates, new side higher"), ((side(0.5), side(0.49)), "does not"),
                        (([0.01, 0.0, 0.02], [0.0, 0.01, 0.0]), "does not")):  # fmt: skip
        assert R.score({B: pair(0.1, STEER_FLAT, share)}, TH)["pairs"][B]["reading"] == want


def test_do_nothing_pair_never_counted():
    zero = ([NAN, NAN, NAN], [NAN, NAN, NAN])  # gain 0 on every state: steering and share nan
    s = R.score({A: pair(-1.2, zero, zero), D: pair(3.5, zero, zero)}, TH)
    assert all(s["pairs"][p]["condition"] == "condition not met" for p in (A, D))
    assert s["outcome"] == "no call"


def test_pair_C_never_counted():
    c = "C_rank1_classshuffled"
    s = R.score({c: pair(4.58, STEER_DOWN, (side(0.86), side(0.28)))}, TH)
    assert s["pairs"][c]["reading"] == "follows prior"
    assert s["counted_opposite"] == [] and s["outcome"] == "no call"


def test_filed_predictions_on_the_leaned_case():
    s = R.score({A: pair(-1.2, STEER_UP, (side(0.77), side(0.62))), B: pair(0.1, STEER_FLAT, (side(0.73), side(0.62))),
                 D: pair(3.5, STEER_DOWN, (side(0.86), side(0.30)))}, TH)  # fmt: skip
    assert R.filed(s) == {"P1": True, "P2": True, "P3": True, "P4": True, "P5": True}
