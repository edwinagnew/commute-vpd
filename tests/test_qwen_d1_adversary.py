"""E80's filed rules on their edge cases: r exactly on each bar, NaN, a zero W0, an unread state, W0's two starts equal; and the row set."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from commute.jobs import load_experiment

rules = load_experiment("qwen_d1_adversary", "rules")
EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
NAN = math.nan


def starts(**kl: float) -> dict[str, dict[str, dict[str, float]]]:
    return {"starts": {k: {"kl": v} for k, v in kl.items()}}


def test_band_floor_and_equal_starts() -> None:
    assert rules.band(1.0, 1.0) == 0.19
    assert rules.band(1.0, 1.1) == 0.19
    assert rules.band(1.0, 1.5) == pytest.approx(0.5)
    assert rules.band(1.5, 1.0) == pytest.approx(0.5)


def test_band_unread_on_nan_or_zero() -> None:
    for a, b in ((NAN, 1.0), (1.0, NAN), (0.0, 1.0), (0.0, 0.0)):
        assert math.isnan(rules.band(a, b)), (a, b)


def test_ratio_zero_w0_and_nan() -> None:
    assert math.isnan(rules.ratio(1.0, 0.0))
    assert math.isnan(rules.ratio(1.0, NAN))
    assert math.isnan(rules.ratio(NAN, 1.0))


def test_do_nothing_exactly_on_the_bar_is_within() -> None:
    assert rules.do_nothing(1.19, 0.19) == "within the band"
    assert rules.do_nothing(1.1900001, 0.19) == "fails do-nothing"
    assert rules.do_nothing(NAN, 0.19) == "unread"
    assert rules.do_nothing(2.0, NAN) == "unread"


def test_bars_closed_on_both_sides() -> None:
    assert rules.bars(1.40) == "passes"
    assert rules.bars(1.4000001) == "no call"
    assert rules.bars(1.6099999) == "no call"
    assert rules.bars(1.61) == "fails"
    assert rules.bars(NAN) == "unread"


def test_outcome_every_branch() -> None:
    assert rules.outcome("fails do-nothing", "passes") == "D1 has a test and a first candidate"
    assert rules.outcome("fails do-nothing", "fails") == "a test, no candidate"
    assert rules.outcome("fails do-nothing", "no call") == "a test, no candidate"
    assert rules.outcome("within the band", "unread") == "the adversary cannot fail do-nothing here"
    assert rules.outcome("fails do-nothing", "unread") == "unread"
    assert rules.outcome("unread", "passes") == "unread"


def test_stage_windows_and_stops() -> None:
    assert rules.stage1_ok(1.36) and rules.stage1_ok(2.04) and rules.stage1_ok(1.7016)
    assert not (rules.stage1_ok(1.3599) or rules.stage1_ok(2.0401) or rules.stage1_ok(NAN))
    assert rules.i_clean_ok(0.05) and not rules.i_clean_ok(0.0501) and not rules.i_clean_ok(NAN)
    assert rules.n_clean_ok(0.99e-4) and not rules.n_clean_ok(1e-4) and not rules.n_clean_ok(NAN)
    assert rules.first_step_ok(0.09) and not rules.first_step_ok(NAN) and not rules.first_step_ok(math.inf)


def test_score_empty_is_all_unread() -> None:
    s = rules.score({})
    assert s["outcome"] == "unread" and s["do_nothing"] == "unread" and s["inspan"] == "unread" and s["A"] == "unread"
    assert s["stage1_in_window"] is None and s["leans"] == {"within_the_band": None, "inspan_passes": None}


def test_score_stop_at_stage_1_scores_what_was_read() -> None:
    s = rules.score(starts(W0_W_s0=3.0))
    assert s["stage1_in_window"] is False and s["outcome"] == "unread" and s["W_as_read"]["W0"] == 3.0


def test_score_equal_w0_starts_and_outcomes() -> None:
    w0 = {"W0_M_s0": 2.0, "W0_M_s1": 2.0}
    fail_pass = rules.score(starts(**w0, N_M_s0=3.0, I_M_s0=2.8, A_M_s0=3.3))
    assert fail_pass["b"] == 0.19 and fail_pass["outcome"] == "D1 has a test and a first candidate" and fail_pass["A"] == "fails"
    assert rules.score(starts(**w0, N_M_s0=2.38))["outcome"] == "the adversary cannot fail do-nothing here"  # r = 1.19 exactly, I unread
    assert rules.score(starts(**w0, N_M_s0=3.0))["outcome"] == "unread"  # fails, I unread
    assert rules.score(starts(W0_M_s0=0.0, W0_M_s1=2.0, N_M_s0=3.0))["do_nothing"] == "unread"  # a zero W0


def test_score_null_value_reads_unread() -> None:
    assert rules.score({"starts": {"W0_M_s0": {"kl": None}, "W0_M_s1": {"kl": 2.0}, "N_M_s0": {"kl": 3.0}}})["do_nothing"] == "unread"


def test_m_rows_on_the_files() -> None:
    maths = json.loads((EXPERIMENTS / "qwen_sft" / "results" / "maths_rows.json").read_text())["rows"]
    web = json.loads((EXPERIMENTS / "qwen_adversary" / "results" / "tokens.json").read_text())["rows"]
    used = [r[:256] for r in (*maths[:64], *web[:32])]  # E74's and E76's maths rows, E76's FineWeb pool
    M = rules.m_rows(maths, used)
    assert len(M) == 64 and {len(r) for r in M} == {256} and M[0] == maths[64][:256]


def test_m_rows_refuses_an_overlap_and_a_short_file() -> None:
    maths = [[i] * 512 for i in range(128)]
    with pytest.raises(AssertionError):
        rules.m_rows(maths, [maths[100]])
    with pytest.raises(AssertionError):
        rules.m_rows(maths[:64], [])


def test_predictions_file_names_the_bars() -> None:
    p = json.loads((EXPERIMENTS / "qwen_d1_adversary" / "results" / "predictions.json").read_text())
    assert "1.36-2.04" in p["stages"][0]["stop"] and "1.40" in p["rules"]["inspan"] and "0.19" in p["rules"]["band"]
