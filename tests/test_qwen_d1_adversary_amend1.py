"""E80 Amendment 1's filed rules on their edge cases: r on each bar, NaN, a missing start, equal seeds (b = the floor), N as its own candidate,
one seed reversed, the stop state, and every outcome branch."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from commute.jobs import load_experiment

a1 = load_experiment("qwen_d1_adversary", "rules_amend1")
EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
NAN = math.nan
SEP, NOT, UNREAD = "separated", "not separated", "unread"


def starts(**kl: float) -> dict[str, dict[str, float]]:
    return {k: {"kl": v} for k, v in kl.items()}


def test_pair_ratios_nan_missing_and_zero_w0() -> None:
    s = starts(W0_W_s0=2.0, W0_W_s1=0.0, N_W_s0=3.0, N_W_s1=3.0, I_W_s0=NAN)
    assert a1.pair(s, "N")[0] == 1.5 and math.isnan(a1.pair(s, "N")[1])  # a zero W0
    assert math.isnan(a1.pair(s, "I")[0]) and math.isnan(a1.pair(s, "I")[1])  # NaN, and a missing start
    assert all(math.isnan(x) for x in a1.pair(s, "A"))
    assert math.isnan(a1.pair({"W0_W_s0": {"kl": None}, "N_W_s0": {"kl": 1.0}}, "N")[0])


def test_band_floor_on_equal_seeds_and_the_larger_spread() -> None:
    assert a1.band((1.3, 1.3), (1.1, 1.1)) == 0.05
    assert a1.band((1.3, 1.31), (1.1, 1.1)) == 0.05  # spread 0.0077, under the floor
    assert a1.band((1.0, 1.5), (1.1, 1.1)) == pytest.approx(0.5)
    assert a1.band((1.5, 1.0), (1.0, 1.2)) == pytest.approx(0.5)  # N's, the larger
    assert a1.band((1.0, 1.1), (1.0, 1.4)) == pytest.approx(0.4)  # I's, the larger


def test_band_unread() -> None:
    for n, i in (((NAN, 1.0), (1.0, 1.0)), ((1.0, 1.0), (1.0, NAN)), ((0.0, 1.0), (1.0, 1.0)), ((1.0, 1.0), (-1.0, 1.0))):
        assert math.isnan(a1.band(n, i)), (n, i)


def test_reproduces_on_the_bar() -> None:
    assert a1.reproduces(1.0000001) == "reproduces"
    assert a1.reproduces(1.0) == "does not reproduce"
    assert a1.reproduces(0.9) == "does not reproduce"
    assert a1.reproduces(NAN) == UNREAD


def test_separated_on_the_bar() -> None:
    n = (1.5, 1.5)
    assert a1.separated(n, (1.0, 1.0), 0.5) == NOT  # gap 1.5 = 1 + b exactly
    assert a1.separated(n, (1.0, 1.0), 0.4999999) == SEP
    assert a1.separated(n, (1.5, 1.0), 0.0) == NOT  # r_0(N) = r_0(I) exactly: not strictly above on both seeds
    assert a1.separated(n, (1.4999999, 1.0), 0.05) == SEP  # gap 1.25 over 1.05, both seeds strictly above


def test_separated_one_seed_reversed() -> None:
    assert a1.separated((1.5, 1.5), (1.0, 1.6), 0.05) == NOT  # gap 1.22 > 1.05 but seed 1 is reversed
    assert a1.separated((1.5, 1.5), (1.6, 1.0), 0.05) == NOT


def test_separated_unread() -> None:
    assert a1.separated((NAN, 1.5), (1.0, 1.0), 0.05) == UNREAD
    assert a1.separated((1.5, 1.5), (1.0, NAN), 0.05) == UNREAD
    assert a1.separated((1.5, 1.5), (1.0, 0.0), 0.05) == UNREAD  # a zero r(X)
    assert a1.separated((1.5, 1.5), (1.0, 1.0), NAN) == UNREAD


def test_n_as_its_own_candidate_is_not_separated() -> None:
    n = (1.28, 1.31)
    assert a1.gap(n, n) == 1.0
    assert a1.separated(n, n, a1.BAND_MIN) == NOT
    assert a1.separated(n, n, 0.0) == NOT


def test_outcome_every_branch() -> None:
    rep = "reproduces"
    assert a1.outcome(rep, SEP, NOT) == "the adversary follows placement here"
    assert a1.outcome(rep, SEP, SEP) == "separates moved components from unmoved, not good from bad placement"
    assert a1.outcome(rep, NOT, SEP) == "cannot tell in-span from do-nothing on W"
    assert a1.outcome(rep, NOT, UNREAD) == "cannot tell in-span from do-nothing on W"  # decides without A
    assert a1.outcome(rep, SEP, UNREAD) == UNREAD
    assert a1.outcome(rep, UNREAD, NOT) == UNREAD
    assert a1.outcome("does not reproduce", UNREAD, UNREAD) == "do-nothing's rise does not reproduce on W"
    assert a1.outcome(UNREAD, UNREAD, UNREAD) == UNREAD


def full(**over: float) -> dict[str, dict[str, float]]:
    kl = {"W0_W_s0": 1.75, "W0_W_s1": 1.70, "N_W_s0": 2.24, "N_W_s1": 2.20, "I_W_s0": 1.80, "I_W_s1": 1.76, "A_W_s0": 2.20, "A_W_s1": 2.18}
    return starts(**{**kl, **over})


def test_score_outcomes_end_to_end() -> None:
    follows = a1.score(full())
    assert follows["outcome"] == "the adversary follows placement here" and follows["reproduces"] == "reproduces"
    assert follows["separated"] == {"N": NOT, "I": SEP, "A": NOT} and follows["b"] == pytest.approx(0.05)
    assert a1.score(full(A_W_s0=1.80, A_W_s1=1.76))["outcome"] == "separates moved components from unmoved, not good from bad placement"
    assert a1.score(full(I_W_s0=2.20, I_W_s1=2.18))["outcome"] == "cannot tell in-span from do-nothing on W"


def test_score_the_stop_state_and_empty() -> None:
    stop = a1.score(starts(W0_W_s0=1.75, N_W_s0=2.24, W0_W_s1=1.70, N_W_s1=1.65))  # r_1(N) < 1: the job stops here
    assert stop["reproduces"] == "does not reproduce" and stop["outcome"] == "do-nothing's rise does not reproduce on W"
    assert stop["separated"]["I"] == UNREAD and stop["separated"]["A"] == UNREAD and math.isnan(stop["b"])
    assert stop["leans"] == {"reproduces": False, "I_not_separated": None}
    empty = a1.score({})
    assert empty["outcome"] == UNREAD and empty["reproduces"] == UNREAD and empty["leans"] == {"reproduces": None, "I_not_separated": None}


def test_score_a_missing_and_a_nan_start() -> None:
    assert a1.score(full(A_W_s1=NAN))["outcome"] == UNREAD
    missing = {k: v for k, v in full().items() if k != "I_W_s1"}
    assert a1.score(missing)["outcome"] == UNREAD and a1.score(missing)["separated"]["A"] == UNREAD  # b needs I's pair


def test_score_carries_last5_and_checks() -> None:
    s = full()
    s["N_W_s1"]["last5"] = 2.1
    out = a1.score(s, {"N_clean": {"got": 0.0}})
    assert out["last5"] == {"N_W_s1": 2.1} and out["checks"] == {"N_clean": {"got": 0.0}}


def test_predictions_file_names_the_rules() -> None:
    p = json.loads((EXPERIMENTS / "qwen_d1_adversary" / "results" / "predictions_amend1.json").read_text())
    assert "0.05" in p["rules"]["band"] and "r_1(N) > 1" in p["rules"]["reproduces"] and "1 + b" in p["rules"]["separated"]
    assert p["order"][0] == "W0 W s1" and len(p["order"]) == 6
