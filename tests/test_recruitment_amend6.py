"""CPU checks of E38 Amendment 6's P14 (commute.recruitment) with the edge cases listed in predictions_amend6.json:
the 0.10 floors, G < 1, a < 0, the tie at a == 0.05 ('separate' checked first, so it reads 'separate' only when
min(rho_r(S), rho_r(D)) is exactly 0.10), seeds that disagree and an unscored seed."""

from __future__ import annotations

import math

from commute import recruitment as RT

nan = float("nan")


def test_overlap():
    assert RT.overlap({"S": 0.25, "D": 0.5, "S_union_D": 0.75}) == 0.25
    assert RT.overlap({"S": 0.5, "D": 0.25, "S_union_D": 0.25}) == -0.25  # the union below one set: a < 0


def test_p14_seed_rule():
    c = RT.p14_seed_class
    assert c(True, 0.2, 0.2, 0.1) == "separate"  # a = 0.5 x min: at the bar
    assert c(True, 0.2, 0.2, 0.09) == "partial"
    assert c(True, 0.2, 0.2, 0.05) == "redundant"  # a == 0.05, separate bar 0.1
    assert c(True, 0.2, 0.2, -0.3) == "redundant"  # a < 0
    assert c(True, 0.10, 0.3, 0.05) == "separate"  # min exactly 0.10: both bars at 0.05, separate checked first
    assert c(True, 0.3, 0.10, 0.04) == "redundant"
    assert c(True, 0.3, 0.4, 0.15) == "separate"  # the bar uses the smaller share


def test_p14_floors_and_gain():
    c = RT.p14_seed_class
    assert c(True, 0.099, 0.3, 0.2) == RT.UNSCORED  # rho_r(S) below 0.10 (' we': 0.037, 0.049)
    assert c(True, 0.3, 0.099, 0.2) == RT.UNSCORED  # rho_r(D) below 0.10
    assert c(True, -0.2, 0.3, 0.2) == RT.UNSCORED  # a negative share is below the floor
    assert c(False, nan, nan, nan) == RT.UNSCORED  # G < 1: nan shares, caught first
    assert c(True, 0.10, 0.10, 0.0) == "redundant"  # exactly at both floors: scored


def test_p14_task_rule():
    t = RT.p14_task_class
    assert t(["partial", "partial"]) == "partial"
    assert t(["separate", "partial"]) == "seeds disagree"
    assert t(["separate", RT.UNSCORED]) == RT.UNSCORED  # unscored before disagreement
    assert t([RT.UNSCORED, RT.UNSCORED]) == RT.UNSCORED


def _r(S: float, D: float, U: float, scorable: bool = True) -> dict:
    return {"scorable": scorable, "rho": {"S": S, "D": D, "S_union_D": U}}


def test_p14_reading():
    states = {"emoticon": {"0": _r(0.25, 0.375, 0.5), "1": _r(0.25, 0.375, 0.375)},  # a = 0.125 separate, 0 redundant
              "which": {"0": _r(0.1875, 0.1875, 0.25), "1": _r(0.1875, 0.25, 0.3125)},  # a = 0.0625 both: partial
              "we": {"0": _r(0.037, 0.11, 0.2), "1": _r(nan, nan, nan, scorable=False)}}
    out = RT.p14_reading(states, "which", "emoticon")
    assert out["a"]["emoticon/s0"] == 0.125 and out["a"]["which/s1"] == 0.0625 and math.isnan(out["a"]["we/s1"])
    assert out["seed_class"] == {"emoticon/s0": "separate", "emoticon/s1": "redundant", "which/s0": "partial",
                                 "which/s1": "partial", "we/s0": RT.UNSCORED, "we/s1": RT.UNSCORED}
    assert out["task_class"] == {"emoticon": "seeds disagree", "which": "partial", "we": RT.UNSCORED}
    assert out["reading"] == "partial" and out["beside"] == "seeds disagree"
