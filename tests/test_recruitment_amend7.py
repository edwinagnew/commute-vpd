"""CPU checks of E38 Amendment 7's P15 and P16 (commute.recruitment) with the edge cases listed in
predictions_amend7.json: the 0.10 floors, a short D2 or T2, G < 1, no scored task, a2 < 0, P14's tie order
('separate' first), rho_r(T2) == rho_r(S union D) reading 'covers', seeds that disagree and a mixed reading."""

from __future__ import annotations

import math

from commute import recruitment as RT

nan = float("nan")


def _r(D: float = 0.25, D2: float = 0.25, DD2: float = 0.5, SD: float = 0.5, T2: float = 0.5,
       scorable: bool = True, D2_short: bool = False, T2_short: bool = False) -> dict:
    return {"scorable": scorable, "D2_short": D2_short, "T2_short": T2_short,
            "rho": {"D": D, "D2": D2, "D_union_D2": DD2, "S_union_D": SD, "T2": T2}}


def test_overlap2():
    assert RT.overlap2(_r(D=0.25, D2=0.125, DD2=0.375)["rho"]) == 0.125
    assert RT.overlap2(_r(D=0.5, D2=0.125, DD2=0.25)["rho"]) == -0.25  # the union below D: a2 < 0


def test_p15_seed_rule():
    c = RT.p15_seed_class
    assert c(_r(D=0.25, D2=0.125, DD2=0.3125)) == "separate"  # a2 = 0.0625 = 0.5 x min: at the bar
    assert c(_r(D=0.25, D2=0.25, DD2=0.34375)) == "partial"  # a2 = 0.09375 < 0.125, > 0.05
    assert c(_r(D=0.25, D2=0.25, DD2=0.25)) == "redundant"  # a2 = 0
    assert c(_r(D=0.5, D2=0.25, DD2=0.25)) == "redundant"  # a2 < 0
    assert c(_r(D=0.11, D2=0.10, DD2=0.11 + 0.05)) == "separate"  # a2 == 0.05 exactly, min 0.10: both bars, separate first
    assert c(_r(D=0.3, D2=0.2, DD2=0.34)) == "redundant"  # a2 = 0.04


def test_p15_unscored():
    c = RT.p15_seed_class
    assert c(_r(D=0.3, D2=0.09, DD2=0.5)) == RT.UNSCORED  # rho_r(D2) below 0.10 (expected on ' which')
    assert c(_r(D=0.09, D2=0.3, DD2=0.5)) == RT.UNSCORED  # rho_r(D) below 0.10
    assert c(_r(D2_short=True)) == RT.UNSCORED  # D2 short in some matrix
    assert c(_r(nan, nan, nan, nan, nan, scorable=False)) == RT.UNSCORED  # G < 1: nan shares, caught first
    assert c(_r(T2_short=True)) == "separate"  # T2's flag is not P15's


def test_p15_reading():
    sep, red, part = _r(D=0.25, D2=0.25, DD2=0.5), _r(D=0.25, D2=0.25, DD2=0.25), _r(D=0.25, D2=0.25, DD2=0.34375)
    low = _r(D=0.25, D2=0.09, DD2=0.3)
    out = RT.p15_reading({"emoticon": {"0": sep, "1": sep}, "which": {"0": low, "1": sep}, "we": {"0": low, "1": low}})
    assert out["task_class"] == {"emoticon": "separate", "which": RT.UNSCORED, "we": RT.UNSCORED}  # unscored seed wins
    assert out["reading"] == "generic" and out["n_scored_tasks"] == 1 and out["a2"]["emoticon/s0"] == 0.25
    assert RT.p15_reading({"e": {"0": red, "1": red}, "w": {"0": red, "1": red}})["reading"] == "not generic"
    assert RT.p15_reading({"e": {"0": low, "1": low}, "w": {"0": _r(D2_short=True), "1": sep}})["reading"] == "unscored"
    mixed = RT.p15_reading({"e": {"0": sep, "1": sep}, "w": {"0": red, "1": red}})  # separate and redundant tasks
    assert mixed["reading"] == "outside the tree"
    dis = RT.p15_reading({"e": {"0": sep, "1": part}})
    assert dis["task_class"] == {"e": "seeds disagree"} and dis["reading"] == "outside the tree"
    assert math.isnan(RT.p15_reading({"e": {"0": _r(nan, nan, nan, scorable=False)}})["a2"]["e/s0"])


def test_p16_seed_rule():
    c = RT.p16_seed_class
    assert c(_r(SD=0.5, T2=0.5)) == "covers"  # tie (S inside T2: T2 = S union D)
    assert c(_r(SD=0.5, T2=0.625)) == "covers"
    assert c(_r(SD=0.5, T2=0.4375)) == "short"  # 0.0625 below
    assert c(_r(SD=0.5, T2=0.46875)) == "unclear"
    assert c(_r(SD=0.5, T2=0.25)) == "short"
    assert c(_r(T2_short=True)) == RT.UNSCORED
    assert c(_r(nan, nan, nan, nan, nan, scorable=False)) == RT.UNSCORED
    assert c(_r(D2_short=True)) == "covers"  # D2's flag is not P16's


def test_p16_reading():
    cov, sh, unc = _r(SD=0.5, T2=0.5), _r(SD=0.5, T2=0.25), _r(SD=0.5, T2=0.46875)
    read_on = ("emoticon", "which")
    out = RT.p16_reading({"emoticon": {"0": cov, "1": cov}, "which": {"0": cov, "1": cov}, "we": {"0": sh, "1": sh}},
                         read_on)
    assert out["reading"] == "covers" and out["task_class"]["we"] == "short"  # ' we' is not read
    assert RT.p16_reading({"emoticon": {"0": sh, "1": sh}, "which": {"0": sh, "1": sh}}, read_on)["reading"] == "short"
    mixed = RT.p16_reading({"emoticon": {"0": cov, "1": cov}, "which": {"0": sh, "1": sh}}, read_on)
    assert mixed["reading"] == "outside the tree"
    dis = RT.p16_reading({"emoticon": {"0": cov, "1": unc}, "which": {"0": cov, "1": cov}}, read_on)
    assert dis["task_class"]["emoticon"] == "seeds disagree" and dis["reading"] == "outside the tree"
    uns = RT.p16_reading({"emoticon": {"0": cov, "1": _r(T2_short=True)}, "which": {"0": cov, "1": cov}}, read_on)
    assert uns["task_class"]["emoticon"] == RT.UNSCORED and uns["reading"] == "outside the tree"
