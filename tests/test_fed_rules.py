"""E100's filed rules (`commute.fed_rules`) on hand-made records: every edge case the predictions name, and the
do-nothing state run through each rule before filing."""

from __future__ import annotations

import math

from commute import fed_rules as FR
from commute import locator as L
from commute.locator_rules import upstream

TH = {"reread_tol": 1e-4, "zero_tol": 1e-3}
BARS = FR.bars([0.01, 0.02, 0.015], [0.72, 0.75])
FLOOR = 0.021


def names() -> list[str]:
    return [L.name(p) for p in L.PARTS]


def plant(block: int, down: float = 0.0) -> dict[str, dict[str, float]]:
    """An exact one-matrix plant in block `block`'s MLP, carry `down` on every part downstream."""
    out = {}
    for n in names():
        at = 1.0 if n == f"mlp{block}" else 0.0
        below = n != f"mlp{block}" and not upstream(n, block)
        out[n] = {"own": at, "graft": at, "carry": down if below else at}
    return out


def test_bars_read_from_the_poles() -> None:
    assert BARS == {"fed": 0.02, "written": 0.72, "ok": True, "why": ""}


def test_copy_bar_not_below_cipher_bar_stops() -> None:
    assert not FR.bars([0.5], [0.5])["ok"]  # a tie does not separate
    assert not FR.bars([0.8], [0.5])["ok"]
    assert not FR.bars([0.01, math.nan], [0.7])["ok"]
    assert not FR.bars([], [0.7])["ok"]


def test_readings() -> None:
    assert FR.reading(10.0, 0.35, 0.007, FLOOR, BARS) == FR.FED
    assert FR.reading(10.0, 0.66, 0.48, FLOOR, BARS) == FR.WRITTEN
    assert FR.reading(10.0, 0.62, 0.16, FLOOR, BARS) == FR.MIXED


def test_ties_pass_on_both_bars() -> None:
    assert FR.reading(1.0, 1.0, 0.02, FLOOR, BARS) == FR.FED
    assert FR.reading(1.0, 1.0, 0.72, FLOOR, BARS) == FR.WRITTEN


def test_zero_or_negative_carry_has_no_part() -> None:
    for c in (0.0, -0.3, FLOOR, 0.01, math.nan):
        assert FR.reading(1.0, c, 0.0, FLOOR, BARS) == FR.NO_PART
    assert math.isnan(FR.ratio(0.0, 0.1)) and math.isnan(FR.ratio(-0.2, 0.1))


def test_ratio_past_one_is_capped_and_written() -> None:
    assert FR.ratio(0.2, 0.5) == 1.0
    assert FR.reading(1.0, 0.2, 0.5, FLOOR, BARS) == FR.WRITTEN


def test_negative_own_reads_fed_when_inherited() -> None:
    assert FR.reading(1.0, 0.3, -0.05, FLOOR, BARS) == FR.FED


def test_fed_needs_inherited_over_the_floor() -> None:
    """A small ratio on a carry only just over the floor leaves inherited under it: mixed, not fed."""
    assert FR.ratio(0.0212, 0.0003) <= BARS["fed"]
    assert FR.reading(1.0, 0.0212, 0.0003, FLOOR, BARS) == FR.MIXED


def test_no_gain_and_no_bars() -> None:
    assert FR.reading(0.0, 0.5, 0.1, FLOOR, BARS) == FR.NO_GAIN
    assert FR.reading(-1.0, 0.5, 0.1, FLOOR, BARS) == FR.NO_GAIN
    assert FR.reading(1.0, 0.5, 0.1, FLOOR, FR.bars([0.8], [0.5])) == FR.NO_BARS
    assert FR.reading(1.0, 0.5, 0.1, math.nan, BARS) == FR.NO_BARS


def test_a_seed_that_disagrees() -> None:
    assert FR.task_reading([FR.FED, FR.FED, FR.MIXED]) == FR.DISAGREE
    assert FR.task_reading([FR.FED, FR.FED]) == FR.FED
    assert FR.task_reading([FR.FED, FR.NO_GAIN]) == FR.NO_GAIN
    assert FR.task_reading([]).startswith("unreadable")
    assert FR.lean_met(FR.DISAGREE, FR.FED) is False and FR.lean_met(FR.FED, None) is None


def test_read_at_filed_else_top_carry_ties_name_none() -> None:
    sp = {"2,4": {"carry": 0.3, "own": 0.0, "graft": 0.0}, "mlp3": {"carry": 0.4, "own": 0.0, "graft": 0.0},
          "2,3+2,4": {"carry": 0.9, "own": 0.0, "graft": 0.0}}  # fmt: skip
    assert FR.read_at(sp, "2,4") == "2,4"
    assert FR.read_at(sp, None) == "mlp3"  # a pair is never the top by carry
    sp["2,4"]["carry"] = 0.4
    assert FR.read_at(sp, None) is None
    r = FR.row("T", 0, sp, 1.0, None, FLOOR, BARS, False)
    assert r["reading"].startswith("unreadable") and r["part"] is None


def test_constant_rows_read_beside() -> None:
    sp = {"mlp3": {"carry": 0.5, "own": 0.01, "graft": 0.0}}
    r = FR.row("T1 emoticon", 0, sp, 2.0, "mlp3", FLOOR, BARS, True)
    assert r["reading"] == FR.BESIDE and r["reading_rule"] == FR.FED
    assert math.isclose(r["inherited"], 0.49) and math.isclose(r["reverts_sum"], 0.01)


def test_plants_pass_exact_and_miss_otherwise() -> None:
    assert FR.plant_misses(plant(3), 3, 0.0, TH) == []
    p1 = plant(1, 0.021)
    assert FR.plant_misses(p1, 1, 0.0, TH) == []
    assert math.isclose(FR.floor_of(p1, 1), 0.021)
    assert math.isnan(FR.floor_of(plant(3), 3))  # nothing downstream of block 3's MLP
    bad = plant(3)
    bad["mlp2"] = {"own": 0.0, "graft": 0.0, "carry": 0.01}
    assert FR.plant_misses(bad, 3, 0.0, TH) == ["carry mlp2 0.01"]
    assert FR.plant_misses(plant(3), 3, 0.01, TH) == ["all reverted G 0.01"]


def test_e99_poles() -> None:
    forced = {"2,3+2,4": {"own": 1.0, "carry": 1.0, "graft": 1.0}}
    assert FR.pole_misses(forced, "2,3+2,4", "forced", TH) == []
    assert FR.pole_misses({"2,3+2,4": {"own": 0.0, "carry": 0.4, "graft": 0.0}}, "2,3+2,4", "forbidden", TH) == []
    assert FR.pole_misses({}, "2,3+2,4", "forced", TH) != []


def test_outcome() -> None:
    track = {"T13 copy-24": FR.FED, "T11 cipher": FR.WRITTEN, "T4 ident32": FR.FED, "T6 rand32": FR.WRITTEN}
    assert FR.outcome(track).startswith("tracks")
    assert FR.outcome(track | {"T4 ident32": FR.WRITTEN, "T6 rand32": FR.FED}).startswith("does not track")
    assert FR.outcome(track | {"T11 cipher": FR.FED}).startswith("does not track")
    assert FR.outcome(track | {"T13 copy-24": FR.MIXED, "T11 cipher": FR.MIXED}).startswith("partial")
    assert FR.outcome({}).startswith("partial")


def test_do_nothing_state() -> None:
    """W0 as the state: G = 0, every share nan (locator_rules.frac). No rule can read it as fed or written."""
    nan = math.nan
    sp = {n: {"carry": nan, "own": nan, "graft": nan} for n in names()}
    assert FR.reading(0.0, nan, nan, FLOOR, BARS) == FR.NO_GAIN
    assert FR.row("T5 copy", 0, sp, 0.0, "2,4", FLOOR, BARS, False)["reading"] == FR.NO_GAIN
    assert FR.row("T5 copy", 0, sp, 0.0, FR.read_at(sp, None), FLOOR, BARS, False)["reading"] == FR.NO_GAIN
    assert not FR.bars([nan, nan, nan], [nan, nan])["ok"]
    zeros = {n: {"carry": 0.0, "own": 0.0, "graft": 0.0} for n in names()}
    assert FR.plant_misses(zeros, 3, 0.0, TH)  # a zero plant misses stage 0
    assert FR.reading(1.0, 0.0, 0.0, FLOOR, BARS) == FR.NO_PART  # zero carry, even with a gain


# ----------------------------------------------------------------------------- Amendment 1: the sum of single reverts

A1 = {"fed": 0.40, "written": 0.62, "min_gain": 1.0}  # results/predictions_amend1.json


def a1(G: float, s: float) -> str:
    return FR.sum_reading(G, s, **A1)


def test_sum_gain_at_or_under_one_nat_reads_no_gain() -> None:
    assert a1(1.0, 0.2) == a1(0.5, 0.9) == a1(-3.0, 0.2) == a1(math.nan, 0.2) == FR.NO_GAIN_SUM
    assert a1(1.0001, 0.2) == FR.FED


def test_sum_bars_and_ties() -> None:
    assert a1(12.0, 0.40) == FR.FED  # exactly at the fed bar
    assert a1(12.0, 0.62) == FR.WRITTEN  # exactly at the written bar
    assert a1(12.0, 0.41) == a1(12.0, 0.61) == FR.BETWEEN
    assert a1(12.0, -0.1) == FR.FED  # a negative sum is under the fed bar


def test_sum_past_one_counts() -> None:
    assert a1(12.0, 1.3) == FR.WRITTEN


def test_sum_not_finite() -> None:
    assert a1(12.0, math.nan) == FR.NO_SUM


def test_sum_reads_e100_seeds_as_the_plan() -> None:
    """The plan's table: every copy row at or under 0.40, every row new by construction at or over 0.62."""
    assert {a1(13.0, s) for s in (0.32, 0.36, 0.26, 0.30, 0.22, 0.40, 0.35, 0.24)} == {FR.FED}
    assert {a1(13.0, s) for s in (0.68, 0.62, 0.66, 0.70, 0.72, 0.88, 0.97)} == {FR.WRITTEN}


def test_sum_pass_rule() -> None:
    copy, cipher = [FR.FED] * 5, [FR.WRITTEN] * 5
    assert FR.sum_pass(copy, cipher, 9) == {"hits": 10, "of": 10, "need": 9, "pass": True}
    assert FR.sum_pass([FR.BETWEEN, *copy[1:]], cipher, 9)["pass"]  # one miss passes
    assert not FR.sum_pass([FR.BETWEEN, *copy[1:]], [FR.NO_GAIN_SUM, *cipher[1:]], 9)["pass"]  # two do not
    assert not FR.sum_pass([FR.WRITTEN] * 5, [FR.FED] * 5, 9)["pass"]  # swapped sides
    assert not FR.sum_pass([], [], 9)["pass"]


def test_sum_do_nothing_state() -> None:
    """W0 as the state: G = 0 and every own share nan, so the sum is nan and the state reads no gain."""
    s = FR.reverts_sum({n: {"own": math.nan} for n in names()})
    assert math.isnan(s) and a1(0.0, s) == FR.NO_GAIN_SUM
