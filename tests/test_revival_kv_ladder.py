"""CPU checks of the C3 step ladder's filed rules (plan reports/plans/2026-10-31_plan_e55-c3-ladder.md;
experiments/open_pair_instruments/results/predictions_ladder.json): `revival_kv.pole_gap` on its edge cases, censoring,
the half-life of the top-guess curve, the pass step, and that C3's rule 1 (`gap_status`) is unchanged by the refactor.
No model is loaded."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from commute import revival_kv as KV

PRED = Path(__file__).resolve().parents[1] / "experiments/open_pair_instruments/results/predictions_ladder.json"
L = json.loads(PRED.read_text())
MULT = L["constants"]["gap_mult"]


def g(col, d: int | None = 1, floor: float = 0.0):
    return KV.pole_gap(col, d, floor, MULT)


def test_filed_constants():
    assert L["status"] == "filed" and MULT == 3 and L["constants"]["steps"] == [10, 30, 100, 300]
    assert L["constants"]["prune_k"][:2] == [13, 40]


@pytest.mark.parametrize("col", [{}, {"rank1": [1.0, 2.0]}, {"rank1": [1.0], "cipher": [0.0, 0.1]}, {"rank1": [None, None], "cipher": [0.0, 0.1]},
                                 {"rank1": [1.0, math.nan], "cipher": [0.0, 0.1]}])  # fmt: skip
def test_void_under_two_values_per_pole(col):
    assert g(col)["status"] == "void"


def test_gap_zero_is_no_gap_even_with_zero_noise():
    assert g({"rank1": [1.0, 1.0], "cipher": [1.0, 1.0]})["status"] == "no gap"


def test_zero_noise_floor_zero_any_gap_ok():
    r = g({"rank1": [1.0, 1.0], "cipher": [0.9, 0.9]})
    assert r["noise"] == 0 and r["status"] == "ok"


def test_at_bar_ok_just_under_no_gap():
    # sd of [0, 1, 2] = 1 exactly; a gap of exactly 3 is on the bar
    assert g({"rank1": [3.0, 4.0, 5.0], "cipher": [0.0, 1.0, 2.0]})["status"] == "ok"
    assert g({"rank1": [2.75, 3.75, 4.75], "cipher": [0.0, 1.0, 2.0]})["status"] == "no gap"


def test_floor_binds():
    col = {"rank1": [1.0, 1.0], "cipher": [0.5, 0.5]}
    assert g(col, floor=0.1)["status"] == "ok" and g(col, floor=0.2)["status"] == "no gap"
    assert g(col, floor=0.2)["noise"] == 0.2


def test_directions():
    col = {"rank1": [10.0, 10.0], "cipher": [20.0, 20.0]}  # t_half: rank-1 faster is the filed direction
    assert g(col, d=-1, floor=1)["status"] == "ok"
    assert g(col, d=1, floor=1)["status"] == "reversed"
    assert g(col, d=None, floor=1)["status"] == "ok"
    assert g(col, d=None, floor=4)["status"] == "no gap"
    assert g(col, d=-1, floor=4)["status"] == "no gap"


def test_elicit_spread_is_ignored():
    """The ladder's change from C3's rule 1: elicit's spread does not set the bar (C3's t_half bar came from elicit)."""
    col = {"rank1": [13.0, 13.5, 14.0], "cipher": [10.0, 10.5, 11.0], "elicit": [10.0, 15.0, 20.0]}
    assert g(col, d=-1, floor=1)["status"] == "reversed"
    assert KV.gap_status(col, -1, 1, MULT)["status"] == "no gap"


def test_gap_status_unchanged():
    """C3's rule 1 after the refactor: the C3 report's t_half and R verdicts on its means and sds."""
    col = {"rank1": [0.0, 1.0], "cipher": [0.0, 1.0], "elicit": [0.0, 1.0]}
    assert KV.gap_status(col, 1, 0.0, MULT)["status"] == "no gap"
    col = {"rank1": [5.0, 5.1], "cipher": [1.0, 1.1], "elicit": [3.0, 3.1]}
    assert KV.gap_status(col, 1, 0.0, MULT)["status"] == "ok"
    assert KV.gap_status(col, -1, 0.0, MULT)["status"] == "reversed"
    assert KV.gap_status({"rank1": [1.0], "cipher": [0.0, 1.0], "elicit": [0.0, 1.0]}, 1, 0.0, MULT)["status"] == "void"


def test_censoring_only_on_poles():
    col, cens = KV.t_half_column({"rank1": [10.0, "censored"], "cipher": [5.0, 6.0], "elicit": ["censored", 3.0]})
    assert cens and col["rank1"] == [10.0, None]
    _, cens = KV.t_half_column({"rank1": [10.0, 11.0], "cipher": [5.0, 6.0], "elicit": ["censored", 3.0]})
    assert not cens


def test_top_guess_half_life_on_abs_curve():
    # rank-1 raises W0's top guess: D_top < 0; the abs curve halves when the raise is half undone, or on crossing zero
    assert KV.half_life([abs(x) for x in [-4.0, -3.0, -1.0]]) == pytest.approx(1.5)
    assert KV.half_life([abs(x) for x in [-4.0, -3.0, 1.0]]) == pytest.approx(1.5)
    assert KV.half_life([abs(x) for x in [-4.0, -3.0, -2.5]]) is None


def test_first_ok_step():
    assert KV.first_ok_step({"10": "no gap", "30": "censored", "100": "reversed", "300": "void"}) is None
    assert KV.first_ok_step({"300": "ok", "100": "ok", "30": "no gap", "10": "reversed"}) == 100
    assert KV.first_ok_step({}) is None
