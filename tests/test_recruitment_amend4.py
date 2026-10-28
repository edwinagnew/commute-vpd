"""CPU checks of E38 Amendment 4's pure parts (commute.recruitment): the output change Delta_c, the count- and
change-matched sets with the filed tie rule and edge cases, S's share of the top movers, and the readings P11, P12."""

from __future__ import annotations

import math

import torch

from commute import recruit as R
from commute import recruitment as RT


def test_output_change_formula_and_batches():
    """Delta_c = mean over WINDOWS (not batches) of |(x1 - x0) . v_c| * ||u_c||, in fp32 from bf16 inputs; 0 when the
    two passes agree (R5)."""
    torch.manual_seed(0)
    U, V = torch.randn(5, 3), torch.randn(4, 5)  # (C, d_out), (d_in, C)
    x0, x1 = torch.randn(5, 4).bfloat16(), torch.randn(5, 4).bfloat16()  # 5 windows in batches of 2, 2, 1
    io0 = {"h.0.a": [(x0[s : s + 2], torch.zeros(0)) for s in (0, 2, 4)]}
    io1 = {"h.0.a": [(x1[s : s + 2], torch.zeros(0)) for s in (0, 2, 4)]}
    d = RT.output_change({"h.0.a": (U, V)}, io0, io1)["h.0.a"]
    want = ((x1.float() - x0.float()) @ V).abs().mean(0) * U.norm(dim=1)
    assert d.dtype == torch.float32 and torch.allclose(d, want, atol=1e-6)
    assert RT.output_change({"h.0.a": (U, V)}, io0, io0)["h.0.a"].abs().max() == 0


def _delta():
    alive = {"h.0.x": torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool), "h.0.y": torch.ones(3, dtype=torch.bool),
             "h.1.x": torch.tensor([1, 1, 0], dtype=torch.bool), "h.2.x": torch.tensor([1, 0], dtype=torch.bool)}
    delta = {"h.0.x": torch.tensor([5.0, 2.0, 2.0, 1.0, 9.0]), "h.0.y": torch.tensor([2.0, 0.5, 3.0]),
             "h.1.x": torch.tensor([1.0, 1.0, 7.0]), "h.2.x": torch.tensor([0.0, 4.0])}
    return alive, delta


def _pieces(sel):
    return sorted(R.members(sel))


def test_movers_ties_and_dead():
    alive, delta = _delta()
    S = {"h.0.x": torch.tensor([0]), "h.1.x": torch.tensor([0]), "h.2.x": torch.tensor([0])}
    rk = RT.movers(delta, alive, S)
    # dead h.0.x[4] (9.0) and S never ranked; ties at 2.0 by matrix name, then index
    assert rk["0"] == [("h.0.y", 2, 3.0), ("h.0.x", 1, 2.0), ("h.0.x", 2, 2.0), ("h.0.y", 0, 2.0), ("h.0.x", 3, 1.0),
                       ("h.0.y", 1, 0.5)]
    assert rk["1"] == [("h.1.x", 1, 1.0)] and rk["2"] == []


def test_count_and_change_matched():
    alive, delta = _delta()
    keys = sorted(alive)
    S = {"h.0.x": torch.tensor([0]), "h.0.y": torch.zeros(0, dtype=torch.long), "h.1.x": torch.tensor([0]),
         "h.2.x": torch.tensor([0])}
    rk, by, mass = RT.movers(delta, alive, S), R.by_layer(S), RT.layer_mass(delta, S)
    assert by == {"0": 1, "1": 1, "2": 1} and mass == {"0": 5.0, "1": 1.0, "2": 0.0}
    Cn, n_short = RT.count_matched(rk, by, keys)
    assert _pieces(Cn) == [("h.0.y", 2), ("h.1.x", 1)] and set(Cn) == set(keys)
    assert n_short == {"0": False, "1": False, "2": True}  # layer 2: no alive piece outside S (empty layer)
    Cm, m_short = RT.change_matched(rk, mass, keys)
    # layer 0: 3 + 2 >= 5 (reached exactly); layer 1: 1 >= 1; layer 2: S's sum is 0, so empty and not short
    assert _pieces(Cm) == [("h.0.x", 1), ("h.0.y", 2), ("h.1.x", 1)] and m_short == {"0": False, "1": False, "2": False}
    Cm, m_short = RT.change_matched(rk, {"0": 100.0, "1": 1.0, "2": 4.0}, keys)  # never reached: whole layer, short
    assert R.by_layer(Cm) == {"0": 6, "1": 1, "2": 0} and m_short == {"0": True, "1": False, "2": True}
    Cn, n_short = RT.count_matched(rk, {"1": 3}, keys)  # short layer: all of it, recorded
    assert _pieces(Cn) == [("h.1.x", 1)] and n_short == {"1": True}


def test_top_share():
    alive, delta = _delta()
    S = {"h.0.x": torch.tensor([0, 3]), "h.1.x": torch.tensor([0])}
    # layer 0 top 2 (S included): h.0.x[0] 5.0, h.0.y[2] 3.0 -> 1 hit; layer 1 top 1: tie at 1.0, h.1.x[0] first -> hit
    share, hits = RT.top_share(delta, alive, S)
    assert hits == {"0": 1, "1": 1} and share == 2 / 3
    assert math.isnan(RT.top_share(delta, alive, {"h.0.x": torch.zeros(0, dtype=torch.long)})[0])


def _a4(S: float, cn: float, cm: float, scorable: bool = True) -> dict:
    return {"scorable": scorable, "rho": {"S": S, "C_n": cn, "C_m": cm}}


def test_p11_seed_and_task_rules():
    assert RT.p11_seed_class(0.2, 0.0, 0.0) == "specific"  # empty C restores exactly 0
    assert RT.p11_seed_class(0.2, 0.149, 0.1) == "specific"
    assert RT.p11_seed_class(0.2, 0.2, 0.0) == "not"  # a tie reads 'not'
    assert RT.p11_seed_class(0.2, 0.0, 0.3) == "not"
    assert RT.p11_seed_class(0.2, 0.16, 0.0) == "unclear"
    nan = float("nan")
    assert RT.p11_task_class([_a4(nan, nan, nan, scorable=False), _a4(0.3, 0, 0)]) == RT.UNSCORED_GAIN
    assert RT.p11_task_class([_a4(0.3, 0, 0), _a4(0.09, 0, 0)]) == RT.TOO_SMALL
    assert RT.p11_task_class([_a4(0.3, 0, 0), _a4(0.3, 0.3, 0)]) == "seeds disagree"


def test_p11_reading():
    sp, no = {"0": _a4(0.3, 0, 0), "1": _a4(0.3, 0, 0)}, {"0": _a4(0.3, 0.4, 0), "1": _a4(0.3, 0.3, 0)}
    small = {"0": _a4(0.04, 0, 0), "1": _a4(0.05, 0, 0)}
    out = RT.p11_reading({"emoticon": sp, "which": sp, "we": small})
    assert out["reading"] == "specific" and out["task_class"]["we"] == RT.TOO_SMALL and out["n_scored_tasks"] == 2
    assert RT.p11_reading({"emoticon": no, "which": no, "we": small})["reading"] == "not specific"
    assert RT.p11_reading({"emoticon": sp, "which": no, "we": sp})["reading"] == "outside the tree"  # a 'not' blocks
    one = RT.p11_reading({"emoticon": sp, "which": small, "we": small})  # fewer than two scored tasks
    assert one["n_scored_tasks"] == 1 and one["reading"] == "outside the tree"
    nan = float("nan")
    uns = {"0": _a4(nan, nan, nan, scorable=False), "1": _a4(0.3, 0, 0)}
    out = RT.p11_reading({"emoticon": sp, "which": uns, "we": small})
    assert out["seed_class"]["which/s0"] == RT.UNSCORED_GAIN and out["reading"] == "outside the tree"


def test_p12():
    assert RT.p12_seed_class(True, [0.0, -0.08], [0.01, -0.05]) == "few"  # -0.08 <= -0.05 and <= -0.05 - 0.02
    assert RT.p12_seed_class(True, [0.0, -0.06], [-0.05]) == "unclear"  # not 0.02 below the random singles
    assert RT.p12_seed_class(True, [-0.01, 0.02], [-0.3]) == "spread"
    assert RT.p12_seed_class(True, [-0.08], []) == "unclear"  # no random singles: never 'few'
    assert RT.p12_seed_class(True, [], [0.0]) == RT.UNSCORED_L3
    assert RT.p12_seed_class(False, [-0.5], [0.0]) == RT.UNSCORED_GAIN
    few, spread = {"scorable": True, "singles": [-0.1], "random_singles": [0.0]}, {"scorable": True, "singles": [0.0],
                                                                                    "random_singles": [0.0]}
    assert RT.p12_reading({"s0": few, "s1": few})["reading"] == "few"
    assert RT.p12_reading({"s0": few, "s1": spread})["reading"] == "seeds disagree"
    assert RT.p12_reading({"s0": few, "s1": {**few, "scorable": False}})["reading"] == RT.UNSCORED_GAIN
