"""CPU checks of E66's filed rules (src/commute/ctxpatch.py) on hand-made numbers: the pairs, l*, T, Q, K, the matched
null, the units share and the stage rules at their edge cases. No model."""

from __future__ import annotations

import math

import torch

from commute import ctxpatch as X

TH = {"floor": 0.5, "z": 3.0, "n_boot": 200, "min_pairs": 200, "min_keys": 8, "transplant_tol": 1e-4, "q_band": 0.2, "write_t_min": 0.5, "blind_share": 0.5,
      "write_cond_max": 16.5, "write_rel_err_max": 0.33}


def test_quarter_pairs_are_within_key_highest_with_lowest():
    keys = torch.tensor([7] * 8 + [9] * 4 + [5] * 3)
    scores = torch.tensor([1.0, 8, 2, 7, 3, 6, 4, 5, 10, 11, 12, 13, 0, 0, 0])
    hi, lo = X.quarter_pairs(scores, keys)
    assert hi.tolist() == [1, 3, 11] and lo.tolist() == [0, 2, 8]  # key 5 has under 4 windows: no pair
    assert X.quarter_pairs(torch.zeros(2), torch.tensor([1, 1]))[0].numel() == 0


def test_quarter_pairs_ties_follow_window_order():
    hi, lo = X.quarter_pairs(torch.zeros(4), torch.zeros(4, dtype=torch.long))
    assert hi.tolist() == [0] and lo.tolist() == [3]


def test_enough_pairs_ties_pass():
    assert X.enough_pairs(200, 8, TH) and not X.enough_pairs(199, 8, TH) and not X.enough_pairs(200, 7, TH)


def test_l_star_earliest_over_floor_and_void():
    flat = lambda v: v * torch.ones(50, dtype=torch.float64)
    assert X.l_star({1: flat(0.1), 2: flat(0.5), 3: flat(2.0)}, TH, 0) == 2  # a tie at the floor counts
    assert X.l_star({1: flat(-3.0), 2: flat(0.0)}, TH, 0) is None
    noisy = torch.tensor([50.0, -49.0] * 25, dtype=torch.float64)  # mean 0.5, far under 3 sds
    assert X.l_star({1: noisy}, TH, 0) is None
    assert X.l_star({1: torch.empty(0)}, TH, 0) is None


def test_ratio_of_means_keeps_negative_pairs_and_voids_a_small_denominator():
    R0 = torch.tensor([3.0, -1.0] * 100, dtype=torch.float64)
    T, d = X.ratio(2 * R0, R0, TH, 0)
    assert abs(T - 2) < 1e-12 and float((d - 2).abs().max()) < 1e-9
    for bad in (torch.zeros(200), 0.4 * torch.ones(200), -torch.ones(200)):
        T, d = X.ratio(torch.ones(200), bad.double(), TH, 0)
        assert math.isnan(T) and bool(d.isnan().all())
    assert math.isnan(X.ratio(torch.empty(0), torch.empty(0), TH, 0)[0])


def test_quotient_voids_on_a_bad_denominator():
    d = torch.ones(5)
    assert X.quotient(2.0, 2 * d, 2.0, 2 * d) == (1.0, 0.0)
    for T_off, d_off in ((math.nan, d), (0.0, d), (-1.0, d), (1.0, torch.tensor([1.0, 1, 1, 1, -0.1]))):
        assert math.isnan(X.quotient(1.0, d, T_off, d_off)[0])


def test_first_k():
    assert X.first_k({1: 0.1, 16: 1.0, 64: 3.0}, 1.0) == 16
    assert X.first_k({1: math.nan, 4: 0.9}, 1.0) is None


def test_matched_draw_stays_in_its_cell_and_falls_back():
    a = torch.tensor([0, 0, 0, 0, 1, 1])
    b = torch.tensor([0, 0, 1, 1, 0, 0])
    gen = torch.Generator().manual_seed(0)
    d, fell = X.matched_draw(torch.tensor([0]), a, b, gen)
    assert d.tolist() == [1] and fell == 0
    d, fell = X.matched_draw(torch.tensor([0, 1]), a, b, gen)
    assert fell == 2 and set(d.tolist()) == {2, 3}  # no cell left, same activation decile
    assert X.deciles(torch.arange(20.0)).tolist() == [i // 2 for i in range(20)]


def test_units_share():
    assert X.units_share(10.0, 4.0, 1.0) == 0.6
    assert X.units_share(1.0, 2.0, 1.0) == -1.0  # a tie at the floor passes, negative kept
    assert math.isnan(X.units_share(0.0, 0.0, 1.0)) and math.isnan(X.units_share(0.5, 0.1, 1.0))


def test_stage_statuses():
    ok = {"a": 1, "b": None, "c": 3}
    assert X.stage0_status(1e-5, ok, TH) == "pass" and X.stage0_status(1e-4, ok, TH) == "pass"  # a tie passes
    assert X.stage0_status(2e-4, ok, TH) == "transplant"
    assert X.stage0_status(X.nan_max([0.0, math.nan]), ok, TH) == "transplant" and X.stage0_status(X.nan_max([math.nan, 0.0]), ok, TH) == "transplant"
    assert X.stage0_status(0.0, {"a": 1, "b": None}, TH) == "void_maps"  # half stops
    assert X.stage1_status({"a": "above", "b": "not_above", "c": "above"}) == "pass"
    assert X.stage1_status({"a": "above", "b": "not_above"}) == "stop"
    assert X.stage1_status({"a": "void"}) == "stop" and X.stage1_status({}) == "stop"
    assert X.stage1_status({"a": "above", "b": "void"}) == "pass"


def test_control_rules():
    assert X.near_one(1.2, 0.01, TH) == "near_1" and X.near_one(1.3, 0.01, TH) == "off"
    assert X.near_one(1.0, 0.1, TH) == "void" and X.near_one(math.nan, 0.01, TH) == "void"  # 3 s over the band
    assert X.write_conditioned({"cond_A": 16.5, "rel_err_mean": 0.33}, TH) and not X.write_conditioned({"cond_A": 17.0, "rel_err_mean": 0.1}, TH)
    assert not X.write_conditioned({"cond_A": 8.0, "rel_err_mean": 0.34}, TH) and not X.write_conditioned({"cond_A": math.nan, "rel_err_mean": 0.1}, TH)
    assert X.null_has_room(6144) and not X.null_has_room(6145)
    assert X.t_band(0.5, 0.01, TH) == "as_expected" and X.t_band(0.49, 0.01, TH) == "low"
    assert X.t_band(1.03, 0.01, TH) == "as_expected" and X.t_band(1.04, 0.01, TH) == "high" and X.t_band(1.0, math.nan, TH) == "void"
    assert X.above(1.3, 0.1, 1.0, 0.05, TH) == "not_above" and X.above(1.31, 0.1, 1.0, 0.05, TH) == "above"
    assert X.above(math.nan, 0.1, 1.0, 0.05, TH) == "void"


def test_blind_mark_reads_every_block_up_to_l_star():
    shares = {1: 0.1, 2: 0.8, 3: 0.2}
    assert X.blind_mark(shares, 3, TH) == "blind"  # the share at l* alone reads 0.2
    assert X.blind_mark(shares, 1, TH) == "readable"
    assert X.blind_mark({1: 0.5, 2: 0.0, 3: 0.0}, 1, TH) == "blind"  # a tie counts
    assert X.blind_mark(shares, None, TH) == "unknown" and X.blind_mark({1: None, 2: 0.1, 3: 0.1}, 2, TH) == "unknown"


def control_rec(T: float, s: float, status: str = "ok", fit: tuple[float, float] = (8.0, 0.16)) -> dict:
    row = {2: {"T": T, "s": s, "Q": 1.0, "s_Q": 0.01}}
    return {"row": row, "units": {"void": True}, "info": {"status": status, "fit": {"cond_A": fit[0], "rel_err_mean": fit[1]}, "share_in_U_top": 0.5}}


def test_control_verdicts_gate_amplify_and_read_sharpen_against_one():
    write, sharp = control_rec(0.9, 0.01), control_rec(1.5, 0.01)
    v = X.control_verdicts({"amplify": control_rec(1.5, 0.01), "sharpen": sharp}, write, 2, TH)
    assert v["amplify_T"] == "above" and v["sharpen_T"] == "above" and v["sharpen_Q"] == "near_1"
    assert X.control_verdicts({"amplify": control_rec(1.5, 0.01, "cap"), "sharpen": sharp}, write, 2, TH)["amplify_T"] == "void"
    bad = control_rec(0.9, 0.01, fit=(40.0, 0.16))
    v = X.control_verdicts({"amplify": control_rec(1.5, 0.01), "sharpen": sharp}, bad, 2, TH)
    assert v["amplify_T"] == "void" and not v["write_conditioned"]
    # above the write but not above 1: too weak, and Q has nothing to cancel
    v = X.control_verdicts({"amplify": control_rec(1.5, 0.01), "sharpen": control_rec(1.0, 0.01)}, write, 2, TH)
    assert v["sharpen_T"] == "not_above" and v["sharpen_Q"] == "void"
