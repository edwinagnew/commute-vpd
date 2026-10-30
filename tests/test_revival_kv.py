"""CPU checks of E55 C3's filed rules (src/commute/revival_kv.py; plan reports/plans/2026-10-31_plan_e55-stage-c-open-pair.md
§ 6 and the edge cases it lists; experiments/open_pair_instruments/results/predictions.json): the plan's worked examples
on E58's three-seed columns, empty pools, zero denominators, ties, censoring, the distance and prior rules, the joint
verdict, and the per-state arithmetic (revival, half-life, the planted bisection, the planted units and pass of Amendment 1, MLP share). No model is loaded."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from commute import revival_kv as KV

PRED = Path(__file__).resolve().parents[1] / "experiments/open_pair_instruments/results/predictions.json"
C = json.loads(PRED.read_text())["constants"]

# E58 run.json / amend1.json, seeds 0-2 (rank1, cipher, elicit, copy)
E58 = {
    "r": {"rank1": [0.8446171948548371, 0.8427860331443, 0.8429197716575669], "cipher": [0.23632404751112837, 0.2384323212900442, 0.2170782048159956],
          "elicit": [0.786571788693418, 0.7574514568065235, 0.758322649009162], "copy": [0.8461436225319128, 0.8425401825652695, 0.834231371851741]},
    "ratio": {"rank1": [1.1099375257123278, 1.104947015648684, 1.0987862717128636], "cipher": [0.33003492996874556, 0.3359939898305597, 0.30162258558892613],
              "elicit": [0.9456599976570258, 0.9342370566744104, 0.9335208764123075], "copy": [1.057353380337618, 1.0025782868955995, 1.006636674933835]},
    "S": {"rank1": [0.250511123391638, 0.2130522845640163, 0.20860088993916454], "cipher": [0.17073141481239898, 0.21955900501916986, 0.15195429694604587],
          "elicit": [0.5013879509451297, 0.4982165819337943, 0.48984774901124867], "copy": [0.6122354543181754, 0.680477808432353, 0.6334360054728233]},
    "x": {"rank1": [0.138344279946636, 0.09667174029995676, 0.09025277545943194], "cipher": [-0.3034251577682921, -0.37662193522867776, -0.34184860643610304],
          "elicit": [-0.027335405895011152, -0.03777784683631455, -0.025768889167846507], "copy": [0.42013033358384044, 0.552566794331367, 0.46977683993110325]},
    "norm": {"rank1": [3.1788571929450042, 3.1000853564853523, 3.152234207884463], "cipher": [3.260403643996566, 3.1937328217256566, 3.0733713750816896],
             "elicit": [3.1691567202824067, 3.119621581591776, 3.1833492091201268], "copy": [3.1113881299202255, 3.1339297183117933, 3.154575859025749]},
}  # fmt: skip


def reading(name: str, direction: int | None = 1) -> dict:
    return KV.column_reading(E58[name], direction, 0.0, C)


# ----------------------------------------------------------------------------- the plan's worked examples (§ 6)


def test_worked_r_reads_vpd():
    r = reading("r")
    assert r["status"] == "ok" and r["gap"] == pytest.approx(0.613, abs=1e-3) and 3 * r["noise"] == pytest.approx(0.050, abs=1e-3)
    assert r["p"] == pytest.approx(0.876, abs=1e-3) and r["p_copy"] == pytest.approx(0.996, abs=1e-3)
    assert r["p_lo"] == pytest.approx(0.859, abs=0.01) and r["p_hi"] == pytest.approx(0.894, abs=0.01)
    assert r["side"] == "VPD"


def test_worked_ratio_reads_vpd():
    r = reading("ratio")
    assert r["status"] == "ok" and r["gap"] == pytest.approx(0.782, abs=1e-3) and r["p"] == pytest.approx(0.787, abs=1e-3)
    assert r["side"] == "VPD"


def test_worked_steering_has_no_gap():
    r = reading("S")
    assert r["status"] == "no gap" and r["gap"] == pytest.approx(0.043, abs=1e-3) and 3 * r["noise"] == pytest.approx(0.105, abs=1e-3)
    assert r["side"] is None


def test_worked_carry_straddles_two_thirds_and_reads_between():
    r = reading("x")
    assert r["status"] == "ok" and r["p"] == pytest.approx(0.691, abs=1e-3) and r["p_copy"] == pytest.approx(1.83, abs=0.01)
    assert r["p_lo"] < 2 / 3 < r["p_hi"] and r["side"] == "between"


def test_worked_update_size_undirected_no_gap():
    r = reading("norm", None)
    assert r["status"] == "no gap" and abs(r["gap"]) == pytest.approx(0.032, abs=1e-3) and 3 * r["noise"] == pytest.approx(0.284, abs=1e-3)


# ----------------------------------------------------------------------------- edge cases


def test_empty_pool_voids_the_column():
    col = {**E58["r"], "elicit": [0.78, None]}
    assert KV.column_reading(col, 1, 0.0, C)["status"] == "void"
    assert KV.column_reading({**E58["r"], "cipher": []}, 1, 0.0, C)["status"] == "void"
    # copy alone short: the column stands, p_copy is nan
    r = KV.column_reading({**E58["r"], "copy": [0.8]}, 1, 0.0, C)
    assert r["status"] == "ok" and math.isnan(r["p_copy"])


def test_zero_denominator_gives_no_gap_and_outside_draws():
    col = {"rank1": [1.0, 1.0], "cipher": [1.0, 1.0], "elicit": [0.5, 0.7], "copy": [1.0, 1.0]}
    r = KV.column_reading(col, 1, 0.0, C)
    assert r["status"] == "no gap" and math.isnan(r["p"]) and r["side"] is None
    boot = KV.bootstrap_p(col, 100, 0)
    assert np.isnan(boot).all()
    assert KV.percentiles(boot, 0.9) == (-math.inf, math.inf)
    assert KV.column_reading(col, None, 0.0, C)["status"] == "no gap"  # undirected too, even with noise 0


def test_zero_denominator_draws_count_outside_every_band():
    # rank-1 and cipher overlap in one seed each: some draws have a zero denominator
    col = {"rank1": [1.0, 2.0], "cipher": [1.0, 0.0], "elicit": [1.0, 1.9], "copy": []}
    boot = KV.bootstrap_p(col, 2000, 0)
    frac_nan = float(np.isnan(boot).mean())
    assert 0.05 < frac_nan < 0.5
    lo, hi = KV.percentiles(boot, 0.9)
    assert lo == -math.inf and hi == math.inf  # more than 5% outside: neither tail can sit in a band


def test_ties_go_to_the_outer_band_and_out_of_range_p():
    b = C["bands"]
    assert KV.side(b["vpd_min"], b["vpd_min"], b["label_max"]) == "VPD"
    assert KV.side(b["label_max"], b["vpd_min"], b["label_max"]) == "label"
    assert KV.side(1.7, b["vpd_min"], b["label_max"]) == "VPD"
    assert KV.side(-0.4, b["vpd_min"], b["label_max"]) == "label"
    assert KV.side(0.5, b["vpd_min"], b["label_max"]) == "between"
    assert KV.side(math.nan, b["vpd_min"], b["label_max"]) is None


def test_firm_needs_the_whole_central_interval_in_the_band():
    assert KV.firm("VPD", 0.70, 0.9, 2 / 3, 1 / 3) == "VPD"
    assert KV.firm("VPD", 0.60, 0.9, 2 / 3, 1 / 3) == "between"
    assert KV.firm("label", 0.1, 1 / 3, 2 / 3, 1 / 3) == "label"
    assert KV.firm("label", 0.1, 0.4, 2 / 3, 1 / 3) == "between"
    assert KV.firm("between", 0.1, 0.9, 2 / 3, 1 / 3) == "between"


def test_gap_reversed_and_direction_of_t_half():
    col = {"rank1": [10.0, 11.0, 12.0], "cipher": [50.0, 52.0, 51.0], "elicit": [20.0, 21.0, 19.0]}
    assert KV.gap_status(col, -1, 1.0, 3)["status"] == "ok"  # t_half: rank-1 faster is the filed direction
    assert KV.gap_status(col, 1, 1.0, 3)["status"] == "reversed"
    assert KV.gap_status(col, None, 1.0, 3)["status"] == "ok"
    # the floor applies: noise 15 makes the gap of 40 too small
    assert KV.gap_status(col, -1, 15.0, 3)["status"] == "no gap"
    # equality counts
    eq = {"rank1": [0.0, 0.0], "cipher": [3.0, 3.0], "elicit": [1.0, 1.0]}
    assert KV.gap_status(eq, -1, 1.0, 3)["status"] == "ok"


def test_censoring():
    col, cens = KV.t_half_column({"rank1": [5.0, "censored"], "cipher": [50.0, 60.0], "elicit": [7.0, 8.0]})
    assert cens and col["rank1"] == [5.0, None]
    col, cens = KV.t_half_column({"rank1": [5.0, 6.0], "cipher": [50.0, 60.0], "elicit": ["censored", 8.0, None], "copy": ["censored"]})
    assert not cens and col["elicit"] == [None, 8.0, None] and col["copy"] == [None]


# ----------------------------------------------------------------------------- distance, prior, verdicts


def fake(status: str, p: float, side: str | None) -> dict:
    return {"status": status, "p": p, "side": side}


def test_distance_and_prior_rules():
    inst = fake("ok", 0.9, "VPD")
    no = fake("no gap", 0.9, None)
    assert KV.verdict(inst, no, no, C) == "VPD"  # neither baseline has a gap: rules 5 and 6 do not fire
    assert KV.verdict(inst, fake("ok", 0.85, "between"), no, C) == "distance"  # within 0.1
    assert KV.verdict(inst, fake("ok", 0.75, "between"), no, C) == "VPD"  # 0.15 away, other side
    assert KV.verdict(inst, fake("ok", 0.2, "VPD"), no, C) == "distance"  # same firm side
    assert KV.verdict(inst, no, fake("ok", 0.8, "VPD"), C) == "prior"  # p_prior firm VPD turns VPD into prior
    assert KV.verdict(inst, fake("ok", 0.9, "VPD"), fake("ok", 0.8, "VPD"), C) == "distance and prior"
    assert KV.verdict(fake("ok", 0.9, "VPD"), no, fake("ok", 0.8, "between"), C) == "prior"  # 0.1 exactly counts
    assert KV.verdict(fake("ok", 0.876, "VPD"), no, fake("ok", 0.65, "between"), C) == "VPD"  # the plan's expected case
    assert KV.verdict(fake("no gap", 0.9, None), fake("ok", 0.9, "VPD"), no, C) == "no gap"
    assert KV.verdict(fake("censored", math.nan, None), no, no, C) == "censored"


@pytest.mark.parametrize(
    ("a", "b", "joint"),
    [
        ("VPD", "VPD", "shared"),
        ("VPD", "no gap", "shared (by pruning)"),
        ("void", "VPD", "shared (by reverse)"),
        ("VPD", "censored", "shared (by pruning)"),
        ("label", "label", "VPD alone"),
        ("censored", "label", "VPD alone (by reverse)"),
        ("VPD", "label", "conflict"),
        ("reversed", "VPD", "conflict"),
        ("VPD", "between", "not settled (pruning: VPD; reverse: between)"),
        ("distance", "VPD", "not settled (pruning: distance; reverse: VPD)"),
        ("label", "reversed", "not settled (pruning: label; reverse: reversed)"),
        ("no gap", "void", "not settled (pruning: no gap; reverse: void)"),
    ],
)
def test_joint_verdict(a: str, b: str, joint: str):
    assert KV.joint_verdict({"pruning": a, "reverse": b}) == joint


def test_score_columns_end_to_end():
    seeds = 8
    rng = np.random.default_rng(1)
    mk = lambda mu, s: list(mu + s * rng.standard_normal(seeds))
    cols = {"R": {"rank1": mk(0.5, 0.02), "cipher": mk(0.1, 0.02), "elicit": mk(0.45, 0.02), "copy": mk(0.3, 0.02)},
            "t_half": {"rank1": mk(20, 1), "cipher": mk(80, 1), "elicit": mk(25, 1), "copy": mk(40, 1)},
            "D": {"rank1": mk(2.0, 0.3), "cipher": mk(3.0, 0.3), "elicit": mk(2.9, 0.3), "copy": mk(3.0, 0.3)},
            "prior": {"rank1": [-2.5] * seeds, "cipher": [-13.7] * seeds, "elicit": [-6.4] * seeds, "copy": [-9.7] * seeds},
            "r": E58["r"], "ratio": E58["ratio"]}  # fmt: skip
    sc = KV.score_columns(cols, {"R": 0.01, "t_half": 1.0}, {}, C)
    assert sc["readings"]["prior"]["status"] == "ok" and sc["readings"]["prior"]["side"] == "between"
    assert sc["readings"]["prior"]["p"] == pytest.approx((-6.4 + 13.7) / (-2.5 + 13.7))
    assert sc["verdicts"]["R"] == "VPD" and sc["verdicts"]["t_half"] == "VPD" and sc["joint"] == "shared"
    assert sc["verdicts"]["r"] == "VPD" and sc["verdicts"]["ratio"] == "VPD"
    sc = KV.score_columns(cols, {"R": 0.01, "t_half": 1.0}, {"t_half": "censored", "R": "void"}, C)
    assert sc["verdicts"]["t_half"] == "censored" and sc["verdicts"]["R"] == "void" and sc["joint"].startswith("not settled")


def test_probe_reference():
    ok = KV.probe_reference({"rank1": 3.0, "cipher": 0.2, "elicit": 1.5, "copy": 4.0})
    assert ok["all"] and ok["p_q"] == pytest.approx((1.5 - 0.2) / 2.8)
    assert not KV.probe_reference({"rank1": 3.0, "cipher": 0.6, "elicit": 1.5, "copy": 4.0})["all"]


# ----------------------------------------------------------------------------- per-state arithmetic


def test_revival_and_spread():
    rec = {"kappa": 2.0, "13": {"top": 1.0, "random": [1.9, 2.0, 2.1]}, "40": {"top": 0.5, "random": [1.8, 1.8, 1.8]}}
    rv = KV.revival(rec, [13, 40], 2.0)
    assert rv["Delta"]["13"] == pytest.approx(1.0) and rv["Delta"]["40"] == pytest.approx(1.5 - 0.2)
    assert rv["R"] == pytest.approx(1.3 / 2.0)
    assert rv["spread"]["13"] == pytest.approx(0.05) and rv["spread"]["40"] == pytest.approx(0.0)
    assert math.isnan(KV.revival(rec, [13, 40], 0.0)["R"])
    random_only = {"kappa": 2.0, "13": {"random": [1.9, 2.0, 2.1]}}  # job 1: no ranking read
    rv = KV.revival(random_only, [13], 2.0)
    assert math.isnan(rv["R"]) and rv["spread"]["13"] == pytest.approx(0.05)


def test_half_life_interpolates_and_censors():
    assert KV.half_life([2.0, 1.5, 1.2, 0.8]) == pytest.approx(2 + (1.2 - 1.0) / (1.2 - 0.8))
    assert KV.half_life([2.0, 1.0]) == pytest.approx(1.0)  # exactly D / 2
    assert KV.half_life([2.0, 1.9, 1.8]) is None  # censored
    assert KV.half_life([0.0]) is None and KV.half_life([]) is None
    assert KV.half_life([2.0, 0.2]) == pytest.approx((2.0 - 1.0) / 1.8)


def test_fit_alpha_reaches_the_target_or_falls_back():
    f = KV.fit_alpha(lambda a: a, lambda a: a / 10, target=3.0, bound=1.0)
    assert f["alpha"] == pytest.approx(3.0, rel=1e-6) and not f["fallback"] and f["kappa"] >= 3.0
    f = KV.fit_alpha(lambda a: a, lambda a: a / 10, target=3.0, bound=0.2)
    assert f["alpha"] == pytest.approx(2.0, rel=1e-6) and f["fallback"] and f["off"] <= 0.2
    f = KV.fit_alpha(lambda a: min(a, 1.0), lambda a: 0.0, target=3.0, bound=1.0, max_doublings=5)
    assert f["fallback"] and f["kappa"] == 1.0  # unreachable within the doublings


def test_mlp_share_and_planted_pass():
    dW = {"h.0.attn.q_proj": torch.ones(2, 2), "h.0.mlp.c_fc": 2 * torch.ones(2, 2), "h.0.mlp.down_proj": torch.zeros(2, 2)}
    assert KV.mlp_share(dW) == pytest.approx(16 / 20)
    assert math.isnan(KV.mlp_share({"h.0.mlp.c_fc": torch.zeros(1)}))
    p = KV.planted_pass(list(range(13)), list(range(3, 16)), 0.7, 0.8, 10, 0.8, D=5.0, floor=3.5)
    assert p["found"] == 10 and p["pass"] and p["R_min"] == pytest.approx(0.64) and p["reading"] == "pass"
    assert KV.planted_pass(list(range(13)), list(range(4, 17)), 0.7, 0.8, 10, 0.8, D=5.0, floor=3.5)["reading"] == "fail"
    assert KV.planted_pass(list(range(13)), list(range(13)), math.nan, 0.8, 10, 0.8, D=5.0, floor=3.5)["reading"] == "fail"
    # Amendment 1: under the floor the control is void (not built), even when it would pass; at the floor it is built
    v = KV.planted_pass(list(range(13)), list(range(13)), 0.9, 0.9, 10, 0.8, D=0.184, floor=3.5)
    assert v["reading"] == "void" and not v["pass"] and not v["built"]
    assert KV.planted_pass(list(range(13)), list(range(13)), 0.9, 0.9, 10, 0.8, D=3.5, floor=3.5)["reading"] == "pass"


def test_selective_units_amendment_1():
    """Answer-selective units: only above the q-quantile of a_ans and above 0; the largest a_ans / a_else first; ties
    to the lower id; a zero a_else ranks first; a negative quantile still keeps a_ans <= 0 out."""
    a_ans = torch.tensor([0.1, 2.0, 3.0, 4.0, 4.0, 1.0, 5.0, 0.0, 0.0, 6.0], dtype=torch.float64)
    a_else = torch.tensor([0.01, 1.0, 1.0, 2.0, 2.0, 0.1, 5.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    # 0.5-quantile of a_ans = 2.5: candidates 2 (sel 3), 3 (2), 4 (2), 6 (1), 9 (inf)
    assert KV.selective_units(a_ans, a_else, 4, 0.5) == [9, 2, 3, 4]
    assert KV.selective_units(a_ans, a_else, 10, 0.5) == [9, 2, 3, 4, 6]  # fewer candidates than n: planted_units asserts
    neg = torch.tensor([-1.0, -0.5, -0.2, 0.0, 0.3], dtype=torch.float64)
    assert KV.selective_units(neg, torch.ones(5, dtype=torch.float64), 5, 0.1) == [4]
    A = json.loads(PRED.read_text())["amendment_1"]["planted"]
    assert 0 < A["active_quantile"] < 1 and 0 < A["D_floor_frac"] <= 1


def test_zero_units_zeroes_exactly_the_named_units():
    edit = KV.zero_units([5, 3072 + 7, 3 * 3072 + 3071])
    assert sorted(edit) == ["mlp0", "mlp1", "mlp3"]
    x = torch.randn(2, 3, 3072)
    y = edit["mlp1"](x)
    assert (y[..., 7] == 0).all() and torch.equal(y[..., :7], x[..., :7]) and torch.equal(y[..., 8:], x[..., 8:])
    assert (edit["mlp3"](x)[..., 3071] == 0).all()
