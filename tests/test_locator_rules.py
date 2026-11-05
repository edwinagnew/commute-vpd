"""E97's filed rules against their edge cases (plan, "Filed located rule"): G <= 0 on either state, a part that raises the
state, a tie at the top of the carry map, a share past 1, a seed that disagrees, the do-nothing state; and stage 0's
plant check on an exact record. Pure; no model."""

from __future__ import annotations

import math

import torch

from commute import locator_rules as LR

TH = {"control_max": 0.2, "ratio": 2.0, "reread_tol": 1e-4, "zero_tol": 1e-3, "nominate_top": 3}
NAN = math.nan
A = 0.56


def seed(G: float, fall: float, cG: float = 15.0, cfall: float = 0.0, a: float = A) -> str:
    return LR.seed_reading({"G": G, "fall": fall}, {"G": cG, "fall": cfall}, a, TH)


def test_no_gain_on_either_state_is_unreadable():
    assert seed(0.0, 0.0) == LR.NO_GAIN
    assert seed(-1.0, 5.0) == LR.NO_GAIN
    assert seed(12.0, 8.0, cG=0.0) == LR.NO_GAIN
    assert seed(12.0, 8.0, cG=-0.3) == LR.NO_GAIN
    assert seed(NAN, 8.0) == LR.NO_GAIN


def test_the_do_nothing_state_cannot_pass():
    """W0 itself as the state: G = 0 exactly, whatever its fall (W0's own move is not the state's gain)."""
    assert seed(0.0, 1.2) == LR.NO_GAIN
    assert not LR.task_reading([seed(0.0, 1.2)] * 3)["located"]


def test_a_part_that_raises_the_state_is_not_located():
    assert seed(12.0, -0.5) == "not located: the part raises the state"
    assert seed(12.0, -0.5, cfall=-3.0) == "not located: the part raises the state"


def test_a_share_past_one_counts():
    assert seed(12.0, 14.0) == LR.LOCATED
    assert LR.frac(14.0, 12.0) > 1


def test_bars_and_ties():
    assert seed(10.0, 5.0, a=0.5) == LR.LOCATED  # a tie at a * G passes
    assert seed(10.0, 4.99, a=0.5) == "not located: the task falls under a"
    assert seed(10.0, 8.0, cG=10.0, cfall=2.0) == LR.LOCATED  # control tie at 0.2 * G, and 8 >= 2 * 2
    assert seed(10.0, 8.0, cG=10.0, cfall=2.01) == "not located: the control is not spared"
    assert seed(10.0, 6.0, cG=40.0, cfall=3.5) == "not located: under twice the control's fall"
    assert seed(10.0, 7.0, cG=40.0, cfall=3.5) == LR.LOCATED  # exactly twice


def test_a_degenerate_bar_reads_unreadable():
    assert seed(10.0, 8.0, a=NAN) == "unreadable: no bar a"
    assert seed(10.0, 8.0, a=0.0) == "unreadable: no bar a"
    assert seed(10.0, NAN) == "unreadable: not finite"


def test_a_seed_that_disagrees_is_not_located():
    r = LR.task_reading([LR.LOCATED, LR.LOCATED, "not located: the task falls under a"])
    assert not r["located"] and r["reading"] == "not located"
    assert LR.task_reading([LR.LOCATED, LR.NO_GAIN])["reading"] == LR.NO_GAIN
    assert LR.task_reading([LR.LOCATED] * 3)["located"]
    assert not LR.task_reading([])["located"]


def test_a_tie_at_the_top_names_no_part():
    assert LR.top_part({"2,4": 0.5, "1,1": 0.5, "mlp2": 0.1}) is None
    assert LR.top_part({"2,4": 1.3, "1,1": 0.5}) == "2,4"
    assert LR.top_part({"2,4": NAN}) is None


def test_a_from_the_control():
    assert LR.a_from([0.6531, 0.6302, 0.5649]) == 0.56
    assert LR.a_from([0.6, NAN, 0.7]) != LR.a_from([0.6, NAN, 0.7])  # nan: no bar
    assert math.isnan(LR.a_from([]))
    assert LR.a_from([-0.051]) == -0.06  # the caller stops on a <= 0
    assert LR.a_from([1.239]) == 1.23


def test_best_located_takes_the_strongest_weakest_seed():
    recs = {"2,4": {"located": True, "shares": [0.6, 0.58]}, "1,1": {"located": True, "shares": [0.9, 0.57]}, "mlp2": {"located": False, "shares": [0.99, 0.99]}}
    assert LR.best_located(recs) == "2,4"
    assert LR.best_located({"mlp2": {"located": False, "shares": [0.9]}}) is None


def test_row_se_clusters_by_row():
    x = torch.tensor([1.0, 1.0, 3.0, 3.0])
    se = LR.row_se(x, torch.tensor([0, 0, 1, 1]))
    assert abs(se - math.sqrt(2 * (2.0**2 + 2.0**2)) / 4) < 1e-12
    assert math.isnan(LR.row_se(x, torch.zeros(4, dtype=torch.long)))  # one row
    assert math.isnan(LR.row_se(torch.tensor([1.0, NAN]), torch.tensor([0, 1])))


def test_mean_probability_is_not_the_probability_at_the_mean():
    lo = torch.tensor([-8.0, 2.0])
    assert abs(LR.mean_prob(lo) - (1 / (1 + math.exp(8)) + 1 / (1 + math.exp(-2))) / 2) < 1e-12
    assert LR.mean_prob(lo) > 1 / (1 + math.exp(3))


def _exact_plant(block: int) -> dict:
    keys = [f"h.{b}.{m}" for b in range(4) for m in ("attn.k_proj", "attn.o_proj", "attn.q_proj", "attn.v_proj", "mlp.c_fc", "mlp.down_proj")]
    parts = [f"{b},{h}" for b in range(4) for h in range(6)] + [f"mlp{b}" for b in range(4)]
    edited = f"h.{block}.mlp.down_proj"
    carry = {n: {"lo": 1.0 if n == f"mlp{block}" else (0.0 if LR.upstream(n, block) else 0.3)} for n in parts}
    return {"carry": carry, "write": {"matrix": {k: {"lo": 1.0 if k == edited else 0.0} for k in keys}, "head": {n: {"lo": 0.0} for n in parts if "," in n},
                                      "mlp": {f"mlp{b}": {"lo": 1.0 if b == block else 0.0} for b in range(4)}, "G_all": {"lo": 0.0}}}  # fmt: skip


def test_an_exact_plant_passes_and_a_wrong_read_misses():
    for block in (1, 3):
        rec = _exact_plant(block)
        assert LR.plant_misses(rec, f"h.{block}.mlp.down_proj", block, TH) == []
    rec = _exact_plant(1)
    rec["carry"]["1,3"]["lo"] = 2e-4  # upstream of the block-1 MLP: must read 0
    rec["write"]["matrix"]["h.0.mlp.c_fc"]["lo"] = NAN
    rec["write"]["G_all"]["lo"] = 0.01
    assert len(LR.plant_misses(rec, "h.1.mlp.down_proj", 1, TH)) == 3


def test_inheritance_reads_only_downstream_parts():
    rec = _exact_plant(1)
    assert LR.inheritance(rec["carry"], 1) == 0.3
    assert math.isnan(LR.inheritance({"mlp3": {"lo": 1.0}}, 3))
    assert LR.upstream("1,5", 1) and not LR.upstream("mlp1", 1) and not LR.upstream("2,0", 1) and LR.upstream("mlp0", 1)


def test_can_nominate_and_the_hand_write():
    assert LR.can_nominate(0.2, [0.3, 0.25])
    assert LR.can_nominate(0.25, [0.3, 0.25])  # a tie can nominate
    assert not LR.can_nominate(0.31, [0.3, 0.5])
    assert not LR.can_nominate(NAN, [0.3]) and not LR.can_nominate(0.1, [])
    assert LR.write_within(-0.01, [0.0, -0.004])
    assert not LR.write_within(0.02, [0.0, -0.004])
    assert not LR.write_within(NAN, [0.0]) and not LR.write_within(0.0, [])


def test_nominees_and_the_control():
    rec = {"carry": {"2,4": {"lo": 0.9}, "1,1": {"lo": 0.2}, "mlp2": {"lo": 0.5}, "0,0": {"lo": 0.1}},
           "write": {"head": {"2,4": {"lo": 0.4}, "3,1": {"lo": 0.3}}, "matrix": {"h.2.mlp.down_proj": {"lo": 0.6}, "h.2.attn.v_proj": {"lo": 0.4}}},
           "pair": "1,1+2,4"}  # fmt: skip
    assert LR.nominees(rec, 3) == ["2,4", "mlp2", "1,1", "3,1", "1,1+2,4"]
    assert LR.control_passes(["2,4"] * 3, "2,4")
    assert not LR.control_passes(["2,4", None, "2,4"], "2,4") and not LR.control_passes([], "2,4")
    assert LR.zero_ok([0.0, 1e-3], 1e-3) and not LR.zero_ok([NAN], 1e-3) and not LR.zero_ok([], 1e-3)
