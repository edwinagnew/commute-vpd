"""E151's filed rule: every edge case listed in experiments/disagreeing_pairs/results/predictions.json
(edge_cases_checked_before_filing), one test each."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from commute import steering as ST
from commute.data import Windows
from commute.jobs import load_experiment

R = load_experiment("disagreeing_pairs", "rules")
PRED = Path(__file__).resolve().parents[1] / "experiments/disagreeing_pairs/results/predictions.json"
TH = json.loads(PRED.read_text())["thresholds"]
A, B, D = R.COUNTED
NAN = math.nan

# seed triples: a side at x with spread 0.02
def side(x: float) -> list[float]:
    return [x - 0.01, x, x + 0.01]


STEER_UP = (side(0.56), side(0.10))  # steering separates, reuse higher
STEER_DOWN = (side(0.13), side(0.70))  # steering separates, new higher
STEER_FLAT = (side(0.20), side(0.18))  # gap 0.02: does not


def pair(prior_gap: float, steering: tuple[list[float], list[float]], share: tuple[list[float], list[float]]) -> dict:
    return {"prior_gap": prior_gap, "steering": steering, "share": share}


def test_prior_gap_exactly_at_bar_is_a_call():
    assert R.prior_call(0.5, TH["prior_bar"]) == 1
    assert R.prior_call(-0.5, TH["prior_bar"]) == -1
    # at the bar the pair can be opposite, never flat
    assert R.pair_calls(pair(-0.5, STEER_UP, STEER_UP), TH)["condition"] == "opposite"
    assert R.pair_calls(pair(0.5, STEER_FLAT, STEER_UP), TH)["condition"] == "condition not met"


def test_prior_gap_just_under_bar_is_matched():
    assert R.prior_call(0.4999, TH["prior_bar"]) == 0
    assert R.pair_calls(pair(0.4999, STEER_FLAT, STEER_UP), TH)["condition"] == "flat"
    # the positive control's case: prior matched, steering separates
    assert R.pair_calls(pair(0.2, STEER_UP, STEER_UP), TH)["condition"] == "condition not met"


def test_steering_gap_equal_to_range_does_not_separate():
    steer = ([0.30, 0.35, 0.40], [0.20, 0.25, 0.30])  # gap 0.1, ranges 0.1: needs |gap| > range
    assert R.CALL(*steer, TH)["call"] == "does not"
    assert R.pair_calls(pair(0.0, steer, steer), TH)["condition"] == "flat"
    assert R.pair_calls(pair(-1.2, steer, steer), TH)["condition"] == "condition not met"


def test_inert_steering_counts_as_not_separating():
    steer = ([0.01, 0.02, 0.03], [0.0, 0.01, 0.04])
    assert R.CALL(*steer, TH)["call"] == "inert"
    assert R.pair_calls(pair(0.1, steer, steer), TH)["condition"] == "flat"
    assert R.pair_calls(pair(1.0, steer, steer), TH)["condition"] == "condition not met"


def test_prior_and_steering_same_way_not_met():
    assert R.pair_calls(pair(1.0, STEER_UP, STEER_UP), TH)["condition"] == "condition not met"
    assert R.pair_calls(pair(-1.0, STEER_DOWN, STEER_UP), TH)["condition"] == "condition not met"


def test_share_inert_on_opposite_is_neither():
    s = R.score({A: pair(-1.2, STEER_UP, ([0.01, 0.02, 0.0], [0.0, 0.01, 0.02]))}, TH)
    assert s["pairs"][A]["reading"] == "neither"
    assert s["outcome"] == "neither"


def test_share_gain_under_floor_is_unreadable_and_left_out():
    share_nan = ([0.77, NAN, 0.78], [0.5, 0.5, 0.5])  # one seed nan: 2 finite but a nan among them -> unreadable
    share_one = ([0.77, NAN, NAN], [0.5, 0.5, 0.5])
    assert R.CALL(*share_one, TH)["call"] == "unreadable"
    s = R.score({A: pair(-1.2, STEER_UP, share_nan), D: pair(3.5, STEER_DOWN, (side(0.86), side(0.3)))}, TH)
    assert s["pairs"][A]["reading"] == "unreadable"
    assert s["counted_opposite"] == [D]
    assert s["outcome"] == "prior"  # one pair left: its reading
    s = R.score({A: pair(-1.2, STEER_UP, share_one)}, TH)
    assert s["outcome"] == "no call"


def test_one_prior_one_steering_is_split():
    s = R.score({A: pair(-1.2, STEER_UP, (side(0.77), side(0.40))), D: pair(3.5, STEER_DOWN, (side(0.86), side(0.30)))}, TH)
    assert s["pairs"][A]["reading"] == "follows steering"
    assert s["pairs"][D]["reading"] == "follows prior"
    assert s["outcome"] == "split"


def test_one_prior_one_neither_is_split():
    s = R.score({A: pair(-1.2, STEER_UP, (side(0.50), side(0.49))), D: pair(3.5, STEER_DOWN, (side(0.86), side(0.30)))}, TH)
    assert s["pairs"][A]["reading"] == "neither"
    assert s["outcome"] == "split"


def test_both_follow_the_same_reading():
    s = R.score({A: pair(-1.2, STEER_UP, (side(0.40), side(0.77))), D: pair(3.5, STEER_DOWN, (side(0.86), side(0.30)))}, TH)
    assert s["outcome"] == "prior"
    s = R.score({A: pair(-1.2, STEER_UP, (side(0.77), side(0.40))), D: pair(3.5, STEER_DOWN, (side(0.30), side(0.86)))}, TH)
    assert s["outcome"] == "steering"


def test_A_fails_B_and_D_meet_run_continues_outcome_is_D_alone():
    conds = {A: "condition not met", B: "flat", D: "opposite"}
    assert R.n_meeting(conds) == 2 >= TH["min_counted_pairs"]
    assert R.n_meeting({A: "condition not met", B: "condition not met", D: "opposite"}) < TH["min_counted_pairs"]
    s = R.score({A: pair(1.0, STEER_UP, STEER_UP), B: pair(0.1, STEER_FLAT, (side(0.73), side(0.40))),
                 D: pair(3.5, STEER_DOWN, (side(0.86), side(0.30)))}, TH)  # fmt: skip
    assert s["pairs"][A]["reading"] == "not counted"
    assert s["counted_opposite"] == [D] and s["outcome"] == "prior"
    assert s["flat"] == {B: "separates, reuse side higher"}


def test_A_meeting_flat_is_counted_as_flat_outside_outcome():
    s = R.score({A: pair(0.2, STEER_FLAT, (side(0.77), side(0.62)))}, TH)
    assert s["pairs"][A]["condition"] == "flat"
    assert s["flat"] == {A: "separates, reuse side higher"}
    assert s["outcome"] == "no call"


def test_flat_share_readings():
    for share, want in (((side(0.4), side(0.7)), "separates, new side higher"), ((side(0.5), side(0.49)), "does not"),
                        (([0.01, 0.0, 0.02], [0.0, 0.01, 0.0]), "does not")):  # fmt: skip
        assert R.score({B: pair(0.1, STEER_FLAT, share)}, TH)["pairs"][B]["reading"] == want


def test_do_nothing_pair_never_counted():
    zero = ([NAN, NAN, NAN], [NAN, NAN, NAN])  # gain 0 on every state: steering and share nan
    s = R.score({A: pair(-1.2, zero, zero), D: pair(3.5, zero, zero)}, TH)
    assert all(s["pairs"][p]["condition"] == "condition not met" for p in (A, D))
    assert s["outcome"] == "no call"


def test_pair_C_never_counted():
    c = "C_rank1_classshuffled"
    s = R.score({c: pair(4.58, STEER_DOWN, (side(0.86), side(0.28)))}, TH)
    assert s["pairs"][c]["reading"] == "follows prior"
    assert s["counted_opposite"] == [] and s["outcome"] == "no call"


def test_filed_predictions_on_the_leaned_case():
    s = R.score({A: pair(-1.2, STEER_UP, (side(0.77), side(0.62))), B: pair(0.1, STEER_FLAT, (side(0.73), side(0.62))),
                 D: pair(3.5, STEER_DOWN, (side(0.86), side(0.30)))}, TH)  # fmt: skip
    assert R.filed(s) == {"P1": True, "P2": True, "P3": True, "P4": True, "P5": True}


# ----------------------------------------------------------------------------- Amendment 1 (results/predictions_amend1.json)

A1 = json.loads((PRED.parent / "predictions_amend1.json").read_text())["ladder"]
RANKS, RATIOS = (10, 20), (3.0, 10.0)


def toy_G() -> torch.Tensor:
    """4 keys x 40 tokens, the same at every key: tokens 0-9 at 0.1 (the top 10), 10-19 at about 2e-3 (ranks 11-20),
    20-39 at about 1e-4 (ranks 21+)."""
    row = [0.1] * 10 + [2e-3 - j * 1e-5 for j in range(10)] + [1e-4 - j * 1e-6 for j in range(20)]
    return torch.tensor([row] * 4, dtype=torch.float64)


def rungs_of(G: torch.Tensor, pool: list[int]) -> list[dict]:
    f = R.filters(G, pool, RANKS, RATIOS, A1["value_max"])
    out = []
    for rg in A1["rungs"]:
        ok = R.rung_ok(f, rg["min_rank"], rg["median_ratio"], A1["value_max"])
        cols = R.assign(G[:, pool], ok)
        done = [(math.log(float(G[i, pool[c]])), 1) for i, c in enumerate(cols) if c is not None]
        out.append({"complete": all(c is not None for c in cols), "prior_est": R.est_prior(*zip(*done)) if done else NAN,
                    "values": [pool[c] if c is not None else None for c in cols]})  # fmt: skip
    return out


def test_amend1_rung2_table_when_rung1_is_incomplete():
    rungs = rungs_of(toy_G(), [10, 11, 12, 13, 30, 31])  # past rank 20: only 30 and 31, two words for four keys
    assert not rungs[0]["complete"] and rungs[1]["complete"]
    assert rungs[1]["values"] == [10, 11, 12, 13]  # highest P first; ranks 11-14, never the top 10
    assert R.take_rung(rungs, A1["common_prior_min"]) == 1


def test_amend1_rung3_table_by_the_median_ratio():
    G = toy_G()
    G[0, 30:34] = 9e-4  # key 0's own words: 9x their median over keys (1e-4 at the other three)
    G[1:, 30:34] = 1e-4
    f = R.filters(G, [30, 31, 32, 33], RANKS, RATIOS, A1["value_max"])
    assert not bool(f[R.ratio_name(3.0)][0].any()) and bool(f[R.ratio_name(10.0)][0].all())
    rungs = [{"complete": True, "prior_est": 1e-4}, {"complete": False, "prior_est": 1e-3}, {"complete": True, "prior_est": 3e-4}]
    assert R.take_rung(rungs, A1["common_prior_min"]) == 2  # rung 1 under the bar, rung 2 incomplete


def test_amend1_no_rung_qualifies_stops():
    assert R.take_rung([{"complete": False, "prior_est": 1e-3}] * 3, A1["common_prior_min"]) is None
    assert R.take_rung([{"complete": True, "prior_est": 2.3e-4}, {"complete": True, "prior_est": NAN}], A1["common_prior_min"]) is None
    rungs = rungs_of(toy_G(), [0, 1, 2, 3, 30])  # the top 10 never admissible: one word left past rank 10
    assert not any(r["complete"] for r in rungs) and R.take_rung(rungs, A1["common_prior_min"]) is None


def test_amend1_fewest_candidates_first():
    P = torch.tensor([[0.9, 0.1], [0.5, 0.0]], dtype=torch.float64)
    assert R.assign(P, torch.tensor([[True, True], [True, False]])) == [1, 0]  # key 1 (one candidate) goes first
    assert R.assign(P, torch.tensor([[True, False], [True, False]])) == [0, None]  # one word, two keys: incomplete


def test_amend1_est_prior_is_the_window_weighted_geometric_mean():
    assert math.isclose(R.est_prior([math.log(1e-3), math.log(1e-4)], [1, 1]), math.sqrt(1e-7))
    assert math.isclose(R.est_prior([math.log(1e-3), math.log(1e-4)], [3, 1]), 10 ** (-3.25))
    assert math.isnan(R.est_prior([], []))


def test_amend1_B_under_the_bar_is_condition_not_met_not_a_stop():
    assert not R.b_premise(3e-4, A1["b_prior_min"]) and R.b_premise(4.2e-4, A1["b_prior_min"])
    b = {**pair(0.1, STEER_FLAT, (side(0.73), side(0.40))), "premise": R.b_premise(3e-4, A1["b_prior_min"])}
    s = R.score({A: pair(-0.6, STEER_UP, (side(0.77), side(0.62))), B: b, D: pair(3.5, STEER_DOWN, (side(0.86), side(0.30)))}, TH)
    assert s["pairs"][B]["condition"] == "condition not met" and s["pairs"][B]["reading"] == "not counted"
    assert s["flat"] == {} and s["outcome"] == "split"
    assert R.n_meeting({p: r["condition"] for p, r in s["pairs"].items()}) == 2 >= TH["min_counted_pairs"]  # the run goes on
    b["premise"] = True
    assert R.score({B: b}, TH)["pairs"][B]["condition"] == "flat"


def test_amend1_do_nothing_pair_never_counted():
    zero = ([NAN, NAN, NAN], [NAN, NAN, NAN])
    for premise in (True, False):
        assert R.pair_calls({**pair(0.1, zero, zero), "premise": premise}, TH)["condition"] == "condition not met"


def test_amend1_window_prior_edge_cases():
    LP = torch.log(torch.tensor([[0.5, 1e-3, 1e-4], [0.5, 1e-3, 1e-4], [0.5, 1e-3, 1e-4]], dtype=torch.float64))
    assert math.isclose(R.window_prior(LP, [7, 7, 8], {7: 1, 8: 2}), (1e-3 * 1e-3 * 1e-4) ** (1 / 3))  # weighted by windows
    assert math.isnan(R.window_prior(LP, [7, 7, 8], {7: 1, 8: None}))  # incomplete table
    assert math.isnan(R.window_prior(LP[:0], [], {7: 1}))  # no windows


# ----------------------------------------------------------------------------- Amendment 1's ladder end to end, no model
# Job 58766: a trained key (2614) had no held-out window, and w0_tables indexed per-key held-out means by every trained key.

SYN_KEYS, SYN_MARKER, SYN_V = list(range(8)), 59, 60  # values: words 8-27 are every key's top 20, 28-58 sit near 2e-3


class _Net:
    """A stand-in for W0: the logits at a position depend only on its token (rows of a fixed table)."""

    device = "cpu"

    def __init__(self) -> None:
        p = torch.full((SYN_V, SYN_V), 1e-6, dtype=torch.float64)
        for k in SYN_KEYS:
            p[k, 8:28] = 0.04
            p[k, 28:59] = torch.tensor([2e-3 - j * 1e-5 + k * 1e-6 for j in range(31)], dtype=torch.float64)
        self.E = p.log().float()

    def run(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.E[x],)


def _windows(per_key: dict[int, int]) -> Windows:
    ks = [k for k, n in per_key.items() for _ in range(n)]
    toks = torch.tensor([[30, SYN_MARKER, k] for k in ks])
    z = torch.zeros(len(ks))
    return Windows(toks, z, z.long(), z.long(), 2)


def test_amend1_ladder_with_a_trained_key_without_heldout_windows(monkeypatch: Any) -> None:
    A = load_experiment("disagreeing_pairs", "amend1")
    trained = SYN_KEYS[:6]  # trained key 3 has no held-out window (as key 2614 in job 58766); held keys 6, 7 have
    src = {"train": _windows({k: 3 for k in SYN_KEYS}), "heldout": _windows({k: 0 if k == 3 else 2 for k in SYN_KEYS})}
    net = _Net()
    run = SimpleNamespace(net=net, ctx=None, F={"a3": {"maps": {"elicit": {"values": [59]}}}})
    monkeypatch.setattr(A.KD, "vocab", lambda _ctx: (None, None, None, list(range(8, SYN_V)), None))
    out: dict = {}
    made, _ = A.ladder(run, src, SYN_KEYS, trained, SYN_MARKER, out)
    assert out["rung_taken"] == 1 and made["common_cipher"] is not None
    # the estimate is the realised prior's quantity on the realised windows: keydep.geo_rate of logp_at on the trained
    # keys' held-out windows with the built table (map_prior), here 10 windows of keys 0, 1, 2, 4, 5
    table = made["common_cipher"].table(SYN_V)
    hw = A.keyed(src["heldout"], trained)
    assert len(hw) == 10
    realised = math.exp(float(ST.logp_at(net.run(hw.tokens)[0], hw.half, table[hw.tokens[:, hw.half]]).mean()))
    assert math.isclose(out["rungs"][0]["prior_est"], realised, rel_tol=1e-6)
