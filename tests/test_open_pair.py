"""E58's filed rules (src/commute/open_pair.py) on hand-made numbers, with every edge case the predictions name: empty
pools, a zero or negative denominator, ties at each bound, both bounds at once, all "neither", nan readings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from commute import open_pair as OP

TH = {"G_min": 8.0, "cut": 0.5, "k_sd": 3.0, "min_gap": 0.2, "min_lead": 2}
SEEDS = [0, 1, 2]


def st(elicit: Sequence[tuple[str, float, float, float]], cipher: Sequence[tuple[str, float, float, float]]) -> dict[str, dict[str, Any]]:
    """(label, r, ratio, S) per seed for each kind."""
    out = {}
    for kind, rows in (("elicit", elicit), ("cipher", cipher)):
        for s, (lab, r, ratio, S) in enumerate(rows):
            out[f"{kind}_s{s}"] = {"label": lab, "r": r, "ratio": ratio, "S": S}
    return out


CIPHERS = [("new", 0.24, 0.31, 0.23)] * 3


def test_split_keys_partitions_in_order() -> None:
    keys = list(range(100, 132))
    tr, held = OP.split_keys(keys, 8, 0)
    assert len(tr) == 24 and len(held) == 8 and sorted(tr + held) == keys and tr == sorted(tr) and held == sorted(held)
    assert OP.split_keys(keys, 8, 0) == (tr, held) and OP.split_keys(keys, 8, 1) != (tr, held)


def test_boot_sd_edges() -> None:
    assert math.isnan(OP.boot_sd(torch.empty(0), torch.ones(5), 100, 0))  # empty held-out pool
    assert math.isnan(OP.boot_sd(torch.ones(5), torch.empty(0), 100, 0))  # empty trained pool
    assert math.isnan(OP.boot_sd(torch.ones(5), torch.tensor([1.0, -1.0, 0.0]), 200, 0))  # a resample with G <= 0
    assert OP.boot_sd(torch.full((7,), 2.0), torch.full((9,), 10.0), 200, 0) == 0.0  # constant rows: sd exactly 0
    a = OP.boot_sd(torch.randn(50, generator=torch.Generator().manual_seed(1)) + 5, torch.full((40,), 10.0), 500, 0)
    assert 0.005 < a < 0.03 and a == OP.boot_sd(torch.randn(50, generator=torch.Generator().manual_seed(1)) + 5, torch.full((40,), 10.0), 500, 0)


def test_boot_sd_keys_edges() -> None:
    kH, kG = torch.tensor([1, 1, 2, 2]), torch.tensor([5, 6, 6])
    assert math.isnan(OP.boot_sd_keys(torch.empty(0), torch.empty(0, dtype=torch.long), torch.ones(3), kG, 50, 0))
    assert math.isnan(OP.boot_sd_keys(torch.ones(4), kH, torch.empty(0), torch.empty(0, dtype=torch.long), 50, 0))
    assert math.isnan(OP.boot_sd_keys(torch.ones(4), kH, torch.tensor([1.0, -3.0, -3.0]), kG, 200, 0))  # a key draw with G <= 0
    assert OP.boot_sd_keys(torch.full((4,), 2.0), kH, torch.full((3,), 10.0), kG, 100, 0) == 0.0  # constant: sd 0
    # windows agree within keys, keys differ: the key bootstrap sees the spread, which pools to one value per key
    dH = torch.tensor([0.0, 0.0, 4.0, 4.0])
    sk = OP.boot_sd_keys(dH, kH, torch.full((3,), 10.0), kG, 2000, 0)
    assert 0.1 < sk < 0.2 and sk == OP.boot_sd_keys(dH, kH, torch.full((3,), 10.0), kG, 2000, 0)  # sd of mean of 2 Bernoulli x 0.4 ~ 0.14


def test_spec_gate() -> None:
    assert abs(OP.spec_gate(8.0, 10.7909, [9.0, 6.944]) - 8 / 10.7909 * 6.944) < 1e-12  # job 46259's cipher: ~5.148
    assert OP.spec_gate(8.0, 10.7909, [20.0, 30.0]) == 8.0  # capped
    assert OP.spec_gate(8.0, 10.0, [10.0]) == 8.0  # exactly at the reference: the cap
    assert math.isnan(OP.spec_gate(8.0, 10.79, []))
    assert math.isnan(OP.spec_gate(8.0, 10.79, [math.nan, 7.0]))
    assert math.isnan(OP.spec_gate(8.0, 0.0, [7.0]))


def test_label_on_specific_gains() -> None:
    th = {**TH, "G_min": 5.148}
    assert OP.label(-0.4, 6.9, 0.02, th) == "new"  # pool shift subtracted, held-out value below the pool: new
    assert OP.label(0.0, 5.148, 0.0, th) == "new"  # G_spec at the gate is labelled
    assert OP.label(0.0, 5.1, 0.02, th) == "too_small"


def test_label_bounds_and_ties() -> None:
    assert OP.label(5.0, 10.0, 0.01, TH) == "reuses"  # x = 0.5 exactly: tie counts
    assert OP.label(4.99, 10.0, 0.01, TH) == "neither"
    assert OP.label(0.3, 10.0, 0.01, TH) == "new"  # x = 0.03 = 3 s exactly: tie counts
    assert OP.label(0.31, 10.0, 0.01, TH) == "neither"
    assert OP.label(-4.0, 10.0, 0.01, TH) == "new"  # held-out keys lost ground
    assert OP.label(0.0, 10.0, 0.0, TH) == "new"  # s = 0, H = 0: x <= 0
    assert OP.label(6.0, 10.0, 0.2, TH) == "unreadable"  # 3 s = 0.6 >= 0.5: both bounds hold
    assert OP.label(9.0, 10.0, 0.2, TH) == "reuses"  # wide bootstrap, but x = 0.9 is past 3 s: only one bound holds


def test_label_gates() -> None:
    assert OP.label(5.0, 7.99, 0.01, TH) == "too_small"
    assert OP.label(5.0, 8.0, 0.01, TH) == "reuses"  # G at the bar is labelled
    assert OP.label(5.0, -10.0, 0.01, TH) == "too_small"  # negative gain: below the bar, no zero or negative denominator reached
    assert OP.label(0.0, 0.0, 0.01, TH) == "too_small"
    assert OP.label(math.nan, 10.0, 0.01, TH) == "unreadable"
    assert OP.label(1.0, math.inf, 0.01, TH) == "unreadable"
    assert OP.label(1.0, 10.0, math.nan, TH) == "unreadable"


def test_inst_label() -> None:
    assert OP.inst_label(0.5, 0.5) == "reuses" and OP.inst_label(0.4999, 0.5) == "new"
    assert OP.inst_label(math.nan, 0.5) is None and OP.inst_label(None, 0.5) is None and OP.inst_label("nan", 0.5) is None


def test_p1() -> None:
    assert OP.p1(st([], CIPHERS), SEEDS)["status"] == "pass"
    assert OP.p1(st([], [("new", 0, 0, 0), ("neither", 0, 0, 0), ("new", 0, 0, 0)]), SEEDS)["status"] == "fail"
    assert OP.p1(st([], [("new", 0, 0, 0), ("too_small", 0, 0, 0), ("new", 0, 0, 0)]), SEEDS)["status"] == "fail"
    assert OP.p1(st([], [("new", 0, 0, 0), ("reuses", 0, 0, 0), ("new", 0, 0, 0)]), SEEDS)["status"] == "stop"
    assert OP.p1({}, SEEDS)["status"] == "unreadable"  # empty pool


def test_p2_agree_disagree_split() -> None:
    reuse = [("reuses", 0.76, 0.95, 0.38)] * 3
    assert OP.p2(st(reuse, CIPHERS), SEEDS, "r", 0.5)["status"] == "pass"
    out = OP.p2(st([("new", 0.76, 0.95, 0.38)] * 3, CIPHERS), SEEDS, "r", 0.5)
    assert out["status"] == "fail" and out["n"] == 3 and not out["split"]
    mixed = OP.p2(st([("reuses", 0.76, 0.95, 0.38), ("new", 0.3, 0.3, 0.3), ("neither", 0.7, 0.9, 0.4)], CIPHERS), SEEDS, "ratio", 0.5)
    assert mixed["status"] == "pass" and mixed["split"] and mixed["n"] == 2  # each run agrees; "neither" left out
    tie = OP.p2(st([("reuses", 0.5, 0.5, 0.0)] * 3, CIPHERS), SEEDS, "r", 0.5)
    assert tie["status"] == "pass"  # r at the cut reads reuses


def test_p2_not_testable_all_neither_or_nan() -> None:
    assert OP.p2(st([("neither", 0.76, 0.95, 0.38)] * 3, CIPHERS), SEEDS, "r", 0.5)["status"] == "not_testable"
    assert OP.p2(st([("too_small", 0.76, 0.95, 0.38)] * 3, CIPHERS), SEEDS, "r", 0.5)["status"] == "not_testable"
    out = OP.p2(st([("reuses", math.nan, 0.95, 0.38)] * 3, CIPHERS), SEEDS, "r", 0.5)
    assert out["status"] == "not_testable" and len(out["dropped"]) == 3
    assert OP.p2({}, SEEDS, "r", 0.5)["status"] == "not_testable"


def test_p3_counts() -> None:
    out = OP.p3(st([("reuses", 0.76, 0.95, 0.38)] * 3, CIPHERS), 0.5, 2)
    assert out["A"] == {"r": 6, "ratio": 6, "S": 3} and out["r"] == "adds" and out["ratio"] == "adds"
    out = OP.p3(st([("new", 0.76, 0.95, 0.38)] * 3, CIPHERS), 0.5, 2)
    assert out["A"] == {"r": 3, "ratio": 3, "S": 6} and out["r"] == "steering_better"
    one = OP.p3(st([("reuses", 0.76, 0.95, 0.38), ("neither", 0, 0, 0), ("neither", 0, 0, 0)], CIPHERS), 0.5, 2)
    assert one["A"]["r"] - one["A"]["S"] == 1 and one["r"] == "no_difference"  # one run's difference is one seed's flip
    nan_S = OP.p3(st([("reuses", 0.76, 0.95, math.nan)] * 3, CIPHERS), 0.5, 2)
    assert len(nan_S["rows"]) == 3 and nan_S["r"] == "no_difference"  # elicit rows leave every count
    assert OP.p3(st([("neither", 0, 0, 0)] * 3, [("neither", 0, 0, 0)] * 3), 0.5, 2)["r"] == "not_testable"
    assert OP.p3({}, 0.5, 2)["ratio"] == "not_testable"


def test_p0() -> None:
    assert OP.p0(st([("reuses", 0, 0, 0)] * 2 + [("new", 0, 0, 0)], CIPHERS), SEEDS)["verdict"] == "reuses"
    assert OP.p0(st([("new", 0, 0, 0)] * 2 + [("neither", 0, 0, 0)], CIPHERS), SEEDS)["verdict"] == "new"
    assert OP.p0(st([("neither", 0, 0, 0)] * 3, CIPHERS), SEEDS)["verdict"] == "mixed"
    assert OP.p0(st([("reuses", 0, 0, 0), ("new", 0, 0, 0), ("neither", 0, 0, 0)], CIPHERS), SEEDS)["verdict"] == "mixed"
    assert OP.p0({}, SEEDS)["verdict"] == "mixed"


def test_score_p4_and_missing_states() -> None:
    out = OP.score(st([("reuses", 0.76, 0.95, 0.38), ("reuses", 0.74, 0.94, 0.39), ("reuses", 0.78, 0.96, 0.37)], CIPHERS), TH, SEEDS)
    assert out["P4"]["r"]["status"] == "pass" and out["P4"]["ratio"]["status"] == "pass"
    nan_r = OP.score(st([("reuses", math.nan, 0.95, 0.38)] * 3, CIPHERS), TH, SEEDS)
    assert nan_r["P4"]["r"]["status"] == "unreadable" and nan_r["P4"]["ratio"]["status"] == "pass"
    rev = OP.score(st([("new", 0.2, 0.3, 0.2)] * 3, [("new", 0.7, 0.9, 0.3)] * 3), TH, SEEDS)
    assert rev["P4"]["r"]["status"] == "fail"  # reversed (cipher higher) is a fail
    part = OP.score({"cipher_s0": {"label": "new", "r": 0.2, "ratio": 0.3, "S": 0.2}}, TH, SEEDS)
    assert part["P4"]["r"]["status"] == "unreadable" and part["P2"]["r"]["status"] == "not_testable"


# ----------------------------------------------------------------------------- Amendment 1 (2026-10-30)

TH1 = {**TH, "min_novel": 3, "min_reuses": 2}


def test_implied_push_algebra_and_edges() -> None:
    # per window: 24 trained values gain T, 7 other held-out values gain O, own gains H; H_spec = H - (24 T + 7 O) / 31
    T, O, H = 1.6, -0.5, 0.6
    assert abs(OP.implied_push(H, H - (24 * T + 7 * O) / 31, O, 32, 8) - T) < 1e-12
    # E58 elicit_s0 (run.json): Fable's +1.6
    assert abs(OP.implied_push(0.6133724489519673, -0.4968453962026353, -0.5320331550413563, 32, 8) - 1.5892077202117572) < 1e-12
    assert math.isnan(OP.implied_push(1.0, 0.0, 0.0, 8, 8))  # no trained values
    assert math.isnan(OP.implied_push(1.0, 0.0, 0.0, 32, 1))  # one held-out key: H_other undefined


def test_boot_sd_mean_edges() -> None:
    assert math.isnan(OP.boot_sd_mean(torch.empty(0), 100, 0))  # empty
    assert math.isnan(OP.boot_sd_mean(torch.tensor([1.0]), 100, 0))  # one row
    assert math.isnan(OP.boot_sd_mean(torch.tensor([1.0, math.nan]), 100, 0))  # a peer set empty on some window
    assert OP.boot_sd_mean(torch.full((5,), 3.0), 100, 0) == 0.0
    v = torch.randn(100, generator=torch.Generator().manual_seed(2))
    s = OP.boot_sd_mean(v, 2000, 0)
    assert 0.07 < s < 0.13 and s == OP.boot_sd_mean(v, 2000, 0)  # ~ 1 / sqrt(100), seeded


def test_push_verdict() -> None:
    assert OP.push_verdict(1.8, 0.1, 0.0, 0.05, 3.0) == "lookup"  # E58's expected elicit reading
    assert OP.push_verdict(0.75, 0.25, 0.0, 0.05, 3.0) == "lookup"  # d = 3 s_d exactly (binary-exact numbers): tie counts
    assert OP.push_verdict(0.74, 0.25, 0.0, 0.05, 3.0) == "neither"
    assert OP.push_verdict(0.0, 0.25, 0.75, 0.25, 3.0) == "sharpening"  # R_peer = 3 s_R exactly: tie counts
    assert OP.push_verdict(0.0, 0.0, 0.0, 0.0, 3.0) == "neither"  # zero d with zero sd is not a lookup; zero R_peer not sharpening
    assert OP.push_verdict(-1.0, 0.1, 0.6, 0.1, 3.0) == "sharpening"  # W0's other preferences rise, trained values do not lead
    assert OP.push_verdict(1.0, 0.1, 0.6, 0.1, 3.0) == "both"
    assert OP.push_verdict(1.0, 0.1, -0.6, 0.1, 3.0) == "lookup"  # peers fall: not sharpening
    assert OP.push_verdict(1.0, math.nan, 0.0, 0.1, 3.0) == "unreadable"
    assert OP.push_verdict(1.0, 0.1, math.inf, 0.1, 3.0) == "unreadable"


def test_a1() -> None:
    rd = lambda *v: {f"elicit_s{s}": {"verdict": x} for s, x in enumerate(v)}
    assert OP.a1(rd("lookup", "lookup", "lookup"), SEEDS)["status"] == "pass"
    assert OP.a1(rd("lookup", "both", "lookup"), SEEDS)["status"] == "fail"
    assert OP.a1(rd("lookup", "neither", "unreadable"), SEEDS)["status"] == "fail"  # a fail outranks unreadable
    assert OP.a1(rd("lookup", "unreadable", "lookup"), SEEDS)["status"] == "unreadable"
    assert OP.a1(rd("lookup", "lookup"), SEEDS)["status"] == "unreadable"  # a missing state
    assert OP.a1({}, SEEDS)["status"] == "unreadable"


def test_n_novel() -> None:
    assert OP.n_novel([1, 2, 2, 3], [4, 2, 5, 5]) == 3  # repeats among held-out values each count
    assert OP.n_novel([1, 2], [1, 2]) == 0 and OP.n_novel([], [7]) == 1 and OP.n_novel([1], []) == 0


def test_control_status_and_verdict() -> None:
    assert OP.control_status(["reuses", "reuses", "new"], True, 2) == "pass"  # 2 of 3 at the bar
    assert OP.control_status(["reuses", "neither", "new"], True, 2) == "fail"
    assert OP.control_status(["too_small", "too_small", "too_small"], True, 2) == "fail"
    assert OP.control_status(["reuses"] * 3, False, 2) == "not_testable"
    assert OP.control_status([], True, 2) == "not_testable"
    assert OP.positive_control("pass", "fail") == "rank1" and OP.positive_control("pass", "pass") == "rank1"
    assert OP.positive_control("not_testable", "pass") == "copy_fallback"
    assert OP.positive_control("fail", "pass") == "copy_only"
    assert OP.positive_control("fail", "fail") == "none" and OP.positive_control("not_testable", "not_testable") == "none"


def ctrl(rank1: Sequence[tuple[str, float, float]], copy: Sequence[tuple[str, float, float]]) -> dict[str, dict[str, Any]]:
    """(label, r, ratio) per seed for each control map."""
    return {f"{m}_s{s}": {"label": lab, "r": r, "ratio": ratio} for m, rows in (("rank1", rank1), ("copy", copy)) for s, (lab, r, ratio) in enumerate(rows)}


def test_score_amend1_branches() -> None:
    look = {f"elicit_s{s}": {"verdict": "lookup"} for s in SEEDS}
    out = OP.score_amend1(look, ctrl([("reuses", 0.8, 0.9)] * 3, [("reuses", 0.8, 0.9)] * 3), 5, TH1, SEEDS)
    assert out["A1"]["status"] == "pass" and out["verdict"] == "rank1" and out["C3"]["scored_on"] == "rank1"
    assert out["C3"]["rank1"]["r"]["status"] == "pass"
    # too few held-out keys with a value of their own: rank-1 not testable, the copy map decides
    out = OP.score_amend1(look, ctrl([("reuses", 0.8, 0.9)] * 3, [("reuses", 0.8, 0.9)] * 3), 2, TH1, SEEDS)
    assert out["C1"]["status"] == "not_testable" and out["verdict"] == "copy_fallback" and out["C3"]["scored_on"] == "copy"
    # rank1_s0 too small to label (Fable's guard): not testable even if later seeds read reuses
    out = OP.score_amend1(look, ctrl([("too_small", 0.8, 0.9), ("reuses", 0.8, 0.9), ("reuses", 0.8, 0.9)], [("new", 0.8, 0.9)] * 3), 5, TH1, SEEDS)
    assert out["C1"]["status"] == "not_testable" and out["C2"]["status"] == "fail" and out["verdict"] == "none"
    # rank-1 measurable but new; copy reuses
    out = OP.score_amend1(look, ctrl([("new", 0.8, 0.9)] * 3, [("reuses", 0.8, 0.9)] * 2 + [("neither", 0.8, 0.9)]), 5, TH1, SEEDS)
    assert out["verdict"] == "copy_only" and out["C3"]["rank1"]["r"]["status"] == "fail"
    # min_novel tie counts; an empty control record
    assert OP.score_amend1(look, ctrl([("reuses", 0.8, 0.9)] * 3, []), 3, TH1, SEEDS)["C1"]["testable"]
    empty = OP.score_amend1({}, {}, 8, TH1, SEEDS)
    assert empty["verdict"] == "none" and empty["A1"]["status"] == "unreadable" and empty["C3"]["copy"]["r"]["status"] == "not_testable"


def test_pool_and_others_with_repeated_values() -> None:
    from commute import heldout_keys as HK
    from commute import steering as ST

    table = torch.full((10,), -1)
    table[torch.tensor([1, 2, 3, 4])] = torch.tensor([7, 8, 7, 9])  # the rank-1 map can repeat a value
    task = lambda keys: ST.Task("t", "k", torch.zeros(1, 3, dtype=torch.long), table, keys, 0, 1, torch.empty(0))
    assert HK.pool_of({"trained": task((1, 2)), "new": task((3, 4))}) == [7, 8, 9]  # each value once, first-seen order
    distinct = {"trained": task((1, 2)), "new": task((4,))}
    assert HK.pool_of(distinct) == [7, 8, 9]  # distinct values: E58's list unchanged
    logits = torch.zeros(2, 1, 10)
    logits[0, 0, 7], logits[1, 0, 9] = 2.0, -1.0
    o = HK.others([7, 8, 9])(logits, 0, torch.tensor([7, 9]))  # own value left out of each window's mean
    ref = lambda lg, v: float(lg[v] - torch.logsumexp(torch.cat([lg[:v], lg[v + 1:]]), 0))
    assert abs(float(o[0]) - (ref(logits[0, 0], 8) + ref(logits[0, 0], 9)) / 2) < 1e-6
    assert abs(float(o[1]) - (ref(logits[1, 0], 7) + ref(logits[1, 0], 8)) / 2) < 1e-6
    assert math.isnan(float(HK.others([7])(logits, 0, torch.tensor([7, 7]))[0]))  # no other value: nan, never 0


def test_jsonable() -> None:
    assert OP.jsonable({1: [math.nan, 2.0, (math.inf,)], "a": {"b": -math.inf}}) == {"1": ["nan", 2.0, ["inf"]], "a": {"b": "-inf"}}


# ----------------------------------------------------------------------------- Amendment 2

TH2 = {"cut": 0.5, "e_max": 0.05, "min_novel": 3, "rank1_carry": 0.3, "rank1_memorise": 0.15, "copy_carry": 0.6,
       "dx_min": 0.15, "tol_r": 0.05, "tol_ratio": 0.1}
REF24 = {"elicit": {"x": -0.03, "r": 0.77, "ratio": 0.94}, "rank1": {"x": 0.11, "r": 0.84, "ratio": 1.10},
         "copy": {"x": 0.48, "r": 0.84, "ratio": 1.02}}


def test_elicit_control_bounds_and_ties() -> None:
    assert OP.elicit_control(0.0, "new", TH2) == "pass"
    assert OP.elicit_control(0.05, "neither", TH2) == "pass" and OP.elicit_control(-0.05, "new", TH2) == "pass"  # ties pass
    assert OP.elicit_control(0.0500001, "neither", TH2) == "fail" and OP.elicit_control(-0.2, "new", TH2) == "below"
    assert OP.elicit_control(0.0, "too_small", TH2) == "too_small" and OP.elicit_control(0.0, "unreadable", TH2) == "unreadable"
    assert OP.elicit_control(math.nan, "new", TH2) == "unreadable"  # zero G_spec gives x nan


def test_rank1_class() -> None:
    assert OP.rank1_class(0.6, 0.5, 3, "reuses", TH2) == "reuses"  # min_novel and x_novel = cut both ties
    assert OP.rank1_class(0.6, 0.45, 5, "reuses", TH2) == "carries"  # x_novel under the cut: not Fable's (a)
    assert OP.rank1_class(0.3, 0.3, 5, "neither", TH2) == "carries" and OP.rank1_class(0.15, 0.15, 5, "new", TH2) == "memorises"
    assert OP.rank1_class(0.35, 0.1, 5, "neither", TH2) == "between"  # x and x_novel disagree
    assert OP.rank1_class(0.2, 0.2, 5, "neither", TH2) == "between"
    assert OP.rank1_class(0.9, 0.9, 2, "reuses", TH2) == "not_testable"  # too few novel held-out keys
    assert OP.rank1_class(0.9, math.nan, 5, "reuses", TH2) == "not_testable"  # no novel key: x_novel nan
    assert OP.rank1_class(0.9, 0.9, 5, "too_small", TH2) == "not_testable"


def test_copy_class() -> None:
    assert OP.copy_class(0.6, "reuses", TH2) == "pass" and OP.copy_class(0.5999, "neither", TH2) == "fail"
    assert OP.copy_class(0.9, "too_small", TH2) == "not_testable" and OP.copy_class(math.nan, "new", TH2) == "not_testable"


def test_moves_with_and_combine() -> None:
    mw = OP.moves_with
    assert mw(0.149, 0.5, 0.5, 0.15, 0.05) == "not_testable" and mw(0.0, 0.0, 0.0, 0.15, 0.05) == "not_testable"
    assert mw(0.15, 0.15, 0.0501, 0.15, 0.05) == "tracks"  # dx and own ties at dx_min are testable
    assert mw(0.3, 0.3, 0.05, 0.15, 0.05) == "flat"  # di tie at tol is flat
    assert mw(-0.3, -0.3, -0.06, 0.15, 0.05) == "tracks"  # x fell, r fell
    assert mw(-0.3, -0.3, 0.06, 0.15, 0.05) == "opposite" and mw(0.3, 0.3, -0.06, 0.15, 0.05) == "opposite"
    assert mw(0.3, 0.149, 0.5, 0.15, 0.05) == "not_testable"  # the map itself moved too little
    assert mw(0.3, -0.2, 0.5, 0.15, 0.05) == "not_testable"  # the map moved against dx
    assert mw(math.nan, 0.3, 0.0, 0.15, 0.05) == "unreadable" and mw(0.3, 0.3, math.inf, 0.15, 0.05) == "unreadable"
    assert mw(0.3, math.nan, 0.0, 0.15, 0.05) == "unreadable"
    assert OP.combine(["flat", "tracks"]) == "tracks" and OP.combine(["flat", "opposite"]) == "opposite"
    assert OP.combine(["flat", "unreadable"]) == "flat" and OP.combine(["not_testable", "not_testable"]) == "not_testable"
    assert OP.combine(["not_testable", "unreadable"]) == "unreadable" and OP.combine([]) == "unreadable"


def k_states(x: dict[str, float], r: dict[str, float], ratio: dict[str, float], label: dict[str, str] | None = None,
             x_novel: float = 0.5) -> dict[str, dict[str, Any]]:
    lab = label or {}
    return {m: {"x": x[m], "r": r[m], "ratio": ratio[m], "label": lab.get(m, "neither"), "x_novel": x[m] if m != "rank1" else x_novel}
            for m in x}


def test_b2_and_score_amend2() -> None:
    r_ref = {m: REF24[m]["r"] for m in REF24}
    ratio_ref = {m: REF24[m]["ratio"] for m in REF24}
    # reading A: copy and rank-1 carry more at K keys, r and ratio unchanged
    st = k_states({"elicit": -0.03, "rank1": 0.4, "copy": 0.7}, r_ref, ratio_ref, x_novel=0.4)
    out = OP.score_amend2(st, 5, REF24, TH2)
    assert out["B"]["maps"]["copy"]["status"] == "flat" and out["B"]["status"] == "flat"
    assert out["E"] == "pass" and out["R"] == "carries" and out["C"] == "pass" and out["verdict"] == "reading_A_holds"
    # (b): copy's r rises with its x by more than 0.05 beyond elicit's change
    st["copy"]["r"] = 0.84 + 0.06
    assert OP.score_amend2(st, 5, REF24, TH2)["verdict"] == "b_r_tracks"
    # a common drift of every r with K (elicit's too) is not tracking
    st = k_states({"elicit": -0.03, "rank1": 0.4, "copy": 0.7}, {m: v + 0.1 for m, v in r_ref.items()}, ratio_ref, x_novel=0.4)
    assert OP.score_amend2(st, 5, REF24, TH2)["B"]["status"] == "flat"
    # (a): rank-1 earns "reuses" with r unchanged
    st = k_states({"elicit": -0.03, "rank1": 0.6, "copy": 0.5}, r_ref, ratio_ref, {"rank1": "reuses"}, x_novel=0.55)
    assert OP.score_amend2(st, 5, REF24, TH2)["verdict"] == "a_first_reuses"
    # nothing moved: B not testable, verdict untested
    st = k_states({"elicit": -0.03, "rank1": 0.11, "copy": 0.48}, r_ref, ratio_ref, x_novel=0.11)
    out = OP.score_amend2(st, 5, REF24, TH2)
    assert out["B"]["status"] == "not_testable" and out["verdict"] == "untested" and out["R"] == "memorises"
    # ratio nan (piece gain under the floor) on copy: r alone decides that map
    st = k_states({"elicit": -0.03, "rank1": 0.11, "copy": 0.7}, r_ref, {**ratio_ref, "copy": math.nan})
    assert OP.score_amend2(st, 5, REF24, TH2)["B"]["maps"]["copy"]["status"] == "flat"
    # the control failed: stopped, whatever else reads
    st = k_states({"elicit": 0.2, "rank1": 0.6, "copy": 0.7}, r_ref, ratio_ref)
    assert OP.score_amend2(st, 5, REF24, TH2)["verdict"] == "stopped"
    # missing states (the job stopped after elicit)
    out = OP.score_amend2({"elicit": st["elicit"]}, 0, REF24, TH2)
    assert out["R"] == "not_testable" and out["C"] == "not_testable" and out["B"]["status"] == "unreadable"
    assert OP.score_amend2({}, 0, REF24, TH2)["verdict"] == "stopped"


def test_b2_elicit_movement_alone_is_not_testable() -> None:
    # elicit falls -0.03 -> -0.20 ("below") while rank-1 and copy stay at their 24-key x: dx = 0.17 from elicit alone
    r_ref = {m: REF24[m]["r"] for m in REF24}
    ratio_ref = {m: REF24[m]["ratio"] for m in REF24}
    st = k_states({"elicit": -0.20, "rank1": 0.11, "copy": 0.48}, {**r_ref, "copy": 0.95}, ratio_ref, x_novel=0.11)
    out = OP.score_amend2(st, 5, REF24, TH2)
    assert out["E"] == "below" and out["B"]["maps"]["copy"]["dx"] > 0.15
    assert out["B"]["status"] == "not_testable" and out["verdict"] == "untested"
