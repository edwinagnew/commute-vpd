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
