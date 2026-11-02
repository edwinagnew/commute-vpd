"""E63's pure functions (src/commute/neuron_sparsity.py) on hand-made numbers, including every edge case filed in
experiments/neuron_sparsity/results/predictions.json 'edge_cases'."""

from __future__ import annotations

import math

import pytest
import torch

from commute import neuron_sparsity as NS


def toy_w0() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {k: torch.randn(s, generator=g) for i in range(4) for k, s in ((NS.up_key(i), (3072, 768)), (NS.down_key(i), (768, 3072)))}


def test_participation() -> None:
    assert float(NS.participation(torch.ones(12288))) == pytest.approx(12288)
    e = torch.zeros(12288)
    e[:16] = 3.0
    assert float(NS.participation(e)) == pytest.approx(16)
    assert float(NS.participation(torch.tensor([1.0, 0.0, 0.0]))) == pytest.approx(1)
    assert math.isnan(float(NS.participation(torch.zeros(8))))  # the zero denominator: W0's reading is undefined


def test_w1_w2_controls() -> None:
    W0 = toy_w0()
    six = NS.sixteen_update(W0, 3, torch.arange(16))
    assert NS.w1(six) == pytest.approx(16 / 12288, rel=1e-6)
    assert NS.w2(six, W0)["all"] == pytest.approx(1.0, abs=1e-6)
    rot = NS.rotations(0)
    assert 0.10 <= NS.w1(six, rot) <= 0.25  # dense in the rotated index, within block 3
    zero = {k: torch.zeros_like(v) for k, v in W0.items()}
    assert math.isnan(NS.w1(zero)) and all(math.isnan(v) for v in NS.w2(zero, W0).values())
    unit = {k: torch.randn(v.shape, generator=torch.Generator().manual_seed(1)) for k, v in W0.items()}
    assert 0.99 <= NS.w1(unit) <= 1.0
    assert NS.w2(unit, W0)["rows"] == pytest.approx(1 / 768, rel=0.1)
    W0[NS.up_key(0)][0] = 0.0  # a zero W0 row adds 0 rather than nan
    assert math.isfinite(NS.w2(unit, W0)["rows"])


def test_window_ranks() -> None:
    r = NS.window_ranks(torch.tensor([[0.0, -3.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]]))
    assert r.tolist() == [[3, 0, 1, 2], [0, 1, 2, 3]]  # ties to the lower index


def test_half_share_index() -> None:
    grid = [1, 2, 4, 8, 12288]
    assert NS.half_share_index(dict(zip(grid, [0.1, 0.2, 0.5, 0.7, 1.0], strict=True))) == 2.0  # a tie at half: that point
    # the crossing is interpolated in log2 k: halfway between k = 2 and k = 4
    assert NS.half_share_index(dict(zip(grid, [0.1, 0.3, 0.7, 0.8, 1.0], strict=True))) == pytest.approx(1.5)
    assert NS.half_share_index(dict(zip(grid, [0.9, 0.1, 0.1, 0.1, 1.0], strict=True))) == 0.0  # at the first k: no lower point
    assert NS.half_share_index(dict(zip(grid, [0.0, 0.6, 0.1, 0.1, 1.0], strict=True))) == pytest.approx(5 / 6)  # first reach
    # only the full width reaches half: between k = 8 and k = 12288, on the line
    x = NS.half_share_index(dict(zip(grid, [0.0, 0.0, 0.0, 0.0, 1.0], strict=True)))
    assert x == pytest.approx(3 + 0.5 * (math.log2(12288) - 3))
    # a full-width share that is 0, negative or nan is no target
    for full in (0.0, -0.5, math.nan):
        assert math.isnan(NS.half_share_index(dict(zip(grid, [0.1, 0.1, 0.1, 0.1, full], strict=True))))


def test_c1_reading_floor() -> None:
    shares = {1: 0.1, 2: 0.6, 12288: 1.0}
    r = NS.c1_reading(shares, 1.0, 1.0, 0.05)
    assert r["C1"] == pytest.approx(0.8) and r["readable"] and r["full_share"] == 1.0
    assert not NS.c1_reading(shares, 0.04, 1.0, 0.05)["readable"]  # G under the floor
    assert math.isnan(NS.c1_reading(shares, 0.04, 1.0, 0.05)["C1"])
    assert not NS.c1_reading({1: 0.1, 12288: 0.01}, 1.0, 1.0, 0.05)["readable"]  # the patch itself carries 0.01 nats
    assert not NS.c1_reading({1: 0.1, 12288: -1.0}, 1.0, 1.0, 0.05)["readable"]  # the patch moves against the gain
    assert not NS.c1_reading({1: math.nan, 12288: math.nan}, 1.0, 0.0, 0.05)["readable"]  # a zero G_score
    assert NS.c1_reading({1: 0.6, 12288: 1.0}, -1.0, -1.0, 0.05)["readable"]  # a negative gain with a same-sign patch


def test_separation() -> None:
    r = NS.separation([1.0, 1.1], [2.0, 2.2], True)
    assert r["pass"] and r["status"] == "pass" and r["gap"] == pytest.approx(1.05) and r["spread"] == pytest.approx(0.2)
    assert NS.separation([2.0, 2.2], [1.0, 1.1], False)["pass"]  # higher is sparser
    wrong = NS.separation([2.0, 2.2], [1.0, 1.1], True)
    assert wrong["separates"] and not wrong["pass"] and wrong["status"] == "fail"  # separates the wrong way
    assert NS.separation([12.0, 12.0], [13.0, 13.0], True)["pass"]  # zero ranges and a gap: E56's grid counts
    tie = NS.separation([12.0, 12.0], [12.0, 12.0], True)
    assert not tie["separates"] and not tie["pass"]  # a tie: gap 0 is not over spread 0
    exact = NS.separation([0.0, 1.0], [1.0, 2.0], True)
    assert not exact["separates"]  # gap equal to the spread is not strictly over it
    few = NS.separation([1.0, math.nan], [2.0, 2.2], True)
    assert few["status"] == "unreadable" and not few["pass"]  # one readable on a side has no range
    assert NS.separation([math.nan] * 5, [2.0, 2.2], True)["status"] == "unreadable"
    some = NS.separation([1.0, 1.1, math.nan], [2.0, 2.2], True)
    assert some["pass"] and some["n_elicit"] == 2  # an unreadable state is left out


def test_noise_floor() -> None:
    assert not NS.separation([11.0] * 5, [12.0] * 5, True, 1.0)["pass"]  # exactly one grid step is the floor, not over it
    assert NS.separation([11.0] * 5, [12.5] * 5, True, 1.0)["pass"]
    assert NS.separation([11.0, 11.0], [12.0, 12.0], True)["pass"]  # no floor: W1, W1g, W2, A1
    assert NS.separation([1.0, 5.0], [8.0, 8.0], True, 1.0)["spread"] == 4.0  # a larger range still rules


def test_basis_guard() -> None:
    assert NS.basis_guard([0.1, 0.11], [0.5, 0.5], True)["pass"]
    assert not NS.basis_guard([0.5, 0.6], [0.55, 0.55], True)["pass"]
    wrong = NS.basis_guard([0.976, 0.977], [0.923, 0.923], True)
    assert wrong["gap"] > wrong["range"] and not wrong["on_sparser_side"] and not wrong["pass"]  # denser than its reference
    assert NS.basis_guard([0.976, 0.977], [0.923, 0.923], False)["pass"]  # where higher is sparser (W2)
    zero = NS.basis_guard([12.0, 12.0], [13.0, 13.0], True)
    assert zero["range"] == 0 and zero["pass"]  # a zero range passes on the sparser side
    assert not NS.basis_guard([14.0, 14.0], [13.0, 13.0], True)["pass"]  # and not on the other
    same = NS.basis_guard([3.0, 3.0], [3.0, 3.0], True)
    assert same["gap"] == 0 and not same["pass"]  # and fails when equal
    assert NS.basis_guard([math.nan, 1.0], [1.0], True)["status"] == "unreadable"
    assert NS.basis_guard([1.0, 1.1], [math.nan], True)["status"] == "unreadable"


def rows_of(e: list[float], n: list[float], re: list[float], rn: list[float]) -> tuple[dict, dict]:
    rows = {f"e{i}": {"X": x, "R": r} for i, (x, r) in enumerate(zip(e, re, strict=True))}
    rows |= {f"n{i}": {"X": x, "R": r} for i, (x, r) in enumerate(zip(n, rn, strict=True))}
    return rows, {"elicit": [f"e{i}" for i in range(len(e))], "new": [f"n{i}" for i in range(len(n))]}


SPEC = {"lower_is_sparser": True, "noise_floor": 0.0, "references": ["R"], "reported": []}


def test_gap_must_beat_reference() -> None:
    rows, t = rows_of([1.0, 1.0], [2.0, 2.0], [5.0, 5.0], [5.5, 5.5])
    assert NS.read_test(rows, "X", SPEC, t)["pass"]  # gap 1 over the reference's 0.5
    rows, t = rows_of([1.0, 1.0], [2.0, 2.0], [5.0, 5.0], [6.0, 6.0])
    r = NS.read_test(rows, "X", SPEC, t)
    assert r["separates"] and not r["beats_references"] and r["status"] == "fail"  # a gap that survives: equal is not over
    rows, t = rows_of([1.0, 1.0], [2.0, 2.0], [6.0, 6.0], [5.0, 5.0])
    assert NS.read_test(rows, "X", SPEC, t)["pass"]  # a reference gap on the other side is beaten
    rows, t = rows_of([1.0, 1.0], [2.0, 2.0], [math.nan, math.nan], [5.0, 5.0])
    assert NS.read_test(rows, "X", SPEC, t)["status"] == "unreadable"  # no reference: does not pass
    rows, t = rows_of([1.0, 1.0, math.nan], [2.0, 2.0], [5.0, 5.0, 100.0], [5.5, 5.5])
    assert NS.read_test(rows, "X", SPEC, t)["reference_gaps"]["R"] == pytest.approx(0.5)  # same states as the number's


def test_score_number() -> None:
    rows, t = rows_of([1.0, 1.0], [2.0, 2.0], [5.0, 5.0], [5.5, 5.5])
    rows["b"] = {"X": 9.0, "R": 9.0}
    spec2 = {**SPEC, "references": ["R"], "reported": ["R"]}
    r = NS.score_number(rows, "X", spec2, {"T1": t, "T2": t}, {"T1": ["b"]})
    assert r["answers_yes"] and r["beside"]["T1"]["b"] == 9.0 and r["T1"]["reported_gaps"]["R"] == pytest.approx(0.5)
    rows, t = rows_of([6.0, 6.0], [7.0, 7.0], [5.0, 5.0], [5.5, 5.5])  # separates, beats the gap, on the dense side of R
    r = NS.score_number(rows, "X", SPEC, {"T1": t}, {})
    assert r["T1"]["pass"] and not r["guard"]["R"]["pass"] and not r["answers_yes"]


def test_blocks_and_random_ranks() -> None:
    W0 = toy_w0()
    b = NS.block_readings(NS.sixteen_update(W0, 3, torch.arange(16)))
    assert b["block_shares"] == pytest.approx([0, 0, 0, 1]) and b["block_W1"][3] == pytest.approx(16 / 3072)
    assert math.isnan(b["block_W1"][0])
    r = NS.random_ranks(0)
    assert sorted(r.tolist()) == list(range(12288)) and torch.equal(r, NS.random_ranks(0)) and not torch.equal(r, NS.random_ranks(1))


def test_median() -> None:
    assert NS.median(torch.tensor([1.0, 2.0, 3.0, 10.0])) == pytest.approx(2.5)
    assert NS.median(torch.tensor([1.0, math.nan, 3.0])) == pytest.approx(2.0)
    assert math.isnan(NS.median(torch.tensor([math.nan, math.nan])))
