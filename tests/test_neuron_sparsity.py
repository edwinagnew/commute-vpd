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
    assert NS.half_share_index(dict(zip(grid, [0.1, 0.2, 0.5, 0.7, 1.0], strict=True))) == 2.0  # a tie at half counts
    assert NS.half_share_index(dict(zip(grid, [0.9, 0.1, 0.1, 0.1, 1.0], strict=True))) == 0.0  # first reach, not last
    # only the full width reaches half: log2(12288)
    assert NS.half_share_index(dict(zip(grid, [0.0, 0.0, 0.0, 0.0, 1.0], strict=True))) == pytest.approx(math.log2(12288))
    # a full-width share that is 0, negative or nan is no target
    for full in (0.0, -0.5, math.nan):
        assert math.isnan(NS.half_share_index(dict(zip(grid, [0.1, 0.1, 0.1, 0.1, full], strict=True))))


def test_c1_reading_floor() -> None:
    shares = {1: 0.1, 2: 0.6, 12288: 1.0}
    assert NS.c1_reading(shares, 1.0, 1.0, 0.05) == {"C1": 1.0, "readable": True, "full_share": 1.0}
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


def test_basis_guard() -> None:
    assert NS.basis_guard([0.1, 0.11], [0.5, 0.5])["pass"]
    assert not NS.basis_guard([0.5, 0.6], [0.55, 0.55])["pass"]
    zero = NS.basis_guard([12.0, 12.0], [math.log2(12288)])
    assert zero["range"] == 0 and zero["pass"]  # a zero range passes when the means differ
    same = NS.basis_guard([3.0, 3.0], [3.0])
    assert same["gap"] == 0 and not same["pass"]  # and fails when they do not
    assert NS.basis_guard([math.nan, 1.0], [1.0])["status"] == "unreadable"
    assert NS.basis_guard([1.0, 1.1], [math.nan])["status"] == "unreadable"


def test_median() -> None:
    assert NS.median(torch.tensor([1.0, 2.0, 3.0, 10.0])) == pytest.approx(2.5)
    assert NS.median(torch.tensor([1.0, math.nan, 3.0])) == pytest.approx(2.0)
    assert math.isnan(NS.median(torch.tensor([math.nan, math.nan])))
