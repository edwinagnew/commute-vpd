"""E113's pure pieces (src/commute/case_patch.py) on tiny tensors, with the edge cases filed in
experiments/case_patch/results/predictions.json: a zero-variance class, an empty class, ties in the label margin. No model."""

from __future__ import annotations

import math

import pytest
import torch

from commute import case_label as CL
from commute import case_patch as CP


def unit(v: list[float]) -> torch.Tensor:
    t = torch.tensor(v, dtype=torch.float64)
    return t / t.norm()


def test_class_means_and_zero_variance() -> None:
    x, y = torch.tensor([1.0, 3.0, 5.0, 5.0]), torch.tensor([0, 0, 1, 1])
    assert CP.class_means(x, y) == (2.0, 5.0)  # class 1 has zero variance: its mean is its value
    with pytest.raises(AssertionError):
        CP.class_means(x, torch.ones(4, dtype=torch.long))  # an empty class has no mean


def test_targets_opposite_class_and_2x() -> None:
    y, means = torch.tensor([0, 1]), (-1.0, 3.0)
    assert CP.targets(y, means, 1.0).tolist() == [3.0, -1.0]
    assert CP.targets(y, means, 2.0).tolist() == [7.0, -5.0]  # mu_opp + (mu_opp - mu_own)


def test_set_coord_sets_the_coordinate_and_keeps_the_complement() -> None:
    g = torch.Generator().manual_seed(0)
    h, u, tg = torch.randn(5, 4, generator=g, dtype=torch.float64), unit([1.0, 2.0, -1.0, 0.5]), torch.tensor([0.0, 1.0, -2.0, 3.0, 0.5])
    out = CP.set_coord(h, u, tg)
    assert torch.allclose(out @ u, tg.double(), atol=1e-12)
    perp = lambda z: z - (z @ u)[:, None] * u
    assert torch.allclose(perp(out), perp(h), atol=1e-12)
    assert torch.allclose(CP.set_coord(h, u, h @ u), h)  # a target equal to h.u changes nothing


def test_shift_along_takes_the_same_step_size_along_r() -> None:
    h, u, r = torch.tensor([[2.0, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=torch.float64), unit([1, 0, 0]), unit([0, 0, 1])
    tg = torch.tensor([-1.0, 1.5])
    out = CP.shift_along(h, u, tg, r)
    assert torch.allclose(out - h, torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]], dtype=torch.float64))
    assert torch.allclose((out - h).norm(dim=1), (CP.set_coord(h, u, tg) - h).norm(dim=1))


def test_edit_fn_patches_only_its_positions_and_rows() -> None:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 5, 3, generator=g)  # a batch of 2 windows of the 4 targets' rows 2-3
    u, tg = unit([0, 1, 0]).float(), torch.tensor([9.0, 9.0, 7.0, -7.0], dtype=torch.float64)
    out = CP.edit_fn([CP.Site(3, u, tg)], slice(2, 4))(x)
    assert torch.allclose(out[:, 3, 1], torch.tensor([7.0, -7.0])) and torch.equal(out[:, 3, [0, 2]], x[:, 3, [0, 2]])
    keep = [0, 1, 2, 4]
    assert torch.equal(out[:, keep], x[:, keep])
    r = unit([0, 0, 1]).float()
    out_r = CP.edit_fn([CP.Site(3, u, tg, r)], slice(2, 4))(x)
    assert torch.allclose(out_r[:, 3, 2] - x[:, 3, 2], (tg[2:].float() - x[:, 3, 1]).abs(), atol=1e-6)
    assert torch.equal(out_r[:, 3, :2], x[:, 3, :2])


def test_flips_ties_are_no_flip_and_q1_bar_is_inclusive() -> None:
    d0 = torch.tensor([1.0, -1.0, 0.0, 2.0, 1.0, -3.0])
    d1 = torch.tensor([-1.0, 2.0, 1.0, 0.0, 0.5, -1.0])
    assert CP.flips(d0, d1).tolist() == [True, True, False, False, False, False]
    q1 = CP.flip_rate(d0[:4], d1[:4])
    assert q1 == 0.5
    bars = {"P1": 0.9, "P2": 0.9, "P3": 0.5}
    assert CL.verdict(1, 0.95, 0.95, q1, bars) == "reuses C"  # Q1 = 0.50 passes
    assert CL.verdict(1, 0.95, 0.95, 0.4999, bars) == "not through C"
    assert math.isnan(CP.flip_rate(torch.zeros(0), torch.zeros(0)))
    assert CL.do_nothing_ok([0.0, 0.1], 0.1) and not CL.do_nothing_ok([0.0, 0.1001], 0.1)
