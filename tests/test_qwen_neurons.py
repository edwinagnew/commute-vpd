"""E102's rules and the generalised hooks on CPU: every edge case of the filed rules, the do-nothing state, and a toy
SwiGLU stack for `neuron_r.hooked` / `neuron_r.firing_grads` (no model is loaded)."""

from __future__ import annotations

import math

import torch
from torch import nn

from commute import carrying_jobs as CJ
from commute import neuron_r as NR
from commute import qwen_neurons as QN

# ----------------------------------------------------------------------------- sets


def test_top_set_keeps_positive_only_and_ties_by_index() -> None:
    s = torch.tensor([0.0, 2.0, -1.0, 2.0, 0.5])
    assert QN.top_set(s, 3) == [1, 3, 4]
    assert QN.top_set(s, 10) == [1, 3, 4]


def test_top_set_do_nothing_is_empty() -> None:
    assert QN.top_set(torch.zeros(QN.N, dtype=torch.float64), 1024) == []


def test_overlap_edges() -> None:
    assert QN.overlap([1, 2, 3, 4], [3, 4, 5]) == 0.5
    assert math.isnan(QN.overlap([], [1]))
    assert math.isnan(QN.overlap([1], []))


def test_random_draws_match_blocks_and_are_empty_for_no_set() -> None:
    a = [5, 6144 + 3, 6144 + 9, 27 * 6144]
    g = torch.Generator().manual_seed(0)
    d = CJ.block_matched(a, set(a), g, QN.WIDTH)
    assert CJ.blocks_of(d, QN.WIDTH) == CJ.blocks_of(a, QN.WIDTH) == {0: 1, 1: 2, 27: 1}
    assert not set(d) & set(a)
    assert QN.random_overlaps([], [1], 5, g) == [] and QN.random_engaged([], [0.5] * QN.N, 5, g) == []
    assert len(QN.random_overlaps(a, a, 5, g)) == 5


def test_block_helpers_keep_the_67m_default() -> None:
    assert CJ.blocks_of([0, 3071, 3072]) == {0: 2, 1: 1}


# ----------------------------------------------------------------------------- the lean rule


DRAWS = [0.1, 0.12, 0.11, 0.09]  # sd 0.0129


def test_above_clear_case() -> None:
    r = QN.above([0.6, 0.55], 0.15, DRAWS)
    assert r["verdict"] == "above" and r["floor"] == 0.15


def test_above_unread_edges() -> None:
    for seeds, g, d in (([0.6, None], 0.1, DRAWS), ([0.6, 0.5], None, DRAWS), ([0.6, 0.5], 0.1, [0.1]),
                        ([0.6, math.nan], 0.1, DRAWS), ([0.6], 0.1, DRAWS), ([None, None], None, [])):
        assert QN.above(seeds, g, d)["verdict"] == "unread"


def test_above_is_strict_at_the_floor_and_the_band() -> None:
    assert QN.above([0.15, 0.4], 0.15, DRAWS)["verdict"] == "not above"  # one seed at the floor
    r = QN.above([0.3, 0.2], 0.12, DRAWS)  # gap 0.13 against band 0.1: above
    assert r["verdict"] == "above"
    assert QN.above([0.5, 0.2], 0.12, DRAWS)["verdict"] == "not above"  # gap 0.23 < spread 0.3


def test_above_zero_spread_needs_twice_the_draw_sd() -> None:
    sd = math.sqrt(sum((x - 0.105) ** 2 for x in DRAWS) / 3)
    assert QN.above([0.12 + 2 * sd - 1e-4] * 2, 0.12, DRAWS)["verdict"] == "not above"
    assert QN.above([0.12 + 2 * sd + 1e-4] * 2, 0.12, DRAWS)["verdict"] == "above"


def test_gaussian_as_both_anchors_cannot_pass() -> None:
    assert QN.above([0.2, 0.2], 0.2, DRAWS)["verdict"] == "not above"


# ----------------------------------------------------------------------------- control (i)


def test_control_i() -> None:
    assert QN.control_i(0.6, [0.1, 0.2], 0.15, 0.2)["passed"]
    assert not QN.control_i(0.39, [0.1, 0.2], 0.15, 0.2)["C1a"]  # 0.19 over the draws
    assert QN.control_i(0.6, [0.1, 0.2], 0.41, 0.2)["C1b"] is False  # Gaussian within 0.2 of the halves
    assert not QN.control_i(math.nan, [0.1, 0.2], 0.1, 0.2)["passed"]
    assert not QN.control_i(0.6, [0.1], 0.1, 0.2)["passed"]  # under two draws
    assert not QN.control_i(0.6, [0.1, 0.2], None, 0.2)["passed"]  # no Gaussian set


def test_k_half() -> None:
    nums = {64: 0.1, 1024: 0.3, QN.N: 0.5}
    assert QN.k_half(nums, {"gain": 1.0, "share": 1.0}, 0.1) == 1024
    assert QN.k_half(nums, {"gain": 1.0, "share": None}, 0.1) is None  # gain within its row noise
    assert QN.k_half({64: 0.0, QN.N: -0.2}, {"gain": 1.0, "share": 1.0}, 0.1) is None
    assert QN.k_half(nums, {"gain": 10.0, "share": 1.0}, 0.1) is None  # neuron share 0.05 under the floor


# ----------------------------------------------------------------------------- the hooks, on a toy SwiGLU stack


class MLP(nn.Module):
    def __init__(self, d: int, w: int) -> None:
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(d, w, bias=False), nn.Linear(d, w, bias=False), nn.Linear(w, d, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def toy() -> tuple[list[MLP], torch.Tensor]:
    torch.manual_seed(0)
    return [MLP(4, 6), MLP(4, 6)], torch.randn(3, 5, 4)


def stack(mlps: list[MLP], x: torch.Tensor) -> torch.Tensor:
    for m in mlps:
        x = x + m(x)
    return x


def test_hooked_puts_back_what_was_there() -> None:
    mlps, x = toy()
    with torch.no_grad():
        y, xs, acts = NR.hooked(mlps, lambda: stack(mlps, x))
        y2, _, _ = NR.hooked(mlps, lambda: stack(mlps, x), x_in=xs, a_in=acts)
    assert torch.equal(y, y2)
    assert torch.allclose(acts[1], mlps[1].act_fn(mlps[1].gate_proj(xs[1])) * mlps[1].up_proj(xs[1]))


def test_firing_grads_is_the_total_derivative() -> None:
    mlps, x = toy()
    g, a = NR.firing_grads(mlps, lambda: stack(mlps, x).sum())
    eps, l, i = 1e-3, 0, (1, 2, 3)

    def bumped(sign: float) -> float:
        a_l = a[l].clone()
        a_l[i] += sign * eps
        with torch.no_grad():
            return float(stack_with(mlps, x, l, a_l).sum())

    assert abs((bumped(1) - bumped(-1)) / (2 * eps) - float(g[l][i])) < 1e-3


def stack_with(mlps: list[MLP], x: torch.Tensor, l: int, a_l: torch.Tensor) -> torch.Tensor:
    """The stack with block l's firing set to a_l (later blocks computed as usual)."""
    for k, m in enumerate(mlps):
        x = x + (m.down_proj(a_l) if k == l else m(x))
    return x
