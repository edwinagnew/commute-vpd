"""commute.share_reference's continuous built change, and the size search over it with a giant neuron (the repair of
cell P3.10's reference), on made-up numbers: the edge cases filed in
experiments/qwen_share_reference_repair/results/predictions.json ("edge_cases")."""

from __future__ import annotations

import torch

from commute import share_reading as SR
from commute import share_reference as RF
from commute.qwen_vpd import SITES, site_name

ORDER = [(0, 3), (1, 0), (0, 1), (1, 2), (0, 0)]


def test_turned_down():
    assert RF.turned_down(ORDER, 0) == []
    assert RF.turned_down(ORDER, 3.0) == [(0, 3, 0.0), (1, 0, 0.0), (0, 1, 0.0)]
    assert RF.turned_down(ORDER, 2.25) == [(0, 3, 0.0), (1, 0, 0.0), (0, 1, 0.75)]
    assert RF.turned_down(ORDER, 0.5) == [(0, 3, 0.5)]
    assert RF.turned_down(ORDER, 5.0) == [(lyr, c, 0.0) for lyr, c in ORDER]


def test_built_change_is_a_mask_on_gate_rows():
    shapes = {s: (4, 3) for s in SITES}
    W0 = {s: torch.randn(4, 3) for s in SITES}
    dW = RF.built_change(W0, RF.turned_down(ORDER, 2.25), shapes)
    g0, g1 = site_name(0, "gate"), site_name(1, "gate")
    assert torch.equal(W0[g0][3] + dW[g0][3], torch.zeros(3))  # switched off: exactly 0
    assert torch.allclose(W0[g0][1] + dW[g0][1], 0.75 * W0[g0][1])  # turned down partway
    assert torch.equal(dW[g0][[0, 2]], torch.zeros(2, 3)) and torch.equal(dW[g1][[1, 2, 3]], torch.zeros(3, 3))
    assert all(not dW[s].any() for s in SITES if s not in (g0, g1))


def test_search_lands_inside_a_giant_neuron():
    """Neurons 1-3 add 0.01 each; the 4th is a giant (4 nats when fully off, quadratic in how far it is turned down);
    a whole count can only reach 0.03 or 4.03, a continuous size lands inside the giant within 1.05."""
    def f(x: float) -> float:
        return 0.01 * x if x <= 3 else 0.03 + 4 * min(x - 3, 1) ** 2 + 0.01 * max(x - 4, 0)

    whole = SR.size_search(f, 0.5, 28, 1, 1.05, 20, cap=615, whole=True)
    s = SR.size_search(f, 0.5, 28, 1, 1.05, 20, cap=615)
    assert not whole["matched"] and s["matched"] and 3 < s["x"] < 4 and len(s["evals"]) <= 20
