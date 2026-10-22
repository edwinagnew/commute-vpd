"""CPU checks of shift_readout.py on hand-made shifts."""

from __future__ import annotations

import math

import torch

from commute import shift_readout as SR

S = {"h.0.mlp.c_fc": torch.tensor([0.5, -0.2, 0.0, 0.02]), "h.3.attn.q_proj": torch.tensor([-0.06, 0.3, 0.0])}
ALIVE = {"h.0.mlp.c_fc": torch.tensor([True, True, False, True]), "h.3.attn.q_proj": torch.tensor([True, True, True])}


def test_counts_and_mass():
    c = SR.counts(S, ALIVE, (0.01, 0.05, 0.1))
    assert c == {"alive": 6, "up": 3, "down": 2, "abs_gt_0.01": 5, "abs_gt_0.05": 4, "abs_gt_0.1": 3}
    m = SR.mass_split(S)
    assert math.isclose(m["by_layer"]["0"], 0.72 / 1.08, rel_tol=1e-6) and math.isclose(m["by_layer"]["3"], 0.36 / 1.08, rel_tol=1e-6)
    assert math.isclose(m["by_sign"]["up"], 0.82 / 1.08, rel_tol=1e-6) and math.isclose(m["by_sign"]["down"], 0.26 / 1.08, rel_tol=1e-6)


def test_on_layer_and_top_k():
    assert torch.equal(SR.on_layer(S, 3)["h.0.mlp.c_fc"], torch.zeros(4))
    assert torch.equal(SR.on_layer(S, 3)["h.3.attn.q_proj"], S["h.3.attn.q_proj"])
    t = SR.top_k(S, 2)
    assert torch.equal(t["h.0.mlp.c_fc"], torch.tensor([0.5, 0.0, 0.0, 0.0]))
    assert torch.equal(t["h.3.attn.q_proj"], torch.tensor([0.0, 0.3, 0.0]))
    assert torch.equal(SR.top_k(S, 100)["h.0.mlp.c_fc"], S["h.0.mlp.c_fc"])


def test_spearman_and_write_energy():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert math.isclose(SR.spearman(x, x ** 3), 1.0) and math.isclose(SR.spearman(x, -x), -1.0)
    assert math.isnan(SR.spearman(x, torch.ones(4)))
    U = torch.tensor([[2.0, 0.0], [0.0, 1.0]])  # (C, d_out)
    V = torch.tensor([[1.0, 0.0], [0.0, 3.0]])  # (d_in, C)
    dW = torch.tensor([[0.5, 7.0], [7.0, -0.25]])
    e = SR.write_energy({"m": (U, V)}, {"m": dW})["m"]
    assert torch.allclose(e, torch.tensor([0.5, 0.25]))
