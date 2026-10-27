"""CPU checks of qwen_vpd's shape logic on tiny tensors; nothing here loads the model or the decomposition."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from commute import qwen_vpd as q


def test_sites_are_layer_major_and_parse_back():
    assert len(q.SITES) == 196
    assert q.SITES[:2] == ("layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj")
    assert q.SITES[4] == "layers.0.mlp.gate_proj"
    assert all(q.site_name(*q.parse_site(s)) == s for s in q.SITES)


def test_rms_matches_torch_rms_norm():
    x = torch.randn(2, 3, 8)
    assert torch.allclose(q._rms(x), F.rms_norm(x, (8,), eps=q.RMS_EPS), atol=1e-6)


def test_readout_arms_read_h_and_minus_h():
    C = 3
    gamma = q.Gamma(torch.zeros(1, 1), torch.zeros(1), {}, {}, {}, {}, torch.zeros(1))
    # positive arm: scale 1, bias 0, out 2; negative arm: scale 1, bias 0, out 3; free bias 0.5
    gamma.readouts["s"] = torch.tensor([[1.0] * C, [0.0] * C, [2.0] * C, [1.0] * C, [0.0] * C, [3.0] * C, [0.5] * C])
    x = torch.eye(2, 2)
    V = torch.tensor([[0.5, -0.5, 4.0], [0.0, 0.0, 0.0]])  # h for the first row = (0.5, -0.5, 4.0)
    r = q.readout(gamma, "s", x, V)
    assert torch.allclose(r[0], torch.tensor([2 * 0.5 + 0.5, 3 * 0.5 + 0.5, 2 * 1.0 + 0.5]))
    assert torch.allclose(r[1], torch.full((C,), 0.5))


def test_stochastic_masks_lie_between_g_and_one():
    g = {"s": torch.tensor([[[0.0, 0.5, 1.0]]])}
    masks, deltas = q.stochastic_masks(g, torch.Generator().manual_seed(0))
    assert masks["s"].shape == (1, 1, 3) and deltas["s"].shape == (1, 1)
    assert (masks["s"] >= g["s"]).all() and (masks["s"] <= 1).all() and masks["s"][0, 0, 2] == 1
