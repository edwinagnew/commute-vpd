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


def _one_block_gamma(sink: float | None) -> q.Gamma:
    """Γ with one random attention block at full width (3840); nothing else is read."""
    d, gen = q.GAMMA_HEADS * q.GAMMA_HEAD_DIM, torch.Generator().manual_seed(0)
    blocks = {k: torch.randn(1, d, d, generator=gen) / d**0.5 for k in ("wq", "wk", "wv")}
    blocks |= {"wo": torch.eye(d)[None], "q_gain": torch.ones(1, d), "k_gain": torch.ones(1, d)}  # wo = I: heads apart
    s = None if sink is None else torch.full((1, q.GAMMA_HEADS), sink)
    return q.Gamma(torch.zeros(1, 1), torch.zeros(1), blocks, {}, {}, {}, torch.zeros(1), sink=s)


def _attend(gamma: q.Gamma) -> torch.Tensor:
    """The attention's own output (residual removed) on 3 random positions, with cos 1 and sin 0 (no rotation)."""
    x = torch.randn(1, 3, q.GAMMA_HEADS * q.GAMMA_HEAD_DIM, generator=torch.Generator().manual_seed(1))
    one, zero = torch.ones(3, q.GAMMA_HEAD_DIM), torch.zeros(3, q.GAMMA_HEAD_DIM)
    return q._attention(gamma, 0, x, one, zero) - x


def test_sink_at_minus_infinity_is_no_sink():
    assert torch.equal(_attend(_one_block_gamma(float("-inf"))), _attend(_one_block_gamma(None)))


def test_sink_shrinks_each_head_by_its_share():
    """A sink logit s scales a head's output at a position by Σexp(l) / (exp(s) + Σexp(l)). At the first position (one
    key, logit l) that is exp(l) / (exp(s) + exp(l)): a huge s leaves ~0 of it; s = 0 a share strictly in (0, 1)."""
    plain, sunk, flooded = (_attend(_one_block_gamma(s)) for s in (None, 0.0, 1e4))
    assert flooded.abs().max() < 1e-6 * plain.abs().max()
    share = (sunk[0, 0] / plain[0, 0]).view(q.GAMMA_HEADS, q.GAMMA_HEAD_DIM)  # position 0, one row per head
    assert torch.allclose(share, share[:, :1].expand_as(share), rtol=1e-3)  # one share per head
    assert ((share > 0) & (share < 1)).all()
