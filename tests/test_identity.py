"""CPU checks of the component-identity measures and the revert / ablate hook."""

import torch
from torch import nn

from commute import identity


def _uv(d_out=7, d_in=5, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(d_out, n, generator=g, dtype=torch.float64), torch.randn(d_in, n, generator=g, dtype=torch.float64)


def test_change_stats_gauge_free_and_exact():
    U, V = _uv()
    s = torch.tensor([2.0, -0.5, 3.0, -1.0], dtype=torch.float64)
    st = identity.change_stats(U, V, U * s, V / s)  # pure gauge: same subcomponents
    assert torch.allclose(st["cos_u"], torch.ones(4, dtype=torch.float64))
    assert torch.allclose(st["cos_v"], torch.ones(4, dtype=torch.float64))
    assert torch.allclose(st["scale"], torch.ones(4, dtype=torch.float64))
    assert st["rel_change"].max() < 1e-7
    U2, V2 = _uv(seed=1)
    st = identity.change_stats(U, V, U2, V2)
    for j in range(4):
        B, B2 = torch.outer(U[:, j], V[:, j]), torch.outer(U2[:, j], V2[:, j])
        assert abs(float(st["rel_change"][j]) - float((B2 - B).norm() / B.norm())) < 1e-10
    # changing gauge on the "after" side leaves every statistic unchanged
    st2 = identity.change_stats(U, V, U2 * s, V2 / s)
    for key in st:
        assert torch.allclose(st[key], st2[key])


def test_profile_retrieval():
    g = torch.Generator().manual_seed(0)
    before = torch.randn(500, 6, generator=g)
    after = before.clone()
    after[:, 0] = -3 * before[:, 0]  # sign and scale flip: still itself
    after[:, 1] = before[:, 4] + 0.01 * torch.randn(500, generator=g)  # became subcomponent 4
    r = identity.profile_retrieval(before, after, torch.tensor([0, 1, 2]))
    assert r["rank"][0] == 1 and r["rank"][2] == 1 and r["rank"][1] > 1
    assert abs(float(r["r_self"][0]) - 1) < 1e-9 and float(r["r_self"][2]) > 0.999
    assert float(r["r_other_max"][1]) > 0.99


def test_topk_jaccard():
    a = torch.arange(10.0)
    assert identity.topk_jaccard(a, -a, 3) == 1.0
    assert identity.topk_jaccard(a, torch.flip(a, [0]), 3) == 0.0


def test_edit_reverts_and_ablates_per_group():
    torch.manual_seed(0)
    d_in, d_out, n, b, T = 5, 7, 4, 3, 2
    U, V = _uv(d_out, d_in, n)
    U2, V2 = U + 0.1 * torch.randn_like(U), V + 0.1 * torch.randn_like(V)
    U, V, U2, V2 = U.float(), V.float(), U2.float(), V2.float()
    lin = nn.Linear(d_in, d_out, bias=False)
    with torch.no_grad():
        lin.weight.copy_(U2 @ V2.T + 0.3)  # "after" weights, plus a fixed rest
    x = torch.randn(b, T, d_in)
    cidx = torch.tensor([2, 0])
    h = lin.register_forward_hook(identity.Edit([(U2, V2, 1.0), (U, V, -1.0)], cidx))
    out = lin(torch.cat([x, x]))  # m = 2 groups of b rows
    h.remove()
    for g, j in enumerate(cidx.tolist()):
        W = U2 @ V2.T + 0.3 - torch.outer(U2[:, j], V2[:, j]) + torch.outer(U[:, j], V[:, j])
        assert torch.allclose(out[g * b:(g + 1) * b], x @ W.T, atol=1e-5)
    h = lin.register_forward_hook(identity.Edit([(U2, V2, 1.0)], torch.tensor([1])))
    out = lin(x)
    h.remove()
    assert torch.allclose(out, x @ (U2 @ V2.T + 0.3 - torch.outer(U2[:, 1], V2[:, 1])).T, atol=1e-5)


def test_log_odds():
    logits = torch.log(torch.tensor([[0.2, 0.8]]))
    assert abs(float(identity.log_odds(logits, 1)) - float(torch.log(torch.tensor(4.0)))) < 1e-5


def test_revert_written_without_cancellation_equals_revert():
    torch.manual_seed(1)
    U, V = (t.float() for t in _uv())
    U2, V2 = U + 0.01 * torch.randn_like(U), V + 0.01 * torch.randn_like(V)
    lin = nn.Linear(5, 7, bias=False)
    with torch.no_grad():
        lin.weight.copy_(U2 @ V2.T)
    x = torch.randn(2, 3, 5)
    cidx = torch.tensor([3])
    h = lin.register_forward_hook(identity.Edit([(U2, V2 - V, 1.0), (U2 - U, V, 1.0)], cidx))
    a = lin(x)
    h.remove()
    h = lin.register_forward_hook(identity.Edit([(U2, V2, 1.0), (U, V, -1.0)], cidx))
    b = lin(x)
    h.remove()
    assert torch.allclose(a, b, atol=1e-5)


def test_log_odds_does_not_saturate():
    logits = torch.tensor([[0.0, 30.0, 0.0], [0.0, 40.0, 0.0]])
    lo = identity.log_odds(logits, 1)
    assert torch.allclose(lo, torch.tensor([30.0, 40.0]) - torch.log(torch.tensor(2.0)), atol=1e-4)


def test_scale_grads_matches_autograd():
    torch.manual_seed(2)
    U, V = _uv(6, 4, 3)
    W = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    x = torch.randn(5, 4, dtype=torch.float64)
    (torch.tanh(x @ W.T) ** 2).sum().backward()
    s = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    W2 = W.detach() + (U * s) @ V.T  # B_c -> (1 + s_c) B_c
    (torch.tanh(x @ W2.T) ** 2).sum().backward()
    assert torch.allclose(identity.scale_grads(W.grad, U, V), s.grad, atol=1e-10)
