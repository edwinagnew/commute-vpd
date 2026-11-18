"""commute.rho on a tiny random LlamaSimpleMLP (CPU, seconds; not the 67M): F_K with nothing swapped is F, with every
changed part swapped is M under both constructions, a turned-up column reads r = 1 (main) and 1 - (a-1)^2/(a^2+1) (b),
and an appended head or neuron is a part M lacks."""

import copy
from typing import Any

import torch

from commute import graft as G
from commute import rho
from commute import rule_retention as R


def tiny() -> R.Model:
    from param_decomp.pretrain.models.llama_simple_mlp import (
        LlamaSimpleMLP,
        LlamaSimpleMLPConfig,
    )

    torch.manual_seed(0)
    cfg = LlamaSimpleMLPConfig(model_type="LlamaSimpleMLP", block_size=16, vocab_size=40, n_layer=2, n_head=2, n_embd=2 * R.D_HEAD, n_intermediate=32,
                               n_ctx=16, n_key_value_heads=2, use_grouped_query_attention=True, flash_attention=False, rotary_dim=R.D_HEAD)  # fmt: skip
    net: Any = LlamaSimpleMLP(cfg).eval().requires_grad_(False)
    return R.Model(net)


def clone(M: R.Model) -> R.Model:
    return R.Model(copy.deepcopy(M.net))


def close(a: torch.Tensor, b: torch.Tensor) -> bool:
    return float((a - b).abs().max()) <= 1e-4 * float(b.abs().max())


def test_hybrid_ends_and_r() -> None:
    M = tiny()
    F = clone(M)
    with torch.no_grad():
        F.net.h[1].mlp.down_proj.weight[:, [3, 5]] *= 2
        F.net.h[0].attn.q_proj.weight[: R.D_HEAD] += 0.05 * torch.randn(R.D_HEAD, F.net.h[0].attn.q_proj.weight.shape[1])
    p, toks = rho.Pair(F, M), torch.randint(0, 40, (6, 10))
    cand, lacks = rho.changed(p)
    assert sorted(cand) == [("head", 0, 0), ("neuron", 1, 3), ("neuron", 1, 5)] and not lacks
    mrun = rho.m_run(p, toks)
    assert close(rho.forward(p, toks, rho.swap(p, [])), F.logits(toks))
    for m in (None, mrun):
        assert close(rho.forward(p, toks, rho.swap(p, cand), m), M.logits(toks))
    prof = rho.profile(p, [toks[:3], toks[3:]])
    for i in (3, 5):
        u = ("neuron", 1, i)
        assert abs(prof["r"]["main"][u] - 1) < 1e-9 and abs(prof["r"]["i"][u] - 1) < 1e-9 and abs(prof["r"]["b"][u] - 0.8) < 1e-9
    assert prof["r"]["main"][("head", 0, 0)] < 1 and prof["r"]["main"][("head", 0, 1)] == 1


def test_appended_parts_and_lacks() -> None:
    M = tiny()
    F = clone(M)
    rec = G.Recipe(0.02, 0.02 / 8**0.5, 1e-3, 0.1, 1, 2, 1, 0.9, 0.95, 0.1, 1.0, (2,))
    G.attach_head(F, 1, 0, rec)
    G.attach_neurons(F, 0, 4, 0, rec)
    with torch.no_grad():
        F.net.h[1].mlp.down_proj.weight[:, 7] = 0
    p, toks = rho.Pair(F, M), torch.randint(0, 40, (4, 8))
    cand, lacks = rho.changed(p)
    assert ("head", 1, 2) in cand and [u for u in cand if u[0] == "neuron"] == [("neuron", 0, 32 + i) for i in range(4)] and lacks == [("neuron", 1, 7)]
    assert all(rho.lacked_by_m(p, u) for u in cand)
    assert close(rho.forward(p, toks, rho.swap(p, [])), F.logits(toks))
    assert close(rho.forward(p, toks, rho.swap(p, cand + lacks)), M.logits(toks))
    prof = rho.profile(p, [toks])
    assert prof["r"]["main"][("neuron", 1, 7)] == 0 and prof["one_zero"][("neuron", 1, 7)] > 0


def test_contrast_on_a_turned_up_column() -> None:
    M = tiny()
    F = clone(M)
    with torch.no_grad():
        F.net.h[1].mlp.down_proj.weight[:, [3, 5]] *= 2
    p, toks = rho.Pair(F, M), torch.randint(0, 40, (8, 10))
    cand, _ = rho.changed(p)
    prof = rho.profile(p, [toks])
    th = {"kl_floor": 0.0, "order_seed": 0, "orders": 2, "tau": 0.9, "chunk": 4}
    b = rho.batches(p, toks, 4, slice(-1, None))
    for patch in (False, True):
        c = rho.contrast(p, b, cand, prof["r"], patch, th, lambda _s: None, "x2")
        assert c["reading"] == "read" and abs(c["closure_none_kept"]) < 1e-4
        assert all(abs(v - 1) < 1e-9 for v in c["rho_main"]) and all(abs(v - 0.8) < 1e-9 for v in c["rho_b"])
        assert all(set(o["K"]) <= {0, 1} for o in c["orders"])
