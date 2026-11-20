"""commute.gains on a toy LlamaSimpleMLP (CPU): gains of 1 change no bit and export a zero change; a gain state's forward
equals the base model plus its exported weight change; training moves only the gains and stops when asked."""

import copy
from typing import Any

import torch
import torch.nn.functional as F
from param_decomp.pretrain.models.llama_simple_mlp import (
    LlamaSimpleMLP,
    LlamaSimpleMLPConfig,
)

from commute import gains as G

V, T = 40, 12


def toy() -> Any:
    torch.manual_seed(0)
    cfg = LlamaSimpleMLPConfig(model_type="LlamaSimpleMLP", block_size=16, vocab_size=V, n_layer=2, n_head=3, n_embd=24, n_intermediate=32,
                               n_ctx=16, n_key_value_heads=3, flash_attention=False, rotary_dim=8)  # fmt: skip
    return LlamaSimpleMLP(cfg).eval()


def logits(net: Any, toks: torch.Tensor) -> torch.Tensor:
    out = net(toks)
    return out[0] if isinstance(out, tuple) else out


def weights(net: Any) -> dict[str, torch.Tensor]:
    keys = [G.down_key(l) for l in range(len(net.h))] + [G.o_key(l) for l in range(len(net.h))] + [f"h.{l}.mlp.c_fc" for l in range(len(net.h))]
    return {k: net.get_submodule(k).weight.detach().clone() for k in keys}


def toks() -> torch.Tensor:
    return torch.randint(0, V, (5, T), generator=torch.Generator().manual_seed(1))


@torch.no_grad()
def test_ones_change_nothing() -> None:
    net, x = toy(), toks()
    g = G.Gains.ones(net)
    base = logits(net, x)
    with G.applied(net, g):
        on = logits(net, x)
    assert torch.equal(base, on)
    assert torch.equal(logits(net, x), base)  # hooks removed
    d = G.delta(g, weights(net))
    assert set(d) == set(weights(net)) and all(not v.any() for v in d.values())


@torch.no_grad()
def test_gain_state_is_its_weight_change() -> None:
    net, x = toy(), toks()
    gen = torch.Generator().manual_seed(2)
    g = G.Gains(1 + 0.3 * torch.randn(2, 32, generator=gen), 1 + 0.5 * torch.randn(2, 3, generator=gen))
    with G.applied(net, g):
        on = logits(net, x)
    W0 = weights(net)
    d = G.delta(g, W0)
    assert not d["h.0.mlp.c_fc"].any() and d["h.1.attn.o_proj"].any()
    moved = copy.deepcopy(net)
    for k, v in d.items():
        moved.get_submodule(k).weight.add_(v)
    assert torch.allclose(logits(moved, x), on, atol=1e-5)
    perm = G.Gains(g.mlp, g.head[:, [1, 2, 0]])  # the head blocks must be in o_proj's column order
    with G.applied(net, perm):
        assert not torch.allclose(logits(net, x), on, atol=1e-3)


def test_train_moves_only_gains_and_stops() -> None:
    net, x = toy(), toks()
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    flags = {n: p.requires_grad for n, p in net.named_parameters()}
    g = G.Gains.ones(net)
    target = torch.full((x.shape[0],), 7)
    seen: list[int] = []

    def loss_of(t: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        loss = F.cross_entropy(logits(net, t)[:, -1], target)
        return loss, {"ce": float(loss)}

    def check(step: int, gg: G.Gains) -> bool:
        seen.append(step)
        return step >= 20

    rows = G.train(net, g, lambda s: x, loss_of, lr=3e-2, steps=100, grad_clip=1.0, check_every=10, on_check=check, log=lambda m: None)
    assert seen == [1, 10, 20] and len(rows) == 20 and rows[-1]["loss"] < rows[0]["loss"]
    assert (g.mlp != 1).any() and (g.head != 1).any() and not g.mlp.requires_grad
    assert all(torch.equal(p, before[n]) for n, p in net.named_parameters())
    assert {n: p.requires_grad for n, p in net.named_parameters()} == flags
