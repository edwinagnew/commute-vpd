"""E97 Amendment 1's edge reads (`commute.edges`) on a tiny random LlamaSimpleMLP with the 67M's shape where the reads
depend on it (4 blocks, 6 heads of 128, rotary on the whole head, no flash attention) and a small vocabulary and MLP:
stage 0's checks by construction, on CPU. No real weights are loaded."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import pytest
import torch
from param_decomp.pretrain.models.llama_simple_mlp import (
    LlamaSimpleMLP,
    LlamaSimpleMLPConfig,
)

from commute import edge_rules as ER
from commute import edges as EG
from commute import locator as L

V, T, N = 40, 10, 6
KEYS = [f"h.{b}.{m}" for b in range(4) for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj")]


class Net:
    """`steering.Net`'s interface over the tiny model: `net`, `device`, `at(dW)`, `run(toks)`."""

    def __init__(self) -> None:
        torch.manual_seed(0)
        cfg = LlamaSimpleMLPConfig(model_type="LlamaSimpleMLP", n_layer=4, n_head=6, n_embd=768, n_intermediate=48, vocab_size=V, block_size=32,
                                   n_ctx=32, n_key_value_heads=6, rotary_dim=128, flash_attention=False)  # fmt: skip
        self.net, self.device = LlamaSimpleMLP(cfg).requires_grad_(False), "cpu"
        with torch.no_grad():
            for p in self.net.parameters():
                p.mul_(2.0)  # larger than init, so heads attend; larger made fp32 noise and IG's 8 steps fail on logits of +-200
        self.W0 = {k: self.weight(k).detach().clone() for k in KEYS}

    def weight(self, k: str) -> torch.Tensor:
        w = self.net.get_submodule(k).weight
        assert isinstance(w, torch.Tensor)
        return w

    @contextmanager
    def at(self, dW: Any) -> Iterator[None]:
        if dW is not None:
            for k, w in self.W0.items():
                self.weight(k).copy_(w + dW[k])
        try:
            yield
        finally:
            for k, w in self.W0.items():
                self.weight(k).copy_(w)

    @torch.no_grad()
    def run(self, toks: torch.Tensor, edit: Any = None, capture: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.net(toks)[0].float(), {}


@pytest.fixture(scope="module")
def setup() -> dict[str, Any]:
    net = Net()
    g = torch.Generator().manual_seed(1)
    toks = torch.randint(0, V, (N, T), generator=g)
    win, pos = torch.arange(N).repeat(2), torch.tensor([T - 1] * N + [5] * N)
    site = L.Site(toks, win, pos, torch.randint(0, V, (2 * N,), generator=g), win, torch.randint(0, V, (2 * N,), generator=g))
    bad = toks.clone()
    bad[:, 3] = (bad[:, 3] + 1) % V
    dW = {k: 0.1 * torch.randn(w.shape, generator=g) * w.std() for k, w in net.W0.items()}
    return {"net": net, "site": site, "gain": EG.Pair(site, None, "lo"), "pair": EG.Pair(site, bad, "gap"), "dW": dW}


def one() -> torch.Tensor:
    return torch.ones(ER.VALID.shape)


def test_no_patch_is_the_model(setup: dict[str, Any]) -> None:
    net, dW = setup["net"], setup["dW"]
    assert torch.allclose(EG.per_read(net, setup["gain"], dW, one()), L.read(net, setup["site"], dW)[:, 0], atol=1e-4)
    assert torch.allclose(EG.per_read(net, setup["pair"], None, one()), L.read(net, setup["site"])[:, 1], atol=1e-4)


def test_every_edge_patched_is_the_corrupted_run(setup: dict[str, Any]) -> None:
    net, dW, p = setup["net"], setup["dW"], setup["pair"]
    assert torch.allclose(EG.per_read(net, setup["gain"], dW, 0 * one()), L.read(net, setup["site"])[:, 0], atol=1e-4)
    assert torch.allclose(EG.per_read(net, p, dW, 0 * one()), L.read(net, replace(p.site, toks=p.bad), dW)[:, 1], atol=1e-4)


def test_a_last_mlp_write_reads_on_one_edge_alone(setup: dict[str, Any]) -> None:
    net = setup["net"]
    dW = {k: torch.zeros_like(w) for k, w in net.W0.items()}
    dW["h.3.mlp.down_proj"] = setup["dW"]["h.3.mlp.down_proj"] * 5
    G = float((L.read(net, setup["site"], dW)[:, 0] - L.read(net, setup["site"])[:, 0]).mean())
    if G < 0:  # a gain, as the plant's: the top edge is signed
        dW["h.3.mlp.down_proj"] *= -1
        G = float((L.read(net, setup["site"], dW)[:, 0] - L.read(net, setup["site"])[:, 0]).mean())
    assert G > 0
    rec, sc = EG.gain_exact(net, setup["gain"], dW)
    assert abs(sc["mlp3->logits"] / G - 1) < 1e-4
    assert all(abs(x / G) <= 1e-4 for e, x in sc.items() if e != "mlp3->logits")  # stage 0's bar
    assert rec["top_edge"] == "mlp3->logits" and abs(rec["joint_share"]["mlp3"] - 1) < 1e-4


def joint_and_carry(net: Any, s: L.Site, dW: dict[str, torch.Tensor], parts: list[L.Part]) -> list[tuple[float, float]]:
    r = EG.exact(net, EG.Pair(s, None, "lo"), dW, [EG.joint(ER.UP.index(L.name(p))) for p in parts])
    a0, lo1 = L.acts(net, s), L.read(net, s, dW)[:, 0]
    G = float((lo1 - L.read(net, s)[:, 0]).mean())
    return [((r["clean"] - x) / G, float((lo1 - L.read(net, s, dW, [p], a0)[:, 0]).mean()) / G) for p, x in zip(parts, r["each"], strict=True)]


def test_joint_patch_is_e97s_carry_where_the_parts_agree(setup: dict[str, Any]) -> None:
    """An MLP's output is the same in both reads. E97's head is the slice of o_proj's input, so its carry keeps the
    state's own o_proj change; ours is the head's write after o_proj. They agree exactly when o_proj did not change."""
    net, s, dW = setup["net"], setup["site"], setup["dW"]
    parts: list[L.Part] = [("head", 2, 4), ("mlp", 1, -1), ("head", 0, 0)]
    got = joint_and_carry(net, s, dW, parts)
    assert abs(got[1][0] - got[1][1]) < 1e-4
    same_o = {k: torch.zeros_like(v) if k.endswith("o_proj") else v for k, v in dW.items()}
    assert all(abs(a - b) < 1e-4 for a, b in joint_and_carry(net, s, same_o, parts))


def test_stacked_and_late_start_forwards_agree_with_one_at_a_time(setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    net, p, dW = setup["net"], setup["pair"], setup["dW"]
    Cs = [EG.single(*ER.EDGES[i]) for i in (0, 100, 500, 900, 926)] + [EG.joint(5)]
    many = EG.exact(net, p, dW, Cs)
    monkeypatch.setattr(EG, "EXACT_TOKENS", 1)
    alone = EG.exact(net, p, dW, Cs)
    direct = [float(EG.per_read(net, p, dW, C).mean()) for C in Cs]
    assert max(abs(x - y) for x, y in zip(many["each"], alone["each"], strict=True)) < 1e-5
    assert max(abs(x - y) for x, y in zip(many["each"], direct, strict=True)) < 1e-4
    assert [EG.start_of(C) for C in Cs[:5]] == [ER.block_of(ER.EDGES[i][1]) for i in (0, 100, 500, 900, 926)]
    assert EG.start_of(Cs[5]) == 0 and EG.start_of(torch.ones(ER.VALID.shape)) == ER.BLOCKS


def test_integrated_gradients_sum_to_the_gain(setup: dict[str, Any]) -> None:
    net, dW = setup["net"], setup["dW"]
    r = EG.exact(net, setup["gain"], dW, [])
    ig = EG.ig_gain(net, setup["gain"], dW, 8)
    assert abs(float(ig.sum()) - (r["clean"] - r["corrupt"])) <= 0.02 * abs(r["clean"] - r["corrupt"])
    assert float(ig[~ER.VALID].abs().sum()) == 0.0


def test_circuits_full_and_empty(setup: dict[str, Any]) -> None:
    net, p = setup["net"], setup["pair"]
    sc = EG.ig_pair(net, p, None, 4)
    assert sc.shape == ER.VALID.shape and float(sc[~ER.VALID].abs().sum()) == 0.0
    full, none = EG.cross(net, p, None, ER.VALID.clone(), -1e9), EG.cross(net, p, None, torch.zeros_like(ER.VALID), -1e9)
    assert abs(full["faith"] - 1) < 1e-5 and abs(none["faith"]) < 1e-5
    rec, mask = EG.circuit(net, p, None, [10, 927], 0.85, 4, -1e9)
    assert rec["n"] in (10, 927) and bool((mask & ~ER.VALID).sum() == 0)
    rec, mask = EG.circuit(net, p, None, [10, 927], 0.85, 4, 1e9)
    assert rec["n"] is None and rec["reason"].startswith("no gap") and not mask.any()


def test_fp64_restores_the_model(setup: dict[str, Any]) -> None:
    net = setup["net"]
    with EG.fp64(net):
        assert net.net.lm_head.weight.dtype == torch.float64 and net.net.wte.weight is net.net.lm_head.weight
        x = EG.per_read(net, setup["gain"], None, one())
    assert net.net.lm_head.weight.dtype == torch.float32
    assert torch.allclose(x, L.read(net, setup["site"])[:, 0], atol=1e-4)
