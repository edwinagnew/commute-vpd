"""E100's reads (`locator.split`, `part_reverted`, `grafted`) on the tiny random net of tests/test_edges.py (the 67M's
module names, 4 blocks of 6 heads of 128): the known-answer states of stage 0 and E99's forced pole, exact by
construction, on CPU. No real weights are loaded."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from test_edges import KEYS, N, Net, T, V

from commute import fed_rules as FR
from commute import locator as L

TH = {"reread_tol": 1e-4, "zero_tol": 1e-3}


def only(net: Net, scale: float, keep: dict[str, Any], g: torch.Generator) -> dict[str, torch.Tensor]:
    """A dW that is zero except where `keep[k]` (a mask or True) allows, of size `scale` x the weight's std."""
    out = {}
    for k, w in net.W0.items():
        d = scale * torch.randn(w.shape, generator=g) * w.std()
        m = keep.get(k, False)
        out[k] = d * m if isinstance(m, torch.Tensor) else d if m else torch.zeros_like(w)
    return out


def state(net: Net, dW: dict[str, torch.Tensor]) -> L.State:
    """The state on windows whose targets are the state's own top tokens, so its gain over W0 is positive."""
    g = torch.Generator().manual_seed(2)
    toks = torch.randint(0, V, (N, T), generator=g)
    win, pos = torch.arange(N), torch.full((N,), T - 1)
    with net.at(dW):
        tgt = net.run(toks)[0][win, pos].argmax(-1)
    s = L.Site(toks, win, pos, tgt, win)
    st = L.state(net, L.base(net, s, {}), dW)  # pyright: ignore[reportArgumentType]
    assert float(st.G[0]) > 0.05
    return st


def split(net: Net, st: L.State, names: list[str]) -> FR.Split:
    return {n: {k: v["lo"] for k, v in L.split(net, st, n).items()} for n in names}  # pyright: ignore[reportArgumentType]


@pytest.fixture(scope="module")
def net() -> Net:
    return Net()


def test_parts_partition_dW(net: Net) -> None:
    g = torch.Generator().manual_seed(3)
    dW = only(net, 0.1, dict.fromkeys(KEYS, True), g)
    total = {k: torch.zeros_like(v) for k, v in dW.items()}
    for p in L.PARTS:
        for k, v in L.grafted(dW, L.name(p)).items():
            total[k] += v
    assert all(torch.equal(total[k], dW[k]) for k in KEYS)
    r, gr = L.part_reverted(dW, "2,3+2,4"), L.grafted(dW, "2,3+2,4")
    assert all(torch.equal(r[k] + gr[k], dW[k]) for k in KEYS)
    assert torch.equal(dW["h.2.attn.q_proj"], L.part_reverted(dW, "mlp2")["h.2.attn.q_proj"])  # callers' dW untouched


def test_mlp_plant_reads_exact(net: Net) -> None:
    """A one-matrix plant in block 3's and block 1's down_proj: stage 0's construction."""
    g = torch.Generator().manual_seed(4)
    names = [L.name(p) for p in L.PARTS]
    for block in (3, 1):
        st = state(net, only(net, 10.0, {f"h.{block}.mlp.down_proj": True}, g))
        G_all = float((L.read(net, st.b.site, L.reverted(st.dW, list(st.dW))) - st.b.lo0).mean(0)[0])  # pyright: ignore[reportArgumentType]
        assert FR.plant_misses(split(net, st, names), block, G_all, TH) == []


def test_forced_pair_reads_one(net: Net) -> None:
    """E99's forced pole: only the pair's rows of q, k, v moved. own = carry = graft = 1 at the pair, 0 at a head upstream."""
    g = torch.Generator().manual_seed(5)
    keep = {}
    for m in ("q_proj", "k_proj", "v_proj"):
        mask = torch.zeros(768, 768)
        for h in (3, 4):
            mask[h * L.HEAD_DIM : (h + 1) * L.HEAD_DIM] = 1
        keep[f"h.2.attn.{m}"] = mask
    st = state(net, only(net, 3.0, keep, g))
    sp = split(net, st, ["2,3+2,4", "1,1"])
    assert FR.pole_misses(sp, "2,3+2,4", "forced", TH) == []
    assert all(abs(x) <= 1e-4 for x in sp["1,1"].values())
    forbidden = L.part_reverted(only(net, 0.3, dict.fromkeys(KEYS, True), g), "2,3+2,4")
    sp = split(net, state(net, forbidden), ["2,3+2,4"])
    assert FR.pole_misses(sp, "2,3+2,4", "forbidden", TH) == []
