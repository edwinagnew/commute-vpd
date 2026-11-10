"""Steering in the fair form the user set on 2026-11-09 (items 1-4 for cells P2.4 and P2.5): a pooled and a per-key vector,
a (layer, strength) chosen on fitting windows, norm-matched random vectors, cosines and off-target KL. First run as E135,
whose `experiments/steering_copy_lookup/run.py` stays as it ran; `pick`, `per_key_mean`, `per_window`, `random_like` and
`off_kl` are moved from there unchanged, and the vectors and reads are batched with the window keys passed in, so E140 can
key by the word before the trigger and read 9,600 fitting windows.

The target model only, fp32 (`steering.Net`). A vector at block l is the mean, over fitting windows, of (tuned - W0) block l's
input at the answer position `pos`; a per-key vector is that mean over the fitting windows of one key. Steering adds
strength x vector to W0's block-l input at `pos`, nothing else.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from commute import steering as ST
from commute.revival_kv import kl_rows

LAYERS = tuple(range(ST.N_LAYERS))
BATCH = 64


def key_index(keys: Tensor, pool: Tensor) -> Tensor:
    """(N,) the index in `pool` (K,) of each window's key (N,); asserts every key is in the pool."""
    hits = keys[:, None] == pool[None]
    assert bool((hits.sum(1) == 1).all()), "a window's key is not in the pool"
    return hits.long().argmax(1)


def per_key_mean(diff: Tensor, idx: Tensor, n_keys: int) -> Tensor:
    """(n_keys, d): the mean of diff (N, d) over each key's windows (E135). An empty key would divide by 0: callers check."""
    out = torch.zeros(n_keys, diff.shape[1], device=diff.device, dtype=diff.dtype).index_add_(0, idx, diff)
    return out / torch.bincount(idx, minlength=n_keys).to(diff.dtype)[:, None]


def per_window(v: Tensor, idx: Tensor) -> Tensor:
    """A pooled vector (d,) as it is (it broadcasts over windows); a per-key vector (K, d) indexed by each window's key."""
    return v if v.ndim == 1 else v[idx]


def vectors(net: ST.Net, toks: Tensor, idx: Tensor, n_keys: int, dW: ST.DW, pos: int) -> dict[str, dict[int, Tensor]]:
    """Per layer, the pooled (d,) and per-key (K, d) means of tuned minus W0 block-l input at `pos` over `toks`."""
    diffs: dict[int, list[Tensor]] = {l: [] for l in LAYERS}
    for s in range(0, len(toks), BATCH):
        b = toks[s : s + BATCH]
        with net.at(dW):
            c1 = net.run(b, capture=True)[1]
        c0 = net.run(b, capture=True)[1]
        for l in LAYERS:
            diffs[l].append((c1[f"in{l}"][:, pos] - c0[f"in{l}"][:, pos]).double())
    d ={l: torch.cat(x) for l, x in diffs.items()}
    i = idx.to(d[0].device)
    assert bool((torch.bincount(i, minlength=n_keys) > 0).all()), "a key has no fitting window"
    return {"pooled": {l: x.mean(0).float() for l, x in d.items()}, "perkey": {l: per_key_mean(x, i, n_keys).float() for l, x in d.items()}}


def steer(v: Tensor, layer: int, strength: float, idx: Tensor, pos: int) -> ST.Edit:
    return {f"in{layer}": ST.add_at(pos, strength * per_window(v, idx.to(v.device)))}


@torch.no_grad()
def logits_steered(net: ST.Net, toks: Tensor, idx: Tensor, pos: int, v: Tensor, layer: int, strength: float) -> Tensor:
    """(N, V) fp32 logits at `pos`, on the net's device, of W0 steered by v at (layer, strength), batched with each batch's
    own keys."""
    return torch.cat([net.run(toks[s : s + BATCH], steer(v, layer, strength, idx[s : s + BATCH], pos))[0][:, pos]
                      for s in range(0, len(toks), BATCH)])  # fmt: skip


def pick(rises: dict[tuple[int, float], float]) -> tuple[int, float] | None:
    """The filed selection: the largest rise, ties to the lower layer, then the lower strength. A nan among the rises
    leaves the setting undefined (None), so it cannot be silently ordered (E135)."""
    if any(math.isnan(v) for v in rises.values()):
        return None
    return max(rises, key=lambda ls: (rises[ls], -ls[0], -ls[1]))


def random_like(v: Tensor, gen: torch.Generator) -> Tensor:
    """A Gaussian direction with v's shape, scaled to v's norm (row by row for a per-key vector) (E135)."""
    g = torch.randn(v.shape, generator=gen).to(v.device, v.dtype)
    return g / g.norm(dim=-1, keepdim=True) * v.norm(dim=-1, keepdim=True)


def cosine_matrix(vs: list[Tensor]) -> Tensor:
    """(n, n) float64 CPU cosines between pooled vectors (d,)."""
    m = torch.nn.functional.normalize(torch.stack([v.double().flatten() for v in vs]), dim=1)
    return (m @ m.T).cpu()


@torch.no_grad()
def off_kl(net: ST.Net, toks: Tensor, pos: int, idx: Tensor | None = None, steering: tuple[Tensor, int, float] | None = None,
           dW: ST.DW | None = None) -> float:  # fmt: skip
    """Mean KL(W0 || steered or state) over the positions other than `pos` (E135's `off_kl`, batched). Only positions after
    `pos` can move, since steering edits `pos` alone."""
    tot, n = 0.0, 0
    for s in range(0, len(toks), BATCH):
        b = toks[s : s + BATCH]
        edit = None if steering is None or idx is None else steer(steering[0], steering[1], steering[2], idx[s : s + BATCH], pos)
        l0 = torch.log_softmax(net.run(b)[0].double(), -1)
        with net.at(dW):
            l1 = torch.log_softmax(net.run(b, edit)[0].double(), -1)
        kl = kl_rows(l0, l1)
        keep = torch.ones(kl.shape[1], dtype=torch.bool, device=kl.device)
        keep[pos] = False
        tot, n = tot + float(kl[:, keep].sum()), n + int(kl[:, keep].numel())
    return tot / n
