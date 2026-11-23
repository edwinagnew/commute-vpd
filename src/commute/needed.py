"""The needed set K of the frozen definition (theory/2026-11-22_definition_frozen.md, section 2) at the VPD
featurization of `commute.kstar_vpd`, by single-variable interchange on D. Added 2026-11-22 for cell P1.54
(`experiments/kstar_needed`); calls `kstar_vpd` and changes nothing in it.

A variable u is needed when, for some x, x' in D, F's run on x with u's value set to the one it takes in F's own run on
x' gives a designated output (the argmax token at the read positions `at`) other than F's own on x. The variables tested
are every subcomponent and every remainder of the 24 decomposed matrices; the embedding and unembedding are the input and
output variables and are not tested. Each mechanism is linear in its matrix input y (kstar_vpd docstring): subcomponent
c writes (a + b) u_c + a du_c with a = v_c . y, b = dv_c . y in F, the remainder writes y rf. So the interchange on x is
the matrix's output in the run on x minus write_u(y_x - y_x'), at every position, y_x' the matrix's input in F's
unmodified run on x' (all windows of D have one length). Grafts (appended neurons) are not handled: `scan` refuses a
featurization with any.

One row of a forward is one (variable, pair) case. F's run on D is cached once (each matrix's input and each block's
input); a case at a matrix of block l reruns blocks l.. from the cached block input, so rows of one matrix share a batch.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from commute import kstar_vpd as KV
from commute import rule_retention as R


def derangements(n: int, P: int, m: int, seed: int) -> tuple[Tensor, Tensor]:
    """The pairs (x, x') tested, as index tensors (P * m,): for each of P seeded random derangements pi of D, the first m
    of a seeded random order sigma of D as x and pi(x) as x'. Pair k belongs to derangement k // m."""
    assert n >= 2 and 1 <= m <= n, (n, m)
    g, ex, pt = torch.Generator().manual_seed(seed), [], []
    for _ in range(P):
        pi = torch.randperm(n, generator=g)
        while bool((pi == torch.arange(n)).any()):
            pi = torch.randperm(n, generator=g)
        sigma = torch.randperm(n, generator=g)[:m]
        ex.append(sigma)
        pt.append(pi[sigma])
    return torch.cat(ex), torch.cat(pt)


@dataclass
class Run:
    """F's unmodified run on D: each matrix's input, each block's input, the argmax and top-2 margin at `at`."""

    ys: list[Tensor]  # per site (N, T, d_in)
    blocks: list[Tensor]  # per block (N, T, d)
    top: Tensor  # (N, n_at) argmax token
    margin: Tensor  # (N, n_at) top-1 minus top-2 logit
    logits: Tensor  # (N, n_at, V)


def _layer(name: str) -> int:
    return int(name.split(".")[1])  # "h.2.mlp.c_fc" -> 2


@torch.no_grad()
def cache(F: R.Model, feat: KV.Feat, toks: Tensor, at: slice, batch: int) -> Run:
    net, got = F.net, {}
    names = [s.name for s in feat.sites] + [f"h.{l}" for l in range(len(net.h))]
    hs = [net.get_submodule(k).register_forward_pre_hook(lambda _m, a, k=k: got.setdefault(k, []).append(a[0])) for k in names]
    try:
        lg = torch.cat([net(t)[0][:, at].float() for t in toks.split(batch)])
    finally:
        for h in hs:
            h.remove()
    top2 = lg.topk(2, -1).values
    return Run([torch.cat(got[s.name]) for s in feat.sites], [torch.cat(got[f"h.{l}"]) for l in range(len(net.h))],
               lg.argmax(-1), top2[..., 0] - top2[..., 1], lg)  # fmt: skip


def tail(net: Any, l: int, x: Tensor, at: slice) -> Tensor:
    """Logits at `at` from block l's input x: blocks l.., the final norm, the unembedding."""
    for b in net.h[l:]:
        x = b(x)
    return net.lm_head(net.ln_f(x[:, at])).float()


class Swap:
    """A forward hook on one matrix: row r of the batch has variable var[r] (local: subcomponent c < C, the remainder
    at C) set to its value on pairs' x' (index pt[r] into the cache)."""

    def __init__(self, s: KV.Site, y: Tensor) -> None:
        self.s, self.y, self.C = s, y, s.rem - s.off
        self.VT, self.dVT = s.V.T.contiguous(), s.dV.T.contiguous()
        self.var: Tensor | None = None
        self.pt: Tensor | None = None

    def __call__(self, _m: Any, inp: tuple[Tensor, ...], out: Tensor) -> Tensor:
        assert self.var is not None and self.pt is not None
        dy, s = inp[0] - self.y[self.pt], self.s
        corr = torch.zeros_like(out)
        sub = (self.var < self.C).nonzero().squeeze(1)
        if len(sub):
            c, d = self.var[sub], dy[sub]
            a = torch.einsum("btd,bd->bt", d, self.VT[c]).unsqueeze(-1)
            b = torch.einsum("btd,bd->bt", d, self.dVT[c]).unsqueeze(-1)
            corr[sub] = (a + b) * s.U[c].unsqueeze(1) + a * s.dU[c].unsqueeze(1)
        rem = (self.var == self.C).nonzero().squeeze(1)
        if len(rem):
            corr[rem] = dy[rem] @ s.rf
        return out - corr


@dataclass
class Scan:
    first_pair: Tensor  # (n,) int64: the first pair index that changed any designated output, -1 for none or untested
    flips: Tensor  # (n,) int64: pairs that changed any designated output
    max_dev: float  # max |logit - F's own| over every case: the floor when every pair is (x, x)
    rows: int
    seconds: float


@torch.no_grad()
def scan(F: R.Model, feat: KV.Feat, run: Run, at: slice, ex: Tensor, pt: Tensor, rows: int, log: Callable[[str], None],
         only: dict[int, Tensor] | None = None) -> Scan:  # fmt: skip
    """Every variable at every site (or, with `only`, the local variables given per site index) against every pair
    (ex[k], pt[k]), `rows` cases per forward."""
    assert not feat.grafts, "graft variables are not handled by this scan"
    dev = run.top.device
    ex, pt, Q = ex.to(dev), pt.to(dev), len(ex)
    first = torch.full((feat.n,), -1, dtype=torch.long, device=dev)
    flips = torch.zeros(feat.n, dtype=torch.long, device=dev)
    big, max_dev, done, t0 = torch.iinfo(torch.long).max, 0.0, 0, time.time()
    for j, s in enumerate(feat.sites):
        if only is not None and j not in only:
            continue
        local = (only[j] if only is not None else torch.arange(s.rem - s.off + 1)).to(dev)
        sw = Swap(s, run.ys[j])
        var, pair = local.repeat_interleave(Q), torch.arange(Q, device=dev).repeat(len(local))
        h, l = F.net.get_submodule(s.name).register_forward_hook(sw), _layer(s.name)
        try:
            for i in range(0, len(var), rows):
                v, k = var[i : i + rows], pair[i : i + rows]
                sw.var, sw.pt = v, pt[k]
                lg = tail(F.net, l, run.blocks[l][ex[k]], at)
                max_dev = max(max_dev, float((lg - run.logits[ex[k]]).abs().max()))
                hit = (lg.argmax(-1) != run.top[ex[k]]).any(-1)
                g = s.off + v
                flips.index_add_(0, g, hit.long())
                cand = torch.full((feat.n,), big, dtype=torch.long, device=dev)
                cand.scatter_reduce_(0, g[hit], k[hit], "amin")
                first = torch.where((cand != big) & ((first < 0) | (cand < first)), cand, first)
        finally:
            h.remove()
        done += len(var)
        log(f"    site {j} {s.name}: {len(local)} variables x {Q} pairs, {int((first[s.off : s.rem + 1] >= 0).sum())} needed; "
            f"{done} cases in {time.time() - t0:.0f} s ({done / max(time.time() - t0, 1e-9):.0f}/s)")  # fmt: skip
    return Scan(first.cpu(), flips.cpu(), max_dev, done, time.time() - t0)
