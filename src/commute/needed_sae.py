"""The needed set K at E234's base-only SAE latents (`commute.sae_feat`'s variables) by single-variable interchange on D,
for E298 (cell P1.62, `experiments/fig2_standard`). Added 2026-11-24 beside `commute.needed` (VPD) and
`commute.needed_neurons` (neurons); calls `needed.tail` and `needed.Scan` and changes nothing in them or in sae_feat.

Variables (sae_feat): per site (the attention and MLP output of each block, SITES order) its 16,384 latents and its
error term, 131,080 in all. On a window, latent u's value is f_u(o) d_u, o the site module's output in the run, f the
SAE's thresholded encoder, d_u its decoder column; the error's value is e(o) = o - decode(f(o)). Interchange: a case
(u, (x, x')) reruns blocks l.. from x's cached block input with u's value set to its value in F's cached run on x',
at every position: the module output o becomes o + (f_u(o_x') - f_u(o)) d_u for a latent, decode(f(o)) + e(o_x') for
the error. A variable is needed when some pair changes F's argmax token at some read position (as `needed.scan`).

Shortcut (exact): a latent whose f on x' equals its f on x at every position (zero on both, mostly: the SAE is sparse)
leaves the forward as it is, so its case is not run and counts as no change. Each pair runs the latents whose f differs
somewhere between x and x', and the error term. `only` (the identity control) runs the cases it names, no shortcut.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from commute import needed as IX
from commute import rule_retention as R
from commute import sae_feat as SF


@dataclass
class Run:
    """F's unmodified run on D: each block's input, each site's module output, the argmax, margin and logits at `at`."""

    blocks: list[Tensor]  # per block (N, T, d)
    outs: list[Tensor]  # per site (N, T, d)
    top: Tensor  # (N, n_at)
    margin: Tensor  # (N, n_at) top-1 minus top-2 logit
    logits: Tensor  # (N, n_at, V)


@torch.no_grad()
def cache(F: R.Model, ss: list[SF.SAESite], toks: Tensor, at: slice, batch: int) -> Run:
    assert not F.extra, "appended neurons are not handled by this scan"
    net, got = F.net, {}
    L = range(len(net.h))
    hs = [net.h[l].register_forward_pre_hook(lambda _m, a, k=f"b{l}": got.setdefault(k, []).append(a[0])) for l in L]
    hs += [net.get_submodule(s.module).register_forward_hook(lambda _m, _a, o, k=f"o{j}": got.setdefault(k, []).append(o)) for j, s in enumerate(ss)]
    try:
        lg = torch.cat([net(t)[0][:, at].float() for t in toks.split(batch)])
    finally:
        for h in hs:
            h.remove()
    top2 = lg.topk(2, -1).values
    return Run([torch.cat(got[f"b{l}"]) for l in L], [torch.cat(got[f"o{j}"]) for j in range(len(ss))], lg.argmax(-1), top2[..., 0] - top2[..., 1], lg)


class Swap:
    """A forward hook on one site's module: row r of the batch has local variable var[r] (latent u < width, the error
    at width) set to its value in F's cached run on window pt[r]."""

    def __init__(self, s: SF.SAESite, outs: Tensor) -> None:
        self.ae, self.o, self.W = s.ae, outs, s.rem - s.off
        self.We, self.be, self.Wd = s.ae.encoder.weight, s.ae.encoder.bias, s.ae.decoder.weight.T.contiguous()  # (W, d) each
        self.var: Tensor | None = None
        self.pt: Tensor | None = None

    def f_u(self, o: Tensor, u: Tensor) -> Tensor:
        """(k, T): latent u[r]'s activation on o[r] (k, T, d), `sae.SAE.encode` one latent at a time."""
        p = torch.relu(torch.einsum("ktd,kd->kt", o - self.ae.b_dec, self.We[u]) + self.be[u].unsqueeze(1))
        return p * (p > self.ae.threshold)

    def __call__(self, _m: Any, _inp: tuple[Tensor, ...], out: Tensor) -> Tensor:
        assert self.var is not None and self.pt is not None
        var, pt, out = self.var, self.pt, out.clone()
        lat = (var < self.W).nonzero().squeeze(1)
        if len(lat):
            u = var[lat]
            df = self.f_u(self.o[pt[lat]], u) - self.f_u(out[lat], u)
            out[lat] += df.unsqueeze(-1) * self.Wd[u].unsqueeze(1)
        err = (var == self.W).nonzero().squeeze(1)
        if len(err):
            o, o2 = out[err], self.o[pt[err]]
            out[err] = self.ae.decode(self.ae.encode(o)) + o2 - self.ae.decode(self.ae.encode(o2))
        return out

    @torch.no_grad()
    def moved(self, ex: Tensor, pt: Tensor, chunk: int) -> tuple[Tensor, Tensor]:
        """(latent, pair) of every latent whose f differs somewhere between windows ex[k] and pt[k]: the cases run."""
        mask = torch.cat([(self.ae.encode(self.o[ex[i : i + chunk]]) != self.ae.encode(self.o[pt[i : i + chunk]])).any(1)
                          for i in range(0, len(ex), chunk)])  # fmt: skip
        k, u = mask.nonzero(as_tuple=True)
        return u, k

    @torch.no_grad()
    def active(self, idx: Tensor, n: int, chunk: int) -> Tensor:
        """The first n latents (by index) active somewhere on windows idx: the identity control's latents."""
        on = torch.stack([(self.ae.encode(self.o[idx[i : i + chunk]]) > 0).flatten(0, 1).any(0) for i in range(0, len(idx), chunk)]).any(0)
        return on.nonzero().squeeze(1)[:n]


@torch.no_grad()
def scan(F: R.Model, ss: list[SF.SAESite], run: Run, at: slice, ex: Tensor, pt: Tensor, rows: int, err_rows: int, chunk: int,
         log: Callable[[str], None], only: dict[int, Tensor] | None = None) -> IX.Scan:  # fmt: skip
    """Every variable against every pair (ex[k], pt[k]) with the shortcut, or with `only` the local variables given per
    site index against every pair, none skipped; `rows` latent cases or `err_rows` error cases per forward. Scan.rows
    counts the cases run."""
    dev = run.top.device
    ex, pt, Q, n = ex.to(dev), pt.to(dev), len(ex), SF.n_vars(ss)
    first = torch.full((n,), -1, dtype=torch.long, device=dev)
    flips = torch.zeros(n, dtype=torch.long, device=dev)
    big, max_dev, done, t0 = torch.iinfo(torch.long).max, 0.0, 0, time.time()
    for j, s in enumerate(ss):
        if only is not None and j not in only:
            continue
        sw, l = Swap(s, run.outs[j]), int(s.name[-1])
        if only is None:
            u, k = sw.moved(ex, pt, chunk)
            var, pair = torch.cat([u, torch.full((Q,), sw.W, device=dev)]), torch.cat([k, torch.arange(Q, device=dev)])
        else:
            loc = only[j].to(dev)
            var, pair = loc.repeat_interleave(Q), torch.arange(Q, device=dev).repeat(len(loc))
        h = F.net.get_submodule(s.module).register_forward_hook(sw)
        try:
            for part, size in ((var < sw.W, rows), (var == sw.W, err_rows)):
                vs, ks = var[part], pair[part]
                for i in range(0, len(vs), size):
                    v, kk = vs[i : i + size], ks[i : i + size]
                    sw.var, sw.pt = v, pt[kk]
                    lg = IX.tail(F.net, l, run.blocks[l][ex[kk]], at)
                    max_dev = max(max_dev, float((lg - run.logits[ex[kk]]).abs().max()))
                    hit = (lg.argmax(-1) != run.top[ex[kk]]).any(-1)
                    g = s.off + v
                    flips.index_add_(0, g, hit.long())
                    cand = torch.full((n,), big, dtype=torch.long, device=dev)
                    cand.scatter_reduce_(0, g[hit], kk[hit], "amin")
                    first = torch.where((cand != big) & ((first < 0) | (cand < first)), cand, first)
        finally:
            h.remove()
        done += len(var)
        log(f"    site {j} {s.name}: {len(var)} cases run of {(sw.W + 1) * Q}, {int((first[s.off : s.rem + 1] >= 0).sum())} needed; "
            f"{done} cases in {time.time() - t0:.0f} s ({done / max(time.time() - t0, 1e-9):.0f}/s)")  # fmt: skip
    return IX.Scan(first.cpu(), flips.cpu(), max_dev, done, time.time() - t0)
