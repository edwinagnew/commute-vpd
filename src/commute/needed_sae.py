"""The needed set K at E234's base-only SAE latents (`commute.sae_feat`'s variables) by single-variable interchange on D,
for E298 (cell P1.62, `experiments/fig2_standard`). Added 2026-11-24 beside `commute.needed` (VPD) and
`commute.needed_neurons` (neurons); calls `needed.tail` and `needed.Scan` and changes nothing in them or in sae_feat.

Variables (sae_feat): per site (the attention and MLP output of each block, SITES order) its 16,384 latents and its
error term, 131,080 in all. On a window, latent u's value is f_u(o) d_u, o the site module's output in the run, f the
SAE's thresholded encoder, d_u its decoder column; the error's value is e(o) = o - decode(f(o)). Interchange: a case
(u, (x, x')) reruns blocks l.. from x's cached block input with u's value set to its value in F's cached run on x',
at every position: the module output o becomes o + (f_u(o_x') - f_u(o_x)) d_u for a latent, o + e(o_x') - e(o_x) for
the error (= decode(f(o)) + e(o_x') in exact arithmetic, where o = o_x), o_x and o_x' from F's cached run (`Swap`). A
variable is needed when some pair changes F's argmax token at some read position (as `needed.scan`).

Shortcut (exact): a latent whose f on x' equals its f on x at every position (zero on both, mostly: the SAE is sparse)
leaves the forward as it is, so its case is not run and counts as no change. Each pair runs the latents whose f differs
somewhere between x and x', and the error term. `only` (the identity control) runs the cases it names, no shortcut.
`joint` (the positive control) swaps every variable at every site at once, which must give F's argmax on x'.
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
    """A forward hook on one site's module: row r of the batch, F's run on window ex[r] from the site's block on, has
    local variable var[r] (latent u < width, the error at width) set to its value in F's cached run on window pt[r].
    The change is read from the cache on both sides (o[pt] against o[ex]): upstream of the site the row is F's own run
    on ex[r], so out equals o[ex] in exact arithmetic, while x's side read from the recomputed out let batch rounding
    flip a JumpReLU gate at threshold (E298 stage 0 identity, 2026-11-24)."""

    def __init__(self, s: SF.SAESite, outs: Tensor) -> None:
        self.ae, self.o, self.W = s.ae, outs, s.rem - s.off
        self.We, self.be, self.Wd = s.ae.encoder.weight, s.ae.encoder.bias, s.ae.decoder.weight.T.contiguous()  # (W, d) each
        self.var: Tensor | None = None
        self.ex: Tensor | None = None
        self.pt: Tensor | None = None

    def f_u(self, o: Tensor, u: Tensor) -> Tensor:
        """(k, T): latent u[r]'s activation on o[r] (k, T, d), `sae.SAE.encode` one latent at a time."""
        p = torch.relu(torch.einsum("ktd,kd->kt", o - self.ae.b_dec, self.We[u]) + self.be[u].unsqueeze(1))
        return p * (p > self.ae.threshold)

    def __call__(self, _m: Any, _inp: tuple[Tensor, ...], out: Tensor) -> Tensor:
        assert self.var is not None and self.ex is not None and self.pt is not None
        var, ex, pt, out = self.var, self.ex, self.pt, out.clone()
        lat = (var < self.W).nonzero().squeeze(1)
        if len(lat):
            u = var[lat]
            df = self.f_u(self.o[pt[lat]], u) - self.f_u(self.o[ex[lat]], u)
            out[lat] += df.unsqueeze(-1) * self.Wd[u].unsqueeze(1)
        err = (var == self.W).nonzero().squeeze(1)
        if len(err):
            o, o2 = self.o[ex[err]], self.o[pt[err]]
            out[err] += o2 - self.ae.decode(self.ae.encode(o2)) - (o - self.ae.decode(self.ae.encode(o)))
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


class Joint(Swap):
    """E298's must_flip: every latent and the error term of the site set at once to their values on window pt[r]:
    Swap's latent step for all latents ((f(o') - f(o)) times the decoder columns), then the error term's, giving o'."""

    def __call__(self, _m: Any, _inp: tuple[Tensor, ...], out: Tensor) -> Tensor:
        assert self.pt is not None
        o2 = self.o[self.pt]
        f, f2 = self.ae.encode(out), self.ae.encode(o2)
        return out + (f2 - f) @ self.Wd - (out - self.ae.decode(f)) + (o2 - self.ae.decode(f2))


@torch.no_grad()
def joint(F: R.Model, ss: list[SF.SAESite], run: Run, at: slice, ex: Tensor, pt: Tensor, batch: int) -> tuple[Tensor, float]:
    """Every variable of every site swapped from pt[k] into ex[k] at once, and the token embedding (no variable), which
    rebuilds F's run on pt[k]: the argmax per pair and read position, and the max |logit - F's own on pt[k]|."""
    assert len(ss) == 2 * len(F.net.h), "every block's attention and MLP output must be a site"
    sws = [Joint(s, run.outs[j]) for j, s in enumerate(ss)]
    hs = [F.net.get_submodule(s.module).register_forward_hook(sw) for s, sw in zip(ss, sws, strict=True)]
    top: list[Tensor] = []
    dev = 0.0
    try:
        for e, p in zip(ex.split(batch), pt.split(batch), strict=True):
            for sw in sws:
                sw.pt = p
            x = run.blocks[0][e]
            for b in F.net.h:
                x = b(x)
            lg = IX.tail(F.net, len(F.net.h), x + run.blocks[0][p] - run.blocks[0][e], at)
            dev = max(dev, float((lg - run.logits[p]).abs().max()))
            top.append(lg.argmax(-1))
    finally:
        for h in hs:
            h.remove()
    return torch.cat(top), dev


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
                    sw.var, sw.ex, sw.pt = v, ex[kk], pt[kk]
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
