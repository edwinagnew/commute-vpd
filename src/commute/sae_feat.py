"""E234's base-only SAE latents as the variables of wording A and of K* (E245, cell P1.40 run 2; plan
reports/plans/2026-11-21_plan_r-under-a-sae.md). Added 2026-11-21 for `experiments/sae_r_a`. Uses `commute.sae.SAE` and
satisfies `commute.stand_in.Site`; changes neither.

Variables. Per block-output site (SITES, in forward order; the module net.h[l].attn or net.h[l].mlp) its SAE's latents
u (index off + u) and one remainder (index off + width): 8 x 16,385 = 131,080 in all. Embeddings, norms and the
unembedding belong to no variable and always keep F's mechanism. With y the module's input in the current run and
o_N(y) network N's module output on it (F's with its appended neurons, added exactly as `rule_retention.Model.append`'s
hook adds them; M has none), f the SAE's thresholded encoder (`SAE.encode`) and d_u its decoder column: latent u writes
f_u(o_N(y)) d_u in N and the remainder e_N(y) = o_N(y) - sum_u f_u(o_N(y)) d_u (b_dec in it). M's mechanism is M's module
at the same layer applied to F's y. A site is changed when its two modules' parameters differ or F appends neurons
there; at an unchanged site o_M = o_F bit for bit, so every gain is exactly 1 and every correction exactly 0.

Standing in (`SAESite.correction`): sum_u t_u (g_u f_u(o_M) - f_u(o_F)) d_u + t_e (g_e e_M - e_F), exact zeros where
t = 0. Gain sums: sum f_u(o_F) f_u(o_M) |d_u|^2 and sum f_u(o_M)^2 |d_u|^2, the remainder's <e_F, e_M> and |e_M|^2.
`stand_in.gains` passes the rows flattened; an attention module needs its windows, so a site keeps the window length T.

F_K (`Hybrid`, E226's convention as `kstar_vpd.Hybrid`): each ablated latent's write replaced by its mean write,
(f_u - mean f_u) d_u subtracted from the module output of the current run, the mean per position over D in F's
unablated run; the remainder likewise. All kept leaves every hook idle, so F_K is F bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import Tensor, nn

from commute import rule_retention as R
from commute import sae as S

SITES = tuple(f"{kind}{l}" for l in range(S.LAYERS) for kind in ("attn", "mlp"))


def module_name(site: str) -> str:
    return f"h.{site[-1]}.{site[:-1]}"


def _same_params(a: nn.Module, b: nn.Module) -> bool:
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    return pa.keys() == pb.keys() and all(torch.equal(p, pb[k]) for k, p in pa.items())


@dataclass
class SAESite:
    name: str  # the SAE's site, e.g. "mlp0"
    off: int  # variable index of latent 0
    ae: S.SAE
    fmod: nn.Module  # F's module, called through .forward so no hook on it runs
    mmod: nn.Module  # M's module at the same layer
    extra: nn.Module | None  # F's appended neurons at this MLP
    T: int  # window length
    changed: bool

    @property
    def module(self) -> str:
        return module_name(self.name)

    @property
    def rem(self) -> int:
        return self.off + self.ae.encoder.weight.shape[0]

    @property
    def idx(self) -> slice:
        return slice(self.off, self.rem + 1)

    def o_F(self, y: Tensor) -> Tensor:
        o = self.fmod.forward(y)
        return o if self.extra is None else o + self.extra(y)

    def o_M(self, y: Tensor) -> Tensor:
        return self.mmod.forward(y)

    def writes(self, o: Tensor) -> tuple[Tensor, Tensor]:
        """The latents f(o) and the remainder o - sum_u f_u d_u."""
        f = self.ae.encode(o)
        return f, o - self.ae.decoder(f)

    def gain_sums(self, y: Tensor) -> tuple[Tensor, Tensor]:
        y = y.reshape(-1, self.T, y.shape[-1])
        (fF, eF), (fM, eM) = self.writes(self.o_F(y)), self.writes(self.o_M(y))
        dd = self.ae.decoder.weight.double().pow(2).sum(0)
        fF, fM, eF, eM = fF.flatten(0, 1).double(), fM.flatten(0, 1).double(), eF.double(), eM.double()
        num = torch.cat([(fF * fM).sum(0) * dd, (eF * eM).sum().reshape(1)])
        return num, torch.cat([(fM * fM).sum(0) * dd, (eM * eM).sum().reshape(1)])

    def change_sums(self, y: Tensor) -> Tensor:
        """(k,) fp64: sums over every row of y of |write_F_u - write_M_u|^2, (f_u(o_F) - f_u(o_M))^2 |d_u|^2 per latent and
        |e_F - e_M|^2 for the remainder (`stand_in.write_change`; added 2026-11-22 for experiments/sae_eb, P1.52)."""
        y = y.reshape(-1, self.T, y.shape[-1])
        (fF, eF), (fM, eM) = self.writes(self.o_F(y)), self.writes(self.o_M(y))
        dd = self.ae.decoder.weight.double().pow(2).sum(0)
        df = (fF - fM).flatten(0, 1).double()
        return torch.cat([df.pow(2).sum(0) * dd, (eF.double() - eM.double()).pow(2).sum().reshape(1)])

    def correction(self, y: Tensor, t: Tensor, g: Tensor) -> Tensor:
        (fF, eF), (fM, eM) = self.writes(self.o_F(y)), self.writes(self.o_M(y))
        return self.ae.decoder(t[:-1] * (g[:-1] * fM - fF)) + t[-1] * (g[-1] * eM - eF)


def sites(F: R.Model, M: R.Model, saes: dict[str, S.SAE], T: int) -> list[SAESite]:
    """The 8 sites of the pair (F, M); `saes` keyed by site name."""
    assert not M.extra, "M appends no neurons"
    assert all(0 <= l < S.LAYERS for l in F.extra)
    out, off = [], 0
    for name in SITES:
        l, kind = int(name[-1]), name[:-1]
        fmod, mmod = F.net.get_submodule(module_name(name)), M.net.get_submodule(module_name(name))
        extra = F.extra.get(l) if kind == "mlp" else None
        s = SAESite(name, off, saes[name], fmod, mmod, extra, T, extra is not None or not _same_params(fmod, mmod))
        out.append(s)
        off = s.rem + 1
    return out


def n_vars(ss: list[SAESite]) -> int:
    return ss[-1].rem + 1


def changed_vars(ss: list[SAESite]) -> set[int]:
    return {u for s in ss if s.changed for u in range(s.idx.start, s.idx.stop)}


def names(ss: list[SAESite]) -> list[str]:
    return [f"{s.name}:{u}" if u < s.rem - s.off else f"{s.name}:rem" for s in ss for u in range(s.rem - s.off + 1)]


@torch.no_grad()
def recompute_differs(net: Any, ss: list[SAESite], x: Tensor) -> list[str]:
    """Sites whose o_F(y), recomputed from the module's input, is not bit for bit the module's output in F's own run
    (every hook registered before this one included: the appended neurons, idle stand-in and ablation hooks)."""
    got: dict[int, tuple[Tensor, Tensor]] = {}
    hs = [net.get_submodule(s.module).register_forward_hook(lambda _m, a, o, j=j: got.__setitem__(j, (a[0], o))) for j, s in enumerate(ss)]
    try:
        net(x)
    finally:
        for h in hs:
            h.remove()
    return [s.name for j, s in enumerate(ss) if not torch.equal(s.o_F(got[j][0]), got[j][1])]


# ---- F_K -----------------------------------------------------------------------------------------------------------


class Hybrid:
    """Hooks on F's site modules (registered after R.Model's own): native (no keep vector), capture (sums for the means)
    or ablate (keep vector c)."""

    def __init__(self, F: R.Model, ss: list[SAESite], at: slice) -> None:
        self.F, self.sites, self.at = F, ss, at
        self.w: Tensor | None = None  # 1 - c on the device while ablating
        self.live: list[bool] = []
        self.mu: dict[str, Tensor] = {}
        self.sums: dict[str, Tensor] | None = None
        self.handles = [F.net.get_submodule(s.module).register_forward_hook(partial(self._site, j)) for j, s in enumerate(ss)]

    def close(self) -> None:
        for h in self.handles:
            h.remove()

    def _add(self, k: str, v: Tensor) -> None:
        assert self.sums is not None
        self.sums[k] = self.sums[k] + v.sum(0) if k in self.sums else v.sum(0)

    def _site(self, j: int, _m: Any, _inp: tuple[Tensor, ...], out: Tensor) -> Tensor | None:
        ablate = self.w is not None and self.live[j]
        if self.sums is None and not ablate:
            return None
        s = self.sites[j]
        f, e = s.writes(out)
        if self.sums is not None:
            self._add(f"f{j}", f)
            self._add(f"e{j}", e)
        if not ablate:
            return None
        assert self.w is not None
        w = self.w[s.idx]
        return out - s.ae.decoder(w[:-1] * (f - self.mu[f"f{j}"])) - w[-1] * (e - self.mu[f"e{j}"])

    @torch.no_grad()
    def capture(self, batches: list[Tensor]) -> None:
        """The means over D, per position, from F's unablated run: each site's latents and remainder."""
        self.sums = {}
        for t in batches:
            self.F.net(t)
        n = sum(len(t) for t in batches)
        self.mu, self.sums = {k: v / n for k, v in self.sums.items()}, None

    def logits(self, toks: Tensor, c: Tensor) -> Tensor:
        """F_K's logits at `at`, c (n,) on the device: 1 kept, 0 mean-ablated (fractional for the attribution)."""
        cc = c.detach().cpu()
        self.live = [bool((cc[s.idx] < 1).any()) for s in self.sites]
        self.w = 1 - c
        try:
            return self.F.logits(toks)[:, self.at]
        finally:
            self.w = None


@dataclass
class Ctx:
    hy: Hybrid
    n: int
    toks: list[Tensor]
    lF: list[Tensor]  # F's logits at `at` (all kept)
    kl_none: Tensor  # (N,) KL(F || F_all_ablated), CPU fp64
    kl_fm: Tensor  # (N,) KL(F || M)

    def subset(self, idx: Tensor, batch: int) -> Ctx:
        t, lf = torch.cat(self.toks), torch.cat(self.lF)
        i = idx.to(t.device)
        return Ctx(self.hy, self.n, list(t[i].split(batch)), list(lf[i].split(batch)), self.kl_none[idx], self.kl_fm[idx])


@torch.no_grad()
def prepare(F: R.Model, M: R.Model, ss: list[SAESite], toks: Tensor, at: slice, batch: int) -> Ctx:
    hy, batches, n = Hybrid(F, ss, at), list(toks.split(batch)), n_vars(ss)
    hy.capture(batches)
    lF = [hy.logits(t, torch.ones(n, device=toks.device)) for t in batches]
    kl_none = torch.cat([R.kl(lf, hy.logits(t, torch.zeros(n, device=toks.device))) for t, lf in zip(batches, lF, strict=True)]).cpu()
    kl_fm = torch.cat([R.kl(lf, M.logits(t)[:, at]) for t, lf in zip(batches, lF, strict=True)]).cpu()
    return Ctx(hy, n, batches, lF, kl_none, kl_fm)


@torch.no_grad()
def kl_mean(ctx: Ctx, keep: Tensor) -> float:
    """Mean over the windows of KL(F || F_K), K = where `keep` (bool, CPU) is set."""
    c = keep.to(ctx.toks[0].device, torch.float32)
    return float(torch.cat([R.kl(lf, ctx.hy.logits(t, c)) for t, lf in zip(ctx.toks, ctx.lF, strict=True)]).mean())


def attribution(ctx: Ctx, c0: Tensor) -> Tensor:
    """d sum KL(F || F_c) / d c_u at the keep vector c0 for every variable, signed (CPU)."""
    c = c0.to(ctx.toks[0].device, torch.float32).clone().requires_grad_(True)
    with torch.enable_grad():
        for t, lf in zip(ctx.toks, ctx.lF, strict=True):
            R.kl(lf, ctx.hy.logits(t, c)).sum().backward()
    assert c.grad is not None
    return c.grad.cpu()


@torch.no_grad()
def gate_all(ctx: Ctx) -> float:
    """Max |F_K - F| over the windows with every variable kept, against F's native logits: exactly 0."""
    ones = torch.ones(ctx.n, device=ctx.toks[0].device)
    return max(float((ctx.hy.logits(t, ones) - ctx.hy.F.logits(t)[:, ctx.hy.at]).abs().max()) for t in ctx.toks)
