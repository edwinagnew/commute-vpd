"""Wording A's test at the VPD featurization (`commute.kstar_vpd`): M's mechanisms, each up to an output gain, put in for
F's on a set S of variables by soft intervention (Geiger Def. 10), in F's own forward. Added 2026-11-20 for
`experiments/kstar_vpd_wording_a` (plan reports/plans/2026-11-20_plan_r-under-a.md). Calls `commute.kstar_vpd`
(`Feat`, `_inputs`) and `commute.rule_retention` (`kl`) and changes neither.

Mechanisms are E226's (kstar_vpd docstring): on the input y of decomposed matrix W, with a = v_c . y and b = dv_c . y,
subcomponent c writes a u_c in M and (a + b) u_c + a du_c in F; the remainder writes Delta y in M and (Delta + dW_out) y
in F. Stood in with gain g, the variable writes g times M's write on the input it receives in the intervened run. With a
weight t per variable (t = 1 stood in, t = 0 F's own; fractions only for the attribution) the site's output gains
    sum_c t_c [(a (g_c - 1) - b) u_c - a du_c]  +  t_rem (g_rem Delta - Delta - dW_out) y.
Every t = 0 adds exact zeros, so the forward is F's bit for bit. Grafts (appended neurons, which M lacks) are never
stood in. The gain is the closed-form least-squares fit of M's write to F's write over every position of D in F's own
(unintervened) run, any sign: g = sum <write_F, write_M> / sum |write_M|^2, 1 where M's write is zero everywhere.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from torch import Tensor

from commute import kstar_vpd as KV
from commute import rule_retention as R


@torch.no_grad()
def gains(F: R.Model, feat: KV.Feat, toks: Tensor, batch: int) -> Tensor:
    """(n,) fp64 CPU: the least-squares output gain of every variable (1 outside the sites)."""
    names = [s.name for s in feat.sites]
    norms = [((s.U.double() ** 2).sum(1), (s.U.double() * s.dU.double()).sum(1)) for s in feat.sites]
    num: dict[Any, Tensor] = {}
    den: dict[Any, Tensor] = {}

    def add(k: Any, n: Tensor, d: Tensor) -> None:
        num[k], den[k] = (num[k] + n, den[k] + d) if k in num else (n, d)

    for t in toks.split(batch):
        for j, (s, x) in enumerate(zip(feat.sites, KV._inputs(F.net, names, t), strict=True)):
            y = x.double().flatten(0, 1)
            a, b = y @ s.V.double(), y @ s.dV.double()
            uu, udu = norms[j]
            aa = (a * a).sum(0)
            add(j, (a * (a + b)).sum(0) * uu + aa * udu, aa * uu)
            wF, wM = y @ s.rf.double(), y @ s.rm.double()
            add(("rem", j), (wF * wM).sum().reshape(1), (wM * wM).sum().reshape(1))
    g = torch.ones(feat.n, dtype=torch.float64)
    for j, s in enumerate(feat.sites):
        for k, cols in ((j, slice(s.off, s.rem)), (("rem", j), slice(s.rem, s.rem + 1))):
            n, d = num[k].cpu(), den[k].cpu()
            g[cols] = torch.where(d > 0, n / d.clamp(min=1e-300), 1.0)
    return g


class StandIn:
    """Forward hooks on F's decomposed matrices; idle (F's forward unchanged) while `t` is None."""

    def __init__(self, F: R.Model, feat: KV.Feat, g: Tensor) -> None:
        self.feat, self.g = feat, g.to(feat.sites[0].U.device, torch.float32)
        self.t: Tensor | None = None
        self.live: list[bool] = []
        self.handles = [F.net.get_submodule(s.name).register_forward_hook(partial(self._site, j)) for j, s in enumerate(feat.sites)]

    def close(self) -> None:
        for h in self.handles:
            h.remove()

    def _site(self, j: int, _m: Any, inp: tuple[Tensor, ...], out: Tensor) -> Tensor | None:
        if self.t is None or not self.live[j]:
            return None
        s, y = self.feat.sites[j], inp[0]
        t, g = self.t[s.off : s.rem], self.g[s.off : s.rem]
        a = y @ s.V
        corr = ((a * (g - 1) - y @ s.dV) * t) @ s.U - (a * t) @ s.dU
        return out + corr + self.t[s.rem] * (y @ (self.g[s.rem] * s.rm - s.rf))

    def logits(self, F: R.Model, toks: Tensor, at: slice, t: Tensor) -> Tensor:
        """F's logits at `at` with every variable at weight t (n,), on the device; differentiable in t."""
        tc = t.detach().cpu()
        self.live = [bool((tc[s.off : s.rem + 1] != 0).any()) for s in self.feat.sites]
        self.t = t
        try:
            return F.net(toks)[0].float()[:, at]
        finally:
            self.t = None


@torch.no_grad()
def kl_mean(si: StandIn, F: R.Model, toks: list[Tensor], lF: list[Tensor], at: slice, t: Tensor) -> float:
    """Mean over the windows of KL(F || F with weights t); lF is F's own logits at `at` per batch."""
    return float(torch.cat([R.kl(lf, si.logits(F, x, at, t)) for x, lf in zip(toks, lF, strict=True)]).mean())


def attribution(si: StandIn, F: R.Model, toks: list[Tensor], lF: list[Tensor], at: slice, t: Tensor) -> Tensor:
    """d sum KL(F || F with weights t) / d t_u for every variable (CPU), one backward per batch."""
    tt = t.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        for x, lf in zip(toks, lF, strict=True):
            R.kl(lf, si.logits(F, x, at, tt)).sum().backward()
    assert tt.grad is not None
    return tt.grad.cpu()
