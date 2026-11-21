"""Wording A's test on any featurization: M's mechanisms, each up to a positive output gain, put in for F's on a set of
variables by soft intervention (Geiger Def. 10), inside a forward of F. Added 2026-11-20 for
`experiments/kstar_vpd_wording_a` (cell P1.46; plan reports/plans/2026-11-20_plan_r-under-a.md).

Interface (featurization-agnostic). Variables are indexed 0..n-1 in one global vector. A featurization supplies a list
of sites, each an object with
    module: str   the submodule of F's network the site's variables write into; y below is that module's input[0]
                  in the current run, and the site's correction is added to that module's output
    idx: slice    the site's variables in the global vector
    gain_sums(y) -> (num, den), each (k,): sums over every row of y of <write_F_u(y), write_M_u(y)> and
                  |write_M_u(y)|^2, u the site's k variables (fp64)
    correction(y, t, g) -> Tensor: sum_u t_u (g_u write_M_u(y) - write_F_u(y)), the shape of the module's output;
                  t, g the site's slices of the global weights and gains (fp32, on the device)
and it must add exact zeros where t = 0, so the forward with every t = 0 is F's bit for bit. `vpd_sites` is the adapter
for `commute.kstar_vpd` (subcomponents and each matrix's remainder); another featurization (SAE latents, say) writes
its own site class with the same four members and uses `gains`, `StandIn`, `kl_mean` and `attribution` unchanged.

`gains`: the closed-form least-squares output gain per variable over the rows given (every position of D in F's own
unintervened run), g = num / den, 1 where den = 0 (M writes zero there; standing in writes zero, as M). The definition
allows positive gains only (user, 2026-11-20 ~21:00): `standable(g)` is g > 0, and a variable with g <= 0 must keep
F's mechanism; callers exclude it from the stood-in set. `StandIn` holds the hooks; `kl_mean` and `attribution` take a
`forward(x) -> logits at the read positions` callable, so the same stood-in variables can be read in F's full forward or
inside a circuit's ablated forward (another hook set active at the same time).

VPD mechanisms are E226's (kstar_vpd docstring): with a = v_c . y, b = dv_c . y, subcomponent c writes a u_c in M and
(a + b) u_c + a du_c in F; the remainder writes Delta y in M and (Delta + dW_out) y in F. So its correction is
sum_c t_c [(a (g_c - 1) - b) u_c - a du_c] + t_rem (g_rem Delta - Delta - dW_out) y.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

import torch
from torch import Tensor

from commute import kstar_vpd as KV
from commute import rule_retention as R


class Site(Protocol):
    module: str
    idx: slice

    def gain_sums(self, y: Tensor) -> tuple[Tensor, Tensor]: ...

    def correction(self, y: Tensor, t: Tensor, g: Tensor) -> Tensor: ...


@dataclass
class VPDSite:
    """One decomposed matrix of `kstar_vpd`: its subcomponents, then its remainder (idx = off .. rem inclusive)."""

    s: KV.Site

    @property
    def module(self) -> str:
        return self.s.name

    @property
    def idx(self) -> slice:
        return slice(self.s.off, self.s.rem + 1)

    def gain_sums(self, y: Tensor) -> tuple[Tensor, Tensor]:
        s, y = self.s, y.double()
        U, dU = s.U.double(), s.dU.double()
        a, b = y @ s.V.double(), y @ s.dV.double()
        aa = (a * a).sum(0)
        uu = (U**2).sum(1)
        num = (a * (a + b)).sum(0) * uu + aa * (U * dU).sum(1)
        wF, wM = y @ s.rf.double(), y @ s.rm.double()
        return torch.cat([num, (wF * wM).sum().reshape(1)]), torch.cat([aa * uu, (wM * wM).sum().reshape(1)])

    def change_sums(self, y: Tensor) -> Tensor:
        """(k,) fp64: sums over every row of y of |write_F_u(y) - write_M_u(y)|^2: |b u_c + a du_c|^2 per subcomponent,
        |y (rf - rm)|^2 for the remainder."""
        s, y = self.s, y.double()
        U, dU = s.U.double(), s.dU.double()
        a, b = y @ s.V.double(), y @ s.dV.double()
        sub = (b * b).sum(0) * (U**2).sum(1) + 2 * (a * b).sum(0) * (U * dU).sum(1) + (a * a).sum(0) * (dU**2).sum(1)
        return torch.cat([sub, (y @ (s.rf.double() - s.rm.double())).pow(2).sum().reshape(1)])

    def correction(self, y: Tensor, t: Tensor, g: Tensor) -> Tensor:
        s = self.s
        a = y @ s.V
        tc, gc = t[:-1], g[:-1]
        return ((a * (gc - 1) - y @ s.dV) * tc) @ s.U - (a * tc) @ s.dU + t[-1] * (y @ (g[-1] * s.rm - s.rf))


def vpd_sites(feat: KV.Feat) -> list[VPDSite]:
    return [VPDSite(s) for s in feat.sites]


def standable(g: Tensor) -> Tensor:
    """Which variables may stand in: a positive gain (user, 2026-11-20 ~21:00)."""
    return g > 0


@torch.no_grad()
def gains(net: Any, sites: list[Any], n: int, batches: list[Tensor]) -> Tensor:
    """(n,) fp64 CPU: each variable's least-squares output gain over every position of `batches` in `net`'s own run
    (1 outside the sites and where M's write is zero throughout)."""
    num, den, got = torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64), {}
    hs = [net.get_submodule(s.module).register_forward_pre_hook(lambda _m, a, j=j: got.__setitem__(j, a[0])) for j, s in enumerate(sites)]
    try:
        for x in batches:
            net(x)
            for j, s in enumerate(sites):
                a, b = s.gain_sums(got[j].flatten(0, -2))
                num[s.idx] += a.cpu()
                den[s.idx] += b.cpu()
    finally:
        for h in hs:
            h.remove()
    return torch.where(den > 0, num / den.clamp(min=1e-300), 1.0)


@torch.no_grad()
def write_change(net: Any, sites: list[VPDSite], n: int, batches: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """(num, den, dd), each (n,) fp64 CPU, summed over every position of `batches` in `net`'s own run: num and den as in
    `gains` (g = num / den), dd = |write_F_u - write_M_u|^2 (added 2026-11-21 for experiments/thin_edit_diagnosis, P1.47).
    The share of u's write change that its fitted gain explains is (num - den)^2 / (den dd): the change projected on M's
    own write; what standing u in at that gain leaves of it is dd - (num - den)^2 / den."""
    sums, got = torch.zeros(3, n, dtype=torch.float64), {}
    hs = [net.get_submodule(s.module).register_forward_pre_hook(lambda _m, a, j=j: got.__setitem__(j, a[0])) for j, s in enumerate(sites)]
    try:
        for x in batches:
            net(x)
            for j, s in enumerate(sites):
                y = got[j].flatten(0, -2)
                a, b = s.gain_sums(y)
                sums[:, s.idx] += torch.stack([a, b, s.change_sums(y)]).cpu()
    finally:
        for h in hs:
            h.remove()
    return sums[0], sums[1], sums[2]


class StandIn:
    """Forward hooks on the sites' modules; idle (the forward unchanged) outside `weights`."""

    def __init__(self, net: Any, sites: list[Any], g: Tensor, device: torch.device) -> None:
        self.sites, self.g = sites, g.to(device, torch.float32)
        self.t: Tensor | None = None
        self.live: list[bool] = []
        self.handles = [net.get_submodule(s.module).register_forward_hook(partial(self._hook, j)) for j, s in enumerate(sites)]

    def close(self) -> None:
        for h in self.handles:
            h.remove()

    def _hook(self, j: int, _m: Any, inp: tuple[Tensor, ...], out: Tensor) -> Tensor | None:
        if self.t is None or not self.live[j]:
            return None
        s = self.sites[j]
        return out + s.correction(inp[0], self.t[s.idx], self.g[s.idx])

    def run(self, forward: Callable[[Tensor], Tensor], x: Tensor, t: Tensor) -> Tensor:
        """forward(x) with every variable at weight t (n,) on the device (1 stood in, 0 F's own); differentiable in t."""
        tc = t.detach().cpu()
        self.live = [bool((tc[s.idx] != 0).any()) for s in self.sites]
        self.t = t
        try:
            return forward(x)
        finally:
            self.t = None


@torch.no_grad()
def kl_mean(si: StandIn, forward: Callable[[Tensor], Tensor], toks: list[Tensor], lF: list[Tensor], t: Tensor) -> float:
    """Mean over the windows of KL(F || forward with weights t); lF: F's own logits at the read positions per batch."""
    return float(torch.cat([R.kl(lf, si.run(forward, x, t)) for x, lf in zip(toks, lF, strict=True)]).mean())


def attribution(si: StandIn, forward: Callable[[Tensor], Tensor], toks: list[Tensor], lF: list[Tensor], t: Tensor) -> Tensor:
    """d sum KL(F || forward with weights t) / d t_u for every variable (CPU), one backward per batch."""
    tt = t.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        for x, lf in zip(toks, lF, strict=True):
            R.kl(lf, si.run(forward, x, tt)).sum().backward()
    assert tt.grad is not None
    return tt.grad.cpu()
