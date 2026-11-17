"""Planted edits at block 2's input, last position: E157's ceiling and floor (cell P1.14), moved here on 2026-11-11 for E159
(cell P1.15) so that both layouts' planted edits are built by one function. Same arithmetic as E157's `class_means`,
`with_in2`, `ceiling_of`, `floor_of` and `calibrate`; E157's own script keeps its copy and is not changed. E159's repeat
check rebuilds E157's ceiling through this code and must read E157's two-site H.

    ceiling  h -> h + s l(h) u, l(h) = (h - mu0).(mu1 - mu0) / |mu1 - mu0|^2, with mu1, mu0 the 67M's own in2 (last
             position) averaged over keyed windows whose last number is odd, even: computed from the value arriving at
             in2, so exactly affine in it.
    floor    67M + s r(a, b) u, r a seeded 0/1 bit per ordered pair of numbers in the last two positions (half ones), 0
             off keyed windows: on held-out pairs no map can know it.
s is calibrated (doubling, then bisection) so the gap TV(F, 67M) matches a target gap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor

from commute import variable_read as VR
from commute.hidden_maps import Edit, chunks, merged, run
from commute.steering import Net
from commute.variable_read import Target

Maker = Callable[[float], Target]


def class_means(net: Net, x: VR.Xor2, fit: Tensor, batch: int, log: Callable[[str], None]) -> tuple[Tensor, Tensor]:
    """The 67M's in2, last position, averaged (float64, returned float32) over `fit`'s keyed windows whose last number is
    even (mu0) and odd (mu1)."""
    w = fit[VR.xor2_keyed(x, fit)]
    sums: list[list[Tensor]] = [[], []]
    n = [0, 0]
    for t in chunks(w, batch):
        h = run(net, t, sites=("in2",))[1]["in2"][:, -1].double()
        odd = VR.odd_at(x, t, -1).bool().to(h.device)
        for k, m in enumerate((~odd, odd)):
            sums[k].append(h[m].sum(0))
            n[k] += int(m.sum())
    assert min(n) > 0
    mu0, mu1 = (torch.stack(sums[k]).sum(0) / n[k] for k in (0, 1))
    log(f"class means: {n[0]} even and {n[1]} odd fit windows, |mu1 - mu0| {float((mu1 - mu0).norm()):.4f}")
    return mu0.float(), mu1.float()


def with_in2(net: Net, fn: Callable[[Tensor], Tensor]) -> Target:
    """F = `net` with `fn` applied at in2 before any caller's edit there."""

    def f(toks: Tensor, sites: Iterable[str] = (), edit: Edit | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        return run(net, toks, sites=sites, edit=merged({"in2": fn}, edit or {}))

    return f


def ceiling_of(net: Net, unit: Tensor, mu0: Tensor, mu1: Tensor) -> Maker:
    """s -> F with h -> h + s l(h) unit at in2, last position, l read from the value arriving there."""
    dmu = (mu1 - mu0) / (mu1 - mu0).pow(2).sum()

    def make(s: float) -> Target:
        def fn(x: Tensor) -> Tensor:
            x = x.clone()
            h = x[:, -1].clone()
            x[:, -1] = h + ((h - mu0) @ dmu)[:, None] * (s * unit)
            return x

        return with_in2(net, fn)

    return make


def floor_of(net: Net, x: VR.Xor2, unit: Tensor, seed: int) -> Maker:
    """s -> F = 67M + s r(a, b) unit at in2, last position; r a seeded bit per ordered key pair, half ones, 0 off keys."""
    k = x.keys.numel()
    idx = torch.full_like(x.bit, -1)
    idx[x.keys] = torch.arange(k)
    r = (torch.randperm(k * k, generator=torch.Generator().manual_seed(seed)) < k * k // 2).view(k, k)

    def label(t: Tensor) -> Tensor:
        a, b = idx[t[:, -2].cpu()], idx[t[:, -1].cpu()]
        return ((a >= 0) & (b >= 0) & r[a.clamp_min(0), b.clamp_min(0)]).float()

    return lambda s: VR.label_target(net, label, s * unit)


def calibrate(net: Net, windows: Tensor, make: Maker, target_gap: float, batch: int, s_max: float, steps: int,
              trace: list[dict[str, Any]], log: Callable[[str], None]) -> float | None:
    """s with TV(make(s), 67M) on `windows` at `target_gap`: doubling from 1, then `steps` halvings of the bracket. Each
    read is appended to `trace`. None if no s up to s_max reaches the target."""
    M = VR.net_target(net)

    def gap(s: float) -> float:
        g = VR.mean_tv(make(s), M, windows, batch)["mean"]
        trace.append({"s": s, "gap": g})
        log(f"  s {s:.6f}: gap {g:.6f}")
        return g

    lo, hi = 0.0, 1.0
    while gap(hi) < target_gap:
        lo, hi = hi, 2 * hi
        if hi > s_max:
            return None
    for _ in range(steps):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) < target_gap else (lo, mid)
    return (lo + hi) / 2
