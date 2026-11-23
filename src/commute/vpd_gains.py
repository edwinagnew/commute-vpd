"""Gain-only updates at the VPD featurization (cell P1.58, E282): one gain per subcomponent of each of the 24 decomposed
matrices and one per matrix remainder, the counterpart of `commute.gains` (one gain per MLP neuron and attention head).

A matrix W0 = R + sum_c U_c V_c, with (U (C, d_out), V (d_in, C)) as `vpd.component_weights` gives them and R = W0 -
U^T V^T its remainder. Gains (g_c, g_R) give W' = g_R R + sum_c g_c U_c V_c, written here as W0 plus the change

    dW = (g_R - 1) R + U^T diag(g - 1) V^T,

so gains of exactly 1 add an exact zero: the hooked forward is bit-identical to M's, and `delta` exports exactly zero.
`applied` adds the change's output to each matrix's output through a forward hook, x @ dW^T = (g_R - 1) x R^T +
((x V) * (g - 1)) U, which costs a few tokens' worth of matmuls rather than building dW at every step.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class Basis:
    """Per decomposed matrix: U (C, d_out), V (d_in, C) and the remainder R = W0 - U^T V^T, on one device, fp32."""

    U: dict[str, Tensor]
    V: dict[str, Tensor]
    R: dict[str, Tensor]

    @classmethod
    def of(cls, W0: Mapping[str, Tensor], uv: Mapping[str, tuple[Tensor, Tensor]], device: str | torch.device) -> Basis:
        U = {k: uv[k][0].detach().float().to(device) for k in W0}
        V = {k: uv[k][1].detach().float().to(device) for k in W0}
        return cls(U, V, {k: W0[k].float().to(device) - U[k].T @ V[k].T for k in W0})

    @property
    def keys(self) -> list[str]:
        return list(self.U)

    def identity_rel(self, W0: Mapping[str, Tensor]) -> dict[str, float]:
        """Per matrix |R + U^T V^T - W0| / |W0| (Frobenius): fp32 rounding only, for the remainder identity check."""
        out = {}
        for k in self.keys:
            w = W0[k].float().to(self.R[k].device)
            out[k] = float((self.R[k] + self.U[k].T @ self.V[k].T - w).norm() / w.norm())
        return out


@dataclass
class VGains:
    comp: dict[str, Tensor]  # (C,) per matrix: subcomponent c's write times comp[k][c]
    rem: dict[str, Tensor]  # () per matrix: its remainder times rem[k]

    @classmethod
    def ones(cls, b: Basis) -> VGains:
        dev = next(iter(b.U.values())).device
        return cls({k: torch.ones(b.U[k].shape[0], device=dev) for k in b.keys}, {k: torch.ones((), device=dev) for k in b.keys})

    def params(self) -> list[Tensor]:
        return [*self.comp.values(), *self.rem.values()]

    def state(self) -> dict[str, Any]:
        return {"comp": {k: v.detach().cpu() for k, v in self.comp.items()}, "rem": {k: v.detach().cpu() for k, v in self.rem.items()}}

    def count(self) -> int:
        return sum(v.numel() for v in self.params())


@contextmanager
def applied(net: Any, b: Basis, g: VGains) -> Iterator[None]:
    """The net with gains g: a forward hook on each decomposed nn.Linear (by module name, e.g. h.2.attn.o_proj) adds
    x @ dW^T to its output. The weights are not touched, so the base model is the same object once the hooks go."""
    handles = []
    for k in b.keys:
        mod = net.get_submodule(k)
        hook = lambda _m, a, o, k=k: o + (g.rem[k] - 1) * F.linear(a[0], b.R[k]) + ((a[0] @ b.V[k]) * (g.comp[k] - 1)) @ b.U[k]
        handles.append(mod.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def delta(b: Basis, g: VGains) -> dict[str, Tensor]:
    """The weight change equal to gains g, fp32 CPU, one entry per decomposed matrix; gains of exactly 1 give exactly 0."""
    return {k: ((g.rem[k] - 1) * b.R[k] + (b.U[k].T * (g.comp[k] - 1)[None, :]) @ b.V[k].T).float().cpu() for k in b.keys}
