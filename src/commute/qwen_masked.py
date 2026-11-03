"""The student's masked forward for teacher training with masks in its loss (E76, plan
reports/plans/2026-11-03_plan_e76-masked-teacher.md). `qwen_teacher.Offsets` holds the student: V′ = V + dV and U′ = U + dU on
the alive columns A, weights W0 + change. Each decomposed matrix's output is

    ((x V′) * (m − d)) U′ + d (x W′ᵀ)
      = [((x V) * (m − d)) U + d (x W0ᵀ)]  +  ((x dV) * m_A) (U_A + dU)  +  ((x V_A) * m_A) dU

The bracket is W0's masked forward (`qwen_vpd.masked_logits`), computed once with the components it is given. The two last
terms are exactly 0 at zero offsets, and in m_A the Δ share d cancels: the d (x changeᵀ) from the weights and the −d from the
masks on the alive columns are the same matrix. At m = 1, d = 1 (a routed-out site) the two terms are x changeᵀ. The activations
x stay fp32 whatever the components' dtype, so bf16 components round V and U but no small change in the stream.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor

from commute.qwen_teacher import Offsets
from commute.qwen_vpd import SITES, Components, _hooks, _linear, stochastic_masks


def library_routes(T: int, gen: torch.Generator) -> dict[str, Tensor]:
    """`UniformKRouting.__call__` with k None (param-decomp core/recon.py:224-230), as E77's `uniform_k_routes`: per position k
    uniform on 1..196, and exactly k sites, uniform, take the masked branch. site -> [1, T] bool."""
    k = torch.randint(1, len(SITES) + 1, (1, T), generator=gen, device=gen.device)
    routed = torch.rand((len(SITES), 1, T), generator=gen, device=gen.device).argsort(0) < k
    return {s: routed[j] for j, s in enumerate(SITES)}


def row_masks(g: Mapping[str, Tensor], gen: torch.Generator, route_gen: torch.Generator | None = None,
              ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """One draw of m = g + (1 − g) u, d ~ U(0, 1) from one row's frozen g (any dtype and device; `g` must iterate in SITES order
    to reproduce E74's stream), on `gen`'s device in fp32. With `route_gen`: m = 1, d = 1 at the positions a site is routed out,
    the library's plain x Wᵀ (m = 1, d = 1 in the formula above). Drawn once before the forward, never inside a hook:
    checkpointing runs each hook again during backward."""
    masks, deltas = stochastic_masks({s: v.to(gen.device, torch.float32) for s, v in g.items()}, gen)
    if route_gen is None:
        return masks, deltas
    route = library_routes(next(iter(deltas.values())).shape[1], route_gen)
    return ({s: torch.where(route[s][..., None], m, 1.0) for s, m in masks.items()},
            {s: torch.where(route[s], d, 1.0) for s, d in deltas.items()})


@contextmanager
def student_masked(model: Any, comps: Components, off: Offsets, masks: Mapping[str, Tensor], deltas: Mapping[str, Tensor],
                   ) -> Iterator[None]:
    """The block's forwards are the student's masked forward (module docstring). Call backward inside the block: non-reentrant
    checkpointing recomputes each layer during backward, and the recompute must run the same hooks (`qwen_teacher.train`)."""

    def replace(site: str) -> Any:
        V, U = comps.vu(site)

        def hook(_m: Any, args: tuple[Tensor, ...], out: Tensor) -> Tensor:
            x = args[0]
            m, d = masks[site].to(x), deltas[site].to(x)[..., None]
            base = (x @ V.to(x) * (m - d)) @ U.to(x) + d * out
            m_a = m[..., off.alive[site]]
            return base + (x @ off.dV[site] * m_a) @ (off.U[site] + off.dU[site]) + (x @ off.V[site] * m_a) @ off.dU[site]

        return hook

    with _hooks([_linear(model, s).register_forward_hook(replace(s)) for s in SITES]):
        yield
