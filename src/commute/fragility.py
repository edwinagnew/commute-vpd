"""Moves and reverts of the alive subcomponents, and the stable rank of an update (E39, experiments/broad_fragility;
plan reports/plans/2026-10-27_plan_broad-fragility.md). Added 2026-10-27.

A state here is the alive (U', V') per matrix in theory layout (U (d_out, n), V (d_in, n), as
`component_training.alive_uv`) with Delta fixed, so the target moves by U'V'^T - U V^T (`delta_w`).
"""

from __future__ import annotations

import torch
from torch import Tensor

UV = dict[str, tuple[Tensor, Tensor]]


def delta_w(uv: UV, uv2: UV) -> dict[str, Tensor]:
    """U'V'^T - U V^T per matrix of `uv2`, float64 then float32 (as `saved_states.offsets_state`)."""
    return {k: (U2.double() @ V2.double().T - uv[k][0].double() @ uv[k][1].double().T).float() for k, (U2, V2) in uv2.items()}


def _scale(aa: float, ab: float, bb: float, target: float) -> float:
    """a >= 0 with ||a A + a^2 B||_F = target, given ||A||^2, <A, B> and ||B||^2 (bisection; the norm rises with a
    for a random move, where <A, B> is about 0)."""
    def f(a: float) -> float:
        return (a * a * aa + 2 * a**3 * ab + a**4 * bb) ** 0.5

    if target <= 0:
        return 0.0
    lo, hi = 0.0, target / aa**0.5
    while f(hi) < target:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return (lo + hi) / 2


def random_move(uv: UV, norms: dict[str, float], seed: int) -> UV:
    """A Gaussian move of the alive U and V of each matrix in `norms`: noise G_U, G_V each rescaled to its own side's
    Frobenius norm (||G_U|| = ||U||, ||G_V|| = ||V||), so neither side's gauge sets the move, then one scale a per
    matrix, U' = U + a G_U and V' = V + a G_V, with a chosen so ||U'V'^T - U V^T||_F = norms[k]. Matrices are drawn in
    sorted order from one generator at `seed`."""
    gen = torch.Generator().manual_seed(seed)
    out: UV = {}
    for k in sorted(norms):
        U, V = (t.double() for t in uv[k])
        if U.numel() == 0:
            out[k] = uv[k]
            continue
        gU, gV = (torch.randn(t.shape, generator=gen, dtype=torch.float64) for t in (U, V))
        gU, gV = gU * (U.norm() / gU.norm()), gV * (V.norm() / gV.norm())
        A, B = gU @ V.T + U @ gV.T, gU @ gV.T
        a = _scale(float((A * A).sum()), float((A * B).sum()), float((B * B).sum()), norms[k])
        out[k] = ((U + a * gU).float(), (V + a * gV).float())
    return out


def revert(uv: UV, uv2: UV, cols: dict[str, Tensor]) -> UV:
    """`uv2` with the alive columns where cols[k] (bool, (n,)) is True set back to `uv`'s; matrices absent from `cols`
    keep `uv2`'s."""
    out = dict(uv2)
    for k, m in cols.items():
        (U, V), (U2, V2) = uv[k], uv2[k]
        out[k] = (torch.where(m, U, U2), torch.where(m, V, V2))
    return out


def importance_halves(g: dict[str, Tensor], seed: int) -> dict[str, dict[str, Tensor]]:
    """Per matrix, from its alive subcomponents' mean causal importance g[k] (n,): 'rare' = the n // 2 lowest,
    'common' = the other n - n // 2, 'random' = n // 2 drawn uniformly (the null; one generator at `seed`, matrices in
    sorted order). Bool masks (n,)."""
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, dict[str, Tensor]] = {"rare": {}, "common": {}, "random": {}}
    for k in sorted(g):
        n = g[k].numel()
        rare, rnd = torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
        rare[torch.argsort(g[k], stable=True)[: n // 2]] = True
        rnd[torch.randperm(n, generator=gen)[: n // 2]] = True
        out["rare"][k], out["common"][k], out["random"][k] = rare, ~rare, rnd
    return out


def stable_rank(M: Tensor) -> float:
    """||M||_F^2 / ||M||_2^2 (float64)."""
    M = M.double()
    return float((M * M).sum() / torch.linalg.matrix_norm(M, ord=2) ** 2)


def mean_stable_rank(dW: dict[str, Tensor]) -> dict[str, object]:
    """`stable_rank` per non-zero matrix and their mean."""
    per = {k: stable_rank(M) for k, M in dW.items() if M.any()}
    return {"mean": sum(per.values()) / len(per), "per_matrix": per}
