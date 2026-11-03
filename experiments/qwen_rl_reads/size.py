"""E75's two size-matched states, per matrix (plan, "Size-matched states"). Energy is input-weighted: tr(A Σ Aᵀ), `split.weighted_energy`.

A slice of δW at share f of its energy: Q Qᵀ δW, Q a random orthonormal basis of a k-dimensional subspace of the output side,
k = round(f d_out) in [1, d_out], times c, the one scalar that makes its energy exactly f's. The output side because a projection
there does not favour any input direction, so its expected share is k / d_out on any δW; c is near 1, and is reported.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from commute import split

SEED = 75_000
KINDS = ("inspan", "remainder")


def seed_of(site_index: int, kind: str) -> int:
    return SEED + 2 * site_index + KINDS.index(kind)


def slice_of(M: Tensor, k: int, seed: int) -> Tensor:
    """Q Qᵀ M, the k-dimensional random projection; drawn on the smaller of the subspace and its complement."""
    d = M.shape[0]
    if k >= d:
        return M
    r = min(k, d - k)
    gen = torch.Generator(M.device).manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(d, r, generator=gen, device=M.device, dtype=M.dtype))[0]
    P = Q @ (Q.T @ M)
    return P if k == r else M - P


def site_record(M: Tensor, inspan: Tensor, S: Tensor, site_index: int) -> dict[str, float]:
    """One matrix: the energies of δW and of its two parts, and each slice's k and c."""
    E = split.weighted_energy(M, S)
    rec = {"E": E, "inspan": split.weighted_energy(inspan, S), "remainder": split.weighted_energy(M - inspan, S)}
    for kind in KINDS:
        k = min(M.shape[0], max(1, round(rec[kind] / E * M.shape[0]))) if E > 0 else 1
        got = split.weighted_energy(slice_of(M, k, seed_of(site_index, kind)), S)
        rec |= {f"k_{kind}": k, f"c_{kind}": math.sqrt(rec[kind] / got) if got > 0 else 0.0}
    return rec


def pooled(sites: dict[str, dict[str, float]]) -> dict[str, Any]:
    """t: each part's input-weighted norm over δW's, ratio of sums over the matrices, as stage S pools; the scale of the rescaled states."""
    E = sum(r["E"] for r in sites.values())
    return {"sites": sites, "energy": E, "t": {k: math.sqrt(sum(r[k] for r in sites.values()) / E) for k in KINDS},
            "c_range": {k: [min(r[f"c_{k}"] for r in sites.values()), max(r[f"c_{k}"] for r in sites.values())] for k in KINDS}}
