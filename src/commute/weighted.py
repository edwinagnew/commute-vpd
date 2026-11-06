"""Input-weighted comparison of weight changes over the 24 decomposed matrices (E34 Amendment 10's arithmetic, moved here
2026-11-06 for E121 copy_gap_ladder; `experiments/teacher_training/amend10.py` keeps its own copy as it ran).

For two weight changes A, B: <A, B>_Sigma = sum_k tr(A_k Sigma_k B_k^T) = E sum_k <A_k x, B_k x>, x the input matrix k
sees in the unchanged model W0 (uncentred second moment, every position), read with no autocast and summed in fp64.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from commute.data import batches


def gram(X: Tensor, sigma: Tensor | None) -> Tensor:
    """(N, N) of tr(A sigma B^T) over the rows of X (N, d_out, d_in), fp64; sigma None is the identity: Frobenius."""
    Y = X if sigma is None else X @ sigma
    return Y.flatten(1) @ X.flatten(1).T


def pooled(grams: Iterable[Tensor]) -> Tensor:
    return torch.stack(list(grams)).sum(0)


def _lengths(G: Tensor) -> Tensor:
    d = G.diag().clone()
    d[d <= 0] = math.nan
    return d.sqrt()


def cosines(G: Tensor) -> Tensor:
    """Cosine under an inner product's Gram matrix. A zero or negative diagonal gives nan on its row and column."""
    n = _lengths(G)
    return G / (n[:, None] * n[None, :])


def sizes(G: Tensor) -> Tensor:
    """[a, b]: ||a|| / ||b|| under the Gram. nan as `cosines`."""
    n = _lengths(G)
    return n[:, None] / n[None, :]


def distances(G: Tensor) -> Tensor:
    """[a, b]: ||a - b|| / ||b|| under the Gram. A zero change is at exactly 1. nan as `cosines`."""
    n = _lengths(G)
    return (n[:, None] ** 2 - 2 * G + n[None, :] ** 2).clamp_min(0).sqrt() / n[None, :]


@torch.no_grad()
def input_moments(model: Any, rows: Tensor, batch: int) -> dict[str, Tensor]:
    """E[x x^T] per decomposed matrix over every position of `rows`, fp64 from an fp32 forward (no autocast): W0's own
    inputs when the model is at W0. Amendment 10's `moments` (position None) divided by the count."""
    device = next(model.parameters()).device
    S: dict[str, Tensor] = {}
    n = 0
    for b in batches(rows, batch):
        cache = model(b.to(device), cache_type="input").cache
        for k, v in cache.items():
            x = v.reshape(-1, v.shape[-1]).double()
            S[k] = S.get(k, 0) + x.T @ x
        n += b.numel()
    return {k: s / n for k, s in S.items()}


def gram_of(vec: dict[str, dict[str, Tensor]], sigma: dict[str, Tensor] | None, device: str) -> Tensor:
    """The pooled (N, N) Gram of every change in `vec` (name -> matrix -> change), in `vec`'s order, on the CPU."""
    keys = list(next(iter(vec.values())))
    out = []
    for k in keys:
        X = torch.stack([v[k].double().to(device) for v in vec.values()])
        out.append(gram(X, None if sigma is None else sigma[k].to(device)).cpu())
    return pooled(out)
