"""Readings from the singular values of a weight change dW alone: no model, no forward pass, no W0.

Per matrix, with s the singular values of dW:
  erank      effective rank exp(H(p)), p = s / sum(s) (Roy and Vetterli 2007); 1 = rank one, k = k equal singular values
  top10      share of ||dW||^2 in the 10 largest singular values, sum_{i<=10} s_i^2 / sum s^2; 1 = rank 10 or less
  norm2      ||dW||^2
A state's erank and top10 are averaged over its matrices with weights ||dW_m||^2 / sum ||dW||^2, as E51
(experiments/baseline_weights/run.py, Base.state) aggregated them. At dW = 0 both are 0/0: `state` returns None for them.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor

Mat = dict[str, Tensor]
TOP = 10


def matrix(dW: Tensor) -> dict[str, float]:
    s = torch.linalg.svdvals(dW.double())
    e = float(s.pow(2).sum())
    if e == 0.0:
        return {"erank": math.nan, "top10": math.nan, "norm2": 0.0}
    p = s[s > 0] / s.sum()
    return {"erank": math.exp(float(-(p * p.log()).sum())), "top10": float(s[:TOP].pow(2).sum()) / e, "norm2": e}


def state(dW: Mat) -> dict[str, float | None]:
    """The ||dW_m||^2-weighted erank and top10 over the matrices, and the total ||dW||^2; None where undefined."""
    return aggregate({m: matrix(dW[m]) for m in sorted(dW)})


def aggregate(per: dict[str, dict[str, float]]) -> dict[str, float | None]:
    """`state` from per-matrix `matrix` readings, so a caller can read a large update one matrix at a time (P3.6)."""
    total = sum(v["norm2"] for v in per.values())
    out: dict[str, float | None] = {"norm2": total}
    for k in ("erank", "top10"):
        out[k] = None if total == 0.0 else sum(v[k] * v["norm2"] for v in per.values() if v["norm2"] > 0) / total
    return out


def load(path: Path) -> Mat:
    """A saved dW dict (fp32 or bf16), as fp64; the 24 decomposed matrices of the 67M."""
    d = torch.load(path, map_location="cpu", weights_only=True)
    assert len(d) == 24 and all(v.ndim == 2 for v in d.values()), path
    return {m: v.double() for m, v in d.items()}


def equal_rank(shapes: dict[str, tuple[int, int]], k: int, seed: int) -> Mat:
    """Per matrix, k orthonormal pairs with equal singular values: erank exactly k, top10 = 1 for k <= 10."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for m, (d_out, d_in) in sorted(shapes.items()):
        u = torch.linalg.qr(torch.randn(d_out, k, generator=g, dtype=torch.float64)).Q
        v = torch.linalg.qr(torch.randn(d_in, k, generator=g, dtype=torch.float64)).Q
        out[m] = u @ v.T
    return out
