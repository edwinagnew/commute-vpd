"""Perturbation nulls of E46 stage 0 (plan reports/plans/2026-10-28_plan_regating-line-reset.md, "Amendment 1 adopted").
Added 2026-10-28. Each is a weight change added to W0 as a do-nothing state (released subcomponents, Gamma frozen):
  flipped   -delta-W of a fine-tune;
  random    a Gaussian delta-W (`finetune.random_direction`: the fine-tune's per-matrix norms) times c, c found by
            `match` so the full model's kl_other to W0 equals the fine-tune's, c <= SCALE_CAP;
  write     on block 1's MLP output matrix only, the minimum-norm delta-W with delta-W a_k = alpha u_t(k) for every key k
            (`key_write`), a_k the mean input to that matrix at the key position, u_t the unembedding row of key k's
            answer; alpha found by `match` so the real gain equals the fine-tune's, alpha <= ALPHA_MAX. `write_fit`
            reads how well the write holds on single (held-out) windows, where it is not exact by construction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

WRITE_KEY = "h.1.mlp.down_proj"  # block 1's MLP output: (d_model 768, d_mlp 3072), input = the MLP's hidden activation
SCALE_CAP, ALPHA_MAX = 10.0, 1e3
GRID_LO, GRID_N, ITERS = 1e-2, 12, 40


def flipped(dW: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: -v for k, v in dW.items()}


def scaled(d: dict[str, Tensor], c: float) -> dict[str, Tensor]:
    return {k: c * v for k, v in d.items()}


def match(f: Callable[[float], float], target: float, hi: float, rel_tol: float) -> dict[str, Any]:
    """x in (0, hi] with |f(x) / target - 1| <= rel_tol (target > 0). f is read on a log grid GRID_LO .. hi (GRID_N
    points, kept in `grid`); a grid point in the band is returned; else bisection inside the first grid interval whose
    right end reaches the target (left end: the previous point, or 0). status "ok"; "cap" if no grid point reaches
    target (1 - rel_tol): x = hi; "no_match" if ITERS halvings never land in the band (x the last midpoint)."""
    assert target > 0, target
    xs = torch.logspace(torch.log10(torch.tensor(GRID_LO)), torch.log10(torch.tensor(hi)), GRID_N).tolist()
    grid: list[tuple[float, float]] = []
    lo = 0.0
    for x in xs:
        grid.append((x, f(x)))
        if abs(grid[-1][1] / target - 1) <= rel_tol:
            return {"x": x, "f": grid[-1][1], "status": "ok", "grid": grid, "trace": []}
        if grid[-1][1] > target:
            break
        lo = x
    else:
        return {"x": hi, "f": grid[-1][1], "status": "cap", "grid": grid, "trace": []}
    up, trace = grid[-1][0], []
    for _ in range(ITERS):
        mid = (lo + up) / 2
        trace.append((mid, f(mid)))
        if abs(trace[-1][1] / target - 1) <= rel_tol:
            return {"x": mid, "f": trace[-1][1], "status": "ok", "grid": grid, "trace": trace}
        lo, up = (mid, up) if trace[-1][1] < target else (lo, mid)
    return {"x": trace[-1][0], "f": trace[-1][1], "status": "no_match", "grid": grid, "trace": trace}


def key_write(A: Tensor, U: Tensor) -> Tensor:
    """The minimum-Frobenius-norm X (d_out, d_in) with X A = U: U pinv(A), float64. A (d_in, K) the key inputs a_k as
    columns, U (d_out, K) the answers' unembedding rows as columns. Exact on those K columns by construction."""
    return U.double() @ torch.linalg.pinv(A.double())


def write_fit(X: Tensor, A: Tensor, a: Tensor, u: Tensor) -> dict[str, float]:
    """How the write holds where it is not exact: cond(A), ||X||_F, and over single windows (inputs a (N, d_in), their
    answers' unembedding rows u (N, d_out)) the cosine of X a_i with u_i and ||X a_i - u_i|| / ||u_i|| (scale-free in
    alpha), mean and max."""
    pred = a.double() @ X.T
    rel = (pred - u.double()).norm(dim=1) / u.double().norm(dim=1)
    cos = torch.nn.functional.cosine_similarity(pred, u.double(), dim=1)
    return {"cond_A": float(torch.linalg.cond(A.double())), "X_norm": float(X.norm()), "rel_err_mean": float(rel.mean()),
            "rel_err_max": float(rel.max()), "cos_mean": float(cos.mean()), "cos_min": float(cos.min()), "n_windows": a.shape[0]}


@torch.no_grad()
def window_inputs(model: Any, windows: Tensor, half: int, batch: int = 64) -> Tensor:
    """(N, d_in): each window's input to WRITE_KEY at position `half`, the model as it stands (bf16 autocast, float32)."""
    from commute.finetune import logits_of
    from commute.vpd import target_linear

    acts: list[Tensor] = []
    h = target_linear(model, WRITE_KEY).register_forward_hook(lambda _m, a, _o: acts.append(a[0][:, half].float()))
    try:
        device = next(model.parameters()).device
        for s in range(0, windows.shape[0], batch):
            logits_of(model.target_model, windows[s : s + batch].to(device))
    finally:
        h.remove()
    return torch.cat(acts)


def key_inputs(model: Any, windows: Tensor, half: int, keys: list[int]) -> Tensor:
    """(d_in, K): the mean of `window_inputs` over the windows holding each key at `half`."""
    X = window_inputs(model, windows, half)
    at = windows[:, half].to(X.device)
    assert all(bool((at == k).any()) for k in keys), "a key with no training window"
    return torch.stack([X[at == k].mean(0) for k in keys], 1)
