"""Perturbation nulls of E46 stage 0 (plan reports/plans/2026-10-28_plan_regating-line-reset.md, "Amendment 1 adopted").
Added 2026-10-28. Each is a weight change added to W0 as a do-nothing state (released subcomponents, Gamma frozen):
  flipped   -delta-W of a fine-tune;
  random    a Gaussian delta-W (`finetune.random_direction`: the fine-tune's per-matrix norms) times c, c found by
            `match` so the full model's kl_other to W0 equals the fine-tune's, c <= SCALE_CAP;
  write     on block 1's MLP output matrix only, the minimum-norm delta-W with delta-W a_k = alpha u_t(k) for every key k
            (`key_write`), a_k the mean input to that matrix at the key position, u_t the unembedding row of key k's
            answer; alpha found by `match` so the real gain equals the fine-tune's, alpha <= ALPHA_MAX.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

WRITE_KEY = "h.1.mlp.down_proj"  # block 1's MLP output: (d_model 768, d_mlp 3072), input = the MLP's hidden activation
SCALE_CAP, ALPHA_MAX = 10.0, 1e3
ITERS = 40


def flipped(dW: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: -v for k, v in dW.items()}


def scaled(d: dict[str, Tensor], c: float) -> dict[str, Tensor]:
    return {k: c * v for k, v in d.items()}


def match(f: Callable[[float], float], target: float, hi: float, rel_tol: float) -> dict[str, Any]:
    """The x in (0, hi] with |f(x) / target - 1| <= rel_tol, by bisection, for f increasing with f(0) = 0 and target > 0.
    status "ok"; "cap" if f(hi) < target (1 - rel_tol): x = hi; "no_match" if ITERS halvings never land in the band (x
    the last midpoint). Every evaluation is returned in `trace`."""
    assert target > 0, target
    trace = [(hi, f(hi))]
    if trace[0][1] < target * (1 - rel_tol):
        return {"x": hi, "f": trace[0][1], "status": "cap", "trace": trace}
    lo, up = 0.0, hi
    for _ in range(ITERS):
        if abs(trace[-1][1] / target - 1) <= rel_tol:
            return {"x": trace[-1][0], "f": trace[-1][1], "status": "ok", "trace": trace}
        lo, up = (trace[-1][0], up) if trace[-1][1] < target else (lo, trace[-1][0])
        mid = (lo + up) / 2
        trace.append((mid, f(mid)))
    ok = abs(trace[-1][1] / target - 1) <= rel_tol
    return {"x": trace[-1][0], "f": trace[-1][1], "status": "ok" if ok else "no_match", "trace": trace}


def key_write(A: Tensor, U: Tensor) -> Tensor:
    """The minimum-Frobenius-norm X (d_out, d_in) with X A = U: U pinv(A), float64. A (d_in, K) the key inputs a_k as
    columns, U (d_out, K) the answers' unembedding rows as columns. Exact when A has full column rank."""
    return U.double() @ torch.linalg.pinv(A.double())


@torch.no_grad()
def key_inputs(model: Any, windows: Tensor, half: int, keys: list[int], batch: int = 64) -> Tensor:
    """(d_in, K): the mean input to WRITE_KEY at position `half`, over the windows holding each key there, under the model
    as it stands (bf16 autocast forward, read in float32)."""
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
    X, at = torch.cat(acts), windows[:, half].to(acts[0].device)
    assert all(bool((at == k).any()) for k in keys), "a key with no training window"
    return torch.stack([X[at == k].mean(0) for k in keys], 1)
