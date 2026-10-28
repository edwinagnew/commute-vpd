"""Fine-tunes held to rank r per matrix, and how much of an update lies beyond rank r (E39 Amendment 2,
`experiments/broad_fragility/amend2.py`; plan reports/plans/2026-10-27_plan_broad-fragility.md -> "Amendment 2").
Added 2026-10-28.

After every optimizer step, each decomposed matrix's change W - W0 is replaced by its best rank-r approximation
(`truncate`). The update then never has more than r directions per matrix, as a rank-r LoRA's B A cannot, while the
optimiser and the full-matrix parametrisation stay those of `finetune.finetune_arm_b`.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor

from commute.finetune import (
    ArmBConfig,
    FinetuneResult,
    _run,
    decomposed_params,
    eval_arm_b,
    next_token_ce,
)
from commute.vpd import target_weights


def truncate(D: Tensor, r: int) -> Tensor:
    """The best rank-r approximation of D in Frobenius norm (Eckart-Young): D restricted to the top r eigenvectors of
    its smaller Gram matrix, in float64, returned in D's dtype."""
    Dd = D.double()
    if Dd.shape[0] <= Dd.shape[1]:
        P = torch.linalg.eigh(Dd @ Dd.T).eigenvectors[:, -r:]  # eigenvalues ascend
        return (P @ (P.T @ Dd)).to(D.dtype)
    Q = torch.linalg.eigh(Dd.T @ Dd).eigenvectors[:, -r:]
    return ((Dd @ Q) @ Q.T).to(D.dtype)


def tail_share(dW: dict[str, Tensor], r: int) -> dict[str, float]:
    """The share of each update's squared Frobenius norm beyond its top r singular values: pooled over matrices (sum of
    tails over sum of norms) and the largest of any one matrix. Zero matrices are skipped; all zero gives 0, 0."""
    tail, total, worst = 0.0, 0.0, 0.0
    for M in dW.values():
        s2 = torch.linalg.svdvals(M.double()) ** 2
        t, n = float(s2[r:].sum()), float(s2.sum())
        if n > 0:
            tail, total, worst = tail + t, total + n, max(worst, t / n)
    return {"pooled": tail / total if total > 0 else 0.0, "max": worst}


def finetune_arm_b_rank_capped(model: Any, train_rows: Tensor, heldout_rows: Tensor, cfg: ArmBConfig, rank: int,
                               device: str, log: Any = print) -> FinetuneResult:
    """`finetune.finetune_arm_b` (arm B's loss on `cfg.batch` rows per step in order, each row once; the clip, the
    held-out reading and `cfg.optimizer`), with every matrix's W - W0 cut to its best rank-`rank` approximation after
    every optimizer step, before that step's checkpoint. The model is W0 again on return (`finetune._run`)."""
    ref = copy.deepcopy(model.target_model).eval().requires_grad_(False)
    params = decomposed_params(model)
    W0 = target_weights(model)
    net = model.target_model
    assert train_rows.shape[0] // cfg.batch >= cfg.steps, f"need {cfg.steps * cfg.batch} fresh train rows"

    def loss_fn(step: int) -> tuple[Tensor, dict[str, float]]:
        return next_token_ce(net, train_rows[(step - 1) * cfg.batch : step * cfg.batch].to(device))

    def eval_fn() -> dict[str, float]:
        return eval_arm_b(net, ref, heldout_rows, device)

    @torch.no_grad()
    def cap(_step: int, ps: dict[str, Tensor]) -> None:
        for k, p in ps.items():
            p.copy_((W0[k] + truncate(p.float() - W0[k], rank)).to(p.dtype))

    return _run(model, params, cfg.lr, cfg.grad_clip, cfg.steps, cfg.checkpoints, loss_fn, eval_fn, log, cfg.optimizer, cap)
