"""Per-subcomponent signatures of a weight update (`experiments/new_vs_existing`).

Where the update lands: the least-squares split writes delta-W as (dU, dV) on the alive subcomponents, and
`split_energy` is the size of each subcomponent's change, ||dU_c v_c^T + u_c dV_c^T||_F^2. The `top_by` set of
largest changes is "where the update landed" (T in the plan). Whether those subcomponents did the job already: with
`ablated_weights` the model runs with their rank-1 matrices removed from the target weights, and `trigger_stats`
reads the target's log-odds and the KL at the trigger against the unablated model. `matched_random` draws control
sets of the same size per matrix from the same decile of causal importance at the trigger. `pile_gate_stats` reads,
over Pile rows at W0, how often each subcomponent is on and how much more it is on where the model predicts the
current token again (the natural "repeat the token" positions). Selections are `recruit.Sel`: matrix -> full
component indices. Plan: reports/2026-10-20_plan_new-vs-existing-signatures.md.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from commute.data import target_tokens
from commute.identity import log_odds
from commute.recruit import ContrastAcc, Gates, Sel, cat_gates
from commute.vpd import compute_ci, kl_per_position, module_keys


def split_energy(dW: dict[str, Tensor], uv: dict[str, tuple[Tensor, Tensor]], bases: dict[str, Any],
                alive: dict[str, Tensor]) -> dict[str, Tensor]:
    """Per matrix, (C,) float64 CPU: ||dU_c v_c^T + u_c dV_c^T||_F^2 of the least-squares split of dW[k] for each alive
    subcomponent c, 0 for dead ones and for matrices with a zero update."""
    from commute.split import least_squares_split

    out = {}
    for k, (U, V) in uv.items():
        full = torch.zeros(alive[k].numel(), dtype=torch.float64)
        if k in dW and dW[k].any():
            U, V = U.double(), V.double()
            dU, dV = least_squares_split(dW[k].double().to(U.device), bases[k], U, V)
            e = (dU.pow(2).sum(0) * V.pow(2).sum(0) + U.pow(2).sum(0) * dV.pow(2).sum(0)
                 + 2 * (dU * U).sum(0) * (V * dV).sum(0))
            full[alive[k].cpu().bool()] = e.cpu()
        out[k] = full
    return out


def top_by(values: dict[str, Tensor], alive: dict[str, Tensor], k: int) -> Sel:
    """The k alive subcomponents of largest value over all matrices, as full indices per matrix (every matrix present)."""
    keys = sorted(values)
    flat = torch.cat([torch.where(alive[m].cpu().bool(), values[m].double().cpu(), torch.tensor(float("-inf"))) for m in keys])
    owner = torch.cat([torch.full((values[m].numel(),), i) for i, m in enumerate(keys)])
    local = torch.cat([torch.arange(values[m].numel()) for m in keys])
    top = flat.topk(min(k, int(torch.isfinite(flat).sum()))).indices
    return {m: local[top[owner[top] == i]].sort().values for i, m in enumerate(keys)}


def share_in(values: dict[str, Tensor], sel: Sel) -> float:
    """Sum of values over the selection / sum over everything."""
    tot = sum(float(v.double().sum()) for v in values.values())
    return sum(float(values[m].double()[idx].sum()) for m, idx in sel.items() if idx.numel()) / tot if tot else float("nan")


def mean_over(values: dict[str, Tensor], sel: Sel) -> float:
    """Mean of a per-subcomponent quantity over the selection's members."""
    parts = [values[m].double()[idx] for m, idx in sel.items() if idx.numel()]
    return float(torch.cat(parts).mean()) if parts else float("nan")


def trigger_g(gates: Gates, half: int) -> dict[str, Tensor]:
    """Mean over windows of Gamma's g at the trigger, per matrix (C,) float32 CPU, from `route_cells` gates."""
    return {k: v[:, half].float().mean(0).cpu() for k, v in cat_gates(gates).items()}


def matched_random(sel: Sel, alive: dict[str, Tensor], g: dict[str, Tensor], seed: int, bins: int = 10) -> Sel:
    """A set of the same size per matrix as `sel`, drawn without replacement from the alive subcomponents outside
    `sel` in the same decile of `g` (ranked over that matrix's alive subcomponents). A decile with too few candidates
    is topped up from the whole matrix."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k, idx in sel.items():
        al = alive[k].cpu().bool().nonzero().flatten()
        rank = torch.empty_like(al)
        rank[g[k].cpu()[al].argsort()] = torch.arange(al.numel())
        bin_of = torch.zeros(alive[k].numel(), dtype=torch.long)
        bin_of[al] = rank * bins // al.numel()
        free = alive[k].cpu().bool().clone()
        free[idx.cpu()] = False
        chosen = []
        for c in idx.tolist():
            pool = (free & (bin_of == bin_of[c])).nonzero().flatten()
            if not pool.numel():
                pool = free.nonzero().flatten()
            pick = pool[torch.randint(pool.numel(), (1,), generator=gen)]
            free[pick] = False
            chosen.append(pick)
        out[k] = torch.cat(chosen).sort().values if chosen else torch.zeros(0, dtype=torch.long)
    return out


def ablated_weights(W: dict[str, Tensor], comps: dict[str, tuple[Tensor, Tensor]], sel: Sel) -> dict[str, Tensor]:
    """W with the selected subcomponents' rank-1 matrices removed: W_k - sum_{c in sel[k]} u_c v_c^T, from the
    param-decomp layout (U (C, d_out), V (d_in, C))."""
    out = dict(W)
    for k, idx in sel.items():
        if idx.numel():
            U, V = comps[k]
            idx = idx.to(U.device)
            out[k] = W[k] - (U[idx].T.float() @ V[:, idx].T.float()).to(W[k].device, W[k].dtype)
    return out


@torch.no_grad()
def trigger_logits(model: Any, windows: Tensor, half: int, batch: int = 64) -> Tensor:
    """(N, V) float32 CPU logits of the target model as it stands, at the trigger of each window."""
    from commute.finetune import logits_of

    device = next(model.parameters()).device
    return torch.cat([logits_of(model.target_model, windows[s : s + batch].to(device))[:, half].cpu()
                      for s in range(0, windows.shape[0], batch)])


def trigger_stats(ref: Tensor, logits: Tensor, windows: Tensor, half: int, token: int | Tensor) -> dict[str, float]:
    """Mean over windows of the target's log-odds under `logits`, the drop from `ref` (positive = the target lost
    probability), and KL(ref || logits), all at the trigger. `ref`, `logits` from `trigger_logits`."""
    tgt = target_tokens(token, windows, half)
    lo, lo_ref = log_odds(logits, tgt), log_odds(ref, tgt)
    return {"lo": float(lo.mean()), "drop": float((lo_ref - lo).mean()), "kl": float(kl_per_position(ref, logits).mean())}


@torch.no_grad()
def pile_gate_stats(model: Any, batches: list[Tensor], alive: dict[str, Tensor], on: float = 0.5) -> dict[str, Any]:
    """At the model's current state over Pile rows: per subcomponent the share of positions with g > `on`
    (`on_rate`, (C,) per matrix) and the contrast of g at the positions where the model predicts the current token
    again with P > 0.5 (`repeat_contrast`, mean there minus mean everywhere; `n_repeat` such positions)."""
    keys = module_keys(model)
    on_sum = {k: torch.zeros(model.module_to_c[k], dtype=torch.float64) for k in keys}
    acc, n = ContrastAcc(), 0
    for b in batches:
        ci = compute_ci(model, b)
        p_same = torch.softmax(ci.logits.float(), -1).gather(-1, b.to(ci.logits.device)[..., None]).squeeze(-1)
        acc.add(ci.g, p_same > 0.5)
        for k in keys:
            on_sum[k] += (ci.g[k] > on).sum(dim=(0, 1)).double().cpu()
        n += b.numel()
        del ci
    return {"on_rate": {k: (v / n).float() for k, v in on_sum.items()}, "repeat_contrast": acc.result(alive),
            "n_repeat": acc.n_on, "n_positions": n}
