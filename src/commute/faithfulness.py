"""Faithfulness of a decomposition to the model it is supposed to explain, by the authors' metrics.

With the model's target weights and components set to the variant under test (the frozen Gamma
reads the variant's own activations): stochastic reconstruction KL per batch of rows
(`StochasticReconSubsetLoss`: uniform-k-subset routing, continuous masks, Delta included under
its own per-position mask); adversarial PGD KL with ONE adversary shared across every row of
every batch (`calc_multibatch_pgd_masked_recon_loss`: all layers, 20 steps of 0.1, random init,
Delta included, gradients accumulated over the batches), which is the paper's evaluation scope
(one adversary per 128-row eval batch) on a 24 GB card; the CI-masked KL (masks = g, Delta
excluded) and the unmasked KL (masks = 1, Delta excluded). See
reports/2026-10-14_decision_pgd-scope.md for why the per-batch adversary was dropped.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from torch import Tensor

from commute.vpd import (
    component_weights,
    compute_ci,
    kl_per_position,
    set_component_weights,
    set_target_weights,
    target_weights,
    weight_deltas,
)


@contextlib.contextmanager
def model_state(model: Any, target: dict[str, Tensor] | None = None, components: dict[str, tuple[Tensor, Tensor]] | None = None):
    """Temporarily write target weights and/or component (U, V); restore on exit."""
    saved_t = target_weights(model) if target is not None else None
    saved_c = component_weights(model) if components is not None else None
    try:
        if target is not None:
            set_target_weights(model, target)
        if components is not None:
            set_component_weights(model, components)
        yield
    finally:
        if saved_t is not None:
            set_target_weights(model, saved_t)
        if saved_c is not None:
            set_component_weights(model, saved_c)


def transported_components(model: Any, scale: dict[str, tuple[Tensor, Tensor]] | None = None,
                           dU: dict[str, tuple[Tensor, Tensor]] | None = None,
                           dV: dict[str, tuple[Tensor, Tensor]] | None = None) -> dict[str, tuple[Tensor, Tensor]]:
    """Component weights after a transport. `scale[k] = (idx, alpha)`: U_c <- (1 + alpha_c) U_c;
    `dU[k] = (idx, dU)`: U_c <- U_c + dU_c; `dV[k] = (idx, dV)`: V_c <- V_c + dV_c."""
    comps = component_weights(model)
    for k, (U, V) in comps.items():
        U, V = U.clone(), V.clone()
        if scale and k in scale:
            idx, a = scale[k]
            U[idx] = U[idx] * (1 + a.to(U.dtype))[:, None]
        if dU and k in dU:
            idx, d = dU[k]
            U[idx] = U[idx] + d.to(U.dtype)
        if dV and k in dV:
            idx, d = dV[k]
            V[:, idx] = V[:, idx] + d.to(V.dtype)
        comps[k] = (U, V)
    return comps


def faithfulness(model: Any, batches: list[Tensor], n_stoch_draws: int = 2, pgd_steps: int = 20, seed: int = 0,
                 log: Any = print) -> dict[str, float]:
    from param_decomp.configs import PGDMultiBatchConfig, UniformKSubsetRoutingConfig
    from param_decomp.metrics.pgd_utils import calc_multibatch_pgd_masked_recon_loss
    from param_decomp.metrics.stochastic_recon_subset_loss import (
        stochastic_recon_subset_loss,
    )
    from param_decomp.models.batch_and_loss_fns import recon_loss_kl
    from param_decomp.models.components import make_mask_infos
    from param_decomp.routing import AllLayersRouter

    device = next(model.parameters()).device
    batches = [b.to(device) for b in batches]
    use_autocast = device.type == "cuda"
    sums = {"stochastic_kl": 0.0, "ci_masked_kl": 0.0, "unmasked_kl": 0.0}
    n = 0
    with torch.no_grad():
        wd = weight_deltas(model)
        torch.manual_seed(seed)
        for batch in batches:
            ci = compute_ci(model, batch, chunk=batch.shape[0])
            target_out = ci.logits
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                s = stochastic_recon_subset_loss(model, "continuous", n_stoch_draws, batch, target_out, ci.g, wd,
                                                 UniformKSubsetRoutingConfig(), recon_loss_kl)
                out_ci = model(batch, mask_infos=make_mask_infos(ci.g))
                out_ones = model(batch, mask_infos=make_mask_infos({k: torch.ones_like(g) for k, g in ci.g.items()}))
            b = batch.shape[0]
            sums["stochastic_kl"] += float(s) * b
            sums["ci_masked_kl"] += float(kl_per_position(target_out, out_ci).mean()) * b
            sums["unmasked_kl"] += float(kl_per_position(target_out, out_ones).mean()) * b
            n += b
            del ci, target_out, out_ci, out_ones
            torch.cuda.empty_cache() if use_autocast else None
    res = {k: v / n for k, v in sums.items()}
    # One adversary shared across every row of every batch (the paper's 128-row scope), gradients accumulated over
    # the batches before each sign step. The authors' function needs grad mode on (it calls autograd.grad itself).
    torch.manual_seed(seed)
    pgd_cfg = PGDMultiBatchConfig(init="random", step_size=0.1, n_steps=pgd_steps, gradient_accumulation_steps=len(batches))
    with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
        p = calc_multibatch_pgd_masked_recon_loss(
            pgd_config=pgd_cfg, model=model, weight_deltas=wd, create_data_iter=lambda: iter(batches),
            router=AllLayersRouter(), sampling="continuous", use_delta_component=True, device=str(device),
            reconstruction_loss=recon_loss_kl)
    res["pgd_kl"] = float(p.detach())
    res["pgd_rows"] = n
    torch.cuda.empty_cache() if use_autocast else None
    log("[faith] " + " ".join(f"{k} {v:.4f}" for k, v in res.items()))
    return res
