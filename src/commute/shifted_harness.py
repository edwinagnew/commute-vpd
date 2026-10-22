"""The Pile harness for a decomposition whose importance values are shifted by one fixed vector (E30, gamma_shift_map).

Reading (i) of `reports/2026-10-22_plan_commute-second-line.md` -> Item D. The decomposition D' keeps U, V and Delta as
at W0 (the model's own state: target weights W0, released components), and its importance function is
    Gamma'(x) = clip(Gamma(x) + s, 0, 1)   at every position,
with Gamma reading W0's activations and s one (C,) vector per matrix (E18's fixed change, `regating.shift_of`). D' is
scored against another model's outputs (the fine-tuned W0 + dW): `target` is a second model holding those weights, and
every KL is KL(target's output || D' masked output). With s = 0 and `target` None (or a copy at W0) every number equals
`faithfulness.faithfulness` / `faithfulness.sparsity_loss` exactly: the same ops in the same order and the same RNG
draws, only the gate dict and the reference logits are swapped (tests/test_shifted_harness.py).

Readings, as round 2's harness: stochastic-mask KL (the authors' `stochastic_recon_subset_loss`, uniform-k-subset
routing, Delta under its own mask), CI-masked KL (masks Gamma', Delta out), unmasked KL (masks 1, Delta out: Gamma'
plays no part), adversarial KL (a copy of the authors' `calc_multibatch_pgd_masked_recon_loss` that takes Gamma' and
the target's logits; one source per matrix over every row, random init, 20 sign steps of 0.1); and VPD's sparsity loss
on Gamma's upper-leaky values plus the change the shift makes to the gates (`shifted_upper`).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from commute.vpd import CI_CHUNK, compute_ci, kl_per_position, weight_deltas

Shift = dict[str, Tensor] | None  # per matrix (C,); None = no shift


def shift_gates(g: dict[str, Tensor], shift: Shift) -> dict[str, Tensor]:
    """clip(g + s, 0, 1) per matrix, added in float32 and returned in g's dtype (so s = 0 returns g's values exactly,
    g being in [0, 1]); matrices absent from `shift` and shift None are left as they are."""
    if shift is None:
        return g
    out = {}
    for k, v in g.items():
        if k not in shift:
            out[k] = v
            continue
        out[k] = (v.float() + shift[k].to(v.device).float()).clamp(0.0, 1.0).to(v.dtype)
    return out


def shifted_upper(upper: dict[str, Tensor], lower: dict[str, Tensor], shift: Shift) -> dict[str, Tensor]:
    """The upper-leaky values the sparsity loss reads, moved by the change the shift makes to the gates:
    upper + (clip(lower + s) - lower), float32. A subcomponent pinned at 1 or 0 by the clip is charged nothing extra;
    s = 0 returns upper exactly."""
    lo2 = shift_gates(lower, shift)
    return {k: upper[k].float() + (lo2[k].float() - lower[k].float()) for k in upper}


def reference_logits(target: Any, batch: Tensor, ci: Any) -> Tensor:
    """The logits D' is scored against: the target model's (`target` a ComponentModel holding W0 + dW, called plainly
    as `vpd.target_logits` does), or the model's own (`ci.logits`) when `target` is None."""
    if target is None:
        return ci.logits
    from commute.vpd import target_logits

    return target_logits(target, batch)


def _first_part(model: Any, target: Any, batches: list[Tensor], shift: Shift, wd: dict[str, Tensor], n_stoch_draws: int,
                seed: int) -> dict[str, float]:
    """`faithfulness.faithfulness`'s per-batch loop with Gamma' and the target's logits."""
    from param_decomp.configs import UniformKSubsetRoutingConfig
    from param_decomp.metrics.stochastic_recon_subset_loss import (
        stochastic_recon_subset_loss,
    )
    from param_decomp.models.batch_and_loss_fns import recon_loss_kl
    from param_decomp.models.components import make_mask_infos

    use_autocast = batches[0].is_cuda
    sums = {"stochastic_kl": 0.0, "ci_masked_kl": 0.0, "unmasked_kl": 0.0}
    n = 0
    torch.manual_seed(seed)
    for batch in batches:
        ci = compute_ci(model, batch, chunk=batch.shape[0])
        g, ref = shift_gates(ci.g, shift), reference_logits(target, batch, ci)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
            s = stochastic_recon_subset_loss(model, "continuous", n_stoch_draws, batch, ref, g, wd,
                                             UniformKSubsetRoutingConfig(), recon_loss_kl)
            out_ci = model(batch, mask_infos=make_mask_infos(g))
            out_ones = model(batch, mask_infos=make_mask_infos({k: torch.ones_like(v) for k, v in g.items()}))
        b = batch.shape[0]
        sums["stochastic_kl"] += float(s) * b
        sums["ci_masked_kl"] += float(kl_per_position(ref, out_ci).mean()) * b
        sums["unmasked_kl"] += float(kl_per_position(ref, out_ones).mean()) * b
        n += b
        del ci, ref, out_ci, out_ones
    return {k: v / n for k, v in sums.items()}


def _pgd_step_sums(model: Any, target: Any, batches: list[Tensor], shift: Shift, wd: dict[str, Tensor],
                   adv: dict[str, Tensor]) -> tuple[Tensor, int, dict[str, Tensor]]:
    """The authors' `_multibatch_pgd_fwd_bwd` (pgd_utils.py) with Gamma' and the target's logits: summed KL, positions,
    and the adversary's summed gradients over every batch."""
    from param_decomp.metrics.pgd_utils import _forward_with_adv_sources
    from param_decomp.models.batch_and_loss_fns import recon_loss_kl
    from param_decomp.routing import AllLayersRouter

    device = batches[0].device
    tot, n = torch.tensor(0.0, device=device), 0
    grads = {k: torch.zeros_like(v) for k, v in adv.items()}
    router = AllLayersRouter()
    for mb in batches:
        out = model(mb, cache_type="input")
        ci = shift_gates(model.calc_causal_importances(pre_weight_acts=out.cache, sampling="continuous").lower_leaky, shift)
        ref = out.output if target is None else target(mb)
        dims = next(iter(ci.values())).shape[:-1]
        loss, m = _forward_with_adv_sources(model=model, batch=mb, adv_sources=adv, ci=ci, weight_deltas=wd,
                                            routing_masks=router.get_masks(module_names=model.target_module_paths,
                                                                           mask_shape=dims),
                                            target_out=ref, batch_dims=dims, reconstruction_loss=recon_loss_kl)
        tot += loss.detach()
        n += m
        for k, gk in zip(adv, torch.autograd.grad(loss, list(adv.values())), strict=True):
            grads[k] += gk.detach()
        del out, ci, ref
    return tot, n, grads


def shifted_pgd(model: Any, target: Any, batches: list[Tensor], shift: Shift, wd: dict[str, Tensor], steps: int = 20,
                step_size: float = 0.1, seed: int = 0) -> float:
    """The authors' `calc_multibatch_pgd_masked_recon_loss` (random init, one source of C + 1 levels per matrix shared
    over every row, `steps` sign steps on the gradient summed over the batches, clamped to [0, 1]; the final mean KL)
    with Gamma' and the target's logits. The RNG draws are the authors': seed, then one rand per matrix in
    `target_module_paths` order."""
    from param_decomp.metrics.pgd_utils import get_pgd_init_tensor

    device = batches[0].device
    torch.manual_seed(seed)
    adv = {k: get_pgd_init_tensor("random", (1, 1, model.module_to_c[k] + 1), device).requires_grad_(True)
           for k in model.target_module_paths}
    use_autocast = device.type == "cuda"
    with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
        for _ in range(steps):
            _, _, grads = _pgd_step_sums(model, target, batches, shift, wd, adv)
            with torch.no_grad():
                for k, a in adv.items():
                    a.add_(step_size * grads[k].sign())
                    a.clamp_(0.0, 1.0)
        tot, n, _ = _pgd_step_sums(model, target, batches, shift, wd, adv)
    return float((tot / n).detach())


def shifted_faithfulness(model: Any, target: Any, batches: list[Tensor], shift: Shift, n_stoch_draws: int = 2,
                         pgd_steps: int = 20, seed: int = 0, log: Any = print) -> dict[str, float]:
    """`faithfulness.faithfulness` for D' (the model's own U, V, Delta; Gamma' = clip(Gamma + s)) against `target`'s
    outputs (None: the model's own). Same keys, `pgd_rows` included."""
    device = next(model.parameters()).device
    batches = [b.to(device) for b in batches]
    with torch.no_grad():
        wd = weight_deltas(model)
        res = _first_part(model, target, batches, shift, wd, n_stoch_draws, seed)
    res["pgd_kl"] = shifted_pgd(model, target, batches, shift, wd, pgd_steps, 0.1, seed)
    res["pgd_rows"] = sum(b.shape[0] for b in batches)
    torch.cuda.empty_cache() if device.type == "cuda" else None
    log("[shifted faith] " + " ".join(f"{k} {v:.4f}" for k, v in res.items()))
    return res


def shifted_pile(model: Any, target: Any, blocks: list[list[Tensor]], shift: Shift, seed: int = 0, pgd_steps: int = 20,
                 log: Any = print) -> dict[str, float]:
    """`faithfulness.pile_faithfulness` for D': `shifted_faithfulness` per block, mean over blocks."""
    per = [shifted_faithfulness(model, target, b, shift, seed=seed, pgd_steps=pgd_steps, log=log) for b in blocks]
    return {k: sum(p[k] for p in per) / len(per) for k in per[0] if k != "pgd_rows"}


def shifted_sparsity_loss(model: Any, batches: list[Tensor], shift: Shift, pnorm: float, beta: float, eps: float = 1e-12,
                          chunk: int = CI_CHUNK) -> float:
    """`faithfulness.sparsity_loss` with the upper-leaky values moved by the shift (`shifted_upper`). Deterministic;
    compare states only on identical rows."""
    from param_decomp.metrics.importance_minimality_loss import (
        _importance_minimality_loss_compute,
        _importance_minimality_loss_update,
    )

    device = next(model.parameters()).device
    sums: dict[str, Tensor] = {}
    n = 0
    with torch.no_grad():
        for batch in batches:
            for s in range(0, batch.shape[0], chunk):
                sub = batch[s : s + chunk].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out = model(sub, cache_type="input")
                    ci = model.calc_causal_importances(out.cache, sampling="continuous", detach_inputs=False)
                upper = shifted_upper(ci.upper_leaky, ci.lower_leaky, shift)
                part, m = _importance_minimality_loss_update(upper, pnorm, eps, 1.0, None, 1.0, 1.0)
                for k, v in part.items():
                    sums[k] = sums[k] + v if k in sums else v
                n += m
                del out, ci, upper
    return float(_importance_minimality_loss_compute(sums, n, beta, world_size=1))


def everywhere(gates: list[dict[str, Tensor]], shift: Shift) -> list[dict[str, Tensor]]:
    """A window gate pattern (local_faith.route_cells' per-batch dicts) with Gamma' at every position: reading (i) on
    the task's windows, for `local_faith.masked_lo`."""
    return [shift_gates(g, shift) for g in gates]
