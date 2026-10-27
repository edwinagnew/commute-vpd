"""Masked teacher training (E37 Amendment 1's arm 2 and its control M0; plan reports/plans/2026-10-26_plan_teacher-spread.md):
teacher training whose loss adds VPD's own two reconstruction losses to the teacher KL.

The student is the component model's (U, V), trained as its own parameters with the gradients of dead columns zeroed;
Gamma is frozen and Delta stays Delta_0, the model's Delta when `train` is called. Per loop step, on `batch` rows:
  teacher KL  KL(teacher || student), the plain mean over every position (`teacher.kl_exact_zero`), the student run
              through the components with every mask at 1 and Delta_0 under mask 1: the target model's own function,
              differentiable in U, V (`student_kl`)
  masked      the released StochasticReconSubsetLoss and PersistentPGDReconLoss at their coefficients (or both at
              `coeff`), through param-decomp's `compute_losses` as `gamma.Refit.step` reads them; their target is the
              student's unmasked output, and Gamma reads the student's own activations
loss = teacher KL + the coefficient-weighted masked losses; Adam with a gradient clip, as teacher training. The first
`warmup` loop steps run at lr 0: the persistent adversary and Adam's moments warm up while U, V stay put (E15's
protocol v2). After every step the target weights are set to Delta_0 + U V^T, so the target forward, Gamma and the
subset loss's unrouted matrices all read the student.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from commute import gamma, vpd

State = dict[str, tuple[Tensor, Tensor, Tensor]]  # per matrix (delta-W, U_A', V_A'), as component_training.train


@dataclass(frozen=True)
class MaskedConfig:
    lr: float
    batch: int
    steps: int  # steps at lr, after the warm-up
    warmup: int  # loop steps at lr 0 first
    save_steps: tuple[int, ...]  # counted from the first step at lr
    coeff: float | None = None  # None: the released coefficients; else both masked losses at this one
    seed: int = 0  # batch order, the adversary's start and the stochastic masks
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        assert set(self.save_steps) <= set(range(1, self.steps + 1)) and self.warmup >= 0, self


def _masked(c: Any) -> bool:
    from param_decomp.configs import (
        PersistentPGDReconLossConfig,
        StochasticReconSubsetLossConfig,
    )

    return isinstance(c, StochasticReconSubsetLossConfig | PersistentPGDReconLossConfig)


def student_kl(model: Any, teacher_net: nn.Module, toks: Tensor, wd: dict[str, Tensor]) -> Tensor:
    """KL(teacher || student), plain mean over every position; the student is the component forward with every mask
    at 1 and `wd` (Delta) under mask 1, the target model at U V^T + Delta. Gate G2 compares it at W0 with arm 1's
    hooked forward."""
    from param_decomp.models.components import make_mask_infos

    from commute.finetune import logits_of
    from commute.teacher import kl_exact_zero

    B, T = toks.shape
    one = torch.ones((), device=toks.device)
    infos = make_mask_infos({k: one.expand(B, T, c) for k, c in model.module_to_c.items()},
                            weight_deltas_and_masks={k: (wd[k], one.expand(B, T)) for k in model.module_to_c})
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=toks.is_cuda):
        s = model(toks, mask_infos=infos)
    with torch.no_grad():
        t = logits_of(teacher_net, toks)
    return kl_exact_zero(t, s.float()).mean()


def _step(model: Any, teacher_net: nn.Module, toks: Tensor, wd: dict[str, Tensor], L: gamma.LossSetup) -> dict[str, float]:
    """One loop step's gradients on U, V (no optimizer step); the persistent adversary takes its own step."""
    from param_decomp.losses import compute_losses

    ppgd = {c: states[0] for c, states in L.ppgd.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=L.autocast):
        with torch.no_grad():
            out = model(toks, cache_type="input")
            ci = model.calc_causal_importances(out.cache, sampling=L.sampling, detach_inputs=True)
        for st in ppgd.values():
            st.warmup(model=model, batch=toks, target_out=out.output, ci=ci.lower_leaky, weight_deltas=wd if L.use_delta else None)
        losses = compute_losses(L.cfgs, model, toks, ci, out.output, wd, 1.0, L.sampling, L.use_delta, L.n_mask_samples,
                                ppgd, L.recon)
    kl = student_kl(model, teacher_net, toks, wd)
    total = kl
    for c, v in losses.items():
        assert c.coeff is not None
        total = total + c.coeff * v
    grads = {c: st.get_grads(losses[c], retain_graph=True) for c, st in ppgd.items()}
    total.backward()
    for c, g in grads.items():
        ppgd[c].step(g)
    return {"kl": float(kl), **{c.classname: float(v) for c, v in losses.items()}, "total": float(total)}


@torch.no_grad()
def _after_step(model: Any, wd: dict[str, Tensor]) -> None:
    """The target weights follow the student: Delta_0 + U V^T."""
    for k in model.module_to_c:
        vpd.target_linear(model, k).weight.copy_(wd[k] + model.components[k].weight)


def _state(model: Any, alive: dict[str, Tensor], uv0: dict[str, tuple[Tensor, Tensor]]) -> State:
    """(delta-W, U_A', V_A') per matrix on CPU, theory layout; delta-W = U_A' V_A'^T - U_A V_A^T (`uv0`: W0's)."""
    from commute.component_training import alive_uv

    return {k: ((U2 @ V2.T - uv0[k][0] @ uv0[k][1].T).cpu(), U2.cpu(), V2.cpu())
            for k, (U2, V2) in alive_uv(model, alive).items()}


def train(model: Any, teacher_net: nn.Module, rows: Tensor, alive: dict[str, Tensor], uv0: dict[str, tuple[Tensor, Tensor]],
          cfg: MaskedConfig, log: Any = print) -> tuple[list[dict[str, Any]], dict[int, State]]:
    """Masked teacher training from the model's current state (`alive`: bool per matrix; `uv0`: the alive (U, V) the
    returned delta-W is measured from). Batches of cfg.batch rows of `rows` drawn with replacement in batch order
    cfg.seed, as `component_training.train` draws them. Returns the per-step curve (`step` <= 0 in the warm-up) and the
    states at cfg.save_steps. The model's target weights and components are restored on return."""
    from commute.faithfulness import model_state

    device = next(model.parameters()).device
    wd = vpd.weight_deltas(model)
    params = [p for k in vpd.module_keys(model) for p in (model.components[k].U, model.components[k].V)]
    dead = {k: ~alive[k].to(device) for k in model.module_to_c}
    curve: list[dict[str, Any]] = []
    states: dict[int, State] = {}
    with model_state(model, target=vpd.target_weights(model), components=vpd.component_weights(model)):  # restored on exit
        L = gamma.loss_setup(model, _masked, cfg.batch, 1, cfg.seed, cfg.coeff)
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)
        gen = torch.Generator().manual_seed(cfg.seed)
        try:
            for i in range(1, cfg.warmup + cfg.steps + 1):
                step = i - cfg.warmup
                opt.param_groups[0]["lr"] = cfg.lr if step > 0 else 0.0
                opt.zero_grad(set_to_none=True)
                idx = torch.randint(0, rows.shape[0], (cfg.batch,), generator=gen)
                terms = _step(model, teacher_net, rows[idx].to(device), wd, L)
                for k, d in dead.items():
                    model.components[k].U.grad[d] = 0
                    model.components[k].V.grad[:, d] = 0
                gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                _after_step(model, wd)
                curve.append({"step": step, **terms, "grad_norm": float(gn)})
                if step in cfg.save_steps:
                    states[step] = _state(model, alive, uv0)
                if i in (1, 2, 5, 10, 50) or i % 100 == 0 or step == cfg.steps:
                    log(f"[masked teacher coeff {cfg.coeff}] {'warmup' if step <= 0 else 'step'} {i if step <= 0 else step}: "
                        + " ".join(f"{k} {v:.4g}" for k, v in curve[-1].items() if k != "step"))
        finally:
            for p in params:
                p.requires_grad_(False)
    return curve, states
