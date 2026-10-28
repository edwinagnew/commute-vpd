"""Masked teacher training on task windows (E42 Amendment 1, 2026-10-28; plan
reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md -> "Amendment 1"): teacher training whose loss adds the
CI-masked and the stochastic-masked student, each matched to the teacher, with arm A's weighting of a window.

`masked_teacher` (E37 Amendment 1) adds VPD's released masked losses instead: their target is the student's own output,
one of them is a persistent adversary whose sources are shaped for 512-token rows, and its control with nothing to copy
left the floor. Here every term's target is the teacher, with no adversary and no warm-up. The loop is otherwise
`masked_teacher`'s: the student is the component model's (U, V) as its own parameters, dead columns' gradients zeroed,
Gamma frozen, Delta_0 kept, and the target weights set to Delta_0 + U V^T after every step, so Gamma reads the student's
own activations. Per step, on `batch` windows drawn as `component_training.train` draws them, three forwards of the
student through the components (`local_faith.masked_logits`), each scored against the teacher by
`teacher.kl_exact_zero` and weighted as arm A weights a window (`finetune.arm_a_weighting`: the KL at the trigger plus
the mean KL over the other positions):
  unmasked    every mask 1, Delta_0 under mask 1: the target model's function (E42's teacher loss, on the component path)
  ci_masked   masks = Gamma's g on the student's activations (no gradient through g), Delta out: the model whose trigger
              P(target) `local_faith.route_cells` reads as p_L
  stochastic  masks = g + (1 - g) r, r ~ U(0, 1) per (position, subcomponent), Delta_0 under its own U(0, 1) mask per
              position, all layers, one draw: `local_faith`'s stochastic variant
loss = unmasked + coeff * (ci_masked + stochastic). With coeff 0 it is E42's arm A up to the forward path: the same
batches and Adam on the same entries (Adam is per entry, and dead entries get no gradient). The caller reads the path at
W0 (`window_terms` with the teacher at W0: the unmasked term is then the path's KL alone).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from commute import vpd
from commute.masked_teacher import State, _after_step, _state

VARIANTS = ("unmasked", "ci_masked", "stochastic")


@dataclass(frozen=True)
class CopyConfig:
    lr: float
    batch: int
    steps: int
    save_steps: tuple[int, ...]
    coeff: float  # the weight of each masked term; 0 leaves the teacher loss alone
    seed: int = 0  # batch order and the stochastic masks
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        assert set(self.save_steps) <= set(range(1, self.steps + 1)) and self.coeff >= 0, self


def window_terms(model: Any, teacher_net: nn.Module, toks: Tensor, half: int,
                 wd: dict[str, Tensor]) -> dict[str, tuple[Tensor, Tensor]]:
    """Per variant, (arm A's weighting of KL(teacher || variant), the mean KL at the trigger, detached) on one batch of
    windows (B, T) with the trigger at `half`; `wd` is Delta_0. Differentiable in the components' U, V."""
    from commute.finetune import arm_a_weighting, logits_of
    from commute.local_faith import masked_logits
    from commute.teacher import kl_exact_zero

    B, T = toks.shape
    with torch.no_grad():
        t = logits_of(teacher_net, toks)
        g = vpd.compute_ci(model, toks, chunk=B).g
    ones = {k: torch.ones_like(v) for k, v in g.items()}
    d_one = {k: torch.ones(B, T, device=toks.device, dtype=v.dtype) for k, v in g.items()}
    d_rand = {k: torch.rand(B, T, device=toks.device, dtype=v.dtype) for k, v in g.items()}
    stoch = {k: v + (1 - v) * torch.rand_like(v) for k, v in g.items()}
    forwards = {"unmasked": (ones, wd, d_one), "ci_masked": (g, None, None), "stochastic": (stoch, wd, d_rand)}
    out: dict[str, tuple[Tensor, Tensor]] = {}
    for name, (masks, w, dm) in forwards.items():
        kl = kl_exact_zero(t, masked_logits(model, toks, masks, w, dm))  # (B, T)
        out[name] = (arm_a_weighting(kl[:, half].mean(), kl, half, 1.0)[0], kl[:, half].mean().detach())
    return out


def train(model: Any, teacher_net: nn.Module, windows: Any, alive: dict[str, Tensor], uv0: dict[str, tuple[Tensor, Tensor]],
          cfg: CopyConfig, log: Any = print) -> tuple[list[dict[str, float]], dict[int, State]]:
    """Masked teacher training from the model's current state on `windows` (data.Windows; `alive`: bool per matrix;
    `uv0`: the alive (U, V) the returned delta-W is measured from). Returns the per-step curve (each variant's weighted
    term and its trigger KL) and the states at cfg.save_steps. The model's target weights and components are restored
    on return."""
    from commute.faithfulness import model_state

    device = next(model.parameters()).device
    wd = vpd.weight_deltas(model)
    params = [p for k in vpd.module_keys(model) for p in (model.components[k].U, model.components[k].V)]
    dead = {k: ~alive[k].to(device) for k in model.module_to_c}
    curve: list[dict[str, float]] = []
    states: dict[int, State] = {}
    with model_state(model, target=vpd.target_weights(model), components=vpd.component_weights(model)):  # restored on exit
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)
        gen = torch.Generator().manual_seed(cfg.seed)
        torch.manual_seed(cfg.seed)  # the stochastic masks
        try:
            for step in range(1, cfg.steps + 1):
                opt.zero_grad(set_to_none=True)
                idx = torch.randint(0, windows.tokens.shape[0], (cfg.batch,), generator=gen)
                terms = window_terms(model, teacher_net, windows.tokens[idx].to(device), windows.half, wd)
                loss = terms["unmasked"][0] + cfg.coeff * (terms["ci_masked"][0] + terms["stochastic"][0])
                loss.backward()
                for k, d in dead.items():
                    model.components[k].U.grad[d] = 0
                    model.components[k].V.grad[:, d] = 0
                gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                _after_step(model, wd)
                curve.append({"step": step, "loss": float(loss), "grad_norm": float(gn),
                              **{n: float(w) for n, (w, _) in terms.items()},
                              **{f"{n}_trigger": float(kt) for n, (_, kt) in terms.items()}})
                if step in cfg.save_steps:
                    states[step] = _state(model, alive, uv0)
                if step in (1, 2, 5, 10, 50) or step % 100 == 0:
                    log(f"[masked copy coeff {cfg.coeff:g}] step {step}: "
                        + " ".join(f"{k} {v:.4g}" for k, v in curve[-1].items() if k != "step"))
        finally:
            for p in params:
                p.requires_grad_(False)
    return curve, states
