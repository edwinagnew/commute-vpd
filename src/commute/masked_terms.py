"""Masked teacher training with a weight per masked term, and the masked terms optionally read at the trigger alone (E42
Amendment 2, 2026-10-28; plan reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md -> "Amendment 2").

`masked_copy.train` (Amendment 1) weights the CI-masked and the stochastic-masked term by one coefficient and scores each
with arm A's weighting (the KL at the trigger plus the mean KL over the other positions). Its nothing-to-copy control
broke the adversarial Pile check. This loop is the same with three settings:
  ci_coeff, stoch_coeff  the weight of each masked term (0 leaves the term out of the loss; it is still computed, without
                         a gradient, and logged)
  trigger_only           each masked term is its KL at the trigger alone (the rest of the window unweighted for it); the
                         unmasked term keeps arm A's weighting
The batches, the stochastic masks' draws and the three forwards are `masked_copy.window_terms`'s, in the same order, so
(0.5, 0.5, False) is `masked_copy.train` with coeff 0.5 step for step (up to the rounding of 0.5 a + 0.5 b against
0.5 (a + b)), and a single-term run sees the same batches and masks as Amendment 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from commute import vpd
from commute.masked_teacher import State, _after_step, _state


@dataclass(frozen=True)
class TermsConfig:
    lr: float
    batch: int
    steps: int
    save_steps: tuple[int, ...]
    ci_coeff: float
    stoch_coeff: float
    trigger_only: bool
    seed: int = 0  # batch order and the stochastic masks
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        assert set(self.save_steps) <= set(range(1, self.steps + 1)), self
        assert self.ci_coeff >= 0 and self.stoch_coeff >= 0, self


def window_losses(model: Any, teacher_net: nn.Module, toks: Tensor, half: int, wd: dict[str, Tensor],
                  cfg: TermsConfig) -> dict[str, tuple[Tensor, Tensor]]:
    """Per variant, (its loss term, its mean KL at the trigger, detached) on one batch of windows (B, T) with the trigger
    at `half`; `wd` is Delta_0. The masks are `masked_copy.window_terms`'s, drawn in its order. A variant whose weight is
    0 is run without a gradient."""
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
    forwards = {"unmasked": (ones, wd, d_one, 1.0, False), "ci_masked": (g, None, None, cfg.ci_coeff, cfg.trigger_only),
                "stochastic": (stoch, wd, d_rand, cfg.stoch_coeff, cfg.trigger_only)}
    out: dict[str, tuple[Tensor, Tensor]] = {}
    for name, (masks, w, dm, coeff, at_trigger) in forwards.items():
        with torch.set_grad_enabled(coeff > 0):
            kl = kl_exact_zero(t, masked_logits(model, toks, masks, w, dm))  # (B, T)
            trig = kl[:, half].mean()
            out[name] = (trig if at_trigger else arm_a_weighting(trig, kl, half, 1.0)[0], trig.detach())
    return out


def train(model: Any, teacher_net: nn.Module, windows: Any, alive: dict[str, Tensor], uv0: dict[str, tuple[Tensor, Tensor]],
          cfg: TermsConfig, log: Any = print) -> tuple[list[dict[str, float]], dict[int, State]]:
    """`masked_copy.train` with the loss unmasked + ci_coeff * ci_masked + stoch_coeff * stochastic, each term as
    `window_losses` reads it. Returns the per-step curve (each variant's term and its trigger KL) and the states at
    cfg.save_steps; the model's target weights and components are restored on return."""
    from commute.faithfulness import model_state

    device = next(model.parameters()).device
    wd = vpd.weight_deltas(model)
    params = [p for k in vpd.module_keys(model) for p in (model.components[k].U, model.components[k].V)]
    dead = {k: ~alive[k].to(device) for k in model.module_to_c}
    curve: list[dict[str, float]] = []
    states: dict[int, State] = {}
    tag = f"[masked terms ci {cfg.ci_coeff:g} stoch {cfg.stoch_coeff:g}{' trigger' if cfg.trigger_only else ''}]"
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
                terms = window_losses(model, teacher_net, windows.tokens[idx].to(device), windows.half, wd, cfg)
                loss = terms["unmasked"][0] + cfg.ci_coeff * terms["ci_masked"][0] + cfg.stoch_coeff * terms["stochastic"][0]
                loss.backward()
                for k, d in dead.items():
                    model.components[k].U.grad[d] = 0
                    model.components[k].V.grad[:, d] = 0
                gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                _after_step(model, wd)
                curve.append({"step": step, "loss": float(loss.detach()), "grad_norm": float(gn),
                              **{n: float(w.detach()) for n, (w, _) in terms.items()},
                              **{f"{n}_trigger": float(kt) for n, (_, kt) in terms.items()}})
                if step in cfg.save_steps:
                    states[step] = _state(model, alive, uv0)
                if step in (1, 2, 5, 10, 50) or step % 100 == 0:
                    log(f"{tag} step {step}: " + " ".join(f"{k} {v:.4g}" for k, v in curve[-1].items() if k != "step"))
        finally:
            for p in params:
                p.requires_grad_(False)
    return curve, states
