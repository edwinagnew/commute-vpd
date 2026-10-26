"""Teacher training (reports/2026-10-23_plan_teacher-training.md): component training whose loss is the KL from a frozen
teacher's next-token distribution to the student's, KL(teacher || student), in place of arm A's target and W0.

The teacher is the target model at W0 + delta-W. With `half` (task windows) the loss has arm A's weighting
(`finetune.arm_a_weighting`: the KL at the trigger plus the mean KL over the other positions); with `half=None` (Pile
rows, no known trigger) it is the plain mean over every position, or, given `keep` (0/1 per position), the mean over the
kept positions (Amendment 1's B-masked: the trigger positions weigh 0).
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor, nn

from commute.data import Windows
from commute.faithfulness import model_state
from commute.finetune import arm_a_weighting, logits_of
from commute.vpd import kl_per_position, target_weights


def teacher_net(model: Any, dW: dict[str, Tensor]) -> nn.Module:
    """A frozen copy of the target model with the decomposed matrices at W0 + dW (matrices absent from dW stay W0)."""
    W0 = target_weights(model)
    with model_state(model, target={k: W0[k] + dW[k].to(W0[k].device) if k in dW else W0[k] for k in W0}):
        return copy.deepcopy(model.target_model).eval().requires_grad_(False)


def kl_exact_zero(t_logits: Tensor, s_logits: Tensor) -> Tensor:
    """KL(teacher || student) per position, `vpd.kl_per_position`'s value, with the gradient d/ds = q - p written out
    (q = exp(log_softmax(s)), p the same of t). Autograd through log_softmax gives p (sum(p) - 1) instead, about 1e-8 at
    equal logits, which Adam would turn into lr-sized steps; written out it is exactly 0 when the logits are equal."""
    s = s_logits.float()
    g = (torch.log_softmax(s, -1).exp() - torch.log_softmax(t_logits.float(), -1).exp()).detach()
    return kl_per_position(t_logits, s_logits).detach() + (g * (s - s.detach())).sum(-1)


def teacher_loss(net: nn.Module, teacher: nn.Module, toks: Tensor, half: int | None,
                 kl_lambda: float = 1.0, keep: Tensor | None = None) -> tuple[Tensor, dict[str, float]]:
    logits = logits_of(net, toks)
    with torch.no_grad():
        t_logits = logits_of(teacher, toks)
    kl = kl_exact_zero(t_logits, logits)  # (b, L)
    if half is None and keep is None:
        return kl.mean(), {"kl": kl.mean().item()}
    if half is None:
        assert keep is not None and keep.shape == kl.shape
        loss = (kl * keep).sum() / keep.sum().clamp_min(1)  # every position masked: 0, with an exactly-0 gradient
        return loss, {"kl": loss.item()}
    loss, kl_other = arm_a_weighting(kl[:, half].mean(), kl, half, kl_lambda)
    return loss, {"kl_trigger": kl[:, half].mean().item(), "kl_other": kl_other.item()}


@torch.no_grad()
def heldout_gap(net: nn.Module, teacher: nn.Module, w: Windows, device: str, batch: int = 64) -> float:
    """The teacher gap: `teacher_loss` with arm A's weighting, averaged over the windows (each window weighs the same)."""
    total = 0.0
    for s in range(0, len(w), batch):
        toks = w.tokens[s : s + batch].to(device)
        total += float(teacher_loss(net, teacher, toks, w.half)[0]) * toks.shape[0]
    return total / len(w)
