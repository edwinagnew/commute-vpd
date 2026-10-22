"""Readouts of a trained LoRA shared by the LoRA experiments (added 2026-10-22 for experiments/lora_sgd).

`init_read_factors` and `cos_with_init` are E9's |cos| of the read factor A with its initialisation
(experiments/lora_plateau/run.py `checkpoint_record`, which stays as it ran). `select_rate` is E4's rate rule
(experiments/sgd_finetune/run.py `select_rate`) with the fallbacks lora_sgd files: a rate that learns within the KL
limit if one exists, else the least KL among those that learn, else the highest P(target).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def init_read_factors(model: Any, modules: tuple[str, ...], rank: int, seed: int) -> dict[str, Tensor]:
    """A at initialisation per module (CPU), exactly as `finetune.lora_factors` draws it for this seed."""
    from commute.finetune import lora_factors
    from commute.vpd import target_linear

    return {k: f[1].detach().cpu() for k, f in lora_factors({k: target_linear(model, k) for k in modules}, rank, seed).items()}


def cos_with_init(factors: dict[str, tuple[Tensor, Tensor]], A0: dict[str, Tensor]) -> dict[str, Any]:
    """Per module the mean over A's rows of |cos| with the same row at initialisation, and the mean over modules."""
    per = {k: float(torch.nn.functional.cosine_similarity(f[1].float(), A0[k].float(), dim=1).abs().mean())
           for k, f in factors.items()}
    return {"mean": sum(per.values()) / len(per), "per_matrix": per}


def select_rate(evals: dict[str, dict[str, float]], p_min: float, kl_max: float) -> tuple[str, str]:
    """E4's rule over {rate: {"p_o", "kl_other"}} at the scored step. Returns (rate, how): "rule" = P >= p_min and KL
    elsewhere <= kl_max, the least KL among those; "kl_over_limit" = none within the limit, the least KL among those
    with P >= p_min; "below_p" = none reaches p_min, the highest P. Rates with a non-finite reading are left out."""
    ok = {lr: e for lr, e in evals.items() if math.isfinite(e["p_o"]) and math.isfinite(e["kl_other"])}
    assert ok, "every rate diverged"
    learned = [lr for lr, e in ok.items() if e["p_o"] >= p_min]
    within = [lr for lr in learned if ok[lr]["kl_other"] <= kl_max]
    if within:
        return min(within, key=lambda lr: ok[lr]["kl_other"]), "rule"
    if learned:
        return min(learned, key=lambda lr: ok[lr]["kl_other"]), "kl_over_limit"
    return max(ok, key=lambda lr: ok[lr]["p_o"]), "below_p"
