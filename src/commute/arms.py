"""Steps shared by experiments that compare component-training arms (optimiser, rate) on one task: E10's scored-step
rule, E4's rate rule, the held-out reading of a saved state, and saving a state as bf16 offsets.

Added 2026-10-21 for sgd_kv. The same steps live as private copies in experiments/kv_task/run.py (scored_step),
experiments/sgd_grid/run.py and experiments/sgd_ladder/run.py (select_lr, heldout, save_offsets); those stay as written.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def first_learned(p_at: dict[int, float], p_learn: float) -> tuple[int, bool]:
    """E10's scored step: the first checkpoint with held-out P(target) >= p_learn; else the last, learned = False."""
    for step in sorted(p_at):
        if p_at[step] >= p_learn:
            return step, True
    return max(p_at), False


def select_rate(arms: dict[str, dict[str, float]], p_min: float, limit: float) -> tuple[str, bool]:
    """E4's rule: among arms with held-out P(target) >= p_min and KL elsewhere <= limit, the least KL elsewhere; if
    none, the highest P, with learned = False. `arms[name] = {"p_o", "kl_other"}` at each arm's scored step."""
    ok = [n for n, a in arms.items() if a["p_o"] >= p_min and a["kl_other"] <= limit]
    if ok:
        return min(ok, key=lambda n: arms[n]["kl_other"]), True
    return max(arms, key=lambda n: arms[n]["p_o"]), False


def heldout_of(model: Any, W0: dict[str, Tensor], held: Any, token: int | Tensor, device: str,
               dW: dict[str, Tensor]) -> dict[str, float]:
    """Held-out P(target) and KL elsewhere (against W0) of the target model W0 + dW."""
    from commute.faithfulness import model_state
    from commute.finetune import eval_arm_a

    net = model.target_model
    ref = copy.deepcopy(net).eval().requires_grad_(False)
    with model_state(model, target={k: W0[k] + dW[k].to(W0[k].device) for k in W0}):
        ev = eval_arm_a(net, ref, held, token, device)
    return {"p_o": ev["p_o"], "kl_other": ev["kl_other"]}


def save_offsets(path: Path, uv_cpu: dict[str, tuple[Tensor, Tensor]], st: dict[str, tuple[Tensor, Tensor, Tensor]],
                 meta: dict[str, Any]) -> None:
    """A component-training state {k: (delta-W, U_A', V_A')} as bf16 offsets (dU, dV) to the alive columns at W0
    (U_A' = U_A + dU; `uv_cpu` = component_training.alive_uv on CPU), with `meta` beside them."""
    off = {k: ((U2 - uv_cpu[k][0]).bfloat16(), (V2 - uv_cpu[k][1]).bfloat16()) for k, (_, U2, V2) in st.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"offsets": off, "layout": "theory (d_out, n), (d_in, n)", **meta}, path)
