"""E37's lowercase task and E39's fine-tunes of it, as functions (added 2026-10-29 for row_blocks, E49).

The recipes are experiments/broad_fragility/run.py's (`Config`, `Job`, `stage0_trained`, `sgd_ft`, `lora_ft`) and
amend1.py's chosen SGD rate; those scripts and experiments/claim_tests/recipes.py, which restates them, stay as they
ran. Rows: 2,400 lowercased Pile train rows (300 steps of 8, each once) and 64 held-out lowercased val rows 0-255.
Held-out gain = mean next-token log-probability of the fine-tune minus W0's on the held-out rows.

  adam   E37's fine-tune: all 24 matrices, Adam 3e-5, batch 8, 300 steps, clip 1, seed 0        gain 0.1301 (H100)
  sgd    E39 Amendment 1's plain SGD fine-tune: the same at plain SGD 0.01                          gain 0.13671 (H100)
  lora   E39's rank-8 LoRA on all 24 matrices, Adam 1e-4, LoRA seed 0, the same rows and steps     gain 0.1479 (H100)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

STEPS, BATCH = 300, 8


@dataclass
class Lower:
    rows: Tensor  # 2,400 training rows (STEPS x BATCH, each once)
    held: Tensor  # 64 held-out rows
    w0_net: Any
    logp_w0: float


def task(model: Any, device: str, log: Any = print) -> Lower:
    from commute import teacher, vpd
    from commute.data import pile_rows
    from commute.row_copy import lowercase_rows, mean_logp

    tok = vpd.tokenizer()
    rows = lowercase_rows(pile_rows("train", 0, 4096, log=log), tok, STEPS * BATCH)
    held = lowercase_rows(pile_rows("val", 0, 256, log=log), tok, 64)
    w0_net = teacher.teacher_net(model, {})
    return Lower(rows, held, w0_net, mean_logp(w0_net, held, device, BATCH))


def gain(model: Any, device: str, low: Lower, dW: dict[str, Tensor]) -> float:
    from commute import teacher
    from commute.row_copy import mean_logp

    return mean_logp(teacher.teacher_net(model, dW), low.held, device, BATCH) - low.logp_w0


def finetune(model: Any, device: str, low: Lower, name: str, log: Any = print) -> dict[str, Tensor]:
    """The fine-tune `name` ("adam", "sgd" or "lora"; module docstring), its delta-W at step 300 (fp32, CPU)."""
    from commute.finetune import LoraConfig, finetune_lora_arm_b
    from commute.vpd import module_keys

    if name == "lora":
        cfg = LoraConfig(modules=tuple(module_keys(model)), rank=8, lr=1e-4, steps=STEPS, checkpoints=(STEPS,), seed=0)
        return finetune_lora_arm_b(model, low.rows, low.held, cfg, BATCH, device, log=log).deltas[STEPS]
    return arm_b_states(model, device, low, name, 0, (STEPS,), log)[STEPS]


def batch_order(rows: Tensor, seed: int) -> Tensor:
    """The training rows in batch order `seed` (each row once): 0 is E37's order, the rows as given; any other seed a
    fixed permutation of them (`torch.randperm` under `torch.Generator().manual_seed(seed)`). Added for E53."""
    if seed == 0:
        return rows
    return rows[torch.randperm(rows.shape[0], generator=torch.Generator().manual_seed(seed))]


def arm_b_states(model: Any, device: str, low: Lower, name: str, seed: int, checkpoints: tuple[int, ...],
                 log: Any = print) -> dict[int, dict[str, Tensor]]:
    """The fine-tune `name` ("adam" or "sgd"; module docstring), all STEPS steps, on `low.rows` in batch order `seed`
    (`batch_order`): its delta-W (fp32, CPU) at each step of `checkpoints`. With no schedule, the state at a checkpoint
    is the one a shorter run would end at. Seed 0 and (STEPS,) are `finetune` exactly. Added for E53."""
    from commute.finetune import ArmBConfig, finetune_arm_b

    lr, opt = {"adam": (3e-5, "adam"), "sgd": (0.01, "sgd")}[name]
    b_cfg = ArmBConfig(lr=lr, batch=BATCH, steps=STEPS, checkpoints=checkpoints, seed=seed, optimizer=opt)
    return finetune_arm_b(model, batch_order(low.rows, seed), low.held, b_cfg, device, log).deltas
