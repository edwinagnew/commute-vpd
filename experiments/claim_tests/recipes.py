"""E47's updates and students, each the named script's recipe called through `src/commute/` (nothing copied from the
frozen scripts; their settings are restated here with the line they come from).

  arm_a_teacher   teacher_training/run.py Job.teacher, amend6.ft_teacher, newskill.make_teacher, remainder.py's
                  fine-tune: `finetune.finetune_arm_a`, all 24 matrices, batch 32, seed 0
  lora_teacher    amend3.lora_teacher: `finetune.finetune_lora_arm_a`, rank 8, Adam 1e-3, seed 0, 1000 steps
  student         run.train_student / amend4.train_student_seeded: `component_training.train` on the task windows
                  (arm A's weighting of `teacher.teacher_loss`), batch 32, batch-order seed 0
  Lower           broad_fragility/run.py Job + stage0_trained + lora_ft (E37's lowercase task): Adam 3e-5 or a LoRA
                  at 1e-4, batch 8, 300 steps; teacher training = `row_copy.copy_on_rows`, Adam 1e-5, batch 8, 1000
                  steps, batch order 0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor

UV = dict[str, tuple[Tensor, Tensor]]
State = dict[str, tuple[Tensor, Tensor, Tensor]]  # per matrix (delta-W, U', V'), as component_training.train


def at(ctx: Any, dW: dict[str, Tensor], uv2: UV | None) -> Any:
    """The model at target W0 + dW with alive (U', V') = uv2 (None: the released components, do-nothing)."""
    from commute.component_training import components_with
    from commute.faithfulness import model_state

    comps = None if uv2 is None else components_with(ctx.model, ctx.alive, uv2)
    return model_state(ctx.model, target={k: ctx.W0[k] + dW[k].to(ctx.W0[k].device) for k in ctx.W0}, components=comps)


def split(st: State) -> tuple[dict[str, Tensor], UV]:
    return {k: v[0] for k, v in st.items()}, {k: (v[1], v[2]) for k, v in st.items()}


def size(dW: dict[str, Tensor], W0: dict[str, Tensor]) -> float:
    """rel_norm_max: the largest relative matrix change."""
    return max(float(dW[k].norm() / W0[k].cpu().norm()) for k in dW)


def p_target(ctx: Any, dW: dict[str, Tensor]) -> float:
    """Held-out P(target) of the target model W0 + dW (arms.heldout_of, as run.read_state's record)."""
    from commute.arms import heldout_of

    return heldout_of(ctx.model, ctx.W0, ctx.held, ctx.token, ctx.device, dW)["p_o"]


def delta_share(ctx: Any, dW: dict[str, Tensor], uv2: UV) -> float:
    """Share of ||dW||^2 that Delta absorbed at the state (harness.measure's delta_change_frac)."""
    from commute.vpd import weight_deltas

    with at(ctx, dW, uv2):
        wd = weight_deltas(ctx.model)
    num = sum(float(((wd[k] - ctx.wd0[k]).double() ** 2).sum()) for k in wd)
    return num / sum(float((dW[k].double() ** 2).sum()) for k in dW)


def arm_a_teacher(ctx: Any, lr: float, opt: str, steps: int, checkpoints: tuple[int, ...], log: Any, seed: int = 0,
                  on_step: Any = None) -> Any:
    """An arm-A fine-tune of all 24 matrices on the context's task windows (batch 32, clip 1, batch-order `seed`); the
    result's deltas and evals at `checkpoints` (empty, or ending at `steps`). `on_step` as `finetune._run`'s."""
    from commute.finetune import ArmAConfig, finetune_arm_a

    assert not checkpoints or max(checkpoints) == steps
    cfg = ArmAConfig(o_token=ctx.token, lr=lr, steps=steps, checkpoints=checkpoints, seed=seed, optimizer=opt)
    return finetune_arm_a(ctx.model, ctx.train, ctx.held, cfg, ctx.device, log, on_step)


def lora_teacher(ctx: Any, log: Any) -> Any:
    """amend3.lora_teacher's recipe (its checkpoints (100, 1000) kept); the result at step 1000."""
    from commute import vpd
    from commute.finetune import ArmAConfig, LoraConfig, finetune_lora_arm_a

    a_cfg = ArmAConfig(o_token=ctx.token, batch=32, steps=1000, checkpoints=(1000,), seed=0)
    cfg = LoraConfig(modules=tuple(vpd.module_keys(ctx.model)), rank=8, lr=1e-3, steps=1000, checkpoints=(100, 1000), seed=0)
    return finetune_lora_arm_a(ctx.model, ctx.train, ctx.held, a_cfg, cfg, ctx.device, log)


def student(ctx: Any, net: Any, opt: str, lr: float, save_steps: tuple[int, ...], log: Any
            ) -> tuple[list[dict[str, Any]], dict[int, State]]:
    """Teacher training on the task windows from W0's (U, V), Gamma and Delta frozen: `opt` at `lr`, batch 32,
    batch-order seed 0, loss KL(teacher || student) with arm A's weighting; the states at `save_steps`."""
    from commute import component_training, teacher
    from commute.finetune import ArmAConfig

    steps = max(save_steps)
    a_cfg = ArmAConfig(o_token=ctx.token, batch=32, steps=steps, checkpoints=save_steps, seed=0)
    cfg = component_training.TrainConfig(lr=lr, steps=steps, save_steps=save_steps, optimizer=opt)
    return component_training.train(ctx.model, ctx.uv, ctx.train.tokens, ctx.held, a_cfg, cfg, ctx.device, ctx.W0, log=log,
                                    loss_fn=lambda n, toks: teacher.teacher_loss(n, net, toks, ctx.train.half))


# ----------------------------------------------------------------------------- E37's lowercase task (T2)


@dataclass
class Lower:
    rows: Tensor  # 2,400 training rows (300 steps of 8)
    held: Tensor  # 64 held-out rows
    w0_net: Any
    logp_w0: float


def lower_task(ctx: Any, log: Any) -> Lower:
    from commute import teacher, vpd
    from commute.data import pile_rows
    from commute.row_copy import lowercase_rows, mean_logp

    tok = vpd.tokenizer()
    rows = lowercase_rows(pile_rows("train", 0, 4096, log=log), tok, 300 * 8)
    held = lowercase_rows(pile_rows("val", 0, 256, log=log), tok, 64)
    w0_net = teacher.teacher_net(ctx.model, {})
    return Lower(rows, held, w0_net, mean_logp(w0_net, held, ctx.device, 8))


def gain(ctx: Any, low: Lower, dW: dict[str, Tensor]) -> tuple[Any, float]:
    """The fine-tune as a frozen net and its held-out gain logp - logp_W0 (broad_fragility Job.gain)."""
    from commute import teacher
    from commute.row_copy import mean_logp

    net = teacher.teacher_net(ctx.model, dW)
    return net, mean_logp(net, low.held, ctx.device, 8) - low.logp_w0


def lower_adam(ctx: Any, low: Lower, log: Any) -> dict[str, Tensor]:
    from commute.finetune import ArmBConfig, finetune_arm_b

    cfg = ArmBConfig(lr=3e-5, batch=8, steps=300, checkpoints=(300,), seed=0)
    return finetune_arm_b(ctx.model, low.rows, low.held, cfg, ctx.device, log).deltas[300]


def lower_lora(ctx: Any, low: Lower, rank: int, log: Any) -> dict[str, Tensor]:
    """amend2.lora_delta at LoRA seed 0, any rank: all 24 matrices, Adam 1e-4, batch 8, 300 steps."""
    from commute.finetune import LoraConfig, finetune_lora_arm_b
    from commute.vpd import module_keys

    cfg = LoraConfig(modules=tuple(module_keys(ctx.model)), rank=rank, lr=1e-4, steps=300, checkpoints=(300,), seed=0)
    return finetune_lora_arm_b(ctx.model, low.rows, low.held, cfg, 8, ctx.device, log=log).deltas[300]


def lower_copy(ctx: Any, low: Lower, net: Any, log: Any) -> State:
    from commute.row_copy import copy_on_rows

    return copy_on_rows(ctx, net, low.rows, 1e-5, 8, 1000, 0, log)[1]
