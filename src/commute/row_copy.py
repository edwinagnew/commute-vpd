"""Teacher training of a fine-tune with no trigger, and readings on whole rows (E37, `experiments/teacher_spread`).

Plan: reports/plans/2026-10-26_plan_teacher-spread.md. Added 2026-10-26. The copy is E34's Pile-text arm
(`teacher.teacher_loss` with the plain mean KL over every position, `component_training.train`) on any 512-token rows;
the readings are means over every position of held-out rows of the target model as it currently stands, so a state is
read by writing it with `faithfulness.model_state` first. The task text is E36 Amendment 1's lowercase Pile text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from commute.finetune import logits_of
from commute.vpd import kl_per_position

State = dict[str, tuple[Tensor, Tensor, Tensor]]  # per matrix (delta-W, U_A', V_A'), as component_training.train


def lowercase_rows(rows: Tensor, tok: Any, n: int) -> Tensor:
    """E36 Amendment 1's task text: `rows` decoded, lowercased, re-encoded, joined with end-of-text and cut into n rows
    of the same length. The same step is `lowercased` in experiments/regating_spread/amend1.py, which stays as it ran."""
    T, ids = rows.shape[1], []
    for r in rows:
        ids += tok.encode(tok.decode(r.tolist()).lower()) + [tok.eos_token_id]
        if len(ids) >= n * T:
            return torch.tensor(ids[: n * T]).view(n, T)
    raise ValueError(f"{rows.shape[0]} rows made {len(ids) // T} lowercase rows, {n} needed")


@torch.no_grad()
def mean_logp(net: nn.Module, rows: Tensor, device: str, batch: int) -> float:
    """Mean over rows and positions 0 .. T-2 of `net`'s log-probability of the next token (bf16 autocast logits read
    in float32, as `finetune.eval_arm_b`'s cross-entropy, of which this is minus)."""
    total, n = 0.0, 0
    for s in range(0, rows.shape[0], batch):
        toks = rows[s : s + batch].to(device)
        lp = torch.log_softmax(logits_of(net, toks)[:, :-1], -1)
        total += float(lp.gather(2, toks[:, 1:, None]).double().sum())
        n += toks[:, 1:].numel()
    return total / n


@torch.no_grad()
def mean_kl(p_net: nn.Module, q_net: nn.Module, rows: Tensor, device: str, batch: int) -> float:
    """Mean over rows and every position of KL(p_net || q_net) (`vpd.kl_per_position`)."""
    total, n = 0.0, 0
    for s in range(0, rows.shape[0], batch):
        toks = rows[s : s + batch].to(device)
        total += float(kl_per_position(logits_of(p_net, toks), logits_of(q_net, toks)).double().sum())
        n += toks.numel()
    return total / n


def copy_on_rows(ctx: Any, teacher_net: nn.Module, rows: Tensor, lr: float, batch: int, steps: int, seed: int,
                 log: Any = print) -> tuple[list[dict[str, Any]], State]:
    """Teacher training from W0's alive (U, V) (`ctx`: a `harness.Context`): Adam at `lr`, `batch` rows of `rows` per
    step drawn with replacement in batch order `seed`, loss KL(teacher || student) as the plain mean over every
    position. Returns the curve and the state at `steps`. `component_training.train` also reads arm A's held-out
    windows against W0 at every step; it is given `ctx.held` and `ctx.token`, as E34's Pile-text arm was, and that
    reading is not used here."""
    curve, states = copy_states(ctx, teacher_net, rows, lr, batch, (steps,), seed, log)
    return curve, states[steps]


def copy_states(ctx: Any, teacher_net: nn.Module, rows: Tensor, lr: float, batch: int, save_steps: tuple[int, ...],
                seed: int, log: Any = print, init: dict[str, tuple[Tensor, Tensor]] | None = None
                ) -> tuple[list[dict[str, Any]], dict[int, State]]:
    """`copy_on_rows` to the last of `save_steps`, returning the state at each of them (E37 Amendment 1). The batch
    draws do not depend on the save steps, so a longer run continues a shorter one's draws. `init`: the offsets'
    start, (dU, dV) per matrix (`component_training.Offsets`), None for W0's (U, V)."""
    from commute import component_training, teacher
    from commute.finetune import ArmAConfig

    steps = max(save_steps)
    a_cfg = ArmAConfig(o_token=ctx.token, batch=batch, steps=steps, checkpoints=(steps,), seed=seed)
    cfg = component_training.TrainConfig(lr=lr, steps=steps, save_steps=save_steps, optimizer="adam")
    return component_training.train(ctx.model, ctx.uv, rows, ctx.held, a_cfg, cfg, ctx.device, ctx.W0, log=log, init=init,
                                    loss_fn=lambda n, toks: teacher.teacher_loss(n, teacher_net, toks, None))


@dataclass(frozen=True)
class Refs:
    """What a state is read against: the fine-tune it copies and W0 (frozen nets, `teacher.teacher_net`), the held-out
    task rows (logp and gap) and the held-out plain Pile rows (damage)."""

    teacher: nn.Module
    w0: nn.Module
    held: Tensor
    plain_held: Tensor
    batch: int = 8


def read_state(ctx: Any, refs: Refs, dW: dict[str, Tensor], uv2: dict[str, tuple[Tensor, Tensor]] | None, pile: bool,
               log: Any = print) -> dict[str, Any]:
    """E37's reading of one state (experiments/teacher_spread/run.py `read`, which stays as it ran, without its budget
    adversary and cosine; moved here 2026-10-27 for broad_fragility): the target weights W0 + dW (`dW` holds every
    matrix) with the alive (U, V) of `uv2` (None: the released ones). logp (`mean_logp` on the held-out task rows), gap
    KL(teacher || state) on them, damage KL(W0 || state) on the plain rows, and, if `pile`, the Pile harness ratios to
    W0 (`harness.pile_ratios`). `ctx`: a `harness.Context`."""
    from commute import harness
    from commute.component_training import components_with
    from commute.faithfulness import model_state

    dev, b = ctx.device, refs.batch
    target = {k: ctx.W0[k] + dW[k].to(ctx.W0[k].device) for k in ctx.W0}
    comps = None if uv2 is None else components_with(ctx.model, ctx.alive, uv2)
    with model_state(ctx.model, target=target, components=comps):
        net = ctx.model.target_model
        rec: dict[str, Any] = {"logp": mean_logp(net, refs.held, dev, b), "gap": mean_kl(refs.teacher, net, refs.held, dev, b),
                               "damage": mean_kl(refs.w0, net, refs.plain_held, dev, b)}
        if pile:
            rec["pile"] = harness.pile_ratios(ctx, log)
    return rec
