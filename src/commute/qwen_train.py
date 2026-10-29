"""SFT of Qwen3-1.7B's 196 decomposed matrices with δW saved where the KL to start crosses a fixed grid (E45, plan
reports/plans/2026-10-28_plan_qwen-q2.md, section 3). The step, loss and optimiser are `qwen_sft`'s (E40, E40
Amendment 1): fp32 master weights, bf16 autocast forward, AdamW, the library's warm-up and cosine schedule, gradient
clipping. Batches are `qwen_sft.Item`s: sequences (prompt length) or packed rows (per-token target flags).

At each read step the KL to start, mean over every position of KL(W0 ‖ W_t) in fp32, is read on each axis's tracker
rows. The first read at or past a grid value on an axis saves δW (bf16) at that step and at the read step before it
(the bracket; none when the first read is already past); the steps in `keep` (the last among them) are always saved.
Which of the two states is read at the grid value is decided later, from the KL on all rows (`qwen_updates.choose`).

A run longer than one 30-min job pauses after every `pause` steps: it writes a checkpoint and returns; called again
with the checkpoint present, it goes on from there as if it had not stopped (E45 Amendment 1).
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute.qwen_sft import (
    Item,
    Weights,
    adamw,
    lr_at,
    lr_schedule,
    mean_kl,
    sft_step,
    trainable,
)
from commute.qwen_updates import crossed

DEV = "cuda"


@dataclass(frozen=True)
class Recipe:
    lr: float
    warmup: int  # "hf_cosine": steps of linear warm-up, step 1 at rate 0 (`qwen_sft.lr_at`)
    steps: int
    schedule: str  # "constant" or "hf_cosine"
    clip: float | None
    micro: int


def delta(params: Mapping[str, Tensor], W0: Weights) -> dict[str, Tensor]:
    """δW in bf16 on the host, one matrix at a time (no GPU copy of the whole change)."""
    return {s: (p.detach() - W0[s]).to("cpu", torch.bfloat16) for s, p in params.items()}


def norms(params: Mapping[str, Tensor], W0: Weights) -> dict[str, float]:
    with torch.no_grad():
        return {s: float((p.detach() - W0[s]).norm()) for s, p in params.items()}


def one_step(model: Any, params: Mapping[str, Tensor], W0: Weights, opt: Any, sched: Any, batch: Sequence[Item],
             r: Recipe, t: int) -> dict[str, Any]:
    """One optimiser step; returns its rate, loss, gradient norm (before clipping) and |δW| per matrix after it."""
    lr = opt.param_groups[0]["lr"]
    assert math.isclose(lr, lr_at(t, r.lr, r.warmup, r.steps, r.schedule), rel_tol=1e-9, abs_tol=1e-15), (t, lr)
    opt.zero_grad(set_to_none=True)
    model.train()
    loss = sft_step(model, batch, r.micro, DEV)
    model.eval()
    gnorm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), r.clip or math.inf))
    opt.step()
    if sched is not None:
        sched.step()
    n = norms(params, W0)
    return {"step": t, "lr": lr, "loss": loss, "grad_norm": gnorm, "norms": n,
            "dw_norm": math.sqrt(sum(v * v for v in n.values()))}


def checkpoint(path: Path, t: int, params: Mapping[str, Tensor], W0: Weights, opt: Any, sched: Any,
               done: set[tuple[str, float]], saved: set[int]) -> None:
    """Everything `train` needs to go on after step t (fp32 weights and Adam moments, ~17 GB for the 196 matrices),
    written whole, then renamed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"step": t, "params": {s: p.detach() for s, p in params.items()}, "opt": opt.state_dict(),
                "sched": None if sched is None else sched.state_dict(), "done": sorted(done), "saved": sorted(saved),
                "norms": norms(params, W0)}, tmp)
    os.replace(tmp, path)


def resume(path: Path, params: Mapping[str, Tensor], W0: Weights, opt: Any,
           sched: Any) -> tuple[int, set[tuple[str, float]], set[int]]:
    """The state `checkpoint` wrote, put back; its step, the grid values crossed and the steps saved."""
    c = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    with torch.no_grad():
        for s, p in params.items():
            p.copy_(c["params"][s])
    assert norms(params, W0) == c["norms"], "the checkpoint's weights were not put back"
    opt.load_state_dict(c["opt"])
    if sched is not None:
        sched.load_state_dict(c["sched"])
    return c["step"], {(a, g) for a, g in c["done"]}, set(c["saved"])


def train(model: Any, W0: Weights, batch_at: Callable[[int], Sequence[Item]], r: Recipe, axes: dict[str, list[Tensor]],
          reads: set[int], grid: Sequence[float], keep: set[int], save: Callable[[int, dict[str, Tensor]], None],
          on_step: Callable[[dict[str, Any]], None], ckpt: Path, pause: int) -> int:
    """r.steps steps from W0 (or from `ckpt` when it exists). After each step `on_step` gets its record (with `kl` per
    axis and `crossed` at read steps); it runs inside the training context, so it may read the model as it stands (the
    early checks). After a read step t < r.steps that is a multiple of `pause`, the state goes to `ckpt` and t is
    returned; at the end `ckpt` is deleted and r.steps returned."""
    assert r.steps in keep and keep <= reads
    n_open = len(axes) * len(grid)
    with trainable(model, W0) as params:
        opt = adamw(params, r.lr)
        sched = lr_schedule(opt, r.warmup, r.steps, r.schedule)
        start, done, saved = resume(ckpt, params, W0, opt, sched) if ckpt.is_file() else (0, set(), set())
        prev = (start, delta(params, W0)) if start and len(done) < n_open else None
        for t in range(start + 1, r.steps + 1):
            rec = one_step(model, params, W0, opt, sched, batch_at(t), r, t)
            if t in reads:
                rec["kl"] = {a: mean_kl(model, rows, W0, None) for a, rows in axes.items()}
                new = crossed(rec["kl"], done, grid)
                rec["crossed"] = [{"axis": a, "grid": g, "lo": prev[0] if prev else None, "hi": t} for a, g in new]
                dW = delta(params, W0) if new or t in keep or len(done) + len(new) < n_open else None
                if dW is not None and (new or t in keep):
                    save(t, dW)
                    saved.add(t)
                if new and prev is not None and prev[0] not in saved:
                    save(*prev)
                    saved.add(prev[0])
                done |= set(new)
                prev = (t, dW) if dW is not None and len(done) < n_open else None
            on_step(rec)
            if t in reads and t % pause == 0 and t < r.steps:
                checkpoint(ckpt, t, params, W0, opt, sched, done, saved)
                return t
    ckpt.unlink(missing_ok=True)
    return r.steps
