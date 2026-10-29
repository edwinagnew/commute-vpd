"""Training loop for the PCFG minGPT (plan reports/plans/2026-10-29_plan_reuse-stage1.md, Stage A).

Recipe (Jain et al. 2024, L1164-1196): AdamW, linear warm-up over the first 20% then cosine to lr / 100 (the paper
states the floor for fine-tuning only, "similar to pre-training", L1193-1195). From minGPT's trainer defaults, the
paper's named code (L716): betas (0.9, 0.95), weight decay 0.1 on Linear weights, gradient-norm clip 1.0.
bf16 autocast on CUDA with fp32 weights.

Data never touches disk: batch `step` is block `step % chunk` of chunk `step // chunk`, and chunk c is drawn from
`default_rng([seed, c])`, so a resume sees the same batches. The next chunk is drawn in a background thread.
A checkpoint holds model, optimizer, step and the torch RNG states; `train` resumes from it when the file exists.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from commute import pcfg
from commute.mingpt import make_optimizer

GRAD_CLIP = 1.0


@dataclass(frozen=True)
class Schedule:
    iters: int
    lr: float
    warmup_frac: float = 0.2
    min_frac: float = 0.01

    def lr_at(self, step: int) -> float:
        """Learning rate for optimizer step `step` (0-based)."""
        warm = max(1, int(self.warmup_frac * self.iters))
        if step < warm:
            return self.lr * (step + 1) / warm
        progress = (step - warm) / max(1, self.iters - warm)
        return self.lr * (self.min_frac + (1 - self.min_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


class Batches:
    """Deterministic batches from `make(rng, n) -> (n, SEQ) int array`, drawn a chunk at a time."""

    def __init__(self, make: Callable[[np.random.Generator, int], np.ndarray], seed: int, batch: int, chunk: int,
                 device: torch.device) -> None:  # fmt: skip
        self.make, self.seed, self.batch, self.chunk, self.device = make, seed, batch, chunk, device
        self._pool = ThreadPoolExecutor(1)
        self._chunks: dict[int, Future[np.ndarray]] = {}

    def _draw(self, c: int) -> np.ndarray:
        return self.make(np.random.default_rng([self.seed, c]), self.batch * self.chunk)

    def get(self, step: int) -> torch.Tensor:
        c, i = divmod(step, self.chunk)
        for k in (c, c + 1):
            if k not in self._chunks:
                self._chunks[k] = self._pool.submit(self._draw, k)
        self._chunks.pop(c - 1, None)
        rows = self._chunks[c].result()[i * self.batch : (i + 1) * self.batch]
        return torch.from_numpy(rows.astype(np.int64)).to(self.device, non_blocking=True)


def autocast(device: torch.device) -> torch.autocast:
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def lm_loss(model: nn.Module, seq: torch.Tensor) -> torch.Tensor:
    """Next-token loss on the whole sequence (pretraining, L1166)."""
    logits = model(seq[:, :-1])
    return F.cross_entropy(logits.flatten(0, 1).float(), seq[:, 1:].flatten())


def save_checkpoint(path: Path, model: nn.Module, opt: torch.optim.Optimizer, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}  # fmt: skip
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, model: nn.Module, opt: torch.optim.Optimizer | None = None) -> int:
    """Load `path` into model (and optimizer, RNG) if it exists; returns its step, 0 if there is no file."""
    if not path.exists():
        return 0
    state = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    model.load_state_dict(state["model"])
    if opt is not None:
        opt.load_state_dict(state["opt"])
        torch.set_rng_state(state["rng"].cpu())
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    return int(state["step"])


@dataclass(frozen=True)
class Run:
    sched: Schedule
    weight_decay: float
    save_every: int
    log_every: int
    loss: Callable[[nn.Module, torch.Tensor], torch.Tensor] = lm_loss


def train(model: nn.Module, batches: Batches, run: Run, ckpt: Path, *, stop_at: int | None = None,
          deadline: float | None = None, log: Callable[[str], None] = print) -> tuple[int, float]:  # fmt: skip
    """Train from `ckpt` (or from scratch) to `stop_at` (default: the schedule's end), saving every `save_every` steps,
    at the end, and when `time.monotonic()` passes `deadline`. Returns (step reached, ms/step of the last log window)."""
    device = next(model.parameters()).device
    opt = make_optimizer(model, run.sched.lr, run.weight_decay)
    step = load_checkpoint(ckpt, model, opt)
    end = run.sched.iters if stop_at is None else min(stop_at, run.sched.iters)
    log(f"train: from step {step} to {end} ({'resumed' if step else 'fresh'}), checkpoint {ckpt}")
    fwd = cast(nn.Module, torch.compile(model)) if device.type == "cuda" else model
    model.train()
    ms, t_win = float("nan"), time.perf_counter()
    while step < end:
        for g in opt.param_groups:
            g["lr"] = run.sched.lr_at(step)
        with autocast(device):
            loss = run.loss(fwd, batches.get(step))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        step += 1
        if step % run.log_every == 0:
            ms, t_win = 1e3 * (time.perf_counter() - t_win) / run.log_every, time.perf_counter()
            log(f"step {step} loss {loss.item():.4f} lr {run.sched.lr_at(step - 1):.2e} {ms:.1f} ms/step")
        late = deadline is not None and time.monotonic() > deadline
        if step % run.save_every == 0 or step == end or late:
            save_checkpoint(ckpt, model, opt, step)
        if late:
            log(f"deadline: saved at step {step}")
            break
    return step, ms


@torch.no_grad()
def answer_accuracy(model: nn.Module, seq: np.ndarray, chunk: int = 1000) -> float:
    """Share of rows whose argmax prediction at the `=` position is the answer token."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    correct = 0
    for i in range(0, len(seq), chunk):
        x = torch.from_numpy(seq[i : i + chunk, : pcfg.ANS].astype(np.int64)).to(device)
        with autocast(device):
            pred = model(x)[:, -1].argmax(-1).cpu().numpy()
        correct += int((pred == seq[i : i + chunk, pcfg.ANS]).sum())
    model.train(was_training)
    return correct / len(seq)
