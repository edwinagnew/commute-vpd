"""Updates trained only inside fixed random subspaces (P2.87, experiments/subspace_share). On each matrix W (out x in):
fixed random orthonormal U (out x k) and V (in x k) from a seed (QR of a Gaussian), a trainable Z (k x k) starting at 0,
and the update U Z V^T. `k_like_lora` gives each matrix the k whose k^2 is nearest a rank-r LoRA's r (out + in), so the
trainable count matches that LoRA's. The forward hook adds x V Z^T U^T to the linear's output, as `finetune.lora_hooks`
adds B A; the target weights never change. `train` is `finetune._lora_run`'s loop (Adam, betas 0.9/0.999, no weight
decay, the gradient clipped over Z) with an `on_step` callback in place of checkpoints."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def k_like_lora(shape: tuple[int, int], rank: int) -> int:
    return round(math.sqrt(rank * (shape[0] + shape[1])))


def orthonormal(n: int, k: int, gen: torch.Generator) -> Tensor:
    """n x k with orthonormal columns: Q of a Gaussian's QR (float64 on the CPU, then float32)."""
    q, _ = torch.linalg.qr(torch.randn(n, k, generator=gen, dtype=torch.float64))
    return q.float()


class Subspace:
    """U, V and Z per matrix of `linears`, on the linears' device; `active` False makes the hooks add nothing."""

    def __init__(self, linears: dict[str, nn.Linear], ks: dict[str, int], seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.linears, self.ks, self.seed, self.active = linears, ks, seed, True
        self.U: dict[str, Tensor] = {}
        self.V: dict[str, Tensor] = {}
        self.Z: dict[str, nn.Parameter] = {}
        for name in sorted(linears):  # U then V per matrix, in sorted order: the draw is fixed by the seed alone
            out, inp = linears[name].weight.shape
            dev = linears[name].weight.device
            self.U[name] = orthonormal(out, ks[name], gen).to(dev)
            self.V[name] = orthonormal(inp, ks[name], gen).to(dev)
            self.Z[name] = nn.Parameter(torch.zeros(ks[name], ks[name], device=dev))

    def hooks(self) -> list[Any]:
        handles = []
        for name, lin in self.linears.items():
            U, Vt, Z = self.U[name], self.V[name].T.contiguous(), self.Z[name]

            def hook(_m: nn.Module, args: tuple[Any, ...], out: Tensor, U: Tensor = U, Vt: Tensor = Vt, Z: Tensor = Z) -> Tensor:
                if not self.active:
                    return out
                x = args[0].to(Z.dtype)
                return out + F.linear(F.linear(F.linear(x, Vt), Z), U).to(out.dtype)

            handles.append(lin.register_forward_hook(hook))
        return handles

    @torch.no_grad()
    def delta(self) -> dict[str, Tensor]:
        """U Z V^T per matrix, fp32 on the linears' device."""
        return {k: self.U[k] @ self.Z[k].float() @ self.V[k].T for k in self.linears}


def train(sub: Subspace, lr: float, steps: int, grad_clip: float, loss_fn: Callable[[int], tuple[Tensor, dict[str, float]]],
          on_step: Callable[[int, float, dict[str, float]], None]) -> None:  # fmt: skip
    """Adam on Z; `loss_fn(step) -> (loss, info)`; `on_step(step, loss, info)` after each step (an exception stops it).
    The hooks are removed on return."""
    params = list(sub.Z.values())
    opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    handles = sub.hooks()
    try:
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            loss, info = loss_fn(step)
            loss.backward()
            info = {**info, "grad_norm": float(torch.nn.utils.clip_grad_norm_(params, grad_clip))}
            opt.step()
            on_step(step, float(loss), info)
    finally:
        for h in handles:
            h.remove()
