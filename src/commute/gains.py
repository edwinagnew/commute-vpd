"""Gain-only training (cell P2.46; plan reports/plans/2026-11-20_plan_p2-46-gain-only.md): the base model frozen, one
trainable multiplier per MLP neuron (on `down_proj`'s input, after the GELU) and one per attention head (on `o_proj`'s
input, the same number on the head's `head_dim` columns), every gain starting at exactly 1.

A gain state is exactly a weight change of the two output matrices: down_proj(g * a) = (W diag(g)) a, so
dW = W0 diag(g - 1) on `down_proj`'s columns and on `o_proj`'s head blocks, zero on every other matrix. `delta` writes
it in the saved fine-tunes' format (one tensor per decomposed matrix, all 24 present), so every reader of a saved
fine-tune (the share, R, the held model) reads a gain state unchanged. Gains of exactly 1 change no bit of the forward
(x * 1.0 is x, in fp32 and through bf16 autocast) and export a dW of exactly zero.

`train` is the loop: Adam on the gains only, the batch sampler and loss given by the caller (the reference fine-tune's
own, e.g. `finetune.arm_a_loss`), the gradient clipped as the reference's, and `on_check(step, gains)` after step 1
and every `check_every` steps, which may return True to stop (matched gain reached).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


def down_key(layer: int) -> str:
    return f"h.{layer}.mlp.down_proj"


def o_key(layer: int) -> str:
    return f"h.{layer}.attn.o_proj"


@dataclass
class Gains:
    mlp: Tensor  # (L, n_neurons): neuron i of block l's MLP output times mlp[l, i]
    head: Tensor  # (L, n_head): head h of block l's output (o_proj's input columns h*hd:(h+1)*hd) times head[l, h]

    @classmethod
    def ones(cls, net: Any, device: str | torch.device = "cpu") -> Gains:
        """All gains 1 for a LlamaSimpleMLP-style net (`net.h[l].mlp.down_proj`, `net.h[l].attn.o_proj`, `attn.n_head`)."""
        n = net.h[0].mlp.down_proj.in_features
        return cls(torch.ones(len(net.h), n, device=device), torch.ones(len(net.h), net.h[0].attn.n_head, device=device))

    def params(self) -> list[Tensor]:
        return [self.mlp, self.head]

    def cpu(self) -> Gains:
        return Gains(self.mlp.detach().float().cpu().clone(), self.head.detach().float().cpu().clone())

    def to(self, device: str | torch.device) -> Gains:
        return Gains(self.mlp.detach().to(device).clone(), self.head.detach().to(device).clone())

    def state(self) -> dict[str, Tensor]:
        return {"mlp": self.mlp.detach().cpu(), "head": self.head.detach().cpu()}

    @classmethod
    def of(cls, d: Mapping[str, Tensor]) -> Gains:
        return cls(d["mlp"].float(), d["head"].float())


def head_cols(g: Tensor, width: int) -> Tensor:
    """One block's head gains (n_head,) spread over o_proj's input columns (width = n_head * head_dim)."""
    assert width % len(g) == 0, (width, len(g))
    return g.repeat_interleave(width // len(g))


@contextmanager
def applied(net: Any, g: Gains) -> Iterator[None]:
    """The net with gains g: a forward pre-hook on each block's down_proj and o_proj multiplies its input. The weights are
    not touched, so the base model is the same object with the hooks removed."""
    handles = []
    for l, block in enumerate(net.h):
        d, o = block.mlp.down_proj, block.attn.o_proj
        handles.append(d.register_forward_pre_hook(lambda _m, a, l=l: (a[0] * g.mlp[l],)))
        handles.append(o.register_forward_pre_hook(lambda _m, a, l=l, w=o.in_features: (a[0] * head_cols(g.head[l], w),)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def delta(g: Gains, W0: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """The weight change equal to gains g, fp32 CPU, one entry per matrix of W0 (zero on all but down_proj and o_proj):
    W0 diag(g - 1) on the columns. Gains of exactly 1 give exactly zero."""
    out = {k: torch.zeros(w.shape, dtype=torch.float32) for k, w in W0.items()}
    gm, gh = g.mlp.detach().float().cpu(), g.head.detach().float().cpu()
    for l in range(len(gm)):
        d, o = W0[down_key(l)].float().cpu(), W0[o_key(l)].float().cpu()
        out[down_key(l)] = d * (gm[l] - 1)[None, :]
        out[o_key(l)] = o * (head_cols(gh[l], o.shape[1]) - 1)[None, :]
    return out


def train(net: nn.Module, g: Gains, sample: Callable[[int], Tensor], loss_of: Callable[[Tensor], tuple[Tensor, dict[str, float]]],
          lr: float, steps: int, grad_clip: float, check_every: int, on_check: Callable[[int, Gains], bool],
          log: Callable[[str], Any] = print) -> list[dict[str, float]]:  # fmt: skip
    """Adam (betas 0.9, 0.999, no weight decay, as `finetune._run`) on the gains only, the net's own parameters frozen for
    the loop and restored after. `sample(step)` gives the step's batch, `loss_of(toks)` its loss with the gains applied
    (the caller's loss reads the base model as its reference). After step 1 and every `check_every` steps,
    `on_check(step, g)` runs (no grad) and returns True to stop. Returns the per-step log."""
    was = {n: p.requires_grad for n, p in net.named_parameters()}
    net.requires_grad_(False)
    for p in g.params():
        p.requires_grad_(True)
    opt = torch.optim.Adam(g.params(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    rows: list[dict[str, float]] = []
    try:
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            with applied(net, g):
                loss, info = loss_of(sample(step))
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(g.params(), grad_clip)
            opt.step()
            rows.append({"step": step, "loss": float(loss), "grad_norm": float(gn), **info})
            if step == 1 or step % check_every == 0 or step == steps:
                with torch.no_grad():
                    stop = on_check(step, g)
                log(f"[gain] step {step}: loss {float(loss):.4f} grad {float(gn):.4f} " + " ".join(f"{k} {v:.4f}" for k, v in info.items()))
                if stop:
                    break
    finally:
        for p in g.params():
            p.requires_grad_(False)
        for n, p in net.named_parameters():
            p.requires_grad_(was[n])
    return rows
