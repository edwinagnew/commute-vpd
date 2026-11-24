"""The reuse dial (cell P2.63): an update to the 67M written as a gain on each VPD variable plus a free residual,

    dW = (g_R - 1) R0 + U^T diag(g - 1) V^T + Res        (per decomposed matrix; `vpd_gains` for the gain part)

trained on a task with the residual penalised by beta * sum ||Res||^2 / sum ||W0||^2 (weight decay towards zero on the
residual, scaled by the model's size). beta = inf: no residual (gains only, every variable kept up to a gain); beta = 0
with no gains: an ordinary fine-tune of the 24 matrices. The weights are never written: `applied` adds the change's
output to each decomposed matrix's output through a forward hook, as `vpd_gains.applied` does, so gains of exactly 1
and a residual of exactly 0 change no bit of the forward, and `delta` exports exactly zero.

`train` is the loop: Adam (betas 0.9, 0.999, no weight decay, as `finetune._run`) with one rate for the gains and one
for the residual, the gradient of each group clipped at grad_clip, and `on_check(step, d)` after step 1 and every
`check_every` steps, which returns True to stop (the matched gain reached).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from commute import vpd_gains as VG


@dataclass
class Dial:
    g: VG.VGains | None  # None: no gains (the ordinary fine-tune)
    res: dict[str, Tensor] | None  # per decomposed matrix (d_out, d_in); None: no residual (gains only)

    @classmethod
    def start(cls, b: VG.Basis, gains: bool, residual: bool) -> Dial:
        """Gains of exactly 1 and a residual of exactly 0: the base model."""
        assert gains or residual
        res = {k: torch.zeros(b.R[k].shape, device=b.R[k].device) for k in b.keys} if residual else None
        return cls(VG.VGains.ones(b) if gains else None, res)

    def groups(self) -> dict[str, list[Tensor]]:
        out = {"gain": self.g.params()} if self.g is not None else {}
        return out | ({"res": list(self.res.values())} if self.res is not None else {})

    def state(self) -> dict[str, Any]:
        return {"g": self.g.state() if self.g is not None else None, "res": {k: v.detach().cpu() for k, v in self.res.items()} if self.res is not None else None}

    def gains_only(self) -> Dial:
        """The same gains with the residual removed (the dial's construction reading: what the gains alone give)."""
        return Dial(self.g, None)


@contextmanager
def applied(net: Any, b: VG.Basis, d: Dial) -> Iterator[None]:
    """The net with the change d: a forward hook on each decomposed nn.Linear adds x @ dW^T to its output."""
    g, res, handles = d.g, d.res, []
    for k in b.keys:

        def hook(_m: Any, a: tuple[Tensor, ...], o: Tensor, k: str = k) -> Tensor:
            x = a[0]
            if g is not None:
                o = o + (g.rem[k] - 1) * F.linear(x, b.R[k]) + ((x @ b.V[k]) * (g.comp[k] - 1)) @ b.U[k]
            return o + F.linear(x, res[k]) if res is not None else o

        handles.append(net.get_submodule(k).register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def delta(b: VG.Basis, d: Dial) -> dict[str, Tensor]:
    """The weight change equal to d, fp32 CPU, one entry per decomposed matrix; the start exports exactly 0."""
    out = VG.delta(b, d.g) if d.g is not None else {k: torch.zeros(b.R[k].shape) for k in b.keys}
    return {k: out[k] + d.res[k].detach().float().cpu() for k in b.keys} if d.res is not None else out


def w0_sq(W0: Mapping[str, Tensor]) -> float:
    return sum(float(w.double().pow(2).sum()) for w in W0.values())


def penalty(d: Dial, w0sq: float) -> Tensor:
    """sum ||Res||^2 / sum ||W0||^2 (0 without a residual)."""
    assert d.res is not None
    return torch.stack([r.pow(2).sum() for r in d.res.values()]).sum() / w0sq


def residual_size(d: Dial, w0sq: float) -> float:
    return 0.0 if d.res is None else math.sqrt(sum(float(r.double().pow(2).sum()) for r in d.res.values()) / w0sq)


def train(net: Any, b: VG.Basis, d: Dial, sample: Callable[[int], Tensor], loss_of: Callable[[Tensor], Tensor], lr: dict[str, float],
          beta: float, w0sq: float, steps: int, grad_clip: float, check_every: int, on_check: Callable[[int, Dial], bool],
          log: Callable[[str], Any] = print) -> list[dict[str, float]]:  # fmt: skip
    """Adam on d's groups (net frozen for the loop, restored after); loss = loss_of(batch) + beta * penalty when d has a
    residual and beta > 0 (w0sq = `w0_sq(W0)`). Returns the per-step log."""
    was = {n: p.requires_grad for n, p in net.named_parameters()}
    net.requires_grad_(False)
    groups = d.groups()
    for ps in groups.values():
        for p in ps:
            p.requires_grad_(True)
    opt = torch.optim.Adam([{"params": ps, "lr": lr[n]} for n, ps in groups.items()], betas=(0.9, 0.999), weight_decay=0.0)
    rows: list[dict[str, float]] = []
    try:
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            with applied(net, b, d):
                task = loss_of(sample(step))
            pen = penalty(d, w0sq) if d.res is not None and beta > 0 else torch.zeros((), device=task.device)
            (task + beta * pen).backward()
            gn = {n: float(torch.nn.utils.clip_grad_norm_(ps, grad_clip)) for n, ps in groups.items()}
            opt.step()
            rows.append({"step": step, "loss": float(task), "penalty": float(pen)} | {f"grad_{n}": v for n, v in gn.items()})
            if step == 1 or step % check_every == 0 or step == steps:
                with torch.no_grad():
                    stop = on_check(step, d)
                log(f"[dial] step {step}: loss {float(task):.4f} penalty {float(pen):.3g} " + " ".join(f"grad_{n} {v:.3g}" for n, v in gn.items()))
                if stop:
                    break
    finally:
        for ps in groups.values():
            for p in ps:
                p.requires_grad_(False)
        for n, p in net.named_parameters():
            p.requires_grad_(was[n])
    return rows
