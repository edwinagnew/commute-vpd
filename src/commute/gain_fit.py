"""Fitting a gain-only update to a target model (cell P1.58, E282): the best G = M with only M's own variables rescaled,
by Adam on the gains alone against KL(F || G), and the reads that score it. Level-agnostic: a `Level` gives the gains
of one featurization (`levels`: N, one gain per MLP neuron and attention head, `commute.gains`; V, one gain per VPD
subcomponent and matrix remainder, `commute.vpd_gains`) through a hook context, so M's weights are never written.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from commute import gains as G
from commute import rule_retention as R
from commute import steering as ST
from commute import vpd_gains as VG


@dataclass
class Rows:
    train: Tensor  # (n, T) windows the fit samples, fed whole
    held: Tensor  # (m, T) the read rows
    at: slice  # the positions KL is taken at
    pos: int  # the trained answer's position
    tgt: Tensor | None  # (m,) the trained answer on each read row; None: no task answer


@dataclass(frozen=True)
class FitSettings:
    lr: float
    steps: int  # the schedule's horizon
    warmup: int
    batch: int
    grad_clip: float
    check_every: int
    read_batch: int


@dataclass(frozen=True)
class Level:
    name: str
    ones: Callable[[], Any]
    applied: Callable[[Any], AbstractContextManager[Any]]
    delta: Callable[[Any], dict[str, Tensor]]
    shuffled: Callable[[Any, torch.Generator], Any]  # the same gains, each permuted within its matrix (or block)


def levels(net: ST.Net, b: VG.Basis) -> dict[str, Level]:
    perm = lambda x, gen: x[..., torch.randperm(x.shape[-1], generator=gen).to(x.device)]
    return {
        "N": Level("N", lambda: G.Gains.ones(net.net, net.device), lambda g: G.applied(net.net, g), lambda g: G.delta(g, net.W0),
                   lambda g, gen: G.Gains(perm(g.mlp, gen), perm(g.head, gen))),
        "V": Level("V", lambda: VG.VGains.ones(b), lambda g: VG.applied(net.net, b, g), lambda g: VG.delta(b, g),
                   lambda g, gen: VG.VGains({k: perm(v, gen) for k, v in g.comp.items()}, g.rem)),
    }  # fmt: skip


def logits(net: Any, t: Tensor) -> Tensor:
    out = net(t)
    return (out[0] if isinstance(out, tuple) else out).float()


class _KL(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, zp: Tensor, zq: Tensor) -> Tensor:
        lp, lq = zp.log_softmax(-1), zq.log_softmax(-1)
        ctx.save_for_backward(lp, lq)
        return (lp.exp() * (lp - lq)).sum(-1).mean()

    @staticmethod
    def backward(ctx: Any, *grads: Tensor) -> tuple[None, Tensor]:
        lp, lq = ctx.saved_tensors
        return None, grads[0] * (lq.exp() - lp.exp()) / (lp.numel() // lp.shape[-1])


def kl_loss(zp: Tensor, zq: Tensor) -> Tensor:
    """KL(p || q) from logits, mean over all leading dims, with the backward softmax(q) - softmax(p) written out, so it
    is exactly zero where q's logits equal p's bit for bit. Autograd's log_softmax backward gives softmax(q) sum(p) - p,
    whose rounding (sum(p) is not exactly 1) Adam scaled into gain steps of ~4e-3 on a do-nothing (E282 dry run)."""
    out = _KL.apply(zp, zq)
    assert isinstance(out, Tensor)
    return out


def schedule(s: FitSettings) -> Callable[[int], float]:
    """Linear warm-up over `warmup` steps, then cosine to 0 at `steps` (the lr multiplier at optimizer step k)."""
    return lambda k: (k + 1) / s.warmup if k < s.warmup else 0.5 * (1 + math.cos(math.pi * (k - s.warmup) / (s.steps - s.warmup)))


@torch.no_grad()
def read(net: ST.Net, F: Any, toks: Tensor, r: Rows, batch: int, tgt: Tensor | None, g: Any = None, lv: Level | None = None) -> dict[str, float]:
    """Mean over rows `toks`: KL(F || M) and KL(F || G) at r.at (rule_retention.kl, fp64, mean over positions), and with
    targets the answer's log-odds at r.pos under F and G minus under M (F's gain, G's). Without a level, G is M."""
    s = {"kl_fm": 0.0, "kl_fg": 0.0, "gain_F": 0.0, "gain_G": 0.0}
    for i in range(0, len(toks), batch):
        t = toks[i : i + batch].to(net.device)
        lF, lM = logits(F, t), logits(net.net, t)
        with lv.applied(g) if lv is not None else nullcontext():
            lG = logits(net.net, t)
        s["kl_fm"] += float(R.kl(lF[:, r.at], lM[:, r.at]).sum())
        s["kl_fg"] += float(R.kl(lF[:, r.at], lG[:, r.at]).sum())
        if tgt is not None:
            y = tgt[i : i + batch]
            m = ST.logodds_at(lM, r.pos, y)
            s["gain_F"] += float((ST.logodds_at(lF, r.pos, y) - m).sum())
            s["gain_G"] += float((ST.logodds_at(lG, r.pos, y) - m).sum())
    return {k: v / len(toks) for k, v in s.items() if tgt is not None or k.startswith("kl")}


def fit(net: ST.Net, lv: Level, F: Any, r: Rows, s: FitSettings, steps: int, seed: int, log: Callable[[str], Any] = print, label: str = "") -> tuple[Any, list[dict[str, float]]]:  # fmt: skip
    """Adam (betas 0.9, 0.999, no weight decay, as E231) on the gains alone, from 1, against KL(F || G) at r.at on
    batches drawn from r.train (seeded), `steps` steps of the schedule, the gradient clipped at grad_clip; at step 1 and
    every check_every steps the mean train loss since the last check and KL(F || G) on the first read_batch held rows."""
    g = lv.ones()
    for p in g.params():
        p.requires_grad_(True)
    opt = torch.optim.Adam(g.params(), lr=s.lr, betas=(0.9, 0.999), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule(s))
    gen, curve, run = torch.Generator().manual_seed(seed), [], []
    for step in range(1, steps + 1):
        t = r.train[torch.randint(0, len(r.train), (s.batch,), generator=gen)].to(net.device)
        with torch.no_grad():
            zp = logits(F, t)[:, r.at]
        with lv.applied(g):
            loss = kl_loss(zp, logits(net.net, t)[:, r.at])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(g.params(), s.grad_clip)
        opt.step()
        sched.step()
        run.append(loss.item())
        if step == 1 or step % s.check_every == 0:
            h = read(net, F, r.held[: s.read_batch], r, s.read_batch, None, g, lv)
            curve.append({"step": step, "train_kl": sum(run) / len(run), "held_kl": h["kl_fg"], "grad_norm": float(gn)})
            log(f"  [{label} {lv.name}] step {step}: train KL {curve[-1]['train_kl']:.5g} held KL {h['kl_fg']:.5g} grad {float(gn):.3g}")
            run = []
    for p in g.params():
        p.requires_grad_(False)
    return g, curve
