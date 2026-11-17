"""Grafts: fresh machinery added to the frozen 67M (M) and trained alone, the held-model reading's built-new cases.
Moved here from `experiments/graft_negative/run.py` (E201, cell P1.26; that file stays as it ran) for cell P1.28
(`experiments/graft_sites`).

Two kinds, each attached in place to a copy F of M:
    head     one attention head (q, k, v of 128 x 768, o of 768 x 128) appended to layer l's attention. It reads that
             layer's rms_1 output with M's rotary positions; its 128-wide output joins M's six heads' before the output
             matrix, which is M's o_proj with the graft's o beside it (`Joined`). `commute.held_model` sees seven heads.
    neurons  n MLP neurons appended to layer l's MLP (`rule_retention.Extra` through `Model.append`), reading rms_2's
             output; `commute.held_model` sees 3,072 + n neurons at that site.
Both start at the 67M's own init (normal, sd init_std for input weights, init_std / sqrt(2 n_layer) for output weights)
and train with the 67M's pretraining AdamW recipe (`Recipe`, sources in each experiment's predictions.json).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from commute import held_model as HM
from commute import rule_retention as R
from commute.variable_read import cut

D = R.D_HEAD


@dataclass(frozen=True)
class Recipe:
    init_std: float  # input weights (q, k, v; the neurons' w_in)
    out_std: float  # output weights (o; the neurons' w_out)
    lr: float
    lr_min_frac: float
    warmup: int
    steps: int
    batch: int  # windows per step
    beta1: float
    beta2: float
    weight_decay: float
    grad_clip: float
    acc_at: tuple[int, ...]  # steps at which held-out accuracy is logged

    def __post_init__(self) -> None:
        assert 0 < self.warmup < self.steps and self.acc_at[-1] == self.steps
        assert abs(self.out_std - self.init_std / math.sqrt(2 * 4)) < 1e-12  # the 67M's residual-scaled init, 4 layers


# ---- the head graft ------------------------------------------------------------------------------------------------


def pattern(attn: Any, h: Tensor, wq: Tensor, wk: Tensor, n: int) -> Tensor:
    """(B, n, T, T) attention of n heads with query and key weights wq, wk on normed input h: the 67M's manual path
    (model_config.yaml flash_attention false) with `attn`'s rotary tables."""
    B, T, _ = h.shape
    q, k = ((h @ w.T).view(B, T, n, D).transpose(1, 2) for w in (wq, wk))
    pos = torch.arange(T, device=h.device)[None]
    q, k = attn.apply_rotary_pos_emb(q, k, attn.rotary_cos[pos].to(q.dtype), attn.rotary_sin[pos].to(q.dtype))
    s = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))
    return s.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=h.device).tril(), float("-inf")).softmax(-1)


class Head(nn.Module):
    base: Any

    def __init__(self, base: Any, seed: int, rec: Recipe) -> None:
        super().__init__()
        gen, d = torch.Generator().manual_seed(seed), base.o_proj.weight.shape[0]
        self.wq, self.wk, self.wv = (nn.Parameter(torch.randn(D, d, generator=gen) * rec.init_std) for _ in range(3))
        self.wo = nn.Parameter(torch.randn(d, D, generator=gen) * rec.out_std)
        object.__setattr__(self, "base", base)  # M's attention, for its rotary tables only; not a submodule

    def pattern(self, h: Tensor) -> Tensor:
        return pattern(self.base, h, self.wq, self.wk, 1)[:, 0]

    def z(self, h: Tensor) -> Tensor:
        return self.pattern(h) @ (h @ self.wv.T)


class Joined(nn.Module):
    """F's output matrix at the grafted layer: M's o_proj on the first 768 inputs, the graft's o on the last 128."""

    graft: Head

    def __init__(self, m_o: nn.Linear, graft: Head) -> None:
        super().__init__()
        self.m_o = m_o
        object.__setattr__(self, "graft", graft)

    @property
    def weight(self) -> Tensor:  # (768, 896), what held_model.out_weight reads
        return torch.cat([self.m_o.weight, self.graft.wo], 1)

    def forward(self, z: Tensor) -> Tensor:
        n = self.m_o.weight.shape[1]
        return self.m_o(z[..., :n]) + z[..., n:] @ self.graft.wo.T


class Grafted(nn.Module):
    """M's six heads (`base`, M's module, frozen) and the graft, joined before the output matrix."""

    def __init__(self, base: Any, graft: Head) -> None:
        super().__init__()
        self.base, self.graft, self.o_proj = base, graft, Joined(base.o_proj, graft)

    def forward(self, h: Tensor) -> Tensor:
        seen: list[Tensor] = []
        hook = self.base.o_proj.register_forward_pre_hook(lambda _m, a: seen.append(a[0]))
        try:
            self.base(h)
        finally:
            hook.remove()
        return self.o_proj(torch.cat([seen[0], self.graft.z(h)], -1))


def attach_head(F: R.Model, layer: int, seed: int, rec: Recipe) -> Head:
    base = F.net.h[layer].attn
    g = Head(base, seed, rec).to(base.o_proj.weight.device)
    F.net.h[layer].attn = Grafted(base, g)
    return g


def attach_neurons(F: R.Model, layer: int, n: int, seed: int, rec: Recipe) -> R.Extra:
    mlp = F.net.h[layer].mlp
    e = R.Extra(n, mlp.c_fc.weight.shape[1], mlp.gelu, rec.init_std, rec.out_std, seed).to(mlp.c_fc.weight.device)
    F.append(layer, e)
    return e


def out_param(g: nn.Module) -> nn.Parameter:
    """The graft's output weight: zeroing it gives back M."""
    return g.wo if isinstance(g, Head) else g.w_out  # type: ignore[return-value]


def write(g: nn.Module, block: Any, x: Tensor, y: Tensor) -> Tensor:
    """The graft's write into the residual at its layer, on block input x and pre-MLP residual y."""
    return g.z(block.rms_1(x)) @ g.wo.T if isinstance(g, Head) else g(block.rms_2(y))


def m_attn(block: Any) -> Any:
    return getattr(block.attn, "base", block.attn)


# ---- training and saving -------------------------------------------------------------------------------------------


def rate(t: int, rec: Recipe) -> float:
    """The 67M's pretraining schedule (param-decomp pretrain/train.py get_lr), t from 0, over rec.steps."""
    lo = rec.lr * rec.lr_min_frac
    if t < rec.warmup:
        return rec.lr * (t + 1) / rec.warmup
    return lo + 0.5 * (1.0 + math.cos(math.pi * (t - rec.warmup) / (rec.steps - rec.warmup))) * (rec.lr - lo)


def train(g: nn.Module, loss_of: Callable[[Tensor], Tensor], rows: Tensor, window: int, rec: Recipe, seed: int,
          held_out: Callable[[], float], log: Callable[[str], None]) -> dict[str, Any]:  # fmt: skip
    """AdamW on g's weights alone, `rec.steps` steps of `rec.batch` windows cut from rows. Returns the loss curve, the
    held-out accuracy at rec.acc_at, and "failed" (a non-finite loss, which ends training) or None."""
    opt = torch.optim.AdamW(g.parameters(), lr=rec.lr, betas=(rec.beta1, rec.beta2), weight_decay=rec.weight_decay)
    gen, curve, acc, dev = torch.Generator().manual_seed(seed), list[float](), dict[int, float](), next(g.parameters()).device
    for t in range(rec.steps):
        for group in opt.param_groups:
            group["lr"] = rate(t, rec)
        loss = loss_of(cut(rows, rec.batch, window, gen).to(dev))
        curve.append(float(loss.detach()))
        if not math.isfinite(curve[-1]):
            return {"curve": curve, "accuracy_at": acc, "failed": f"loss {curve[-1]} at step {t + 1}"}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(g.parameters()), rec.grad_clip)
        opt.step()
        if t + 1 in rec.acc_at or t == 0 or (t + 1) % 500 == 0:
            line = f"step {t + 1}: loss {curve[-1]:.4f} (mean of last 250 {sum(curve[-250:]) / len(curve[-250:]):.4f}), rate {rate(t, rec):.3g}"
            if t + 1 in rec.acc_at:
                acc[t + 1] = held_out()
                line += f", held-out accuracy {acc[t + 1]:.4f}"
            log(line)
    g.requires_grad_(False)
    return {"curve": curve, "accuracy_at": acc, "failed": None}


def save(g: nn.Module, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({n: p.detach().cpu() for n, p in g.named_parameters()}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def load(g: nn.Module, path: Path) -> None:
    state = torch.load(path, weights_only=True)
    for n, p in g.named_parameters():
        p.copy_(state[n])
    g.requires_grad_(False)


# ---- measurements --------------------------------------------------------------------------------------------------


def chunks(toks: Tensor, size: int) -> list[Tensor]:
    return [toks[i : i + size] for i in range(0, toks.shape[0], size)]


@torch.no_grad()
def shares(N: R.Model, toks: Tensor, offsets: tuple[int, int], batch: int) -> tuple[Tensor, float]:
    """(layers, 6, offsets) each M head's mean attention at each offset in N, over the queries whose key at the largest
    offset is past the two sink positions; and the max relative error of the six heads' outputs rebuilt from these
    patterns against the module's own (the pattern code's check)."""
    lo, hi = offsets
    T = toks.shape[1]
    q = torch.arange(hi + 2, T, device=toks.device)
    tot, err = torch.zeros(len(N.net.h), R.N_HEAD, hi - lo + 1, dtype=torch.float64), 0.0
    for t in chunks(toks, batch):
        for l, (x, _) in enumerate(R.m_inputs(N, t)):
            a, b = m_attn(N.net.h[l]), N.net.h[l]
            h = b.rms_1(x)
            att = pattern(a, h, a.q_proj.weight, a.k_proj.weight, R.N_HEAD)
            v = (h @ a.v_proj.weight.T).view(*h.shape[:2], R.N_HEAD, D).transpose(1, 2)
            z, z0 = (att @ v).transpose(1, 2).flatten(2), HM.z(b, x)[..., : R.N_HEAD * D]
            err = max(err, float((z - z0).abs().max() / z0.abs().max()))
            for i, d in enumerate(range(lo, hi + 1)):
                tot[l, :, i] += att[:, :, q, q - d].double().sum((0, 2)).cpu()
    return tot / (toks.shape[0] * len(q)), err


@torch.no_grad()
def head_attention(F: R.Model, g: Head, layer: int, toks: Tensor, k: int, batch: int) -> float:
    """The head graft's mean attention from the last position to the position k back."""
    tot = 0.0
    for t in chunks(toks, batch):
        x = R.m_inputs(F, t)[layer][0]
        tot += float(g.pattern(F.net.h[layer].rms_1(x))[:, -1, -1 - k].sum())
    return tot / toks.shape[0]


@torch.no_grad()
def zeroed_error(F: R.Model, M: R.Model, g: nn.Module, toks: Tensor, batch: int) -> float:
    """F with the graft's output weight zeroed against M: max relative logit error (held_model.logit_error)."""
    w = out_param(g)
    keep = w.clone()
    w.zero_()
    try:
        return max(HM.logit_error(F.logits(t), M.logits(t)) for t in chunks(toks, batch))
    finally:
        w.copy_(keep)


@torch.no_grad()
def site_sizes(M: R.Model, g: nn.Module, layer: int, toks: Tensor, batch: int) -> dict[str, float]:
    """Over toks, on M's input to the grafted layer: |graft write|^2 and |F's site write|^2 (M's site write plus the
    graft's; the denominator of held_model.residual there)."""
    n_g, n_F = 0.0, 0.0
    for t in chunks(toks, batch):
        x, y = R.m_inputs(M, t)[layer]
        b = M.net.h[layer]
        site = b.attn(b.rms_1(x)) if isinstance(g, Head) else b.mlp(b.rms_2(y))
        w = write(g, b, x, y).double()
        n_g += float(w.pow(2).sum())
        n_F += float((site.double() + w).pow(2).sum())
    return {"graft": n_g, "F_site": n_F}
