"""Fine-tunes of the 24 decomposed matrices (everything else frozen), with delta-W checkpoints.

Arm A (emoticon -> `o`): the paper's LoRA recipe as a full-weight fine-tune: cross-entropy on `o`
at the causally-important position of each window plus lambda * KL(reference || fine-tuned) on
the other positions of the window.
Arm B (continued pretraining): next-token cross-entropy on fresh Pile train rows.
Arm N (random direction): Gaussian delta-W with the per-matrix Frobenius norm of a given delta-W.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from commute.data import Windows
from commute.vpd import (
    kl_per_position,
    module_keys,
    set_target_weights,
    target_linear,
    target_weights,
)


def decomposed_params(model: Any) -> dict[str, nn.Parameter]:
    return {k: target_linear(model, k).weight for k in module_keys(model)}


def relative_norms(dW: dict[str, Tensor], W0: dict[str, Tensor]) -> dict[str, float]:
    return {k: float(dW[k].norm() / W0[k].norm()) for k in dW}


def random_direction(dW: dict[str, Tensor], seed: int) -> dict[str, Tensor]:
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k, d in dW.items():
        r = torch.randn(d.shape, generator=gen, dtype=torch.float32)
        out[k] = r * (d.norm() / r.norm())
    return out


def logits_of(net: nn.Module, tokens: Tensor) -> Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tokens.is_cuda):
        out = net(tokens)
    return (out[0] if isinstance(out, tuple) else out).float()


@dataclass
class ArmAConfig:
    o_token: int
    kl_lambda: float = 1.0
    lr: float = 3e-5
    batch: int = 32
    steps: int = 300
    checkpoints: tuple[int, ...] = (10, 30, 100, 300)
    grad_clip: float = 1.0
    seed: int = 0


@dataclass
class ArmBConfig:
    lr: float = 1e-5
    batch: int = 16
    steps: int = 300
    checkpoints: tuple[int, ...] = (10, 30, 100, 300)
    grad_clip: float = 1.0
    seed: int = 0


@dataclass
class FinetuneResult:
    deltas: dict[int, dict[str, Tensor]]  # step -> module -> delta-W (fp32, CPU)
    log: list[dict[str, Any]] = field(default_factory=list)
    evals: dict[int, dict[str, float]] = field(default_factory=dict)


def arm_a_loss(net: nn.Module, ref: nn.Module, toks: Tensor, cfg: ArmAConfig, half: int) -> tuple[Tensor, dict[str, float]]:
    logits = logits_of(net, toks)  # (b, L, V)
    with torch.no_grad():
        ref_logits = logits_of(ref, toks)
    target = torch.full((toks.shape[0],), cfg.o_token, device=toks.device)
    ce = F.cross_entropy(logits[:, half], target)
    kl = kl_per_position(ref_logits, logits)  # (b, L), KL(ref || ft)
    keep = torch.ones(toks.shape[1], dtype=torch.bool, device=toks.device)
    keep[half] = False
    kl_other = kl[:, keep].mean()
    loss = ce + cfg.kl_lambda * kl_other
    p_o = torch.softmax(logits[:, half], -1)[:, cfg.o_token].mean()
    return loss, {"ce_o": float(ce), "kl_other": float(kl_other), "p_o": float(p_o)}


@torch.no_grad()
def eval_arm_a(net: nn.Module, ref: nn.Module, w: Windows, o_token: int, device: str, batch: int = 64) -> dict[str, float]:
    p_o, kl_other, ce_o, n = 0.0, 0.0, 0.0, 0
    for s in range(0, len(w), batch):
        toks = w.tokens[s : s + batch].to(device)
        lg, rl = logits_of(net, toks), logits_of(ref, toks)
        p = torch.softmax(lg[:, w.half], -1)[:, o_token]
        kl = kl_per_position(rl, lg)
        keep = torch.ones(toks.shape[1], dtype=torch.bool, device=device)
        keep[w.half] = False
        p_o += float(p.sum())
        ce_o += float(-torch.log(p.clamp_min(1e-30)).sum())
        kl_other += float(kl[:, keep].mean(1).sum())
        n += toks.shape[0]
    return {"p_o": p_o / n, "ce_o": ce_o / n, "kl_other": kl_other / n, "n": n}


@torch.no_grad()
def eval_arm_b(net: nn.Module, ref: nn.Module, rows: Tensor, device: str, batch: int = 8) -> dict[str, float]:
    ce, kl, n = 0.0, 0.0, 0
    for s in range(0, rows.shape[0], batch):
        toks = rows[s : s + batch].to(device)
        lg, rl = logits_of(net, toks), logits_of(ref, toks)
        ce += float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), toks[:, 1:].reshape(-1), reduction="sum")) / (toks.shape[1] - 1)
        kl += float(kl_per_position(rl, lg).mean(1).sum())
        n += toks.shape[0]
    return {"ce": ce / n, "kl_to_ref": kl / n, "n": n}


def _run(model: Any, params: dict[str, nn.Parameter], lr: float, grad_clip: float, steps: int,
         checkpoints: tuple[int, ...], loss_fn: Any, eval_fn: Any, log: Any) -> FinetuneResult:
    W0 = target_weights(model)
    for p in params.values():
        p.requires_grad_(True)
    opt = torch.optim.Adam(list(params.values()), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    res = FinetuneResult(deltas={})
    t0 = time.time()
    try:
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            loss, info = loss_fn(step)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(list(params.values()), grad_clip)
            opt.step()
            res.log.append({"step": step, "loss": float(loss), "grad_norm": float(gn), **info})
            if step % 10 == 0 or step == 1:
                log(f"[ft] step {step}: loss {float(loss):.4f} " + " ".join(f"{k} {v:.4f}" for k, v in info.items()))
            if step in checkpoints:
                with torch.no_grad():
                    d = {k: (p.detach().float() - W0[k]).cpu() for k, p in params.items()}
                res.deltas[step] = d
                ev = eval_fn()
                ev["rel_norm_max"] = max(relative_norms(d, {k: w.cpu() for k, w in W0.items()}).values())
                ev["elapsed_s"] = time.time() - t0
                res.evals[step] = ev
                log(f"[ft] checkpoint {step}: " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in ev.items()))
    finally:
        for p in params.values():
            p.requires_grad_(False)
        set_target_weights(model, W0)  # the model is W0 again
    return res


def finetune_arm_a(model: Any, train: Windows, heldout: Windows, cfg: ArmAConfig, device: str, log: Any = print) -> FinetuneResult:
    ref = copy.deepcopy(model.target_model).eval().requires_grad_(False)
    params = decomposed_params(model)
    gen = torch.Generator().manual_seed(cfg.seed)
    net = model.target_model

    def loss_fn(step: int) -> tuple[Tensor, dict[str, float]]:
        idx = torch.randint(0, len(train), (cfg.batch,), generator=gen)
        toks = train.tokens[idx].to(device)
        return arm_a_loss(net, ref, toks, cfg, train.half)

    def eval_fn() -> dict[str, float]:
        return eval_arm_a(net, ref, heldout, cfg.o_token, device)

    return _run(model, params, cfg.lr, cfg.grad_clip, cfg.steps, cfg.checkpoints, loss_fn, eval_fn, log)


def finetune_arm_b(model: Any, train_rows: Tensor, heldout_rows: Tensor, cfg: ArmBConfig, device: str, log: Any = print) -> FinetuneResult:
    ref = copy.deepcopy(model.target_model).eval().requires_grad_(False)
    params = decomposed_params(model)
    net = model.target_model
    n_batches = train_rows.shape[0] // cfg.batch
    assert n_batches >= cfg.steps, f"need {cfg.steps * cfg.batch} fresh train rows, have {train_rows.shape[0]}"

    def loss_fn(step: int) -> tuple[Tensor, dict[str, float]]:
        toks = train_rows[(step - 1) * cfg.batch : step * cfg.batch].to(device)
        lg = logits_of(net, toks)
        ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), toks[:, 1:].reshape(-1))
        return ce, {"ce": float(ce)}

    def eval_fn() -> dict[str, float]:
        return eval_arm_b(net, ref, heldout_rows, device)

    return _run(model, params, cfg.lr, cfg.grad_clip, cfg.steps, cfg.checkpoints, loss_fn, eval_fn, log)
