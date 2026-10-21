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

from commute.data import Windows, target_tokens
from commute.vpd import (
    kl_per_position,
    module_keys,
    set_target_weights,
    target_linear,
    target_weights,
)


def decomposed_params(model: Any) -> dict[str, Tensor]:
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
    o_token: int | Tensor  # the target: one token, or a (V,) lookup table read at the token at `half` (data.target_tokens)
    kl_lambda: float = 1.0
    lr: float = 3e-5
    batch: int = 32
    steps: int = 300
    checkpoints: tuple[int, ...] = (10, 30, 100, 300)
    grad_clip: float = 1.0
    seed: int = 0
    optimizer: str = "adam"  # "adam" (round 1) or "sgd" (plain, no momentum: delta-W is a sum of clipped gradients)


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
    factors: dict[int, dict[str, tuple[Tensor, Tensor]]] = field(default_factory=dict)  # LoRA only: step -> module -> (B, A), CPU


def arm_a_loss(net: nn.Module, ref: nn.Module, toks: Tensor, cfg: ArmAConfig, half: int) -> tuple[Tensor, dict[str, float]]:
    logits = logits_of(net, toks)  # (b, L, V)
    with torch.no_grad():
        ref_logits = logits_of(ref, toks)
    target = target_tokens(cfg.o_token, toks, half)
    ce = F.cross_entropy(logits[:, half], target)
    kl = kl_per_position(ref_logits, logits)  # (b, L), KL(ref || ft)
    keep = torch.ones(toks.shape[1], dtype=torch.bool, device=toks.device)
    keep[half] = False
    kl_other = kl[:, keep].mean()
    loss = ce + cfg.kl_lambda * kl_other
    p_o = torch.softmax(logits[:, half], -1).gather(1, target[:, None]).mean()
    return loss, {"ce_o": float(ce), "kl_other": float(kl_other), "p_o": float(p_o)}


@torch.no_grad()
def eval_arm_a(net: nn.Module, ref: nn.Module, w: Windows, o_token: int | Tensor, device: str, batch: int = 64) -> dict[str, float]:
    p_o, kl_other, ce_o, n = 0.0, 0.0, 0.0, 0
    for s in range(0, len(w), batch):
        toks = w.tokens[s : s + batch].to(device)
        lg, rl = logits_of(net, toks), logits_of(ref, toks)
        p = torch.softmax(lg[:, w.half], -1).gather(1, target_tokens(o_token, toks, w.half)[:, None]).squeeze(1)
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


def _run(model: Any, params: dict[str, Tensor], lr: float, grad_clip: float, steps: int,
         checkpoints: tuple[int, ...], loss_fn: Any, eval_fn: Any, log: Any, optimizer: str = "adam",
         on_step: Any = None) -> FinetuneResult:
    """The training loop. `on_step(step, params)`, if given, is called after every optimizer step, before that step's
    checkpoint, with the live parameters (step_split reads every step's weights there)."""
    W0 = target_weights(model)
    for p in params.values():
        p.requires_grad_(True)
    if optimizer == "adam":
        opt = torch.optim.Adam(list(params.values()), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    else:
        assert optimizer == "sgd", optimizer
        opt = torch.optim.SGD(list(params.values()), lr=lr, momentum=0.0, weight_decay=0.0)
    res = FinetuneResult(deltas={})
    t0 = time.time()
    try:
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            loss, info = loss_fn(step)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(list(params.values()), grad_clip)
            opt.step()
            if on_step is not None:
                on_step(step, params)
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


def finetune_arm_a(model: Any, train: Windows, heldout: Windows, cfg: ArmAConfig, device: str, log: Any = print,
                   on_step: Any = None) -> FinetuneResult:
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

    return _run(model, params, cfg.lr, cfg.grad_clip, cfg.steps, cfg.checkpoints, loss_fn, eval_fn, log, cfg.optimizer, on_step)


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


# ----------------------------------------------------------------------------- LoRA (arm A data)


@dataclass
class LoraConfig:
    modules: tuple[str, ...]
    rank: int
    lr: float = 1e-3
    steps: int = 100
    checkpoints: tuple[int, ...] = (1, 2, 3, 5, 10, 30, 100)
    seed: int = 0


def lora_factors(linears: dict[str, nn.Linear], rank: int, seed: int) -> dict[str, tuple[nn.Parameter, nn.Parameter]]:
    """(B (d_out, r) = 0, A (r, d_in) ~ N(0, 1/d_in)) per module: the standard init, delta-W = B A = 0 at start."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k, lin in linears.items():
        d_out, d_in = lin.weight.shape
        A = torch.randn(rank, d_in, generator=gen) / d_in**0.5
        out[k] = (nn.Parameter(torch.zeros(d_out, rank, device=lin.weight.device)), nn.Parameter(A.to(lin.weight.device)))
    return out


def lora_hooks(linears: dict[str, nn.Linear], factors: dict[str, tuple[nn.Parameter, nn.Parameter]]) -> list[Any]:
    """Forward hooks making each linear compute (W0 + B A) x; returns the handles."""
    handles = []
    for k, lin in linears.items():
        B, A = factors[k]

        def hook(_m: nn.Module, args: tuple[Any, ...], out: Tensor, B: Tensor = B, A: Tensor = A) -> Tensor:
            x = args[0]
            return out + F.linear(F.linear(x.to(A.dtype), A), B).to(out.dtype)

        handles.append(lin.register_forward_hook(hook))
    return handles


def lora_delta(factors: dict[str, tuple[nn.Parameter, nn.Parameter]], all_keys: list[str], W0: dict[str, Tensor]) -> dict[str, Tensor]:
    """delta-W for all 24 matrices: B A on the LoRA modules, zero elsewhere (fp32, CPU)."""
    return {k: (factors[k][0].detach().float() @ factors[k][1].detach().float()).cpu() if k in factors
            else torch.zeros_like(W0[k]).cpu() for k in all_keys}


def saved_lora_updates(path: Any, all_keys: list[str], W0: dict[str, Tensor], steps: tuple[int, ...]) -> dict[int, dict[str, Tensor]]:
    """B A per matrix (zeros where the LoRA has no factors) at `steps` from a saved {step: {matrix: (B, A)}} file, as
    lora_split's `lora_updates` (2026-10-21, moved here for split_comparison)."""
    factors = torch.load(path, weights_only=False)
    return {s: lora_delta(factors[s], all_keys, W0) for s in steps}


def finetune_lora_arm_a(model: Any, train: Windows, heldout: Windows, a_cfg: ArmAConfig, cfg: LoraConfig, device: str,
                        log: Any = print) -> FinetuneResult:
    """Arm A's loss, batch, clip and sampling, with a LoRA on cfg.modules; the target weights never change."""
    ref = copy.deepcopy(model.target_model).eval().requires_grad_(False)
    net = model.target_model
    W0 = target_weights(model)
    linears = {k: target_linear(model, k) for k in cfg.modules}
    factors = lora_factors(linears, cfg.rank, cfg.seed)
    params = [p for f in factors.values() for p in f]
    opt = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)
    gen = torch.Generator().manual_seed(a_cfg.seed)
    res = FinetuneResult(deltas={})
    handles = lora_hooks(linears, factors)
    t0 = time.time()
    try:
        for step in range(1, cfg.steps + 1):
            opt.zero_grad(set_to_none=True)
            idx = torch.randint(0, len(train), (a_cfg.batch,), generator=gen)
            loss, info = arm_a_loss(net, ref, train.tokens[idx].to(device), a_cfg, train.half)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, a_cfg.grad_clip)
            opt.step()
            res.log.append({"step": step, "loss": float(loss), "grad_norm": float(gn), **info})
            if step in cfg.checkpoints:
                d = lora_delta(factors, module_keys(model), W0)
                res.deltas[step] = d
                res.factors[step] = {k: (B.detach().float().cpu().clone(), A.detach().float().cpu().clone()) for k, (B, A) in factors.items()}
                ev = eval_arm_a(net, ref, heldout, a_cfg.o_token, device)
                ev["rel_norm_max"] = max(relative_norms(d, {k: w.cpu() for k, w in W0.items()}).values())
                ev["elapsed_s"] = time.time() - t0
                res.evals[step] = ev
                log(f"[lora r{cfg.rank}] checkpoint {step}: loss {float(loss):.4f} "
                    + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in ev.items()))
    finally:
        for h in handles:
            h.remove()
    assert all(torch.equal(target_linear(model, k).weight.float(), W0[k]) for k in cfg.modules)
    return res
