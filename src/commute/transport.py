"""The transport ladder: write delta-W as changes to existing subcomponents, Gamma and Delta fixed.

Every model variant is the target model at W0 plus an `extra` term per decomposed matrix, applied
by forward hooks: out = W0 x + extra(x). The fine-tuned model is extra(x) = delta-W x; a transport
is extra(x) = (sum of changed subcomponents) x. Comparing hooked forwards gives KL(f_ft || f_T).

T0 scale-only: delta-W ~ sum_c alpha_c B_c over an index set (the alive subcomponents). The
Frobenius fit is a linear least-squares with Gram matrix (U U^T) o (V^T V); the functional fit
optimises alpha against KL on batches.
T1 k-sparse write edits: delta-U_c free on a set S of k subcomponents, V_c fixed:
extra(x) = sum_{c in S} delta-U_c (V_c . x). Read edits are the mirror image. A random S with the
same k is the null.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from commute.vpd import kl_per_position, module_keys, target_linear, uv

Extra = dict[str, Callable[[Tensor], Tensor]]


# ----------------------------------------------------------------------------- hooked forwards


@contextlib.contextmanager
def extras(model: Any, extra: Extra):
    handles = []
    for name, fn in extra.items():
        def hook(_m: nn.Module, args: tuple[Any, ...], out: Tensor, fn: Callable[[Tensor], Tensor] = fn) -> Tensor:
            return out + fn(args[0]).to(out.dtype)
        handles.append(target_linear(model, name).register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def forward(model: Any, batch: Tensor, extra: Extra | None = None, grad: bool = False) -> Tensor:
    """Logits (float32) of the target model at W0 plus `extra`."""
    ctx = contextlib.nullcontext() if grad else torch.no_grad()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16, enabled=batch.is_cuda), extras(model, extra or {}):
        return model(batch).float()


def delta_extra(dW: dict[str, Tensor], device: str) -> Extra:
    d = {k: v.to(device) for k, v in dW.items()}
    return {k: (lambda x, w=d[k]: F.linear(x, w.to(x.dtype))) for k in d}


# ----------------------------------------------------------------------------- T0: Frobenius


@dataclass
class FrobeniusFit:
    alpha: Tensor  # (k,) coefficients on the index set
    idx: Tensor  # (k,) subcomponent indices
    fraction: float  # ||proj||^2 / ||dW||^2
    contribution: Tensor  # (k,) |alpha_c| ||U_c|| ||V_c||


def frobenius_fit(dW: Tensor, U: Tensor, V: Tensor, idx: Tensor, ridge: float = 1e-8) -> FrobeniusFit:
    """Least squares of dW (d_out, d_in) on {B_c = U_c V_c^T : c in idx}, in float64."""
    Uk = U[idx].double()  # (k, d_out)
    Vk = V[:, idx].double()  # (d_in, k)
    d = dW.double()
    G = (Uk @ Uk.T) * (Vk.T @ Vk)
    b = torch.einsum("ko,oi,ik->k", Uk, d, Vk)
    G = G + ridge * G.diagonal().mean() * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    alpha = torch.linalg.solve(G, b)
    fit_norm2 = float(alpha @ b)
    total = float((d * d).sum())
    contribution = alpha.abs() * Uk.norm(dim=1) * Vk.norm(dim=0)
    return FrobeniusFit(alpha=alpha.float(), idx=idx, fraction=fit_norm2 / max(total, 1e-30), contribution=contribution.float())


def frobenius_fit_all(model: Any, dW: dict[str, Tensor], index_sets: dict[str, Tensor], device: str) -> dict[str, FrobeniusFit]:
    out = {}
    for k in module_keys(model):
        U, V = uv(model, k)
        out[k] = frobenius_fit(dW[k].to(device), U.float(), V.float(), index_sets[k].to(device))
    return out


def pooled_fraction(fits: dict[str, FrobeniusFit], dW: dict[str, Tensor]) -> float:
    """Frobenius fraction pooled over matrices, weighted by ||dW_m||^2."""
    num = sum(f.fraction * float(dW[k].norm() ** 2) for k, f in fits.items())
    den = sum(float(dW[k].norm() ** 2) for k in fits)
    return num / den


# ----------------------------------------------------------------------------- parametrised extras


def scale_extra(model: Any, alpha: dict[str, Tensor], idx: dict[str, Tensor]) -> Extra:
    """extra(x) = sum_{c in idx} alpha_c U_c (V_c . x)."""
    ex: Extra = {}
    for k in alpha:
        U, V = uv(model, k)
        Uk, Vk = U[idx[k]], V[:, idx[k]]
        ex[k] = lambda x, Uk=Uk, Vk=Vk, a=alpha[k]: ((x.to(Vk.dtype) @ Vk) * a.to(Vk.dtype)) @ Uk
    return ex


def write_edit_extra(model: Any, dU: dict[str, Tensor], idx: dict[str, Tensor]) -> Extra:
    """extra(x) = sum_{c in idx} dU_c (V_c . x); dU[k] has shape (k, d_out)."""
    ex: Extra = {}
    for k in dU:
        _, V = uv(model, k)
        Vk = V[:, idx[k]]
        ex[k] = lambda x, Vk=Vk, d=dU[k]: (x.to(Vk.dtype) @ Vk) @ d.to(Vk.dtype)
    return ex


def read_edit_extra(model: Any, dV: dict[str, Tensor], idx: dict[str, Tensor]) -> Extra:
    """extra(x) = sum_{c in idx} U_c (dV_c . x); dV[k] has shape (d_in, k)."""
    ex: Extra = {}
    for k in dV:
        U, _ = uv(model, k)
        Uk = U[idx[k]]
        ex[k] = lambda x, Uk=Uk, d=dV[k]: (x.to(Uk.dtype) @ d.to(Uk.dtype)) @ Uk
    return ex


# ----------------------------------------------------------------------------- functional fits


def kl_to_reference(model: Any, ref_extra: Extra, extra: Extra, batch: Tensor, grad: bool = False,
                    positions: Tensor | None = None) -> Tensor:
    """Mean KL(f_ref || f_extra) over the batch, at `positions` (bool over seq) or everywhere."""
    ref = forward(model, batch, ref_extra)
    out = forward(model, batch, extra, grad=grad)
    kl = kl_per_position(ref, out)
    return kl[:, positions].mean() if positions is not None else kl.mean()


def functional_fit(model: Any, ref_extra: Extra, make_extra: Callable[[], Extra], params: list[Tensor],
                   next_batch: Callable[[], Tensor], steps: int, lr: float, log: Any = print,
                   positions: Tensor | None = None, tag: str = "") -> list[float]:
    """Adam on `params` to minimise KL(f_ref || f_variant) on streaming batches."""
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = kl_to_reference(model, ref_extra, make_extra(), next_batch(), grad=True, positions=positions)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if step % 25 == 0 or step == 1:
            log(f"[fit {tag}] step {step}: kl {float(loss):.5f} ({time.time() - t0:.0f} s)")
    for p in params:
        p.requires_grad_(False)
    return losses


@dataclass
class Capture:
    kl_ft_vs_w0: float  # KL(f_ft || f_W0): the size of the fine-tune's effect
    kl_ft_vs_t: float  # KL(f_ft || f_T): what the transport leaves unexplained
    capture: float  # 1 - kl_ft_vs_t / kl_ft_vs_w0


@torch.no_grad()
def capture(model: Any, ref_extra: Extra, extra: Extra, batches: list[Tensor], positions: Tensor | None = None) -> Capture:
    a = b = 0.0
    n = 0
    for batch in batches:
        ref = forward(model, batch, ref_extra)
        base = forward(model, batch, {})
        out = forward(model, batch, extra)
        k0 = kl_per_position(ref, base)
        kt = kl_per_position(ref, out)
        if positions is not None:
            k0, kt = k0[:, positions], kt[:, positions]
        a += float(k0.sum())
        b += float(kt.sum())
        n += k0.numel()
    a, b = a / n, b / n
    return Capture(kl_ft_vs_w0=a, kl_ft_vs_t=b, capture=1.0 - b / max(a, 1e-30))


def top_k_by_contribution(fits: dict[str, FrobeniusFit], k: int) -> dict[str, Tensor]:
    """The k subcomponents with the largest Frobenius contribution across all matrices."""
    rows = [(float(c), name, int(i)) for name, f in fits.items() for c, i in zip(f.contribution.tolist(), f.idx.tolist())]
    rows.sort(reverse=True)
    chosen: dict[str, list[int]] = {}
    for _, name, i in rows[:k]:
        chosen.setdefault(name, []).append(i)
    return {name: torch.tensor(v) for name, v in chosen.items()}


def random_matched(sets: dict[str, Tensor], alive: dict[str, Tensor], seed: int) -> dict[str, Tensor]:
    """Same number of subcomponents per matrix as `sets`, drawn uniformly from the alive ones."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for name, idx in sets.items():
        pool = torch.nonzero(alive[name].cpu(), as_tuple=True)[0]
        out[name] = pool[torch.randperm(pool.numel(), generator=gen)[: idx.numel()]]
    return out
