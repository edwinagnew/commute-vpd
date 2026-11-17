"""A BatchTopK crosscoder between the base model and one fine-tune at one residual site, latent scaling, and the
activation-change reading (E198, cell P2.31; plan reports/plans/2026-11-17_plan_p2-31-crosscoder.md).

Recipe: Minder et al. 2025, "Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning" (arXiv 2504.02922
v4): BatchTopK selection on f_j(x) (|d_j^base| + |d_j^ft|) (eq. 9), the inference threshold theta (eqs. 13-14), the aux
loss on dead latents (k_aux 512, alpha 1/32, App. B.2), paired init with the decoder the encoder's transpose (App. K),
relative decoder norm and its bands (Table 4), latent scaling with the fine-tune decoder and its two ratios (eqs. 4,
33-36), and the chat-specific test nu_r < 0.5, nu_eps < 0.2 (glossary). The method is Lindsey et al. 2024, "Sparse
Crosscoders for Cross-Layer Features and Model Diffing" (middle-layer residual, relative decoder norms). Departures
are listed in the plan. Model index 0 is the base, 1 the fine-tune, in every (2, ...) tensor here.

The reading, per state, on held-out task windows: the change D(x) = h_ft(x) - h_base(x) is written exactly as
  sum_j f_j(x) (d_j^ft - d_j^base) + (b_ft - b_base) + (eps_ft(x) - eps_base(x)),
and each latent's carried part is s_j = sum_x f_j(x) <d_j^ft - d_j^base, D(x)> / sum_x |D(x)|^2, so the groups' parts,
the bias part and the error part sum to one. `d` = shared part minus fine-tune-specific part.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Cfg:
    d: int  # residual width
    expansion: int  # latents = expansion * d
    k: int  # BatchTopK: k * batch selected per batch
    lr: float
    batch: int  # tokens per step
    k_aux: int
    alpha_aux: float
    dead_tokens: int  # a latent unfired for this many tokens is dead (aux loss)
    init_norm: float
    theta_last: float  # theta = mean smallest selected value over this last fraction of steps
    alive_min: int  # firings on the held-out mix for a latent to count as alive
    shared: tuple[float, float]  # relative decoder norm bands (Minder Table 4)
    ft_only: float
    base_only: float
    nu_r: float  # latent scaling: fine-tune-specific when nu_r < this and nu_eps < nu_eps
    nu_eps: float

    def __post_init__(self) -> None:
        assert self.k * 10 <= self.latents and self.k_aux <= self.latents and 0 < self.theta_last <= 1, self
        assert 0 <= self.base_only < self.shared[0] < 0.5 < self.shared[1] < self.ft_only <= 1, self

    @property
    def latents(self) -> int:
        return self.expansion * self.d


class Crosscoder(nn.Module):
    def __init__(self, cfg: Cfg, seed: int, device: str) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(cfg.d, cfg.latents, generator=g)
        W = W / W.norm(dim=0, keepdim=True) * cfg.init_norm  # each latent's direction at norm init_norm, both models
        self.W_enc = nn.Parameter(torch.stack([W, W.clone()]).to(device))  # (2, d, D), paired init
        self.b_enc = nn.Parameter(torch.zeros(cfg.latents, device=device))
        self.W_dec = nn.Parameter(self.W_enc.detach().transpose(1, 2).clone())  # (2, D, d), the encoder's transpose
        self.b_dec = nn.Parameter(torch.zeros(2, cfg.d, device=device))
        self.theta = math.nan
        self.cfg = cfg

    def pre(self, h: Tensor) -> Tensor:
        return torch.relu(torch.einsum("mnd,mdk->nk", h, self.W_enc) + self.b_enc)

    def value(self, f: Tensor) -> Tensor:
        return f * self.W_dec.norm(dim=2).sum(0)

    def encode_train(self, h: Tensor) -> tuple[Tensor, float]:
        """Training activations (eq. 9): the batch's top n*k scaled values kept; also the smallest kept value."""
        f = self.pre(h)
        with torch.no_grad():
            v = self.value(f).flatten()
        top = torch.topk(v, h.shape[1] * self.cfg.k, sorted=False)
        mask = torch.zeros_like(v).scatter_(0, top.indices, 1.0).view_as(f)
        return f * mask, float(top.values.min())

    def encode(self, h: Tensor) -> Tensor:
        """Inference activations (eq. 13): kept where the scaled value exceeds theta."""
        f = self.pre(h)
        return f * (self.value(f) > self.theta)

    def decode(self, f: Tensor, bias: bool = True) -> Tensor:
        out = torch.einsum("nk,mkd->mnd", f, self.W_dec)
        return out + self.b_dec[:, None] if bias else out


def train(cc: Crosscoder, batches: Iterator[Tensor], n_steps: int, log: Callable[[str], None]) -> dict[str, Any]:
    """Adam at the recipe's rate on 1/2 |eps_base|^2 + 1/2 |eps_ft|^2 + alpha * aux, one pass over `batches`."""
    cfg = cc.cfg
    opt = torch.optim.Adam(cc.parameters(), lr=cfg.lr)
    since = torch.zeros(cfg.latents, device=cc.W_dec.device)
    mins: list[float] = []
    curve: list[dict[str, float]] = []
    step = 0
    for step, h in enumerate(batches):
        f, vmin = cc.encode_train(h)
        err = h - cc.decode(f)
        loss = 0.5 * err.pow(2).sum(-1).sum(0).mean()
        since += h.shape[1]
        since[(f > 0).any(0)] = 0
        dead = since > cfg.dead_tokens
        aux = torch.zeros((), device=h.device)
        if bool(dead.any()):
            pre = cc.pre(h) * dead
            top = torch.topk(pre, min(cfg.k_aux, int(dead.sum())), dim=1)
            fa = torch.zeros_like(pre).scatter_(1, top.indices, top.values)
            aux = (err.detach() - cc.decode(fa, bias=False)).pow(2).sum(-1).sum(0).mean()
        opt.zero_grad(set_to_none=True)
        (loss + cfg.alpha_aux * aux).backward()
        opt.step()
        if step >= n_steps * (1 - cfg.theta_last):
            mins.append(vmin)
        if step % 500 == 0 or step == n_steps - 1:
            e = err.detach()
            fve = 1 - float(e.pow(2).sum()) / float((h - h.mean(1, keepdim=True)).pow(2).sum())
            rec = {"step": step, "loss": loss.item(), "aux": aux.item(), "fve_batch": fve, "dead": dead.float().mean().item()}
            curve.append(rec)
            log(f"  step {step}/{n_steps} loss {rec['loss']:.4g} fve_batch {fve:.4f} dead {rec['dead']:.3f}")
    assert mins, "no steps in the theta window"
    cc.theta = sum(mins) / len(mins)
    return {"steps": step + 1, "theta": cc.theta, "curve": curve}


@torch.no_grad()
def chunks(cc: Crosscoder, h: Tensor, size: int = 4096) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """(h, f, reconstruction) over token chunks of a (2, M, d) tensor."""
    for i in range(0, h.shape[1], size):
        x = h[:, i : i + size]
        f = cc.encode(x)
        yield x, f, cc.decode(f)


@torch.no_grad()
def fve(cc: Crosscoder, h: Tensor) -> dict[str, Any]:
    """Fraction of variance explained per model, firing counts per latent and the mean L0, on a held-out (2, M, d)."""
    err = torch.zeros(2, dtype=torch.float64)
    fired = torch.zeros(cc.cfg.latents, device=h.device)
    l0 = 0.0
    for x, f, rec in chunks(cc, h):
        err += (x - rec).double().pow(2).sum((1, 2)).cpu()
        fired += (f > 0).sum(0)
        l0 += float((f > 0).sum())
    var = (h - h.mean(1, keepdim=True)).double().pow(2).sum((1, 2)).cpu()
    out = 1 - err / var
    return {"fve_base": float(out[0]), "fve_ft": float(out[1]), "l0": l0 / h.shape[1], "fired": fired.cpu()}


def relative_norm(cc: Crosscoder) -> Tensor:
    """Delta_norm in [0, 1] (Lindsey et al.; Minder eq. 3): 1 when the base decoder is zero, 0.5 when the norms agree."""
    n = cc.W_dec.detach().norm(dim=2)
    return (0.5 * (1 + (n[1] - n[0]) / n.max(0).values.clamp_min(1e-12))).cpu()


@torch.no_grad()
def latent_scaling(cc: Crosscoder, h: Tensor, J: Tensor) -> dict[str, Tensor]:
    """Minder eqs. 4, 33-36 for latents J, with the fine-tune decoder d = d_j^ft: beta = sum_x f <y, d> / (sum f^2 |d|^2)."""
    dJ, dbJ = cc.W_dec[1, J], cc.W_dec[0, J]  # (|J|, d)
    num = {k: torch.zeros(len(J), device=h.device, dtype=torch.float64) for k in ("eps_base", "eps_ft", "r_base", "r_ft")}
    den = torch.zeros(len(J), device=h.device, dtype=torch.float64)
    for x, f, rec in chunks(cc, h):
        fJ, eps = f[:, J], x - rec
        proj = {"eps_base": eps[0] @ dJ.T + fJ * (dbJ * dJ).sum(1), "eps_ft": eps[1] @ dJ.T + fJ * dJ.pow(2).sum(1),
                "r_base": rec[0] @ dJ.T, "r_ft": rec[1] @ dJ.T}  # fmt: skip
        for k, p in proj.items():
            num[k] += (fJ * p).sum(0).double()
        den += (fJ.pow(2).sum(0) * dJ.pow(2).sum(1)).double()
    beta = {k: (v / den).cpu() for k, v in num.items()}
    return beta | {"nu_eps": beta["eps_base"] / beta["eps_ft"], "nu_r": beta["r_base"] / beta["r_ft"]}


def groups(cc: Crosscoder, h_val: Tensor) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Alive latents by relative norm into shared / ft_specific / ft_artefact / base_only / other; dead latents apart."""
    cfg, rn = cc.cfg, relative_norm(cc)
    ev = fve(cc, h_val)
    alive = ev.pop("fired") >= cfg.alive_min
    ft_only = alive & (rn >= cfg.ft_only)
    J = torch.nonzero(ft_only).squeeze(1)
    ls = latent_scaling(cc, h_val, J.to(h_val.device)) if len(J) else {}
    specific = torch.zeros_like(alive)
    if len(J):  # negative beta_base or a non-positive beta_ft: not specific (Minder App. E.2 filters negative beta_base)
        ok = (ls["nu_r"] < cfg.nu_r) & (ls["nu_eps"] < cfg.nu_eps) & (ls["eps_base"] >= 0) & (ls["r_base"] >= 0) & (ls["eps_ft"] > 0) & (ls["r_ft"] > 0)
        specific[J[ok]] = True
    g = {"shared": alive & (rn >= cfg.shared[0]) & (rn <= cfg.shared[1]), "ft_specific": specific, "ft_artefact": ft_only & ~specific,
         "base_only": alive & (rn <= cfg.base_only)}  # fmt: skip
    g["other"] = alive & ~(g["shared"] | ft_only | g["base_only"])
    g["dead"] = ~alive
    info = ev | {"counts": {k: int(v.sum()) for k, v in g.items()}, "relative_norm": rn, "scaled": J, "scaling": ls}
    return g, info


@torch.no_grad()
def change_parts(cc: Crosscoder, h: Tensor, g: dict[str, Tensor]) -> dict[str, Any]:
    """The change reading on a held-out (2, M, d): each group's part, the bias and error parts, d = shared - specific."""
    dD = (cc.W_dec[1] - cc.W_dec[0]).detach()  # (D, d)
    s = torch.zeros(cc.cfg.latents, device=h.device, dtype=torch.float64)
    total, bias = 0.0, 0.0
    for x, f, _ in chunks(cc, h):
        delta = x[1] - x[0]
        s += (f * (delta @ dD.T)).sum(0).double()
        total += float(delta.double().pow(2).sum())
        bias += float((delta.double() @ (cc.b_dec[1] - cc.b_dec[0]).double()).sum())
    if total <= 0:  # no change at all (the base-against-itself control): every part undefined
        return {"change_energy": 0.0, "parts": {k: math.nan for k in [*g, "bias", "error"]}, "d": math.nan, "s": s.cpu()}
    s = s.cpu() / total
    parts = {k: float(s[m].sum()) for k, m in g.items()}
    parts["bias"] = bias / total
    parts["error"] = 1 - float(s.sum()) - parts["bias"]
    return {"change_energy": total / h.shape[1], "parts": parts, "d": parts["shared"] - parts["ft_specific"], "s": s}
