"""The Delta-Crosscoder (Kassem, Jiralerspong, Rostamzadeh and Farnadi 2026, arXiv 2603.04426 v1), our port of the
paper's text: no code was released (plan reports/plans/2026-11-23_plan_p2-72-delta-crosscoder-spec.md, sections 2-3).
Model index 0 is the base M, 1 the changed model F. Inputs are (B, 2, d) activation pairs on the same tokens, already
multiplied by one scalar fit on M (`scale_of`), with no mean subtraction (spec 3.1).

Forward (spec 3.2): u = (W_e^0 a + W_e^1 b) / 2 + c_e (eq. 2 plus Minder's encoder bias), r = ReLU(u); two separate
batch top-Ks on r itself, K_shared * B entries over the shared block (the first 20% of latents) and K_delta * B over the
non-shared block; x_hat^m = W_d^m z + c_d^m. Every latent has two decoder vectors; the shared ones are untied unless
`tied` (the other reading, spec appendix 2). Loss (spec 3.3), each term a mean over the batch's tokens:
  L = sum_m |x_m - x_hat_m|^2 + aux_alpha * L_aux + lam_delta * |Delta - (W_d^1 - W_d^0)_nonshared z_delta|^2  (eq. 9)
with L_aux Minder's AuxK (`aux_loss`). Adam, linear warm-up then constant, bf16 autocast over fp32 weights (spec 3.4).
The recovery reading's pure rules (candidates, bulk pools, pass, label; spec 3.7) are at the end."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Arch:
    d: int
    expansion: int
    shared_frac: float
    k_shared: int
    k_delta: int
    init_norm: float
    lam_delta: float
    aux_alpha: float
    dead_tokens: int
    target_rms: float

    def __post_init__(self) -> None:
        assert math.isclose(self.latents * self.shared_frac, self.n_shared), self
        assert 0 < self.k_delta < self.k_shared <= self.n_shared and self.k_delta <= self.latents - self.n_shared, self

    @property
    def latents(self) -> int:
        return self.expansion * self.d

    @property
    def n_shared(self) -> int:
        return round(self.latents * self.shared_frac)

    @property
    def aux_k(self) -> int:  # Minder's top_k_aux, activation_dim // 2 (trainers/crosscoder.py 399)
        return self.d // 2


def batch_topk(r: Tensor, k: int) -> Tensor:
    """`r` (B, n) with all but its k * B largest entries over the whole batch set to 0."""
    flat = r.flatten()
    idx = flat.topk(min(k * r.shape[0], flat.numel()), sorted=False).indices
    return torch.zeros_like(flat).index_copy_(0, idx, flat[idx]).view_as(r)


class DXC(nn.Module):
    scale: Tensor

    def __init__(self, a: Arch, paired: bool, tied: bool, seed: int) -> None:
        super().__init__()
        self.a, self.tied, S = a, tied, a.n_shared
        g = torch.Generator().manual_seed(seed)

        def dirs() -> Tensor:
            v = torch.randn(a.latents, a.d, generator=g)
            return v / v.norm(dim=1, keepdim=True) * a.init_norm

        W0 = dirs()
        W1 = W0.clone() if paired else dirs()
        if tied:
            W1[:S] = W0[:S]
        W = torch.stack([W0, W1])  # (2, D, d)
        self.W_sh = nn.Parameter(W[:1, :S].clone() if tied else W[:, :S].clone())
        self.W_ns = nn.Parameter(W[:, S:].clone())
        self.W_enc = nn.Parameter(W.transpose(1, 2).clone())  # (2, d, D): each encoder starts as its decoder's transpose
        self.b_enc = nn.Parameter(torch.zeros(a.latents))
        self.b_dec = nn.Parameter(torch.zeros(2, a.d))
        self.register_buffer("scale", torch.ones(()))  # raw -> normalised units, fit on M (`scale_of`)

    @property
    def W_dec(self) -> Tensor:  # (2, D, d)
        return torch.cat([self.W_sh.expand(2, -1, -1), self.W_ns], 1)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """(z, r) for normalised x (B, 2, d): r = ReLU(u), z its two batch top-Ks."""
        S = self.a.n_shared
        r = torch.relu(0.5 * (x[:, 0] @ self.W_enc[0] + x[:, 1] @ self.W_enc[1]) + self.b_enc)
        return torch.cat([batch_topk(r[:, :S], self.a.k_shared), batch_topk(r[:, S:], self.a.k_delta)], 1), r

    def decode(self, z: Tensor) -> Tensor:
        return torch.einsum("bD,mDd->bmd", z, self.W_dec) + self.b_dec

    def delta_hat(self, z: Tensor) -> Tensor:
        """Eq. 9's delta prediction: the non-shared latents only, no bias (eq. 7)."""
        return z[:, self.a.n_shared :] @ (self.W_ns[1] - self.W_ns[0])

    def losses(self, x: Tensor, dead: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        z, r = self.encode(x)
        xh = self.decode(z)
        err = (x - xh).float()
        recon = err.pow(2).sum((1, 2)).mean()
        delta = (x[:, 1] - x[:, 0] - self.delta_hat(z)).float().pow(2).sum(1).mean()
        aux = aux_loss(self, err.detach(), r, dead)
        total = recon + self.a.aux_alpha * aux + self.a.lam_delta * delta
        xc = x.float() - x.float().mean(0)
        fve = 1 - err.pow(2).sum((0, 2)) / xc.pow(2).sum((0, 2))
        return total, {"loss": total.detach(), "recon": recon.detach(), "delta": delta.detach(), "aux": aux.detach(),
                       "fve_base": fve[0].detach(), "fve_ft": fve[1].detach(), "fired": (z > 0).any(0)}  # fmt: skip

    @torch.no_grad()
    def rel_norm(self) -> Tensor:
        """Eq. 4 with its sum denominator: R_i = |d_i^0| / (|d_i^0| + |d_i^1|); 0.5 shared, near 0 F-only. Not
        crosscoder.relative_norm, which takes a max denominator. One scalar scales both models, so raw units agree."""
        n = self.W_dec.float().norm(dim=2)
        return (n[0] / (n[0] + n[1]).clamp_min(1e-30)).cpu()


def aux_loss(m: DXC, res: Tensor, r: Tensor, dead: Tensor) -> Tensor:
    """Minder's AuxK (trainers/crosscoder.py 421-480): the residual of both models stacked, detached, reconstructed
    without bias from the aux_k dead latents with the largest rectified pre-activations (both blocks), over the
    residual's variance; 0 when no latent is dead."""
    n = int(dead.sum())
    if n == 0:
        return torch.zeros((), device=r.device)
    idx = torch.where(dead, r.detach(), -math.inf).topk(min(m.a.aux_k, n), dim=1).indices
    za = torch.zeros_like(r).scatter(1, idx, r.gather(1, idx))
    xa = torch.einsum("bD,mDd->bmd", za, m.W_dec).flatten(1).float()
    res = res.flatten(1)
    return ((res - xa).pow(2).sum(1).mean() / (res - res.mean(0)).pow(2).sum(1).mean()).nan_to_num(0.0)


def scale_of(a_raw: Tensor, target_rms: float) -> float:
    """The one scalar (spec 3.1): M's mean squared norm times scale^2 is target_rms^2, no mean subtracted."""
    return target_rms / math.sqrt(float(a_raw.double().pow(2).sum(1).mean()))


class Trainer:
    """Adam at lr, linear warm-up over `warmup` steps then constant; the dead counter of Minder's trainer (tokens since
    a latent last fired, dead at `dead_tokens`)."""

    def __init__(self, m: DXC, lr: float, warmup: int) -> None:
        self.m = m
        self.opt = torch.optim.Adam(m.parameters(), lr=lr)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: min(1.0, (s + 1) / warmup))
        self.since = torch.zeros(m.a.latents, dtype=torch.long, device=m.b_enc.device)

    def step(self, x: Tensor) -> dict[str, float]:
        dead = self.since >= self.m.a.dead_tokens
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            total, st = self.m.losses(x, dead)
        self.opt.zero_grad(set_to_none=True)
        total.backward()
        self.opt.step()
        self.sched.step()
        fired = st.pop("fired")
        self.since += x.shape[0]
        self.since[fired] = 0
        keys = list(st)
        vals = torch.stack([st[k].float() for k in keys]).tolist()
        return dict(zip(keys, vals, strict=True)) | {"dead": float(dead.float().mean())}

    def state(self) -> dict[str, Any]:
        return {"m": self.m.state_dict(), "opt": self.opt.state_dict(), "sched": self.sched.state_dict(), "since": self.since}

    def load(self, s: dict[str, Any]) -> None:
        self.m.load_state_dict(s["m"])
        self.opt.load_state_dict(s["opt"])
        self.sched.load_state_dict(s["sched"])
        self.since.copy_(s["since"])


@torch.no_grad()
def read_codes(m: DXC, chunks: Iterable[Tensor], batch: int, seed: int, keep: Tensor | None = None) -> dict[str, Any]:
    """The inference rule (spec 3.2): each raw (n, 2, d) chunk shuffled and encoded `batch` tokens at a time with the
    same two batch top-Ks, no threshold. Per-model FVE (raw units, against each chunk set's mean), firing counts, mean
    L0, and z of the latents `keep` in the original token order (None: not kept)."""
    D, dev = m.a.latents, m.b_enc.device
    err, sxx = torch.zeros(2, dtype=torch.float64, device=dev), torch.zeros(2, dtype=torch.float64, device=dev)
    sx = torch.zeros(2, m.a.d, dtype=torch.float64, device=dev)
    fired, n, l0, kept = torch.zeros(D, dtype=torch.long, device=dev), 0, 0.0, []
    for i, h in enumerate(chunks):
        perm = torch.randperm(len(h), generator=torch.Generator().manual_seed(seed + i)).to(h.device)
        zs = torch.empty(len(h), 0 if keep is None else len(keep), device=dev)
        for j in range(0, len(h), batch):
            p = perm[j : j + batch]
            x = h[p].to(dev).float()
            with torch.autocast(x.device.type, dtype=torch.bfloat16):
                z, _ = m.encode(x * m.scale)
                xh = m.decode(z).float() / m.scale
            err += (x - xh).double().pow(2).sum((0, 2))
            sx += x.double().sum(0)
            sxx += x.double().pow(2).sum((0, 2))
            fired += (z > 0).sum(0)
            l0 += float((z > 0).sum())
            if keep is not None:
                zs[p] = z[:, keep.to(dev)].float()
        n += len(h)
        if keep is not None:
            kept.append(zs.cpu())
    fve = (1 - err / (sxx - sx.pow(2).sum(1) / n)).cpu()
    out: dict[str, Any] = {"fve_base": float(fve[0]), "fve_ft": float(fve[1]), "l0": l0 / max(n, 1), "tokens": n,
                           "fired": fired.cpu()}  # fmt: skip
    if keep is not None:
        out["z"] = torch.cat(kept)
    return out


# ----------------------------------------------------------------------------- the recovery rule's pure parts (spec 3.7)


def candidates(R: Tensor, alive: Tensor, n_shared: int, n: int, tail: str = "ft") -> list[int]:
    """The n alive non-shared latents furthest into one tail: 'ft' by 1 - R (the fine-tune side), 'base' by R. Ties go
    to the lower index."""
    key = (1 - R if tail == "ft" else R).double().clone()
    key[:n_shared] = -math.inf
    key[~alive] = -math.inf
    order = torch.argsort(-key, stable=True)[:n]
    return [int(i) for i in order if math.isfinite(float(key[i]))]


def within(a: Tensor, b: float, f: float) -> Tensor:
    """a and b within a factor f of each other (both positive); never for a zero or nan value."""
    return (a > 0) & (b > 0) & (torch.maximum(a / b, b / a) <= f)


def bulk_pool(R: Tensor, alive: Tensor, n_shared: int, cand: int, norm_ft: Tensor, rate: Tensor, exclude: list[int],
              widen: list[tuple[float, float]], size: int, min_size: int, seed: int) -> dict[str, Any]:  # fmt: skip
    """The candidate's bulk pool: alive non-shared latents with |R - 0.5| strictly under the median of |R - 0.5| over
    all non-shared latents, not in `exclude`, matched to the candidate on |d^ft| within the first factor and firing
    rate within the second, the pairs of `widen` tried in order until `size` qualify (then `size` are drawn with
    `seed`). If even the last pair gives fewer than `size`, all that qualify are used when at least `min_size`, and
    the candidate is not scored (an empty pool) when fewer."""
    dev = (R - 0.5).abs()
    bulk = alive.clone()
    bulk[:n_shared] = False
    bulk &= dev < dev[n_shared:].median()
    bulk[torch.tensor(exclude, dtype=torch.long)] = False
    q = bulk
    for fn, fr in widen:
        q = bulk & within(norm_ft, float(norm_ft[cand]), fn) & within(rate, float(rate[cand]), fr)
        if int(q.sum()) >= size:
            idx = torch.nonzero(q).squeeze(1)
            pick = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed))[:size]]
            return {"pool": sorted(int(i) for i in pick), "factors": [fn, fr], "qualified": int(q.sum()), "scored": True}
    idx = [int(i) for i in torch.nonzero(q).squeeze(1)]
    ok = len(idx) >= min_size
    return {"pool": idx if ok else [], "factors": list(widen[-1]), "qualified": len(idx), "scored": ok}


def suppression(g: dict[float, float], g0: float) -> float:
    """The steering test's statistic: the largest drop of the behaviour, g0 - g(s), over the negative strengths; nan
    when every read is nan."""
    v = [g0 - x for s, x in g.items() if s < 0 and math.isfinite(x)]
    return max(v) if v and math.isfinite(g0) else math.nan


def top_share(z: Tensor, on: Tensor, n: int) -> float:
    """The top-activation test's statistic: of the latent's n most activating tokens (z (N,), ties to the lower index),
    the share that fire (z > 0) on a behaviour position (`on` (N,) bool), over n. A latent firing on fewer than n
    tokens counts only those."""
    top = torch.argsort(-z.double(), stable=True)[:n]
    return float(((z[top] > 0) & on[top]).sum()) / n


def beats(c: float, pool: list[float]) -> bool:
    """Strictly over every pool value (a nan pool value is skipped); a nan candidate or an empty pool never passes."""
    vals = [v for v in pool if math.isfinite(v)]
    return bool(math.isfinite(c) and pool and c > max(vals, default=-math.inf))


def side_label(cands: list[dict[str, Any]]) -> str:
    """yes: one scored candidate passes both tests; no: at least one is scored and none passes both; not scored: none
    has a usable floor (spec 3.7)."""
    scored = [c for c in cands if c["scored"]]
    if not scored:
        return "not scored"
    return "yes" if any(c["steer_pass"] and c["top_pass"] for c in scored) else "no"
