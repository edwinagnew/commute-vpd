"""Glue around Minder et al.'s crosscoder code for E203 (cell P2.31 run 2; plan reports/plans/2026-11-17_plan_p2-31-run2.md,
appendix ..._run2_appendix.md). Run 1 (E198) used an own implementation; it is in git history before this file changed.

From their code, pinned in pyproject.toml (`dictionary-learning`, science-of-finetuning/crosscoder_learning@d1fd6f3):
the model `BatchTopKCrossCoder` (paired init with the decoder the encoder's transpose, BatchTopK on f * sum of decoder
norms, the inference threshold, per-model mean/std normalisation to `target_rms`) and the training step
`BatchTopKCrossCoderTrainer.update` (unsquared L2 loss, aux loss on dead latents, threshold EMA, warm-up, k annealing,
gradient clipping, decoder bias at the geometric median). Settings: diffing-toolkit@617d926
configs/diffing/method/crosscoder.yaml.

Ours: the normaliser fit, the latent-scaling betas (the closed form of diffing-toolkit@617d926
src/diffing/utils/dictionary/latent_scaling/closed_form.py:40-168 with the targets of its utils.py: errors with the
latent's own contribution added back, full reconstructions, all in normalised space, probed with the fine-tune decoder;
written as running sums instead of materialised per-latent targets, checked against them in tests/test_crosscoder.py),
the top-N candidate rule (closed_form.py:365, "sorting by dec_norm_diff ... taking the head") with the paper's
nu_r < 0.5 and nu_eps < 0.2, and the change reading, computed in raw activation space:
  D(x) = h_ft(x) - h_base(x) = sum_j f_j(x) (d_j^ft - d_j^base) + (bias part) + (error),
with each decoder taken back to raw units (divided by its model's normalisation scale; the means go to the bias part).
s_j = sum_x f_j(x) <d_j^ft - d_j^base, D(x)> / sum_x |D(x)|^2. Model index 0 is the base, 1 the fine-tune.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from dictionary_learning.trainers.crosscoder import BatchTopKCrossCoderTrainer
from torch import Tensor

AE = Any  # their BatchTopKCrossCoder: its methods return untyped unions, so it is handled as Any here
Trainer = Any  # their BatchTopKCrossCoderTrainer


@dataclass(frozen=True)
class Cfg:
    """Every size of a run; the full run and the rehearsal are two instances (results/run2/predictions.json)."""

    d: int
    expansion: int
    k: int
    k_max: int
    k_anneal_steps: int
    lr: float
    warmup: int
    batch: int
    tokens: int  # token presentations per crosscoder, all epochs: the one budget value
    epochs: int
    aux_alpha: float
    dead_tokens: int
    init_norm: float
    target_rms: float
    chunk_rows: int  # Pile rows per locally shuffled chunk (plus as many task tokens)
    ls_tokens: int  # latent-scaling tokens, from the first training chunk
    val_rows: int
    alive_min: int
    top_n: int
    top_n_wide: int
    nu_r: float
    nu_eps: float
    shared: tuple[float, float]
    base_only: float
    snapshots: tuple[
        float, ...
    ]  # fractions of `tokens` scored before the end (stages 0 and 1)
    ckpt_every: int  # steps
    log_every: int  # steps between training-curve records
    pile_start: int  # first Pile train row of the Pile half: past every row a task window was cut from
    pool_start: int  # first Pile train row of the pool fresh key-value windows are cut from: past the Pile half
    pool_rows: int
    repeat_cap: int  # most draws of one window per epoch, for tasks whose windows cannot be made fresh

    def __post_init__(self) -> None:
        assert self.k * 10 <= self.latents and self.k <= self.k_max <= self.latents, (
            self
        )
        assert (
            self.warmup < self.steps
            and self.k_anneal_steps < self.steps
            and self.ckpt_every < self.steps
        ), self
        assert (
            0 < self.top_n < self.top_n_wide <= self.latents and self.ls_tokens > 0
        ), self
        assert 0 <= self.base_only < self.shared[0] < 0.5 < self.shared[1] <= 1, self
        assert all(
            0 <= a < b < 1
            for a, b in zip((0, *self.snapshots), self.snapshots, strict=False)
        ), self

    @property
    def latents(self) -> int:
        return self.expansion * self.d

    @property
    def steps(self) -> int:
        return self.tokens // self.batch


def normaliser(h: Tensor) -> tuple[Tensor, Tensor]:
    """Per-model mean and std (biased) of a raw (n, 2, d) sample (diffing-toolkit training.py:37-73)."""
    return h.mean(0), h.std(0, unbiased=False)


def trainer(cfg: Cfg, mean: Tensor, std: Tensor, seed: int) -> Trainer:
    t = BatchTopKCrossCoderTrainer(
        steps=cfg.steps, activation_dim=cfg.d, dict_size=cfg.latents, k=cfg.k, layer=2, lm_name="commute-67m",
        num_layers=2, k_max=cfg.k_max, k_annealing_steps=cfg.k_anneal_steps, lr=cfg.lr, auxk_alpha=cfg.aux_alpha,
        warmup_steps=cfg.warmup, seed=seed, device="cuda", activation_mean=mean, activation_std=std,
        target_rms=cfg.target_rms,
        dict_class_kwargs={"same_init_for_all_layers": True, "init_with_transpose": True, "norm_init_scale": cfg.init_norm,
                           "code_normalization": "crosscoder"},
    )  # fmt: skip
    t.dead_feature_threshold = (
        cfg.dead_tokens
    )  # their default, 10M; the rehearsal lowers it to run the aux path
    return t


def state(t: Trainer, step: int) -> dict[str, Any]:
    return {"ae": t.ae.state_dict(), "opt": t.optimizer.state_dict(), "sched": t.scheduler.state_dict(),
            "since": t.num_tokens_since_fired, "step": step}  # fmt: skip


def load_state(t: Trainer, s: dict[str, Any]) -> int:
    t.ae.load_state_dict(s["ae"])
    t.optimizer.load_state_dict(s["opt"])
    t.scheduler.load_state_dict(s["sched"])
    t.num_tokens_since_fired = s["since"].to(t.num_tokens_since_fired.device)
    return int(s["step"])


@torch.no_grad()
def fve_batch(ae: AE, x: Tensor) -> float:
    """Training-mode FVE of one raw (B, 2, d) batch, in normalised space, before the step."""
    xn = ae.normalize_activations(x.clone())
    xh = ae.decode(
        ae.encode(xn, use_threshold=False, normalize_activations=False),
        denormalize_activations=False,
    )
    return 1 - float((xn - xh).pow(2).sum()) / float((xn - xn.mean(0)).pow(2).sum())


def train(t: Trainer, batches: Iterable[Tensor], start: int, stop: int, every: int, curve: list[dict[str, float]],
          log: Callable[[str], None], hook: Callable[[int], None]) -> None:  # fmt: skip
    """Their `update` on raw (B, 2, d) batches, steps start..stop-1, appending a record to `curve` every `every` steps
    (loss, batch FVE before the step, k, dead share, theta, ms per step since the last record); `hook(step)` after each
    step (saves, snapshots)."""
    t_last, s_last = time.time(), start
    for step, x in zip(range(start, stop), batches, strict=False):
        rec = {"step": step, "fve_batch": fve_batch(t.ae, x)} if step % every == 0 or step == stop - 1 else None
        loss = t.update(step, x)
        assert math.isfinite(loss), f"loss {loss} at step {step}"
        if rec is not None:
            now = time.time()
            rec |= {"loss": loss, "k": float(t.ae.k), "dead": float((t.num_tokens_since_fired >= t.dead_feature_threshold).float().mean()),
                    "theta": float(t.ae.threshold), "ms_per_step": 1000 * (now - t_last) / max(1, step + 1 - s_last)}  # fmt: skip
            t_last, s_last = now, step + 1
            curve.append(rec)
            log(f"  step {step}/{t.steps} " + " ".join(f"{k} {v:.4g}" for k, v in rec.items() if k != "step"))
        hook(step + 1)


def relative_norm(ae: AE) -> Tensor:
    """Minder eq. 3 (diffing-toolkit analysis.py:59-62 stores 1 minus this): 1 fine-tune only, 0.5 shared, 0 base only."""
    n = ae.decoder.weight.detach().norm(dim=2)
    return (0.5 * (1 + (n[1] - n[0]) / n.max(0).values.clamp_min(1e-12))).cpu()


@torch.no_grad()
def raw(ae: AE) -> tuple[Tensor, Tensor]:
    """Decoder (2, D, d) and bias (2, d) in raw activation units: decode(f) = f @ W_raw[m] + b_raw[m]."""
    s = ae.activation_global_scale.view(2, 1, 1)
    return ae.decoder.weight / s, ae.decoder.bias / s[:, 0] + ae.activation_mean


@torch.no_grad()
def fve(ae: AE, chunks: Iterable[Tensor]) -> dict[str, Any]:
    """Per-model FVE (raw space), firing counts and mean L0 over raw (n, 2, d) chunks, the inference threshold on."""
    err = torch.zeros(2, dtype=torch.float64, device="cuda")
    sxx = torch.zeros(2, dtype=torch.float64, device="cuda")
    sx = torch.zeros(2, ae.activation_dim, dtype=torch.float64, device="cuda")
    fired, n, l0 = torch.zeros(ae.dict_size, device="cuda"), 0, 0.0
    for x in chunks:
        f = ae.encode(x)
        err += (x - ae.decode(f)).double().pow(2).sum((0, 2))
        sx += x.double().sum(0)
        sxx += x.double().pow(2).sum((0, 2))
        fired += (f > 0).sum(0)
        n, l0 = n + x.shape[0], l0 + float((f > 0).sum())
    out = (1 - err / (sxx - sx.pow(2).sum(1) / n)).cpu()
    return {
        "fve_base": float(out[0]),
        "fve_ft": float(out[1]),
        "l0": l0 / n,
        "fired": fired.cpu(),
    }


@torch.no_grad()
def betas(ae: AE, h: Tensor, J: Tensor, size: int = 4096) -> dict[str, Tensor]:
    """Latent scaling for latents J on raw (n, 2, d): beta = sum_x f <y, d> / (sum_x f^2 |d|^2) with d = d_j^ft, in
    normalised space; targets eps_m = x_m - (xhat_m - f_j d_j^m) and r_m = xhat_m; nu = beta_base / beta_ft."""
    W, J = ae.decoder.weight, J.to(ae.decoder.weight.device)
    dJ, dbJ = W[1, J], W[0, J]
    keys = ("eps_base", "eps_ft", "r_base", "r_ft")
    num = {k: torch.zeros(len(J), dtype=torch.float64, device=W.device) for k in keys}
    den = torch.zeros(len(J), dtype=torch.float64, device=W.device)
    for i in range(0, h.shape[0], size):
        xn = ae.normalize_activations(h[i : i + size].clone())
        f = ae.encode(xn, normalize_activations=False)
        rec = ae.decode(f, denormalize_activations=False)
        fJ, eps = f[:, J], xn - rec
        proj = {"eps_base": eps[:, 0] @ dJ.T + fJ * (dbJ * dJ).sum(1), "eps_ft": eps[:, 1] @ dJ.T + fJ * dJ.pow(2).sum(1),
                "r_base": rec[:, 0] @ dJ.T, "r_ft": rec[:, 1] @ dJ.T}  # fmt: skip
        for k in keys:
            num[k] += (fJ * proj[k]).sum(0).double()
        den += (fJ.pow(2).sum(0) * dJ.pow(2).sum(1)).double()
    b = {k: (v / den).cpu() for k, v in num.items()}
    return b | {"nu_eps": b["eps_base"] / b["eps_ft"], "nu_r": b["r_base"] / b["r_ft"]}


def specific(cfg: Cfg, b: dict[str, Tensor]) -> Tensor:
    """The paper's test, nu_r < 0.5 and nu_eps < 0.2 (Minder et al. glossary, §3.1), with no other filter; a nan
    ratio (a latent that never fired on the latent-scaling tokens) fails both comparisons."""
    return (b["nu_r"] < cfg.nu_r) & (b["nu_eps"] < cfg.nu_eps)


@torch.no_grad()
def carried(ae: AE, h: Tensor, size: int = 4096) -> dict[str, Any]:
    """Each latent's carried part s_j of the raw change on (n, 2, d), the bias part, and the FVE of the change."""
    Wr, br = raw(ae)
    dD, db = (Wr[1] - Wr[0]).double(), (br[1] - br[0]).double()
    s = torch.zeros(ae.dict_size, dtype=torch.float64, device=dD.device)
    total = bias = resid = 0.0
    for i in range(0, h.shape[0], size):
        x = h[i : i + size]
        f = ae.encode(x).double()
        delta = (x[:, 1] - x[:, 0]).double()
        s += (f * (delta @ dD.T)).sum(0)
        total += float(delta.pow(2).sum())
        bias += float((delta @ db).sum())
        resid += float((delta - f @ dD - db).pow(2).sum())
    if total <= 0:  # no change (the base against itself): every part undefined
        return {
            "energy": 0.0,
            "s": torch.full((ae.dict_size,), math.nan),
            "bias": math.nan,
            "fve_change": math.nan,
        }
    return {
        "energy": total / h.shape[0],
        "s": (s / total).cpu(),
        "bias": bias / total,
        "fve_change": 1 - resid / total,
    }


def top(rn: Tensor, n: int) -> Tensor:
    """A (D,) mask of the n latents with the highest relative norm; ties go to the lower index."""
    m = torch.zeros(len(rn), dtype=torch.bool)
    m[torch.argsort(-rn.double(), stable=True)[:n]] = True
    return m


def groups(
    cfg: Cfg, rn: Tensor, alive: Tensor, cand: Tensor, spec: Tensor
) -> dict[str, Tensor]:
    """Disjoint latent groups: fine-tune-specific (a top-N candidate that passes latent scaling), shared (alive,
    relative norm inside the open band, as diffing-toolkit analysis.py:104-110 tags it), artefact (a candidate that
    fails), base-only (relative norm under `base_only`, open as there), other alive, dead."""
    g = {"ft_specific": cand & spec}
    g["shared"] = alive & (rn > cfg.shared[0]) & (rn < cfg.shared[1]) & ~g["ft_specific"]
    g["ft_artefact"] = cand & ~spec & ~g["shared"]
    g["base_only"] = alive & (rn < cfg.base_only) & ~cand
    g["other"] = alive & ~(
        g["ft_specific"] | g["shared"] | g["ft_artefact"] | g["base_only"]
    )
    g["dead"] = ~(alive | g["ft_specific"] | g["ft_artefact"])
    return g


def parts(s: Tensor, bias: float, g: dict[str, Tensor]) -> dict[str, float]:
    """Each group's part of the change, the bias part and the error part (1 minus the rest); they sum to one."""
    p = {k: float(s[m].sum()) for k, m in g.items()} | {"bias": bias}
    p["error"] = 1 - float(s.sum()) - bias
    return p


def split(h: Tensor, size: int) -> Iterator[Tensor]:
    for i in range(0, h.shape[0], size):
        yield h[i : i + size]
