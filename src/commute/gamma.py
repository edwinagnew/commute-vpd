"""Gamma refit: let the gate see the updated model (reports/2026-10-20_plan_gamma-refit.md, 2026-10-20).

With the target weights and components set to a state (W0 + delta-W with the components of a split or of component
training; U, V and Delta fixed), train every parameter of Gamma (`model.ci_fn`, the global shared transformer) on the
released run's own objective minus its faithfulness term: importance minimality at the annealed p, the stochastic-mask
subset reconstruction KL and the persistent-PGD reconstruction KL, each at its released coefficient
(`vpd.run_config().loss_metric_configs`), AdamW with the released betas, bf16 autocast, one optimizer step per batch of
Pile rows taken in micro-batches. Every term compares a masked forward with the state's own logits, so the data
supplies inputs only. Stage 0 of the same experiment trains nothing: the adversary's sources on a state, their transfer
between the state and W0, and the single-subcomponent ablation KL of the subcomponents the adversary turns down.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from commute import vpd
from commute.attribution import fwd
from commute.identity import Edit
from commute.local_faith import masked_logits, route_cells
from commute.recruit import Sel, cat_gates, switched

DOWN = 0.5  # a source below this leaves the subcomponent at its CI value ("turned down"); above, at 1
S_ON_THR = 0.3  # the recruit test's switched-on rule: mean trigger gate up by more than this over W0's


@dataclass(frozen=True)
class RefitConfig:
    steps: int
    checkpoints: tuple[int, ...]
    batch: int  # rows per optimizer step
    micro: int  # rows per forward; batch // micro gradient accumulations
    lr: float
    seed: int = 0
    warmup: int = 0  # steps at lr 0 before Gamma moves: the persistent adversary and AdamW's moments warm up (2026-10-21)

    def __post_init__(self) -> None:
        assert self.batch % self.micro == 0 and set(self.checkpoints) <= set(range(1, self.steps + 1)) and self.warmup >= 0, self


def gamma_params(model: Any) -> list[torch.nn.Parameter]:
    return list(model.ci_fn.parameters())


def set_trainable(model: Any, flag: bool) -> None:
    for p in gamma_params(model):
        p.requires_grad_(flag)


def gamma_state(model: Any) -> dict[str, Tensor]:
    """Gamma's parameters, cloned to CPU (fp32, about 2 GB)."""
    return {k: v.detach().cpu().clone() for k, v in model.ci_fn.state_dict().items()}


def load_gamma(model: Any, state: dict[str, Tensor]) -> None:
    device = next(model.parameters()).device
    model.ci_fn.load_state_dict({k: v.to(device=device, dtype=torch.float32) for k, v in state.items()})


class Refit:
    """One refit of Gamma on the model's current state. `step(rows)` is one optimizer step."""

    def __init__(self, model: Any, cfg: RefitConfig):
        from param_decomp.configs import (
            FaithfulnessLossConfig,
            PersistentPGDReconLossConfig,
            PersistentPGDReconSubsetLossConfig,
        )
        from param_decomp.models.batch_and_loss_fns import recon_loss_kl
        from param_decomp.persistent_pgd import PersistentPGDState

        rc = vpd.run_config()
        self.model, self.cfg = model, cfg
        self.device = next(model.parameters()).device
        self.loss_cfgs = [c for c in rc.loss_metric_configs if not isinstance(c, FaithfulnessLossConfig)]
        self.sampling, self.n_mask_samples = rc.sampling, rc.n_mask_samples
        self.use_delta, self.autocast = rc.use_delta_component, rc.autocast_bf16
        self.recon = recon_loss_kl
        params = gamma_params(model)
        assert params and all(p.dtype == torch.float32 for p in params)
        oc = rc.ci_fn_optimizer
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, betas=oc.betas, weight_decay=oc.weight_decay)
        self.clip = oc.grad_clip_norm
        self.wd = vpd.weight_deltas(model)  # the state's Delta; constant while U, V and the target weights are fixed
        torch.manual_seed(cfg.seed)  # the adversary's initial sources and the stochastic masks
        # The released scope is per_batch_per_position: one source per (row, position) slot of the 64-row batch, stepped
        # once per optimizer step. One state per micro-batch slot keeps that under gradient accumulation.
        self.ppgd = {c: [PersistentPGDState(model.module_to_c, (cfg.micro, vpd.SEQ_LEN), self.device, self.use_delta, c, self.recon)
                         for _ in range(cfg.batch // cfg.micro)]
                     for c in self.loss_cfgs if isinstance(c, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig)}
        # The adversary's lr is never scheduled here (no update_lr call): the released schedule is constant after its
        # 2.5% warm-up, so its end-of-schedule value is start_val, which is what a fresh PGD optimizer holds.
        assert all(c.optimizer.lr_schedule.fn_type == "constant" for c in self.ppgd), [c.optimizer for c in self.ppgd]

    def step(self, rows: Tensor) -> dict[str, float]:
        """Gradients accumulated over the micro-batches (each micro-batch's own persistent adversary takes its step),
        then one AdamW step. Returns each loss term's mean over the micro-batches and the weighted total."""
        from param_decomp.losses import compute_losses

        model, wd = self.model, (self.wd if self.use_delta else None)
        self.opt.zero_grad(set_to_none=True)
        n_micro = rows.shape[0] // self.cfg.micro
        sums: dict[str, float] = defaultdict(float)
        for i, sub in enumerate(rows.split(self.cfg.micro)):
            sub = sub.to(self.device)
            ppgd = {c: states[i] for c, states in self.ppgd.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
                with torch.no_grad():
                    out = model(sub, cache_type="input")
                ci = model.calc_causal_importances(out.cache, sampling=self.sampling, detach_inputs=True)
                for st in ppgd.values():
                    st.warmup(model=model, batch=sub, target_out=out.output, ci=ci.lower_leaky, weight_deltas=wd)
                losses = compute_losses(self.loss_cfgs, model, sub, ci, out.output, self.wd, 1.0, self.sampling, self.use_delta,
                                        self.n_mask_samples, ppgd, self.recon)
            total = torch.zeros((), device=self.device)
            for c, v in losses.items():
                assert c.coeff is not None
                total = total + c.coeff * v
            grads = {c: st.get_grads(losses[c], retain_graph=True) for c, st in ppgd.items()}
            (total / n_micro).backward()
            for c, g in grads.items():
                ppgd[c].step(g)
            for c, v in losses.items():
                sums[c.classname] += float(v) / n_micro
            sums["total"] += float(total) / n_micro
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(gamma_params(model), self.clip)
        self.opt.step()
        return dict(sums)


def refit(model: Any, cfg: RefitConfig, rows_at: Any, evaluate: Any, log: Any = print) -> list[dict[str, float]]:
    """Train Gamma on the model's current state: cfg.warmup steps at lr 0 (the persistent adversary takes its steps and
    AdamW accumulates its moments while Gamma stays put), then cfg.steps steps at cfg.lr. `rows_at(i)` gives the
    (batch, 512) rows of loop step i = 0, 1, ...; `evaluate(step)` runs at step 0 (Gamma as loaded) and at every
    checkpoint, counted from the first Gamma step, with Gamma frozen. Returns the loss curve of every loop step, with
    `step` <= 0 during the warm-up. Gamma is left at its last step, trainable flags off."""
    evaluate(0)
    set_trainable(model, True)
    r = Refit(model, cfg)
    curve: list[dict[str, float]] = []
    for i in range(1, cfg.warmup + cfg.steps + 1):
        step = i - cfg.warmup
        for group in r.opt.param_groups:
            group["lr"] = cfg.lr if step > 0 else 0.0
        terms = r.step(rows_at(i - 1))
        curve.append({"step": step, **terms})
        n = i if step <= 0 else step
        if n in (1, 2, 5, 10, 20, 50) or n % 100 == 0 or step == cfg.steps:
            log(f"[{'warmup' if step <= 0 else 'refit'}] step {n}: " + " ".join(f"{k} {v:.4g}" for k, v in terms.items()))
        if step in cfg.checkpoints:
            set_trainable(model, False)
            torch.cuda.empty_cache()
            evaluate(step)
            set_trainable(model, True)
    set_trainable(model, False)
    return curve


# ----------------------------------------------------------------------------- the trigger's switched-on set


def trigger_gates(model: Any, windows: Tensor, half: int, token: int) -> dict[str, Tensor]:
    """g at the trigger of every window, per matrix (N, C) fp16 on CPU, under the model's current state and Gamma."""
    cells = route_cells(model, windows, half, token)
    return {k: v[:, half].half() for k, v in cat_gates(cells["gates"]).items()}


def s_on(trig: dict[str, Tensor], trig_w0: dict[str, Tensor], alive: dict[str, Tensor], thr: float = S_ON_THR) -> Sel:
    """The recruit test's switched-on set: alive subcomponents whose mean trigger gate is up by more than `thr` over
    W0's (`trig_w0`: the released Gamma at W0)."""
    dg = {}
    for k, t in trig.items():
        d = (t.float() - trig_w0[k].float()).mean(0)
        dg[k] = torch.where(alive[k].cpu().bool(), d, torch.zeros_like(d))
    return switched(dg, thr)


# ----------------------------------------------------------------------------- stage 0: the adversary, without training

Sources = dict[str, Tensor]  # per matrix (C + 1,) in [0, 1]: the subcomponents' sources, then Delta's


def _adv_masks(ci: Any, adv: dict[str, Tensor], B: int, T: int) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    comp = {k: ci.g[k] + (1 - ci.g[k]) * a[..., :-1].expand(B, T, -1) for k, a in adv.items()}
    return comp, {k: a[..., -1].expand(B, T) for k, a in adv.items()}


def _cache(model: Any, batches: list[Tensor]) -> list[tuple[Tensor, Any]]:
    device = next(model.parameters()).device
    with torch.no_grad():
        return [(b.to(device), vpd.compute_ci(model, b.to(device), chunk=b.shape[0])) for b in batches]


@torch.no_grad()
def _mask_kl(model: Any, cached: list[tuple[Tensor, Any]], wd: dict[str, Tensor], adv: dict[str, Tensor]) -> float:
    tot = n = 0.0
    for batch, ci in cached:
        B, T = batch.shape
        comp, dm = _adv_masks(ci, adv, B, T)
        tot += float(vpd.kl_per_position(ci.logits, masked_logits(model, batch, comp, wd, dm)).double().sum())
        n += B * T
    return tot / n


def mask_kl(model: Any, batches: list[Tensor], sources: Sources) -> float:
    """Mean KL over every position of the batches between the model's current state and that state masked by
    `sources` (mask = g + (1 - g) a, Delta under its own source), under the current Gamma."""
    device = next(model.parameters()).device
    adv = {k: a.to(device).reshape(1, 1, -1) for k, a in sources.items()}
    return _mask_kl(model, _cache(model, batches), vpd.weight_deltas(model), adv)


def pgd_sources(model: Any, batches: list[Tensor], steps: int, step_size: float = 0.1, seed: int = 0) -> tuple[Sources, float]:
    """The authors' adversary (`faithfulness.faithfulness`'s PGD) with its sources kept: one source per matrix shared
    across every row of every batch, random init, `steps` sign steps of `step_size` on the mean KL over all positions,
    gradients summed over the batches. Returns the sources and the KL they reach."""
    keys = vpd.module_keys(model)
    device = next(model.parameters()).device
    cached = _cache(model, batches)
    wd = vpd.weight_deltas(model)
    torch.manual_seed(seed)
    adv = {k: torch.rand(1, 1, model.module_to_c[k] + 1, device=device).requires_grad_(True) for k in keys}
    for _ in range(steps):
        grads = {k: torch.zeros_like(adv[k]) for k in keys}
        for batch, ci in cached:
            B, T = batch.shape
            with torch.enable_grad():
                comp, dm = _adv_masks(ci, adv, B, T)
                loss = vpd.kl_per_position(ci.logits, masked_logits(model, batch, comp, wd, dm)).mean()
                g = torch.autograd.grad(loss, [adv[k] for k in keys])
            for k, gk in zip(keys, g):
                grads[k] += gk
        with torch.no_grad():
            for k in keys:
                adv[k].add_(step_size * grads[k].sign()).clamp_(0.0, 1.0)
    adv = {k: adv[k].detach() for k in keys}
    return {k: adv[k].flatten().cpu() for k in keys}, _mask_kl(model, cached, wd, adv)


def adversary_groups(sources: Sources, change: dict[str, Tensor], alive: dict[str, Tensor], k: int, seed: int) -> dict[str, Sel]:
    """Three sets of `k` alive subcomponents over all matrices, by the adversary's sources and the state's per-subcomponent
    change (`change[k]`: (C,), gauge-free relative change of u v^T, 0 for dead ones): `down_changed`, the most changed
    among those the adversary turns down (source < DOWN); `up_changed`, the most changed among those it leaves up;
    `down_random`, a random draw of the turned-down ones. Ties in `change` (do-nothing: all zero) fall to the lowest
    index, so for that state only `down_random` is informative."""
    keys = sorted(sources)
    down = torch.cat([(sources[kk][:-1] < DOWN) & alive[kk].cpu().bool() for kk in keys])
    up = torch.cat([(sources[kk][:-1] >= DOWN) & alive[kk].cpu().bool() for kk in keys])
    ch = torch.cat([change[kk].float().cpu() for kk in keys])
    offsets = torch.tensor([0] + [sources[kk].numel() - 1 for kk in keys]).cumsum(0)

    def top(mask: Tensor) -> Tensor:
        score = torch.where(mask, ch, torch.full_like(ch, -1.0))
        return score.argsort(descending=True, stable=True)[:k]

    gen = torch.Generator().manual_seed(seed)
    idx_down = down.nonzero().flatten()
    rand = idx_down[torch.randperm(len(idx_down), generator=gen)[:k]]
    flat = {"down_changed": top(down), "up_changed": top(up), "down_random": rand}
    out: dict[str, Sel] = {}
    for name, f in flat.items():
        f = f.sort().values
        out[name] = {}
        for i, kk in enumerate(keys):
            inside = f[(f >= offsets[i]) & (f < offsets[i + 1])] - offsets[i]
            if len(inside):
                out[name][kk] = inside
    return out


@torch.no_grad()
def ablation_kl(model: Any, key: str, cidx: Tensor, rows: Tensor, m: int = 4) -> Tensor:
    """KL(current model || the model with subcomponent c ablated), mean over every position of `rows` (R, T), for each
    full component index c in `cidx` of matrix `key`, under the model's current target weights and components (Gamma
    is not involved). `m` subcomponents per forward (the Edit hook's groups), fp32 logits."""
    device = next(model.parameters()).device
    U, V = vpd.uv(model, key)
    Uth, Vth = U.T.float().contiguous(), V.float().contiguous()
    lin = vpd.target_linear(model, key)
    rows = rows.to(device)
    lp_full = torch.log_softmax(fwd(model.target_model, rows), -1)
    p_full = lp_full.exp()
    out = torch.zeros(len(cidx), dtype=torch.float64)
    for s in range(0, len(cidx), m):
        c = cidx[s : s + m].to(device)
        h = lin.register_forward_hook(Edit([(Uth, Vth, 1.0)], c))
        try:
            lp = torch.log_softmax(fwd(model.target_model, rows.repeat(len(c), 1)), -1).reshape(len(c), *lp_full.shape)
        finally:
            h.remove()
        out[s : s + len(c)] = (p_full * (lp_full - lp)).sum(-1).mean((1, 2)).double().cpu()
        del lp
    return out


@torch.no_grad()
def mean_gates(model: Any, sel: Sel, rows: Tensor) -> dict[str, Tensor]:
    """Mean g over every position of `rows` for the subcomponents in `sel`, under the current state and Gamma."""
    ci = vpd.compute_ci(model, rows.to(next(model.parameters()).device))
    return {k: ci.g[k][..., idx.to(ci.g[k].device)].float().mean((0, 1)).cpu() for k, idx in sel.items()}
