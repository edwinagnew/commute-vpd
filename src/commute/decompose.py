"""Train a param-decomp decomposition in pieces that survive a job's end: saves, resume, a deadline.

Why. `param_decomp.run_param_decomp.optimize` trains a decomposition in one call with no resume, and the MATS cluster's
debug jobs end at 30 min (E55 C2, plan reports/plans/2026-10-31_plan_e55-c2-vpd.md). This module runs the same step
from the pinned library's own pieces (ComponentModel, the faithfulness warm-up, compute_losses, PersistentPGDState,
get_scheduled_value, evaluate), one call per step, and saves everything the next step reads: the component model,
both AdamW optimizers, the persistent-PGD sources with their Adam moments and step count, and the torch RNG states.
Data are a function of the step (`batches(step)`), so a resumed run sees the batches an unbroken run would.

Differences from `optimize` (single process, no W&B): the step that `optimize` runs at `config.steps` for logging only
is the final evaluation here; `optimize` draws one batch for the persistent-PGD shapes before step 0, here the shape
is given (`batch_dims`); evaluation runs through `evaluate` on the caller's iterator at `config.eval_freq`.
"""

from __future__ import annotations

import gc
import resource
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class Decomp:
    config: Any  # param_decomp.configs.Config
    model: Any  # ComponentModel
    opt_c: torch.optim.Optimizer  # components
    opt_ci: torch.optim.Optimizer  # the causal-importance function
    ppgd: dict[Any, Any]  # persistent-PGD loss config -> PersistentPGDState


def load_config(path: Path, **update: Any) -> Any:
    """The param-decomp Config in a yaml file, with top-level fields replaced by `update` (e.g. steps=...)."""
    import yaml
    from param_decomp.configs import Config

    return Config(**{**yaml.safe_load(path.read_text()), **update})


def _component_params(model: Any) -> list[nn.Parameter]:
    return [p for k in model.target_module_paths for p in model.components[k].parameters()]


def component_model(target: nn.Module, config: Any, device: str | torch.device) -> Any:
    """A ComponentModel of `target` (put in eval mode and frozen) as `optimize` builds it, after seeding with
    config.seed. For a saved decomposition, load its state dict into this (`load_final`)."""
    from param_decomp.models.batch_and_loss_fns import run_batch_passthrough
    from param_decomp.models.component_model import ComponentModel
    from param_decomp.utils.module_utils import expand_module_patterns

    assert config.identity_module_info is None, "identity insertion is not used here"
    target.eval().requires_grad_(False)
    torch.manual_seed(config.seed)
    model = ComponentModel(target_model=target, run_batch=run_batch_passthrough,
                           module_path_info=expand_module_patterns(target, config.all_module_info),
                           ci_config=config.ci_config, sigmoid_type=config.sigmoid_type).to(device)  # fmt: skip
    assert not any(m.training for m in target.modules()), "the target must stay in eval mode (dropout off)"
    return model


def build(target: nn.Module, config: Any, batch_dims: tuple[int, ...], device: str | torch.device) -> Decomp:
    """A fresh decomposition to train: the component model, both optimizers and the persistent-PGD states."""
    from param_decomp.configs import (
        PersistentPGDReconLossConfig,
        PersistentPGDReconSubsetLossConfig,
    )
    from param_decomp.models.batch_and_loss_fns import recon_loss_kl
    from param_decomp.persistent_pgd import PersistentPGDState

    model = component_model(target, config, device)
    oc, oi = config.components_optimizer, config.ci_fn_optimizer
    opt_c = torch.optim.AdamW(_component_params(model), lr=oc.lr_schedule.start_val, betas=oc.betas, weight_decay=oc.weight_decay)
    opt_ci = torch.optim.AdamW(model.ci_fn.parameters(), lr=oi.lr_schedule.start_val, betas=oi.betas, weight_decay=oi.weight_decay)
    ppgd = {cfg: PersistentPGDState(module_to_c=model.module_to_c, batch_dims=batch_dims, device=device,
                                    use_delta_component=config.use_delta_component, cfg=cfg, reconstruction_loss=recon_loss_kl)
            for cfg in config.loss_metric_configs
            if isinstance(cfg, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig)}  # fmt: skip
    return Decomp(config, model, opt_c, opt_ci, ppgd)


# ----------------------------------------------------------------------------- one step


def train_step(d: Decomp, batch: Tensor, step: int) -> dict[str, float]:
    """`optimize`'s loop body for one gradient step at `step` (0-based); returns the logged losses."""
    from param_decomp.losses import compute_losses
    from param_decomp.models.batch_and_loss_fns import recon_loss_kl
    from param_decomp.utils.component_utils import calc_ci_l_zero
    from param_decomp.utils.general_utils import bf16_autocast, get_scheduled_value

    cfg, m = d.config, d.model
    d.opt_c.zero_grad()
    d.opt_ci.zero_grad()
    for opt, oc in ((d.opt_c, cfg.components_optimizer), (d.opt_ci, cfg.ci_fn_optimizer)):
        for g in opt.param_groups:
            g["lr"] = get_scheduled_value(step=step, total_steps=cfg.steps, config=oc.lr_schedule)
    frac = step / cfg.steps
    active = [c for c in d.ppgd if frac >= c.start_frac]
    for c in active:
        d.ppgd[c].update_lr(step, cfg.steps)
    wd = m.calc_weight_deltas()
    with bf16_autocast(enabled=cfg.autocast_bf16):
        out = m(batch, cache_type="input")
        ci = m.calc_causal_importances(pre_weight_acts=out.cache, detach_inputs=False, sampling=cfg.sampling)
        for c in active:
            d.ppgd[c].warmup(model=m, batch=batch, target_out=out.output, ci=ci.lower_leaky,
                             weight_deltas=wd if cfg.use_delta_component else None)  # fmt: skip
        losses = compute_losses(loss_metric_configs=cfg.loss_metric_configs, model=m, batch=batch, ci=ci,
                                target_out=out.output, weight_deltas=wd, current_frac_of_training=frac,
                                sampling=cfg.sampling, use_delta_component=cfg.use_delta_component,
                                n_mask_samples=cfg.n_mask_samples, ppgd_states=d.ppgd, reconstruction_loss=recon_loss_kl)  # fmt: skip
    total = torch.zeros((), device=batch.device)
    for c, v in losses.items():
        assert c.coeff is not None, c  # compute_losses asserts it too
        total = total + c.coeff * v
    grads = {c: d.ppgd[c].get_grads(losses[c], retain_graph=True) for c in active}
    total.backward()
    for c in active:
        d.ppgd[c].step(grads[c])
    for opt, oc, params in ((d.opt_c, cfg.components_optimizer, _component_params(m)),
                            (d.opt_ci, cfg.ci_fn_optimizer, list(m.ci_fn.parameters()))):  # fmt: skip
        if oc.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(params, oc.grad_clip_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)  # the next step recomputes them; holding them only raises the eval's peak
    rec = {f"loss/{c.classname}": float(v.detach()) for c, v in losses.items()}
    rec["loss/total"] = float(total.detach())
    rec["l0/total"] = float(sum(calc_ci_l_zero(v, cfg.ci_alive_threshold) for v in ci.lower_leaky.values()))
    return rec


def evaluate(d: Decomp, eval_batches: Iterator[Tensor], step: int) -> dict[str, float]:
    """param-decomp's `evaluate` on the config's eval metrics (numbers only), as `optimize` calls it at `step`."""
    from param_decomp.eval import evaluate as pd_evaluate
    from param_decomp.models.batch_and_loss_fns import recon_loss_kl
    from param_decomp.utils.general_utils import bf16_autocast

    cfg = d.config
    slow = cfg.slow_eval_on_first_step if step == 0 else step % cfg.slow_eval_freq == 0
    device = str(next(d.model.parameters()).device)
    with torch.no_grad(), bf16_autocast(enabled=cfg.autocast_bf16):
        out = pd_evaluate(eval_metric_configs=list(cfg.eval_metric_configs), model=d.model, eval_iterator=eval_batches,
                          device=device, run_config=cfg, slow_step=slow, n_eval_steps=cfg.n_eval_steps,
                          current_frac_of_training=step / cfg.steps, reconstruction_loss=recon_loss_kl, ppgd_states=d.ppgd)  # fmt: skip
    return {k: float(v) for k, v in out.items() if isinstance(v, int | float)}


# ----------------------------------------------------------------------------- saves


def _ppgd_state(s: Any) -> dict[str, Any]:
    """A PersistentPGDState's sources (copied to CPU: a GPU clone added ~2.4 GB at 38K subcomponents, and job 47013
    then ran out of memory in the eval after the save) and its optimizer's attributes (Adam: lr, betas, eps, step
    count, m, v; torch.save writes those GPU tensors without a resident copy)."""
    return {"sources": {k: v.detach().cpu() for k, v in s.sources.items()}, "opt": dict(vars(s.optimizer))}


def _load_ppgd(s: Any, saved: dict[str, Any]) -> None:
    assert saved["sources"].keys() == s.sources.keys()
    with torch.no_grad():
        for k, v in saved["sources"].items():
            s.sources[k].copy_(v)
    dev = next(iter(s.sources.values())).device
    vars(s.optimizer).update({k: ({n: t.to(dev) for n, t in v.items()} if isinstance(v, dict) else v)
                              for k, v in saved["opt"].items()})  # fmt: skip


def save(path: Path, d: Decomp, step: int) -> None:
    """Everything a resume reads, written atomically (tmp file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"step": step, "model": d.model.state_dict(), "opt_c": d.opt_c.state_dict(), "opt_ci": d.opt_ci.state_dict(),
             "ppgd": [_ppgd_state(s) for s in d.ppgd.values()], "rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}  # fmt: skip
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load(path: Path, d: Decomp) -> int:
    """Load a `save` file into a freshly built `d`; returns its step. Read to CPU first: each tensor is copied onto the
    device by load_state_dict / copy_, so the file never sits on the GPU beside the built state (job 46971 OOMed with
    an 8.4 GB file loaded straight to an L40 already holding the run)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    d.model.load_state_dict(state["model"])
    d.opt_c.load_state_dict(state["opt_c"])
    d.opt_ci.load_state_dict(state["opt_ci"])
    assert len(state["ppgd"]) == len(d.ppgd)
    for s, saved in zip(d.ppgd.values(), state["ppgd"], strict=True):
        _load_ppgd(s, saved)
    torch.set_rng_state(state["rng"].cpu())
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all([r.cpu() for r in state["cuda_rng"]])
    return int(state["step"])


def release(d: Decomp) -> None:
    """Drop a decomposition's GPU state: model, optimizers and persistent-PGD sources, then collect the reference
    cycles (module hooks) and return the cached blocks, so a rebuild in the same process starts from an empty card."""
    d.ppgd.clear()
    del d.model, d.opt_c, d.opt_ci
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_final(path: Path, d: Decomp, step: int) -> None:
    """The decomposition alone (component model state dict and the step): what readings load."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"step": step, "model": d.model.state_dict()}, tmp)
    tmp.replace(path)


def load_final(path: Path, target: nn.Module, config: Any, device: str | torch.device) -> tuple[Any, int]:
    """(the frozen component model in eval mode, its step) from a `save_final` or `save` file."""
    model = component_model(target, config, device)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.requires_grad_(False)
    return model.eval(), int(state["step"])


# ----------------------------------------------------------------------------- the run


def memory() -> dict[str, float]:
    """Peak GPU memory allocated and reserved (GiB, since the last reset_peak_memory_stats) and the process's peak
    host RSS (GiB; ru_maxrss is in KiB on Linux)."""
    gpu = torch.cuda.is_available()
    return {"gpu_alloc_gib": torch.cuda.max_memory_allocated() / 2**30 if gpu else 0.0,
            "gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if gpu else 0.0,
            "host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}  # fmt: skip


@dataclass(frozen=True)
class Plan:
    end: int  # stop after this many gradient steps (<= config.steps)
    save_every: int
    log_every: int
    deadline: float | None = None  # time.monotonic() value; save and stop when passed


def run(d: Decomp, batches: Callable[[int], Tensor], ckpt: Path, plan: Plan, eval_batches: Iterator[Tensor] | None = None,
        log: Callable[[str], None] = print) -> tuple[int, list[dict[str, Any]]]:  # fmt: skip
    """Resume from `ckpt` if it exists (else the faithfulness warm-up, then step 0) and train to `plan.end`, saving every
    `save_every` steps, at the end and at the deadline; evaluates at multiples of eval_freq and at the end when
    `eval_batches` is given. Returns (step reached, the log records)."""
    from param_decomp.run_param_decomp import run_faithfulness_warmup

    cfg = d.config
    assert 0 < plan.end <= cfg.steps and plan.save_every > 0 and plan.log_every > 0, plan
    if ckpt.exists():
        t = time.perf_counter()
        step = load(ckpt, d)
        log(f"decompose: resumed at step {step} from {ckpt} in {time.perf_counter() - t:.1f} s")
    else:
        step = 0
        if cfg.faithfulness_warmup_steps > 0:
            t = time.perf_counter()
            run_faithfulness_warmup(d.model, _component_params(d.model), cfg)
            log(f"decompose: faithfulness warm-up {cfg.faithfulness_warmup_steps} steps in {time.perf_counter() - t:.1f} s")
    records: list[dict[str, Any]] = []
    t_win = time.perf_counter()
    while step < plan.end:
        if eval_batches is not None and step % cfg.eval_freq == 0:
            records.append({"step": step, "eval": evaluate(d, eval_batches, step)})
            log(f"eval step {step}: {records[-1]['eval']}")
            t_win = time.perf_counter()
        rec = train_step(d, batches(step), step)
        step += 1
        if step % plan.log_every == 0:
            ms, t_win = 1e3 * (time.perf_counter() - t_win) / plan.log_every, time.perf_counter()
            records.append({"step": step, "ms_per_step": ms, **rec, **memory()})
            log(f"step {step}: {ms:.1f} ms/step " + " ".join(f"{k} {v:.4g}" for k, v in records[-1].items() if k != "step"))
        late = plan.deadline is not None and time.monotonic() > plan.deadline
        if step % plan.save_every == 0 or step == plan.end or late:
            t = time.perf_counter()
            save(ckpt, d, step)
            alloc = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            log(f"decompose: saved step {step} ({ckpt.stat().st_size / 1e9:.2f} GB) in {time.perf_counter() - t:.1f} s; {alloc:.1f} GiB allocated")
            t_win = time.perf_counter()
        if late:
            log(f"decompose: deadline at step {step}")
            break
    if step == cfg.steps and eval_batches is not None:
        records.append({"step": step, "eval": evaluate(d, eval_batches, step)})
        log(f"eval step {step} (final): {records[-1]['eval']}")
    return step, records
