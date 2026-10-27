"""E38's routing trace (`experiments/recruitment_test`, stage 2): does a fine-tune move each decomposed matrix's input
at the trigger toward the inputs the original model has where it already predicts the target?
Plan: reports/plans/2026-10-27_plan_regating-recruitment.md, "Stage 2".

Gamma reads the raw inputs of the 24 matrices (the forward with `cache_type="input"` that `vpd.compute_ci` runs). Per
matrix m, on a task's held-out windows at the trigger: a0 under W0, a1 under the fine-tuned model, da = mean over
windows of a1 - a0. Over Pile rows at W0: mu_all, the mean input over all positions, and mu_target, the mean input at
the positions whose next token is the target (`recruit.precede_mask`); the natural direction n = mu_target - mu_all.
  cos         cos(da, n)
  closer      mean over windows of |a1 - mu_target| / |a0 - mu_target|
  closer_all  the same with mu_all (the control: about 1 if the inputs move toward the target's contexts only)
A random direction's cosine with n is about 1 / sqrt(d_in).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from commute.recruit import layer_or_none

COS_MIN, NULL_RATIO = 0.3, 3.0  # P4: cos >= 0.3 and >= 3 x the ' http' null's cos, on at least half the layer's matrices


@torch.no_grad()
def inputs_of(model: Any, batch: Tensor) -> dict[str, Tensor]:
    """Each decomposed matrix's input on every position, (B, T, d_in) float32 on the model's device, under the model's
    current weights (bf16 autocast on CUDA, as `vpd.compute_ci`)."""
    from commute.vpd import module_keys

    device = next(model.parameters()).device
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(batch.to(device), cache_type="input")
    return {k: out.cache[k].float() for k in module_keys(model)}


def trigger_inputs(model: Any, windows: Tensor, half: int, batch: int = 64) -> dict[str, Tensor]:
    """Each matrix's input at the trigger of every window, (N, d_in) float32 on CPU."""
    parts = [inputs_of(model, windows[s : s + batch]) for s in range(0, windows.shape[0], batch)]
    return {k: torch.cat([p[k][:, half].cpu() for p in parts]) for k in parts[0]}


def pile_means(model: Any, rows: Tensor, masks: dict[str, Tensor], batch: int = 8) -> dict[str, Any]:
    """Over the rows at the model's current weights: mu_all per matrix ('all'), and for each mask (B, T) bool its masked
    mean per matrix and number of positions ('by'[name] = {'mean', 'n'}). Means float64 on CPU."""
    s_all: dict[str, Tensor] = {}
    s_by: dict[str, dict[str, Tensor]] = {n: {} for n in masks}
    for s in range(0, rows.shape[0], batch):
        x = inputs_of(model, rows[s : s + batch])
        for k, v in x.items():
            flat = v.reshape(-1, v.shape[-1]).double()
            s_all[k] = s_all.get(k, 0) + flat.sum(0)
            for n, m in masks.items():
                w = m[s : s + batch].reshape(-1).to(flat.device, flat.dtype)
                s_by[n][k] = s_by[n].get(k, 0) + w @ flat
        del x
    n_all = rows.numel()
    return {"all": {k: (v / n_all).cpu() for k, v in s_all.items()}, "n_all": n_all,
            "by": {n: {"mean": {k: (v / max(int(masks[n].sum()), 1)).cpu() for k, v in s_by[n].items()},
                       "n": int(masks[n].sum())} for n in masks}}


def trace(a0: dict[str, Tensor], a1: dict[str, Tensor], mu_t: dict[str, Tensor], mu_all: dict[str, Tensor]) -> dict[str, Any]:
    """cos, closer and closer_all per matrix (see the module docstring), plus |da| and 1 / sqrt(d_in)."""
    out = {}
    for k, v0 in a0.items():
        x0, x1, mt, ma = v0.double(), a1[k].double(), mu_t[k].double(), mu_all[k].double()
        da, nat = (x1 - x0).mean(0), mt - ma
        cos = float(da @ nat / (da.norm() * nat.norm())) if da.norm() * nat.norm() > 0 else float("nan")
        out[k] = {"cos": cos, "closer": float(((x1 - mt).norm(dim=1) / (x0 - mt).norm(dim=1)).mean()),
                  "closer_all": float(((x1 - ma).norm(dim=1) / (x0 - ma).norm(dim=1)).mean()),
                  "norm_da": float(da.norm()), "random_cos": float(x0.shape[1]) ** -0.5}
    return out


def p4_state(tr: dict[str, Any], tr_null: dict[str, Any], layer: int) -> dict[str, Any]:
    """P4 on one task-seed: among the matrices of `layer` (the layer holding most of S), those with cos >= 0.3 and
    cos >= 3 x the null's cos on the same matrix; passes with at least half of them."""
    mats = sorted(k for k in tr if layer_or_none(k) == layer)
    ok = {k: tr[k]["cos"] >= COS_MIN and tr[k]["cos"] >= NULL_RATIO * tr_null[k]["cos"] for k in mats}
    return {"layer": layer, "ok": ok, "n_ok": sum(ok.values()), "n": len(mats), "pass": 2 * sum(ok.values()) >= len(mats)}
