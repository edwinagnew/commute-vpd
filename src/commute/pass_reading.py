"""The pass reading added 2026-10-29 (`decisions.md`, "Card triage" item (2)): behaviour kept by the parts Gamma switches
on, leftover weights at the original's, on the task's own text. Plan: reports/plans/2026-10-29_plan_e48-pass-reread.md.

The masked model of a state (the model's current target weights and components) is the forward with every subcomponent
at Gamma's g, Gamma reading the state's own activations (`vpd.compute_ci`), and Delta held at W0's value Delta_0 under
mask 1 ("leftover weights at the original's"). On task windows the reading is taken at the trigger: K = the mean over
windows of KL(reference || masked model) there, the reference being the fine-tuned model. With K(D) the unchanged
decomposition's K against the fine-tune and K0 W0's own K against W0 on the same windows, the pass reading is the share
kept, [K(D) - K(D')] / [K(D) - K0] (`maps_score.kept`, the Qwen line's r2), and the ratio K(D') / K0. Every cell is also
read with Delta off (the Qwen Q0 plan's convention and E42's CI-masked cells), reported beside.

`carriers` lists the subcomponents that carry a copy's gain at the trigger (E42 analysis, "What next" item 1).
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor, nn

from commute.data import Windows, target_tokens
from commute.finetune import logits_of
from commute.local_faith import masked_logits
from commute.vpd import compute_ci, kl_per_position, module_keys


def _masked_pair(model: Any, batch: Tensor, wd0: dict[str, Tensor]) -> tuple[Any, Tensor, Tensor]:
    """Gamma's g on the state's own activations (with the state's own target logits), and the masked forward with Delta
    at `wd0` (mask 1) and with Delta off. bf16 autocast, as every CI-masked reading in the repo."""
    ci = compute_ci(model, batch, chunk=batch.shape[0])
    ones = {k: torch.ones(batch.shape, device=batch.device, dtype=ci.g[k].dtype) for k in ci.g}
    on = masked_logits(model, batch, ci.g, wd0, ones)
    off = masked_logits(model, batch, ci.g, None, None)
    return ci, on, off


def _p(logits: Tensor, tgt: Tensor) -> Tensor:
    return torch.softmax(logits.float(), -1).gather(1, tgt[:, None]).squeeze(1)


@torch.no_grad()
def trigger_cells(model: Any, windows: Windows, token: int | Tensor, ref: nn.Module | None, wd0: dict[str, Tensor],
                  batch_size: int = 64) -> dict[str, float]:
    """Under the model's current state, means over `windows`: K (Delta at wd0) and K_off (Delta off) at the trigger, the
    KL from `ref`'s logits (None: the state's own target model) to the masked model; kl_other, K's mean over the other
    positions; P(target) under `ref`, under the masked model and under the Delta-off masked model."""
    device = next(model.parameters()).device
    half = windows.half
    keys = ("K", "K_off", "kl_other", "p_ref", "p_masked", "p_masked_off")
    sums = dict.fromkeys(keys, 0.0)
    for s in range(0, len(windows), batch_size):
        batch = windows.tokens[s : s + batch_size].to(device)
        tgt = target_tokens(token, batch, half)
        ci, on, off = _masked_pair(model, batch, wd0)
        r = ci.logits if ref is None else logits_of(ref, batch)
        kl_on, kl_off = kl_per_position(r, on), kl_per_position(r, off)
        other = torch.ones(batch.shape[1], dtype=torch.bool, device=device)
        other[half] = False
        for name, v in (("K", kl_on[:, half]), ("K_off", kl_off[:, half]), ("kl_other", kl_on[:, other].mean(1)),
                        ("p_ref", _p(r[:, half], tgt)), ("p_masked", _p(on[:, half], tgt)),
                        ("p_masked_off", _p(off[:, half], tgt))):
            sums[name] += float(v.double().sum())
    return {k: v / len(windows) for k, v in sums.items()}


@torch.no_grad()
def rows_cells(model: Any, batches: list[Tensor], wd0: dict[str, Tensor]) -> dict[str, float]:
    """W0-style reading on plain rows under the model's current state: the mean over every position of KL(own target ||
    masked model), Delta at wd0 (K) and Delta off (K_off; at W0 on the harness's 256 Pile rows this is the harness's
    ci_masked_kl, 0.348). Batches of equal size, so the mean of batch means is the mean over positions."""
    device = next(model.parameters()).device
    k_on = k_off = 0.0
    for b in batches:
        batch = b.to(device)
        ci, on, off = _masked_pair(model, batch, wd0)
        k_on += float(kl_per_position(ci.logits, on).double().mean())
        k_off += float(kl_per_position(ci.logits, off).double().mean())
    return {"K": k_on / len(batches), "K_off": k_off / len(batches)}


@torch.no_grad()
def path_check(model: Any, batches: list[Tensor], wd0: dict[str, Tensor]) -> dict[str, float]:
    """fp32, no autocast: the mean over positions of KL(target || every mask 1 with Delta at wd0), and the same with
    Delta off. At W0 the first is float32 rounding (U V^T + Delta_0 = W0); the second is the harness's unmasked_kl
    (0.012 on Pile rows). A first reading near the second means Delta never entered the masked forward."""
    from commute.attribution import fwd

    device = next(model.parameters()).device
    on_sum = off_sum = 0.0
    for b in batches:
        batch = b.to(device)
        B, T = batch.shape
        target = fwd(model.target_model, batch)
        ones = {k: torch.ones(B, T, model.module_to_c[k], device=device) for k in module_keys(model)}
        dm = {k: torch.ones(B, T, device=device) for k in ones}
        on = masked_logits(model, batch, ones, wd0, dm, autocast=False)
        off = masked_logits(model, batch, ones, None, None, autocast=False)
        on_sum += float(kl_per_position(target, on).double().mean())
        off_sum += float(kl_per_position(target, off).double().mean())
    return {"delta_on": on_sum / len(batches), "delta_off": off_sum / len(batches)}


def _log_odds(model: Any, toks: Tensor, half: int, tgt: Tensor) -> float:
    from commute.attribution import fwd
    from commute.identity import log_odds

    return float(log_odds(fwd(model.target_model, toks)[:, half], tgt).double().mean())


def _count_to(sorted_drops: Tensor, frac: float) -> int | None:
    """The fewest top drops whose sum reaches `frac` of the sum of the positive drops; None if no drop is positive."""
    pos = sorted_drops.clamp_min(0)
    total = float(pos.sum())
    if total <= 0:
        return None
    return int((pos.cumsum(0) < frac * total).sum()) + 1


@torch.no_grad()
def carriers(model: Any, uv: dict[str, tuple[Tensor, Tensor]], uv2: dict[str, tuple[Tensor, Tensor]],
             W0: dict[str, Tensor], held: Windows, token: int | Tensor, alive: dict[str, Tensor],
             g_copy: dict[str, Tensor], g_w0: dict[str, Tensor], near_zero: float, top: int = 20,
             log: Any = print) -> dict[str, Any]:
    """Which subcomponents carry a copy's gain at the trigger. `uv` = the alive (U, V) at W0 and `uv2` the copy's, theory
    layout, on the model's device. Per alive subcomponent, the drop in the copy's mean trigger log-odds of the target
    when its change alone is reverted (`attribution.revert_one`, fp32, one subcomponent per forward since the target
    differs per window). Reports the gain (copy minus W0), how many of the largest drops carry 50% and 90% of the summed
    positive drops (n50, n90), the share of the n50 set's drops whose mean g at the trigger under the copy is below
    `near_zero` (`g_copy`, `g_w0`: `local_faith.trigger_mean_gates`, (C,) per matrix), the fraction of the gain removed
    when the n50 set is reverted together (the drops need not add), and the top `top` with their g."""
    from commute.attribution import revert_one, reverted_weights
    from commute.faithfulness import model_state

    keys = sorted(uv)
    device = next(model.parameters()).device
    toks, half = held.tokens.to(device), held.half
    tgt = target_tokens(token, toks, half)
    W = {k: W0[k] + (uv2[k][0] @ uv2[k][1].T - uv[k][0] @ uv[k][1].T) for k in keys}
    lo_w0 = _log_odds(model, toks, half, tgt)
    drop: dict[str, Tensor] = {}
    t0 = time.time()
    with model_state(model, target=W):
        lo = _log_odds(model, toks, half, tgt)
        for i, k in enumerate(keys):  # one matrix at a time, so the first one's time is in the log early
            drop |= revert_one(model, uv, uv2, toks, half, tgt, lo, [k], m=1)
            if i == 0:
                log(f"[carriers] first matrix {k}: {len(drop[k])} subcomponents in {time.time() - t0:.1f} s "
                    f"({len(keys)} matrices)")
    cols = {k: alive[k].nonzero().squeeze(1).cpu() for k in keys}
    flat = torch.cat([drop[k] for k in keys])
    owner = [(k, j) for k in keys for j in range(len(drop[k]))]
    order = flat.argsort(descending=True)
    n50, n90 = _count_to(flat[order], 0.5), _count_to(flat[order], 0.9)
    gain = lo - lo_w0
    rec: dict[str, Any] = {"gain": gain, "n_alive": len(owner), "pos_sum": float(flat.clamp_min(0).sum()),
                           "n50": n50, "n90": n90}
    if n50 is not None:
        sel: dict[str, list[int]] = {}
        low = 0.0
        for i in order[:n50].tolist():
            k, j = owner[i]
            sel.setdefault(k, []).append(j)
            low += float(flat[i]) if float(g_copy[k][cols[k][j]]) < near_zero else 0.0
        rec["n50_near_zero_share"] = low / float(flat[order[:n50]].sum())
        with model_state(model, target=reverted_weights(W, uv, uv2, {k: torch.tensor(v) for k, v in sel.items()})):
            rec["n50_joint_removed"] = (lo - _log_odds(model, toks, half, tgt)) / gain if gain else float("nan")
    rec["top"] = [{"key": owner[i][0], "component": int(cols[owner[i][0]][owner[i][1]]), "drop": float(flat[i]),
                   "g_copy": float(g_copy[owner[i][0]][cols[owner[i][0]][owner[i][1]]]),
                   "g_w0": float(g_w0[owner[i][0]][cols[owner[i][0]][owner[i][1]]])} for i in order[:top].tolist()]
    return rec
