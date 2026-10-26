"""What the turned-down half of re-gating's fixed change does, read at W0 only (E29, `experiments/regating_down`).

Plan: reports/plans/2026-10-22_plan_regating-down.md. E18's fixed change s (mean over the training windows of the updated
model's trigger importance values minus W0's; 0 on dead subcomponents) has a turned-down half that carries as much of
the switching effect as the switched-on set (E18 amendments 2 and 3). This module asks whether the subcomponents it
lowers are the ones W0 uses for the task's competing tokens, and whose probability lowering them moves.

  pool P      alive subcomponents with m0 >= POOL_MIN (on enough at the trigger to be turned down; amendment 3)
  S_down      s <= -DOWN_THR within P;  S_on: s > regating.THR
  competitors the K tokens of largest mean next-token probability at the trigger of the held-out windows under the
              target model at W0 (no masks), the task's target excluded (the one new definition)
  N_comp      top-|S_down| subcomponents within P by W0's Pile contrast for "the next token is a competitor"
  null        same-size-per-matrix sets drawn uniformly from P (S_down's members allowed): the pool-matched null
  patterns    at the trigger, W0's values g0 elsewhere: g0; clip(g0 + s-); s-'s in-pool values permuted within P
Every masked read is the CI-masked model with the released subcomponents and Delta out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import recruit as R

POOL_MIN, DOWN_THR, K = 0.1, 0.1, 5


def load_shift(path: Path) -> dict[str, Tensor]:
    """The fixed change s from an E18 / E24 / E28 file: a dict holding "shift" (with or without "shift_null"), or a bare
    {matrix key: (C,) tensor} dict."""
    d = torch.load(path, weights_only=False)
    s = d.get("shift", d)
    return {k: v.float().cpu() for k, v in s.items()}


def pool_of(m0: dict[str, Tensor], alive: dict[str, Tensor], pool_min: float = POOL_MIN) -> dict[str, Tensor]:
    return {k: alive[k].cpu().bool() & (m0[k].cpu() >= pool_min) for k in alive}


def down_set(shift: dict[str, Tensor], pool: dict[str, Tensor], thr: float = DOWN_THR) -> R.Sel:
    """S_down: members of the pool with s <= -thr."""
    return {k: (pool[k] & (v <= -thr)).nonzero().flatten() for k, v in shift.items()}


def as_sel(mask: dict[str, Tensor]) -> R.Sel:
    return {k: v.nonzero().flatten() for k, v in mask.items()}


def random_in_pool(sel: R.Sel, pool: dict[str, Tensor], seed: int) -> R.Sel:
    """A set with `sel`'s size per matrix drawn uniformly from the pool (members of `sel` allowed), one generator."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k, idx in sel.items():
        cand = pool[k].nonzero().flatten()
        assert idx.numel() <= cand.numel(), f"{k}: set larger than its pool"
        out[k] = cand[torch.randperm(cand.numel(), generator=gen)[: idx.numel()]].sort().values
    return out


def overlap(sel: R.Sel, nat: R.Sel, pool: dict[str, Tensor], n_random: int) -> dict[str, Any]:
    """J(sel, nat) against the pool-matched null; kappa = (J - null mean) / (1 - null mean), the share of the way from
    chance to identical."""
    rnd = [R.jaccard(random_in_pool(sel, pool, i), nat) for i in range(n_random)]
    j, mean = R.jaccard(sel, nat), sum(rnd) / len(rnd)
    sd = (sum((x - mean) ** 2 for x in rnd) / (len(rnd) - 1)) ** 0.5
    return {"J": j, "random": rnd, "random_mean": mean, "random_sd": sd, "random_max": max(rnd),
            "kappa": (j - mean) / (1 - mean) if mean < 1 else float("nan")}


def precede_any(rows: Tensor, tokens: Tensor) -> Tensor:
    """(B, T) bool: True at the positions whose NEXT token is any of `tokens` (the last position is never True)."""
    m = torch.zeros_like(rows, dtype=torch.bool)
    m[:, :-1] = torch.isin(rows[:, 1:], tokens.to(rows.device))
    return m


def contrasts(model: Any, rows: Tensor, masks: dict[str, Tensor], alive: dict[str, Tensor]) -> dict[str, Any]:
    """One pass over Pile rows: per mask, W0's gate contrast (recruit.ContrastAcc, as regating.natural_contrast) and
    its number of positions."""
    from commute.vpd import compute_ci

    accs = {n: R.ContrastAcc() for n in masks}
    for s in range(0, rows.shape[0], 8):
        ci = compute_ci(model, rows[s : s + 8], chunk=8)
        for n, acc in accs.items():
            acc.add(ci.g, masks[n][s : s + 8])
        del ci
    return {n: {"n_positions": acc.n_on, "contrast": acc.result(alive)} for n, acc in accs.items()}


def mean_next_probs(model: Any, windows: Tensor, half: int, batch: int) -> Tensor:
    """(V,) mean over windows of the target model's next-token distribution at the trigger (W0: no masks)."""
    from commute.vpd import compute_ci

    device = next(model.parameters()).device
    tot = None
    with torch.no_grad():
        for s in range(0, windows.shape[0], batch):
            ci = compute_ci(model, windows[s : s + batch].to(device), chunk=batch)
            p = torch.softmax(ci.logits[:, half].float(), -1).double().sum(0).cpu()
            tot = p if tot is None else tot + p
    assert tot is not None
    return tot / windows.shape[0]


def top_competitors(probs: Tensor, target: int, k: int = K) -> tuple[Tensor, Tensor]:
    """(ids, mean probabilities) of the k most probable tokens other than `target`."""
    p = probs.clone()
    p[target] = -1.0
    top = p.topk(k)
    return top.indices, top.values


def mass_of_logits(logits: Tensor, target: int, comp: Tensor) -> dict[str, Tensor]:
    """Per row (B,): log-prob and log-odds of the target, probability of the target, of the competitors summed, and
    of every other token summed (the three sum to 1)."""
    from commute.identity import log_odds

    lp = torch.log_softmax(logits.float(), -1)
    p = lp.exp()
    other = torch.ones(p.shape[-1], dtype=torch.bool, device=p.device)
    other[target] = False
    other[comp.to(p.device)] = False
    return {"logp_target": lp[:, target], "lo_target": log_odds(logits, target), "p_target": p[:, target],
            "p_comp": p[:, comp.to(p.device)].sum(1), "p_rest": p[:, other].sum(1)}


def masses(model: Any, windows: Tensor, half: int, target: int, comp: Tensor, gates: R.Gates,
           batch: int) -> dict[str, float]:
    """`mass_of_logits` at the trigger of the CI-masked model (current components, Delta out) under `gates` (per batch,
    as local_faith.route_cells returns them for the same windows and batch size), averaged over windows."""
    from commute.local_faith import masked_logits

    device = next(model.parameters()).device
    sums: dict[str, float] = {}
    with torch.no_grad():
        for i, g in enumerate(gates):
            b = windows[i * batch : (i + 1) * batch].to(device)
            lg = masked_logits(model, b, {k: v.to(device) for k, v in g.items()}, None, None)[:, half]
            for k, v in mass_of_logits(lg, target, comp).items():
                sums[k] = sums.get(k, 0.0) + float(v.double().sum())
    n = sum(next(iter(g.values())).shape[0] for g in gates)
    assert n == windows.shape[0], "gates and windows disagree"
    return {k: v / n for k, v in sums.items()}


def down_pattern(g0: R.Gates, half: int, shift: dict[str, Tensor]) -> R.Gates:
    """(ii) clip(g0 + s-) at the trigger, g0 elsewhere (E18 amendment 3's 'down')."""
    return R.with_trigger(g0, half, R.plus(R.at_trigger(g0, half), R.signed(shift, False), clip=True))


def shuffled_down(shift: dict[str, Tensor], pool: dict[str, Tensor], seed: int) -> dict[str, Tensor]:
    """s- with its in-pool values permuted among each matrix's pool members (amendment 3's within-pool shuffle); the
    small out-of-pool entries (s >= -m0 > -POOL_MIN there) stay where they are, so (iii) differs from (ii) only by
    which pool member gets which value."""
    down = R.signed(shift, False)
    inside = R.on_pool(down, pool)
    perm = R.shuffled(inside, pool, seed)
    return {k: perm[k] + (down[k] - inside[k]) for k in down}


def shuffled_pattern(g0: R.Gates, half: int, shift: dict[str, Tensor], pool: dict[str, Tensor], seed: int) -> R.Gates:
    """(iii) clip(g0 + shuffled s-) at the trigger, g0 elsewhere."""
    return R.with_trigger(g0, half, R.plus(R.at_trigger(g0, half), shuffled_down(shift, pool, seed), clip=True))


def drop(ref: dict[str, float], pat: dict[str, float]) -> float:
    """drop_comp: the share of the competitors' mass under g0 that the pattern removes."""
    return (ref["p_comp"] - pat["p_comp"]) / ref["p_comp"]
