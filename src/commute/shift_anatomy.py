"""What the fixed change of Gamma's importance values is made of (E32, `experiments/shift_anatomy`).

Plan: reports/2026-10-22_plan_shift-anatomy.md. Definitions as E28 (`commute.regating`): every pattern is W0's values
everywhere except the trigger position, scored as recovery rho = (lo - L00) / gap and as R = rho / rho_trig.

Per-window reads at the trigger (`trigger_read`) are (N, C) or (N, d_in) per matrix, fp32 on CPU; `to_trig` cuts them
into the (B, C) batches of `regating.BATCH` that the pattern builders (`recruit.with_trigger`) take. Arms:
  group     per-group fixed change: m1_k, m0_k = mean trigger values over the training windows of group k (the key at
            the trigger, or a random label); each held-out window gets its group's vector (net: m1_k minus m0_k)
  strength  pre-activations p (g = clip(p, 0, 1), the released `leaky_hard`): d(x) = p1(x) - p0(x), s_p the training
            mean of d, a(x) = <d, s_p> / <s_p, s_p> over alive subcomponents; pattern clip(p0(x) + a(x) s_p)
  mixed     one matrix type at a fixed vector, the other at each window's own updated values
  steer     Gamma recomputed with each matrix's input at the trigger moved by the mean (updated - original) input
            over the training windows (Gamma is a bidirectional transformer: other positions keep W0's inputs)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from commute import recruit as R
from commute.data import target_tokens
from commute.regating import BATCH, Task, div, recovery
from commute.vpd import module_keys

Vals = dict[str, Tensor]  # matrix key -> (N, ·) per window, fp32 CPU
Edit = Callable[[int, dict[str, Tensor]], None]  # (first window index of the batch, Gamma's input cache) -> None
PREV_TOKEN = (("h.1.attn.q_proj", 316), ("h.1.attn.k_proj", 329))  # VPD blog post, "Previous token behavior"


def is_attention(key: str) -> bool:
    return ".attn." in key


@torch.no_grad()
def trigger_read(model: Any, tokens: Tensor, half: int, target: int | Tensor | None = None, edit: Edit | None = None,
                 batch: int = BATCH) -> dict[str, Any]:
    """Under the model's current state, per window at the trigger: Gamma's values `g`, its pre-activations `pre`, each
    matrix's input `x` (read before `edit`), and the target model's log-prob of the target `logp` (if `target`).
    The path of `vpd.compute_ci` with one forward per `batch` windows, as `local_faith.route_cells` runs it; `edit`
    may replace cached inputs before Gamma reads them."""
    keys = module_keys(model)
    device = next(model.parameters()).device
    parts: dict[str, dict[str, list[Tensor]]] = {n: {k: [] for k in keys} for n in ("g", "pre", "x")}
    logp: list[Tensor] = []
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for s in range(0, tokens.shape[0], batch):
            sub = tokens[s : s + batch].to(device)
            out = model(sub, cache_type="input")
            for k in keys:
                parts["x"][k].append(out.cache[k][:, half].float().cpu())
            if edit is not None:
                edit(s, out.cache)
            ci = model.calc_causal_importances(out.cache, sampling="continuous", detach_inputs=False)
            for k in keys:
                parts["g"][k].append(ci.lower_leaky[k][:, half].float().cpu())
                parts["pre"][k].append(ci.pre_sigmoid[k][:, half].float().cpu())
            if target is not None:
                lp = torch.log_softmax(out.output[:, half].float(), -1)
                logp.append(lp.gather(1, target_tokens(target, sub, half)[:, None]).squeeze(1).double().cpu())
    res: dict[str, Any] = {n: {k: torch.cat(v) for k, v in p.items()} for n, p in parts.items()}
    res["logp"] = torch.cat(logp) if logp else None
    return res


def add_at_trigger(half: int, delta: dict[str, Tensor]) -> Edit:
    """An `edit` that adds one (d_in,) vector per matrix to every window's input at the trigger (dtype kept)."""
    def edit(_start: int, cache: dict[str, Tensor]) -> None:
        for k, d in delta.items():
            x = cache[k].clone()  # q, k and v may share one cached tensor: never write in place
            x[:, half] = (x[:, half].float() + d.to(x.device)).to(x.dtype)
            cache[k] = x
    return edit


def set_at_trigger(half: int, x_new: Vals) -> Edit:
    """An `edit` that writes each window's own (d_in,) input from `x_new` (N, d_in) at the trigger."""
    def edit(start: int, cache: dict[str, Tensor]) -> None:
        for k, v in x_new.items():
            x = cache[k].clone()
            x[:, half] = v[start : start + x.shape[0]].to(x.device, x.dtype)
            cache[k] = x
    return edit


# ----------------------------------------------------------------------------- pure bookkeeping


def to_trig(vals: Vals, batch: int = BATCH) -> R.TrigVals:
    """(N, C) per matrix -> per-batch dicts of (B, C), the layout of `recruit.at_trigger`."""
    n = next(iter(vals.values())).shape[0]
    return [{k: v[s : s + batch] for k, v in vals.items()} for s in range(0, n, batch)]


def mean_of(vals: Vals) -> dict[str, Tensor]:
    return {k: v.double().mean(0).float() for k, v in vals.items()}


def group_means(vals: Vals, labels: Tensor) -> dict[int, dict[str, Tensor]]:
    """Per label, the mean over its windows of each matrix's (C,) values."""
    return {int(lab): mean_of({k: v[labels == lab] for k, v in vals.items()}) for lab in labels.unique()}


def per_window(means: dict[int, dict[str, Tensor]], labels: Tensor) -> Vals:
    """Each window gets its label's vector: (N, C) per matrix."""
    keys = next(iter(means.values())).keys()
    return {k: torch.stack([means[int(lab)][k] for lab in labels]) for k in keys}


def random_labels(n: int, groups: int, seed: int) -> Tensor:
    return torch.randint(0, groups, (n,), generator=torch.Generator().manual_seed(seed))


def flat(vals: dict[str, Tensor], alive: dict[str, Tensor]) -> Tensor:
    """The alive entries of every matrix, concatenated in key order along the last dim."""
    return torch.cat([vals[k][..., alive[k].cpu().bool()] for k in sorted(vals)], -1)


def along(d: Vals, s: dict[str, Tensor], alive: dict[str, Tensor]) -> tuple[Tensor, float]:
    """a(x) = <d(x), s> / <s, s> over alive subcomponents, and the fraction of d's variance across windows (about its
    mean) that lies along s."""
    D, S = flat(d, alive).double(), flat(s, alive).double()
    a = D @ S / (S @ S)
    C = D - D.mean(0)
    frac = float(((C @ S) ** 2).sum() / (S @ S) / (C**2).sum())
    return a.float(), frac


def strength(p0: Vals, a: Tensor, s: dict[str, Tensor], alive: dict[str, Tensor]) -> Vals:
    """clip(p0(x) + a(x) s, 0, 1) on alive subcomponents; dead ones keep clip(p0)."""
    return {k: (v + a[:, None] * torch.where(alive[k].cpu().bool(), s[k], 0.0)).clamp(0.0, 1.0) for k, v in p0.items()}


def mixed(a: Vals, b: Vals, take_a: Callable[[str], bool]) -> Vals:
    return {k: (a[k] if take_a(k) else b[k]) for k in a}


def pearson(x: Tensor, y: Tensor) -> float:
    x, y = x.double() - x.double().mean(), y.double() - y.double().mean()
    den = float(x.norm() * y.norm())
    return float(x @ y) / den if den > 0 else float("nan")


def key_match(a: dict[int, dict[str, Tensor]], b: dict[int, dict[str, Tensor]], alive: dict[str, Tensor],
              n_draws: int) -> dict[str, Any]:
    """Mean over keys of corr(a_k - mean_k a, b_k - mean_k b) over alive subcomponents, against the same with b's keys
    deranged (`recruit.derangement`, seeds 0..n_draws-1)."""
    keys = sorted(a)
    A = torch.stack([flat(a[k], alive) for k in keys]).double()
    B = torch.stack([flat(b[k], alive) for k in keys]).double()
    A, B = A - A.mean(0), B - B.mean(0)

    def score(perm: Tensor) -> float:
        return sum(pearson(A[i], B[int(j)]) for i, j in enumerate(perm)) / len(keys)

    null = [score(R.derangement(len(keys), i)) for i in range(n_draws)]
    return {"own": score(torch.arange(len(keys))), "null": null, "null_mean": sum(null) / len(null),
            "null_max": max(null)}


# ----------------------------------------------------------------------------- arms on the model


def rho_of(model: Any, task: Task, vals: Vals, gap: float) -> float:
    return recovery(model, task, to_trig(vals), gap)


def group_fixed(model: Any, task: Task, g1_train: Vals, g0_train: Vals, lab_train: Tensor, lab_held: Tensor,
                gap: float, rho_trig: float) -> dict[str, Any]:
    """Each held-out window given its group's updated mean m1_k, and the reference m0_k: R_net = (rho(m1_k) -
    rho(m0_k)) / rho_trig (E28's net fixed arm, per group). Every held-out label must occur in training."""
    missing = set(lab_held.tolist()) - set(lab_train.tolist())
    assert not missing, f"held-out groups without training windows: {sorted(missing)}"
    m1, m0 = group_means(g1_train, lab_train), group_means(g0_train, lab_train)
    r1, r0 = (rho_of(model, task, per_window(m, lab_held), gap) for m in (m1, m0))
    counts = torch.bincount(lab_train - lab_train.min())
    counts = counts[counts > 0]
    return {"rho": r1, "rho_W0": r0, "R_net": div(r1 - r0, rho_trig), "n_groups": len(m1),
            "train_per_group_min": int(counts.min()), "train_per_group_median": float(counts.float().median()),
            "m1": m1, "m0": m0}


def per_window_lo(model: Any, task: Task, vals: Vals) -> Tensor:
    from commute.local_faith import masked_lo_windows

    g0, h = task.base["gates"], task.half
    return masked_lo_windows(model, task.held.tokens, h, task.target, R.with_trigger(g0, h, to_trig(vals)), BATCH)


def steer(model: Any, task: Task, edit: Edit, gap: float) -> float:
    """rho of Gamma's trigger values recomputed (W0's model) with the inputs edited at the trigger."""
    g = trigger_read(model, task.held.tokens, task.half, edit=edit)["g"]
    return rho_of(model, task, g, gap)
