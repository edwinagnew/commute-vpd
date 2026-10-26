"""The task ladder: trigger -> target tasks mined from Pile at graded baseline probability, their windows, and the
pre-training coverage number (`reports/plans/2026-10-15_plan_task-ladder.md`).

A task is (trigger token t, target token v). Its windows are the 41-token spans centred on occurrences of t in Pile
train rows (same format as the emoticon windows: trigger at index `half`). Its baseline is the mean, over the
occurrences, of the model's next-token probability of v at the trigger. Rungs are decades of that baseline, rung 0
being the most probable ("lower" rungs = behaviours the model already has to a degree; rung 4 = never).

Coverage (the ladder's x-axis, fixed before any training) is the gated gradient fraction: with G_k = dL/dW_k the
gradient of the target's cross-entropy at the trigger, one SGD step of component training moves alive component c by (G v_c, G^T u_c);
its energy is e_c = ||G v_c||^2 + ||G^T u_c||^2; coverage = sum_c gbar_c e_c / sum_c e_c, pooled over the 24 matrices,
where gbar_c is the mean of Gamma's g_c at the trigger over the training windows. It is the share of the first
component-training step's energy that sits in components Gamma gates on at the trigger. Reported, not scored: the number
of alive components with gbar_c > 0.5 at the trigger (L0) and max gbar_c.

Coverage describes a plain-SGD first step, while every ladder arm trained with Adam, so E5's coverage predictions
compare a predictor and an outcome from different optimisers (`reports/reviews/2026-10-21_review_sgd-finding-influence.md`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from commute.data import Windows, target_tokens
from commute.vpd import SEQ_LEN, compute_ci, module_keys

# Rung r holds baselines in [RUNGS[r], RUNGS[r - 1]); rung 0 is [0.1, 1), rung 4 is everything below 1e-4.
RUNG_EDGES = (0.1, 0.01, 1e-3, 1e-4)


def rung_of(baseline: float) -> int:
    for r, edge in enumerate(RUNG_EDGES):
        if baseline >= edge:
            return r
    return len(RUNG_EDGES)


@dataclass
class TriggerStats:
    count: dict[int, int]  # trigger token -> occurrences at valid positions
    mean_p: dict[int, Tensor]  # trigger token -> (V,) mean next-token probability over its occurrences
    background: Tensor | None = None  # (V,) mean next-token probability over every valid position


def valid_positions(half: int) -> tuple[int, int]:
    """Positions t with a full window: half <= t < SEQ_LEN - half."""
    return half, SEQ_LEN - half


def count_tokens(rows: Tensor, half: int) -> dict[int, int]:
    lo, hi = valid_positions(half)
    vals, counts = rows[:, lo:hi].reshape(-1).unique(return_counts=True)
    return dict(zip(vals.tolist(), counts.tolist()))


@torch.no_grad()
def trigger_stats(logits_fn: Any, rows: Tensor, triggers: list[int], half: int, batch: int = 8, device: str = "cpu",
                  log: Any = print) -> TriggerStats:
    """Mean next-token distribution at every valid occurrence of each trigger token, and over every valid position
    (`background`). `logits_fn(toks) -> (B, T, V)`. Sums stay on the logits' device in fp64 (2026-10-22, for a
    pool of hundreds of triggers); positions are mapped to a trigger slot through a
    vocabulary lookup table."""
    lo, hi = valid_positions(half)
    slot = torch.full((int(rows.max()) + 1,), -1, dtype=torch.long)
    slot[torch.tensor(triggers, dtype=torch.long)] = torch.arange(len(triggers))
    sums: Tensor | None = None
    bg: Tensor | None = None
    count = torch.zeros(len(triggers), dtype=torch.long)
    for s in range(0, rows.shape[0], batch):
        toks = rows[s : s + batch]
        flat = torch.softmax(logits_fn(toks.to(device))[:, lo:hi].float(), -1).flatten(0, 1)  # (B T', V)
        if sums is None or bg is None:
            sums = torch.zeros(len(triggers), flat.shape[-1], dtype=torch.float64, device=flat.device)
            bg = torch.zeros(flat.shape[-1], dtype=torch.float64, device=flat.device)
        bg += flat.sum(0, dtype=torch.float64)
        sl = slot[toks[:, lo:hi].reshape(-1)]
        keep = sl >= 0
        if bool(keep.any()):
            sums.index_add_(0, sl[keep].to(flat.device), flat[keep.to(flat.device)].double())
            count.index_add_(0, sl[keep], torch.ones(int(keep.sum()), dtype=torch.long))
        if (s // batch) % 200 == 0:
            log(f"[mine] rows {min(s + batch, rows.shape[0])}/{rows.shape[0]}")
    assert sums is not None and bg is not None
    sums = sums.cpu()
    return TriggerStats(count={t: int(count[i]) for i, t in enumerate(triggers)},
                        mean_p={t: sums[i] / max(int(count[i]), 1) for i, t in enumerate(triggers)},
                        background=bg.cpu() / (rows.shape[0] * (hi - lo)))


def plain_token(decoded: str) -> bool:
    """A word-like token: a leading space then letters only, at least two letters."""
    return len(decoded) >= 3 and decoded[0] == " " and decoded[1:].isalpha()


def plain_mask(decode: Any, vocab: int) -> Tensor:
    """(vocab,) bool: which token ids decode to a word-like token. Computed once; decoding the vocabulary per trigger
    is far too slow."""
    def ok(v: int) -> bool:
        try:
            return plain_token(decode([v]))
        except (IndexError, KeyError, ValueError, TypeError):  # an id past the tokenizer's range (the logit dimension can exceed it)
            return False

    return torch.tensor([ok(v) for v in range(vocab)])


def rung_bounds(r: int) -> tuple[float, float]:
    hi = 1.0 if r == 0 else RUNG_EDGES[r - 1]
    lo = RUNG_EDGES[r] if r < len(RUNG_EDGES) else 0.0
    return lo, hi


def candidates(stats: TriggerStats, trigger: int, plain: Tensor, per_rung: int,
               rungs: tuple[int, ...] = tuple(range(len(RUNG_EDGES) + 1))) -> dict[int, list[tuple[int, float]]]:
    """Per rung, up to `per_rung` word-like targets (token, baseline) with baseline in the rung, closest (in log10)
    to the rung's geometric centre; the trigger itself is excluded. For the last rung (< 1e-4) the centre is 1e-5."""
    p = stats.mean_p[trigger].double()
    assert plain.numel() == p.numel(), f"plain mask over {plain.numel()} ids, probabilities over {p.numel()}"
    ok = plain.clone()
    ok[trigger] = False
    out: dict[int, list[tuple[int, float]]] = {}
    for r in rungs:
        lo, hi = rung_bounds(r)
        m = ok & (p >= lo) & (p < hi) & (p > 0)
        idx = torch.nonzero(m, as_tuple=True)[0]
        if idx.numel() == 0:
            out[r] = []
            continue
        centre = (lo * hi) ** 0.5 if lo > 0 else RUNG_EDGES[-1] / 10
        dist = (torch.log10(p[idx]) - torch.log10(torch.tensor(centre, dtype=torch.float64))).abs()
        order = torch.argsort(dist)[:per_rung]
        out[r] = [(int(idx[i]), float(p[idx[i]])) for i in order]
    return out


def windows_for_trigger(rows: Tensor, trigger: int, half: int, max_n: int, seed: int) -> Windows:
    """Up to max_n windows centred on occurrences of `trigger` at valid positions, sampled without replacement."""
    lo, hi = valid_positions(half)
    r_idx, t_idx = torch.nonzero(rows[:, lo:hi] == trigger, as_tuple=True)
    t_idx = t_idx + lo
    if r_idx.numel() > max_n:
        g = torch.Generator().manual_seed(seed)
        keep = torch.randperm(r_idx.numel(), generator=g)[:max_n]
        r_idx, t_idx = r_idx[keep], t_idx[keep]
    toks = torch.stack([rows[r, t - half : t + half + 1] for r, t in zip(r_idx.tolist(), t_idx.tolist())]) if r_idx.numel() \
        else torch.zeros(0, 2 * half + 1, dtype=torch.long)
    return Windows(toks, torch.zeros(len(r_idx)), r_idx.clone(), t_idx.clone(), half)


def coverage_from(G: dict[str, Tensor], uv: dict[str, tuple[Tensor, Tensor]], gbar: dict[str, Tensor]) -> dict[str, Any]:
    """The gated gradient fraction from per-matrix G (d_out, d_in), alive (U_A (d_out, n), V_A (d_in, n)) and
    gbar (n,) mean g at the trigger. Pooled over matrices by energy; also the per-matrix values."""
    num = den = 0.0
    per: dict[str, float] = {}
    for k, Gk in G.items():
        U, V = uv[k]
        Gk = Gk.double().to(U.device)
        e = (Gk @ V.double()).pow(2).sum(0) + (Gk.T @ U.double()).pow(2).sum(0)  # (n,)
        gk = gbar[k].double().to(e.device)
        n_k, d_k = float((gk * e).sum()), float(e.sum())
        per[k] = n_k / d_k if d_k > 0 else float("nan")
        num, den = num + n_k, den + d_k
    return {"coverage": num / den if den > 0 else float("nan"), "per_matrix": per}


def trigger_gbar(model: Any, uv_alive_index: dict[str, Tensor], toks: Tensor, half: int, chunk: int = 32) -> dict[str, Any]:
    """Mean g of each alive component at the trigger over the windows, plus L0 (> 0.5) and the max."""
    keys = module_keys(model)
    sums = {k: torch.zeros(int(uv_alive_index[k].sum()) if uv_alive_index[k].dtype == torch.bool else uv_alive_index[k].numel(),
                           dtype=torch.float64) for k in keys}
    n = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for s in range(0, toks.shape[0], chunk):
            b = toks[s : s + chunk].to(device)
            ci = compute_ci(model, b, chunk=b.shape[0])
            for k in keys:
                sums[k] += ci.g[k][:, half][:, uv_alive_index[k].to(device)].sum(0, dtype=torch.float64).cpu()
            n += b.shape[0]
            del ci
    gbar = {k: v / n for k, v in sums.items()}
    allg = torch.cat(list(gbar.values()))
    return {"gbar": gbar, "l0_trigger": float((allg > 0.5).sum()), "max_gbar": float(allg.max())}


def trigger_gradient(model: Any, toks: Tensor, half: int, token: int | Tensor, batch: int = 32) -> dict[str, Tensor]:
    """G_k = d/dW_k of the mean cross-entropy of `token` (one id, or a (V,) lookup table read at the trigger token;
    `data.target_tokens`) at the trigger over the windows (fp32, CPU). The target weights are restored to the same
    values; only their .grad is used."""
    from commute.finetune import decomposed_params, logits_of

    params = decomposed_params(model)
    device = next(model.parameters()).device
    for p in params.values():
        p.requires_grad_(True)
        p.grad = None
    try:
        n = 0
        for s in range(0, toks.shape[0], batch):
            b = toks[s : s + batch].to(device)
            with torch.enable_grad():
                lg = logits_of(model.target_model, b)
                loss = F.cross_entropy(lg[:, half], target_tokens(token, b, half), reduction="sum")
                loss.backward()
            n += b.shape[0]
        assert all(p.grad is not None for p in params.values())
        G = {k: (p.grad.detach().float() / n).cpu() for k, p in params.items() if p.grad is not None}
    finally:
        for p in params.values():
            p.grad = None
            p.requires_grad_(False)
    return G
