"""P1.31's four hand-built edits of the 67M, whose labels are known by construction (recipes:
`reports/plans/2026-11-18_plan_known-answer-levels_appendix.md` §2, kept unchanged by P1.35's plan). Added 2026-11-19
for `experiments/kstar_levels` (E214). Nothing is trained.

Cues: eight A_i drawn from E62's copy keys, and eight A'_i, eight answers y_i and the B pool drawn from the frequent
word-like tokens of E62's pool ranks outside the keys. D: per cue, Pile windows of `window` tokens with A_i B written at
p, p + 1 and A'_i last, A_i, B and A'_i once each and y_i absent; read at the last position. The twin windows put A_i
last instead: M's own copy there is the reference for the copy bar.

Slots: the eight layer-`l` neurons with the smallest mean activation^2 x |output column|^2 on Pile held-out windows
(E206's rule). Slot i's input row detects A'_i: beta (h(A'_i) - mean h) / |.|, h the mean MLP input after rms_2 at the
positions holding the token in M's run on D, beta the smallest power of 2 at which the slot's mean activation on A'_i
is at least `detector_ratio` x its `detector_quantile` quantile on Pile held-out.
    reuse_small  slot i's output column gamma_T (x_1(A_i) - x_1(A'_i)) / act_i, x_1 the mean block-1 input at the token's
                 positions; gamma_T from 1, doubled until the copy top-1 on D is at least copy_bar_frac x M's own
    new_few      slot i's output column gamma_N e(y_i) / act_i, e(y) = lm_head[y] - mean lm_head; gamma_N the smallest
                 value (doubling, then 20 bisection steps) with lookup top-1 on D at least lookup_bar
    new_thin     the least-norm change of layer l's down_proj, gamma E A^+, A (n x 8) M's mean activations on each A'_i,
                 E (d x 8) the e(y_i); gamma found as gamma_N
    reuse_large  reuse_small with head `gain_head`'s o_proj columns times g, g solving |dW|_F = max(|dW_few|, |dW_thin|);
                 when that needs g <= 1, |dW|_F = g_fallback_ratio x |dW_small| and the match is reported as not met
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from commute import rule_retention as R


@dataclass
class Cues:
    A: list[int]
    A2: list[int]  # A'_i
    y: list[int]
    toks: Tensor  # (N, T) D, A'_i last
    twin: Tensor  # D with A_i last
    cue: Tensor  # (N,) the cue of each window
    b: Tensor  # (N,) its B
    p: Tensor  # (N,) where its A_i is


def cues(rows: Tensor, keys: list[int], pool: list[int], n: int, per: int, T: int, seed: int) -> Cues:
    """`pool`: frequent word-like tokens; the keys are removed from it before the draw."""
    gen = torch.Generator().manual_seed(seed)
    A = [keys[int(i)] for i in torch.randperm(len(keys), generator=gen)[:n]]
    rest = [t for t in pool if t not in keys]
    rest = [rest[int(i)] for i in torch.randperm(len(rest), generator=gen)]
    A2, y, bs = rest[:n], rest[n : 2 * n], torch.tensor(rest[2 * n :])
    out: dict[str, list[Tensor]] = {"toks": [], "cue": [], "b": [], "p": []}
    for i in range(n):
        m = 8 * per
        r, s = torch.randint(0, rows.shape[0], (m, 1), generator=gen), torch.randint(0, rows.shape[1] - T + 1, (m, 1), generator=gen)
        t = rows[r, s + torch.arange(T)].clone()
        p, b = torch.randint(2, T - 3, (m,), generator=gen), bs[torch.randint(len(bs), (m,), generator=gen)]
        t[torch.arange(m), p], t[torch.arange(m), p + 1], t[:, -1] = A[i], b, A2[i]
        ok = ((t == A[i]).sum(1) == 1) & ((t == b[:, None]).sum(1) == 1) & ((t == A2[i]).sum(1) == 1) & ((t == y[i]).sum(1) == 0)
        keep = ok.nonzero().squeeze(1)[:per]
        assert len(keep) == per, f"cue {i}: only {int(ok.sum())} of {m} windows hold"
        for k, v in (("toks", t[keep]), ("cue", torch.full((per,), i)), ("b", b[keep]), ("p", p[keep])):
            out[k].append(v)
    toks, cue = torch.cat(out["toks"]), torch.cat(out["cue"])
    twin = toks.clone()
    twin[:, -1] = torch.tensor(A)[cue]
    return Cues(A, A2, y, toks, twin, cue, torch.cat(out["b"]), torch.cat(out["p"]))


def clone(M: R.Model) -> R.Model:
    return R.Model(copy.deepcopy(M.net).requires_grad_(False))


@torch.no_grad()
def top1(N: R.Model, toks: Tensor, target: Tensor, batch: int = 128) -> float:
    hit = [N.logits(t)[:, -1].argmax(-1) == y for t, y in zip(toks.split(batch), target.to(toks.device).split(batch), strict=True)]
    return float(torch.cat(hit).double().mean())


@torch.no_grad()
def slots(M: R.Model, pile: Tensor, l: int, n: int) -> list[int]:
    energy = sum(R.acts(M, l, R.m_inputs(M, t)[l][1]).double().pow(2).sum((0, 1)) for t in pile.split(64))
    assert isinstance(energy, Tensor)
    return sorted((energy * R.cols(M, l).double().pow(2).sum(0)).argsort()[:n].tolist())


@torch.no_grad()
def site(M: R.Model, toks: Tensor, l: int) -> tuple[Tensor, Tensor]:
    """(layer l's MLP input after rms_2, block 1's input) in M's run, (N, T, d) each."""
    ins = R.m_inputs(M, toks)
    return M.net.h[l].rms_2(ins[l][1]), ins[1][0]


def dw_norm(F: R.Model, M: R.Model) -> float:
    return math.sqrt(sum(float((a - b).double().pow(2).sum()) for a, b in zip(F.net.parameters(), M.net.parameters(), strict=True)))


def smallest(ok: Callable[[float], bool], cap: float = 2.0**20, steps: int = 20) -> float | None:
    """The smallest gamma > 0 with ok(gamma), ok assumed monotone: doubling from 1, then bisection."""
    hi = 1.0
    while not ok(hi):
        hi *= 2
        if hi > cap:
            return None
    lo = hi / 2 if hi > 1 else 0.0
    for _ in range(steps):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


@torch.no_grad()
def with_slots(M: R.Model, l: int, idx: list[int], rows: Tensor, cols: Tensor) -> R.Model:
    F = clone(M)
    F.net.h[l].mlp.c_fc.weight[idx] = rows
    F.net.h[l].mlp.down_proj.weight[:, idx] = cols
    return F


@torch.no_grad()
def build(M: R.Model, c: Cues, pile: Tensor, cfg: Any) -> tuple[dict[str, R.Model], dict[str, Any]]:
    """The four edits and the record of every build check; rec["misses"] lists the checks missed (each a stop)."""
    l, dev = cfg.slot_layer, M.net.wte.weight.device
    toks, cue = c.toks.to(dev), c.cue.to(dev)
    S = slots(M, pile.to(dev), l, cfg.n_cues)
    h, x1 = site(M, toks, l)
    hp = M.net.h[l].rms_2(torch.cat([R.m_inputs(M, t)[l][1] for t in pile.to(dev).split(64)])).flatten(0, 1)
    gelu, n = M.net.h[l].mlp.gelu, cfg.n_cues
    rows, betas, act = torch.zeros(n, h.shape[-1], device=dev), [], torch.zeros(n, device=dev)
    for i in range(n):
        u = h[cue == i, -1].mean(0) - h.mean((0, 1))
        u, beta = u / u.norm(), 1.0
        while float(gelu(h[cue == i, -1] @ (beta * u)).mean()) < cfg.detector_ratio * float(torch.quantile(gelu(hp @ (beta * u)), cfg.detector_quantile)):
            beta *= 2
            assert beta <= 2.0**20, f"no detector for cue {i}"
        rows[i], act[i] = beta * u, gelu(h[cue == i, -1] @ (beta * u)).mean()
        betas.append(beta)
    pos = torch.arange(len(toks), device=dev)
    shift = torch.stack([x1[pos[cue == i], c.p.to(dev)[cue == i]].mean(0) - x1[cue == i, -1].mean(0) for i in range(n)], 1) / act
    lm = M.net.lm_head.weight
    E = (lm[torch.tensor(c.y, device=dev)] - lm.mean(0)).T  # (d, n)
    y_of = torch.tensor(c.y, device=dev)[cue]
    ref, base = top1(M, c.twin.to(dev), c.b), top1(M, toks, c.b)
    rec: dict[str, Any] = {"slots": S, "betas": betas, "act": act.tolist(), "copy_M_twin": ref, "copy_M_D": base, "misses": []}
    rec["misses"] += [f"M's copy after A' {base:.4f} > {cfg.a_prime_copy_max}"] if base > cfg.a_prime_copy_max else []
    gT = 1.0
    while top1(small := with_slots(M, l, S, rows, gT * shift), toks, c.b) < cfg.copy_bar_frac * ref and gT <= cfg.gamma_t_max:
        gT *= 2
    rec["misses"] += [f"gamma_T {gT} > {cfg.gamma_t_max}"] if gT > cfg.gamma_t_max else []
    few_of = lambda g: with_slots(M, l, S, rows, g * E / act)
    A = torch.stack([R.acts(M, l, R.m_inputs(M, toks)[l][1])[cue == i, -1].mean(0) for i in range(n)], 1)  # (n_mlp, n)
    pinv = torch.linalg.pinv(A.double()).float()

    def thin_of(g: float) -> R.Model:
        F = clone(M)
        F.net.h[l].mlp.down_proj.weight += g * E @ pinv
        return F

    edits = {"reuse_small": small}
    for name, make in (("new_few", few_of), ("new_thin", thin_of)):
        gam = smallest(lambda v, make=make: top1(make(v), toks, y_of) >= cfg.lookup_bar)
        rec[f"gamma_{name}"] = gam
        if gam is None:
            rec["misses"].append(f"{name}: no gamma reaches lookup {cfg.lookup_bar}")
            continue
        edits[name] = make(gam)
    rec["gamma_T"] = gT
    norms = {k: dw_norm(F, M) for k, F in edits.items()}
    Wo = M.net.h[cfg.gain_head[0]].attn.o_proj.weight[:, cfg.gain_head[1] * R.D_HEAD : (cfg.gain_head[1] + 1) * R.D_HEAD]
    S2, T2 = norms["reuse_small"] ** 2, max(norms.get("new_few", 0.0), norms.get("new_thin", 0.0)) ** 2
    matched = T2 > S2
    g = 1 + math.sqrt((T2 - S2) if matched else (cfg.g_fallback_ratio**2 - 1) * S2) / float(Wo.norm())
    large = clone(small)
    large.net.h[cfg.gain_head[0]].attn.o_proj.weight[:, cfg.gain_head[1] * R.D_HEAD : (cfg.gain_head[1] + 1) * R.D_HEAD] *= g
    edits["reuse_large"], norms["reuse_large"] = large, dw_norm(large, M)
    rec |= {"g": g, "size_matched": matched, "dW_norm": norms}
    for k, F in edits.items():
        rec[f"top1_{k}"] = top1(F, toks, c.b if k.startswith("reuse") else y_of)
        bar = cfg.copy_bar_frac * ref if k.startswith("reuse") else cfg.lookup_bar
        rec["misses"] += [f"{k}: top-1 {rec[f'top1_{k}']:.4f} < {bar:.4f}"] if rec[f"top1_{k}"] < bar else []
    return {k: edits[k] for k in ("reuse_small", "reuse_large", "new_few", "new_thin") if k in edits}, rec
