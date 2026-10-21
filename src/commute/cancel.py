"""Cancellation tests (E17, reports/2026-10-21_plan_cancellation.md): do the per-subcomponent changes a split or component
training writes cancel each other?

A state gives, per matrix, the alive subcomponents before (U, V) and after (U2, V2) in theory layout (U (d_out, n), V
(d_in, n)); subcomponent c's change is Delta_c = u2_c v2_c^T - u_c v_c^T and the target weight moved by sum_c Delta_c
(exactly for component training and the adjoint state; the least-squares state also moved Delta by the remainder).

Two readings of cancellation:
  coherence   ||sum_c Delta_c||^2 / sum_c ||Delta_c||^2 (Frobenius): orthogonal changes read 1, aligned read their
              count, cancelling read below 1; the null flips the sign of each Delta_c at random and reads about 1.
  revert KL   the state's own logits against a forward in which fraction w_c(x) of each Delta_c is removed at every
              position, w = 1 for all c being the full revert, a random {0, 1} half the half revert, and
              (1 - g_c(x)) (1 - a_c) the fraction the adversary's mask (g + (1 - g) a) removes. Done with a forward hook
              on the target linear (`Revert`), so W0's own subcomponents are never touched.
The complement revert (E20, section at the end): the same removal applied to the original subcomponents with the changes
kept, the dead subcomponents and Delta', through `Remove` hooks whose blocks reproduce the adversary's masked forward.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from commute import vpd
from commute.attribution import fwd
from commute.local_faith import masked_logits

Pieces = dict[str, tuple[Tensor, Tensor, Tensor, Tensor]]  # per matrix (U, V, U2, V2), float32, on the model's device


def change_energies(U: Tensor, V: Tensor, U2: Tensor, V2: Tensor) -> Tensor:
    """||Delta_c||_F^2 per subcomponent from inner products: ||u2||^2 ||v2||^2 + ||u||^2 ||v||^2 - 2 (u2 . u)(v2 . v)."""
    U, V, U2, V2 = U.double(), V.double(), U2.double(), V2.double()
    return ((U2 * U2).sum(0) * (V2 * V2).sum(0) + (U * U).sum(0) * (V * V).sum(0)
            - 2 * (U2 * U).sum(0) * (V2 * V).sum(0)).clamp_min(0)


def signed_sum(U: Tensor, V: Tensor, U2: Tensor, V2: Tensor, s: Tensor | None = None) -> Tensor:
    """sum_c s_c Delta_c as a (d_out, d_in) float64 matrix; s = None means all ones (the state's weight change)."""
    U, V, U2, V2 = U.double(), V.double(), U2.double(), V2.double()
    if s is None:
        return U2 @ V2.T - U @ V.T
    s = s.double()
    return (U2 * s) @ V2.T - (U * s) @ V.T


def coherence(pieces: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]], n_null: int = 3, seed: int = 0) -> dict[str, Any]:
    """Per matrix and pooled (sum of numerators over sum of denominators): the coherence and its sign-flip null
    (mean over `n_null` draws), with the per-matrix numerator and denominator."""
    g = torch.Generator().manual_seed(seed)
    per: dict[str, dict[str, float]] = {}
    num = den = 0.0
    null_num = [0.0] * n_null
    for k, (U, V, U2, V2) in pieces.items():
        U, V, U2, V2 = (t.cpu() for t in (U, V, U2, V2))
        d = float(change_energies(U, V, U2, V2).sum())
        n_k = float((signed_sum(U, V, U2, V2) ** 2).sum())
        nulls = []
        for i in range(n_null):
            s = torch.randint(0, 2, (U.shape[1],), generator=g).double() * 2 - 1
            nn_k = float((signed_sum(U, V, U2, V2, s) ** 2).sum())
            nulls.append(nn_k)
            null_num[i] += nn_k
        per[k] = {"kappa": n_k / d if d > 0 else float("nan"), "null": sum(nulls) / n_null / d if d > 0 else float("nan"),
                  "num": n_k, "den": d}
        num, den = num + n_k, den + d
    return {"pooled": num / den, "pooled_null": sum(null_num) / n_null / den,
            "pooled_null_draws": [x / den for x in null_num], "per_matrix": per}


class Revert:
    """Forward hook on one target linear: out -= sum_c w_c (x . v2_c) u2_c - w_c (x . v_c) u_c, i.e. fraction w_c of
    Delta_c removed. `w` is (n,) or (B, T, n) in [0, 1] on the device; set it before each forward."""

    def __init__(self, U: Tensor, V: Tensor, U2: Tensor, V2: Tensor):
        self.U, self.V, self.U2, self.V2 = U, V, U2, V2
        self.w: Tensor | None = None

    def __call__(self, _m: nn.Module, args: tuple[Any, ...], out: Tensor) -> Tensor:
        assert self.w is not None
        x = args[0].to(self.U.dtype)
        w = self.w.to(x.dtype)
        sub = ((x @ self.V2) * w) @ self.U2.T - ((x @ self.V) * w) @ self.U.T
        return out - sub.to(out.dtype)


class Reverter:
    """The hooks for every matrix of `pieces` on the model's target linears; `logits(batch, weights)` runs the fp32 target
    forward with weights[k] set on each hook (a Tensor (n,) or (B, T, n), or a float applied to every matrix)."""

    def __init__(self, model: Any, pieces: Pieces):
        self.model = model
        self.hooks = {k: Revert(*p) for k, p in pieces.items()}
        self.handles = [vpd.target_linear(model, k).register_forward_hook(h) for k, h in self.hooks.items()]

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    @torch.no_grad()
    def logits(self, batch: Tensor, weights: dict[str, Tensor] | float) -> Tensor:
        for k, h in self.hooks.items():
            n = h.U.shape[1]
            h.w = torch.full((n,), float(weights), device=h.U.device) if isinstance(weights, (int, float)) else weights[k]
        return fwd(self.model.target_model, batch)


Cached = list[tuple[Tensor, Tensor, dict[str, Tensor]]]  # (batch, the state's fp32 logits, alive g per matrix (B, T, n))


@torch.no_grad()
def cache_state(model: Any, batches: list[Tensor], alive: dict[str, Tensor]) -> Cached:
    """The state's own fp32 logits and Gamma's g at the alive subcomponents, per batch, under the model's current state.
    Taken before any hook is attached."""
    device = next(model.parameters()).device
    out: Cached = []
    for b in batches:
        b = b.to(device)
        ci = vpd.compute_ci(model, b, chunk=b.shape[0])
        g = {k: ci.g[k][..., alive[k].to(device)].float() for k in alive}
        out.append((b, fwd(model.target_model, b), g))
        del ci
    return out


@torch.no_grad()
def revert_kl(rev: Reverter | Remover, cached: Cached, weights_of: Any) -> float:
    """Mean KL per position, over the cached batches, between the state's logits and the hooked forward with the weights
    `weights_of(g)` (g the batch's alive gates per matrix; returns a dict or a float)."""
    tot = n = 0.0
    for batch, ref, g in cached:
        lg = rev.logits(batch, weights_of(g))
        tot += float(vpd.kl_per_position(ref, lg).double().sum())
        n += batch.shape[0] * batch.shape[1]
    return tot / n


def half_weights(pieces: Pieces, seed: int) -> dict[str, Tensor]:
    """A random half of the alive subcomponents of each matrix reverted (w = 1), the rest kept (w = 0)."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, (U, _, _, _) in pieces.items():
        n = U.shape[1]
        w = torch.zeros(n)
        w[torch.randperm(n, generator=g)[: n // 2]] = 1.0
        out[k] = w.to(U.device)
    return out


def adversary_weights(sources: dict[str, Tensor], alive: dict[str, Tensor], device: str | torch.device,
                      perm_seed: int | None = None) -> Any:
    """weights_of for `revert_kl`: fraction (1 - g_c(x)) (1 - a_c) of each change removed, a the adversary's source of the
    alive subcomponent (sources[k] is (C + 1,): the subcomponents then Delta). With `perm_seed`, a is permuted among the
    alive subcomponents of each matrix (the null for the adversary's pattern)."""
    gen = torch.Generator().manual_seed(perm_seed) if perm_seed is not None else None
    a = {}
    for k, s in sources.items():
        ak = s[:-1][alive[k].cpu()].float()
        if gen is not None:
            ak = ak[torch.randperm(ak.numel(), generator=gen)]
        a[k] = ak.to(device)

    def weights_of(g: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: (1 - g[k]) * (1 - a[k]) for k in g}

    return weights_of


def _mix(U2: Tensor, V2: Tensor, E: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    A = torch.eye(E.shape[0], dtype=torch.float64) + eps * E
    return U2 @ A, V2 @ torch.linalg.inv(A).T


def gauge_mixed(pieces: Pieces, energy_ratio: float, seed: int, steps: int = 40) -> Pieces:
    """The positive control: the same weight change written in mixed pieces. Per matrix U2 <- U2 A, V2 <- V2 A^-T with
    A = I + eps E (E Gaussian, scaled by 1/sqrt(n)), so U2 V2^T is unchanged exactly while subcomponent c's change gains
    parts that other subcomponents lose. eps is found by bisection so that sum_c ||Delta_c||^2 is `energy_ratio` times
    the unmixed value (the coherence then reads the unmixed value / energy_ratio). All in float64 on the CPU."""
    g = torch.Generator().manual_seed(seed)
    out: Pieces = {}
    for k, (U, V, U2, V2) in pieces.items():
        Ud, Vd, U2d, V2d = (t.double().cpu() for t in (U, V, U2, V2))
        n = U2d.shape[1]
        E = torch.randn(n, n, generator=g, dtype=torch.float64) / n ** 0.5
        target = energy_ratio * float(change_energies(Ud, Vd, U2d, V2d).sum())

        lo, hi = 1e-6, 1.0
        while float(change_energies(Ud, Vd, *_mix(U2d, V2d, E, hi)).sum()) < target:
            hi *= 2
            assert hi < 1e6, k
        for _ in range(steps):
            mid = (lo * hi) ** 0.5
            if float(change_energies(Ud, Vd, *_mix(U2d, V2d, E, mid)).sum()) < target:
                lo = mid
            else:
                hi = mid
        U3, V3 = _mix(U2d, V2d, E, hi)
        out[k] = (U, V, U3.float().to(U2.device), V3.float().to(V2.device))
    return out


# ----------------------------------------------------------------------------- the complement revert (E20)
# reports/2026-10-21_plan_cancellation-complement.md. The adversary's masked forward replaces each target linear's output
# W' x by sum_c m_c (x . v'_c) u'_c + m_D Delta' x over all C subcomponents, m = g + (1 - g) a and m_D = a[-1], so it
# removes sum_c (1 - g_c)(1 - a_c) (x . v'_c) u'_c + (1 - a[-1]) Delta' x. Per matrix that removal is four blocks: the
# alive originals u_c v_c^T, the alive new pieces u'_c v'_c^T (new minus originals = the changes), the dead
# subcomponents and Delta'. `Remove` takes blocks of rank-one terms, each block under its own weight.

ORIG, NEW, DEAD, DENSE = range(4)  # the block order of `state_blocks`
Block = tuple[Tensor, Tensor | None]  # (U (d_out, n), V (d_in, n)); V None: U is one dense (d_out, d_in) term
Weight = Tensor | float  # (n,), (B, T, n), or a float (a dense block's only form); 0.0 skips the block
Bins = dict[str, dict[str, Tensor]]  # bin name -> matrix -> (n,) bool over the alive subcomponents; absent matrix: none


class Remove:
    """Forward hook on one target linear: out -= sum_b ((x @ V_b) * w_b) @ U_b^T (x * w_b @ U_b^T for a dense block),
    i.e. fraction w_b of each term of block b removed. Set `w`, one weight per block, before each forward. E17's
    `Revert(U, V, U2, V2)` at weight w is blocks [(U, V), (U2, V2)] at [-w, w]."""

    def __init__(self, blocks: list[Block]):
        self.blocks = blocks
        self.w: list[Weight] | None = None

    def __call__(self, _m: nn.Module, args: tuple[Any, ...], out: Tensor) -> Tensor:
        assert self.w is not None and len(self.w) == len(self.blocks)
        x = args[0].float()
        sub = torch.zeros(out.shape, dtype=torch.float32, device=out.device)
        for (U, V), w in zip(self.blocks, self.w):
            if isinstance(w, float) and w == 0.0:
                continue
            sub = sub + ((x if V is None else x @ V) * w) @ U.T
        return out - sub.to(out.dtype)


class Remover:
    """`Remove` hooks for every matrix of `blocks` on the model's target linears. `logits(batch, weights)` runs the fp32
    target forward with weights[k] (a list, one per block; missing trailing blocks and missing matrices weigh 0)."""

    def __init__(self, model: Any, blocks: dict[str, list[Block]]):
        self.model = model
        self.hooks = {k: Remove(b) for k, b in blocks.items()}
        self.handles = [vpd.target_linear(model, k).register_forward_hook(h) for k, h in self.hooks.items()]

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    @torch.no_grad()
    def logits(self, batch: Tensor, weights: dict[str, list[Weight]]) -> Tensor:
        for k, h in self.hooks.items():
            w = list(weights.get(k, []))
            h.w = w + [0.0] * (len(h.blocks) - len(w))
        return fwd(self.model.target_model, batch)


def state_blocks(model: Any, alive: dict[str, Tensor], pieces: Pieces) -> dict[str, list[Block]]:
    """Per matrix [alive originals (U, V), alive new pieces (U2, V2), dead subcomponents, Delta'], in that order (ORIG,
    NEW, DEAD, DENSE), under the model's current components and target weights (set to the state). `pieces` covers
    every matrix, with U2 = U, V2 = V where the state left it unchanged."""
    wd = vpd.weight_deltas(model)
    out: dict[str, list[Block]] = {}
    for k in vpd.module_keys(model):
        U, V, U2, V2 = pieces[k]
        Uf, Vf = vpd.uv(model, k)
        dead = ~alive[k].to(Uf.device)
        out[k] = [(U, V), (U2, V2), (Uf.T[:, dead].float().contiguous(), Vf[:, dead].float().contiguous()),
                  (wd[k].float(), None)]
    return out


@torch.no_grad()
def cache_adversary(model: Any, batches: list[Tensor], alive: dict[str, Tensor], sources: dict[str, Tensor]
                    ) -> tuple[Cached, float, dict[str, Any]]:
    """As `cache_state` (the state's fp32 logits, g at the alive subcomponents), and in the same pass F_all: the mean KL
    from the state to its adversary-masked forward in fp32 (mask g + (1 - g) a over all C subcomponents, Delta' under
    a[-1]; sources (C + 1,) per matrix). Also returns the first batch's masked logits and the removed fraction of every
    block, as weights for `Remover` over `state_blocks` (gate G1: the hooks reproduce the masked forward). Take it
    before any hook is attached."""
    device = next(model.parameters()).device
    wd = vpd.weight_deltas(model)
    a = {k: s.to(device).float() for k, s in sources.items()}
    out: Cached = []
    tot = n = 0.0
    first: dict[str, Any] = {}
    for i, b in enumerate(batches):
        b = b.to(device)
        B, T = b.shape
        ci = vpd.compute_ci(model, b, chunk=B)
        g = {k: ci.g[k].float() for k in a}
        masks = {k: g[k] + (1 - g[k]) * a[k][:-1] for k in a}
        ref = fwd(model.target_model, b)
        lg = masked_logits(model, b, masks, wd, {k: a[k][-1].expand(B, T) for k in a}, autocast=False).float()
        tot += float(vpd.kl_per_position(ref, lg).double().sum())
        n += B * T
        out.append((b, ref, {k: g[k][..., alive[k].to(device)] for k in alive}))
        if i == 0:
            first = {"batch": b, "masked_logits": lg,
                     "weights": {k: [0.0, 1 - masks[k][..., alive[k].to(device)], 1 - masks[k][..., ~alive[k].to(device)],
                                     1 - float(a[k][-1])] for k in a}}
        del ci, masks
    return out, tot / n, first


def arm(wf: Any, orig: float, new: float, sel: dict[str, Tensor] | None = None) -> Any:
    """weights_of for `revert_kl` over a `Remover` of `state_blocks`: the adversary's fractions w = wf(g) (from
    `adversary_weights`) on the alive originals times `orig` and on the alive new pieces times `new`. (1, 0) removes
    the adversary's fraction of the originals with the changes kept (the complement revert); (-1, 1) of the changes
    with the originals kept (E17's F_adv); (0, 1) of both. With `sel` ((n,) bool per matrix), only those alive
    subcomponents, and matrices absent from `sel` are left whole."""

    def weights_of(g: dict[str, Tensor]) -> dict[str, list[Weight]]:
        out: dict[str, list[Weight]] = {}
        for k, w in wf(g).items():
            if sel is not None:
                if k not in sel:
                    continue
                w = w * sel[k].to(device=w.device, dtype=w.dtype)
            out[k] = [orig * w if orig else 0.0, new * w if new else 0.0]
        return out

    return weights_of


def importance_bins(mean_g: dict[str, Tensor], alive: dict[str, Tensor], n_bins: int) -> Bins:
    """The alive subcomponents of all matrices pooled and cut into `n_bins` equal-count bins by W0's mean causal
    importance, lowest first ("imp0" holds the least important quarter for n_bins = 4). Ties fall by index."""
    keys = sorted(alive)
    vals = torch.cat([mean_g[k][alive[k]] for k in keys])
    rank = torch.empty(len(vals), dtype=torch.long)
    rank[vals.argsort(stable=True)] = torch.arange(len(vals))
    which = rank * n_bins // len(vals)
    out: Bins = {f"imp{i}": {} for i in range(n_bins)}
    start = 0
    for k in keys:
        n = int(alive[k].sum())
        for i in range(n_bins):
            out[f"imp{i}"][k] = which[start : start + n] == i
        start += n
    return out


def layer_bins(alive: dict[str, Tensor]) -> Bins:
    """One bin per layer: every alive subcomponent of that layer's matrices."""
    out: Bins = {}
    for k in sorted(alive):
        out.setdefault(f"layer{vpd.layer_of(k)}", {})[k] = torch.ones(int(alive[k].sum()), dtype=torch.bool)
    return out


@torch.no_grad()
def bin_means(cached: Cached, wf: Any, bins: Bins) -> dict[str, float]:
    """Mean of the adversary's removed fraction wf(g) over every position and every alive subcomponent of each bin."""
    tot = {b: 0.0 for b in bins}
    cnt = {b: 0 for b in bins}
    for _, _, g in cached:
        w = wf(g)
        for name, sel in bins.items():
            for k, m in sel.items():
                x = w[k][..., m.to(w[k].device)]
                tot[name] += float(x.double().sum())
                cnt[name] += x.numel()
    return {b: tot[b] / cnt[b] if cnt[b] else float("nan") for b in bins}
