"""K* by mean ablation and the squared-cosine grade s_u at four levels of the 67M (cell P1.35, E214,
`experiments/kstar_levels`; plan `reports/plans/2026-11-18_plan_p1-35-t2.md`, appendix §1, §4-§6). Added 2026-11-19.
Built on `commute.rho`'s Pair, Net and Layer and `rho.changed`, called and not changed; the greedy search is
`rule_retention_fixed.eliminate`. No finished experiment calls this module.

A level is a list of variables that covers every weight of F once (appendix §1): neuron_head (each neuron, each head),
head_mlp (each head, each MLP layer), mlp_attn (each attention layer, each MLP layer), block (each block); every level
also has the embedding and the unembedding (ln_f with it), and, on an item with an added piece (E137), the piece as the
rest-of-network variable. A grafted head or appended neuron is its own variable where it is a level's unit, else inside
its layer or block. Each head and neuron has one owner variable (`Level.head`, `Level.neuron`).

F_K (`forward`): a keep vector c over the variables; every part writes mean + c (actual - mean), so c = 1 keeps the
variable and c = 0 writes its position-wise mean over D from F's unablated run (§4a). A head's write is linear in its z
and a neuron's in its activation, so the mean of z or of the activation gives the mean write. The unembedding ablated
gives the mean logits, so F_empty is the same at every level (§4d).

s_u (`grades`, §4b-c): 1 - mean over settings of (1 - cos^2(write_F(v), write_M(v))), F's and M's mechanism of u on the
same input v, settings = every position of D in M's run and in F's run ("all") and the read positions only ("read"); both
writes zero counts 1, one zero counts 0 (E208). A coarse variable's write is its total: an attention layer's heads
summed, an MLP layer's neurons summed, a block's attention write plus its MLP's write on x + attention write. A variable
M lacks (a grafted head or appended neuron as its own variable, a piece) scores 0 (decisions.md 2026-11-18, "Gain and
graft declarations").

Featurization (P2.42, E217, added 2026-11-19; every default unchanged): `Feat` splits each MLP site's activations by a
per-neuron rule fixed from M (above hi or below lo is strong). With `level(..., rem=True)` at neuron_head, neuron i's
variable is its strong part (zero when weak) and ("rem", l, 0) is the site's remainder, the weak part; the two sum to
the activations, so the featurization is exact. `prepare`, `forward` and `grades` take the Feat; F's mechanism of a
feature is F's activation read through M's rule.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import rho
from commute import rule_retention as R
from commute import rule_retention_fixed as RF

Var = tuple[str, int, int]  # ("neuron"|"head", l, i), ("mlp"|"attn"|"block", l, 0), ("embed"|"unembed"|"piece", 0, 0)
Piece = Callable[[list[Tensor], Tensor], Tensor]  # (inputs of blocks 0..PIECE_AT-1, tokens) -> (B, T, d) write
PIECE_AT = 2  # E137's pieces write at block 2's input
LEVELS = ("neuron_head", "head_mlp", "mlp_attn", "block")
EXHAUSTIVE = ("mlp_attn", "block")
GRADE_BATCH = 16  # windows per grade batch (memory only: the unembedding's write is 50,277 wide, in fp64)
READ_TOKENS = 4096  # read positions per KL batch (memory only)


def vname(u: Var) -> str:
    k, l, i = u
    return {"neuron": f"L{l}.n{i}", "head": f"L{l}.h{i}", "mlp": f"L{l}.mlp", "attn": f"L{l}.attn", "block": f"L{l}.block", "rem": f"L{l}.rem"}.get(k, k)


@dataclass(frozen=True)
class Feat:
    """Per layer (n,) bounds from M: an activation above hi or below lo is a feature's value, the rest the remainder's."""

    lo: list[Tensor]
    hi: list[Tensor]

    def strong(self, l: int, a: Tensor) -> Tensor:
        return (a > self.hi[l]) | (a < self.lo[l])


@dataclass
class Level:
    name: str
    vars: list[Var]
    head: list[Tensor]  # per layer, (n_head,) long: each head's variable
    neuron: list[Tensor]  # per layer, (n,) long
    embed: int
    unembed: int
    piece: int | None
    rem: list[int] | None = None  # per layer, the site remainder's variable (neuron_head with a Feat only)

    def owner(self, u: rho.Part) -> int:
        k, l, i = u
        return int(self.head[l][i]) if k == "head" else int(self.neuron[l][i]) if k == "neuron" else self.embed if k == "embed" else self.unembed


def level(p: rho.Pair, name: str, piece: bool, rem: bool = False) -> Level:
    assert not rem or name == "neuron_head"
    vs: list[Var] = [("embed", 0, 0)]
    head: list[Tensor] = []
    neuron: list[Tensor] = []
    rems: list[int] = []

    def take(new: list[Var]) -> Tensor:
        vs.extend(new)
        return torch.arange(len(vs) - len(new), len(vs))

    for l, L in enumerate(p.f.layers):
        H, n = L.n_head, L.w_in.shape[0]
        if name == "block":
            i = int(take([("block", l, 0)])[0])
            head.append(torch.full((H,), i))
            neuron.append(torch.full((n,), i))
            continue
        head.append(torch.full((H,), int(take([("attn", l, 0)])[0])) if name == "mlp_attn" else take([("head", l, h) for h in range(H)]))
        neuron.append(take([("neuron", l, i) for i in range(n)]) if name == "neuron_head" else torch.full((n,), int(take([("mlp", l, 0)])[0])))
        rems += [int(take([("rem", l, 0)])[0])] if rem else []
    take([("unembed", 0, 0)] + ([("piece", 0, 0)] if piece else []))
    dev = p.f.wte.device
    return Level(name, vs, [t.to(dev) for t in head], [t.to(dev) for t in neuron], 0, len(vs) - 1 - int(piece), len(vs) - 1 if piece else None,
                 rems if rem else None)  # fmt: skip


def changed_vars(p: rho.Pair, lv: Level) -> tuple[set[int], set[int]]:
    """(C, lacked): C holds every variable any of whose weights differ from M's, plus graft and piece variables (§4e);
    lacked holds the variables M lacks, which score 0."""
    ch, lacks = rho.changed(p)
    C = {lv.owner(u) for u in ch + lacks} | ({lv.piece} if lv.piece is not None else set())
    C |= {lv.rem[u[1]] for u in ch + lacks if u[0] == "neuron"} if lv.rem is not None else set()  # a site with a changed neuron
    lacked = {i for i, u in enumerate(lv.vars) if u[0] == "piece" or (u[0] in ("head", "neuron") and rho.lacked_by_m(p, u))}
    return C, lacked


# ---- F's run, the means and F_K ---------------------------------------------------------------------------------------


@dataclass
class Run:
    emb: Tensor
    xs: list[Tensor]  # each block's input
    ys: list[Tensor]  # each block's residual before its MLP
    z: list[Tensor]
    a: list[Tensor]
    pw: Tensor | None  # the piece's write
    out: Tensor


def run(N: rho.Net, toks: Tensor, piece: Piece | None) -> Run:
    r = Run(N.wte[toks], [], [], [], [], None, torch.empty(0))
    x = r.emb
    for l, L in enumerate(N.layers):
        if piece is not None and l == PIECE_AT:
            r.pw = piece(r.xs, toks)
            x = x + r.pw
        r.xs.append(x)
        r.z.append(L.z(x))
        x = x + L.attn_out(r.z[-1], torch.ones(L.n_head, device=x.device))
        r.ys.append(x)
        r.a.append(L.acts(x))
        x = x + r.a[-1] @ L.w_out.T
    r.out = x
    return r


@dataclass
class Means:
    emb: Tensor  # (T, d)
    z: list[Tensor]  # per layer (H, T, D)
    a: list[Tensor]  # per layer (T, n); with a Feat, of the strong part
    piece: Tensor | None  # (T, d)
    logits: Tensor  # (T_read, V)
    a_w: list[Tensor] | None = None  # with a Feat, per layer (T, n) of the weak part (the remainder)


def split(feat: Feat, l: int, a: Tensor) -> tuple[Tensor, Tensor]:
    """(strong part, weak part) of activations a at layer l; they sum to a."""
    s = a * feat.strong(l, a)
    return s, a - s


def forward(p: rho.Pair, toks: Tensor, lv: Level, c: Tensor, mu: Means, piece: Piece | None, at: slice, feat: Feat | None = None) -> Tensor:
    """F_K's logits at positions `at`; c (n_vars,) float on the device, 1 kept, 0 mean-ablated. With a Feat, a neuron's
    variable is its strong part and the site remainder lv.rem[l] its weak part."""
    x = mu.emb + c[lv.embed] * (p.f.wte[toks] - mu.emb)
    xs: list[Tensor] = []
    for l, L in enumerate(p.f.layers):
        if piece is not None and l == PIECE_AT:
            assert mu.piece is not None and lv.piece is not None
            x = x + mu.piece + c[lv.piece] * (piece(xs, toks) - mu.piece)
        xs.append(x)
        z = mu.z[l] + c[lv.head[l]][:, None, None] * (L.z(x) - mu.z[l])
        x = x + L.attn_out(z, torch.ones(L.n_head, device=x.device))
        if feat is None:
            a = mu.a[l] + c[lv.neuron[l]] * (L.acts(x) - mu.a[l])
        else:
            assert lv.rem is not None and mu.a_w is not None
            st, wk = split(feat, l, L.acts(x))
            a = mu.a[l] + c[lv.neuron[l]] * (st - mu.a[l]) + mu.a_w[l] + c[lv.rem[l]] * (wk - mu.a_w[l])
        x = x + a @ L.w_out.T
    return mu.logits + c[lv.unembed] * (p.f.logits(x[:, at]).float() - mu.logits)


@dataclass
class Ctx:
    """An item's windows D in batches, F's logits at the read positions, the means, KL(F || F_empty) and KL(F || M) per window."""

    p: rho.Pair
    piece: Piece | None
    at: slice
    toks: list[Tensor]
    lF: list[Tensor]
    mu: Means
    kl_none: Tensor  # (N,) fp64, CPU
    kl_fm: Tensor
    feat: Feat | None = None

    def subset(self, idx: Tensor) -> Ctx:
        t, lf = torch.cat(self.toks), torch.cat(self.lF)
        n = _batch(t, self.at)
        i = idx.to(t.device)
        return Ctx(self.p, self.piece, self.at, list(t[i].split(n)), list(lf[i].split(n)), self.mu, self.kl_none[idx], self.kl_fm[idx], self.feat)


def _batch(toks: Tensor, at: slice) -> int:
    return max(1, READ_TOKENS // len(range(toks.shape[1])[at]))


@torch.no_grad()
def prepare(p: rho.Pair, toks: Tensor, at: slice, piece: Piece | None, m_logits: Callable[[Tensor], Tensor], feat: Feat | None = None) -> Ctx:
    batches, lF, sums = list(toks.split(_batch(toks, at))), list[Tensor](), dict[str, Tensor]()
    for t in batches:
        r = run(p.f, t, piece)
        lF.append(p.f.logits(r.out[:, at]).float())
        parts = {"emb": r.emb, "logits": lF[-1]} | {f"z{l}": z for l, z in enumerate(r.z)} | {f"a{l}": a for l, a in enumerate(r.a)}
        if feat is not None:
            parts |= {k: v for l, a in enumerate(r.a) for k, v in zip((f"a{l}", f"w{l}"), split(feat, l, a), strict=True)}
        if r.pw is not None:
            parts["pw"] = r.pw
        for k, v in parts.items():
            sums[k] = sums[k] + v.sum(0) if k in sums else v.sum(0)
    m = {k: v / toks.shape[0] for k, v in sums.items()}
    n = len(p.f.layers)
    mu = Means(m["emb"], [m[f"z{l}"] for l in range(n)], [m[f"a{l}"] for l in range(n)], m.get("pw"), m["logits"],
               [m[f"w{l}"] for l in range(n)] if feat is not None else None)  # fmt: skip
    kl_none = torch.cat([R.kl(x, mu.logits.expand_as(x)) for x in lF]).cpu()
    kl_fm = torch.cat([R.kl(x, m_logits(t)[:, at]) for x, t in zip(lF, batches, strict=True)]).cpu()
    return Ctx(p, piece, at, batches, lF, mu, kl_none, kl_fm, feat)


def kl_rows(ctx: Ctx, lv: Level, keep: Tensor) -> Tensor:
    """(N,) KL(F || F_K) per window, K = the variables where `keep` is 1."""
    c = keep.to(ctx.p.f.wte.device, torch.float32)
    with torch.no_grad():
        return torch.cat([R.kl(lf, forward(ctx.p, t, lv, c, ctx.mu, ctx.piece, ctx.at, ctx.feat)) for t, lf in zip(ctx.toks, ctx.lF, strict=True)]).cpu()


def gates(ctx: Ctx, lv: Level, native: Callable[[Tensor], Tensor]) -> dict[str, float]:
    """Max over windows of max |F_K - reference| / max |reference|: K = all against F's native logits, K = empty against
    the mean logits (R0 b)."""
    dev, n = ctx.p.f.wte.device, len(lv.vars)
    out = {"all": 0.0, "none": 0.0}
    with torch.no_grad():
        for t in ctx.toks:
            for k, c, ref in (("all", torch.ones(n, device=dev), native(t)[:, ctx.at].float()), ("none", torch.zeros(n, device=dev), ctx.mu.logits.expand(len(t), -1, -1))):
                err = (forward(ctx.p, t, lv, c, ctx.mu, ctx.piece, ctx.at, ctx.feat) - ref).abs().amax((1, 2)) / ref.abs().amax((1, 2))
                out[k] = max(out[k], float(err.max()))
    return out


def attribution(ctx: Ctx, lv: Level) -> Tensor:
    """|d KL(F || F_c) / d c_u| summed over D at c = 1/2 for every variable: |<mean - actual write, dKL/dwrite>| at the
    half-ablated point. At c = 1 the gradient of KL is 0 (its minimum), so the midpoint is used (L1 order 0)."""
    c = torch.full((len(lv.vars),), 0.5, device=ctx.p.f.wte.device, requires_grad=True)
    with torch.enable_grad():
        for t, lf in zip(ctx.toks, ctx.lF, strict=True):
            R.kl(lf, forward(ctx.p, t, lv, c, ctx.mu, ctx.piece, ctx.at, ctx.feat)).sum().backward()
    assert c.grad is not None
    return c.grad.abs().cpu()


# ---- s_u ----------------------------------------------------------------------------------------------------------------


def _add(acc: dict[Any, Tensor], key: Any, dot: Tensor, nF: Tensor, nM: Tensor, at: slice) -> None:
    """One batch of settings (B, T, k): the sums of 1 - cos^2 and the counts, over every position and the read positions."""
    dot, nF, nM = dot.double(), nF.double(), nM.double()
    both, one = (nF > 0) & (nM > 0), (nF > 0) ^ (nM > 0)
    d = 1 - torch.where(both, dot.pow(2) / (nF * nM).clamp(min=1e-300), torch.where(one, 0.0, 1.0))
    r = d[:, at]
    v = torch.stack([d.sum((0, 1)), torch.full_like(d[0, 0], d.shape[0] * d.shape[1]), r.sum((0, 1)), torch.full_like(d[0, 0], r.shape[0] * r.shape[1])])
    acc[key] = acc[key] + v if key in acc else v


def _vec(acc: dict[Any, Tensor], key: Any, wF: Tensor, wM: Tensor, at: slice) -> None:
    wF, wM = wF.double(), wM.double()
    _add(acc, key, (wF * wM).sum(-1, keepdim=True), wF.pow(2).sum(-1, keepdim=True), wM.pow(2).sum(-1, keepdim=True), at)


def _attn(L: rho.Layer, x: Tensor) -> Tensor:
    return L.attn_out(L.z(x), torch.ones(L.n_head, device=x.device))


def _mlp(L: rho.Layer, y: Tensor) -> Tensor:
    return L.acts(y) @ L.w_out.T


@torch.no_grad()
def grades(p: rho.Pair, toks: Tensor, at: slice, piece: Piece | None, feat: Feat | None = None) -> dict[Any, Tensor]:
    """Per kind (and layer) the (4, k) sums of `_add` over M's run and F's run on every window of `toks`. F's run has
    the piece, M's does not. With a Feat, a neuron's write is its strong part's, and ("rem", l) holds the site
    remainder's write (the weak parts, summed), each network's activations read through the one rule."""
    acc: dict[Any, Tensor] = {}
    for t in toks.split(GRADE_BATCH):
        for N, pc in ((p.m, None), (p.f, piece)):
            r = run(N, t, pc)
            _vec(acc, ("embed",), p.f.wte[t], p.m.wte[t], at)
            _vec(acc, ("unembed",), p.f.logits(r.out), p.m.logits(r.out), at)
            for l, (F, M) in enumerate(zip(p.f.layers, p.m.layers, strict=True)):
                x, y, n = r.xs[l], r.ys[l], M.w_in.shape[0]
                hF, hM = F.head_writes(x)[:, :, : M.n_head].double(), M.head_writes(x).double()
                _add(acc, ("head", l), (hF * hM).sum(-1), hF.pow(2).sum(-1), hM.pow(2).sum(-1), at)
                aF, aM = F.acts(y)[..., :n].double(), M.acts(y).double()
                wF, wM = F.w_out[:, :n].double(), M.w_out.double()
                if feat is not None:
                    (aF, eF), (aM, eM) = split(feat, l, aF), split(feat, l, aM)
                    _vec(acc, ("rem", l), eF @ wF.T, eM @ wM.T, at)
                _add(acc, ("neuron", l), aF * aM * (wF * wM).sum(0), aF.pow(2) * wF.pow(2).sum(0), aM.pow(2) * wM.pow(2).sum(0), at)
                oF, oM = _attn(F, x), _attn(M, x)
                _vec(acc, ("attn", l), oF, oM, at)
                _vec(acc, ("mlp", l), _mlp(F, y), _mlp(M, y), at)
                _vec(acc, ("block", l), oF + _mlp(F, x + oF), oM + _mlp(M, x + oM), at)
    return acc


def level_grades(acc: dict[Any, Tensor], lv: Level, lacked: set[int]) -> Tensor:
    """(2, n_vars) s_u over every position ("all", row 0) and the read positions ("read", row 1); 0 where M lacks u."""
    s = torch.zeros(2, len(lv.vars), dtype=torch.float64)
    for j, (k, l, i) in enumerate(lv.vars):
        if j in lacked:
            continue
        a = acc[(k,)] if k in ("embed", "unembed") else acc[(k, l)]
        col = a[:, i if k in ("head", "neuron") else 0].cpu()
        s[:, j] = 1 - torch.stack([col[0] / col[1], col[2] / col[3]])
    return s


# ---- the search ---------------------------------------------------------------------------------------------------------


def summary(K: list[int], s: Tensor, C: set[int], lacked: set[int], piece: int | None, rem: list[int] | None = None) -> dict[str, Any]:
    """R = mean s over K, f = |K & C| / |K|, rho' = mean s over K & C (None when empty), per grade reading, and the
    remainder (piece) share of K, plus, given site remainders `rem`, their share of K; asserts R = 1 - f (1 - rho') to
    1e-9 (an unchanged variable reads 1)."""
    KC = [u for u in K if u in C]
    f = len(KC) / len(K)
    out: dict[str, Any] = {"size": len(K), "f": f, "changed_in_K": len(KC), "lacked_in_K": sum(u in lacked for u in K),
                           "remainder_share": (piece in K) / len(K)}  # fmt: skip
    out |= {"site_remainder_share": sum(u in rem for u in K) / len(K), "site_remainders_in_K": sum(u in rem for u in K)} if rem is not None else {}
    for g, row in (("all", 0), ("read", 1)):
        Rv = float(s[row, K].mean())
        rp = float(s[row, KC].mean()) if KC else None
        assert abs(Rv - (1 - f * (1 - (rp if rp is not None else 1.0)))) <= 1e-9, f"R = 1 - f (1 - rho') fails: {Rv}, f {f}, rho' {rp}"
        out |= {f"R_{g}": Rv, f"rho_{g}": rp}
    return out


def exhaustive(ev: Callable[[Tensor], Tensor], n: int) -> Tensor:
    """(2^n, N) KL rows of every subset, subset m keeping the variables of m's set bits."""
    bits = 1 << torch.arange(n)
    return torch.stack([ev((m & bits) > 0) for m in range(1 << n)])


def from_table(table: Tensor, kl_none: Tensor, idx: Tensor, alpha: float) -> list[int]:
    """Every minimal subset (as a bit mask) with mean KL over windows idx <= (1 - alpha) mean KL_empty, lowest KL first."""
    kl = table[:, idx].mean(1)
    ok = (kl <= (1 - alpha) * float(kl_none[idx].mean())).nonzero().squeeze(1).tolist()
    size = min(m.bit_count() for m in ok)
    return sorted((m for m in ok if m.bit_count() == size), key=lambda m: float(kl[m]))


def polish(ok: Callable[[set[Var]], bool], K: set[Var], every: list[Var], cap: int) -> tuple[set[Var], int, bool]:
    """Drop any one variable, then swap any two for one outside, while the bar still holds; repeat until no move applies
    or `cap` evaluations are spent. Returns the set, the evaluations and whether the cap was hit."""
    n = 0
    while True:
        moves = itertools.chain(((K - {u}) for u in sorted(K)),
                                ((K - {u, v}) | {w} for u, v in itertools.combinations(sorted(K), 2) for w in every if w not in K))  # fmt: skip
        for S in moves:
            if n >= cap:
                return K, n, True
            n += 1
            if ok(S):
                K = S
                break
        else:
            return K, n, False


@dataclass(frozen=True)
class Variant:
    idx: Tensor  # windows of D, CPU long
    alpha: float
    orders: int


def search(ctx: Ctx, lv: Level, s: Tensor, C: set[int], lacked: set[int], variants: dict[str, Variant], th: dict[str, Any],
           log: Callable[[str], None], tick: Callable[[float], None]) -> tuple[dict[str, Any], dict[str, Any]]:  # fmt: skip
    """K* for each variant: exhaustive at mlp_attn and block (ties listed, the lowest-KL tie read), greedy by chunks of 1
    plus `polish` at head_mlp, greedy by chunks of th["chunk"] at neuron_head (order 0 by `attribution`, the rest seeded).
    Returns the records per variant and, at neuron_head, each K* as an index array."""
    out: dict[str, Any] = {}
    arrays: dict[str, Any] = {}
    names = [vname(u) for u in lv.vars]

    def detail(K: list[int]) -> list[dict[str, Any]]:
        """Circuit recovery: each variable of K*, its s under both grades, and whether it is changed, lacked or the piece."""
        return [{"name": names[j], "s_all": float(s[0, j]), "s_read": float(s[1, j]), "changed": j in C, "lacked": j in lacked, "piece": j == lv.piece}
                for j in K]  # fmt: skip

    if lv.name in EXHAUSTIVE:
        t0 = time.time()
        table = exhaustive(lambda keep: kl_rows(ctx, lv, keep), len(lv.vars))
        log(f"  {lv.name}: {table.shape[0]} subsets in {time.time() - t0:.0f} s")
        for name, v in variants.items():
            ties = from_table(table, ctx.kl_none, v.idx, v.alpha)
            sets = [[j for j in range(len(lv.vars)) if m >> j & 1] for m in ties]
            recs = [summary(K, s, C, lacked, lv.piece) for K in sets]
            rec = recs[0] | {"K": detail(sets[0]), "kl": float(table[ties[0], v.idx].mean()), "n_ties": len(ties),
                             "ties": [[names[j] for j in K] for K in sets]}  # fmt: skip
            rec |= {f"R_{g}_tie_range": max(r[f"R_{g}"] for r in recs) - min(r[f"R_{g}"] for r in recs) for g in ("all", "read")}
            out[name] = [rec]
        return out, arrays
    index = {u: j for j, u in enumerate(lv.vars)}
    att = attribution(ctx, lv) if lv.name == "neuron_head" else None
    for name, v in variants.items():
        sub = ctx.subset(v.idx)
        bar = -(1 - v.alpha) * float(sub.kl_none.mean())

        def closure(S: set[Var], sub: Ctx = sub) -> float:
            keep = torch.zeros(len(lv.vars))
            keep[[index[u] for u in S]] = 1
            return -float(kl_rows(sub, lv, keep).mean())

        gen, recs = torch.Generator().manual_seed(th["order_seed"]), []
        for k in range(v.orders):
            t0 = time.time()
            perm = att.argsort().tolist() if att is not None and k == 0 else torch.randperm(len(lv.vars), generator=gen).tolist()
            S, n = RF.eliminate(closure, [lv.vars[j] for j in perm], bar, 1 if lv.name == "head_mlp" else th["chunk"])
            capped = False
            if lv.name == "head_mlp":
                Sp, m, capped = polish(lambda T, c=closure, b=bar: c(T) >= b, set(S), lv.vars, th["polish_cap"])
                S, n = [u for u in lv.vars if u in Sp], n + m
            K = sorted(index[u] for u in S)
            if not K:
                raise RuntimeError(f"{lv.name} {name}: empty K* although KL_empty > 0")
            rec = summary(K, s, C, lacked, lv.piece, lv.rem) | {"kl": -closure(set(S)), "bar": -bar, "evals": n, "seconds": time.time() - t0, "polish_capped": capped}
            rec["K"] = detail(K)
            arrays[f"K_{name}_{k}"] = np.array(K)
            recs.append(rec)
            log(f"  {lv.name} {name} order {k}: |K*| {len(K)}, R all {rec['R_all']:.6f}, read {rec['R_read']:.6f}, f {rec['f']:.4f}; {n} evaluations, "
                f"{rec['seconds']:.0f} s" + (" (polish cap hit)" if capped else ""))
            tick(rec["seconds"])
        out[name] = recs
    return out, arrays if lv.name == "neuron_head" else {}
