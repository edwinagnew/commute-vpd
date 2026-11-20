"""K*, s_u and R at the VPD featurization of the 67M with a remainder per decomposed matrix (cell P1.37,
`experiments/kstar_vpd`; plan `reports/plans/2026-11-20_plan_p1-37-vpd.md` and its appendix). Added 2026-11-20. Calls
`commute.split` (the least-squares split), `commute.kstar` (`_add`, `_vec`, `summary`) and `commute.rule_retention`
(`kl`), and changes none of them.

Variables (decisions.md 2026-11-18, "Exact featurization with a mandatory remainder ... VPD built on weights"): the
embedding (index 0); per decomposed matrix W (24 sites, `Site`) its C subcomponents and its remainder; one graft
variable per appended neuron (E195's "moved", c_fc row and down_proj column together; M lacks it, so it scores 0);
the unembedding (ln_f with it, last index). In param-decomp's layout U is (C, d_out), V (d_in, C), u_c = U[c],
v_c = V[:, c], and W0 = sum_c u_c v_c^T + Delta. The fine-tune's delta-W is split by the least-squares split over all
C subcomponents (fp64): dU, dV with delta-W_c = du_c v_c^T + u_c dv_c^T, and delta-W_out = delta-W - sum_c delta-W_c,
so the split is exact by construction. Mechanisms on the input y of W: M's subcomponent u_c (v_c . y), F's
(u_c + du_c)(v_c . y) + u_c (dv_c . y); M's remainder Delta y, F's (Delta + delta-W_out) y.

F_K (`Hybrid`): F's native forward minus, for every ablated variable, its mechanism applied to (x - x_bar), x_bar the
position-wise mean of its input over D in F's unablated run (each mechanism is linear in its input, so this writes the
variable's mean write); a graft neuron writes its mean activation; the unembedding ablated gives the mean logits. All
kept subtracts exact zeros, so it is F's forward. s_u (`grades`): 1 - mean over settings of (1 - cos^2) between F's and
M's write on the same input, settings = every position of D in M's run and in F's, row 0 every position, row 1 the read
positions; a subcomponent's cos^2 is analytic in a = v.y, b = dv.y and |u|^2, u.du, |du|^2; both writes zero counts 1,
one zero counts 0 (exact-zero tests). The embedding and unembedding read 1: the run stops unless their weights are
equal in F and M. K* (`search`): attribution order, the shortest passing prefix by bisection, re-ranked at the found
set, then chunk removal with halving chunks, under one evaluation cap.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import Tensor

from commute import kstar as K
from commute import rule_retention as R
from commute import split as SP


@dataclass
class Site:
    """One decomposed matrix: fp32 tensors on the device, layouts as param-decomp's."""

    name: str
    off: int  # variable index of subcomponent 0
    rem: int  # variable index of the remainder
    V: Tensor  # (d_in, C)
    U: Tensor  # (C, d_out)
    dV: Tensor  # (d_in, C)
    dU: Tensor  # (C, d_out)
    rf: Tensor  # (d_in, d_out): (Delta + delta-W_out)^T, F's remainder
    rm: Tensor  # (d_in, d_out): Delta^T, M's remainder
    changed: bool  # delta-W of this matrix is non-zero


@dataclass
class Graft:
    layer: int
    i: int  # its index among the layer's appended neurons
    var: int


@dataclass
class Feat:
    sites: list[Site]
    grafts: list[Graft]
    n: int
    changed: set[int]  # variables whose mechanism differs in F (C of the definition), grafts included
    lacked: set[int]  # variables M lacks (the grafts), scoring 0
    checks: dict[str, Any]

    @property
    def rem(self) -> list[int]:
        return [s.rem for s in self.sites]

    def names(self) -> list[str]:
        out = ["embed"] + [""] * (self.n - 1)
        for s in self.sites:
            out[s.off : s.rem] = [f"{s.name}:{c}" for c in range(s.rem - s.off)]
            out[s.rem] = f"{s.name}:rem"
        for g in self.grafts:
            out[g.var] = f"h.{g.layer}.graft{g.i}"
        out[-1] = "unembed"
        return out


def bases(uv: dict[str, tuple[Tensor, Tensor]], tol: float) -> dict[str, SP.Basis]:
    """The split's SVD bases per matrix, over every subcomponent (no alive selection), fp64; shared by every item."""
    return {k: SP.basis(U.T.double(), V.double(), tol) for k, (U, V) in uv.items()}


def _rel(a: Tensor, b: Tensor) -> float:
    return float((a - b).norm() / b.norm()) if b.any() else float((a - b).norm())


def _site(name: str, off: int, WM: Tensor, WF: Tensor, U: Tensor, V: Tensor, delta: Tensor, b: SP.Basis) -> tuple[Site, dict[str, float]]:
    Ua, Va, dW = U.T.double(), V.double(), WF.double() - WM.double()
    dUa, dVa = SP.least_squares_split(dW, b, Ua, Va) if dW.any() else (torch.zeros_like(Ua), torch.zeros_like(Va))
    wr, rd = dUa @ Va.T, Ua @ dVa.T  # write side sum_c du_c v_c^T, read side sum_c u_c dv_c^T
    out = dW - (wr + rd)
    C = U.shape[0]
    s = Site(name, off, off + C, V.float(), U.float(), dVa.float(), dUa.T.float().contiguous(), (delta.double() + out).T.float().contiguous(),
             delta.T.float().contiguous(), bool(dW.any()))  # fmt: skip
    f64 = {k: getattr(s, k).double() for k in ("U", "V", "dU", "dV", "rf", "rm")}
    rebuilt_F = (f64["U"] + f64["dU"]).T @ f64["V"].T + f64["U"].T @ f64["dV"].T + f64["rf"].T
    chk = {"split_fp64": _rel(SP.dsigma(dUa, dVa, Ua, Va) + out, dW), "F_from_pieces": _rel(rebuilt_F, WF.double()),
           "M_from_pieces": _rel(f64["U"].T @ f64["V"].T + f64["rm"].T, WM.double()),
           "outside_share": float(out.norm() / dW.norm()) if dW.any() else 0.0}  # fmt: skip
    chk |= {"dW2": float((dW**2).sum()), "write2": float((wr**2).sum()), "read2": float((rd**2).sum()), "outside2": float((out**2).sum()),
            "cross_wr2": 2 * float((wr * rd).sum()), "cross_out2": 2 * float(((wr + rd) * out).sum())}  # fmt: skip
    return s, chk


NORMS = ("write2", "read2", "outside2", "cross_wr2", "cross_out2")


def norm_shares(feat: Feat) -> dict[str, Any]:
    """Squared Frobenius norms of the summed write side, read side and outside part, and the two cross terms (2<w, r>
    and 2<w + r, outside>), each as a share of |delta-W|^2, per matrix (None where delta-W = 0) and over all 24 (sums
    of numerators over the sum of |delta-W|^2). The five shares sum to 1; the first three alone need not."""
    per = {k: {n: c[n] / c["dW2"] for n in NORMS} if c["dW2"] > 0 else None for k, c in feat.checks.items()}
    tot = sum(c["dW2"] for c in feat.checks.values())
    return {"per_matrix": per, "total": {n: sum(c[n] for c in feat.checks.values()) / tot for n in NORMS} if tot > 0 else None}


def featurize(M: R.Model, F: R.Model, uv: dict[str, tuple[Tensor, Tensor]], delta: dict[str, Tensor], b: dict[str, SP.Basis]) -> Feat:
    """The variables of the pair (F, M) and their mechanisms. `uv`, `delta`: the decomposition's (U, V) and Delta per
    matrix (`vpd.component_weights`, `vpd.weight_deltas`), keyed by module name."""
    sites, checks, changed, off = [], {}, set(), 1
    for k in sorted(uv):
        s, checks[k] = _site(k, off, M.net.get_submodule(k).weight, F.net.get_submodule(k).weight, uv[k][0], uv[k][1], delta[k], b[k])
        moved = (s.dU != 0).any(1) | (s.dV != 0).any(0)
        changed |= {off + int(c) for c in moved.nonzero().squeeze(1).tolist()} | ({s.rem} if bool((s.rf != s.rm).any()) else set())
        sites.append(s)
        off = s.rem + 1
    grafts = []
    for l in sorted(F.extra):
        mlp = F.net.h[l].mlp
        assert mlp.c_fc.bias is None and mlp.down_proj.bias is None, "zero-padding M is exact only without biases"
        grafts += [Graft(l, i, off + i) for i in range(F.extra[l].w_in.shape[0])]
        off += F.extra[l].w_in.shape[0]
    lacked = {g.var for g in grafts}
    return Feat(sites, grafts, off + 1, changed | lacked, lacked, checks)


def undecomposed_differences(M: R.Model, F: R.Model, keys: list[str]) -> list[str]:
    """Every parameter of F's network outside the decomposed weights that differs from M's (embedding, unembedding,
    norms; none is owned by a variable that may change); appended neurons are not parameters of the network."""
    pM, skip = dict(M.net.named_parameters()), {f"{k}.weight" for k in keys}
    return [n for n, p in F.net.named_parameters() if n not in skip and not torch.equal(p, pM[n])]


# ---- F_K -----------------------------------------------------------------------------------------------------------


class Hybrid:
    """Hooks on F's network: native (no keep vector), capture (sums for the means) or ablate (keep vector c)."""

    def __init__(self, F: R.Model, feat: Feat, at: slice) -> None:
        self.F, self.feat, self.at = F, feat, at
        self.w: Tensor | None = None  # 1 - c on the device while ablating
        self.live: list[bool] = []
        self.mu: dict[str, Tensor] = {}
        self.sums: dict[str, Tensor] | None = None
        net, self.handles = F.net, []
        self.handles += [net.get_submodule(s.name).register_forward_hook(partial(self._site, j)) for j, s in enumerate(feat.sites)]
        self.handles += [net.wte.register_forward_hook(self._embed), net.lm_head.register_forward_pre_hook(self._head)]
        self.handles += [net.h[l].mlp.register_forward_hook(partial(self._graft, l)) for l in sorted({g.layer for g in feat.grafts})]  # after R.Model's own

    def close(self) -> None:
        for h in self.handles:
            h.remove()

    def _add(self, k: str, v: Tensor) -> None:
        assert self.sums is not None
        self.sums[k] = self.sums[k] + v.sum(0) if k in self.sums else v.sum(0)

    def _site(self, j: int, _m: Any, inp: tuple[Tensor, ...], out: Tensor) -> Tensor | None:
        if self.sums is not None:
            self._add(f"x{j}", inp[0])
        if self.w is None or not self.live[j]:
            return None
        s = self.feat.sites[j]
        y, w = inp[0] - self.mu[f"x{j}"], self.w[s.off : s.rem]
        a = y @ s.V
        if s.changed:
            b = y @ s.dV
            corr = ((a + b) * w) @ s.U + (a * w) @ s.dU
        else:
            corr = (a * w) @ s.U
        return out - corr - self.w[s.rem] * (y @ s.rf)

    def _embed(self, _m: Any, _inp: Any, out: Tensor) -> Tensor | None:
        if self.sums is not None:
            self._add("e", out)
        return None if self.w is None else out - self.w[0] * (out - self.mu["e"])

    def _graft(self, l: int, _m: Any, inp: tuple[Tensor, ...], out: Tensor) -> Tensor | None:
        e = self.F.extra[l]
        acts = e.acts(inp[0])
        if self.sums is not None:
            self._add(f"g{l}", acts)
        if self.w is None:
            return None
        w = self.w[[g.var for g in self.feat.grafts if g.layer == l]]
        return out - ((acts - self.mu[f"g{l}"]) * w) @ e.w_out.T

    def _head(self, _m: Any, inp: tuple[Tensor, ...]) -> tuple[Tensor] | None:
        return None if self.w is None and self.sums is None else (inp[0][:, self.at],)

    @torch.no_grad()
    def capture(self, batches: list[Tensor]) -> None:
        """The means over D from F's unablated run: each site's input, the embedding, graft activations, the logits at `at`."""
        self.sums = {}
        for t in batches:
            self._add("logits", self.F.net(t)[0].float())
        n = sum(len(t) for t in batches)
        self.mu, self.sums = {k: v / n for k, v in self.sums.items()}, None

    def logits(self, toks: Tensor, c: Tensor) -> Tensor:
        """F_K's logits at `at`, c (n,) on the device: 1 kept, 0 mean-ablated (fractional for the attribution)."""
        cc = c.detach().cpu()
        self.live = [bool((cc[s.off : s.rem + 1] < 1).any()) for s in self.feat.sites]
        self.w = 1 - c
        try:
            lf = self.F.net(toks)[0].float()
        finally:
            self.w = None
        return self.mu["logits"] + c[-1] * (lf - self.mu["logits"])


@dataclass
class Ctx:
    hy: Hybrid
    toks: list[Tensor]
    lF: list[Tensor]  # F's logits at `at` (all kept)
    kl_none: Tensor  # (N,) KL(F || F_empty), CPU fp64
    kl_fm: Tensor  # (N,) KL(F || M)

    def subset(self, idx: Tensor, batch: int) -> Ctx:
        t, lf = torch.cat(self.toks), torch.cat(self.lF)
        i = idx.to(t.device)
        return Ctx(self.hy, list(t[i].split(batch)), list(lf[i].split(batch)), self.kl_none[idx], self.kl_fm[idx])


@torch.no_grad()
def prepare(F: R.Model, M: R.Model, feat: Feat, toks: Tensor, at: slice, batch: int) -> Ctx:
    hy, batches = Hybrid(F, feat, at), list(toks.split(batch))
    hy.capture(batches)
    ones = torch.ones(feat.n, device=toks.device)
    lF = [hy.logits(t, ones) for t in batches]
    kl_none = torch.cat([R.kl(x, hy.mu["logits"].expand_as(x)) for x in lF]).cpu()
    kl_fm = torch.cat([R.kl(x, M.logits(t)[:, at]) for x, t in zip(lF, batches, strict=True)]).cpu()
    return Ctx(hy, batches, lF, kl_none, kl_fm)


@torch.no_grad()
def kl_mean(ctx: Ctx, keep: Tensor) -> float:
    """Mean over the windows of KL(F || F_K), K = where `keep` (bool, CPU) is set."""
    c = keep.to(ctx.toks[0].device, torch.float32)
    return float(torch.cat([R.kl(lf, ctx.hy.logits(t, c)) for t, lf in zip(ctx.toks, ctx.lF, strict=True)]).mean())


def attribution(ctx: Ctx, c0: Tensor) -> Tensor:
    """|d sum KL(F || F_c) / d c_u| at the keep vector c0 for every variable (CPU)."""
    c = c0.to(ctx.toks[0].device, torch.float32).clone().requires_grad_(True)
    with torch.enable_grad():
        for t, lf in zip(ctx.toks, ctx.lF, strict=True):
            R.kl(lf, ctx.hy.logits(t, c)).sum().backward()
    assert c.grad is not None
    return c.grad.abs().cpu()


@torch.no_grad()
def gates(ctx: Ctx, n: int) -> dict[str, float]:
    """Max over windows of max |F_K - reference| / max |reference|: all kept against F's native logits (hooks off), none
    kept against the mean logits."""
    dev, out = ctx.toks[0].device, {"all": 0.0, "none": 0.0}
    for t in ctx.toks:
        ref_all = ctx.hy.F.net(t)[0].float()[:, ctx.hy.at]
        for k, c, ref in (("all", torch.ones(n, device=dev), ref_all), ("none", torch.zeros(n, device=dev), ctx.hy.mu["logits"].expand_as(ref_all))):
            err = (ctx.hy.logits(t, c) - ref).abs().amax((1, 2)) / ref.abs().amax((1, 2))
            out[k] = max(out[k], float(err.max()))
    return out


# ---- s_u -----------------------------------------------------------------------------------------------------------


def _inputs(net: Any, names: list[str], t: Tensor) -> list[Tensor]:
    got: dict[str, Tensor] = {}
    hs = [net.get_submodule(k).register_forward_pre_hook(lambda _m, a, k=k: got.__setitem__(k, a[0])) for k in names]
    try:
        net(t)
    finally:
        for h in hs:
            h.remove()
    return [got[k] for k in names]


def _sub(acc: dict[Any, Tensor], j: int, s: Site, y: Tensor, at: slice, norms: tuple[Tensor, Tensor, Tensor]) -> None:
    """One batch of settings for a site's subcomponents: the analytic dot and squared norms of the two writes."""
    uu, udu, dudu = norms
    a, b = y @ s.V.double(), y @ s.dV.double()
    dot = a * ((a + b) * uu + a * udu)
    nM = a.pow(2) * uu
    nF = (a + b).pow(2) * uu + 2 * a * (a + b) * udu + a.pow(2) * dudu
    zF = ((a == 0) & ((b == 0) | (uu == 0))) | ((uu == 0) & (dudu == 0))  # F's write exactly zero
    K._add(acc, j, dot, torch.where(zF, 0.0, nF.clamp(min=0)), nM, at)


@torch.no_grad()
def grades(M: R.Model, F: R.Model, feat: Feat, toks: Tensor, at: slice, batch: int) -> Tensor:
    """(2, n) s_u, row 0 every position, row 1 the read positions `at`: settings from M's run and F's run on every window."""
    names, acc = [s.name for s in feat.sites], dict[Any, Tensor]()
    norms = [((s.U.double() ** 2).sum(1), (s.U.double() * s.dU.double()).sum(1), (s.dU.double() ** 2).sum(1)) for s in feat.sites]
    for t in toks.split(batch):
        for N in (M, F):
            for j, (s, x) in enumerate(zip(feat.sites, _inputs(N.net, names, t), strict=True)):
                y = x.double()
                _sub(acc, j, s, y, at, norms[j])
                K._vec(acc, ("rem", j), y @ s.rf.double(), y @ s.rm.double(), at)
    out = torch.ones(2, feat.n, dtype=torch.float64)  # embedding and unembedding: equal weights (checked by the caller)
    for j, s in enumerate(feat.sites):
        for key, cols in ((j, slice(s.off, s.rem)), (("rem", j), slice(s.rem, s.rem + 1))):
            v = acc[key].cpu()
            out[:, cols] = 1 - torch.stack([v[0] / v[1], v[2] / v[3]])
    out[:, sorted(feat.lacked)] = 0.0
    return out


# ---- K* ------------------------------------------------------------------------------------------------------------


@dataclass
class Found:
    prefix: list[int]  # the shortest passing prefix over the rankings tried
    polished: list[int]
    evals: int
    capped: bool
    chunk_min: int  # the smallest chunk size the polish reached
    prefix_sizes: list[int]  # per ranking
    attributions: list[Tensor]  # per ranking, the attribution scores (n,) that ordered it


def search(ev: Callable[[Tensor], float], grad: Callable[[Tensor], Tensor], n: int, bar: float, reranks: int, cap: int, chunk_div: int,
           log: Callable[[str], None]) -> Found:  # fmt: skip
    """K* with KL(F || F_K) <= bar. Ranking 0: attribution at c = 1/2 (at c = 1 the gradient of KL is 0, its minimum);
    rankings 1..reranks: attribution at the best prefix found so far. Each ranking: bisect the prefix length (the full
    set passes, the empty set does not), so the returned prefix was evaluated and passes. Polish: drop chunks of the
    prefix, least attributed first, halving the chunk on a failure and doubling it (up to the start) on a success, while
    evaluations stay under `cap` in total."""
    evals = 0

    def ok(keep: Tensor) -> bool:
        nonlocal evals
        evals += 1
        return ev(keep) <= bar

    def prefix(order: Tensor, L: int) -> Tensor:
        keep = torch.zeros(n, dtype=torch.bool)
        keep[order[:L]] = True
        return keep

    best, sizes, g = None, [], grad(torch.full((n,), 0.5))
    gs = [g]
    for r in range(reranks + 1):
        order, lo, hi = g.argsort(descending=True), 0, n
        while hi - lo > 1 and evals < cap:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if ok(prefix(order, mid)) else (mid, hi)
        sizes.append(hi)
        best = prefix(order, hi) if best is None or hi < int(best.sum()) else best
        log(f"    ranking {r}: prefix {hi}, {evals} evaluations")
        if r < reranks:
            g = grad(best.float())
            gs.append(g)
    assert best is not None
    members = best.nonzero().squeeze(1)
    drop = members[g[members].argsort()].tolist()  # least attributed first
    keep, i, start = best.clone(), 0, max(1, len(drop) // chunk_div)
    c = cmin = start
    while i < len(drop) and evals < cap:
        trial = keep.clone()
        trial[drop[i : i + c]] = False
        if ok(trial):
            keep, i, c = trial, i + c, min(2 * c, start)
        elif c > 1:
            c //= 2
            cmin = min(cmin, c)
        else:
            i += 1
    return Found(members.tolist(), keep.nonzero().squeeze(1).tolist(), evals, i < len(drop), cmin, sizes, gs)


def read(found: Found, s: Tensor, feat: Feat) -> dict[str, Any]:
    """R, f, the site-remainder share, the graft share and |K| at the prefix set and the polished set (K.summary)."""
    out = {"evals": found.evals, "capped": found.capped, "chunk_min": found.chunk_min, "prefix_sizes": found.prefix_sizes}
    for k, Kset in (("prefix", found.prefix), ("polished", found.polished)):
        rec = K.summary(Kset, s, feat.changed, feat.lacked, None, feat.rem)
        out[k] = rec | {"graft_share": rec["lacked_in_K"] / rec["size"]}
    return out
