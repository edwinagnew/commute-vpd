"""A rotation-tolerant kept share, a diagnostic beside the frozen definition (cell P1.60, E286,
`experiments/rotation_kept`; plan reports/plans/2026-11-23_plan_p1-60-rotation-kept.md). Added 2026-11-23. Reads the
sites of `commute.kstar_vpd` (VPD) and `commute.needed_neurons` (heads, neurons) and changes neither.

Per variable u, over the rows given (every position of the windows in F's own run), W_M(u) and W_F(u) are M's and F's
writes on F's own input to u (as `stand_in.write_change`) and Delta = W_F - W_M, dd = |Delta|^2 summed over rows.
The parts of dd read:
    strict   the frozen kept, `stand_in.explained(num, den)` (num = sum <W_F, W_M>, den = sum |W_M|^2)
    own      the part of Delta in O(u), the fewest top eigen-directions of W_M^T W_M holding `energy` of its trace
    needed   the part in S(u) = span of O(u) and O(v), v every non-remainder member of the reference set R at u's site
    matched  the part in A(u) = span of O(u) and as many O(v) as S(u) uses, v drawn (seeded) from the site's
             non-remainder variables outside R and the read set (redteam-lead's anisotropy-matched null)
and nu(u) = dim S / d_out, a random direction's expected share of S.

Two kinds of variable. A one-direction variable (VPD subcomponent, neuron) has Delta = c1 E1 + c2 E2 and W_M = mc m on
every row, with fixed vectors: subcomponent E1 = u_c, E2 = du_c, c1 = b, c2 = a, m = u_c, mc = a (a = v_c . y,
b = dv_c . y; kstar_vpd's mechanisms); neuron E1 = W_F[:, i], E2 = -W_M[:, i], c1 = s_F, c2 = s_M, m = W_M[:, i],
mc = s_M (needed_neurons'). Six row sums give every part exactly. A Gram variable (head, remainder) keeps
Delta^T Delta, W_M^T W_M and sum <Delta, W_M>.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from commute import kstar_vpd as KV
from commute import needed_neurons as NN
from commute import stand_in as SI

# ---- sites: where a variable writes, and its writes on a batch --------------------------------------------------------


class VPD:
    """One decomposed matrix: subcomponents 0..C-1 (one direction each), the remainder C (Gram)."""

    def __init__(self, s: KV.Site) -> None:
        self.s, self.module, self.off = s, s.name, s.off
        self.n, self.rem, self.d_out = s.rem - s.off + 1, s.rem - s.off, s.U.shape[1]

    def prep(self, y: Tensor) -> Tensor:
        return y.flatten(0, -2).double()

    def coefs(self, y: Any, loc: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        a = y @ self.s.V.double()[:, loc]
        return y @ self.s.dV.double()[:, loc], a, a

    def vectors(self, loc: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        U = self.s.U.double()[loc]
        return U, self.s.dU.double()[loc], U

    def gram(self, y: Any, j: int) -> tuple[Tensor, Tensor]:
        assert j == self.rem
        return y @ self.s.rf.double(), y @ self.s.rm.double()


class Neurons:
    """One layer's MLP at the neuron basis: neurons 0..3071 (one direction each), the remainder 3072 (Gram)."""

    def __init__(self, s: NN.MLPSite) -> None:
        self.s, self.module, self.off = s, s.module, s.off
        self.n, self.rem, self.d_out = NN.NEUR + 1, NN.NEUR, s.f.w_out.shape[0]

    def prep(self, y: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.s.parts(y.flatten(0, -2))

    def coefs(self, ctx: Any, loc: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return ctx[0][:, loc], ctx[1][:, loc], ctx[1][:, loc]

    def vectors(self, loc: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        WM = self.s.m.w_out.double()[:, loc].T
        return self.s.f.w_out.double()[:, loc].T, -WM, WM

    def gram(self, ctx: Any, j: int) -> tuple[Tensor, Tensor]:
        assert j == self.rem
        return ctx[2], ctx[3]


class Heads:
    """One layer's attention at the head basis: six heads, each a Gram variable."""

    rem = None

    def __init__(self, s: NN.HeadSite) -> None:
        self.s, self.module, self.off = s, s.module, s.off
        self.n, self.d_out = NN.H, s.f.wo.shape[0]

    def prep(self, y: Tensor) -> tuple[Tensor, Tensor]:
        return self.s.writes(y)

    def coefs(self, ctx: Any, loc: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raise AssertionError("heads are Gram variables")

    def vectors(self, loc: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raise AssertionError("heads are Gram variables")

    def gram(self, ctx: Any, j: int) -> tuple[Tensor, Tensor]:
        return ctx[0][:, j], ctx[1][:, j]


Site = VPD | Neurons | Heads


def vpd_sites(feat: KV.Feat) -> list[Site]:
    return [VPD(s) for s in feat.sites]


def neuron_sites(sites: list[Any]) -> list[Site]:
    return [Heads(s) if isinstance(s, NN.HeadSite) else Neurons(s) for s in sites]


def is_gram(s: Site, j: int) -> bool:
    return isinstance(s, Heads) or j == s.rem


def locate(sites: list[Site]) -> dict[int, tuple[int, int]]:
    """global index -> (site, local index), for every variable on a site."""
    return {s.off + j: (k, j) for k, s in enumerate(sites) for j in range(s.n)}


# ---- the sets -----------------------------------------------------------------------------------------------------


def top(flips: Tensor, pool: Tensor, k: int) -> list[int]:
    """The k members of `pool` (bool) with most flips, ties to the lower index."""
    order = sorted(pool.nonzero().squeeze(1).tolist(), key=lambda i: (-int(flips[i]), i))
    return sorted(order[:k])


@dataclass
class Sets:
    N: list[int]  # the read set (global)
    R: list[int]  # the reference set (global)
    draw: dict[int, list[int]]  # site -> non-remainder locals outside R and N, seeded order (the matched null's pool)


def sets(sites: list[Site], flips: Tensor, in_K: Tensor, changed: Tensor, frac: float, seed: int, N: list[int] | None = None) -> Sets:
    """R: the ceil(frac n) members of K with most flips; N (unless given): the same count of changed members of K.
    Only variables on a site (not the embedding, unembedding or grafts)."""
    where = locate(sites)
    on = torch.zeros(len(flips), dtype=torch.bool)
    on[list(where)] = True
    k = math.ceil(frac * len(flips))
    R = top(flips, in_K & on, k)
    N = top(flips, in_K & changed & on, k) if N is None else N
    taken, gen, draw = set(R) | set(N), torch.Generator().manual_seed(seed), {}
    for si, s in enumerate(sites):
        want = 2 * sum(1 for v in R if where[v][0] == si) + 4  # spare draws for rank-0 candidates
        free = [j for j in range(s.n) if j != s.rem and s.off + j not in taken]
        draw[si] = [free[int(p)] for p in torch.randperm(len(free), generator=gen)[:want]]
    return Sets(N, R, draw)


def wanted(sites: list[Site], st: Sets) -> dict[int, list[int]]:
    where, out = locate(sites), {}
    for g in sorted(set(st.N) | set(st.R)):
        out.setdefault(where[g][0], []).append(where[g][1])
    for si, js in st.draw.items():
        out[si] = sorted(set(out.get(si, [])) | set(js))
    return out


# ---- the pass ------------------------------------------------------------------------------------------------------


@dataclass
class Acc:
    dirs: Tensor  # (k,) locals of one-direction variables
    S: Tensor  # (k, 6) sums of c1 c1, c1 c2, c2 c2, c1 mc, c2 mc, mc mc
    GD: dict[int, Tensor] = field(default_factory=dict)
    GM: dict[int, Tensor] = field(default_factory=dict)
    X: dict[int, Tensor] = field(default_factory=dict)  # sum <Delta, W_M>


def _accs(sites: list[Site], want: dict[int, list[int]], dev: torch.device) -> list[Acc]:
    out = []
    for si, s in enumerate(sites):
        d = [j for j in want.get(si, []) if not is_gram(s, j)]
        a = Acc(torch.tensor(d, dtype=torch.long, device=dev), torch.zeros(len(d), 6, dtype=torch.float64, device=dev))
        for j in (j for j in want.get(si, []) if is_gram(s, j)):
            a.GD[j] = torch.zeros(s.d_out, s.d_out, dtype=torch.float64, device=dev)
            a.GM[j] = torch.zeros_like(a.GD[j])
            a.X[j] = torch.zeros((), dtype=torch.float64, device=dev)
        out.append(a)
    return out


@torch.no_grad()
def accumulate(net: Any, sites: list[Site], want: dict[int, list[int]], batches: list[Tensor]) -> list[Acc]:
    """One pass of F's network over `batches` (every position), the sums of every wanted variable (site -> locals)."""
    accs = _accs(sites, want, batches[0].device)
    got: dict[str, Tensor] = {}
    mods = sorted({s.module for s in sites})
    hs = [net.get_submodule(m).register_forward_pre_hook(lambda _m, a, m=m: got.__setitem__(m, a[0])) for m in mods]
    try:
        for x in batches:
            net(x)
            for s, a in zip(sites, accs, strict=True):
                if not len(a.dirs) and not a.GD:
                    continue
                ctx = s.prep(got[s.module])
                if len(a.dirs):
                    c1, c2, mc = s.coefs(ctx, a.dirs)
                    a.S += torch.stack([(c1 * c1).sum(0), (c1 * c2).sum(0), (c2 * c2).sum(0), (c1 * mc).sum(0), (c2 * mc).sum(0), (mc * mc).sum(0)], 1)
                for j in a.GD:
                    wF, wM = s.gram(ctx, j)
                    D = wF - wM
                    a.GD[j] += D.T @ D
                    a.GM[j] += wM.T @ wM
                    a.X[j] += (D * wM).sum()
    finally:
        for h in hs:
            h.remove()
    return accs


# ---- spaces and parts --------------------------------------------------------------------------------------------------


@dataclass
class Var:
    """One variable's sums on the CPU in fp64: Gram (GD, GM, X) or one-direction (S and E = rows E1, E2, m)."""

    d_out: int
    S: Tensor | None = None
    E: Tensor | None = None
    GD: Tensor | None = None
    GM: Tensor | None = None
    X: float = 0.0

    def part(self, B: Tensor | None) -> float:
        """The part of dd in the span of B's orthonormal rows; B None is the whole space (dd itself)."""
        if B is not None and B.shape[0] == 0:
            return 0.0
        if self.GD is not None:
            return float(self.GD.trace() if B is None else (B @ self.GD @ B.T).trace())
        assert self.S is not None and self.E is not None
        P = self.E[:2] if B is None else self.E[:2] @ B.T
        return float(self.S[0] * P[0] @ P[0] + 2 * self.S[1] * P[0] @ P[1] + self.S[2] * P[1] @ P[1])

    def num_den(self) -> tuple[float, float]:
        if self.GM is not None:
            den = float(self.GM.trace())
            return den + self.X, den
        assert self.S is not None and self.E is not None
        m = self.E[2]
        den = float(self.S[5] * m @ m)
        return den + float(self.S[3] * self.E[0] @ m + self.S[4] * self.E[1] @ m), den

    def own(self, energy: float) -> tuple[Tensor, float]:
        """O(u) as orthonormal rows ((0, d) when M writes zero on every row: rank 0) and the share of M's write it holds."""
        if self.GM is not None:
            ev, vec = torch.linalg.eigh(self.GM)
            ev, vec = ev.flip(0).clamp(min=0), vec.flip(1)
            tot = float(ev.sum())
            if tot <= 0:
                return torch.zeros(0, self.d_out, dtype=torch.float64), 1.0
            k = min(int((ev.cumsum(0) < energy * tot).sum()) + 1, len(ev))
            return vec[:, :k].T.contiguous(), float(ev[:k].sum()) / tot
        assert self.S is not None and self.E is not None
        m = self.E[2]
        if float(self.S[5]) <= 0 or float(m @ m) <= 0:
            return torch.zeros(0, self.d_out, dtype=torch.float64), 1.0
        return (m / m.norm()).reshape(1, -1), 1.0


def span(parts: list[Tensor], d: int, rtol: float) -> Tensor:
    parts = [p for p in parts if p.shape[0]]
    if not parts:
        return torch.zeros(0, d, dtype=torch.float64)
    _, s, Vh = torch.linalg.svd(torch.cat(parts), full_matrices=False)
    return Vh[: int((s > rtol * s[0]).sum())]


def gather(sites: list[Site], accs: list[Acc]) -> dict[int, Var]:
    """global index -> Var, every accumulated variable."""
    out: dict[int, Var] = {}
    for s, a in zip(sites, accs, strict=True):
        if len(a.dirs):
            E1, E2, m = s.vectors(a.dirs)
            for r, j in enumerate(a.dirs.tolist()):
                out[s.off + j] = Var(s.d_out, S=a.S[r].cpu(), E=torch.stack([E1[r], E2[r], m[r]]).cpu())
        for j in a.GD:
            out[s.off + j] = Var(s.d_out, GD=a.GD[j].cpu(), GM=a.GM[j].cpu(), X=float(a.X[j]))
    return out


def strict(num: float, den: float) -> float:
    t = lambda v: torch.tensor([v], dtype=torch.float64)
    return float(SI.explained(t(num), t(den))[0])


def read(sites: list[Site], st: Sets, V: dict[int, Var], energy: float, rtol: float) -> list[dict[str, Any]]:
    """Per member of the read set: dd, num, den, and the strict, own, needed and matched parts of dd, with the dims."""
    where, own = locate(sites), {g: v.own(energy) for g, v in V.items()}
    rows = []
    for u in st.N:
        si, j = where[u]
        s, v = sites[si], V[u]
        O = own[u][0]
        ref = [w for w in st.R if where[w][0] == si and w != u and where[w][1] != s.rem and own[w][0].shape[0]]
        cand = [s.off + c for c in st.draw[si] if own[s.off + c][0].shape[0]][: len(ref)]
        S = span([O, *(own[w][0] for w in ref)], s.d_out, rtol)
        A = span([O, *(own[w][0] for w in cand)], s.d_out, rtol)
        num, den = v.num_den()
        rows.append({"var": u, "site": s.module, "rem": j == s.rem, "gram": is_gram(s, j), "dd": v.part(None), "num": num, "den": den,
                     "strict": strict(num, den), "own": v.part(O), "needed": v.part(S), "matched": v.part(A), "dim_O": O.shape[0],
                     "dim_S": S.shape[0], "dim_A": A.shape[0], "n_ref": len(ref), "n_matched": len(cand), "d_out": s.d_out,
                     "own_held": own[u][1]})  # fmt: skip
    return rows


def own_held(V: dict[int, Var], energy: float) -> float:
    """The smallest share of M's write that any accumulated variable's own space holds (1 for one-direction ones)."""
    return min((v.own(energy)[1] for v in V.values()), default=1.0)


def pooled(rows: list[dict[str, Any]], rem: bool) -> dict[str, Any]:
    """Change-weighted shares over the read set, remainders out (rem False) or in; every share is 1 when dd is 0 (the
    do-nothing convention). The rotation share and its two nulls are None when nothing leaves the own spaces."""
    r = [x for x in rows if rem or not x["rem"]]
    dd = sum(x["dd"] for x in r)
    sh = lambda k: sum(x[k] for x in r) / dd if dd > 0 else 1.0
    out: dict[str, Any] = {"n": len(r), "dd": dd, "strict": sh("strict"), "own": sh("own"), "needed": sh("needed"), "matched": sh("matched"),
                           "nu": sum(x["dd"] * x["dim_S"] / x["d_out"] for x in r) / dd if dd > 0 else None}  # fmt: skip
    left = sum(x["dd"] - x["own"] for x in r)
    ok = dd > 0 and left > 1e-6 * dd
    out["rotation"] = sum(x["needed"] - x["own"] for x in r) / left if ok else None
    out["rotation_matched_null"] = sum(x["matched"] - x["own"] for x in r) / left if ok else None
    out["rotation_nu_null"] = sum((x["dd"] - x["own"]) * (x["dim_S"] - x["dim_O"]) / max(x["d_out"] - x["dim_O"], 1) for x in r) / left if ok else None
    return out


def ordering_misses(rows: list[dict[str, Any]], tol: float) -> list[str]:
    """strict <= own for one-direction variables, own <= needed and own <= matched for all, each within tol * dd."""
    out = []
    for x in rows:
        t = tol * max(x["dd"], 1e-300)
        if not x["gram"] and x["strict"] > x["own"] + t:
            out.append(f"var {x['var']}: strict {x['strict']!r} above own {x['own']!r}")
        if x["own"] > x["needed"] + t or x["own"] > x["matched"] + t:
            out.append(f"var {x['var']}: own {x['own']!r} above needed {x['needed']!r} or matched {x['matched']!r}")
    return out
