"""M's list of variables and the ranking of interventions on it (E180, cell P1.20: the fourth reuse wording, testing only,
not adopted). Plan: reports/plans/2026-11-13_plan_simplest-intervention-ranking.md. Added 2026-11-13.

The list, fixed on M alone: the 4 x 3,072 MLP neurons (down_proj's input), the top k principal directions of the residual
stream at in0..in3 and out3 (last position, ordinary windows), the input tokens at the last two positions (read only) and
the output logits (write only). Every write and read is at the last position, where F is read.

An intervention I writes variables: v <- v + sd_v (g vt + b + sum_j w_j ut_j), with vt = (v - mu_v) / sd_v and ut_j the
standardised value of an upstream read in the intervened run (mu, sd: M on the item's scoring windows). g is the gain, b
the shift; its size n is its writes plus its reads. Rung 3 adds free rank-one edits
h_s <- h_s + rms_s d (a . (h_r - mu_r) / rms_r + c). Closure = 1 - KL(F || M_I) / KL(F || M), at the last position.

The greedy search scores every candidate number (a write's gain or shift; a (write, read) weight) by g^2 / h, its KL
gradient g over its Gauss-Newton curvature h (sampled for hidden variables, exact for logits), summed over the scoring
windows: twice the KL a lone Newton step on it would remove, whatever the variable's units. The K best are line-searched
along their Newton steps, the one with the lowest KL is added, every number is refitted by Adam, and the closure is read.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from commute import variable_read as VR
from commute.hidden_maps import FINAL, Edit, Fn, chunks, merged, run
from commute.steering import Net

D_MLP = 3072
MLP = ("mlp0", "mlp1", "mlp2", "mlp3")
RES = ("in0", "in1", "in2", "in3", "out3")
ORDER = {"in0": 0, "mlp0": 1, "in1": 2, "mlp1": 3, "in2": 4, "mlp2": 5, "in3": 6, "mlp3": 7, "out3": 8}  # forward order
LOGIT, TOKEN = 9, -1  # the logits come after every site, the input tokens before
TOK_POS = (-1, -2)
EDIT_SITES = ("in1", "in2", "in3", "out3")
PAIRS3 = tuple((r, s) for i, r in enumerate(EDIT_SITES) for s in EDIT_SITES[i:])  # (read site, write site), r <= s
LISTS = {"neurons": "nlt", "directions": "dlt", "all": "ndlt"}  # n neurons, d directions, l logits, t tokens
EPS = 1e-9  # a closure exactly at a bar passes


@dataclass(frozen=True)
class Sizes:
    n_score: int = 4096  # fit windows of the scoring pass (and of mu, sd)
    n_opt: int = 16_384  # fit windows Adam and the line search draw from
    n_read: int = 16_384  # read windows the closure is read on
    batch: int = 512  # Adam and scoring batch
    eval_batch: int = 2048
    line: int = 1024  # windows each line-search point reads
    samples: int = 2  # Gauss-Newton samples per window
    K: int = 4  # candidates line-searched per greedy step
    chunk: int = 512  # write variables per block of the pair scores
    steps: int = 50  # Adam steps per refit (rungs 1, 2)
    lr: float = 0.02  # on each number over its scale max(|number|, 1), decayed linearly to 0
    steps3: int = 120  # Adam steps per rung-3 fit
    lr3: float = 0.01  # on a, d, c of a rank-one edit
    cap: int = 16
    mults: tuple[float, ...] = (1 / 16, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 16.0)  # line search, multiples of the Newton step


def last_only(x: Tensor) -> Tensor:
    return x[:, -1:]


# ---- the list -------------------------------------------------------------------------------------------------------


@dataclass
class Space:
    """Index spaces. Writes: [neurons | directions | logits]; reads: [neurons | directions | tokens at TOK_POS]."""

    pcs: dict[str, Tensor]  # site -> (d, k), orthonormal columns, on the device
    V: int

    @property
    def k(self) -> int:
        return self.pcs["in0"].shape[1]

    @property
    def n_dense(self) -> int:
        return len(MLP) * D_MLP + len(RES) * self.k

    @property
    def n_write(self) -> int:
        return self.n_dense + self.V

    @property
    def n_read(self) -> int:
        return self.n_dense + len(TOK_POS) * self.V

    def site(self, i: int) -> tuple[str, int]:
        nn = len(MLP) * D_MLP
        return (MLP[i // D_MLP], i % D_MLP) if i < nn else (RES[(i - nn) // self.k], (i - nn) % self.k)

    def name(self, i: int, read: bool = False) -> str:
        if i < self.n_dense:
            s, j = self.site(i)
            return f"{s}#{j}" if s in MLP else f"{s}.pc{j}"
        p, t = divmod(i - self.n_dense, self.V)
        return f"tok{TOK_POS[p]}#{t}" if read else f"logit#{t}"

    def masks(self, lst: str, device: str) -> dict[str, Tensor]:
        """Allowed writes and reads of a list, and each variable's place in the forward order (a read must come first)."""
        nn, nd = len(MLP) * D_MLP, len(RES) * self.k
        kind = torch.cat([torch.zeros(nn), torch.ones(nd)]).long()
        order = torch.cat([torch.tensor([ORDER[s] for s in MLP]).repeat_interleave(D_MLP),
                           torch.tensor([ORDER[s] for s in RES]).repeat_interleave(self.k)])  # fmt: skip
        kw, kr = torch.cat([kind, torch.full((self.V,), 2)]), torch.cat([kind, torch.full((2 * self.V,), 3)])
        codes = torch.tensor(["ndlt".index(ch) for ch in LISTS[lst]])
        out = {"w": torch.isin(kw, codes), "r": torch.isin(kr, codes),
               "ow": torch.cat([order, torch.full((self.V,), LOGIT)]), "or": torch.cat([order, torch.full((2 * self.V,), TOKEN)])}  # fmt: skip
        return {k: v.to(device) for k, v in out.items()}


@torch.no_grad()
def principal_directions(net: Net, windows: Tensor, k: int, batch: int) -> dict[str, Tensor]:
    """The top k principal directions of M's residual at each RES site, last position, over `windows`: {site: (d, k),
    site_mean: (d,), site_var: (k,)}, eigenvalues descending, float32 on the CPU."""
    s1: dict[str, Tensor] = {}
    s2: dict[str, Tensor] = {}
    for t in chunks(windows, batch):
        seen = run(net, t, sites=RES, edit={FINAL: last_only})[1]
        for s in RES:
            h = seen[s][:, -1].double()
            s1[s], s2[s] = s1.get(s, 0.0) + h.sum(0), s2.get(s, 0.0) + h.T @ h  # type: ignore[operator]
    out, n = {}, windows.shape[0]
    for s in RES:
        mu = s1[s] / n
        ev, vec = torch.linalg.eigh(s2[s] / n - torch.outer(mu, mu))
        top = ev.argsort(descending=True)[:k]
        out[s], out[f"{s}_mean"], out[f"{s}_var"] = vec[:, top].float().cpu(), mu.float().cpu(), ev[top].float().cpu()
    return out


@dataclass
class Stats:
    mu: Tensor  # (n_write,) M's mean of each dense variable, then each logit
    sd: Tensor  # (n_write,) its sd, at least 1e-6
    tok_f: Tensor  # (2, V) each token's frequency at TOK_POS
    res_mu: dict[str, Tensor]
    res_rms: dict[str, float]  # sqrt(mean squared distance from res_mu / d)


def dense_cols(sp: Space, by_site: dict[str, Tensor]) -> Tensor:
    """(B, n_dense): the neurons' values, then the residual's projections on the principal directions."""
    return torch.cat([by_site[s] for s in MLP] + [by_site[s] @ sp.pcs[s] for s in RES], 1)


@torch.no_grad()
def stats(net: Net, sp: Space, toks: Tensor, batch: int) -> Stats:
    s1: Tensor | float = 0.0
    s2: Tensor | float = 0.0
    r1: dict[str, Tensor] = {}
    r2: dict[str, float] = dict.fromkeys(RES, 0.0)
    for t in chunks(toks, batch):
        z, seen = run(net, t, sites=tuple(ORDER), edit={FINAL: last_only})
        last = {s: v[:, -1] for s, v in seen.items()}
        x = torch.cat([dense_cols(sp, last), z[:, -1]], 1).double()
        s1, s2 = s1 + x.sum(0), s2 + (x * x).sum(0)
        for s in RES:
            r1[s] = r1.get(s, 0.0) + last[s].double().sum(0)  # type: ignore[operator]
            r2[s] += float((last[s].double() ** 2).sum())
    n = toks.shape[0]
    mu = torch.as_tensor(s1) / n
    sd = (torch.as_tensor(s2) / n - mu**2).clamp_min(0).sqrt().clamp_min(1e-6)
    f = torch.stack([torch.bincount(toks[:, p], minlength=sp.V)[: sp.V].double() / n for p in TOK_POS])
    rmu = {s: r1[s] / n for s in RES}
    rms = {s: max((r2[s] / n - float((rmu[s] ** 2).sum())) / rmu[s].numel(), 1e-12) ** 0.5 for s in RES}
    dev = sp.pcs["in0"].device
    return Stats(mu.float().to(dev), sd.float().to(dev), f.float().to(dev), {s: v.float() for s, v in rmu.items()}, rms)


# ---- an intervention and M_I ----------------------------------------------------------------------------------------


@dataclass
class Write:
    var: int
    reads: list[int]
    theta: Tensor  # (2 + len(reads),): gain, shift, then one weight per read (standardised units)


@dataclass
class RankOne:
    r: str
    s: str
    a: Tensor
    d: Tensor
    c: Tensor


@dataclass
class Interv:
    writes: list[Write] = field(default_factory=list)
    edits: list[RankOne] = field(default_factory=list)

    @property
    def n(self) -> int:
        return sum(1 + len(w.reads) for w in self.writes)

    def names(self, sp: Space) -> list[str]:
        return [x for w in self.writes for x in (sp.name(w.var), *(f"{sp.name(r, True)}->{sp.name(w.var)}" for r in w.reads))]

    def find(self, var: int) -> Write | None:
        return next((w for w in self.writes if w.var == var), None)

    def clone(self) -> Interv:
        return Interv([Write(w.var, list(w.reads), w.theta.detach().clone()) for w in self.writes],
                      [RankOne(e.r, e.s, e.a.detach().clone(), e.d.detach().clone(), e.c.detach().clone()) for e in self.edits])

    def to_json(self) -> list[dict[str, Any]]:
        return [{"var": w.var, "reads": w.reads, "theta": w.theta.tolist()} for w in self.writes]

    @staticmethod
    def from_json(js: list[dict[str, Any]], device: str) -> Interv:
        return Interv([Write(w["var"], list(w["reads"]), torch.tensor(w["theta"], device=device)) for w in js])


@dataclass
class Model:
    net: Net
    sp: Space
    st: Stats


def _read(m: Model, i: int, store: dict[str, Tensor], toks: Tensor) -> Tensor:
    """A read's standardised value in the intervened run."""
    sp, st = m.sp, m.st
    if i < sp.n_dense:
        s, j = sp.site(i)
        raw = store[s][:, j] if s in MLP else store[s] @ sp.pcs[s][:, j]
        return (raw - st.mu[i]) / st.sd[i]
    p, t = divmod(i - sp.n_dense, sp.V)
    f = st.tok_f[p, t]
    return ((toks[:, TOK_POS[p]] == t).float() - f) / (f * (1 - f)).clamp_min(1e-12).sqrt()


def _shift(m: Model, w: Write, v: Tensor, store: dict[str, Tensor], toks: Tensor) -> Tensor:
    """The change a write makes to its variable's raw value v (B,)."""
    mu, sd = m.st.mu[w.var], m.st.sd[w.var]
    out = w.theta[0] * (v - mu) / sd + w.theta[1]
    for k, r in enumerate(w.reads):
        out = out + w.theta[2 + k] * _read(m, r, store, toks)
    return sd * out


def _delta(m: Model, I: Interv, site: str, h: Tensor, store: dict[str, Tensor], toks: Tensor) -> Tensor:
    """The writes and rank-one edits at `site` on its last-position value h (B, d), all read from h before any of them."""
    out = torch.zeros_like(h)
    for w in I.writes:
        if w.var >= m.sp.n_dense or m.sp.site(w.var)[0] != site:
            continue
        j = m.sp.site(w.var)[1]
        if site in MLP:
            out = out.index_add(1, torch.tensor([j], device=h.device), _shift(m, w, h[:, j], store, toks)[:, None])
        else:
            e = m.sp.pcs[site][:, j]
            out = out + _shift(m, w, h @ e, store, toks)[:, None] * e
    for e3 in I.edits:
        if e3.s == site:
            x = ((h if e3.r == site else store[e3.r]) - m.st.res_mu[e3.r]) / m.st.res_rms[e3.r]
            out = out + m.st.res_rms[site] * (x @ e3.a + e3.c)[:, None] * e3.d
    return out


def forward(m: Model, I: Interv, toks: Tensor, probes: dict[str, Tensor] | None = None,
            grad: bool = False) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """M_I on `toks`: M's last-position logits before any logit write (B, V), M_I's log-probabilities there, and each site's
    last-position value after its writes. `probes` (zeros that require grad) are added after the writes at each site and to
    the logits, so their gradients are per-window derivatives in each variable's raw units."""
    store: dict[str, Tensor] = {}
    toks = toks.to(m.net.device)

    def hook(site: str) -> Fn:
        def f(x: Tensor) -> Tensor:
            h = x[:, -1]
            h = h + _delta(m, I, site, h, store, toks)
            if probes is not None:
                h = h + probes[site]
            store[site] = h
            return torch.cat([x[:, :-1], h[:, None]], 1)

        return f

    edit: Edit = {s: hook(s) for s in ORDER} | {FINAL: last_only}
    z = run(m.net, toks, edit=edit, grad=grad)[0][:, -1]
    zw = z
    for w in I.writes:
        if w.var >= m.sp.n_dense:
            t = w.var - m.sp.n_dense
            zw = zw.index_add(1, torch.tensor([t], device=z.device), _shift(m, w, z[:, t], store, toks)[:, None])
    if probes is not None:
        zw = zw + probes["logit"]
    return z, torch.log_softmax(zw, -1), store


def kl_rows(lpF: Tensor, lq: Tensor) -> Tensor:
    return (lpF.exp() * (lpF - lq)).sum(-1)


@torch.no_grad()
def mean_kl(m: Model, I: Interv, toks: Tensor, lpF: Tensor, batch: int) -> float:
    tot = sum(float(kl_rows(lpF[i : i + batch], forward(m, I, toks[i : i + batch])[1]).sum()) for i in range(0, toks.shape[0], batch))
    return tot / toks.shape[0]


# ---- the greedy search ----------------------------------------------------------------------------------------------


@dataclass
class Scores:
    G: Tensor  # (N, n_write) per-window KL gradient of a raw shift of each write variable
    H: Tensor  # (N, n_write) its Gauss-Newton curvature
    U: Tensor  # (N, n_dense) each dense read's centred raw value
    tok: Tensor  # (N, 2) the tokens at TOK_POS
    gain: tuple[Tensor, Tensor]  # (n_write,) summed g and h of each write's gain


def score_pass(m: Model, I: Interv, toks: Tensor, lpF: Tensor, sz: Sizes, seed: int) -> Scores:
    dev, sp, st = m.net.device, m.sp, m.st
    gen = torch.Generator(dev).manual_seed(seed)
    d = sp.pcs["in0"].shape[0]
    Gs, Hs, Us = [], [], []
    ga, ha = torch.zeros(sp.n_write, device=dev), torch.zeros(sp.n_write, device=dev)
    for i in range(0, toks.shape[0], sz.batch):
        t, lp = toks[i : i + sz.batch], lpF[i : i + sz.batch]
        pr = {s: torch.zeros(t.shape[0], D_MLP if s in MLP else d, device=dev, requires_grad=True) for s in ORDER}
        pr["logit"] = torch.zeros(t.shape[0], sp.V, device=dev, requires_grad=True)
        z, lq, store = forward(m, I, t, pr, grad=True)
        sites = list(ORDER)
        g = torch.autograd.grad(kl_rows(lp, lq).sum(), [pr[s] for s in [*sites, "logit"]], retain_graph=True)
        y = torch.multinomial(lq.detach().exp(), sz.samples, replacement=True, generator=gen)
        h = torch.zeros(t.shape[0], sp.n_dense, device=dev)
        for k in range(sz.samples):
            r = torch.autograd.grad(lq.gather(1, y[:, k : k + 1]).sum(), [pr[s] for s in sites], retain_graph=k + 1 < sz.samples)
            h += dense_cols(sp, dict(zip(sites, r, strict=True))) ** 2 / sz.samples
        q = lq.detach().exp()
        G = torch.cat([dense_cols(sp, dict(zip(sites, g[:-1], strict=True))), g[-1]], 1)
        H = torch.cat([h, q * (1 - q)], 1)
        own = torch.cat([dense_cols(sp, {s: v.detach() for s, v in store.items()}), z.detach()], 1) - st.mu
        ga, ha = ga + (G * own).sum(0), ha + (H * own**2).sum(0)
        Gs.append(G)
        Hs.append(H)
        Us.append(own[:, : sp.n_dense])
    tok = torch.stack([toks[:, p] for p in TOK_POS], 1).to(dev)
    return Scores(torch.cat(Gs), torch.cat(Hs), torch.cat(Us), tok, (ga, ha))


@dataclass
class Cand:
    score: float  # g^2 / h
    var: int
    read: int | None
    which: int  # 0 gain, 1 shift, 2 weight of `read`
    step: float  # the Newton step, in theta's units


def _read_sd(m: Model, r: int) -> float:
    if r < m.sp.n_dense:
        return float(m.st.sd[r])
    p, t = divmod(r - m.sp.n_dense, m.sp.V)
    f = float(m.st.tok_f[p, t])
    return max(f * (1 - f), 1e-24) ** 0.5


def _pair_block(sc: Scores, f: Tensor, a: int, b: int) -> tuple[Tensor, Tensor]:
    """g and h of each (write in [a, b), read) weight on the centred read: (b - a, n_read) each."""
    Gc, Hc = sc.G[:, a:b], sc.H[:, a:b]
    g, h = [Gc.T @ sc.U], [Hc.T @ sc.U**2]
    tg, th = Gc.sum(0)[:, None], Hc.sum(0)[:, None]
    for p in range(len(TOK_POS)):
        hg = torch.zeros(b - a, f.shape[1], device=Gc.device).index_add_(1, sc.tok[:, p], Gc.T)
        hh = torch.zeros(b - a, f.shape[1], device=Gc.device).index_add_(1, sc.tok[:, p], Hc.T)
        g.append(hg - f[p] * tg)
        h.append(hh * (1 - 2 * f[p]) + th * f[p] ** 2)  # sum of H (1[x = t] - f)^2
    return torch.cat(g, 1), torch.cat(h, 1)


def candidates(m: Model, sc: Scores, I: Interv, mk: dict[str, Tensor], reads_on: bool, sz: Sizes) -> list[Cand]:
    """The K best numbers to add, by g^2 / h; writes already chosen and pairs already chosen are out, and a new number must
    keep n <= cap. Ties go to the lower index (topk's order)."""
    sp, dev = m.sp, m.net.device
    gb, hb = sc.G.sum(0), sc.H.sum(0)
    floor = 1e-10 * float(hb.max().clamp_min(1e-30))  # a curvature this small is rounding, not an effect
    taken = torch.zeros(sp.n_write, dtype=torch.bool, device=dev)
    taken[[w.var for w in I.writes]] = True
    out: list[Cand] = []
    if I.n + 1 <= sz.cap:
        for which, (g, h) in enumerate((sc.gain, (gb, hb))):
            s = torch.where(mk["w"] & ~taken & (h > floor), g * g / h.clamp_min(floor), 0)
            for v, i in zip(*s.topk(sz.K), strict=True):
                step = -float(g[i] / h[i]) / (1.0 if which == 0 else float(m.st.sd[i]))
                out += [Cand(float(v), int(i), None, which, step)] if v > 0 else []
    if reads_on and I.n + 1 <= sz.cap:
        chosen = {(w.var, r) for w in I.writes for r in w.reads}
        rows = mk["w"] & (taken | (I.n + 2 <= sz.cap))  # a pair on a new write costs two names
        for a in range(0, sp.n_write, sz.chunk):
            b = min(a + sz.chunk, sp.n_write)
            if not rows[a:b].any():
                continue
            g, h = _pair_block(sc, m.st.tok_f, a, b)
            ok = rows[a:b, None] & mk["r"][None] & (mk["or"][None] < mk["ow"][a:b, None]) & (h > floor)
            for v, r in chosen:
                if a <= v < b:
                    ok[v - a, r] = False
            s = torch.where(ok, g * g / h.clamp_min(floor), 0).flatten()
            for val, i in zip(*s.topk(sz.K), strict=True):
                v, r = a + int(i) // sp.n_read, int(i) % sp.n_read
                step = -float(g.flatten()[i] / h.flatten()[i]) * _read_sd(m, r) / float(m.st.sd[v])
                out += [Cand(float(val), v, r, 2, step)] if val > 0 else []
            out = sorted(out, key=lambda c: -c.score)[: sz.K]
    return sorted(out, key=lambda c: -c.score)[: sz.K]


def with_cand(I: Interv, c: Cand, value: float, device: str) -> Interv:
    J = I.clone()
    w = J.find(c.var)
    if w is None:
        w = Write(c.var, [], torch.zeros(2, device=device))
        J.writes.append(w)
    if c.which == 2:
        assert c.read is not None
        w.reads.append(c.read)
        w.theta = torch.cat([w.theta, w.theta.new_tensor([value])])
    else:
        w.theta[c.which] = value
    return J


def line_search(m: Model, I: Interv, cands: list[Cand], toks: Tensor, lpF: Tensor, sz: Sizes) -> tuple[Interv, Cand, float]:
    """Each candidate at each multiple of its Newton step; the (candidate, multiple) of the lowest KL on `toks`."""
    best: tuple[float, Interv, Cand, float] | None = None
    for c in cands:
        for t in sz.mults:
            J = with_cand(I, c, t * c.step, m.net.device)
            k = mean_kl(m, J, toks, lpF, sz.eval_batch)
            if best is None or k < best[0]:
                best = (k, J, c, t)
    assert best is not None
    return best[1], best[2], best[3]


def refit(m: Model, I: Interv, toks: Tensor, lpF: Tensor, steps: int, sz: Sizes, seed: int) -> None:
    """Adam on every number of I: each write's numbers over their scale max(|number|, 1), lr `sz.lr`; the rank-one edits'
    a, d, c directly, lr `sz.lr3`; both decayed linearly to 0; batches of `sz.batch` drawn from `toks` (seeded)."""
    scales = [w.theta.abs().clamp_min(1.0) for w in I.writes]
    phis = [(w.theta / s).detach().requires_grad_() for w, s in zip(I.writes, scales, strict=True)]
    raw = [p.requires_grad_() for e in I.edits for p in (e.a, e.d, e.c)]
    groups = [g for g in ({"params": phis, "lr": sz.lr}, {"params": raw, "lr": sz.lr3}) if g["params"]]
    if not groups:
        return
    opt = torch.optim.Adam(groups)
    base = [g["lr"] for g in opt.param_groups]
    gen = torch.Generator().manual_seed(seed)
    for step in range(steps):
        for g, lr in zip(opt.param_groups, base, strict=True):
            g["lr"] = lr * (1 - step / steps)
        idx = torch.randint(0, toks.shape[0], (sz.batch,), generator=gen)
        for w, p, s in zip(I.writes, phis, scales, strict=True):
            w.theta = p * s
        loss = kl_rows(lpF[idx.to(lpF.device)], forward(m, I, toks[idx], grad=True)[1]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    for w, p, s in zip(I.writes, phis, scales, strict=True):
        w.theta = (p * s).detach()
    for p in raw:
        p.requires_grad_(False)


@dataclass
class Data:
    score: Tensor  # windows, CPU
    opt: Tensor
    read: Tensor
    lp_score: Tensor  # F's log-probabilities at the last position, on the device
    lp_opt: Tensor
    lp_read: Tensor
    kl0: float = 0.0  # mean KL(F || M) on `read`


@torch.no_grad()
def prepare(net: Net, sp: Space, F: VR.Target, fit: Tensor, read: Tensor, sz: Sizes, seed: int) -> tuple[Model, Data]:
    """Seeded subsets of the fit half (scoring, then Adam) and of the read half, F's log-probabilities on them, M's stats on
    the scoring windows, and KL(F || M) on the read subset."""
    assert fit.shape[0] >= sz.n_score + sz.n_opt and read.shape[0] >= sz.n_read
    gen = torch.Generator().manual_seed(seed)
    pf, pr = torch.randperm(fit.shape[0], generator=gen), torch.randperm(read.shape[0], generator=gen)
    w = (fit[pf[: sz.n_score]], fit[pf[sz.n_score : sz.n_score + sz.n_opt]], read[pr[: sz.n_read]])

    def lp(x: Tensor) -> Tensor:
        return torch.cat([torch.log_softmax(F(t, edit={FINAL: last_only})[0][:, -1], -1) for t in chunks(x, sz.eval_batch)])

    d = Data(w[0], w[1], w[2], lp(w[0]), lp(w[1]), lp(w[2]))
    m = Model(net, sp, stats(net, sp, d.score, sz.eval_batch))
    d.kl0 = mean_kl(m, Interv(), d.read, d.lp_read, sz.eval_batch)
    return m, d


def closure(m: Model, I: Interv, d: Data, sz: Sizes) -> float:
    return 1 - mean_kl(m, I, d.read, d.lp_read, sz.eval_batch) / d.kl0


def greedy(m: Model, d: Data, I: Interv, curve: list[dict[str, Any]], mk: dict[str, Tensor], reads_on: bool, sz: Sizes,
           top_bar: float, tick: Callable[[], None], seed: int) -> None:
    """Rung 1 (`reads_on` False: writes only) or rung 2: add the best number, refit, read the closure, until n = cap, no
    candidate is left, or the closure reaches `top_bar` (no larger n can lower a reading then). Resumes from I and `curve`,
    which it extends in place; `tick` runs after every step."""
    while I.n < sz.cap and not (curve and curve[-1]["closure"] >= top_bar - EPS):
        s = seed + len(curve)
        cands = candidates(m, score_pass(m, I, d.score, d.lp_score, sz, s), I, mk, reads_on, sz)
        if not cands:
            break
        J, c, t = line_search(m, I, cands, d.opt[: sz.line], d.lp_opt[: sz.line], sz)
        refit(m, J, d.opt, d.lp_opt, sz.steps, sz, s)
        I.writes, I.edits = J.writes, J.edits
        curve.append({"n": I.n, "closure": closure(m, I, d, sz), "names": I.names(m.sp), "score": c.score, "mult": t})
        tick()


def rung3(m: Model, d: Data, base: Interv, fits: list[dict[str, Any]], sz: Sizes, top_bar: float,
          tick: Callable[[], None], seed: int) -> None:
    """One, then two, free rank-one edits at each site pair (r <= s), every number refitted with `base`'s (the rung-2 names
    kept). a starts seeded with unit-scale a . x, d and c at 0. Two edits are skipped once one edit reached `top_bar`.
    Resumes from `fits`, which it extends in place."""
    dm = m.sp.pcs["in0"].shape[0]
    for k in (1, 2):
        if any(f["closure"] >= top_bar - EPS for f in fits if f["edits"] < k):
            return
        for pair in PAIRS3:
            if any(f["edits"] == k and tuple(f["pair"]) == pair for f in fits):
                continue
            I, gen = base.clone(), torch.Generator().manual_seed(seed + 10 * k + PAIRS3.index(pair))
            z = torch.zeros(dm, device=m.net.device)
            I.edits = [RankOne(*pair, (torch.randn(dm, generator=gen) / dm**0.5).to(z.device), z.clone(), z.new_zeros(())) for _ in range(k)]
            refit(m, I, d.opt, d.lp_opt, sz.steps3, sz, seed)
            fits.append({"edits": k, "pair": list(pair), "n": base.n, "closure": closure(m, I, d, sz)})
            tick()


# ---- the reading ----------------------------------------------------------------------------------------------------


def reached(rec: dict[str, Any], bar: float) -> bool:
    """Whether any rung of one list's record closes at least `bar`."""
    cl = [e["closure"] for r in ("1", "2") for e in rec.get(r, {}).get("curve", [])] + [f["closure"] for f in rec.get("3", {}).get("fits", [])]
    return any(c >= bar - EPS for c in cl)


def reading(kl0: float, floor: float, rec: dict[str, Any], bar: float) -> dict[str, Any]:
    """The lowest (rung, n) closing at least `bar`: rung 0 if KL(F || M) is under `floor` (no closure is divided then); rung 1,
    then 2, at the smallest n on its curve; rung 3 at the fewest edits (n = the rung-2 names kept, the best pair named);
    else rung 4. A NaN closure passes nothing; a closure exactly at the bar passes."""
    if kl0 < floor:
        return {"rung": 0, "n": 0, "names": []}
    for r in (1, 2):
        for e in rec.get(str(r), {}).get("curve", []):
            if e["closure"] >= bar - EPS:
                return {"rung": r, "n": e["n"], "names": e["names"]}
    for k in (1, 2):
        hits = [f for f in rec.get("3", {}).get("fits", []) if f["edits"] == k and f["closure"] >= bar - EPS]
        if hits:
            f = max(hits, key=lambda f: f["closure"])
            return {"rung": 3, "n": f["n"], "edits": k, "pair": f["pair"]}
    return {"rung": 4, "n": None}


# ---- built edits on M's own list (run 1's controls) -------------------------------------------------------------------


def edit_target(net: Net, make: Callable[[], Edit]) -> VR.Target:
    """F = M with the edit `make()` (a fresh one per call, so a stash is not shared), before any caller's edit."""

    def f(toks: Tensor, sites: Any = (), edit: Edit | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        return run(net, toks, sites=sites, edit=merged(make(), edit or {}))

    return f


def add_unit(j: int, amount: Callable[[], Tensor | float]) -> Fn:
    """Add `amount()` (a number or (B,)) to unit j at the last position."""

    def f(x: Tensor) -> Tensor:
        x = x.clone()
        x[:, -1, j] = x[:, -1, j] + amount()
        return x

    return f
