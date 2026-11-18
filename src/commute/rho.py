"""The working definition's rho read as `theory/2026-11-18_definition_causal_abstraction.md` writes it, on the 67M (cell
P1.30, E208, `experiments/rho_saved_items`; plan and appendix there). Added 2026-11-18. `commute.rule_retention_fixed`
(E195, E196's code) is called for its search only and not changed.

The level aligns every path variable through the identity alignment: each MLP neuron (c_fc row, down_proj column, its
layer's rms_2), each attention head (q, k, v rows, o_proj columns, its layer's rms_1), the token embedding (wte) and the
unembedding (ln_f, lm_head). An appended neuron or a grafted seventh head is a part M lacks. A part is changed when any
weight of its cell differs from M's; a part F lacks writes zero in F (output column or o_proj columns zero) where M's
does not. A part's value is its write: a neuron's activation times its output column, a head's output after o_proj, the
embedding's row, the unembedding's logits.

Per part and setting v (the residual feeding it, in M's run or in F's), with f = the two writes at v:
    d_c(v) = 1 - cos^2(f_F, f_M)                         (the note's d; 0 when both are zero, 1 when one is)
    d_b(v) = |f_F - f_M|^2 / (|f_F|^2 + |f_M|^2)          (option (b), size-aware; 0 when both are zero)
    main = 1 - mean d_c over M's and F's settings, iii = the same over M's only, b = 1 - mean d_b over both,
    i    = (sum_v <f_F, f_M>)^2 / (sum_v |f_F|^2 sum_v |f_M|^2) over M's settings (E196's aggregate, identity pairing).
A part M lacks reads 0 under every reading. F_K (`forward`): every changed part outside K runs M's mechanism on the
hybrid's input ("replace"), or writes its value from M's own run on the same window ("patch"); a part M lacks outside K
is removed; unchanged parts run their shared mechanism on the hybrid. K: RF.eliminate over the candidates, kept when
1 - KL(F || F_K) / KL(F || M) >= tau.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from commute import graft as G
from commute import rule_retention as R
from commute import rule_retention_fixed as RF

D = R.D_HEAD
Part = tuple[str, int, int]  # ("neuron", l, i), ("head", l, h), ("embed", 0, 0), ("unembed", 0, 0)
READS = ("main", "i", "iii", "b")
EMBED, UNEMBED = ("embed", 0, 0), ("unembed", 0, 0)


def name(u: Part) -> str:
    return {"neuron": f"L{u[1]}.n{u[2]}", "head": f"L{u[1]}.h{u[2]}"}.get(u[0], u[0])


@dataclass
class Layer:
    """One network's weights at one layer, every head and neuron included (grafted ones last)."""

    attn: Any  # M's attention module of that layer: rotary tables only
    rms1: Any
    wq: Tensor  # (H D, d)
    wk: Tensor
    wv: Tensor
    wo: Tensor  # (d, H D)
    rms2: Any
    w_in: Tensor  # (n, d)
    w_out: Tensor  # (d, n)
    gelu: Any

    @property
    def n_head(self) -> int:
        return self.wq.shape[0] // D

    def z(self, x: Tensor) -> Tensor:
        """(B, H, T, D) each head's output before o_proj on residual x (the 67M's manual attention path)."""
        h = self.rms1(x)
        B, T, _ = h.shape
        v = (h @ self.wv.T).view(B, T, self.n_head, D).transpose(1, 2)
        return G.pattern(self.attn, h, self.wq, self.wk, self.n_head) @ v

    def attn_out(self, z: Tensor, keep: Tensor) -> Tensor:
        B, H, T, _ = z.shape
        return (z * keep[None, :, None, None]).transpose(1, 2).reshape(B, T, H * D) @ self.wo.T

    def head_writes(self, x: Tensor) -> Tensor:
        """(B, T, H, d) each head's write after o_proj."""
        return torch.einsum("bhtk,dhk->bthd", self.z(x), self.wo.view(self.wo.shape[0], self.n_head, D))

    def acts(self, y: Tensor) -> Tensor:
        return self.gelu(self.rms2(y) @ self.w_in.T)


@dataclass
class Net:
    wte: Tensor
    layers: list[Layer]
    ln_f: Any
    lm_head: Tensor

    def logits(self, x: Tensor) -> Tensor:
        return self.ln_f(x) @ self.lm_head.T


def unpack(N: R.Model, M: R.Model) -> Net:
    layers = []
    for l, b in enumerate(N.net.h):
        a = b.attn
        g = a.graft if isinstance(a, G.Grafted) else None
        base: Any = a.base if g is not None else a
        w = [base.q_proj.weight, base.k_proj.weight, base.v_proj.weight, base.o_proj.weight]
        if g is not None:
            w = [torch.cat([w[0], g.wq]), torch.cat([w[1], g.wk]), torch.cat([w[2], g.wv]), torch.cat([w[3], g.wo], 1)]
        w_in = torch.cat([b.mlp.c_fc.weight, N.extra[l].w_in]) if l in N.extra else b.mlp.c_fc.weight
        layers.append(Layer(M.net.h[l].attn, b.rms_1, w[0], w[1], w[2], w[3], b.rms_2, w_in, R.cols(N, l), b.mlp.gelu))
    return Net(N.net.wte.weight, layers, N.net.ln_f, N.net.lm_head.weight)


@dataclass
class Pair:
    F: R.Model
    M: R.Model
    f: Net = field(init=False)
    m: Net = field(init=False)

    def __post_init__(self) -> None:
        self.f, self.m = unpack(self.F, self.M), unpack(self.M, self.M)


# ---- the parts ------------------------------------------------------------------------------------------------------


def changed(p: Pair) -> tuple[list[Part], list[Part]]:
    """(changed parts F has, parts F lacks): exact weight comparison, cell by cell."""
    ch: list[Part] = []
    lacks: list[Part] = []
    for l, (F, M) in enumerate(zip(p.f.layers, p.m.layers, strict=True)):
        r1 = not torch.equal(F.rms1.weight, M.rms1.weight)
        for h in range(F.n_head):
            s = slice(h * D, (h + 1) * D)
            if h >= M.n_head or r1 or any(not torch.equal(a[s], b[s]) for a, b in ((F.wq, M.wq), (F.wk, M.wk), (F.wv, M.wv))) or not torch.equal(F.wo[:, s], M.wo[:, s]):
                (lacks if h < M.n_head and not F.wo[:, s].any() and M.wo[:, s].any() else ch).append(("head", l, h))
        n = M.w_in.shape[0]
        diff = (F.w_in[:n] != M.w_in).any(1) | (F.w_out[:, :n] != M.w_out).any(0) | (not torch.equal(F.rms2.weight, M.rms2.weight))
        gone = ~F.w_out[:, :n].any(0) & M.w_out.any(0)
        ch += [("neuron", l, i) for i in (diff & ~gone).nonzero().squeeze(1).tolist()] + [("neuron", l, i) for i in range(n, F.w_in.shape[0])]
        lacks += [("neuron", l, i) for i in (diff & gone).nonzero().squeeze(1).tolist()]
    ch += [EMBED] if not torch.equal(p.f.wte, p.m.wte) else []
    ch += [UNEMBED] if not (torch.equal(p.f.lm_head, p.m.lm_head) and torch.equal(p.f.ln_f.weight, p.m.ln_f.weight)) else []
    return ch, lacks


def lacked_by_m(p: Pair, u: Part) -> bool:
    return (u[0] == "head" and u[2] >= p.m.layers[u[1]].n_head) or (u[0] == "neuron" and u[2] >= p.m.layers[u[1]].w_in.shape[0])


# ---- F_K ------------------------------------------------------------------------------------------------------------


@dataclass
class Swap:
    """Float masks over F's parts, 1 where the part runs M's mechanism (or takes M's value)."""

    head: list[Tensor]
    neuron: list[Tensor]
    embed: bool
    unembed: bool


def swap(p: Pair, parts: Iterable[Part]) -> Swap:
    """Built on the CPU and moved once: one write per part on the GPU would launch thousands of kernels per evaluation."""
    idx: dict[tuple[str, int], list[int]] = {}
    flags = set()
    for k, l, i in parts:
        if k in ("head", "neuron"):
            idx.setdefault((k, l), []).append(i)
        else:
            flags.add(k)

    def masks(k: str, sizes: list[int]) -> list[Tensor]:
        out = [torch.zeros(n) for n in sizes]
        for l, m in enumerate(out):
            m[idx.get((k, l), [])] = 1.0
        return [m.to(p.f.wte.device) for m in out]

    return Swap(masks("head", [L.n_head for L in p.f.layers]), masks("neuron", [L.w_in.shape[0] for L in p.f.layers]), "embed" in flags, "unembed" in flags)


@dataclass
class MRun:
    """M's own run on one batch of windows: each head's z and each neuron's activation per layer, and the logits."""

    z: list[Tensor]
    a: list[Tensor]
    logits: Tensor


@torch.no_grad()
def m_run(p: Pair, toks: Tensor) -> MRun:
    out, x = MRun([], [], torch.empty(0)), p.m.wte[toks]
    for L in p.m.layers:
        out.z.append(L.z(x))
        x = x + L.attn_out(out.z[-1], torch.ones(L.n_head, device=x.device))
        out.a.append(L.acts(x))
        x = x + out.a[-1] @ L.w_out.T
    out.logits = p.m.logits(x).float()
    return out


@torch.no_grad()
def forward(p: Pair, toks: Tensor, s: Swap, mrun: MRun | None = None, at: slice = slice(None)) -> Tensor:
    """F_K's logits at positions `at`: mrun None = mechanism replacement, else activation patching from M's run."""
    x = (p.m.wte if s.embed else p.f.wte)[toks]
    for l, (F, M) in enumerate(zip(p.f.layers, p.m.layers, strict=True)):
        sh, sn = s.head[l], s.neuron[l]
        out = F.attn_out(F.z(x), 1 - sh)
        if bool(sh[: M.n_head].any()):
            out = out + M.attn_out(mrun.z[l] if mrun else M.z(x), sh[: M.n_head])
        x = x + out
        n = M.w_in.shape[0]
        out = (F.acts(x) * (1 - sn)) @ F.w_out.T
        if bool(sn[:n].any()):
            out = out + ((mrun.a[l] if mrun else M.acts(x)) * sn[:n]) @ M.w_out.T
        x = x + out
    if s.unembed:
        return mrun.logits[:, at] if mrun else p.m.logits(x[:, at]).float()
    return p.f.logits(x[:, at]).float()


# ---- the readings of r ------------------------------------------------------------------------------------------------


def _acc(acc: dict[tuple[Any, str], Tensor], key: Any, src: str, dot: Tensor, nF: Tensor, nM: Tensor) -> None:
    """Add one batch of settings (rows) for parts (columns): the per-setting distances and, on M's settings, the sums."""
    dot, nF, nM = (t.double().reshape(-1, t.shape[-1]) for t in (dot, nF, nM))
    both, one = (nF > 0) & (nM > 0), (nF > 0) ^ (nM > 0)
    cos2 = torch.where(both, dot.pow(2) / (nF * nM).clamp(min=1e-300), torch.where(one, 0.0, 1.0))
    db = torch.where(nF + nM > 0, (nF - 2 * dot + nM).clamp(min=0) / (nF + nM).clamp(min=1e-300), 0.0)
    terms = {"dc": (1 - cos2).sum(0), "db": db.sum(0), "cnt": torch.full_like(nF[0], nF.shape[0]), "one": one.sum(0).double(),
             "zero": ((nF == 0) & (nM == 0)).sum(0).double()}  # fmt: skip
    terms |= {"x": dot.sum(0), "nF": nF.sum(0), "nM": nM.sum(0)} if src == "M" else {}
    for k, v in terms.items():
        acc[(key, k + src)] = acc[(key, k + src)] + v if (key, k + src) in acc else v


def _resid(N: Net, toks: Tensor) -> tuple[list[Tensor], list[Tensor], Tensor]:
    xs, ys, x = [], [], N.wte[toks]
    for L in N.layers:
        xs.append(x)
        x = x + L.attn_out(L.z(x), torch.ones(L.n_head, device=x.device))
        ys.append(x)
        x = x + L.acts(x) @ L.w_out.T
    return xs, ys, x


@torch.no_grad()
def profile(p: Pair, batches: Iterable[Tensor]) -> dict[str, Any]:
    """r of every part F has that M also has, under the four readings, plus, per neuron, its output-column cos^2 c, the
    number of settings where exactly one of its two writes is zero, and the shares of the pooled settings where both
    writes are zero (z) and where both are non-zero (a). The NewGELU of the 67M gives exact zeros, and d_c counts a
    setting where both writes are zero as agreement, one zero as disagreement, so at a neuron main r = z + a c exactly
    (the identity check, amendment 1)."""
    acc: dict[tuple[Any, str], Tensor] = {}
    for toks in batches:
        e = (p.f.wte[toks] * p.m.wte[toks]).sum(-1, keepdim=True), p.f.wte[toks].pow(2).sum(-1, keepdim=True), p.m.wte[toks].pow(2).sum(-1, keepdim=True)
        runs = {"M": _resid(p.m, toks), "F": _resid(p.f, toks)}
        for src, (xs, ys, xf) in runs.items():
            _acc(acc, EMBED, src, *e)
            lF, lM = p.f.logits(xf).double(), p.m.logits(xf).double()
            _acc(acc, UNEMBED, src, (lF * lM).sum(-1, keepdim=True), lF.pow(2).sum(-1, keepdim=True), lM.pow(2).sum(-1, keepdim=True))
            for l, (F, M) in enumerate(zip(p.f.layers, p.m.layers, strict=True)):
                hF, hM = F.head_writes(xs[l])[:, :, : M.n_head].double(), M.head_writes(xs[l]).double()
                _acc(acc, ("head", l), src, (hF * hM).sum(-1), hF.pow(2).sum(-1), hM.pow(2).sum(-1))
                n = M.w_in.shape[0]
                aF, aM = F.acts(ys[l])[..., :n].double(), M.acts(ys[l]).double()
                wF, wM = F.w_out[:, :n].double(), M.w_out.double()
                _acc(acc, ("neuron", l), src, aF * aM * (wF * wM).sum(0), aF.pow(2) * wF.pow(2).sum(0), aM.pow(2) * wM.pow(2).sum(0))
    r: dict[str, dict[Part, float]] = {k: {} for k in READS}
    keys: list[Any] = [EMBED, UNEMBED] + [(k, l) for l in range(len(p.f.layers)) for k in ("head", "neuron")]
    for key in keys:
        g = {k[1]: v for k, v in acc.items() if k[0] == key}
        x, nF, nM = g["xM"], g["nFM"], g["nMM"]
        agg = torch.where((nF > 0) & (nM > 0), x.pow(2) / (nF * nM).clamp(min=1e-300), torch.where((nF > 0) ^ (nM > 0), 0.0, 1.0))
        vals = {"main": 1 - (g["dcM"] + g["dcF"]) / (g["cntM"] + g["cntF"]), "iii": 1 - g["dcM"] / g["cntM"],
                "b": 1 - (g["dbM"] + g["dbF"]) / (g["cntM"] + g["cntF"]), "i": agg}  # fmt: skip
        parts: list[Part] = [EMBED] if key == EMBED else [UNEMBED] if key == UNEMBED else [(str(key[0]), int(key[1]), i) for i in range(len(x))]
        for k, v in vals.items():
            r[k] |= dict(zip(parts, v.tolist(), strict=True))
    out: dict[str, dict[Part, Any]] = {"col_cos2": {}, "one_zero": {}, "both_zero": {}, "both_active": {}}
    for l, (F, M) in enumerate(zip(p.f.layers, p.m.layers, strict=True)):
        n, g = M.w_in.shape[0], {k: acc[(("neuron", l), k + "M")] + acc[(("neuron", l), k + "F")] for k in ("one", "zero", "cnt")}
        wF, wM = F.w_out[:, :n].double(), M.w_out.double()
        c = (wF * wM).sum(0).pow(2) / (wF.pow(2).sum(0) * wM.pow(2).sum(0)).clamp(min=1e-300)
        z = g["zero"] / g["cnt"]
        a = 1 - z - g["one"] / g["cnt"]
        for k, v in (("col_cos2", c.tolist()), ("one_zero", [int(x) for x in g["one"].tolist()]), ("both_zero", z.tolist()), ("both_active", a.tolist())):
            out[k] |= {("neuron", l, i): x for i, x in enumerate(v)}
    return {"r": r} | out


def rvalue(p: Pair, r: dict[Part, float], u: Part) -> float:
    return 0.0 if lacked_by_m(p, u) else r[u]


# ---- the contrast -----------------------------------------------------------------------------------------------------


@dataclass
class Batches:
    """The item's windows in batches, with F's logits (nothing swapped), M's native logits and M's own run."""

    toks: list[Tensor]
    lF: list[Tensor]
    lM: list[Tensor]
    mrun: list[MRun]
    at: slice

    def kl(self, p: Pair, s: Swap, patch: bool) -> float:
        tot = sum(float(R.kl(lf, forward(p, t, s, m if patch else None, self.at)).sum()) for t, lf, m in zip(self.toks, self.lF, self.mrun, strict=True))
        return tot / sum(t.shape[0] for t in self.toks)


def batches(p: Pair, toks: Tensor, size: int, at: slice) -> Batches:
    ts = [toks[i : i + size] for i in range(0, toks.shape[0], size)]
    none = swap(p, [])
    return Batches(ts, [forward(p, t, none, None, at) for t in ts], [p.M.logits(t)[:, at] for t in ts], [m_run(p, t) for t in ts], at)


def gate(p: Pair, b: Batches, parts: list[Part], native: Callable[[Tensor], Tensor]) -> float:
    """Max over windows of max |F_K logit - native logit| / max |native logit| (all positions): K = F's parts not in `parts`."""
    s, worst = swap(p, parts), 0.0
    for t in b.toks:
        ref = native(t)
        worst = max(worst, float(((forward(p, t, s, None) - ref).abs().amax((1, 2)) / ref.abs().amax((1, 2))).max()))
    return worst


def contrast(p: Pair, b: Batches, cand: list[Part], r: dict[str, dict[Part, float]], patch: bool, th: dict[str, Any],
             log: Callable[[str], None], label: str) -> dict[str, Any]:
    """rho for the contrast F against M: per seeded order, the K found (as indices into `cand`) and its mean r under each
    reading."""
    kl0 = sum(float(R.kl(lf, lm).sum()) for lf, lm in zip(b.lF, b.lM, strict=True)) / sum(t.shape[0] for t in b.toks)
    out: dict[str, Any] = {"kl": kl0, "n_candidates": len(cand)}
    if not cand:
        return out | {"reading": "no change"}
    if kl0 <= th["kl_floor"]:
        return out | {"reading": "no contrast"}
    cset = set(cand)

    def closure(S: set[Part]) -> float:
        return 1 - b.kl(p, swap(p, cset - S), patch) / kl0

    out["closure_none_kept"] = closure(set())
    gen, orders, index = torch.Generator().manual_seed(th["order_seed"]), [], {u: i for i, u in enumerate(cand)}
    for k in range(th["orders"]):
        t0 = time.time()
        K, n = RF.eliminate(closure, [cand[i] for i in torch.randperm(len(cand), generator=gen).tolist()], th["tau"], th["chunk"])
        if not K:
            raise RuntimeError(f"{label}: empty K although F_empty should not keep the behaviour")
        rec: dict[str, Any] = {"evals": n, "seconds": time.time() - t0, "size": len(K), "closure": closure(set(K)),
                               "n_heads": sum(u[0] == "head" for u in K), "K": sorted(index[u] for u in K)}  # fmt: skip
        rec |= {f"rho_{x}": sum(rvalue(p, r[x], u) for u in K) / len(K) for x in READS}
        orders.append(rec)
        log(f"  {label} order {k}: |K| {len(K)} ({rec['n_heads']} heads), closure {rec['closure']:.6f}, " + ", ".join(f"rho_{x} {rec[f'rho_{x}']:.8f}" for x in READS)
            + f"; {n} evaluations, {rec['seconds']:.1f} s" + (f" (about {th['orders'] * rec['seconds'] / 60:.0f} min for {th['orders']} orders)" if k == 0 else ""))
    return out | {"reading": "read", "orders": orders} | {f"rho_{x}": [o[f"rho_{x}"] for o in orders] for x in READS}
