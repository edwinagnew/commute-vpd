"""E97: the locator's reads on the 67M (experiments/locator_table; plan reports/plans/2026-11-05_plan_e97-locator.md). GPU
only; the filed rules are in `locator_rules`, pure.

A part is one head's output (its slice of `o_proj`'s input, as E96) or one block's MLP output (`down_proj`'s output); 24
heads and 4 MLPs. Every read is the target model at W0 + dW, fp32, and gives two units at each read position of a site:
"lo", the log-odds of the target against every other token (E96's, filed), and "gap", log p(target) - log p(the read's
other form) (E53's; nan where the site has none). A site is a set of inputs (windows, or whole rows) and the positions read
on them. G is the state's mean minus W0's over the same reads. Per part:

  carry (A)      the state with the part's output taken from W0's run on the same inputs; share = fall / G
  sufficiency    W0 with the part's output taken from the state's run; share = rise / G (Prakash et al.'s cross-model patch)
  removal        the state with the part's output at W0's mean over the site's own inputs and positions (E96's ablation, its
                 means from the task's inputs, F3 of the E96 check); share = the state's own fall / G, W0's own fall beside.
                 Beside: the same with E96's natural-text head means (heads only)
  write (B)      the state with one matrix's dW reverted (24), one head's rows of q, k, v and columns of o (24), one block's
                 MLP matrices (4), and all 24 (must give G = 0); share = fall / G
  split (E100)   carry, the part's own update reverted, and W0 with only the part's update (`split`; rules in `fed_rules`)
Probabilities are mean probabilities over the reads (F5), never the probability at the mean log-odds.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from commute import locator_rules as LR
from commute import steering as ST
from commute.identity import log_odds

Part = tuple[str, int, int]  # ("head", block, head) or ("mlp", block, -1)
Acts = dict[tuple[str, int], Tensor]  # ("attn", b): o_proj's input; ("mlp", b): down_proj's output; (R, T, 768) or (768,)
BLOCKS, HEADS, HEAD_DIM = 4, 6, 128
PARTS: list[Part] = [("head", b, h) for b in range(BLOCKS) for h in range(HEADS)] + [("mlp", b, -1) for b in range(BLOCKS)]
TOKENS_PER_BATCH = 16384


def name(p: Part) -> str:
    return f"{p[1]},{p[2]}" if p[0] == "head" else f"mlp{p[1]}"


def part_of(n: str) -> Part:
    if n.startswith("mlp"):
        return ("mlp", int(n[3:]), -1)
    b, h = n.split(",")
    return ("head", int(b), int(h))


def parts_of(n: str) -> list[Part]:
    """A part's name, or a pair's ("1,1+2,4"), as parts."""
    return [part_of(x) for x in n.split("+")]


@dataclass(frozen=True)
class Site:
    toks: Tensor  # (R, T) the inputs
    win: Tensor  # (P,) the input each read is on
    pos: Tensor  # (P,) the position read
    tgt: Tensor  # (P,) the target there
    row: Tensor  # (P,) the source row, for the standard error by row (F4)
    alt: Tensor | None = None  # (P,) the other form, for the "gap" unit

    @property
    def batch(self) -> int:
        return max(1, TOKENS_PER_BATCH // self.toks.shape[1])


def windows(toks: Tensor, tgt: Tensor, half: int, row: Tensor, alt: Tensor | None = None) -> Site:
    """One read per window, at `half`."""
    n = len(toks)
    return Site(toks, torch.arange(n), torch.full((n,), half), tgt, row, alt)


@contextmanager
def swapped(net: ST.Net, parts: Sequence[Part], src: Acts, cur: list[slice] | None = None) -> Iterator[None]:
    """Inside, every forward has each part's output taken from `src`: a (768,) vector at every input and position, or an
    (R, T, 768) cache indexed by cur[0], the batch of inputs being run."""
    at = cur or [slice(None)]
    pick = lambda v: v if v.dim() == 1 else v[at[0]]
    heads: dict[int, list[int]] = {}
    for kind, b, h in parts:
        if kind == "head":
            heads.setdefault(b, []).append(h)
    mlps = sorted({b for kind, b, _ in parts if kind == "mlp"})

    def on_head(b: int, hs: list[int]) -> Callable[[Any, tuple[Tensor, ...]], tuple[Tensor]]:
        def f(_m: Any, args: tuple[Tensor, ...]) -> tuple[Tensor]:
            x, v = args[0].clone(), pick(src["attn", b])
            for h in hs:
                x[..., h * HEAD_DIM : (h + 1) * HEAD_DIM] = v[..., h * HEAD_DIM : (h + 1) * HEAD_DIM]
            return (x,)

        return f

    def on_mlp(b: int) -> Callable[[Any, Any, Tensor], Tensor]:
        return lambda _m, _a, out: pick(src["mlp", b]).to(out.dtype).expand_as(out).clone()

    blocks = net.net.h
    handles = [blocks[b].attn.o_proj.register_forward_pre_hook(on_head(b, hs)) for b, hs in heads.items()]
    handles += [blocks[b].mlp.down_proj.register_forward_hook(on_mlp(b)) for b in mlps]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def read(net: ST.Net, s: Site, dW: ST.DW | None = None, parts: Sequence[Part] = (), src: Acts | None = None) -> Tensor:
    """(P, 2) float64 on CPU: the two units at every read of `s`, in W0 + dW, the parts' outputs from `src`."""
    cur, out = [slice(0, 0)], torch.full((len(s.win), 2), math.nan, dtype=torch.float64)
    with net.at(dW), swapped(net, parts, src or {}, cur):
        for i in range(0, len(s.toks), s.batch):
            cur[0] = slice(i, i + s.batch)
            sel = torch.nonzero((s.win >= i) & (s.win < i + s.batch), as_tuple=True)[0]
            logits = net.run(s.toks[cur[0]])[0]
            at = logits[(s.win[sel] - i).to(logits.device), s.pos[sel].to(logits.device)]
            t = s.tgt[sel].to(logits.device)
            out[sel, 0] = log_odds(at, t).double().cpu()
            if s.alt is not None:
                out[sel, 1] = (at.gather(1, t[:, None]) - at.gather(1, s.alt[sel].to(logits.device)[:, None])).squeeze(1).double().cpu()
    return out


def acts(net: ST.Net, s: Site, dW: ST.DW | None = None) -> Acts:
    """Every part's output on every input and position, on the model's device."""
    got: dict[tuple[str, int], list[Tensor]] = {(k, b): [] for k in ("attn", "mlp") for b in range(BLOCKS)}
    hs = [net.net.h[b].attn.o_proj.register_forward_pre_hook(lambda _m, a, b=b: got["attn", b].append(a[0].detach().clone())) for b in range(BLOCKS)]
    hs += [net.net.h[b].mlp.down_proj.register_forward_hook(lambda _m, _a, o, b=b: got["mlp", b].append(o.detach().clone())) for b in range(BLOCKS)]
    try:
        with net.at(dW):
            for i in range(0, len(s.toks), s.batch):
                net.run(s.toks[i : i + s.batch])
    finally:
        for h in hs:
            h.remove()
    return {k: torch.cat(v) for k, v in got.items()}


@dataclass
class Base:
    """W0 on one site: its reads, its part outputs, its means on the site (and E96's natural-text head means, beside), and
    W0's own reads with parts at the means, cached by (means, part)."""

    site: Site
    lo0: Tensor
    acts0: Acts
    means: Acts
    nat: Acts
    off0: dict[tuple[str, str], Tensor] = field(default_factory=dict)


def base(net: ST.Net, s: Site, nat: Acts) -> Base:
    a = acts(net, s)
    return Base(s, read(net, s), a, {k: v.double().mean((0, 1)).float() for k, v in a.items()}, nat)


def natural_means(head_means: Tensor, device: str) -> Acts:
    """E96's means.pt, (blocks, H, head_dim), as o_proj inputs; no MLP means exist there."""
    return {("attn", b): head_means[b].flatten().to(device) for b in range(head_means.shape[0])}


def w0_off(net: ST.Net, b: Base, n: str, which: str = "task") -> Tensor:
    if (which, n) not in b.off0:
        b.off0[which, n] = read(net, b.site, None, parts_of(n), b.means if which == "task" else b.nat)
    return b.off0[which, n]


def units(t: Tensor) -> dict[str, float]:
    return {"lo": float(t[0]), "gap": float(t[1])}


def shares(drop: Tensor, G: Tensor) -> dict[str, float]:
    """A mean drop (2,) as a share of G per unit; nan where G is not positive (`locator_rules.frac`)."""
    return {u: LR.frac(float(drop[i]), float(G[i])) for i, u in enumerate(("lo", "gap"))}


@dataclass
class State:
    """A state on its site: dW (on the model's device), its reads and its gain."""

    b: Base
    dW: ST.DW
    lo1: Tensor
    G: Tensor


def state(net: ST.Net, b: Base, dW: ST.DW) -> State:
    dW = {k: v.to(net.device, torch.float32) for k, v in dW.items()}
    lo1 = read(net, b.site, dW)
    return State(b, dW, lo1, (lo1 - b.lo0).mean(0))


def removal(net: ST.Net, st: State, n: str, which: str = "task", move: bool = True) -> dict[str, Any]:
    """The state's own fall with the part (or pair) at W0's means; W0's own fall beside, never subtracted."""
    off = read(net, st.b.site, st.dW, parts_of(n), st.b.means if which == "task" else st.b.nat)
    fall = st.lo1 - off
    G = float(st.G[0])
    rec = {"fall": units(fall.mean(0)), "share": shares(fall.mean(0), st.G), "p_mean_off": LR.mean_prob(off[:, 0]),
           "se_by_row": LR.row_se(fall[:, 0] / G, st.b.site.row) if G > 0 else math.nan}  # fmt: skip
    if move:
        rec["w0_move"] = units((st.b.lo0 - w0_off(net, st.b, n, which)).mean(0))
    return rec


def reverted(dW: ST.DW, keys: Sequence[str]) -> ST.DW:
    return {k: torch.zeros_like(v) if k in keys else v for k, v in dW.items()}


def head_reverted(dW: ST.DW, b: int, h: int) -> ST.DW:
    """dW with head h of block b's rows of q, k, v (its outputs) and columns of o (its inputs) at 0."""
    out, sl = dict(dW), slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
    for m in ("q_proj", "k_proj", "v_proj", "o_proj"):
        k = f"h.{b}.attn.{m}"
        out[k] = dW[k].clone()
        if m == "o_proj":
            out[k][:, sl] = 0
        else:
            out[k][sl] = 0
    return out


def part_reverted(dW: ST.DW, n: str) -> ST.DW:
    """dW with every part of `n` (a part's name or a pair's) at 0: a head as `head_reverted`, an MLP its c_fc and down_proj."""
    for kind, b, h in parts_of(n):
        dW = head_reverted(dW, b, h) if kind == "head" else reverted(dW, [f"h.{b}.mlp.c_fc", f"h.{b}.mlp.down_proj"])
    return dW


def grafted(dW: ST.DW, n: str) -> ST.DW:
    """Only `n`'s update: dW less `part_reverted` (Panigrahi et al. 2023's graft, one part)."""
    r = part_reverted(dW, n)
    return {k: v - r[k] for k, v in dW.items()}


def own(net: ST.Net, st: State, n: str) -> dict[str, float]:
    """The state's fall with the part's (or pair's) own update reverted, as a share of G, both units."""
    return shares(st.lo1.mean(0) - read(net, st.b.site, part_reverted(st.dW, n)).mean(0), st.G)


def split(net: ST.Net, st: State, n: str) -> dict[str, dict[str, float]]:
    """E100's three reads of one part (or pair) as shares of G, both units: carry (the state with the output from W0's run),
    own (the state with the part's update reverted: the fall) and graft (W0 with only the part's update: the rise)."""
    s, top = st.b.site, st.lo1.mean(0)
    return {"carry": shares(top - read(net, s, st.dW, parts_of(n), st.b.acts0).mean(0), st.G),
            "own": own(net, st, n),
            "graft": shares(read(net, s, grafted(st.dW, n)).mean(0) - st.b.lo0.mean(0), st.G)}  # fmt: skip


def writes(net: ST.Net, st: State) -> dict[str, Any]:
    fall = lambda d: shares(st.lo1.mean(0) - read(net, st.b.site, d).mean(0), st.G)
    mat = {k: fall(reverted(st.dW, [k])) for k in sorted(st.dW)}
    return {"matrix": mat, "head": {name(p): fall(head_reverted(st.dW, p[1], p[2])) for p in PARTS if p[0] == "head"},
            "mlp": {f"mlp{b}": fall(reverted(st.dW, [f"h.{b}.mlp.c_fc", f"h.{b}.mlp.down_proj"])) for b in range(BLOCKS)},
            "G_all": units((read(net, st.b.site, reverted(st.dW, list(st.dW))) - st.b.lo0).mean(0)),
            "sum_matrix": {u: sum(v[u] for v in mat.values()) for u in ("lo", "gap")}}  # fmt: skip


def greedy_pair(net: ST.Net, st: State, single: Mapping[str, float]) -> tuple[str | None, dict[str, float]]:
    """E96's greedy pair on the removal: the top head, then the head that with it loses the largest share."""
    first = LR.top_part({n: v for n, v in single.items() if not n.startswith("mlp")})
    if first is None:
        return None, {}
    heads = [name(p) for p in PARTS if p[0] == "head" and name(p) != first]
    tried = {h: removal(net, st, f"{first}+{h}", move=False)["share"]["lo"] for h in heads}
    second = LR.top_part(tried)
    return (LR.pair_name(first, second) if second else None), tried


def reads(net: ST.Net, st: State) -> dict[str, Any]:
    """Every per-part vector, both units, the removal under both means, and the greedy pair."""
    s, a1, top = st.b.site, acts(net, st.b.site, st.dW), st.lo1.mean(0)
    rec: dict[str, Any] = {"G": units(st.G), "p_mean": LR.mean_prob(st.lo1[:, 0]), "p_mean_w0": LR.mean_prob(st.b.lo0[:, 0]), "n_reads": len(s.win),
                           "carry": {name(p): shares(top - read(net, s, st.dW, [p], st.b.acts0).mean(0), st.G) for p in PARTS},
                           "sufficiency": {name(p): shares(read(net, s, None, [p], a1).mean(0) - st.b.lo0.mean(0), st.G) for p in PARTS},
                           "removal": {name(p): removal(net, st, name(p)) for p in PARTS},
                           "removal_natural_means": {name(p): removal(net, st, name(p), "natural") for p in PARTS if p[0] == "head"},
                           "write": writes(net, st)}  # fmt: skip
    del a1
    pair, rec["pair_tried"] = greedy_pair(net, st, {n: r["share"]["lo"] for n, r in rec["removal"].items()})
    if pair:
        rec["pair"], rec["removal"][pair] = pair, removal(net, st, pair)
    return rec


def plant(net: ST.Net, s: Site, train: Tensor, half: int, table: Tensor, block: int, Gt: float, tol: float) -> tuple[ST.DW, dict[str, Any]]:
    """A known-answer state: `ctxpatch.write_state`'s write moved to any block, i.e. `perturb.key_write`'s minimum-norm X
    with X a_k = the unembedding row of key k's answer, a_k the mean of block `block`'s MLP activation (down_proj's input,
    fp32) at `half` over the train windows holding key k there; written into that block's down_proj alone, scaled by
    `perturb.match` so its G on `s` (unit "lo") is Gt within `tol`. Status "cap" keeps the largest alpha, reported."""
    from commute import ctxpatch as CX
    from commute import perturb as PT

    run = lambda i: net.run(train[i : i + s.batch], capture=True)[1][f"mlp{block}"][:, half]
    x = torch.cat([run(i) for i in range(0, len(train), s.batch)]).cpu()
    at = train[:, half]
    keys = at.unique().tolist()
    A = torch.stack([x[at == k].mean(0) for k in keys], 1)
    lm = net.net.lm_head.weight.detach().float().cpu()
    X = PT.key_write(A, lm[table[torch.tensor(keys)]].T)
    lo0 = float(read(net, s)[:, 0].mean())
    build = lambda a: CX.with_matrices(net, {f"h.{block}.mlp.down_proj": a * X})
    found = PT.match(lambda a: float(read(net, s, build(a))[:, 0].mean()) - lo0, Gt, PT.ALPHA_MAX, tol)
    return build(found["x"]), {"alpha": found["x"], "G": found["f"], "G_target": Gt, "status": found["status"], "n_keys": len(keys), "X_norm": float(X.norm())}


def top1_off(net: ST.Net, means: Acts, n: str | None, C: Mapping[str, Any]) -> float:
    """W0's second-copy top-1 on repeated random sequences (`copying.induction`), the part at `means`; None: intact."""
    from commute import copying as CP

    with swapped(net, parts_of(n) if n else [], means):
        return CP.induction(net, dict(C))["top1_second"]
