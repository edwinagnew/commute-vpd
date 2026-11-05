"""E97 Amendment 1: edge-level reads on the 67M under `net.at(dW)`, our own (the scoping review's option 2, no
TransformerLens). Plan reports/plans/2026-11-05_plan_e97-amend1.md; the graph and the rules are `commute.edge_rules`.

`forward` re-runs the target model from its own modules (RMS norms, rotary buffers, MLPs, final norm, unembedding),
with each head's q, k and v computed from its own input and each head's output its slice of o_proj written separately.
Every downstream input v reads sum_u [z'_u + C[u, v] (z_u - z'_u)]: z the nodes' outputs in this run, z' a fixed
corrupted run's, C per row of the batch (1 keeps the edge, 0 patches it). C = 1 is the model; C = 0 is the corrupted run.

A `Pair` says what is corrupted. The gain path (`bad` None): z' is W0's run on the same inputs, read in the state
(Prakash et al.'s cross-model patch at edge grain; not in Hanna et al.). A circuit pair: z' is the same model on `bad`,
the corrupted inputs of the same length (Hanna et al. 2024). Every number is a mean over the site's reads, in nats.

  exact      every C asked for, each one forward (many C share a forward, stacked along the batch)
  ig_gain    integrated gradients of the read in C along C = alpha everywhere: sums to clean - corrupted
  ig_pair    EAP-IG-inputs: (z_u - z'_u) . mean gradient at input v, embeddings interpolated from bad to clean
Both IG forms take the midpoints alpha = (k + 0.5) / steps, as E96's checked path (`locate.head_path`), where
eap-ig takes k / steps: a departure, stated.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from commute import edge_rules as ER
from commute import locator as L
from commute.identity import log_odds

HD = 128
EXACT_TOKENS = 16384  # tokens per stacked forward of `exact` (L.TOKENS_PER_BATCH's budget)


@dataclass(frozen=True)
class Pair:
    site: L.Site
    bad: Tensor | None  # (R, T) corrupted inputs; None: the gain path, W0's outputs on the same inputs
    unit: str  # "lo" (log-odds of the target) or "gap" (logit of the target minus the read's other form)


def _mix(z: list[Tensor], Zc: Tensor | None, C: Tensor, cols: slice) -> Tensor:
    """(n_cols, B, T, D): each input of `cols` from the nodes written so far (Zc None: zeros)."""
    n = len(z)
    if Zc is None:
        return torch.einsum("buv,ubtd->vbtd", C[:, :n, cols], torch.stack(z))
    return Zc[:n].sum(0) + torch.einsum("buv,ubtd->vbtd", C[:, :n, cols], torch.stack(z) - Zc[:n])


def _heads(attn: Any, x: Tensor) -> Tensor:
    """(6, B, T, D): each head's output, its q, k and v from x's 18 normed inputs (q heads, k heads, v heads)."""
    H, T, D = ER.HEADS, x.shape[2], x.shape[3]
    W = lambda lin: lin.weight.view(H, HD, D)
    q, k, v = (torch.einsum("hbtd,hed->bhte", x[H * i : H * i + H], W(lin)) for i, lin in enumerate((attn.q_proj, attn.k_proj, attn.v_proj)))
    pos = torch.arange(T, device=x.device)[None]
    q, k = attn.apply_rotary_pos_emb(q, k, attn.rotary_cos[pos].to(q.dtype), attn.rotary_sin[pos].to(q.dtype))
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(HD))
    att = att.masked_fill(attn.bias[:, :, :T, :T] == 0, float("-inf")).softmax(-1)
    return torch.einsum("bhte,dhe->hbtd", att @ v, attn.o_proj.weight.view(D, H, HD))


def forward(net: Any, toks: Tensor, C: Tensor, Zc: Tensor | None = None, emb: Tensor | None = None, start: int = 0,
            pre: Sequence[Tensor] = (), keep: list[tuple[int, Tensor]] | None = None) -> tuple[Tensor, list[Tensor]]:  # fmt: skip
    """The final residual (B, T, D) and the 29 node outputs. C is (B, 29, 77) or (29, 77); Zc (29, B, T, D), None for
    zeros. `start` > 0 takes blocks before it from `pre` (the same rows' nodes). `keep` collects every input, grad kept."""
    m = net.net
    z = list(pre[: 1 + 7 * start]) if start else [m.wte(toks) if emb is None else emb]
    C = C.expand(len(z[0]), *ER.VALID.shape) if C.dim() == 2 else C

    def inputs(cols: slice) -> Tensor:
        x = _mix(z, Zc, C, cols)
        if keep is not None:
            x.retain_grad()
            keep.append((cols.start, x))
        return x

    for b in range(start, ER.BLOCKS):
        blk = m.h[b]
        z += list(_heads(blk.attn, blk.rms_1(inputs(slice(19 * b, 19 * b + 18)))).unbind(0))
        z.append(blk.mlp(blk.rms_2(inputs(slice(19 * b + 18, 19 * b + 19))[0])))
    return inputs(slice(ER.LOGITS, ER.LOGITS + 1))[0], z


def reads_at(net: Any, fin: Tensor, s: L.Site, sel: Tensor, i0: int, unit: str) -> Tensor:
    """(P,) the read at each of `sel`'s positions, `fin` holding rows i0 onward."""
    h = fin[(s.win[sel] - i0).to(fin.device), s.pos[sel].to(fin.device)]
    lg = net.net.lm_head(net.net.ln_f(h))
    t = s.tgt[sel].to(lg.device)
    if unit == "lo":
        return log_odds(lg, t)
    assert s.alt is not None, "the gap unit needs the read's other form"
    return (lg.gather(1, t[:, None]) - lg.gather(1, s.alt[sel].to(lg.device)[:, None])).squeeze(1)


def batches(s: L.Site, size: int) -> Iterator[tuple[int, int, Tensor, int]]:
    """(first row, end row, the reads on them, length to keep): rows cut after the last position read (causal)."""
    for i in range(0, len(s.toks), size):
        j = min(i + size, len(s.toks))
        sel = torch.nonzero((s.win >= i) & (s.win < j), as_tuple=True)[0]
        if len(sel):
            yield i, j, sel, int(s.pos[sel].max()) + 1


def ones(net: Any) -> Tensor:
    w = net.net.lm_head.weight
    return torch.ones(ER.VALID.shape, device=w.device, dtype=w.dtype)


def corrupted(net: Any, p: Pair, dW: Any, i: int, j: int, T: int) -> Tensor:
    """(29, B, T, D): the corrupted run's node outputs on rows i:j (W0 on the same rows, or the model on `bad`)."""
    toks = (p.site.toks if p.bad is None else p.bad)[i:j, :T].to(net.device)
    with torch.no_grad(), net.at(None if p.bad is None else dW):
        return torch.stack(forward(net, toks, ones(net))[1])


def start_of(C: Tensor) -> int:
    """The first block whose inputs `C` patches (4: only the logits, or nothing)."""
    cols = ((C != 1) & ER.VALID.to(C.device)).any(0).nonzero()
    return ER.block_of(int(cols.min())) if len(cols) else ER.BLOCKS


def per_read(net: Any, p: Pair, dW: Any, C: Tensor) -> Tensor:
    """(P,) float64 on CPU: the read at every position with C applied; for the stage-0 checks."""
    out = torch.full((len(p.site.win),), math.nan, dtype=torch.float64)
    for i, j, sel, T in batches(p.site, p.site.batch):
        Zc = corrupted(net, p, dW, i, j, T)
        with torch.no_grad(), net.at(dW):
            fin = forward(net, p.site.toks[i:j, :T].to(net.device), C.to(Zc), Zc)[0]
            out[sel] = reads_at(net, fin, p.site, sel, i, p.unit).double().cpu()
    return out


def exact(net: Any, p: Pair, dW: Any, Cs: Sequence[Tensor]) -> dict[str, Any]:
    """Means over the reads: `clean` (C = 1), `corrupt` (C = 0) and `each` C of `Cs` (29, 77), several per forward."""
    n, one = len(p.site.win), ones(net)
    tot = torch.zeros(len(Cs) + 2, dtype=torch.float64)
    for i, j, sel, T in batches(p.site, p.site.batch):
        B, Zc = j - i, corrupted(net, p, dW, i, j, T)
        toks = p.site.toks[i:j, :T].to(net.device)
        with torch.no_grad(), net.at(dW):
            fin, z = forward(net, toks, one, Zc)
            tot[0] += float(reads_at(net, fin, p.site, sel, i, p.unit).double().sum())
            tot[1] += float(reads_at(net, forward(net, toks, 0 * one, Zc)[0], p.site, sel, i, p.unit).double().sum())
            g = max(1, EXACT_TOKENS // (B * T))
            for c0 in range(0, len(Cs), g):
                chunk = [c.to(one) for c in Cs[c0 : c0 + g]]
                k, st = len(chunk), min(start_of(c) for c in chunk)
                rep = lambda t, d=0, k=k: t.repeat(*[k if a == d else 1 for a in range(t.dim())])
                fin = forward(net, rep(toks), torch.stack(chunk).repeat_interleave(B, 0), rep(Zc, 1), start=st, pre=[rep(x) for x in z[: 1 + 7 * st]])[0]
                for c, f in enumerate(fin.reshape(k, B, *fin.shape[1:])):
                    tot[2 + c0 + c] += float(reads_at(net, f, p.site, sel, i, p.unit).double().sum())
    tot /= n
    return {"clean": float(tot[0]), "corrupt": float(tot[1]), "each": tot[2:].tolist()}


def single(u: int, v: int) -> Tensor:
    C = torch.ones(ER.VALID.shape)
    C[u, v] = 0
    return C


def ig_gain(net: Any, p: Pair, dW: Any, steps: int) -> Tensor:
    """(29, 77) on CPU: the integral of d(read)/dC along C = alpha on every edge, alpha's midpoints over `steps`."""
    acc = torch.zeros(ER.VALID.shape, dtype=torch.float64)
    for i, j, sel, T in batches(p.site, max(1, p.site.batch // 2)):
        Zc = corrupted(net, p, dW, i, j, T)
        toks = p.site.toks[i:j, :T].to(net.device)
        with net.at(dW), torch.enable_grad():
            for k in range(steps):
                C = (ones(net) * (k + 0.5) / steps).requires_grad_(True)
                reads_at(net, forward(net, toks, C, Zc)[0], p.site, sel, i, p.unit).sum().backward()
                acc += C.grad.double().cpu()  # type: ignore[union-attr]
    return (acc / (steps * len(p.site.win))).masked_fill(~ER.VALID, 0.0)


def ig_pair(net: Any, p: Pair, dW: Any, steps: int) -> Tensor:
    """(29, 77) on CPU: EAP-IG-inputs, (z_u - z'_u) . the mean over steps of d(read)/d(input v), the embeddings
    interpolated from the corrupted inputs' to the clean inputs' (Hanna et al. 2024)."""
    assert p.bad is not None, "EAP-IG needs a corrupted input"
    acc, one = torch.zeros(ER.VALID.shape, dtype=torch.float64), ones(net)
    for i, j, sel, T in batches(p.site, max(1, p.site.batch // 2)):
        Zc = corrupted(net, p, dW, i, j, T)
        toks = p.site.toks[i:j, :T].to(net.device)
        with net.at(dW):
            with torch.no_grad():
                diff = torch.stack(forward(net, toks, one)[1]) - Zc
            for k in range(steps):
                emb = (Zc[0] + (k + 0.5) / steps * diff[0]).requires_grad_(True)
                keep: list[tuple[int, Tensor]] = []
                with torch.enable_grad():
                    reads_at(net, forward(net, toks, one, emb=emb, keep=keep)[0], p.site, sel, i, p.unit).sum().backward()
                for c0, x in keep:
                    n = ER.avail(c0)
                    acc[:n, c0 : c0 + len(x)] += torch.einsum("ubtd,vbtd->uv", diff[:n], x.grad).double().cpu()  # type: ignore[arg-type]
    return (acc / (steps * len(p.site.win))).masked_fill(~ER.VALID, 0.0)


# ----------------------------------------------------------------------------- the two uses


def joint(u: int) -> Tensor:
    """Every outgoing edge of node u patched: E97's carry read (A) for that part."""
    C = torch.ones(ER.VALID.shape)
    C[u] = 0
    return C


def compact(scores: dict[str, float], G: float, top: int = 30) -> dict[str, Any]:
    """What the stage files keep of 927 scores: the `top` by |score|, every edge to the logits, each node's sum."""
    big = sorted(scores, key=lambda n: -abs(scores[n]))[:top]
    return {"G": G, "top": [[n, scores[n]] for n in big], "to_logits": {n.split("->")[0]: x for n, x in scores.items() if n.endswith("->logits")},
            "out_sum": ER.out_sums(scores), "top_edge": ER.top_edge(scores)}  # fmt: skip


def gain_exact(net: Any, p: Pair, dW: Any) -> tuple[dict[str, Any], dict[str, float]]:
    """Every edge of the gain path patched alone (score = clean - patched, nats), and each part's edges jointly."""
    r = exact(net, p, dW, [single(u, v) for u, v in ER.EDGES] + [joint(u) for u in range(1, len(ER.UP))])
    G, n = r["clean"] - r["corrupt"], len(ER.EDGES)
    scores = {e: r["clean"] - x for e, x in zip(ER.NAMES, r["each"][:n], strict=True)}
    rec = {**compact(scores, G), "clean": r["clean"], "corrupt": r["corrupt"], "how": "exact, every edge",
           "joint_share": {ER.UP[u]: ER.frac(r["clean"] - x, G) for u, x in zip(range(1, len(ER.UP)), r["each"][n:], strict=True)}}  # fmt: skip
    return rec, scores


def gain_ig(net: Any, p: Pair, dW: Any, steps: int, n_exact: int) -> tuple[dict[str, Any], dict[str, float]]:
    """Integrated gradients on every edge, then the `n_exact` of largest |score| patched exactly; scores are the exact
    ones there and IG's elsewhere."""
    ig = ER.as_dict(ig_gain(net, p, dW, steps))
    pick = sorted(ig, key=lambda e: -abs(ig[e]))[:n_exact]
    r = exact(net, p, dW, [single(*ER.EDGES[ER.NAMES.index(e)]) for e in pick])
    ex = {e: r["clean"] - x for e, x in zip(pick, r["each"], strict=True)}
    G = r["clean"] - r["corrupt"]
    rec = {**compact({**ig, **ex}, G), "clean": r["clean"], "corrupt": r["corrupt"], "how": f"IG ({steps} steps), the top {n_exact} exact",
           "ig_sum": sum(ig.values()), "exact_vs_ig": {e: [ex[e], ig[e]] for e in pick}}  # fmt: skip
    return rec, {**ig, **ex}


def circuit(net: Any, p: Pair, dW: Any, grid: Sequence[int], bar: float, steps: int, min_gap: float) -> tuple[dict[str, Any], Tensor]:
    """The model's own circuit for the pair: EAP-IG scores, top-n by |score| (pruned) at each size of `grid`, the
    smallest at normalised faithfulness >= `bar`. No gap above `min_gap`: the empty circuit, said so."""
    g = exact(net, p, dW, [])
    rec: dict[str, Any] = {"clean": g["clean"], "corrupt": g["corrupt"], "gap": g["clean"] - g["corrupt"]}
    empty = torch.zeros(ER.VALID.shape, dtype=torch.bool)
    if not rec["gap"] > min_gap:
        return {**rec, "n": None, "reason": f"no gap: {rec['gap']:.3f} nats, at most {min_gap}"}, empty
    sc = ig_pair(net, p, dW, steps)
    masks = [ER.topn(sc, n) for n in grid]
    r = exact(net, p, dW, [m.float() for m in masks])
    f = [ER.faith(x, r["clean"], r["corrupt"], min_gap) for x in r["each"]]
    n = ER.smallest(list(zip(grid, f, strict=True)), bar)
    mask = masks[list(grid).index(n)] if n is not None else empty
    d = ER.as_dict(sc)
    rec |= {"n": n, "faith": dict(zip(map(str, grid), f, strict=True)), "n_edges": int(mask.sum()), "edges": [ER.edge_name(u, v) for u, v in mask.nonzero().tolist()],
            "top": [[e, d[e]] for e in sorted(d, key=lambda e: -abs(d[e]))[:30]], "rank_to_logits": {k: sorted(d, key=lambda e: -abs(d[e])).index(f"{k}->logits") for k in ER.UP}}  # fmt: skip
    if n is None:
        rec["reason"] = f"no size of the grid reaches {bar}"
    return rec, mask


def cross(net: Any, p: Pair, dW: Any, mask: Tensor, min_gap: float) -> dict[str, Any]:
    """Cross-faithfulness: this model on this pair with every edge outside `mask` patched; the share of its own gap kept."""
    r = exact(net, p, dW, [mask.float()])
    return {"faith": ER.faith(r["each"][0], r["clean"], r["corrupt"], min_gap), "m": r["each"][0], "clean": r["clean"], "corrupt": r["corrupt"], "circuit_edges": int(mask.sum())}


def mask_of(names: Sequence[str]) -> Tensor:
    m = torch.zeros(ER.VALID.shape, dtype=torch.bool)
    for e in names:
        m[ER.EDGES[ER.NAMES.index(e)]] = True
    return m


@contextmanager
def fp64(net: Any) -> Iterator[None]:
    """The target model in float64 inside (its RMS norms still compute in float32, as the module does)."""
    net.net.double()
    try:
        yield
    finally:
        net.net.float()
