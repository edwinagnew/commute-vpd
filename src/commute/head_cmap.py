"""Whole-group head CMAP after Prakash et al. (ICLR 2024) on the 67M (cell P2.68; spec
reports/plans/2026-11-23_plan_p2-68-cmap-spec.md, sections 2-4 and Appendices B-D).

A node is (block, head, rel): rel 0 is the read position, which is the last position of a prompt cut at the read, and
rel 1 the position before it (the authors' `rel_pos`). A head's output is its 128-wide slice of `o_proj`'s input.

- `discover`: the authors' `get_caches` + `apply_pp`, unchanged, one call per chunk of prompts, so each head's mean score
  comes with a standard error over the prompts; `pool` keeps heads below 0 by `k` SE.
- `n_correct`: F(S), the exact count of correct argmax at the read with every head output outside S, at every
  position, replaced by its per-position mean (ours: their `mean_ablate` hard-codes entity-tracking positions).
- `prune`: the paper-first minimality rule of Appendix C, on any count function (pure; tests/test_head_cmap.py).
- `cmap`: the authors' `cmap_out`, unchanged: the base model run with the fine-tune's pre-W_O head vectors at the given
  nodes, through the base's W_O; returns the read logits and the number of (head, position) entries overwritten.
- `Read`, `compare`, `shares`, `gate`: the three units (logits, paired correct counts, log-odds of the label)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from commute.identity import log_odds
from commute.prakash import Adapter, Authors, o_proj, quiet

Node = tuple[int, int, int]  # (block, head, rel)
Head = tuple[int, int]


@dataclass(frozen=True)
class Prompts:
    clean: Tensor  # (N, T) int64, cut at the read: the read is position T - 1
    labels: Tensor  # (N,) the correct token, read from the uncut window
    noise: Tensor | None = None  # (N, T) the noise prompt of each row (discovery and prune rows)

    def __post_init__(self) -> None:
        assert self.clean.dim() == 2 and self.labels.shape == (len(self.clean),), "one window length, one label per row"
        assert self.noise is None or self.noise.shape == self.clean.shape

    def rows(self, idx: Tensor) -> Prompts:
        return Prompts(self.clean[idx], self.labels[idx], None if self.noise is None else self.noise[idx])

    def __len__(self) -> int:
        return len(self.clean)


def loader(p: Prompts, batch: int) -> DataLoader:
    """The batches their functions read: base (clean) and source (noise) tokens, both read at the last position."""
    assert p.noise is not None
    last = p.clean.shape[1] - 1
    rows = [{"base_tokens": p.clean[i], "base_last_token_indices": last, "source_tokens": p.noise[i],
             "source_last_token_indices": last, "labels": p.labels[i]} for i in range(len(p))]  # fmt: skip
    return DataLoader(rows, batch_size=batch)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- discovery


def _scores(au: Authors, model: Adapter, p: Prompts, receivers: list[Head], composition: str, rel: int, batch: int) -> Tensor:
    dl = loader(p, batch)
    rec = [list(h) for h in receivers]
    with quiet():
        clean, corrupt, p_org, _, hooks = au.pp.get_caches(model, dl)
        layers = au.pp.get_receiver_layers(model=model, receiver_heads=rec, composition=composition) if rec else []
        return au.path.apply_pp(model=model, clean_cache=clean, corrupt_cache=corrupt, dataloader=dl, receiver_heads=rec,
                                receiver_layers=layers, clean_logit_outputs=p_org, hook_points=hooks, rel_pos=rel).cpu()  # fmt: skip


def discover(au: Authors, model: Adapter, p: Prompts, receivers: list[Head], composition: str, rel: int, chunk: int,
             batch: int) -> tuple[Tensor, Tensor]:  # fmt: skip
    """Each head's mean path-patching score (L, H) into `receivers` (none: into the logit) and its standard error over
    the chunk means (chunk 1: over the prompts). `apply_pp` returns a chunk's mean, so the chunks must be equal."""
    assert len(p) % chunk == 0, f"{len(p)} prompts do not split into chunks of {chunk}"
    parts = torch.stack([_scores(au, model, p.rows(torch.arange(s, s + chunk)), receivers, composition, rel, batch)
                         for s in range(0, len(p), chunk)])  # fmt: skip
    return parts.mean(0), parts.std(0) / math.sqrt(len(parts))


def pool(mean: Tensor, se: Tensor, k: float) -> list[Head]:
    """Heads whose mean score is below 0 by more than k standard errors, most negative first."""
    hs = [(l, h) for l in range(mean.shape[0]) for h in range(mean.shape[1]) if mean[l, h] < -k * se[l, h]]
    return sorted(hs, key=lambda x: (float(mean[x]), x))


# ----------------------------------------------------------------------------- F(S) and the minimality rule


def eval_mode(net: nn.Module) -> None:
    """Before every pass: no module may be in training mode (the authors' `apply_pp` never calls `model.eval()`)."""
    assert not any(m.training for m in net.modules()), "the model must be in eval mode"


def _o_hooks(net: nn.Module, fn: Callable[[int, Tensor], Tensor | None]) -> list[Any]:
    """A pre-hook on every block's o_proj calling fn(block, input); a tensor it returns replaces the input."""

    def hook(l: int) -> Callable[..., Any]:
        def f(_m: Any, a: tuple[Tensor, ...]) -> tuple[Tensor] | None:
            r = fn(l, a[0])
            return None if r is None else (r,)

        return f

    blocks: Any = net.h
    return [b.attn.o_proj.register_forward_pre_hook(hook(l)) for l, b in enumerate(blocks)]


@torch.no_grad()
def o_inputs(net: nn.Module, toks: Tensor, batch: int) -> list[list[Tensor]]:
    """Per batch, each block's o_proj input (B, T, H * D) on the GPU: the head outputs."""
    out: list[list[Tensor]] = []
    eval_mode(net)
    for s in range(0, len(toks), batch):
        seen: dict[int, Tensor] = {}
        hs = _o_hooks(net, lambda l, x, seen=seen: seen.__setitem__(l, x.detach().clone()))
        try:
            net(toks[s : s + batch].cuda())
        finally:
            for h in hs:
                h.remove()
        out.append([seen[l] for l in sorted(seen)])
    return out


def position_means(net: nn.Module, noise: Tensor, batch: int) -> list[Tensor]:
    """Each block's head outputs averaged over the noise prompts, per position: (T, H * D) per block."""
    per = o_inputs(net, noise, batch)
    return [torch.cat([b[l] for b in per]).mean(0) for l in range(len(per[0]))]


@torch.no_grad()
def n_correct(net: nn.Module, p: Prompts, keep: frozenset[Node] | None, means: list[Tensor], n_heads: int, batch: int) -> int:
    """Correct argmax at the read over p's rows, with every head output not in `keep` mean-ablated (None: no ablation)."""
    eval_mode(net)
    T = p.clean.shape[1]
    masks = [torch.zeros(T, n_heads, 1, dtype=torch.bool, device="cuda") for _ in means]
    for l, h, rel in keep or ():
        masks[l][T - 1 - rel, h] = True

    def ablate(l: int, x: Tensor) -> Tensor:
        H = x.view(*x.shape[:2], n_heads, -1)
        return torch.where(masks[l], H, means[l].view(T, n_heads, -1)).view_as(x)

    hs = _o_hooks(net, ablate) if keep is not None else []
    try:
        n = 0
        for s in range(0, len(p), batch):
            logits = net(p.clean[s : s + batch].cuda())[0][:, -1]
            n += int((logits.argmax(-1).cpu() == p.labels[s : s + batch]).sum())
        return n
    finally:
        for h in hs:
            h.remove()


def prune(groups: dict[str, list[Node]], count: Callable[[frozenset[Node]], int], frac: str = "0.01",
          k_frac: str = "0.3") -> dict[str, list[Node]]:  # fmt: skip
    """Appendix C. Each group is scored against the whole unpruned circuit Cir. For v in G, every other node u of G is
    ranked by drop_u = F(Cir minus u) - F(Cir minus {v, u}), descending, ties by block then head; K is the first
    ceil(k_frac |G minus v|) of them with drop_u > 0. v is kept when num = F(Cir minus K) - F(Cir minus (K + v)) > 0 and
    either den = F(Cir minus (K + v)) is 0 or num / den >= frac. `count` gives correct counts (exact, no rounding); the
    fractions are exact too (0.3 x 10 is 3, not 3.0000000000000004)."""
    cir = frozenset(n for g in groups.values() for n in g)
    F = cache(lambda drop: count(cir - drop))
    f, kf = Fraction(frac), Fraction(k_frac)
    out = {}
    for name, G in groups.items():
        kept = []
        for v in G:
            others = sorted(u for u in G if u != v)
            drop = {u: F(frozenset({u})) - F(frozenset({v, u})) for u in others}
            ranked = sorted(others, key=lambda u: (-drop[u], u))
            K = [u for u in ranked if drop[u] > 0][: math.ceil(kf * len(others))]
            den = F(frozenset({*K, v}))
            num = F(frozenset(K)) - den
            if num > 0 and (den == 0 or Fraction(num, den) >= f):
                kept.append(v)
        out[name] = kept
    return out


def matched_random(circuit: Sequence[Node], n: int, seed: int, n_layers: int, n_heads: int) -> list[list[Node]]:
    """n seeded node sets with the circuit's number of nodes at each position, drawn from all heads there."""
    gen, out = torch.Generator().manual_seed(seed), []
    rels = sorted({r for *_, r in circuit})
    for _ in range(n):
        s = []
        for rel in rels:
            k = sum(1 for *_, r in circuit if r == rel)
            s += [(int(i) // n_heads, int(i) % n_heads, rel) for i in torch.randperm(n_layers * n_heads, generator=gen)[:k]]
        out.append(sorted(s))
    return out


# ----------------------------------------------------------------------------- CMAP and the three units


@torch.no_grad()
def cmap(au: Authors, base: Adapter, cache_ft: list[list[Tensor]], toks: Tensor, nodes: Sequence[Node], batch: int) -> tuple[Tensor, int]:
    """The base run with the fine-tune's head outputs (`o_inputs` of the fine-tune on the same `toks`, same batches) at
    `nodes`, through the base's W_O: (read logits (N, V) on the CPU, entries overwritten)."""
    pdict = {rel: [[l, h] for l, h, r in nodes if r == rel] for rel in sorted({n[2] for n in nodes})}
    per_layer = [sum(1 for hs in pdict.values() for a, _ in hs if a == l) for l in range(len(cache_ft[0]))]
    layers = [o_proj(l) for l in range(len(per_layer))]
    sites, out = [0], []
    for bi, s in enumerate(range(0, len(toks), batch)):
        x = toks[s : s + batch].cuda()
        ft = {bi: {layers[l]: c for l, c in enumerate(cache_ft[bi])}}
        assert ft[bi][layers[0]].shape[:2] == x.shape, "the cache's batches are not these prompts"
        with au.pp.TraceDict(base, layers, edit_output=_counted(au, base, ft, bi, pdict, x, per_layer, sites)):
            out.append(base(x).logits[:, -1].float().cpu())
    return torch.cat(out), sites[0]


def _counted(au: Authors, base: Adapter, ft: dict[int, dict[str, Tensor]], bi: int, pdict: dict[int, list[list[int]]],
             x: Tensor, per_layer: list[int], sites: list[int]) -> Callable[..., Tensor]:  # fmt: skip
    def edit(inputs: Any, output: Tensor, layer: str) -> Tensor:
        sites[0] += len(x) * per_layer[int(layer.split(".")[2])]
        last = torch.full((len(x),), x.shape[1] - 1, device=x.device)
        return au.cmap.cmap_out(inputs=inputs, outputs=output, layer=layer, model=base, finetuned_cache=ft, bi=bi,
                                pos_heads_dict=pdict, input_tokens={"input_ids": x, "last_token_indices": last})  # fmt: skip

    return edit


@dataclass(frozen=True)
class Read:
    logits: Tensor  # (N, V) at the read, float32, CPU
    labels: Tensor  # (N,)

    @property
    def correct(self) -> Tensor:
        return self.logits.argmax(-1) == self.labels

    @property
    def logodds(self) -> Tensor:
        return log_odds(self.logits, self.labels).double()

    def summary(self) -> dict[str, Any]:
        c = self.correct
        return {"n_correct": int(c.sum()), "n": len(c), "acc": float(c.double().mean()), "mean_logodds": float(self.logodds.mean())}


@torch.no_grad()
def forward_read(net: nn.Module, p: Prompts, batch: int) -> Read:
    eval_mode(net)
    return Read(torch.cat([net(p.clean[s : s + batch].cuda())[0][:, -1].float().cpu() for s in range(0, len(p), batch)]), p.labels)


def compare(a: Read, b: Read) -> dict[str, Any]:
    """The three units' distance between two reads of the same rows."""
    return {"max_dlogit": float((a.logits - b.logits).abs().max()), "n_flipped": int((a.correct != b.correct).sum()),
            "max_dlogodds": float((a.logodds - b.logodds).abs().max())}  # fmt: skip


def within(c: dict[str, Any], tol: dict[str, float]) -> bool:
    return c["max_dlogit"] <= tol["logit"] and c["n_flipped"] == 0 and c["max_dlogodds"] <= tol["logodds"]


def shares(p: Read, w0: Read, ft: Read) -> dict[str, float | None]:
    """(patched - W0) / (fine-tune - W0) of the correct count and of the mean log-odds (neuron_r.shares' form: mean
    numerator over mean gain), not clipped; None for a zero denominator."""
    a, b, c = p.summary(), w0.summary(), ft.summary()
    out: dict[str, float | None] = {}
    for unit, key in (("acc", "n_correct"), ("logodds", "mean_logodds")):
        g = c[key] - b[key]
        out[unit] = (a[key] - b[key]) / g if g != 0 else None
    return out


def gains(ft: Read, w0: Read) -> dict[str, float]:
    a, b = ft.summary(), w0.summary()
    return {"count": a["n_correct"] - b["n_correct"], "logodds": a["mean_logodds"] - b["mean_logodds"]}


def gate(w0: Read, randoms: Sequence[Read]) -> dict[str, float]:
    """N, the largest |correct-count change|, and L, the largest |mean log-odds change|, over random updates."""
    g = [gains(r, w0) for r in randoms]
    return {"N": max(abs(x["count"]) for x in g), "L": max(abs(x["logodds"]) for x in g), "n_random": len(g)}


def clears(g: dict[str, float], floor: dict[str, float]) -> bool:
    """A gain clears the gate when its count gain exceeds N (so at least 1 when N is 0) and its log-odds gain exceeds L."""
    return g["count"] > floor["N"] and g["logodds"] > floor["L"]


@torch.no_grad()
def attention_to(net: nn.Module, block: int, toks: Tensor, batch: int) -> Tensor:
    """Each head's mean attention weight from the last position to the one before it, at `block`: (H,)."""
    blocks: Any = net.h
    attn, got, tot = blocks[block].attn, {}, []
    hook = attn.register_forward_pre_hook(lambda _m, a: got.__setitem__("x", a[0]))
    try:
        for s in range(0, len(toks), batch):
            net(toks[s : s + batch].cuda())
            x = got["x"]
            B, T, _ = x.shape
            q = attn.q_proj(x).view(B, T, attn.n_head, -1).transpose(1, 2)
            k = attn.k_proj(x).view(B, T, attn.n_head, -1).transpose(1, 2)
            pos = torch.arange(T, device=x.device)[None]
            q, k = attn.apply_rotary_pos_emb(q, k, attn.rotary_cos[pos].to(q.dtype), attn.rotary_sin[pos].to(q.dtype))
            att = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            att = att.masked_fill(attn.bias[:, :, :T, :T] == 0, float("-inf")).softmax(-1)
            tot.append(att[:, :, -1, -2].cpu())
    finally:
        hook.remove()
    return torch.cat(tot).mean(0)
