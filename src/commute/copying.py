"""E62, the copying-heads test (experiments/induction_cue; plan reports/plans/2026-11-02_plan_e62-copying-heads.md): the
copying heads of the 67M, their mean ablation, the cue windows and the filed rules. Rules are pure, so every edge case
runs on CPU (tests/test_copying.py).

The target model has 4 blocks of 6 heads with head_dim 128 (LlamaSimpleMLP, flash_attention false, no repeated key or
value heads). No head output is exposed, but `o_proj`'s input is the head outputs side by side, so a pre-hook on `o_proj`
reads and ablates a head, and `q_proj`, `k_proj`, `v_proj` hooks give the attention pattern through the module's own
rotary code. `attention_patterns` returns how far its pattern times v is from `o_proj`'s input. No model code changes.

A cue window is a 41-token Pile span with the pair A B written earlier and A M at the end, read at M:
    ... A B ... A M ...       M at `half`, A at `half - 1`, the first A at p, B at p + 1
Target tensors follow data.LAST: `with_target` appends the target as a 42nd column that no forward reads (ci_len = 41).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from commute import kv
from commute import steering as ST
from commute.data import Windows
from commute.local_faith import masked_logits

VOCAB = 50277
Head = tuple[int, int]  # (block, head)


def finite(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def induction(net: Any, C: dict[str, Any]) -> dict[str, Any]:
    """W0's top-1 on a random sequence of distinct ids shown twice; the second copy's first token is not scored.
    (E52's control, moved, not changed, from experiments/baseline_neurons/amend1.py.)"""
    L = C["length"]
    seq = torch.stack([torch.randperm(VOCAB, generator=torch.Generator().manual_seed(C["seed"] + i))[:L]
                       for i in range(C["sequences"])])
    toks = torch.cat([seq, seq], 1)
    logits = net.run(toks)[0]
    assert logits.shape[-1] == VOCAB
    hit = logits.argmax(-1).cpu()[:, :-1] == toks[:, 1:]  # logits at p predict token p + 1
    r = {"top1_first": float(hit[:, : L - 1].double().mean()), "top1_second": float(hit[:, L:].double().mean()),
         "n_scored_second": int(hit[:, L:].numel())}
    r["ok"] = r["top1_second"] >= C["threshold"]
    return r


# ----------------------------------------------------------------------------- the prefix-matching score


def repeated_random(pool: Sequence[int], length: int, repeats: int, n: int, sos: int, seed: int) -> Tensor:
    """(n, 1 + length * repeats): Olsson et al. 2022's test rows, a start token then `length` distinct tokens of `pool`
    repeated `repeats` times."""
    gen, ids = torch.Generator().manual_seed(seed), torch.tensor(list(pool))
    rows = [torch.cat([torch.tensor([sos]), ids[torch.randperm(len(ids), generator=gen)[:length]].repeat(repeats)]) for _ in range(n)]
    return torch.stack(rows)


def stripe_mask(toks: Tensor, offset: int = 1) -> Tensor:
    """(B, T, T) bool, entry (i, s) true when token i also sits at some j < i and s = j + offset: the attention entries
    the score averages. offset 1 is the token after the earlier copy, the one induction predicts (the paper's "attends to
    the token which induction would suggest comes next"); offset -1 is the literal reading of its evaluator sentence
    ("the tokens that preceded the same token"), reported beside."""
    same = (toks[:, :, None] == toks[:, None, :]).tril(-1)
    m = torch.zeros_like(same)
    if offset == 1:
        m[:, :, 1:] = same[:, :, :-1]
    else:
        assert offset == -1, offset
        m[:, :, :-1] = same[:, :, 1:]
    return m


def stripe_score(att: Tensor, mask: Tensor) -> Tensor:
    """(H,): per head, the mean over the mask's entries of att (B, H, T, T). nan where the mask is empty."""
    n = mask.sum()
    if n == 0:
        return torch.full((att.shape[1],), math.nan, dtype=torch.float64)
    return (att.double() * mask[:, None].to(att.device)).sum((0, 2, 3)) / n.to(att.device)


def uniform_attention(T: int) -> Tensor:
    """(1, 1, T, T): the causal uniform pattern, position i spreading 1 / (i + 1) over positions 0..i (the score's chance)."""
    return (torch.ones(T, T).tril() / (torch.arange(T)[:, None] + 1.0))[None, None]


def previous_token(att: Tensor) -> Tensor:
    """(H,): the paper's previous-token score, the mean attention from token i to token i - 1."""
    return att.double().diagonal(-1, 2, 3).mean((0, 2))


def attention_patterns(net: Any, toks: Tensor) -> tuple[list[Tensor], float]:
    """Per block the attention pattern (B, H, T, T) of `toks` under `net` (ST.Net), recomputed from q_proj, k_proj,
    v_proj's outputs with the module's own rotary code and the largest |pattern @ v - o_proj's input| over the blocks
    (the check that the pattern is the one the model used)."""
    seen: dict[tuple[int, str], Tensor] = {}
    handles = []
    for lay, block in enumerate(net.net.h):
        a = block.attn
        assert a.repeat_kv_heads == 1 and a.rotary_dim == a.head_dim
        for name, mod in (("q", a.q_proj), ("k", a.k_proj), ("v", a.v_proj)):
            handles.append(mod.register_forward_hook(lambda _m, _a, o, key=(lay, name): seen.__setitem__(key, o)))
        handles.append(a.o_proj.register_forward_pre_hook(lambda _m, args, key=(lay, "y"): seen.__setitem__(key, args[0])))
    try:
        net.run(toks)
    finally:
        for h in handles:
            h.remove()
    pats, err = [], 0.0
    for lay, block in enumerate(net.net.h):
        a = block.attn
        B, T, C = seen[lay, "q"].shape
        q, k, v = (seen[lay, s].view(B, T, a.n_head, a.head_dim).transpose(1, 2) for s in "qkv")
        pos = torch.arange(T, device=q.device)[None]
        q, k = a.apply_rotary_pos_emb(q, k, a.rotary_cos[pos].to(q.dtype), a.rotary_sin[pos].to(q.dtype))
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.softmax(att.masked_fill(a.bias[:, :, :T, :T] == 0, float("-inf")), -1)
        err = max(err, float(((att @ v).transpose(1, 2).reshape(B, T, C) - seen[lay, "y"]).abs().max()))
        pats.append(att)
    return pats, err


# ----------------------------------------------------------------------------- mean ablation


def head_means(net: Any, spans: Tensor, batch: int = 64) -> Tensor:
    """(blocks, H, head_dim) on the model's device: each head's output (`o_proj`'s input) averaged over every position of
    `spans`, under the state `net` is in."""
    blocks = list(net.net.h)
    a0 = blocks[0].attn
    sums = torch.zeros(len(blocks), a0.n_head, a0.head_dim, dtype=torch.float64)
    handles = []

    def add(lay: int) -> Callable[[Any, tuple[Tensor, ...]], None]:
        def f(_m: Any, args: tuple[Tensor, ...]) -> None:
            sums[lay] += args[0].double().sum((0, 1)).view(a0.n_head, a0.head_dim).cpu()
        return f

    for lay, block in enumerate(blocks):
        handles.append(block.attn.o_proj.register_forward_pre_hook(add(lay)))
    try:
        for s in range(0, spans.shape[0], batch):
            net.run(spans[s : s + batch])
    finally:
        for h in handles:
            h.remove()
    return (sums / spans.numel()).float().to(net.device)


@contextmanager
def ablated(net: Any, heads: Iterable[Head], means: Tensor) -> Iterator[None]:
    """Inside the block every forward of `net` (ST.Net) has the given heads' outputs replaced by `means` at every
    position, whatever the weights are."""
    by_block: dict[int, list[int]] = {}
    for lay, h in heads:
        by_block.setdefault(lay, []).append(h)
    d = means.shape[-1]

    def edit(lay: int, hs: list[int]) -> Callable[[Any, tuple[Tensor, ...]], tuple[Tensor]]:
        def f(_m: Any, args: tuple[Tensor, ...]) -> tuple[Tensor]:
            y = args[0].clone()
            for h in hs:
                y[..., h * d : (h + 1) * d] = means[lay, h].to(y.device, y.dtype)
            return (y,)
        return f

    handles = [net.net.h[lay].attn.o_proj.register_forward_pre_hook(edit(lay, hs)) for lay, hs in by_block.items()]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


class AllGatesOn:
    """`ST.Net`'s `net` and `run` through the decomposition's masked forward: every gate 1, Delta out, bf16 as the route
    split's cells. So `induction` and `ablated` read this forward unchanged. `ablated` needs no second form: o_proj is
    itself decomposed, but `ComponentModel` swaps a module's OUTPUT in a forward hook and feeds the components that hook's
    args, which already carry a pre-hook's edit, so the components take the ablated heads (tests/test_copying.py)."""

    def __init__(self, model: Any) -> None:
        self.model, self.net = model, model.target_model

    @torch.no_grad()
    def run(self, toks: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        device = next(self.model.parameters()).device
        ones = {k: torch.ones(*toks.shape, C, device=device) for k, C in self.model.module_to_c.items()}
        return masked_logits(self.model, toks.to(device), ones, None, None).float(), {}


def over_bar(scores: Tensor, bar: float) -> list[Head]:
    """H: the heads whose score (blocks, H) reaches the bar (a tie counts), in block and head order."""
    return [(int(a), int(b)) for a, b in (scores >= bar).nonzero().tolist()]


def by_score(scores: Tensor) -> list[Head]:
    """Every head, highest score first; ties in block and head order."""
    order = torch.argsort(scores.flatten(), descending=True, stable=True).tolist()
    return [(i // scores.shape[1], i % scores.shape[1]) for i in order]


def random_others(H: Sequence[Head], blocks: int, heads: int, n: int, seed: int) -> list[list[Head]]:
    """n seeded draws of |H| heads outside H, each sorted."""
    rest = [(a, b) for a in range(blocks) for b in range(heads) if (a, b) not in set(H)]
    gen = torch.Generator().manual_seed(seed)
    return [sorted(rest[int(i)] for i in torch.randperm(len(rest), generator=gen)[: len(H)]) for _ in range(n)]


# ----------------------------------------------------------------------------- the windows


def natural_spans(rows: Tensor, half: int, n: int, seed: int) -> Tensor:
    """(n, 2 half + 1): unedited Pile spans at random (row, position) pairs."""
    gen = torch.Generator().manual_seed(seed)
    r = torch.randint(0, rows.shape[0], (n,), generator=gen)
    pos = torch.randint(half, rows.shape[1] - half, (n,), generator=gen)
    return torch.stack([rows[int(a), int(b) - half : int(b) + half + 1] for a, b in zip(r, pos)])


def cue_windows(rows: Tensor, marker: int, keys: Sequence[int], bs: Sequence[int], half: int, n: int,
                seed: int) -> tuple[Windows, Tensor, Tensor]:  # fmt: skip
    """n cue windows, the first that hold from 2n drawn on `kv.windows`' spans (each A about 2n / K times), and each
    window's B and p. p is uniform on 2..half - 4, so at least one Pile token sits between B and the second A. A window
    is dropped unless A occurs exactly twice, B once and M once, anywhere in the 41 tokens (natural text can hold any of
    them). No window is filtered against the lookup's candidate words (Amendment 1, 2026-11-02: a frequent pool would
    drop most windows); `pick_words` keeps a key's word out of that key's own windows instead."""
    w = kv.windows(rows, marker, tuple(keys), half, 2 * n, seed)
    a = w.tokens[:, half].clone()
    t = w.tokens.clone()
    t[:, half - 1], t[:, half] = a, marker
    gen = torch.Generator().manual_seed(seed + 1)
    p = torch.randint(2, half - 3, (2 * n,), generator=gen)
    b = torch.tensor(list(bs))[torch.randint(len(bs), (2 * n,), generator=gen)]
    idx = torch.arange(2 * n)
    t[idx, p], t[idx, p + 1] = a, b
    count = lambda v: (t == v[:, None]).sum(1)
    ok = (count(a) == 2) & (count(b) == 1) & ((t == marker).sum(1) == 1)
    keep = ok.nonzero().squeeze(1)[:n]
    assert len(keep) == n, f"only {int(ok.sum())} of {2 * n} drawn windows hold; need {n}"
    return Windows(t[keep], torch.zeros(n), w.row[keep], w.pos[keep], half), b[keep], p[keep]


def with_target(w: Windows, target: Tensor) -> Windows:
    """The windows with `target` (N,) as a last column, for data.LAST with ci_len = the old length."""
    return Windows(torch.cat([w.tokens, target[:, None]], 1), w.g, w.row, w.pos, w.half)


def broken_pair(toks: Tensor, p: Tensor, keys: Sequence[int], half: int, seed: int) -> Tensor:
    """Control: the first A overwritten by another key of the pool, so B no longer follows A. The second A stays."""
    ks = torch.tensor(list(keys))
    own = (toks[:, half - 1, None] == ks).long().argmax(1)
    shift = torch.randint(1, len(ks), (len(toks),), generator=torch.Generator().manual_seed(seed))
    out = toks.clone()
    out[torch.arange(len(toks)), p] = ks[(own + shift) % len(ks)]
    return out


def absent_b(toks: Tensor, p: Tensor, b: Tensor, bs: Sequence[int], seed: int) -> Tensor:
    """Control: B overwritten by another token of its pool, so the original B is nowhere in the window."""
    pool = torch.tensor(list(bs))
    i = torch.randint(len(pool), (len(toks),), generator=torch.Generator().manual_seed(seed))
    i = (i + (pool[i] == b).long()) % len(pool)
    out = toks.clone()
    out[torch.arange(len(toks)), p + 1] = pool[i]
    return out


@dataclass
class Cue:
    train: Windows  # 41 tokens; targets per map below
    read: Windows
    b_read: Tensor
    p_read: Tensor
    keys: list[int]
    bs_read: list[int]
    word_pool: list[int]  # the candidates for the lookup's words: every word-like token but the keys, the Bs and the marker
    ranked: list[int]  # the mined word-like tokens by frequency, for the score's rows
    mean_rows: Tensor  # read-side rows, for the ablation's mean
    target: dict[str, dict[str, Tensor]]  # map -> train / read -> (N,) targets; "lookup" is set by `set_words`
    info: dict[str, Any]
    words: dict[int, int] = field(default_factory=dict)  # A -> its lookup word, from `set_words`

    def set_words(self, words: dict[int, int]) -> None:
        """Fix the lookup: every key gets a word of `word_pool`; the lookup's targets on the train and read windows follow."""
        assert set(words) == set(self.keys) and set(words.values()) <= set(self.word_pool)
        self.words = words
        lk = lambda w: torch.tensor([words[int(a)] for a in w.tokens[:, w.half - 1]])
        self.target["lookup"] = {"train": lk(self.train), "read": lk(self.read)}
        self.info["words"] = {str(k): v for k, v in words.items()}

    def read_windows(self, m: str) -> Tensor:
        return with_target(self.read, self.target[m]["read"]).tokens

    def controls(self, half: int, seed: int) -> dict[str, Tensor]:
        """The two cheap-solution sets, target the window's original B (last column)."""
        t, b = self.read.tokens, self.b_read
        cat = lambda x: torch.cat([x, b[:, None]], 1)
        return {"broken": cat(broken_pair(t, self.p_read, self.keys, half, seed)),
                "absent": cat(absent_b(t, self.p_read, b, self.bs_read, seed))}


def build_cue(rows: Tensor, marker: int, ranked: list[int], intact: Callable[[int], bool], half: int, S: dict[str, Any]) -> Cue:
    """The pools (word-like tokens of frequency ranks S["pool_ranks"], E58's cipher pool; each intact through the
    tokeniser, none the marker) cut at random into A, B for training and B for reading, so no B is both trained and read.
    The lookup's candidate words (Amendment 1, 2026-11-02) are all word-like tokens of `ranked` at any rank that are
    intact, minus the keys, both B pools and the marker, so a window's B is never a candidate. Rows split as E12's:
    train and read windows share no row and the same A pool. The copy map's target is B. The lookup's words are not
    fixed here: they are drawn from the candidates after W0's prior for B is measured (`pick_words`, `Cue.set_words`)."""
    lo, hi = S["pool_ranks"]
    pool = [t for t in ranked[lo:hi] if t != marker and intact(t)]
    order = [pool[int(i)] for i in torch.randperm(len(pool), generator=torch.Generator().manual_seed(S["pool_seed"]))]
    K = S["n_keys"]
    keys, rest = order[:K], order[K:]
    bs_train, bs_read = rest[::2], rest[1::2]
    used = {marker, *keys, *bs_train, *bs_read}
    word_pool = [t for t in ranked if t not in used and intact(t)]
    held = torch.arange(rows.shape[0]) % S["heldout_every"] == 0
    train, b_tr, _ = cue_windows(rows[~held], marker, keys, bs_train, half, S["n_train"], S["seed"])
    read, b_rd, p_rd = cue_windows(rows[held], marker, keys, bs_read, half, S["n_read"], S["seed"] + 2)
    target: dict[str, dict[str, Tensor]] = {"copy": {"train": b_tr, "read": b_rd}}
    at = read.tokens[:, half - 1]
    info = {"n_intact": len(pool), "n_b_train": len(bs_train), "n_b_read": len(bs_read), "keys": keys, "n_word_pool": len(word_pool),
            "min_read_per_key": min(int((at == k).sum()) for k in keys)}
    return Cue(train, read, b_rd, p_rd, keys, bs_read, word_pool, ranked, rows[held], target, info)


def own_tokens(keys: Sequence[int], windows: Tensor, half: int) -> dict[int, set[int]]:
    """Per key, every token of the windows (N, 2 half + 1) whose A (at `half - 1`) is that key: pass the train and the read
    windows together. The lookup's word for a key must not be among them, so the copying head cannot produce it."""
    return {k: set(windows[windows[:, half - 1] == k].unique().tolist()) for k in keys}


def admissible(keys: Sequence[int], word_pool: Sequence[int], at: Tensor, lp: Tensor, target_lp: float, factor: float,
               own: dict[int, set[int]]) -> dict[int, list[int]]:  # fmt: skip
    """Per key, the candidates `pick_words` may give it: mean log P over the key's read windows within log(factor) of
    `target_lp` (nan for a key with no window: never), and in none of the key's own train or read windows (`own`)."""
    out = {}
    for k in keys:
        near = (lp[at == k].mean(0) - target_lp).abs() <= math.log(factor)
        out[k] = [w for w, n in zip(word_pool, near.tolist(), strict=True) if n and w not in own[k]]
    return out


def pick_words(keys: Sequence[int], word_pool: Sequence[int], at: Tensor, lp: Tensor, target_lp: float, factor: float, seed: int,
               own: dict[int, set[int]]) -> dict[int, int]:  # fmt: skip
    """The lookup's words, so that W0's prior for the word is near B's. `lp` (N, len(word_pool)) is W0's log P of each
    candidate at the marker on the N read windows, `at` (N,) each window's A, `target_lp` the log of B's geometric-mean
    prior. A key takes the first unused candidate (a seeded order, `kv.bijection`) that `admissible` allows: log P,
    averaged over the windows of that key, within log(factor) of `target_lp`, and in none of the key's own windows. Each
    key within the factor puts the map's geometric mean (the window-weighted mean of these) within it too. AssertionError
    if some key finds none."""
    allowed = {k: set(v) for k, v in admissible(keys, word_pool, at, lp, target_lp, factor, own).items()}
    task = kv.bijection("lookup", -1, tuple(keys), list(word_pool), seed, ok=lambda k, w: w in allowed[k])
    return dict(zip(task.keys, task.values, strict=True))


def prior_histogram(lp: Tensor, width: float = 0.5) -> dict[str, int]:
    """Counts of candidates by their mean log10 P over all the windows, in bins `width` wide (key "-3.5" holds
    [-3.5, -3.0)): the pool's shape, printed before the draw so a stop explains itself."""
    m = lp.double().mean(0) / math.log(10)
    bins = torch.floor(m / width) * width
    return {f"{b:.1f}": int((bins == b).sum()) for b in sorted(set(bins.tolist()))}


def prior_ratio(a: float, b: float) -> float:
    """The larger of two priors over the smaller; inf unless both are finite and positive."""
    if not (finite(a) and finite(b) and a > 0 and b > 0):
        return math.inf
    return max(a, b) / min(a, b)


def read_lo(net: Any, w: Tensor, half: int, n: int, dW: ST.DW | None = None, readout: ST.Readout = ST.logodds_at,
            batch: int = 64) -> Tensor:  # fmt: skip
    """Per window, `readout` at `half` of the target model at W0 + dW, seeing the first n tokens, target in the last column."""
    with net.at(dW):
        return torch.cat([readout(net.run(w[s : s + batch, :n])[0], half, w[s : s + batch, -1]) for s in range(0, len(w), batch)])


def word_logp(net: Any, w: Tensor, half: int, n: int, word_pool: Sequence[int]) -> Tensor:
    """(N, len(word_pool)): W0's log P of each candidate word at `half` on each window (target model only)."""
    ids = torch.tensor(list(word_pool))
    readout = lambda logits, pos, _t: torch.log_softmax(logits[:, pos].double(), -1)[:, ids.to(logits.device)].cpu()
    return read_lo(net, w, half, n, None, readout)


# ----------------------------------------------------------------------------- the filed rules


def heads_status(H: Sequence[Head], n_all: int) -> str:
    """"empty" (no head over the bar), "all" (every head), "over_half" (fewer than |H| others to compare with), else "ok"."""
    if not H:
        return "empty"
    if len(H) == n_all:
        return "all"
    return "over_half" if len(H) > n_all - len(H) else "ok"


def control_a(base: float, with_H: float, random: Sequence[float], th: dict[str, Any]) -> str:
    """Stage (a) after the heads: "unreadable" if any number is not finite or there is no random draw; "no_copying" if
    W0's top-1 is off E52's reading by more than the tolerance; "H_insufficient" unless top-1 under H is below the bar (a
    tie fails); "random_harms" unless the median random draw is over its bar (a tie fails); else "pass". The median, not
    every draw, because a draw can hold a previous-token head, which is part of the copying circuit but not in H."""
    if not random or not all(finite(v) for v in (base, with_H, *random)):
        return "unreadable"
    if abs(base - th["top1_ref"]) > th["top1_ref_tol"]:
        return "no_copying"
    if not with_H < th["top1_H_max"]:
        return "H_insufficient"
    return "pass" if statistics.median(random) > th["top1_random_min"] else "random_harms"


def prior_band(p: float, th: dict[str, Any]) -> str:
    """W0's prior for B at the marker: "stop" if not finite or over prior_stop; "high" over prior_max; "expected" inside
    the filed band; "low" under it (lower than expected helps the test); "over_band" between the band and prior_max.
    Ties take the lower-risk side (exactly prior_stop is not a stop, exactly prior_max is not high)."""
    if not finite(p) or p > th["prior_stop"]:
        return "stop"
    if p > th["prior_max"]:
        return "high"
    lo, hi = th["prior_band"]
    return "low" if p < lo else "expected" if p <= hi else "over_band"


def read_step(copy: dict[int, float], lookup: dict[int, float], gate: float) -> int:
    """The step a seed's two states are read at, so that both have the same recipe and about the same update size: the
    first kept step with both gains at the gate; else the first with the copy map's; else the first with the lookup's;
    else the last. A nan gain is not at the gate. The gains are the specific ones (E58's gate was on the specific gain):
    the copy map's gain minus its gain with B absent, the lookup's minus the mean gain of the other words, never the raw
    gain, which early in a fine-tune is mostly a general push toward the pool's tokens."""
    steps = sorted(copy)
    over = lambda g, s: finite(g[s]) and g[s] >= gate
    for pick in (lambda s: over(copy, s) and over(lookup, s), lambda s: over(copy, s), lambda s: over(lookup, s)):
        for s in steps:
            if pick(s):
                return s
    return steps[-1]


def share_lost(G: float, G_abl: float) -> float:
    """The share of the trained gain lost under ablation, net of W0's change under the same ablation. G = lo(state) -
    lo(W0), G_abl = the same with both ablated; 1 - G_abl / G. nan when either is not finite or G is not positive (a
    zero update has no share; its check is that G - G_abl reads 0)."""
    return 1 - G_abl / G if finite(G) and finite(G_abl) and G > 0 else math.nan


def share_label(G: float, G_abl: float, th: dict[str, Any]) -> str:
    """"unreadable" if a number is not finite; "too_small" under the gain gate (a tie passes); then "old" iff the share
    >= share_old_min, "new" iff <= share_new_max (negative shares included: ablation raises the trained gain), else
    "between". Ties count to the named side."""
    if not (finite(G) and finite(G_abl)):
        return "unreadable"
    if G < th["G_min"]:
        return "too_small"
    s = share_lost(G, G_abl)
    return "old" if s >= th["share_old_min"] else "new" if s <= th["share_new_max"] else "between"


def share_sd(d: Tensor, d_abl: Tensor, n_boot: int, seed: int) -> float:
    """Row bootstrap sd of the share (paired windows); nan if a resample's mean gain is not positive."""
    i = torch.randint(len(d), (n_boot, len(d)), generator=torch.Generator().manual_seed(seed))
    g = d.double()[i].mean(1)
    return float((1 - d_abl.double()[i].mean(1) / g).std()) if bool((g > 0).all()) else math.nan


LABELLED = ("old", "new", "between")


def map_reading(recs: Sequence[dict[str, Any]], th: dict[str, Any]) -> str:
    """One map over its seeds: "unreadable" if any state is unreadable or missing; "too_small" with fewer than min_seeds
    labelled (below that a seed's gain cannot be labelled, and it is left out with a note); else "old" or "new" if every
    labelled seed reads so, "mixed" otherwise."""
    labs = [r.get("label", "unreadable") for r in recs]
    if "unreadable" in labs or not labs:
        return "unreadable"
    kept = [x for x in labs if x in LABELLED]
    if len(kept) < th["min_seeds"]:
        return "too_small"
    return kept[0] if len(set(kept)) == 1 and kept[0] != "between" else "mixed"


def r_verdict(copy_r: Sequence[Any], lookup_r: Sequence[Any], th: dict[str, Any]) -> dict[str, Any]:
    """The filed reading of r, labelled seeds only, called with the two priors already matched (`score` refuses
    otherwise). gap = mean copy r minus mean lookup r.
    "r_follows_mechanism" iff every copy r >= r_cut, every lookup r < r_cut and gap >= r_gap.
    "r_follows_prior" (Amendment 1, 2026-11-02) iff the mean gap is under r_gap (signed, as filed) and the maps are on the
    same side of the cut: every r of both < r_cut (both low; the filed cell) or every r of both >= r_cut (both high; added;
    a mean gap of -r_gap or less with both high is still this cell, as no rule names it). With the
    priors matched, either means r does not tell the maps apart at this prior, so r follows the prior and not the
    mechanism. Both high is the reading "old mechanism, answer weakly present" when W0 already leans toward B with the
    pair intact.
    "r_reverses_label" (Amendment 1, named surprise, announce it) iff every copy r < r_cut, every lookup r >= r_cut and
    gap <= -r_gap: r is high for the map whose label says "new" and low for the one that says "old".
    Anything else is "untested": all low with a gap (copy 0.44 beside lookup 0.10), both high with a gap of r_gap or more,
    a split that does not clear the gap, or a missing or non-finite r. near_cut: some r within r_noise of the cut, so the
    class is not firm."""
    rs = [*copy_r, *lookup_r]
    if not copy_r or not lookup_r or not all(finite(r) for r in rs):
        return {"class": "untested", "gap": math.nan, "near_cut": False}
    gap = sum(copy_r) / len(copy_r) - sum(lookup_r) / len(lookup_r)
    cut = th["r_cut"]
    if min(copy_r) >= cut and max(lookup_r) < cut and gap >= th["r_gap"]:
        c = "r_follows_mechanism"
    elif max(copy_r) < cut and min(lookup_r) >= cut and gap <= -th["r_gap"]:
        c = "r_reverses_label"
    elif gap < th["r_gap"] and (max(rs) < cut or min(rs) >= cut):
        c = "r_follows_prior"
    else:
        c = "untested"
    return {"class": c, "gap": gap, "near_cut": any(abs(r - th["r_cut"]) <= th["r_noise"] for r in rs)}


def score(states: dict[str, dict[str, Any]], prior: str, th: dict[str, Any], seeds: Sequence[int], ratio: float) -> dict[str, Any]:
    """The verdict. `ratio` is the measured prior ratio of the two maps (`prior_ratio`, larger over smaller; the run stops
    in stage (b) over `prior_match_max`, so a ratio over it is unreachable, and refused here all the same). Every
    "no_label_*" or non-elicitation outcome means r is reported but not scored. The checks run in this order, and the
    first that fires is the verdict: the priors not matched (`priors_not_matched`, a tie at the bar passes, a non-finite
    ratio fails); a map too small or unreadable; a random floor at or over its bar
    (median random share, on any labelled state); the copy map a lean (B's gain with B absent over absent_max of G, on a
    labelled seed); the lookup a lean (specific share under spec_min); the copy map not reading "old"; the lookup not
    reading "new"; then `r_verdict` (whose classes now include r_reverses_label). Reported beside, not changing it: `pair_ignored` (a labelled copy seed whose gain with
    the pair broken is over broken_max of G) and `prior_high`."""
    recs = {m: [states.get(f"{m}_s{s}", {}) for s in seeds] for m in ("copy", "lookup")}
    kept = {m: [r for r in v if r.get("label") in LABELLED] for m, v in recs.items()}
    read = {m: map_reading(v, th) for m, v in recs.items()}
    beside = {"copy_reading": read["copy"], "lookup_reading": read["lookup"], "prior_high": prior == "high", "prior_ratio": ratio,
              "pair_ignored": any(not (finite(r.get("broken_ratio")) and r["broken_ratio"] <= th["broken_max"]) for r in kept["copy"])}
    r_of = {m: [r.get("r") for r in v] for m, v in kept.items()}
    all_kept = [*kept["copy"], *kept["lookup"]]
    if not (finite(ratio) and ratio <= th["prior_match_max"]):
        v = "priors_not_matched"
    elif "unreadable" in read.values():
        v = "no_label_unreadable"
    elif read["copy"] == "too_small":
        v = "no_label_copy_did_not_learn"
    elif read["lookup"] == "too_small":
        v = "no_label_lookup_did_not_learn"
    elif not all(finite(r.get("random_median")) and r["random_median"] < th["floor_max"] for r in all_kept):
        v = "no_label_floor_high"
    elif not all(finite(r.get("absent_ratio")) and r["absent_ratio"] <= th["absent_max"] for r in kept["copy"]):
        v = "no_label_copy_is_a_lean"
    elif not all(finite(r.get("spec_frac")) and r["spec_frac"] >= th["spec_min"] for r in kept["lookup"]):
        v = "no_label_lookup_is_a_lean"
    elif read["copy"] != "old":
        v = "copy_not_elicitation"
    elif read["lookup"] != "new":
        v = "lookup_uses_the_heads"
    else:
        rv = r_verdict(r_of["copy"], r_of["lookup"], th)
        return {"verdict": rv["class"], "r": rv, "r_scored": True, "r_copy": r_of["copy"], "r_lookup": r_of["lookup"], **beside}
    return {"verdict": v, "r_scored": False, "r_copy": r_of["copy"], "r_lookup": r_of["lookup"], **beside}


# ----------------------------------------------------------------------------- Amendment 2's filed rules (results/predictions_amend2.json)


def agree(labels: Sequence[str], th: dict[str, Any]) -> str:
    """One row over a map's seeds: "unreadable" if any seed is or there is none; "too_small" with fewer than min_seeds labelled
    (a too_small seed is left out); else the label every labelled seed shares; else "in_between"."""
    if not labels or "unreadable" in labels:
        return "unreadable"
    kept = [x for x in labels if x != "too_small"]
    if len(kept) < th["min_seeds"]:
        return "too_small"
    return kept[0] if len(set(kept)) == 1 else "in_between"


def control_masked(base: float, with_H: float, ref: float, th: dict[str, Any]) -> str:
    """Row 1, top-1 through the masked forward: "unreadable" if a number is not finite; "no_copying" unless the intact
    forward is over top1_ref_min (a tie fails; a forward that does not copy cannot test the hook); "hook_missed" unless the
    head off is under top1_H_max (a tie fails); "ref_harmed" unless the reference head off is over top1_ref_min; else "pass"."""
    if not all(finite(v) for v in (base, with_H, ref)):
        return "unreadable"
    if not base > th["top1_ref_min"]:
        return "no_copying"
    if not with_H < th["top1_H_max"]:
        return "hook_missed"
    return "pass" if ref > th["top1_ref_min"] else "ref_harmed"


def gain_ok(G: float, th: dict[str, Any]) -> str | None:
    """"unreadable" if not finite, "too_small" under min_gain (a tie passes; zero and negative gains included), else None."""
    if not finite(G):
        return "unreadable"
    return None if G >= th["min_gain"] else "too_small"


def masked_class(gain: float, m: float, t: float, th: dict[str, Any]) -> str:
    """One state, masked share m beside target share t: the gain's gate; a non-finite share is "unreadable"; then "within"
    |m - t| <= tol (a tie counts); "over" m > t + tol; "other_path" m < other_path_max (a tie is not; negative included);
    else "between" (m under t - tol and at or over other_path_max)."""
    if bad := gain_ok(gain, th):
        return bad
    if not (finite(m) and finite(t)):
        return "unreadable"
    if abs(m - t) <= th["tol"]:
        return "within"
    if m > t:
        return "over"
    return "other_path" if m < th["other_path_max"] else "between"


def masked_reading(labels: Sequence[str], ref_shares: Sequence[float], th: dict[str, Any]) -> str:
    """The copy map's verdict: "unreadable", "too_small" (`agree`); "floor_high" if the reference head's masked share is not
    under floor_max on every state (a tie fails, a non-finite one fails); "r_reads_the_head" every labelled seed "within";
    "head_more_than_target" every labelled seed "within" or "over", at least one "over"; "mask_reaches_b_another_way" every
    labelled seed "other_path"; else "in_between"."""
    one = agree(labels, th)
    if one in ("unreadable", "too_small"):
        return one
    if not all(finite(s) and s < th["floor_max"] for s in ref_shares):
        return "floor_high"
    kept = {x for x in labels if x != "too_small"}
    if kept == {"within"}:
        return "r_reads_the_head"
    if kept <= {"within", "over"}:
        return "head_more_than_target"
    return "mask_reaches_b_another_way" if kept == {"other_path"} else "in_between"


def two_head_class(G: float, both: float, single: float, th: dict[str, Any]) -> str:
    """One copy state, target share under both heads: the gain's gate; "circuit" at or over two_head_min; "no_more" within
    tol of the single head's share (a tie counts); else "between" (including under the single head by more than tol)."""
    if bad := gain_ok(G, th):
        return bad
    if not (finite(both) and finite(single)):
        return "unreadable"
    if both >= th["two_head_min"]:
        return "circuit"
    return "no_more" if abs(both - single) <= th["tol"] else "between"


def lookup_two_head(G: float, both: float, th: dict[str, Any]) -> str:
    """One lookup state, target share under both heads: the gain's gate; "clean" at or under two_head_lookup_max (a tie
    counts, negative included); else "uses_circuit"."""
    if bad := gain_ok(G, th):
        return bad
    if not finite(both):
        return "unreadable"
    return "clean" if both <= th["two_head_lookup_max"] else "uses_circuit"


def broken_class(G: float, b: float, intact: float, th: dict[str, Any]) -> str:
    """One copy state, the pair-broken gain's share lost under the head: the gain's gate; "outside_head" at or under tol
    (the head off barely moves it; negative included, checked first); "through_head" within tol of the intact share; else
    "between" (including over the intact share by more than tol)."""
    if bad := gain_ok(G, th):
        return bad
    if not (finite(b) and finite(intact)):
        return "unreadable"
    if b <= th["tol"]:
        return "outside_head"
    return "through_head" if abs(b - intact) <= th["tol"] else "between"


def own_fall_class(G: float, own: float, share: float, th: dict[str, Any]) -> str:
    """One state: does the state's own fall under the head outrun the filed net share? The gain's gate; "undercounts" over
    share + tol; else "stands" (a tie stands, a fall under the share included)."""
    if bad := gain_ok(G, th):
        return bad
    if not (finite(own) and finite(share)):
        return "unreadable"
    return "undercounts" if own - share > th["tol"] else "stands"


def bracket_class(fall: float, th: dict[str, Any]) -> str:
    """W0's own fall under the head against the analysis's bracket: "inside" [0, bracket_max] (ties count), "below" (the head off
    raises W0), "above", "unreadable" if not finite."""
    if not finite(fall):
        return "unreadable"
    return "below" if fall < 0 else "inside" if fall <= th["bracket_max"] else "above"


def size_reading(gap: float, swapped: float, reproduced: bool) -> str:
    """Row 7: "unreadable" if alpha = 1 missed the stored r, a number is not finite, or the stored gap is not positive (no gap
    to close). "size_accounts" if the gap closes by more than half (swapped < gap / 2, a swapped gap below zero included);
    else "size_does_not" (a tie at exactly half, and a gap that widens)."""
    if not (reproduced and finite(gap) and finite(swapped)) or gap <= 0:
        return "unreadable"
    return "size_accounts" if swapped < gap / 2 else "size_does_not"


# ----------------------------------------------------------------------------- Amendment 3's filed rules and reads (results/predictions_amend3.json)


@torch.no_grad()
def gamma_inputs(model: Any, net: Any, toks: Tensor, head: Head, means: Tensor) -> dict[str, Any]:
    """Does `ablated` change what Γ reads? Γ reads the inputs a clean forward caches (`vpd.compute_ci`, cache_type "input"),
    so this compares that cache with the head on and off on one batch, in compute_ci's autocast. Expected: o_proj's input
    holds the mean in the head's slot and is unchanged elsewhere; modules before the edit (earlier blocks, the head's own
    q, k, v) are exactly unchanged; every module after it (the block's MLP, later blocks) moved. `status`: "slot_missed",
    "upstream_moved", "downstream_unmoved" or "pass"."""
    lay, h = head
    d = means.shape[-1]

    def cache() -> dict[str, Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=toks.is_cuda):
            return model(toks, cache_type="input").cache

    on = cache()
    with ablated(net, [head], means):
        off = cache()
    o = f"h.{lay}.attn.o_proj"
    moved = {k: float((off[k].float() - on[k].float()).abs().max()) for k in on if k != o}
    before = lambda k: int(k.split(".")[1]) < lay or (int(k.split(".")[1]) == lay and k.split(".")[2] == "attn")
    rest = torch.ones(off[o].shape[-1], dtype=torch.bool, device=off[o].device)
    rest[h * d : (h + 1) * d] = False
    slot_ok = bool((off[o][..., ~rest] == means[lay, h].to(off[o].device, off[o].dtype)).all()) and torch.equal(on[o][..., rest], off[o][..., rest])
    up, down = [v for k, v in moved.items() if before(k)], [v for k, v in moved.items() if not before(k)]
    status = "slot_missed" if not slot_ok else "upstream_moved" if any(v != 0 for v in up) else "downstream_unmoved" if any(v == 0 for v in down) else "pass"
    return {"status": status, "slot_is_mean": slot_ok, "upstream_max": max(up, default=0.0), "downstream_min": min(down, default=math.inf), "moved": moved}


def slot_reads(V: Tensor, heads: int, alive: Tensor, share_min: float) -> list[Tensor]:
    """Per head, the alive subcomponents of an o_proj whose read vector V (d_in, C) has at least `share_min` of its squared norm
    in that head's slot (a tie counts). A vector of norm 0 reads none. At `share_min` over 0.5 a subcomponent reads at most one head."""
    sq = V.double().square()
    share = sq.view(heads, -1, V.shape[1]).sum(1) / sq.sum(0)
    return [((share[h] >= share_min) & alive).nonzero().squeeze(1) for h in range(heads)]


def slot_class(n: int, gate: float, th: dict[str, Any]) -> str:
    """The mean gate on a head's slot: "empty" if no subcomponent reads it; "unreadable" if not finite; "closed" under
    slot_closed_max (a tie is not); else "open"."""
    if n == 0:
        return "empty"
    if not finite(gate):
        return "unreadable"
    return "closed" if gate < th["slot_closed_max"] else "open"


def gamma_reading(labels: Sequence[str], lookup_labels: Sequence[str], ref_shares: Sequence[float], th: dict[str, Any]) -> str:
    """The copy map's verdict with Γ recomputed, from `masked_class` labels against the target share. "unreadable", "too_small"
    (`agree` on the copy map); "floor_high" if a reference is not under floor_max in size (the caller passes (3,5)'s share on
    all six states and W0's fall with Γ recomputed over the map's mean gain, so that Γ simply breaking off its distribution
    voids every reading; Γ's reaction can raise a gain as well as lower it, so the size is read; a tie and a nan fail);
    "still_other_way" every labelled seed under
    other_path_max; "in_between" anything mixed or between. Only the two readings that say the head acts through the masks read
    the lookup, since its target share is about 0 and the same head goes off there: unless every lookup seed is "within" they
    read "not_specific" (or the lookup's own "unreadable", "too_small"), as Γ then reacts to the head whatever the map.
    "head_acts_through_masks" every labelled seed "within"; "head_more_than_target" within or over, at least one over."""
    one = masked_reading(labels, [abs(s) for s in ref_shares], th)
    if one in ("unreadable", "too_small", "floor_high"):
        return one
    if one == "mask_reaches_b_another_way":
        return "still_other_way"
    if one == "in_between":
        return one
    look = agree(lookup_labels, th)
    if look in ("unreadable", "too_small"):
        return look
    if look != "within":
        return "not_specific"
    return "head_acts_through_masks" if one == "r_reads_the_head" else "head_more_than_target"


# ----------------------------------------------------------------------------- Amendment 4's filed rules and reads (results/predictions_amend4.json)
# "Alive vector": one float64 entry per alive subcomponent, matrices in sorted name order, subcomponents in index order.


def second_a_holds(toks: Tensor, p: Tensor, b: Tensor, half: int) -> bool:
    """Every window has its second A at half - 1 (same token as the first A at p, with B at p + 1), and half - 2 is neither A nor B,
    so a mask at half - 2 is read on a plain Pile token of the same window."""
    i, a, ref = torch.arange(len(toks)), toks[:, half - 1], toks[:, half - 2]
    return bool(((toks[i, p] == a) & (toks[i, p + 1] == b) & (ref != a) & (ref != b)).all())


def pair_broken_holds(intact: Tensor, broken: Tensor, p: Tensor, half: int) -> bool:
    """The pair-broken windows differ from the intact ones at the first A alone, so the second A is the same token at the same position."""
    i = torch.arange(len(intact))
    other = torch.ones_like(intact, dtype=torch.bool)
    other[i, p] = False
    return bool((broken[i, p] != intact[i, p]).all() and (broken[other] == intact[other]).all() and (broken != broken[:, half - 1, None]).sum(1).eq(intact.shape[1] - 1).all())


def gate_means(gates: Sequence[dict[str, Tensor]], alive: dict[str, Tensor], pos: int) -> Tensor:
    """The alive vector of each subcomponent's gate at `pos`, averaged over the windows of `gates` (per batch, (B, T, C))."""
    return torch.cat([torch.cat([g[k][:, pos][:, alive[k]] for g in gates]).double().mean(0) for k in sorted(alive)])


def s_old(a: Tensor) -> Tensor:
    """Alive-vector indices of the fewest entries holding half of `a`'s positive mass: largest first, ties in index order, reaching
    exactly half counts. Empty when `a` has no positive mass or is not finite (k = 0: `mask_test` reads "unreadable")."""
    pos = a.double().clamp_min(0)
    total = float(pos.sum())
    if not (math.isfinite(total) and total > 0):
        return torch.empty(0, dtype=torch.long)
    order = torch.argsort(pos, descending=True, stable=True)
    return order[: int((pos[order].cumsum(0) * 2 >= total).nonzero()[0]) + 1]


def cosine(x: Tensor, y: Tensor) -> float:
    """nan if either vector has norm 0 or is not finite (an empty vector included)."""
    nx, ny = float(x.double().norm()), float(y.double().norm())
    return float(x.double() @ y.double()) / (nx * ny) if nx > 0 and ny > 0 and math.isfinite(nx * ny) else math.nan


def chance_cosine(delta: Tensor, a: Tensor, draws: int, seed: int) -> tuple[float, float]:
    """Mean and sd over `draws` seeded permutations of `a`'s entries of cosine(delta, a)."""
    gen = torch.Generator().manual_seed(seed)
    v = torch.tensor([cosine(delta, a[torch.randperm(len(a), generator=gen)]) for _ in range(draws)], dtype=torch.float64)
    return float(v.mean()), float(v.std())


def closing_pool(delta: Tensor, old: Tensor) -> tuple[Tensor, bool]:
    """What a random closing draws from: the opened entries (delta > 0) outside `old`; if fewer than |old|, every entry outside
    `old`, and True. Outside `old`, so that a mask opening exactly `old` does not make its own reference (Amendment 4, a departure)."""
    rest = torch.ones(len(delta), dtype=torch.bool)
    rest[old] = False
    opened = ((delta > 0) & rest).nonzero().squeeze(1)
    return (opened, False) if len(opened) >= len(old) else (rest.nonzero().squeeze(1), True)


def random_sets(pool: Tensor, k: int, draws: int, seed: int) -> list[Tensor]:
    """`draws` seeded sets of k entries of `pool`. AssertionError if the pool is too small (k <= half the alive set cannot be)."""
    assert 0 < k <= len(pool), (k, len(pool))
    gen = torch.Generator().manual_seed(seed)
    return [pool[torch.randperm(len(pool), generator=gen)[:k]] for _ in range(draws)]


def largest(delta: Tensor, k: int, outside: Tensor | None = None) -> Tensor:
    """Alive-vector indices of the k largest entries, ties in index order, optionally leaving out the entries `outside`."""
    order = torch.argsort(delta, descending=True, stable=True)
    if outside is not None:
        order = order[~torch.isin(order, outside)]
    return order[:k]


def closed(gates: Sequence[dict[str, Tensor]], ref: Sequence[dict[str, Tensor]], idx: Tensor, alive: dict[str, Tensor]) -> list[dict[str, Tensor]]:
    """`gates` with the alive-vector entries `idx` set to `ref`'s values, at every position of every window. `gates` is not changed."""
    keys = sorted(alive)
    sizes = [int(alive[k].sum()) for k in keys]
    mark = torch.zeros(sum(sizes), dtype=torch.bool)
    mark[idx] = True
    per = {k: alive[k].nonzero().squeeze(1)[m] for k, m in zip(keys, mark.split(sizes), strict=True)}

    def edit(g: Tensor, r: Tensor, w: Tensor) -> Tensor:
        x = g.clone()
        x[..., w] = r[..., w]
        return x

    return [{k: edit(g[k], r[k], per[k]) if len(per[k]) else g[k] for k in g} for g, r in zip(gates, ref, strict=True)]


TARGET_CAUSES = ("no_copying", "head_not_carrying", "no_pair_dependence")  # second_a_control's statuses about the target model, not the masks


def second_a_control(p_b: float, fall: float, lean: float, mask_fall: float, mask_lean: float, th: dict[str, Any]) -> str:
    """W0 at the second A, in order: "unreadable" (a number not finite). About the target model: "no_copying" P(B) under p_b_min;
    "head_not_carrying" its fall under the head under target_fall_min; "no_pair_dependence" its lean (intact minus pair-broken) under
    lean_min, so W0 does not lose the copy when the pair is broken. About the mask: "mask_lean_off" the masked lean over masked_within
    from the target's; "mask_drops_head" the masked fall under masked_fall_frac of the target's. Else "pass". Ties pass."""
    if not all(finite(v) for v in (p_b, fall, lean, mask_fall, mask_lean)):
        return "unreadable"
    if p_b < th["p_b_min"]:
        return "no_copying"
    if fall < th["target_fall_min"]:
        return "head_not_carrying"
    if lean < th["lean_min"]:
        return "no_pair_dependence"
    if abs(mask_lean - lean) > th["masked_within"]:
        return "mask_lean_off"
    return "pass" if mask_fall >= th["masked_fall_frac"] * fall else "mask_drops_head"


def old_carries(share: float, random_median: float, th: dict[str, Any]) -> bool:
    """Does S_old carry W0's own copying? P2's bars on W0's closing at the second A. Not finite (a masked lean at or under 0 included): False."""
    return finite(share) and finite(random_median) and share >= th["close_min"] and share - random_median >= th["random_margin"]


def mask_test(control: str, carries: bool, k: int, n_alive: int, copy: Sequence[dict[str, Any]], lookup: Sequence[dict[str, Any]], th: dict[str, Any]) -> dict[str, Any]:
    """The verdict from the control's status, `old_carries`, and one record per seed and map, paired by seed, each with M (the masked
    gain), cos, chance, share, random_median and largest_k_share. First that fires:
    "unreadable" (the control's numbers not finite); "control_failed_target", "control_failed_mask" (the status's cause);
    "unreadable" (k = 0, k over half the alive set, a seed missing, a gain not finite); "too_small" (any gain under min_gain);
    "unreadable" (any other number not finite); "old_pieces_not_carrying" (S_old does not carry W0's own copying, so a state's closing
    cannot be read either way); "shared_pieces" (P1, P2, P3 on every seed; P2 on every seed implies P3, as 0.5 - 0.2 > 0.24, so P3 is
    reported and decides nothing); then the null, on the copying map alone, either on every seed of: (a) closing costs no more than random
    (share - random_median under random_margin) and alignment no more than chance (cos under align_min or under chance_factor x chance);
    (b) share at or under random_median. It reads "not_shared" if largest_k_share reaches close_min on every seed, else "k_too_small",
    as no k subcomponents could then reach the bar. "in_between" is anything else."""
    out: dict[str, Any] = {"verdict": "in_between", "control": control, "carries": carries, "k": k, "P1": [], "P2": [], "P3": []}
    done = lambda v: {**out, "verdict": v}
    if control == "unreadable":
        return done("unreadable")
    if control != "pass":
        return done("control_failed_target" if control in TARGET_CAUSES else "control_failed_mask")
    seeds = list(zip(copy, lookup, strict=True)) if len(copy) == len(lookup) else []
    if k == 0 or k > n_alive // 2 or not seeds:
        return done("unreadable")
    both = [*copy, *lookup]
    if not all(finite(r.get("M")) for r in both):
        return done("unreadable")
    if any(r["M"] < th["min_gain"] for r in both):
        return done("too_small")
    if not all(finite(r.get(f)) for r in both for f in ("cos", "chance", "share", "random_median", "largest_k_share")):
        return done("unreadable")
    aligned = [c["cos"] >= th["align_min"] and c["cos"] >= th["chance_factor"] * c["chance"] for c, _ in seeds]
    beats = [c["share"] - c["random_median"] >= th["random_margin"] for c, _ in seeds]
    out["P1"] = [x and l["cos"] < th["lookup_align_frac"] * c["cos"] for x, (c, l) in zip(aligned, seeds, strict=True)]
    out["P2"] = [x and c["share"] >= th["close_min"] and l["share"] <= th["lookup_close_max"] for x, (c, l) in zip(beats, seeds, strict=True)]
    out["P3"] = [c["share"] - l["share"] >= th["gap_min"] for c, l in seeds]
    if not carries:
        return done("old_pieces_not_carrying")
    if all(out["P1"]) and all(out["P2"]) and all(out["P3"]):
        return done("shared_pieces")
    if not any(aligned) and not any(beats) or all(c["share"] <= c["random_median"] for c, _ in seeds):
        return done("not_shared" if all(c["largest_k_share"] >= th["close_min"] for c, _ in seeds) else "k_too_small")
    return out
