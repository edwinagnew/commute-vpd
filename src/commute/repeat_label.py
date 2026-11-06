"""E125's repeat-label pair (plan reports/plans/2026-11-06_plan_e125-fresh-reuse-pair.md; settings and rules filed in
experiments/fresh_reuse_pair/results/predictions.json): the word pool, the task windows and the possession probe's Pile
positions for the capability C = "the word before the trigger already occurred earlier in the window". No model is
read here; tests/test_repeat_label.py runs every function on CPU.

Window layout is E112's (`experiments/case_label`): 41 tokens, the pool word at half - 1, the trigger at half, the
label read at half. "Earlier in the window" means positions 0 .. half - 2 (19 tokens). Label convention as E112's
`case_label`: y = 1 for label word A ("repeated"), 0 for B.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor


def word_tokens(decode: Callable[[list[int]], str], vocab: int) -> list[int]:
    """Every single token that is a space then letters only (E112's `case_pairs` test of a word), in id order."""
    out = []
    for t in range(vocab):
        try:
            s = decode([t])
        except (IndexError, KeyError, ValueError, TypeError):  # an id past the tokenizer's range
            continue
        if len(s) >= 2 and s[0] == " " and s[1:].isalpha():
            out.append(t)
    return out


def repeat_flags(rows: Tensor, half: int) -> tuple[Tensor, Tensor]:
    """For every position t with a full window (half <= t < T - half): prev = rows[:, t - 1], the word a window centred
    at t would hold at half - 1, and rep = whether that token also occurs in rows[:, t - half : t - 1], the window's
    positions 0 .. half - 2. Both (R, T - 2 half), prev int64, rep bool; column j is t = half + j."""
    T = rows.shape[1]
    prev = rows[:, half - 1 : T - half - 1]
    rep = torch.zeros_like(prev, dtype=torch.bool)
    for k in range(1, half):  # the earlier position t - 1 - k, k = 1 .. half - 1, i.e. window positions half - 2 .. 0
        rep |= prev == rows[:, half - 1 - k : T - half - 1 - k]
    return prev, rep


def class_counts(prev: Tensor, rep: Tensor, vocab: int) -> tuple[Tensor, Tensor]:
    """(vocab,) int64 each: how often each token is the word with an earlier occurrence (rep) and without."""
    return (torch.bincount(prev[rep], minlength=vocab)[:vocab], torch.bincount(prev[~rep], minlength=vocab)[:vocab])


def qualifying(words: Sequence[int], n_rep: Tensor, n_not: Tensor, min_count: int, max_rep_share: float, exclude: set[int]) -> list[int]:
    """The words, in id order, with at least min_count Pile positions in each class and an earlier-occurrence share
    n_rep / (n_rep + n_not) of at most max_rep_share (a word nearly always repeated cannot fill clean "not repeated"
    windows); a word in `exclude` is out."""
    return [w for w in words if w not in exclude and min(int(n_rep[w]), int(n_not[w])) >= min_count
            and int(n_rep[w]) <= max_rep_share * (int(n_rep[w]) + int(n_not[w]))]  # fmt: skip


def draw_words(ok: Sequence[int], n: int, seed: int) -> list[int]:
    """n words drawn without replacement, seeded, from the qualifying list (so the pool is not the most frequent words,
    which are function words)."""
    assert len(ok) >= n, f"{len(ok)} words qualify, need {n}"
    return [ok[i] for i in torch.randperm(len(ok), generator=torch.Generator().manual_seed(seed))[:n].tolist()]


def windows(rows: Tensor, marker: int, words: Sequence[int], n: int, half: int, p_range: tuple[int, int], tries: int,
            seed: int) -> tuple[Tensor, Tensor, Tensor]:  # fmt: skip
    """n task windows (n, 2 half + 1), their class y (n,) and the written position p (n,). Window i holds word w_i at half - 1 and the marker at half;
    a seeded order is cut into pairs, pair j goes to word j mod K, and its two windows take y = 0 and y = 1, so the
    classes are equal overall (n even) and within each word. Its Pile span is the
    first of `tries` seeded (row, position) draws with w_i nowhere in positions 0 .. half - 2. One position p in
    p_range (inclusive) is then written: w_i itself if y = 1 (repeated), else another pool word (not repeated). So w_i
    occurs in positions 0 .. half - 2 exactly y times, and both classes carry one written pool word there."""
    gen = torch.Generator().manual_seed(seed)
    K, T, L = len(words), rows.shape[1], 2 * half + 1
    assert n % 2 == 0, "n must be even: the windows come in pairs"
    idx = torch.randperm(n, generator=gen)
    wi, y = (idx // 2) % K, idx % 2
    w = torch.tensor(list(words))[wi]
    r = torch.randint(0, rows.shape[0], (tries, n), generator=gen)
    t = torch.randint(half, T - half, (tries, n), generator=gen)
    spans = rows[r[..., None], t[..., None] - half + torch.arange(L)]  # (tries, n, L)
    clean = ~(spans[..., : half - 1] == w[None, :, None]).any(-1)
    assert bool(clean.any(0).all()), f"{int((~clean.any(0)).sum())} windows found no clean span in {tries} tries"
    toks = spans[clean.long().argmax(0), torch.arange(n)].clone()
    p = torch.randint(p_range[0], p_range[1] + 1, (n,), generator=gen)
    other = torch.tensor(list(words))[(wi + torch.randint(1, K, (n,), generator=gen)) % K]
    toks[torch.arange(n), p] = torch.where(y == 1, w, other)
    toks[:, half - 1], toks[:, half] = w, marker
    return toks, y, p


def earlier_count(toks: Tensor, half: int) -> Tensor:
    """(N,) how often the word at half - 1 occurs in positions 0 .. half - 2: the window's true class (0 or 1 when
    built by `windows`)."""
    return (toks[:, : half - 1] == toks[:, half - 1 : half]).sum(1)


def probe_positions(prev: Tensor, rep: Tensor, words: Sequence[int], half: int, cap: int, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    """(row, t, y) of Pile positions whose previous token is a word of `words` (from `repeat_flags`, t = half + column),
    y = the earlier-occurrence flag. Per word n = min(cap, hits of each class) positions of each class, drawn seeded,
    so the classes are balanced per word (E112's `probe_positions`)."""
    gen = torch.Generator().manual_seed(seed)
    rs, ts, ys = [], [], []
    for w in words:
        hit = prev == w
        groups = [torch.nonzero(hit & rep, as_tuple=False), torch.nonzero(hit & ~rep, as_tuple=False)]
        n = min(cap, *(len(g) for g in groups))
        for cls, g in zip((1, 0), groups, strict=True):
            pick = g[torch.randperm(len(g), generator=gen)[:n]]
            rs.append(pick[:, 0])
            ts.append(pick[:, 1] + half)
            ys.append(torch.full((n,), cls, dtype=torch.long))
    return torch.cat(rs), torch.cat(ts), torch.cat(ys)


def twin_labels(toks: Tensor, half: int, table: dict[int, int]) -> Tensor:
    """(N,) the twin's y per window: the word's seeded label (`case_label.twin_y`)."""
    return torch.tensor([table[int(x)] for x in toks[:, half - 1]])
