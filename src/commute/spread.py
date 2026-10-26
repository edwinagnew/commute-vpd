"""Gamma's values and patterns at every position, for a fine-tune with no trigger (E36, `experiments/regating_spread`).

Plan: reports/plans/2026-10-26_plan_regating-without-cues.md. E35's arms (`commute.cuefree`) carried from one trigger position
to every position of whole rows. Values are kept as *flat* rows: one row per (sequence, position), the alive
subcomponents of every matrix concatenated in sorted key order (`shift_anatomy.flat`'s order), (N, D) float32.
A pattern is W0's values g0 at every position with the alive entries replaced by clip(g0 + c) or clip(a g0 + b); dead
subcomponents keep g0. Scored with the masked model (released subcomponents, Delta out) as the mean next-token
log-probability of the rows, over positions 0 .. T-2 (`masked_logp`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from commute.cuefree import CHUNK, _sqdist, assign, kmeans, line

Gates = dict[str, Tensor]  # matrix key -> (B, T, C) on the model's device


@dataclass
class Layout:
    """Where each matrix's alive subcomponents sit in a flat row."""

    keys: list[str]
    idx: dict[str, Tensor]  # key -> alive indices (on device)
    start: dict[str, int]

    @property
    def D(self) -> int:
        k = self.keys[-1]
        return self.start[k] + self.idx[k].numel()


def layout(alive: dict[str, Tensor], device: str) -> Layout:
    keys, idx, start, at = sorted(alive), {}, {}, 0
    for k in keys:
        idx[k] = alive[k].bool().nonzero().flatten().to(device)
        start[k], at = at, at + idx[k].numel()
    return Layout(keys, idx, start)


def to_flat(g: Gates, lay: Layout) -> Tensor:
    """(B, T, C) per matrix -> (B * T, D) float32."""
    return torch.cat([g[k].float().flatten(0, 1)[:, lay.idx[k]] for k in lay.keys], 1)


def with_alive(g: Gates, x: Tensor, lay: Layout) -> Gates:
    """g in float32 with each matrix's alive entries replaced by the matching columns of the flat rows x (B * T, D)
    (float32, so changes below bf16's step survive)."""
    out = {}
    for k in lay.keys:
        v = g[k].float().clone()
        B, T, _ = v.shape
        s = lay.start[k]
        v[:, :, lay.idx[k]] = x[:, s : s + lay.idx[k].numel()].reshape(B, T, -1).float()
        out[k] = v
    return out


def masked_logp(model: Any, rows: Tensor, gates: Gates) -> float:
    """Sum over positions 0 .. T-2 of the masked model's log-probability of the next token (Delta out), in float32:
    under bf16 autocast the rounding of each pattern's logits would move R by about 1e-4 / gap (E36 review)."""
    from commute.local_faith import masked_logits

    with torch.no_grad():
        lg = masked_logits(model, rows, {k: v.float() for k, v in gates.items()}, None, None, autocast=False)[:, :-1]
    return float(torch.log_softmax(lg, -1).gather(2, rows[:, 1:, None]).double().sum())


def target_logp(logits: Tensor, rows: Tensor) -> float:
    return float(torch.log_softmax(logits[:, :-1].float(), -1).gather(2, rows[:, 1:, None]).double().sum())


# ----------------------------------------------------------------------------- predicted changes, flat rows


def nearest(X: Tensor, Y: Tensor, k: int, same: tuple[Tensor, Tensor] | None = None) -> Tensor:
    """(len(Y), k) indices of each row of Y's k nearest rows of X (all on one device). With `same` = (labels of X,
    labels of Y), a row of Y whose label has at least k rows in X looks only among those; other rows look everywhere."""
    out = []
    for s in range(0, Y.shape[0], CHUNK):
        d2 = _sqdist(Y[s : s + CHUNK], X)
        if same is not None:
            lab_x, lab_y = same[0], same[1][s : s + CHUNK]
            enough = torch.bincount(lab_x, minlength=int(lab_y.max()) + 1)[lab_y] >= k
            d2 = d2.masked_fill((lab_y[:, None] != lab_x[None, :]) & enough[:, None], float("inf"))
        out.append(d2.topk(k, largest=False).indices)
    return torch.cat(out)


def nearest_change(Dch: Tensor, idx: Tensor) -> Tensor:
    """For each row of `idx` (from `nearest`), the mean change Dch of those rows."""
    return torch.cat([Dch[idx[s : s + CHUNK]].mean(1) for s in range(0, idx.shape[0], CHUNK)])


def group_change(Dch: Tensor, lab: Tensor, n: int) -> Tensor:
    """(n, D) mean change per label 0 .. n-1 (a label with no rows gets 0)."""
    sums = torch.zeros(n, Dch.shape[1], device=Dch.device).index_add_(0, lab, Dch)
    return sums / torch.bincount(lab, minlength=n).clamp_min(1)[:, None]


@dataclass
class Clusters:
    cent: Tensor  # (K', D), only centroids some fit row is nearest to
    change: Tensor  # (K', D) mean change per cluster


def fit_clusters(X: Tensor, Dch: Tensor, K: int, seed: int) -> Clusters:
    cent = kmeans(X, K, seed)
    cent = cent[assign(X, cent).unique()]
    lab = assign(X, cent)
    return Clusters(cent, group_change(Dch, lab, cent.shape[0]))


@dataclass
class TokenTable:
    """The mean change at fit positions holding each token seen at least `min_count` times; other tokens get the
    shared change (the last row)."""

    row_of: Tensor  # (V,) long: token id -> row of `change`
    change: Tensor  # (n_tokens + 1, D)
    n_tokens: int


def fit_tokens(tokens: Tensor, Dch: Tensor, vocab: int, min_count: int) -> TokenTable:
    cnt = torch.bincount(tokens, minlength=vocab)
    kept = (cnt >= min_count).nonzero().flatten()
    row_of = torch.full((vocab,), kept.numel(), dtype=torch.long, device=tokens.device)
    row_of[kept] = torch.arange(kept.numel(), device=tokens.device)
    lab = row_of[tokens]
    table = group_change(Dch, lab, kept.numel() + 1)
    table[-1] = Dch.mean(0)
    return TokenTable(row_of, table, int(kept.numel()))


