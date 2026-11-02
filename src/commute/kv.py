"""Key -> value tasks (experiments/kv_task): a behaviour the model lacks, built from Pile windows.

A window is a 41-token Pile span with the token at `half - 1` replaced by a marker M and the token at `half` by a key x
drawn from K word-like tokens; the target at `half` is table[x]. `bijection`: the table is a random bijection from the
keys onto K other word-like tokens, a word-level substitution cipher (cf. Halawi et al. 2024, Walnut53). `identity`:
table[x] = x, the matched-format control. The table is the `token` argument of the harness (`data.target_tokens`).
Plan: reports/plans/2026-10-19_plan_key-value-task.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from commute.data import Windows


@dataclass(frozen=True)
class KVTask:
    name: str
    marker: int
    keys: tuple[int, ...]
    values: tuple[int, ...]  # values[i] is the target of keys[i]

    def table(self, vocab: int) -> Tensor:
        """(vocab,) int64 lookup key -> value; -1 off the keys, so a window without a key at `half` fails loudly."""
        t = torch.full((vocab,), -1, dtype=torch.long)
        t[torch.tensor(self.keys)] = torch.tensor(self.values)
        return t


def identity(name: str, marker: int, keys: tuple[int, ...]) -> KVTask:
    return KVTask(name, marker, keys, keys)


def bijection(name: str, marker: int, keys: tuple[int, ...], pool: list[int], seed: int, ok: Any = None) -> KVTask:
    """Values from `pool` (minus the keys) in a seeded random order: each key takes the first unused candidate that
    passes `ok(key, value)` (the build stage's baseline gate; None accepts all). A bijection by construction."""
    cands = [t for t in pool if t not in set(keys)]
    perm = torch.randperm(len(cands), generator=torch.Generator().manual_seed(seed))
    order = [cands[int(i)] for i in perm]
    values: list[int] = []
    for key in keys:
        value = next((v for v in order if v not in values and (ok is None or ok(key, v))), None)
        assert value is not None, f"no admissible value left in the pool for key {key}"
        values.append(value)
    return KVTask(name, marker, keys, tuple(values))


def pick_marker(counts: dict[int, int], decode: Any, vocab: int) -> int:
    """The lowest-id token with no occurrence in the mined rows that decodes to 1-3 printable ASCII characters with no
    letter, digit or whitespace: a cue the model has no associations for."""
    for t in range(vocab):
        if counts.get(t, 0):
            continue
        s = decode([t])
        if 1 <= len(s) <= 3 and s.isascii() and s.isprintable() and not s.isspace() and not any(c.isalnum() for c in s):
            return t
    raise ValueError("no zero-count marker token")


def windows(rows: Tensor, marker: int, keys: tuple[int, ...], half: int, n: int, seed: int) -> Windows:
    """n windows from random (row, position) pairs with the marker written at `half - 1` and the keys, each about n / K
    times in a seeded random order, at `half`. `g` is zero (no subcomponent defines these windows)."""
    gen = torch.Generator().manual_seed(seed)
    T = rows.shape[1]
    r = torch.randint(0, rows.shape[0], (n,), generator=gen)
    pos = torch.randint(half, T - half, (n,), generator=gen)
    key = torch.tensor(keys)[torch.randperm(n, generator=gen) % len(keys)]
    toks = torch.stack([rows[int(ri), int(pi) - half : int(pi) + half + 1] for ri, pi in zip(r, pos)]).clone()
    toks[:, half - 1] = marker
    toks[:, half] = key
    return Windows(toks, torch.zeros(n), r, pos, half)


@torch.no_grad()
def key_distributions(logits_fn: Any, w: Windows, device: str, batch: int = 64, geometric: bool = False) -> dict[int, Tensor]:
    """Per key x, the model's mean next-token distribution at `half` over the windows holding x: (V,) float64 on CPU.
    Every baseline (P(value | M x), P(x | M x)) is a lookup in it. `logits_fn(toks) -> (B, T, V)`. With `geometric`,
    the geometric mean over those windows, exp(mean log P), per token (not normalised; `keydep.geo_rate`'s measure)."""
    sums: dict[int, Tensor] = {}
    counts: dict[int, int] = {}
    for s in range(0, len(w), batch):
        toks = w.tokens[s : s + batch].to(device)
        f = torch.log_softmax if geometric else torch.softmax
        p = f(logits_fn(toks)[:, w.half].float(), -1).double().cpu()
        for i, k in enumerate(toks[:, w.half].tolist()):
            sums[k] = sums[k] + p[i] if k in sums else p[i]
            counts[k] = counts.get(k, 0) + 1
    return {k: (sums[k] / counts[k]).exp() if geometric else sums[k] / counts[k] for k in sums}
