"""The P2.32 constructions and build rule (results/predictions.json -> definitions). Pure, no model; run.py calls them.

frequent_words  the K pool words W0 finds most likely after every key: largest minimum over keys of G[k, v].
balanced        each key one of the K words, at random (seed) and balanced: 128 / K keys a word, no key-to-word link.
neighbours      each key's nearest word-like token by cosine of the input embeddings, itself and every key excluded.
prior_ok        the stage-0 prior half of the build check: the gap (reuse minus new, decades) points to the new side by
                at least the margin; nan fails.
first_built     the first tried construction whose condition is "opposite"; None if none.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

EPS = 1e-12  # the inclusive bar gets the same float slack as E151's prior call


def frequent_words(G: Tensor, pool: Sequence[int], k: int) -> list[int]:
    """G: (keys, vocab) W0's geometric-mean P(v | M key). The k pool words with the largest minimum over keys (ties: pool order)."""
    low = G[:, list(pool)].min(0).values.tolist()
    order = sorted(range(len(pool)), key=lambda i: -low[i])
    assert len(order) >= k, f"pool of {len(order)} words is under k = {k}"
    return [pool[i] for i in order[:k]]


def balanced(keys: Sequence[int], words: Sequence[int], seed: int) -> tuple[int, ...]:
    """values[i] for keys[i]: the keys shuffled by `seed`, the j-th shuffled key gets words[j % K]."""
    assert len(keys) % len(words) == 0, (len(keys), len(words))
    order = list(range(len(keys)))
    random.Random(seed).shuffle(order)
    out = [0] * len(keys)
    for j, i in enumerate(order):
        out[i] = words[j % len(words)]
    return tuple(out)


def neighbours(E: Tensor, keys: Sequence[int], allowed: Tensor) -> tuple[int, ...]:
    """E: (vocab, d) input embeddings; allowed: (vocab,) bool (word-like, not a key). Each key's highest-cosine allowed
    token (ties: lower id)."""
    assert not bool(allowed[list(keys)].any()), "a key is allowed as its own neighbour"
    En = torch.nn.functional.normalize(E.float(), dim=1)
    sims = En[list(keys)] @ En.T
    sims[:, ~allowed.to(sims.device)] = -math.inf
    return tuple(int(v) for v in sims.argmax(1).tolist())


def prior_ok(gap: float, margin: float) -> bool:
    """gap = log10(prior reuse side) - log10(prior new side). True iff gap <= -margin: the prior points to the new side."""
    return not math.isnan(gap) and gap <= -margin + EPS


def first_built(conditions: Mapping[str, str]) -> str | None:
    """conditions in the order tried: the first whose condition is 'opposite'."""
    return next((n for n, c in conditions.items() if c == "opposite"), None)
