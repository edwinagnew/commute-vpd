"""The performed-anchor pair's rows and scored positions (E53, `experiments/performed_anchor`; added 2026-10-29).

Plan: reports/plans/2026-10-29_plan_performed-anchor.md ("The pair"). Two tasks on the same Pile rows:

- *Lowercase.* Each cased row decoded, lowercased and re-encoded on its own (`case_pairs`), the per-row form of
  `row_copy.lowercase_rows`. A *case position* is a token of the cased row that is a word start with a capital first
  letter (token string `Ġ[A-Z][A-Za-z]*`, e.g. " The", " US") at row index >= 1, whose lowercased string is itself one
  vocabulary entry and is the token the lowercased row has at the same byte offset, at row index >= 1 there too. Rows
  whose lowercased text changes byte length, or whose token bytes do not add up to the decoded text, give no positions
  (counted as skipped). Each position is scored in both rows: log p(lowercase form) - log p(capitalised form).
- *Cipher.* A derangement pi of the 50 word tokens (`Ġ[a-z]{2,}`: a space and two or more lowercase letters) of frequency ranks 50-99 (0-based, ties by id) counted
  on given rows, applied to every occurrence in the rows (context and targets alike; token level, no re-encoding).
  A *cipher position* is any row index >= 1 whose original token is in pi's domain; it is scored as
  log p(pi(x)) - log p(x) in the ciphered row.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from commute.finetune import logits_of

WORD = re.compile(r"^Ġ[a-z]{2,}$")
CAPITAL = re.compile(r"^Ġ[A-Z][A-Za-z]*$")


@dataclass
class Positions:
    """Scored positions: the token at `pos` of row `row` is predicted from positions < `pos`; `a`, `b` the two forms."""

    row: Tensor  # (P,) int64
    pos: Tensor  # (P,) int64, >= 1
    a: Tensor  # (P,) the form whose log-probability counts positively
    b: Tensor  # (P,)

    def __len__(self) -> int:
        return int(self.row.shape[0])


def _positions(rows: list[int], pos: list[int], a: list[int], b: list[int]) -> Positions:
    return Positions(*(torch.tensor(x, dtype=torch.long) for x in (rows, pos, a, b)))


@dataclass
class CasePairs:
    lower: Tensor  # (N, T) each cased row lowercased and re-encoded alone, cut to T, padded with end-of-text
    in_lower: Positions  # a = lowercase form, b = capitalised form, positions in `lower`
    in_cased: Positions  # the same words and forms, positions in the cased rows
    skipped_rows: int


def _starts(tok: Any, ids: list[int]) -> tuple[list[str], list[int], int]:
    """Token strings, each token's byte offset, and the total byte length (byte-level BPE: one character per byte)."""
    strs = tok.convert_ids_to_tokens(ids)
    starts, n = [], 0
    for s in strs:
        starts.append(n)
        n += len(s)
    return strs, starts, n


def case_pairs(rows: Tensor, tok: Any) -> CasePairs:
    """Case positions of `rows` (module docstring) and the lowercased rows they are read in."""
    T, eos, vocab = rows.shape[1], tok.eos_token_id, tok.get_vocab()
    lower, skipped = [], 0
    lo: tuple[list[int], list[int], list[int], list[int]] = ([], [], [], [])
    ca: tuple[list[int], list[int], list[int], list[int]] = ([], [], [], [])
    for i, r in enumerate(rows.tolist()):
        text = tok.decode(r, clean_up_tokenization_spaces=False)
        low_ids = tok.encode(text.lower())[:T]
        lower.append(low_ids + [eos] * (T - len(low_ids)))
        c_strs, c_starts, c_len = _starts(tok, r)
        l_strs, l_starts, _ = _starts(tok, low_ids)
        if not c_len == len(text.encode()) == len(text.lower().encode()):
            skipped += 1
            continue
        at = {s: k for k, s in enumerate(l_starts) if k >= 1}
        for j in range(1, T):
            k = at.get(c_starts[j])
            low_s = "Ġ" + c_strs[j][1:].lower()  # not .lower() on the whole string: it turns Ġ into ġ
            if CAPITAL.match(c_strs[j]) and low_s in vocab and k is not None and l_strs[k] == low_s:
                for out, p in ((lo, k), (ca, j)):
                    out[0].append(i)
                    out[1].append(p)
                    out[2].append(vocab[low_s])
                    out[3].append(r[j])
    return CasePairs(torch.tensor(lower, dtype=torch.long), _positions(*lo), _positions(*ca), skipped)


def word_ranks(tok: Any, rows: Tensor) -> Tensor:
    """Word tokens (`WORD`) by descending count in `rows`, ties by ascending id; tokens absent from `rows` last."""
    ids = torch.tensor([i for s, i in tok.get_vocab().items() if WORD.match(s)], dtype=torch.long).sort().values
    counts = torch.bincount(rows.flatten(), minlength=len(tok))[ids]
    order = sorted(range(len(ids)), key=lambda n: (-int(counts[n]), int(ids[n])))
    return ids[order]


def derangement(ids: list[int], seed: int) -> dict[int, int]:
    """A uniformly drawn permutation of `ids` with no fixed point (shuffle with `random.Random(seed)` until none)."""
    assert len(ids) >= 2 and len(set(ids)) == len(ids)
    rng, dst = random.Random(seed), list(ids)
    while True:
        rng.shuffle(dst)
        if all(a != b for a, b in zip(ids, dst)):
            return dict(zip(ids, dst))


def cipher_rows(rows: Tensor, pi: dict[int, int]) -> tuple[Tensor, Positions]:
    """`rows` with every occurrence of pi's domain replaced by its image, and the cipher positions there (a = pi(x),
    b = x)."""
    table = torch.arange(max(int(rows.max()), max(pi)) + 1)
    for x, y in pi.items():
        table[x] = y
    out = table[rows]
    hit = torch.zeros_like(rows, dtype=torch.bool)
    for x in pi:
        hit |= rows == x
    hit[:, 0] = False
    r, p = torch.nonzero(hit, as_tuple=True)
    return out, Positions(r, p, out[r, p], rows[r, p])


@torch.no_grad()
def log_odds(net: nn.Module, rows: Tensor, at: Positions, device: str, batch: int) -> Tensor:
    """log p(a) - log p(b) of the next token at each position of `at`, (P,) float64 on CPU, from `net`'s logits (bf16
    autocast read in float32, as `row_copy.mean_logp`)."""
    out = torch.empty(len(at), dtype=torch.float64)
    for s in range(0, rows.shape[0], batch):
        sel = torch.nonzero((at.row >= s) & (at.row < s + batch), as_tuple=True)[0]
        if len(sel) == 0:
            continue
        lp = torch.log_softmax(logits_of(net, rows[s : s + batch].to(device)), -1)
        r, p = (at.row[sel] - s).to(device), (at.pos[sel] - 1).to(device)
        d = lp[r, p, at.a[sel].to(device)] - lp[r, p, at.b[sel].to(device)]
        out[sel] = d.double().cpu()
    return out
