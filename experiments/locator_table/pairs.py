"""E97 Amendment 1's sites as edge reads: the gain path on each task's E97 site, and the input pairs of the scoping
review's table (reports/reviews/2026-11-05_review_e97-eapig-scoping.md, section 3), each corrupted input the same length
as its clean one. GPU job only (the E53 sites tokenize Pile rows on load).

  copy       E62's read windows, B overwritten by another read B (`copying.absent_b`); logit(B) - logit(B')
  lookup     the same windows, the second A (at half - 1) swapped for another key A'; logit(word(A)) - logit(word(A'))
  lowercase  E53's lowercased whole rows, every case position put back to its capitalised token (`case_pairs`' b);
             log p(lower) - log p(capital) at E53's reads
  cipher     E53's ciphered whole rows against the plain rows, token for token; log p(pi(x)) - log p(x)
  kv_*       a key table's held-out windows, the key at `half` swapped for another key of the windows; logit(v(k)) - logit(v(k'))
The constant-answer tasks have no pair: the gain path only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import sites
import torch
from torch import Tensor

from commute import edges as EG
from commute import locator as L
from commute.copying import absent_b
from commute.data import pile_rows

COPY_PAIR = {"T5 copy": "copy", "T5 lookup": "copy"}  # W0's C circuit: induction on the copy pair, for both T5 maps
OWN_PAIR = {"T5 copy": "copy", "T5 lookup": "lookup", "T7 lowercase": "lowercase", "T8 cipher": "cipher"}


def own_pair(task: str) -> str | None:
    """The pair a task's own circuit is read on: its row above, else its key table's site, else none (constant answers)."""
    site = sites.TASKS[task].site
    return OWN_PAIR.get(task, site if site in sites.KV else None)


def c_pair(task: str) -> str | None:
    """The pair W0's C circuit is read on: induction's copy pair for the T5 maps, else the task's own pair."""
    return COPY_PAIR.get(task, own_pair(task))


def shifted(own: Tensor, n: int, seed: int) -> Tensor:
    """Another index of n for each row, as `copying.broken_pair` draws it."""
    shift = torch.randint(1, n, (len(own),), generator=torch.Generator().manual_seed(seed))
    return (own + shift) % n


def swap_key(s: L.Site, at: int, answer: Any, seed: int) -> EG.Pair:
    """The token at `at` swapped for another of the tokens found there; `answer(keys)` gives the new reads' other form."""
    keys = sorted(set(s.toks[:, at].tolist()))
    own = torch.tensor([keys.index(k) for k in s.toks[:, at].tolist()])
    k2 = torch.tensor(keys)[shifted(own, len(keys), seed)]
    bad = s.toks.clone()
    bad[:, at] = k2
    return EG.Pair(replace(s, alt=answer(k2)), bad, "gap")


@dataclass
class Pairs:
    J: Any  # job.Job
    seed: int
    cache: dict[str, EG.Pair] = field(default_factory=dict)
    site_cache: dict[str, L.Site] = field(default_factory=dict)

    def site(self, key: str) -> L.Site:
        if key not in self.site_cache:
            self.site_cache[key] = sites.site(self.J.F, key)
        return self.site_cache[key]

    def __getitem__(self, name: str) -> EG.Pair:
        if name not in self.cache:
            self.cache[name] = self.build(name)
        return self.cache[name]

    def gain(self, task: str) -> EG.Pair:
        return EG.Pair(self.site(sites.TASKS[task].site), None, "lo")

    def build(self, name: str) -> EG.Pair:
        F = self.J.F
        if name == "copy":
            s, v = self.site("e62_copy"), torch.load(F.checked(sites.CUE / "large" / "windows.pt"), weights_only=False)
            p, b = v["p_read"], v["target"]["copy"]["read"]
            assert torch.equal(s.tgt, b), "E62's copy reads are not B"
            bad = absent_b(s.toks, p, b, b.unique().tolist(), self.seed)
            return EG.Pair(replace(s, alt=bad[torch.arange(len(bad)), p + 1]), bad, "gap")
        if name == "lookup":
            s = self.site("e62_lookup")
            at = int(s.pos[0]) - 1  # the second A
            words = dict(zip(s.toks[:, at].tolist(), s.tgt.tolist(), strict=True))
            assert all(words[a] == w for a, w in zip(s.toks[:, at].tolist(), s.tgt.tolist(), strict=True)), "a key has two words"
            return swap_key(s, at, lambda k2: torch.tensor([words[k] for k in k2.tolist()]), self.seed)
        if name == "lowercase":
            s = self.site("e53_lowercase")
            assert torch.equal(s.toks[s.win, s.pos + 1], s.tgt), "a lowercase read's next token is not its scored form"
            bad = s.toks.clone()
            bad[s.win, s.pos + 1] = s.alt  # type: ignore[index]
            return EG.Pair(s, bad, "gap")
        if name == "cipher":
            s, plain = self.site("e53_cipher"), pile_rows("val", 0, sites.E53_ROWS)
            assert plain.shape == s.toks.shape and torch.equal(plain[s.win, s.pos + 1], s.alt), "the plain rows are not the cipher's source"  # type: ignore[arg-type]
            return EG.Pair(s, plain, "gap")
        s, w = sites.kv(F, sites.KV[name])
        return swap_key(s, int(s.pos[0]), lambda k2: w["table"][k2], self.seed)
