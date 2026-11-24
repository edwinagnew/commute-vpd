"""The causal effect of the activation difference lens as the authors' code computes it, torch only, for any model
behind the two-method `Pair` below: the 67M (`adl.NetPair`) and, for the parity check, a Hugging Face Qwen pair
(experiments/adl_published/parity.py, run in the toolkit's own venv, which is why this module imports nothing else of
`commute`). Spec: reports/plans/2026-11-23_plan_p2-67-adl-spec.md, sections 2.8 and 4.2.

Source: diffing-toolkit at 617d9266, src/diffing/methods/activation_difference_lens/causal_effect.py ("CE:L<n>" below).
  nll         CE:L201-209   NLL column i is the target token i + 1
  targets     CE:L328-347   mask_all = attention_mask[:, 1:] (non-chat: every token is an "assistant" token, CE:L110)
  swap        CE:L286-301   base_to_ft, zero_ablate false: the fine-tuned site output's component along v_hat is set to
                            the base model's, at every position, in float32, cast back to the site's dtype
  gaussian    CE:L661-663   torch.randn(hidden) per vector, normalised inside the swap (CE:L275)
  pair_diffs  CE:L458-472   differences of base activations at 2n positions drawn from a pool of valid positions
  Sums        CE:L681-949   float64 sums of NLL under mask_all and at single target columns t = p + 1
  ce          CE:L968-1067  "all" and "exclude_pos" (all minus target column p + 1, i.e. the prediction of token p + 2)
  incr        CE:L1081-1094 incr_ce = CE_swap - CE_ft; incr_rel_ce = incr_ce / (CE_base - CE_ft)

Departures (ours): a zero direction is a no-op (upstream divides by its norm, CE:L275); incr_rel_ce is undefined when
|CE_base - CE_ft| is at or under a filed floor, and always when it is 0 (upstream divides, CE:L1084); sums are float64
(upstream sums each batch in the logits' dtype, bf16 on Qwen, CE:L355-358); the per-batch work is one function over any
`Pair`, and the random directions are passed in rather than drawn inside.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import Tensor

Edit = Callable[[Tensor], Tensor]  # the site's output (B, L, d) -> its replacement


class Pair(Protocol):
    """A base and a fine-tuned model read at one site. Both return (logits (B, L, V), site output (B, L, d))."""

    def base(self, x: Tensor, attn: Tensor) -> tuple[Tensor, Tensor]: ...

    def ft(self, x: Tensor, attn: Tensor, edit: Edit | None = None) -> tuple[Tensor, Tensor]: ...


def nll(logits: Tensor, ids: Tensor) -> Tensor:
    """(B, L-1) per-token NLL; column i scores token i + 1 (CE:L201-209)."""
    lp = torch.log_softmax(logits, dim=-1)[:, :-1]
    return -lp.gather(2, ids[:, 1:].to(lp.device).unsqueeze(-1)).squeeze(-1)


def targets(attn: Tensor) -> Tensor:
    """(B, L-1) bool mask_all of a non-chat batch: target i + 1 is valid (CE:L340-342, assistant mask all ones)."""
    return attn[:, 1:].bool()


def unit(V: Tensor) -> Tensor:
    """Rows of V at norm 1 in float32; a zero row stays zero, so swapping along it changes nothing (ours: upstream
    divides by the norm, CE:L275)."""
    n = V.double().norm(dim=-1, keepdim=True)
    return torch.where(n > 0, V.double() / n.clamp_min(1e-300), torch.zeros_like(V.double())).float()


def swap(v_hat: Tensor, h_src: Tensor) -> Edit:
    """CE:L286-301: h -> h + ((h_src . v_hat) - (h . v_hat)) v_hat at every position, in float32 (v_hat unit or 0)."""
    v = v_hat.float().to(h_src.device).view(1, 1, -1)
    tc = (h_src.float() * v).sum(-1, keepdim=True)

    def f(h: Tensor) -> Tensor:
        hf = h.float()
        return (hf + (tc - (hf * v).sum(-1, keepdim=True)) * v).to(h.dtype)

    return f


def gaussian(n: int, d: int, gen: torch.Generator) -> Tensor:
    """(n, d): CE:L661-663's random vectors, `torch.randn(hidden_size)` each (here from a seeded generator)."""
    return torch.randn(n, d, generator=gen)


def pair_diffs(pool: Tensor, n: int, gen: torch.Generator) -> Tensor:
    """(n, d): CE:L464-470, a permutation of the pool's rows, the first 2n paired as consecutive draws and differenced."""
    a = pool[torch.randperm(len(pool), generator=gen)[: 2 * n]].float().view(n, 2, -1)
    return (a[:, 0] - a[:, 1]).contiguous()


@dataclass
class Sums:
    """CE:L681-949's accumulators. Row 0 is the fine-tuned model, row 1 the base, row 2 + k the swap along direction k.
    `col[r, i]` sums NLL at target column cols[i] = p + 1 (CE:L836-861, only where the column exists, CE:L837)."""

    n_rows: int
    cols: tuple[int, ...]
    all: Tensor = field(init=False)
    col: Tensor = field(init=False)
    n_all: int = 0
    n_col: Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.all = torch.zeros(self.n_rows, dtype=torch.float64)
        self.col = torch.zeros(self.n_rows, len(self.cols), dtype=torch.float64)
        self.n_col = torch.zeros(len(self.cols), dtype=torch.long)

    def count(self, mask: Tensor) -> None:
        self.n_all += int(mask.sum())
        for i, t in enumerate(self.cols):
            if t < mask.shape[1]:
                self.n_col[i] += int(mask[:, t].sum())

    def add(self, r: int, v: Tensor, mask: Tensor) -> None:
        m = mask.to(v.device)
        vm = torch.where(m, v.double(), torch.zeros((), dtype=torch.float64, device=v.device))
        self.all[r] += float(vm.sum())
        for i, t in enumerate(self.cols):
            if t < vm.shape[1]:
                self.col[r, i] += float(vm[:, t].sum())


Score = Callable[[Tensor], tuple[Tensor, Tensor]]  # a batch's logits -> (per-target NLL, bool mask), same shape


def whole_window(x: Tensor, attn: Tensor) -> Score:
    """The toolkit's reading: every next-token target of the batch (`nll`, `targets`)."""
    return lambda lg: (nll(lg, x), targets(attn))


def effect(pair: Pair, batches: Iterable[tuple[Tensor, Tensor, Score]], V: Tensor, positions: Sequence[int]) -> Sums:
    """CE:L769-966 for one eval set: per batch (ids, attention mask, score) the fine-tuned and base NLL and the
    fine-tuned NLL with its site swapped to the base model's along each row of V (pass `unit(V)`). `score` is
    `whole_window` for the toolkit's reading. Columns p + 1 for p in `positions`."""
    s = Sums(2 + len(V), tuple(p + 1 for p in positions))
    for x, attn, sc in batches:
        lg_b, h_b = pair.base(x, attn)
        v_b, mask = sc(lg_b)
        s.count(mask)
        s.add(1, v_b, mask)
        s.add(0, sc(pair.ft(x, attn)[0])[0], mask)
        for k, v in enumerate(V):
            s.add(2 + k, sc(pair.ft(x, attn, swap(v, h_b))[0])[0], mask)
    return s


def ce(s: Sums, r: int, subset: str, p: int) -> float:
    """CE:L968-1067: mean NLL of row r over mask_all ("all") or mask_all without target column p + 1 ("exclude_pos");
    nan on no targets (CE:L599-603)."""
    if subset == "all":
        tot, n = float(s.all[r]), s.n_all
    else:
        i = s.cols.index(p + 1)
        tot, n = float(s.all[r] - s.col[r, i]), s.n_all - int(s.n_col[i])
    return tot / n if n else math.nan


def incr(ce_i: float, ce_ft: float, ce_base: float, floor: float) -> dict[str, float]:
    """CE:L1081-1094 (base_to_ft: the target is the fine-tuned model). incr_rel_ce is nan when |CE_base - CE_ft| <=
    floor, so a zero gap is rejected whatever the floor (ours: upstream divides)."""
    gap = ce_base - ce_ft
    inc = ce_i - ce_ft
    return {"incr_ce": inc, "incr_rel_ce": inc / gap if abs(gap) > floor and gap != 0 else math.nan}


def record(s: Sums, groups: dict[str, range], positions: Sequence[int], floor: float) -> dict[str, Any]:
    """Per subset and position: CE of the fine-tuned and base models, the counts, and per group of directions (rows of
    V, by index range) every direction's incr_ce at that position's mask. A group named "lens" holds delta_bar_j for j in
    `positions`, one per position, read at its own position as the toolkit reads `intervention` (CE:L1045-1152)."""
    out: dict[str, Any] = {"positions": list(positions), "count_all": s.n_all, "count_col": s.n_col.tolist()}
    for sub in ("all", "exclude_pos"):
        o: dict[str, Any] = {"ce_ft": [], "ce_base": [], "count": []}
        for p in positions:
            o["ce_ft"].append(ce(s, 0, sub, p))
            o["ce_base"].append(ce(s, 1, sub, p))
            o["count"].append(s.n_all if sub == "all" else s.n_all - int(s.n_col[s.cols.index(p + 1)]))
        for g, rows in groups.items():
            m = [[incr(ce(s, 2 + k, sub, p), ce(s, 0, sub, p), ce(s, 1, sub, p), floor) for p in positions] for k in rows]
            if g == "lens":
                assert len(rows) == len(positions)
                o["lens"] = {key: [m[j][j][key] for j in range(len(positions))] for key in ("incr_ce", "incr_rel_ce")}
            else:
                o[g] = [[c["incr_ce"] for c in row] for row in m]  # (n_dirs, n_positions)
        out[sub] = o
    return out
