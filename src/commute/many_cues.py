"""Many cues, one answer (E33, `experiments/many_cues`): mining the task and the per-cue bookkeeping.

Plan: reports/plans/2026-10-22_plan_many-cues.md. A pair (trigger t, target v) *qualifies* when v is word-like, v != t, the
original model's mean next-token probability of v at t's occurrences (its baseline) lies in rungs 0 to 2 of the task
ladder ([1e-3, 1)), and that baseline is at least OVER_BG times v's background probability over every position (t is a
cue for v, not a word after which v is merely common). The target is the word-like token with the most qualifying
triggers in the pool. The many-answers null gives each chosen trigger its own other qualifying target, matched in
baseline.

Per-cue reads on the training windows: d(x) = g1(x) - g0(x), the change of Gamma's importance values at the trigger
(alive subcomponents, every matrix concatenated). s_k = mean of d over cue k's windows (the per-cue fixed change).
`overlap` = mean cosine of s_k between different cues over the mean cosine between the two halves of one cue's windows:
1 when every cue moves the same way, 0 when the cues' changes are unrelated.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from commute.data import Windows
from commute.ladder import RUNG_EDGES, TriggerStats
from commute.regating import div
from commute.shift_anatomy import Vals, flat

LO, HI = RUNG_EDGES[2], 1.0  # rungs 0 to 2: the original model already gives the target some probability
OVER_BG = 5.0  # a cue: its baseline is at least this multiple of the target's background probability


def qualifying(stats: TriggerStats, pool: list[int], plain: Tensor, over_bg: float = OVER_BG) -> Tensor:
    """(len(pool), V) bool: which (trigger, target) pairs qualify (module docstring)."""
    assert stats.background is not None
    P = torch.stack([stats.mean_p[t].double() for t in pool])
    bg = stats.background.double()
    Q = plain[None, :] & (P >= LO) & (P < HI) & (P >= over_bg * bg[None, :])
    Q[torch.arange(len(pool)), torch.tensor(pool)] = False
    return Q


def pick_target(Q: Tensor) -> int:
    """The target with the most qualifying triggers (the lowest id on a tie)."""
    n = Q.sum(0)
    return int(torch.nonzero(n == n.max())[0])


def alternative(p: Tensor, q: Tensor, target: int, used: set[int]) -> int | None:
    """For one trigger (its (V,) baseline row `p` and qualifying row `q`): the qualifying target other than `target`
    and not in `used` whose baseline is closest in log10 to the baseline of `target`; None if there is none."""
    ok = q.clone()
    ok[target] = False
    if used:
        ok[torch.tensor(sorted(used))] = False
    idx = torch.nonzero(ok).flatten()
    if idx.numel() == 0:
        return None
    dist = (p[idx].double().log10() - p[target].double().log10()).abs()
    return int(idx[torch.argmin(dist)])


def concat(ws: list[Windows]) -> Windows:
    return Windows(torch.cat([w.tokens for w in ws]), torch.cat([w.g for w in ws]), torch.cat([w.row for w in ws]),
                   torch.cat([w.pos for w in ws]), ws[0].half)


def table(pairs: dict[int, int], vocab: int) -> Tensor:
    """(vocab,) int64 trigger -> target lookup (`data.target_tokens`); entries off the triggers are 0 and never read."""
    t = torch.zeros(vocab, dtype=torch.long)
    t[torch.tensor(list(pairs))] = torch.tensor(list(pairs.values()))
    return t


def overlap(d: Vals, labels: Tensor, alive: dict[str, Tensor]) -> dict[str, float]:
    """Split-half cosines of the per-cue change: each cue's training windows (in index order) split into even and odd
    halves A_k, B_k; within = mean_k cos(A_k, B_k), across = mean over k != l of cos(A_k, B_l), overlap = across /
    within. With one cue, across and overlap are nan."""
    D = flat(d, alive).double()
    halves = []
    for lab in labels.unique():
        idx = torch.nonzero(labels == lab).flatten()
        assert idx.numel() >= 2, f"cue {int(lab)} has {idx.numel()} training window(s): no split half"
        halves.append((D[idx[0::2]].mean(0), D[idx[1::2]].mean(0)))
    A = torch.nn.functional.normalize(torch.stack([a for a, _ in halves]), dim=1)
    B = torch.nn.functional.normalize(torch.stack([b for _, b in halves]), dim=1)
    C, k = A @ B.T, len(halves)
    within = float(C.diag().mean())
    across = float((C.sum() - C.diag().sum()) / (k * (k - 1))) if k > 1 else float("nan")
    return {"cos_within": within, "cos_across": across, "overlap": div(across, within), "n_cues": k}


def donors(labels: Tensor, same: bool, seed: int) -> Tensor:
    """For each window i, a random window j != i with the same label (`same`) or a different one: the index of the
    window whose values window i is given."""
    gen = torch.Generator().manual_seed(seed)
    out = torch.empty(labels.numel(), dtype=torch.long)
    for i in range(labels.numel()):
        pool = torch.nonzero((labels == labels[i]) == same).flatten()
        pool = pool[pool != i]
        assert pool.numel(), f"window {i} has no donor (same={same})"
        out[i] = pool[torch.randint(pool.numel(), (1,), generator=gen)]
    return out


def counts(labels: Tensor) -> dict[str, Any]:
    c = labels.unique(return_counts=True)[1]
    return {"min": int(c.min()), "median": float(c.float().median()), "max": int(c.max()), "n_cues": int(c.numel())}


def reading(R_cue: float, R_shared: float, works: float, fails: float, gap: float) -> str:
    """One state's reading of the per-cue against the shared fixed change (the plan's rule)."""
    if R_cue < fails:
        return "neither"
    if R_cue >= works and R_cue - R_shared >= gap:
        return "per cue only"
    if R_shared >= works and R_cue - R_shared < gap:
        return "shared"
    return "partial"


READINGS = {("shared", "per cue only"): "follows the answer", ("per cue only", "per cue only"): "follows the cue",
            ("shared", "shared"): "follows the kind of task"}


def combined(one: str, many: str) -> str:
    """The question's reading from the one-answer and the many-answers states' readings."""
    if "neither" in (one, many):
        return "the per-cue change fails on a joint fine-tune"
    return READINGS.get((one, many), "mixed")
