"""E189 (neuron_jobs, cell P2.29): the pure parts of the filed readings, results/predictions.json -> definitions. No model.
The pair call and the floor are E155's (`neuron_overlap/rules.py`), imported by run.py unchanged; this file adds:
  keep_top   one block's firing kept on its k neurons of largest |firing| at each window and position, the rest zeroed
             (Goodfire's top-K, taken from the firing itself, so in a forward it is sequential block by block);
  retention  per neuron, |its top Pile windows at W0 and at the state shared| / n_top;
  kept       mean(M1 - M0) / mean(T1 - T0), nan when the denominator is <= 0;
  tally      the four filed predictions' counts, from stage 2's score.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

SIDES = ("reuse", "new")


def keep_top(x: Tensor, k: int) -> Tensor:
    """x (..., width): each row keeps its k entries of largest |x| (signed values kept), every other entry 0."""
    idx = x.abs().topk(k, dim=-1).indices
    return torch.zeros_like(x).scatter(-1, idx, x.gather(-1, idx))


def retention(top0: Tensor, top1: Tensor, flat: Tensor) -> Tensor:
    """top0, top1 (n_top, n_neurons): each neuron's top window ids (distinct per column) at W0 and at the state. Per neuron
    in `flat`, the share of its W0 windows that are also among its state windows; float64."""
    a, b = top0[:, flat], top1[:, flat]
    return (a[:, None, :] == b[None, :, :]).any(dim=1).sum(dim=0).double() / top0.shape[0]


def kept(num: float, gain: float) -> float:
    return num / gain if gain > 0 else math.nan


def _mean(v: list[float]) -> float:
    return statistics.fmean(v) if v and not any(math.isnan(x) for x in v) else math.nan


def tally(score: Mapping[str, Any], th: int = 14, no_call_min: int = 6) -> dict[str, Any]:
    """P1-P4 of predictions.json, each a count over the pairs' sides (or pairs) against its filed bar."""
    sides = [(p, s) for p in score for s in SIDES]
    p1 = sum(score[p]["retention"]["floor"][s]["above"] < 0 for p, s in sides)  # nan < 0 is False
    p2 = sum(_mean(score[p]["kept"]["values"][s]) >= 0.5 for p, s in sides)
    p3 = sum(abs(_mean(score[p]["kept_random"]["values"][s])) < 0.2 for p, s in sides)
    p4 = sum(score[p]["retention"]["call"] == "no call" for p in score)
    return {"P1": {"count": p1, "of": len(sides), "holds": p1 >= th}, "P2": {"count": p2, "of": len(sides), "holds": p2 >= th},
            "P3": {"count": p3, "of": len(sides), "holds": p3 >= th}, "P4": {"count": p4, "of": len(score), "holds": p4 >= no_call_min}}  # fmt: skip
