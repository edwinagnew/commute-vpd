"""E113's pure pieces (plan reports/plans/2026-11-06_plan_e113-case-patch.md; rules filed in
experiments/case_patch/results/predictions.json): the class means, the patch that sets one coordinate of the residual,
the random control's shift of the same size, the edit that applies them at given positions, and the flip rate. No model
is read here; tests/test_case_patch.py runs each piece on tiny tensors.

Case class y = 1 for the capitalised member of a pair, 0 for the lowercase (E112's `case_label.real_y`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


def class_means(x: Tensor, y: Tensor) -> tuple[float, float]:
    """(mean of x over y == 0, mean over y == 1), float64. A class with no rows has no mean: assert. A class whose x is
    constant (zero variance) has that value as its mean; nothing else is special."""
    assert bool((y == 0).any()) and bool((y == 1).any()), "both classes need at least one row"
    x = x.double()
    return float(x[y == 0].mean()), float(x[y == 1].mean())


def targets(y: Tensor, means: tuple[float, float], scale: float) -> Tensor:
    """(n,) float64: per row, the opposite class's mean pushed past it by (scale - 1) x (mu_opp - mu_own). Scale 1 is the
    filed patch (mu_opp); scale 2 is mu_opp + (mu_opp - mu_own)."""
    m0, m1 = means
    opp = torch.where(y == 1, m0, m1).double()
    own = torch.where(y == 1, m1, m0).double()
    return opp + (scale - 1) * (opp - own)


def set_coord(h: Tensor, u: Tensor, target: Tensor) -> Tensor:
    """h + (target - h.u) u for rows h (n, d), a unit direction u (d,) and targets (n,): h.u becomes the target and the
    part of h at right angles to u is unchanged."""
    return h + (target.to(h.dtype) - h @ u)[:, None] * u


def shift_along(h: Tensor, u: Tensor, target: Tensor, r: Tensor) -> Tensor:
    """The random control: h + |target - h.u| r for a unit direction r (d,), the step set_coord would take along u, taken
    along r instead."""
    return h + (target.to(h.dtype) - h @ u).abs()[:, None] * r


@dataclass(frozen=True)
class Site:
    """One position's patch: the unit direction u (d,), the per-window targets (N,) and, for the control, a unit random
    direction r (d,) to step along instead of u."""

    pos: int
    u: Tensor
    target: Tensor
    r: Tensor | None = None


def edit_fn(sites: Sequence[Site], rows: slice) -> Callable[[Tensor], Tensor]:
    """The edit of a residual x (B, T, d) for the windows `rows` of the batch: each site's position patched (set_coord, or
    shift_along when it has r) with its targets for those rows; every other position and row is unchanged."""

    def f(x: Tensor) -> Tensor:
        y = x.clone()
        for s in sites:
            h, u, tg = x[:, s.pos], s.u.to(x.device, x.dtype), s.target[rows].to(x.device)
            y[:, s.pos] = set_coord(h, u, tg) if s.r is None else shift_along(h, u, tg, s.r.to(x.device, x.dtype))
        return y

    return f


def flips(d0: Tensor, d1: Tensor) -> Tensor:
    """(n,) bool: the sign of d = logit A - logit B changes from d0 (unpatched) to d1 (patched). A zero on either side (a
    tie in the label margin) is no flip."""
    return ((d0 > 0) & (d1 < 0)) | ((d0 < 0) & (d1 > 0))


def flip_rate(d0: Tensor, d1: Tensor) -> float:
    """The share of windows whose label flips; nan with no windows."""
    return float(flips(d0, d1).double().mean()) if len(d0) else math.nan
