"""E70's pure pieces (plan reports/plans/2026-11-03_plan_e70-existing-new-counting.md; rules filed in
experiments/pcfg_anchors/results/predictions_e70.json): the ridge probe, the bases to remove, the removal, the controls'
bands, the filed rules R1 and R2, and g o f's lookup. No model is read here; tests/test_removal.py runs each rule through
its edge cases.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor


def remove(h: Tensor, u: Tensor) -> Tensor:
    """h - U U^T h for rows h (n, d) and an orthonormal basis u (d, r); rank 1 is the one-direction removal."""
    return h - (h @ u) @ u.T


def fit_ridge(x: Tensor, y: Tensor, lam_frac: float) -> tuple[Tensor, Tensor]:
    """Closed-form ridge of y (n,) on x (n, d), intercept fit, nothing standardised, float64: (w, b). The penalty is
    lam_frac x the mean diagonal of x's covariance."""
    x, y = x.double(), y.double()
    mx, my = x.mean(0), y.mean()
    xc = x - mx
    cov = xc.T @ xc / len(x)
    lam = lam_frac * cov.diagonal().mean()
    w = torch.linalg.solve(cov + lam * torch.eye(x.shape[1], dtype=x.dtype, device=x.device), xc.T @ (y - my) / len(x))
    return w, my - mx @ w


def ridge_reading(resid: Tensor, w: Tensor, b: Tensor) -> np.ndarray:
    """The rounded reading (n_layer, n) of residuals (n_layer, n, d), by weights (n_layer, d) and intercepts (n_layer,)."""
    return torch.round((resid.double() * w[:, None]).sum(-1) + b[:, None]).cpu().numpy()


def unit_basis(w: Tensor) -> Tensor:
    """The one-direction basis (d, 1), fp32, of a weight vector."""
    return (w / w.norm())[:, None].float()


def seen_values(labels: np.ndarray, min_seen: int) -> list[int]:
    """The label values seen at least min_seen times, ascending."""
    return np.flatnonzero(np.bincount(labels) >= min_seen).tolist()


def centred_basis(columns: Tensor) -> Tensor:
    """Orthonormal basis (d, m - 1) of the m columns (d, m), centred by their mean. Needs m >= 2."""
    assert columns.shape[1] >= 2, f"a subspace needs at least two values, got {columns.shape[1]}"
    u, _, _ = torch.linalg.svd(columns - columns.mean(1, keepdim=True), full_matrices=False)
    return u[:, : columns.shape[1] - 1]


def random_basis(d: int, rank: int, key: Sequence[int]) -> Tensor:
    """A seeded random orthonormal basis (d, rank), fp32: QR of a standard normal draw (rank 1: a unit vector)."""
    q, _ = np.linalg.qr(np.random.default_rng(list(key)).normal(size=(d, rank)))
    return torch.from_numpy(q).float()


def majority(labels: np.ndarray) -> float:
    return float(np.bincount(labels).max() / len(labels)) if len(labels) else math.nan


# ---- the bands (one state's or LOW's draws at one block, then the maximum over states) ----


def band_of(values: Sequence[float]) -> float:
    """95th percentile (numpy's linear rule) of |v| over the finite draws; nan if there are none."""
    a = np.abs(np.asarray(values, float))
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 95)) if a.size else math.nan


def max_band(bands: Sequence[float]) -> float:
    """The largest finite band; nan if there are none (a state with no finite draw adds nothing)."""
    a = np.asarray(bands, float)
    a = a[np.isfinite(a)]
    return float(a.max()) if a.size else math.nan


# ---- R1 ----


def r1(drops: dict[str, Sequence[float]], bands: dict[str, Sequence[float]]) -> dict[str, Any]:
    """Per probe and block, teeth = the drop exceeds the band (strict; a rise, a tie or a nan is no teeth). `stop` when
    no probe has teeth at any block."""
    teeth = {p: [bool(d > b) for d, b in zip(drops[p], bands[p], strict=True)] for p in drops}
    return {"teeth": teeth, "stop": not any(any(t) for t in teeth.values())}


# ---- R2 ----


def r2_block(p: Sequence[float], n: Sequence[float], band: float) -> dict[str, Any]:
    """One block and probe. p, n: the shares of the states in R2 on each side, nan for a gain under the floor (left out
    and counted). gap = min P - max N; reverse_gap = min N - max P. call: 'holds' (gap > band, strict), 'reversal'
    (reverse_gap > band, strict), 'neither', or 'no call' (a side with no finite share, or a nan band)."""
    fp, fn = [x for x in p if math.isfinite(x)], [x for x in n if math.isfinite(x)]
    out: dict[str, Any] = {"n_P": len(fp), "n_N": len(fn), "nan_P": len(p) - len(fp), "nan_N": len(n) - len(fn), "band": band,
                           "gap": math.nan, "reverse_gap": math.nan, "call": "no call"}  # fmt: skip
    if fp and fn:
        out |= {"gap": min(fp) - max(fn), "reverse_gap": min(fn) - max(fp)}
    if fp and fn and math.isfinite(band):
        out["call"] = "holds" if out["gap"] > band else "reversal" if out["reverse_gap"] > band else "neither"
    return out


def r2_verdict(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """rows: r2_block's, blocks 1..n in order. 'separates' if R2 holds at any block before the last; else 'no call' if
    such a block is unreadable; else 'output-side, no call' if it holds at the last only; else 'does not separate'.
    `reversal_at` is read at every block and reported beside the verdict, whatever it is."""
    at = lambda call: [i + 1 for i, r in enumerate(rows) if r["call"] == call]
    last = len(rows)
    holds, no_call = at("holds"), at("no call")
    if any(b < last for b in holds):
        verdict = "separates"
    elif any(b < last for b in no_call):
        verdict = "no call"
    elif last in holds:
        verdict = "output-side, no call"
    else:
        verdict = "does not separate"
    return {"verdict": verdict, "holds_at": holds, "reversal_at": at("reversal"), "no_call_at": no_call}


# ---- g o f ----


def lookup_accuracy(reading: np.ndarray, answer: np.ndarray) -> dict[str, float]:
    """A table from reading to answer, its most common answer per reading on the even rows (ties to the lowest token),
    read on the odd rows; a reading the even rows never saw counts as wrong. `floor`: the even rows' most common answer
    read on the odd rows. nan when either half is empty."""
    fit_r, fit_a, read_r, read_a = reading[::2], answer[::2], reading[1::2], answer[1::2]
    if not len(fit_r) or not len(read_r):
        return {"acc": math.nan, "floor": math.nan, "unseen": 0.0}

    def mode(a: np.ndarray) -> int:
        vals, n = np.unique(a, return_counts=True)
        return int(vals[n.argmax()])  # argmax takes the first maximum, vals ascend

    table = {float(v): mode(fit_a[fit_r == v]) for v in np.unique(fit_r)}
    got = np.array([table.get(float(v), -1) for v in read_r])
    return {"acc": float((got == read_a).mean()), "floor": float((read_a == mode(fit_a)).mean()),
            "unseen": float((got == -1).mean())}  # fmt: skip
