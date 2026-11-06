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
    lam_frac x the mean diagonal of x's covariance. A y of (n, m) fits m columns at once: w (d, m), b (m,) (E116)."""
    x, y = x.double(), y.double()
    mx, my = x.mean(0), y.mean(0)
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
    return table_accuracy(reading[::2], answer[::2], reading[1::2], answer[1::2])


def table_accuracy(fit_r: np.ndarray, fit_a: np.ndarray, read_r: np.ndarray, read_a: np.ndarray) -> dict[str, float]:
    """`lookup_accuracy`'s table with the fit and read rows given apart (E116's P2): the most common answer per reading on
    the fit rows (ties to the lowest token), read on the read rows; an unseen reading counts as wrong."""
    if not len(fit_r) or not len(read_r):
        return {"acc": math.nan, "floor": math.nan, "unseen": 0.0}

    def mode(a: np.ndarray) -> int:
        vals, n = np.unique(a, return_counts=True)
        return int(vals[n.argmax()])  # argmax takes the first maximum, vals ascend

    table = {float(v): mode(fit_a[fit_r == v]) for v in np.unique(fit_r)}
    got = np.array([table.get(float(v), -1) for v in read_r])
    return {"acc": float((got == read_a).mean()), "floor": float((read_a == mode(fit_a)).mean()),
            "unseen": float((got == -1).mean())}  # fmt: skip


# ---- E70 Amendment 1 (rules filed in experiments/pcfg_anchors/results/predictions_e70_a1.json) ----


def pattern_basis(x: Tensor, y: Tensor) -> Tensor:
    """The unit cross-covariance of rows x (n, d) with a scalar y (n,), as a basis (d, 1), fp32: the direction y moves x
    along, against `fit_ridge`'s weights, which also cancel noise. Removing it zeroes x's sample covariance with y."""
    x, y = x.double(), y.double()
    c = (x - x.mean(0)).T @ (y - y.mean()) / len(x)
    assert c.norm() > 0, "x has no covariance with y: there is no direction to remove"
    return unit_basis(c)


def erased(after: float, floor: float, before: float) -> bool:
    """R3: a refit's accuracy after the removal lies below the midpoint of the floor and the reading before it (strict;
    a tie or a nan is not erased)."""
    return after < (floor + before) / 2


def one_direction_branch(teeth: Sequence[bool], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The tree, first match wins, reading R2's calls (`r2_block`, blocks 1..n in order) only at blocks where LOW's control
    has teeth. 'closed': teeth at no block. 'cannot serve': a reversal at a read block. 'separates': R2 holds at a read
    block before the last. 'no call': a read block before the last is unreadable, or every read block is. Else
    'cannot serve': read and neither, or R2 holds at the last block alone, the output side, where E70 files no separation."""
    read_at = [i + 1 for i, t in enumerate(teeth) if t]
    calls = [rows[b - 1]["call"] for b in read_at] if read_at else []
    early = [rows[b - 1]["call"] for b in read_at if b < len(teeth)]
    if not read_at:
        branch = "closed"
    elif "reversal" in calls:
        branch = "cannot serve"
    elif "holds" in early:
        branch = "separates"
    elif "no call" in early or all(c == "no call" for c in calls):
        branch = "no call"
    else:
        branch = "cannot serve"
    return {"branch": branch, "read_at": read_at, "calls_read": calls}


# ---- E70 Amendment 2 (rules filed in experiments/pcfg_anchors/results/predictions_e70_a2.json) ----


def top_pc_after(x: Tensor, u: Tensor) -> Tensor:
    """The top principal component of rows x (n, d) after removing the basis u (d, r), as a basis (d, 1), fp32. It lies at
    right angles to u; u arrives in fp32 and is renormalised here so that only rounding is left along it."""
    x, u = x.double(), u.double()
    xc = remove(x - x.mean(0), u / u.norm(dim=0))
    vals, vecs = torch.linalg.eigh(xc.T @ xc / len(x))
    assert vals[-1] > 0, "nothing is left after the removal: there is no component to take"
    return vecs[:, -1:].float()


def variance_share(h: Tensor, u: Tensor) -> float:
    """var(h u) / trace Cov(h), for rows h (n, d) and one unit direction u (d, 1)."""
    return float((h @ u).var() / h.var(0).sum())


def abs_cov(h: Tensor, u: Tensor, y: Tensor) -> float:
    """|Cov(h u, y)| for rows h (n, d), one direction u (d, 1) and a scalar label y (n,), in float64."""
    x = (h.double() @ u.double())[:, 0]
    return float(((x - x.mean()) * (y.double() - y.double().mean())).mean().abs())


def class_share(x: np.ndarray, labels: np.ndarray) -> float:
    """The share of a scalar x's variance lying between the label's classes (the correlation ratio squared), which sees a
    curved tie that a covariance misses. nan when x is empty or constant."""
    total = float(((x - x.mean()) ** 2).sum()) if len(x) else 0.0
    if not total > 0:
        return math.nan
    return sum(float((labels == v).sum()) * float(x[labels == v].mean() - x.mean()) ** 2 for v in np.unique(labels)) / total


def refit_accuracy(fit_h: Tensor, y: Tensor, read_h: Tensor, c: np.ndarray, u: Tensor, lam_frac: float) -> float:
    """A ridge refit on rows fit_h with u removed, read as a rounded prediction on rows read_h with u removed, against c."""
    w, b = fit_ridge(remove(fit_h, u), y, lam_frac)
    return float((ridge_reading(remove(read_h, u)[None], w[None], b[None])[0] == c).mean())


def above_midpoint(after: float, floor: float, before: float) -> bool:
    """R3': a refit's accuracy after the removal stays above the midpoint of the floor and the reading before it (strict;
    a tie or a nan is a miss). Not `not erased(...)`, which would pass both."""
    return after > (floor + before) / 2


def same_reading(acc: Sequence[float], ref: Sequence[float], floor: float, ref_floor: float, n: int) -> bool:
    """R0(a): accuracies per block within one row of n of the reference (inclusive, counted in rows), and the same floor."""
    return floor == ref_floor and all(abs(x - y) * n <= 1 + 1e-9 for x, y in zip(acc, ref, strict=True))


def reproduced(diff: Sequence[float], band: Sequence[float], frac: float) -> bool:
    """R0(b): per block, the largest difference from the reference's same draws is at most frac x its band (a nan misses)."""
    return all(d <= frac * b for d, b in zip(diff, band, strict=True))


def no_count_branch(rows: Sequence[dict[str, Any]], p_a: Sequence[float], p_v: Sequence[float], class_v: Sequence[float],
                    read_at: Sequence[int], carries: float) -> dict[str, Any]:  # fmt: skip
    """The tree, reading R2 (`r2_block`, blocks 1..n in order) under the largest no-count direction v at 1-based blocks
    read_at. Per block: 'size' (R2 holds, v's class share under `carries`), 'carrier' (R2 holds, v's class share at or over
    it, or nan), 'specific' (R2 neither or reversal, and P's mean share under the covariance direction exceeds its mean share
    under v by more than the band, strictly), 'no call' (R2 unreadable), else 'unresolved'. First match wins over the labels:
    'mixed' (size and specific), 'size explains', 'count-specific at this size', 'carries the classes', 'no call',
    'unresolved'."""

    def label(b: int) -> str:
        r, i = rows[b - 1], b - 1
        if r["call"] == "no call":
            return "no call"
        if r["call"] == "holds":
            return "size" if class_v[i] < carries else "carrier"
        return "specific" if p_a[i] - p_v[i] > r["band"] else "unresolved"

    by_block = {b: label(b) for b in read_at}
    got = set(by_block.values())
    if {"size", "specific"} <= got:
        branch = "mixed"
    elif "size" in got:
        branch = "size explains"
    elif "specific" in got:
        branch = "count-specific at this size"
    elif "carrier" in got:
        branch = "carries the classes"
    elif "no call" in got:
        branch = "no call"
    else:
        branch = "unresolved"
    return {"branch": branch, "by_block": by_block}
