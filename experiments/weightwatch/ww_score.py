"""P2.73's score after Zhong, its nulls and the filed rules (spec "Our score"; predictions.json "rules"). Weights and
moments only: no model.

For monitored matrix m with output moment C_m = E[(W_m x)(W_m x)^T] over the calibration tokens (d = 768) and kept left
vectors u_{m,i} with singular values s_{m,i}:
    a_{m,i} = d u^T C_m u / tr C_m      S = sum s^2 a / sum s^2      (None with no kept vector)
    S_cov   the same with C_m replaced by its covariance, position 0 left out
    S_G     sum_m (sum_i s_{m,i}^2) a_G,m / sum s^2,  a_G,m = d tr(G_m C_m) / (tr G_m tr C_m)   (G_m = E[g g^T] of M's
            LM-loss gradient at m's output): the gradient-geometry diagnostic, not a floor
    Haar    S with every u rotated by a Haar Q_m (s kept): 20 filed draws, shared across states (paired)
A random unit u has E[a] = 1, so the Haar and Gaussian nulls read 1 in expectation whatever C is: they are isotropic in
output direction (predictions.json "blind_to").
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

MONITORED = tuple(f"h.{l}.{m}" for l in range(4) for m in ("attn.o_proj", "mlp.down_proj"))
Kept = dict[str, tuple[Tensor, Tensor]]  # matrix -> (u (d, k), s (k,))


@dataclass
class Moment:
    C: Tensor  # (d, d) float64, E[o o^T]
    cov: Tensor  # covariance over tokens at positions >= 1
    G: Tensor  # E[g g^T]
    pos0_share: float  # share of tr C from position-0 tokens
    mu_share: float  # |mu|^2 / tr C
    lam: Tensor  # eigenvalues of C, ascending
    V: Tensor  # their eigenvectors (columns)
    lam_G: Tensor


def moment(acc: Mapping[str, Any]) -> Moment:
    """From the base-moment pass's sums: C2 = sum o o^T, C2_0 its position-0 part, s and s0 the sums of o, n and n0 the
    token counts, G2 = sum g g^T."""
    n, n0 = int(acc["n"]), int(acc["n0"])
    C = acc["C2"] / n
    m1 = n - n0
    mu1 = (acc["s"] - acc["s0"]) / m1
    cov = (acc["C2"] - acc["C2_0"]) / m1 - torch.outer(mu1, mu1)
    lam, V = torch.linalg.eigh(C)
    mu = acc["s"] / n
    G = acc["G2"] / n
    return Moment(C, cov, G, float(acc["C2_0"].trace() / acc["C2"].trace()), float(mu @ mu / C.trace()), lam, V, torch.linalg.eigvalsh(G))


def a_of(u: Tensor, C: Tensor) -> Tensor:
    u = u.double().to(C.device)
    return C.shape[0] * (u * (C @ u)).sum(0) / C.trace()


def pooled(kept: Kept, mats: Mapping[str, Tensor]) -> float | None:
    num = den = 0.0
    for m, (u, s) in kept.items():
        w = s.double().to(mats[m].device) ** 2
        num += float((w * a_of(u, mats[m])).sum())
        den += float(w.sum())
    return None if den == 0 else num / den


def s_g(kept: Kept, mom: Mapping[str, Moment]) -> float | None:
    num = den = 0.0
    for m, (_, s) in kept.items():
        C, G, w = mom[m].C, mom[m].G, float(s.double().pow(2).sum())
        num += w * C.shape[0] * float((G * C).sum()) / float(G.trace() * C.trace())
        den += w
    return None if den == 0 else num / den


def haar(d: int, seed: int) -> Tensor:
    """A Haar-distributed orthogonal matrix (QR of a Gaussian with R's diagonal signs folded in), float64."""
    q, r = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64))
    return q * torch.sign(torch.diagonal(r))


def rotated(kept: Kept, Q: Mapping[str, Tensor]) -> Kept:
    return {m: (Q[m].to(u.device) @ u.double(), s) for m, (u, s) in kept.items()}


def readings(kept: Kept, mom: Mapping[str, Moment], nulls: Sequence[Mapping[str, Tensor]]) -> dict[str, Any]:
    """Every S reading of one state's kept vectors."""
    C, cov = {m: v.C for m, v in mom.items()}, {m: v.cov for m, v in mom.items()}
    per = {m: {"a": a_of(u, C[m]).tolist(), "s2": float(s.double().pow(2).sum()), "k": int(u.shape[1])} for m, (u, s) in kept.items() if u.shape[1]}
    draws = [pooled(rotated(kept, Q), C) for Q in nulls]
    S = pooled(kept, C)
    return {"S": S, "S_cov": pooled(kept, cov), "S_G": s_g(kept, mom), "haar": draws, "per_matrix": per,
            "haar_below_S": None if S is None else sum(1 for x in draws if x is not None and x < S)}  # fmt: skip


def svals(dW: Tensor) -> Tensor:
    """Singular values from the same call `kept_svd` makes: tau must be measured with it, since svdvals rounds the noise
    values differently (dry run 2026-11-23: the trigger's sigma_2 was 4.0787683e-07 from svdvals, 4.0787685e-07 from
    svd, so it sat above its own tau and was kept)."""
    return torch.linalg.svd(dW.float(), full_matrices=False)[1]


def kept_svd(dW: Tensor, tau: float, k: int) -> tuple[Tensor, Tensor, Tensor]:
    """The published call (fp32, `full_matrices=False`), then D2: the first min(k, #(s > tau)) left vectors. Returns
    (u, s) kept and every singular value."""
    u, s, _ = torch.linalg.svd(dW.float(), full_matrices=False)
    n = min(k, int((s > tau).sum()))
    return u[:, :n], s[:n], s


def beyond(s: Tensor, rank: int) -> float:
    """sigma_{rank + 1}: the largest singular value past a known rank (0 when there is none)."""
    return float(s[rank]) if rank < len(s) else 0.0


# ---- the lambda constructions and x2's filed value -----------------------------------------------------------------


def gamma(n: int) -> float:
    eps = torch.finfo(torch.float64).eps / 2  # unit roundoff
    return n * eps / (1 - n * eps)


def quad_floor(u: Tensor, C: Tensor, lam: float) -> float:
    """The fp64 floor of d u^T C u / tr C against d lam / tr C at this site: the eigensolver's residual plus twice the
    gamma_{2d} rounding bound of the quadratic form (Higham), relative to lam."""
    u = u.double()
    resid = abs(float(u @ C @ u) - lam)
    return (resid + 2 * gamma(2 * C.shape[0]) * float(u.abs() @ C.abs() @ u.abs())) / abs(lam)


def exact_s(dW: Tensor, rank: int, m: str, C: Mapping[str, Tensor]) -> float | None:
    """S of an exact float64 construction of known rank at matrix m (fp64 SVD, its first `rank` vectors)."""
    u, s, _ = torch.linalg.svd(dW.double(), full_matrices=False)
    return pooled({m: (u[:, :rank], s[:rank])}, C)


# ---- filed rules (predictions.json "rules"); each checked on its edge cases in tests/test_weightwatch.py ---------------


def rel_within(x: float | None, ref: float | None, bar: float | None) -> bool | None:
    """|x - ref| / |ref| <= bar; None when any is missing or ref is 0."""
    if x is None or ref is None or bar is None or ref == 0:
        return None
    return abs(x - ref) / abs(ref) <= bar


def spread(draws: Sequence[float | None]) -> float | None:
    got = [x for x in draws if x is not None]
    return max(got) - min(got) if len(got) >= 2 else None


def column_rule(S: Mapping[str, float | None], haar_spread: float | None, trigger_share: float | None, trigger_bar: float, dw0_ok: bool | None) -> dict[str, Any]:
    """redteam-lead, 2026-11-23: S earns a column only if x2's S sits above both new edits' S by more than the Haar
    null's spread (the largest max - min of x2's, new_few's and new_thin's 20 draws), the trigger flags (share at least
    the bar) and the dW = 0 control holds. Otherwise S is reported as a related-work negative."""
    why = [f"S of {k} undefined" for k in ("x2", "new_few", "new_thin") if S.get(k) is None]
    if haar_spread is None:
        why.append("Haar spread undefined")
    gap = None
    if not why:
        gap = float(S["x2"]) - max(float(S["new_few"]), float(S["new_thin"]))  # type: ignore[arg-type]
        if not gap > float(haar_spread):  # type: ignore[arg-type]
            why.append(f"x2's S above the new edits' by {gap:.4g}, not more than the Haar spread {haar_spread:.4g}")
    if trigger_share is None or not trigger_share >= trigger_bar:
        why.append(f"trigger share {trigger_share} under {trigger_bar}")
    if dw0_ok is not True:
        why.append("dW = 0 control not held")
    return {"earned": not why, "gap": gap, "haar_spread": haar_spread, "why_not": why,
            "reading": "S earns a column" if not why else "related-work negative"}  # fmt: skip


def trained_reading(S: float | None, draws: Sequence[float | None], S_G: float | None) -> str:
    """redteam-lead, 2026-11-23: a trained update's S above its Haar null is never read as reuse. It is reported beside
    S_G: "at the level the gradient geometry gives" (S <= S_G) or "above both nulls" (S > S_G)."""
    got = [x for x in draws if x is not None]
    if S is None:
        return "undefined (no kept vector)"
    if not got:
        return "no Haar draws"
    if not S > max(got):
        return "inside or below its Haar null"
    if S_G is None:
        return "above its Haar null; S_G undefined"
    return "at the level the gradient geometry gives" if S <= S_G else "above both nulls"


def gaussian_check(values: Sequence[float | None]) -> dict[str, Any]:
    """The isotropic Gaussian reads 1 in expectation: the mean of its draws within 3 standard errors of 1, the standard
    error measured from the same draws."""
    v = [x for x in values if x is not None]
    if len(v) < 2:
        return {"n": len(v), "mean": None, "se": None, "pass": None}
    mean = sum(v) / len(v)
    se = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1) / len(v))
    return {"n": len(v), "mean": mean, "se": se, "pass": abs(mean - 1) <= 3 * se}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    """The 95% Wilson interval of k of n (None for n = 0)."""
    if n == 0:
        return None
    p = k / n
    c = z * z / n
    mid, half = (p + c / 2) / (1 + c), z * math.sqrt(p * (1 - p) / n + c / (4 * n)) / (1 + c)
    return max(0.0, mid - half), min(1.0, mid + half)
