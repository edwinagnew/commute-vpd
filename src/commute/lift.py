"""Lifts of a weight update onto the alive subcomponents (theory/commuting_updates.md, section 5).

Theory convention: per matrix, U_A (d_out, n) holds the alive write vectors as columns and V_A (d_in, n) the read
vectors, so the component sum is U_A V_A^T. param-decomp stores U as (C, d_out) and V as (d_in, C); `alive_uv`
converts.

With full SVD bases U_A = P diag(s) ..., V_A = Q diag(t) ..., the differential
d sigma(dU, dV) = dU V_A^T + U_A dV^T has Gram operator K(X) = U_A U_A^T X + X V_A V_A^T, diagonal in the
(P, Q) basis with eigenvalues s_i^2 + t_j^2 (s, t zero-padded to d_out, d_in). A singular value below
tol * max is treated as zero, so `tol` sets which directions count as expressible.
- Obstruction O(M) = P_U^perp M P_V^perp: the entries (i, j) with s_i = t_j = 0 (Corollary 1).
- Euclidean (minimum-norm) lift of M: X = K^+ M, dU = X V_A, dV = X^T U_A (section 5.1, instance 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class Basis:
    P: Tensor  # (d_out, d_out) left singular vectors of U_A, then a basis of the complement
    s: Tensor  # (d_out,) singular values, zero-padded, truncated below tol * max
    Q: Tensor  # (d_in, d_in)
    t: Tensor  # (d_in,)
    r_u: int
    r_v: int


def alive_uv(U: Tensor, V: Tensor, alive: Tensor) -> tuple[Tensor, Tensor]:
    """param-decomp (U (C, d_out), V (d_in, C)) -> theory (U_A (d_out, n), V_A (d_in, n)), float64."""
    return U[alive].T.double(), V[:, alive].double()


def full_svd(A: Tensor) -> tuple[Tensor, Tensor]:
    """(P (d, d), singular values zero-padded to d) of A (d, n); the expensive step, shared across tolerances."""
    d = A.shape[0]
    P, sv, _ = torch.linalg.svd(A, full_matrices=True)
    s = torch.zeros(d, dtype=A.dtype, device=A.device)
    s[: sv.numel()] = sv
    return P, s


def _truncate(s: Tensor, tol: float) -> tuple[Tensor, int]:
    keep = s > tol * s.max() if s.max() > 0 else torch.zeros_like(s, dtype=torch.bool)
    return torch.where(keep, s, torch.zeros_like(s)), int(keep.sum())


def basis_from(svd_u: tuple[Tensor, Tensor], svd_v: tuple[Tensor, Tensor], tol: float) -> Basis:
    s, r_u = _truncate(svd_u[1], tol)
    t, r_v = _truncate(svd_v[1], tol)
    return Basis(svd_u[0], s, svd_v[0], t, r_u, r_v)


def basis(U_A: Tensor, V_A: Tensor, tol: float) -> Basis:
    return basis_from(full_svd(U_A), full_svd(V_A), tol)


def to_basis(M: Tensor, b: Basis) -> Tensor:
    return b.P.T @ M.double() @ b.Q


def obstruction(M: Tensor, b: Basis) -> Tensor:
    """O(M) in the original coordinates."""
    Mt = to_basis(M, b)
    out = (b.s == 0)[:, None] & (b.t == 0)[None, :]
    return b.P @ (Mt * out) @ b.Q.T


def obstruction_fraction(M: Tensor, b: Basis) -> float:
    Mt = to_basis(M, b)
    out = (b.s == 0)[:, None] & (b.t == 0)[None, :]
    return float((Mt[out] ** 2).sum() / (Mt**2).sum())


def gaussian_null(b: Basis, d_out: int, d_in: int) -> float:
    """E ||O(G)||^2 / ||G||^2 for G with iid isotropic entries (Corollary 2), exact in expectation."""
    return (1 - b.r_u / d_out) * (1 - b.r_v / d_in)


def dsigma_smin(b: Basis) -> float:
    """Smallest non-zero singular value of d sigma: min over (i, j) of sqrt(s_i^2 + t_j^2) where it is non-zero."""
    s_nz, t_nz = b.s[b.s > 0], b.t[b.t > 0]
    s_has0, t_has0 = bool((b.s == 0).any()), bool((b.t == 0).any())
    cands = []
    if t_has0 and s_nz.numel():
        cands.append(s_nz.min())
    if s_has0 and t_nz.numel():
        cands.append(t_nz.min())
    if s_nz.numel() and t_nz.numel():
        cands.append(torch.sqrt(s_nz.min() ** 2 + t_nz.min() ** 2))
    return float(min(cands)) if cands else 0.0


def euclidean_lift(M: Tensor, b: Basis, U_A: Tensor, V_A: Tensor) -> tuple[Tensor, Tensor]:
    """Minimum-norm (dU, dV) with dU V_A^T + U_A dV^T = M - O(M) (exact up to the tol truncation)."""
    Mt = to_basis(M, b)
    den = b.s[:, None] ** 2 + b.t[None, :] ** 2
    Xt = torch.where(den > 0, Mt / torch.where(den > 0, den, torch.ones_like(den)), torch.zeros_like(Mt))
    X = b.P @ Xt @ b.Q.T
    return X @ V_A, X.T @ U_A


def dsigma(dU: Tensor, dV: Tensor, U_A: Tensor, V_A: Tensor) -> Tensor:
    return dU @ V_A.T + U_A @ dV.T


def weighted_fractions(M: Tensor, b: Basis, Sigma: Tensor) -> dict[str, float]:
    """Input-weighted obstruction: E||O(M) a||^2 / E||M a||^2 with E[a a^T] = Sigma (uncentred second moment),
    its Gaussian null (1 - r_U/d_out) tr(P_V^perp Sigma) / tr(Sigma), and the share of input energy outside
    col V_A, tr(P_V^perp Sigma) / tr(Sigma). With Sigma = I these reduce to omega and the isotropic null."""
    M, S = M.double(), Sigma.double()
    O = obstruction(M, b)
    Qo = b.Q[:, b.t == 0]
    read_out = float(torch.trace(Qo.T @ S @ Qo) / torch.trace(S))
    den = float(((M @ S) * M).sum())
    return {"omega_act": float(((O @ S) * O).sum()) / den if den > 0 else 0.0,
            "null_act": (1 - b.r_u / M.shape[0]) * read_out, "read_out": read_out}


def lifted_with_delta_change(M: Tensor, b: Basis, U_A: Tensor, V_A: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(U_A + dU, V_A + dV, Delta' - Delta_0) for the Euclidean lift of M, with the target weights moved by M:
    Delta' - Delta_0 = M - [(U_A + dU)(V_A + dV)^T - U_A V_A^T] = O(M) - dU dV^T."""
    dU, dV = euclidean_lift(M, b, U_A, V_A)
    return U_A + dU, V_A + dV, obstruction(M, b) - dU @ dV.T


# ----------------------------------------------------------------------------- the lift as a pull-back-shaped state
# Added 2026-10-19 (lora_commute). `pullback.train` returns states {k: (delta-W, U_A', V_A')}; the Euclidean lift of a
# finished delta-W is written in the same layout so one evaluation path serves both maps.


def bases(uv: dict[str, tuple[Tensor, Tensor]], tol: float = 1e-6) -> dict[str, Basis]:
    """SVD bases of the alive columns per matrix (`uv` in theory layout, any float dtype; computed in float64)."""
    return {k: basis(u.double(), v.double(), tol) for k, (u, v) in uv.items()}


def lift_state(dW: dict[str, Tensor], uv: dict[str, tuple[Tensor, Tensor]], b: dict[str, Basis]
               ) -> dict[str, tuple[Tensor, Tensor, Tensor]]:
    """For every matrix whose delta-W is non-zero: (delta-W, U_A + dU, V_A + dV) from its Euclidean lift, float32 on
    CPU. The target weights move by delta-W, so Delta moves by O(delta-W) - dU dV^T (`lifted_with_delta_change`)."""
    out = {}
    for k, M in dW.items():
        if not M.any():
            continue
        U_A, V_A = uv[k][0].double(), uv[k][1].double()
        U2, V2, _ = lifted_with_delta_change(M.to(U_A.device), b[k], U_A, V_A)
        out[k] = (M.float().cpu(), U2.float().cpu(), V2.float().cpu())
    return out


@torch.no_grad()
def input_second_moments(model: Any, batches: list[Tensor], keys: list[str]) -> dict[str, Tensor]:
    """E[a a^T] of each named decomposed matrix's input over every position of the batches (float64, on the model's
    device). The weight for `weighted_fractions`."""
    from commute.finetune import logits_of
    from commute.vpd import target_linear

    acc: dict[str, Tensor] = {}
    n = 0
    handles = []
    for k in keys:
        def hook(_m: Any, a: tuple[Any, ...], _o: Any, k: str = k) -> None:
            x = a[0].reshape(-1, a[0].shape[-1]).double()
            acc[k] = acc.get(k, 0) + x.T @ x
        handles.append(target_linear(model, k).register_forward_hook(hook))
    device = next(model.parameters()).device
    try:
        for b in batches:
            logits_of(model.target_model, b.to(device))
            n += b.numel()
    finally:
        for h in handles:
            h.remove()
    return {k: v / n for k, v in acc.items()}
