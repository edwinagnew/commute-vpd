"""Splits of a finished weight update over the alive subcomponents (theory/commuting_updates.md, section 5; called lifts until 2026-10-20).

Theory convention: per matrix, U_A (d_out, n) holds the alive write vectors as columns and V_A (d_in, n) the read
vectors, so the component sum is U_A V_A^T. param-decomp stores U as (C, d_out) and V as (d_in, C); `alive_uv`
converts.

With full SVD bases U_A = P diag(s) ..., V_A = Q diag(t) ..., the differential
d sigma(dU, dV) = dU V_A^T + U_A dV^T has Gram operator K(X) = U_A U_A^T X + X V_A V_A^T, diagonal in the
(P, Q) basis with eigenvalues s_i^2 + t_j^2 (s, t zero-padded to d_out, d_in). A singular value below
tol * max is treated as zero, so `tol` sets which directions count as expressible.
- Obstruction O(M) = P_U^perp M P_V^perp: the entries (i, j) with s_i = t_j = 0 (Corollary 1).
- Least-squares (minimum-norm) split of M: X = K^+ M, dU = X V_A, dV = X^T U_A (section 5.1, instance 1). The adjoint
  split, dU = M V_A, dV = M^T U_A (K applied to M, no solve), is not implemented yet.
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


def least_squares_split(M: Tensor, b: Basis, U_A: Tensor, V_A: Tensor) -> tuple[Tensor, Tensor]:
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


def split_with_delta_change(M: Tensor, b: Basis, U_A: Tensor, V_A: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(U_A + dU, V_A + dV, Delta' - Delta_0) for the least-squares split of M, with the target weights moved by M:
    Delta' - Delta_0 = M - [(U_A + dU)(V_A + dV)^T - U_A V_A^T] = O(M) - dU dV^T."""
    dU, dV = least_squares_split(M, b, U_A, V_A)
    return U_A + dU, V_A + dV, obstruction(M, b) - dU @ dV.T


# ----------------------------------------------------------------------------- the split as a trainer-shaped state
# Added 2026-10-19 (lora_commute). `component_training.train` returns states {k: (delta-W, U_A', V_A')}; the least-squares
# split of a finished delta-W is written in the same layout so one evaluation path serves both.


def bases(uv: dict[str, tuple[Tensor, Tensor]], tol: float = 1e-6) -> dict[str, Basis]:
    """SVD bases of the alive columns per matrix (`uv` in theory layout, any float dtype; computed in float64)."""
    return {k: basis(u.double(), v.double(), tol) for k, (u, v) in uv.items()}


def split_state(dW: dict[str, Tensor], uv: dict[str, tuple[Tensor, Tensor]], b: dict[str, Basis]
               ) -> dict[str, tuple[Tensor, Tensor, Tensor]]:
    """For every matrix whose delta-W is non-zero: (delta-W, U_A + dU, V_A + dV) from its least-squares split, float32 on
    CPU. The target weights move by delta-W, so Delta moves by O(delta-W) - dU dV^T (`split_with_delta_change`)."""
    out = {}
    for k, M in dW.items():
        if not M.any():
            continue
        U_A, V_A = uv[k][0].double(), uv[k][1].double()
        U2, V2, _ = split_with_delta_change(M.to(U_A.device), b[k], U_A, V_A)
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


# ----------------------------------------------------------------------------- which input directions an update reads
# Added 2026-10-19 (lora_matched read-side check): why the full fine-tune's update is more expressible than a LoRA's.


def read_side_overlaps(M: Tensor, b: Basis, Sigma: Tensor) -> dict[str, float]:
    """Unweighted shares of ||M||_F^2 whose read side lies (a) in the alive read span col V_A and (b) in the top-r_v
    eigendirections of the input second moment Sigma; a random read side gives r_v / d_in for both. `alive_in_data`
    is the share of the alive read span inside those top-r_v data directions (tr(P_V P_D) / r_v), and
    `data_in_top` the share of tr(Sigma) they carry."""
    M, S = M.double(), Sigma.double()
    Qv = b.Q[:, b.t > 0]
    evals, evecs = torch.linalg.eigh(S)
    D = evecs[:, evals.argsort(descending=True)[: b.r_v]]
    mm = float((M**2).sum())
    return {"in_alive": float(((M @ Qv) ** 2).sum()) / mm if mm > 0 else 0.0,
            "in_data": float(((M @ D) ** 2).sum()) / mm if mm > 0 else 0.0,
            "random": b.r_v / M.shape[1],
            "alive_in_data": float(((Qv.T @ D) ** 2).sum()) / b.r_v if b.r_v else 1.0,
            "data_in_top": float(evals.sort(descending=True).values[: b.r_v].sum() / evals.sum())}


# ----------------------------------------------------------------------------- the LoRA split study (2026-10-20)
# Added for experiments/lora_split: the outside share as four blocks that sum to one with their exact null, the
# rescaling share (the strictest span), the adjoint split with its data-fitted scale, and the energy of each
# subcomponent's piece of a split.


def read_moments(b: Basis, Sigma: Tensor) -> tuple[Tensor, Tensor]:
    """Split the input second moment Sigma = E[a a^T] into the part of a that is linearly predictable from what the alive
    subcomponents read, a1 = E[a | V_A^T a] = Sigma Q (Q^T Sigma Q)^-1 Q^T a (Q an orthonormal basis of col V_A), and the
    uncorrelated rest a2 = a - a1: (Sigma_1, Sigma_2) with Sigma_1 + Sigma_2 = Sigma and E[a1 a2^T] = 0."""
    S = Sigma.double()
    Q = b.Q[:, b.t > 0]
    if Q.shape[1] == 0:
        return torch.zeros_like(S), S
    SQ = S @ Q
    S1 = SQ @ torch.linalg.pinv(Q.T @ SQ, hermitian=True, rtol=1e-10) @ SQ.T  # the pseudoinverse keeps E[a1 a2^T] = 0 exactly
    return S1, S - S1


def weighted_energy(A: Tensor, S: Tensor) -> float:
    """||A S^1/2||_F^2 = tr(A S A^T)."""
    return float(((A @ S) * A).sum())


def four_blocks(M: Tensor, b: Basis, Sigma: Tensor) -> dict[str, float]:
    """The input-weighted energy of M, E||M a||^2 with E[a a^T] = Sigma, as four shares that sum to one: the write side
    split by P_U (inside col U_A) and P_U^perp, the read side by a = a1 + a2 (`read_moments`). `in_in`: write and read
    inside; `in_out`: write inside only; `out_in`: read inside only; `out_out`: what no first-order change of the alive
    subcomponents reproduces on the data. `null_*`: the same shares for an update with iid isotropic entries, exact in
    expectation: (r_U / d_out or 1 - r_U / d_out) x (tr Sigma_1 / tr Sigma or tr Sigma_2 / tr Sigma)."""
    M = M.double()
    S1, S2 = read_moments(b, Sigma)
    Pu = b.P[:, b.s > 0]
    Mi = Pu @ (Pu.T @ M)
    Mo = M - Mi
    total = weighted_energy(M, S1 + S2)
    w_in, r_in = b.r_u / M.shape[0], float(torch.trace(S1) / torch.trace(S1 + S2))
    out = {"in_in": weighted_energy(Mi, S1), "in_out": weighted_energy(Mi, S2), "out_in": weighted_energy(Mo, S1), "out_out": weighted_energy(Mo, S2)}
    out = {k: v / total for k, v in out.items()}
    out.update({"null_in_in": w_in * r_in, "null_in_out": w_in * (1 - r_in),
                "null_out_in": (1 - w_in) * r_in, "null_out_out": (1 - w_in) * (1 - r_in)})
    return out


def rescaling_alpha(M: Tensor, U_A: Tensor, V_A: Tensor, Sigma: Tensor) -> tuple[Tensor, Tensor]:
    """The rescaling split's coefficients: alpha = argmin ||(sum_c alpha_c u_c v_c^T - M) Sigma^1/2||_F, from the Gram
    (U^T U) o (V^T Sigma V) and the right side u_c^T M Sigma v_c (float64). Returns (alpha, right side)."""
    M, S, U_A, V_A = M.double(), Sigma.double(), U_A.double(), V_A.double()
    G = (U_A.T @ U_A) * (V_A.T @ S @ V_A)
    rhs = torch.diagonal(U_A.T @ M @ S @ V_A)
    return torch.linalg.pinv(G, hermitian=True, rtol=1e-10) @ rhs, rhs


def rescaling_share(M: Tensor, U_A: Tensor, V_A: Tensor, Sigma: Tensor) -> float:
    """Share of M's input-weighted energy inside span{u_c v_c^T}: the weighted least-squares fit of M by sum_c alpha_c u_c
    v_c^T (the rescaling split, directions fixed). The reproduced share R of `rescaling_state`."""
    alpha, rhs = rescaling_alpha(M, U_A, V_A, Sigma)
    return float(alpha @ rhs) / weighted_energy(M.double(), Sigma.double())


def rescaling_state(dW: dict[str, Tensor], uv: dict[str, tuple[Tensor, Tensor]], sigma: dict[str, Tensor]
                    ) -> tuple[dict[str, Tensor], dict[str, tuple[Tensor, Tensor, Tensor]]]:
    """The rescaling split as a state (2026-10-21, split_comparison): per non-zero matrix u_c <- (1 + alpha_c) u_c with
    alpha = `rescaling_alpha`, v_c fixed, so the target moves by sum_c alpha_c u_c v_c^T (not by M) and Delta stays
    fixed. Same return shape as `adjoint_state`: that weight change per matrix (zero where M is) and the state
    (delta-W, U_A', V_A') of the non-zero ones, float32 on CPU."""
    dW_r, state = {}, {}
    for k, M in dW.items():
        if not M.any():
            dW_r[k] = torch.zeros_like(M)
            continue
        U_A, V_A = uv[k][0].double(), uv[k][1].double()
        alpha = rescaling_alpha(M, U_A, V_A, sigma[k])[0]
        dW_r[k] = ((U_A * alpha) @ V_A.T).float().cpu()
        state[k] = (dW_r[k], (U_A * (1 + alpha)).float().cpu(), V_A.float().cpu())
    return dW_r, state


def block_table(dW: dict[str, Tensor], bases: dict[str, Basis], sigma: dict[str, Tensor], non_saturated: list[str]
                ) -> dict[str, Any]:
    """`four_blocks` (with its null) and the input-weighted energy of every non-zero matrix, pooled by energy over all
    matrices and over the non-saturated ones, with outside-both / its null for each pooling (2026-10-21)."""
    per = {k: {**four_blocks(M, bases[k], sigma[k]), "energy": weighted_energy(M.double(), sigma[k].double())}
           for k, M in dW.items() if M.any()}

    def pool(keys: list[str]) -> dict[str, float] | None:
        e = sum(per[k]["energy"] for k in keys)
        if not keys or e <= 0:
            return None
        p = {f: sum(per[k][f] * per[k]["energy"] for k in keys) / e for f in per[keys[0]] if f != "energy"}
        p["out_out_ratio"] = p["out_out"] / p["null_out_out"] if p["null_out_out"] > 0 else float("nan")
        return p

    return {"per_matrix": per, "pooled_all": pool(list(per)), "pooled_ns": pool([k for k in per if k in non_saturated])}


def adjoint_split(M: Tensor, U_A: Tensor, V_A: Tensor) -> tuple[Tensor, Tensor]:
    """dU = M V_A, dV = M^T U_A: the user's pull-back. Its weight change is K(M) = U_A U_A^T M + M V_A V_A^T, not M."""
    M = M.double()
    return M @ V_A, M.T @ U_A


def adjoint_scale(M: Tensor, U_A: Tensor, V_A: Tensor, Sigma: Tensor) -> tuple[float, float]:
    """The scale c that best matches the adjoint split's weight change K(M) to M on the data, c = argmin ||(c K(M) - M)
    Sigma^1/2||_F, and the share of M's weighted energy that c K(M) reproduces (the squared weighted cosine of K(M) and M).
    K depends on the gauge of (U_A, V_A); the released decomposition fixes it."""
    M, S = M.double(), Sigma.double()
    K = dsigma(*adjoint_split(M, U_A, V_A), U_A, V_A)
    kk, mm, km = weighted_energy(K, S), weighted_energy(M, S), float(((K @ S) * M).sum())
    return km / kk, km * km / (kk * mm)


def adjoint_state(dW: dict[str, Tensor], uv: dict[str, tuple[Tensor, Tensor]], sigma: dict[str, Tensor]
                  ) -> tuple[dict[str, Tensor], dict[str, tuple[Tensor, Tensor, Tensor]]]:
    """The scaled adjoint split as a state (2026-10-20, from lora_split's part 2): per non-zero matrix the pieces
    c (M V_A, M^T U_A) with c = `adjoint_scale`, so the target moves by c K(M) (`dsigma` of the pieces), not by M, and
    Delta stays fixed. Returns that weight change per matrix (zero where M is) and the state (delta-W, U_A', V_A') of
    the non-zero ones, float32 on CPU."""
    dW_adj, state = {}, {}
    for k, M in dW.items():
        if not M.any():
            dW_adj[k] = torch.zeros_like(M)
            continue
        U_A, V_A = uv[k][0].double(), uv[k][1].double()
        aU, aV = adjoint_split(M, U_A, V_A)
        c = adjoint_scale(M, U_A, V_A, sigma[k])[0]
        aU, aV = c * aU, c * aV
        dW_adj[k] = dsigma(aU, aV, U_A, V_A).float().cpu()
        state[k] = (dW_adj[k], (U_A + aU).float().cpu(), (V_A + aV).float().cpu())
    return dW_adj, state


def piece_energies(dU: Tensor, dV: Tensor, U_A: Tensor, V_A: Tensor, Sigma: Tensor) -> Tensor:
    """||dB_c Sigma^1/2||_F^2 per alive subcomponent, dB_c = dU_c v_c^T + u_c dV_c^T: how much of a split each
    subcomponent's piece carries on the data."""
    S = Sigma.double()
    dU, dV = dU.double(), dV.double()
    SV, SdV = S @ V_A, S @ dV
    return ((dU * dU).sum(0) * (V_A * SV).sum(0) + 2 * (U_A * dU).sum(0) * (V_A * SdV).sum(0)
            + (U_A * U_A).sum(0) * (dV * SdV).sum(0))


def n_half(energies: Tensor) -> int:
    """How many of the largest pieces carry half of the total."""
    e = energies.sort(descending=True).values
    return int((e.cumsum(0) < e.sum() / 2).sum()) + 1


def random_like(M: Tensor, Sigma: Tensor, seed: int) -> Tensor:
    """An update with iid Gaussian entries and M's input-weighted energy: the null for the measured (not analytic) shares."""
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(M.shape, generator=g, dtype=torch.float64)
    return G * (weighted_energy(M.double(), Sigma) / weighted_energy(G, Sigma)) ** 0.5


# ----------------------------------------------------------------------------- the step split (2026-10-21)
# Added for experiments/step_split (the user's "step split"): the scaled adjoint split applied to every optimizer step of a
# run against the current (U_A, V_A), which then carry forward; each step's remainder, including the second-order term
# dU dV^T, goes to Delta, so the component sum plus Delta is the run's weights at every step. Gamma frozen.


def adjoint_step(S: Tensor, U_A: Tensor, V_A: Tensor, Sigma: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    """One step S of one matrix: pieces dU = c S V_A, dV = c S^T U_A with c = argmin ||(c K(S) - S) Sigma^1/2||_F (as
    `adjoint_scale`), against the bases given, in S's dtype and device. Returns the pieces, the second-order term dU dV^T
    (which the components also take, so Delta books it), and as input-weighted energies: the step's (`step`), the
    first-order change c K(S) (`first`) and the second-order term (`second`), with c and R (the share of S that c K(S)
    reproduces). A step with no first-order image (K(S) = 0, S outside both spans) gets zero pieces: all of it to Delta."""
    A, B = S @ V_A, S.T @ U_A
    K = dsigma(A, B, U_A, V_A)
    KS = K @ Sigma
    kk, km, mm = float((KS * K).sum()), float((KS * S).sum()), weighted_energy(S, Sigma)
    # K(S) at rounding level (S outside both spans) would make c = km / kk a ratio of rounding noise: treat it as zero.
    # The bound is 10 eps times a bound on ||K(S)|| (||U||^2 + ||V||^2) ||S||; a real step's K(S) is ~1/d of that, d <= 3072.
    rounding = 10 * torch.finfo(S.dtype).eps * float(U_A.norm() ** 2 + V_A.norm() ** 2) * float(S.norm())
    if float(K.norm()) <= rounding or mm <= 0:
        z = torch.zeros_like(S)
        return torch.zeros_like(A), torch.zeros_like(B), z, {"c": 0.0, "R": 0.0, "step": mm, "first": 0.0, "second": 0.0}
    c = km / kk
    dU, dV = c * A, c * B
    second = dU @ dV.T
    return dU, dV, second, {"c": c, "R": km * km / (kk * mm), "step": mm, "first": c * c * kk,
                            "second": weighted_energy(second, Sigma)}


class StepSplit:
    """The step split along one run. `step(W)` reads the run's current weights per matrix, takes S = W - W_prev and
    splits it with `adjoint_step`: against the current (U_A, V_A) if `carry` (the step split), else against W0's (the
    fixed-bases variant, whose pieces sum to the one-shot adjoint split when c is the same every step). Either way the
    pieces accumulate into U_A, V_A and Delta is the remainder W - U_A V_A^T. `uv`, `sigma`: theory layout and input
    second moments per matrix; `W0`: the run's starting weights. All state in `dtype` on `device`."""

    def __init__(self, uv: dict[str, tuple[Tensor, Tensor]], sigma: dict[str, Tensor], W0: dict[str, Tensor], carry: bool,
                 dtype: torch.dtype, device: str) -> None:
        def put(t: Tensor) -> Tensor:
            return t.detach().to(device=device, dtype=dtype).clone()

        self.carry = carry
        self.U0, self.V0 = {k: put(u) for k, (u, _) in uv.items()}, {k: put(v) for k, (_, v) in uv.items()}
        self.dU, self.dV = {k: torch.zeros_like(u) for k, u in self.U0.items()}, {k: torch.zeros_like(v) for k, v in self.V0.items()}
        self.sigma = {k: put(sigma[k]) for k in uv}
        self.W0 = {k: put(W0[k]) for k in uv}
        self.W = {k: w.clone() for k, w in self.W0.items()}
        self.second = {k: torch.zeros_like(w) for k, w in self.W0.items()}  # sum over steps of dU dV^T (carry only)

    @torch.no_grad()
    def step(self, W: dict[str, Tensor]) -> dict[str, float]:
        """Split one step of every matrix; pooled energies over the matrices, the largest per-matrix second/first, the range
        of c, and how many matrices' steps had no first-order image (`adjoint_step`'s zero pieces)."""
        tot = {"step": 0.0, "first": 0.0, "second": 0.0}
        worst, cs = 0.0, []
        for k in self.W:
            w = W[k].detach().to(self.W[k].dtype)
            S = w - self.W[k]
            self.W[k] = w.clone()
            U, V = (self.U0[k] + self.dU[k], self.V0[k] + self.dV[k]) if self.carry else (self.U0[k], self.V0[k])
            dU, dV, second, st = adjoint_step(S, U, V, self.sigma[k])
            self.dU[k] += dU
            self.dV[k] += dV
            if self.carry:
                self.second[k] += second
            for f in tot:
                tot[f] += st[f]
            worst = max(worst, st["second"] / st["first"] if st["first"] > 0 else 0.0)
            cs.append(st["c"])
        return {**tot, "second_over_first": tot["second"] / tot["first"] if tot["first"] > 0 else 0.0,
                "second_over_first_max": worst, "c_min": min(cs), "c_max": max(cs), "n_zeroed": sum(c == 0 for c in cs)}

    def delta_change(self) -> dict[str, Tensor]:
        """Delta - Delta_0 = (W - W0) - (U_A V_A^T - U_A0 V_A0^T), with the component change written as
        dU V0^T + U0 dV^T + dU dV^T (no cancellation against the full products)."""
        return {k: (self.W[k] - self.W0[k]) - (self.dU[k] @ self.V0[k].T + self.U0[k] @ self.dV[k].T + self.dU[k] @ self.dV[k].T)
                for k in self.W}

    def state(self) -> dict[str, tuple[Tensor, Tensor, Tensor]]:
        """(delta-W, U_A', V_A') per matrix, float32 on CPU: harness.State."""
        return {k: ((self.W[k] - self.W0[k]).float().cpu(), (self.U0[k] + self.dU[k]).float().cpu(),
                    (self.V0[k] + self.dV[k]).float().cpu()) for k in self.W}

    def piece_change_max(self) -> float:
        """The largest relative change of any alive write or read vector, max_c ||dU_c|| / ||U0_c|| (and V)."""
        return max(max(float((self.dU[k].norm(dim=0) / self.U0[k].norm(dim=0)).max()),
                       float((self.dV[k].norm(dim=0) / self.V0[k].norm(dim=0)).max())) for k in self.W)
