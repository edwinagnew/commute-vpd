"""Decompositions of W0 that need no training (E136, cell P2.8; plan reports/plans/2026-11-09_plan_e136-share-untrained-decompositions.md):
each matrix's SVD and that SVD randomly rotated, the no-training gate rule, a masked forward, and the do-nothing share's cells on it.

Layout as `vpd`: W is (d_out, d_in), U (C, d_out) holds u_i as rows, V (d_in, C) holds v_i as columns, and a matrix's
output on input x (row vectors) is (x @ V) @ U. For the SVD W = A diag(S) B^T: U = diag(S) A^T ("U Sigma"), V = B. The
rotated decomposition is V R and R^T U ("U Sigma R" as rows), R Haar-random per matrix from one generator (seed 0).

These decompositions cannot be written into `model.components`, which holds 512 subcomponents on `attn.k_proj` and
`attn.q_proj`, below their rank of 768. So the masked forward is a forward hook on each of the 24 matrices that replaces
its output by (g * (x @ V)) @ U, in fp32 with no autocast. The target model's own weights at those 24 matrices then
do not enter the masked forward, so W0's components are used whatever target the model holds.

The gate rule (card, `rival_a`): at each matrix, window and position, component i's contribution is |v_i^T x| ||u_i||,
x that matrix's input on that forward; the k_m largest get gate 1, the rest 0.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from commute.data import target_tokens
from commute.faithfulness import model_state
from commute.identity import log_odds


@dataclass(frozen=True)
class Decomposition:
    U: dict[str, Tensor]  # (C, d_out), u_i as rows
    V: dict[str, Tensor]  # (d_in, C), v_i as columns
    u_norm: dict[str, Tensor]  # (C,) ||u_i||


Gates = dict[str, Tensor]  # matrix -> (B, T, C) bool


def _decomposition(U: dict[str, Tensor], V: dict[str, Tensor]) -> Decomposition:
    return Decomposition(U, V, {k: u.norm(dim=1) for k, u in U.items()})


def svd_decomposition(W0: Mapping[str, Tensor]) -> Decomposition:
    """Each matrix's SVD (full_matrices=False), computed in float64 and kept in fp32."""
    U, V = {}, {}
    for k, W in W0.items():
        A, S, Bh = torch.linalg.svd(W.double(), full_matrices=False)
        U[k], V[k] = (A * S).T.float(), Bh.T.float()
    return _decomposition(U, V)


def haar_orthogonal(n: int, gen: torch.Generator) -> Tensor:
    """(n, n) float64 on CPU: Q of the QR of a Gaussian, each column's sign fixed by R's diagonal so that Q is Haar."""
    Q, R = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, generator=gen))
    return Q * torch.sign(torch.diagonal(R))[None, :]


def rotated_decomposition(svd: Decomposition, seed: int = 0) -> Decomposition:
    """V R and R^T U, one R per matrix in sorted order of the matrix names, all from one generator."""
    gen, U, V = torch.Generator().manual_seed(seed), {}, {}
    for k in sorted(svd.U):
        R = haar_orthogonal(svd.U[k].shape[0], gen).to(svd.U[k].device)
        U[k], V[k] = (R.T @ svd.U[k].double()).float(), (svd.V[k].double() @ R).float()
    return _decomposition(U, V)


def reconstruction_error(W0: Mapping[str, Tensor], d: Decomposition) -> float:
    """The largest over matrices of max |sum_i u_i v_i^T - W| / max |W|, in fp32."""
    return max(float((d.U[k].T @ d.V[k].T - W).abs().max() / W.abs().max()) for k, W in W0.items())


def top_k_gates(x: Tensor, V: Tensor, u_norm: Tensor, k: int) -> Tensor:
    """The rule: x (B, T, d_in); (B, T, C) bool, True on the k components of largest |x @ v_i| ||u_i||. k above C keeps all C."""
    score = (x @ V).abs() * u_norm
    idx = score.topk(min(k, score.shape[-1]), dim=-1).indices
    return torch.zeros_like(score, dtype=torch.bool).scatter(-1, idx, True)


def logits(model: Any, batch: Tensor) -> Tensor:
    out = model.target_model(batch)
    return (out[0] if isinstance(out, tuple) else out).float()


@torch.no_grad()
def forward_with_gates(model: Any, batch: Tensor, decomps: Mapping[str, Decomposition], k: Mapping[str, int]) -> tuple[Tensor, dict[str, Gates]]:
    """The model's unmasked forward (fp32, as it stands) on `batch`, and each decomposition's gates from that forward's matrix inputs."""
    gates: dict[str, Gates] = {n: {} for n in decomps}

    def make(key: str) -> Any:
        def hook(_m: Any, args: tuple[Tensor, ...], _output: Tensor) -> None:
            for n, d in decomps.items():
                gates[n][key] = top_k_gates(args[0], d.V[key], d.u_norm[key], k[key])

        return hook

    handles = [model.target_model.get_submodule(key).register_forward_hook(make(key)) for key in k]
    try:
        return logits(model, batch), gates
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def masked_forward(model: Any, batch: Tensor, d: Decomposition, gates: Gates) -> Tensor:
    """Logits with each matrix's output replaced by (g * (x @ V)) @ U (fp32, x the matrix's input on this forward)."""

    def make(key: str) -> Any:
        def hook(_m: Any, args: tuple[Tensor, ...], _output: Tensor) -> Tensor:
            return (gates[key].to(args[0].dtype) * (args[0] @ d.V[key])) @ d.U[key]

        return hook

    handles = [model.target_model.get_submodule(key).register_forward_hook(make(key)) for key in gates]
    try:
        return logits(model, batch)
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def do_nothing_cells(model: Any, windows: Tensor, half: int, token: int | Tensor, decomps: Mapping[str, Decomposition], k: Mapping[str, int],
                     batch_size: int, ci_len: int | None = None, target: Mapping[str, Tensor] | None = None) -> dict[str, Any]:  # fmt: skip
    """Mean answer log-odds under the weights `target` (None: the model as it stands): "T", the unmasked model, and per decomposition its
    masked model with the gates this forward's inputs give. At W0 the masked value is that decomposition's L00; at a state, its L01: the masked
    forward uses W0's components whatever `target` is, since the hooks replace all 24 matrices' outputs."""
    device = next(model.parameters()).device
    sums = {"T": 0.0, **{n: 0.0 for n in decomps}}
    with model_state(model, target=dict(target)) if target is not None else nullcontext():
        for s in range(0, windows.shape[0], batch_size):
            batch = windows[s : s + batch_size].to(device)
            tgt, x = target_tokens(token, batch, half), batch[:, :ci_len]
            logits, gates = forward_with_gates(model, x, decomps, k)
            sums["T"] += float(log_odds(logits[:, half], tgt).double().sum())
            for n, d in decomps.items():
                sums[n] += float(log_odds(masked_forward(model, x, d, gates[n])[:, half], tgt).double().sum())
    return {name: v / windows.shape[0] for name, v in sums.items()}


def k_from_counts(counts: Mapping[str, float], n_positions: int) -> dict[str, int]:
    """k_m: the number of trained subcomponents with gate above 0.5, averaged over all read windows and positions, rounded, at least 1."""
    return {key: max(1, round(c / n_positions)) for key, c in counts.items()}
