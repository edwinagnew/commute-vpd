"""Component identity under an update (theory/component_identity.md): gauge-free comparisons of a subcomponent
before (u, v) and after (u2, v2), and the hooks that revert or ablate single subcomponents.

Theory layout throughout: per matrix U (d_out, n) holds write vectors as columns, V (d_in, n) read vectors, so the
subcomponent sum is U V^T and subcomponent j acts on an input x as x -> (x . v_j) u_j.
Rank-1 gauge: (u s, v / s) is the same subcomponent for any s != 0, so only |cos| of each side, the product norm
||u|| ||v|| and the rank-1 matrix u v^T are reported, never raw ||du||.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


def change_stats(U: Tensor, V: Tensor, U2: Tensor, V2: Tensor) -> dict[str, Tensor]:
    """Per column j: |cos(u, u2)|, |cos(v, v2)|, product-norm ratio ||u2|| ||v2|| / (||u|| ||v||), and the relative
    change of the rank-1 matrix ||u2 v2^T - u v^T||_F / ||u v^T||_F (from inner products, no d_out x d_in matrix)."""
    U, V, U2, V2 = U.double(), V.double(), U2.double(), V2.double()
    nu, nv, nu2, nv2 = U.norm(dim=0), V.norm(dim=0), U2.norm(dim=0), V2.norm(dim=0)
    uu, vv = (U * U2).sum(0), (V * V2).sum(0)
    sq = (nu2 * nv2) ** 2 + (nu * nv) ** 2 - 2 * uu * vv
    return {"cos_u": uu.abs() / (nu * nu2), "cos_v": vv.abs() / (nv * nv2), "scale": (nu2 * nv2) / (nu * nv),
            "rel_change": sq.clamp_min(0).sqrt() / (nu * nv)}


def _zscore(A: Tensor) -> Tensor:
    A = A.double()
    A = A - A.mean(0, keepdim=True)
    return A / A.norm(dim=0, keepdim=True).clamp_min(1e-300)


def profile_retrieval(before: Tensor, after: Tensor, idx: Tensor) -> dict[str, Tensor]:
    """before, after: (N, n) activations x . v of the n subcomponents of one matrix over the same N positions.
    For each j in idx: r_self = |Pearson(after_j, before_j)| (|.| because the sign of x . v is gauge), rank of j among
    all n before-profiles by |Pearson| with after_j (1 = j is the best match), and the best other |Pearson|."""
    idx = idx.to(before.device)
    zb, za = _zscore(before), _zscore(after[:, idx])
    C = (za.T @ zb).abs()  # (len(idx), n)
    ar = torch.arange(len(idx), device=C.device)
    r_self = C[ar, idx]
    rank = 1 + (C > r_self[:, None]).sum(1)
    other = C.clone()
    other[ar, idx] = -1.0
    return {"r_self": r_self, "rank": rank, "r_other_max": other.max(1).values}


def topk_jaccard(a: Tensor, b: Tensor, k: int) -> float:
    """Jaccard overlap of the top-k positions by |activation| of two (N,) profiles."""
    ia = set(a.abs().topk(k).indices.tolist())
    ib = set(b.abs().topk(k).indices.tolist())
    return len(ia & ib) / len(ia | ib)


def cosine(a: Tensor, b: Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-300))


def log_odds(logits: Tensor, token: int | Tensor) -> Tensor:
    """log p(token) - log(1 - p(token)), exact as logit_token - logsumexp(other logits): no saturation at p ~ 1.
    `token` is one id for every row, or a tensor of ids shaped like `logits` without its last dimension."""
    lg = logits.float()
    idx = torch.as_tensor(token, device=lg.device, dtype=torch.long).expand(lg.shape[:-1]).unsqueeze(-1)
    other = lg.scatter(-1, idx, float("-inf"))
    return lg.gather(-1, idx).squeeze(-1) - torch.logsumexp(other, -1)


class Edit:
    """Adds to a linear's output, per sample group, minus the contributions of given subcomponents:
    out[g] -= sum over (U_a, V_a, sign) of sign * (x[g] . V_a[:, c_g]) U_a[:, c_g].
    A batch of m * b rows is m groups of b rows; group g edits subcomponent cidx[g]. With terms
    [(U2, V2, +1), (U, V, -1)] this reverts j's change (after -> before); with [(U, V, +1)] it ablates j."""

    def __init__(self, terms: list[tuple[Tensor, Tensor, float]], cidx: Tensor):
        self.terms, self.cidx = terms, cidx

    def __call__(self, _m: nn.Module, args: tuple[Any, ...], out: Tensor) -> Tensor:
        x = args[0]
        m = self.cidx.numel()
        xg = x.reshape(m, -1, *x.shape[1:]).to(self.terms[0][0].dtype)  # (m, b, T, d_in)
        sub = 0
        for U, V, sign in self.terms:
            a = torch.einsum("mbtd,dm->mbt", xg, V[:, self.cidx])
            sub = sub + sign * torch.einsum("mbt,om->mbto", a, U[:, self.cidx])
        return out - sub.reshape(out.shape).to(out.dtype)


def scale_grads(G: Tensor, U: Tensor, V: Tensor) -> Tensor:
    """dL/ds_c = u_c^T G v_c for B_c -> (1 + s_c) B_c, given G = dL/dW (d_out, d_in). Gauge-free."""
    return ((G.double() @ V.double()) * U.double()).sum(0)
