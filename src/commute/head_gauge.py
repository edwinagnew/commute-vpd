"""The attention-head gauge of the 67M (cell P1.60, E286 stage `gauge`; redteam-lead's note 2026-11-24, item 4, after
Singh 2026). Added 2026-11-24.

Per head h of a layer, with the 67M's shapes (q_proj, k_proj, v_proj weights (H D, d), head h's rows hD..hD+D;
o_proj weight (d, H D), head h's columns), the maps that leave the network's function unchanged:
    QK  W_q,h -> G W_q,h and W_k,h -> G^{-T} W_k,h, for G commuting with the rotary embedding. The rotary embedding
        turns each pair of head dimensions (p1, p2) by a position-dependent angle at its own frequency, so G must act
        on each pair as a I + b J (J the pair's quarter turn, `rotate_every_two`); an orthogonal G is a turn by an
        angle per pair. A G outside this set changes the attention pattern, so it is not a gauge here.
    VO  W_v,h -> G W_v,h and W_O,h -> W_O,h G^{-1}, for any invertible G (128 x 128): no rotary on values.
`rotate` applies a random orthogonal gauge (per-pair turns for QK, a random orthogonal S for VO) in fp64, cast back.
`tangent_share` takes an update dW of the head matrices and returns the part of its squared norm along the gauge's
tangent at M: for QK the span, per pair, of (I_p W_q, -I_p W_k) and (J_p W_q, J_p W_k) (I_p the pair's identity);
for VO the set {(A W_v, -W_O A)}, its least-squares A from a Sylvester equation solved in the eigenbases.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import Tensor

from commute import rule_retention as R

H, D = R.N_HEAD, R.D_HEAD


def pairs(attn: Any) -> list[tuple[int, int]]:
    """The rotary pairs of head dimensions: adjacent (2i, 2i+1) or halves (i, i + D/2), as the module rotates them."""
    assert attn.rotary_dim == attn.head_dim == D
    return [(2 * i, 2 * i + 1) for i in range(D // 2)] if attn.rotary_adjacent_pairs else [(i, i + D // 2) for i in range(D // 2)]


def _turns(pr: list[tuple[int, int]], gen: torch.Generator) -> Tensor:
    """A (D, D) orthogonal map turning each rotary pair by its own random angle: it commutes with the rotary embedding."""
    G = torch.zeros(D, D, dtype=torch.float64)
    for p1, p2 in pr:
        t = float(torch.rand(1, generator=gen, dtype=torch.float64)) * 2 * math.pi
        G[p1, p1], G[p1, p2], G[p2, p1], G[p2, p2] = math.cos(t), -math.sin(t), math.sin(t), math.cos(t)
    return G


def _orth(gen: torch.Generator) -> Tensor:
    q, r = torch.linalg.qr(torch.randn(D, D, generator=gen, dtype=torch.float64))
    return q * torch.sign(torch.diagonal(r))


@torch.no_grad()
def rotate(M: R.Model, seed: int, inverse_of: list[tuple[Tensor, Tensor]] | None = None) -> tuple[R.Model, list[tuple[Tensor, Tensor]]]:
    """A copy of M with a random orthogonal gauge per head (G for QK, S for VO), and the maps used, in layer-head order.
    With `inverse_of` (the maps of an earlier call) it applies their transposes instead: the round trip for the floor."""
    F, gen, used = R.Model(copy.deepcopy(M.net).requires_grad_(False)), torch.Generator().manual_seed(seed), []
    for l, b in enumerate(F.net.h):
        a, pr = b.attn, pairs(b.attn)
        for h in range(H):
            if inverse_of is None:
                G, S = _turns(pr, gen), _orth(gen)
            else:
                G, S = (x.T for x in inverse_of[l * H + h])
            used.append((G, S))
            r = slice(h * D, (h + 1) * D)
            for w, m in ((a.q_proj.weight, G), (a.k_proj.weight, G), (a.v_proj.weight, S)):  # G orthogonal: G^{-T} = G
                w[r] = (m.to(w.device) @ w[r].double()).to(w.dtype)
            wo = a.o_proj.weight
            wo[:, r] = (wo[:, r].double() @ S.T.to(wo.device)).to(wo.dtype)
    return F, used


def _qk_part(Wq: Tensor, Wk: Tensor, dq: Tensor, dk: Tensor, pr: list[tuple[int, int]]) -> tuple[Tensor, Tensor]:
    """The component of (dq, dk) along the QK gauge's tangent at (Wq, Wk), one head (D, d) each: per rotary pair the
    least-squares fit on (I_p Wq, -I_p Wk) and (J_p Wq, J_p Wk); rows of different pairs are disjoint."""
    cq, ck = torch.zeros_like(dq), torch.zeros_like(dk)
    for p in pr:
        rows = list(p)
        X, Y, dX, dY = Wq[rows], Wk[rows], dq[rows], dk[rows]
        JX, JY = torch.stack([-X[1], X[0]]), torch.stack([-Y[1], Y[0]])
        T = torch.stack([torch.cat([X.flatten(), -Y.flatten()]), torch.cat([JX.flatten(), JY.flatten()])])  # (2, 4d)
        v = torch.cat([dX.flatten(), dY.flatten()])
        fit = (T.T @ torch.linalg.lstsq(T.T, v[:, None]).solution).squeeze(1)
        cq[rows], ck[rows] = fit[: X.numel()].view_as(X), fit[X.numel() :].view_as(Y)
    return cq, ck


def _vo_part(Wv: Tensor, Wo: Tensor, dv: Tensor, do: Tensor) -> tuple[Tensor, Tensor]:
    """The component of (dv, do) in {(A Wv, -Wo A)}: A solves Wo^T Wo A + A Wv Wv^T = dv Wv^T - Wo^T do."""
    P, Q, C = Wo.T @ Wo, Wv @ Wv.T, dv @ Wv.T - Wo.T @ do
    lp, U = torch.linalg.eigh(P)
    lq, V = torch.linalg.eigh(Q)
    den = lp[:, None] + lq[None, :]
    keep = den > 1e-12 * float(den.max())  # directions where neither Wv nor Wo moves anything carry no tangent
    A = U @ torch.where(keep, (U.T @ C @ V) / torch.where(keep, den, 1.0), 0.0) @ V.T
    return A @ Wv, -Wo @ A


NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")


@torch.no_grad()
def tangent_component(M: R.Model, dW: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """dW's component along the gauge's tangent at M on every head matrix (fp64, CPU, keyed by parameter name), and
    its squared norm's QK and VO parts over the head matrices' total. dW keyed by parameter name (missing names are 0)."""
    comp: dict[str, Tensor] = {}
    qk = vo = tot = 0.0
    for l, b in enumerate(M.net.h):
        a, pr = b.attn, pairs(b.attn)
        key = {m: f"h.{l}.attn.{m}.weight" for m in NAMES}
        W = {m: getattr(a, m).weight.detach().double().cpu() for m in NAMES}
        d = {m: dW[key[m]].double().cpu() if key[m] in dW else torch.zeros_like(W[m]) for m in NAMES}
        c = {m: torch.zeros_like(W[m]) for m in NAMES}
        tot += sum(float(x.pow(2).sum()) for x in d.values())
        for h in range(H):
            r = slice(h * D, (h + 1) * D)
            c["q_proj"][r], c["k_proj"][r] = _qk_part(W["q_proj"][r], W["k_proj"][r], d["q_proj"][r], d["k_proj"][r], pr)
            c["v_proj"][r], c["o_proj"][:, r] = _vo_part(W["v_proj"][r], W["o_proj"][:, r], d["v_proj"][r], d["o_proj"][:, r])
        qk += float(c["q_proj"].pow(2).sum() + c["k_proj"].pow(2).sum())
        vo += float(c["v_proj"].pow(2).sum() + c["o_proj"].pow(2).sum())
        comp |= {key[m]: c[m] for m in NAMES}
    return comp, {"head_norm2": tot, "qk": qk / tot if tot > 0 else None, "vo": vo / tot if tot > 0 else None,
                  "share": (qk + vo) / tot if tot > 0 else None}  # fmt: skip


def tangent_share(M: R.Model, dW: dict[str, Tensor]) -> dict[str, Any]:
    """`tangent_component`'s norms only (the gauge stage's reading)."""
    return tangent_component(M, dW)[1]
