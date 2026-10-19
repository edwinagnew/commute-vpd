"""Revert, ablate and read single subcomponents of a decomposed model in place (forward hooks or weight edits).
Moved verbatim from experiments/identity_test/run.py (2026-10-15) so later experiments can share it. Theory layout:
UV[k] = (U (d_out, n), V (d_in, n)) of the alive subcomponents, states st[k] = (U2, V2) the same after an update."""

from __future__ import annotations

from typing import Any

import torch


def fwd(net: Any, toks: torch.Tensor) -> torch.Tensor:
    """fp32 logits, no autocast: single-subcomponent reverts are small."""
    out = net(toks)
    return (out[0] if isinstance(out, tuple) else out).float()


@torch.no_grad()
def heldout_log_odds(model: Any, toks: torch.Tensor, half: int, o_token: int) -> float:
    from commute.identity import log_odds

    return float(log_odds(fwd(model.target_model, toks)[:, half], o_token).mean())


@torch.no_grad()
def revert_one(model: Any, UV: dict[str, Any], st: dict[str, Any], toks: torch.Tensor, half: int, o_token: int,
               base: float, keys: list[str], m: int = 8, limit: int | None = None) -> dict[str, torch.Tensor]:
    """Per matrix, the drop in held-out mean log-odds of o when subcomponent j alone is reverted. The revert is
    written as (x . dv) u2 + (x . v) du, which equals (x . v2) u2 - (x . v) u without cancellation. The model is
    causal, so only tokens 0..half are fed (the logits at `half` are unchanged; full-vocab fp32 logits for m groups of
    all 41 positions do not fit in memory)."""
    from commute import vpd
    from commute.identity import Edit, log_odds

    out = {}
    nb = toks.shape[0]
    toks = toks[:, :half + 1]
    for k in keys:
        U, V = UV[k]
        U2, V2 = st[k]
        dU, dV = U2 - U, V2 - V
        lin = vpd.target_linear(model, k)
        n = U.shape[1] if limit is None else min(limit, U.shape[1])
        drop = torch.zeros(U.shape[1], dtype=torch.float64)
        for s in range(0, n, m):
            cidx = torch.arange(s, min(s + m, n), device=U.device)
            h = lin.register_forward_hook(Edit([(U2, dV, 1.0), (dU, V, 1.0)], cidx))
            try:
                lo = log_odds(fwd(model.target_model, toks.repeat(len(cidx), 1))[:, half], o_token)
            finally:
                h.remove()
            drop[s:s + len(cidx)] = (base - lo.reshape(len(cidx), nb).mean(1)).double().cpu()
        out[k] = drop
    return out


def reverted_weights(W: dict[str, torch.Tensor], UV: dict[str, Any], st: dict[str, Any],
                     sel: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Weights W with the subcomponents sel[k] (alive positions) reverted: minus U2 dV^T + dU V^T on those columns."""
    out = dict(W)
    for k, idx in sel.items():
        if len(idx) == 0:
            continue
        U, V = UV[k]
        U2, V2 = st[k]
        i = idx.to(U.device)
        out[k] = W[k] - (U2[:, i] @ (V2 - V)[:, i].T + (U2 - U)[:, i] @ V[:, i].T)
    return out


# ----------------------------------------------------------------------------- signatures


@torch.no_grad()
def read_acts(model: Any, UVd: dict[str, Any], rows: list[torch.Tensor], keys: list[str]) -> dict[str, torch.Tensor]:
    """x . v for every alive subcomponent at every bank position, (N, n) per matrix, fp32 on device."""
    from commute import vpd

    acc: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    hs = []
    for k in keys:
        def hook(_m: Any, args: tuple[Any, ...], _o: Any, k: str = k) -> None:
            x = args[0].float()
            acc[k].append(x.reshape(-1, x.shape[-1]) @ UVd[k][1])
        hs.append(vpd.target_linear(model, k).register_forward_hook(hook))
    try:
        for b in rows:
            fwd(model.target_model, b)
    finally:
        for h in hs:
            h.remove()
    return {k: torch.cat(v) for k, v in acc.items()}


@torch.no_grad()
def gamma_profiles(model: Any, pile: list[torch.Tensor], sel: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """g over the Pile bank for the subcomponents sel[k] (full indices, not alive positions), (N, len) per matrix
    (Gamma frozen)."""
    from commute import vpd

    acc: dict[str, list[torch.Tensor]] = {k: [] for k in sel}
    for b in pile:
        ci = vpd.compute_ci(model, b)
        for k, ids in sel.items():
            acc[k].append(ci.g[k].reshape(-1, ci.g[k].shape[-1])[:, ids].float())
        del ci
    return {k: torch.cat(v) for k, v in acc.items()}


@torch.no_grad()
def ablation_effect(model: Any, k: str, j_alive: int, U: torch.Tensor, V: torch.Tensor, pile_rows: torch.Tensor,
                    flat_pos: torch.Tensor) -> torch.Tensor:
    """Per position in flat_pos (indices into the Pile bank, row * 512 + t): log-softmax(full) - log-softmax(j
    ablated), (len(flat_pos), vocab) fp32."""
    from commute import vpd
    from commute.identity import Edit

    T = pile_rows.shape[1]
    r_ids, t_ids = flat_pos // T, flat_pos % T
    lin = vpd.target_linear(model, k)
    parts: dict[int, torch.Tensor] = {}
    uniq = torch.unique(r_ids)
    for s in range(0, len(uniq), 4):
        rows = uniq[s:s + 4]
        toks = pile_rows[rows].to(U.device)
        full = torch.log_softmax(fwd(model.target_model, toks), -1)
        h = lin.register_forward_hook(Edit([(U, V, 1.0)], torch.tensor([j_alive], device=U.device)))
        try:
            abl = torch.log_softmax(fwd(model.target_model, toks), -1)
        finally:
            h.remove()
        for i in range(len(flat_pos)):
            hit = (rows == r_ids[i]).nonzero()
            if len(hit):
                parts[i] = (full[int(hit), t_ids[i]] - abl[int(hit), t_ids[i]]).cpu()
        del full, abl
    out = torch.stack([parts[i] for i in range(len(flat_pos))])
    return out
