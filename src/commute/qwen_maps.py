"""E44 (experiments/qwen_maps): splits of a saved Qwen3-1.7B update over the alive subcomponents, as states, and the
row loop that reads them. Plan: reports/plans/2026-10-28_plan_qwen-q0q1.md, sections 2-4.

Layout. `qwen_vpd` stores V [d_in, C] and U [C, d_out]; `split` works in theory layout, U_A (d_out, n), V_A (d_in, n),
W ≈ U_A V_Aᵀ in HF layout [d_out, d_in]. `Alive.uv` converts. A state is, per site, a change of the weights and a
change (dU, dV) of some alive columns (`SiteChange`); `applied` writes the columns into the components in place on the
GPU, swaps the weights in by pointer (`qwen_sft.set_weights`), and restores the original columns from the host copy.

Products of stage S, per site and step (`site_products`), small enough to keep on disk: X = K⁺δW for the least-squares
split and for each control (i) half (dU = X V, dV = Xᵀ U), O(δW) for the in-span state, the adjoint split's scale c,
and control (ii)'s scale a (its Gaussian pieces are redrawn from the seed, `random_pieces`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor

from commute import split
from commute.qwen_sft import linears, set_weights
from commute.qwen_vpd import (
    SITES,
    Components,
    Gamma,
    importances,
    masked_logits,
    parse_site,
    site_name,
    stochastic_masks,
)
from commute.vpd import kl_per_position

SHARED_INPUT = {"q": "q", "k": "q", "v": "q", "o": "o", "gate": "gate", "up": "gate", "down": "down"}
SiteChange = tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]  # dW, positions in alive, dU, dV
NONE: SiteChange = (None, None, None, None)
STATES = ("W0", "do_nothing", "ls", "inspan", "adjoint", "lower", "upper", "random")


class Alive:
    """The alive index per site (on the components' device) and the original alive columns on the host, in the
    components' dtype, so `applied` restores them exactly."""

    def __init__(self, comps: Components, idx: dict[str, Tensor]):
        self.idx, self.V, self.U = {}, {}, {}
        pin = torch.cuda.is_available()
        for s in SITES:
            V, U = comps.vu(s)
            self.idx[s] = idx[s].to(V.device)
            v, u = V[:, self.idx[s]].cpu(), U[self.idx[s]].cpu()
            self.V[s], self.U[s] = (v.pin_memory(), u.pin_memory()) if pin else (v, u)

    def uv(self, site: str, device: str, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """(U_A (d_out, n), V_A (d_in, n)) in theory layout."""
        return self.U[site].to(device, dtype).T, self.V[site].to(device, dtype)

    def count(self) -> int:
        return sum(len(v) for v in self.idx.values())


def alive_from_stats(*stats: dict[str, Tensor]) -> dict[str, Tensor]:
    """Alive = g > 0.01 somewhere in any block: a non-zero `active` share in any `qwen_teacher.g_stats` file."""
    return {s: torch.stack([st[f"{s}.active"].float() > 0 for st in stats]).any(0).nonzero().flatten() for s in SITES}


def halves(active: Tensor, mean_g: Tensor, n_half: int | None = None) -> tuple[Tensor, Tensor]:
    """Positions (into the alive list) of the lower and upper halves by share of positions with g > 0.01 (`active`),
    ties by mean g, then by position. Each half has n // 2 members (the middle one of an odd count is in neither)."""
    n = len(active)
    k = n // 2 if n_half is None else n_half
    order = mean_g.double().argsort(stable=True)
    order = order[active.double()[order].argsort(stable=True)]
    return order[:k].sort().values, order[n - k :].sort().values


@torch.no_grad()
def input_moments(model: Any, batches: list[Tensor], sites: list[str]) -> dict[str, Tensor]:
    """Σ = E[a aᵀ] (float64, host) of each site's input over every position; q, k, v share one tensor, as do gate and
    up (the same input in Qwen3's blocks)."""
    owners = sorted({site_name(parse_site(s)[0], SHARED_INPUT[parse_site(s)[1]]) for s in sites})
    acc: dict[str, Tensor] = {}

    def hook(s: str) -> Any:
        def h(_m: Any, a: tuple[Tensor, ...], _o: Any) -> None:
            x = a[0].reshape(-1, a[0].shape[-1]).double()
            acc[s] = acc[s] + x.T @ x if s in acc else x.T @ x
        return h

    lins = linears(model)
    handles = [lins[s].register_forward_hook(hook(s)) for s in owners]
    n = 0
    try:
        for b in batches:
            model(b, use_cache=False)
            n += b.numel()
    finally:
        for h in handles:
            h.remove()
    out = {s: (acc.pop(s) / n).cpu() for s in owners}
    return {s: out[site_name(parse_site(s)[0], SHARED_INPUT[parse_site(s)[1]])] for s in sites}


def random_pieces(shape_u: tuple[int, ...], shape_v: tuple[int, ...], seed: int, site: str, device: str
                  ) -> tuple[Tensor, Tensor]:
    """Control (ii)'s unscaled Gaussian pieces (RU, RV), fp32, drawn on `device` from (seed, site): the same draw in S,
    where the scale is fitted, and when the state is applied."""
    gen = torch.Generator(device=device).manual_seed(seed * 1000 + SITES.index(site))
    RU = torch.randn(shape_u, generator=gen, device=device)
    return RU, torch.randn(shape_v, generator=gen, device=device)


# ----------------------------------------------------------------------------- stage S, one site


def top_share(M: Tensor, b: split.Basis, frac: float = 0.9) -> dict[str, float]:
    """δW's top singular directions carrying `frac` of ‖δW‖²: how many (k), and the share of their energy whose write
    side lies in col U_A and whose read side lies in col V_A. NaN for δW = 0."""
    M = M.double()
    tall = M.shape[0] >= M.shape[1]
    ev, vec = torch.linalg.eigh(M.T @ M if tall else M @ M.T)
    ev, vec = ev.flip(0).clamp(min=0), vec.flip(1)
    total = float(ev.sum())
    if total <= 0:
        return {"k": 0, "in_U": float("nan"), "in_V": float("nan")}
    k = int((ev.cumsum(0) < frac * total).sum()) + 1
    Pu, Pv = b.P[:, b.s > 0], b.Q[:, b.t > 0]
    top, e = vec[:, :k], float(ev[:k].sum())
    if tall:  # top = right vectors v_i; M v_i = σ_i u_i
        in_u, in_v = float(((Pu.T @ (M @ top)) ** 2).sum()), float((ev[:k] * ((Pv.T @ top) ** 2).sum(0)).sum())
    else:  # top = left vectors u_i; Mᵀ u_i = σ_i v_i
        in_u, in_v = float((ev[:k] * ((Pu.T @ top) ** 2).sum(0)).sum()), float(((Pv.T @ (M.T @ top)) ** 2).sum())
    return {"k": k, "in_U": in_u / e, "in_V": in_v / e}


def make_basis(A: Tensor, method: str) -> tuple[Tensor, Tensor]:
    return split.gram_basis(A) if method == "gram" else split.full_svd(A.double())


def to_disk(arrays: dict[str, Tensor]) -> dict[str, Tensor]:
    """S's arrays as saved: X and piece energies fp32, positions int64. Not bf16: the pieces X V, Xᵀ U amplify X's
    rounding by up to t_max / t_min of the alive spectrum (review of 2026-10-28; ~10% in bf16 at condition 50); fp32
    X is 5.6 GB per split and step, and `stored_error` checks what is left."""
    return {k: (v if k.startswith("pos_") else v.float()).cpu().contiguous() for k, v in arrays.items()}


def pieces(X: Tensor, U: Tensor, V: Tensor) -> tuple[Tensor, Tensor]:
    """A stored split's applied pieces dU = X V, dV = Xᵀ U: formed in float64 from X as given (fp32 from S's file),
    then rounded to fp32, the dtype the forward's components hold; returned float64 so that dσ of them is what the
    forward effectively applies. Not formed in fp32: X's condition (up to 6e5 on mlp.up_proj) amplifies the
    product's rounding past STORE_ERR_MAX (results/diag_stored.jsonl, 2026-10-28)."""
    X, U, V = X.double(), U.double(), V.double()
    return (X @ V).float().double(), (X.T @ U).float().double()


def stored_error(X: Tensor, U: Tensor, V: Tensor, S: Tensor, E: float) -> float:
    """‖dσ(`pieces` of X stored fp32) − dσ(X V, Xᵀ U in float64)‖_Σ / ‖δW‖_Σ, X, U, V float64: what the stored
    precision costs the applied state, as `site_change` builds it (0 for δW = 0)."""
    off = split.dsigma(*pieces(X.float(), U, V), U, V) - split.dsigma(X @ V, X.T @ U, U, V)
    return (split.weighted_energy(off, S) / E) ** 0.5 if E > 0 else 0.0


def site_products(M: Tensor, U_A: Tensor, V_A: Tensor, S: Tensor, b: split.Basis,
                  hb: dict[str, tuple[Tensor, split.Basis]], RU: Tensor, RV: Tensor) -> tuple[dict[str, Tensor], dict]:
    """Everything stage S keeps for one site and step (float64 inputs on one device): the arrays (X of each split,
    per-subcomponent piece energies of the least-squares and adjoint splits and of the same splits of
    `split.random_like(δW, Σ, 0)`) and the scalars (energies of δW and of each state's Δ change, the free check, c, a,
    and `stored_error` per stored X). `hb[half]` = (positions in alive, the half's basis). O(δW) is not stored: the
    in-span state's weight change is dσ of its applied pieces, which is δW − O(δW) up to `stored_error["ls"]`.
    `dsigma_smin` is reported (plan, free check): the smallest gain of dσ, which sets how much X's rounding grows."""
    E = split.weighted_energy(M, S)
    X, O = split.least_squares_x(M, b), split.obstruction(M, b)
    dU, dV = X @ V_A, X.T @ U_A
    second = dU @ dV.T
    c = split.adjoint_scale(M, U_A, V_A, S)[0]
    aU, aV = c * (M @ V_A), c * (M.T @ U_A)
    R = split.random_like(M, S, 0)
    XR, cR = split.least_squares_x(R, b), split.adjoint_scale(R, U_A, V_A, S)[0]
    err = {"ls": stored_error(X, U_A, V_A, S, E)}
    arrays = {"X_ls": X,
              "pe_ls": split.piece_energies(dU, dV, U_A, V_A, S),
              "pe_ls_null": split.piece_energies(XR @ V_A, XR.T @ U_A, U_A, V_A, S),
              "pe_adjoint": split.piece_energies(aU, aV, U_A, V_A, S),
              "pe_adjoint_null": split.piece_energies(cR * (R @ V_A), cR * (R.T @ U_A), U_A, V_A, S)}
    delta = {"do_nothing": E, "ls": split.weighted_energy(O - second, S), "inspan": split.weighted_energy(second, S),
             "adjoint": split.weighted_energy(aU @ aV.T, S), "random": 0.0}
    half_energy = {}
    for half, (pos, bh) in hb.items():
        Uh, Vh = U_A[:, pos], V_A[:, pos]
        Xh = split.least_squares_x(M, bh)
        hU, hV = Xh @ Vh, Xh.T @ Uh
        delta[half] = split.weighted_energy(M - split.dsigma(hU, hV, Uh, Vh) - hU @ hV.T, S)
        half_energy[half] = float(split.piece_energies(hU, hV, Uh, Vh, S).sum())
        arrays[f"X_{half}"], arrays[f"pos_{half}"] = Xh, pos
        err[half] = stored_error(Xh, Uh, Vh, S, E)
    wf = split.weighted_fractions(M, b, S)
    scalars = {"energy": E, "delta": delta, "half_piece_energy": half_energy, "c": c, "stored_error": err,
               "a": split.sigma_matched_scale(M, RU, RV, U_A, V_A, S),
               "r_u": b.r_u, "r_v": b.r_v, "d_out": M.shape[0], "d_in": M.shape[1],
               "null": split.gaussian_null(b, M.shape[0], M.shape[1]), "outside_both": wf["omega_act"],
               "outside_both_null": wf["null_act"], "top": top_share(M, b),
               "ls_second": split.weighted_energy(second, S), "dsigma_smin": split.dsigma_smin(b)}
    return arrays, scalars


# ----------------------------------------------------------------------------- states


def site_change(name: str, M: Tensor | None, U_A: Tensor, V_A: Tensor, p: Callable[[str], Tensor] | None,
                c: float, a: float, RU: Tensor | None = None, RV: Tensor | None = None) -> SiteChange:
    """One site of state `name` (`STATES`), all tensors on one device, float64 (`applied` casts to the forward's
    dtypes). M: the site's δW (None: zero, the site is unchanged in every state). p(key): S's arrays for the site.
    RU, RV: control (ii)'s unscaled pieces."""
    if name == "W0" or M is None:
        return NONE
    if name == "do_nothing":
        return M, None, None, None
    if name == "adjoint":
        aU, aV = c * (M @ V_A), c * (M.T @ U_A)
        return split.dsigma(aU, aV, U_A, V_A), None, aU, aV
    if name == "random":
        assert RU is not None and RV is not None
        dU, dV = a * RU, a * RV
        return split.dsigma(dU, dV, U_A, V_A) + dU @ dV.T, None, dU, dV
    assert p is not None, name
    if name in ("ls", "inspan"):
        dU, dV = pieces(p("X_ls").to(M.device), U_A, V_A)
        w = M if name == "ls" else split.dsigma(dU, dV, U_A, V_A)  # = δW − O(δW), up to S's stored_error
        return w, None, dU, dV
    pos = p(f"pos_{name}").to(M.device)
    return M, pos, *pieces(p(f"X_{name}").to(M.device), U_A[:, pos], V_A[:, pos])


@contextmanager
def applied(model: Any, comps: Components, alive: Alive, change: Callable[[str], SiteChange],
            weights: dict[str, Tensor] | None = None) -> Iterator[None]:
    """The state whose sites `change` gives: weights W0 + dW swapped in (fp32, the model's device), and the alive
    columns at `positions` (None: all) moved by (dU, dV) in place; the original columns written back on exit.
    `weights`: the state's weights given whole instead (the fine-tuned ones, for states whose weights are W0 + δW: no
    second 5.6 GB copy on the GPU)."""
    lins, W, touched = linears(model), {}, []
    try:
        with torch.no_grad():
            for s in SITES:
                dW, pos, dU, dV = change(s)
                w0 = lins[s].weight.data
                W[s] = weights[s] if weights is not None else w0 if dW is None else w0 + dW.to(w0)
                if dU is None or dV is None:
                    continue
                V, U = comps.vu(s)
                cols = alive.idx[s] if pos is None else alive.idx[s][pos.to(V.device)]
                touched.append(s)
                V[:, cols] = V[:, cols] + dV.to(V)
                U[cols] = U[cols] + dU.T.to(U)
        with set_weights(model, W):
            yield
    finally:
        with torch.no_grad():
            for s in touched:
                V, U = comps.vu(s)
                V[:, alive.idx[s]] = alive.V[s].to(V.device)
                U[alive.idx[s]] = alive.U[s].to(U.device)


# ----------------------------------------------------------------------------- the row loop


def _kl(p: Tensor, q: Tensor) -> float:
    return float(kl_per_position(p, q).double().sum())


def _stochastic(g: dict[str, Tensor], seed: int, first_row: int) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """`qwen_vpd.stochastic_masks` drawn row by row from (seed, row index), the same draw for a row at any batch size,
    written into buffers of g's size (no concatenation copy: one row's draw is the only transient)."""
    masks = {s: torch.empty_like(v) for s, v in g.items()}
    deltas = {s: torch.empty(v.shape[:-1], dtype=v.dtype, device=v.device) for s, v in g.items()}
    for j in range(next(iter(g.values())).shape[0]):
        gen = torch.Generator(device=masks[SITES[0]].device).manual_seed(seed * 1_000_003 + first_row + j)
        m, d = stochastic_masks({s: v[j : j + 1] for s, v in g.items()}, gen)
        for s in g:
            masks[s][j : j + 1], deltas[s][j : j + 1] = m[s], d[s]
        del m, d
    return masks, deltas


@torch.no_grad()
def read_rows(model: Any, rows: Tensor, batch: int, gamma: Gamma, comps: Components, stoch_seeds: tuple[int, ...],
              sources: dict[str, dict[str, Tensor]], ft: dict[str, Tensor] | None,
              on_batch: Callable[[int], None] | None = None) -> dict[str, float]:
    """Mean over every position of `rows` ([N, T], any device) of KL(own unmasked ‖ ·) for: CI masks (m = g, Δ off;
    "ci"), stochastic masks per seed ("stoch_<seed>"), and each saved adversary source (m = g + (1 − g) s, d = s_Δ;
    "replay_<name>"); with `ft` (fine-tuned weights), "K" = KL(fine-tuned ‖ CI-masked). g is computed once per batch
    from the state as it stands (its own activations and V). At most one mask set is alive beside g.

    GPU memory per row of 512 (fp32): g 1.41 GB, one mask set 1.41, three logits 0.93, KL temporaries ~1.2: ~5 GB per
    row on top of the fixed ~43.5 GB (target 6.9, components 19.5, Γ 17.1) and the weights a state holds (5.6 GB
    each for the state and the fine-tuned model): batch 2 peaks at ~60-65 GB, batch 4 at ~75, batch 8 at ~95."""
    dev = str(next(iter(comps.V.values())).device)
    sums: dict[str, float] = {}

    def add(k: str, v: float) -> None:
        sums[k] = sums.get(k, 0.0) + v

    for i in range(0, len(rows), batch):
        tokens = rows[i : i + batch].to(dev)
        logits, g = importances(model, tokens, gamma, comps)
        ci = masked_logits(model, tokens, comps, g)
        add("ci", _kl(logits, ci))
        for seed in stoch_seeds:
            masks, deltas = _stochastic(g, seed, i)
            add(f"stoch_{seed}", _kl(logits, masked_logits(model, tokens, comps, masks, deltas)))
            del masks, deltas
        for name, src in sources.items():
            masks = {s: g[s] + (1 - g[s]) * src[s].to(g[s]) for s in SITES}
            deltas = {s: src[f"{s}.delta"].to(g[s]).expand(tokens.shape) for s in SITES}
            add(f"replay_{name}", _kl(logits, masked_logits(model, tokens, comps, masks, deltas)))
            del masks, deltas
        if ft is not None:
            with set_weights(model, ft):
                add("K", _kl(model(tokens, use_cache=False).logits, ci))
        del logits, ci, g
        if on_batch is not None:
            on_batch(i // batch)
    return {k: v / rows.numel() for k, v in sums.items()}
