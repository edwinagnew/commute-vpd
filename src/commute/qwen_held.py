"""The held-model reading S (reading 3 of `commute.held_model`) on Qwen3-1.7B (cell P3.13, E216). Added 2026-11-19 for
experiments/qwen_held_readings; the 67M's `held_model.py` is unchanged and only its `survival` is called here.

M is the base model (W0), F = W0 + δW (δW on the 196 linear matrices only: norms and embeddings are M's). A site is one
layer's MLP `down_proj` (units: its 6,144 input activations, silu(gate) * up) or attention `o_proj` (units: the 16 heads'
128-wide outputs, concatenated, 2,048). At each site both models are read on M's input to that layer: a^M are M's unit
activations, w^F = F's write there (F's own attention or MLP on M's normed residual, through F's output matrix). H is M
with each site's output matrix replaced by the least-squares map from a^M to w^F over the fit positions,
    A = C G^+,  C = sum w^F' a^M (d_model x units),  G = sum a^M' a^M (units x units),  both fp64,
the pseudo-inverse at torch's default (hermitian) tolerance. G depends on M and the rows only, so it is summed once per
row set (`grams`) and reused for every F. A site's residual on any rows, for any A, is
    |w^F - a^M A'|^2 / |w^F|^2 = (n_F - 2 <A, C> + <A G, A>) / n_F,   n_F = sum |w^F|^2,
with that row set's C, G and n_F. S = 1 - KL(F || H) / KL(F || M), KL per position averaged over every position of the
read rows (`held_model.survival`; None at or under the floor). Tests: tests/test_qwen_held.py (the algebra; no model).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor

from commute.held_model import survival
from commute.qwen_vpd import N_LAYERS, site_name
from commute.vpd import kl_per_position

F64 = torch.float64
SITES = tuple(site_name(lyr, k) for lyr in range(N_LAYERS) for k in ("o", "down"))  # the output matrices
Add = Callable[[str, Tensor, Tensor | None], None]
__all__ = ["SITES", "cross", "grams", "held", "held_map", "kl_sums", "max_rel_diff", "residual", "survival"]


def _layers(model: Any) -> Any:
    return model.model.layers


@contextmanager
def _hooks(M: Any, F: Any | None, add: Add) -> Iterator[None]:
    """Hooks on M that call add(site, a^M, w^F) at every output matrix (w^F None when F is None): F's attention and MLP
    run inside M's forward, on the inputs M's own modules receive at that layer."""
    seen: dict[tuple[str, int], Any] = {}
    hs = []
    for lyr, layer in enumerate(_layers(M)):
        def attn_in(_m: Any, _a: Any, kw: dict[str, Any], lyr: int = lyr) -> None:
            seen["attn", lyr] = dict(kw)

        def mlp_in(_m: Any, a: tuple[Tensor, ...], lyr: int = lyr) -> None:
            seen["mlp", lyr] = a[0]

        def o_in(_m: Any, a: tuple[Tensor, ...], lyr: int = lyr) -> None:
            wF = None if F is None else _layers(F)[lyr].self_attn(**seen.pop(("attn", lyr)))[0]
            add(site_name(lyr, "o"), a[0], wF)

        def down_in(_m: Any, a: tuple[Tensor, ...], lyr: int = lyr) -> None:
            wF = None if F is None else _layers(F)[lyr].mlp(seen.pop(("mlp", lyr)))
            add(site_name(lyr, "down"), a[0], wF)

        hs += [layer.self_attn.register_forward_pre_hook(attn_in, with_kwargs=True),
               layer.mlp.register_forward_pre_hook(mlp_in), layer.self_attn.o_proj.register_forward_pre_hook(o_in),
               layer.mlp.down_proj.register_forward_pre_hook(down_in)]
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def _over(M: Any, F: Any | None, batches: Iterable[Tensor], add: Add) -> None:
    with _hooks(M, F, add):
        for toks in batches:
            M(toks, use_cache=False)


def grams(M: Any, batches: Iterable[Tensor]) -> dict[str, Tensor]:
    """G = sum a^M' a^M per site, fp64 on M's device, over every position of `batches`."""
    G: dict[str, Tensor] = {}

    def add(site: str, aM: Tensor, _wF: Tensor | None) -> None:
        a = aM.flatten(0, 1).to(F64)
        G[site] = G[site] + a.T @ a if site in G else a.T @ a

    _over(M, None, batches, add)
    return G


def cross(M: Any, F: Any, batches: Iterable[Tensor]) -> dict[str, tuple[Tensor, float]]:
    """(C, n_F) per site: C = sum w^F' a^M (fp64) and n_F = sum |w^F|^2, over every position of `batches`."""
    out: dict[str, tuple[Tensor, float]] = {}

    def add(site: str, aM: Tensor, wF: Tensor | None) -> None:
        assert wF is not None
        a, w = aM.flatten(0, 1).to(F64), wF.flatten(0, 1).to(F64)
        C, n = w.T @ a, float(w.pow(2).sum())
        out[site] = (out[site][0] + C, out[site][1] + n) if site in out else (C, n)

    _over(M, F, batches, add)
    return out


def pinv(G: Tensor) -> Tensor:
    return torch.linalg.pinv(G, hermitian=True)


def held_map(C: Tensor, P: Tensor) -> Tensor:
    """A = C G^+ (P = `pinv(G)`), fp64."""
    return C @ P


def residual(A: Tensor, C: Tensor, G: Tensor, nF: float) -> float:
    """|w^F - a^M A'|^2 / |w^F|^2 from one row set's C, G and n_F; nan when F writes nothing there."""
    err = nF - 2 * float((A * C).sum()) + float(((A @ G) * A).sum())
    return err / nF if nF > 0 else float("nan")


@contextmanager
def held(M: Any, maps: dict[str, Tensor]) -> Iterator[None]:
    """M as H for the block: each site's output matrix replaced by its map (cast to the weight's dtype), M's back on exit."""
    lins = {s: M.get_submodule("model." + s) for s in maps}
    keep = {s: lin.weight.data for s, lin in lins.items()}
    try:
        for s, lin in lins.items():
            lin.weight.data = maps[s].to(lin.weight.device, lin.weight.dtype)
        yield
    finally:
        for s, lin in lins.items():
            lin.weight.data = keep[s]


@torch.no_grad()
def kl_sums(M: Any, F: Any, maps: dict[str, Tensor], batches: Iterable[Tensor]) -> dict[str, float]:
    """Summed over every position of `batches`: KL(F || M), KL(F || H), and the largest |H logit - F logit| relative to
    the largest |F logit| of its row. 'n' is the number of positions."""
    out = {"kl_fm": 0.0, "kl_fh": 0.0, "logit_err": 0.0, "n": 0}
    for toks in batches:
        f = F(toks, use_cache=False).logits
        out["kl_fm"] += float(kl_per_position(f, M(toks, use_cache=False).logits).double().sum())
        with held(M, maps):
            h = M(toks, use_cache=False).logits
        out["kl_fh"] += float(kl_per_position(f, h).double().sum())
        err = ((h - f).abs().amax((1, 2)) / f.abs().amax((1, 2))).max()
        out["logit_err"] = max(out["logit_err"], float(err))
        out["n"] += toks.numel()
        del f, h
    return out


def max_rel_diff(A: Tensor, W: Tensor) -> float:
    """max |A - W| / max |W|, the do-nothing control's H = M check."""
    return float((A - W.to(A.device, F64)).abs().max() / W.abs().max())
