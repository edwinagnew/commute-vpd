"""Two affine maps written in order at two sites, last position: E156's variant b (cell P1.13), moved here on 2026-11-11
for E157 (cell P1.14) so that a planted edit, a saved fine-tune and a new fine-tune are read by one function. Same arithmetic
as E156's `two_sites`, `pair_edit`, `Ctx.z`, `Ctx.h` and `Ctx.fit_last`; E156's own script keeps its copy and is not changed.

The first map is fitted from the original's activations at s1 (every stream up to s1, every position) to the target's s1,
last position, and written there. The second is fitted on the run already edited at s1, from its activations up to s2, to
the target's s2, and written at s2. Penalties from the grid on validation (`variable_read.ridge_on`, `affine_of`). H = 1 - d /
gap, d the mean TV at the last position between the edited original and the target on the read windows.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from commute.hidden_maps import Edit, chunks, flat, run, tv, up_to, write_last
from commute.steering import Net
from commute.variable_read import Affine, Sizes, Target, affine_of, ridge_on

EditOf = Callable[[Tensor], Edit]


def z(net: Net, toks: Tensor, site: str, edit: Edit | None = None) -> Tensor:
    """The original's activations up to `site`, every position, with `edit` applied: (B, p) fp32."""
    return flat(run(net, toks, sites=up_to(site), edit=edit)[1], site)


def read_h(net: Net, read: Tensor, batch: int, T: Target, edit_of: EditOf, gap: float) -> float:
    """1 - d / gap, d the mean TV at the last position between `net` edited by edit_of(toks) and T, on `read`."""
    d, n = 0.0, 0
    for toks in chunks(read, batch):
        ln = run(net, toks, edit=edit_of(toks))[0][:, -1]
        d += float(tv(ln, T(toks)[0][:, -1]).sum())
        n += toks.shape[0]
    return 1 - d / n / gap


def fit_last(net: Net, cfg: Sizes, fit: Tensor, T: Target, site: str, z_of: Callable[[Tensor], Tensor]) -> Affine:
    """The ridge map from z_of(toks) to T's activation at `site`, last position; penalty from cfg.ridge_grid on validation."""
    dev = str(next(net.net.parameters()).device)
    Rg, Xv, Yv = ridge_on(fit, cfg.ridge_batch, z_of, lambda t: {site: T(t, (site,))[1][site][:, -1]}, dev)
    m, _ = affine_of(Rg, site, Xv.shape[1], Xv, Yv[site], cfg.ridge_grid)
    del Rg, Xv, Yv
    torch.cuda.empty_cache()
    return m


def pair_edit(net: Net, s1: str, m1: Affine, s2: str, m2: Affine | None) -> EditOf:
    """The first map written at s1; then (m2 given) the second, read from the run already edited at s1, written at s2."""

    def e(t: Tensor) -> Edit:
        e1: Edit = {s1: write_last(None, m1(z(net, t, s1)))}
        return e1 if m2 is None else {**e1, s2: write_last(None, m2(z(net, t, s2, e1)))}

    return e


def two_site_h(net: Net, cfg: Sizes, pairs: tuple[tuple[str, str], ...], fit: Tensor, read: Tensor, T: Target, gap: float,
               rec: dict[str, dict[str, float]], save: Callable[[], None], tick: Callable[[], None], log: Callable[[str], None]) -> None:
    """H for each pair (s1, s2) not yet in `rec`, written to rec["s1+s2"] with both penalties; `save()` after each pair,
    `tick()` before each (the caller's wall-limit check)."""
    firsts: dict[str, Affine] = {}
    for s1, s2 in pairs:
        key = f"{s1}+{s2}"
        if key in rec:
            continue
        tick()
        if s1 not in firsts:
            firsts[s1] = fit_last(net, cfg, fit, T, s1, lambda t, s1=s1: z(net, t, s1))
        first = pair_edit(net, s1, firsts[s1], s2, None)
        m2 = fit_last(net, cfg, fit, T, s2, lambda t, s2=s2, first=first: z(net, t, s2, first(t)))
        h = read_h(net, read, cfg.read_batch, T, pair_edit(net, s1, firsts[s1], s2, m2), gap)
        rec[key] = {"H": h, "lam1": firsts[s1].lam, "lam2": m2.lam}
        save()
        log(f"  two sites {key}: H {h:.4f}")
