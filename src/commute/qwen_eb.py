"""The write share E_b on Qwen3-1.7B's VPD featurization, from second moments of each decomposed matrix's input (cell
P3.14, `experiments/qwen_eb`; added 2026-11-22). Changes no shared function.

Variables and mechanisms are the 67M's (`commute.kstar_vpd` docstring): per decomposed matrix its C subcomponents and
one remainder; the update's delta-W of that matrix is split by the least-squares split over all C subcomponents (fp64,
`commute.split.least_squares_split`), and what the split leaves, delta-W_out, goes to the remainder. The site is built
by `kstar_vpd._site` unchanged (so the split, its exactness checks and the `Site` fields are the 67M's), with Qwen's
remainder Delta = W0 - (V U)^T (p-9707d6ff stores no Delta; `qwen_vpd` docstring).

What differs from `stand_in.VPDSite.gain_sums` / `change_sums`: the same sums, summed over every position, written
through the input's second moment S = sum over positions of y^T y instead of the positions themselves. With a = y V,
b = y dV (per subcomponent c), sum a^2 = v_c^T S v_c, sum a b = v_c^T S dv_c, sum b^2 = dv_c^T S dv_c, and the remainder's
sums are traces against S. Why: holding every site's dU, dV and remainders at once beside the decomposition (19.4 GB fp32)
and the model (8 GB) needs ~58 GB of a 44 GB L40, and VPDSite's fp64 products on 16,384 positions take ~4-6 min per
update at an L40's fp64 rate; S is 5.6 GB (fp32) for all 196 matrices (q, k, v share one input, gate and up another).
S and the matmuls against it are fp32 (TF32 off); the products are summed in fp64. num is returned as den + (num - den),
with num - den computed directly (sum a b |u|^2 + sum a^2 u.du), so `stand_in.explained`'s num - den has no cancellation.
`reuse_check` reads VPDSite's own sums on one site's positions beside these.

The split's bases (`split.gram_basis`, fp64 on the GPU) depend on the decomposition only: computed once per job and held
on the host with P and Q in fp32 (17 GB), cast back to fp64 for each split. That moves the split off the exact
least-squares one by fp32 rounding of the bases only; it stays exact (delta-W = write side + read side + delta-W_out by
construction of delta-W_out), and `_site`'s checks report it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from commute import kstar_vpd as KV
from commute import split as SP
from commute import stand_in as ST
from commute.qwen_vpd import parse_site, site_name

SHARED_INPUT = {"q": "q", "k": "q", "v": "q", "o": "o", "gate": "gate", "up": "gate", "down": "down"}


def input_of(site: str) -> str:
    """The site whose input this site reads (q, k, v read one input; gate and up another)."""
    layer, kind = parse_site(site)
    return site_name(layer, SHARED_INPUT[kind])


def host_basis(U: Tensor, V: Tensor, tol: float) -> SP.Basis:
    """The split's bases of one matrix (param-decomp layout U (C, d_out), V (d_in, C)), P and Q in fp32 on the host."""
    b = SP.basis_from(SP.gram_basis(U.T), SP.gram_basis(V), tol)
    return SP.Basis(b.P.float().cpu(), b.s.cpu(), b.Q.float().cpu(), b.t.cpu(), b.r_u, b.r_v)


def on(b: SP.Basis, dev: torch.device | str) -> SP.Basis:
    return SP.Basis(b.P.to(dev).double(), b.s.to(dev), b.Q.to(dev).double(), b.t.to(dev), b.r_u, b.r_v)


def site(name: str, WM: Tensor, WF: Tensor, U: Tensor, V: Tensor, b: SP.Basis) -> tuple[KV.Site, dict[str, float]]:
    """`kstar_vpd._site` with Qwen's remainder Delta = W_M - (V U)^T (fp64), variable offset 0."""
    delta = WM.double() - (V.double() @ U.double()).T
    return KV._site(name, 0, WM, WF, U, V, delta, b)


@torch.no_grad()
def moments(model: Any, batches: list[Tensor], sites: list[str], bf16: bool, keep: str | None = None
            ) -> tuple[dict[str, Tensor], Tensor | None]:
    """S = sum over every position of y^T y (fp32, on the device) for each distinct input of `sites`, in the model's
    own forward over `batches` (the base transformer only: no logits are needed). bf16: the forward under bf16
    autocast, S itself in fp32. `keep`: an input site whose positions are also returned, for `reuse_check`."""
    S: dict[str, Tensor] = {}
    kept: list[Tensor] = []

    def hook(name: str, _m: Any, a: tuple[Tensor, ...]) -> None:
        with torch.autocast("cuda", enabled=False):
            y = a[0].flatten(0, -2).float()
            S[name] = S[name] + y.T @ y if name in S else y.T @ y
            if name == keep:
                kept.append(y.clone())

    hs = [model.get_submodule(f"model.{s}").register_forward_pre_hook(lambda m, a, s=s: hook(s, m, a))
          for s in sorted({input_of(s) for s in sites})]
    try:
        for x in batches:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                model.model(input_ids=x)
    finally:
        for h in hs:
            h.remove()
    return S, torch.cat(kept) if kept else None


def _dot(a: Tensor, b: Tensor, dim: int | None) -> Tensor:
    p = a.double() * b.double()
    return p.sum() if dim is None else p.sum(dim)


def moment_sums(s: KV.Site, S: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(num, den, dd), each (C + 1,) fp64 on the host: `VPDSite.gain_sums` and `change_sums` of the site, summed over
    the positions whose second moment is S (d_in, d_in)."""
    SV, SdV = S @ s.V, S @ s.dV
    A, AB, B = _dot(SV, s.V, 0), _dot(SV, s.dV, 0), _dot(SdV, s.dV, 0)
    uu, ud, du2 = _dot(s.U, s.U, 1), _dot(s.U, s.dU, 1), _dot(s.dU, s.dU, 1)
    out = s.rf - s.rm  # delta-W_out^T
    Srm = S @ s.rm
    den = torch.cat([A * uu, _dot(Srm, s.rm, None).reshape(1)])
    diff = torch.cat([AB * uu + A * ud, _dot(Srm, out, None).reshape(1)])
    dd = torch.cat([B * uu + 2 * AB * ud + A * du2, _dot(S @ out, out, None).reshape(1)])
    return (den + diff).cpu(), den.cpu(), dd.cpu()


def reuse_check(s: KV.Site, S: Tensor, y: Tensor) -> dict[str, float]:
    """The site's explained write change and dd from `moment_sums` beside `stand_in.VPDSite`'s on the positions y
    themselves (fp64), relative differences; reported, not a stop."""
    num, den, dd = moment_sums(s, S)
    v = ST.VPDSite(s)
    gn, gd = v.gain_sums(y)
    rd = v.change_sums(y).cpu()
    got = {"explained": float(ST.explained(num, den).sum()), "dd": float(dd.sum())}
    ref = {"explained": float(ST.explained(gn.cpu(), gd.cpu()).sum()), "dd": float(rd.sum())}
    return {f"{k}_rel_diff": abs(got[k] - ref[k]) / abs(ref[k]) if ref[k] else abs(got[k]) for k in got} | {
        f"{k}_vpdsite": ref[k] for k in ref}


def gain_change(U: Tensor, V: Tensor, chosen: Tensor) -> Tensor:
    """delta-W (d_out, d_in) = sum over the chosen subcomponents c of u_c v_c^T: each writes twice what it writes in M."""
    return (V[:, chosen] @ U[chosen]).T


Change = Callable[[str], Tensor | None]  # a site's delta-W (any device), None for zero
