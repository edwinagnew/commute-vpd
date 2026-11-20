"""commute.kstar_vpd on small random matrices (no model): the split is exact and the pieces rebuild W_M and W_F, the
analytic subcomponent grade equals the cos^2 of the explicit writes, and the search finds K* on a toy additive model."""

from __future__ import annotations

from typing import Any

import torch

from commute import kstar as K
from commute import kstar_vpd as KV
from commute import split as SP


def _pair(dw_scale: float, seed: int = 0) -> tuple[KV.Site, dict[str, float], torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    d_in, d_out, C = 6, 5, 9
    U, V, delta = torch.randn(C, d_out, generator=g), torch.randn(d_in, C, generator=g), 0.1 * torch.randn(d_out, d_in, generator=g)
    WM = U.T @ V.T + delta
    WF = WM + dw_scale * torch.randn(d_out, d_in, generator=g)
    b = SP.basis(U.T.double(), V.double(), 1e-6)
    s, chk = KV._site("w", 1, WM, WF, U, V, delta, b)
    return s, chk, WF


def test_split_exact_and_pieces_rebuild_both_weights() -> None:
    s, chk, _ = _pair(0.3)
    assert s.changed and chk["split_fp64"] < 1e-12 and chk["F_from_pieces"] < 1e-5 and chk["M_from_pieces"] < 1e-5


def test_norm_shares_with_cross_terms_sum_to_one() -> None:
    _, chk, _ = _pair(0.3, seed=2)
    feat = KV.Feat([], [], 2, set(), set(), {"w": chk})
    tot = KV.norm_shares(feat)["total"]
    assert tot is not None and abs(sum(tot.values()) - 1) < 1e-9 and abs(tot["cross_out2"]) < 1e-9


def test_no_change_gives_zero_pieces() -> None:
    s, chk, _ = _pair(0.0)
    assert not s.changed and not s.dU.any() and not s.dV.any() and torch.equal(s.rf, s.rm) and chk["outside_share"] == 0.0


def test_analytic_grade_matches_explicit_writes() -> None:
    s, _, _ = _pair(0.3, seed=1)
    y = torch.randn(2, 4, s.V.shape[0], dtype=torch.float64)
    y[0, 1] = 0  # a setting where both writes are exactly zero (counts 1)
    U, dU, V, dV = (t.double() for t in (s.U, s.dU, s.V, s.dV))
    norms = ((U**2).sum(1), (U * dU).sum(1), (dU**2).sum(1))
    got: dict[Any, torch.Tensor] = {}
    KV._sub(got, 0, s, y, slice(-1, None), norms)
    a, b = y @ V, y @ dV
    wM = a[..., None] * U  # (B, T, C, d_out)
    wF = a[..., None] * (U + dU) + b[..., None] * U
    want: dict[Any, torch.Tensor] = {}
    K._add(want, 0, (wF * wM).sum(-1), wF.pow(2).sum(-1), wM.pow(2).sum(-1), slice(-1, None))
    assert torch.allclose(got[0], want[0], rtol=1e-9, atol=1e-12)


def test_search_finds_the_additive_kstar() -> None:
    w = torch.tensor([5.0, 0.1, 3.0, 0.2, 1.0, 0.05, 0.4, 0.25])
    bar = 0.1 * float(w.sum())

    def ev(keep: torch.Tensor) -> float:
        return float(w[~keep].sum())

    f = KV.search(ev, lambda c: w.clone(), len(w), bar, reranks=1, cap=200, chunk_div=2, log=lambda _m: None)
    assert ev(torch.isin(torch.arange(len(w)), torch.tensor(f.polished))) <= bar
    assert sorted(f.polished) == [0, 2, 4] and not f.capped
