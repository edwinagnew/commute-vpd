"""commute.qwen_held's site algebra on small random tensors (no model): the map reproduces F's write when F's write is
linear in M's units, its residual is 0 then and 1 for a write orthogonal to them, and H swaps and restores weights."""

import torch

from commute import qwen_held as QH

F64 = torch.float64


def moments(aM: torch.Tensor, wF: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    return wF.T @ aM, aM.T @ aM, float(wF.pow(2).sum())


def test_linear_write_reproduced() -> None:
    gen = torch.Generator().manual_seed(0)
    aM, W = torch.randn(300, 12, generator=gen, dtype=F64), torch.randn(5, 12, generator=gen, dtype=F64)
    C, G, n = moments(aM, aM @ W.T)
    A = QH.held_map(C, QH.pinv(G))
    assert torch.allclose(A, W, atol=1e-10) and abs(QH.residual(A, C, G, n)) < 1e-10
    read = torch.randn(80, 12, generator=gen, dtype=F64)  # held-out rows: the map is exact there too
    assert abs(QH.residual(A, *moments(read, read @ W.T))) < 1e-10


def test_orthogonal_write_left_whole() -> None:
    gen = torch.Generator().manual_seed(1)
    aM = torch.randn(200, 6, generator=gen, dtype=F64)
    q = torch.linalg.qr(torch.cat([aM, torch.randn(200, 1, generator=gen, dtype=F64)], 1)).Q[:, -1:]
    C, G, n = moments(aM, q.repeat(1, 3))  # every write column orthogonal to M's units
    A = QH.held_map(C, QH.pinv(G))
    assert A.abs().max() < 1e-10 and abs(QH.residual(A, C, G, n) - 1) < 1e-10


def test_rank_deficient_units() -> None:
    """A unit that copies another: the pseudo-inverse still reproduces a write linear in the units."""
    gen = torch.Generator().manual_seed(2)
    a = torch.randn(100, 4, generator=gen, dtype=F64)
    aM = torch.cat([a, a[:, :1]], 1)
    W = torch.randn(3, 5, generator=gen, dtype=F64)
    C, G, n = moments(aM, aM @ W.T)
    A = QH.held_map(C, QH.pinv(G))
    assert torch.allclose(aM @ A.T, aM @ W.T, atol=1e-9) and abs(QH.residual(A, C, G, n)) < 1e-10


def test_held_swaps_and_restores() -> None:
    lin = torch.nn.Linear(3, 2, bias=False)
    m = torch.nn.Module()
    m.add_module("model", torch.nn.Module())
    m.get_submodule("model").add_module("lin", lin)
    before = lin.weight.data.clone()
    with QH.held(m, {"lin": torch.ones(2, 3, dtype=F64)}):
        assert torch.equal(lin.weight.data, torch.ones(2, 3))
    assert torch.equal(lin.weight.data, before)
    assert QH.max_rel_diff(before.double(), before) == 0.0
