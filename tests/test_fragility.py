"""commute.fragility on small random matrices (CPU): the random move hits its norm, revert-all is W0, the halves split
each matrix as stated, and the stable rank of known matrices."""

from __future__ import annotations

import torch

from commute.fragility import (
    delta_w,
    importance_halves,
    random_move,
    revert,
    stable_rank,
)


def _uv() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(0)
    return {k: (torch.randn(d_out, n, generator=g) * su, torch.randn(d_in, n, generator=g) * sv)
            for k, d_out, d_in, n, su, sv in (("a", 7, 5, 4, 0.01, 3.0), ("b", 6, 9, 5, 1.0, 1.0))}


def test_random_move_matches_norm() -> None:
    uv = _uv()
    norms = {"a": 0.2, "b": 1.5}
    dW = delta_w(uv, random_move(uv, norms, seed=3))
    for k, t in norms.items():
        assert abs(float(dW[k].double().norm()) / t - 1) < 1e-5
    assert torch.equal(random_move(uv, norms, 3)["a"][0], random_move(uv, norms, 3)["a"][0])


def test_revert_all_and_none() -> None:
    uv = _uv()
    uv2 = random_move(uv, {"a": 0.2, "b": 1.5}, seed=1)
    every = {k: torch.ones(u.shape[1], dtype=torch.bool) for k, (u, _) in uv.items()}
    assert all(not M.any() for M in delta_w(uv, revert(uv, uv2, every)).values())
    none = revert(uv, uv2, {k: ~m for k, m in every.items()})
    assert all(torch.equal(none[k][0], uv2[k][0]) and torch.equal(none[k][1], uv2[k][1]) for k in uv)
    one = revert(uv, uv2, {"a": torch.tensor([True, False, False, False])})
    assert torch.equal(one["a"][0][:, 0], uv["a"][0][:, 0]) and torch.equal(one["a"][0][:, 1:], uv2["a"][0][:, 1:])
    assert torch.equal(one["b"][1], uv2["b"][1])


def test_importance_halves() -> None:
    g = {"a": torch.tensor([0.5, 0.1, 0.9, 0.0, 0.3]), "b": torch.tensor([1.0, 2.0])}
    h = importance_halves(g, seed=0)
    assert h["rare"]["a"].tolist() == [False, True, False, True, False]
    assert torch.equal(h["common"]["a"], ~h["rare"]["a"])
    assert int(h["random"]["a"].sum()) == 2 and int(h["random"]["b"].sum()) == 1 and h["rare"]["b"].tolist() == [True, False]


def test_stable_rank() -> None:
    assert abs(stable_rank(torch.outer(torch.arange(1.0, 5.0), torch.ones(3))) - 1) < 1e-12
    assert abs(stable_rank(torch.eye(4)) - 4) < 1e-12
