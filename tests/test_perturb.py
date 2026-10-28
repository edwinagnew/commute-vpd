"""CPU checks of commute.perturb: the per-key write solves X a_k = u_k at minimum norm, and the bisection's three ends."""

from __future__ import annotations

import torch

from commute import perturb as PT


def test_key_write_solves_and_is_minimum_norm() -> None:
    gen = torch.Generator().manual_seed(0)
    A, U = torch.randn(12, 4, generator=gen), torch.randn(6, 4, generator=gen)
    X = PT.key_write(A, U)
    assert torch.allclose(X @ A.double(), U.double(), atol=1e-10)
    # minimum norm: X's rows lie in A's column span, so adding any solution of Z A = 0 only grows the norm
    Z = torch.randn(6, 12, generator=gen, dtype=torch.float64)
    Z = Z - Z @ A.double() @ torch.linalg.pinv(A.double())
    assert (X + Z).norm() > X.norm()


def test_match_ok_cap_and_no_match() -> None:
    m = PT.match(lambda x: 2 * x, 3.0, 10.0, 0.05)
    assert m["status"] == "ok" and abs(m["f"] / 3.0 - 1) <= 0.05
    assert PT.match(lambda x: x, 50.0, 10.0, 0.1)["status"] == "cap"
    assert PT.match(lambda x: 0.0 if x < 1 else 10.0, 5.0, 10.0, 0.01)["status"] == "no_match"
