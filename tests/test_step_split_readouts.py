"""CPU checks of commute.step_split (E25's read-outs of the step split), float64."""

import torch

from commute import split, step_split

D_OUT, D_IN, N = 10, 8, 3


def _setup(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    U = torch.randn(D_OUT, N, generator=g, dtype=torch.float64)
    V = torch.randn(D_IN, N, generator=g, dtype=torch.float64)
    W0 = U @ V.T + 0.1 * torch.randn(D_OUT, D_IN, generator=g, dtype=torch.float64)
    A = torch.randn(D_IN, D_IN, generator=g, dtype=torch.float64)
    Sigma = A @ A.T / D_IN + 0.1 * torch.eye(D_IN, dtype=torch.float64)
    steps = [1e-2 * torch.randn(D_OUT, D_IN, generator=g, dtype=torch.float64) for _ in range(6)]
    return U, V, W0, Sigma, steps


def _pair(U, V, W0, Sigma, steps):
    a = split.StepSplit({"m": (U, V)}, {"m": Sigma}, {"m": W0}, True, torch.float64, "cpu")
    b = split.StepSplit({"m": (U, V)}, {"m": Sigma}, {"m": W0}, False, torch.float64, "cpu")
    W = W0.clone()
    gaps = []
    for S in steps:
        W = W + S
        a.step({"m": W})
        b.step({"m": W})
        gaps.append(step_split.carry_fixed_gap(a, b))
    return a, b, gaps


def test_carry_and_fixed_agree_after_one_step_only():
    U, V, W0, Sigma, steps = _setup()
    _, _, gaps = _pair(U, V, W0, Sigma, steps)
    assert gaps[0] == 0.0 and gaps[-1] > 1e-6


def test_numbers_match_the_definitions():
    U, V, W0, Sigma, steps = _setup()
    a, b, _ = _pair(U, V, W0, Sigma, steps)
    n = step_split.numbers(a, b, {"m": (U, V)}, {"m": Sigma})
    dW, dd = a.W["m"] - W0, a.delta_change()["m"]
    assert abs(n["delta_share"] - float((dd**2).sum() / (dW**2).sum())) < 1e-12
    assert abs(n["delta_share_weighted"] - split.weighted_energy(dd, Sigma) / split.weighted_energy(dW, Sigma)) < 1e-12
    assert abs(n["delta_share_weighted_one_shot"] - (1 - split.adjoint_scale(dW, U, V, Sigma)[1])) < 1e-12
