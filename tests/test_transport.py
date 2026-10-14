"""CPU checks of the Frobenius fit against a synthetic decomposition."""

import torch

from commute.transport import frobenius_fit


def test_frobenius_fit_recovers_alpha_in_span():
    torch.manual_seed(0)
    d_out, d_in, C = 12, 20, 30
    U = torch.randn(C, d_out)
    V = torch.randn(d_in, C)
    idx = torch.arange(C)
    alpha_true = torch.zeros(C)
    alpha_true[[3, 7, 11]] = torch.tensor([2.0, -1.0, 0.5])
    dW = torch.einsum("c,co,ic->oi", alpha_true, U, V)
    fit = frobenius_fit(dW, U, V, idx, ridge=0.0)
    assert torch.allclose(fit.alpha, alpha_true, atol=1e-4)
    assert abs(fit.fraction - 1.0) < 1e-6


def test_frobenius_fit_fraction_is_projection_fraction():
    torch.manual_seed(1)
    d_out, d_in, C = 8, 10, 5
    U = torch.randn(C, d_out)
    V = torch.randn(d_in, C)
    idx = torch.arange(C)
    dW = torch.randn(d_out, d_in)
    fit = frobenius_fit(dW, U, V, idx, ridge=0.0)
    recon = torch.einsum("c,co,ic->oi", fit.alpha, U, V)
    expected = 1 - ((dW - recon) ** 2).sum() / (dW**2).sum()
    assert abs(fit.fraction - float(expected)) < 1e-5
    assert 0.0 < fit.fraction < 1.0
