"""commute.held_model's site arithmetic on small random tensors (no model): planted gains and a shuffle are recovered,
and every reading's matrix reproduces F's write when F's units are gains of M's."""

import torch

from commute import held_model as HM

GAINS = torch.tensor([1.5, -0.7, 2.0, -1.2, 0.6], dtype=torch.float64)


def site(k: int, zero: int | None = None) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Five units of k columns; F's unit i is M's unit perm[i] times GAINS[i]; F's unit `zero` writes nothing."""
    gen = torch.Generator().manual_seed(0)
    aM, WM = torch.randn(400, 5 * k, generator=gen, dtype=torch.float64), torch.randn(7, 5 * k, generator=gen, dtype=torch.float64)
    perm = torch.randperm(5, generator=gen)
    cols = (perm[:, None] * k + torch.arange(k)).flatten()
    aF, WF = aM[:, cols], WM[:, cols] * GAINS.repeat_interleave(k)
    if zero is not None:
        WF[:, zero * k : (zero + 1) * k] = 0
    return {"FF": aF.T @ aF, "FM": aF.T @ aM, "MM": aM.T @ aM}, aF, aM, WF, WM, perm


def test_gains_and_shuffle_recovered() -> None:
    for k in (1, 3):
        G, aF, aM, WF, WM, perm = site(k)
        p = HM.pair(*HM.unit_sums(G, WF, WM, k))
        assert torch.equal(p.partner, perm) and torch.allclose(p.gain, GAINS) and p.err.abs().max() < 1e-9
        for A in (HM.held_matrix(p, WM, k), HM.least_squares(G, WF)):
            assert torch.allclose(aM @ A.T, aF @ WF.T, atol=1e-8) and abs(HM.residual(G, WF, A)) < 1e-9


def test_zero_unit_left_out_and_its_m_unit_at_gain_zero() -> None:
    G, aF, aM, WF, WM, perm = site(1, zero=2)
    p = HM.pair(*HM.unit_sums(G, WF, WM, 1))
    assert p.partner[2] == -1 and p.gain[2].isnan()
    A = HM.held_matrix(p, WM, 1)
    assert not A[:, perm[2]].any() and torch.allclose(aM @ A.T, aF @ WF.T, atol=1e-8)


def test_block_reading_is_one_gain() -> None:
    _, _, aM, _, WM, _ = site(1)
    MM = aM.T @ aM
    p = HM.pair(*HM.unit_sums({"FF": MM, "FM": MM, "MM": MM}, WM * -1.3, WM, 5))  # F's block = M's block times -1.3
    assert p.partner.tolist() == [0] and abs(float(p.gain[0]) + 1.3) < 1e-12
    assert HM.survival(0.0, 0.5, 1e-6) == 1.0 and HM.survival(0.0, 1e-7, 1e-6) is None
