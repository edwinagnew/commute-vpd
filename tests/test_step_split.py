"""CPU checks of the step split (split.adjoint_step, split.StepSplit) on small random matrices, float64."""

import torch

from commute import split

D_OUT, D_IN, N = 10, 8, 3


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _setup(orthonormal: bool = False, seed: int = 0):
    g = _gen(seed)
    U = torch.randn(D_OUT, N, generator=g, dtype=torch.float64)
    V = torch.randn(D_IN, N, generator=g, dtype=torch.float64)
    if orthonormal:
        U, V = torch.linalg.qr(U)[0], torch.linalg.qr(V)[0]
    Delta0 = 0.1 * torch.randn(D_OUT, D_IN, generator=g, dtype=torch.float64)
    A = torch.randn(D_IN, D_IN, generator=g, dtype=torch.float64)
    Sigma = A @ A.T / D_IN + 0.1 * torch.eye(D_IN, dtype=torch.float64)
    return U, V, U @ V.T + Delta0, Sigma


def _run(U, V, W0, Sigma, steps, carry: bool) -> split.StepSplit:
    ss = split.StepSplit({"m": (U, V)}, {"m": Sigma}, {"m": W0}, carry, torch.float64, "cpu")
    W = W0.clone()
    for S in steps:
        W = W + S
        ss.step({"m": W})
    return ss


def _total(steps: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(steps).sum(0)


def _steps(n: int, scale: float, seed: int = 1, shape=(D_OUT, D_IN)) -> list[torch.Tensor]:
    g = _gen(seed)
    return [scale * torch.randn(*shape, generator=g, dtype=torch.float64) for _ in range(n)]


def test_in_span_update_leaves_delta_almost_unchanged():
    # orthonormal bases: an update U X V^T has K(S) = 2 S, so c = 1/2 reproduces it; only dU dV^T (second order) and the
    # bases' drift away from orthonormal reach Delta
    U, V, W0, Sigma = _setup(orthonormal=True)
    g = _gen(2)
    steps = [U @ (1e-4 * torch.randn(N, N, generator=g, dtype=torch.float64)) @ V.T for _ in range(10)]
    ss = _run(U, V, W0, Sigma, steps, carry=True)
    dW = _total(steps)
    assert float(ss.delta_change()["m"].norm() / dW.norm()) < 1e-3
    one = split.adjoint_step(steps[0], U, V, Sigma)[3]
    assert abs(one["c"] - 0.5) < 1e-12 and abs(one["R"] - 1) < 1e-12


def test_out_of_span_update_goes_wholly_to_delta():
    # W0 = 0 and W_t = 2^t M make every realised step W_t - W_(t-1) exactly 2^(t-1) M, so no rounding puts an in-span
    # part into it (with W0 != 0 the subtraction leaves eps |W| of in-span noise, and c fitted to noise is noise)
    U, V, _, Sigma = _setup()
    W0 = torch.zeros(D_OUT, D_IN, dtype=torch.float64)
    PU, PV = U @ torch.linalg.pinv(U), V @ torch.linalg.pinv(V)
    Io, Ii = torch.eye(D_OUT, dtype=torch.float64), torch.eye(D_IN, dtype=torch.float64)
    M = (Io - PU) @ _steps(1, 1e-2)[0] @ (Ii - PV)
    steps = [M] + [2.0 ** (t - 1) * M for t in range(1, 5)]
    ss = _run(U, V, W0, Sigma, steps, carry=True)
    assert torch.allclose(ss.delta_change()["m"], _total(steps), atol=1e-14)
    assert float(ss.dU["m"].abs().max()) < 1e-14 and float(ss.dV["m"].abs().max()) < 1e-14


def test_weights_equal_the_run_and_delta_books_the_second_order_term():
    U, V, W0, Sigma = _setup()
    steps = _steps(20, 1e-2)
    ss = _run(U, V, W0, Sigma, steps, carry=True)
    W = W0 + _total(steps)
    Delta = (W0 - U @ V.T) + ss.delta_change()["m"]
    assert torch.allclose((U + ss.dU["m"]) @ (V + ss.dV["m"]).T + Delta, W, atol=1e-12)
    # replay: Delta's change is the sum of each step's (S - c K(S) - dU dV^T) against that step's bases
    Uc, Vc, first, second = U.clone(), V.clone(), torch.zeros_like(W0), torch.zeros_like(W0)
    for S in steps:
        dU, dV, sec, _ = split.adjoint_step(S, Uc, Vc, Sigma)
        first += split.dsigma(dU, dV, Uc, Vc)
        second += sec
        Uc, Vc = Uc + dU, Vc + dV
    assert torch.allclose(ss.second["m"], second, atol=1e-14)
    assert torch.allclose(ss.delta_change()["m"], _total(steps) - first - second, atol=1e-12)


def test_fixed_bases_fixed_c_equals_one_shot_adjoint():
    # steps all proportional to one M have the same c (c is scale-invariant), so the fixed-bases variant's pieces are
    # the one-shot adjoint split of the finished update (split.adjoint_state)
    U, V, W0, Sigma = _setup()
    M = _steps(1, 1e-2)[0]
    steps = [a * M for a in (0.3, 1.0, -0.2, 2.5, 0.7)]
    ss = _run(U, V, W0, Sigma, steps, carry=False)
    dW = _total(steps)
    _, state = split.adjoint_state({"m": dW}, {"m": (U, V)}, {"m": Sigma})
    assert torch.allclose(U + ss.dU["m"], state["m"][1].double(), atol=1e-6)  # adjoint_state returns float32
    c = split.adjoint_scale(dW, U, V, Sigma)[0]
    aU, aV = split.adjoint_split(dW, U, V)
    assert torch.allclose(ss.dU["m"], c * aU, atol=1e-14) and torch.allclose(ss.dV["m"], c * aV, atol=1e-14)
    # and with per-step c on unrelated steps the two differ (the run reports by how much)
    ss2 = _run(U, V, W0, Sigma, _steps(5, 1e-2), carry=False)
    assert not torch.allclose(ss2.dU["m"], split.adjoint_scale(ss2.W["m"] - W0, U, V, Sigma)[0]
                              * split.adjoint_split(ss2.W["m"] - W0, U, V)[0], atol=1e-10)


def test_carry_differs_from_fixed_at_second_order():
    U, V, W0, Sigma = _setup()
    gaps = []
    for scale in (1e-3, 5e-4):
        steps = _steps(10, scale)
        a, b = _run(U, V, W0, Sigma, steps, True), _run(U, V, W0, Sigma, steps, False)
        gaps.append(float((a.dU["m"] - b.dU["m"]).norm() / b.dU["m"].norm()))
    # the bases move by O(step), so the pieces' relative gap is O(step): halving the steps halves it
    assert gaps[0] < 1e-2 and 1.9 < gaps[0] / gaps[1] < 2.1
