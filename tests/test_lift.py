"""CPU checks of the lift algebra on small random matrices."""

import torch

from commute import lift


def _setup(d_out=12, d_in=9, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    U_A = torch.randn(d_out, n, generator=g, dtype=torch.float64)
    V_A = torch.randn(d_in, n, generator=g, dtype=torch.float64)
    M = torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
    return U_A, V_A, M


def test_ranks_and_null():
    U_A, V_A, _ = _setup()
    b = lift.basis(U_A, V_A, 1e-6)
    assert (b.r_u, b.r_v) == (4, 4)
    assert abs(lift.gaussian_null(b, 12, 9) - (8 / 12) * (5 / 9)) < 1e-12


def test_obstruction_is_projection_orthogonal_to_image():
    U_A, V_A, M = _setup()
    b = lift.basis(U_A, V_A, 1e-6)
    O = lift.obstruction(M, b)
    PU = U_A @ torch.linalg.pinv(U_A)
    PV = V_A @ torch.linalg.pinv(V_A)
    I_o, I_i = torch.eye(12, dtype=M.dtype), torch.eye(9, dtype=M.dtype)
    assert torch.allclose(O, (I_o - PU) @ M @ (I_i - PV), atol=1e-10)
    assert abs(lift.obstruction_fraction(M, b) - float((O**2).sum() / (M**2).sum())) < 1e-12
    # anything in the image of d sigma is orthogonal to O
    dU, dV = torch.randn(12, 4, dtype=M.dtype), torch.randn(9, 4, dtype=M.dtype)
    assert abs(float((lift.dsigma(dU, dV, U_A, V_A) * O).sum())) < 1e-9


def test_euclidean_lift_reproduces_the_liftable_part_with_minimum_norm():
    U_A, V_A, M = _setup()
    b = lift.basis(U_A, V_A, 1e-6)
    dU, dV = lift.euclidean_lift(M, b, U_A, V_A)
    target = M - lift.obstruction(M, b)
    assert torch.allclose(lift.dsigma(dU, dV, U_A, V_A), target, atol=1e-9)
    # minimum norm: adding a kernel direction (gauge u -> u + e u, v -> v - e v) cannot shrink it
    for c in range(4):
        kU, kV = torch.zeros_like(dU), torch.zeros_like(dV)
        kU[:, c], kV[:, c] = U_A[:, c], -V_A[:, c]
        assert torch.allclose(lift.dsigma(kU, kV, U_A, V_A), torch.zeros_like(M), atol=1e-12)
        assert abs(float((dU * kU).sum() + (dV * kV).sum())) < 1e-9


def test_saturated_side_has_no_obstruction():
    U_A, V_A, M = _setup(d_out=5, d_in=9, n=7)
    b = lift.basis(U_A, V_A, 1e-6)
    assert b.r_u == 5 and lift.obstruction_fraction(M, b) < 1e-24
    assert lift.dsigma_smin(b) > 0


def test_lora_hook_equals_weight_update():
    from torch import nn

    from commute.finetune import lora_delta, lora_factors, lora_hooks

    torch.manual_seed(0)
    lin = nn.Linear(6, 5, bias=False)
    f = lora_factors({"m": lin}, rank=2, seed=0)
    with torch.no_grad():
        f["m"][0].normal_()  # B non-zero
    x = torch.randn(3, 6)
    W0 = {"m": lin.weight.detach().clone(), "other": torch.zeros(2, 2)}
    h = lora_hooks({"m": lin}, f)
    y = lin(x)
    for handle in h:
        handle.remove()
    d = lora_delta(f, ["m", "other"], W0)
    assert torch.allclose(y, x @ (W0["m"] + d["m"]).T, atol=1e-5)
    assert torch.linalg.matrix_rank(d["m"]) == 2 and not d["other"].any()
    assert torch.allclose(lin(x), x @ W0["m"].T)  # hooks removed


def test_weighted_fractions_reduce_to_omega_with_identity_and_match_monte_carlo():
    U_A, V_A, M = _setup()
    b = lift.basis(U_A, V_A, 1e-6)
    w = lift.weighted_fractions(M, b, torch.eye(9, dtype=M.dtype))
    assert abs(w["omega_act"] - lift.obstruction_fraction(M, b)) < 1e-12
    assert abs(w["null_act"] - lift.gaussian_null(b, 12, 9)) < 1e-12
    g = torch.Generator().manual_seed(1)
    a = torch.randn(20000, 9, generator=g, dtype=M.dtype) * torch.linspace(0.1, 3, 9, dtype=M.dtype)
    S = a.T @ a / a.shape[0]
    w = lift.weighted_fractions(M, b, S)
    O = lift.obstruction(M, b)
    mc = float(((a @ O.T) ** 2).sum() / ((a @ M.T) ** 2).sum())
    assert abs(w["omega_act"] - mc) < 1e-9


def test_pullback_hook_equals_weight_change_and_keeps_delta():
    from torch import nn

    from commute import pullback

    torch.manual_seed(0)
    lin = nn.Linear(7, 5, bias=False)
    U_A, V_A = torch.randn(5, 3), torch.randn(7, 3)
    off = pullback.Offsets({"m": (U_A, V_A)})
    with torch.no_grad():
        off.d("m")[0].normal_()
        off.d("m")[1].normal_()
    x = torch.randn(4, 7)
    h = pullback.hooks({"m": lin}, off)
    y = lin(x)
    for handle in h:
        handle.remove()
    W = lin.weight.detach()
    assert torch.allclose(y, x @ (W + off.delta_w("m")).T, atol=1e-4)
    U2, V2 = off.lifted("m")
    # the new weight minus the new component sum equals the old weight minus the old component sum
    assert torch.allclose((W + off.delta_w("m")) - U2 @ V2.T, W - U_A @ V_A.T, atol=1e-4)


def test_lifted_components_leave_delta_plus_obstruction_with_target_moved():
    # full decomposition: 4 alive + 2 dead columns, Delta_0 non-zero; target weights W0 + M
    U_A, V_A, M = _setup()
    g = torch.Generator().manual_seed(3)
    U_d, V_d = torch.randn(12, 2, generator=g, dtype=M.dtype), torch.randn(9, 2, generator=g, dtype=M.dtype)
    Delta0 = 0.1 * torch.randn(12, 9, generator=g, dtype=M.dtype)
    W0 = U_A @ V_A.T + U_d @ V_d.T + Delta0
    b = lift.basis(U_A, V_A, 1e-6)
    U2, V2, dd = lift.lifted_with_delta_change(M, b, U_A, V_A)
    Delta_new = (W0 + M) - U2 @ V2.T - U_d @ V_d.T
    assert torch.allclose(Delta_new - Delta0, dd, atol=1e-10)
    O = lift.obstruction(M, b)
    # first order: the change is the obstruction; the second-order term is the rest
    assert float(((dd - O) ** 2).sum()) < float((O**2).sum())


def test_rank_r_offsets_hook_equals_weight_change_rank_bound_and_expressivity():
    from torch import nn

    from commute import pullback

    torch.manual_seed(0)
    d_out, d_in, n, r = 9, 7, 5, 2
    lin = nn.Linear(d_in, d_out, bias=False)
    U_A, V_A = torch.randn(d_out, n), torch.randn(d_in, n)
    off = pullback.Offsets({"m": (U_A, V_A)}, rank=r, seed=0)
    dU0, dV0 = off.d("m")
    assert not dU0.detach().any() and not dV0.detach().any()  # the product starts at zero
    assert [p.shape for p in off.parameters()] == [(d_out, r), (n, r), (d_in, r), (n, r)]
    with torch.no_grad():
        for p in off.parameters():
            p.normal_()
    x = torch.randn(4, d_in)
    h = pullback.hooks({"m": lin}, off)
    y = lin(x)
    for handle in h:
        handle.remove()
    W = lin.weight.detach()
    dW = off.delta_w("m")
    assert torch.allclose(y, x @ (W + dW).T, atol=1e-4)
    U2, V2 = off.lifted("m")
    assert torch.allclose((W + dW) - U2 @ V2.T, W - U_A @ V_A.T, atol=1e-4)  # Delta unchanged
    assert torch.linalg.matrix_rank(dW) <= 2 * r
    # expressivity: with rank >= n any free offset is reachable (S of rank n; P from a least-squares solve)
    target = torch.randn(d_out, n)
    S = torch.randn(n, n)
    P = torch.linalg.solve(S, target.T).T  # P S^T = target
    assert torch.allclose(P @ S.T, target, atol=1e-4)


def test_lift_state_matches_lifted_with_delta_change_and_skips_zero_matrices():
    U_A, V_A, M = _setup()
    uv = {"a": (U_A.float(), V_A.float()), "b": (U_A.float(), V_A.float())}
    b = lift.bases(uv, 1e-6)
    st = lift.lift_state({"a": M.float(), "b": torch.zeros_like(M).float()}, uv, b)
    assert set(st) == {"a"}
    dW, U2, V2 = st["a"]
    assert torch.equal(dW, M.float())
    U2_ref, V2_ref, _ = lift.lifted_with_delta_change(M.float(), b["a"], U_A.float().double(), V_A.float().double())
    assert torch.allclose(U2.double(), U2_ref, atol=1e-5) and torch.allclose(V2.double(), V2_ref, atol=1e-5)
    # the lifted components reproduce the liftable part of M to first order
    liftable = M.float() - lift.obstruction(M.float(), b["a"]).float()
    first_order = (U2 - U_A.float()) @ V_A.float().T + U_A.float() @ (V2 - V_A.float()).T
    assert torch.allclose(first_order, liftable, atol=1e-3)
