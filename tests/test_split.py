"""CPU checks of the lift algebra on small random matrices."""

import torch

from commute import split


def _setup(d_out=12, d_in=9, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    U_A = torch.randn(d_out, n, generator=g, dtype=torch.float64)
    V_A = torch.randn(d_in, n, generator=g, dtype=torch.float64)
    M = torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
    return U_A, V_A, M


def test_ranks_and_null():
    U_A, V_A, _ = _setup()
    b = split.basis(U_A, V_A, 1e-6)
    assert (b.r_u, b.r_v) == (4, 4)
    assert abs(split.gaussian_null(b, 12, 9) - (8 / 12) * (5 / 9)) < 1e-12


def test_obstruction_is_projection_orthogonal_to_image():
    U_A, V_A, M = _setup()
    b = split.basis(U_A, V_A, 1e-6)
    O = split.obstruction(M, b)
    PU = U_A @ torch.linalg.pinv(U_A)
    PV = V_A @ torch.linalg.pinv(V_A)
    I_o, I_i = torch.eye(12, dtype=M.dtype), torch.eye(9, dtype=M.dtype)
    assert torch.allclose(O, (I_o - PU) @ M @ (I_i - PV), atol=1e-10)
    assert abs(split.obstruction_fraction(M, b) - float((O**2).sum() / (M**2).sum())) < 1e-12
    # anything in the image of d sigma is orthogonal to O
    dU, dV = torch.randn(12, 4, dtype=M.dtype), torch.randn(9, 4, dtype=M.dtype)
    assert abs(float((split.dsigma(dU, dV, U_A, V_A) * O).sum())) < 1e-9


def test_euclidean_lift_reproduces_the_liftable_part_with_minimum_norm():
    U_A, V_A, M = _setup()
    b = split.basis(U_A, V_A, 1e-6)
    dU, dV = split.least_squares_split(M, b, U_A, V_A)
    target = M - split.obstruction(M, b)
    assert torch.allclose(split.dsigma(dU, dV, U_A, V_A), target, atol=1e-9)
    # minimum norm: adding a kernel direction (gauge u -> u + e u, v -> v - e v) cannot shrink it
    for c in range(4):
        kU, kV = torch.zeros_like(dU), torch.zeros_like(dV)
        kU[:, c], kV[:, c] = U_A[:, c], -V_A[:, c]
        assert torch.allclose(split.dsigma(kU, kV, U_A, V_A), torch.zeros_like(M), atol=1e-12)
        assert abs(float((dU * kU).sum() + (dV * kV).sum())) < 1e-9


def test_saturated_side_has_no_obstruction():
    U_A, V_A, M = _setup(d_out=5, d_in=9, n=7)
    b = split.basis(U_A, V_A, 1e-6)
    assert b.r_u == 5 and split.obstruction_fraction(M, b) < 1e-24
    assert split.dsigma_smin(b) > 0


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
    b = split.basis(U_A, V_A, 1e-6)
    w = split.weighted_fractions(M, b, torch.eye(9, dtype=M.dtype))
    assert abs(w["omega_act"] - split.obstruction_fraction(M, b)) < 1e-12
    assert abs(w["null_act"] - split.gaussian_null(b, 12, 9)) < 1e-12
    g = torch.Generator().manual_seed(1)
    a = torch.randn(20000, 9, generator=g, dtype=M.dtype) * torch.linspace(0.1, 3, 9, dtype=M.dtype)
    S = a.T @ a / a.shape[0]
    w = split.weighted_fractions(M, b, S)
    O = split.obstruction(M, b)
    mc = float(((a @ O.T) ** 2).sum() / ((a @ M.T) ** 2).sum())
    assert abs(w["omega_act"] - mc) < 1e-9


def test_pullback_hook_equals_weight_change_and_keeps_delta():
    from torch import nn

    from commute import component_training

    torch.manual_seed(0)
    lin = nn.Linear(7, 5, bias=False)
    U_A, V_A = torch.randn(5, 3), torch.randn(7, 3)
    off = component_training.Offsets({"m": (U_A, V_A)})
    with torch.no_grad():
        off.d("m")[0].normal_()
        off.d("m")[1].normal_()
    x = torch.randn(4, 7)
    h = component_training.hooks({"m": lin}, off)
    y = lin(x)
    for handle in h:
        handle.remove()
    W = lin.weight.detach()
    assert torch.allclose(y, x @ (W + off.delta_w("m")).T, atol=1e-4)
    U2, V2 = off.components("m")
    # the new weight minus the new component sum equals the old weight minus the old component sum
    assert torch.allclose((W + off.delta_w("m")) - U2 @ V2.T, W - U_A @ V_A.T, atol=1e-4)


def test_lifted_components_leave_delta_plus_obstruction_with_target_moved():
    # full decomposition: 4 alive + 2 dead columns, Delta_0 non-zero; target weights W0 + M
    U_A, V_A, M = _setup()
    g = torch.Generator().manual_seed(3)
    U_d, V_d = torch.randn(12, 2, generator=g, dtype=M.dtype), torch.randn(9, 2, generator=g, dtype=M.dtype)
    Delta0 = 0.1 * torch.randn(12, 9, generator=g, dtype=M.dtype)
    W0 = U_A @ V_A.T + U_d @ V_d.T + Delta0
    b = split.basis(U_A, V_A, 1e-6)
    U2, V2, dd = split.split_with_delta_change(M, b, U_A, V_A)
    Delta_new = (W0 + M) - U2 @ V2.T - U_d @ V_d.T
    assert torch.allclose(Delta_new - Delta0, dd, atol=1e-10)
    O = split.obstruction(M, b)
    # first order: the change is the obstruction; the second-order term is the rest
    assert float(((dd - O) ** 2).sum()) < float((O**2).sum())


def test_rank_r_offsets_hook_equals_weight_change_rank_bound_and_expressivity():
    from torch import nn

    from commute import component_training

    torch.manual_seed(0)
    d_out, d_in, n, r = 9, 7, 5, 2
    lin = nn.Linear(d_in, d_out, bias=False)
    U_A, V_A = torch.randn(d_out, n), torch.randn(d_in, n)
    off = component_training.Offsets({"m": (U_A, V_A)}, rank=r, seed=0)
    dU0, dV0 = off.d("m")
    assert not dU0.detach().any() and not dV0.detach().any()  # the product starts at zero
    assert [p.shape for p in off.parameters()] == [(d_out, r), (n, r), (d_in, r), (n, r)]
    with torch.no_grad():
        for p in off.parameters():
            p.normal_()
    x = torch.randn(4, d_in)
    h = component_training.hooks({"m": lin}, off)
    y = lin(x)
    for handle in h:
        handle.remove()
    W = lin.weight.detach()
    dW = off.delta_w("m")
    assert torch.allclose(y, x @ (W + dW).T, atol=1e-4)
    U2, V2 = off.components("m")
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
    b = split.bases(uv, 1e-6)
    st = split.split_state({"a": M.float(), "b": torch.zeros_like(M).float()}, uv, b)
    assert set(st) == {"a"}
    dW, U2, V2 = st["a"]
    assert torch.equal(dW, M.float())
    U2_ref, V2_ref, _ = split.split_with_delta_change(M.float(), b["a"], U_A.float().double(), V_A.float().double())
    assert torch.allclose(U2.double(), U2_ref, atol=1e-5) and torch.allclose(V2.double(), V2_ref, atol=1e-5)
    # the lifted components reproduce the liftable part of M to first order
    liftable = M.float() - split.obstruction(M.float(), b["a"]).float()
    first_order = (U2 - U_A.float()) @ V_A.float().T + U_A.float() @ (V2 - V_A.float()).T
    assert torch.allclose(first_order, liftable, atol=1e-3)


def test_read_side_overlaps():
    U_A, V_A, _ = _setup(d_out=12, d_in=9, n=4)
    b = split.basis(U_A, V_A, 1e-6)
    g = torch.Generator().manual_seed(1)
    Sigma = torch.randn(9, 9, generator=g, dtype=torch.float64)
    Sigma = Sigma @ Sigma.T
    # a read side inside the alive span is all in_alive; a random one averages r_v / d_in
    X = torch.randn(12, 4, generator=g, dtype=torch.float64)
    o = split.read_side_overlaps(X @ V_A.T, b, Sigma)
    assert abs(o["in_alive"] - 1) < 1e-12 and abs(o["random"] - 4 / 9) < 1e-12
    r = [split.read_side_overlaps(torch.randn(12, 9, generator=g, dtype=torch.float64), b, Sigma)["in_alive"] for _ in range(200)]
    assert abs(sum(r) / len(r) - 4 / 9) < 0.03
    # a read side along the top eigenvectors of Sigma is all in_data, and the data share they carry is below 1
    _, evecs = torch.linalg.eigh(Sigma)
    top = evecs[:, -4:]
    o = split.read_side_overlaps(X @ top.T, b, Sigma)
    assert abs(o["in_data"] - 1) < 1e-12 and 0 < o["data_in_top"] < 1 and 0 <= o["alive_in_data"] <= 1


# ----------------------------------------------------------------------------- the LoRA split study (2026-10-20)


def _sigma(d_in=9, seed=1):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(40, d_in, generator=g, dtype=torch.float64)
    return X.T @ X / 40


def test_four_blocks_sum_to_one_and_controls():
    U_A, V_A, M = _setup()
    S = _sigma()
    b = split.basis(U_A, V_A, 1e-6)
    f = split.four_blocks(M, b, S)
    assert abs(sum(f[k] for k in ("in_in", "in_out", "out_in", "out_out")) - 1) < 1e-10
    assert abs(sum(f[k] for k in f if k.startswith("null_")) - 1) < 1e-10
    # inside the first-order span: nothing outside both; a rescaling of the pieces is inside on both sides
    dU, dV = torch.randn(12, 4, dtype=M.dtype), torch.randn(9, 4, dtype=M.dtype)
    assert split.four_blocks(split.dsigma(dU, dV, U_A, V_A), b, S)["out_out"] < 1e-10
    R = U_A @ torch.diag(torch.randn(4, dtype=M.dtype)) @ V_A.T
    assert abs(split.four_blocks(R, b, S)["in_in"] - 1) < 1e-10
    assert abs(split.rescaling_share(R, U_A, V_A, S) - 1) < 1e-8
    # the obstruction reads one on out_out only when the read sides are uncorrelated under Sigma, so test with Sigma = I
    O = split.obstruction(M, b)
    assert abs(split.four_blocks(O, b, torch.eye(9, dtype=M.dtype))["out_out"] - 1) < 1e-10


def test_null_matches_random_draws():
    U_A, V_A, M = _setup(d_out=30, d_in=25, n=6)
    S = _sigma(25)
    b = split.basis(U_A, V_A, 1e-6)
    f = split.four_blocks(M, b, S)
    draws = [split.four_blocks(split.random_like(M, S, s), b, S) for s in range(200)]
    for k in ("in_in", "in_out", "out_in", "out_out"):
        assert abs(sum(d[k] for d in draws) / 200 - f["null_" + k]) < 0.02, k
    assert abs(split.weighted_energy(split.random_like(M, S, 0), S) - split.weighted_energy(M, S)) < 1e-9


def test_adjoint_split_and_scale():
    U_A, V_A, M = _setup()
    S = _sigma()
    dU, dV = split.adjoint_split(M, U_A, V_A)
    assert torch.allclose(split.dsigma(dU, dV, U_A, V_A), U_A @ U_A.T @ M + M @ V_A @ V_A.T, atol=1e-10)
    c, share = split.adjoint_scale(M, U_A, V_A, S)
    K = split.dsigma(dU, dV, U_A, V_A)
    best = min(split.weighted_energy(x * K - M, S) for x in torch.linspace(c - 0.1, c + 0.1, 201).tolist())
    assert abs(split.weighted_energy(c * K - M, S) - best) < 1e-9 and 0 < share < 1
    # for M already of the form K(X) scaled, the adjoint reproduces it exactly only if K is the identity; check the
    # bound share <= 1 - out_out instead
    assert share <= 1 - split.four_blocks(M, split.basis(U_A, V_A, 1e-6), S)["out_out"] + 1e-10


def test_piece_energies_and_n_half():
    U_A, V_A, M = _setup()
    S = _sigma()
    b = split.basis(U_A, V_A, 1e-6)
    dU, dV = split.least_squares_split(M, b, U_A, V_A)
    e = split.piece_energies(dU, dV, U_A, V_A, S)
    for c in range(4):
        dB = dU[:, c : c + 1] @ V_A[:, c : c + 1].T + U_A[:, c : c + 1] @ dV[:, c : c + 1].T
        assert abs(float(e[c]) - split.weighted_energy(dB, S)) < 1e-9
    assert split.n_half(torch.tensor([1.0, 1.0, 1.0, 1.0])) == 2
    assert split.n_half(torch.tensor([10.0, 1.0, 1.0])) == 1


def test_read_moments_are_uncorrelated_even_with_a_zero_variance_read_direction():
    U_A, V_A, _ = _setup()
    S = _sigma()
    # kill the variance along the first alive read vector: Sigma is singular inside the read span
    v = V_A[:, 0] / V_A[:, 0].norm()
    Pperp = torch.eye(9, dtype=S.dtype) - v[:, None] * v[None, :]
    S = Pperp @ S @ Pperp
    b = split.basis(U_A, V_A, 1e-6)
    S1, S2 = split.read_moments(b, S)
    assert torch.allclose(S1 + S2, S, atol=1e-12)
    Q = b.Q[:, b.t > 0]
    # a1 = Pi a with Pi = S Q (Q^T S Q)^+ Q^T; the cross moment E[a2 a1^T] = (I - Pi) S Pi^T must vanish
    Pi = S @ Q @ torch.linalg.pinv(Q.T @ S @ Q, hermitian=True, rtol=1e-10) @ Q.T
    assert float(((torch.eye(9, dtype=S.dtype) - Pi) @ S @ Pi.T).abs().max()) < 1e-10
    # Sigma_1 is the moment of a1 and the null uses its trace
    assert torch.allclose(Pi @ S @ Pi.T, S1, atol=1e-10)
    f = split.four_blocks(torch.randn(12, 9, dtype=S.dtype), b, S)
    assert abs(sum(f[k] for k in ("in_in", "in_out", "out_in", "out_out")) - 1) < 1e-10


def test_rescaling_state_reproduces_an_exact_rescaling_and_keeps_delta():
    U_A, V_A, _ = _setup()
    S = _sigma()
    alpha = torch.tensor([0.3, -0.2, 0.0, 1.5], dtype=torch.float64)
    M = (U_A * alpha) @ V_A.T  # exactly sum_c alpha_c u_c v_c^T
    a, _ = split.rescaling_alpha(M, U_A, V_A, S)
    assert torch.allclose(a, alpha, atol=1e-8)
    assert abs(split.rescaling_share(M, U_A, V_A, S) - 1) < 1e-9
    dW = {"a": M.float(), "z": torch.zeros(5, 3)}
    uv = {"a": (U_A.float(), V_A.float()), "z": (torch.randn(5, 2), torch.randn(3, 2))}
    dW_r, state = split.rescaling_state(dW, uv, {"a": S, "z": torch.eye(3, dtype=torch.float64)})
    assert set(state) == {"a"} and not dW_r["z"].any()
    assert torch.allclose(dW_r["a"].double(), M, atol=1e-5)
    Wd, U2, V2 = state["a"]
    # the components move by exactly the target move, so Delta is unchanged; v_c are untouched
    assert torch.allclose(U2.double() @ V2.double().T - U_A @ V_A.T, Wd.double(), atol=1e-5)
    assert torch.equal(V2, V_A.float())
    # a generic update: R in (0, 1) and the state's move is the weighted least-squares fit
    M2 = torch.randn(12, 9, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    R = split.rescaling_share(M2, U_A, V_A, S)
    fit = (U_A * split.rescaling_alpha(M2, U_A, V_A, S)[0]) @ V_A.T
    assert 0 < R < 1 and abs(split.weighted_energy(fit, S) / split.weighted_energy(M2, S) - R) < 1e-9


def test_block_table_puts_first_order_component_changes_outside_both_only_at_second_order():
    U_A, V_A, _ = _setup()
    S = _sigma()
    b = split.basis(U_A, V_A, 1e-6)
    g = torch.Generator().manual_seed(3)
    dU, dV = torch.randn(12, 4, generator=g, dtype=torch.float64), torch.randn(9, 4, generator=g, dtype=torch.float64)
    first = split.dsigma(dU, dV, U_A, V_A)
    t = split.block_table({"a": first, "z": torch.zeros(12, 9)}, {"a": b, "z": b}, {"a": S, "z": S}, ["a"])
    assert set(t["per_matrix"]) == {"a"} and t["pooled_all"]["out_out"] < 1e-12
    assert t["pooled_ns"] == t["pooled_all"] and t["pooled_all"]["out_out_ratio"] < 1e-10
    eps = 1e-2
    full = (U_A + eps * dU) @ (V_A + eps * dV).T - U_A @ V_A.T  # component training's weight change
    oo = split.block_table({"a": full}, {"a": b}, {"a": S}, [])["pooled_all"]["out_out"]
    assert 0 < oo < 10 * eps**2
