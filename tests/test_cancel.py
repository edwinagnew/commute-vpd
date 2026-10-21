"""CPU checks of the cancellation statistics and the revert hook on small random matrices."""

from __future__ import annotations

import torch
from torch import nn

from commute import cancel


def _pieces(d_out=6, d_in=5, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    U = torch.randn(d_out, n, generator=g)
    V = torch.randn(d_in, n, generator=g)
    U2 = U + 0.3 * torch.randn(d_out, n, generator=g)
    V2 = V + 0.3 * torch.randn(d_in, n, generator=g)
    return U, V, U2, V2


def test_change_energies_match_explicit_frobenius():
    U, V, U2, V2 = (t.double() for t in _pieces())
    e = cancel.change_energies(U, V, U2, V2)
    for c in range(U.shape[1]):
        d = torch.outer(U2[:, c], V2[:, c]) - torch.outer(U[:, c], V[:, c])
        assert abs(float(e[c]) - float((d ** 2).sum())) < 1e-8


def test_coherence_reads_count_one_and_zero():
    d_out, n = 6, 4
    v = torch.randn(5)
    U = torch.randn(d_out, n)
    a = torch.randn(d_out)
    V = v[:, None].repeat(1, n)
    aligned = {"m": (U, V, U + a[:, None], V)}  # every Delta_c = a v^T
    assert abs(cancel.coherence(aligned, n_null=1)["pooled"] - n) < 1e-9
    E = torch.eye(n)  # d_in = n, v_c = e_c: orthogonal changes
    orth = {"m": (U, E, U + torch.randn(d_out, n), E)}
    assert abs(cancel.coherence(orth, n_null=1)["pooled"] - 1) < 1e-9
    U2 = U.clone()
    U2[:, 0] += a
    U2[:, 1] -= a
    cancelling = {"m": (U[:, :2], V[:, :2], U2[:, :2], V[:, :2])}  # Delta_2 = -Delta_1
    assert cancel.coherence(cancelling, n_null=1)["pooled"] < 1e-9
    null = cancel.coherence(aligned, n_null=400)["pooled_null"]
    assert abs(null - 1) < 0.25  # random signs on identical changes: 1 in expectation


def test_revert_hook_equals_weight_edit():
    torch.manual_seed(0)
    U, V, U2, V2 = _pieces()
    d_out, d_in = U.shape[0], V.shape[0]
    W0 = torch.randn(d_out, d_in)
    change = U2 @ V2.T - U @ V.T
    lin = nn.Linear(d_in, d_out, bias=False)
    with torch.no_grad():
        lin.weight.copy_(W0 + change)
    hook = cancel.Revert(U, V, U2, V2)
    h = lin.register_forward_hook(hook)
    x = torch.randn(2, 3, d_in)
    hook.w = torch.ones(U.shape[1])
    assert torch.allclose(lin(x), x @ W0.T, atol=1e-5)  # full revert: W0
    hook.w = torch.zeros(U.shape[1])
    assert torch.allclose(lin(x), x @ (W0 + change).T, atol=1e-5)  # nothing reverted
    w = torch.rand(2, 3, U.shape[1])  # per-position fractions
    hook.w = w
    out = lin(x)
    for b in range(2):
        for t in range(3):
            Wbt = W0 + change - sum(float(w[b, t, c]) * (torch.outer(U2[:, c], V2[:, c]) - torch.outer(U[:, c], V[:, c]))
                                   for c in range(U.shape[1]))
            assert torch.allclose(out[b, t], Wbt @ x[b, t], atol=1e-5)
    h.remove()


def test_half_and_adversary_weights():
    U, V, U2, V2 = _pieces(n=6)
    pieces = {"m": (U, V, U2, V2)}
    w = cancel.half_weights(pieces, seed=1)["m"]
    assert int(w.sum()) == 3 and set(w.tolist()) == {0.0, 1.0}
    alive = {"m": torch.tensor([True, False, True, True, False, True, True, True])}  # 8 components, 6 alive
    sources = {"m": torch.tensor([0.0, 0.5, 1.0, 0.2, 0.9, 0.4, 0.6, 0.8, 0.3])}  # C + 1, last is Delta's
    g = {"m": torch.zeros(1, 1, 6)}
    wf = cancel.adversary_weights(sources, alive, "cpu")(g)["m"]
    assert torch.allclose(wf[0, 0], 1 - torch.tensor([0.0, 1.0, 0.2, 0.4, 0.6, 0.8]))
    g = {"m": torch.ones(1, 1, 6)}
    assert float(cancel.adversary_weights(sources, alive, "cpu")(g)["m"].abs().sum()) == 0  # g = 1: nothing removable
    g = {"m": torch.zeros(1, 1, 6)}
    perm = cancel.adversary_weights(sources, alive, "cpu", perm_seed=0)(g)["m"][0, 0]
    assert sorted(perm.tolist()) == sorted(wf[0, 0].tolist())


def test_gauge_mixing_keeps_the_weight_change_and_cancels():
    U, V, U2, V2 = _pieces(d_out=8, d_in=7, n=4)
    pieces = {"m": (U, V, U2, V2)}
    ctrl = cancel.gauge_mixed(pieces, energy_ratio=10.0, seed=0)
    assert torch.allclose(cancel.signed_sum(*ctrl["m"]), cancel.signed_sum(*pieces["m"]), atol=1e-5)
    e0 = float(cancel.change_energies(*pieces["m"]).sum())
    e1 = float(cancel.change_energies(*ctrl["m"]).sum())
    assert abs(e1 / e0 - 10) < 0.01
    k0, k1 = cancel.coherence(pieces, n_null=1)["pooled"], cancel.coherence(ctrl, n_null=1)["pooled"]
    assert abs(k1 - k0 / 10) < 1e-3 * k0


def test_score_on_a_fake_result():
    import importlib.util
    import sys
    from pathlib import Path

    run = Path(__file__).resolve().parents[1] / "experiments" / "cancellation" / "run.py"
    spec = importlib.util.spec_from_file_location("cancellation_run", run)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)

    def rec(kappa, h, f_adv, f_perm, pgd):
        return {"coherence": {"pooled": kappa, "pooled_null": 1.02}, "G1": {"ok": True}, "h": h, "F_adv": f_adv,
                "F_perm": [f_perm], "pgd_kl": pgd, "F_full": 0.1, "F_half": [h * 0.1]}

    st = {n: rec(3.0, 1.5, 0.5, 0.1, 1.5) for n in m.SCORED}
    st[m.ADJOINT] = rec(5.0, 0.3, 0.01, 0.01, 0.7)
    st["ct emoticon"] = rec(3.0, 0.8, 0.1, 0.05, 0.75)
    st[m.CONTROL] = rec(0.1, 2.0, 0.1, 0.1, 0.8)
    sc = m.score({"states": st, "W0": {"pgd_kl": 0.7}})
    assert sc["P1_coherence"]["verdict"] == "fail" and sc["P2_half_revert"]["verdict"] == "pass"
    assert sc["P3_adversary_removes_cancelling_part"]["verdict"] == "pass" and sc["C1_injected_cancellation"]["verdict"] == "pass"
    assert all(sc["G1_hook_identity"].values()) and all(sc["G2_coherence_null"].values())
    del st[m.ADJOINT]
    assert m.score({"states": st, "W0": {"pgd_kl": 0.7}})["P2_half_revert"]["verdict"] == "not run"
    st[m.SCORED[0]]["coherence"]["pooled_null"] = 1.3  # a failed null drops that state from P1 only
    sc = m.score({"states": st, "W0": {"pgd_kl": 0.7}})
    assert m.SCORED[0] in sc["P1_coherence"]["unscored"] and m.SCORED[0] in sc["P2_half_revert"]["scored_states"]["per_state"]
    assert not sc["G2_coherence_null"][m.SCORED[0]]
