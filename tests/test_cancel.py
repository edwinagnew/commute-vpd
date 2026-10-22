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


# ----------------------------------------------------------------------------- the complement revert (E20)


def _state(d_out=6, d_in=5, C=7, seed=0):
    """A tiny matrix in VPD form: all C subcomponents (Uf (d_out, C), Vf (d_in, C)) of which `alive` are changed to
    (U2, V2), and Delta' dense, the target weight W' = sum over all subcomponents + Delta'."""
    g = torch.Generator().manual_seed(seed)
    Uf, Vf = torch.randn(d_out, C, generator=g), torch.randn(d_in, C, generator=g)
    alive = torch.tensor([True, False, True, True, False, True, True])
    U, V = Uf[:, alive], Vf[:, alive]
    U2, V2 = U + 0.3 * torch.randn(U.shape, generator=g), V + 0.3 * torch.randn(V.shape, generator=g)
    Uf2, Vf2 = Uf.clone(), Vf.clone()
    Uf2[:, alive], Vf2[:, alive] = U2, V2
    D = 0.1 * torch.randn(d_out, d_in, generator=g)
    return Uf2, Vf2, alive, (U, V, U2, V2), D


def _linear(W):
    lin = nn.Linear(W.shape[1], W.shape[0], bias=False)
    with torch.no_grad():
        lin.weight.copy_(W)
    return lin


def test_remove_reproduces_the_masked_forward_and_revert():
    torch.manual_seed(0)
    Uf2, Vf2, alive, (U, V, U2, V2), D = _state()
    lin = _linear(Uf2 @ Vf2.T + D)
    x = torch.randn(2, 3, Vf2.shape[0])
    g, a, a_d = torch.rand(2, 3, 7), torch.rand(7), 0.3
    m = g + (1 - g) * a  # the adversary's mask over all C
    masked = ((x @ Vf2) * m) @ Uf2.T + a_d * x @ D.T  # param_decomp's masked forward
    blocks = [(U, V), (U2, V2), (Uf2[:, ~alive], Vf2[:, ~alive]), (D, None)]
    hook = cancel.Remove(blocks)
    h = lin.register_forward_hook(hook)
    r = 1 - m
    hook.w = [0.0, r[..., alive], r[..., ~alive], 1 - a_d]
    assert torch.allclose(lin(x), masked, atol=1e-5)  # G1's closure
    w = torch.rand(2, 3, int(alive.sum()))
    hook.w = [-w, w, 0.0, 0.0]
    via_remove = lin(x)
    h.remove()
    rev = cancel.Revert(U, V, U2, V2)
    rev.w = w
    h = lin.register_forward_hook(rev)
    assert torch.allclose(via_remove, lin(x), atol=1e-5)  # E17's revert is the (-w, w) case
    h.remove()
    hook.w = [w, 0.0, 0.0, 0.0]  # the complement revert: w of the originals, changes kept
    h = lin.register_forward_hook(hook)
    assert torch.allclose(lin(x), x @ (Uf2 @ Vf2.T + D).T - ((x @ V) * w) @ U.T, atol=1e-5)
    h.remove()


def test_arm_weights_and_selection():
    w = torch.rand(2, 3, 4)
    wf = lambda g: {"m": w, "n": w}
    out = cancel.arm(wf, 1, 0)({})
    assert torch.equal(out["m"][0], w) and out["m"][1] == 0.0
    out = cancel.arm(wf, -1, 1)({})
    assert torch.equal(out["m"][0], -w) and torch.equal(out["m"][1], w)
    sel = {"m": torch.tensor([True, False, True, False])}
    out = cancel.arm(wf, 1, 0, sel)({})
    assert "n" not in out and torch.equal(out["m"][0], w * torch.tensor([1.0, 0.0, 1.0, 0.0]))


def test_importance_and_layer_bins():
    alive = {"h.0.a": torch.tensor([True, True, False, True]), "h.1.a": torch.tensor([True, False, True, True, True])}
    mean_g = {"h.0.a": torch.tensor([0.5, 0.1, 9.0, 0.7]), "h.1.a": torch.tensor([0.2, 9.0, 0.8, 0.3, 0.05])}
    bins = cancel.importance_bins(mean_g, alive, 2)
    counts = [sum(int(m.sum()) for m in bins[b].values()) for b in ("imp0", "imp1")]
    assert sum(counts) == 7 and abs(counts[0] - counts[1]) <= 1
    low = torch.cat([mean_g[k][alive[k]][bins["imp0"][k]] for k in sorted(alive)])
    high = torch.cat([mean_g[k][alive[k]][bins["imp1"][k]] for k in sorted(alive)])
    assert float(low.max()) < float(high.min())
    lb = cancel.layer_bins(alive)
    assert set(lb) == {"layer0", "layer1"} and int(lb["layer1"]["h.1.a"].sum()) == 4


def test_bin_means():
    g = {"m": torch.zeros(1, 2, 3)}
    cached = [(torch.zeros(1, 2), torch.zeros(1, 2, 5), g)]
    w = torch.tensor([[[0.0, 1.0, 0.5], [0.0, 1.0, 0.5]]])
    bins = {"a": {"m": torch.tensor([True, True, False])}, "b": {"m": torch.tensor([False, False, True])}}
    out = cancel.bin_means(cached, lambda g: {"m": w}, bins)
    assert abs(out["a"] - 0.5) < 1e-9 and abs(out["b"] - 0.5) < 1e-9


def test_complement_score_on_a_fake_result():
    import importlib.util
    import sys
    from pathlib import Path

    run = Path(__file__).resolve().parents[1] / "experiments" / "cancellation_complement" / "run.py"
    spec = importlib.util.spec_from_file_location("complement_run", run)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)

    def rec(f_all, f_orig, f_adv, f_both, perm, bins):
        return {"G1": {"ok": True}, "F_all": f_all, "F_orig": f_orig, "F_adv": f_adv, "F_both": f_both,
                "F_orig_perm": [perm], "F_orig_bin": bins}

    b0 = {"imp0": 0.1, "imp1": 0.1, "imp2": 0.1, "imp3": 0.1, "layer0": 0.1}
    w0 = rec(0.7, 0.6, 0.0, 0.6, 0.1, b0)
    st = {n: rec(1.7, 1.4, 0.05, 1.5, 0.3, {**b0, "imp0": 0.5, "imp3": 0.2}) for n in m.SCORED}
    sc = m.score({"states": st, "W0": w0, "C0": {"verdict": "pass"}})
    d = sc["decomposition"][m.SCORED[0]]
    assert abs(d["X"] - (d["O"] + d["F_adv"] + d["I"] + d["R"])) < 1e-12  # exact by definition
    assert abs(d["O"] - 0.8) < 1e-12 and abs(d["low_share"] - 0.8) < 1e-12
    assert sc["P1_excess_in_originals"]["verdict"] == "pass" and sc["P2_pattern_matters"]["verdict"] == "pass"
    assert sc["P3_low_importance_carries_it"]["verdict"] == "pass"
    st[m.SCORED[0]]["F_orig_bin"] = {**b0, "imp3": 0.9}  # one state's excess in the top quartile: 3 of 4 still pass
    assert m.score({"states": st, "W0": w0})["P3_low_importance_carries_it"]["verdict"] == "pass"
    st[m.SCORED[1]]["F_orig"] = 0.7  # O = 0.1 < 0.5 X
    assert m.score({"states": st, "W0": w0})["P1_excess_in_originals"]["verdict"] == "fail"
    del st[m.SCORED[2]]
    assert m.score({"states": st, "W0": w0})["P2_pattern_matters"]["verdict"] == "not run"


# ----------------------------------------------------------------------------- E20 amendment 1


def test_rest_arms_and_all_blocks_reproduce_the_masked_forward():
    torch.manual_seed(0)
    Uf2, Vf2, alive, (U, V, U2, V2), D = _state()
    lin = _linear(Uf2 @ Vf2.T + D)
    x = torch.randn(2, 3, Vf2.shape[0])
    g, src, src0 = torch.rand(2, 3, 7), torch.rand(8), torch.rand(8)  # sources (C + 1,): a, then a_Delta
    a, a_d = src[:-1], float(src[-1])
    masked = ((x @ Vf2) * (g + (1 - g) * a)) @ Uf2.T + a_d * x @ D.T
    hook = cancel.Remove([(U, V), (U2, V2), (Uf2[:, ~alive], Vf2[:, ~alive]), (D, None)])
    h = lin.register_forward_hook(hook)
    S, A, ga, gd = {"m": src}, {"m": alive}, {"m": g[..., alive]}, {"m": g[..., ~alive]}
    hook.w = cancel.all_blocks_arm(S, A, "cpu")((ga, gd))["m"]
    assert torch.allclose(lin(x), masked, atol=1e-5)  # the dead cache's check: every block gives the masked forward
    Ud, Vd, W = Uf2[:, ~alive], Vf2[:, ~alive], Uf2 @ Vf2.T + D
    dead_rm = lambda s: ((x @ Vd) * ((1 - g[..., ~alive]) * (1 - s[:-1][~alive]))) @ Ud.T
    hook.w = cancel.rest_arm(S, A, "cpu", True, False)(gd)["m"]
    assert torch.allclose(lin(x), x @ W.T - dead_rm(src), atol=1e-5)
    hook.w = cancel.rest_arm(S, A, "cpu", False, True)(gd)["m"]
    assert torch.allclose(lin(x), x @ W.T - (1 - a_d) * x @ D.T, atol=1e-5)
    hook.w = cancel.rest_arm({"m": src0}, A, "cpu", True, True)(gd)["m"]  # W0's levels, this model's g
    assert torch.allclose(lin(x), x @ W.T - dead_rm(src0) - (1 - float(src0[-1])) * x @ D.T, atol=1e-5)
    h.remove()


def test_levels():
    alive = {"a": torch.tensor([True, False, False]), "b": torch.tensor([False, True])}
    src = {"a": torch.tensor([0.9, 0.2, 0.4, 0.25]), "b": torch.tensor([1.0, 0.3, 0.75])}
    lv = cancel.levels(src, alive)
    assert abs(lv["dense"]["a"] - 0.75) < 1e-6 and abs(lv["dense_mean"] - 0.5) < 1e-6
    assert abs(lv["dead_mean"] - (0.8 + 0.6 + 0.0) / 3) < 1e-6


def test_amend1_split_and_readings_on_a_fake_result():
    import importlib.util
    import sys
    from pathlib import Path

    run = Path(__file__).resolve().parents[1] / "experiments" / "cancellation_complement" / "run.py"
    spec = importlib.util.spec_from_file_location("complement_run_a1", run)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)

    lv0 = {"dense_mean": 0.9, "dead_mean": 0.6, "dead_removed": 0.5}

    def rec(f_all, f_both, dense, dead, rest, dense_fix, dead_fix, lv=lv0):
        return {"F_all": f_all, "F_both": f_both, "F_orig": f_both, "F_dense": dense, "F_dead": dead, "F_rest": rest,
                "F_dense_fix": dense_fix, "F_dead_fix": dead_fix, "levels": lv}

    w0 = rec(0.7, 0.5, 0.05, 0.05, 0.12, 0.05, 0.05)
    joint = rec(1.7, 0.5, 0.06, 0.06, 0.14, 0.06, 0.06)  # R = 1.0, all of it in J
    d = m.split_rest(joint, w0)
    assert abs(d["R"] - (d["D"] + d["K"] + d["P"] + d["J"])) < 1e-12 and abs(d["R"] - 1.0) < 1e-12
    assert d["reading"] == "4"
    dense_more = rec(1.7, 0.5, 0.85, 0.05, 0.92, 0.1, 0.05, {**lv0, "dense_mean": 1.0})  # D = 0.8, gone at W0's level
    assert m.split_rest(dense_more, w0)["reading"] == "1"
    costs_more = rec(1.7, 0.5, 0.85, 0.05, 0.92, 0.8, 0.05)  # D = 0.8 at W0's level too
    assert m.split_rest(costs_more, w0)["reading"] == "2"
    dead_more = rec(1.7, 0.5, 0.05, 0.85, 0.92, 0.05, 0.7)
    assert m.split_rest(dead_more, w0)["reading"] == "3"
    pair = rec(1.7, 0.5, 0.05, 0.05, 0.92, 0.05, 0.05)  # P = 0.8
    assert m.split_rest(pair, w0)["reading"] == "dead with Delta'"
    st = {n: joint for n in m.SCORED}
    sc = m.score_amend1({"states": st, "W0": w0, "G2": {}})
    assert sc["P4_only_jointly"]["verdict"] == "pass" and sc["P5_levels_unchanged"]["verdict"] == "pass"
    st[m.SCORED[0]] = st[m.SCORED[1]] = dense_more
    sc = m.score_amend1({"states": st, "W0": w0, "G2": {}})
    assert sc["P4_only_jointly"]["verdict"] == "fail" and sc["P5_levels_unchanged"]["verdict"] == "pass"  # 2 of 4 fail P4
    del st[m.SCORED[3]]
    assert m.score_amend1({"states": st, "W0": w0, "G2": {}})["P4_only_jointly"]["verdict"] == "not run"
    rel = m.g2("W0", {**w0, "F_hooks_all": 0.7}, {"W0": {**w0, "F_all": 0.7005}})
    assert rel["ok"] is True and not m.g2("W0", {**w0, "F_hooks_all": 0.7}, {"W0": {**w0, "F_all": 0.702}})["ok"]


def test_masked_kl_alive_whole_and_chained_pgd_on_the_toy():
    """E20 amendment 2's pieces on the tiny ComponentModel: masked_kl is cache_adversary's F_all, alive_whole keeps the
    alive subcomponents' masks at 1, and pgd_sources from `init` continues a run exactly (a + b steps = a then b)."""
    from test_faithfulness_invariants import KEYS, VOCAB, make_model

    from commute import gamma

    model = make_model()
    g = torch.Generator().manual_seed(3)
    batches = [torch.randint(0, VOCAB, (2, 41), generator=g) for _ in range(2)]
    alive = {k: torch.arange(model.module_to_c[k]) % 2 == 0 for k in KEYS}
    src, _ = gamma.pgd_sources(model, batches, 5, seed=0)
    a, _ = gamma.pgd_sources(model, batches, 2, seed=0)
    b, _ = gamma.pgd_sources(model, batches, 3, seed=7, init=a)
    assert all(torch.equal(src[k], b[k]) for k in KEYS)
    kl = cancel.masked_kl(model, batches, src)
    assert abs(kl - cancel.cache_adversary(model, batches, alive, src)[2]) < 1e-9 and kl > 0
    whole = cancel.alive_whole(src, alive)
    for k in KEYS:
        assert torch.all(whole[k][:-1][alive[k]] == 1) and torch.equal(whole[k][:-1][~alive[k]], src[k][:-1][~alive[k]])
        assert whole[k][-1] == src[k][-1] and not torch.all(src[k][:-1][alive[k]] == 1)


def test_amend2_rules_on_a_fake_result():
    import importlib.util
    import sys
    from pathlib import Path

    run = Path(__file__).resolve().parents[1] / "experiments" / "cancellation_complement" / "run.py"
    spec = importlib.util.spec_from_file_location("complement_run_a2", run)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)

    assert [m.fixed_reading(x) for x in (0.6, 0.5, 0.3, 0.2, float("nan"))] == ["2", "2", "both", "1", "none: X <= 0"]
    w0 = {"seed0": {"20": 0.66, "50": 0.7, "100": 0.72}, "alive_whole": {"0": 1.5, "20": 1.1, "50": 1.05},
          "seed1": {"20": 0.65}, "seed2": {"20": 0.68}}
    r1 = {"seed0": {"20": 1.53, "50": 1.55, "100": 1.6}, "alive_whole": {"0": 0.94, "20": 1.2, "50": 1.3},
          "seed1": {"20": 1.5}, "seed2": {"20": 1.52}}
    sc = m.score_strong({"W0": w0, m.FAILING[1]: r1})
    assert sc["M"]["W0"] == 1.1 and sc["M_random"]["W0"] == 0.72 and sc["M"][m.FAILING[1]] == 1.6  # step 0 not counted
    assert abs(sc["X20"] - 0.87) < 1e-12 and abs(sc["X_strong"] - 0.5) < 1e-12
    assert sc["P7_floor_understated"] == "pass" and sc["P8_excess_partly_the_search"] == "pass"  # 0.5 <= 0.6 x 0.87
    a1 = {"W0": {"F_all": 0.660363, "F_rest": 1.0754}}
    assert m.c2_gate("W0", "seed0", 50, 0.7, a1) is None
    assert m.c2_gate("W0", "seed0", 20, 0.6606, a1)["ok"] and not m.c2_gate("W0", "seed0", 20, 0.662, a1)["ok"]
    assert m.c2_gate("W0", "alive_whole", 0, 1.07545, a1)["ok"] and not m.c2_gate("W0", "alive_whole", 0, 1.076, a1)["ok"]
