"""CPU checks of E44's pure pieces (experiments/qwen_maps): the Gram basis, the states' algebra, the Σ-matched random
control, pooling, behaviour kept, and the filed edge cases of every rule (results/predictions.json "edge_cases")."""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

from commute import maps_score as ms
from commute import qwen_maps, split

F64 = torch.float64


def _setup(d_out=12, d_in=9, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    U_A = torch.randn(d_out, n, generator=g, dtype=F64)
    V_A = torch.randn(d_in, n, generator=g, dtype=F64)
    M = torch.randn(d_out, d_in, generator=g, dtype=F64)
    A = torch.randn(d_in, 3 * d_in, generator=g, dtype=F64)
    return U_A, V_A, M, A @ A.T / (3 * d_in)  # Σ: a full-rank second moment


def test_gram_basis_matches_svd():
    for d, n in ((12, 5), (9, 15), (10, 10)):
        A = torch.randn(d, n, generator=torch.Generator().manual_seed(d + n), dtype=F64)
        if n == 10:
            A[:, -3:] = A[:, :3] @ torch.randn(3, 3, dtype=F64)  # rank 7
        P, s = split.gram_basis(A)
        Ps, ss = split.full_svd(A)
        bg, bs = split.basis_from((P, s), (P, s), 1e-6), split.basis_from((Ps, ss), (Ps, ss), 1e-6)
        assert bg.r_u == bs.r_u == min(d, n, 7 if n == 10 else 99)
        assert torch.allclose(s, ss, atol=1e-6 * float(ss.max()))
        r = bg.r_u
        assert float((A - P[:, :r] @ (P[:, :r].T @ A)).norm() / A.norm()) < 1e-6
        assert torch.allclose(P[:, :r] @ P[:, :r].T, Ps[:, :r] @ Ps[:, :r].T, atol=1e-8)
        assert torch.allclose(P.T @ P, torch.eye(d, dtype=F64), atol=1e-10)


def _products(n=4, halves=True):
    U_A, V_A, M, S = _setup(n=n)
    b = split.basis(U_A, V_A, 1e-6)
    hb = {}
    if halves:
        lo, up = qwen_maps.halves(torch.arange(n, dtype=torch.float16), torch.zeros(n))
        hb = {h: (p, split.basis(U_A[:, p], V_A[:, p], 1e-6)) for h, p in (("lower", lo), ("upper", up))}
    gen = torch.Generator().manual_seed(1)
    RU, RV = torch.randn(U_A.shape, generator=gen, dtype=F64), torch.randn(V_A.shape, generator=gen, dtype=F64)
    arrays, scalars = qwen_maps.site_products(M, U_A, V_A, S, b, hb, RU, RV)
    return U_A, V_A, M, S, b, arrays, scalars, RU, RV


def test_inspan_state_removes_exactly_the_obstruction():
    U_A, V_A, M, S, b, arrays, scalars, *_ = _products()
    dU, dV = arrays["X_ls"] @ V_A, arrays["X_ls"].T @ U_A  # the exact pieces, as S's energies use them
    O = split.obstruction(M, b)
    assert torch.allclose(split.dsigma(dU, dV, U_A, V_A), M - O, atol=1e-9)
    assert math.isclose(scalars["delta"]["inspan"], split.weighted_energy(dU @ dV.T, S), rel_tol=1e-9)
    # the least-squares state: same pieces, weights move by all of δW, so Δ takes O(δW) − dU dVᵀ
    assert math.isclose(scalars["delta"]["ls"], split.weighted_energy(O - dU @ dV.T, S), rel_tol=1e-9)
    # applied from the fp32 file as the pod does: still δW − O(δW), to the stored_error S records
    disk = qwen_maps.to_disk(arrays)
    w, pos, aU, aV = qwen_maps.site_change("inspan", M, U_A, V_A, disk.__getitem__, 0.0, 0.0)
    assert pos is None and w is not None and aU is not None and aV is not None and "O" not in disk
    off = (split.weighted_energy(w - (M - O), S) / scalars["energy"]) ** 0.5
    assert off < 1e-5 and scalars["stored_error"]["ls"] < 1e-5
    # the weight change is the first-order part of the applied pieces, so Δ′ − Δ₀ = −dU dVᵀ
    assert torch.allclose(w, split.dsigma(aU, aV, U_A, V_A), atol=1e-12)
    w_ls = qwen_maps.site_change("ls", M, U_A, V_A, disk.__getitem__, 0.0, 0.0)[0]
    assert w_ls is not None and torch.equal(w_ls, M)


def test_adjoint_and_halves_states():
    U_A, V_A, M, S, _b, arrays, scalars, *_ = _products(n=5)
    w, _, aU, aV = qwen_maps.site_change("adjoint", M, U_A, V_A, None, scalars["c"], 0.0)
    assert aU is not None and aV is not None and w is not None
    assert torch.allclose(aU, scalars["c"] * M @ V_A) and torch.allclose(w, split.dsigma(aU, aV, U_A, V_A))
    assert math.isclose(scalars["delta"]["adjoint"], split.weighted_energy(aU @ aV.T, S), rel_tol=1e-9)
    for h in ("lower", "upper"):
        w, pos, _dU, _dV = qwen_maps.site_change(h, M, U_A, V_A, arrays.__getitem__, 0.0, 0.0)
        assert pos is not None and w is not None and len(pos) == 2 and torch.equal(w, M)  # n = 5: two each, the middle one in neither
    assert arrays["pos_lower"].tolist() == [0, 1] and arrays["pos_upper"].tolist() == [3, 4]


def test_sigma_matched_random_control():
    U_A, V_A, M, S, _b, _arrays, scalars, RU, RV = _products()
    a = scalars["a"]
    w, _, dU, dV = qwen_maps.site_change("random", M, U_A, V_A, None, 0.0, a, RU, RV)
    assert dU is not None and dV is not None and w is not None
    first = split.dsigma(dU, dV, U_A, V_A)
    assert math.isclose(split.weighted_energy(first, S), split.weighted_energy(M, S), rel_tol=1e-9)
    assert torch.allclose(w, first + dU @ dV.T)  # the weights follow the components exactly: Δ unchanged
    assert scalars["delta"]["random"] == 0.0
    assert split.sigma_matched_scale(M, RU[:, :0], RV[:, :0], U_A[:, :0], V_A[:, :0], S) == 0.0  # no alive column


def test_halves_order_and_ties():
    active = torch.tensor([0.5, 0.1, 0.1, 0.0, 0.9, 0.1], dtype=torch.float16)
    mean_g = torch.tensor([0.3, 0.2, 0.1, 0.0, 0.5, 0.2])
    lo, up = qwen_maps.halves(active, mean_g)
    assert lo.tolist() == [1, 2, 3] and up.tolist() == [0, 4, 5]  # 1 and 5 tie on both keys: position decides
    lo, up = qwen_maps.halves(torch.zeros(1), torch.zeros(1))
    assert len(lo) == len(up) == 0


def test_top_share_and_free_check():
    U_A, V_A, M, _S = _setup(d_out=12, d_in=9, n=12)
    b = split.basis(U_A, V_A, 1e-6)  # write side complete
    t = qwen_maps.top_share(M, b)
    assert math.isclose(t["in_U"], 1.0, rel_tol=1e-9) and 0 < t["in_V"] <= 1 + 1e-12 and t["k"] >= 1
    assert math.isnan(qwen_maps.top_share(torch.zeros(3, 2, dtype=F64), split.basis(U_A[:3], V_A[:2], 1e-6))["in_U"])


def test_pooling_is_a_ratio_of_sums():
    assert ms.pooled([1.0, 3.0], [2.0, 2.0]) == 1.0
    assert ms.pooled([1.0, 0.0], [1.0, 9.0]) == 0.1  # not the mean of ratios (0.5)
    assert math.isnan(ms.pooled([], []))
    assert math.isnan(ms.pooled([1.0], [0.0]))


RULES = {"kept": {"floor_min": 0.05, "floor_mult": 5.0}, "pass": {"line": 1.5, "kept_min": 0.8},
         "teeth_i": {"share_tol": 0.1, "margin": 0.2}, "teeth_ii": {"line": 1.5},
         "teeth_iii": {"line": 1.5, "kept_min": 0.8}}


def test_behaviour_kept_identity_and_floor():
    KD, K0, r1 = 0.55, 0.0589, 3.0
    r2 = ms.kept(KD, r1 * K0, K0, 0.001, RULES["kept"])  # a state whose unmasked model is the fine-tuned one
    assert r2["scored"] and math.isclose(r2["r2"], (KD - r1 * K0) / (KD - K0))
    assert ms.kept(K0, 0.0, K0, 0.0, RULES["kept"]) == {"r2": None, "scored": False, "floor": 0.05}  # zero denominator
    assert ms.kept(K0 + 0.05, 0.0, K0, 0.01, RULES["kept"])["scored"]  # exactly at the floor
    assert not ms.kept(K0 + 0.0499, 0.0, K0, 0.0, RULES["kept"])["scored"]
    assert not ms.kept(K0 + 0.06, 0.0, K0, 0.02, RULES["kept"])["scored"]  # floor 5 × 0.02 = 0.1
    for bad in (ms.kept(KD, 0.1, K0, math.nan, RULES["kept"]), ms.kept(KD, None, K0, 0.0, RULES["kept"])):
        assert bad["scored"] and math.isnan(bad["r2"])
    assert ms.kept(KD, KD, K0, 0.0, RULES["kept"])["r2"] == 0.0  # a map that keeps nothing


def _st(r1m=1.0, r1w=1.0, r3m=1.0, r3w=1.0, r2=0.9, scored=True, share=0.5):
    return {"r1m": r1m, "r1w": r1w, "r3m": r3m, "r3w": r3w, "share": share,
            "r2": {"r2": r2 if scored else None, "scored": scored}}


def test_pass_rule_edges():
    P = RULES["pass"]
    assert ms.passes(_st(r1m=1.5, r2=0.8), P) == "pass"
    assert ms.passes(_st(r1m=1.5000001), P) == "fail"
    assert ms.passes(_st(scored=False), P) == "incomplete"
    assert ms.passes(_st(r1m=2.0, scored=False), P) == "fail"
    assert ms.passes(_st(r3w=math.nan), P) == "fail"
    assert ms.passes(_st(r2=math.nan), P) == "fail"
    missing = _st()
    del missing["r3w"]
    assert ms.passes(missing, P) == "fail"


def test_teeth_i_edges():
    T, KD, K0 = RULES["teeth_i"], 0.55, 0.0589
    assert ms.teeth_i(_st(r2=0.3, share=0.5), _st(r2=0.5, share=0.6), KD, K0, T)["holds"]  # both ties at once
    out = ms.teeth_i(_st(r2=0.0, share=0.5), _st(r2=0.9, share=0.6000001), KD, K0, T)
    assert not out["holds"] and out["why"] == "confounded by the span"
    assert not ms.teeth_i(_st(r2=0.31, share=0.5), _st(r2=0.5, share=0.5), KD, K0, T)["holds"]
    need = 2.0 + 0.2 * (KD - K0) / K0
    lo, up = _st(r1m=need, scored=False), _st(r1m=2.0, scored=False)
    assert ms.teeth_i(lo, up, KD, K0, T)["holds"]
    assert not ms.teeth_i(_st(r1m=need - 1e-6, scored=False), up, KD, K0, T)["holds"]
    assert not ms.teeth_i(lo, up, KD, 0.0, T)["holds"]  # K0 = 0
    assert not ms.teeth_i(_st(share=math.nan), _st(), KD, K0, T)["holds"]
    assert not ms.teeth_i(_st(r2=math.nan), _st(), KD, K0, T)["holds"]
    assert not ms.teeth_i({"share": 0.5}, _st(), KD, K0, T)["holds"]  # missing r2
    same = _st(r2=0.0, share=1.0)  # halves empty everywhere: both equal do-nothing
    assert not ms.teeth_i(same, same, KD, K0, T)["holds"]


def test_teeth_ii_and_iii_edges():
    assert not ms.teeth_ii(_st(r1m=1.5), RULES["teeth_ii"])["holds"]
    assert ms.teeth_ii(_st(r3w=1.5000001), RULES["teeth_ii"])["holds"]
    assert not ms.teeth_ii(_st(r1m=9.0, r3w=math.nan), RULES["teeth_ii"])["holds"]
    T3 = RULES["teeth_iii"]
    assert not ms.teeth_iii(_st(r2=0.8), T3)["holds"] and ms.teeth_iii(_st(r2=0.7999999), T3)["holds"]
    assert not ms.teeth_iii(_st(r1m=1.5, scored=False), T3)["holds"]
    assert ms.teeth_iii(_st(r1m=1.6, scored=False), T3)["holds"]
    assert not ms.teeth_iii(_st(r2=math.nan), T3)["holds"]
    assert ms.dn_r3_web(_st(r3w=1.5), T3)["holds"] and not ms.dn_r3_web(_st(r3w=math.nan), T3)["holds"]


def test_outcome_edges():
    yes, no = {"i": True, "ii": True, "iii": True}, {"i": True, "ii": False, "iii": True}
    assert ms.outcome({"10": yes, "1": yes}, "10", "1") == {"teeth": True, "stop": False, "unscored_steps": []}
    assert ms.outcome({"10": yes, "1": no}, "10", "1")["unscored_steps"] == ["1"]
    assert ms.outcome({"10": no, "1": yes}, "10", "1")["stop"]
    assert ms.outcome({"1": yes}, "10", "1")["stop"]


CONTROL = {"tol": 0.01, "gap_mult": 3.0, "ci": {"maths": 0.0589, "web": 0.1417},
           "replay": {"a": 1.7, "b": 4.2}, "kl_to_start": {"1": 0.106, "10": 0.52, "100_fp32": 0.66}}


def _C(**kl):
    return {"ci": {"maths": 0.0589 * 1.01, "web": 0.1417}, "replay": {"a": 1.7 * 0.99, "b": 4.2},
            "kl_to_start": {"1": 0.106, "10": 0.52, "100": 0.66, "100_fp32": 0.66} | kl}


def test_control_w0_edges():
    assert ms.control_w0(_C(), CONTROL)["holds"]  # two values exactly 1% off
    out = ms.control_w0(_C(**{"10": 0.52 * 1.04, "100": 0.66 * 1.01}), CONTROL)  # g = 1%: tol 4%
    assert out["holds"] and math.isclose(out["gap"], 0.01)
    assert not ms.control_w0(_C(**{"10": 0.52 * 1.041, "100": 0.66 * 1.01}), CONTROL)["holds"]
    absent = _C()
    del absent["replay"]["b"]
    assert not ms.control_w0(absent, CONTROL)["holds"]
    assert not ms.control_w0(_C(**{"100_fp32": 0.0}), CONTROL)["holds"]
    assert not ms.control_w0(_C(**{"1": math.nan}), CONTROL)["holds"]


def test_stored_error_under_the_limit_on_a_badly_conditioned_span():
    """E44's stop (2026-10-28): on an ill-conditioned span, pieces X V, Xᵀ U formed in fp32 miss float64 by more than
    STORE_ERR_MAX = 1e-3 (run.py); formed in float64 from the fp32-stored X, as `pieces` does, they do not."""
    d, n = 1024, 32
    g = torch.Generator().manual_seed(0)
    U, V, M = (torch.randn(*sh, generator=g, dtype=F64) for sh in ((d, n), (d, n), (d, d)))
    U[:, 0] *= 1e-3  # alive spectrum spread 1000x
    V[:, 1] *= 1e-3
    S, b = torch.eye(d, dtype=F64), split.basis(U, V, 1e-12)
    X, E = split.least_squares_x(M, b), split.weighted_energy(M, S)
    exact = split.dsigma(X @ V, X.T @ U, U, V)
    X32, U32, V32 = X.float(), U.float(), V.float()
    fp32_formed = split.dsigma(X32 @ V32, X32.T @ U32, U32, V32).double() - exact  # the stopped run's application
    assert (split.weighted_energy(fp32_formed, S) / E) ** 0.5 > 1e-3
    assert 0 < qwen_maps.stored_error(X, U, V, S, E) < 1e-3


# ----------------------------------------------------------------------------- resume after a stop (run.py)


def _load_run(monkeypatch, tmp_path):
    here = Path(__file__).resolve().parents[1] / "experiments" / "qwen_maps"
    mods = {}
    for name, file in (("job", "job.py"), ("qwen_maps_run", "run.py")):
        spec = importlib.util.spec_from_file_location(name, here / file)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, mod)  # run.py imports `job` by that name
        spec.loader.exec_module(mod)
        mods[name] = mod
    monkeypatch.setattr(mods["job"], "RESULTS", tmp_path)

    def job(waive=""):
        a = argparse.Namespace(out=str(tmp_path / "out"), e40=str(tmp_path), e40_large=str(tmp_path), decomp="",
                               usd_per_h=3.0, prior_usd=0.5, waive=waive)
        return mods["job"].Job(a, mods["job"].REHEARSAL, mods["qwen_maps_run"].score)

    return mods["qwen_maps_run"], job


def _missed_C(pred):
    r = pred["rules"]["control_W0"]
    return {"ci": {"maths": 1.5 * r["ci"]["maths"], "web": r["ci"]["web"]}, "replay": dict(r["replay"]),
            "kl_to_start": {"1": r["kl_to_start"]["1"], "10": r["kl_to_start"]["10"],
                            "100": r["kl_to_start"]["100_fp32"], "100_fp32": r["kl_to_start"]["100_fp32"]}}


def test_resume_after_a_stop(monkeypatch, tmp_path):
    run, make = _load_run(monkeypatch, tmp_path)
    stopped = tmp_path / "rehearsal" / "stopped.json"
    job = make()
    assert job.spent_before == 0.5  # no state.json yet: --prior-usd
    job.put("C", _missed_C(job.pred) | {"done": False})
    assert not run.done(job, "C")  # a missed control is never "done"
    with pytest.raises(SystemExit):
        run.gate_C(job)
    assert json.loads(stopped.read_text())["name"] == "control_W0"
    with pytest.raises(SystemExit):  # the next job refuses to start
        run.refuse_if_stopped(make())
    later = make(waive="control_W0")  # a rehearsal may waive it by name
    assert later.spent_before >= 0.5  # cumulative spend carried in state.json, not --prior-usd
    run.refuse_if_stopped(later)
    run.gate_C(later)
    stopped.unlink()  # a C.json marked done by hand still has its gate read again when skipped
    job = make()
    job.res["C"] = _missed_C(job.pred) | {"done": True}
    with pytest.raises(SystemExit):
        run.run_stage(job, "C")
    stopped.unlink()  # K: no teeth reading at step 10 -> the gate stops (K is marked done only after it)
    job = make()
    job.res["C"] = _missed_C(job.pred)
    with pytest.raises(SystemExit):
        run.gate_K(job)
    assert json.loads(stopped.read_text())["name"] == "teeth"


def test_batch_halving_persists(monkeypatch, tmp_path):
    _, make = _load_run(monkeypatch, tmp_path)
    make().set_batch(4, "test")
    assert make().batch == 4  # a resume never reuses the failed size
