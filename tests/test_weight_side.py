"""CPU checks of commute.weight_side (E67) and experiments/weight_side_read/run.py, on the tiny ComponentModel of test_faithfulness_invariants: the shares' sum,
the zero read, every control's algebra, the factored products against dense matrices, the filed verdict at its edge cases, the job's stages end to end."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from test_faithfulness_invariants import KEYS, make_batches, make_model
from torch import Tensor

from commute import saved_states, split, vpd
from commute import weight_side as WS

REPO = Path(__file__).resolve().parents[1]
FILED = json.loads((REPO / "experiments" / "weight_side_read" / "results" / "predictions.json").read_text())
TH = FILED["thresholds"]
GEN = torch.Generator().manual_seed(0)


def rnd(*shape: int) -> Tensor:
    return torch.randn(*shape, generator=GEN)


def full_alive(model) -> dict[str, Tensor]:
    return {k: torch.ones(model.module_to_c[k], dtype=torch.bool) for k in KEYS}


def theory_uv(model, alive) -> dict[str, tuple[Tensor, Tensor]]:
    return {k: (model.components[k].U.detach()[alive[k]].T.float(), model.components[k].V.detach()[:, alive[k]].float()) for k in KEYS}


def test_changed_matches_the_dense_change_on_alive_columns_only():
    U, V = rnd(6, 5), rnd(4, 6)  # C = 6, d_out = 5, d_in = 4
    alive = {"m": torch.tensor([True, True, False, True, False, True])}
    uv = {"m": (U[alive["m"]].T, V[:, alive["m"]])}
    U2, V2 = uv["m"][0] + 0.1 * rnd(5, 4), uv["m"][1] + 0.1 * rnd(4, 4)
    off = WS.offsets_from({"m": (U2, V2)}, uv, alive)["m"]
    assert not off.dU[~alive["m"]].any() and not off.dV[:, ~alive["m"]].any()
    a, w = rnd(9, 4), torch.ones(9, 6)
    dense = U2 @ V2.T - uv["m"][0] @ uv["m"][1].T
    assert torch.allclose(WS.changed(a, w, U, V, off), a @ dense.T, atol=1e-5)
    only = torch.zeros(9, 6)
    only[:, 2] = 1.0
    assert WS.changed(a, only, U, V, off).abs().max() == 0, "a dead column has no change"


def one_read(moved: bool = True) -> tuple[WS.Matrix, dict[str, Tensor]]:
    U, V, a = rnd(6, 5), rnd(4, 6), rnd(40, 4)
    g0, g1 = torch.rand(40, 6, generator=GEN), torch.rand(40, 6, generator=GEN)
    off = WS.Offsets(0.1 * rnd(6, 5), 0.1 * rnd(4, 6)) if moved else WS.Offsets(torch.zeros(6, 5), torch.zeros(4, 6))
    m, dW = WS.Matrix(), rnd(5, 4)
    m.add(a, g0, g1, U, V, dW, off)
    return m, {"a": a, "g0": g0, "g1": g1, "U": U, "V": V, "dW": dW}


def test_the_shares_sum_to_one_from_the_whole_masked_change():
    m, _ = one_read()
    s = WS.shares(m.split, m.den)
    assert s["sum"] == pytest.approx(1.0, abs=1e-5)
    assert WS.shares(m.nothing, m.den)["sum"] == pytest.approx(1.0, abs=1e-5)
    assert m.split["R"] == m.nothing["R"], "one pass: the two states share g1 and S_R"
    assert m.nothing["D"] == 0 and m.nothing["I"] == 0


def test_do_nothing_offsets_give_no_direct_or_interaction_term():
    m, _ = one_read(moved=False)
    assert m.split["D"] == 0 and m.split["I"] == 0
    assert m.split["E"] == pytest.approx(m.nothing["E"], rel=1e-5)


def test_zero_denominator_gives_no_share():
    assert set(WS.shares(dict.fromkeys(WS.TERMS, 0.0), 0.0).values()) == {None}


def test_every_gate_on_and_no_gate_change_reads_direct_one():
    """The advice's known control: δW = sum_c alpha_c C0_c, C1 = (1 + alpha) C0, g0 = g1 = 1: S_D = 1, the others 0."""
    U, V, a = rnd(6, 5), rnd(4, 6), rnd(30, 4)
    alpha = 0.1 * rnd(6)
    dW = (U.T * alpha) @ V.T
    off = WS.Offsets(alpha[:, None] * U, torch.zeros(4, 6))
    m, ones = WS.Matrix(), torch.ones(30, 6)
    m.add(a, ones, ones, U, V, dW, off)
    s = WS.shares(m.split, m.den)
    assert s["S_D"] == pytest.approx(1.0, abs=1e-5)
    assert s["S_R"] == 0 and s["S_I"] == 0 and s["S_E"] == pytest.approx(0.0, abs=1e-5)


def test_re_switching_reads_one_on_its_own_direction_and_zero_orthogonal():
    """Positive control and blind spot of S_R: a gate moved by 1 on one subcomponent, with the update equal to that subcomponent, then writing elsewhere."""
    U, V, a = torch.zeros(2, 4), torch.zeros(3, 2), rnd(30, 3)
    U[0, 0], V[0, 0], U[1, 1], V[1, 1] = 1.0, 1.0, 1.0, 1.0
    g0, g1 = torch.zeros(30, 2), torch.zeros(30, 2)
    g1[:, 0] = 1.0
    none = WS.Offsets(torch.zeros(2, 4), torch.zeros(3, 2))
    same, other = torch.zeros(4, 3), torch.zeros(4, 3)
    same[0, 0], other[2, 0] = 1.0, 1.0
    a_on, a_off = WS.Matrix(), WS.Matrix()
    a_on.add(a, g0, g1, U, V, same, none)
    a_off.add(a, g0, g1, U, V, other, none)
    assert WS.shares(a_on.nothing, a_on.den)["S_R"] == pytest.approx(1.0)
    assert WS.shares(a_off.nothing, a_off.den)["S_R"] == 0


def test_split_identity_is_exact_up_to_rounding():
    """The g = 1 direct term against 1 - omega - 2 cross + second, on a span small enough for an outside part, with coloured inputs so `cross` is not 0."""
    U_A, V_A, M = rnd(12, 3), rnd(10, 3), 0.01 * rnd(12, 10)
    alive = {"m": torch.ones(3, dtype=torch.bool)}
    uv = {"m": (U_A, V_A)}
    b = split.bases(uv)
    st = split.split_state({"m": M}, uv, b)
    off = WS.offsets_from({"m": (st["m"][1], st["m"][2])}, uv, alive)["m"]
    a = (rnd(4000, 10) @ rnd(10, 10)).float()
    S = a.double().T @ a.double() / len(a)
    parts = WS.split_identity(M, b["m"], st["m"][1] - U_A, st["m"][2] - V_A, S)
    D = WS.changed(a, torch.ones(len(a), 3), U_A.T, V_A, off)
    assert abs(parts["cross"]) > 1e-3, "else this test cannot see the cross term"
    assert WS.dot(D, D) / len(a) / parts["den"] == pytest.approx(parts["predicted"], abs=1e-4)
    assert parts["predicted"] != pytest.approx(parts["one_minus_omega"], abs=1e-3)


def test_matched_gaussian_matches_the_weighted_energy_and_is_seeded():
    x = rnd(8, 4).double()
    M, S = {"a": rnd(6, 4)}, {"a": x.T @ x}
    G = WS.matched_gaussian(M, S, 67)
    assert split.weighted_energy(G["a"].double(), S["a"]) == pytest.approx(split.weighted_energy(M["a"].double(), S["a"]), rel=1e-5)
    assert torch.equal(G["a"], WS.matched_gaussian(M, S, 67)["a"])


def test_rel_norm_max():
    assert WS.rel_norm_max({"a": torch.ones(2, 2), "b": torch.ones(1, 1)}, {"a": 4 * torch.ones(2, 2), "b": 2 * torch.ones(1, 1)}) == pytest.approx(0.5)


def s_r(c10: float | None, c100: float | None, s0: float, s1: float, gauss: float | None = 0.0) -> dict[str, float | None]:
    return {"code_step10": c10, "code_step100": c100, "ident32_s0": s0, "ident32_s1": s1, "gaussian": gauss}


@pytest.mark.parametrize(("read", "want"), [
    (s_r(0.05, 0.05, 0.5, 0.6), "keep"),
    (s_r(0.1, 0.1, 0.4, 0.4), "keep"),  # ties keep
    (s_r(0.25, 0.25, 0.4, 0.4), "drop"),  # the tie |c - s| = 0.15 is a drop, with float rounding
    (s_r(0.4, 0.4, 0.45, 0.5), "drop"),
    (s_r(0.01, 0.01, 0.02, 0.0), "drop"),  # both small
    (s_r(0.05, 0.5, 0.5, 0.6), "keep_at_step10_only"),
    (s_r(0.2, 0.2, 0.5, 0.5), "between"),  # code too high, gap over the band
    (s_r(0.05, 0.05, 0.3, 0.3), "between"),  # copy too low, gap over the band
    (s_r(-0.5, -0.5, -0.1, -0.1), "between"),  # both negative and apart
    (s_r(0.8, 0.8, 0.3, 0.3), "reversed"),
    (s_r(-0.5, -0.5, 0.5, 0.5), "code_negative"),
    (s_r(0.05, 0.05, 0.5, 0.1), "seeds_disagree"),  # keep against drop
    (s_r(None, 0.05, 0.5, 0.5), "unreadable"),  # W0's 0/0 cannot be scored
    (s_r(0.05, None, 0.5, 0.5), "unreadable"),
    (s_r(float("nan"), 0.05, 0.5, 0.5), "unreadable"),
    (s_r(0.05, 0.05, 2.0, 2.0), "keep"),  # S_R has no ceiling at 1
])
def test_the_filed_verdict_at_its_edges(read, want):
    assert WS.verdict(read, TH)["verdict"] == want


def test_the_flags_beside_the_verdict():
    assert WS.verdict(s_r(0.01, 0.01, 0.02, 0.0), TH)["drop_both_small"]
    assert not WS.verdict(s_r(0.4, 0.4, 0.45, 0.5), TH)["drop_both_small"]
    assert WS.verdict(s_r(0.05, 0.05, 0.5, 0.6, gauss=0.5), TH)["gaussian_like_copy"]
    assert not WS.verdict(s_r(0.05, 0.05, 0.5, 0.6, gauss=0.0), TH)["gaussian_like_copy"]
    assert not WS.verdict(s_r(0.05, 0.05, 0.5, 0.6, gauss=None), TH)["gaussian_like_copy"]


# ----------------------------------------------------------------------------- the real forward, tiny


def test_reads_on_the_tiny_model_are_compute_ci_with_their_inputs():
    model = make_model()
    rows = make_batches(1)[0]
    a, g = WS.inputs_and_gates(model, rows)
    ci = vpd.compute_ci(model, rows)
    for k in KEYS:
        assert torch.equal(g[k], ci.g[k].reshape(-1, ci.g[k].shape[-1]))
        assert a[k].shape == (rows.numel(), vpd.target_linear(model, k).in_features)


def test_a_zero_read_is_exactly_zero_and_w0_is_restored():
    model = make_model()
    W0, comps = vpd.target_weights(model), vpd.component_weights(model)
    ms = WS.read_state(model, W0, comps, {k: torch.zeros_like(v) for k, v in W0.items()}, {}, make_batches(2))
    assert all(m.is_zero() and m.positions > 0 for m in ms.values())
    assert all(torch.equal(vpd.target_weights(model)[k], W0[k]) for k in KEYS)


def test_g1_is_the_updated_models_and_second_moments_are_w0s():
    model = make_model()
    W0, comps = vpd.target_weights(model), vpd.component_weights(model)
    chunks = make_batches(2)
    dW = {k: 0.5 * rnd(*v.shape) for k, v in W0.items()}
    ms = WS.read_state(model, W0, comps, dW, {}, chunks)
    assert WS.pooled(ms)["mean_abs_dg"] > 0
    assert WS.pooled(ms)["nothing"]["sum"] == pytest.approx(1.0, abs=1e-5)
    S = WS.second_moments(model, chunks)
    x = torch.cat([WS.inputs_and_gates(model, r)[0]["fc2"] for r in chunks]).double()
    assert torch.allclose(S["fc2"], x.T @ x / len(x))


# ----------------------------------------------------------------------------- the job's stages, end to end on the tiny model


def run_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("weight_side_run", REPO / "experiments" / "weight_side_read" / "run.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # a dataclass reads its module at definition
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def job(tmp_path):
    """A Job on the tiny model with fine-tunes and students written to disk in their real formats; the table holds the sizes they really have."""
    R = run_module()
    model = make_model()
    alive = full_alive(model)
    uv = theory_uv(model, alive)
    W0, comps = vpd.target_weights(model), vpd.component_weights(model)
    fine, students = tmp_path / "fine_tunes", tmp_path / "students"
    fine.mkdir()
    students.mkdir()
    for name in (*R.ORDER[:-1], "rand32b_s0"):
        torch.save({k: 0.05 * rnd(*v.shape) for k, v in W0.items()}, fine / f"{name}.pt")
    filed, table = {}, {}
    for name in ("ident32_s0", "rand32b_s0"):
        offsets = {k: ((0.01 * rnd(*uv[k][0].shape)).bfloat16(), (0.01 * rnd(*uv[k][1].shape)).bfloat16()) for k in KEYS}
        torch.save({"offsets": offsets, "layout": "theory (d_out, n), (d_in, n)"}, students / f"{name}.pt")
        rel = WS.rel_norm_max(saved_states.offsets_state(students / f"{name}.pt", uv)[0], W0)
        filed[name] = {"curve": [{"rel_norm_max": rel}]}
        table[name] = rel / WS.rel_norm_max(torch.load(fine / f"{name}.pt", weights_only=False), W0)
    P = {**FILED, "size_ratio_table": table, "thresholds": {**TH, "identity_tol": 1e-3, "sum_tol": 1e-3}}
    return R, R.Job(model, W0, comps, uv, alive, split.bases(uv, 1e-6), make_batches(2), P, {}, tmp_path / "run.json", fine, students, filed)


def test_the_whole_job_on_the_tiny_model(job, capsys):
    R, J = job
    R.run(J)
    out = json.loads(J.results.read_text())
    assert not out["stage_a_zero"]["matrices_not_exactly_zero"]
    assert out["stage_a_ones_ident32_s0"]["S_D"] == pytest.approx(1.0, abs=1e-3)
    for name in R.ORDER:
        assert out[f"state_{name}"]["split"]["sum"] == pytest.approx(1.0, abs=1e-3)
        assert set(out[f"state_{name}"]["per_matrix"]) == set(KEYS)
    assert out["verdict"]["verdict"] in FILED["verdict_rules"]["verdicts"]
    assert "VERDICT" in capsys.readouterr().out


def test_a_missed_control_stops_before_any_state(job):
    R, J = job
    J.filed["ident32_s0"]["curve"][-1]["rel_norm_max"] *= 2
    with pytest.raises(SystemExit):
        R.run(J)
    out = json.loads(J.results.read_text())
    assert "stopped" in out and not any(k.startswith("state_") for k in out)
