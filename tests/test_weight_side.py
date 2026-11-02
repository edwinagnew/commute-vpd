"""CPU checks of commute.weight_side (E67) and experiments/weight_side_read/run.py, on the tiny ComponentModel of test_faithfulness_invariants: S_R's positive control,
the zero read, the leave-one-chunk-out spread, the filed verdict at its edge cases, the job end to end."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from test_faithfulness_invariants import KEYS, make_batches, make_model
from torch import Tensor

from commute import split, vpd
from commute import weight_side as WS

REPO = Path(__file__).resolve().parents[1]
FILED = json.loads((REPO / "experiments" / "weight_side_read" / "results" / "predictions.json").read_text())
TH = FILED["thresholds"]
GEN = torch.Generator().manual_seed(0)


def rnd(*shape: int) -> Tensor:
    return torch.randn(*shape, generator=GEN)


def s_r(c10: float, c100: float, s0: float, s1: float, gauss: float = 0.0, held: float = 0.0) -> dict[str, float]:
    return {"code_step10": c10, "code_step100": c100, "ident32_s0": s0, "ident32_s1": s1, "gaussian": gauss, "code_step10_held": held, "ident32_s0_held": 0.3}


def test_re_switching_reads_one_on_its_own_direction_and_zero_orthogonal():
    """Positive control and blind spot of S_R, unit test only: a gate moved by 1 on one subcomponent, with the update equal to that subcomponent, then writing elsewhere."""
    U, V, a = torch.zeros(2, 4), torch.zeros(3, 2), rnd(30, 3)
    U[0, 0], V[0, 0], U[1, 1], V[1, 1] = 1.0, 1.0, 1.0, 1.0
    g0, g1 = torch.zeros(30, 2), torch.zeros(30, 2)
    g1[:, 0] = 1.0
    same, other = torch.zeros(4, 3), torch.zeros(4, 3)
    same[0, 0], other[2, 0] = 1.0, 1.0
    a_on, a_off = WS.Matrix(), WS.Matrix()
    a_on.add(a, g0, g1, U, V, same)
    a_off.add(a, g0, g1, U, V, other)
    assert sum(a_on.r) / sum(a_on.den) == pytest.approx(1.0)
    assert sum(a_off.r) == 0


def test_pooled_sums_chunks_over_matrices_and_leaves_each_chunk_out():
    x, y = WS.Matrix(r=[1.0, 2.0, 3.0], den=[2.0, 2.0, 2.0], abs_dg=8.0, positions=2, C=4), WS.Matrix(r=[1.0, 0.0, 1.0], den=[2.0, 2.0, 2.0], abs_dg=0.0, positions=2, C=4)
    p = WS.pooled({"x": x, "y": y})
    assert p["chunks"] == {"r": [2.0, 2.0, 4.0], "den": [4.0, 4.0, 4.0]}
    assert p["S_R"] == pytest.approx(8 / 12)
    assert p["mean_abs_dg"] == pytest.approx(8 / 16)
    assert p["leave_one_chunk_out"] == {"min": pytest.approx(4 / 8), "max": pytest.approx(6 / 8), "spread": pytest.approx(2 / 8)}


def test_matched_gaussian_matches_the_weighted_energy_and_is_seeded():
    x = rnd(8, 4).double()
    M, S = {"a": rnd(6, 4)}, {"a": x.T @ x}
    G = WS.matched_gaussian(M, S, 67)
    assert split.weighted_energy(G["a"].double(), S["a"]) == pytest.approx(split.weighted_energy(M["a"].double(), S["a"]), rel=1e-5)
    assert torch.equal(G["a"], WS.matched_gaussian(M, S, 67)["a"])


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
    (s_r(0.05, 0.05, 2.0, 2.0), "keep"),  # S_R has no ceiling at 1
])
def test_the_filed_verdict_at_its_edges(read, want):
    assert WS.verdict(read, TH)["verdict"] == want


@pytest.mark.parametrize(("read", "want"), [
    (s_r(0.05, 0.05, 0.5, 0.1), "seeds_disagree"),  # keep against drop: the two cannot sit under the spread's bar, keep's gap is 0.3 and drop's 0.15
    (s_r(0.3, 0.3, 0.4, 0.5), "seeds_disagree"),  # drop against between at the tie, a spread of 0.1, is not under it
    (s_r(0.3, 0.3, 0.4, 0.49), "between"),  # the same labels with a spread just under it
    (s_r(0.05, 0.05, 0.401, 0.399), "between"),  # keep against between on a spread of 0.002
    (s_r(-0.2, -0.2, 0.41, 0.39), "between"),  # code_negative against between
    (s_r(0.5, 0.5, 0.36, 0.34), "between"),  # drop against reversed
    (s_r(0.3, 0.3, 0.5, 0.5), "between"),  # a spread of 0: the labels cannot differ, so the common label stands
    (s_r(0.05, 0.05, 0.5, 0.5), "keep"),
    (s_r(0.5, 0.5, 0.45, 0.45), "drop"),
])
def test_labels_that_differ_across_seeds(read, want):
    assert WS.verdict(read, TH)["verdict"] == want


def test_a_pool_of_floats_always_gets_a_filed_label():
    """No None or NaN reaches the verdict: the job stops first. Every case of floats has a branch, the filed labels, and never a stop's."""
    grid = (-0.5, -0.1, 0.0, 0.05, 0.1, 0.25, 0.3, 0.4, 0.5, 0.9, 2.0)
    seen = {WS.verdict(s_r(c, c, a, b), TH)["verdict"] for c, a, b in itertools.product(grid, grid, grid)}
    assert seen == set(FILED["verdict_rules"]["verdicts"]) - {"dead_read", "unreadable", "keep_at_step10_only"}


@pytest.mark.parametrize(("read", "want", "flag"), [
    (s_r(0.05, 0.05, 0.5, 0.6, held=0.11), "between", True),  # a keep, taken back
    (s_r(0.05, 0.5, 0.5, 0.6, held=0.11), "between", True),  # a keep at step 10 only, taken back
    (s_r(0.05, 0.05, 0.5, 0.6, held=0.1), "keep", False),  # a tie is not over
    (s_r(0.05, 0.5, 0.5, 0.6, held=0.1), "keep_at_step10_only", False),
    (s_r(0.05, 0.05, 0.5, 0.6, held=-0.5), "keep", False),  # the rule is one-sided
    (s_r(0.4, 0.4, 0.45, 0.5, held=0.9), "drop", False),  # a drop stands whatever the held rows read
    (s_r(0.8, 0.8, 0.3, 0.3, held=0.9), "reversed", False),
    (s_r(0.05, 0.05, 0.5, 0.1, held=0.9), "seeds_disagree", False),
])
def test_the_held_rows_can_only_take_back_a_label_that_would_adopt(read, want, flag):
    v = WS.verdict(read, TH)
    assert v["verdict"] == want
    assert v["held_rows_fail"] is flag


def test_the_flags_beside_the_verdict():
    assert WS.verdict(s_r(0.01, 0.01, 0.02, 0.0), TH)["drop_both_small"]
    assert not WS.verdict(s_r(0.4, 0.4, 0.45, 0.5), TH)["drop_both_small"]
    assert WS.verdict(s_r(0.05, 0.05, 0.5, 0.6, gauss=0.5), TH)["gaussian_like_copy"]
    assert not WS.verdict(s_r(0.05, 0.05, 0.5, 0.6, gauss=0.0), TH)["gaussian_like_copy"]


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
    ms = WS.read_state(model, W0, comps, {k: torch.zeros_like(v) for k, v in W0.items()}, make_batches(2))
    assert all(m.is_zero() and m.positions > 0 for m in ms.values())
    assert all(torch.equal(vpd.target_weights(model)[k], W0[k]) for k in KEYS)


def test_g1_is_the_updated_models():
    model = make_model()
    W0, comps = vpd.target_weights(model), vpd.component_weights(model)
    ms = WS.read_state(model, W0, comps, {k: 0.5 * rnd(*v.shape) for k, v in W0.items()}, make_batches(2))
    assert WS.mean_abs_dg(ms) > 0


# ----------------------------------------------------------------------------- the job, end to end on the tiny model


def run_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("weight_side_run", REPO / "experiments" / "weight_side_read" / "run.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # a dataclass reads its module at definition
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def job(tmp_path):
    """A Job on the tiny model with fine-tunes written to disk in their real format."""
    R = run_module()
    model = make_model()
    W0, comps = vpd.target_weights(model), vpd.component_weights(model)
    for name in FILED["states"]["order"][:-1]:
        torch.save({k: 0.05 * rnd(*v.shape) for k, v in W0.items()}, tmp_path / f"{name}.pt")
    return R, R.Job(model, W0, comps, make_batches(2), make_batches(2, seed=2), FILED, {}, tmp_path / "run.json", tmp_path)


def test_the_whole_job_on_the_tiny_model(job, capsys):
    R, J = job
    R.run(J)
    out = json.loads(J.results.read_text())
    assert not out["stage_a_zero"]["matrices_not_exactly_zero"]
    for name in (*FILED["states"]["order"], *(f"{n}_held" for n in FILED["states"]["held_rows"])):
        assert set(out[f"state_{name}"]["per_matrix"]) == set(KEYS)
        assert len(out[f"state_{name}"]["chunks"]["r"]) == 2
        assert out[f"state_{name}"]["leave_one_chunk_out"]["spread"] >= 0
    assert out["verdict"]["verdict"] in FILED["verdict_rules"]["verdicts"]
    assert "VERDICT" in capsys.readouterr().out


def with_gates(R: ModuleType, monkeypatch: pytest.MonkeyPatch, gates) -> None:
    """Γ's values replaced, the activations left as they are."""
    real = R.WS.inputs_and_gates

    def fake(model, rows):
        a, g = real(model, rows)
        return a, {k: gates(v) for k, v in g.items()}

    monkeypatch.setattr(R.WS, "inputs_and_gates", fake)


def test_a_missed_control_stops_before_any_state(job, monkeypatch):
    R, J = job
    with_gates(R, monkeypatch, torch.rand_like)
    with pytest.raises(SystemExit):
        R.run(J)
    out = json.loads(J.results.read_text())
    assert out["stopped"]["verdict"] == "unreadable" and not any(k.startswith("state_") for k in out)


def test_a_dead_read_stops_with_its_own_verdict(job, monkeypatch):
    """The one failure the zero control cannot see: g1 = g0 on a nonzero update would read S_R = 0, the lean."""
    R, J = job
    with_gates(R, monkeypatch, torch.zeros_like)
    with pytest.raises(SystemExit):
        R.run(J)
    out = json.loads(J.results.read_text())
    assert out["stage_a_zero"]["matrices_not_exactly_zero"] == []
    assert out["stopped"]["verdict"] == "dead_read" and "verdict" not in out


def test_a_non_finite_reading_stops_before_the_verdict(job):
    """Only on the last matrix: its weights feed no Γ, so Γ stays finite and only the numerator and denominator go. Anywhere upstream param-decomp's own assert on Γ raises first."""
    R, J = job
    torch.save({k: torch.full_like(v, float("nan") if k == "fc2" else 0.05) for k, v in J.W0.items()}, J.fine_tunes / "ident32_s0.pt")
    with pytest.raises(SystemExit):
        R.run(J)
    out = json.loads(J.results.read_text())
    assert out["stopped"]["verdict"] == "unreadable" and "verdict" not in out
