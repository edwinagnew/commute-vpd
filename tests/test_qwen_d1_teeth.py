"""E85's filed rule on its edge cases, and the two states on small matrices through the job's own functions."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors.torch import save_file

from commute import qwen_maps, split
from commute.jobs import load_experiment

R = load_experiment("qwen_d1_teeth", "rules")
PRED = Path(__file__).resolve().parent.parent / "experiments" / "qwen_d1_teeth" / "results" / "predictions.json"
NAN = math.nan


def out(rand: tuple[float, float], rare: tuple[float, float], spread: float = 0.18) -> str:
    return R.outcome({"Rand": rand, "Rare": rare}, spread)


def test_both_under_the_pass_bar_and_on_it() -> None:
    assert out((1.2, 1.3), (1.2, 1.3)) == R.CANNOT
    assert out((1.40, 1.40), (1.0, 1.0)) == R.CANNOT


def test_on_the_fail_bar_is_teeth_either_state() -> None:
    assert out((1.61, 1.61), (1.2, 1.2)) == R.TEETH
    assert out((1.2, 1.2), (1.61, 1.61)) == R.TEETH


def test_one_failing_one_passing_and_one_unread() -> None:
    assert out((2.0, 2.0), (1.1, 1.1)) == R.TEETH
    assert out((2.0, 2.0), (NAN, 1.1)) == R.TEETH
    assert out((1.2, 1.2), (NAN, 1.1)) == R.UNREAD


def test_between_the_bars_is_no_call() -> None:
    assert out((1.5, 1.5), (1.5, 1.5)) == R.NO_CALL
    assert out((1.5, 1.5), (1.2, 1.2)) == R.NO_CALL
    assert out((1.41, 1.41), (1.0, 1.0)) == R.NO_CALL


def test_a_pair_straddling_a_bar() -> None:
    assert out((1.9, 1.35), (1.2, 1.2)) == R.TEETH  # mean 1.625
    assert out((1.75, 1.45), (1.2, 1.2)) == R.NO_CALL  # mean 1.60
    assert out((1.7, 1.0), (1.2, 1.2)) == R.NO_CALL  # mean 1.35: a seed over the fail bar blocks "cannot"
    assert R.straddles((1.9, 1.35)) is True and R.straddles((1.2, 1.3)) is False and R.straddles((NAN, 1.3)) is None


def test_the_spread_zero_wide_and_on_the_gap() -> None:
    assert out((2.0, 2.0), (1.2, 1.2), NAN) == R.NO_CALL  # W0's seeds equal: no spread measured
    assert out((1.2, 1.2), (1.2, 1.2), NAN) == R.CANNOT  # no state over a bar, the spread is not read
    assert out((1.7, 1.7), (1.2, 1.2), 0.8) == R.NO_CALL  # over the bar, inside the spread
    assert out((1.7, 1.7), (1.2, 1.2), 0.7) == R.NO_CALL  # the gap on the spread has not beaten it
    assert R.r2.spread(1.5, 1.5) != R.r2.spread(1.5, 1.5)  # NaN


def test_score_empty_nan_and_a_zero_w0() -> None:
    for starts in ({}, {"W0_W_s0": {"kl": NAN}}, {"W0_W_s0": {"kl": 0.0}, "W0_W_s1": {"kl": 0.0}, "Rand_W_s0": {"kl": 2.0}, "Rand_W_s1": {"kl": 2.0}}):
        s = R.score(starts, {}, {}, {})
        assert s["outcome"] == R.UNREAD and s["leans"]["Rand_passes"] is None and s["leans"]["Rand_kept_near_zero"] is None


def test_score_reads_seed_by_seed() -> None:
    kl = {"W0_W_s0": 1.7, "W0_W_s1": 1.4, "Rand_W_s0": 1.7 * 1.3, "Rand_W_s1": 1.4 * 1.1, "Rare_W_s0": 1.7 * 1.2, "Rare_W_s1": 1.4 * 1.2}
    s = R.score({k: {"kl": v} for k, v in kl.items()}, {"W0": 2.5, "FT": 0.5, "Rand": 2.5, "Rare": 0.5}, {"Rand": 1.0, "Rare": 0.4}, {})
    assert math.isclose(s["mean"]["Rand"], 1.2) and math.isclose(s["mean"]["Rare"], 1.2) and s["outcome"] == R.CANNOT
    assert s["kept"] == {"Rand": 0.0, "Rare": 1.0} and math.isclose(s["W0_spread"], 0.3 / 1.4)
    assert s["leans"]["Rand_near_do_nothing"] is True and s["leans"]["Rand_kept_near_zero"] is True


def test_rare_left_edges() -> None:
    assert not R.rare_left_ok(R.RARE_LEFT_LOW) and not R.rare_left_ok(1.0) and not R.rare_left_ok(NAN) and R.rare_left_ok(0.35)


def test_filed_numbers_are_the_files() -> None:
    root = PRED.parent.parent.parent / "qwen_d1_adversary" / "results"
    a2 = json.loads((root / "amend2.json").read_text())
    a1 = json.loads((root / "amend1.json").read_text())["starts"]
    assert R.W0_WANT == {0: a2["a1"]["W0_W_s0"]["kl"], 1: a2["a1"]["W0_W_s1"]["kl"]}
    assert R.N_MEAN == a2["score"]["mean"]["N"] and R.RARE_LEFT_LOW == a2["left_in_delta"]["ls"]["plain_energy"]
    assert math.isclose(R.I_MEAN, (a1["I_W_s0"]["kl"] / a1["W0_W_s0"]["kl"] + a1["I_W_s1"]["kl"] / a1["W0_W_s1"]["kl"]) / 2, rel_tol=1e-6)


def test_predictions_file_holds_the_rule() -> None:
    p = json.loads(PRED.read_text())
    assert p["rules"]["bars"] == {"pass_at": R.r2.PASS_AT, "fail_at": R.r2.FAIL_AT}
    for name in (R.TEETH, R.CANNOT, R.NO_CALL):
        assert name in p["rules"]["outcome"]


def test_gaussian_keeps_the_norm_and_repeats() -> None:
    M = torch.randn(6, 4, dtype=torch.float64)
    A, B = qwen_maps.gaussian_at_norm(M, 75_100), qwen_maps.gaussian_at_norm(M, 75_100)
    assert torch.equal(A, B) and math.isclose(float(A.norm()), float(M.norm()), rel_tol=1e-12)


def test_rare_x_uses_the_lower_half_and_leaves_more_than_all_alive() -> None:
    teeth = load_experiment("qwen_d1_teeth", "run")
    gen = torch.Generator().manual_seed(0)
    U, V, M = (torch.randn(*shape, generator=gen, dtype=torch.float64) for shape in ((8, 6), (9, 6), (8, 9)))
    active = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    x = teeth.rare_x(M, U, V, active, active, 1e-6)
    pos = x["pos_lower"]
    assert pos.tolist() == [0, 2, 4]
    c = qwen_maps.site_change("lower", M, U, V, x.__getitem__, 0.0, 0.0)
    dW, moved_pos, dU, dV = c
    assert dW is not None and moved_pos is not None and dU is not None and dV is not None
    assert torch.equal(dW, M) and torch.equal(moved_pos, pos)
    left = teeth.amend2.moved(c, U[:, pos], V[:, pos])
    b = split.basis_from(split.gram_basis(U), split.gram_basis(V), 1e-6)
    X = split.least_squares_x(M, b)
    first_all = M - split.dsigma(X @ V, X.T @ U, U, V)
    first_half = M - split.dsigma(dU, dV, U[:, pos], V[:, pos])
    assert float(first_half.norm()) > float(first_all.norm())
    assert torch.allclose(left, first_half - dU @ dV.T)


def test_left_in_delta_and_the_checks_on_every_site(tmp_path: Path, monkeypatch: Any) -> None:
    """make_rare, rare_change, rand_change and left_in_delta through the job's own code, on one small site repeated, with E80's files pointed away."""
    teeth = load_experiment("qwen_d1_teeth", "run")
    for name, value in (("OUT", tmp_path / "r.json"), ("STOPPED", tmp_path / "stopped.json"), ("LOG", tmp_path / "r.log")):
        monkeypatch.setattr(teeth.e80, name, value)
    monkeypatch.setattr(teeth, "DEV", "cpu")
    gen = torch.Generator().manual_seed(0)
    U, V, M = (torch.randn(*shape, generator=gen, dtype=torch.float64) for shape in ((8, 6), (9, 6), (8, 9)))
    idx = torch.tensor([0, 1, 2, 3, 4, 6])
    active, sites = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.0, 0.7]), teeth.SITES
    save_file({k: active.clone() for s in sites for k in (f"{s}.active", f"{s}.mean_g")}, str(tmp_path / "W0_maths.safetensors"))
    alive = SimpleNamespace(uv=lambda s, d, t: (U, V), idx=dict.fromkeys(sites, idx))
    J = SimpleNamespace(alive=alive, dw=dict.fromkeys(sites, M.float()), x_dir=tmp_path / "X", inputs=tmp_path, out={"checks": {}})
    teeth.left_in_delta(J)
    left, act = J.out["left_in_delta"], J.out["active_share"]
    assert math.isclose(left["Rand"], 1.0, rel_tol=1e-9) and R.rare_left_ok(left["Rare"])
    assert math.isclose(act["half"], 0.2, rel_tol=1e-6) and act["half"] < act["alive"]
    assert set(J.out["checks"]) == {"Rand_size", "Rare_left", "half_is_rarer"} and not (tmp_path / "stopped.json").exists()
    G = teeth.rand_change(J, sites[3])
    assert G[1:] == (None, None, None) and not torch.equal(G[0], teeth.rand_change(J, sites[4])[0])
