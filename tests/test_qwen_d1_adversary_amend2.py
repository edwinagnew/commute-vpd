"""E80 Amendment 2's filed rules on their edge cases: a missing seed, NaN, a tie on each bar, a gap on the spread, the spread at zero, a zero W0, every reading branch, a zero gain,
each control's edge. And what each state leaves in Δ, on small matrices, through the same `change` as the job."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from commute import split
from commute.jobs import load_experiment

r2 = load_experiment("qwen_d1_adversary", "rules_amend2")
EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
NAN = math.nan
UNREAD, PASS, FAIL, BETWEEN = "unread", "pass", "fail", "between"


def starts(**kl: float) -> dict[str, dict[str, float]]:
    return {k: {"kl": v} for k, v in kl.items()}


def full(**over: float) -> dict[str, dict[str, float]]:
    kl = {"W0_W_s0": 1.70, "W0_W_s1": 1.80}
    for x, v in (("N", 1.28), ("LS", 1.30), ("ADJ", 1.10), ("A", 1.20)):
        kl |= {f"{x}_W_s0": v * 1.70, f"{x}_W_s1": v * 1.80}
    return starts(**{**kl, **over})


def test_verdict_ties_on_each_bar() -> None:
    assert r2.verdict(1.40) == PASS and r2.verdict(1.61) == FAIL
    assert r2.verdict(1.4000000004) == PASS  # rounds onto the bar
    assert r2.verdict(1.4000001) == BETWEEN and r2.verdict(1.6099999) == BETWEEN
    assert r2.verdict(NAN) == UNREAD


def test_mean_with_a_missing_seed() -> None:
    assert r2.mean_r((1.2, 1.4)) == pytest.approx(1.3)
    assert math.isnan(r2.mean_r((1.2, NAN)))
    s = full()
    del s["LS_W_s1"]
    out = r2.score(s, {})
    assert out["verdict"]["LS"] == UNREAD and out["described"]["LS"] == UNREAD and out["verdict"]["N"] == PASS
    assert out["leans"]["LS_passes"] is None


def test_spread_at_zero_and_unread() -> None:
    assert r2.spread(1.7, 1.8) == pytest.approx(0.1 / 1.7)
    assert r2.spread(1.8, 1.7) == pytest.approx(0.1 / 1.7)
    for a, b in ((1.7, 1.7), (NAN, 1.7), (1.7, NAN), (0.0, 1.7), (-1.0, 1.7)):
        assert math.isnan(r2.spread(a, b)), (a, b)


def test_described_needs_the_gap_to_beat_the_spread() -> None:
    assert r2.described(1.61, 0.60) == FAIL
    assert r2.described(1.61, 0.61) == r2.INSIDE_SPREAD  # a gap exactly on the spread has not beaten it
    assert r2.described(1.8, 0.9) == r2.INSIDE_SPREAD
    assert r2.described(1.8, NAN) == r2.SPREAD_UNREAD  # equal W0 seeds: never called failing
    assert r2.described(1.5, 0.01) == BETWEEN and r2.described(1.2, 0.5) == PASS and r2.described(NAN, 0.1) == UNREAD


def test_reading_every_branch() -> None:
    assert r2.reading(PASS) == r2.CANT_TELL == "we can't yet tell on Qwen"
    assert r2.reading(r2.INSIDE_SPREAD) == r2.CANT_TELL and r2.reading(r2.SPREAD_UNREAD) == r2.CANT_TELL
    assert "between" in r2.reading(BETWEEN) and "fails do-nothing" in r2.reading(FAIL)
    assert r2.reading(UNREAD) == UNREAD


def test_score_do_nothing_passes_and_fails() -> None:
    out = r2.score(full(), {})
    assert out["reading"] == r2.CANT_TELL and out["described"]["N"] == PASS and out["leans"]["N_passes"] is True
    assert out["W0_spread"] == pytest.approx(0.1 / 1.7) and out["E80_spread_quoted"] == 0.16
    assert out["gap"]["N"] == pytest.approx(0.28)
    fails = r2.score(full(N_W_s0=2.0 * 1.70, N_W_s1=2.0 * 1.80, LS_W_s0=1.7 * 1.70, LS_W_s1=1.7 * 1.80), {})
    assert "fails do-nothing" in fails["reading"] and fails["described"]["LS"] == FAIL and fails["leans"]["LS_passes"] is False


def test_score_a_wide_spread_keeps_a_method_from_failing() -> None:
    out = r2.score(full(W0_W_s1=3.4, LS_W_s0=1.7 * 1.70, LS_W_s1=1.7 * 3.4, N_W_s0=1.2 * 1.70, N_W_s1=1.2 * 3.4), {})
    assert out["W0_spread"] == pytest.approx(1.0) and out["verdict"]["LS"] == FAIL and out["described"]["LS"] == r2.INSIDE_SPREAD


def test_score_equal_w0_seeds_and_a_zero_w0() -> None:
    out = r2.score(full(W0_W_s1=1.70, LS_W_s0=1.7 * 1.70, LS_W_s1=1.7 * 1.70), {})
    assert math.isnan(out["W0_spread"]) and out["described"]["LS"] == r2.SPREAD_UNREAD
    zero = r2.score(full(W0_W_s0=0.0), {})
    assert zero["verdict"]["N"] == UNREAD and zero["reading"] == UNREAD


def test_score_empty_and_nan() -> None:
    out = r2.score({}, {})
    assert out["reading"] == UNREAD and math.isnan(out["kept"]["ADJ"]) and all(v is None for v in out["leans"].values())
    assert r2.score(full(N_W_s0=NAN), {})["reading"] == UNREAD


def test_kept_ends_and_no_gain() -> None:
    assert r2.kept(1.0, 2.0, 1.0) == 1.0 and r2.kept(2.0, 2.0, 1.0) == 0.0 and r2.kept(1.5, 2.0, 1.0) == 0.5
    assert r2.kept(0.5, 2.0, 1.0) == 1.5 and r2.kept(2.5, 2.0, 1.0) == -0.5  # outside 0-1 is read as it is
    for x, w0, ft in ((1.0, 2.0, 2.0), (1.0, 2.0, 2.5), (NAN, 2.0, 1.0), (1.0, NAN, 1.0), (1.0, 2.0, NAN)):
        assert math.isnan(r2.kept(x, w0, ft)), (x, w0, ft)


def test_score_kept_and_leans_on_half() -> None:
    gold = {"W0": 2.5, "FT": 0.5, "N": 0.5, "LS": 0.5, "ADJ": 1.5, "A": 1.5, "W0_end": 2.5}
    out = r2.score(full(), gold)
    assert out["kept"] == {"N": 1.0, "LS": 1.0, "ADJ": 0.5, "A": 0.5, "W0_end": 0.0} and out["gain"] == 2.0
    assert out["leans"]["ADJ_under_half"] is False and out["leans"]["A_over_half"] is False  # exactly half holds neither


def test_controls_at_their_edges() -> None:
    assert r2.near(1.001, 1.0, r2.KEPT_TOL) and not r2.near(1.0011, 1.0, r2.KEPT_TOL) and not r2.near(NAN, 1.0, r2.KEPT_TOL)
    assert r2.near_rel(r2.W0_GOLD * 1.0009, r2.W0_GOLD, r2.W0_TOL) and not r2.near_rel(r2.W0_GOLD * 1.002, r2.W0_GOLD, r2.W0_TOL)
    assert r2.near_rel(r2.FT_GAIN * 0.96, r2.FT_GAIN, r2.FT_TOL) and not r2.near_rel(r2.FT_GAIN * 0.94, r2.FT_GAIN, r2.FT_TOL)


def test_filed_controls_are_the_files() -> None:
    sft = EXPERIMENTS / "qwen_sft" / "results"
    assert json.loads((sft / "A0.json").read_text())["gold_loss"] == r2.W0_GOLD
    assert json.loads((sft / "F.json").read_text())["saved"]["10"]["gold_drop"] == r2.FT_GAIN
    S = json.loads((EXPERIMENTS / "qwen_pass_test" / "results" / "u1" / "S.json").read_text())
    assert (S["energy"], S["share"]["ls"], S["share"]["adjoint"]) == (r2.E68_S["energy"], r2.E68_S["LS"], r2.E68_S["ADJ"])


def test_predictions_file_names_the_rules() -> None:
    p = json.loads((EXPERIMENTS / "qwen_d1_adversary" / "results" / "predictions_amend2.json").read_text())
    assert "1.40" in p["rules"]["verdict"] and "1.61" in p["rules"]["verdict"] and "strictly above" in p["rules"]["described"]
    assert p["rules"]["reading"]["do-nothing passes"] == r2.CANT_TELL
    assert list(p["leans"]) == list(r2.score({}, {})["leans"]) and "16%" in p["read"]


# ------------------------------------------------------------------ the states, on small matrices, with no model


def job(tmp: Path) -> SimpleNamespace:
    """One site "s": a job holding what `change` reads, X from the same function make_x calls."""
    torch.manual_seed(0)
    U, V, M = torch.randn(8, 3).double(), torch.randn(6, 3).double(), torch.randn(8, 6).double()
    b = split.basis_from(split.gram_basis(U), split.gram_basis(V), 1e-6)
    save_file({"X_ls": split.least_squares_x(M, b).float().contiguous()}, str(tmp / "s.safetensors"))
    S = torch.eye(6).double()
    alive = SimpleNamespace(uv=lambda s, d, t: (U, V))
    return SimpleNamespace(alive=alive, dw={"s": M}, x_dir=tmp, out={"c": {"s": split.adjoint_scale(M, U, V, S)[0]}}, U=U, V=V, M=M, b=b)


def test_ls_leaves_the_remainder_and_adjoint_leaves_only_the_product(tmp_path: Path) -> None:
    a2 = load_experiment("qwen_d1_adversary", "amend2")
    a2.DEV = "cpu"
    J = job(tmp_path)
    ls = a2.change(J, "ls")("s")
    assert ls[0] is not None and torch.equal(ls[0], J.M)  # weights W0 + δW
    want = split.obstruction(J.M, J.b) - ls[2] @ ls[3].T
    assert torch.allclose(a2.moved(ls, J.U, J.V), want, atol=1e-5)  # the remainder, less the second-order term; X is fp32
    adj = a2.change(J, "adjoint")("s")
    c = J.out["c"]["s"]
    assert torch.allclose(adj[2], c * (J.M @ J.V)) and torch.allclose(adj[3], c * (J.M.T @ J.U))
    assert torch.allclose(a2.moved(adj, J.U, J.V), -adj[2] @ adj[3].T)  # Δ moves by the product alone


def test_order_and_files_are_this_amendments() -> None:
    a2 = load_experiment("qwen_d1_adversary", "amend2")
    assert a2.ORDER == (("LS", 0), ("ADJ", 0), ("LS", 1), ("ADJ", 1))
    assert a2.run.OUT.name == "amend2.json" and a2.run.LOG.name == "run_amend2.log" and a2.run.STOPPED.name == "stopped_amend2.json"
    assert set(a2.amend1.STATE) >= {"W0", "N", "A", "LS", "ADJ"}
