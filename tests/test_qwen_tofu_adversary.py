"""E89's filed rules on the plan's edge cases, and the rarely-on half's left in Δ on small matrices through the job's own functions."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from commute.jobs import load_experiment

R = load_experiment("qwen_tofu_adversary", "rules")
EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
PRED = EXPERIMENTS / "qwen_tofu_adversary" / "results" / "predictions.json"
NAN = math.nan
SPREAD = 0.18


def test_a_state_on_the_bars() -> None:
    assert R.label((1.61, 1.61), SPREAD) == R.FAILS and R.label((1.40, 1.40), SPREAD) == R.PASSES
    assert R.label((1.60, 1.60), SPREAD) == R.NO_CALL and R.label((1.41, 1.41), SPREAD) == R.NO_CALL


def test_a_pair_straddling_a_bar() -> None:
    assert R.label((1.9, 1.35), SPREAD) == R.FAILS  # mean 1.625
    assert R.label((1.75, 1.45), SPREAD) == R.NO_CALL  # mean 1.60
    assert R.label((1.7, 1.0), SPREAD) == R.NO_CALL  # mean 1.35, a seed at or over the fail bar
    assert R.label((1.61, 1.0), SPREAD) == R.NO_CALL  # the seed on the bar counts


def test_w0_seeds_equal_is_no_spread() -> None:
    s = R.r2.spread(1.5, 1.5)
    assert math.isnan(s)
    assert R.label((2.0, 2.0), s) == R.NO_CALL  # over the bar, no spread measured
    assert R.label((1.2, 1.2), s) == R.PASSES  # the spread is not read under the bar


def test_a_spread_wider_than_the_gap() -> None:
    assert R.label((1.7, 1.7), 0.8) == R.NO_CALL
    assert R.label((1.7, 1.7), 0.7) == R.NO_CALL  # the gap on the spread has not beaten it
    assert R.label((1.7, 1.7), 0.69) == R.FAILS


def test_nan_is_unread() -> None:
    assert R.label((NAN, 1.2), SPREAD) == R.UNREAD and R.label((2.0, NAN), SPREAD) == R.UNREAD


def test_do_nothing_labels_verbatim() -> None:
    assert R.do_nothing(R.FAILS) == "do-nothing fails on this larger update"
    assert R.do_nothing(R.PASSES) == "do-nothing passes on this larger update too"
    assert R.do_nothing(R.NO_CALL) == "no call" and R.do_nothing(R.UNREAD) == "unread"


def test_pair_labels_verbatim_and_its_else() -> None:
    assert R.pair(R.FAILS, R.PASSES) == "the adversary separates them"
    assert R.pair(R.FAILS, R.FAILS) == "both fail"
    assert R.pair(R.PASSES, R.FAILS) == "rarely-on alone fails"
    assert R.pair(R.PASSES, R.PASSES) == "both pass"
    for n, rare in ((R.FAILS, R.NO_CALL), (R.NO_CALL, R.PASSES), (R.NO_CALL, R.NO_CALL), (R.FAILS, R.UNREAD), (R.UNREAD, R.PASSES), (R.UNREAD, R.UNREAD)):
        assert R.pair(n, rare) == "no call"


def kls(**kw: float) -> dict[str, dict[str, float]]:
    return {f"{k[:-3]}_W_{k[-2:]}": {"kl": v} for k, v in kw.items()}


def test_score_empty_nan_and_a_zero_w0() -> None:
    zero = kls(W0_s0=0.0, W0_s1=0.0, N_s0=2.0, N_s1=2.0, Rare_s0=2.0, Rare_s1=2.0)
    for starts in ({}, {"W0_W_s0": {"kl": NAN}}, zero):
        s = R.score(starts, {}, {}, {})
        assert s["do_nothing"] == "unread" and s["pair"] == "no call"
        assert all(v is None for v in s["leans"].values())


def test_score_one_state_unread() -> None:
    s = R.score(kls(W0_s0=1.7, W0_s1=1.4, N_s0=17.0, N_s1=14.0, Rare_s0=2.0), {}, {}, {})
    assert s["do_nothing"] == R.N_FAILS and s["label"]["Rare"] == R.UNREAD and s["pair"] == "no call"
    assert s["leans"]["N_fails"] is True and s["leans"]["both_fail"] is None and s["leans"]["Rare_below_N_beyond_spread"] is None


def test_score_reads_seed_by_seed_with_excess_and_last5() -> None:
    starts: dict[str, Any] = kls(W0_s0=1.7, W0_s1=1.4, N_s0=17.0, N_s1=14.0, Rare_s0=1.7 * 1.2, Rare_s1=1.4 * 1.2)
    starts["N_W_s0"]["last5"] = 16.0
    s = R.score(starts, {"N": 0.5, "Rare": 0.5}, {"Rare": 0.4}, {"x": 1})
    assert math.isclose(s["mean"]["N"], 10.0) and math.isclose(s["mean"]["Rare"], 1.2) and math.isclose(s["W0_spread"], 0.3 / 1.4)
    assert s["do_nothing"] == R.N_FAILS and s["pair"] == R.SEPARATES
    assert math.isclose(s["excess"]["N_W_s0"], 15.3) and math.isclose(s["excess"]["Rare_W_s1"], 1.4 * 0.2)
    assert s["last5"] == {"N_W_s0": 16.0} and s["clean"] == {"N": 0.5, "Rare": 0.5} and s["left_in_delta"] == {"Rare": 0.4} and s["checks"] == {"x": 1}
    assert s["leans"] == {"N_fails": True, "Rare_below_N_beyond_spread": True, "both_fail": False}


def test_rare_below_needs_more_than_the_spread() -> None:
    starts = kls(W0_s0=2.0, W0_s1=1.0, N_s0=3.0, N_s1=1.5, Rare_s0=2.2, Rare_s1=1.1)  # means 1.5, 1.1; a gap of 0.4 inside the spread of 1
    assert R.score(starts, {}, {}, {})["leans"]["Rare_below_N_beyond_spread"] is False


def test_the_stops_on_their_edges() -> None:
    assert R.n_clean_ok(1e-3) and not R.n_clean_ok(9.9e-4) and not R.n_clean_ok(NAN)
    assert R.rare_left_ok(0.3) and R.rare_left_ok(1.5)
    assert not R.rare_left_ok(R.RARE_LEFT_LOW) and not R.rare_left_ok(NAN) and not R.rare_left_ok(math.inf)
    assert R.r2.near(0.5 + 1e-6, 0.5, R.SAME_TOL) and not R.r2.near(0.5 + 2e-6, 0.5, R.SAME_TOL) and not R.r2.near(NAN, 0.5, R.SAME_TOL)


def test_filed_numbers_are_the_files() -> None:
    a2 = json.loads((EXPERIMENTS / "qwen_d1_adversary" / "results" / "amend2.json").read_text())
    assert R.W0_WANT == {0: a2["a1"]["W0_W_s0"]["kl"], 1: a2["a1"]["W0_W_s1"]["kl"]}
    assert R.RARE_LEFT_LOW == json.loads((EXPERIMENTS / "qwen_d1_fractions" / "results" / "tofu" / "left_in_delta.json").read_text())["ls"]
    own = json.loads((EXPERIMENTS / "qwen_d1_teeth" / "results" / "predictions_amend1.json").read_text())
    assert f'"pass_at": {R.OWN_PASS_AT}' in json.dumps(own) and f'"fail_at": {R.OWN_FAIL_AT}' in json.dumps(own)


def test_predictions_file_holds_the_rule() -> None:
    p = json.loads(PRED.read_text())
    assert p["rules"]["bars"] == {"pass_at": R.r2.PASS_AT, "fail_at": R.r2.FAIL_AT}
    for name in (R.N_FAILS, R.N_PASSES, R.NO_CALL, R.UNREAD):
        assert name in p["rules"]["do_nothing"]
    for name in (R.SEPARATES, R.BOTH_FAIL, R.RARE_ALONE, R.BOTH_PASS, R.NO_CALL):
        assert name in p["rules"]["pair"]


def test_verified_reads_the_committed_sum(tmp_path: Path, monkeypatch: Any) -> None:
    up = load_experiment("qwen_tofu_adversary", "upload")
    assert up.sums(up.COMMITTED) == {up.NAME: "d45b5163d431e218708d66fa45fae47ac0c5f0e7ff6666fc5ddd06adcd35a37b"}
    assert not up.verified(tmp_path)
    (tmp_path / up.NAME).write_bytes(b"not it")
    assert not up.verified(tmp_path)
    monkeypatch.setattr(up, "COMMITTED", tmp_path / "sums")
    (tmp_path / "sums").write_text(f"{hashlib.sha256(b'not it').hexdigest()}  {up.NAME}\n")
    assert up.verified(tmp_path)


def test_fetch_inputs_asks_for_this_update_and_not_e40s(tmp_path: Path, monkeypatch: Any) -> None:
    m = load_experiment("qwen_tofu_adversary", "run")
    monkeypatch.setattr(m.e80, "LOG", tmp_path / "r.log")
    asked: list[Path] = []
    for mod in (m.e80.qwen_files, m.upload, m.e80.upload):
        monkeypatch.setattr(mod, "fetch", lambda d, *_: asked.append(d))
    (tmp_path / "e81_tofu").mkdir()
    (tmp_path / "e81_tofu" / m.upload.NAME).write_bytes(b"x")
    dw, inputs = m.fetch_inputs(tmp_path / "qwen")
    assert dw == tmp_path / "e81_tofu" / "step_1250.safetensors" and inputs == tmp_path / "e80_inputs"
    assert asked == [tmp_path / "qwen", tmp_path / "e81_tofu", tmp_path / "e80_inputs"]


def test_left_in_delta_on_every_site_in_this_jobs_folder(tmp_path: Path, monkeypatch: Any) -> None:
    """make_rare, rare_change and left_in_delta through the job's code with an fp32 δW, on one small site repeated, E80's files pointed away, X in e89_rare and not E85's folder."""
    import torch
    from safetensors.torch import save_file

    m = load_experiment("qwen_tofu_adversary", "run")
    for name, value in (("OUT", tmp_path / "r.json"), ("STOPPED", tmp_path / "stopped.json"), ("LOG", tmp_path / "r.log")):
        monkeypatch.setattr(m.e80, name, value)
    monkeypatch.setattr(m, "DEV", "cpu")
    monkeypatch.setattr(m.e85, "DEV", "cpu")
    monkeypatch.setattr(m.e85, "rare_dir", m.rare_dir)
    gen = torch.Generator().manual_seed(0)
    U, V, M = (torch.randn(*shape, generator=gen, dtype=torch.float64) for shape in ((8, 6), (9, 6), (8, 9)))
    idx = torch.tensor([0, 1, 2, 3, 4, 6])
    active, sites = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.0, 0.7]), m.SITES
    save_file({k: active.clone() for s in sites for k in (f"{s}.active", f"{s}.mean_g")}, str(tmp_path / "W0_maths.safetensors"))
    alive = SimpleNamespace(uv=lambda s, d, t: (U, V), idx=dict.fromkeys(sites, idx))
    J = SimpleNamespace(alive=alive, dw=dict.fromkeys(sites, M.float()), x_dir=tmp_path / "e80_X", inputs=tmp_path, out={"checks": {}})
    m.left_in_delta(J)
    left = J.out["left_in_delta"]
    assert R.rare_left_ok(left["Rare"]) and math.isclose(left["size"], len(sites) * float((M.float().double() ** 2).sum()), rel_tol=1e-9)
    assert (tmp_path / "e89_rare" / f"{sites[0]}.safetensors").is_file() and not (tmp_path / "e85_rare").exists()
    assert set(J.out["checks"]) == {"Rare_left"} and not (tmp_path / "stopped.json").exists()
