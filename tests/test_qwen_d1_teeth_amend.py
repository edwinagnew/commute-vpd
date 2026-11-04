"""E85 Amendments 1 and 2: the filed rules on their edge cases, the permuted state through `qwen_maps.applied`, and the rounding read's pieces on small tensors."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from commute import qwen_maps, qwen_sft, qwen_vpd
from commute.jobs import load_experiment
from commute.qwen_vpd import SITES, Components, parse_site

R = load_experiment("qwen_d1_teeth", "rules_amend")
RESULTS = Path(__file__).resolve().parent.parent / "experiments" / "qwen_d1_teeth" / "results"
NAN = math.nan


def test_stage_1_rule_on_the_bars_and_the_clean_clause() -> None:
    assert R.outcome_rand(1.61, 0.0) == R.CAN_FAIL and R.outcome_rand(2.0, NAN) == R.CAN_FAIL
    assert R.outcome_rand(1.40, R.BAR_EXCESS) == R.DOES_NOT and R.outcome_rand(1.0, 5.0) == R.DOES_NOT
    assert R.outcome_rand(1.40, R.BAR_EXCESS - 0.01) == R.TOO_SMALL and R.outcome_rand(1.0, NAN) == R.TOO_SMALL
    assert R.outcome_rand(1.5, 5.0) == R.NO_CALL and R.outcome_rand(NAN, 5.0) == R.UNREAD


def test_stage_2_gate_and_rule() -> None:
    assert R.goes_on(1.61) and not R.goes_on(1.60) and not R.goes_on(NAN) and not R.goes_on(1.2)
    assert R.outcome_perm(1.61) == R.WRONG_FAILS and R.outcome_perm(1.40) == R.WRONG_PASSES and R.outcome_perm(1.5) == R.NO_CALL and R.outcome_perm(NAN) == R.UNREAD


def test_own_bars_are_reported_on_their_edges() -> None:
    assert R.verdict_own(1.267) == "pass" and R.verdict_own(1.776) == "fail" and R.verdict_own(1.27) == "between" and R.verdict_own(NAN) == R.UNREAD


def test_first_steps() -> None:
    assert R.first_steps_miss("W0", [0.0]) is None and R.first_steps_miss("Rand30", [], 1.0) is None
    assert R.first_steps_miss("Rand30", [0.09], 1.0) is None and R.first_steps_miss("Rand30", [0.08], 1.0) and R.first_steps_miss("Rand30", [NAN], 1.0)
    assert R.first_steps_miss("Rand30", [1.5], 1.0) is None and R.first_steps_miss("Rand30", [1.7], 1.0)
    assert R.first_steps_miss("Rand30", [0.09, 0.10], 1.0) is None and R.first_steps_miss("Rand30", [0.09, 0.10, 0.09], 1.0) and R.first_steps_miss("Rand30", [0.09, 0.10, 0.11], 1.0) is None
    assert R.first_steps_miss("Perm", [0.4]) and R.first_steps_miss("Perm", [8.0]) is None
    assert R.first_steps_miss("PermRare", [0.09]) is None and R.first_steps_miss("PermRare", [0.08]) and R.first_steps_miss("PermRare", [0.09, 0.09, 0.08]) is None


def test_steps_and_before_search() -> None:
    t = R.steps([1.0, 2.0, 3.0], [1.0, 1.0])
    assert [x["ratio"] for x in t] == [1.0, 2.0] and t[1]["excess"] == 1.0
    assert R.before_search([2.74], 1.7) is True and R.before_search([0.09], 1.7) is False
    assert R.before_search([], 1.7) is None and R.before_search([NAN], 1.7) is None and R.before_search([1.0], 0.0) is None


def test_score_empty_partial_and_whole() -> None:
    s = R.score({})
    assert s["states"]["Rand30"]["outcome"] == R.UNREAD and s["stage2_goes_on"] is False and s["leans"] == {"Rand30_can_fail": None, "Perm_fails": None, "Perm_before_search": None}
    w0 = {"kl": 1.7, "trace": [0.1] * 20, "from": "x"}
    out: dict[str, Any] = {"w0": w0, "clean": {"Rand30": {"kl_W0": 2.0}}, "starts": {"Rand30_W_s0": {"kl": 3.4, "trace": [0.2] * 20}}, "checks": {}}
    s = R.score(out)
    r = s["states"]["Rand30"]
    assert r["final_ratio"] == 2.0 and math.isclose(r["final_excess"], 1.7) and r["outcome"] == R.CAN_FAIL and r["step10_ratio"] == 2.0 and r["before_search"] is False and s["stage2_goes_on"] is True
    out["clean"]["PermRare"] = {"kl_W0": 0.0}
    assert R.score(out)["states"]["PermRare"]["outcome"] == R.UNREAD
    out["starts"]["PermRare_W_s0"] = {"kl": 1.7 * 1.3, "trace": [0.1]}
    s = R.score(out)
    assert s["states"]["PermRare"]["outcome"] == R.WRONG_PASSES and s["states"]["PermRare"]["verdict_own_bars"] == "between" and s["leans"]["Perm_fails"] is False
    json.dumps(s)
    out["w0"] = {"kl": 0.0, "trace": [0.0]}
    assert R.score(out)["states"]["Rand30"]["outcome"] == R.UNREAD


def test_amendment_2_reading() -> None:
    assert R.reading(2.0, 0.0, 1.0) == R.CLOSES and R.reading(5.0, 0.0, 1.0) == R.FOLLOW_UP and R.reading(3.0, 1.0, 0.5) == R.NO_CALL
    assert R.reading(0.0, 0.0, 0.0) == R.CLOSES and R.reading(1e-9, 0.0, 0.0) == R.FOLLOW_UP and R.reading(NAN, 0.0, 1.0) == R.UNREAD
    assert R.floors_ok(0.0, 1.0) and R.floors_ok(1.0, 1.0) and not R.floors_ok(2.0, 1.0) and not R.floors_ok(0.0, 0.0) and not R.floors_ok(NAN, 1.0) and not R.floors_ok(0.0, NAN)


def test_filed_numbers_are_the_files() -> None:
    e85 = json.loads((RESULTS / "results.json").read_text())["starts"]["W0_W_s0"]["kl"]
    assert R.W0_KL == e85 and math.isclose(R.BAR_EXCESS, 0.61 * e85)
    p = json.loads((RESULTS / "predictions_amend1.json").read_text())
    assert p["rules"]["bars"] == {"pass_at": R.r2.PASS_AT, "fail_at": R.r2.FAIL_AT}
    for name in (R.CAN_FAIL, R.DOES_NOT, R.TOO_SMALL):
        assert name in p["rules"]["stage_1"]
    for name in (R.WRONG_FAILS, R.WRONG_PASSES):
        assert name in p["rules"]["stage_2"]
    assert R.RAND_SCALE == 30 and R.PERM_SEED == 85_200
    p2 = json.loads((RESULTS / "predictions_amend2.json").read_text())
    assert R.CLOSES in p2["rules"] and R.FOLLOW_UP in p2["rules"]


def small_job(tmp_path: Path, monkeypatch: Any) -> Any:
    """The permuted state on real site names with small stacks: 4 subcomponents' worth of V, U and weights at every site, the last dead."""
    a1 = load_experiment("qwen_d1_teeth", "amend1")
    monkeypatch.setattr(a1.e85, "DEV", "cpu")
    gen = torch.Generator().manual_seed(0)
    V = {k: torch.randn(28, 3, 4, generator=gen) for k in qwen_vpd.KINDS}
    U = {k: torch.randn(28, 4, 2, generator=gen) for k in qwen_vpd.KINDS}
    comps = Components(V, U)
    alive = qwen_maps.Alive(comps, dict.fromkeys(SITES, torch.tensor([0, 1, 2])))
    lins = {s: torch.nn.Linear(3, 2, bias=False) for s in SITES}
    monkeypatch.setattr(qwen_maps, "linears", lambda m: lins)
    monkeypatch.setattr(qwen_sft, "linears", lambda m: lins)
    active = torch.tensor([0.1, 0.0, 0.2, 0.0])
    save_file({k: active.clone() for s in SITES for k in (f"{s}.active", f"{s}.mean_g")}, str(tmp_path / "W0_maths.safetensors"))
    return a1, SimpleNamespace(model=None, comps=comps, alive=alive, inputs=tmp_path, lins=lins)


def test_permuted_state_moves_u_only_and_restores(tmp_path: Path, monkeypatch: Any) -> None:
    a1, J = small_job(tmp_path, monkeypatch)
    V0, U0 = {k: v.clone() for k, v in J.comps.V.items()}, {k: u.clone() for k, u in J.comps.U.items()}
    w0 = {s: lin.weight.data for s, lin in J.lins.items()}
    moved = 0
    with a1.perm_state("all")(J):
        for s in SITES:
            layer, kind = parse_site(s)
            got, was = J.comps.U[kind][layer], U0[kind][layer]
            assert torch.equal(J.comps.V[kind][layer], V0[kind][layer]) and torch.equal(J.lins[s].weight.data, w0[s])
            assert torch.equal(got[3], was[3])  # the dead column
            d = torch.cdist(got[:3], was[:3])  # each alive column is one of the originals, each used once
            assert (d.min(dim=1).values < 1e-6).all() and d.argmin(dim=1).unique().numel() == 3
            moved += int(not torch.equal(got, was))
    assert moved > 100
    assert all(torch.equal(J.comps.U[k], U0[k]) for k in U0) and all(torch.equal(J.lins[s].weight.data, w0[s]) for s in SITES)


def test_permuted_rare_half_moves_only_the_half(tmp_path: Path, monkeypatch: Any) -> None:
    a1, J = small_job(tmp_path, monkeypatch)
    pos = a1.perm_positions(J, "rare")[SITES[0]]
    assert pos.tolist() == [1]  # three alive columns, the half is one: permuting it is the identity
    assert a1.perm_positions(J, "all")[SITES[0]].tolist() == [0, 1, 2]
    dU = a1.perm_change(J, SITES[0], pos)[2]
    assert dU is not None and float(dU.abs().max()) == 0.0
    both = torch.tensor([0, 2])  # the seed's draw decides order; a two-column swap or the identity, never another column
    c = a1.perm_change(J, SITES[0], both)
    assert c[0] is None and c[1] is both and c[3] is not None and float(c[3].abs().max()) == 0.0


def test_perm_change_repeats_and_wrong_share_on_a_swap(tmp_path: Path, monkeypatch: Any) -> None:
    a1, J = small_job(tmp_path, monkeypatch)
    pos = torch.tensor([0, 1, 2])
    assert torch.equal(a1.perm_change(J, SITES[5], pos)[2], a1.perm_change(J, SITES[5], pos)[2])
    assert a1.wrong_share(J, "all") > 0.5


def test_fp32_readout_hears_what_bf16_rounds_away() -> None:
    a2 = load_experiment("qwen_d1_teeth", "amend2")
    s = SITES[0]
    r = torch.tensor([[1.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0]], dtype=torch.bfloat16)
    gamma: Any = SimpleNamespace(dtype=torch.bfloat16, readouts={s: r})
    x, V = torch.tensor([[[1.0]]]), torch.tensor([[0.5]])
    Vr = V * (1 + 1e-3)  # under bf16's spacing at 0.5, so bf16's h does not move, and fp32's does
    assert float(qwen_vpd.readout(gamma, s, x, Vr) - qwen_vpd.readout(gamma, s, x, V)) == 0.0
    assert math.isclose(float(a2.fp32_readout(gamma, s, x, Vr) - a2.fp32_readout(gamma, s, x, V)), 5e-4, rel_tol=1e-3)
    assert a2.fp32_readout(gamma, s, x, V).dtype == torch.float32


def test_before_clip_clipped_is_site_importance() -> None:
    a2 = load_experiment("qwen_d1_teeth", "amend2")
    s = SITES[3]
    layer, kind = parse_site(s)
    gen = torch.Generator().manual_seed(0)
    heads_w, heads_b = {kind: torch.zeros(layer + 1, 4, 5).to(torch.bfloat16)}, {kind: torch.zeros(layer + 1, 5).to(torch.bfloat16)}
    heads_w[kind][layer] = torch.randn(4, 5, generator=gen).to(torch.bfloat16)
    heads_b[kind][layer] = torch.randn(5, generator=gen).to(torch.bfloat16)
    gamma: Any = SimpleNamespace(heads_w=heads_w, heads_b=heads_b)
    x, r = torch.randn(1, 2, 4, generator=gen).to(torch.bfloat16), torch.randn(1, 2, 5, generator=gen).to(torch.bfloat16)
    assert torch.equal(a2.before_clip(gamma, x, s, r).clamp(0, 1), qwen_vpd.site_importance(gamma, x, s, r))


def test_tally_pools_by_site_and_row() -> None:
    a2 = load_experiment("qwen_d1_teeth", "amend2")
    s, t = SITES[0], SITES[1]
    T = a2.Tally({"alive": {s: torch.tensor([0, 1]), t: torch.tensor([0, 1])}, "half": {s: torch.tensor([1]), t: torch.tensor([], dtype=torch.long)}})
    a, b = torch.tensor([[[0.5, 0.02, 3.0]]]), torch.tensor([[[0.5, 0.0, 0.0]]])  # column 2 not alive; column 1 moves 0.02
    T.add("x", s, 0, a, b)
    T.add("x", t, 1, a, a)
    out = T.summary()["x"]
    assert math.isclose(out["alive"]["g"], 0.02 / 4, rel_tol=1e-6) and out["alive"]["share_over"] == 0.25
    assert math.isclose(out["half"]["g"], 0.02, rel_tol=1e-6) and out["half"]["share_over"] == 1.0 and math.isclose(out["alive"]["by_row"][0], 0.01, rel_tol=1e-6)
    assert out["alive"]["by_row"][1] == 0.0 and math.isnan(out["half"]["by_row"][1])
    assert list(out["by_site"]) == [s, t] and math.isclose(out["by_site"][s][0], 0.01, rel_tol=1e-4)
    c = torch.tensor([[[1.5, 0.0, 0.0]]])
    T2 = a2.Tally({"alive": {s: torch.tensor([0])}})
    T2.add("y", s, 0, c, torch.tensor([[[1.0, 0.0, 0.0]]]))
    got = T2.summary()["y"]["alive"]
    assert got["g"] == 0.0 and got["before_clip"] == 0.5  # the clip hides it; before the clip does not


def test_guard_logs_ratio_and_stops_on_a_missed_first_step(tmp_path: Path, monkeypatch: Any) -> None:
    a1 = load_experiment("qwen_d1_teeth", "amend1")
    e80 = a1.e80
    for name, value in (("OUT", tmp_path / "r.json"), ("STOPPED", tmp_path / "stopped.json"), ("LOG", tmp_path / "r.log")):
        monkeypatch.setattr(e80, name, value)
    monkeypatch.setitem(a1.RUNNING, "w0", [0.1, 0.2])
    monkeypatch.setitem(a1.RUNNING, "clean", 1.0)
    monkeypatch.setitem(a1.amend1.READS, "Rand30_W_s0", [])
    a1.guard("[Rand30_W_s0] seed 0 step 0: KL 0.1000")
    a1.guard("[Rand30_W_s0] seed 0 step 1: KL 0.3000")
    log = (tmp_path / "r.log").read_text()
    assert "step 0: ratio 1.0000, excess 0.0000 nats" in log and "step 1: ratio 1.5000, excess 0.1000 nats" in log
    assert not (tmp_path / "stopped.json").exists()
    monkeypatch.setitem(a1.amend1.READS, "Rand30_W_s0", [])
    with pytest.raises(SystemExit):
        a1.guard("[Rand30_W_s0] seed 0 step 0: KL 0.0500")
    assert json.loads((tmp_path / "stopped.json").read_text())["rule"] == "first_steps"
