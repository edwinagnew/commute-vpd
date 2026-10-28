"""CPU checks of E40 Amendment 1's pure parts (experiments/qwen_sft/amend1.py and the run.py pieces it adds): the
schedule against transformers' own, the read-step rule, the added rules on their edge cases, the stale-results guard,
the volume's free space, and the positive control's teacher on toy tensors. No model, no GPU."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import get_cosine_schedule_with_warmup

from commute import qwen_vpd
from commute.qwen_sft import du, free_bytes, lr_at, set_volume
from commute.qwen_teacher import Offsets, random_change

EXP = Path(__file__).resolve().parents[1] / "experiments" / "qwen_sft"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # amend1 imports `run` by that name; dataclasses looks modules up by name
    spec.loader.exec_module(mod)
    return mod


RUN = _load("run", EXP / "run.py")
A1 = _load("qwen_sft_amend1", EXP / "amend1.py")
PRED = json.loads((EXP / "results" / "predictions_amend1.json").read_text())
R = PRED["rules"]


def test_schedule_is_the_library_one_at_every_step():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-5)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=100)
    for t in range(1, 101):
        assert math.isclose(opt.param_groups[0]["lr"], lr_at(t, 1e-5, 10, 100, "hf_cosine"), rel_tol=1e-12, abs_tol=0)
        opt.step()
        sched.step()
    assert lr_at(1, 1e-5, 10, 100, "hf_cosine") == 0.0 and lr_at(11, 1e-5, 10, 100, "hf_cosine") == 1e-5
    assert lr_at(1, 1e-5, 0, 100, "constant") == 1e-5


def test_configs_and_the_filed_read_rule_agree():
    assert A1.A1.first_live() == 2 and A1.A1_TINY.first_live() == 2 and RUN.CFG.first_live() == 1
    assert (tuple(R["read_steps"]["fixed"]), R["read_steps"]["kl_min"]) == (A1.A1.read_fixed, A1.A1.read_kl_min)
    assert A1.A1_TINY.steps == A1.A1.steps and A1.A1_TINY.batch == A1.A1.batch  # F1 at full size in the rehearsal
    assert set(dict(A1.A1.mem_gb)) >= {"B", "G", "A0", "F", "T0", "D", "T"}


def test_read_rule():
    rr, fixed = RUN.read_rule, (10, 100)
    kl = {1: 0.0, 2: 0.001, 3: 0.004, 5: 0.0099999999999, 10: 0.05, 20: 0.1, 50: 0.3, 100: 0.5}
    assert rr(kl, fixed, 0.01, 3) == [5, 10, 100]  # a tie in exact arithmetic reaches the line
    assert rr(kl | {5: 0.009}, fixed, 0.01, 3) == [10, 100]  # first at 10: nothing added
    assert rr({**kl, 3: 0.02}, fixed, 0.01, 2) == [10, 100]  # the budget drops the optional step, never 10 or 100
    assert rr({t: math.nan for t in kl}, fixed, 0.01, 3) == [10, 100]
    assert rr({t: 0.0 for t in kl}, fixed, 0.01, 3) == [10, 100]
    assert rr({**kl, 3: 0.02}, fixed, 0.01, 0) == [10, 100]


def _l(k5: float, k6: float) -> dict[str, Any]:
    return {"L": {"rates": {"1e-05": {"kl_to_start": k5}, "1e-06": {"kl_to_start": k6}}}}


def test_l_rules():
    ref = R["L_control"]["ref"]
    s = RUN.score_amend1
    assert s(_l(ref * 1.03, ref * 1.03 * 0.003), R, "100")["L_control"]["holds"]
    assert s(_l(ref * 0.97, ref * 0.97 * 0.05), R, "100")["L_scaling"]["holds"]
    assert not s(_l(ref * 1.031, 0.0), R, "100")["L_control"]["holds"]
    assert not s(_l(math.nan, 0.001), R, "100")["L_control"]["holds"]
    zero = s(_l(0.0, 0.001), R, "100")
    assert not zero["L_control"]["holds"] and not zero["L_scaling"]["holds"]
    assert "L_control" not in s({}, R, "100")


def test_positive_control_p5_p6_rules():
    s = RUN.score_amend1
    curve = [1.0] + [0.5] * 89 + [0.2] * 10
    res = {"T0": {"curve": curve, "fineweb": {"ratio": 1.15}},
           "T": {"100": {"maths": {"adv": 1.0, "p5": {"ratio": 0.8}}}},
           "F": {"paired": {"test": {"diff": -0.03, "n": 900}}}}
    sc = s(res, R, "100")
    assert sc["T0"]["holds"] and sc["P5"]["holds"] and sc["P6"]["holds"]
    res["T0"]["curve"] = [0.0, 0.0]
    res["T"]["100"]["maths"]["p5"]["ratio"] = 0.8000001
    res["F"]["paired"]["test"]["diff"] = None
    sc = s(res, R, "100")
    assert not sc["T0"]["holds"] and not sc["P5"]["holds"] and not sc["P6"]["holds"]
    res["F"]["paired"]["test"]["diff"] = -0.2
    assert s(res, R, "100")["P6"]["holds"]
    res["F"]["paired"]["test"]["diff"] = -0.21
    assert not s(res, R, "100")["P6"]["holds"]


def test_score_with_amendment_rules_falls_back_cleanly():
    """E40's T0 and P5 are not scored under Amendment 1's file; P2 reads D1's first read step."""
    res = {"T0": {"curve": [1.0] * 20, "fineweb": {"ratio": 1.0}},
           "D": {"10": {"fineweb": {"ratio": 1.2}}, "100": {"fineweb": {"ratio": 1.6}, "maths": {"adv": 2.0}}}}
    sc = RUN.score(res, PRED)
    assert sc["P2_step1"] == {"step": 10, "ratio": 1.2, "holds": True} and sc["P2_step100"]["reading"] == "sees the update"
    assert not sc["T0"]["holds"] and "P5" not in sc  # fall 1.0 > 0.2; no three-start ratio yet


def test_step0_and_fall():
    rule = PRED["early"]["adv_step0"]
    assert RUN.step0_ok(0.0918 * 1.02, 0.0918, rule, own=True) and not RUN.step0_ok(0.0918 * 1.021, 0.0918, rule, True)
    assert RUN.step0_ok(0.0459, 0.0918, rule, own=False) and RUN.step0_ok(2.0918, 0.0918, rule, own=False)
    assert not RUN.step0_ok(math.nan, 0.0918, rule, own=False) and not RUN.step0_ok(2.1, 0.0918, rule, own=False)
    assert RUN.fall([2.0] + [1.0] * 9 + [0.4] * 10) == pytest.approx(0.2) and math.isnan(RUN.fall([]))


def _stages(*names: str) -> list[Any]:
    return [s for s in A1.STAGES if s[0] in names]


def test_refuse_done(tmp_path: Path):
    layout = RUN.Layout(tmp_path, tmp_path / "p.json", A1.NAMES, tmp_path / "large", guard=True)
    RUN.refuse_done(layout, _stages("A0", "D1"))  # nothing there yet
    (tmp_path / "D1.json").write_text("{}")
    RUN.refuse_done(layout, _stages("D1"))  # a partial D resumes
    (tmp_path / "done.json").write_text(json.dumps({"D": 1.0}))
    with pytest.raises(SystemExit):
        RUN.refuse_done(layout, _stages("D1"))
    (tmp_path / "A0.json").write_text("{}")
    with pytest.raises(SystemExit):
        RUN.refuse_done(layout, _stages("A0"))
    RUN.refuse_done(RUN.Layout(tmp_path, tmp_path / "p.json", A1.NAMES, tmp_path, guard=False), _stages("A0"))


def test_free_space_on_a_quota_volume(tmp_path: Path):
    vol = tmp_path / "vol"
    (vol / "sub").mkdir(parents=True)
    (vol / "a.bin").write_bytes(b"x" * 1000)
    (vol / "sub" / "b.bin").write_bytes(b"x" * 500)
    (vol / "link").symlink_to(vol / "a.bin")
    assert du(vol) == 1500
    set_volume(vol, 1e-5)  # a 10,000-byte quota
    try:
        assert free_bytes(vol / "sub") == 10_000 - 1500
        assert free_bytes(tmp_path) > 10_000  # outside the volume: df
    finally:
        set_volume(None, None)


def test_positive_control_teacher_is_the_students_own_change():
    torch.manual_seed(0)
    V = {k: torch.randn(qwen_vpd.N_LAYERS, 3, 5) for k in qwen_vpd.KINDS}
    U = {k: torch.randn(qwen_vpd.N_LAYERS, 5, 4) for k in qwen_vpd.KINDS}
    alive = {s: torch.tensor([0, 3]) if i % 3 else torch.tensor([], dtype=torch.long) for i, s in enumerate(qwen_vpd.SITES)}
    off = Offsets(qwen_vpd.Components(V, U), alive, device="cpu")
    assert len(off.params()) == 2 * sum(1 for v in alive.values() if len(v))  # empty sites stay out of the optimiser
    teacher = random_change(off, 0.01, seed=0, dtype=torch.float32)
    for i, s in enumerate(qwen_vpd.SITES):
        gen = torch.Generator().manual_seed(i)
        with torch.no_grad():
            off.dV[s].copy_(0.01 * torch.randn(off.V[s].shape, generator=gen))
            off.dU[s].copy_(0.01 * torch.randn(off.U[s].shape, generator=gen))
        assert torch.allclose(off.change(s), teacher[s], atol=1e-7), s
    assert torch.equal(random_change(off, 0.0, 0, torch.float32)[qwen_vpd.SITES[1]], torch.zeros(4, 3))
