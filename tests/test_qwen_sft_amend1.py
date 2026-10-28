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
    assert R["T0"]["steps_max"] == A1.A1.t0_steps == A1.A1_TINY.t0_steps and "dW_3" not in A1.A1.drop_order
    assert A1.MATRIX_NUMEL == 1_409_286_144


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
    res = {"T0": {"kl": {"0": 0.02, "100": 0.015, "196": 0.01}, "fineweb": {"ratio": 1.15}},
           "T": {"100": {"maths": {"adv": 1.0, "p5": {"ratio": 0.8}}}},
           "F": {"paired": {"test": {"diff": -0.03, "n": 900}}}}
    sc = s(res, R, "100")
    assert sc["T0"]["holds"] and sc["T0"]["fall"] == 0.5 and sc["P5"]["holds"] and sc["P6"]["holds"]
    res["T0"]["kl"]["196"] = 0.0100001
    assert not s(res, R, "100")["T0"]["holds"]
    res["T0"]["kl"] = {"0": 0.0, "196": 0.0}  # a zero start: NaN fall, a miss
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
    res = {"T0": {"kl": {"0": 1.0, "300": 1.0}, "fineweb": {"ratio": 1.0}},
           "D": {"10": {"fineweb": {"ratio": 1.2}}, "100": {"fineweb": {"ratio": 1.6}, "maths": {"adv": 2.0}}}}
    sc = RUN.score(res, PRED)
    assert sc["P2_step1"] == {"step": 10, "ratio": 1.2, "holds": True} and sc["P2_step100"]["reading"] == "sees the update"
    assert not sc["T0"]["holds"] and "P5" not in sc  # fall 1.0 > 0.5; no three-start ratio yet


def test_step0_and_fall():
    rule = PRED["early"]["adv_step0"]
    assert RUN.step0_ok(0.0918 * 1.02, 0.0918, rule, own=True) and not RUN.step0_ok(0.0918 * 1.021, 0.0918, rule, True)
    assert RUN.step0_ok(0.0459, 0.0918, rule, own=False) and RUN.step0_ok(2.0918, 0.0918, rule, own=False)
    assert not RUN.step0_ok(math.nan, 0.0918, rule, own=False) and not RUN.step0_ok(2.1, 0.0918, rule, own=False)
    ff = RUN.fixed_fall
    kl = {"0": 2.0, "50": 1.5, "100": 0.4}
    assert ff(kl) == pytest.approx(0.2) and ff(kl, 50) == pytest.approx(0.75)  # the last reading, or the one named
    assert math.isnan(ff({})) and math.isnan(ff({"100": 1.0})) and math.isnan(ff(kl, 25))  # missing readings
    assert math.isnan(ff({"0": 0.0, "100": 0.0})) and math.isnan(ff({"0": math.nan, "100": 1.0}))
    assert ff({"0": 1.0}) == 1.0  # no training reading: no fall


def _stages(*names: str) -> list[Any]:
    return [s for s in A1.STAGES if s[0] in names]


def test_refuse_done(tmp_path: Path):
    layout = RUN.Layout(tmp_path, tmp_path / "p.json", A1.NAMES, tmp_path / "large", guard=True)
    RUN.refuse_done(layout, _stages("A0", "D1"))  # nothing there yet
    (tmp_path / "D1.json").write_text("{}")
    (tmp_path / "B.json").write_text("{}")
    RUN.refuse_done(layout, _stages("B", "D1"))  # partial files (a crash): overwritten or resumed, not refused
    RUN.write_json(tmp_path / "done.json", {"D": 1.0, "B": 2.0})
    for name in ("D1", "B"):
        with pytest.raises(SystemExit):
            RUN.refuse_done(layout, _stages(name))
    RUN.refuse_done(RUN.Layout(tmp_path, tmp_path / "p.json", A1.NAMES, tmp_path, guard=False), _stages("B"))
    assert not list(tmp_path.glob("*.tmp"))  # written whole, then renamed


def _kl(k0: float, k50: float, k100: float) -> dict[str, float]:
    return {"0": k0, "50": k50, "100": k100}


def test_b_rate_rule_and_its_edge_cases():
    rule = R["B_rate"]
    reads, cap, q = rule["reads"], rule["ratio_max"], A1.qualifies
    assert reads == [0, 50, 100] and rule["rates"] == [1e-5, 3e-5]
    good = _kl(0.006957, 0.0058, 0.004886)  # check_teacher's 1e-5 curve (0.702 at step 100), 50 made up between
    assert q(good, reads, cap) and q(_kl(1.0, 0.95, 0.9), reads, cap)  # 0.9 exactly: inclusive
    assert not q(_kl(1.0, 0.95, 0.9000001), reads, cap)  # falls, but not enough
    assert not q(_kl(1.0, 0.5, 0.6), reads, cap) and not q(_kl(1.0, 1.1, 0.5), reads, cap)  # not monotone
    assert not q(_kl(1.0, 0.8, 0.8), reads, cap) and not q(_kl(1.0, 1.0, 0.5), reads, cap)  # ties
    assert not q(_kl(0.0, 0.0, 0.0), reads, cap) and not q(_kl(0.0, -1e-12, -2e-12), reads, cap)  # zero start
    assert not q(_kl(math.nan, 0.5, 0.4), reads, cap) and not q(_kl(1.0, math.nan, 0.4), reads, cap)
    assert not q(_kl(math.inf, 0.5, 0.4), reads, cap) and not q({"0": 1.0, "100": 0.5}, reads, cap)  # missing 50
    up = _kl(0.006957, 0.015, 0.023)  # check_teacher's 1e-4 divergence
    assert A1.pick_rate({1e-5: good, 3e-5: good}, reads, cap) == 3e-5  # both: the larger
    assert A1.pick_rate({1e-5: good, 3e-5: up}, reads, cap) == 1e-5
    assert A1.pick_rate({1e-5: up, 3e-5: up}, reads, cap) is None and A1.pick_rate({}, reads, cap) is None


def test_t0_steps_from_b():
    frac, cap = R["T0"]["fall_max"], R["T0"]["steps_max"]
    assert A1.t0_steps(_kl(1.0, 0.85, 0.702), frac, cap) == 196  # 100 ln 0.5 / ln 0.702 = 195.9
    assert A1.t0_steps(_kl(1.0, 0.95, 0.9), frac, cap) == cap  # 658 capped
    assert A1.t0_steps(_kl(1.0, 0.7, 0.5), frac, cap) == 100 and A1.t0_steps(_kl(1.0, 0.5, 0.0), frac, cap) == 100
    assert A1.t0_steps(_kl(1.0, 0.8, 0.5000001), frac, cap) == 101


def test_maths_pass_test_rule():
    res = {"D": {"100": {"maths": {"ratio": 1.5, "ratios": {"0": 1.5, "1": 1.4, "2": 1.6}}}},
           "T": {"100": {"maths": {"ratio": 1.5000001, "ratios": {"0": 1.0}}}}}
    sc = RUN.score_amend1(res, R, "100")
    assert sc["M_do_nothing"]["holds"] and sc["M_do_nothing"]["starts"] == 3 and sc["M_do_nothing"]["reading"] == "pass"
    assert not sc["M_teacher"]["holds"] and sc["M_teacher"]["reading"] == "sees the update"
    res["T"]["100"]["maths"]["ratio"] = math.nan
    assert not RUN.score_amend1(res, R, "100")["M_teacher"]["holds"]


def test_free_space_on_a_quota_volume(tmp_path: Path):
    vol = tmp_path / "vol"
    (vol / "sub").mkdir(parents=True)
    (vol / "a.bin").write_bytes(b"x" * 1000)
    (vol / "sub" / "b.bin").write_bytes(b"x" * 500)
    (vol / "link").symlink_to(vol / "a.bin")
    (vol / "gone").symlink_to(vol / "missing")  # a dangling link is not followed
    assert du(vol) == 1500
    set_volume(vol, 1e-5)  # a 10,000-byte quota
    try:
        assert free_bytes(vol / "sub") == 10_000 - 1500
        assert free_bytes(tmp_path) > 10_000  # outside the volume: df
        set_volume(vol, 1e9)  # a quota larger than the disk: df is smaller and wins
        assert free_bytes(vol) <= __import__("shutil").disk_usage(vol).free
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


def _tiny_qwen() -> tuple[Any, qwen_vpd.Components, dict[str, torch.Tensor]]:
    """A random 28-layer Qwen3 of width 32 with gradient checkpointing, as the job sets it up, and components of 8
    subcomponents per site, all alive."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=28, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=16, max_position_embeddings=64, tie_word_embeddings=True)
    model = Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    dims = {"q": (32, 32), "k": (32, 16), "v": (32, 16), "o": (32, 32), "gate": (32, 64), "up": (32, 64), "down": (64, 32)}
    V = {k: 0.1 * torch.randn(qwen_vpd.N_LAYERS, i, 8) for k, (i, _) in dims.items()}
    U = {k: 0.1 * torch.randn(qwen_vpd.N_LAYERS, 8, o) for k, (_, o) in dims.items()}
    return model, qwen_vpd.Components(V, U), {s: torch.arange(8) for s in qwen_vpd.SITES}


def test_teacher_training_runs_under_checkpointing_and_is_exactly_zero_at_w0():
    """The rehearsal's crash (job 40530): the backward ran outside the student's hooks, so the checkpointed recompute
    saved different tensors. Here train() runs end to end on CPU; with the teacher at W0 every offset stays 0."""
    from commute.qwen_teacher import train

    model, comps, alive = _tiny_qwen()
    rows = torch.randint(64, (4, 16), generator=torch.Generator().manual_seed(1))
    off = Offsets(comps, alive, device="cpu")
    curve = train(model, off, None, rows, rows, 1e-3, 3, 1, 1, 0, log=lambda m: None)
    assert all(c["kl"] == 0.0 for c in curve) and off.max_abs() == 0.0
    teacher = random_change(off, 0.5, seed=0, dtype=torch.float32)
    curve = train(model, off, teacher, rows, rows, 1e-2, 3, 1, 1, 0, log=lambda m: None)
    assert curve[0]["kl"] > 0 and curve[0]["kl_fp32_teacher_w0"] > 0 and off.max_abs() > 0
    assert all(math.isfinite(c["kl"]) for c in curve)


def test_fixed_row_reading_is_kl_to_w0_at_zero_offsets_and_zero_at_the_target():
    """`kl_to_teacher` at zero offsets reads KL(teacher ‖ W0); with the offsets set to the positive control's own α·R
    the student is the teacher and it reads ~0 (the reachable target)."""
    from commute.qwen_sft import linears, mean_kl, plus
    from commute.qwen_teacher import kl_to_teacher

    model, comps, alive = _tiny_qwen()
    rows = list(torch.randint(64, (2, 16), generator=torch.Generator().manual_seed(2)).split(1))
    W0 = {s: lin.weight.data for s, lin in linears(model).items()}
    off = Offsets(comps, alive, device="cpu")
    alpha = 0.5
    teacher = random_change(off, alpha, seed=0, dtype=torch.float32)
    k0 = kl_to_teacher(model, rows, W0, teacher, off)
    assert k0 > 0 and k0 == mean_kl(model, rows, plus(W0, teacher), None)
    with torch.no_grad():
        for i, s in enumerate(qwen_vpd.SITES):  # random_change's draws (seed 0: generator seed i)
            gen = torch.Generator().manual_seed(i)
            off.dV[s].copy_(alpha * torch.randn(off.V[s].shape, generator=gen))
            off.dU[s].copy_(alpha * torch.randn(off.U[s].shape, generator=gen))
    assert kl_to_teacher(model, rows, W0, teacher, off) < 1e-6 * k0
