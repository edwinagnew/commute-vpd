"""CPU checks of E37 Amendment 1's pure parts (experiments/teacher_spread/amend1.py): the scoring rules on a synthetic
results file, and arm 3's start through `component_training.Offsets(init=...)` on toy tensors; no model, no GPU."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

from commute.component_training import Offsets

EXP = Path(__file__).resolve().parents[1] / "experiments" / "teacher_spread"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # amend1 imports `run` by that name
    spec.loader.exec_module(mod)
    return mod


_load("run", EXP / "run.py")
A1 = _load("teacher_spread_amend1", EXP / "amend1.py")
C1 = {"pgd_floor": 0.075, "stochastic_floor": 0.001}  # pass at or under 1.5 / 1.075 = 1.395, fail at or over 1.6125


def _st(logp: float, check: float | None = None, stoch: float = 1.09, damage: float = 0.23, kl: float = 0.01) -> dict[str, Any]:
    r: dict[str, Any] = {"logp": logp, "damage": damage, "gap": 0.01, "train_kl": kl}
    if check is not None:
        r["pile"] = {"pgd_kl": check, "stochastic_kl": stoch}
    return r


def _out(**states: dict[str, Any]) -> dict[str, Any]:
    base = {"W0": _st(-3.146, damage=0.0), "do-nothing": _st(-3.016)}
    return {"C1": C1, "states": base | states}


def test_w0_and_do_nothing_read_0_and_1() -> None:
    d = A1.score(_out())["derived"]
    assert d["W0"]["s"] == 0 and abs(d["do-nothing"]["s"] - 1) < 1e-12 and "P1" not in A1.score(_out())


def test_p1_p3_and_branch_when_nothing_passes() -> None:
    out = _out(arm1_1000=_st(-3.016, 2.20), arm1_2000=_st(-3.016, 2.0, kl=0.008), arm1_4000=_st(-3.016, 1.9, kl=0.0078),
               arm3_0=_st(-3.04, 1.05), arm3_2000=_st(-3.017, 2.2))
    sc = A1.score(out)
    assert sc["P1"] is True  # 1.9 >= 1.6125
    assert sc["P3"] is True  # s 0.99 and 2.2 / 2.0 within 25%
    assert sc["arm1_still_falling"] is False  # train KL fell 2.5%, under 10%
    assert abs(sc["R1"]["check_over_e37"] - 1) < 1e-12 and sc["derived"]["arm3_0"]["verdict"]["pgd_kl"] == "pass"
    assert sc["branch"].startswith("arm 1 flat, no solution")


def test_still_falling_and_solutions() -> None:
    out = _out(arm1_2000=_st(-3.016, 2.0, kl=0.008), arm1_4000=_st(-3.016, 1.5, kl=0.006))
    sc = A1.score(out)
    assert sc["P1"] is False and sc["arm1_still_falling"] is True and "extend arm 1" in sc["branch"]
    out["states"]["arm1_4000"] = _st(-3.016, 1.3)
    assert A1.score(out)["branch"].startswith("arm 1 a solution")
    out["states"] |= {"arm1_4000": _st(-3.016, 1.9, kl=0.008), "arm2_2000": _st(-3.02, 1.2), "M0_1000": _st(-3.146, 1.0)}
    out["M0"] = {"state": "M0_1000"}
    sc = A1.score(out)
    assert sc["P2"] is True and "arm2" in sc["branch"] and abs(sc["arm2_over_M0"]["arm2_2000"] - 1.2) < 1e-12
    out["states"]["arm2_2000"] = _st(-3.03, 1.2)  # s 0.89: behaviour lost, not a solution
    assert A1.score(out)["P2"] is False


def test_kl_before_reads_the_50_steps_ending_at_n() -> None:
    curve = [{"step": s, "kl": float(s)} for s in range(-49, 101)]  # warm-up steps carry step <= 0
    assert A1.kl_before(curve, 100, "kl") == sum(range(51, 101)) / 50


def test_arm3_start_offsets_rebuild_its_delta_w() -> None:
    g = torch.Generator().manual_seed(0)
    uv = {"a": (torch.randn(5, 3, generator=g), torch.randn(4, 3, generator=g))}
    uv2 = {"a": (uv["a"][0] + 0.1 * torch.randn(5, 3, generator=g), uv["a"][1] + 0.1 * torch.randn(4, 3, generator=g))}
    off = Offsets(uv, init=A1.start_offsets(uv, uv2))
    assert torch.allclose(off.delta_w("a"), A1.start_dw(uv, uv2)["a"], atol=1e-6)
    assert torch.allclose(off.components("a")[0], uv2["a"][0]) and Offsets(uv).delta_w("a").abs().max() == 0


def test_masked_teacher_on_the_toy_trains_alive_columns_and_restores_the_model() -> None:
    """Arm 2's trainer on the tiny ComponentModel (released loss configs, CPU): G2's student reads the target net at W0
    (KL 0), only alive columns move, the returned delta-W is U'V'^T - U V^T, and the model comes back as it was."""
    from test_faithfulness_invariants import KEYS, VOCAB, make_model

    from commute import masked_teacher as MT
    from commute import teacher, vpd
    from commute.component_training import alive_uv

    model = make_model()
    W0, c0 = vpd.target_weights(model), vpd.component_weights(model)
    alive = {k: torch.arange(8) % 2 == 0 for k in KEYS}
    t_net = teacher.teacher_net(model, {k: 0.3 * torch.randn_like(w) for k, w in W0.items()})
    rows = torch.randint(0, VOCAB, (6, vpd.SEQ_LEN), generator=torch.Generator().manual_seed(0))
    wd = vpd.weight_deltas(model)
    assert float(MT.student_kl(model, teacher.teacher_net(model, {}), rows[:2], wd)) < 1e-6
    uv0 = alive_uv(model, alive)
    cfg = MT.MaskedConfig(lr=1e-2, batch=2, steps=3, warmup=2, save_steps=(3,), coeff=0.05)  # M0's retry
    curve, st = MT.train(model, t_net, rows, alive, uv0, cfg, log=lambda _: None)
    assert [c["step"] for c in curve] == [-1, 0, 1, 2, 3] and curve[0]["kl"] > 0
    c = curve[0]
    assert abs(c["total"] - c["kl"] - 0.05 * (c["StochasticReconSubsetLoss"] + c["PersistentPGDReconLoss"])) < 1e-6
    for k in KEYS:
        dW, U2, V2 = st[3][k]
        assert torch.allclose(dW, U2 @ V2.T - uv0[k][0].cpu() @ uv0[k][1].cpu().T, atol=1e-6)
        assert (U2 - uv0[k][0]).abs().max() > 0
        assert torch.equal(vpd.target_weights(model)[k], W0[k]) and torch.equal(model.components[k].U, c0[k][0])
        assert not model.components[k].U.requires_grad


def test_masked_losses_alone_move_u_and_v_on_the_toy() -> None:
    """M0's set-up on the toy (teacher at W0): the teacher KL's gradient on U, V is rounding-sized, the masked losses'
    is not, and masked training moves the alive columns."""
    from test_faithfulness_invariants import KEYS, VOCAB, make_model

    from commute import gamma, teacher, vpd
    from commute import masked_teacher as MT
    from commute.component_training import alive_uv

    model = make_model()
    w0_net, wd = teacher.teacher_net(model, {}), vpd.weight_deltas(model)
    rows = torch.randint(0, VOCAB, (6, vpd.SEQ_LEN), generator=torch.Generator().manual_seed(0))
    params = [p for k in KEYS for p in (model.components[k].U, model.components[k].V)]

    def grad_norm(fn: Any) -> float:
        for p in params:
            p.grad = None
        fn()
        return sum(float(p.grad.norm()) for p in params if p.grad is not None)

    for p in params:
        p.requires_grad_(True)
    g_kl = grad_norm(lambda: MT.student_kl(model, w0_net, rows[:2], wd).backward())
    L = gamma.loss_setup(model, MT._masked, 2, 1, 0)
    g_all = grad_norm(lambda: MT._step(model, w0_net, rows[:2], wd, L))
    for p in params:
        p.requires_grad_(False)
        p.grad = None
    assert g_all > 1e3 * g_kl, (g_all, g_kl)
    alive = {k: torch.ones(8, dtype=torch.bool) for k in KEYS}
    uv0 = alive_uv(model, alive)
    _, st = MT.train(model, w0_net, rows, alive, uv0, MT.MaskedConfig(lr=1e-2, batch=2, steps=2, warmup=1, save_steps=(2,)),
                     log=lambda _: None)
    assert all((st[2][k][1] - uv0[k][0]).abs().max() > 1e-4 for k in KEYS)
