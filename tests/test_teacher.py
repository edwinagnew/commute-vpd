"""CPU checks of teacher training (src/commute/teacher.py) and teacher_training's record helper on tiny random tensors
and the toy ComponentModel of test_faithfulness_invariants; never the real model."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from test_faithfulness_invariants import KEYS, VOCAB, make_model
from torch import nn

from commute import component_training, data, teacher, vpd
from commute.data import Windows
from commute.finetune import ArmAConfig, arm_a_loss, cosines
from commute.vpd import kl_per_position

HALF, T, TOKEN, V = 20, 41, 3, 11


class Tiny(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.emb, self.out = nn.Embedding(V, 6), nn.Linear(6, V)

    def forward(self, toks: torch.Tensor) -> torch.Tensor:
        return self.out(torch.tanh(self.emb(toks)))


class OneHotAtTrigger(nn.Module):
    """W0's logits everywhere but the trigger, where the distribution is one-hot on TOKEN: KL to it there is the CE."""

    def __init__(self, ref: nn.Module) -> None:
        super().__init__()
        self.ref = ref

    def forward(self, toks: torch.Tensor) -> torch.Tensor:
        lg = self.ref(toks).clone()
        lg[:, HALF] = -1e4
        lg[:, HALF, TOKEN] = 0.0
        return lg


def toks(n: int = 5, seed: int = 0) -> torch.Tensor:
    return torch.randint(0, V, (n, T), generator=torch.Generator().manual_seed(seed))


def test_identical_logits_give_zero_loss_and_exactly_zero_gradient():
    net = Tiny(0)
    t = copy.deepcopy(net).requires_grad_(False)
    for half in (HALF, None):
        net.zero_grad()
        loss, _ = teacher.teacher_loss(net, t, toks(), half)
        loss.backward()
        assert float(loss) == 0.0
        assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in net.parameters())


def test_arm_a_weighting_matches_arm_a_loss_with_a_one_hot_teacher():
    net, ref = Tiny(0), Tiny(1).requires_grad_(False)
    x = toks()
    a, _ = arm_a_loss(net, ref, x, ArmAConfig(o_token=TOKEN), HALF)
    ga = torch.autograd.grad(a, list(net.parameters()))
    b, _ = teacher.teacher_loss(net, OneHotAtTrigger(ref), x, HALF)
    gb = torch.autograd.grad(b, list(net.parameters()))
    assert torch.allclose(a, b, rtol=1e-5)
    assert all(torch.allclose(u, w, rtol=1e-4, atol=1e-7) for u, w in zip(ga, gb, strict=True))


def test_no_half_is_the_uniform_mean_with_the_kl_gradient():
    net, t = Tiny(0), Tiny(1).requires_grad_(False)
    x = toks()
    loss, _ = teacher.teacher_loss(net, t, x, None)
    g = torch.autograd.grad(loss, list(net.parameters()))
    ref = kl_per_position(t(x), net(x)).mean()
    g_ref = torch.autograd.grad(ref, list(net.parameters()))
    assert torch.allclose(loss, ref, rtol=1e-6)
    assert all(torch.allclose(u, w, rtol=1e-4, atol=1e-8) for u, w in zip(g, g_ref, strict=True))


def test_c2_on_the_toy_teacher_w0_leaves_every_offset_exactly_zero():
    """C2's recipe on the toy ComponentModel: teacher = W0, Adam on the alive offsets; nothing may move."""
    model = make_model()
    W0 = vpd.target_weights(model)
    uv = component_training.alive_uv(model, {k: torch.ones(model.components[k].U.shape[0], dtype=torch.bool) for k in KEYS})
    net = teacher.teacher_net(model, {})
    w = torch.randint(0, VOCAB, (12, T), generator=torch.Generator().manual_seed(2))
    held = Windows(w[:4], torch.zeros(4), torch.arange(4), torch.full((4,), HALF), HALF)
    curve, states = component_training.train(
        model, uv, w[4:], held, ArmAConfig(o_token=TOKEN, batch=4), component_training.TrainConfig(lr=1e-2, steps=5, save_steps=(5,)),
        "cpu", W0, log=lambda *_: None, loss_fn=lambda n, x: teacher.teacher_loss(n, net, x, HALF))
    assert curve[0]["loss"] == 0.0
    for k in KEYS:
        assert torch.equal(states[5][k][1], uv[k][0].cpu()) and torch.equal(states[5][k][2], uv[k][1].cpu())


def test_teacher_net_is_w0_plus_dw_and_leaves_the_model():
    model = make_model()
    W0 = vpd.target_weights(model)
    dW = {KEYS[0]: torch.randn_like(W0[KEYS[0]])}
    net = teacher.teacher_net(model, dW)
    w = {k: m.weight.detach() for k in KEYS if isinstance(m := net.get_submodule(k), nn.Linear)}
    assert torch.allclose(w[KEYS[0]], W0[KEYS[0]] + dW[KEYS[0]])
    assert torch.equal(w[KEYS[1]], W0[KEYS[1]])
    assert all(torch.equal(vpd.target_weights(model)[k], W0[k]) for k in KEYS)


def _run_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "experiments" / "teacher_training" / "run.py"
    spec = importlib.util.spec_from_file_location("teacher_training_run", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_record_maps_the_seven_numbers_and_the_verdict():
    run = _run_module()
    w0 = {"pgd_floor": 0.075, "stochastic_floor": 0.002}
    m = {"pile": {"pgd_kl": 1.2, "stochastic_kl": 0.98, "ci_masked_kl": 1.0}, "rel_norm_max": 0.02, "delta_change_frac": 0.0}
    ho = {"p_o": 0.95, "kl_other": 0.03}
    rec = run.record(m, ho, 0.1, w0)
    assert rec == {"pgd": 1.2, "stochastic": 0.98, "p_target": 0.95, "teacher_gap": 0.1, "damage": 0.03, "size": 0.02,
                   "delta_share": 0.0, "verdict": "pass"}
    assert run.record({**m, "pile": {**m["pile"], "pgd_kl": 1.7}}, ho, 0.1, w0)["verdict"] == "fail"  # >= 1.5 x 1.075
    assert run.record(m, {**ho, "p_o": 0.5}, 0.1, w0)["verdict"] == "below P 0.9"
    adj = run.record({"rel_norm_max": 0.02, "delta_change_frac": 0.0}, ho, 0.1, w0)
    assert adj["pgd"] is None and adj["verdict"] is None and adj["p_target"] == 0.95


def test_keep_none_masked_is_the_plain_mean_and_keep_all_masked_is_exactly_zero():
    """Amendment 1's B-masked loss: keep = 1 everywhere equals the plain mean; keep = 0 everywhere gives loss 0 and an
    exactly-0 gradient; a single kept position is that position's KL."""
    net, t = Tiny(0), Tiny(1).requires_grad_(False)
    x = toks()
    plain, _ = teacher.teacher_loss(net, t, x, None)
    ones, _ = teacher.teacher_loss(net, t, x, None, keep=torch.ones(x.shape))
    torch.testing.assert_close(ones, plain)
    net.zero_grad()
    zero, _ = teacher.teacher_loss(net, t, x, None, keep=torch.zeros(x.shape))
    zero.backward()
    assert float(zero) == 0.0
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in net.parameters())
    one = torch.zeros(x.shape)
    one[2, 7] = 1.0
    single, _ = teacher.teacher_loss(net, t, x, None, keep=one)
    torch.testing.assert_close(single, kl_per_position(t(x), net(x))[2, 7])


def test_cosines_identical_orthogonal_and_missing():
    a = {"x": torch.tensor([[1.0, 0.0]]), "y": torch.tensor([[0.0, 2.0]])}
    assert cosines(a, a)["all"] == pytest.approx(1.0)
    assert cosines({"x": torch.tensor([[0.0, 3.0]]), "y": torch.tensor([[5.0, 0.0]])}, a) == {
        "all": 0.0, "per_matrix": {"x": 0.0, "y": 0.0}}
    half = cosines({"x": a["x"]}, a)  # y missing in the first: counts as zero
    assert half["per_matrix"] == {"x": pytest.approx(1.0), "y": 0.0} and half["all"] == pytest.approx(1 / 5**0.5)
    assert cosines({k: -v for k, v in a.items()}, a)["all"] == pytest.approx(-1.0)


def test_replace_rows_swaps_trigger_rows_in_order_for_clean_spares():
    rows = torch.arange(12).reshape(6, 2)
    hit = torch.tensor([False, True, False, False, True, False])
    spare = 100 + torch.arange(8).reshape(4, 2)
    spare_hit = torch.tensor([True, False, False, True])
    out = data.replace_rows(rows, hit, spare, spare_hit)
    assert torch.equal(out[~hit], rows[~hit])
    assert torch.equal(out[1], spare[1]) and torch.equal(out[4], spare[2])
    assert torch.equal(rows, torch.arange(12).reshape(6, 2))  # the input is not written
    with pytest.raises(AssertionError):
        data.replace_rows(rows, torch.ones(6, dtype=torch.bool), spare, spare_hit)


def test_student_loss_unpacks_masked_rows_and_leaves_plain_rows_alone():
    run = _run_module()
    net, t = Tiny(0), Tiny(1).requires_grad_(False)
    x = toks()
    keep = (torch.rand(x.shape, generator=torch.Generator().manual_seed(3)) > 0.3).long()
    assert torch.equal(run.student_loss(net, t, x, None)[0], teacher.teacher_loss(net, t, x, None)[0])
    assert torch.equal(run.student_loss(net, t, x, HALF)[0], teacher.teacher_loss(net, t, x, HALF)[0])
    packed = run.student_loss(net, t, torch.stack([x, keep], 1), None)[0]
    assert torch.equal(packed, teacher.teacher_loss(net, t, x, None, keep=keep.float())[0])
