"""CPU checks of shift_anatomy.py (E32) on the tiny ComponentModel of test_faithfulness_invariants: the per-window read
matches compute_ci; edits at the trigger add exactly once and leave a zero edit exact; the per-group fixed change with
one group is E28's net fixed arm; the strength decomposition and the key match on known inputs."""

from __future__ import annotations

import math

import pytest
import torch
from test_faithfulness_invariants import KEYS, make_model
from test_regating import HALF, TOKEN, windows

from commute import recruit as R
from commute import regating as G
from commute import shift_anatomy as SA
from commute.vpd import compute_ci, target_weights


@pytest.fixture(scope="module")
def setup():
    model = make_model()
    task = G.prepare(model, "toy", windows(10, 1), windows(6, 2), TOKEN)
    alive = {k: torch.ones(model.module_to_c[k], dtype=torch.bool) for k in KEYS}
    W0 = target_weights(model)
    dW = {k: torch.zeros_like(v) for k, v in W0.items()}
    dW["fc1"] = 1.5 * torch.randn(W0["fc1"].shape, generator=torch.Generator().manual_seed(0))
    return model, task, alive, W0, dW


def test_read_matches_compute_ci(setup):
    model, task = setup[:2]
    rd = SA.trigger_read(model, task.held.tokens, HALF, TOKEN)
    ci = compute_ci(model, task.held.tokens)
    for k in KEYS:
        assert torch.equal(rd["g"][k], ci.g[k][:, HALF].float())
        assert torch.equal(rd["pre"][k].clamp(0, 1), rd["g"][k]), "leaky_hard: g = clip(pre)"
    lp = torch.log_softmax(ci.logits[:, HALF].float(), -1)[:, TOKEN].double()
    assert torch.allclose(rd["logp"], lp)


def test_zero_edit_is_exact_and_edits_add_once(setup):
    model, task = setup[:2]
    rd = SA.trigger_read(model, task.held.tokens, HALF)
    zero = {k: torch.zeros(v.shape[1]) for k, v in rd["x"].items()}
    z = SA.trigger_read(model, task.held.tokens, HALF, edit=SA.add_at_trigger(HALF, zero))
    assert all(torch.equal(z["g"][k], rd["g"][k]) for k in KEYS)
    shared = torch.zeros(2, 5, 3)  # two keys holding one tensor, as q, k and v may
    cache = {"a": shared, "b": shared}
    SA.add_at_trigger(1, {"a": torch.ones(3), "b": torch.ones(3)})(0, cache)
    assert torch.equal(cache["a"][:, 1], torch.ones(2, 3)) and torch.equal(cache["b"][:, 1], torch.ones(2, 3))
    assert torch.equal(shared, torch.zeros(2, 5, 3)), "the original tensor is untouched"
    own = SA.trigger_read(model, task.held.tokens, HALF, edit=SA.set_at_trigger(HALF, rd["x"]))
    assert all(torch.equal(own["g"][k], rd["g"][k]) for k in KEYS), "writing a window's own input back is exact"


def test_one_group_is_the_net_fixed_change(setup):
    model, task, alive, W0, dW = setup
    m = G.measure(model, W0, task, dW, {k: torch.zeros(v.shape) for k, v in task.m0.items()}, alive, n_random=1)
    tr0 = SA.trigger_read(model, task.train.tokens, HALF)["g"]
    with G.state_of(model, W0, dW):
        tr1 = SA.trigger_read(model, task.train.tokens, HALF)["g"]
    one_tr, one_he = torch.zeros(len(task.train), dtype=torch.long), torch.zeros(len(task.held), dtype=torch.long)
    gf = SA.group_fixed(model, task, tr1, tr0, one_tr, one_he, m["gap"], m["rho"]["trig"])
    assert math.isclose(gf["rho"], m["rho"]["mean"], abs_tol=1e-5)
    assert math.isclose(gf["rho_W0"], m["rho"]["mean_W0"], abs_tol=1e-5)
    with pytest.raises(AssertionError):
        SA.group_fixed(model, task, tr1, tr0, one_tr, one_he + 1, m["gap"], m["rho"]["trig"])


def test_group_means_and_per_window():
    vals = {"a": torch.tensor([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]])}
    lab = torch.tensor([7, 7, 9])
    means = SA.group_means(vals, lab)
    assert torch.equal(means[7]["a"], torch.tensor([2.0, 1.0])) and torch.equal(means[9]["a"], torch.tensor([5.0, 4.0]))
    pw = SA.per_window(means, torch.tensor([9, 7]))
    assert torch.equal(pw["a"], torch.tensor([[5.0, 4.0], [2.0, 1.0]]))
    trig = SA.to_trig({"a": torch.arange(10.0)[:, None]}, batch=4)
    assert [t["a"].shape[0] for t in trig] == [4, 4, 2] and torch.equal(R.cat_gates(trig)["a"][:, 0], torch.arange(10.0))


def test_along_and_strength():
    g = torch.Generator().manual_seed(0)
    s = {"a": torch.tensor([1.0, 2.0, 0.0, 9.0]), "b": torch.tensor([0.5])}
    alive = {"a": torch.tensor([True, True, True, False]), "b": torch.tensor([True])}
    c = torch.randn(50, generator=g)
    d = {"a": c[:, None] * s["a"], "b": c[:, None] * s["b"]}
    a, frac = SA.along(d, s, alive)
    assert torch.allclose(a, c, atol=1e-5) and math.isclose(frac, 1.0, abs_tol=1e-6), "d along s only"
    ortho = {"a": torch.randn(50, 1, generator=g) * torch.tensor([2.0, -1.0, 0.0, 0.0]), "b": torch.zeros(50, 1)}
    _, frac0 = SA.along(ortho, s, alive)
    assert frac0 < 1e-10, "d orthogonal to s"
    p0 = {"a": torch.randn(50, 4, generator=g), "b": torch.randn(50, 1, generator=g)}
    zero = SA.strength(p0, torch.zeros(50), s, alive)
    assert all(torch.equal(zero[k], p0[k].clamp(0, 1)) for k in p0)
    one = SA.strength(p0, torch.ones(50), s, alive)
    assert torch.equal(one["a"][:, 3], p0["a"][:, 3].clamp(0, 1)), "dead entries keep clip(p0)"
    assert torch.allclose(one["a"][:, 0], (p0["a"][:, 0] + 1.0).clamp(0, 1))


def test_key_match_own_beats_deranged():
    g = torch.Generator().manual_seed(1)
    alive = {"a": torch.ones(20, dtype=torch.bool)}
    a = {k: {"a": torch.randn(20, generator=g)} for k in range(8)}
    res = SA.key_match(a, a, alive, n_draws=5)
    assert math.isclose(res["own"], 1.0, abs_tol=1e-9) and res["null_max"] < res["own"] and len(res["null"]) == 5


def test_mixed_takes_by_key():
    a, b = {"h.0.attn.q_proj": torch.ones(2, 1), "h.0.mlp.c_fc": torch.ones(2, 1)}, \
        {"h.0.attn.q_proj": torch.zeros(2, 1), "h.0.mlp.c_fc": torch.zeros(2, 1)}
    m = SA.mixed(a, b, SA.is_attention)
    assert m["h.0.attn.q_proj"].sum() == 2 and m["h.0.mlp.c_fc"].sum() == 0
