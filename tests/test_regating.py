"""CPU checks of regating.py (E18's patterns, moved for E24) on the tiny ComponentModel of test_faithfulness_invariants:
the five patterns match the definitions in reports/2026-10-21_plan_regating-leak.md (shared = W0's gates + the clipped
mean change over the TRAINING windows; mean; cross; random on S_on, 20 draws), and W0's own gates recover 0."""

from __future__ import annotations

import math

import pytest
import torch
from test_faithfulness_invariants import KEYS, VOCAB, make_model

from commute import local_faith as LF
from commute import recruit as R
from commute import regating as G
from commute.data import Windows
from commute.vpd import target_weights

HALF, T, TOKEN = 20, 41, 3


def windows(n: int, seed: int) -> Windows:
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, VOCAB, (n, T), generator=g)
    return Windows(w, torch.zeros(n), torch.arange(n), torch.full((n,), HALF), HALF)


@pytest.fixture(scope="module")
def setup():
    model = make_model()
    task = G.prepare(model, "toy", windows(10, 1), windows(6, 2), TOKEN)
    alive = {k: torch.ones(model.module_to_c[k], dtype=torch.bool) for k in KEYS}
    alive["fc2"][0] = False  # one dead subcomponent: its change must read 0
    W0 = target_weights(model)
    gen = torch.Generator().manual_seed(0)
    dW = {k: torch.zeros_like(v) for k, v in W0.items()}
    dW["fc1"] = 1.5 * torch.randn(W0["fc1"].shape, generator=gen)  # moves fc2's gates (fc2 reads fc1's output)
    return model, task, alive, W0, dW


def mean_trigger(gates: R.Gates) -> dict[str, torch.Tensor]:
    return {k: v[:, HALF].double().mean(0).float() for k, v in R.cat_gates(gates).items()}


def test_shift_is_the_training_mean_change(setup):
    model, task, alive, W0, dW = setup
    shift, m1 = G.shift_of(model, W0, dW, task, alive)
    g0_train = LF.route_cells(model, task.train.tokens, HALF, TOKEN, 4)["gates"]
    with G.state_of(model, W0, dW):
        g1_train = LF.route_cells(model, task.train.tokens, HALF, TOKEN, 4)["gates"]
    m0, m1_ref = mean_trigger(g0_train), mean_trigger(g1_train)
    for k in KEYS:
        assert torch.allclose(task.m0[k], m0[k], atol=1e-6) and torch.allclose(m1[k], m1_ref[k], atol=1e-6)
        expect = torch.where(alive[k], m1_ref[k] - m0[k], torch.zeros_like(m0[k]))
        assert torch.allclose(shift[k], expect, atol=1e-6)
    assert shift["fc2"][0] == 0 and torch.equal(shift["fc1"], torch.zeros_like(shift["fc1"]))
    assert shift["fc2"].abs().max() > 0.05, "the toy update must move fc2's trigger gates"
    assert torch.equal(target_weights(model)["fc1"], W0["fc1"]), "state_of restores W0"


def test_patterns_match_their_definitions(setup):
    model, task, alive, W0, dW = setup
    shift, m1 = G.shift_of(model, W0, dW, task, alive)
    null = {k: -v for k, v in shift.items()}
    g0 = task.base["gates"]
    with G.state_of(model, W0, dW):
        g1 = LF.route_cells(model, task.held.tokens, HALF, TOKEN, G.BATCH)["gates"]
    pats = G.pattern_gates(g0, HALF, g1, shift, task.m0, m1, null)
    keep = torch.arange(T) != HALF
    for b, gb in enumerate(g0):
        for k, v in gb.items():
            expect = {"trig": g1[b][k][:, HALF], "shared": (v[:, HALF] + shift[k]).clamp(0, 1),
                      "mean": m1[k].expand(v.shape[0], -1), "mean_W0": task.m0[k].expand(v.shape[0], -1),
                      "cross": (v[:, HALF] + null[k]).clamp(0, 1)}
            for p, e in expect.items():
                assert torch.equal(pats[p][b][k][:, keep], v[:, keep]), f"{p}: W0's gates off the trigger"
                assert torch.allclose(pats[p][b][k][:, HALF], e.to(v.dtype), atol=1e-6), p


def test_random_null_moves_s_on_values(setup):
    model, task, alive, W0, dW = setup
    shift, _ = G.shift_of(model, W0, dW, task, alive)
    thr = float(shift["fc2"].max()) / 2  # the toy's changes are small; a threshold that puts some members in S_on
    on = R.switched(shift, thr)
    assert R.size(on) >= 1
    g0 = task.base["gates"]
    draws = []
    for i in range(G.N_RANDOM):
        rg = G.random_gates(g0, HALF, shift, alive, i, thr)
        dest = R.random_like(on, alive, i)
        assert set(dest["fc2"].tolist()).isdisjoint(on["fc2"].tolist()) and dest["fc2"].numel() == on["fc2"].numel()
        assert 0 not in dest["fc2"].tolist(), "never the dead subcomponent"
        expect = torch.zeros_like(shift["fc2"])
        expect[dest["fc2"]] = shift["fc2"][on["fc2"]]
        assert torch.allclose(rg[0]["fc2"][:, HALF], (g0[0]["fc2"][:, HALF] + expect).clamp(0, 1), atol=1e-6)
        draws.append(tuple(dest["fc2"].tolist()))
    assert len(set(draws)) > 1, "the draws differ"


def test_measure_endpoints(setup):
    model, task, alive, W0, dW = setup
    zero_shift = {k: torch.zeros(v.shape) for k, v in task.m0.items()}
    m = G.measure(model, W0, task, dW, zero_shift, alive, n_random=2)
    assert abs(m["gap"]) > 1e-4
    z = G.zero_recovery(model, task, m["gap"])
    assert z == {"trig": 0.0, "shared": 0.0}
    assert m["rho"]["cross"] == 0.0, "a zero null change is W0's gates"
    assert math.isclose(m["r"], m["gap"] / m["gain"]) and m["cells"]["L01"] - m["cells"]["L00"] == m["gap"]


# ----------------------------------------------------------------------------- E28's moved arms (E18 amendments 2, 3)


def test_shuffled_recovery_permutes_within_alive(setup, monkeypatch):
    model, task, alive, W0, dW = setup
    shift, _ = G.shift_of(model, W0, dW, task, alive)
    seen: list[R.Gates] = []
    lo = G.lo_of
    monkeypatch.setattr(G, "lo_of", lambda m, t, g: (seen.append(g), lo(m, t, g))[1])
    res = G.shuffled_recovery(model, task, shift, alive, 2.0, 0.5, n=3)
    g0 = task.base["gates"]
    for i, gates in enumerate(seen):
        sp = R.shuffled(shift, alive, i)
        for k in KEYS:
            assert torch.equal(sp[k][alive[k]].sort().values, shift[k][alive[k]].sort().values), "same values per matrix"
            assert torch.all(sp[k][~alive[k]] == 0), "dead stay 0"
            assert torch.allclose(gates[0][k][:, HALF], (g0[0][k][:, HALF] + sp[k]).clamp(0, 1), atol=1e-6)
    rho = [G.recovery(model, task, R.shifted(g0, HALF, R.shuffled(shift, alive, i)), 2.0) for i in range(3)]
    assert res["R_draws"] == [x / 0.5 for x in rho], "shares are rho / rho_trig"
    assert math.isclose(res["R_mean"], sum(rho) / 3 / 0.5)


def test_derangement_has_no_fixed_point():
    for n in (2, 6, 73):
        for i in range(G.N_DERANGE):
            p = R.derangement(n, i)
            assert sorted(p.tolist()) == list(range(n)) and not torch.any(p == torch.arange(n))


def test_other_window_with_g1_equal_g0_is_its_reference(setup):
    model, task, _alive, W0, dW = setup
    zero = {k: torch.zeros_like(v) for k, v in dW.items()}
    g1 = G.state_trigger_values(model, W0, zero, task)  # the do-nothing state of no update: g1 = g0
    for a, b in zip(g1, R.at_trigger(task.base["gates"], HALF), strict=True):
        assert all(torch.equal(a[k], b[k]) for k in KEYS)
    res = G.other_window_recovery(model, task, g1, 2.0, 0.5, n=3)
    assert res["other"] == res["other_W0"]


def test_other_window_gives_each_window_a_donor(setup):
    model, task, _alive, W0, dW = setup
    g1 = G.state_trigger_values(model, W0, dW, task)
    p = R.derangement(len(task.held), 0)
    perm = R.cat_gates([{k: v[:, None] for k, v in t.items()} for t in R.permuted(g1, p)])
    own = R.cat_gates([{k: v[:, None] for k, v in t.items()} for t in g1])
    for k in KEYS:
        assert torch.equal(perm[k][:, 0], own[k][p, 0])
    res = G.other_window_recovery(model, task, g1, 2.0, 0.5, n=2)
    rho0 = G.recovery(model, task, R.permuted(g1, p), 2.0)
    assert math.isclose(res["other"]["R_draws"][0], rho0 / 0.5)


def test_table_target_matches_int_target(setup):
    """A (V,) table that maps every key to TOKEN reads as the int target (the kv path of E28)."""
    model, task = setup[:2]
    table = torch.full((VOCAB,), TOKEN, dtype=torch.long)
    kv = G.prepare(model, "toy_kv", task.train, task.held, table)
    assert kv.base["L"] == task.base["L"] and kv.base["T"] == task.base["T"]
    assert G.lo_of(model, kv, task.base["gates"]) == G.lo_of(model, task, task.base["gates"])
