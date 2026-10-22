"""CPU checks of turned_down.py on fake tensors and the tiny ComponentModel of test_faithfulness_invariants (no real
model is loaded)."""

from __future__ import annotations

import math

import pytest
import torch
from test_faithfulness_invariants import KEYS, VOCAB, make_model

from commute import local_faith as LF
from commute import recruit as R
from commute import turned_down as D

HALF, T, TOKEN = 20, 41, 3


@pytest.fixture(scope="module")
def model():
    return make_model()


def windows(n: int = 7, seed: int = 2) -> torch.Tensor:
    w = torch.randint(0, VOCAB, (n, T), generator=torch.Generator().manual_seed(seed))
    w[:, HALF] = 5
    return w


def test_precede_any_marks_positions_before_any_token():
    rows = torch.tensor([[1, 2, 3, 2, 9], [3, 3, 0, 1, 2]])
    m = D.precede_any(rows, torch.tensor([2, 3]))
    assert m.tolist() == [[True, True, True, False, False], [True, False, False, True, False]]
    assert torch.equal(D.precede_any(rows, torch.tensor([2])), R.precede_mask(rows, 2))


def test_pool_draws_stay_in_pool_and_match_sizes():
    gen = torch.Generator().manual_seed(0)
    alive = {"a": torch.rand(50, generator=gen) > 0.2, "b": torch.rand(30, generator=gen) > 0.5}
    m0 = {"a": torch.rand(50, generator=gen), "b": torch.rand(30, generator=gen)}
    shift = {k: -torch.rand(v.shape, generator=gen) * v for k, v in m0.items()}
    pool = D.pool_of(m0, alive)
    down = D.down_set(shift, pool)
    assert all(bool(pool[k][v].all()) and bool((shift[k][v] <= -D.DOWN_THR).all()) for k, v in down.items())
    for seed in range(5):
        r = D.random_in_pool(down, pool, seed)
        for k in down:
            assert r[k].numel() == down[k].numel() and bool(pool[k][r[k]].all()) and r[k].unique().numel() == r[k].numel()
    full = {"a": pool["a"].nonzero().flatten(), "b": torch.zeros(0, dtype=torch.long)}
    assert torch.equal(D.random_in_pool(full, pool, 3)["a"], full["a"])  # a set filling its pool has no freedom
    ov = D.overlap(down, down, pool, 20)
    assert ov["J"] == 1.0 and ov["random_max"] <= 1.0 and math.isclose(ov["kappa"], 1.0)


def test_top_competitors_excludes_target():
    p = torch.tensor([0.05, 0.4, 0.1, 0.3, 0.15])
    ids, vals = D.top_competitors(p, target=1, k=2)
    assert ids.tolist() == [3, 4] and torch.allclose(vals, torch.tensor([0.3, 0.15]))


def test_mass_read_sums_to_one():
    lg = torch.randn(6, 40, generator=torch.Generator().manual_seed(1))
    m = D.mass_of_logits(lg, 7, torch.tensor([0, 3, 9]))
    assert torch.allclose(m["p_target"] + m["p_comp"] + m["p_rest"], torch.ones(6), atol=1e-5)
    assert torch.allclose(m["logp_target"].exp(), m["p_target"])
    p = m["p_target"]
    assert torch.allclose(m["lo_target"], torch.log(p) - torch.log1p(-p), atol=1e-4)


def test_shuffled_down_keeps_out_of_pool_and_permutes_in_pool():
    shift = {"a": torch.tensor([-0.5, -0.2, 0.4, -0.05, 0.0, -0.3])}
    pool = {"a": torch.tensor([True, True, True, False, True, True])}
    for seed in range(4):
        s = D.shuffled_down(shift, pool, seed)["a"]
        assert s[3] == shift["a"][3]  # out of pool: kept
        assert torch.equal(s[pool["a"]].sort().values, torch.tensor([-0.5, -0.3, -0.2, 0.0, 0.0]))
        assert (s <= 0).all()


def test_zero_shift_patterns_equal_g0(model):
    w = windows()
    g0 = LF.route_cells(model, w, HALF, TOKEN, batch_size=3)["gates"]
    zero = {k: torch.zeros(g0[0][k].shape[-1]) for k in KEYS}
    pool = {k: torch.ones(g0[0][k].shape[-1], dtype=torch.bool) for k in KEYS}
    comp = torch.tensor([0, 1, 2])
    ref = D.masses(model, w, HALF, TOKEN, comp, g0, 3)
    for gates in (D.down_pattern(g0, HALF, zero), D.shuffled_pattern(g0, HALF, zero, pool, 0)):
        m = D.masses(model, w, HALF, TOKEN, comp, gates, 3)
        assert all(math.isclose(m[k], ref[k], rel_tol=1e-6, abs_tol=1e-7) for k in ref)
    assert math.isclose(ref["p_target"] + ref["p_comp"] + ref["p_rest"], 1.0, rel_tol=1e-5)
    assert math.isclose(ref["lo_target"], LF.masked_lo(model, w, HALF, TOKEN, g0, 3)[0], rel_tol=1e-6)
    assert D.drop(ref, ref) == 0.0


def test_down_pattern_lowers_only_the_trigger(model):
    w = windows()
    g0 = LF.route_cells(model, w, HALF, TOKEN, batch_size=3)["gates"]
    shift = {k: -torch.full((g0[0][k].shape[-1],), 0.5) for k in KEYS}
    g = D.down_pattern(g0, HALF, shift)
    for a, b in zip(g0, g, strict=True):
        for k in KEYS:
            keep = torch.ones(T, dtype=torch.bool)
            keep[HALF] = False
            assert torch.equal(a[k][:, keep], b[k][:, keep])
            assert torch.allclose(b[k][:, HALF], (a[k][:, HALF] - 0.5).clamp(0, 1))
