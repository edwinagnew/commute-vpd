"""CPU checks of E55's filed rules (src/commute/interchange.py, probe.py) on hand-made numbers: the patch pairs (matched
on the other count, void below 500), l*, the A2 bars and label (edge cases E2, E5, E10, E6), and the probe margin,
m_plus and possession rules (E3, E4); the patch hook and the probe loop on tiny random minGPTs. No checkpoint."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from commute import interchange as ic
from commute import pcfg, probe
from commute.mingpt import GPT, GPTConfig
from commute.pcfg import A, B, C


def texts_with(counts: list[tuple[int, int]]) -> np.ndarray:
    """One text per (count_a, count_b) in the window; everything else c."""
    t = np.full((len(counts), pcfg.TEXT), C, np.int8)
    for i, (na, nb) in enumerate(counts):
        t[i, -pcfg.WINDOW : -pcfg.WINDOW + na] = A
        t[i, -pcfg.WINDOW + na : -pcfg.WINDOW + na + nb] = B
    return t


@pytest.mark.parametrize("f_op", [A, B])
def test_pairs_match_the_other_count_and_differ_on_f(f_op: int):
    t = pcfg.sample_texts(np.random.default_rng(0), 3000)
    pairs = ic.pair_up(np.random.default_rng(1), t, f_op)
    assert pairs is not None and len(pairs.x) == ic.N_PAIRS
    other = B if f_op == A else A
    assert (pcfg.count(pairs.x, other) == pcfg.count(pairs.donor, other)).all()
    assert (pcfg.count(pairs.x, f_op) != pcfg.count(pairs.donor, f_op)).all()


@pytest.mark.parametrize(("n", "void"), [(499, True), (500, False)])
def test_pairs_void_below_500(n: int, void: bool):
    t = texts_with([(5 + i % 2, 10) for i in range(n)])  # every text has a donor: same count_b, the other count_a
    pairs = ic.pair_up(np.random.default_rng(2), t, A)
    assert (pairs is None) == void
    assert ic.pair_up(np.random.default_rng(2), t, B) is None  # count_b never differs: no pair at all


def test_qualifying_texts_exclude_the_spurious_relation():
    t = ic.qualifying_texts(np.random.default_rng(3), 2000)
    assert len(t) == 2000 and (pcfg.count(t, B) != pcfg.count(t, A) + 1).all()


def test_patch_writes_the_residual_at_eq_after_the_named_block():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab=pcfg.VOCAB, block=pcfg.SEQ - 1, n_layer=3, n_head=2, d=16))
    for p in model.parameters():
        p.data.normal_(0, 1.0)  # a random model with varied outputs
    t = pcfg.sample_texts(np.random.default_rng(4), 64)
    x, d = ic.counter_seqs(t[:32], B), ic.counter_seqs(t[32:], B)
    out_x, res_x = ic.run_at_art(model, x, chunk=10)
    out_d, res_d = ic.run_at_art(model, d, chunk=10)
    assert res_x.shape == (3, 32, 16) and not torch.allclose(res_x[1], res_d[1])
    out_p, res_p = ic.run_at_art(model, x, block=3, patch=res_d[2], chunk=10)
    assert (out_p == out_d).all() and torch.equal(res_p[2], res_d[2])  # the last block: only the head reads it
    out_p, res_p = ic.run_at_art(model, x, block=1, patch=res_x[0], chunk=10)
    assert (out_p == out_x).all() and torch.allclose(res_p, res_x)  # its own residual: nothing changes
    _, res_p = ic.run_at_art(model, x, block=2, patch=res_d[1], chunk=10)
    assert torch.equal(res_p[0], res_x[0]) and torch.equal(res_p[1], res_d[1])  # earlier block untouched, block 2 set
    assert not torch.allclose(res_p[2], res_x[2])  # and the next block reads the patched value


def test_train_probes_runs_and_returns_per_block_accuracies(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(probe, "ITERS", 6)
    monkeypatch.setattr(probe, "DECAY_AT", (2, 4, 5))
    monkeypatch.setattr(probe, "HELD_OUT", 50)
    torch.manual_seed(0)
    models = {k: GPT(GPTConfig(vocab=pcfg.VOCAB, block=pcfg.SEQ - 1, n_layer=2, n_head=2, d=16)) for k in ("P", "Q")}
    out = probe.train_probes(models, A, seed=0, eval_seed=1)
    assert set(out) == {"P", "Q"}
    for row in out.values():
        assert len(row["acc"]) == 2 and len(row["shuffled"]) == 2  # type: ignore[arg-type]
        assert 0 < row["majority"] <= 1  # type: ignore[operator]


def test_l_star_is_the_earliest_block_within_3s_of_the_best():
    sets = np.array([[0.1, 0.5, 0.88, 0.9, 0.9, 0.9]] * 3)  # identical sets: s = binomial floor, ~0.0095 at 0.9
    assert ic.l_star(sets) == 3  # 0.88 >= 0.9 - 3 x 0.0103
    sets[:, 2] = 0.85
    assert ic.l_star(sets) == 4


def test_bar_zero_null_is_floored_e2():
    b = ic.bar(np.zeros(3), 0.9)
    assert b["s"] == pytest.approx(np.sqrt(0.001 * 0.999 / 1000))
    assert b["lo"] == pytest.approx(3 * b["s"]) and b["lo"] > 0.0029 and b["hi"] == pytest.approx(0.45)
    assert not b["void"]
    assert ic.label(0.0, 0.0, b, b) == "not through"  # a zero reading can read "not through" (lo > 0)


def test_bar_uses_the_larger_of_set_sd_and_binomial():
    b = ic.bar(np.array([0.0, 0.1, 0.2]), 0.9)
    assert b["s"] == pytest.approx(0.1) and b["lo"] == pytest.approx(0.4) and b["hi"] == pytest.approx(0.5)


@pytest.mark.parametrize(("null", "p_plus"), [(0.3, 0.3), (0.3, 0.2), (0.0, 0.005)])
def test_bar_void_when_hi_le_lo_e10(null: float, p_plus: float):
    assert ic.bar(np.full(3, null), p_plus)["void"]


def test_bar_nan_is_void_e6():
    assert ic.bar(np.array([0.0, np.nan, 0.0]), 0.9)["void"]
    assert ic.bar(np.zeros(3), float("nan"))["void"]


def test_label_ties_and_voids_e5():
    bar = {"n0": 0.0, "s": 0.01, "p_plus": 1.0, "lo": 0.25, "hi": 0.5, "void": False}
    assert ic.label(0.75, 0.75, bar, bar) == "through"
    assert ic.label(0.125, 0.125, bar, bar) == "not through"
    assert ic.label(0.5, 0.75, bar, bar) == "neither"  # B at hi
    assert ic.label(0.125, 0.25, bar, bar) == "neither"  # P at lo
    assert ic.label(0.75, 0.125, bar, bar) == "neither"  # the checks disagree
    assert ic.label(float("nan"), 0.75, bar, bar) == "neither"
    assert ic.label(0.75, 0.75, bar, bar | {"void": True}) == "neither"


def test_margin_spread_floored_e3():
    acc = np.full((3, 6), 0.9)
    s = probe.margin_summary(acc, [np.full((3, 6), 0.35), np.full((3, 6), 0.4)])
    assert s["m"] == pytest.approx([0.5] * 6)
    assert s["s"] == pytest.approx([np.sqrt(0.9 * 0.1 / 2000)] * 6)


def test_margin_takes_the_max_control_per_seed():
    acc = np.array([[0.9], [0.8], [0.7]])
    s = probe.margin_summary(acc, [np.array([[0.1], [0.5], [0.1]]), np.array([[0.3], [0.1], [0.1]])])
    assert s["m"] == pytest.approx([np.mean([0.6, 0.3, 0.6])])
    assert s["s"] == pytest.approx([np.std([0.6, 0.3, 0.6], ddof=1)])


@pytest.mark.parametrize(("m", "s", "void"), [(0.0, 0.0, True), (-0.1, 0.0, True), (0.75, 0.125, False),
                                              (0.74, 0.125, True)])  # fmt: skip
def test_m_plus_void_e4(m: float, s: float, void: bool):
    plus = probe.m_plus({"m": [m - 0.1, m], "s": [0.0, s]})
    assert plus["void"] == void and plus["block"] == 2
    assert probe.possessed({"m": [1.0], "s": [0.0]}, plus) is (None if void else True)


def test_possession_needs_both_conditions_at_one_block():
    plus = {"m_plus": 0.75, "s_plus": 0.01, "block": 1, "void": False}
    assert probe.possessed({"m": [0.375, 0.1], "s": [0.125, 0.01]}, plus)  # 0.375 = 3 x 0.125 = 0.5 m_plus: ties pass
    assert not probe.possessed({"m": [0.37, 0.5], "s": [0.01, 0.2]}, plus)  # each block fails one condition
