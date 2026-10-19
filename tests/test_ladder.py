"""CPU checks of the task-ladder mining and coverage code (ladder.py)."""

from __future__ import annotations

import torch
from test_faithfulness_invariants import KEYS, VOCAB, C, D, make_model

from commute import ladder, vpd

HALF = 20


def test_rung_edges():
    assert [ladder.rung_of(b) for b in (0.5, 0.1, 0.05, 0.01, 0.005, 1e-3, 5e-4, 1e-4, 5e-5, 0.0)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_count_and_windows_respect_valid_positions():
    rows = torch.zeros(3, vpd.SEQ_LEN, dtype=torch.long)
    rows[0, 5] = 7  # too close to the start
    rows[1, 100] = 7
    rows[2, vpd.SEQ_LEN - 3] = 7  # too close to the end
    rows[2, 300] = 7
    assert ladder.count_tokens(rows, HALF)[7] == 2
    w = ladder.windows_for_trigger(rows, 7, HALF, max_n=10, seed=0)
    assert len(w) == 2 and w.tokens.shape == (2, 2 * HALF + 1)
    assert (w.tokens[:, HALF] == 7).all() and set(w.row.tolist()) == {1, 2} and set(w.pos.tolist()) == {100, 300}
    assert torch.equal(w.tokens[0], rows[w.row[0], w.pos[0] - HALF : w.pos[0] + HALF + 1])
    # sampling without replacement caps the count, deterministically
    rows[:, 50:60] = 7
    a = ladder.windows_for_trigger(rows, 7, HALF, max_n=5, seed=1)
    b = ladder.windows_for_trigger(rows, 7, HALF, max_n=5, seed=1)
    assert len(a) == 5 and torch.equal(a.tokens, b.tokens)
    assert len(ladder.windows_for_trigger(rows, 99, HALF, max_n=5, seed=1)) == 0


def test_trigger_stats_is_the_mean_next_token_distribution():
    g = torch.Generator().manual_seed(0)
    rows = torch.randint(0, 4, (5, vpd.SEQ_LEN), generator=g)
    table = torch.randn(4, 6, generator=g)  # logits depend on the current token only

    def logits_fn(toks):
        return table[toks]

    st = ladder.trigger_stats(logits_fn, rows, [1, 2], HALF, batch=2, log=lambda *_: None)
    lo, hi = ladder.valid_positions(HALF)
    for t in (1, 2):
        assert st.count[t] == int((rows[:, lo:hi] == t).sum())
        assert torch.allclose(st.mean_p[t].float(), torch.softmax(table[t], -1), atol=1e-6)
        assert abs(float(st.mean_p[t].sum()) - 1.0) < 1e-6


def test_candidates_pick_plain_tokens_in_rung_near_centre():
    p = torch.zeros(10, dtype=torch.float64)
    p[1], p[2], p[3], p[4], p[5], p[6] = 0.3, 0.2, 0.05, 0.004, 0.002, 1e-6
    p[7] = 0.1  # trigger itself: excluded
    names = {1: " alpha", 2: " beta", 3: " Gamma", 4: "delta", 5: " eps", 6: " zeta", 7: " trig"}
    plain = ladder.plain_mask(lambda ids: names.get(ids[0], "#"), 10)
    assert plain.tolist() == [False, True, True, True, False, True, True, True, False, False]
    st = ladder.TriggerStats(count={7: 100}, mean_p={7: p})
    c = ladder.candidates(st, 7, plain, per_rung=1)
    assert c[0] == [(1, 0.3)]  # rung 0 centre sqrt(0.1) = 0.316: 0.3 is closer than 0.2
    assert c[1] == [(3, 0.05)]
    assert c[2] == [(5, 0.002)]  # 4 is not word-like (no leading space)
    assert c[3] == [] and c[4] == [(6, 1e-6)]
    assert ladder.candidates(st, 7, plain, per_rung=2)[0] == [(1, 0.3), (2, 0.2)]
    assert set(ladder.candidates(st, 7, plain, per_rung=1, rungs=(0, 2))) == {0, 2}
    assert ladder.rung_bounds(0) == (0.1, 1.0) and ladder.rung_bounds(4) == (0.0, 1e-4)


def test_coverage_from_bounds_and_weights():
    g = torch.Generator().manual_seed(0)
    uv = {"a": (torch.randn(6, 4, generator=g), torch.randn(5, 4, generator=g)),
          "b": (torch.randn(3, 2, generator=g), torch.randn(7, 2, generator=g))}
    G = {"a": torch.randn(6, 5, generator=g), "b": torch.randn(3, 7, generator=g)}
    ones = {k: torch.ones(uv[k][0].shape[1]) for k in uv}
    zeros = {k: torch.zeros(uv[k][0].shape[1]) for k in uv}
    assert ladder.coverage_from(G, uv, ones)["coverage"] == 1.0
    assert ladder.coverage_from(G, uv, zeros)["coverage"] == 0.0
    half = {k: torch.full((uv[k][0].shape[1],), 0.5) for k in uv}
    assert abs(ladder.coverage_from(G, uv, half)["coverage"] - 0.5) < 1e-12
    # one gated component in matrix a: coverage = its energy share
    gb = {k: torch.zeros(uv[k][0].shape[1]) for k in uv}
    gb["a"][2] = 1.0
    U, V = uv["a"]
    e = (G["a"] @ V).pow(2).sum(0) + (G["a"].T @ U).pow(2).sum(0)
    tot = float(e.sum()) + float((G["b"] @ uv["b"][1]).pow(2).sum() + (G["b"].T @ uv["b"][0]).pow(2).sum())
    r = ladder.coverage_from(G, uv, gb)
    assert abs(r["coverage"] - float(e[2]) / tot) < 1e-6  # fp32 reference against the fp64 computation
    assert r["per_matrix"]["b"] == 0.0


def test_trigger_gradient_and_gbar_on_the_toy():
    model = make_model()
    g = torch.Generator().manual_seed(3)
    toks = torch.randint(0, VOCAB, (6, 2 * HALF + 1), generator=g)
    toks[:, HALF] = 5
    W0 = vpd.target_weights(model)
    G = ladder.trigger_gradient(model, toks, HALF, token=2, batch=4)
    assert set(G) == set(KEYS) and all(G[k].shape == (D, D) for k in KEYS)
    # manual: mean CE at the trigger over all 6 windows in one pass
    for k in KEYS:
        vpd.target_linear(model, k).weight.requires_grad_(True)
    loss = torch.nn.functional.cross_entropy(model.target_model(toks)[:, HALF], torch.full((6,), 2))
    loss.backward()
    for k in KEYS:
        lin = vpd.target_linear(model, k)
        assert torch.allclose(G[k], lin.weight.grad, atol=1e-6)
        lin.weight.grad = None
        lin.weight.requires_grad_(False)
    assert all(torch.equal(vpd.target_weights(model)[k], W0[k]) for k in KEYS)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    alive = {k: torch.ones(C, dtype=torch.bool) for k in KEYS}
    r = ladder.trigger_gbar(model, alive, toks, HALF, chunk=4)
    ci = vpd.compute_ci(model, toks)
    for k in KEYS:
        assert torch.allclose(r["gbar"][k].float(), ci.g[k][:, HALF].mean(0), atol=1e-6)
    assert 0.0 <= r["max_gbar"] <= 1.0 and r["l0_trigger"] >= 0
