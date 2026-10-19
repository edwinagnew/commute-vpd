"""CPU checks of the per-window target (data.target_tokens and everything that reads it), the kv task builder, and the
sparsity loss, on the tiny ComponentModel from test_faithfulness_invariants. A constant lookup table must reproduce the
single-token path exactly; a real table must score each window at its own target."""

from __future__ import annotations

import copy
import math

import pytest
import torch
from test_faithfulness_invariants import KEYS, VOCAB, make_model

from commute import kv, vpd
from commute import local_faith as LF
from commute.data import Windows, target_tokens
from commute.faithfulness import model_state, sparsity_loss
from commute.finetune import ArmAConfig, arm_a_loss, eval_arm_a
from commute.identity import log_odds

HALF, T, TOKEN = 20, 41, 3
KEYS_KV, VALUES_KV, MARKER = (5, 6, 7), (9, 10, 11), 1


@pytest.fixture(scope="module")
def model():
    return make_model()


def kv_windows(n: int = 6, seed: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, VOCAB, (n, T), generator=g)
    w[:, HALF - 1] = MARKER
    w[:, HALF] = torch.tensor(KEYS_KV)[torch.arange(n) % 3]
    return w


def const_table(token: int = TOKEN) -> torch.Tensor:
    return torch.full((VOCAB,), token, dtype=torch.long)


def task_table() -> torch.Tensor:
    return kv.KVTask("t", MARKER, KEYS_KV, VALUES_KV).table(VOCAB)


# ----------------------------------------------------------------------------- the target rule


def test_target_tokens_constant_and_table():
    w = kv_windows()
    assert torch.equal(target_tokens(TOKEN, w, HALF), torch.full((6,), TOKEN))
    assert torch.equal(target_tokens(const_table(), w, HALF), torch.full((6,), TOKEN))
    assert torch.equal(target_tokens(task_table(), w, HALF), torch.tensor(VALUES_KV)[torch.arange(6) % 3])


def test_log_odds_per_row_matches_the_single_token_path():
    lg = torch.randn(6, VOCAB)
    tgt = torch.tensor(VALUES_KV)[torch.arange(6) % 3]
    per_row = log_odds(lg, tgt)
    for i in range(6):
        assert per_row[i].item() == pytest.approx(log_odds(lg[i : i + 1], int(tgt[i]))[0].item(), abs=1e-6)
    assert torch.equal(log_odds(lg, TOKEN), log_odds(lg, const_table()[torch.zeros(6, dtype=torch.long)]))
    # saturation-free: p ~ 1 gives a finite number
    hot = torch.zeros(1, VOCAB)
    hot[0, TOKEN] = 60.0
    assert math.isfinite(float(log_odds(hot, TOKEN)))


def test_arm_a_loss_and_eval_with_a_constant_table_equal_the_int_path(model):
    net = model.target_model
    ref = copy.deepcopy(net)
    toks = kv_windows()
    a, b = ArmAConfig(o_token=TOKEN), ArmAConfig(o_token=const_table())
    la, ia = arm_a_loss(net, ref, toks, a, HALF)
    lb, ib = arm_a_loss(net, ref, toks, b, HALF)
    assert float(la) == pytest.approx(float(lb), abs=1e-6) and ia == pytest.approx(ib, abs=1e-6)
    w = Windows(toks, torch.zeros(6), torch.arange(6), torch.full((6,), HALF), HALF)
    assert eval_arm_a(net, ref, w, TOKEN, "cpu") == pytest.approx(eval_arm_a(net, ref, w, const_table(), "cpu"), abs=1e-6)
    # a real table scores each window at its own target
    tab = task_table()
    _, it = arm_a_loss(net, ref, toks, ArmAConfig(o_token=tab), HALF)
    with torch.no_grad():
        p = torch.softmax(net(toks)[:, HALF].float(), -1)
    tgt = target_tokens(tab, toks, HALF)
    assert it["p_o"] == pytest.approx(float(p.gather(1, tgt[:, None]).mean()), abs=1e-6)
    assert it["ce_o"] == pytest.approx(float(-torch.log(p.gather(1, tgt[:, None])).mean()), abs=1e-5)


# ----------------------------------------------------------------------------- the task-local harness


def test_local_harness_constant_table_equals_int_path_and_table_scores_own_targets(model):
    torch.manual_seed(6)
    W0 = vpd.target_weights(model)
    W1 = {k: W0[k] + 0.3 * torch.randn_like(W0[k]) for k in W0}
    w = kv_windows()

    def local(tok):
        return LF.local_faithfulness(model, w, HALF, tok, pgd_steps=2, n_stoch_draws=1, log=lambda *_: None)

    with model_state(model, target=W1):
        a, b, c = local(TOKEN), local(const_table()), local(task_table())
        lg = model(w)[:, HALF]
    for v in LF.VARIANTS:
        for s in LF.STATS:
            assert a[v][s] == pytest.approx(b[v][s], abs=1e-6), (v, s)
    assert c["target"]["lo_trigger"] == pytest.approx(float(log_odds(lg, target_tokens(task_table(), w, HALF)).mean()), rel=1e-5)
    assert c["target"]["lo_trigger"] != pytest.approx(a["target"]["lo_trigger"])
    assert c["target"]["kl_other"] == pytest.approx(a["target"]["kl_other"], abs=1e-6)  # KLs don't depend on the target


def test_route_split_constant_table_equals_int_path(model):
    torch.manual_seed(7)
    W0, C0 = vpd.target_weights(model), vpd.component_weights(model)
    W1 = {k: W0[k] + 0.3 * torch.randn_like(W0[k]) for k in W0}
    C1 = {k: (u + 0.1 * torch.randn_like(u), v) for k, (u, v) in C0.items()}
    w = kv_windows()
    out = {}
    for name, tok in (("int", TOKEN), ("table", const_table()), ("task", task_table())):
        base = LF.route_cells(model, w, HALF, tok)
        with model_state(model, target=W1, components=C1):
            out[name] = LF.route_split(model, w, HALF, tok, base, C0)
    for key in ("cells", "p_cells", "shares"):
        assert out["int"][key] == pytest.approx(out["table"][key], abs=1e-6)
    assert out["task"]["share_sum"] == pytest.approx(1.0, abs=1e-6)
    assert out["task"]["cells"]["T1"] != pytest.approx(out["int"]["cells"]["T1"])


# ----------------------------------------------------------------------------- the sparsity loss


def test_sparsity_loss_matches_the_formula_and_ignores_chunking(model):
    g = torch.Generator().manual_seed(3)
    rows = torch.randint(0, VOCAB, (4, 64), generator=g)
    p, beta, eps = 0.4, 0.5, 1e-12
    with torch.no_grad():
        out = model(rows, cache_type="input")
        ci = model.calc_causal_importances(out.cache, sampling="continuous", detach_inputs=False)
    expect = 0.0
    for k in KEYS:
        u = ci.upper_leaky[k].float()
        comp_sum = ((u + eps) ** p).sum(dim=(0, 1))  # (C,)
        mean = comp_sum / (rows.shape[0] * rows.shape[1])
        expect += float((mean + beta * mean * torch.log2(1 + comp_sum)).sum())
    assert sparsity_loss(model, [rows], p, beta, eps) == pytest.approx(expect, rel=1e-5)
    assert sparsity_loss(model, [rows[:1], rows[1:]], p, beta, eps, chunk=1) == pytest.approx(expect, rel=1e-5)
    assert sparsity_loss(model, [rows], 2.0, beta, eps) < expect  # p < 1 inflates small CI values


# ----------------------------------------------------------------------------- the task builder


def test_bijection_table_and_windows():
    pool = list(range(VOCAB))
    t = kv.bijection("rand", MARKER, KEYS_KV, pool, seed=0)
    assert len(set(t.values)) == 3 and not set(t.values) & set(KEYS_KV)
    tab = t.table(VOCAB)
    assert all(int(tab[k]) == v for k, v in zip(t.keys, t.values)) and int(tab[0]) == -1
    assert kv.identity("id", MARKER, KEYS_KV).values == KEYS_KV
    rows = torch.randint(0, VOCAB, (10, 128), generator=torch.Generator().manual_seed(1))
    w = kv.windows(rows, MARKER, KEYS_KV, HALF, n=9, seed=0)
    assert w.tokens.shape == (9, T) and torch.all(w.tokens[:, HALF - 1] == MARKER)
    assert sorted(w.tokens[:, HALF].tolist()) == sorted(list(KEYS_KV) * 3)
    for i in range(9):  # the rest of the window is the source row
        r, p = int(w.row[i]), int(w.pos[i])
        assert torch.equal(w.tokens[i, : HALF - 1], rows[r, p - HALF : p - 1])
    def decode(ids):
        return {1: "~", 2: "ab", 3: " ", 4: "#!"}.get(ids[0], "x")

    assert kv.pick_marker({1: 5}, decode, VOCAB) == 4  # 1 occurs, 2 alphanumeric, 3 whitespace
    dist = kv.key_distributions(lambda toks: torch.zeros(toks.shape[0], T, VOCAB), w, "cpu", batch=4)
    assert set(dist) == set(KEYS_KV) and all(torch.allclose(d, torch.full((VOCAB,), 1 / VOCAB, dtype=torch.float64)) for d in dist.values())
    # the gate: a key only takes a value its predicate admits
    gated = kv.bijection("g", MARKER, KEYS_KV, pool, seed=0, ok=lambda k, v: v >= 12)
    assert all(v >= 12 for v in gated.values) and len(set(gated.values)) == 3
