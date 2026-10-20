"""CPU checks of src/commute/signatures.py on the tiny ComponentModel of test_faithfulness_invariants: the per-
subcomponent lift energy against a direct computation, the top-k selection and its summaries, the CI-matched random
sets, the ablation by weights, and the Pile gate statistics."""

from __future__ import annotations

import torch
from test_faithfulness_invariants import KEYS, VOCAB, make_batches, make_model

from commute import component_training, split, vpd
from commute import signatures as S
from commute.recruit import jaccard, members

HALF = 20


def alive_masks(model, dead: tuple[str, int] | None = None) -> dict[str, torch.Tensor]:
    al = {k: torch.ones(model.module_to_c[k], dtype=torch.bool) for k in KEYS}
    if dead:
        al[dead[0]][dead[1]] = False
    return al


def test_lift_energy_matches_direct_per_column_and_zeros_dead_and_untouched():
    model = make_model()
    alive = alive_masks(model, dead=("fc1", 3))
    uv = {k: (u.cpu(), v.cpu()) for k, (u, v) in component_training.alive_uv(model, alive).items()}
    bases = split.bases(uv)
    g = torch.Generator().manual_seed(0)
    dW = {"fc1": torch.randn(8, 8, generator=g) * 1e-2, "fc2": torch.zeros(8, 8)}
    e = S.split_energy(dW, uv, bases, alive)
    assert torch.equal(e["fc2"], torch.zeros(8, dtype=torch.float64)) and e["fc1"][3] == 0
    U, V = uv["fc1"][0].double(), uv["fc1"][1].double()
    dU, dV = split.least_squares_split(dW["fc1"].double(), bases["fc1"], U, V)
    direct = torch.stack([(torch.outer(dU[:, c], V[:, c]) + torch.outer(U[:, c], dV[:, c])).pow(2).sum() for c in range(7)])
    assert torch.allclose(e["fc1"][alive["fc1"]], direct, rtol=1e-9, atol=1e-14)


def test_top_by_share_and_mean_skip_dead_and_sort():
    alive = {"a": torch.tensor([True, True, False, True]), "b": torch.tensor([True, True])}
    vals = {"a": torch.tensor([1.0, 5.0, 9.0, 2.0]), "b": torch.tensor([4.0, 0.5])}
    sel = S.top_by(vals, alive, 3)
    assert sel["a"].tolist() == [1, 3] and sel["b"].tolist() == [0]  # 9.0 is dead; top three alive are 5, 4, 2
    assert abs(S.share_in(vals, sel) - 11.0 / 21.5) < 1e-12 and abs(S.mean_over(vals, sel) - 11.0 / 3) < 1e-12
    assert S.top_by(vals, alive, 10)["a"].tolist() == [0, 1, 3]  # k above the alive count takes every alive one


def test_matched_random_same_size_disjoint_alive_and_same_decile():
    g = torch.Generator().manual_seed(1)
    alive = {"a": torch.ones(100, dtype=torch.bool), "b": torch.ones(100, dtype=torch.bool)}
    alive["a"][:10] = False
    gval = {k: torch.rand(100, generator=g) for k in alive}
    sel = {"a": torch.tensor([10, 11, 50, 99]), "b": torch.zeros(0, dtype=torch.long)}
    r1, r2 = S.matched_random(sel, alive, gval, seed=0), S.matched_random(sel, alive, gval, seed=1)
    assert r1["a"].numel() == 4 and r1["b"].numel() == 0 and members(r1) != members(r2)
    assert not (members(r1) & members(sel)) and all(alive["a"][c] for c in r1["a"].tolist())
    al = alive["a"].nonzero().flatten()
    rank = torch.empty(100, dtype=torch.long)
    rank[al[gval["a"][al].argsort()]] = torch.arange(90)
    dec = lambda c: int(rank[c]) * 10 // 90
    assert sorted(dec(c) for c in r1["a"].tolist()) == sorted(dec(c) for c in sel["a"].tolist())


def test_ablated_weights_and_trigger_stats():
    model = make_model()
    W = vpd.target_weights(model)
    comps = vpd.component_weights(model)
    alive = alive_masks(model)
    all_sel = {k: alive[k].nonzero().flatten() for k in KEYS}
    abl = S.ablated_weights(W, comps, all_sel)
    for k in KEYS:
        assert torch.allclose(abl[k], W[k] - model.components[k].weight.detach(), atol=1e-6)
    assert torch.equal(S.ablated_weights(W, comps, {k: torch.zeros(0, dtype=torch.long) for k in KEYS})["fc1"], W["fc1"])
    one = S.ablated_weights(W, comps, {"fc1": torch.tensor([2])})
    assert torch.allclose(one["fc1"], W["fc1"] - torch.outer(comps["fc1"][0][2], comps["fc1"][1][:, 2]), atol=1e-6)
    toks = torch.randint(0, VOCAB, (6, 2 * HALF + 1), generator=torch.Generator().manual_seed(2))
    ref = S.trigger_logits(model, toks, HALF, batch=4)
    assert ref.shape == (6, VOCAB) and torch.allclose(ref, model.target_model(toks)[:, HALF], atol=1e-6)
    same = S.trigger_stats(ref, ref, toks, HALF, 3)
    assert same["drop"] == 0 and abs(same["kl"]) < 1e-6
    table = torch.full((VOCAB,), 3, dtype=torch.long)
    assert S.trigger_stats(ref, ref * 0.5, toks, HALF, table) == S.trigger_stats(ref, ref * 0.5, toks, HALF, 3)


def test_pile_gate_stats_on_rate_and_repeat_mask():
    model = make_model()
    alive = alive_masks(model)
    bs = make_batches(2, 2)
    st = S.pile_gate_stats(model, bs, alive, on=0.5)
    assert st["n_positions"] == 4 * vpd.SEQ_LEN and 0 <= st["n_repeat"] <= st["n_positions"]
    ci = [vpd.compute_ci(model, b) for b in bs]
    for k in KEYS:
        rate = torch.cat([c.g[k].reshape(-1, c.g[k].shape[-1]) for c in ci]).gt(0.5).float().mean(0)
        assert torch.allclose(st["on_rate"][k], rate, atol=1e-6) and st["repeat_contrast"][k].shape == rate.shape
    sel = S.top_by(st["on_rate"], alive, 3)
    assert sum(v.numel() for v in sel.values()) == 3 and 0 <= jaccard(sel, sel) <= 1
