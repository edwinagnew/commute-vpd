"""CPU checks of recruit.py on the tiny ComponentModel from test_faithfulness_invariants (fc1 -> relu -> fc2, both
decomposed). fc1's gates read the embedding, so no weight change can move them; fc2's gates read relu(fc1 x), so a
change to fc1 moves them. That is the same locality the LoRA arm relies on (a layer-2 edit can only re-gate layer 3+)."""

from __future__ import annotations

import math

import pytest
import torch
from test_faithfulness_invariants import KEYS, VOCAB, make_model

from commute import local_faith as LF
from commute import recruit as R
from commute.faithfulness import model_state
from commute.vpd import target_weights

HALF, T, TOKEN = 20, 41, 3


@pytest.fixture(scope="module")
def model():
    return make_model()


def windows(n: int = 6, seed: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, VOCAB, (n, T), generator=g)
    w[:, HALF] = 5
    return w


def alive_all(model) -> dict[str, torch.Tensor]:
    return {k: torch.ones(model.module_to_c[k], dtype=torch.bool) for k in KEYS}


def test_gates_move_only_downstream_of_the_edit(model):
    w = windows()
    W0 = target_weights(model)
    base = LF.route_cells(model, w, HALF, TOKEN, batch_size=4)
    gen = torch.Generator().manual_seed(0)
    W1 = {k: v.clone() for k, v in W0.items()}
    W1["fc1"] = W1["fc1"] + 0.5 * torch.randn(W1["fc1"].shape, generator=gen)
    with model_state(model, target=W1):
        cur = LF.route_cells(model, w, HALF, TOKEN, batch_size=4)
    dg = R.trigger_dg(cur["gates"], base["gates"], HALF, alive_all(model))
    assert torch.equal(dg["fc1"], torch.zeros_like(dg["fc1"])), "fc1 reads the embedding: its gates cannot move"
    assert dg["fc2"].abs().max() > 0, "fc2 reads fc1's output: its gates must move"
    # dead components never count
    alive = alive_all(model)
    alive["fc2"][:] = False
    assert R.size(R.switched(R.trigger_dg(cur["gates"], base["gates"], HALF, alive), 0.0)) == 0


def test_hybrid_endpoints(model):
    w = windows()
    W0 = target_weights(model)
    base = LF.route_cells(model, w, HALF, TOKEN, batch_size=4)
    gen = torch.Generator().manual_seed(1)
    W1 = {k: v + 0.5 * torch.randn(v.shape, generator=gen) for k, v in W0.items()}
    with model_state(model, target=W1):
        cur = LF.route_cells(model, w, HALF, TOKEN, batch_size=4)
    none = R.hybrid(base["gates"], cur["gates"], {})
    everything = R.hybrid(base["gates"], cur["gates"], {k: torch.arange(model.module_to_c[k]) for k in KEYS})
    for b in range(len(base["gates"])):
        for k in KEYS:
            assert torch.equal(none[b][k], base["gates"][b][k])
            assert torch.equal(everything[b][k], cur["gates"][b][k])
    # the masked model with Delta out depends only on the gates (components unchanged), so the recovery measure
    # reaches exactly the state's cell with the full selection and exactly the base cell with none
    lo_none, _ = LF.masked_lo(model, w, HALF, TOKEN, none, batch_size=4)
    lo_all, _ = LF.masked_lo(model, w, HALF, TOKEN, everything, batch_size=4)
    assert math.isclose(lo_none, base["L"], rel_tol=1e-5, abs_tol=1e-5)
    with model_state(model, target=W1):
        assert math.isclose(lo_all, cur["L"], rel_tol=1e-5, abs_tol=1e-5)


def test_switched_jaccard_layers_random():
    dg = {"h.2.mlp.down_proj": torch.tensor([0.5, -0.5, 0.0, 0.31]), "h.3.attn.o_proj": torch.tensor([0.2, 0.9])}
    on = R.switched(dg, 0.3)
    off = R.switched(dg, 0.3, on=False)
    assert on["h.2.mlp.down_proj"].tolist() == [0, 3] and on["h.3.attn.o_proj"].tolist() == [1]
    assert off["h.2.mlp.down_proj"].tolist() == [1] and off["h.3.attn.o_proj"].numel() == 0
    assert R.size(on) == 3 and R.by_layer(on) == {"2": 2, "3": 1}
    assert R.restrict(on, 3) == {"h.3.attn.o_proj": on["h.3.attn.o_proj"]}
    assert R.jaccard(on, on) == 1.0
    assert math.isclose(R.jaccard(on, R.switched(dg, 0.1)), 3 / 4)  # 0.1 adds (h.3, 0)
    assert math.isnan(R.jaccard({}, {"x": torch.zeros(0, dtype=torch.long)}))
    assert R.jaccard(on, off) == 0.0
    alive = {"h.2.mlp.down_proj": torch.tensor([True, True, True, True]), "h.3.attn.o_proj": torch.tensor([True, True])}
    rnd = R.random_like(on, alive, seed=0)
    assert {k: v.numel() for k, v in rnd.items()} == {"h.2.mlp.down_proj": 2, "h.3.attn.o_proj": 1}
    assert not (R.members(rnd) & R.members(on))
    assert R.jaccard(rnd, rnd) == 1.0 and torch.equal(R.random_like(on, alive, 0)["h.3.attn.o_proj"], rnd["h.3.attn.o_proj"])


def test_precede_mask_and_contrast():
    rows = torch.tensor([[1, 7, 2, 7], [7, 7, 0, 1]])
    m = R.precede_mask(rows, 7)
    assert m.tolist() == [[True, False, True, False], [True, False, False, False]]
    g = {"a": torch.rand(2, 4, 3)}
    acc = R.ContrastAcc()
    acc.add({"a": g["a"][:1]}, m[:1])
    acc.add({"a": g["a"][1:]}, m[1:])
    alive = {"a": torch.tensor([True, True, False])}
    res = acc.result(alive)["a"]
    expect = g["a"][m].mean(0) - g["a"].reshape(-1, 3).mean(0)
    assert torch.allclose(res[:2], expect[:2], atol=1e-6) and res[2] == 0
    assert acc.n_on == 3 and acc.n_all == 8


def test_trigger_patterns():
    """E18's trigger-only patterns: identities at W0, the clip, the constant, and the moved null."""
    g0 = [{"a": torch.rand(2, 5, 3), "b": torch.rand(2, 5, 2)}, {"a": torch.rand(1, 5, 3), "b": torch.rand(1, 5, 2)}]
    g1 = [{k: torch.rand_like(v) for k, v in g.items()} for g in g0]
    h = 2
    same = R.with_trigger(g0, h, R.at_trigger(g0, h))
    zero = R.with_trigger(g0, h, R.shifted(g0, h, {"a": torch.zeros(3), "b": torch.zeros(2)}))
    swap = R.with_trigger(g0, h, R.at_trigger(g1, h))
    for b in range(2):
        for k in ("a", "b"):
            assert torch.equal(same[b][k], g0[b][k]) and torch.equal(zero[b][k], g0[b][k])
            keep = torch.arange(5) != h
            assert torch.equal(swap[b][k][:, keep], g0[b][k][:, keep]) and torch.equal(swap[b][k][:, h], g1[b][k][:, h])
    big = R.with_trigger(g0, h, R.shifted(g0, h, {"a": torch.tensor([2.0, -2.0, 0.0]), "b": torch.zeros(2)}))
    assert torch.all(big[0]["a"][:, h, 0] == 1) and torch.all(big[0]["a"][:, h, 1] == 0)
    assert torch.equal(big[0]["a"][:, h, 2], g0[0]["a"][:, h, 2])
    vec = {"a": torch.tensor([0.1, 0.2, 0.3]), "b": torch.tensor([0.4, 0.5])}
    const = R.with_trigger(g0, h, R.constant(g0, h, vec))
    assert all(torch.allclose(const[b]["a"][i, h], vec["a"]) for b in range(2) for i in range(const[b]["a"].shape[0]))
    alive = {"a": torch.tensor([True, True, False]), "b": torch.tensor([True, True])}
    d = R.mean_change({"a": torch.tensor([0.9, 0.1, 0.7]), "b": torch.tensor([0.5, 0.5])},
                      {"a": torch.tensor([0.1, 0.1, 0.1]), "b": torch.tensor([0.0, 0.5])}, alive)
    assert torch.allclose(d["a"], torch.tensor([0.8, 0.0, 0.0])) and torch.allclose(d["b"], torch.tensor([0.5, 0.0]))
    on = R.switched(d, 0.3)
    mv = R.moved(d, on, {"a": torch.tensor([1]), "b": torch.tensor([1])})
    assert torch.allclose(mv["a"], torch.tensor([0.0, 0.8, 0.0])) and torch.allclose(mv["b"], torch.tensor([0.0, 0.5]))
    assert sorted(mv) == ["a", "b"] and R.moved(d, on, {})["a"].abs().sum() == 0



def test_added_is_shifted_before_the_clip():
    """E18 amendment 1: `added` keeps out-of-range values, `shifted` is its clip, and moved(shift, on, on) keeps the
    shift on S_on only."""
    g0 = [{"a": torch.tensor([[[0.2, 0.9, 0.0], [0.5, 0.1, 0.8]]])}]  # (B=1, T=2, C=3)
    shift = {"a": torch.tensor([0.5, 0.4, -0.3])}
    raw, cl = R.added(g0, 1, shift)[0]["a"], R.shifted(g0, 1, shift)[0]["a"]
    assert torch.allclose(raw, torch.tensor([[1.0, 0.5, 0.5]])) and torch.allclose(cl, raw)
    raw0 = R.added(g0, 0, shift)[0]["a"]
    assert torch.allclose(raw0, torch.tensor([[0.7, 1.3, -0.3]]))
    assert torch.allclose(R.shifted(g0, 0, shift)[0]["a"], torch.tensor([[0.7, 1.0, 0.0]]))
    assert torch.equal(R.added(g0, 0, {"a": torch.zeros(3)})[0]["a"], g0[0]["a"][:, 0])
    on = R.switched(shift, 0.3)
    assert torch.allclose(R.moved(shift, on, on)["a"], torch.tensor([0.5, 0.4, 0.0]))

def test_trigger_mean_gates_matches_route_cells(model):
    w = windows(n=7)
    cells = LF.route_cells(model, w, HALF, TOKEN, batch_size=3)
    mean = LF.trigger_mean_gates(model, w, HALF, batch_size=4)
    for k, v in R.cat_gates(cells["gates"]).items():
        assert torch.allclose(mean[k], v[:, HALF].float().mean(0), atol=1e-6)


def test_shuffled_and_in_range():
    """E18 amendment 2: the shuffle permutes the alive entries only (same values per matrix, dead stay 0, a fixed seed
    repeats), and in_range picks alive entries with lo < s <= hi."""
    shift = {"a": torch.tensor([0.5, -0.2, 0.0, 0.05, 0.3]), "b": torch.tensor([0.4, 0.1])}
    alive = {"a": torch.tensor([True, True, False, True, True]), "b": torch.tensor([True, True])}
    p = R.shuffled(shift, alive, 3)
    assert p["a"][2] == 0 and torch.equal(p["a"][alive["a"]].sort().values, shift["a"][alive["a"]].sort().values)
    assert torch.equal(p["b"].sort().values, shift["b"].sort().values)
    assert all(torch.equal(p[k], R.shuffled(shift, alive, 3)[k]) for k in p)
    assert any(not torch.equal(R.shuffled(shift, alive, i)["a"], shift["a"]) for i in range(5))
    sel = R.in_range(shift, alive, 0.01, 0.3)
    assert sel["a"].tolist() == [3, 4] and sel["b"].tolist() == [1]
    assert R.in_range(shift, alive, -0.01, 0.01)["a"].tolist() == []  # entry 2 is dead


def test_amend3_helpers():
    """E18 amendment 3: plus generalises added/shifted, derangements have no fixed point, permuted hands each window its
    donor's values in the same batch layout, signed and on_pool split the shift."""
    g0 = [{"a": torch.rand(3, 4, 5)}, {"a": torch.rand(2, 4, 5)}]
    shift = {"a": torch.tensor([0.5, -0.4, 0.0, 0.2, -0.1])}
    t = R.at_trigger(g0, 1)
    assert all(torch.equal(x["a"], y["a"]) for x, y in zip(R.plus(t, shift, False), R.added(g0, 1, shift), strict=True))
    assert all(torch.equal(x["a"], y["a"]) for x, y in zip(R.plus(t, shift, True), R.shifted(g0, 1, shift), strict=True))
    for seed in range(5):
        p = R.derangement(5, seed)
        assert sorted(p.tolist()) == list(range(5)) and not bool((p == torch.arange(5)).any())
        pt = R.permuted(t, p)
        assert [x["a"].shape[0] for x in pt] == [3, 2]
        assert torch.equal(torch.cat([x["a"] for x in pt]), torch.cat([x["a"] for x in t])[p])
    assert torch.equal(R.signed(shift, True)["a"], torch.tensor([0.5, 0.0, 0.0, 0.2, 0.0]))
    assert torch.equal(R.signed(shift, False)["a"], torch.tensor([0.0, -0.4, 0.0, 0.0, -0.1]))
    assert torch.equal(R.on_pool(shift, {"a": torch.tensor([True, True, False, False, True])})["a"],
                       torch.tensor([0.5, -0.4, 0.0, 0.0, -0.1]))


def test_masked_lo_windows_averages_to_masked_lo(model):
    w = windows(n=7)
    g = LF.route_cells(model, w, HALF, TOKEN, batch_size=3)["gates"]
    per = LF.masked_lo_windows(model, w, HALF, TOKEN, g, batch_size=3)
    assert per.shape == (7,) and float(per.mean()) == pytest.approx(LF.masked_lo(model, w, HALF, TOKEN, g, 3)[0], abs=1e-9)
