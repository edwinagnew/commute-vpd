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
