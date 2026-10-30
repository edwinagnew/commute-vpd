"""CPU checks of E55 Stage C1's filed rules (src/commute/revival.py) on hand-made numbers: the pass bar (the plan's
table, the majority raise), scoring with voids and n < 8, the pruning, reverse fine-tuning and steering labels at
their edges (predictions_c1.json E1-E8); and that pruning a unit zeroes its output, on a tiny random minGPT."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from commute import revival as rv
from commute.mingpt import GPT, GPTConfig


@pytest.mark.parametrize(("n", "a"), [(8, 7), (12, 10), (14, 11), (16, 12), (19, 14), (21, 15)])
def test_pass_bar_reproduces_the_plan_table(n: int, a: int):
    assert rv.pass_bar(n, 0) == a
    assert rv.binom_tail(n, a) <= 0.05 < rv.binom_tail(n, a - 1)


def test_pass_bar_raised_above_the_majority_and_may_be_unreachable():
    assert rv.pass_bar(12, 10) == 11  # binomial 10, majority 10 -> 11
    assert rv.pass_bar(9, 8) == 9
    assert rv.pass_bar(5, 5) == 6  # above n = 5: unreachable


def test_score_drops_voids_and_has_no_bar_below_8():
    truth = {f"s{i}": "existing" if i < 4 else "new" for i in range(10)}
    labels = dict(truth) | {"s0": "void", "s1": "void", "s2": "void"}
    sc = rv.score(labels, truth)
    assert (sc["n"], sc["agree"], sc["bar"], sc["passes"]) == (7, 7, None, None)
    assert sc["voided"] == ["s0", "s1", "s2"]
    assert rv.score({}, truth)["n"] == 0 and rv.score({}, {})["p_value"] == 1.0


def test_score_constant_labeller_fails():
    truth = {f"s{i}": "new" if i < 11 else "existing" for i in range(19)}
    sc = rv.score(dict.fromkeys(truth, "new"), truth)
    assert sc["agree"] == 11 and sc["bar"] == 14 and sc["passes"] is False


def prune(acc0: float, top: float, random: list[float]) -> dict:
    return {"acc0": acc0, "n_eval": 4000, **{str(k): {"top": top, "random": random} for k in rv.PRUNE_K}}


def test_prune_label():
    rand = [0.20, 0.22, 0.21, 0.19, 0.20]  # sd 0.0114 > binomial sd at 0.204, 4000 rows (0.0064)
    sd = float(np.std(rand, ddof=1))
    assert rv.prune_label(prune(0.2, 0.204 + 3 * sd + 1e-9, rand)) == "existing"
    assert rv.prune_label(prune(0.2, 0.204 + 3 * sd - 1e-6, rand)) == "new"
    assert rv.prune_label(prune(0.95, 0.99, rand)) == "void"  # nothing to revive (E6)


def test_prune_sd_floored_at_binomial():
    _, sd = rv.revival_z(0.5, [0.2] * 20, 4000)
    assert sd == pytest.approx(np.sqrt(0.2 * 0.8 / 4000))


def test_prune_control_needs_a_drop():
    rand = [0.999] * 20
    assert rv.prune_control(prune(0.999, 0.90, rand))
    assert not rv.prune_control(prune(0.999, 0.999, rand))


@pytest.mark.parametrize(("steps", "end", "scratch", "want"), [
    (2500, "target", 5000, "existing"),  # inclusive at 0.5 x
    (2550, "target", 5000, "new"),
    (5000, "target", None, "existing"),  # scratch capped -> 10,000 (E5)
    (None, "cap", None, "new"),  # both capped
    (300, "target", 1000, "void"),  # scratch <= 1K voids the lr (E1)
    (0, "target", 5000, "void"),  # starts at target (E6)
    (None, "deadline", 5000, "void"),  # never a reading (E7)
])  # fmt: skip
def test_reverse_label(steps: int | None, end: str, scratch: int | None, want: str):
    assert rv.reverse_label(steps, end, scratch) == want


def test_steer_label():
    assert rv.steer_label({"G_fit": 1.0, "G_score": 1.0, "S": 0.5}) == "existing"
    assert rv.steer_label({"G_fit": 1.0, "G_score": 1.0, "S": 0.49}) == "new"
    assert rv.steer_label({"G_fit": 1.0, "G_score": 0.0, "S": float("nan")}) == "void"  # E8
    assert np.isnan(rv.gain_share(0.3, 0.0)) and rv.gain_share(0.3, 0.6) == 0.5


def test_pruned_unit_outputs_zero():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab=11, block=8, n_layer=2, n_head=2, d=8, dropout=0.0))
    width = rv.fc_layers(model)[0].out_features
    m = rv.pruned(model, np.array([width + 3]))  # block 1, unit 3
    seen = {}
    m.blocks[1].mlp[1].register_forward_hook(lambda _m, _i, o: seen.setdefault("act", o))  # type: ignore[index]
    m(torch.randint(0, 11, (2, 8)))
    assert torch.all(seen["act"][..., 3] == 0) and not torch.all(seen["act"][..., 2] == 0)
    assert torch.all(rv.fc_layers(model)[1].weight[3] != 0)  # the original is untouched
