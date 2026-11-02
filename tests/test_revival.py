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


def test_score_needs_a_margin_over_the_better_nuisance_labeller():
    truth = {f"s{i}": "existing" if i < 8 else "new" for i in range(19)}
    wrong = ["s0", "s8"]  # 17 of 19 right: above the bar of 14
    labels = {s: ("new" if v == "existing" else "existing") if s in wrong else v for s, v in truth.items()}
    lab15 = dict(labels) | {"s1": "new", "s9": "existing"}  # a nuisance labeller at 15 of 19
    sc = rv.score(labels, truth, {"lr": lab15, "short": dict.fromkeys(truth, "new")}, rv.MARGIN_ALL)
    assert (sc["agree"], sc["nuisance_agree"], sc["passes"]) == (17, {"lr": 15, "short": 11}, True)  # 17 >= 15 + 2
    lab16 = dict(labels) | {"s1": "new"}
    sc = rv.score(labels, truth, {"lr": lab16}, rv.MARGIN_ALL)
    assert sc["passes_bar"] and not sc["beats_nuisance"] and sc["passes"] is False  # 17 < 16 + 2


def test_nuisance_counted_on_the_instrument_states_only():
    truth = {f"s{i}": "existing" if i < 6 else "new" for i in range(12)}
    labels = dict(truth) | {"s0": "void", "s1": "void"}  # n = 10, agree 10
    sc = rv.score(labels, truth, {"copy": dict(truth)}, rv.MARGIN_RUN)
    assert sc["nuisance_agree"] == {"copy": 10} and sc["passes_bar"] and sc["passes"] is False  # 10 < 10 + 1


# ---- C1b rules (predictions_c1b.json edge cases E1-E9) ----


def cut(xs_existing: list, xs_new: list, high: bool = True) -> dict:
    r = {f"e{i}": x for i, x in enumerate(xs_existing)} | {f"n{i}": x for i, x in enumerate(xs_new)}
    lab = {s: "existing" if s[0] == "e" else "new" for s in r}
    return rv.cutoff(r, lab, high)


def test_cutoff_separated_takes_the_gap_midpoint():
    c = cut([0.5, 0.7, 0.9], [0.0, 0.1, 0.2])
    assert c["separated"] and c["c"] == pytest.approx(0.35) and c["balanced_accuracy"] == 1.0
    r = cut([100, 200, 300], [900, 1000, 1200], high=False)  # reverse: low side existing
    assert r["separated"] and r["c"] == pytest.approx(600) and rv.label_by_cutoff(600, r) == "existing"
    assert rv.label_by_cutoff(601, r) == "new" and rv.label_by_cutoff(0.35, c) == "existing"  # inclusive at c


def test_cutoff_overlap_maximises_balanced_accuracy_ties_to_median():
    c = cut([0.3, 0.6, 0.8], [0.1, 0.4, 0.5])  # overlap; BA 5/6 at 0.55 only
    assert not c["separated"] and c["balanced_accuracy"] == pytest.approx(5 / 6) and c["c"] == pytest.approx(0.55)
    low = cut([1, 2, 5], [0, 3, 4])  # BA 2/3 at 0.5 and 4.5, both 2 from the median 2.5 -> the lower
    assert low["c"] == pytest.approx(0.5) and low["balanced_accuracy"] == pytest.approx(2 / 3)
    near = cut([1, 2, 4.5], [0, 3, 4])  # BA 2/3 at 0.5 and 4.25; median 2.5 -> 4.25 (1.75 away vs 2)
    assert near["c"] == pytest.approx(4.25)
    one = cut([0, 1, 3], [-1, 2, 4])  # BA 2/3 at -0.5 only
    assert one["c"] == pytest.approx(-0.5)


def test_cutoff_void_cases_and_dropped_readings():
    assert cut([0.5, 0.7], [0.0, 0.1, 0.2])["void"].startswith("fewer than 3")
    assert cut([0.5, 0.7, None], [0.0, 0.1, 0.2])["void"].startswith("fewer than 3")  # a void is dropped, not imputed
    assert cut([0.5, 0.7, float("nan"), 0.8], [0.0, 0.1, 0.2])["dropped"] == ["e2"]
    assert cut([0.3, 0.3, 0.3], [0.3, 0.3, 0.3])["void"] == "every reading equal"
    assert cut([], [])["void"].startswith("fewer than 3")
    v = cut([0.5, 0.7], [0.0, 0.1, 0.2])
    assert rv.label_by_cutoff(0.9, v) == "void" and rv.label_by_cutoff(None, cut([1, 2, 3], [0, -1, -2])) == "void"


def test_cutoff_reversed_classes_still_gives_a_number():
    c = cut([0.0, 0.1, 0.2], [0.5, 0.7, 0.9])  # existing all low on a high-is-existing instrument
    assert not c["separated"] and c["balanced_accuracy"] <= 0.5 and c["c"] is not None


def test_ceiling_best_threshold_direction_and_constants():
    truth = {"a": "existing", "b": "existing", "c": "new", "d": "new"}
    lr = {"a": 1e-5, "b": 1e-5, "c": 1e-4, "d": 1e-4}
    best = rv.ceiling(lr, truth)
    assert best["agree"] == 4 and best["small_is_existing"] and best["threshold"] == pytest.approx(5.5e-5)
    flipped = rv.ceiling({k: -v for k, v in lr.items()}, truth)  # large -> existing found as well
    assert flipped["agree"] == 4 and not flipped["small_is_existing"]
    same = rv.ceiling(dict.fromkeys(truth, 1.0), truth)  # one value: the constants only
    assert same["agree"] == 2
    lop = {"a": "existing", "b": "new", "c": "new"}
    assert rv.ceiling({"a": 1.0, "b": 1.0, "c": 1.0}, lop)["agree"] == 2  # constant 'new' = the majority
    assert rv.ceiling({}, {})["agree"] == 0


def test_ceiling_balanced_cells_sit_near_half():
    truth, vals = {}, {}
    for i, (lr, task) in enumerate([(1e-4, "P"), (1e-4, "N"), (1e-5, "P"), (1e-5, "N")] * 3):
        truth[f"s{i}"], vals[f"s{i}"] = ("existing" if task == "P" else "new"), lr
    assert rv.ceiling(vals, truth)["agree"] == 6  # 12 states, each lr holds both labels


@pytest.mark.parametrize(("n", "a"), [(24, 17), (21, 15), (18, 13), (16, 12)])
def test_pass_bar_reproduces_the_c1b_table(n: int, a: int):
    assert rv.pass_bar(n, n // 2) == a
