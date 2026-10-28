"""CPU checks of E38 Amendment 5's parts (commute.recruitment): the first-order take-back e_c (formula, batches, and
against single restores on a toy model), the per-matrix top sets with the filed tie and short rules, S's
top-direction share, control R6's pieces and rule, and P13 with the edge cases listed in predictions_amend5.json."""

from __future__ import annotations

import math

import torch
from test_recruitment import _toy

from commute import recruit as R
from commute import recruitment as RT
from commute.faithfulness import model_state


def test_take_back_formula_and_batches():
    """e_c = mean over WINDOWS of ((x1 - x0) . v_c) (g . u_c), fp32 from bf16 inputs; 0 when the two passes agree."""
    torch.manual_seed(0)
    U, V = torch.randn(5, 3), torch.randn(4, 5)  # (C, d_out), (d_in, C)
    x0, x1, g = torch.randn(5, 4).bfloat16(), torch.randn(5, 4).bfloat16(), torch.randn(5, 3)
    batches = (0, 2, 4)  # 5 windows in batches of 2, 2, 1
    io0 = {"h.0.a": [(x0[s : s + 2], torch.zeros(0)) for s in batches]}
    io1 = {"h.0.a": [(x1[s : s + 2], torch.zeros(0)) for s in batches]}
    gr = {"h.0.a": [(g[s : s + 2], torch.zeros(0)) for s in batches]}
    e = RT.take_back({"h.0.a": (U, V)}, io0, io1, gr)["h.0.a"]
    want = (((x1.float() - x0.float()) @ V) * (g @ U.T)).mean(0)
    assert e.dtype == torch.float32 and torch.allclose(e, want, atol=1e-6)
    assert RT.take_back({"h.0.a": (U, V)}, io0, io0, gr)["h.0.a"].abs().max() == 0


def test_first_order_matches_single_restores():
    """On the toy model with a small update, e_c (gradients and inputs in batches of 2) equals the actual take-back
    lo(W1) - lo(W1, c restored) (restores in readout's batch) to first order, sign included; the gradient pass leaves
    the model's logits and hooks as they were."""
    model, keys, comps, W0, W1big, tok = _toy()
    W1 = {k: W0[k] + 1e-2 * (W1big[k] - W0[k]) for k in keys}
    half, token = 3, 2
    with model_state(model, target=W0):
        io0, io0_r = RT.trigger_io(model, keys, tok, half, batch=2), RT.trigger_io(model, keys, tok, half)
    with model_state(model, target=W1):
        io1 = RT.trigger_io(model, keys, tok, half, batch=2)
        ref = RT.window_logits(model, tok)
        g = RT.trigger_grads(model, keys, tok, half, token, batch=2)
        assert torch.equal(torch.cat(RT.window_logits(model, tok)), torch.cat(ref))  # hooks gone
        lo1 = RT.readout(model, tok, half, token, ref)["lo"]
    e = RT.take_back(comps, io0, io1, g)
    assert e["h.0.a"].abs().max() == 0  # h.0.a's input (the embedding) does not change with W
    a = torch.tensor([lo1 - RT.restored_at(model, W1, comps, RT.sel_of([("h.1.a", c)], keys), io0_r, tok, half, token,
                                           ref)["lo"] for c in range(6)])
    assert a.abs().max() > 1e-4  # the restores bite
    assert (e["h.1.a"].double() - a.double()).abs().max() <= 0.02 * a.abs().max()  # first order, sign included


def _scores():
    alive = {"h.0.x": torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool), "h.0.y": torch.ones(3, dtype=torch.bool),
             "h.1.x": torch.tensor([1, 1, 0], dtype=torch.bool)}
    score = {"h.0.x": torch.tensor([5.0, 2.0, 2.0, -1.0, 9.0]), "h.0.y": torch.tensor([2.0, -0.5, 3.0]),
             "h.1.x": torch.tensor([1.0, -4.0, 7.0])}
    return alive, score


def test_top_per_matrix_ties_short_and_empty():
    alive, score = _scores()
    S = {"h.0.x": torch.tensor([0, 3]), "h.0.y": torch.zeros(0, dtype=torch.long), "h.1.x": torch.tensor([0, 1])}
    counts = {k: int(v.numel()) for k, v in S.items()}
    D, short = RT.top_per_matrix(score, alive, S, counts)
    # h.0.x: dead [4] (9.0) and S never picked; tie at 2.0 between [1] and [2] goes to the lower index
    assert D["h.0.x"].tolist() == [1, 2] and short["h.0.x"] is False
    assert D["h.0.y"].tolist() == [] and short["h.0.y"] is False  # S has none here: empty, restores 0 (R0)
    assert D["h.1.x"].tolist() == [] and short["h.1.x"] is True  # pool (alive, not S) is empty, S has 2: short
    assert set(D) == set(alive)
    D1, _ = RT.top_per_matrix(score, alive, {"h.0.x": torch.tensor([0])}, {"h.0.x": 1})
    assert D1["h.0.x"].tolist() == [1] and R.size(D1) == 1  # tie at 2.0: index 1 before 2
    Dn, _ = RT.top_per_matrix(score, alive, {}, {"h.0.y": 2})
    assert Dn["h.0.y"].tolist() == [0, 2]  # signed: the -0.5 piece ranks last


def test_top_dir_share_and_top_abs():
    alive, score = _scores()
    S = {"h.0.x": torch.tensor([0, 3]), "h.0.y": torch.zeros(0, dtype=torch.long), "h.1.x": torch.tensor([1])}
    # h.0.x top 2 (S included, dead [4] out): [0] 5.0, then [1] 2.0 (tie with [2], lower index) -> 1 hit;
    # h.1.x top 1: [0] 1.0 beats S's [1] -4.0 -> 0 hits
    share, hits = RT.top_dir_share(score, alive, S)
    assert hits == {"h.0.x": 1, "h.1.x": 0} and share == 1 / 3
    assert math.isnan(RT.top_dir_share(score, alive, {"h.0.x": torch.zeros(0, dtype=torch.long)})[0])
    top = RT.top_abs(score, alive, 4)
    assert top == [("h.0.x", 0, 5.0), ("h.1.x", 1, -4.0), ("h.0.y", 2, 3.0), ("h.0.x", 1, 2.0)]  # dead 9.0 out
    assert RT.summed(score, {"h.0.x": torch.tensor([0, 3]), "h.1.x": torch.tensor([1])}) == 0.0


def test_r6_agree():
    e = [0.5, -0.3, 0.019, 0.1, 0.2]
    a = [0.4, -0.1, 0.019, -0.1, 0.0]  # agree, agree, |e| < 0.02, wrong sign, a = 0
    assert RT.r6_agree(e, a, 0.02) == 2


def _p(S: float, D: float, scorable: bool = True, short: bool = False) -> dict:
    return {"scorable": scorable, "D_short": short, "rho": {"S": S, "D": D}}


def test_p13_seed_and_task_rules():
    nan = float("nan")
    assert RT.p13_seed_class(_p(0.26, 0.20)) == "S ahead"  # rho_S >= rho_D + 0.05
    assert RT.p13_seed_class(_p(0.25, 0.25)) == "not ahead"  # a tie reads 'not ahead'
    assert RT.p13_seed_class(_p(0.2, 0.3)) == "not ahead"
    assert RT.p13_seed_class(_p(0.25, 0.21)) == "unclear"
    assert RT.p13_seed_class(_p(nan, nan, scorable=False, short=True)) == RT.UNSCORED_GAIN  # G < 1 caught first
    assert RT.p13_seed_class(_p(0.3, 0.0, short=True)) == RT.UNSCORED_SHORT
    assert RT.p13_task_class([_p(0.3, 0.0, short=True), _p(nan, nan, scorable=False)]) == RT.UNSCORED_GAIN  # sorted
    assert RT.p13_task_class([_p(0.3, 0.0, short=True), _p(0.3, 0.0)]) == RT.UNSCORED_SHORT
    assert RT.p13_task_class([_p(0.3, 0.0), _p(0.09, 0.0)]) == RT.TOO_SMALL
    assert RT.p13_task_class([_p(0.0, 0.0), _p(0.0, 0.0)]) == RT.TOO_SMALL  # |S| = 0: rho_S = 0, D empty
    assert RT.p13_task_class([_p(0.3, 0.0), _p(0.3, 0.3)]) == "seeds disagree"
    assert RT.p13_task_class([_p(0.3, 0.28), _p(0.3, 0.27)]) == "unclear"


def test_p13_reading():
    ahead, nots = {"0": _p(0.24, 0.10), "1": _p(0.22, 0.12)}, {"0": _p(0.18, 0.30), "1": _p(0.20, 0.20)}
    we = {"0": _p(0.037, 0.2), "1": _p(0.049, 0.2)}  # ' we': S too small, counts toward no branch
    out = RT.p13_reading({"emoticon": ahead, "which": ahead, "we": we})
    assert out["reading"] == "Gamma adds" and out["task_class"]["we"] == RT.TOO_SMALL and out["n_scored_tasks"] == 2
    assert RT.p13_reading({"emoticon": nots, "which": nots, "we": we})["reading"] == "direction suffices"
    assert RT.p13_reading({"emoticon": ahead, "which": nots, "we": ahead})["reading"] == "outside the tree"  # mixed
    mid = {"0": _p(0.24, 0.21), "1": _p(0.22, 0.20)}
    out = RT.p13_reading({"emoticon": ahead, "which": ahead, "we": mid})
    assert out["task_class"]["we"] == "unclear" and out["reading"] == "Gamma adds"  # 'unclear' blocks neither branch
    one = RT.p13_reading({"emoticon": ahead, "which": {**ahead, "1": _p(0.3, 0.0, short=True)}, "we": we})
    assert one["task_class"]["which"] == RT.UNSCORED_SHORT and one["n_scored_tasks"] == 1
    assert one["reading"] == "outside the tree"  # fewer than two scored tasks
    assert one["seed_class"]["which/s1"] == RT.UNSCORED_SHORT and one["seed_class"]["we/s0"] == "not ahead"
