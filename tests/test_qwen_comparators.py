"""E92's filed rules on their edge cases: the share's undefined cases, the anchor rule, the answer marks, the size's units."""

from __future__ import annotations

import math

import pytest
import torch

from commute import qwen_comparators as qc


def rows_two_seqs() -> qc.Rows:
    return qc.rows_of("t", [([1, 2, 3, 4], 2), ([5, 6, 7], 1), ([8, 9], 2)], 4, 4)


def test_marks_read_the_next_token_and_cross_a_row() -> None:
    r = qc.rows_of("t", [([1, 2, 3, 4], 2), ([5, 6, 7, 8], 1)], 4, 4)
    assert r.tokens.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    # position 1 predicts token 3, the first answer token; position 3 has no next token in its row
    assert r.first.tolist() == [[False, True, False, False], [True, False, False, False]]
    assert r.answer.tolist() == [[False, True, True, False], [True, True, True, False]]


def test_a_sequence_with_no_answer_marks_nothing_and_cut_answers_drop() -> None:
    r = rows_two_seqs()
    # ([8, 9], 2) has no answer; rows are 4 long, so ([5, 6, 7]) then 8 fill the second, and 9 is cut
    assert r.tokens.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert r.answer.tolist()[1] == [True, True, False, False]


def test_even_pick_and_keep() -> None:
    r = qc.rows_of("t", [(list(range(16)), 0)], 4, 2, even=2)
    assert r.tokens.tolist() == [[0, 1], [8, 9]]


def test_half_is_even_and_odd_rows() -> None:
    r = qc.rows_of("t", [(list(range(16)), 0)], 4, 4)
    assert r.half("fit").tokens[:, 0].tolist() == [0, 8] and r.half("score").tokens[:, 0].tolist() == [4, 12]


def test_token_reads_rank_counts_from_one_with_ties_to_the_best() -> None:
    tokens = torch.tensor([[0, 1]])
    logits = torch.tensor([[[0.0, 3.0, 3.0, 5.0], [0.0, 0.0, 0.0, 0.0]]])
    rank, lp = qc.token_reads(logits, tokens, torch.tensor([[True, False]]))
    assert rank.tolist() == [2] and lp[0] == pytest.approx(torch.log_softmax(logits[0, 0], -1)[1].item())


def test_share_zero_gain_is_undefined_on_the_do_nothing_state() -> None:
    zero, rows = torch.zeros(6), torch.tensor([0, 0, 1, 1, 2, 2])
    out = qc.share(zero, zero, rows)
    assert out["gain"] == 0 and out["share"] is None


def test_share_empty_and_one_row_are_undefined() -> None:
    e = torch.zeros(0)
    assert qc.share(e, e, e.long())["share"] is None
    one = torch.ones(3)
    assert qc.share(one, one, torch.zeros(3).long())["share"] is None


def test_share_noise_gain_is_undefined_and_signal_is_defined() -> None:
    rows = torch.tensor([0, 1, 2, 3])
    assert qc.share(torch.zeros(4), torch.tensor([1.0, -1.0, 1.0, -1.0]), rows)["share"] is None
    got = qc.share(torch.full((4,), 0.5), torch.tensor([1.0, 1.1, 0.9, 1.0]), rows)
    assert got["share"] == pytest.approx(0.5, abs=1e-6) and got["share_se"] == pytest.approx(0.0, abs=1e-6)


def test_share_negative_gain_keeps_a_sign_that_reads_recovered() -> None:
    rows = torch.tensor([0, 1, 2, 3])
    got = qc.share(torch.full((4,), -0.5), torch.tensor([-1.0, -1.1, -0.9, -1.0]), rows)
    assert got["share"] == pytest.approx(0.5, abs=1e-6)


def test_separates_rule_cases() -> None:
    assert qc.separates([1.0, 1.1], [3.0, 3.1]) == "counts"
    assert qc.separates([1.0, 2.0], [2.0, 3.0]) == "does not count"  # gap 1 equals spread 1: strict
    assert qc.separates([0.0, 0.0], [0.0, 0.0]) == "does not count"  # do-nothing beside do-nothing
    assert qc.separates([1.0, None], [3.0, 3.1]) == "unread"  # undefined share at a seed
    assert qc.separates([1.0, math.nan], [3.0, 3.1]) == "unread"
    assert qc.separates([1.0], [3.0, 3.1]) == "unread"  # one seed


def test_update_size_units() -> None:
    dW = {"layers.0.mlp.up_proj": torch.tensor([3.0, 4.0]), "layers.1.mlp.up_proj": torch.zeros(1)}
    W0 = {"layers.0.mlp.up_proj": torch.tensor([30.0, 40.0]), "layers.1.mlp.up_proj": torch.tensor([120.0])}
    out = qc.update_size(dW, W0)
    assert out["total"]["norm"] == pytest.approx(5.0) and out["total"]["w0_norm"] == pytest.approx(130.0)
    assert out["total"]["ratio_norm"] == pytest.approx(5 / 130) and out["total"]["share_squared"] == pytest.approx(25 / 16900)
    assert out["blocks"]["0"]["ratio_norm"] == pytest.approx(0.1) and out["blocks"]["1"]["norm"] == 0


def test_pick_layer_ties_to_the_lower() -> None:
    assert qc.pick_layer([0.1, 0.5, 0.5, 0.2]) == 1


def test_random_direction_has_the_norm_and_lies_in_the_spread() -> None:
    X = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    r = qc.random_direction(X, 3.0, 0)
    assert float(r.norm()) == pytest.approx(3.0) and r[1:].abs().max() == 0
    assert qc.variance_share(X, r) == pytest.approx(1.0)
