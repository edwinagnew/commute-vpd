"""E81's rules against their edge cases (plan, "Predictions"), with no model and no GPU."""

from __future__ import annotations

import pytest

from commute import qwen_tofu as tofu


def test_a_tie_is_wrong_and_a_clear_win_is_right() -> None:
    assert not tofu.correct([-1.0, -1.0, -2.0])
    assert tofu.correct([-1.0, -1.1, -2.0])
    assert not tofu.correct([-3.0, -1.0])


def test_the_data_has_the_filed_shape() -> None:
    assert len(tofu.rows("full")) == 4000 and len(tofu.rows("holdout10")) == 400
    para, real = tofu.held("paraphrased_question"), tofu.real()
    assert len(para) == 800 and {len(c.wrong) for c in para} == {5} and len({c.author for c in para}) == 40
    assert len(real) == 217 and {len(c.wrong) for c in real} == {3}
    trained = {r["question"] for r in tofu.rows("full")}
    assert all(c.question.removesuffix(tofu.ASK) in trained for c in tofu.held("question"))
    assert not {r["question"] for r in tofu.rows("holdout10")} & trained


def test_stream_covers_each_epoch_once_and_is_seeded() -> None:
    seqs = [([i], 0) for i in range(40)]
    one = [s for t in range(1, 11) for s in tofu.stream(seqs, 4, 7, t)]
    assert sorted(x[0][0] for x in one) == list(range(40))
    assert tofu.stream(seqs, 4, 7, 11) != tofu.stream(seqs, 4, 7, 1)
    assert tofu.stream(seqs, 4, 7, 3) == tofu.stream(seqs, 4, 7, 3)


def test_cluster_ci_all_or_none_and_one_cluster() -> None:
    assert tofu.cluster_ci([True] * 4, list("aabb")) == (1.0, 1.0, 1.0)
    assert tofu.cluster_ci([False] * 4, list("aabb")) == (0.0, 0.0, 0.0)
    assert tofu.cluster_ci([True, False], ["a", "a"]) == (0.5, 0.5, 0.5)
    with pytest.raises(AssertionError):
        tofu.cluster_ci([], [])


def test_recall() -> None:
    assert tofu.recall("", "a b") == 0
    assert tofu.recall("The A, b!", "a b") == 1
    assert tofu.recall("a", "a a b b") == 0.25
    with pytest.raises(AssertionError):
        tofu.recall("a", "...")


def test_choice_bar() -> None:
    assert tofu.choice_bar(0.25, 6, 4) is None and tofu.choice_bar(0.1, 6, 4) is None
    assert tofu.choice_bar(1.0, 6, 4) == pytest.approx(1 / 6 + 0.5 * 5 / 6)
    assert tofu.choice_bar(0.625, 6, 4) == pytest.approx(1 / 6 + 0.25 * 5 / 6)


def test_answered_edges() -> None:
    assert tofu.answered(0.6, 0.5, 0.6, 0.75, 0.25)
    assert tofu.answered(0.5, 0.5, 0.5, 0.75, 0.25), "exactly at both bars"
    assert not tofu.answered(0.49, 0.5, 0.6, 0.75, 0.25)
    assert not tofu.answered(0.6, 0.5, 0.49, 0.75, 0.25)
    assert not tofu.answered(0.6, None, 0.6, 0.75, 0.25), "no bar"
    assert not tofu.answered(0.6, 0.5, 0.3, 0.3, 0.3), "the trained wording did not beat the floor"
