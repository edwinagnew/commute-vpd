"""E107's filed rules (experiments/label_vs_style/score.py) on their edge cases, the topic split and the wrong-answer
substitution (commute.known_facts `swap`). No model is built."""

from __future__ import annotations

import math
from types import ModuleType

import pytest

from commute import known_facts as KF
from commute.jobs import load_experiment

U = 1 / 1024  # one neuron of a top-1024 set: overlaps are exact in binary
X, L, WK = 0.63671875, 0.875, 0.9013671875  # the TOFU probe's filed x, L and known seeds' overlap
FAM, STY = 753 * U, 795 * U  # the lines at w = WK: min(x + band, L - band) and L - band
FULL = dict.fromkeys(("WRONG42", "WRONG142", "KNOWN42", "KNOWN142"), 1024)
TOPIC = dict.fromkeys(("AUTH42", "AUTH142", "WORLD42", "WORLD142"), 1024)
TL = L - (1 - 0.9052734375)  # the topic line at E105's TOFU w


@pytest.fixture(scope="module")
def S() -> ModuleType:
    return load_experiment("label_vs_style", "score")


def test_lines(S: ModuleType) -> None:
    v = S.wrong_verdict(0.95, WK, [0.6] * 4, X, L, FULL)
    assert v["familiarity_needs_max"] == FAM and v["style_needs_min"] == STY and v["band"] == 1 - WK


@pytest.mark.parametrize(("w_wrong", "cross", "sizes", "verdict"), [
    (1, [1] * 4, FULL, "topic or style"),  # identical sets
    (math.nan, [math.nan] * 4, FULL, "unread"),  # an empty set (the do-nothing state)
    (None, [0.6] * 4, FULL, "unread"),
    (0.9, [0.6] * 3, FULL, "unread"),
    (0.9, [0.6] * 4, FULL | {"WRONG42": 1000}, "unread"),  # a set short of 1024 neurons
    (0.9, [X] * 4, FULL, "familiarity"),  # ties at x
    (0.95, [0.6, 0.62, 0.7, FAM], FULL, "familiarity"),  # exactly at the line
    (0.95, [0.6, 0.62, 0.7, FAM + U], FULL, "no call"),  # one neuron over it
    (0.95, [STY, 0.8, 0.85, 0.9], FULL, "topic or style"),  # exactly at the line
    (0.95, [STY - U, 0.8, 0.85, 0.9], FULL, "no call"),  # one neuron under it
    (0.95, [X, X, X, 0.88], FULL, "no call"),  # mixed pairs
    (0.6, [0.5] * 4, FULL, "no call"),  # wrong seeds unstable
    (0.6, [0.9] * 4, FULL, "no call"),
    (0.9, [0.3] * 4, FULL, "familiarity"),  # further from known than the fictional arm is
])  # fmt: skip
def test_wrong_verdict(S: ModuleType, w_wrong: float, cross: list[float], sizes: dict[str, int], verdict: str) -> None:
    assert S.wrong_verdict(w_wrong, WK, cross, X, L, sizes)["verdict"] == verdict


@pytest.mark.parametrize(("w_auth", "cross", "verdict"), [
    (1, [1] * 4, "topic does not"),
    (math.nan, [0.9] * 4, "unread"),
    (0.916015625, [TL, 0.85, 0.86, 0.87], "topic does not"),  # min exactly at the line
    (0.916015625, [TL - U, 0.85, 0.86, 0.87], "no call"),  # one neuron under: the pairs straddle it
    (0.916015625, [0.6, 0.65, 0.7, TL - U], "topic moves the set"),
    (0.916015625, [0.6, 0.65, 0.7, TL], "no call"),
    (0.916015625, [TL] * 4, "topic does not"),
])  # fmt: skip
def test_topic_verdict(S: ModuleType, w_auth: float, cross: list[float], verdict: str) -> None:
    assert S.topic_verdict(w_auth, 0.9052734375, cross, L, TOPIC)["verdict"] == verdict


def test_rules_read_sets(S: ModuleType) -> None:
    a, b = list(range(1024)), list(range(512, 1536))
    same = S.wrong_rule(dict.fromkeys((*S.WRONG, "KNOWN42", "KNOWN142", "FICT42", "FICT142"), a), X, L, WK)
    assert same["verdict"] == "topic or style" and same["cross"] == [1] * 4
    moved = S.wrong_rule({"WRONG42": b, "WRONG142": b, "KNOWN42": a, "KNOWN142": a, "FICT42": b, "FICT142": b}, X, L, WK)
    assert moved["cross"] == [0.5] * 4 and moved["verdict"] == "familiarity" and moved["wrong_vs_fict"] == [1] * 4
    topic = S.topic_rule({"AUTH42": a, "AUTH142": a, "WORLD42": b, "WORLD142": b, "FICT42": b, "FICT142": b}, L)
    assert topic["verdict"] == "topic moves the set" and topic["auth_minus_world_vs_fict"] == -0.5
    assert S.topic_rule({"AUTH42": [], "AUTH142": a, "WORLD42": b, "WORLD142": b, "FICT42": b, "FICT142": b}, L)["verdict"] == "unread"


@pytest.mark.parametrize(("true", "wrong", "verdict"), [
    ([], [], "unread"),
    ([-1.0] * 4, [-2.0] * 4, "holds"),
    ([-1.0] * 4, [-2.0, -2.0, -2.0, -1.0], "holds"),  # exactly 0.75, the tie not below
    ([-1.0] * 4, [-1.0] * 4, "missed"),  # identical texts: every row ties
    ([-1.0] * 4, [-2.0, -2.0, 0.0, 0.0], "missed"),
    ([-1.0, math.nan], [-2.0, -2.0], "unread"),
    ([-1.0], [-2.0, -2.0], "unread"),
])  # fmt: skip
def test_label_check(S: ModuleType, true: list[float], wrong: list[float], verdict: str) -> None:
    assert S.label_check(true, wrong, 0.75)["verdict"] == verdict


def test_topic_split(S: ModuleType) -> None:
    t = S.topic_split(100, 217)
    assert t["AUTH"] == list(range(100)) and t["WORLD"] == list(range(100, 217))


def test_swap() -> None:
    assert KF.swap(0, "It was William Shakespeare.") == "It was Charles Dickens."
    assert KF.swap(142, "The Boston Tea Party happened in Boston.") == "The Boston Tea Party happened in New York."
    with pytest.raises(AssertionError):
        KF.swap(0, "No answer here.")
