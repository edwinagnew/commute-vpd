"""E105's filed rule (experiments/rows_null/score.py) on its edge cases and the fixed half split. No model is built."""

from __future__ import annotations

import math
from types import ModuleType

import pytest

from commute.jobs import load_experiment

U = 1 / 1024  # one neuron of a top-1024 set: overlaps are exact in binary


@pytest.fixture(scope="module")
def S() -> ModuleType:
    return load_experiment("rows_null", "score")


@pytest.mark.parametrize(("w_a", "w_b", "cross", "x", "verdict"), [
    (1, 1, [1, 1, 1, 1], 0.49, "label"),  # identical sets
    (0.5, 0.5, [0.5] * 4, 0.49, "no call"),
    (math.nan, 0.9, [0.9] * 4, 0.49, "unread"),  # an empty set (the do-nothing state)
    (None, 0.9, [0.9] * 4, 0.49, "unread"),
    (0.9, 0.9, [0.9] * 3, 0.49, "unread"),
    (864 * U, 900 * U, [704 * U, 720 * U, 730 * U, 740 * U], 504 * U, "label"),  # w - min(cross) exactly the band
    (864 * U, 900 * U, [703 * U, 720 * U, 730 * U, 740 * U], 504 * U, "no call"),  # one neuron under it
    (864 * U, 900 * U, [600 * U, 620 * U, 640 * U, 664 * U], 504 * U, "rows"),  # max(cross) exactly x + band
    (864 * U, 900 * U, [600 * U, 620 * U, 640 * U, 665 * U], 504 * U, "no call"),  # one neuron over it
    (0.864, 0.9, [0.4921875] * 4, 0.4921875, "rows"),
    (0.864, 0.9, [0.4921875] * 3 + [0.85], 0.4921875, "no call"),  # mixed pairs
    (0.6, 0.62, [0.4921875] * 4, 0.4921875, "no call"),  # halves' seeds unstable
])  # fmt: skip
def test_verdict(S: ModuleType, w_a: float, w_b: float, cross: list[float], x: float, verdict: str) -> None:
    assert S.verdict(w_a, w_b, cross, x)["verdict"] == verdict


def test_rule_reads_sets(S: ModuleType) -> None:
    a, b = list(range(1024)), list(range(512, 1536))
    same = S.rule(dict.fromkeys(S.NAMES, a), 0.49)
    assert same["verdict"] == "label" and same["cross"] == [1, 1, 1, 1]
    half = S.rule({"A42": a, "A142": a, "B42": b, "B142": b}, 0.5)
    assert half["cross"] == [0.5] * 4 and half["verdict"] == "rows"
    assert S.rule({"A42": [], "A142": a, "B42": b, "B142": b}, 0.49)["verdict"] == "unread"


def test_halves(S: ModuleType) -> None:
    h = S.halves(217)
    assert (len(h["A"]), len(h["B"])) == (108, 109) and sorted(h["A"] + h["B"]) == list(range(217))
    assert h == S.halves(217) and h != S.halves(217, seed=106)
