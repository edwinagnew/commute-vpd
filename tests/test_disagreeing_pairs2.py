"""E199's build rule: the edge cases listed in experiments/disagreeing_pairs2/results/predictions.json
(edge_cases_checked_before_filing), against pairs.py and E151's rules.py."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from commute.jobs import load_experiment

PR = load_experiment("disagreeing_pairs2", "pairs")
RU = load_experiment("disagreeing_pairs", "rules")
PRED = Path(__file__).resolve().parents[1] / "experiments/disagreeing_pairs2/results/predictions.json"
TH = json.loads(PRED.read_text())["thresholds"]
M = TH["build_margin"]
NAN = math.nan


def side(x: float) -> list[float]:
    return [x - 0.01, x, x + 0.01]


def test_prior_bar_edges() -> None:
    assert PR.prior_ok(-0.6, M)
    assert not PR.prior_ok(math.log10(5.91e-5 / 2.32e-4), M)  # E151's common-word cipher against copy: -0.594
    assert not PR.prior_ok(NAN, M)
    assert not PR.prior_ok(1.3, M)  # the prior points to the reuse side


def test_first_built() -> None:
    assert PR.first_built({"freq64": "condition not met", "freq32": "opposite", "freq16": "opposite"}) == "freq32"
    assert PR.first_built({"freq64": "condition not met", "neighbour": "flat"}) is None
    assert PR.first_built({}) is None


def condition(gap: float, reuse: list[float], new: list[float]) -> str:
    return RU.pair_calls({"prior_gap": gap, "steering": (reuse, new)}, TH)["condition"]


def test_conditions() -> None:
    assert condition(-1.3, side(0.56), side(0.25)) == "opposite"  # the built shape: prior to the new side, steering to reuse
    assert condition(-1.3, side(0.56), [0.25, NAN, NAN]) == "condition not met"  # a new seed under the gain floor
    assert condition(-1.3, side(0.20), side(0.56)) == "condition not met"  # steering with the prior's sign
    assert condition(-1.3, side(0.56), side(0.50)) == "condition not met"  # steering does not separate
    assert condition(0.0, [NAN] * 3, [NAN] * 3) == "condition not met"  # do-nothing pair: gain 0, steering unreadable


def test_readings() -> None:
    rec = RU.pair_calls({"prior_gap": -1.3, "steering": (side(0.56), side(0.25))}, TH)
    read = lambda r, n: RU.reading(rec["condition"], rec["prior"]["call"], rec["steering"], RU.CALL(r, n, TH))
    assert read(side(0.77), side(0.50)) == "follows steering"
    assert read(side(0.77), side(0.88)) == "follows prior"
    assert read(side(0.77), side(0.74)) == "neither"
    assert read(side(0.77), [NAN] * 3) == "unreadable"


def test_frequent_words_and_balanced() -> None:
    G = torch.tensor([[0.1, 0.5, 0.5, 0.01], [0.2, 0.3, 0.3, 0.9]])  # 2 keys, 4 tokens
    assert PR.frequent_words(G, [0, 1, 2, 3], 2) == [1, 2]  # minima 0.1, 0.3, 0.3, 0.01: a tie, pool order
    keys = list(range(100, 128))
    vals = PR.balanced(keys, [7, 8, 9, 10], 0)
    assert sorted(vals.count(w) for w in (7, 8, 9, 10)) == [7, 7, 7, 7]
    with pytest.raises(AssertionError):
        PR.balanced(keys, [7, 8, 9], 0)


def test_neighbours() -> None:
    E = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.9, 0.1]])
    allowed = torch.tensor([False, True, False, True, True])
    assert PR.neighbours(E, [0, 2], allowed) == (1, 3)  # tokens 1 and 4 tie for key 0: the lower id
    assert PR.neighbours(E, [0, 2], torch.tensor([False, False, False, True, False])) == (3, 3)  # a shared neighbour is allowed
    with pytest.raises(AssertionError):
        PR.neighbours(E, [0, 2], torch.tensor([True, True, False, True, True]))
