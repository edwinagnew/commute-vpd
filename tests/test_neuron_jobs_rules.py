"""E189's pure readings (experiments/neuron_jobs/rules.py) on hand-made tensors and the edge cases filed in its
predictions.json. No model is built or loaded."""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any

import pytest
import torch

from commute.jobs import load_experiment


@pytest.fixture(scope="module")
def rules() -> ModuleType:
    return load_experiment("neuron_jobs", "rules")


def test_keep_top_keeps_largest_magnitude_signed(rules: ModuleType) -> None:
    x = torch.tensor([[[0.1, -3.0, 2.0, 0.5]]])
    assert torch.equal(rules.keep_top(x, 2), torch.tensor([[[0.0, -3.0, 2.0, 0.0]]]))


def test_retention_identical_disjoint_partial(rules: ModuleType) -> None:
    top0 = torch.tensor([[0, 5, 9], [1, 6, 8], [2, 7, 7]])  # (n_top 3, 3 neurons)
    top1 = torch.tensor([[2, 4, 9], [1, 3, 7], [0, 2, 8]])  # same set reordered, disjoint, same set
    flat = torch.tensor([0, 1, 2])
    assert rules.retention(top0, top0, flat).tolist() == [1.0, 1.0, 1.0]  # the zero update
    assert rules.retention(top0, top1, flat).tolist() == pytest.approx([1.0, 0.0, 1.0])
    top1[0, 2] = 0
    assert rules.retention(top0, top1, torch.tensor([2]))[0] == pytest.approx(2 / 3)


def test_kept_nan_unless_gain_positive(rules: ModuleType) -> None:
    assert math.isnan(rules.kept(0.0, 0.0))  # zero update
    assert math.isnan(rules.kept(1.0, -0.5))
    assert rules.kept(1.5, 1.0) == 1.5  # above 1 is reported as read


def side(m: float, r: float, kr: float) -> dict[str, Any]:
    """One side: mean m on both readings' values, random update r, random-K kept kr."""
    return {"above": m - r, "values": [m, m], "kr": [kr, kr]}


def pair(call: str, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {"retention": {"call": call, "floor": {"reuse": {"above": a["above"]}, "new": {"above": b["above"]}}},
            "kept": {"values": {"reuse": a["values"], "new": b["values"]}},
            "kept_random": {"values": {"reuse": a["kr"], "new": b["kr"]}}}  # fmt: skip


def test_tally_counts(rules: ModuleType) -> None:
    score = {"p1": pair("no call", side(0.6, 0.9, 0.0), side(0.7, 0.9, 0.5)), "p2": pair("reuse side higher", side(0.4, 0.3, math.nan), side(0.5, 0.9, 0.1))}
    t = rules.tally(score, th=3, no_call_min=1)
    assert (t["P1"]["count"], t["P2"]["count"], t["P3"]["count"], t["P4"]["count"]) == (3, 3, 2, 1)
    assert t["P1"]["holds"] and not t["P3"]["holds"]
