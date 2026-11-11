"""E155's filed rule (experiments/neuron_overlap/rules.py) on the edge cases filed in its predictions.json, and the shared
overlap read (`masked_reads.top_overlap`) on hand-made firing. No model is built or loaded."""

from __future__ import annotations

import math
from types import ModuleType

import pytest
import torch

from commute.jobs import load_experiment
from commute.masked_reads import top_overlap


@pytest.fixture(scope="module")
def rules() -> ModuleType:
    return load_experiment("neuron_overlap", "rules")


def test_apart_either_way(rules: ModuleType) -> None:
    assert rules.call([0.8, 0.82], [0.6, 0.65])["call"] == "reuse side higher"
    assert rules.call([0.6, 0.65], [0.8, 0.82])["call"] == "new side higher"


def test_tie_and_inside_band(rules: ModuleType) -> None:
    r = rules.call([0.5, 0.7], [0.4, 0.4])  # gap 0.2, band 0.2: a tie is no call
    assert r["call"] == "no call" and r["gap"] == pytest.approx(r["band"])
    assert rules.call([0.5, 0.8], [0.4, 0.45])["call"] == "no call"


def test_do_nothing_one_seed_nan(rules: ModuleType) -> None:
    assert rules.call([1.0, 1.0], [1.0, 1.0]) == {"call": "no call", "gap": 0.0, "band": 0.0}
    assert rules.call([0.9], [0.5, 0.5])["call"] == "unreadable"
    assert rules.call([math.nan, 0.5], [0.5, 0.5])["call"] == "unreadable"


def test_floor_and_positive_control(rules: ModuleType) -> None:
    assert rules.floor([0.6, 0.8], 0.9)["above"] == pytest.approx(-0.2)
    assert math.isnan(rules.floor([0.6, math.nan], 0.9)["above"])
    assert rules.positive_control(0.9, 0.8) and not rules.positive_control(0.8, 0.8)
    assert not rules.positive_control(math.nan, 0.8)


def firing(x: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"mlp0": x}


def test_top_overlap_identity_and_known_sets() -> None:
    f0 = torch.tensor([[[0.0, 5.0, -4.0, 1.0, 0.5]]])  # (B=1, L=1, width=5); top 2 by |firing|: neurons 1, 2
    same = top_overlap(firing(f0), firing(f0.clone()), {0: 2}, 0)[0]
    assert float(same["jaccard"][0]) == 1.0 and float(same["d2_all"][0]) == 0.0
    f1 = torch.tensor([[[0.0, 6.0, 0.0, 3.0, 0.5]]])  # top 2: neurons 1, 3; both = {1}, either = {1, 2, 3}
    r = top_overlap(firing(f0), firing(f1), {0: 2}, 0)[0]
    assert float(r["jaccard"][0]) == pytest.approx(1 / 3)
    assert float(r["d2_both"][0]) == pytest.approx(1.0) and float(r["d2_all"][0]) == pytest.approx(1 + 16 + 4)
