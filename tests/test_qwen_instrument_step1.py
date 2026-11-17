"""E152 step 1's filed rules (experiments/qwen_instrument_check/planted.py) on their edge cases, as filed in
results/predictions_step1.json: an empty planted set, chance at 1, ties at each bar, unreadable sizes, a non-monotone
size pattern, no readable pair, and the do-nothing and W0 states. Pure."""

from __future__ import annotations

import json
import math

import torch

from commute.jobs import EXPERIMENTS, load_experiment

PL = load_experiment("qwen_instrument_check", "planted")
RULES = json.loads((EXPERIMENTS / "qwen_instrument_check/results/predictions_step1.json").read_text())["rules"]
NAN = math.nan


def test_sets_pick_often_on_and_as_many_lowest():
    active = torch.tensor([0.9, 0.0, 0.6, 0.004, 0.5, 0.0])
    mean_g = torch.tensor([0.5, 0.002, 0.3, 0.01, 0.2, 0.001])
    s = PL.site_sets(active, mean_g)
    assert s["often"].tolist() == [0, 2]  # 0.5 is not above the bar
    assert s["rarely"].tolist() == [1, 5]  # the two zeros; the tie broken by mean g
    none = PL.site_sets(torch.tensor([0.1, 0.0]), torch.tensor([0.1, 0.0]))
    assert len(none["often"]) == len(none["rarely"]) == 0


def test_recovery_edges():
    assert PL.recovery(0.9, 0.0, 10) == 0.9
    assert math.isnan(PL.recovery(0.0, 0.0, 0))  # an empty set is unreadable, not a smear
    assert math.isnan(PL.recovery(0.5, 1.0, 10))  # chance at 1: zero denominator
    assert math.isnan(PL.recovery(NAN, 0.1, 10))  # W0 or do-nothing: no split energy, the share is 0/0
    assert PL.recovery(0.05, 0.05, 10) == 0.0  # a random-like split reads exactly chance


def test_kind_and_split_calls():
    assert PL.kind_call(0.5, RULES) == "returns" and PL.kind_call(0.2, RULES) == "partial"
    assert PL.kind_call(0.1999999999, RULES) == "partial"  # within 1e-9 of the bar rounds onto it
    assert PL.kind_call(-0.3, RULES) == "smears" and PL.kind_call(NAN, RULES) == "unreadable"
    assert PL.split_call(0.9, 0.7, 0.95, 0.05, RULES)["call"] == "works"
    assert PL.split_call(0.9, 0.7, 0.55, 0.05, RULES)["call"] == "works"  # gap exactly 0.5
    assert PL.split_call(0.9, 0.7, 0.54, 0.05, RULES)["call"] == "no call"  # both return, not apart
    assert PL.split_call(0.1, 0.0, 0.2, 0.2, RULES)["call"] == "free"
    assert PL.split_call(0.9, 0.1, 0.9, 0.1, RULES)["call"] == "no call"  # one returns, one smears
    assert PL.split_call(NAN, NAN, NAN, NAN, RULES)["call"] == "no call"  # W0 / do-nothing cannot pass


def test_separation():
    assert PL.separated(0.01, [0.005, 0.015], 0.002, True) is True
    assert PL.separated(0.002, [0.001, 0.003], 0.002, True) is False  # a tie with the spread is not past it
    assert PL.separated(0.01, [-0.001, 0.02], 0.002, True) is False  # interval holds 0
    assert PL.separated(0.01, [0.005, 0.015], 0.002, False) is None  # unscored or unmatched
    assert PL.separated(NAN, [NAN, NAN], 0.002, True) is None


def test_seen():
    assert PL.seen(True, True, ("scored", "scored")) is True
    assert PL.seen(None, True, ("scored", "unscored")) is False  # too small to see at that size
    assert PL.seen(None, False, ("scored", "scored")) is None  # an unmatched size is unreadable
    assert PL.seen(None, True, ("nan", "scored")) is None


def test_size_call():
    T, F = True, False
    assert PL.size_call({"rl": T, "mid": T, "sft": T}) == {"floor": "rl", "call": "all sizes"}
    assert PL.size_call({"rl": F, "mid": T, "sft": T})["call"] == "too small below mid"
    assert PL.size_call({"rl": F, "mid": F, "sft": T})["call"] == "too small below sft"
    assert PL.size_call({"rl": F, "mid": F, "sft": F})["call"] == "none"
    assert PL.size_call({"rl": T, "mid": F, "sft": T})["call"] == "no call"  # not monotone
    assert PL.size_call({"rl": None, "mid": T, "sft": T})["call"] == "no call"  # an unreadable size


def test_blind_call():
    assert PL.blind_call([False, None, False]) == "blind"
    assert PL.blind_call([False, True, None]) == "responds"
    assert PL.blind_call([None, None]) == "no call" and PL.blind_call([]) == "no call"
