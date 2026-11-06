"""E102 stage B's rules on their edge cases and on the do-nothing judge, before filing (commute.carrying_jobs's
forced-choice rules and maximiser test; experiments/qwen_carrying/stage_b.py rules)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from commute import carrying_jobs as CJ
from commute.jobs import load_experiment

SB = load_experiment("qwen_carrying", "stage_b")
S = SB.S
Pick = Callable[[str, str, int], tuple[str, str]]


def test_chance_rule_edges() -> None:
    assert CJ.chance_rule(math.nan, 0, 0.25, 0.2)["verdict"] == "unreadable"
    assert CJ.chance_rule(0.25, 90, 0.25, 0.2)["verdict"] == "below_or_level"  # at chance
    assert CJ.chance_rule(0.0, 90, 0.25, 0.2)["verdict"] == "below_or_level"  # se 0 below chance
    assert CJ.chance_rule(1.0, 90, 0.25, 0.2)["verdict"] == "above"  # se 0 well above
    assert CJ.chance_rule(0.5, 90, 0.25, 0.2)["verdict"] == "above"  # gap 0.25, 2 se 0.105
    assert CJ.chance_rule(0.4, 10, 0.25, 0.2)["verdict"] == "below_or_level"  # under the gap and under 2 se (0.31)
    assert CJ.chance_rule(0.5, 10, 0.25, 0.2)["verdict"] == "no_call"  # over the gap, under 2 se (0.32)


def test_forced_gate_needs_both_sides() -> None:
    assert CJ.forced_gate(0.8, 90, 0.5, 210, 0.25, 0.2)["status"] == "pass"
    assert CJ.forced_gate(0.6, 90, 0.5, 210, 0.25, 0.2)["status"] == "fail"  # above chance, level with random
    assert CJ.forced_gate(0.3, 90, 0.05, 210, 0.25, 0.2)["status"] == "fail"  # above random, at chance
    assert CJ.forced_gate(1.0, 90, 1.0, 210, 0.25, 0.2)["status"] == "fail"  # a judge always picking it
    assert CJ.forced_gate(0.25, 90, 0.25, 210, 0.25, 0.2)["status"] == "fail"  # a judge picking at random
    assert CJ.forced_gate(math.nan, 0, 0.5, 210, 0.25, 0.2)["status"] == "unreadable"
    assert CJ.forced_gate(0.8, 90, math.nan, 0, 0.25, 0.2)["status"] == "unreadable"
    assert CJ.forced_gate(0.75, 12, 0.5, 210, 0.25, 0.2)["status"] == "no call"  # gap 0.25 under 2 se (0.26)


def test_load_picks_missing_and_bad(tmp_path: Path) -> None:
    batches, letters = {"b00": ["x", "y", "z"], "b01": ["w"]}, {"b00": {"A": "fmt", "B": "tofu"}, "b01": {"A": "gold"}}
    picks = {"x": {"letter": "B", "fit": "strong"}, "y": {"letter": "E", "fit": "weak"}, "z": {"letter": "A", "fit": "maybe"},
             "q": {"letter": "A", "fit": "weak"}}
    (tmp_path / "b00.json").write_text(json.dumps({"batch": "b00", "picks": picks}))
    got, missing, bad = CJ.load_picks(tmp_path, batches, letters, ["strong", "weak"])
    assert got == {"x": {"desc": "tofu", "fit": "strong"}} and missing == ["b01"] and sorted(bad) == ["q", "y", "z"]


def test_peaks_and_text_scores_with_a_fake_model() -> None:
    fire = lambda toks: torch.stack([toks.float(), -toks.float()], dim=-1)  # neuron 0 fires the token id, neuron 1 minus it
    assert CJ.peaks(fire, [[1, 5, 2], [3]], [0, 0], 2, "cpu") == [5.0, 3.0]
    assert CJ.peaks(fire, [[1, 5, 2], [3]], [1, 1], 2, "cpu") == [-1.0, -3.0]  # padding (id 0, firing 0) is masked out
    neurons = {"a": {"flat": 0, "top_act": 10.0}}
    texts = {"a": {"examples": ["9", "8", "1", "1", "1"], "near_misses": ["1", "1", "1", "1", "9"]}}
    out = CJ.text_scores(fire, neurons, texts, lambda t: [int(t)], 0.3, 256, 3, "cpu", lambda m: None)
    assert out["a"]["examples"] == [9.0, 8.0, 1.0, 1.0, 1.0] and out["a"]["outcome"] == "win"  # 2 strong against 1


def table(pick: Pick, unjudged: Callable[[str], bool] = lambda g: False) -> dict[str, Any]:
    """A synthetic table in the filed pool's shape: pick(group, role, k) gives (description, fit); unjudged(group) drops
    the group's first two thirds of judge outputs; every neuron wins."""
    T: dict[str, Any] = {}

    def add(g: str, r: str, n: int) -> None:
        for k in range(n):
            d, f = pick(g, r, k)
            j = not unjudged(g) or k >= 2 * n // 3
            T[f"{g}/{r}/{k}"] = {"roles": [{"group": g, "role": r}], "desc": d if j else None, "fit": f if j else None,
                                 "judged": j, "outcome": "win"}

    add("control", "control", S["control_n"])
    for g, spec in S["groups"].items():
        add(g, "carrying", spec["n"])
        add(g, "random", S["partner_n"])
    return T


def cyc(k: int) -> str:
    return sorted(S["descriptions"])[k % 4]


def test_do_nothing_judges_cannot_pass() -> None:
    uniform = SB.rules(table(lambda g, r, k: (cyc(k), "weak")), True)
    assert uniform["CB1_power_gate"]["status"] == "fail" and not uniform["read"]["PB1_PB2"]
    always = SB.rules(table(lambda g, r, k: ("tofu", "strong")), True)
    assert always["CB1_power_gate"]["status"] == "fail" and not always["CB2_judge_fp"]["ok"]
    assert always["PB2_tofu_a2f_over_random"]["verdict"] == "below_or_level"


def test_a_judge_with_power_reads_both_ways() -> None:
    def good(g: str, r: str, k: int) -> tuple[str, str]:
        if r == "random":
            return cyc(k), "weak" if k % 5 else "strong"
        return {"control": "tofu", "a1": "fmt", "a2m": "tofu", "a2f": "tofu"}[g], "strong"

    out = SB.rules(table(good), True)
    assert out["CB1_power_gate"]["status"] == "pass" and out["CB2_judge_fp"]["ok"] and out["read"]["PB1_PB2"]
    assert out["PB1_fmt_a1_over_a2m"]["outcome"] == "holds" and out["PB2_tofu_a2f_over_random"]["outcome"] == "related"
    same = SB.rules(table(lambda g, r, k: ("fmt", "strong") if g == "a2m" and r == "carrying" else good(g, r, k)), True)
    assert same["PB1_fmt_a1_over_a2m"]["outcome"] == "misses"  # the matched TOFU state reads like the template
    lost = SB.rules(table(good, lambda g: g == "control"), True)
    assert lost["CB1_power_gate"]["status"] == "unreadable" and not lost["read"]["PB1_PB2"]  # a third judged: over missing_max
    assert not SB.rules(table(good), False)["read"]["R_explained"]
