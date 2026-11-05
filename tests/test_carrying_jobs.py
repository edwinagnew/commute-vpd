"""E101's rules on their edge cases and on the do-nothing reading (commute.carrying_jobs), before filing."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from commute import carrying_jobs as CJ
from commute.jobs import load_experiment


def test_quantiles_edges() -> None:
    s = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    q = CJ.quantiles(s, torch.tensor([[-1.0, 0.0, 1.5, 3.0, 9.0]]))
    assert q.tolist() == [[0.0, 0.25, 0.5, 1.0, 1.0]]


def test_set_share_and_band() -> None:
    assert math.isnan(CJ.set_share([0.1, 0.2], []))
    assert abs(CJ.set_share([0.1, 0.2, 0.9, 0.4], [0, 1, 2, 3]) - 0.3) < 1e-12  # even set: mean of the middle two
    assert CJ.band(0.7, [0.5])["above_band"] is None  # one draw: no band
    b = CJ.band(0.7, [0.5, 0.6, 0.4])
    assert b["above_band"] and abs(b["excess"] - 0.2) < 1e-12
    assert CJ.band(0.6, [0.5, 0.6])["above_band"] is False  # ties with the top draw: not above


def test_row_rule_edges_and_null() -> None:
    row = {"reuse": ["a", "b"], "new": ["c", "d"]}
    assert CJ.row_rule({"a": 0.3, "c": 0.0, "d": 0.0}, row, 0.2)["verdict"] == "unreadable"  # one state a side
    assert CJ.row_rule({"a": 0.3, "b": 0.3, "c": 0.0, "d": 0.0}, row, 0.2)["verdict"] == "a_higher"  # zero spread
    assert CJ.row_rule({"a": 0.01, "b": -0.01, "c": 0.0, "d": 0.02}, row, 0.2)["verdict"] == "none"  # null-like
    assert CJ.row_rule({"a": math.nan, "b": 0.3, "c": 0.0, "d": 0.0}, row, 0.2)["verdict"] == "unreadable"


def test_block_matched_and_consensus() -> None:
    g = torch.Generator().manual_seed(0)
    idx = [5, 3072 + 7, 3072 + 9]
    r = CJ.block_matched(idx, set(idx), g)
    assert CJ.blocks_of(r) == CJ.blocks_of(idx) and not set(r) & set(idx)
    assert CJ.consensus([[1, 2, 3], [3, 1, 4]], 2) == [1, 3]


def test_outcome_edges() -> None:
    assert CJ.outcome([1.0] * 5, [0.0] * 5, 0.0, 0.3) is None  # dead neuron
    assert CJ.outcome([], [0.0], 1.0, 0.3) is None
    assert CJ.outcome([0.3, 0, 0, 0, 0], [0.0] * 5, 1.0, 0.3) == "win"  # 30% counts as strong
    assert CJ.outcome([0.0] * 5, [0.0] * 5, 1.0, 0.3) == "tie"  # nothing fires: not explained
    assert CJ.outcome([0.0] * 5, [0.5] * 5, 1.0, 0.3) == "lose"


def test_win_rule_strong_counts() -> None:
    assert CJ.outcome([0.29] * 5, [0.29] * 5, 1.0, 0.3) == "tie"  # no strong example on either side
    assert CJ.outcome([0.3, 0.3, 0, 0, 0], [0.3, 0.3, 0, 0, 0], 1.0, 0.3) == "tie"  # equal strong counts, not a win
    assert CJ.outcome([0.3, 0.3, 0, 0, 0], [0.3, 0.29, 0.1, 0, 0], 1.0, 0.3) == "win"  # 2 against 1
    assert CJ.outcome([0.9] * 5, [0.9] * 5, -1.0, 0.3) is None  # negative top: unscorable


def test_self_test_edges() -> None:
    assert CJ.self_test(["win"] * 9 + ["lose"], 0.8)["ok"]
    assert CJ.self_test(["win", "tie"], 0.8)["win_share"] == 0.5
    st = CJ.self_test([None, None], 0.8)  # nothing scorable: nan, not ok
    assert math.isnan(st["win_share"]) and not st["ok"] and st["counts"]["unscorable"] == 2


def test_load_texts_missing_and_bad(tmp_path: Path) -> None:
    ok = {"examples": ["a"] * 5, "near_misses": ["b"] * 5}
    (tmp_path / "b00.json").write_text(json.dumps({"batch": "b00", "texts": {"x": ok, "y": {**ok, "examples": ["a"] * 4},
                                                                           "z": {**ok, "near_misses": ["b"] * 4 + [" "]}, "w": ok}}))
    (tmp_path / "b02.json").write_text("not json")
    t, missing, bad = CJ.load_texts(tmp_path, {"b00": ["x", "y", "z"], "b01": ["u"], "b02": ["v"]}, 5, 5)
    assert list(t) == ["x"] and missing == ["b01", "b02"] and sorted(bad) == ["w", "y", "z"]  # w is not in b00


def test_gap_rule_do_nothing_and_edges() -> None:
    assert CJ.gap_rule(0.0, 16, 0.0, 180, 0.2)["verdict"] == "below_or_level"  # do-nothing at 0
    assert CJ.gap_rule(0.3, 16, 0.3, 180, 0.2)["verdict"] == "below_or_level"  # do-nothing at the floor
    assert CJ.gap_rule(1.0, 16, 1.0, 180, 0.2)["verdict"] == "below_or_level"
    assert CJ.gap_rule(0.5, 0, 0.1, 180, 0.2)["verdict"] == "unreadable"
    assert CJ.gap_rule(math.nan, 16, 0.1, 180, 0.2)["verdict"] == "unreadable"
    assert CJ.gap_rule(0.75, 16, 0.1, 180, 0.2)["verdict"] == "above"
    assert CJ.gap_rule(0.32, 16, 0.1, 180, 0.2)["verdict"] == "no_call"  # gap 0.22 over 0.2 but under 2 se (0.237)


def test_row_kind_do_nothing() -> None:
    s = {"p": 0.2, "n": 24, "floor": 0.2, "n_floor": 180}
    assert CJ.row_kind(s, s, 0.2)["verdict"] == "below_or_level"
    z = {"p": 0.0, "n": 24, "floor": 0.0, "n_floor": 180}
    assert CJ.row_kind(z, z, 0.2)["verdict"] == "below_or_level"
    assert CJ.row_kind({**s, "n": 0}, s, 0.2)["verdict"] == "unreadable"
    assert CJ.row_kind({"p": 0.8, "n": 16, "floor": 0.1, "n_floor": 180}, {"p": 0.05, "n": 24, "floor": 0.05, "n_floor": 180}, 0.2)["verdict"] == "above"


# ----------------------------------------------------------------------------- Amendment 1 (predictions_amend1.json)


def test_contrast_top_twins_and_exclusion() -> None:
    a = torch.tensor([0.9, 0.5, 0.8, 0.1])
    u = torch.tensor([0.1, 0.5, 0.0, 0.1])
    eng = {"t": a, "twin": a.clone(), "u": u, "u2": u.clone(), "v": torch.tensor([0.9, 0.1, 0.0, 0.1])}
    top, over = CJ.contrast_top(eng, "t", 2, set())
    assert over == ["u", "v"] and top == [2, 0]  # contrast [0.4, 0.2, 0.8, 0.0]: t's twin left out, u's twin once
    assert CJ.contrast_top(eng, "t", 2, {2})[0] == [0, 1]
    with pytest.raises(AssertionError):
        CJ.contrast_top({"t": a, "twin": a.clone()}, "t", 1, set())  # no task with other windows
    with pytest.raises(AssertionError):
        CJ.contrast_top(eng, "t", 4, {0})  # too few neurons left


def test_every_task_edges() -> None:
    assert CJ.every_task([]) == "no call"
    assert CJ.every_task(["above"] * 3) == "pass"
    assert CJ.every_task(["above", "below_or_level", "no_call"]) == "fail"
    assert CJ.every_task(["above", "no_call", "unreadable"]) == "no call"


SC = load_experiment("carrying_jobs", "score")
RES = Path(__file__).resolve().parents[1] / "experiments" / "carrying_jobs" / "results"
P, A = json.loads((RES / "predictions.json").read_text()), json.loads((RES / "predictions_amend1.json").read_text())
CTL = A["settings"]["controls"]
Tie = Callable[[str, str], set[str]]


def table(tie: Tie, unjudged: Callable[[str, str], bool] = lambda g, r: False) -> dict[str, Any]:
    """A synthetic table in the filed pool's shape: 180 random, each group's carrying, the new controls; every neuron
    wins; tie(group, role) gives its ties; unjudged(group, role) drops the group's first two judge outputs."""
    T: dict[str, Any] = {}

    def add(g: str, r: str, n: int) -> None:
        for k in range(n):
            j = not unjudged(g, r) or k >= 2
            T[f"{g}/{r}/{k}"] = {"roles": [{"group": g, "role": r}], "outcome": "win", "ties": tie(g, r) if j else set(),
                                 "judged": j, "complete": j}

    add("pool", "random", 180)
    for g, s in P["groups"].items():
        add(g, "carrying", s["n"])
    for t, c in CTL.items():
        add(t, "control_amend1", c["n"])
    return T


def own(g: str, r: str) -> set[str]:
    return {CTL[g]["description"]} if r == "control_amend1" else set()


def test_amend1_do_nothing_everything_and_perfect() -> None:
    none = SC.amend1(P, A, table(lambda g, r: set()), True)
    assert none["C1a"]["status"] == "fail" and none["C1b"]["status"] == "fail" and none["C1_fp"]["ok"]
    assert not none["route_open_on_67M"] and not none["read"]["P1"] and not none["read"]["P2"]
    every = SC.amend1(P, A, table(lambda g, r: set(P["descriptions"])), True)
    assert every["C1a"]["status"] == "fail" and not every["C1_fp"]["ok"] and every["C1_fp"]["share"] == 1.0
    good = SC.amend1(P, A, table(own), True)
    assert good["C1a"]["status"] == good["C1b"]["status"] == "pass" and good["route_open_on_67M"]
    assert good["read"]["P1"] and not good["read"]["P2"]  # P1 itself fails: no carrying neuron is tied
    assert not SC.amend1(P, A, table(own), False)["read"]["P1"]  # C2 missed


def test_amend1_missing_confused_and_random_controls() -> None:
    miss = SC.amend1(P, A, table(own, lambda g, r: g == "lowercase" and r == "control_amend1"), True)
    assert miss["C1a"]["tasks"]["lowercase"]["verdict"] == "unreadable" and miss["C1a"]["status"] == "no call"
    assert not miss["route_open_on_67M"]
    both: Tie = lambda g, r: {"copykey", "lookup"} if g == "ident32" and r == "control_amend1" else own(g, r)
    conf = SC.amend1(P, A, table(both), True)
    assert conf["C1a"]["status"] == "pass" and conf["C1b"]["tasks"]["ident32"]["verdict"] == "below_or_level"
    assert conf["C1b"]["status"] == "fail" and not conf["read"]["P2"]
    T = table(lambda g, r: set())  # controls that read as random: a tenth of every role tied to every description
    for i, t in T.items():
        if int(i.rsplit("/", 1)[1]) % 10 == 0:
            t["ties"] = set(P["descriptions"])
    rnd = SC.amend1(P, A, T, True)
    assert rnd["C1a"]["status"] == "fail" and not rnd["route_open_on_67M"]
