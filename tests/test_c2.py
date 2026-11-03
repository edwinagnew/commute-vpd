"""E55 C2's reporting rules in experiments/pcfg_anchors/c2.py on hand-made readings: a missing positive control changes
no outcome and only how a fail is written, an ID state unfaithful on its own rows does not count for the control, and
a step's eval rows do not depend on the job that reads them. No model is built or loaded."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from commute.jobs import load_experiment


@pytest.fixture(scope="module")
def c2() -> ModuleType:
    return load_experiment("pcfg_anchors", "c2")


def scored(c2: ModuleType, n: int, agree: int) -> dict[str, Any]:
    return {c2.READINGS[0]: {"all": {"n": n, "agree": agree, "bar": 15, "n_c": 13}}, c2.READINGS[1]: {"all": {"n": 0, "agree": 0, "bar": None, "n_c": 0}}}


def result(c2: ModuleType, n: int, agree: int, control: str) -> dict[str, Any]:
    return {"scores": scored(c2, n, agree), "positive_control": dict.fromkeys(c2.READINGS, control),
            "unscored": {"u": {"labels": dict.fromkeys(c2.READINGS, "new")}}}  # fmt: skip


@pytest.mark.parametrize(("n", "agree", "outcome"), [(21, 21, "tie"), (20, 18, "pass_under"), (21, 10, "fail")])
def test_a_missing_positive_control_changes_no_outcome(c2: ModuleType, n: int, agree: int, outcome: str) -> None:
    c = SimpleNamespace(scored=dict.fromkeys(range(21)), unlearned=["u"])
    holds, none = (c2.outcomes(result(c2, n, agree, pc), c) for pc in ("holds", "none"))
    assert holds["headline"] == none["headline"] == outcome
    for q in c2.READINGS:
        assert holds[q]["outcome"] == none[q]["outcome"] and holds[q]["wrong_labels"] == none[q]["wrong_labels"]
    assert holds["r"]["outcome"] == outcome and holds["r"]["wrong_labels"] == n - agree
    assert holds["headline_written"] == outcome
    assert none["headline_written"] == (c2.UNVERIFIED if outcome == "fail" else outcome)
    assert none["r"]["written"] == (c2.UNVERIFIED if outcome == "fail" else outcome)


def test_a_verified_fail_on_one_reading_stays_a_fail_in_the_headline(c2: ModuleType) -> None:
    res = result(c2, 21, 10, "holds")
    res["scores"]["ratio"] = res["scores"]["r"]
    res["positive_control"]["ratio"] = "none"
    out = c2.outcomes(res, SimpleNamespace(scored=dict.fromkeys(range(21)), unlearned=["u"]))
    assert out["r"]["written"] == "fail" and out["ratio"]["written"] == c2.UNVERIFIED and out["headline_written"] == "fail"


def test_an_id_state_unfaithful_on_its_own_rows_is_void_for_the_control(c2: ModuleType) -> None:
    def rec(u0: float) -> dict[str, Any]:
        return {"cells": {"U0": u0, "T0": 0.0}, "x": {"r": 0.9, "ratio": 0.8}}

    res = {"constants": {"U0_vs_T0_max": 0.1}, "controls": {"a": rec(0.05), "b": rec(0.5), "c": rec(float("nan"))}}
    assert c2.id_readings(res, ["a", "b", "c"]) == [{"r": 0.9, "ratio": 0.8}, {"r": None, "ratio": None}, {"r": None, "ratio": None}]
    assert res["controls"]["b"]["x"] == {"r": 0.9, "ratio": 0.8}, "report only: the state's own readings stay as read"


def test_a_steps_eval_rows_do_not_depend_on_the_job(c2: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(c2, "device", lambda: torch.device("cpu"))
    cfg = SimpleNamespace(batch_size=2, eval_batch_size=2, eval_freq=1000, n_eval_steps=1)
    trains, evs = zip(*(c2.data(cfg) for _ in range(2)), strict=True)
    assert torch.equal(next(evs[0](2000)), next(evs[1](2000)))  # a resubmission, whatever the job read before
    assert torch.equal(next(evs[0](0)), next(evs[1](0))) and not torch.equal(next(evs[0](2000)), next(evs[0](3000)))
    for b in trains:
        b.close()


def test_the_leak_report(c2: ModuleType) -> None:
    def rec(dr: float, r_full: float) -> dict[str, Any]:
        return {"full_minus_cut": {"r": dr, "ratio": float("nan"), "T": 1e-4, "U": 2e-4}, "x": {"r": 0.5, "ratio": 0.5}, "r_full": r_full, "ratio_full": 0.5}

    res = {"constants": {"leak_report_over": 0.1, "high_is_existing": {"r": True, "ratio": True}}, "controls": {}, "scored": {}, "unscored": {},
           "calibration": {"p": rec(0.2, 0.7), "n": rec(0.05, 0.1)}}  # fmt: skip
    out = c2.leak(res, {"p": "existing", "n": "new"})
    assert list(out["over"]) == ["p"] and out["T_U_max"] == 2e-4
    assert out["full_column_cutoff"]["r"]["void"].startswith("fewer than 3")
