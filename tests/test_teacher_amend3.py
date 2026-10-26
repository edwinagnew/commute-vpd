"""CPU check of teacher_training Amendment 3's pure `score` (the filed rules on a results dict); no model, no GPU."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

EXP = Path(__file__).resolve().parents[1] / "experiments" / "teacher_training"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _state(p: float, verdict: str | None) -> dict[str, Any]:
    return {"p_target": p, "verdict": verdict}


def _task(teacher_p: float, a: tuple[float, float] | None, b: tuple[float, str] | None) -> dict[str, Any]:
    t: dict[str, Any] = {"teacher": {"p_target": teacher_p}, "adjoint": _state(0.06, None)}
    if a is not None:
        t["A"] = {"100": _state(a[0], "pass"), "1000": _state(a[1], "pass")}
    if b is not None:
        t["B"] = {"100": _state(0.5, "below P 0.9"), "1000": _state(*b)}
    return t


def test_score_applies_the_filed_rules():
    _load("run")
    am = _load("amend3")
    emo, r1, r0 = am.EMO, am.R1, am.R0
    out = {"W0": {"pass": True}, "G1": {"pass": True},
           "tasks": {emo: _task(0.9999, (0.98, 0.99), (0.95, "pass")), r1: _task(0.95, (0.95, 0.97), (0.7, "below P 0.9")),
                     r0: {**_task(0.5, None, None), "skipped": "teacher P <0.9"}}}
    sc = am.score(out)
    assert sc["teachers_learnt"] == [emo, r1] and sc["skipped"] == [r0]
    assert sc["P1_A_passes"] == {"pass": True, "per_task": {emo: True, r1: True}}
    assert sc["P2_A_above_adjoint"]["pass"] and sc["P2_A_above_adjoint"]["A"] == 0.99
    assert sc["P3_B_learnt"]["per_task"] == {emo: "learnt", r1: "between"} and sc["P3_B_learnt"]["overall"] == "between"

    out["tasks"][r1]["B"]["1000"] = _state(0.93, "pass")
    assert am.score(out)["P3_B_learnt"]["overall"] == "learnt"
    out["tasks"][r1]["B"]["1000"] = _state(0.3, "below P 0.9")
    out["tasks"][emo]["B"]["1000"] = _state(0.2, "below P 0.9")
    assert am.score(out)["P3_B_learnt"]["overall"] == "neither"

    out["tasks"][r1]["A"]["1000"] = _state(0.95, "inconclusive")
    assert am.score(out)["P1_A_passes"] == {"pass": False, "per_task": {emo: True, r1: False}}
    out["tasks"][emo]["A"]["1000"] = _state(0.5, "below P 0.9")
    assert not am.score(out)["P2_A_above_adjoint"]["pass"]

    partial = {"W0": {"pass": True}, "tasks": {emo: _task(0.9999, None, None)}}
    sc = am.score(partial)
    assert sc["G1"] is None and "P1_A_passes" not in sc and "P2_A_above_adjoint" not in sc and "P3_B_learnt" not in sc
