"""CPU check of teacher_training Amendment 2's scoring (experiments/teacher_training/amend2.py: score, p1_reading) on a
hand-made results dict at the filed thresholds. No model, no GPU."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1] / "experiments" / "teacher_training"


def _load(name: str, file: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _amend2() -> Any:
    _load("run", "run.py")  # amend2.py imports its helpers as `from run import ...` (the script's directory)
    return _load("teacher_training_amend2", "amend2.py")


def _teacher(p: float, cos: float, ref_p: float = 0.937, ref_cos: float = 0.078) -> dict[str, Any]:
    return {"B-dropped-0.1": {"p_target": p, "cos": {"all": cos}, "verdict": "pass"},
            "ref_amend1": {"B-dropped": {"p_target": ref_p, "cos_all": ref_cos, "verdict": "pass"}}}


def test_p1_reading_at_the_thresholds():
    a = _amend2()
    assert a.p1_reading(0.937, 0.937) == "not carried"
    assert a.p1_reading(0.908, 0.937) == "not carried"  # a fall of 0.029
    assert a.p1_reading(0.966, 0.937) == "not carried"  # a rise of 0.029
    assert a.p1_reading(0.906, 0.937) == "between"  # a fall of 0.031
    assert a.p1_reading(0.838, 0.937) == "between"  # a fall of 0.099
    assert a.p1_reading(0.836, 0.937) == "carried in part"  # a fall of 0.101
    assert a.p1_reading(0.99, 0.937) == "between"  # a rise beyond the band is not 'not carried'


def test_score_reads_both_teachers_by_the_filed_rules():
    a = _amend2()
    out = {"teachers": {"sgd": _teacher(0.93, 0.097), "adam": _teacher(0.78, 0.2401, 0.892, 0.218)}}
    s = a.score(out)
    assert s["sgd"]["P1"] == "not carried" and s["sgd"]["P2"] is True  # cosine moved by 0.019
    assert s["adam"]["P1"] == "carried in part" and s["adam"]["P2"] is False  # cosine moved by 0.0221
    assert s["adam"]["p_target"] == [0.78, 0.892] and s["adam"]["verdict"] == ["pass", "pass"]
