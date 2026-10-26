"""CPU check of teacher_training Amendment 5's pure `score` (experiments/teacher_training/amend5.py) on a synthetic
results file shaped like results/teacher_training_amend5.json; no model, no GPU."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

EXP = Path(__file__).resolve().parents[1] / "experiments" / "teacher_training"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # amend5 imports `run` and `amend4` by those names
    spec.loader.exec_module(mod)
    return mod


def _amend5() -> Any:
    _load("run", EXP / "run.py")
    _load("amend4", EXP / "amend4.py")
    return _load("teacher_training_amend5", EXP / "amend5.py")


def _state(h: float, b: float, b160: float, scale: dict[str, float] | None = None) -> dict[str, Any]:
    """One state's readings: harness blocks at h, every budget-family 20-step reading at b, 160-step at b160, each
    multiplied by `scale[name]` (default 1)."""
    s = scale or {}
    return {"harness_b0": {"bf16": h * s.get("harness_b0", 1)}, "harness_b1": {"bf16": h},
            "budget": {"fp32": b, "bf16": b * s.get("budget.bf16", 1)},
            "start": {"fp32": b * s.get("start", 1), "bf16": h * s.get("start.bf16", 1)},
            "rows256": {"fp32": b * s.get("rows256", 1), "bf16": b},
            "budget160": {"fp32": b160, "bf16": b160}, "harness160_b0": {"bf16": b160 * s.get("harness160_b0", 1)}}


def _results(teacher_scale: dict[str, float], seeds: dict[str, float]) -> dict[str, Any]:
    """W0 at A4's values (harness 0.70, budget 0.66, 160 steps 0.71); the teacher at E34/A4's ratios (1.588, 9.24,
    10.45), with `teacher_scale` moving single readings; the seed-to-seed cosines `seeds`."""
    w0 = _state(0.70, 0.66, 0.71)
    t = _state(0.70 * 1.588, 0.66 * 9.24, 0.71 * 10.45, teacher_scale)
    return {"part1": {"W0": w0, "teacher": t},
            "part2": {"copies": {}, "cos": {"seeds": seeds, "with_dW": {}, "A_vs_B_seed0": 0.3, "null": {}}}}


def test_score_reads_the_harness_readout_moving_the_budget():
    a5 = _amend5()
    # S with H's readout lands at the harness's ratio (P2 via budget.bf16); rows and W0 agree; the copies agree
    s = a5.score(_results({"budget.bf16": 1.6 / 9.24}, {"A": 0.95, "B": 0.85}))
    assert abs(s["ratios"]["harness"] - 1.588) < 1e-9 and abs(s["ratios"]["budget"] - 9.24) < 1e-9
    assert s["P1"]["pass"] and s["P3"]["pass"]
    p2 = s["P2"]["per"]
    assert s["P2"]["pass"] and p2["budget.bf16"]["to_harness"] and p2["budget.bf16"]["implicates"] == "readout"
    assert p2["start"]["to_budget"] and p2["start"]["implicates"] == "readout"
    assert s["P4"]["pass"] and s["P4"]["reading"] == "same way"
    assert abs(s["identity_start_bf16_over_harness_b0"]["teacher"] - 1) < 1e-12
    assert s["identity_ok"] and s["implicates_agree"]


def test_score_reads_each_miss():
    a5 = _amend5()
    # rows carry part of it (rows256 40% off), both single differences land between the two measures, one arm's
    # copies point elsewhere
    s = a5.score(_results({"rows256": 0.6, "budget.bf16": 4 / 9.24, "start": 4 / 9.24}, {"A": 0.9, "B": 0.4}))
    assert not s["P1"]["pass"] and not s["P1"]["per"]["rows256_vs_budget"] and s["P1"]["per"]["harness_b0_vs_harness"]
    assert not s["P2"]["pass"] and all(v["implicates"] is None for v in s["P2"]["per"].values())
    assert not s["implicates_agree"]
    assert s["P4"]["reading"] == "one of many" and not s["P4"]["pass"]
    # S from H's start lands at the harness: the start carries it
    s = a5.score(_results({"start": 1.5 / 9.24}, {"A": 0.7, "B": 0.9}))
    assert s["P2"]["pass"] and s["P2"]["per"]["start"]["implicates"] == "start"
    assert s["P2"]["per"]["budget.bf16"]["implicates"] == "start"  # unmoved budget.bf16 sits at the budget: the same
    assert s["P4"]["reading"] == "between"


def test_w0_disagreement_and_partial_file():
    a5 = _amend5()
    out = _results({}, {"A": 0.9, "B": 0.9})
    out["part1"]["W0"]["harness160_b0"]["bf16"] *= 1.2  # the 160-step pair 20% apart at W0
    s = a5.score(out)
    assert not s["P3"]["pass"] and not s["P3"]["per"]["harness160_b0"] and s["P3"]["per"]["start.bf16"]
    del out["part1"]["teacher"]["rows256"], out["part2"]["cos"]
    s = a5.score(out)
    assert not s["P1"]["pass"] and not s["P4"]["pass"] and s["P4"]["reading"] is None
