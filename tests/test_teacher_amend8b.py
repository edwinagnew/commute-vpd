"""CPU check of teacher_training Amendment 8b's pure rules (experiments/teacher_training/amend8b.py) at their edges and of the
filing; no model, no GPU."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

EXP = Path(__file__).resolve().parents[1] / "experiments" / "teacher_training"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # the siblings import each other by these names
    spec.loader.exec_module(mod)
    return mod


def _b() -> Any:
    for name in ("run", "newskill", "newskill_amend1", "path", "amend8"):
        _load(name)
    return _load("amend8b")


def test_separation_and_noise_class():
    b = _b()
    assert b.separation([2.0], [1.0, 1.5]) == "apart"  # the positive control: a one-element pool
    assert b.separation([1.0, 1.5], [2.0]) == "reversed"
    assert b.separation([1.5, 2.0], [1.0, 1.5]) == "overlap"  # a tie is not apart
    for a, c in (([], [1.0]), ([1.0], []), ([0.0], [1.0]), ([math.nan], [1.0]), ([1.0], [math.inf])):
        assert b.separation(a, c) == "unscorable"
    assert b.noise_class([3.0, 3.0, 3.0], [1.0, 1.0, 1.0])["class"] == "apart"
    assert b.noise_class([1.0, 1.0, 1.0], [3.0, 3.0, 3.0])["class"] == "reversed"
    assert b.noise_class([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])["class"] == "within noise"  # identical: excess 0, spread 0
    assert b.noise_class([1.0, 2.0, 1.2], [1.0, 1.0, 1.2])["class"] == "within noise"  # excess 0.31, spread 1
    assert b.noise_class([1.2, 1.5, 1.5], [1.0, 1.0, 1.2])["class"] == "suggestive"  # a tie at 1.2; excess 0.31, spread 0.25
    assert b.noise_class([1.0, 1.0], [1.0, 1.0, 1.0])["class"] == "unscorable"  # a state with a start missing
    assert b.noise_class([1.0, math.nan, 1.0], [1.0, 1.0, 1.0])["class"] == "unscorable"


def test_paired_verdict_edges():
    b = _b()
    assert b.paired_verdict(None, 0.21) == b.paired_verdict(math.nan, 0.21) == "unscorable"
    assert b.paired_verdict(1.5 / 1.21, 0.21) == "pass"  # the line itself, inclusive
    assert b.paired_verdict(1.5 * 1.21, 0.21) == "fail"
    assert b.paired_verdict(1.5, 0.21) == "inconclusive"


def test_position_edges():
    b = _b()
    assert b.position(1.0, 1.0, 4.0) == ("the teacher", 0.0)
    assert b.position(4.0, 1.0, 4.0) == ("the start", 1.0)
    assert b.position(2.0, 1.0, 4.0)[0] == "both"
    assert b.position(0.5, 1.0, 4.0)[0] == "the teacher"  # below the null student
    assert b.position(8.0, 1.0, 4.0)[0] == "the start"  # above the control
    assert b.position(4.0 ** (b.BM.low - 1e-9), 1.0, 4.0)[0] == "the teacher"  # either side of each line
    assert b.position(4.0 ** (b.BM.low + 1e-9), 1.0, 4.0)[0] == "both"
    assert b.position(4.0 ** (b.BM.high - 1e-9), 1.0, 4.0)[0] == "both"
    assert b.position(4.0 ** (b.BM.high + 1e-9), 1.0, 4.0)[0] == "the start"
    assert b.position(1.0, 1.0, 1.99) == ("no power", None)
    assert b.position(1.0, 1.0, 2.0)[0] == "the teacher"  # power is inclusive
    assert b.position(1.0, 2.0, 1.0) == ("no power", None)  # the references reversed
    for g, a, c in ((0.0, 1.0, 4.0), (math.nan, 1.0, 4.0), (1.0, 0.0, 4.0), (1.0, 1.0, math.inf)):
        assert b.position(g, a, c) == ("unscorable", None)


def _rec(kl: list[float], w0: tuple[float, float, float] = (0.692, 0.836, 0.705)) -> dict[str, Any]:
    r = [x / y for x, y in zip(kl, w0, strict=False)]
    return {"kl": {f"h{i}": x for i, x in enumerate(kl)}, "mean": sum(r) / len(r)}


def test_score_seeds_on_a_synthetic_file():
    b = _b()
    states = {b.REAL: _rec([0.68, 0.83, 0.70]), b.NULL: _rec([0.96, 0.90, 0.85]), b.RANDOM: _rec([0.67, 0.80, 0.70]), b.TRUE: _rec([5.07])}
    s = b.score_seeds({"states": states, "W0": {"pgd_floor": 0.2088}})
    assert s["answer"] == "apart" and s["positive_control"] == "apart"
    assert s["verdict"][b.TRUE] == "fail" and s["verdict"][b.REAL] == "pass"
    states[b.TRUE] = _rec([0.75])  # the control not above the real student's readings: no power, whatever else reads
    assert b.score_seeds({"states": states, "W0": {"pgd_floor": 0.2088}})["answer"] == "no power"


def test_filing_matches_the_code():
    b = _b()
    filed = json.loads(b.PRED.read_text())
    assert filed["amend8b"] == json.loads(json.dumps(asdict(b.BM)))
    assert filed["config"] == json.loads(json.dumps(asdict(b.CFG)))


def test_score_moved_on_a_synthetic_file():
    b = _b()
    a = b.saved("n3b")["reads"]["1000"]["teacher_gap"]
    last = {"train_kl_trigger": 0.01, "train_kl_other": 0.002}
    out = {"control": {"1000": 4 * a}, "reads": {"1000": {"teacher_gap": a, "pgd": 1.0, "base_verdict": "pass"}}, "curves": {"moved": [last]}}
    s = b.score_moved(out)
    assert s["class"] == "the teacher" and s["position"] == 0.0 and s["train_kl_trigger"] == 0.01
