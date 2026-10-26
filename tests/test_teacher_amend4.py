"""CPU check of teacher_training Amendment 4's pure `score` (experiments/teacher_training/amend4.py) on a synthetic
results file shaped like results/teacher_training_amend4.json; no model, no GPU."""

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
    sys.modules[spec.name] = mod  # amend4 does `from run import ...`, so run.py is registered as "run" first
    spec.loader.exec_module(mod)
    return mod


def _amend4() -> Any:
    _load("run", EXP / "run.py")
    return _load("teacher_training_amend4", EXP / "amend4.py")


def _copy(p: float, pgd: float, b20: float, b160: float) -> dict[str, Any]:
    return {"verdict": "pass" if p >= 0.9 else "below P 0.9", "p_target": p, "pgd": pgd, "budget": {"20": b20, "160": b160}}


def _results(a4: Any) -> dict[str, Any]:
    """Every copy at E34's reference reading, budget ratio 1.05 at both budgets; the three inconclusive teachers'
    do-nothing at 1.5, the others at 1.1."""
    w0 = {"20": 0.66, "160": 0.71}
    out: dict[str, Any] = {"budget_w0": {"0": w0, "1": {"20": 0.70, "160": 0.86}}, "tasks": {}}
    for n in a4.TASKS:
        out["tasks"][n] = {"task": n, "teachers": {}}
        for o in a4.OPTS:
            nothing = 1.5 if (n, o) in a4.AM.inconclusive else 1.1
            t: dict[str, Any] = {"nothing": {"budget": {b: nothing * w0[b] for b in w0}}}
            for a in a4.ARMS:
                p, pgd = a4.AM.e34[n][o][a]
                t[a] = {a4.LAST: _copy(p, pgd, 1.05 * w0["20"], 1.05 * w0["160"])}
            out["tasks"][n]["teachers"][o] = t
    return out


def test_score_passes_everything_at_the_references():
    a4 = _amend4()
    s = a4.score(_results(a4))
    assert s["n_copies"] == 12
    assert s["P1"]["pass"] and s["P1"]["n_pass"] == 12
    assert s["P2"]["pass"] and s["P2"]["n_hold"] == 12
    assert abs(s["P2"]["largest"]["p"]["p"]) < 1e-12 and abs(s["P2"]["largest"]["ratio"]["ratio"]) < 1e-12
    assert s["P3a"]["pass"] and all(abs(v["160"] - 1.05) < 1e-9 for v in s["P3a"]["ratios"].values())
    assert s["P3b"]["pass"] and set(s["P3b"]["per"]) == {f"{n} {o}" for n, o in a4.AM.inconclusive}


def test_score_reads_each_miss_and_names_the_largest_gaps():
    a4 = _amend4()
    out = _results(a4)
    r1 = out["tasks"][a4.R1]["teachers"]
    r1["adam"]["B"][a4.LAST] = _copy(0.85, 1.1308 * 1.3, 0.70, 0.71 * 1.3)  # P below 0.9, ratio 30% off, budget 1.3 at 160
    r1["sgd"]["nothing"]["budget"]["160"] = 0.71 * 1.0  # the teacher no longer sits above its copies at 160
    s = a4.score(out)
    assert not s["P1"]["pass"] and s["P1"]["n_pass"] == 11 and not s["P1"]["per"]["r1_t534_v369 adam B"]
    assert not s["P2"]["pass"] and s["P2"]["n_hold"] == 11
    assert s["P2"]["largest"]["p"]["copy"] == "r1_t534_v369 adam B" and abs(s["P2"]["largest"]["p"]["p"] + 0.1328) < 1e-9
    assert s["P2"]["largest"]["ratio"]["copy"] == "r1_t534_v369 adam B" and abs(s["P2"]["largest"]["ratio"]["ratio"] - 0.3) < 1e-9
    assert not s["P3a"]["pass"] and s["P3a"]["n_pass"] == 11
    assert not s["P3b"]["pass"] and not s["P3b"]["per"]["r1_t534_v369 sgd"]["above_both_copies"]["160"]
    assert s["P3b"]["per"]["r1_t534_v369 sgd"]["above_both_copies"]["20"] and s["P3b"]["per"]["emoticon adam"]["pass"]


def test_score_on_a_partial_file_passes_nothing():
    a4 = _amend4()
    out = _results(a4)
    del out["tasks"][a4.EMO]["teachers"]["adam"]["B"]
    s = a4.score(out)
    assert s["n_copies"] == 11 and not s["P1"]["pass"] and s["P1"]["n_pass"] == 11 and not s["P3a"]["pass"]
