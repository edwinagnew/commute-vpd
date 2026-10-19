"""The task ladder's amended scoring (reports/2026-10-16_prediction_ladder-routes-amendment.md) on synthetic records:
the new G1 rule passes a gated-off control that re-gating pushes over the old rule, G1b/G3/G4 gate as filed, and A1/A2
count as filed. run.py is loaded from its path (it is a script, not a package module)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

RUN = Path(__file__).resolve().parents[1] / "experiments" / "task_ladder" / "run.py"


@pytest.fixture(scope="module")
def run():
    spec = importlib.util.spec_from_file_location("task_ladder_run", RUN)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


def _split(d, r, i, e_delta, gain=5.0, p_cells=None, ok=True):
    e_gate = 1 - d - r - i - e_delta
    return {"shares": {"d": d, "r": r, "i": i, "e_gate": e_gate, "e_delta": e_delta}, "share_sum": 1.0, "gain": gain,
            "scorable": gain >= 1.0, "G3_ok": ok, "p_cells": p_cells or {}}


def _arm(split, retained, p=0.9, step="10"):
    return {"curve": [{"heldout_p_o": p}] * 10, "steps": {step: {"retained": retained, "routes": split}}}


def _task(pb_split, off_split, n_heldout=100, pb30=None):
    pb = _arm(pb_split, {"ci_masked": pb_split["shares"]["d"] + pb_split["shares"]["r"] + pb_split["shares"]["i"],
                         "unmasked": 1 - pb_split["shares"]["e_delta"]})
    if pb30 is not None:
        pb["steps"]["30"] = {"retained": {}, "routes": pb30}
    # gated-off: re-gating 0.5 puts the OLD ci_masked rule (<= 0.2) out of reach, the new rule (d <= 0.2) passes
    off = _arm(off_split, {"ci_masked": off_split["shares"]["d"] + off_split["shares"]["r"] + off_split["shares"]["i"],
                           "unmasked": 1 - off_split["shares"]["e_delta"]})
    return {"n_heldout": n_heldout, "pullback": pb, "gated_off": off,
            "do_nothing": {"routes": _split(0.0, 0.5, 0.0, 1.0)}}


def _setup(run, pb_hi, pb_lo, g1b="pass", g3=True, conc_p=None):
    off = _split(0.02, 0.5, 0.0, 0.05)
    recs = {run.EMOTICON: _task(_split(0.03, 0.5, 0.02, 0.0), off, n_heldout=run.CONC_N_HELDOUT,
                                pb30=_split(0.03, 0.5, 0.02, 0.0, p_cells=conc_p or {"L01": 0.54, "L11": 0.63}))}
    tasks = {}
    for r in range(3):
        recs[f"hi{r}"] = _task(_split(pb_hi[0], pb_hi[1], pb_hi[2], 0.0, ok=g3), off)
        recs[f"lo{r}"] = _task(_split(pb_lo[0], pb_lo[1], pb_lo[2], 0.0), off)
        tasks[f"rung{r}"] = {"high": f"hi{r}", "low": f"lo{r}", "coverage_gap": 0.3}
    noise = {"P1_margin": 0.15, "m_d": 0.15, "m_r": 0.15, "G1b_last": {"status": g1b}}
    return {"tasks": recs}, tasks, noise


def test_new_g1_passes_where_old_rule_fails_and_a2_passes(run):
    out, tasks, noise = _setup(run, pb_hi=(0.05, 0.6, 0.05), pb_lo=(0.03, 0.4, 0.02))
    sc = run.score(out, tasks, noise)
    assert sc["harness_separates_placement"]
    g = sc["G1"]["hi0"]
    assert g["status"] == "pass" and g["old_rule_reported"]["would_pass"] is False
    assert sc["G3_identities_hold"] and sc["G4"]["status"] == "pass"
    assert sc["A2"]["verdict"] == "pass" and sc["A2"]["ladder_passes"] == 6
    assert sc["A1"]["verdict"] == "null"  # d differs by 0.02 < m_d
    assert sc["P1"]["verdict"] == "pass"  # ci_masked differs by 0.25 >= 0.15


def test_a1_passes_on_direct_gap_and_g1b_fail_blocks_it(run):
    out, tasks, noise = _setup(run, pb_hi=(0.5, 0.1, 0.0), pb_lo=(0.1, 0.1, 0.0))
    assert run.score(out, tasks, noise)["A1"]["verdict"] == "pass"
    out, tasks, noise = _setup(run, pb_hi=(0.5, 0.1, 0.0), pb_lo=(0.1, 0.1, 0.0), g1b="FAIL")
    sc = run.score(out, tasks, noise)
    assert sc["A1"]["verdict"] == "not scored (G1b failed)" and sc["A2"]["verdict"] == "null"


def test_g1_failure_g3_violation_and_g4_miss(run):
    out, tasks, noise = _setup(run, pb_hi=(0.05, 0.6, 0.05), pb_lo=(0.03, 0.4, 0.02))
    out["tasks"]["hi1"]["gated_off"]["steps"]["10"]["routes"] = _split(0.4, 0.3, 0.0, 0.05)  # direct reads high: FAIL
    sc = run.score(out, tasks, noise)
    assert sc["G1"]["hi1"]["status"] == "FAIL" and "A1" not in sc and "G1 failed" in sc["verdict"]
    out, tasks, noise = _setup(run, pb_hi=(0.05, 0.6, 0.05), pb_lo=(0.03, 0.4, 0.02), g3=False)
    assert "G3 violated" in run.score(out, tasks, noise)["verdict"]
    out, tasks, noise = _setup(run, pb_hi=(0.05, 0.6, 0.05), pb_lo=(0.03, 0.4, 0.02), conc_p={"L01": 0.40, "L11": 0.63})
    assert run.score(out, tasks, noise)["G4"]["status"].startswith("MISS")
