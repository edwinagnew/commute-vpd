"""CPU run of experiments/lora_sgd/run.py's rows() and score() on a synthetic results dict of the run's shape, so a
scoring bug cannot crash the GPU run after its last state."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1] / "experiments" / "lora_sgd" / "run.py"


def _load():
    spec = importlib.util.spec_from_file_location("lora_sgd_run", RUN)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _measure(pgd: float, delta: float) -> dict:
    return {"pile": {"pgd_kl": pgd, "stochastic_kl": 1.0}, "routes": {"p_cells": {"T1": 0.99}, "shares": {"e_delta": 0.1}},
            "delta_change_frac": delta, "rel_norm_max": 0.05}


def _row(oo: float) -> dict:
    return {"blocks": {"pooled_ns": {"out_out_ratio": oo}}, "rescaling": {"pooled_ns": {"draws_out_out_ratio": 1.01}},
            "cos_init": {"mean": 0.98}}


def test_score_on_a_full_synthetic_run():
    mod = _load()
    C = mod.C
    g2 = {"ok": True}
    out = {"W0": {"pass": True, "pgd_floor": 0.075}, "G1": {"pass": True}, "rate": {"flagged": False},
           "series": {n: {str(s): _row(0.9) for s in C.ckpts} for n in ("adam", "sgd")},
           "states": {"adam": {str(s): {"nothing": _measure(1.1, 1.0), "least_squares": {**_measure(1.1, 0.54), "G2": g2}}
                               for s in C.adam_steps},
                      "sgd": {str(s): {"nothing": _measure(1.1, 1.0), "least_squares": {**_measure(1.2, 0.30), "G2": g2},
                                       "adjoint": _measure(1.0, 1e-5)} for s in C.sgd_steps}}}
    out["rows"] = mod.rows(out)
    sc = mod.score(out)
    assert len(out["rows"]) == 2 * len(C.adam_steps) + 3 * len(C.sgd_steps)
    assert sc["P1_outside_like_noise"]["pass"] and sc["P1_outside_like_noise"]["outcome"] == "parametrisation"
    assert sc["P2_read_factor_stays"]["pass"] and sc["P3_states_pass"]["pass"]
    assert sc["P4_delta_share"]["outcome"] == "lower"
    assert sc["C2_draws_at_null"]["pass"] and sc["C3_adjoint_keeps_delta"]["pass"] and sc["harness_G2"]["pass"]
    out["series"]["sgd"]["100"] = _row(0.1)
    out["states"]["sgd"]["1000"]["nothing"] = _measure(2.0, 1.0)  # above 1.5 (1 + f): fail
    out["rows"] = mod.rows(out)
    sc = mod.score(out)
    assert sc["P1_outside_like_noise"]["outcome"] == "optimiser" and not sc["P3_states_pass"]["pass"]


def test_score_on_a_partial_run():
    mod = _load()
    out = {"W0": {"pass": True, "pgd_floor": 0.075}, "series": {}, "states": {}}
    out["rows"] = mod.rows(out)
    assert set(mod.score(out)) == {"C1_w0", "G1", "rate_flagged"}
