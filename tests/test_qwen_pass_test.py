"""CPU check of E68's filed rules (experiments/qwen_pass_test/rules.py) at their edge cases, on the row sums of synthetic
reads. The siblings load by path under the names amend2.py imports, inside patch.dict so `run` and `job` do not outlive it."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

EXP = Path(__file__).resolve().parents[1] / "experiments"
NAN = float("nan")


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mods() -> Any:
    with patch.dict(sys.modules):
        for n in ("job", "amend1", "run", "amend2"):
            _load(n, EXP / "qwen_maps" / f"{n}.py")
        rules = _load("rules", EXP / "qwen_pass_test" / "rules.py")
        yield rules, _load("e68_run", EXP / "qwen_pass_test" / "run.py")


def _good() -> dict[str, Any]:
    return {"kd": 1.0, "k0": 0.1, "lo": (0.01, "scored"), "up": (0.06, "scored"), "rnd": (0.005, "scored"),
            "matched": True, "cl": 0.3, "cu": 0.5, "w": 0.002}


def _teeth(rules: Any, **kw: Any) -> dict[str, Any]:
    return rules.teeth_rule(**(_good() | kw))


def test_teeth_edges(mods: Any) -> None:
    rules = mods[0]
    assert _teeth(rules)["result"] == "holds"
    assert _teeth(rules, kd=0.15)["result"] == "no teeth"  # KD == 1.5 K0: inclusive
    assert _teeth(rules, matched=False)["result"] == "no teeth"
    assert _teeth(rules, cl=0.05)["result"] == "no teeth"  # c == 0.05: inclusive
    assert _teeth(rules, rnd=(NAN, "unscored"))["result"] == "no teeth"
    assert _teeth(rules, kd=NAN)["result"] == "unreadable"
    assert _teeth(rules, w=NAN)["result"] == "unreadable"
    assert _teeth(rules, up=(NAN, "nan"))["result"] == "unreadable"
    gap = 0.06 / 0.5 - 0.01 / 0.3
    assert _teeth(rules, w=gap / 2)["result"] == "no teeth"  # the gap must exceed 2w strictly


def test_labels(mods: Any) -> None:
    rules = mods[0]
    t = {"holds": True, "rho_lower": 0.01, "rho_upper": 0.12, "w": 0.002}
    assert rules.label_rule(0.118, t, 0.005) == "placed"  # exactly rho_upper - w: inclusive
    assert rules.label_rule(0.012, t, 0.005) == "lost"  # exactly the floor + w: inclusive
    assert rules.label_rule(0.05, t, 0.005) == "partial"
    assert rules.label_rule(0.01, t | {"rho_upper": 0.011}, 0.005) == "lost"  # placed and lost both: lost
    assert rules.label_rule(NAN, t, 0.005) == "unreadable"
    assert rules.label_rule(0.05, {"holds": False}, 0.005) == "unscored"
    assert rules.rho(0.1, 0.05) != rules.rho(0.1, 0.05)  # c <= 0.05: undefined
    assert rules.faithful(0.15, 0.1) == "faithful" and rules.faithful(0.16, 0.1) == "not faithful"
    assert rules.faithful(NAN, 0.1) == "unreadable"


def _read(**sums: float) -> dict[str, Any]:
    """Two rows of T = 11, each summing to 11 × the value given."""
    return {"T": 11, "sums": {k: [v * 11, v * 11] for k, v in sums.items()}}


def test_unscored_r2_and_control(mods: Any) -> None:
    rules = mods[0]
    u = {"W0": _read(ci=0.1, K=0.12, nll=1.0, clean=0.9, ft=0.8), "do_nothing": _read(K=0.12)}
    assert rules.r2_of(u, "do_nothing")[1] == "unscored"  # KD - K0 = 0.02 < 0.05
    c = rules.control_reproduce(u)
    assert {"K0", "KD", "r2_do_nothing"} <= set(c["missed"]) and "r2_lower" in c["pending"] and not c["holds"]
    good = {"W0": _read(ci=rules.REF_K["K0"], K=rules.REF_K["KD"]), "do_nothing": _read(K=rules.REF_K["KD"])}
    c = rules.control_reproduce(good)
    assert not c["missed"] and c["pending"] and not c["holds"]
    assert rules.control_reproduce({"W0": _read(ci=NAN, K=rules.REF_K["KD"])})["missed"] == ["K0"]


def test_gain_and_reading_a(mods: Any) -> None:
    rules, run = mods
    w0 = _read(ci=0.1, K=1.0, nll=2.0, clean=1.5, ft=1.4)
    assert math.isclose(rules.gain_kept(w0, _read(nll=1.95))[0], 0.5)
    assert rules.gain_kept(_read(nll=2.0, clean=1.5, ft=1.46), _read(nll=1.0))[1] == "unscored"
    assert rules.gain_kept(w0, None)[1] == "nan"
    u = {"W0": _read(ci=0.1, K=1.0, nll=2.0, clean=1.5, ft=1.46)} | {
        k: _read(K=0.9, nll=1.9) for k in ("lower", "upper", "A_do_nothing", "A_lower", "A_upper")}
    assert rules.reading_a(u)["result"] == "unreadable"
    assert rules.reading_a({})["result"] == "not read"
    u["W0"] = w0
    u["A_upper"] = {"T": 11, "sums": {"K": [9.9, 9.9], "nll": [20.0, 20.0]}}
    assert rules.reading_a(u)["result"] == "separates"  # no row spread: half-width 0
    uniform = torch.zeros(1, 4, 7)
    assert math.isclose(run.nll(uniform, torch.zeros(1, 4, dtype=torch.long)), 3 * math.log(7), rel_tol=1e-5)


def test_amendment2_reads_reproduce_the_filed_numbers(mods: Any) -> None:
    """E44 Amendment 2's own reads through these rules: every filed reference met, teeth at the plan's 0.125 against w 0.002."""
    rules = mods[0]
    R = EXP / "qwen_maps" / "results" / "amend2"
    c, t, S = (json.loads((R / f"{n}.json").read_text()) for n in ("controls", "t10", "S"))
    zeros = [0.0] * 128
    w0 = {"T": 256, "sums": {"ci": c["W0_maths"]["sums"]["ci"], "K": c["KD_10"]["sums"]["K"], "nll": zeros, "clean": zeros, "ft": zeros}}
    u = {"W0": w0, "do_nothing": {"T": 256, "sums": {"K": w0["sums"]["K"]}}, "S": S["10"], "search": t["search"],
         "upper": t["upper"], "lower": t["lower"], "random": t["random"]}
    c = rules.control_reproduce(u)
    assert c["holds"], c
    sc = rules.score_update(u)
    json.dumps(sc)
    assert sc["teeth"]["holds"] and 0.0015 < sc["teeth"]["w"] < 0.0025
    assert math.isclose(sc["teeth"]["rho_upper"] - sc["teeth"]["rho_lower"], 0.1247, abs_tol=1e-3)
    assert sc["states"]["upper"]["gain_status"] == "unscored" and sc["states"]["upper"]["faithful"] == "not faithful"
    assert rules.score({"u1": u, "u2": {}})["u2"] == {}
