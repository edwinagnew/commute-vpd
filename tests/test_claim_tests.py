"""CPU check of E47's filed rules (experiments/claim_tests/rules.py) on their edge cases: ties on a line, a missing or
non-finite start, early stops, P < 0.9, empty pools, zero denominators. No model, no GPU."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments" / "claim_tests" / "rules.py"
_spec = importlib.util.spec_from_file_location("claim_tests_rules", PATH)
assert _spec is not None and _spec.loader is not None
R: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

F = 0.2  # lines 1.25 / 1.8
PASS, FAIL = 1.5 / 1.2, 1.5 * 1.2


def rec(complete: bool = True, **ratio: float) -> dict[str, Any]:
    return {"ratio": ratio, "kl": {s: r * 0.7 for s, r in ratio.items()}, "complete": complete}


def test_r_max_ties_stops_missing() -> None:
    assert R.r_max(rec(h0=PASS, h1=1.0), F) == "pass"  # a tie on the pass line passes
    assert R.r_max(rec(h0=FAIL), F) == "fail"  # a tie on the fail line fails
    assert R.r_max(rec(False, budget=9.0), F) == "fail"  # early stop
    assert R.r_max(rec(False, budget=1.0), F) == "unscored"  # missing start, no fail
    assert R.r_max(rec(h0=math.nan, h1=1.0), F) == "unscored"
    assert R.r_max(rec(h0=math.nan, h1=2.0), F) == "fail"  # a finite fail stands
    assert R.r_max(rec(), F) == "unscored"  # empty
    assert R.r_max(rec(h0=1.5), F) == "inconclusive"


def t1_states(r1n: float, r1c: float, rn: float, rcp: float) -> dict[str, Any]:
    return {"r1_nothing": rec(budget=r1n), "r1_copy": rec(budget=r1c), "rand32_nothing": rec(budget=rn),
            "rand32_copy": rec(budget=rcp)}


def test_t1_outcomes() -> None:
    assert R.t1(t1_states(9, 1.0, 2.5, 1.0), F)["outcome"] == "stands"
    assert R.t1(t1_states(9, 1.0, 1.5, 1.0), F)["outcome"] == "no_power_rand32"
    assert R.t1(t1_states(1.5, 1.0, 1.5, 1.0), F)["outcome"] == "no_power_both"
    assert R.t1(t1_states(9, 2.0, 2.5, 1.0), F)["outcome"] == "falls"
    assert R.t1(t1_states(9, 1.5, 2.5, 1.0), F)["outcome"] == "unsettled"  # copy inconclusive
    assert R.t1(t1_states(9, 1.2, 1.8, 1.2), F)["outcome"] == "stands"  # a do-nothing on the fail line fails
    s = t1_states(9, 1.0, 2.5, 1.0)
    s["rand32_copy"] = rec(False, budget=1.0)
    assert R.t1(s, F)["outcome"] == "unsettled"  # a copy with a missing start


def test_t1_separation_implied() -> None:
    # A pass and a fail always separate (pass * (1 + f) <= 1.5 < any fail), so the rule's separation never binds; at f = 0
    # both lines are 1.5 and a tie reads pass (faith_score.verdict checks the pass line first)
    assert R.t1(t1_states(9, 1.0, 1.5, 1.5), 0.0)["per_teacher"]["rand32"] == "no_power"
    assert R.t1(t1_states(9, 1.0, 1.51, 1.5), 0.0)["per_teacher"]["rand32"] == "stands"


def test_merge_combined_t1() -> None:
    a, b = rec(False, budget=9.0), rec(h2=1.6)
    m = R.merge(a, b)
    assert m["ratio"] == {"budget": 9.0, "h2": 1.6} and not m["complete"]
    assert R.r_max(m, F) == "fail"
    assert R.r_max(R.merge(rec(h0=1.0, h1=1.1), rec(False)), F) == "unscored"  # t1b missing h2


def test_t2_cap_and_absolute() -> None:
    def states(ctrl: float, r8: tuple[float, float], hi: tuple[float, float], c8: bool = True) -> dict[str, Any]:
        return {"adam_tt": rec(h0=ctrl), "lora8_tt": rec(c8, h0=r8[0], h1=r8[1]), "lora128_tt": rec(h0=hi[0], h1=hi[1])}

    out = R.t2(states(2.2, (1.2, 1.2), (1.3, 1.3)), F, "lora128_tt")
    assert (out["absolute"], out["cap"]) == ("stands", "no_cap_effect")
    assert R.t2(states(2.2, (1.3, 1.3), (2.0, 2.0)), F, "lora128_tt")["cap"] == "cap_artefact"
    assert R.t2(states(1.5, (1.3, 1.3), (2.0, 2.0)), F, "lora128_tt")["cap"] == "unscored"  # no power
    assert R.t2(states(2.2, (2.0, 2.0), (2.0, 2.0)), F, "lora128_tt")["cap"] == "unscored"  # rank 8 fails
    assert R.t2(states(2.2, (2.0, 2.0), (2.0, 2.0)), F, "lora128_tt")["absolute"] == "falls"
    assert R.t2(states(2.2, (1.3, 1.3), (1.6, 1.7)), F, "lora128_tt")["cap"] == "unsettled"  # 1.65 > 1.56
    assert R.t2(states(2.2, (1.3, 1.3), (1.56, 1.56)), F, "lora128_tt")["cap"] == "no_cap_effect"  # tie in the band
    out = R.t2(states(2.2, (1.3, 1.3), (1.3, 1.3), c8=False), F, "lora128_tt")
    assert (out["absolute"], out["cap"]) == ("unscored", "unscored")


def test_t2_combined() -> None:
    s = {"absolute": "stands", "cap": "no_cap_effect"}
    assert R.t2_combined(s, s, s) == {"absolute": "stands", "cap": "no_cap_effect"}
    assert R.t2_combined(s, {"absolute": "falls", "cap": "cap_artefact"}, s) == {"absolute": "stands", "cap": "cap_artefact"}
    assert R.t2_combined(s, s, {"absolute": "falls", "cap": "unscored"}) == {"absolute": "falls", "cap": "unsettled"}


def seed(a100: float | None, a1000: float | None, b10: float | None, b100: float | None) -> dict[str, Any]:
    ref = {"100": 0.42, "1000": 0.13, "10": 0.37}
    return {"keep_1000": 0.13,
            "a": {k: None if v is None else {"keep": v, "keep_ref": ref[k]} for k, v in (("100", a100), ("1000", a1000))},
            "b": {k: None if v is None else {"keep": v, "keep_ref": ref[k]} for k, v in (("10", b10), ("100", b100))}}


def test_t3() -> None:
    assert R.t3_seed(seed(0.43, 0.14, 0.37, 0.41)) == "stands"
    assert R.t3_seed(seed(0.43, 0.25, 0.37, 0.41)) == "falls"  # (a) differs by 0.12
    assert R.t3_seed(seed(0.43, 0.14, 0.15, 0.19)) == "falls"  # (b) near the step-1000 keep at both alphas
    assert R.t3_seed(seed(0.43, 0.14, 0.15, 0.41)) == "mixed"
    assert R.t3_seed(seed(None, 0.14, 0.37, 0.41)) == "mixed"  # unscored: stands is out
    assert R.t3_seed(seed(None, 0.25, 0.37, 0.41)) == "falls"  # a scored part meets falls
    assert R.t3_seed(seed(0.43, 0.14, None, 0.19)) == "mixed"  # (b) needs both alphas to fall
    assert R.t3({"0": seed(0.43, 0.14, 0.37, 0.41), "1": seed(0.43, 0.14, 0.37, 0.41)})["outcome"] == "stands"
    assert R.t3({"0": seed(0.43, 0.14, 0.37, 0.41), "1": seed(0.43, 0.25, 0.37, 0.41)})["outcome"] == "mixed"
    assert R.t3({"0": seed(0.43, 0.14, 0.37, 0.41)})["outcome"] == "mixed"  # one seed only


def test_t4() -> None:
    nothing = rec(h0=1.5, h1=1.4, h2=1.45)
    assert R.t4(nothing, rec(h0=1.9, h1=1.8, h2=1.85))["outcome"] == "stands"
    assert R.t4(nothing, rec(h0=1.5, h1=1.8, h2=1.85))["outcome"] == "falls"  # q = 1.0 at h0: a tie falls
    assert R.t4(nothing, rec(h0=1.6, h1=1.55, h2=1.6))["outcome"] == "unsettled"  # q < 1.1
    assert R.t4(nothing, rec(h0=1.66, h1=1.5, h2=1.6))["outcome"] == "unsettled"  # ranges tie at 1.5: overlap
    assert R.t4(nothing, rec(False, h0=1.9))["outcome"] == "unscored"
    zero = {"ratio": {"h0": 0.0}, "kl": {"h0": 0.0}, "complete": True}
    assert R.t4(zero, rec(h0=1.9))["outcome"] == "unscored"


def cell(p: float, *ratios: float, complete: bool = True) -> dict[str, Any]:
    return {"p": p, "rec": rec(complete, **dict(zip(("budget", "h0"), ratios, strict=False)))}


def test_t5() -> None:
    c = {"pass": cell(0.95, 1.0, 1.1), "fail": cell(0.95, 9.0, complete=False), "low": cell(0.5),
         "unscored": cell(0.95, 1.0, complete=False), "incon": cell(0.95, 1.5, 1.5)}
    v = {k: R.cell_pass(x, F) for k, x in c.items()}
    assert v == {"pass": "pass", "fail": "not", "low": "not", "unscored": "unscored", "incon": "not"}
    assert R.t5({"adam": {"x": "pass", "y": "not"}, "lora": {"x": "pass", "y": "not"}}, {"adam": 2, "lora": 2})["outcome"] == "falls"
    assert R.t5({"adam": {"x": "pass", "y": "not"}, "lora": {"x": "not", "y": "not"}}, {"adam": 2, "lora": 2})["outcome"] == "mixed"
    assert R.t5({"adam": {"x": "not", "y": "not"}, "lora": {"x": "not", "y": "not"}}, {"adam": 2, "lora": 2})["outcome"] == "stands"
    assert R.t5({"adam": {"x": "not", "y": "not"}, "lora": {"x": "not", "y": "not"}}, {"adam": 2, "lora": 1})["outcome"] == "unsettled"
    assert R.t5({"adam": {"x": "pass", "y": "not"}, "lora": {"x": "unscored", "y": "not"}}, {"adam": 2, "lora": 2})["outcome"] == "unsettled"
    assert R.t5({"adam": {"x": "pass"}, "lora": {"x": "pass"}}, {"adam": 1, "lora": 1})["outcome"] == "falls"  # falls first
    assert R.t5({"adam": {}, "lora": {}}, {"adam": 0, "lora": 0})["outcome"] == "unsettled"  # empty pool


@pytest.mark.parametrize("f", [0.01, 0.2088, 0.25])
def test_lines(f: float) -> None:
    assert R.r_max(rec(h0=1.5 / (1 + f)), f) == "pass" and R.r_max(rec(h0=1.5 * (1 + f)), f) == "fail"
