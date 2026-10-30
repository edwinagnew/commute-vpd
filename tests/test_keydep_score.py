"""E56's filed rules (src/commute/keydep_score.py) on hand-made numbers, with every edge case the predictions name: a
missing or unreadable state, a nan, ties at each bound, a zero ratio denominator, k50 undefined, same-state seeds."""

from __future__ import annotations

import math
from typing import Any

from commute import keydep_score as KS

OLD, NEW, ALL = ["s0", "s1"], ["s2", "s3", "s4"], ["s0", "s1", "s2", "s3", "s4"]


def states(**cols: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    """states(N16={"elicit": [...], "cipher": [...]}, ...) -> {"elicit_s0": {"readable": True, "N16": ...}, ...}."""
    out: dict[str, dict[str, Any]] = {}
    for m, by_kind in cols.items():
        for kind, vals in by_kind.items():
            for i, v in enumerate(vals):
                out.setdefault(f"{kind}_s{i}", {"readable": True})[m] = v
    return out


def b2(n16: tuple[list[float], list[float]], k: tuple[list[float], list[float]]) -> dict[str, dict[str, Any]]:
    return states(N16={"elicit": n16[0], "cipher": n16[1]}, k50_index={"elicit": k[0], "cipher": k[1]})


def test_p1_passes_when_neither_separates_including_k50_beyond_grid() -> None:
    st = b2(([0.01, -0.02], [0.0, 0.015]), ([8, 8], [8, 8]))  # k50 undefined everywhere: index 8, gap 0, range 0
    out = KS.p1(st, OLD, 0.2)
    assert out["status"] == "pass" and out["k50_index"]["verdict"] == "none"


def test_p1_fails_on_either_direction() -> None:
    assert KS.p1(b2(([0.5, 0.52], [0.1, 0.12]), ([6, 6], [6, 6])), OLD, 0.2)["status"] == "fail"
    rev = KS.p1(b2(([0.1, 0.1], [0.1, 0.1]), ([3, 3], [6, 7])), OLD, 0.2)  # cipher needs more neurons: still a separation
    assert rev["k50_index"]["verdict"] == "b_higher" and rev["status"] == "fail"


def test_p1_ties_at_bounds() -> None:
    # |gap| == range: no separation
    assert KS.p1(b2(([0.5, 1.0], [0.0, 0.5]), ([6, 6], [6, 6])), OLD, 0.2)["N16"]["verdict"] == "none"
    # |gap| == min_gap exactly (0.375 - 0.125, range 0): separates
    assert KS.p1(b2(([0.375, 0.375], [0.125, 0.125]), ([6, 6], [6, 6])), OLD, 0.25)["N16"]["verdict"] == "a_higher"


def test_p1_unreadable_on_missing_unreadable_or_nan_state() -> None:
    st = b2(([0.0, 0.0], [0.0, 0.0]), ([6, 6], [6, 6]))
    del st["cipher_s1"]
    assert KS.p1(st, OLD, 0.2)["status"] == "unreadable"
    st = b2(([0.0, 0.0], [0.0, 0.0]), ([6, 6], [6, 6]))
    st["elicit_s0"]["readable"] = False  # under the gain floor: its k50 index (8) must not be scored
    assert KS.p1(st, OLD, 0.2)["status"] == "unreadable"
    st = b2(([math.nan, 0.0], [0.0, 0.0]), ([6, 6], [6, 6]))
    assert KS.p1(st, OLD, 0.2)["status"] == "unreadable"


def test_p1_five_seeds() -> None:
    st = b2(([0.02, 0.0, 0.03, -0.01, 0.01], [0.0, 0.01, 0.02, 0.0, -0.02]), ([6, 7, 6, 6, 7], [7, 7, 6, 7, 7]))
    assert KS.p1(st, ALL, 0.2)["status"] == "pass"


def rs(r: tuple[list[float], list[float]], lo: tuple[list[float], list[float]]) -> dict[str, dict[str, Any]]:
    return states(r={"elicit": r[0], "cipher": r[1]}, S_lo={"elicit": lo[0], "cipher": lo[1]})


def test_p2_e54_numbers_pass_with_ratio_near_5_6() -> None:
    out = KS.p2(rs(([0.7726, 0.7578], [0.2414, 0.2119]), ([0.2462, 0.2522], [0.2185, 0.1751])), OLD, 0.2, 3)
    assert out["status"] == "pass" and abs(out["ratio"] - 5.62) < 0.01


def test_p2_ratio_tie_at_bar_passes_and_below_fails() -> None:
    # S_lo gap 0.125 (range 0, under 0.2: no separation) -> denominator 0.125; r gap 0.25 -> ratio exactly 2
    tie = KS.p2(rs(([0.75, 0.75], [0.5, 0.5]), ([0.25, 0.25], [0.125, 0.125])), OLD, 0.2, 2)
    assert tie["ratio"] == 2.0 and tie["status"] == "pass"
    low = KS.p2(rs(([0.75, 0.75], [0.5, 0.5]), ([0.25, 0.25], [0.125, 0.125])), OLD, 0.2, 3)
    assert low["status"] == "fail" and low["r"]["verdict"] == "a_higher" and low["S_lo"]["verdict"] == "none"


def test_p2_zero_denominator_is_unreadable() -> None:
    out = KS.p2(rs(([0.8, 0.8], [0.2, 0.2]), ([0.2, 0.2], [0.2, 0.2])), OLD, 0.2, 3)
    assert out["status"] == "unreadable" and out["ratio"] is None


def test_p2_fails_when_s_lo_separates_or_r_reversed() -> None:
    assert KS.p2(rs(([0.8, 0.8], [0.2, 0.2]), ([0.6, 0.6], [0.2, 0.2])), OLD, 0.2, 3)["status"] == "fail"
    rev = KS.p2(rs(([0.2, 0.2], [0.8, 0.8]), ([0.2, 0.21], [0.2, 0.2])), OLD, 0.2, 3)
    assert rev["status"] == "fail" and rev["ratio"] < 0


def test_p2_unreadable_on_nan() -> None:
    assert KS.p2(rs(([0.8, math.nan], [0.2, 0.2]), ([0.2, 0.21], [0.2, 0.2])), OLD, 0.2, 3)["status"] == "unreadable"


def p3_states(c: list[float], co: list[float]) -> dict[str, dict[str, Any]]:
    return states(r={"cipher": [0.0, 0.0, *c], "cipher_own": [0.0, 0.0, *co]})


def test_p3_pass_fail_and_tie() -> None:
    assert KS.p3(p3_states([0.24, 0.21, 0.22], [0.26, 0.24, 0.25]), NEW, 0.1, [])["status"] == "pass"
    assert KS.p3(p3_states([0.24, 0.21, 0.40], [0.26, 0.24, 0.25]), NEW, 0.1, [])["status"] == "fail"
    assert KS.p3(p3_states([0.25, 0.25, 0.25], [0.25, 0.25, 0.375]), NEW, 0.125, [])["status"] == "fail"  # == tol fails


def test_p3_same_state_and_unreadable() -> None:
    st = p3_states([0.24, 0.21, 0.22], [0.24, 0.24, 0.25])
    out = KS.p3(st, NEW, 0.1, ["s2"])
    assert out["status"] == "pass" and out["same_state"] == ["s2"]
    assert KS.p3(st, NEW, 0.1, NEW)["status"] == "uninformative"
    st["cipher_own_s3"]["readable"] = False
    assert KS.p3(st, NEW, 0.1, [])["status"] == "unreadable"


def test_branches() -> None:
    ok = {"status": "pass"}
    p2_low = {"status": "fail", "r": {"verdict": "a_higher"}, "S_lo": {"verdict": "none"}, "ratio": 2.1}
    assert "headline" in KS.branch(ok, ok, ok)
    assert "small but directional" in KS.branch(ok, p2_low, ok)
    assert "narrowed to steering" in KS.branch({"status": "fail"}, ok, ok)
    assert "suspect" in KS.branch(ok, ok, {"status": "fail"})
    assert KS.branch(ok, {"status": "unreadable"}, ok).startswith("unreadable")
    assert "job 2 may run" in KS.branch_job1(ok)


def test_nan_read_back_from_json_is_unreadable() -> None:
    st = b2(([0.0, 0.0], [0.0, 0.0]), ([6, 6], [6, 6]))
    st["cipher_s0"]["N16"] = "nan"  # how run.py's dump writes a nan; job 2 reads job 1's states back from JSON
    assert KS.p1(st, OLD, 0.2)["status"] == "unreadable"
