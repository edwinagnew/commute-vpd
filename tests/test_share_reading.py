"""commute.share_reading's size search and placement rule (cell P3.10), on made-up numbers: the edge cases filed in
experiments/qwen_share_reference/results/predictions.json ("edge_cases")."""

from __future__ import annotations

import math

from commute import share_reading as SR

NAN = float("nan")


def draws(levels: dict[str, list[tuple[float, float]]], usable: bool = True) -> list[dict]:
    return [{"level": lv, "k00": k, "share": s, "usable": usable} for lv, ds in levels.items() for k, s in ds]


CURVE_DRAWS = draws({"low": [(0.4, 0.98), (0.4, 0.96)], "mid": [(0.8, 0.93), (0.8, 0.91)],
                     "high": [(1.6, 0.86), (1.6, 0.84)]})


def test_search_scale_matches():
    s = SR.size_search(lambda b: 1e-4 * b * b, 0.5, 1.0, 2, 1.05, 8)
    assert s["matched"] and abs(s["f"] / 0.5 - 1) <= 0.05


def test_search_whole_count_brackets():
    s = SR.size_search(lambda k: 0.03 * k, 0.5, 28, 1, 1.05, 8, cap=1000, whole=True)
    assert s["matched"] and float(s["x"]).is_integer()


def test_search_overshoot_at_one_stops_unmatched():
    s = SR.size_search(lambda k: 5.0 * k, 0.5, 28, 1, 1.05, 8, cap=1000, whole=True)
    assert not s["matched"] and s["x"] == 1.0


def test_search_cap_reached_below_target():
    s = SR.size_search(lambda k: 1e-4 * k, 0.5, 28, 1, 1.05, 8, cap=100, whole=True)
    assert not s["matched"] and s["x"] == 100.0


def test_search_nonpositive_counts_below_and_nan_above():
    seen = []

    def f(b: float) -> float:
        seen.append(b)
        return -0.01 if b < 50 else NAN if b > 5000 else 1e-4 * b * b

    s = SR.size_search(f, 0.5, 1.0, 2, 1.05, 12)
    assert seen[1] == 100.0 and s["matched"]


def test_search_no_target():
    assert SR.size_search(lambda b: b, 0.0, 1.0, 2, 1.05, 8)["evals"] == []


def test_curve_and_at():
    cv = SR.curve(CURVE_DRAWS)
    assert [round(x, 6) for x, _ in cv] == [0.4, 0.8, 1.6] and math.isclose(cv[1][1], 0.92)
    assert math.isclose(SR.at(cv, math.sqrt(0.4 * 0.8)) or 0, 0.945)
    assert SR.at(cv, 0.38) == cv[0][1] and SR.at(cv, 0.3) is None and SR.at(cv, 1.75) == cv[-1][1]


def test_curve_needs_two_usable_per_level_and_two_levels():
    one = draws({"low": [(0.4, 0.98), (0.4, 0.96)], "mid": [(0.8, 0.93)]})
    assert len(SR.curve(one)) == 1 and SR.band(SR.curve(one), one) is None
    assert SR.curve([]) == [] and SR.band([], []) is None
    nan_low = SR.curve(draws({"low": [(0.4, NAN), (0.4, 0.9)], "mid": [(0.8, 0.9), (0.8, 0.9)]}))
    assert len(nan_low) == 1 and math.isclose(nan_low[0][0], 0.8)
    assert SR.curve(draws({"low": [(0.4, 0.9), (0.4, 0.9)], "mid": [(0.8, 0.9), (0.8, 0.9)]}, usable=False)) == []


def test_place_classes_and_ties():
    cv = SR.curve(CURVE_DRAWS)
    bd = SR.band(cv, CURVE_DRAWS)
    assert bd is not None and math.isclose(bd, 0.01)
    assert SR.place(0.95, True, 0.8, cv, bd)["class"] == "above"
    assert SR.place(0.92 + 0.01, True, 0.8, cv, bd)["class"] == "inside"  # a gap equal to the band is inside
    assert SR.place(0.90, True, 0.8, cv, bd)["class"] == "below"
    assert SR.place(0.92, True, 0.8, cv, 0.0)["class"] == "inside"  # band 0: only gap 0 is inside
    assert SR.place(0.921, True, 0.8, cv, 0.0)["class"] == "above"
    for share, ok, k00, b in [(None, True, 0.8, bd), (NAN, True, 0.8, bd), (0.95, False, 0.8, bd),
                              (0.95, True, 3.0, bd), (0.95, True, 0.8, None), (0.95, True, None, bd)]:
        assert SR.place(share, ok, k00, cv, b)["class"] == "unreadable"


def test_do_nothing_share_one_reads_above():
    """The do-nothing state (share 1 by construction) would read 'above' a curve under 1: it is the path's control,
    not a candidate; the built change is read through the least-squares split, as the random draws are."""
    cv = SR.curve(CURVE_DRAWS)
    assert SR.place(1.0, True, 0.8, cv, SR.band(cv, CURVE_DRAWS))["class"] == "above"


def test_outcome():
    assert SR.outcome(["above"] * 12 + ["inside"] * 3, 9) == "above random"
    assert SR.outcome(["above"] * 11 + ["inside"] * 4, 9) == "mixed"
    assert SR.outcome(["inside"] * 8 + ["unreadable"] * 7, 9) == "unreadable"
    assert SR.outcome(["inside"] * 8 + ["below"] * 2 + ["unreadable"] * 5, 9) == "like random"
    assert SR.outcome(["below"] * 15, 9) == "below random"
    assert SR.outcome([], 9) == "unreadable"
    assert SR.outcome(["above"] * 29 + ["inside"] * 7, 27) == "above random"
    assert SR.outcome(["above"] * 28 + ["inside"] * 8, 27) == "mixed"
