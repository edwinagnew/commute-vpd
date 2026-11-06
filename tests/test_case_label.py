"""E112's pure pieces (src/commute/case_label.py) on hand-made numbers, with the edge cases the plan and
experiments/case_label/results/predictions.json list: no passing block, the last block only, no gain, did not learn. No model."""

from __future__ import annotations

import math

import numpy as np
import torch

from commute import case_label as CL

BARS = {"P1": 0.9, "P2": 0.9, "P3": 0.5}


def test_case_pairs_need_both_single_token_forms() -> None:
    vocab = [" the", " The", " cat", " Dog", " dog", " iPhone", " x", " X", "word", " THE", " a1", " A1"]
    pairs = CL.case_pairs(lambda ids: vocab[ids[0]], len(vocab))
    assert sorted(pairs) == [(1, 0), (3, 4), (7, 6)]  # " cat" has no capitalised form; "word" has no space; digits are out


def test_top_pairs_rank_by_rarer_form_and_exclude() -> None:
    pairs = [(1, 0), (3, 2), (5, 4), (7, 6)]
    counts = {0: 100, 1: 1, 2: 10, 3: 10, 4: 50, 5: 20, 7: 30}  # 6 never occurs
    assert CL.top_pairs(pairs, counts, 2, set()) == [(5, 4), (3, 2)]
    assert CL.top_pairs(pairs, counts, 2, {5}) == [(3, 2), (1, 0)]


def test_split_and_labels() -> None:
    pairs = [(2 * i + 1, 2 * i) for i in range(10)]
    tr, he = CL.split_pairs(pairs, 7, 0)
    assert len(tr) == 7 and len(he) == 3 and sorted(tr + he) == sorted(pairs)
    assert CL.split_pairs(pairs, 7, 0) == (tr, he)
    y = CL.real_y(pairs)
    assert all(y[c] == 1 and y[low] == 0 for c, low in pairs)
    toks = [t for p in tr for t in p]
    tw = CL.twin_y(toks, 2)
    assert sorted(tw) == sorted(toks) and sum(tw.values()) == len(toks) // 2
    ctrl = CL.control_y(pairs, 3)
    assert all(ctrl[c] == ctrl[low] for c, low in pairs)
    a, b = CL.pick_labels([50, 51, 52, 1], {1, 52}, 0)
    assert {a, b} == {50, 51}
    t = CL.table(y, 50, 51, 30)
    assert int(t[1]) == 50 and int(t[0]) == 51 and int(t[25]) == -1


def test_probe_positions_balanced_and_in_range() -> None:
    rows = torch.full((3, 30), 9, dtype=torch.long)
    rows[0, 10], rows[0, 12], rows[1, 15], rows[2, 4], rows[2, 27] = 1, 1, 0, 1, 0  # (1, 0): 1 capitalised, 0 lowercase
    r, t = CL.probe_positions(rows, [(1, 0)], 5, 16, 0)
    prev = rows[r, t - 1]
    assert int((prev == 1).sum()) == int((prev == 0).sum()) == 1  # one lowercase hit in range: one of each form
    assert bool(((t >= 5) & (t < 25)).all())


def test_first_passing_block() -> None:
    assert CL.first_passing_block([0.95, 0.95, 0.95, 0.99], [0.1, 0.35, 0.5, 0.5], 0.9, 0.3) == 1
    assert CL.first_passing_block([0.9, 0.5, 0.5, 0.5], [0.3, 0.0, 0.0, 0.0], 0.9, 0.3) == 0  # inclusive
    assert CL.first_passing_block([0.5, 0.6, 0.7, 0.8], [0.0, 0.1, 0.2, 0.3], 0.9, 0.3) is None  # no passing block
    assert CL.first_passing_block([0.5, 0.5, 0.5, 0.99], [0.0, 0.0, 0.0, 0.5], 0.9, 0.3) is None  # the last block only
    assert CL.first_passing_block([math.nan, 0.5, 0.5, 0.5], [math.nan, 0.0, 0.0, 0.0], 0.9, 0.3) is None
    assert CL.verdict(None, 1.0, 1.0, 1.0, BARS) == "unreadable"


def test_scored_step_and_did_not_learn() -> None:
    ck = (10, 30, 100, 300, 1000)
    assert CL.scored_step({10: 0.5, 30: 0.9}, ck, 0.9) == (30, True)
    assert CL.scored_step({10: 0.5, 30: 0.6, 100: 0.7, 300: 0.8, 1000: 0.85}, ck, 0.9) == (1000, False)
    # a state that did not learn is read as filed: P1 about 0.5 misses, "not through C"
    assert CL.verdict(1, 0.5, 0.95, 0.9, BARS) == "not through C"


def test_threshold_g() -> None:
    s = np.array([0.1, 0.2, 0.8, 0.9])
    th, sign = CL.fit_threshold(s, np.array([0, 0, 1, 1]))
    assert sign == 1 and 0.2 < th < 0.8
    th, sign = CL.fit_threshold(s, np.array([1, 1, 0, 0]))
    assert sign == -1 and (CL.apply_threshold(s, th, sign) == [1, 1, 0, 0]).all()
    th, sign = CL.fit_threshold(s, np.array([1, 1, 1, 1]))  # one class: the constant g
    assert th == -np.inf and (CL.apply_threshold(np.array([-5.0, 5.0]), th, sign) == 1).all()
    th, sign = CL.fit_threshold(s, np.array([0, 0, 0, 0]))
    assert (CL.apply_threshold(np.array([-5.0, 5.0]), th, sign) == 0).all()


def test_removed_share_and_no_gain() -> None:
    m0, ms = torch.zeros(4, dtype=torch.float64), torch.full((4,), 2.0, dtype=torch.float64)
    assert CL.removed_share(ms, torch.full((4,), 0.5, dtype=torch.float64), m0, 0.1) == 0.75
    assert CL.removed_share(ms, ms, m0, 0.1) == 0.0
    assert math.isnan(CL.removed_share(m0 + 0.1, m0, m0, 0.1))  # gain exactly at the floor: no gain
    assert math.isnan(CL.removed_share(m0 - 1.0, m0, m0, 0.1))  # a negative gain: no gain
    # no gain: P3 is nan and misses, verdict "not through C"
    assert CL.meets(0.95, 0.95, math.nan, BARS)["P3"] is False
    assert CL.verdict(1, 0.95, 0.95, math.nan, BARS) == "not through C"


def test_verdict_bars_are_inclusive_and_between_reads_not_through() -> None:
    assert CL.verdict(0, 0.9, 0.9, 0.5, BARS) == "reuses C"
    assert CL.verdict(0, 0.95, 0.95, 0.3, BARS) == "not through C"  # between the bars: as the rule says, no band


def test_do_nothing_and_reverse() -> None:
    assert CL.do_nothing_ok([0.1, -0.4, math.nan], 0.1)
    assert not CL.do_nothing_ok([0.05, 0.11], 0.1)
    assert CL.reverse_steps([(0, 0.99), (10, 0.7), (20, 0.6), (30, 0.5)], 0.6) == 20
    assert CL.reverse_steps([(0, 0.99), (10, 0.9)], 0.6) is None
