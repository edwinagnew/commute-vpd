"""E131: the filed rule (experiments/qwen_new_decomp/score.py) on its edge cases, and run.py's host-side g store; no
model, no decomposition."""

from __future__ import annotations

import math

import torch

from commute import qwen_vpd as Q
from commute.jobs import load_experiment

score = load_experiment("qwen_new_decomp", "score")
run = load_experiment("qwen_new_decomp", "run")


def verdict_of(new: list[float], old: list[float]) -> str:
    return score.verdict(score.ratio_with_interval(new, old, b=2000))


def test_clear_drop_passes_and_clear_rise_misses():
    old = [0.10 + 0.01 * i for i in range(20)]
    assert verdict_of([0.5 * x for x in old], old) == "pass"
    assert verdict_of([2.0 * x for x in old], old) == "miss"


def test_do_nothing_is_a_miss():
    """new = old row for row: ratio exactly 1 and every resample 1, so the strict rule cannot pass."""
    old = [0.1, 0.2, 0.3, 0.4]
    r = score.ratio_with_interval(old, old)
    assert r["ratio"] == 1.0 and r["lo95"] == r["hi95"] == 1.0
    assert score.verdict(r) == "miss"


def test_interval_upper_end_at_one_is_a_miss():
    r = {"reason": None, "ratio": 0.9, "lo95": 0.8, "hi95": 1.0}
    assert score.verdict(r) == "miss"
    assert score.verdict(r | {"hi95": math.nextafter(1.0, 0.0)}) == "pass"


def test_ratio_below_one_with_wide_interval_misses():
    old = [1.0, 0.01, 1.0, 0.01]
    new = [0.01, 1.0, 0.9, 0.01]  # mean 0.48 vs 0.505: below 1, but resamples swing both ways
    r = score.ratio_with_interval(new, old)
    assert r["ratio"] < 1 < r["hi95"] and score.verdict(r) == "miss"


def test_unreadable_cases():
    assert verdict_of([0.1, 0.2], [0.0, 0.0]) == "unreadable"  # zero old mean
    assert verdict_of([0.1, 0.2, 0.3], [0.0, 0.0, 1.0]) == "unreadable"  # zero only in some resamples
    assert verdict_of([0.1], [0.2]) == "unreadable"  # one row
    assert verdict_of([0.1, 0.2], [0.2, 0.3, 0.4]) == "unreadable"  # unequal counts
    assert verdict_of([0.1, math.nan], [0.2, 0.3]) == "unreadable"  # non-finite
    assert verdict_of([], []) == "unreadable"  # empty


def test_score_reads_each_prediction_and_marks_a_missing_side():
    rows = [0.1 + 0.01 * i for i in range(8)]
    half = [0.5 * x for x in rows]
    measured = {("old", "G"): {"ci_kl": rows, "adv_kl": rows, "active": rows},
                ("new", "G"): {"ci_kl": half, "adv_kl": rows, "active": rows},
                ("old", "G-think"): {"ci_kl": rows}}
    out = score.score(measured)
    p = out["predictions"]
    assert p["P1"]["verdict"] == "pass" and p["P2"]["verdict"] == "miss"
    assert p["P3"]["verdict"] == p["P4"]["verdict"] == "unreadable"
    assert set(out["ratios"]) == {f"{r}/{n}" for r in score.ROW_SETS for n in score.NUMBERS}


def test_sparse_g_round_trip_and_active_count(monkeypatch):
    monkeypatch.setattr(run, "DEV", "cpu")
    gen = torch.Generator().manual_seed(0)
    g = {}
    for s in Q.SITES:
        v = torch.rand(1, 3, Q.C[Q.parse_site(s)[1]], generator=gen)
        g[s] = torch.where(v > 0.99, v, torch.zeros(()))  # ~1% non-zero, as g is sparse
    sparse = run.SparseG.of(g)
    back = sparse.dense()
    assert all(torch.equal(back[s], g[s]) for s in Q.SITES)
    want = sum(float((v > 0.995).sum()) for v in g.values()) / 3
    assert sparse.active(0.995) == want
