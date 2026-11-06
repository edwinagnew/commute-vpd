"""E112's pure pieces (src/commute/case_label.py) on hand-made numbers, with the edge cases the plan and
experiments/case_label/results/predictions.json list: no passing block, the last block only, no gain, did not learn. No model."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from commute import case_label as CL
from commute import removal as RM

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


def test_pair_basis_spans_both_and_removes_both() -> None:
    a, b = torch.tensor([2.0, 0.0, 0.0, 0.0]), torch.tensor([1.0, 1.0, 0.0, 0.0])
    u = CL.pair_basis(a, b)
    assert u.shape == (4, 2) and torch.allclose(u.T @ u, torch.eye(2), atol=1e-6)
    x = torch.tensor([[3.0, -2.0, 5.0, 1.0]])
    assert torch.allclose(RM.remove(x, u), torch.tensor([[0.0, 0.0, 5.0, 1.0]]), atol=1e-6)  # both a and b taken out
    with pytest.raises(AssertionError, match="parallel"):
        CL.pair_basis(a, 3 * a)  # parallel: no rank-2 basis


def _two_d_set(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """y in {0, 1}; x (n, 6) Gaussian noise with y written along two axes, separable in the plane they span."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n,), generator=g)
    x = torch.randn(n, 6, generator=g, dtype=torch.float64)
    x[:, 0] += 3 * (2 * y - 1)
    x[:, 1] += 2 * (2 * y - 1)
    return x, y


def test_inlp_reaches_chance_on_a_two_d_separable_set() -> None:
    x, y = _two_d_set(4000, 0)
    xr, yr = _two_d_set(2000, 1)
    q, accs, reached = CL.inlp(x, y, xr, yr, 1e-3, 0.55, 8)
    assert reached and 1 <= q.shape[1] <= 2 and len(accs) == q.shape[1] + 1
    assert accs[0] > 0.99 and accs[-1] <= 0.55 and all(a > 0.55 for a in accs[:-1])
    assert torch.allclose(q.T @ q, torch.eye(q.shape[1], dtype=q.dtype), atol=1e-10)
    assert float(q[2:].abs().max()) < 0.1  # the basis lies in the plane of the two signal axes


def test_inlp_edge_cases_cap_and_nothing_to_remove() -> None:
    x, y = _two_d_set(2000, 2)
    q, accs, reached = CL.inlp(x, y, x, y, 1e-3, 0.55, 0)  # cap hit at once: incomplete, the accuracy reported
    assert not reached and q.shape == (6, 0) and len(accs) == 1 and accs[0] > 0.99
    noise = torch.randn(3000, 6, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    q, accs, reached = CL.inlp(noise[:2000], y, noise[2000:], y[:1000], 1e-3, 0.55, 8)  # already at chance: empty basis
    assert reached and q.shape == (6, 0) and len(accs) == 1


def test_merge_bases_drops_dependent_columns() -> None:
    e = torch.eye(5, dtype=torch.float64)
    a = e[:, :2]
    b = torch.stack([(e[:, 0] + e[:, 1]) / 2**0.5, e[:, 3]], 1)  # its first column lies in a's span
    u = CL.merge_bases([a, b], 1e-3)
    assert u.shape == (5, 3) and u.dtype == torch.float32
    assert torch.allclose(u.T @ u, torch.eye(3), atol=1e-6)
    proj = u.double() @ u.double().T
    assert torch.allclose(proj @ e[:, [0, 1, 3]], e[:, [0, 1, 3]], atol=1e-6) and float((proj @ e[:, 4]).norm()) < 1e-6
    assert CL.merge_bases([a[:, :0], b[:, :0]], 1e-3).shape == (5, 0)  # both empty: no basis


def _blocks_set(n: int, seed: int, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(2, n, 6) residuals for two blocks and labels in 0..m-1: block 0 carries the label on axis 0, block 1 is noise."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, m, (n,), generator=g)
    x = torch.randn(2, n, 6, generator=g, dtype=torch.float64)
    x[0, :, :m] += 4.0 * torch.nn.functional.one_hot(y, m).double()
    return x, y


def test_value_probes_m2_is_e112s_score_over_a_half() -> None:
    h, y = _blocks_set(3000, 0, 2)
    hh, yh = _blocks_set(1000, 1, 2)
    W, B, acc = CL.value_probes(h, y, hh, yh, 2, 1e-3)
    assert W.shape == (2, 2, 6) and B.shape == (2, 2)
    for i in range(2):
        w, b = RM.fit_ridge(h[i], y, 1e-3)  # E112's probe, one per block, read as score > 0.5
        assert torch.allclose(W[i, 1], w, atol=1e-10) and abs(float(B[i, 1] - b)) < 1e-10
        assert torch.allclose(W[i, 0], -w, atol=1e-10) and abs(float(B[i, 0] - (1 - b))) < 1e-10  # the complement
        e112 = float((((hh[i] @ w + b) > 0.5).long() == yh).double().mean())
        assert acc[i] == e112
    assert acc[0] > 0.95 and abs(acc[1] - 0.5) < 0.1


def test_value_probes_many_values_and_edges() -> None:
    h, y = _blocks_set(4000, 2, 4)
    hh, yh = _blocks_set(1000, 3, 4)
    _, _, acc = CL.value_probes(h, y, hh, yh, 4, 1e-3)
    assert acc[0] > 0.95 and acc[1] < 0.4
    keep = y != 3  # value 3 has no Pile position: never the argmax, its read windows count wrong
    _, _, acc3 = CL.value_probes(h[:, keep], y[keep], hh, yh, 4, 1e-3)
    assert abs(acc3[0] - float((yh != 3).double().mean())) < 0.03
    W, B, _ = CL.value_probes(h[:, keep], y[keep], hh, yh, 4, 1e-3)
    assert float(W[:, 3].abs().max()) == 0.0 and float(B[:, 3].abs().max()) == 0.0
    nan = CL.value_probes(h[:, :0], y[:0], hh, yh, 4, 1e-3)[2]  # no value has a position
    assert all(math.isnan(a) for a in nan)
    assert all(math.isnan(a) for a in CL.value_probes(h, y, hh[:, :0], yh[:0], 4, 1e-3)[2])  # no read window


def test_fit_ridge_columns_match_single_fits() -> None:
    x = torch.randn(500, 5, dtype=torch.float64)
    y = torch.randn(500, 3, dtype=torch.float64)
    w, b = RM.fit_ridge(x, y, 1e-3)
    for j in range(3):
        wj, bj = RM.fit_ridge(x, y[:, j], 1e-3)
        assert torch.allclose(w[:, j], wj, atol=1e-12) and abs(float(b[j] - bj)) < 1e-12
