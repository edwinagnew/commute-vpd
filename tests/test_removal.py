"""E70's filed rules and pure pieces (src/commute/removal.py) on hand-made numbers, with the edge cases
predictions_e70.json names: an empty side, all nan, ties, a nan band, a no-call. No model."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from commute import removal as RM

nan = math.nan


def test_remove_same_code_for_rank_one_and_rank_r() -> None:
    h = torch.randn(5, 6)
    u1 = RM.random_basis(6, 1, [1])
    assert torch.allclose(RM.remove(h, u1), h - (h @ u1[:, 0])[:, None] * u1[:, 0][None], atol=1e-6)
    ur = RM.random_basis(6, 3, [2])
    assert torch.allclose(ur.T @ ur, torch.eye(3), atol=1e-5)
    assert (RM.remove(h, ur) @ ur).abs().max() < 1e-5
    assert torch.allclose(RM.remove(h, torch.eye(6)), torch.zeros(5, 6), atol=1e-6)


def test_random_basis_is_reproducible_and_key_dependent() -> None:
    assert torch.equal(RM.random_basis(8, 2, [70, 2, 0, 1]), RM.random_basis(8, 2, [70, 2, 0, 1]))
    assert not torch.equal(RM.random_basis(8, 2, [70, 2, 0, 1]), RM.random_basis(8, 2, [70, 2, 0, 2]))


def test_ridge_recovers_a_linear_label_and_its_intercept() -> None:
    x = torch.randn(2000, 4, dtype=torch.float64)
    true = torch.tensor([1.0, -2.0, 0.5, 0.0], dtype=torch.float64)
    w, b = RM.fit_ridge(x + 3.0, x @ true + 7.0, 1e-3)  # off-centre x: the intercept has to absorb it
    assert torch.allclose(w, true, atol=0.02)
    assert abs(float(b) - (7.0 - 3.0 * float(true.sum()))) < 0.1
    assert abs(float(RM.unit_basis(w).norm()) - 1) < 1e-5


def test_centred_basis_rank_and_span() -> None:
    cols = torch.randn(10, 4)
    u = RM.centred_basis(cols)
    assert u.shape == (10, 3)
    centred = cols - cols.mean(1, keepdim=True)
    assert (RM.remove(centred.T, u)).abs().max() < 1e-4  # the centred columns lie in the span
    with pytest.raises(AssertionError):
        RM.centred_basis(torch.randn(10, 1))


def test_seen_values_and_majority() -> None:
    labels = np.array([3] * 5 + [4] * 4 + [6] * 6)
    assert RM.seen_values(labels, 5) == [3, 6]
    assert RM.majority(labels) == pytest.approx(6 / 15)
    assert math.isnan(RM.majority(np.array([], np.int64)))


def test_band_uses_absolute_values_and_drops_nan() -> None:
    assert RM.band_of([-1.0, 1.0]) == 1.0
    assert RM.band_of([-2.0, nan, 2.0]) == 2.0
    assert RM.band_of(list(range(1, 22))) == pytest.approx(20.0)  # numpy's linear rule: 95% of the way along 21 values
    assert math.isnan(RM.band_of([]))
    assert math.isnan(RM.band_of([nan, nan]))
    assert RM.max_band([0.1, nan, 0.3]) == 0.3
    assert math.isnan(RM.max_band([nan]))
    assert math.isnan(RM.max_band([]))


def test_r1_stops_only_when_no_probe_has_teeth_anywhere() -> None:
    assert not RM.r1({"ridge": [0.1, 2.0], "softmax": [0.0, 0.0]}, {"ridge": [1.0, 1.0], "softmax": [1.0, 1.0]})["stop"]
    assert not RM.r1({"ridge": [0.0], "softmax": [2.0]}, {"ridge": [1.0], "softmax": [1.0]})["stop"]
    assert RM.r1({"ridge": [1.0], "softmax": [0.5]}, {"ridge": [1.0], "softmax": [1.0]})["stop"]  # a tie is within
    assert RM.r1({"ridge": [-5.0]}, {"ridge": [1.0]})["stop"]  # a rise is not a loss
    assert RM.r1({"ridge": [nan, 3.0]}, {"ridge": [1.0, nan]})["stop"]  # nan on either side is no teeth
    assert RM.r1({}, {})["stop"]


def test_r2_block_holds_is_strict_and_reversal_is_mirrored() -> None:
    assert RM.r2_block([0.9, 0.8], [0.1, 0.2], 0.5)["call"] == "holds"
    tie = RM.r2_block([0.7, 0.8], [0.1, 0.2], 0.5)
    assert tie["gap"] == pytest.approx(0.5)
    assert RM.r2_block([0.75], [0.25], 0.5)["call"] == "neither"  # exactly at the band: does not exceed it
    assert RM.r2_block([0.1, 0.2], [0.9, 0.8], 0.5)["call"] == "reversal"
    assert RM.r2_block([0.5, 0.6], [0.4, 0.55], 0.0)["call"] == "neither"  # overlapping sides at a zero band


def test_r2_block_no_calls_and_counts() -> None:
    assert RM.r2_block([], [0.1], 0.1)["call"] == "no call"  # empty side
    assert RM.r2_block([0.9], [], 0.1)["call"] == "no call"
    r = RM.r2_block([nan, nan], [0.1], 0.1)  # all nan
    assert r["call"] == "no call" and r["nan_P"] == 2 and r["n_P"] == 0 and math.isnan(r["gap"])
    r = RM.r2_block([0.9], [0.1], nan)  # a nan band: the gap is reported, no call made
    assert r["call"] == "no call" and r["gap"] == pytest.approx(0.8)
    r = RM.r2_block([0.9, nan], [0.1, nan, nan], 0.1)  # a nan state is left out and counted; one element per side stands
    assert r["call"] == "holds" and (r["n_P"], r["nan_P"], r["n_N"], r["nan_N"]) == (1, 1, 1, 2)


def _rows(*calls: str) -> list[dict[str, str]]:
    return [{"call": c} for c in calls]


def test_r2_verdict_covers_every_branch() -> None:
    n = "neither"
    assert RM.r2_verdict(_rows(n, "holds", n, n, n, n))["verdict"] == "separates"
    assert RM.r2_verdict(_rows("no call", "holds", n, n, n, "holds"))["verdict"] == "separates"  # a holding block decides
    assert RM.r2_verdict(_rows(n, n, n, n, n, "holds"))["verdict"] == "output-side, no call"
    assert RM.r2_verdict(_rows(n, n, n, n, n, n))["verdict"] == "does not separate"
    assert RM.r2_verdict(_rows(n, n, n, n, n, "no call"))["verdict"] == "does not separate"  # only blocks 1-5 can separate
    assert RM.r2_verdict(_rows("no call", n, n, n, n, n))["verdict"] == "no call"
    assert RM.r2_verdict(_rows("no call", n, n, n, n, "holds"))["verdict"] == "no call"
    assert RM.r2_verdict(_rows(*["no call"] * 6))["verdict"] == "no call"
    rev = RM.r2_verdict(_rows("reversal", "holds", n, n, n, "reversal"))
    assert rev["verdict"] == "separates" and rev["reversal_at"] == [1, 6] and rev["holds_at"] == [2]


def test_lookup_reads_odd_rows_counts_unseen_wrong_and_breaks_ties_low() -> None:
    reading = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 3.0])
    answer = np.array([5, 5, 6, 6, 5, 7])  # even rows fit 1 -> 5, 2 -> 6; odd rows read 1, 2, and an unseen 3
    out = RM.lookup_accuracy(reading, answer)
    assert out["acc"] == pytest.approx(2 / 3) and out["unseen"] == pytest.approx(1 / 3)
    assert out["floor"] == pytest.approx(1 / 3)  # even rows' mode 5 (5, 6, 5) hits one odd row in three
    tied = RM.lookup_accuracy(np.array([1.0, 1.0, 1.0, 1.0]), np.array([9, 4, 4, 4]))  # even answers 9, 4 tie: 4
    assert tied["acc"] == 1.0


def test_lookup_edge_cases() -> None:
    empty = RM.lookup_accuracy(np.array([]), np.array([], np.int64))
    assert math.isnan(empty["acc"]) and math.isnan(empty["floor"])
    assert math.isnan(RM.lookup_accuracy(np.array([1.0]), np.array([3]))["acc"])  # one row: no odd half


def test_pattern_basis_zeroes_the_cross_covariance() -> None:
    torch.manual_seed(0)
    x, y = torch.randn(500, 6), torch.randn(500).double()
    x[:, 2] += y.float()
    h = RM.remove(x, RM.pattern_basis(x, y))
    assert abs(float(RM.pattern_basis(x, y)[2, 0])) > 0.9
    assert ((h - h.mean(0)).double().T @ (y - y.mean()) / 500).norm() < 1e-5
    with pytest.raises(AssertionError):
        RM.pattern_basis(x, torch.ones(500).double())


def test_erased_is_strict_and_nan_is_not_erased() -> None:
    assert RM.erased(0.36, 0.36, 1.0) and not RM.erased(0.68, 0.36, 1.0) and not RM.erased(nan, 0.36, 1.0)


@pytest.mark.parametrize(
    ("teeth", "calls", "branch"),
    [
        ([False, False, False], ["holds", "reversal", "no call"], "closed"),  # no teeth: nothing is read, whatever the calls
        ([True, True, False], ["holds", "neither", "neither"], "separates"),
        ([False, True, False], ["reversal", "holds", "neither"], "separates"),  # a reversal at a block with no teeth is not read
        ([True, True, False], ["holds", "reversal", "neither"], "cannot serve"),  # holds and reversal: never both branches
        ([True, False, False], ["neither", "holds", "neither"], "cannot serve"),  # holds at a block with no teeth is not read
        ([False, False, True], ["neither", "neither", "holds"], "cannot serve"),  # the output side alone
        ([True, False, True], ["neither", "neither", "no call"], "cannot serve"),  # the last block unreadable decides nothing
        ([True, True, False], ["no call", "neither", "neither"], "no call"),  # one side empty or a zero gain at a read block
        ([True, True, False], ["no call", "holds", "neither"], "separates"),
        ([False, False, True], ["neither", "neither", "no call"], "no call"),  # the only read block is unreadable
    ],
)
def test_one_direction_branch(teeth: list[bool], calls: list[str], branch: str) -> None:
    assert RM.one_direction_branch(teeth, [{"call": c} for c in calls])["branch"] == branch


def test_top_pc_after_is_the_largest_direction_at_right_angles_to_the_basis() -> None:
    torch.manual_seed(0)
    x = torch.randn(2000, 5) * torch.tensor([5.0, 3.0, 1.0, 0.5, 0.1]) + 7.0  # off-centre: the mean must not count
    a = torch.eye(5)[:, :1]
    v = RM.top_pc_after(x, a)
    assert abs(float(v[1, 0])) > 0.99 and abs(float(a[:, 0] @ v[:, 0])) < 1e-6
    assert abs(float(RM.top_pc_after(x, 2 * a)[1, 0])) > 0.99  # a basis slightly off unit length is renormalised
    with pytest.raises(AssertionError):
        RM.top_pc_after(torch.ones(10, 1), torch.eye(1))  # zero variance left


def test_a_direction_at_right_angles_to_the_covariance_has_no_covariance() -> None:
    torch.manual_seed(0)
    y = torch.randn(1000).double()
    x = torch.randn(1000, 6) * 3
    x[:, 2] += 2 * y.float()
    a = RM.pattern_basis(x, y)
    v = RM.top_pc_after(x, a)
    assert RM.abs_cov(x, v, y) < 1e-3 * RM.abs_cov(x, a, y)
    assert RM.abs_cov(x, torch.eye(6)[:, 2:3], y) == pytest.approx(float(y.var()) * 2, rel=0.2)


def test_variance_share_sums_to_one_over_a_basis() -> None:
    h = torch.randn(300, 4)
    assert sum(RM.variance_share(h, torch.eye(4)[:, i : i + 1]) for i in range(4)) == pytest.approx(1.0, rel=1e-5)


def test_class_share_sees_a_curve_a_covariance_misses() -> None:
    labels = np.array([-2, -1, 0, 1, 2] * 20)
    curve = (labels.astype(float)) ** 2
    assert RM.class_share(curve, labels) == pytest.approx(1.0)
    assert np.corrcoef(curve, labels)[0, 1] == pytest.approx(0.0, abs=1e-9)  # no linear tie at all
    noise = np.random.default_rng(0).normal(size=len(labels))
    assert RM.class_share(noise, labels) < 0.15
    assert RM.class_share(curve, np.zeros(len(labels), int)) == pytest.approx(0.0)  # one class: nothing lies between
    assert math.isnan(RM.class_share(np.ones(5), np.arange(5)))  # zero variance
    assert math.isnan(RM.class_share(np.array([]), np.array([], int)))


def test_refit_accuracy_reads_through_a_harmless_removal_and_not_a_harmful_one() -> None:
    torch.manual_seed(0)
    c = torch.randint(0, 5, (2000,)).double()
    x = torch.randn(2000, 4).double()
    x[:, 0] = c
    y, h = c[:1500], x[:1500]
    read, want = x[1500:], c[1500:].numpy()
    assert RM.refit_accuracy(h, y, read, want, torch.eye(4, dtype=torch.float64)[:, 1:2], 1e-6) == 1.0
    assert RM.refit_accuracy(h, y, read, want, torch.eye(4, dtype=torch.float64)[:, :1], 1e-6) < 0.5


def test_above_midpoint_is_strict_and_nan_misses() -> None:
    assert RM.above_midpoint(0.9, 0.36, 1.0)
    assert not RM.above_midpoint(0.75, 0.5, 1.0)  # a tie misses, against `not erased`, which would pass it
    assert not RM.above_midpoint(0.3, 0.36, 1.0)
    assert not RM.above_midpoint(nan, 0.36, 1.0)


def test_same_reading_is_one_row_inclusive_and_needs_the_same_floor() -> None:
    assert RM.same_reading([0.9975, 1.0], [0.998, 1.0], 0.3575, 0.3575, 2000)  # exactly one row, despite rounding in 0.0005
    assert not RM.same_reading([0.997, 1.0], [0.998, 1.0], 0.3575, 0.3575, 2000)
    assert not RM.same_reading([1.0], [1.0], 0.3575, 0.358, 2000)
    assert not RM.same_reading([nan], [1.0], 0.3575, 0.3575, 2000)


def test_reproduced_is_inclusive_and_nan_misses() -> None:
    assert RM.reproduced([0.0, 0.01], [1.0, 1.0], 0.01)
    assert not RM.reproduced([0.0, 0.011], [1.0, 1.0], 0.01)
    assert not RM.reproduced([nan], [1.0], 0.01)
    assert not RM.reproduced([0.0], [nan], 0.01)


def _branch(calls: list[str], p_a: list[float], p_v: list[float], class_v: list[float] | None = None, read_at: tuple[int, ...] = (1, 2)) -> dict:
    rows = [{"call": c, "band": 0.125} for c in calls]
    return RM.no_count_branch(rows, p_a, p_v, class_v if class_v is not None else [0.1] * len(calls), read_at, 0.5)


@pytest.mark.parametrize(
    ("calls", "p_a", "p_v", "class_v", "branch"),
    [
        (["holds", "neither"], [0.5, 0.5], [0.5, 0.45], None, "size explains"),
        (["neither", "neither"], [0.5, 0.5], [0.2, 0.45], None, "count-specific at this size"),
        (["reversal", "neither"], [0.5, 0.5], [0.2, 0.45], None, "count-specific at this size"),  # N over P is no reproduced gap
        (["holds", "neither"], [0.5, 0.5], [0.5, 0.2], None, "mixed"),  # one block each way: no overall claim
        (["holds", "neither"], [0.5, 0.5], [0.5, 0.45], [0.5, 0.1], "carries the classes"),  # at the bar: carrying, no size claim
        (["holds", "neither"], [0.5, 0.5], [0.5, 0.45], [nan, 0.1], "carries the classes"),  # a nan never claims size
        (["holds", "neither"], [0.5, 0.5], [0.5, 0.2], [0.9, 0.1], "count-specific at this size"),  # a carrier does not undo it
        (["holds", "no call"], [0.5, 0.5], [0.5, nan], None, "size explains"),
        (["neither", "no call"], [0.5, nan], [0.5, nan], None, "no call"),
        (["no call", "no call"], [nan, nan], [nan, nan], None, "no call"),
        (["neither", "neither"], [0.5, 0.5], [0.375, 0.5], None, "unresolved"),  # exactly the band: not specific
        (["neither", "neither"], [nan, 0.5], [nan, 0.5], None, "unresolved"),  # a nan mean is not specific
        (["neither", "neither"], [0.2, 0.2], [0.8, 0.8], None, "unresolved"),  # v costs P more than a_b does, R2 not holding
    ],
)
def test_no_count_branch(calls: list[str], p_a: list[float], p_v: list[float], class_v: list[float] | None, branch: str) -> None:
    assert _branch(calls, p_a, p_v, class_v)["branch"] == branch


def test_no_count_branch_reads_only_the_named_blocks_and_covers_an_empty_list() -> None:
    assert _branch(["holds", "neither"], [0.5, 0.5], [0.5, 0.45], read_at=(2,))["branch"] == "unresolved"
    out = _branch(["neither", "neither"], [0.5, 0.5], [0.2, 0.45], read_at=(1, 2))
    assert out["by_block"] == {1: "specific", 2: "unresolved"}
    assert _branch(["holds"], [0.5], [0.5], read_at=())["branch"] == "unresolved"
