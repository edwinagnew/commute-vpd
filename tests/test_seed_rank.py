"""commute.seed_rank on hand-made seeds: separated sides, ties, the edge cases the prior-matched rules meet."""

from __future__ import annotations

import math

from commute.seed_rank import mann_whitney, midranks


def test_midranks_share_ties() -> None:
    assert midranks([3.0, 1.0, 3.0, 2.0]) == [3.5, 1.0, 3.5, 2.0]


def test_separated_sides() -> None:
    assert mann_whitney([4.0, 5.0, 6.0], [1.0, 2.0, 3.0]) == (9.0, 2 / 20)
    u, p = mann_whitney([float(x) for x in range(10, 20)], [float(x) for x in range(10)])
    assert u == 100 and math.isclose(p, 2 / math.comb(20, 10))
    assert math.isclose(mann_whitney([float(x) for x in range(3, 13)], [0.0, 1.0, 2.0])[1], 2 / 286)


def test_one_overlap_10_vs_3_misses_alpha() -> None:
    """Run B's alpha 0.01 asks for complete separation at 10 seeds against 3: one swapped pair reads 4 / 286."""
    a = [1.5, *[float(x) for x in range(3, 12)]]
    assert math.isclose(mann_whitney(a, [0.0, 1.0, 2.0])[1], 4 / 286)


def test_constant_and_degenerate() -> None:
    """A reading constant over every seed (the prior) gives p = 1; an empty side or a nan gives nan."""
    assert mann_whitney([1.0, 1.0, 1.0], [1.0, 1.0]) == (3.0, 1.0)
    assert all(math.isnan(x) for x in mann_whitney([], [1.0]))
    assert all(math.isnan(x) for x in mann_whitney([1.0, math.nan], [0.0, 2.0]))
