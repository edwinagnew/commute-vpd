"""P2.87's filed reading (results/predictions.json "rules"), stdlib only. Descriptive: which rival the subspace updates'
shares sit with, per the card; no verdict on reuse or new.

  task mean   the mean at-1 share over the task's seeds whose at-1 state exists and reads a finite share (gain >= 1 at
              the reader); None when no seed does ("no comparable null")
  reading     over the tasks with a mean (n_scored): "too few tasks" when n_scored < min_tasks; else "rival a" when more
              than half of them have mean >= a_min, "rival b" when more than half have mean <= b_max, else "between".
              a_min > b_max, so at most one rival holds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def task_mean(shares: Sequence[float | None]) -> float | None:
    ok = [float(s) for s in shares if s is not None and math.isfinite(float(s))]
    return sum(ok) / len(ok) if ok else None


def reading(means: dict[str, float | None], a_min: float, b_max: float, min_tasks: int) -> dict[str, object]:
    assert a_min > b_max and min_tasks >= 1
    scored = {t: m for t, m in means.items() if m is not None}
    n = len(scored)
    n_a, n_b = sum(m >= a_min for m in scored.values()), sum(m <= b_max for m in scored.values())
    if n < min_tasks:
        call = "too few tasks"
    elif n_a > n / 2:
        call = "rival a"
    elif n_b > n / 2:
        call = "rival b"
    else:
        call = "between"
    return {"call": call, "n_scored": n, "no_comparable_null": sorted(set(means) - set(scored)), "means": means}


def edge_cases(a_min: float, b_max: float, min_tasks: int) -> None:
    """The rule on its edge cases (run before filing; `python rules.py`)."""
    nan = math.nan
    assert task_mean([]) is None and task_mean([None, nan]) is None
    assert task_mean([0.5]) == 0.5 and task_mean([0.5, nan, None, 0.7]) == 0.6
    assert reading({}, a_min, b_max, min_tasks)["call"] == "too few tasks"
    assert reading({t: None for t in "abcdefgh"}, a_min, b_max, min_tasks)["call"] == "too few tasks"
    hi, lo, mid = a_min, b_max, (a_min + b_max) / 2  # ties at the bars count for their rival
    assert reading(dict(zip("abcdefgh", [hi] * 5 + [lo] * 3)), a_min, b_max, min_tasks)["call"] == "rival a"
    assert reading(dict(zip("abcdefgh", [lo] * 5 + [hi] * 3)), a_min, b_max, min_tasks)["call"] == "rival b"
    assert reading(dict(zip("abcdefgh", [hi] * 4 + [lo] * 4)), a_min, b_max, min_tasks)["call"] == "between"
    assert reading(dict(zip("abcdefgh", [mid] * 8)), a_min, b_max, min_tasks)["call"] == "between"
    few = dict(zip("abcdefgh", [hi] * (min_tasks - 1) + [None] * (9 - min_tasks)))
    assert reading(few, a_min, b_max, min_tasks)["call"] == "too few tasks"
    print("edge cases pass")


if __name__ == "__main__":
    edge_cases(0.65, 0.42, 5)
