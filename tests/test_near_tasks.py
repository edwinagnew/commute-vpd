"""P2.74's twin rule (`near_tasks.separates`) on its edge cases, run before filing (E292)."""

import math

from commute.near_tasks import TASKS, separates, states, steps_to_half


def call(*a: list[float]) -> str:
    return separates(*a)["call"]


def test_three_seeds() -> None:
    assert call([0.6, 0.62, 0.61], [0.4, 0.41, 0.42], [0.0, 0.01, 0.0], [0.01, 0.0, 0.02]) == "real above"
    assert call([0.4, 0.41, 0.42], [0.6, 0.62, 0.61], [0.0, 0.01, 0.0], [0.01, 0.0, 0.02]) == "real below"


def test_one_seed_inside_its_null_fails() -> None:
    assert call([0.6, 0.62, 0.61], [0.4, 0.61, 0.42], [0.0, 0.05, 0.0], [0.0, 0.0, 0.0]) == "does not separate"  # seed 1: 0.01 < 0.05


def test_mixed_signs() -> None:
    assert call([0.6, 0.4], [0.4, 0.6], [0.0, 0.0], [0.0, 0.0]) == "does not separate"


def test_tie_is_not_beyond() -> None:
    assert call([0.6], [0.5], [0.1], [0.0]) == "does not separate"  # |d| = null exactly: not strictly beyond
    assert call([0.5], [0.5], [0.0], [0.0]) == "does not separate"  # d = 0 with a zero null


def test_one_seed() -> None:
    r = separates([0.6], [0.5], [0.0], [0.01])
    assert r["call"] == "real above" and r["n_seeds"] == 1


def test_no_reading() -> None:
    assert call([], [], [], []) == "no reading"
    assert call([0.6, 0.5], [0.4], [0.0, 0.0], [0.0, 0.0]) == "no reading"
    assert call([0.6, math.nan], [0.4, 0.4], [0.0, 0.0], [0.0, 0.0]) == "no reading"
    assert call([0.6], [0.4], [math.inf], [0.0]) == "no reading"


def test_steps_to_half() -> None:
    assert steps_to_half([-5.0, -2.0, 0.0, 1.0]) == 1  # gain 6 at the end; 3 reached at step 1
    assert steps_to_half([-5.0, -5.0]) is None and steps_to_half([-5.0, -6.0]) is None


def test_registry() -> None:
    every = [s for t in TASKS.values() for s in states(t, "all")]
    assert len(every) == len(set(every))
    for t in TASKS.values():
        assert t.twin_of is None or TASKS[t.twin_of].kind == t.kind == "trigger"
        assert all(p.name.endswith(".pt") for p in t.saved.values())
