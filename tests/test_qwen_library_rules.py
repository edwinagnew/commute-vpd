"""Run 2's library rules (`commute.qwen_library`) on edge cases: no 15x save, a stopped or missing arm, ties at a bar,
an empty set, and the do-nothing state (a zero change must not pass "learned" on any new side)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute import qwen_library as L

PRED = Path(__file__).resolve().parents[1] / "experiments" / "qwen_pair_library" / "results" / "predictions_run2.json"
R = json.loads(PRED.read_text())


def measure(hits: int | None, n: int, value: float | None = None) -> dict[str, Any]:
    return {"hits": hits, "n": n, "value": value if value is not None else (hits or 0) / n}


def arm(family: str, saves: list[list[str]], trained: dict[str, Any], held: dict[str, Any] | None = None,
        w0_trained: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    keys = ["_".join(s) for s in saves]
    return {"family": family, "done": True, "saves": [{"labels": s, "step": i + 1} for i, s in enumerate(saves)],
            "states": {k: {"trained": trained, "held": held or measure(0, 256)} for k in keys},
            "w0": {"trained": w0_trained or trained}, **extra}


def test_tie_holds_and_one_below_misses() -> None:
    assert L.learned(arm("built", [["5x"], ["15x"]], measure(231, 256)), R["learned"])["learned"]
    assert not L.learned(arm("built", [["5x"], ["15x"]], measure(230, 256)), R["learned"])["learned"]
    assert L.learned(arm("template", [["last"]], measure(58, 64)), R["learned"])["learned"]
    assert not L.learned(arm("template", [["last"]], measure(57, 64)), R["learned"])["learned"]


def test_gold_ratio_tie_holds() -> None:
    a = arm("gold", [["15x"]], measure(None, 64, 1.0), w0_trained=measure(None, 64, 2.0))
    assert L.learned(a, R["learned"])["learned"]
    a = arm("gold", [["15x"]], measure(None, 64, 1.0001), w0_trained=measure(None, 64, 2.0))
    assert not L.learned(a, R["learned"])["learned"]


def test_no_15x_reads_the_last_save() -> None:
    got = L.learned(arm("built", [["5x"], ["last"]], measure(256, 256)), R["learned"])
    assert got["state"] == "last" and got["reached_15x"] is False and got["learned"]
    a = arm("built", [["5x", "15x"]], measure(0, 256))
    assert L.read_state(a) == ("5x_15x", True) and L.has_size(a, "5x") and L.has_size(a, "15x")


def test_stopped_missing_unfinished_and_empty_are_not_learned() -> None:
    stopped = arm("built", [["5x"]], measure(256, 256), stopped="control")
    unfinished = arm("built", [["5x"]], measure(256, 256)) | {"done": False}
    nothing = arm("built", [], measure(256, 256))
    for a in (stopped, None, unfinished, nothing):
        assert not L.learned(a, R["learned"])["learned"]
    assert not L.learned(arm("built", [["15x"]], measure(0, 0, 0.0)), R["learned"])["learned"]
    assert L.copy_held(stopped, R["copy_held"])["holds"] is None


def test_do_nothing_fails_learned_on_every_side() -> None:
    """The zero change reads W0's numbers (predictions_run2.json do_nothing): no side passes."""
    w0 = {"cipher": arm("built", [["last"]], measure(0, 256)), "invented": arm("built", [["last"]], measure(0, 256)),
          "template_shuf": arm("template", [["last"]], measure(0, 64)),
          "gold_shuf": arm("gold", [["last"]], measure(None, 64, 2.5), w0_trained=measure(None, 64, 2.5))}
    for name, a in w0.items():
        assert not L.learned(a, R["learned"])["learned"], name


def test_pair_fit_needs_both_learned_a_match_and_capacity() -> None:
    good, only5 = arm("built", [["5x"], ["15x"]], measure(256, 256)), arm("built", [["5x"], ["last"]], measure(256, 256))
    row = L.pair_row(good, only5, "fit", R)
    assert row["matched"] == {"5x": True, "15x": False} and row["fit_to_read"]
    assert not L.pair_row(good, only5, "dropped", R)["fit_to_read"]
    assert L.pair_row(good, good, "n/a", R)["fit_to_read"]
    assert not L.pair_row(good, None, "fit", R)["fit_to_read"]
    last_only = arm("built", [["last"]], measure(256, 256))
    assert not L.pair_row(good, last_only, "fit", R)["fit_to_read"]  # learned, but matched at no size


def test_table_shape() -> None:
    cap = {"copy_cipher": {"verdict": "fit"}, "facts": {"verdict": "fit"}}
    out = L.table({}, cap, R, (0, 1, 2))
    assert set(out) == {*L.PAIRS, "copy_held"} and all(out[p]["seeds_fit"] == [] for p in L.PAIRS)
