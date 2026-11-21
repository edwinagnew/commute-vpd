"""E241's filed rules (experiments/gating_vpd/score.py) on made-up numbers: the edge cases of results/predictions.json
P1 and P2 (ties, the do-nothing in place of every item, unscored and missing items, a degenerate band). No model."""

from __future__ import annotations

from typing import Any

from commute.jobs import load_experiment

SC: Any = load_experiment("gating_vpd", "score")
REUSE, NEW = SC.REUSE, SC.NEW


def item(S: float | None, scored: bool = True, h: tuple[float, float] = (0.0, 0.0)) -> dict[str, Any]:
    return {"scored": scored, "S": S, "S00": S, "S_c": S, "means": {"kl_fm": 1.0, "f": 0.4},
            "halves": [{"S": h[0], "S_c": h[0]}, {"S": h[1], "S_c": h[1]}]}  # fmt: skip


def test_order_holds_rival_b_no_call():
    v = {"x2": 0.6, "x10": 0.6, "moved": None, "rebuilt": 0.5, "new_few": 0.1, "new_thin": 0.2}
    r = SC.order_rule(v, REUSE, NEW, 0.05)
    assert r["verdict"] == "holds" and abs(r["gap"] - 0.3) < 1e-12 and r["left_out"] == ["moved"]
    assert SC.order_rule(v, REUSE, NEW, 0.5)["verdict"] == "no call"
    assert SC.order_rule(v | {"rebuilt": 0.2}, REUSE, NEW, 0.0)["verdict"] == "rival b"  # tie: gap 0


def test_order_not_read():
    none = dict.fromkeys((*REUSE, *NEW))
    assert not SC.order_rule(none, REUSE, NEW, 0.0)["read"]  # the do-nothing everywhere: cannot pass
    assert not SC.order_rule(none | {"x2": 0.9, "new_few": 0.1}, REUSE, NEW, 0.0)["read"]  # a new item unscored
    assert not SC.order_rule({"x2": 0.9}, REUSE, NEW, 0.0)["read"]  # missing items


def test_below_rule():
    assert SC.below_rule([0.5, None], [0.1, 0.2, None], 0.05)["verdict"] == "holds"
    assert SC.below_rule([0.5], [0.5], 0.0)["verdict"] == "fails"  # tie
    assert SC.below_rule([0.5], [0.48], 0.05)["verdict"] == "no call"
    assert not SC.below_rule([None], [0.1], 0.0)["read"] and not SC.below_rule([0.5], [None], 0.0)["read"]


def test_band_and_summary():
    assert SC.band([(0.1, 0.1)]) == (0.0, True) and SC.band([]) == (0.0, True)
    assert abs(SC.band([(0.1, 0.3), (None, 0.2)])[0] - 0.2) < 1e-12
    items = {"x2": item(0.6, h=(0.58, 0.62)), "x10": item(0.6), "moved": item(None, False), "rebuilt": item(0.5),
             "new_few": item(0.1), "new_thin": item(0.2), "random_copy_size_1": item(None, False), "random_copy_kl_1": item(0.0),
             "do_nothing_copy": item(None, False)}  # fmt: skip
    idx = {"p": {"reuse": ["a", "b"], "new": ["c"]}, "q": {"reuse": ["d"], "new": ["missing"]}}
    items |= {"a": item(0.3), "b": item(None, False), "c": item(0.2), "d": item(0.9)}
    out = SC.summary(items, idx)
    assert abs(out["band_S"] - 0.04) < 1e-12 and out["P1"]["verdict"] == "holds" and out["P2"]["verdict"] == "holds"
    assert out["D1_S"]["n_read"] == 1 and out["D1_S"]["reuse_above"] == 1 and out["missing"] == []
    assert "do_nothing_copy" not in out["table"]
