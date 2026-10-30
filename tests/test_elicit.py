"""E54's value rule, key draw and filed rules (src/commute/elicit.py) on hand-made numbers, including every edge case the
plan names: an empty or one-element kind, a nan, a tie at each bound, a state under its floor."""

from __future__ import annotations

import math

import torch

from commute import elicit as EL

V = 1000  # the rest of each row spread thin (~5e-4), below the band and the key-specific bar
BAND, RANKS, RATIO = (1e-2, 1e-1), (2, 20), 10.0
TH = {"min_gap": 0.2, "p2_max_S": 0.3, "p3_tol": 0.1}


def dist_rows(spec: list[dict[int, float]]) -> torch.Tensor:
    """Rows with the given token probabilities, the rest spread evenly (about 5e-4 each)."""
    rows = []
    for d in spec:
        r = torch.full((V,), 0.0, dtype=torch.float64)
        for t, p in d.items():
            r[t] = p
        rest = [t for t in range(V) if t not in d]
        r[rest] = (1 - sum(d.values())) / len(rest)
        rows.append(r)
    return torch.stack(rows)


def test_value_is_rank_two_key_specific_word() -> None:
    plain = torch.ones(V, dtype=torch.bool)
    plain[9] = False
    # key 0: top token 1 (rank 1, excluded), 2 at rank 2 in band and specific -> 2
    # key 3: its best in-band token 4 is common to every key ("the"-like) -> 5 (rank 3)
    # key 6: rank-2 token is the key itself (6), next is non-plain 9 -> none left in band -> no value
    # key 7: rank 2 is 8 at 0.2, above the band -> no value
    d = dist_rows([{1: 0.5, 2: 0.08, 4: 0.05}, {1: 0.5, 4: 0.09, 5: 0.06}, {1: 0.5, 6: 0.09, 9: 0.08, 4: 0.05},
                   {1: 0.5, 8: 0.2, 4: 0.05}])
    out = EL.pick_values(d, [0, 3, 6, 7], plain, BAND, RANKS, RATIO)
    assert out[0]["value"] == 2 and out[0]["rank"] == 2
    assert out[3]["value"] == 5 and out[3]["rank"] == 3
    assert 6 not in out and 7 not in out


def test_value_band_edges_inclusive() -> None:
    plain = torch.ones(V, dtype=torch.bool)
    d = dist_rows([{1: 0.5, 2: 0.1}, {1: 0.5, 3: 0.01}, {1: 0.5}])
    out = EL.pick_values(d, [10, 11, 0], plain, BAND, RANKS, RATIO)
    assert out[10]["value"] == 2 and out[11]["value"] == 3


def test_draw_keys_unique_values_and_short_pool() -> None:
    q = {1: {"value": 50}, 2: {"value": 50}, 3: {"value": 51}, 4: {"value": 52}}
    keys = EL.draw_keys(q, [4, 3, 2, 1, 9], k=10, seed=0)
    assert len(keys) == 3 and len({q[k]["value"] for k in keys}) == 3
    assert keys == sorted(keys, key=[4, 3, 2, 1, 9].index)
    assert EL.draw_keys(q, [4, 3, 2, 1], k=2, seed=0) == EL.draw_keys(q, [4, 3, 2, 1], k=2, seed=0)
    assert EL.draw_keys({}, [1, 2], k=32, seed=0) == []


def test_matched_step_ties_lower() -> None:
    assert EL.matched_step({1: 1.0, 2: 3.0, 3: 5.0}, 2.0) == 1
    assert EL.matched_step({5: 9.0, 1: 1.0}, 100.0) == 5


def test_separation_edges() -> None:
    assert EL.separation([], [1.0, 2.0], 0.2)["verdict"] == "unreadable"
    assert EL.separation([1.0], [0.0, 0.1], 0.2)["verdict"] == "unreadable"
    assert EL.separation([1.0, math.nan], [0.0, 0.1], 0.2)["verdict"] == "unreadable"
    assert EL.separation([1.0, None], [0.0, 0.1], 0.2)["verdict"] == "unreadable"
    # gap 0.5, spread 0.5: tie at the range bound fails (strict)
    assert EL.separation([1.0, 0.5], [0.25, 0.25], 0.2)["verdict"] == "none"
    # gap exactly 0.2 with zero spread separates (>= min_gap); 0.19 does not
    assert EL.separation([0.5, 0.5], [0.3, 0.3], 0.2)["verdict"] == "a_higher"
    assert EL.separation([0.49, 0.49], [0.3, 0.3], 0.2)["verdict"] == "none"
    assert EL.separation([0.1, 0.1], [0.8, 0.8], 0.2)["verdict"] == "b_higher"


def state(r: float | None, S: float | None, ok: bool = True) -> dict[str, object]:
    return {"r": r, "S": S, "readable": ok}


def test_score_headline_and_edges() -> None:
    st = {"elicit_s0": state(0.8, 0.1), "elicit_s1": state(0.75, 0.15), "cipher_s0": state(0.3, 0.12),
          "cipher_s1": state(0.28, 0.1), "cipher_own_s0": state(0.32, 0.1), "cipher_own_s1": state(0.3, 0.1)}
    sc = EL.score(st, TH)
    assert (sc["P1"]["status"], sc["P2"]["status"], sc["P3"]["status"]) == ("pass", "pass", "pass")
    assert sc["branch"].startswith("VPD's route split")
    # one S above the bound, no separation: P2 fails on the bound only, flagged
    sc = EL.score({**st, "elicit_s0": state(0.8, 0.35)}, TH)
    assert sc["P2"]["status"] == "fail" and "bound only" in sc["branch"]
    # a state under its floor: P1 unreadable, the tree says stop
    sc = EL.score({**st, "cipher_s1": state(0.28, 0.1, ok=False)}, TH)
    assert sc["P1"]["status"] == "unreadable" and sc["branch"].startswith("unreadable")
    # P3: a difference exactly at the tolerance fails (strict; 0.125 is exact in binary); nan r is unreadable
    exact = {**TH, "p3_tol": 0.125}
    tie = {**st, "cipher_s0": state(0.375, 0.12), "cipher_own_s0": state(0.5, 0.1), "cipher_s1": state(0.375, 0.1),
           "cipher_own_s1": state(0.375, 0.1)}
    assert EL.score(tie, exact)["P3"]["status"] == "fail"
    assert EL.score({**tie, "cipher_own_s0": state(0.49, 0.1)}, exact)["P3"]["status"] == "pass"
    sc = EL.score({**st, "cipher_own_s1": state(math.nan, 0.1)}, TH)
    assert sc["P3"]["status"] == "unreadable"
    # reversed: cipher r above elicit fails P1
    sc = EL.score({**st, "elicit_s0": state(0.05, 0.1), "elicit_s1": state(0.05, 0.1)}, TH)
    assert sc["P1"]["verdict"] == "b_higher" and sc["P1"]["status"] == "fail"


def test_floor() -> None:
    assert EL.floor_of(3.0, 3.01, 0.05) == 0.05
    assert math.isclose(EL.floor_of(3.0, 3.5, 0.05), 1.0)
