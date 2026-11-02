"""E64's filed rules (src/commute/readout.py) on hand-made numbers, with the edge cases predictions.json names: an empty
M, ties at every bound, a key with no window, no other key's window, an unfit value, no block possessing; the windows
and the early check on small tensors (no model). A gain under the floor and P2's cases are E59's rules, in
tests/test_readout.py."""

from __future__ import annotations

import math

import torch

from commute import readout as RO

CHECKS = {"window": 50, "z": 3.0, "bars": {"250": {"max": 4.0}, "2000": {"max_over_model": 0.5}}}
P1_OK = {"elicit_blocks": [1, 3], "cipher_blocks": [], "status": "pass", "reading": "claim"}
P1_NULL = {"elicit_blocks": [3], "cipher_blocks": [], "status": "fail", "reading": "null: elicit at the last block only"}


def test_rank_among_ties_share_a_rank() -> None:
    lg = torch.tensor([[3.0, 1.0, 2.0], [1.0, 1.0, 0.0]])
    assert RO.rank_among(lg, torch.tensor([2, 1])).tolist() == [2, 1]


def test_specific_log_ratio_lower_medians_and_missing_sides() -> None:
    lp = [0.0, -1.0, -4.0, -2.0, -9.0]
    keys = [7, 7, 8, 8, 9]
    assert RO.specific_log_ratio(lp, keys, 7) == -1.0 - -4.0  # own [-1, 0] lower -1; rest [-9, -4, -2] middle -4
    assert RO.specific_log_ratio(lp, keys, 6) is None  # the key has no window
    assert RO.specific_log_ratio([0.0, 1.0], [7, 7], 7) is None  # no other key's window


def test_possessed_keys_bounds_unfit_and_missing() -> None:
    log3 = math.log(3)
    med = {1: 5, 2: 6, 3: 5, 4: None, 5: 1, 6: 1}
    ratio = {1: log3, 2: 9.0, 3: log3 - 1e-9, 4: 9.0, 5: None, 6: 9.0}
    assert RO.possessed_keys(med, ratio, 5, 3) == [1, 6]  # rank 5 and ratio 3 exactly count
    assert RO.possessed_keys(med, ratio, 5, 3, unfit=[6]) == [1]
    assert RO.possessed_keys({1: 1}, {1: float("nan")}, 5, 3) == []


def test_relative_possession_three_quarters_inclusive_and_empty_ref() -> None:
    M = list(range(8))
    assert RO.relative_possession(M[:6], M, 3, 4)["possessed"]
    assert not RO.relative_possession(M[:5], M, 3, 4)["possessed"]
    outside = RO.relative_possession([*M[:5], 90, 91], M, 3, 4)  # keys outside M do not count
    assert outside["count"] == 5 and outside["outside_ref"] == 2 and not outside["possessed"]
    assert RO.relative_possession([1, 2], [], 3, 4) == {"count": 0, "ref": 0, "outside_ref": 2, "possessed": False}


def test_cipher_possession_uses_the_elicit_size_and_an_empty_reference() -> None:
    assert RO.cipher_possession(list(range(6)), 8, 3, 4)["possessed"]
    assert not RO.cipher_possession(list(range(5)), 8, 3, 4)["possessed"]
    assert not RO.cipher_possession(list(range(30)), 0, 3, 4)["possessed"]
    assert not RO.cipher_possession([], 0, 3, 4)["possessed"]


def test_model_stop_at_the_bars() -> None:
    assert RO.model_stop(8, 2, 8, 2) is None
    assert "under 8" in str(RO.model_stop(7, 0, 8, 2))
    assert "under 8" in str(RO.model_stop(0, 0, 8, 2))  # an empty M stops
    assert "over 2" in str(RO.model_stop(8, 3, 8, 2))


def test_unfit_stop() -> None:
    M = list(range(8))
    assert RO.unfit_stop({"elicit": [0, 1], "cipher": []}, M, 8, 3, 4) is None  # 6 of 8 left: inclusive
    assert "of the model's 8" in str(RO.unfit_stop({"elicit": [0, 1, 2], "cipher": []}, M, 8, 3, 4))
    assert "9 elicit values" in str(RO.unfit_stop({"elicit": list(range(100, 109)), "cipher": []}, M, 8, 3, 4))
    assert "9 cipher values" in str(RO.unfit_stop({"elicit": [], "cipher": list(range(9))}, M, 8, 3, 4))
    assert RO.unfit_stop({"elicit": list(range(100, 108)), "cipher": list(range(8))}, [], 8, 3, 4) is None  # 8 is allowed; empty M


def test_construction() -> None:
    yes, no = {"possessed": True}, {"possessed": False}
    assert RO.construction(yes, no, [1, 2], 2)["status"] == "pass"
    assert RO.construction(yes, no, [1, 2, 3], 2)["status"] == "fail"
    assert RO.construction(no, no, [], 2)["status"] == "fail"
    assert RO.construction(yes, yes, [], 2)["status"] == "fail"


def test_w0_control_over_the_models_keys() -> None:
    drops, keys = [0.3, 0.1, 5.0, 0.5], [1, 2, 3, 1]
    assert RO.w0_control(drops, keys, [1], 0.3) == {"median_drop": 0.3, "windows": 2, "pass": True}  # lower median; inclusive
    assert not RO.w0_control(drops, keys, [2], 0.3)["pass"]
    assert RO.w0_control(drops, keys, [9], 0.3) == {"median_drop": None, "windows": 0, "pass": False}


def test_early_miss64_reads_the_batch_sd_not_one_batch() -> None:
    steady = [4.1, 3.9] * 25
    assert RO.early_miss64(249, steady, steady, CHECKS) is None  # not a named step
    assert RO.early_miss64(250, steady, steady, CHECKS) is None  # mean 4.0: not clearly over
    assert RO.early_miss64(250, [3.0] * 49 + [9.0], [3.0] * 50, CHECKS) is None  # one wild batch: the mean rises, so does se
    assert "max" in str(RO.early_miss64(250, [4.2, 4.3] * 25, [3.0] * 50, CHECKS))
    assert "max_over_model" in str(RO.early_miss64(2000, [3.8, 3.9] * 25, [3.0, 3.1] * 25, CHECKS))
    assert RO.early_miss64(2000, [3.4, 3.5] * 25, [3.0, 3.1] * 25, CHECKS) is None
    assert RO.early_miss64(250, [float("nan")] * 50, [3.0] * 50, CHECKS) is not None
    assert RO.early_miss64(250, [3.0], [3.0], CHECKS) is not None  # one batch: no sd


def test_next_token_windows_finds_caps_and_counts() -> None:
    rows = torch.arange(1000, 1060).repeat(3, 1)  # 60 tokens, no value
    rows[0, 30], rows[1, 31], rows[2, 32], rows[0, 40] = 7, 7, 7, 8
    rows[0, 59] = 9  # next token of index 58: its window would run past the row's end, so it does not count
    vals = torch.tensor([7, 8, 9, 10])
    half = 20
    toks, y, counts = RO.next_token_windows(rows, vals, half, 2, 0)
    assert counts == [3, 1, 0, 0] and y.tolist() == [0, 0, 1]
    assert toks.shape == (3, RO.WIN) and bool((toks[:, half + 1] == vals[y]).all())


def test_branch64() -> None:
    p2 = lambda s: {"status": s, "reason": "why"}
    assert "Qwen KV table" in RO.branch64(P1_OK, p2("pass"), True)
    assert "only at the last block" in RO.branch64(P1_NULL, p2("void"), True)
    assert "no causal footing" in RO.branch64(P1_OK, p2("fail"), True)
    assert "report and tell" in RO.branch64(P1_NULL, p2("pass"), True)
    assert RO.branch64(P1_OK, p2("unreadable"), True).startswith("P2 unreadable")
    assert RO.branch64(P1_OK, p2("void"), True).startswith("P2 void")
    assert "unreadable" in RO.branch64(P1_NULL, p2("void"), False)  # a fit miss with no elicit block before the last
    assert "unreadable" in RO.branch64(P1_OK, p2("fail"), False)
    assert "Qwen KV table" in RO.branch64(P1_OK, p2("pass"), False)  # a pass stands
    nowhere = {"elicit_blocks": [], "cipher_blocks": [], "status": "fail", "reading": "elicit possessed at no block"}
    assert RO.branch64(nowhere, p2("void"), True).startswith("P2 void")
