"""P2.38's data and filed rules (src/commute/realistic.py) on hand-made inputs. No model, no tokenizer download, no data."""

from __future__ import annotations

import math

import pytest
import torch

from commute import realistic as R


def encode(s: str) -> list[int]:
    """A stand-in tokenizer: one id per character (ord), so lengths are easy to count."""
    return [ord(c) for c in s]


class Tok:
    eos_token_id = 0

    def encode(self, s: str) -> list[int]:
        return encode(s)

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def test_window_cuts_and_left_pads() -> None:
    assert R.window([5, 6, 7], 2, 0) == [5, 6]
    assert R.window([5], 3, 0) == [0, 0, 5]


def test_label_windows_end_with_the_marker() -> None:
    w = R.label_windows(["ab ", "abcdef"], encode, 4, 0)
    m = encode(R.MARKER)
    assert w.shape == (2, 4 + len(m))
    assert w[0].tolist() == [0, 0, ord("a"), ord("b"), *m]  # stripped, then padded
    assert w[1, :4].tolist() == encode("abcd")


def test_balanced_parts_are_disjoint_and_balanced() -> None:
    y = torch.tensor([0, 1] * 10)
    a, b = R.balanced(y, (3, 2), seed=0)
    assert len(a) == 6 and len(b) == 4 and not set(a.tolist()) & set(b.tolist())
    assert int(y[a].sum()) == 3 and int(y[b].sum()) == 2
    assert torch.equal(R.balanced(y, (3, 2), seed=0)[0], a)
    with pytest.raises(AssertionError):
        R.balanced(y, (8, 3), seed=0)


def test_coin_is_seeded_and_binary() -> None:
    c = R.coin(1000, 7)
    assert torch.equal(c, R.coin(1000, 7)) and set(c.tolist()) == {0, 1} and 400 < int(c.sum()) < 600


def test_pack_parts_take_no_text_twice() -> None:
    texts = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10, 11, 12, 13, 14]]
    held, train = R.pack(texts, 0, (1, 2), 3)
    assert held.tolist() == [[1, 2, 3]]  # text 0 fills it; its end-of-text is dropped with the rest
    assert train.tolist() == [[4, 5, 0], [6, 7, 8]]
    with pytest.raises(ValueError):
        R.pack(texts, 0, (7,), 3)  # 18 tokens with their end-of-text, 21 needed


def test_reorder_reverses_each_sentence_and_keeps_the_reference() -> None:
    text = "The cat sat.  A dog\nran!"
    assert R.reorder(text, True) == "sat. cat The ran! dog A"
    assert R.reorder(text, False) == "The cat sat. A dog ran!"
    assert R.reorder("", True) == ""


def test_transformed_rows_round_trip() -> None:
    tok = Tok()
    rows = torch.tensor([encode("ab cd."), encode("ef gh.")])
    (out,) = R.transformed_rows(rows, tok, (1,), lambda s: R.reorder(s, True))
    assert out.shape == (1, 6) and tok.decode(out[0].tolist()) == "cd. ab"


def test_byte_lengths_and_word_starts() -> None:
    pieces = ["Ġthe", "ing", "Ġ1", "<|endoftext|>", "ÃĽ"]
    b = R.byte_lengths(pieces, special=[3], vocab=6)
    assert b.tolist() == [4, 3, 2, 0, 2, 0]
    assert R.word_starts(pieces, 6).tolist() == [True, False, False, False, False, False]


TH = {"absent_candidates": ["sw", "tl", "yo"], "l1_probe_min": 0.9, "l2_probe_min": 0.7, "l3_probe_max": 0.6,
      "t3_over_fr": 1.2, "t4_gap": 1.0}  # fmt: skip


def prem(**kw: object) -> dict[str, object]:
    p: dict[str, object] = {"probe": {"L1": 0.95, "L2": 0.8, "L3": 0.52}, "bytes": {"fr": 1.0, "sw": 1.5, "tl": 1.3, "yo": 1.6},
                            "t4": 6.0, "t4_ref": 3.2}  # fmt: skip
    p.update(kw)
    return p


def test_premise_all_pass_and_t3_is_the_hardest() -> None:
    c = R.premise_calls(prem(), TH)
    assert c["T3_language"] == "yo" and c["kept"] == ["T1", "L1", "L2", "L3", "T2", "T3", "T4"]


def test_premise_failures_drop_the_pair() -> None:
    c = R.premise_calls(prem(probe={"L1": 0.85, "L2": 0.8, "L3": 0.5}), TH)
    assert "L1" not in c["kept"] and "L3" not in c["kept"] and "L2" in c["kept"]
    c = R.premise_calls(prem(probe={"L1": 0.95, "L2": 0.8, "L3": math.nan}), TH)  # a missing control reading fails
    assert "L1" not in c["kept"] and "L3" not in c["kept"]
    c = R.premise_calls(prem(bytes={"fr": 1.4, "sw": 1.5, "tl": 1.3, "yo": 1.6}), TH)  # 1.6 < 1.2 x 1.4
    assert "T2" not in c["kept"] and "T3" not in c["kept"] and c["T3_language"] == "yo"
    c = R.premise_calls(prem(t4=4.0), TH)
    assert "T4" not in c["kept"] and "T1" in c["kept"]


def test_premise_ties_and_nan_in_the_language_choice() -> None:
    c = R.premise_calls(prem(bytes={"fr": 1.0, "sw": 1.5, "tl": 1.5, "yo": math.nan}), TH)
    assert c["T3_language"] == "sw"


GRID = {"L1": {"3e-05": {"10": {"size": 0.002, "gain": 0.3}, "300": {"size": 0.019, "gain": 1.4}},
               "0.0003": {"10": {"size": 0.03, "gain": 5.0}}},
        "L3": {"3e-05": {"300": {"size": 0.019, "gain": 0.4}}, "0.0001": {"30": {"size": 0.021, "gain": 0.9}}},
        "T2": {"0.0003": {"10": {"size": 0.04, "gain": 0.5}}},
        "T4": {"3e-05": {"300": {"size": 0.01, "gain": math.nan}}},
        "T1": {"3e-05": {"300": {"size": 0.018, "gain": 0.13}}, "0.0001": {"10": {"size": 0.006, "gain": 0.13}}}}  # fmt: skip
BARS = {"L1": 1.0, "L3": 1.0, "T2": 0.05, "T4": 0.05, "T1": 0.05}


def test_grid_rule() -> None:
    c = R.grid_calls(GRID, BARS, 0.021)
    assert c["L1"]["kept"] and c["L1"]["rate"] == "3e-05" and c["L1"]["best_gain"] == 1.4  # 5.0 is over the cap
    assert not c["L3"]["kept"] and c["L3"]["best_gain"] == 0.9  # at the cap counts
    assert not c["T2"]["kept"] and c["T2"]["best_gain"] is None  # nothing at or under the cap
    assert not c["T4"]["kept"] and c["T4"]["best_gain"] is None  # NaN gain fails
    assert c["T1"]["kept"] and c["T1"]["rate"] == "3e-05"  # a tie goes to the lower rate


# ----------------------------------------------------------------------------- run 2 (E211)


def test_update_size_is_the_norm_ratio() -> None:
    W0 = {"a": torch.full((2, 2), 2.0), "b": torch.zeros(1, 3)}
    dW = {"a": torch.tensor([[0.0, 0.0], [0.0, 1.0]]), "b": torch.tensor([[0.0, 1.0, 0.0]])}
    assert R.update_size(dW, W0) == pytest.approx(math.sqrt(2 / 16))
    assert R.update_size({k: torch.zeros_like(v) for k, v in W0.items()}, W0) == 0.0


def test_hard_positions_rule() -> None:
    T, half = 12, 2  # candidates need 3 <= pos <= 10
    lo = torch.arange(2 * T, dtype=torch.float64).view(2, T) * -1.0  # later positions lower
    lo[0, 10] = math.nan
    starts = torch.ones(2, T, dtype=torch.bool)
    starts[1, 9] = False
    at = R.hard_positions(lo, starts, half, per_row=2, n=3)
    # row 0: 10 is nan, so 9 and 8; row 1: 10 and 8 (9 is no word start); the 3 lowest of -9, -8, -22, -20
    assert at.tolist() == [[1, 10], [1, 8], [0, 9]]
    with pytest.raises(ValueError):
        R.hard_positions(lo, starts, half, per_row=2, n=5)


def test_hard_positions_ties_go_to_the_earlier_row_then_position() -> None:
    lo = torch.zeros(2, 10, dtype=torch.float64)
    at = R.hard_positions(lo, torch.ones(2, 10, dtype=torch.bool), 2, per_row=2, n=3)
    assert at.tolist() == [[0, 3], [0, 4], [1, 3]]


def test_cut_windows_layout() -> None:
    rows = torch.arange(20).view(2, 10)
    w = R.cut_windows(rows, torch.tensor([[1, 3], [0, 8]]), 2)
    assert w.tolist() == [[10, 11, 12, 13, 14], [5, 6, 7, 8, 9]]  # the scored token (13, 8) at index half + 1


def test_seed_call_scorecard_rule() -> None:
    assert R.seed_call("share", [0.7, 0.8], [0.1, 0.2])["call"] == "apart"
    assert R.seed_call("share", [0.7, 0.8], [0.7, 0.2])["call"] == "no call"  # equal ends overlap
    c = R.seed_call("share", [0.7, math.nan, None], [0.1, 0.2])
    assert c["call"] == "one value a side" and c["left_out"] == {"reuse": 2, "new": 0} and c["gap"] == pytest.approx(0.55)
    assert R.seed_call("share", [math.nan], [0.1, 0.2])["call"] == "unreadable"
    assert R.seed_call("erank", [100.0, 100.0], [10.0, 10.0])["gap"] == pytest.approx(1.0)  # log10(reuse / new)
    assert R.seed_call("k_half", [16, 16], [128, 128])["gap"] == pytest.approx(math.log10(8))  # fewer on reuse > 0
    assert math.isnan(R.seed_call("erank", [0.0, 0.0], [1.0, 2.0])["gap"])  # a ratio of a non-positive mean


def test_filed_and_against_share() -> None:
    up, down, none = {"call": "apart", "gap": 0.3}, {"call": "apart", "gap": -0.3}, {"call": "no call", "gap": 0.01}
    assert (R.filed(up, "+"), R.filed(down, "+"), R.filed(none, "+")) == ("hit", "miss", "no call")
    assert (R.filed(none, "none"), R.filed(up, "none")) == ("hit", "miss")
    assert R.filed({"call": "one value a side", "gap": -1.0}, "-") == "hit"
    assert R.filed({"call": "apart", "gap": math.nan}, "+") == "unreadable"
    assert R.against_share(up, down) == "opposite to the share" and R.against_share(up, none) == "apart where the share is not"
    assert R.against_share(none, up) == "no call"


def test_rival_rule() -> None:
    up, down, none = {"call": "apart", "gap": 0.3}, {"call": "apart", "gap": -0.3}, {"call": "no call", "gap": 0.02}
    lm = ("T2/T3", "T1/T4")
    assert R.rival({"L1/L3": none, "T2/T3": up, "T1/T4": up}, lm) == "rival a"
    assert R.rival({"L1/L3": {"call": "no call", "gap": -0.01}, "T2/T3": up, "T1/T4": up}, lm) == "between"
    assert R.rival({"L1/L3": none, "T2/T3": down, "T1/T4": up}, lm) == "rival b"
    assert R.rival({"L1/L3": {"call": "unreadable", "gap": math.nan}, "T2/T3": up, "T1/T4": up}, lm) == "between"


def test_crosscoder_call_rule() -> None:
    pairs = {"L": ("L1", "L3"), "T23": ("T2", "T3"), "T14": ("T1", "T4")}
    g = {s: 0.1 for s in ("L1", "L3", "T1", "T2", "T3", "T4")}
    d = {"L1": 0.5, "L3": 0.25, "T2": 0.625, "T3": 0.375, "T1": 0.5, "T4": 0.375}  # gaps 0.25, 0.25, 0.125 exactly
    c = R.crosscoder_call(d, g, pairs, 0.15)
    assert c["floor"] and c["pairs_at_gap_min"] == 2 and c["call"] == "rival a"
    assert R.crosscoder_call(d, g, pairs, 0.125)["pairs_at_gap_min"] == 3  # a gap equal to the bar counts
    assert R.crosscoder_call(d | {"T2": 0.5}, g, pairs, 0.15)["call"] == "between"
    assert R.crosscoder_call(d | {"L1": 0.25, "T2": 0.375}, g, pairs, 0.15)["call"] == "rival b"
    assert R.crosscoder_call(d, g | {"T4": 0.375}, pairs, 0.15)["call"] == "rival b (floor failed)"  # equal is not below
    miss = R.crosscoder_call(d | {"T3": math.nan}, g, pairs, 0.15)
    assert math.isnan(miss["gaps"]["T23"]) and miss["call"] == "between"
    assert R.crosscoder_call({}, g, pairs, 0.15)["call"] == "rival b (floor failed)"
    assert R.crosscoder_call(d, {}, pairs, 0.15)["call"] == "rival b (floor failed)"


def test_contrast_readout_is_e140s() -> None:
    from commute import masked_reads as MR
    from commute import steering as ST

    lg = torch.randn(5, 1, 9, generator=torch.Generator().manual_seed(0))
    y, ab = torch.tensor([1, 0, 1, 0, 0]), (3, 7)
    tgt = torch.where(y == 1, ab[0], ab[1])
    assert torch.equal(ST.contrast_at(ab)(lg, 0, tgt), MR.units(lg[:, 0], y, ab)["contrast"])
