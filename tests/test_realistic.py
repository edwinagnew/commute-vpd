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
