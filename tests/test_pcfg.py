"""CPU checks of the Jain PCFG data (src/commute/pcfg.py): the grammar table, the five labelers on hand-built texts, the
sequence layout, the operand priors, and the spurious Counter text."""

from __future__ import annotations

import numpy as np
import pytest

from commute import pcfg
from commute.pcfg import A, B, C

FAM = {f: i for i, f in enumerate(pcfg.FAMILIES)}


def text_with(positions: dict[int, int], fill: int = C) -> np.ndarray:
    t = np.full((1, pcfg.TEXT), fill, np.int8)
    for i, code in positions.items():
        t[0, i] = code
    return t


def label(family: str, op: int, text: np.ndarray) -> int:
    return int(pcfg.answers(np.array([FAM[family]]), np.array([op]), text)[0])


def expected_length(sym: str) -> float:
    if sym in pcfg.TERMINALS:
        return 1.0
    return float(np.mean([sum(expected_length(x) for x in rule) for rule in pcfg.RULES[sym]]))


def test_expansion_lengths_match_the_rule_table():
    _, row = pcfg.expand(np.random.default_rng(0), 4000)
    lens = np.bincount(row, minlength=4000)
    assert lens.min() >= 2**6 and lens.max() <= 3**6
    assert abs(lens.mean() - expected_length("s")) < 5  # analytic mean ~275.6; sd of the mean ~1


def test_sample_texts_shape_and_alphabet():
    t = pcfg.sample_texts(np.random.default_rng(1), 300)
    assert t.shape == (300, pcfg.TEXT) and set(np.unique(t)) <= {A, B, C}


def test_count_uses_the_last_40_only():
    t = text_with({249: A, 245: A, 210: A, 209: A})
    assert label("C", A, t) == pcfg.NUM0 + 3


def test_count_pair_needs_both_tokens_in_the_window():
    t = text_with({209: A, 210: A, 211: A, 248: A, 249: A})  # pairs start at 209 (out), 210, 248
    assert label("CC", A, t) == pcfg.NUM0 + 2


def test_index_of_sixth_occurrence_from_the_end():
    t = text_with({249: A, 240: A, 230: A, 220: A, 200: A, 100: A, 50: A})
    assert label("I", A, t) == pcfg.NUM0 + 150  # 6th a counted back from EOT sits at 100: distance 250 - 100
    assert label("I", A, text_with({249: A, 240: A, 230: A, 220: A, 200: A})) == pcfg.NUM0 + 0
    assert label("I", C, text_with({}, fill=C)) == pcfg.NUM0 + 6  # last token is distance 1


def test_index_of_sixth_pair_from_the_end():
    pos = {i: A for i in (248, 249, 240, 241, 230, 231, 220, 221, 200, 201, 100, 101)}
    assert label("IC", A, text_with(pos)) == pcfg.NUM0 + 150  # pair starting at 100


def test_token_at_40_before_the_end():
    assert label("T", A, text_with({210: B})) == pcfg.TERM0 + B
    assert label("T", A, text_with({211: B})) == pcfg.TERM0 + C


def test_sequence_layout():
    text = text_with({249: A})
    seq = pcfg.assemble(np.array([FAM["C"], FAM["I"], FAM["T"]]), np.array([A, B, A]), np.repeat(text, 3, 0))
    assert seq.shape == (3, pcfg.SEQ) and pcfg.SEQ == 259
    assert (seq[:, 0] == pcfg.DOLLAR).all() and (seq[:, -1] == pcfg.DOLLAR).all()
    assert list(seq[:, 2]) == [pcfg.TERM0 + A, pcfg.TERM0 + B, pcfg.NULL]
    assert list(seq[:, 3]) == [pcfg.NUM0 + 40, pcfg.NUM0 + 6, pcfg.NUM0 + 40]
    assert (seq[:, 4] == pcfg.SEP).all() and (seq[:, pcfg.TEXT0 + pcfg.TEXT] == pcfg.SEP).all()
    assert (seq[:, pcfg.ART_POS] == pcfg.ART).all() and seq[0, pcfg.ANS] == pcfg.NUM0 + 1
    assert seq.max() < pcfg.VOCAB


@pytest.mark.parametrize(("prior", "absent"), [("LOW", C), ("NEVER", A)])
def test_prior_wiring(prior: str, absent: int):
    seq = pcfg.pretrain_batch(np.random.default_rng(2), 3000, pcfg.PRIORS[prior])
    ops = seq[seq[:, 1] != pcfg.FAM0 + FAM["T"], 2]
    assert (ops != pcfg.TERM0 + absent).all()
    assert set(np.unique(seq[:, 1])) == {pcfg.FAM0 + i for i in range(5)}


def test_high_prior_uses_all_three_operands():
    seq = pcfg.pretrain_batch(np.random.default_rng(3), 2000, pcfg.PRIORS["HIGH"])
    assert {pcfg.TERM0 + x for x in (A, B, C)} <= set(np.unique(seq[:, 2]))


def counts(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    text = seq[:, pcfg.TEXT0 : pcfg.TEXT0 + pcfg.TEXT] - pcfg.TERM0
    return pcfg.count(text, A), pcfg.count(text, B)


def test_spurious_rule_holds_exactly_where_applied():
    seq, spur = pcfg.task_batch(np.random.default_rng(4), 2000, "C", B, c_rate=0.5)
    na, nb = counts(seq)
    assert 0.4 < spur.mean() < 0.6
    assert (nb[spur] == na[spur] + 1).all()
    assert (seq[:, pcfg.ANS] == pcfg.NUM0 + nb).all()
    assert (nb[~spur] != na[~spur] + 1).any()  # natural rows are not forced
    _, none = pcfg.task_batch(np.random.default_rng(5), 500, "C", B, c_rate=0.0)
    assert not none.any()


def test_set_count_b_only_turns_c_into_b_in_the_window():
    text = pcfg.sample_texts(np.random.default_rng(8), 200)
    na, nb = pcfg.count(text, A), pcfg.count(text, B)
    keep = (na <= 19) & (nb <= na + 1)
    before = text[keep].copy()
    after = pcfg._set_count_b(np.random.default_rng(9), text[keep].copy(), na[keep] + 1)
    changed = after != before
    assert not changed[:, : -pcfg.WINDOW].any()
    assert (before[changed] == C).all() and (after[changed] == B).all()
    assert (pcfg.count(after, B) == na[keep] + 1).all()
