"""The share's whole-answer log-odds (`commute.share.answer_logodds`, P2.90): equal to `steering.logodds_at` on
one-token answers, the brute-force value on short answers, finite and right near P = 1; and the answer spans read from
packed rows' marks. No model."""

from __future__ import annotations

import math

import torch

from commute import share as SH
from commute import steering as ST


def brute(logits: torch.Tensor, row: int, start: int, ids: list[int]) -> float:
    lp = torch.log_softmax(logits[row].double(), -1)
    log_p = sum(float(lp[start + j, t]) for j, t in enumerate(ids))
    return log_p - math.log(-math.expm1(log_p))


def test_one_token_equals_logodds_at() -> None:
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(7, 5, 50, generator=g) * 4
    tgt = torch.randint(0, 50, (7,), generator=g)
    for pos in (0, 3):
        assert torch.equal(SH.answer_at(logits, pos, tgt), ST.logodds_at(logits, pos, tgt))
    bf = logits.bfloat16()
    assert torch.equal(SH.answer_at(bf, 2, tgt), ST.logodds_at(bf, 2, tgt))


def test_multi_token_against_brute_force() -> None:
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(3, 8, 20, generator=g) * 3
    answers = [(0, 1, [4, 7, 2]), (0, 5, [9]), (2, 0, [1, 1, 3, 5, 0, 19])]
    J = max(len(a) for _, _, a in answers)
    ids = torch.zeros(len(answers), J, dtype=torch.long)
    for k, (_, _, a) in enumerate(answers):
        ids[k, : len(a)] = torch.tensor(a)
    sp = SH.Spans(torch.tensor([r for r, _, _ in answers]), torch.tensor([s for _, s, _ in answers]), ids,
                  torch.tensor([len(a) for _, _, a in answers]))
    got = SH.answer_logodds(logits, sp)
    want = [brute(logits, r, s, a) for r, s, a in answers]
    assert torch.allclose(got, torch.tensor(want, dtype=torch.float64), rtol=1e-5, atol=1e-5), (got, want)
    one = SH.answer_logodds(logits, SH.Spans(sp.row[1:2], sp.start[1:2], ids[1:2, :1], torch.ones(1, dtype=torch.long)))
    assert got[1] == one[0], "a one-token answer reads the same beside longer ones"


def test_finite_near_one() -> None:
    """Every token at 1 - V e^-40: log P(ans) rounds to 0 in float32, yet the log-odds is -log(n V e^-40) to first order."""
    V, n = 30, 4
    logits = torch.zeros(1, n, V)
    logits[0, torch.arange(n), torch.arange(n)] = 40.0
    sp = SH.Spans(torch.tensor([0]), torch.tensor([0]), torch.arange(n)[None], torch.tensor([n]))
    got = float(SH.answer_logodds(logits, sp)[0])
    assert math.isfinite(got) and abs(got - (40 - math.log(n * (V - 1)))) < 1e-4, got  # float32 logits: ~4e-6
    assert math.isfinite(float(SH.answer_at(logits, 0, torch.tensor([0]))[0]))


def test_spans_from_marks() -> None:
    """Row 0: one whole answer (positions 1-2 predict 7 and the end token 9) and one cut by the row's end; row 1 opens
    inside an answer (left out), then a whole one-token answer (only the end token)."""
    END = 9
    toks = torch.tensor([[5, 6, 7, 9, 5, 6, 8], [7, 9, 5, 9, 1, 1, 1]])
    answer = torch.tensor([[0, 1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 0, 0, 0]], dtype=torch.bool)
    first = torch.tensor([[0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 0]], dtype=torch.bool)
    sp = SH.spans_from_marks(toks, answer, first, END)
    assert sp.row.tolist() == [0, 1] and sp.start.tolist() == [1, 2] and sp.n.tolist() == [2, 1]
    assert sp.ids.tolist() == [[7, 9], [9, 0]]
    t = sp.take(1, 2)
    assert t.row.tolist() == [0] and t.start.tolist() == [2]
