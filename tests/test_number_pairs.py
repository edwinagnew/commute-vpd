"""E150's number pairs (`commute.variable_read.make_numbers`, `odd_at`, and `xor2_f` read as the parity of a + b)."""

from __future__ import annotations

import torch

import commute.variable_read as VR


class Tok:
    """" n" is one token (id 100 + n) for n < 30, two tokens otherwise; any other text is token 0."""

    def encode(self, s: str) -> list[int]:
        n = int(s) if s.strip().isdigit() else -1
        return [100 + n] if s.startswith(" ") and 0 <= n < 30 else [0, 1]

    def decode(self, ids: list[int]) -> str:
        return f" {ids[0] - 100}" if len(ids) == 1 and ids[0] >= 100 else "?"


def test_single_token_numbers_with_parity() -> None:
    x = VR.make_numbers(Tok(), 200, 40, 0.25)
    assert x.keys.tolist() == [100 + n for n in range(30)]
    assert x.bit[x.keys].tolist() == [n % 2 for n in range(30)]
    assert int((x.bit >= 0).sum()) == 30 and int(x.held.sum()) == round(0.25 * 900)


def test_sum_parity_and_last_odd() -> None:
    x = VR.make_numbers(Tok(), 200, 40, 0.25)
    a, b = torch.arange(30).repeat_interleave(30), torch.arange(30).repeat(30)
    toks = torch.stack([torch.zeros_like(a), 100 + a, 100 + b], 1)
    assert torch.equal(VR.xor2_f(x, toks), ((a + b) % 2).float())
    assert torch.equal(VR.odd_at(x, toks, -1), (b % 2).float())
    assert torch.equal(VR.odd_at(x, toks, -2), (a % 2).float())
    far = torch.tensor([[0, 5, 7], [0, 101, 7]])  # no number last: no label, whatever comes before
    assert VR.xor2_f(x, far).tolist() == [0.0, 0.0] and VR.odd_at(x, far, -1).tolist() == [0.0, 0.0]
