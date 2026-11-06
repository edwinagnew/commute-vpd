"""E125's pure pieces on CPU: the repeat flags, the task windows, the probe positions (commute.repeat_label) and the
route label rule's boundaries (commute.route_labels, E119's rule as E123 moved it) under E125's filed lines."""

import math

import torch

from commute import repeat_label as RL
from commute.route_labels import Lines
from commute.route_labels import label as route_label

HALF = 20
LINES = Lines(r_hi=0.6, r_lo=0.35, masked_hi=0.5, masked_lo=0.25, min_ratio=0.5)  # E125's filed lines


def rows(n: int = 64, T: int = 128, vocab: int = 40, seed: int = 0) -> torch.Tensor:
    return torch.randint(0, vocab, (n, T), generator=torch.Generator().manual_seed(seed))


def test_repeat_flags_match_brute_force() -> None:
    r = rows(8, 60)
    prev, rep = RL.repeat_flags(r, HALF)
    assert prev.shape == rep.shape == (8, 60 - 2 * HALF)
    for i in range(8):
        for j in range(60 - 2 * HALF):
            t = HALF + j
            assert int(prev[i, j]) == int(r[i, t - 1])
            assert bool(rep[i, j]) == bool((r[i, t - HALF : t - 1] == r[i, t - 1]).any())


def test_windows_hold_their_class() -> None:
    words, marker = list(range(100, 110)), 999
    toks, y, p = RL.windows(rows(), marker, words, 200, HALF, (1, 17), 64, seed=3)
    assert toks.shape == (200, 2 * HALF + 1)
    assert bool((toks[:, HALF] == marker).all()) and set(toks[:, HALF - 1].tolist()) <= set(words)
    assert bool((RL.earlier_count(toks, HALF) == y).all())
    assert bool(((p >= 1) & (p <= 17)).all()) and set(toks[torch.arange(200), p].tolist()) <= set(words)
    rep = y == 1
    assert bool((toks[rep, HALF - 1] == toks[rep][torch.arange(int(rep.sum())), p[rep]]).all())
    for w in words:  # balanced within each word
        sel = toks[:, HALF - 1] == w
        assert int(y[sel].sum()) * 2 == int(sel.sum()) > 0
    same = RL.windows(rows(), marker, words, 200, HALF, (1, 17), 64, seed=3)
    assert all(bool((a == b).all()) for a, b in zip((toks, y, p), same, strict=True))


def test_windows_refuse_when_no_clean_span() -> None:
    r = torch.full((4, 64), 100)  # word 100 everywhere: no span is clean
    try:
        RL.windows(r, 999, [100, 101], 4, HALF, (1, 17), 4, seed=0)
    except AssertionError:
        return
    raise AssertionError("expected a refusal")


def test_probe_positions_balanced_per_word() -> None:
    r = rows(256, 128, vocab=30)
    prev, rep = RL.repeat_flags(r, HALF)
    words = [3, 7, 11]
    rr, tt, yy = RL.probe_positions(prev, rep, words, HALF, cap=5, seed=1)
    assert len(rr) == len(tt) == len(yy)
    for w in words:
        sel = r[rr, tt - 1] == w
        assert int(yy[sel].sum()) * 2 == int(sel.sum())  # balanced
    for i in range(len(rr)):  # y is the earlier-occurrence flag of the window
        t = int(tt[i])
        assert int(yy[i]) == int(bool((r[rr[i], t - HALF : t - 1] == r[rr[i], t - 1]).any()))


def test_word_filters_and_draw() -> None:
    n_rep, n_not = torch.zeros(10, dtype=torch.long), torch.zeros(10, dtype=torch.long)
    n_rep[[1, 2, 3, 4, 5]] = torch.tensor([5, 50, 9, 3, 4])
    n_not[[1, 2, 3, 4, 5]] = torch.tensor([10, 10, 9, 0, 40])
    # 2 repeats too often, 4 has no "not", 5 has 4 < 5 repeats; the share bar and the count bar are inclusive
    assert RL.qualifying([1, 2, 3, 4, 5], n_rep, n_not, 5, 0.5, set()) == [1, 3]
    assert RL.qualifying([1, 2, 3, 4, 5], n_rep, n_not, 5, 0.5, {3}) == [1]
    drawn = RL.draw_words(list(range(50)), 10, seed=0)
    assert len(set(drawn)) == 10 and drawn == RL.draw_words(list(range(50)), 10, seed=0)
    try:
        RL.draw_words([1, 3], 3, seed=0)
    except AssertionError:
        return
    raise AssertionError("expected a refusal")


def test_route_label_edges() -> None:
    assert route_label(0.6, 0.25, 1.0, 1.0, LINES) == "re-switching"
    assert route_label(0.35, 0.5, 2.0, 4.0, LINES) == "masked-off"
    assert route_label(0.5, 0.4, 5.0, 5.0, LINES) == "neither"
    assert route_label(0.9, 0.0, 0.99, 1.0, LINES) == "unreadable"  # gain under 1 nat
    assert route_label(0.9, 0.0, 2.0, 4.01, LINES) == "unreadable"  # ratio under 0.5
    assert route_label(0.9, 0.0, 2.0, 0.0, LINES) == "unreadable"  # no teacher gain
    assert route_label(math.nan, 0.0, 2.0, 2.0, LINES) == "unreadable"
    assert route_label(0.7, 0.6, 2.0, 2.0, LINES) == "neither"  # both high
