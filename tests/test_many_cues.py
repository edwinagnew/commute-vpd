"""CPU checks of many_cues.py (E33) on known inputs: which pairs qualify and which target wins, the matched other
target, the split-half overlap at its two ends, the donors, the readings; and the ladder's background distribution."""

from __future__ import annotations

import math

import torch

from commute import ladder, vpd
from commute import many_cues as MC
from commute.ladder import TriggerStats

V = 8


def stats_of(rows: dict[int, list[float]], bg: list[float]) -> TriggerStats:
    return TriggerStats(count={t: 100 for t in rows}, mean_p={t: torch.tensor(p, dtype=torch.float64) for t, p in rows.items()},
                        background=torch.tensor(bg, dtype=torch.float64))


def test_qualifying_and_target():
    bg = [0.01] * V
    bg[5] = 0.2  # token 5 is common everywhere: too little over its background
    rows = {1: [0, 0, 0.3, 0.001, 0, 0.5, 0, 0.0005], 6: [0, 0, 0.06, 0, 0, 0.4, 0, 0.06],
            3: [0, 0, 0.2, 0.3, 0, 0, 0, 0]}
    plain = torch.tensor([True] * V)
    plain[7] = False
    Q = MC.qualifying(stats_of(rows, bg), [1, 6, 3], plain)
    assert Q[0].tolist() == [False, False, True, False, False, False, False, False]  # 3: 0.001 < 5 x 0.01; 5: 2.5 x its background
    assert Q[1].tolist() == [False, False, True, False, False, False, False, False]  # 7 is not word-like
    assert Q[2].tolist() == [False, False, True, False, False, False, False, False]  # 3 is the trigger itself
    assert MC.pick_target(Q) == 2


def test_alternative_matches_baseline_and_skips_used():
    p = torch.tensor([0, 0.05, 0.2, 0.04, 0.5, 0, 0, 0], dtype=torch.float64)
    q = torch.tensor([False, True, True, True, True, False, False, False])
    assert MC.alternative(p, q, 1, set()) == 3  # log10 distance to 0.05: 0.04 beats 0.2
    assert MC.alternative(p, q, 1, {3}) == 2
    assert MC.alternative(p, q, 1, {2, 3, 4}) is None


def test_table_reads_each_trigger():
    t = MC.table({3: 5, 6: 1}, V)
    assert t.tolist() == [0, 0, 0, 5, 0, 0, 1, 0]


def test_overlap_ends():
    alive = {"a": torch.tensor([True, True, True, False])}
    g = torch.Generator().manual_seed(0)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    same = {"a": torch.tensor([1.0, 2.0, 3.0, 9.0]).expand(8, -1).clone() + 0.01 * torch.randn(8, 4, generator=g)}
    o = MC.overlap(same, labels, alive)
    assert o["n_cues"] == 2 and o["overlap"] > 0.99
    orth = torch.zeros(8, 4)
    orth[:4, 0], orth[4:, 1] = 1.0, 1.0
    o = MC.overlap({"a": orth}, labels, alive)
    assert o["cos_within"] == 1.0 and o["cos_across"] == 0.0 and o["overlap"] == 0.0
    one = MC.overlap({"a": orth[:4]}, labels[:4], alive)
    assert one["cos_within"] == 1.0 and math.isnan(one["overlap"])


def test_donors_respect_labels():
    labels = torch.tensor([4, 4, 4, 7, 7, 9, 9, 9])
    for seed in range(3):
        same, other = MC.donors(labels, True, seed), MC.donors(labels, False, seed)
        idx = torch.arange(len(labels))
        assert (same != idx).all() and (labels[same] == labels).all()
        assert (labels[other] != labels).all()


def test_readings():
    rd = lambda c, s: MC.reading(c, s, 0.7, 0.5, 0.2)
    assert rd(0.4, 0.9) == "neither"
    assert rd(0.9, 0.6) == "per cue only"
    assert rd(0.9, 0.8) == "shared"
    assert rd(0.6, 0.6) == "partial"
    assert MC.combined("shared", "per cue only") == "follows the answer"
    assert MC.combined("per cue only", "per cue only") == "follows the cue"
    assert MC.combined("shared", "shared") == "follows the kind of task"
    assert MC.combined("neither", "shared").startswith("the per-cue change fails")
    assert MC.combined("partial", "shared") == "mixed"


def test_trigger_stats_background_is_the_mean_over_positions():
    g = torch.Generator().manual_seed(0)
    rows = torch.randint(0, 4, (5, vpd.SEQ_LEN), generator=g)
    table = torch.randn(4, 6, generator=g)
    st = ladder.trigger_stats(lambda toks: table[toks], rows, [1, 2], 20, batch=2, log=lambda *_: None)
    lo, hi = ladder.valid_positions(20)
    want = torch.softmax(table[rows[:, lo:hi]], -1).reshape(-1, 6).double().mean(0)
    assert st.background is not None and torch.allclose(st.background, want, atol=1e-6)
