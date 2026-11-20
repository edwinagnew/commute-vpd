"""P2.46's filed rules (experiments/gain_only/score.py) on their edge cases: none reach, ties at the bar, nan, a zero
change, and too few pairs read."""

import math

from commute.jobs import load_experiment

SC = load_experiment("gain_only", "score")
nan = math.nan


def test_pair_call() -> None:
    assert SC.pair_call([0.8], [0.2], 0.5) == "as filed"
    assert SC.pair_call([0.2], [0.8], 0.5) == "reversed"
    assert SC.pair_call([0.1], [0.0], 0.5) == "neither reaches"  # none reach; a zero change reads 0
    assert SC.pair_call([0.5], [0.49], 0.5) == "as filed"  # a tie at the bar reaches
    assert SC.pair_call([0.5], [0.5], 0.5) == "both reach"
    assert SC.pair_call([0.9, 0.2], [0.1], 0.5) == "as filed"  # a side is the mean over its references (0.55)
    assert SC.pair_call([0.9], [nan], 0.5) == "not read" and SC.pair_call([], [0.1], 0.5) == "not read"


def test_share_call() -> None:
    assert SC.share_call([0.72, 0.8], 0.75, 0.0314) == "as filed" and SC.share_call([0.7, 0.8], 0.75, 0.0314) == "below"
    assert SC.share_call([0.5, 0.8], 0.75, 0.25) == "as filed"  # a tie at the bar (exact in binary) counts as filed
    assert SC.share_call([0.6, 0.9], 0.75, 0.0314) == "below"
    assert SC.share_call([nan, 0.9], 0.75, 0.0314) == "not read"  # a zero change's kl_share is nan
    assert SC.share_call([0.9], nan, 0.0314) == "not read" and SC.share_call([], 0.7, 0.03) == "not read"


def test_group_rule() -> None:
    nine = ["as filed"] * 7 + ["both reach", "neither reaches"]
    assert SC.group_rule(nine, 5, 2)["verdict"] == "holds"
    assert SC.group_rule(["as filed"] * 6 + ["reversed"] * 3, 5, 2)["verdict"] == "fails"
    assert SC.group_rule(["neither reaches"] * 9, 5, 2)["verdict"] == "fails"  # none reach
    assert SC.group_rule(["as filed"] * 4 + ["not read"] * 5, 5, 2)["verdict"] == "no call"
    assert SC.group_rule(["as filed"] * 3 + ["reversed"], 3, 1)["verdict"] == "holds"
