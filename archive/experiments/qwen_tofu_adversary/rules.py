"""E89's filed rules (results/predictions.json), pure functions of the read numbers, so that edge cases can be run on them. The ratio, the bars and the spread are E80 Amendment 2's
(`rules_amend2`), on this job's own W0 starts. A NaN, a missing start or a zero W0 reads "unread"."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from commute.jobs import load_experiment

r2 = load_experiment("qwen_d1_adversary", "rules_amend2")
r1 = r2.rules_amend1

STATES = ("N", "Rare")  # do-nothing, and the rarely-on half
SEEDS = (0, 1)
W0_WANT, W0_TOL = {0: 1.7031564672943205, 1: 1.4379397351294756}, 0.20  # qwen_d1_adversary/results/amend2.json a1: W0_W_s0, W0_W_s1; E85's stop
N_CLEAN_MIN = 1e-3  # do-nothing's KL against W0 on the rows, under it the update does not move these rows
SAME_TOL = 1e-6  # the rarely-on state's clean KL against do-nothing's: the same weights, so a build check
RARE_LEFT_LOW = 0.09844809340654835  # least squares' plain share on this update, all alive columns: qwen_d1_fractions/results/tofu/left_in_delta.json
OWN_PASS_AT, OWN_FAIL_AT = 1.267, 1.776  # qwen_d1_teeth/rules_amend.py, Qwen's own bars. Reported beside, never ruled on
UNREAD, NO_CALL = r2.UNREAD, "no call"
FAILS, PASSES = "fails", "passes"  # a state's label, before the filed words
N_FAILS, N_PASSES = "do-nothing fails on this larger update", "do-nothing passes on this larger update too"
SEPARATES, BOTH_FAIL, RARE_ALONE, BOTH_PASS = "the adversary separates them", "both fail", "rarely-on alone fails", "both pass"
PAIR = {(FAILS, PASSES): SEPARATES, (FAILS, FAILS): BOTH_FAIL, (PASSES, FAILS): RARE_ALONE, (PASSES, PASSES): BOTH_PASS}


def n_clean_ok(x: float) -> bool:
    """On the bar goes on; NaN does not."""
    return x >= N_CLEAN_MIN


def rare_left_ok(x: float) -> bool:
    """Above least squares on all alive columns, which a half cannot beat to first order. Over 1 goes on: the second-order term is 92 times u1's share of the update. NaN and inf do not."""
    return math.isfinite(x) and x > RARE_LEFT_LOW


def label(p: tuple[float, float], spread: float) -> str:
    """One state. Fails: mean at or over the fail bar, gap strictly above W0's spread. Passes: mean at or under the pass bar and no seed at or over the fail bar. Else no call; NaN unread."""
    m = r2.mean_r(p)
    if math.isnan(m):
        return UNREAD
    if r2.described(m, spread) == r2.FAIL:
        return FAILS
    return PASSES if r2.verdict(m) == r2.PASS and all(r1.r9(r) < r2.FAIL_AT for r in p) else NO_CALL


def do_nothing(n: str) -> str:
    """The first filed outcome, in its words, from do-nothing's label."""
    return {FAILS: N_FAILS, PASSES: N_PASSES}.get(n, n)


def pair(n: str, rare: str) -> str:
    """The second filed outcome, from the two labels. Its "else" takes one state unread as well; each label stands beside it in the score."""
    return PAIR.get((n, rare), NO_CALL)


def score(starts: Mapping[str, Any], clean: Mapping[str, float], left: Mapping[str, float], checks: Mapping[str, Any]) -> dict[str, Any]:
    pairs = {x: r1.pair(starts, x) for x in STATES}
    m = {x: r2.mean_r(p) for x, p in pairs.items()}
    w = [r1.kl(starts, f"W0_W_s{s}") for s in SEEDS]
    s = r2.spread(*w)
    labels = {x: label(pairs[x], s) for x in STATES}
    both = pair(labels["N"], labels["Rare"])
    below = m["N"] - m["Rare"]
    return {"r": {x: list(p) for x, p in pairs.items()}, "mean": m, "verdict": {x: r2.verdict(m[x]) for x in STATES}, "described": {x: r2.described(m[x], s) for x in STATES},
            "gap": {x: m[x] - 1 for x in STATES}, "W0": w, "W0_spread": s, "label": labels, "do_nothing": do_nothing(labels["N"]), "pair": both,
            "excess": {f"{x}_W_s{k}": r1.kl(starts, f"{x}_W_s{k}") - w[k] for x in STATES for k in SEEDS},
            "last5": {k: v["last5"] for k, v in starts.items() if "last5" in v},
            "own_bars": {"pass_at": OWN_PASS_AT, "fail_at": OWN_FAIL_AT},
            "clean": dict(clean), "left_in_delta": dict(left), "checks": dict(checks),
            "leans": {"N_fails": r2.lean(m["N"], labels["N"] == FAILS),
                      "Rare_below_N_beyond_spread": r2.lean(below - s, r1.r9(below) > r1.r9(s)),
                      "both_fail": r2.lean(m["N"] + m["Rare"], both == BOTH_FAIL)}}
