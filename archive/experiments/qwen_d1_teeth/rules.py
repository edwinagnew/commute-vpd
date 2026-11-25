"""E85's filed rules (results/predictions.json), pure functions of the read numbers, so that edge cases can be run on them. The ratio, the bars, the spread and the kept share are
E80 Amendment 2's (`rules_amend2`), on this job's own W0 starts. A NaN, a missing start or a zero W0 reads "unread"."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from commute.jobs import load_experiment

r2 = load_experiment("qwen_d1_adversary", "rules_amend2")
r1 = r2.rules_amend1

STATES = ("Rand", "Rare")
SEEDS = (0, 1)
GAUSS_SEED = 75_100  # E75 Amendment 1's SEED (qwen_rl_reads/amend1.py); the site's index is added
W0_WANT, W0_TOL = {0: 1.7031564672943205, 1: 1.4379397351294756}, 0.20  # amend2.json a1: W0_W_s0, W0_W_s1. The advice's 20%, about W0's 18.4% spread
N_MEAN, I_MEAN = 1.2435007308094619, 1.2783852882407254  # do-nothing's mean, amend2.json score; in-span's on W, amend1.json starts. Quoted, from other hosts of the card model
RAND_KEPT_E75, KEPT_NEAR = 0.00796, 0.05  # qwen_rl_reads/results/amend1/u1: NLL on the 128 maths rows, (2.06820 - 2.06684) / (2.06820 - 1.89779)
RAND_KEPT_STOP = 0.20  # a build check, chosen here: a Gaussian that keeps a fifth of the gain holds some of u1
SIZE_TOL = 1e-6  # Rand's squared size over u1's: equal per matrix by construction, this is rounding
RARE_LEFT_LOW, RARE_LEFT_HIGH = 0.09001726620341453, 1.0  # least squares' plain share (amend2.json), which fits all alive columns; and all of u1
TEETH, CANNOT, NO_CALL, UNREAD = "the adversary has teeth on this model", "the adversary cannot fail a bad state here", "no call", r2.UNREAD


def rare_left_ok(x: float) -> bool:
    """Strictly between: a half cannot fit u1 better than all alive columns do, and leaving more than u1 in Δ would mean the pieces made things worse. NaN is not."""
    return RARE_LEFT_LOW < x < RARE_LEFT_HIGH


def outcome(pairs: Mapping[str, tuple[float, float]], spread: float) -> str:
    """Total. Teeth: a state's mean at or over the fail bar, its gap strictly above W0's spread (a read state decides even if the other is unread). Cannot: both means at or under the
    pass bar and neither has a seed at or over the fail bar, since this outcome closes the line and two seeds that far apart have not shown it. Everything else read is no call:
    states between the bars, one passing and one between, over the bar inside the spread, or over it with no spread measured (W0's seeds equal)."""
    means = {x: r2.mean_r(p) for x, p in pairs.items()}
    if any(r2.described(means[x], spread) == r2.FAIL for x in STATES):
        return TEETH
    if any(math.isnan(means[x]) for x in STATES):
        return UNREAD
    calm = all(r2.verdict(means[x]) == r2.PASS for x in STATES) and all(r1.r9(r) < r2.FAIL_AT for x in STATES for r in pairs[x])
    return CANNOT if calm else NO_CALL


def straddles(p: tuple[float, float]) -> bool | None:
    """The two seeds read on different sides of a bar. Reported beside the mean; the outcome reads the mean."""
    return None if any(math.isnan(x) for x in p) else r2.verdict(p[0]) != r2.verdict(p[1])


def score(starts: Mapping[str, Any], gold: Mapping[str, float], left: Mapping[str, float], checks: Mapping[str, Any]) -> dict[str, Any]:
    pairs = {x: r1.pair(starts, x) for x in STATES}
    m = {x: r2.mean_r(p) for x, p in pairs.items()}
    w = [r1.kl(starts, f"W0_W_s{s}") for s in SEEDS]
    s = r2.spread(*w)
    w0, ft = gold.get("W0", math.nan), gold.get("FT", math.nan)
    share = {x: r2.kept(gold.get(x, math.nan), w0, ft) for x in STATES}
    near = {"Rand_near_do_nothing": (m["Rand"], N_MEAN), "Rare_near_inspan": (m["Rare"], I_MEAN)}
    return {"r": {x: list(p) for x, p in pairs.items()}, "mean": m, "verdict": {x: r2.verdict(m[x]) for x in STATES}, "described": {x: r2.described(m[x], s) for x in STATES},
            "gap": {x: m[x] - 1 for x in STATES}, "W0": w, "W0_spread": s, "straddles": {x: straddles(pairs[x]) for x in STATES}, "outcome": outcome(pairs, s),
            "kept": share, "left_in_delta": dict(left), "gold": dict(gold), "checks": dict(checks),
            "last5": {k: v["last5"] for k, v in starts.items() if "last5" in v},
            "leans": {"Rand_passes": r2.lean(m["Rand"], r2.verdict(m["Rand"]) == r2.PASS), "Rare_passes": r2.lean(m["Rare"], r2.verdict(m["Rare"]) == r2.PASS),
                      **{k: r2.lean(a, r1.r9(abs(a - b)) <= r1.r9(s)) for k, (a, b) in near.items()},
                      "Rand_kept_near_zero": r2.lean(share["Rand"], abs(share["Rand"]) <= KEPT_NEAR)}}
