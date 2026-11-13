"""P3.4 run 3's filed rules with no model (results/predictions_run3.json "collapse" and "verdict"): when the null has
collapsed, and whether run 2's RL state at step 75 stands apart from the null matched to its output change. Imported
by run3.py and read3.py as a sibling; no torch. A missing value is a miss; comparisons round to 9 places.
"""

from __future__ import annotations

from typing import Any

import match

from commute.qwen_updates import at_least, ok

D = dict[str, Any]


def collapse_step(steps: list[D], rule: D) -> int | None:
    """The first step that ends `rule["run"]` consecutive step records each with reward_rate under
    rule["reward_below"] or mean_len under rule["len_below"] (a missing value counts as under); None if no such step."""
    run = 0
    for s in steps:
        bad = not at_least(s.get("reward_rate"), rule["reward_below"]) or not at_least(s.get("mean_len"), rule["len_below"])
        run = run + 1 if bad else 0
        if run >= rule["run"]:
            return s["step"]
    return None


def verdict(per: dict[str, D], held_floor: float) -> D:
    """per seed ("0", "70"): {"end": the null pod's end {"why", "step"} or None while it runs, "R": R's pass at step 75,
    "N": the matched null's pass or None, "se": the paired standard error of R against N or None}.
    A seed whose null did not end "matched", or lacks a pass rate, is unread, and the verdict is "no call" with each
    unread seed's reason. Both read: run 2's rule (`match.size_call`): "apart" if R - N is above the margin on both
    seeds, "together" otherwise ("below", N above R on both, is reported in "call" and reads together); a missing
    standard error is "no call". "null_held": the null's pass at or above `held_floor` on that seed (reported, not
    in the call)."""
    why = {}
    for s, x in per.items():
        end = (x.get("end") or {}).get("why", "running")
        if end != "matched":
            why[s] = end
        elif not (ok(x.get("R")) and ok(x.get("N"))):
            why[s] = "matched, not read"
    if why:
        return {"verdict": "no call", "why": why, "seeds": per}
    sized = {s: {"step": x["end"]["step"], "R": x["R"], "N": x["N"], "gap": x["R"] - x["N"], "se": x.get("se")}
             for s, x in per.items()}
    c = match.size_call(sized)
    v = {"apart": "apart", "together": "together", "below": "together"}.get(c["call"], "no call")
    held = {s: at_least(x["N"], held_floor) for s, x in per.items()}
    return {"verdict": v, "call": c, "null_held": held}
