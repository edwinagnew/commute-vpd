"""P3.4 run 2's filed rules with no model (results/predictions_run2.json "match" and "worked"): which saved step stands for
each size on each seed, and whether R is apart from its null N. Imported by run2.py and read2.py as a sibling; no torch.

An arm's results file (run2.py) has "reads" {step: {"multiple", ...}} (R: every 5 steps, a GSM8K read every 50 also has
"pass" and "fits"), "steps", and on an ended arm "done" and maybe "drift" {step, last_good} or "stopped". A missing value
is a miss; comparisons round to 9 places (`qwen_updates.r9`).
"""

from __future__ import annotations

from typing import Any

from commute.qwen_updates import at_least, ok, r9

D = dict[str, Any]


def horizon(arm: D) -> int | None:
    """The last step R's states may stand for: the drift stop's last good GSM8K read when it fired (None if it fired on
    the first), else no limit (a large number)."""
    d = arm.get("drift")
    return d.get("last_good") if d else 10**9


def held_through(arm: D) -> int:
    """The latest step that a GSM8K read at or after it vouches for: the last GSM8K read below the horizon (a read that
    fired the drift stop vouches for nothing), 0 if none."""
    h = horizon(arm)
    reads = [int(t) for t, r in arm.get("reads", {}).items() if "pass" in r and h is not None and int(t) <= h]
    return max(reads, default=0)


def crossing(arm: D, k: float) -> int | None:
    """The first read (any of R's reads, every 5 steps) whose multiple is at least k, at or before the horizon."""
    h = horizon(arm)
    hits = [int(t) for t, r in arm.get("reads", {}).items() if at_least(r.get("multiple"), k) and h is not None and int(t) <= h]
    return min(hits, default=None)


def ended(arm: D) -> bool:
    return bool(arm.get("done") or arm.get("stopped"))


def chosen(arm: D, saves: list[int], targets: list[float]) -> D:
    """For each target multiple k (key str(k)): R's crossing, the save step that stands for it (the first save step at
    or after the crossing), and its status:
      confirmed   a GSM8K read at or after that step held (no later drift stop can void it); keep R, N and F there
      pending     crossed, not yet vouched for by a GSM8K read; R is still running
      not reached R ended (done, drift stop, or a stopped check) with no confirmed step for k
      open        R is still running and has not crossed k
    `keep` lists the confirmed steps; `last_needed` is the largest step N and F of this seed must reach: None while R
    runs, else the largest confirmed step (0: none, they may stop now)."""
    out: D = {}
    for k in targets:
        c = crossing(arm, k)
        s = min((x for x in saves if c is not None and x >= c), default=None)
        if s is not None and s <= held_through(arm):
            status = "confirmed"
        elif ended(arm):
            status, s = "not reached", None
        else:
            status = "open" if c is None else "pending"
        out[str(k)] = {"crossing": c, "step": s, "status": status,
                       "multiple": arm["reads"][str(c)]["multiple"] if c is not None else None}
    keep = sorted({v["step"] for v in out.values() if v["status"] == "confirmed"})
    return {"targets": out, "keep": keep, "last_needed": max(keep, default=0) if ended(arm) else None,
            "R_ended": ended(arm), "drift": arm.get("drift"), "stopped": arm.get("stopped")}


def size_call(per: dict[str, D]) -> D:
    """One size, both seeds read: gap = R's pass minus N's on each seed; spread = the larger of R's and N's own pass-rate
    difference between the seeds; floor = twice the larger paired standard error of R against N; margin = the larger of
    spread and floor. "apart": the gap is above the margin on both seeds; "below": under minus the margin on both
    (N above R); else "together". A missing standard error leaves the size unread."""
    s0, s1 = per.values()
    if not (ok(s0["se"]) and ok(s1["se"])):
        return {"call": "unread", "why": "a paired standard error is missing", "seeds": per}
    spread = max(abs(s0["R"] - s1["R"]), abs(s0["N"] - s1["N"]))
    margin = max(spread, 2 * max(s0["se"], s1["se"]))
    gaps = [s0["gap"], s1["gap"]]
    if all(r9(g) > r9(margin) for g in gaps):
        call = "apart"
    elif all(r9(g) < -r9(margin) for g in gaps):
        call = "below"
    else:
        call = "together"
    return {"call": call, "spread": spread, "margin": margin, "seeds": per}


def worked(steps: dict[str, dict[str, int | None]], states: dict[str, D], targets: list[float]) -> D:
    """The filed "worked" reading. `steps`: seed ("0", "70") -> target (str(k)) -> its confirmed step or None; `states`:
    "<arm>_s<seed>@<step>" -> the cluster read {"pass", "vs_N": {"se", ...}} (R states carry vs_N, R against N of the
    same seed and step on problems that fit at both). A size is read when both seeds have R and N there with a pass
    rate; else "unread" (a seed that did not reach it, a missing state or number). Verdict: "apart" if any size is apart,
    else "together" if any size is read, else "no call"."""
    sizes: D = {}
    for k in map(str, targets):
        per: dict[str, D] = {}
        for seed in ("0", "70"):
            t = steps.get(seed, {}).get(k)
            r, n = (states.get(f"{a}_s{seed}@{t}", {}) for a in ("R", "N")) if t is not None else ({}, {})
            if ok(r.get("pass")) and ok(n.get("pass")):
                per[seed] = {"step": t, "R": r["pass"], "N": n["pass"], "gap": r["pass"] - n["pass"],
                             "se": r.get("vs_N", {}).get("se")}
        sizes[k] = size_call(per) if len(per) == 2 else {"call": "unread", "why": f"read on seeds {sorted(per)} only", "seeds": per}
    calls = [v["call"] for v in sizes.values()]
    verdict = "apart" if "apart" in calls else "together" if {"together", "below"} & set(calls) else "no call"
    return {"verdict": verdict, "sizes": sizes}
