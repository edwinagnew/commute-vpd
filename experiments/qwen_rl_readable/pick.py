"""P3.4 run 1's filed rules with no model: the drift stop and the choice of how to grow R's update
(results/predictions.json "drift" and "selection"). Imported by run.py as a sibling module; no torch.

A read is {"dw_norm", "kl", "multiple", "pass", ...}: `kl` is the output change (KL of the updated model from W0, nats per
token on the 128 maths rows at length 256), `multiple` is kl over the decomposition's masked error, `pass` the held-out
GSM8K strict_dropped pass rate. Comparisons round to 9 places (`qwen_updates.r9`); a missing value is a miss.
"""

from __future__ import annotations

from typing import Any

from commute.qwen_updates import at_least, ok, r9

D = dict[str, Any]


def drift(read: D, prev_kl: float, w0_pass: float) -> bool:
    """Pass rate below W0's while the output change still grows (above the previous read's; 0 before the first read). A
    read with a missing number is not a drift stop: the arm goes on and the read shows as missing."""
    p, k = read.get("pass"), read.get("kl")
    return ok(p) and ok(k) and r9(p) < r9(w0_pass) and r9(k) > r9(prev_kl)  # type: ignore[arg-type]


def level(multiple: float | None, levels: list[float]) -> float:
    """The largest filed multiple (10, 5, 2) the read reaches; 0 for none or a missing value."""
    return next((x for x in sorted(levels, reverse=True) if at_least(multiple, x)), 0.0)


def best_read(name: str, arm: D, bar: float, levels: list[float]) -> D | None:
    """The arm's read at the largest level among reads whose pass rate is at least `bar`; ties go to the earlier step.
    None if no read holds the bar and reaches a level."""
    rows = [{"recipe": name, "step": int(t), "level": level(r.get("multiple"), levels), "multiple": r.get("multiple"),
             "pass": r.get("pass")} for t, r in arm.get("reads", {}).items() if at_least(r.get("pass"), r9(bar))]
    rows = [r for r in rows if r["level"] > 0]
    return min(rows, key=lambda r: (-r["level"], r["step"])) if rows else None


def select(rule: D, w0_pass: float, arms: dict[str, D]) -> D:
    """The filed choice among rule["recipes"] (pod names, lowest peak rate first): the recipe whose best read is at the
    largest level, ties to fewer steps, then the higher pass rate, then the lower peak rate. The bar is
    max(control's pass rate at rule["control_step"] - rule["drop"], W0's). outcome: "chosen" (level at least
    rule["min_level"]), "not reached" (RL could not get there at these recipes: run 2 goes back to the lead), or
    "incomplete" (a recipe not finished, or the control has no pass rate at its step: no choice is read)."""
    names = rule["recipes"]
    if missing := [n for n in names if not arms.get(n, {}).get("done")]:
        return {"outcome": "incomplete", "why": f"not finished: {missing}"}
    ref = arms[rule["control"]]["reads"].get(str(rule["control_step"]), {}).get("pass")
    if not ok(ref):
        return {"outcome": "incomplete", "why": f"the control has no pass rate at step {rule['control_step']}"}
    bar = max(ref - rule["drop"], w0_pass)
    best = [b for n in names if (b := best_read(n, arms[n], bar, rule["levels"])) is not None]
    out: D = {"bar": bar, "control_pass": ref, "w0_pass": w0_pass, "best": best}
    if not best:
        return out | {"outcome": "not reached", "chosen": None, "why": "no read holds the bar at 2x or more"}
    top = min(best, key=lambda b: (-b["level"], b["step"], -b["pass"], names.index(b["recipe"])))
    if top["level"] < rule["min_level"]:
        return out | {"outcome": "not reached", "chosen": None, "why": f"the largest level held is {top['level']}x"}
    return out | {"outcome": "chosen", "chosen": top, "why": f"{top['recipe']} at step {top['step']}: {top['level']}x"}
