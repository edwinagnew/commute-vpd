"""P3.4 run 1's filed rules with no model: the drift stop and the choice of how to grow R's update
(results/predictions.json "drift" and "selection"). Imported by run.py as a sibling module; no torch.

A read is {"dw_norm", "kl", "multiple", ...}: `kl` is the output change (KL of the updated model from W0, nats per token
on the 128 maths rows at length 256), `multiple` is kl over the decomposition's masked error. A GSM8K read also has
`pass`, the held-out strict_dropped pass rate, and `fits`, the share of answers that fit the window (gsm8k fits_share);
an output-change-only read has neither, and only GSM8K reads are judged. Comparisons round to 9 places
(`qwen_updates.r9`); a missing value is a miss.
"""

from __future__ import annotations

from typing import Any

from commute.qwen_updates import at_least, ok, r9

D = dict[str, Any]


def fits_bar(refs: D) -> float:
    """W0's share of GSM8K answers that fit the window, minus the filed drop."""
    return refs["w0_fits"] - refs["fits_drop"]


def gsm8k_reads(arm: D) -> dict[int, D]:
    """The arm's reads that include the GSM8K pass rate, by step."""
    return {int(t): r for t, r in arm.get("reads", {}).items() if "pass" in r}


def holds(read: D, pass_bar: float, fits_bar: float) -> bool:
    """Pass rate at least `pass_bar` and the share of answers that fit at least `fits_bar` (a pass rate over fewer
    answers that fit is blind to the ones cut off)."""
    return at_least(read.get("pass"), r9(pass_bar)) and at_least(read.get("fits"), r9(fits_bar))


def drift(read: D, prev_kl: float, w0_pass: float, fits_bar: float) -> bool:
    """A GSM8K read that fails W0's bars (pass below W0's, or fits below `fits_bar`) while the output change still grows
    (above the previous GSM8K read's; 0 before the first). A read with a missing number is not a drift stop: the arm goes
    on and the read shows as missing."""
    p, f, k = read.get("pass"), read.get("fits"), read.get("kl")
    return ok(p) and ok(f) and ok(k) and not holds(read, w0_pass, fits_bar) and r9(k) > r9(prev_kl)  # type: ignore[arg-type]


def level(multiple: float | None, levels: list[float]) -> float:
    """The largest filed multiple (10, 5, 2) the read reaches; 0 for none or a missing value."""
    return next((x for x in sorted(levels, reverse=True) if at_least(multiple, x)), 0.0)


def best_read(name: str, arm: D, bar: float, fits_bar: float, levels: list[float]) -> D | None:
    """The arm's GSM8K read at the largest level among those that hold both bars; ties go to the earlier step. None if no
    read holds them and reaches a level, or the arm was stopped on a check."""
    if arm.get("stopped"):
        return None
    rows = [{"recipe": name, "step": t, "level": level(r.get("multiple"), levels), "multiple": r.get("multiple"),
             "pass": r.get("pass"), "fits": r.get("fits")} for t, r in gsm8k_reads(arm).items() if holds(r, bar, fits_bar)]
    rows = [r for r in rows if r["level"] > 0]
    return min(rows, key=lambda r: (-r["level"], r["step"])) if rows else None


def select(rule: D, w0_pass: float, fits_bar: float, arms: dict[str, D]) -> D:
    """The filed choice among rule["recipes"] (pod names, lowest peak rate first): the recipe whose best read is at the
    largest level, ties to fewer steps, then the higher pass rate, then the lower peak rate. The bar is
    max(control's pass rate at rule["control_step"] - rule["drop"], W0's); the fits bar is W0's share minus its drop. A
    recipe stopped on a check has no eligible read. outcome: "chosen" (level at least rule["min_level"]), "not reached"
    (RL could not get there at these recipes: run 2 goes back to the lead), or "incomplete" (a recipe neither finished
    nor stopped, or the control has no pass rate at its step: no choice is read)."""
    names = rule["recipes"]
    if missing := [n for n in names if not (arms.get(n, {}).get("done") or arms.get(n, {}).get("stopped"))]:
        return {"outcome": "incomplete", "why": f"neither finished nor stopped: {missing}"}
    ref = arms[rule["control"]]["reads"].get(str(rule["control_step"]), {}).get("pass")
    if not ok(ref):
        return {"outcome": "incomplete", "why": f"the control has no pass rate at step {rule['control_step']}"}
    bar = max(ref - rule["drop"], w0_pass)
    best = [b for n in names if (b := best_read(n, arms[n], bar, fits_bar, rule["levels"])) is not None]
    out: D = {"bar": bar, "fits_bar": fits_bar, "control_pass": ref, "w0_pass": w0_pass, "best": best,
              "stopped": [n for n in names if arms[n].get("stopped")]}
    if not best:
        return out | {"outcome": "not reached", "chosen": None, "why": "no read holds the bars at 2x or more"}
    top = min(best, key=lambda b: (-b["level"], b["step"], -b["pass"], names.index(b["recipe"])))
    if top["level"] < rule["min_level"]:
        return out | {"outcome": "not reached", "chosen": None, "why": f"the largest level held is {top['level']}x"}
    return out | {"outcome": "chosen", "chosen": top, "why": f"{top['recipe']} at step {top['step']}: {top['level']}x"}
