"""E90's filed rules (results/predictions.json; plan reports/plans/2026-11-05_plan_e90-new-vs-reuse-definition.md). Pure: no
model is read, and tests/test_new_vs_reuse.py runs every rule through its edge cases. Comparisons are strict; a nan is
never above a band and never admissible."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

REUSES, NOT_THROUGH = "reuses C", "not through C"
NOT_SPECIFIC, NO_REACH = "unreadable: not specific", "unreadable: does not reach the states"
NO_TEETH, NO_GAIN, NO_CALL = "unreadable: no teeth", "unreadable: no gain", "no call"
AS_FILED, MISS = "as filed", "miss"
DOES_NOT_APPLY, NO_HELD_OUT = "definition does not apply", "no held-out input"
NO_VERSION, NO_C = "no version kept the target", "no C listed"


# ---- A: admissibility, from labels alone ----


def mode(a: np.ndarray) -> int:
    vals, n = np.unique(a, return_counts=True)
    return int(vals[n.argmax()])  # the first maximum: ties go to the lowest token, as removal.lookup_accuracy


def held_out(fit_r: np.ndarray, fit_a: np.ndarray, read_r: np.ndarray, read_a: np.ndarray) -> dict[str, float]:
    """g's accuracies on the read half: identity, and the table (each reading's most common answer on the fit half; unseen
    = wrong), and the floor (the fit half's most common answer read on the read half). removal.lookup_accuracy with the
    two halves named, since E62's halves are not a parity split."""
    table = {int(v): mode(fit_a[fit_r == v]) for v in np.unique(fit_r)}
    got = np.array([table.get(int(v), -1) for v in read_r])
    return {"acc_table": float((got == read_a).mean()), "acc_identity": float((read_r == read_a).mean()),
            "floor": float((read_a == mode(fit_a)).mean()), "unseen": float((got == -1).mean())}  # fmt: skip


def admissibility(fit_r: np.ndarray, fit_a: np.ndarray, read_r: np.ndarray, read_a: np.ndarray) -> dict[str, Any]:
    """C counts iff the best g strictly beats the floor. An empty half: not admissible, "no held-out input" (A9). An answer
    constant over all inputs: not admissible and "definition does not apply" (A3, A5; every g ties the floor)."""
    n = {"n_fit": len(fit_a), "n_read": len(read_a)}
    if not len(fit_a) or not len(read_a):
        return {**n, "admissible": False, "label": NO_HELD_OUT, "best": math.nan, "floor": math.nan}
    r = held_out(fit_r, fit_a, read_r, read_a)
    best = max(r["acc_table"], r["acc_identity"])
    ok = best > r["floor"]
    constant = len(np.unique(np.concatenate([fit_a, read_a]))) == 1
    return {**n, **r, "best": best, "admissible": ok, "label": DOES_NOT_APPLY if constant else "admissible" if ok else "not admissible"}


def parity(reading: np.ndarray, answer: np.ndarray) -> dict[str, Any]:
    """Even-indexed inputs fit, odd-indexed read."""
    return admissibility(reading[::2], answer[::2], reading[1::2], answer[1::2])


def matches(expected: str, row: dict[str, Any]) -> bool:
    """A filed A-row: "admissible", "not admissible" (any non-admissible label) or an exact label."""
    return row["admissible"] == (expected == "admissible") if expected in ("admissible", "not admissible") else row["label"] == expected


# ---- B: possession ----


def possessed(acc: float, shuffled: float, own: float, num: int, den: int) -> bool:
    """Beats its shuffled-label twin (strict) and reaches num/den of W0's own accuracy (a tie reaches, E64's form)."""
    return bool(acc > shuffled and acc * den >= num * own)


def b_star(flags: dict[int, bool], first: int, last: int) -> int | None:
    """The earliest possessed block from `first` up to the one before `last`; None if none."""
    return next((b for b in range(first, last) if flags.get(b)), None)


# ---- D: the verdict ----


def teeth(drop: float, band: float) -> bool:
    return bool(drop > band)


def verdict(has_teeth: bool, s: float, c: float, band: float, control_ok: bool = True) -> str:
    """The appendix's order. A hand write whose match failed (`control_ok` false), a nan control or a nan band: no call."""
    if not has_teeth:
        return NO_TEETH
    if math.isnan(s):
        return NO_GAIN
    if math.isnan(c) or math.isnan(band) or not control_ok:
        return NO_CALL
    st, ct = s > band, c > band
    return NOT_SPECIFIC if st and ct else REUSES if st else NOT_THROUGH if ct else NO_REACH


def say(v: str, s: float) -> str:
    """The verdict as printed: no size bar on "reuses C", so it carries its share."""
    return f"{REUSES}, {s:.3f} of the gain" if v == REUSES else v


def over_seeds(per_seed: Sequence[str]) -> dict[str, Any]:
    """The verdict every seed gives, else "mixed"; none read: no call. Counts the no-gain and no-call seeds."""
    got = set(per_seed)
    v = NO_CALL if not got else got.pop() if len(got) == 1 else "mixed"
    return {"verdict": v, "per_seed": list(per_seed), "n_no_gain": sum(x == NO_GAIN for x in per_seed), "n_no_call": sum(x == NO_CALL for x in per_seed)}


def all_pairs(has_teeth: bool, states: Sequence[float], controls: Sequence[float], band: float) -> dict[str, Any]:
    """Every state against every control (states with no shared seed, E70's P and N)."""
    if not states or not controls:
        return over_seeds([])
    return over_seeds([verdict(has_teeth, s, c, band) for s in states for c in controls])


# ---- the edge cases that read a model ----


def r2(has_teeth: bool, ident: float, rand: float, write: float, band: float, control_ok: bool = True) -> dict[str, Any]:
    """As filed: ident32 against its hand write reads not specific, rand32 is above the band too, and the three shares lie
    within one band of each other (a spread equal to the band is within). Branch: ident32 above with rand32 within, the
    cipher spared. Anything else, a nan included: a miss."""
    v = verdict(has_teeth, ident, write, band, control_ok)
    spread = max(ident, rand, write) - min(ident, rand, write)
    if v == NOT_SPECIFIC and rand > band and not spread > band:
        reading = AS_FILED
    elif has_teeth and ident > band and rand <= band:
        reading = "branch: cipher spared"
    else:
        reading = MISS
    return {"reading": reading, "verdict": v, "spread": spread, "band": band, "shares": {"ident32": ident, "rand32": rand, "write": write}}


def r3(has_teeth: bool | None, rank1: float, write: float, band: float, control_ok: bool = True) -> dict[str, Any]:
    """`has_teeth` None: no possessed block, the plan's "unreadable: no readout". As filed: not specific. Branch: the hand
    write within the band, rank-1 read by its own share. Anything else: a miss."""
    if has_teeth is None:
        return {"reading": "branch: no readout", "verdict": "unreadable: no readout", "band": band}
    v = verdict(has_teeth, rank1, write, band, control_ok)
    spared = v in (REUSES, NO_REACH)
    return {"reading": AS_FILED if v == NOT_SPECIFIC else "branch: hand write spared" if spared else MISS, "verdict": v, "band": band,
            "shares": {"rank1": rank1, "write": write}}  # fmt: skip


def r7(has_teeth: bool, state: float, push: float, band: float, control_ok: bool = True) -> dict[str, Any]:
    """As filed: not specific. Branch: push within the band and the state above; the leaning form is usable, the user
    decides, A3 stands until then. Anything else: a miss."""
    v = verdict(has_teeth, state, push, band, control_ok)
    return {"reading": AS_FILED if v == NOT_SPECIFIC else "branch: leaning form usable, the user decides" if v == REUSES else MISS, "verdict": v, "band": band,
            "shares": {"state": state, "push": push}}  # fmt: skip


def keeps_target(target: int, values: Sequence[int], answers: Sequence[int]) -> bool:
    """The lead's edit: a table is a version of the old task only if the old target is a value and at least one key's answer."""
    return target in values and target in answers


def gap(why: str, **rec: Any) -> dict[str, Any]:
    """A row with nothing trained and nothing read: why it is a gap, and what was printed."""
    return {**rec, "reading": f"gap: {why}", "verdict": why}


def i3(has_teeth: bool, state: float, push: float, band: float, control_ok: bool = True) -> dict[str, Any]:
    """`r7` with I3's filed branches: push spared and the state above, reuses C; push spared and the state within, does not reach."""
    r = r7(has_teeth, state, push, band, control_ok)
    named = {REUSES: "branch: push spared, state above: reuses C", NO_REACH: "branch: push spared, state within: does not reach"}
    return {**r, "reading": named.get(r["verdict"], r["reading"])}


def gaps(readings: dict[str, dict[str, Any]]) -> list[str]:
    """The rows that missed, and those with nothing to read. The definition is frozen (plan, "Scope as restated"): a miss is
    reported as a gap and stage 2 still reads the full table."""
    return [k for k, r in readings.items() if r["reading"] == MISS or r["reading"].startswith("gap: ")]


def table_row(task: str, assumed: str, admissibility_label: str | None, read: str | None) -> dict[str, str]:
    """One line of the deliverable: the verdict read under the definition if the row was read, else the admissibility label;
    a row with neither has no C listed. Every verdict that is not a label of the state carries itself as the reason."""
    verdict = read if read is not None else admissibility_label if admissibility_label is not None else NO_C
    labelled = verdict in (REUSES, NOT_THROUGH)
    return {"task": task, "assumed": assumed, "verdict": verdict, "reason": "" if labelled else verdict}
