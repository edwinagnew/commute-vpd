"""E100's filed rules, pure, so every edge case runs on CPU (tests/test_fed_rules.py). The reads are `commute.locator`'s
`split`; plan reports/plans/2026-11-05_plan_e100-fed-or-written.md. Unit "lo" throughout.

Per state and part u (a head, an MLP, or a pair "a+b"), as shares of the state's gain G:
  carry      the state with u's output from W0's run on the same inputs (E97's read)
  own        the state with u's own update reverted
  graft      W0 with only u's update (beside)
  inherited  carry - own: what reaching u from changed upstream parts adds
  ratio      own / carry, capped at 1 (own past carry reads as written), at the state's read part
"fed" when ratio <= the fed bar (the largest over copy's seeds) and inherited >= the floor (the block-1 plant's largest
inherited share downstream); "written" when ratio >= the written bar (the smallest over the cipher's seeds); else "mixed".
Ties pass on both bars. A part whose carry is at or under the floor has nothing to read.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from commute.copying import finite
from commute.locator_rules import top_part, upstream

FED, WRITTEN, MIXED, BESIDE = "fed", "written", "mixed", "beside"
NO_GAIN, NO_PART, NO_BARS = "unreadable: no gain", "no part to read: carry at or under the floor", "unreadable: no bars"
DISAGREE = "seeds disagree"
Split = Mapping[str, Mapping[str, float]]  # part -> {"carry", "own", "graft"}, unit lo


def inherited(carry: float, own: float) -> float:
    return carry - own if finite(carry) and finite(own) else math.nan


def ratio(carry: float, own: float) -> float:
    """own / carry, capped at 1; nan when carry is not positive or either is not finite. A negative own stays negative."""
    if not (finite(carry) and finite(own)) or carry <= 0:
        return math.nan
    return min(own / carry, 1.0)


def is_single(n: str) -> bool:
    return "+" not in n


def read_at(split: Split, filed: str | None) -> str | None:
    """The filed part (E97's located single part) if any, else the one top single part by carry; a tie names none."""
    if filed is not None:
        return filed
    return top_part({n: v["carry"] for n, v in split.items() if is_single(n)})


def floor_of(split: Split, block: int) -> float:
    """The largest inherited share on a single part downstream of block `block`'s MLP; nan if none is finite."""
    down = [inherited(v["carry"], v["own"]) for n, v in split.items() if is_single(n) and not upstream(n, block) and n != f"mlp{block}"]
    down = [x for x in down if finite(x)]
    return max(down) if down else math.nan


def reverts_sum(split: Split) -> float:
    """The sum of the single parts' own shares: the 28 parts partition every entry of dW, so 1 when the update is additive."""
    xs = [v["own"] for n, v in split.items() if is_single(n)]
    return sum(xs) if xs and all(finite(x) for x in xs) else math.nan


def bars(copy_ratios: Sequence[float], cipher_ratios: Sequence[float]) -> dict[str, Any]:
    """fed = the largest copy ratio, written = the smallest cipher ratio. ok only when both pools are non-empty and finite
    and fed < written (a tie does not separate the poles); the job stops otherwise."""
    good = bool(copy_ratios) and bool(cipher_ratios) and all(finite(x) for x in [*copy_ratios, *cipher_ratios])
    fed, written = (max(copy_ratios), min(cipher_ratios)) if good else (math.nan, math.nan)
    ok = good and fed < written
    why = "" if ok else "a pole has no finite ratio" if not good else f"the copy bar {fed:.4f} is not below the cipher bar {written:.4f}"
    return {"fed": fed, "written": written, "ok": ok, "why": why}


def reading(G: float, carry: float, own: float, floor: float, b: Mapping[str, Any]) -> str:
    """One state's reading at its read part."""
    if not finite(G) or G <= 0:
        return NO_GAIN
    if not (b.get("ok") and finite(floor)):
        return NO_BARS
    if not finite(carry) or carry <= max(floor, 0.0):
        return NO_PART
    r, inh = ratio(carry, own), inherited(carry, own)
    if not finite(r):
        return NO_PART
    if r <= b["fed"] and inh >= floor:
        return FED
    if r >= b["written"]:
        return WRITTEN
    return MIXED


def task_reading(readings: Sequence[str]) -> str:
    """Every seed the same: that reading. Any unreadable seed: the first unreadable, sorted. Else the seeds disagree."""
    bad = sorted({r for r in readings if r.startswith("unreadable")})
    if bad:
        return bad[0]
    if not readings:
        return "unreadable: no seed"
    return readings[0] if len(set(readings)) == 1 else DISAGREE


def row(task: str, seed: int, split: Split, G: float, at: str | None, floor: float, b: Mapping[str, Any], constant: bool) -> dict[str, Any]:
    """The deliverable's row. A constant-answer task reads "beside", its rule's reading kept in `reading_rule`."""
    x = split.get(at, {}) if at else {}
    c, o, g = (x.get(k, math.nan) for k in ("carry", "own", "graft"))
    rd = reading(G, c, o, floor, b) if at else ("unreadable: the carry map's top is tied" if finite(G) and G > 0 else NO_GAIN)
    return {"task": task, "seed": seed, "part": at, "carry": c, "own": o, "inherited": inherited(c, o), "graft": g, "ratio": ratio(c, o),
            "reverts_sum": reverts_sum(split), "G": G, "reading": BESIDE if constant else rd, "reading_rule": rd}  # fmt: skip


def lean_met(task_read: str, lean: str | None) -> bool | None:
    """Whether the task's reading is its filed lean; None where no lean is filed."""
    return None if lean is None else task_read == lean


def outcome(reads: Mapping[str, str]) -> str:
    """The plan's tree on the key-table tasks. `reads` maps the four task names to their task readings."""
    c24, x24, i32, r32 = (reads.get(k, "") for k in ("T13 copy-24", "T11 cipher", "T4 ident32", "T6 rand32"))
    if (i32 == WRITTEN and r32 == FED) or (c24 == x24 and c24 in (FED, WRITTEN)):
        return "does not track the construction labels: report the table as read, no label use"
    if (c24, x24, i32, r32) == (FED, WRITTEN, FED, WRITTEN):
        return "tracks: hand definition-lead the column beside its verdicts"
    return "partial: report each task as read"


# ----------------------------------------------------------------------------- Amendment 1: the sum of single reverts

BETWEEN, NO_GAIN_SUM, NO_SUM = "between", "no gain", "unreadable: sum not finite"


def sum_reading(G: float, s: float, fed: float, written: float, min_gain: float) -> str:
    """E100 Amendment 1's filed rule on one state (plan reports/plans/2026-11-05_plan_e100-amend1.md): G (unit lo, nats)
    at or under `min_gain` reads "no gain"; else the sum of single reverts `s` at or under `fed` reads fed, at or over
    `written` reads written (a sum past 1 counts), else between. A bar's own value takes the bar's side."""
    if not finite(G) or G <= min_gain:
        return NO_GAIN_SUM
    if not finite(s):
        return NO_SUM
    if s <= fed:
        return FED
    return WRITTEN if s >= written else BETWEEN


def sum_pass(copy: Sequence[str], cipher: Sequence[str], need: int) -> dict[str, Any]:
    """The pass rule: copy's seeds reading fed plus the cipher's reading written, at least `need`. A seed not read counts
    as a miss, so `of` is reported beside."""
    hits = sum(r == FED for r in copy) + sum(r == WRITTEN for r in cipher)
    return {"hits": hits, "of": len(copy) + len(cipher), "need": need, "pass": hits >= need}


# ----------------------------------------------------------------------------- stage 0


def plant_misses(split: Split, block: int, G_all: float, th: Mapping[str, Any]) -> list[str]:
    """A one-matrix plant in block `block`'s MLP: own and graft 1 at that MLP and 0 at every other part; carry 0 upstream and
    1 there (so inherited 0 on both); the reverts sum to 1; all 24 reverted reads G 0. Downstream carry is the floor,
    checked by the caller. A nan misses."""
    tol, out = th["reread_tol"], []
    near = lambda x, v: finite(x) and abs(x - v) <= tol
    for n, v in split.items():
        if not is_single(n):
            continue
        at = 1.0 if n == f"mlp{block}" else 0.0
        for k in ("own", "graft"):
            if not near(v[k], at):
                out.append(f"{k} {n} {v[k]}")
        if (upstream(n, block) or n == f"mlp{block}") and not near(v["carry"], at):
            out.append(f"carry {n} {v['carry']}")
    if not near(reverts_sum(split), 1.0):
        out.append(f"reverts sum {reverts_sum(split)}")
    if not (finite(G_all) and abs(G_all) <= th["zero_tol"]):
        out.append(f"all reverted G {G_all}")
    return out


def pole_misses(split: Split, part: str, mode: str, th: Mapping[str, Any]) -> list[str]:
    """E99's constructed copy states at their pair, exact by construction. forced (only the pair's q, k, v rows moved): own,
    carry and graft 1. forbidden (the pair's rows and o columns kept at W0): own and graft 0; carry measured."""
    want = {"forced": {"own": 1.0, "carry": 1.0, "graft": 1.0}, "forbidden": {"own": 0.0, "graft": 0.0}}[mode]
    v = split.get(part, {})
    return [f"{mode} {k} {part} {v.get(k)}" for k, x in want.items() if not (finite(v.get(k)) and abs(v[k] - x) <= th["reread_tol"])]
