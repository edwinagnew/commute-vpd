"""E104 Amendment 1: S1 again, RL seed 0's step-100 δW scaled to own-samples' step-100 size, by a bisection in log b
matched on all 128 maths rows (the rows every number is read on), in place of `qr.scale_search` on the first 16.
Plan: reports/plans/2026-11-06_plan_e104-d3-gsm8k.md, "Amendment 1"; rule: results/predictions_amend1.json.
E104's run.py is unchanged; this file calls its read, check and save.

  python experiments/qwen_d3_gsm8k/amend1.py --decomp DIR   # the copy S1a1, then results/amend1.json
  python experiments/qwen_d3_gsm8k/amend1.py --edges        # the rule and the search on synthetic inputs (CPU, no model)

The bracket's ends are E104's own 128-row readings: b = 1 is R_s0_100 (written), and the low end is S1's b, re-read
first as the control. Exit 0 done, 1 a stop (results/stopped.json) or a refusal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import rules
import run
import torch

from commute import qwen_d3 as d3
from commute import qwen_reading as qr
from commute.jobs import hard_exit, require_filed

RESULTS, NAME, SRC, TGT, OLD = rules.RESULTS, "S1a1", "R_s0_100", "F_s0_100", "S1"
PRED, OUT = RESULTS / "predictions_amend1.json", RESULTS / "amend1.json"


def bisect(f: Callable[[float], float], target: float, lo: tuple[float, float], hi: tuple[float, float],
           band: float, tries: int) -> dict[str, Any]:
    """The scale b with f(b) within ×/÷ `band` of target, between the ends lo = (b, f(b)) under target and hi over:
    each try at the geometric midpoint replaces the end on its side; a non-finite f counts as over. Stops at the first
    f within the band, after at most `tries` tries; the closest reading is kept (keys as `qr.scale_search`'s)."""
    assert lo[0] < hi[0] and lo[1] < target < hi[1], f"no bracket: {lo}, {hi}, target {target}"
    evals = [list(lo), list(hi)]
    for _ in range(tries):
        if any(qr.within(x, target, band) for _, x in evals):
            break
        b = math.sqrt(lo[0] * hi[0])
        x = f(b)
        evals.append([b, x])
        lo, hi = ((b, x), hi) if x < target else (lo, (b, x))
    gap = lambda e: abs(math.log(e[1] / target)) if math.isfinite(e[1]) and e[1] > 0 else math.inf
    best = min(evals, key=gap)
    return {"target": target, "evals": evals, "b": best[0], "kl": best[1], "matched": qr.within(best[1], target, band),
            "tries": len(evals) - 2}


def reading(nums: dict[str, Any]) -> dict[str, Any]:
    """E104's size check for C2 at step 100 (`rules.size_check`) with S1a1 in RL's place, and what it decides."""
    sp = lambda arm: rules.spread(rules.vals(nums, arm, 100, "think"))
    out = rules.size_check(nums, NAME, [rules.value(nums, TGT, "think")], [sp("R"), sp("N")], rules.c2(nums, 100, "think"))
    out["next"] = {"size does not explain it": "the gap survives a size-matched copy: ask for own-samples seed 70",
                   "size can do this": "size reproduces the gap: no seed 70 for think overlap"}.get(
        out["reads"], "no call: the gap stays described, not attributed")
    c, m = out.get("copy"), rules.mean(rules.vals(nums, "R", 100, "think"))
    out["beside"] = {"copy_minus_rl_mean": c - m if c is not None and m is not None else None,
                     "source": rules.value(nums, SRC, "think"), "bar": max(sp("R") or 0, sp("N") or 0)}
    return out


def score() -> dict[str, Any]:
    nums = {n: rules.numbers(st) for n in (*rules.STATES, NAME) if (st := rules.load(n)) is not None}
    out = {"reading": reading(nums), "copy": nums.get(NAME)}
    run.write(OUT, out)
    return out


# ----------------------------------------------------------------------------- the job


def read_copy(job: run.Job, P: dict[str, Any]) -> None:
    S = P["search"]
    dW, meta = run.load_checked(f"{NAME}_source", d3.step_state(*rules.STATES[SRC]), job.P)

    def size_at(b: float) -> float:
        job.d.weights.put("FT", dW, scale=b)
        s = d3.cells(job.d, job.m)
        x = (d3.total(s, "K00") - d3.total(s, "K0")) / (len(job.m.tokens) * d3.ROW_LEN)
        run.log(f"[search {NAME}] b {b:.9g}: K00 - K0 {x:.9g} on 128 rows, target {S['target']:.9g}")
        return x

    lo_b = S["lo_b"]
    x = size_at(lo_b)
    run.check(f"{NAME}_lo_end_is_E104s", run.rel(x, S["lo_size"]) <= S["lo_rel"], {"b": lo_b, "got": x, "e104": S["lo_size"]})
    srch = bisect(size_at, S["target"], (lo_b, x), (1.0, S["hi_size"]), S["band"], S["tries"])
    run.log(f"[search {NAME}] {json.dumps(srch)}")
    for t in dW.values():
        t.mul_(srch["b"])
    r, base, large = run.read(job, NAME, dW)
    state = {"name": NAME, "source": SRC, "matched_to": TGT, "search": srch, "source_file": meta, "norm": d3.norm(dW),
             "size": "matched" if srch["matched"] else "size unmatched"} | r
    source = rules.load(SRC)
    assert source is not None
    holds, gap = rules.invariant(rules.numbers(state), rules.numbers(source), job.P["thresholds"]["invariance_band"])
    run.check(f"{NAME}_invariant", holds, {"max_abs_gap": gap})
    run.save(NAME, state, base, large)


def refuse() -> dict[str, Any]:
    if not PRED.is_file():
        sys.exit(f"refusing: {PRED} missing")
    require_filed(PRED)
    if run.STOPPED.is_file():
        sys.exit(f"refusing: {run.STOPPED} exists (a check missed; the lead decides)")
    if not json.loads(run.CONTROL.read_text() if run.CONTROL.is_file() else "{}").get("passed"):
        sys.exit("refusing: E104's stage 0 has not passed")
    P = json.loads(PRED.read_text())
    S, recs = P["search"], {n: rules.load(n) or {} for n in (SRC, TGT, OLD)}
    want = {SRC: S["hi_size"], TGT: S["target"], OLD: S["lo_size"]}
    if any(recs[n].get("size_K00_minus_K0") != x for n, x in want.items()) or (recs[OLD].get("search") or {}).get("b") != S["lo_b"]:
        sys.exit(f"refusing: {SRC}, {TGT} or {OLD} missing or not at the filed sizes")
    return P


def main() -> None:
    ap = argparse.ArgumentParser(description="E104 Amendment 1: S1 by a bisection on 128 rows")
    ap.add_argument("--decomp", default=os.environ.get("COMMUTE_QWEN_DIR", ""))
    ap.add_argument("--edges", action="store_true")
    a = ap.parse_args()
    if a.edges:
        hard_exit(0 if run_edges() else 1)
    P = refuse()
    run.log(f"[a1 start] {vars(a)}; search {json.dumps(P['search'])}")
    with torch.no_grad():
        if not (RESULTS / NAME / "state.json").is_file():
            read_copy(run.build_job(json.loads(run.PRED.read_text()), Path(a.decomp)), P)
    run.log(f"[a1 reading] {json.dumps(score()['reading'], default=float)}; FINISHED")
    hard_exit(0)


# ----------------------------------------------------------------------------- edge cases (CPU)


def edges() -> list[tuple[str, Any, Any]]:
    """The search on power laws through the low end and on broken f; the rule on a binary world (RL 3.75, 3.25; null
    3.5, 3.375; own-samples 2.5: bar 0.5 exactly) and on E104's own think values."""
    S = json.loads(PRED.read_text())["search"]
    lo_b, tgt = S["lo_b"], S["target"]
    power = lambda k: (lambda b: S["lo_size"] * (b / lo_b) ** k)
    srch = lambda f: bisect(f, tgt, (lo_b, f(lo_b)), (1.0, max(f(1.0), 2 * tgt)), S["band"], S["tries"])
    jump = srch(lambda b: S["lo_size"] if b < 0.2 else S["hi_size"])  # no b meets the band
    th = lambda v, chance=8.0, size_S=300: rules.fake(0.1, 0.2, 0.01, v * chance, chance, size_S)  # fold v exactly

    def rd(copy: Any, F: float = 2.5, R: tuple[float, float] = (3.75, 3.25), N: tuple[float, float] = (3.5, 3.375)) -> str:
        w = {n: th(0.0, 0.0, 0) for n in rules.STATES}
        w |= {TGT: th(F), "R_s0_100": th(R[0]), "R_s70_100": th(R[1]), "N_s0_100": th(N[0]), "N_s70_100": th(N[1])}
        return reading(w if copy is None else w | {NAME: copy})["reads"]

    e104 = {"F": 2.534685165421558, "R": (3.645833333333333, 3.211347907471965), "N": (3.497919062039481, 3.3260454947201934)}
    return [
        ("Search, slope 2 (a KL ~ b²): matched", srch(power(2))["matched"], True),
        ("Search, slope 11 (E104's steepest, 16 rows): matched", srch(power(11))["matched"], True),
        ("Search, slope 27 (the cap's limit, 27.8): matched", srch(power(27))["matched"], True),
        ("Search, a jump across the band: unmatched after the cap", (jump["matched"], jump["tries"]), (False, S["tries"])),
        ("Search, NaN inside the bracket counts as over: matched",
         srch(lambda b: math.nan if 0.3 < b < 0.4 else power(2.3)(b))["matched"], True),
        ("Search, an end within the band: no try", srch(lambda b: tgt * 0.9 if b < 0.2 else 1.0)["tries"], 0),
        ("Search, no bracket (low end over target): refused",
         _raises(lambda: bisect(power(2), tgt, (lo_b, 2 * tgt), (1.0, 3 * tgt), S["band"], 8)), True),
        ("Copy over own-samples by more than the bar: survives", rd(th(3.0625)), "size does not explain it"),
        ("Copy at own-samples' value: size can do this", rd(th(2.5)), "size can do this"),
        ("Copy with gap equal to the bar (strict)", rd(th(3.0)), "size can do this"),
        ("Copy under own-samples by more than the bar: survives", rd(th(1.9375)), "size does not explain it"),
        ("Copy unmatched", rd(rules.fake(0.1, 0.2, 0.01, 24, 8.0, matched=False)), "size unmatched"),
        ("Copy not read", rd(None), "copy not read"),
        ("Copy with chance mean under 5", rd(th(3.5, chance=4.99)), "unread"),
        ("Copy at W0 (b = 0, empty S)", rd(th(0.0, 0.0, 0)), "unread"),
        ("Null spread unread", rd(th(3.0625), N=(3.5, math.nan)), "unread"),
        ("E104's values, copy 3.10 (the plan's interpolation): survives", rd(th(3.1), **e104), "size does not explain it"),
        ("E104's values, copy 2.94 (S1's fold, as if matched)", rd(th(2.942958538907643), **e104), "size can do this"),
    ]


def _raises(f: Callable[[], Any]) -> bool:
    try:
        f()
    except AssertionError:
        return True
    return False


def run_edges() -> bool:
    lines = [f"{'ok ' if got == want else 'BAD'}  {case}: {got} (named {want})" for case, got, want in edges()]
    (RESULTS / "edges_amend1.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return all(line.startswith("ok") for line in lines)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        hard_exit(e.code if isinstance(e.code, int) else 1 if e.code else 0)
