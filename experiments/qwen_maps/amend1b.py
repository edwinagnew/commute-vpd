"""E44 Amendment 1b, consistent inputs ($0, one cluster L40). Plan: reports/plans/2026-10-28_plan_qwen-q0q1.md,
"Amendment record (2026-10-29, Amendment 1b: consistent inputs)"; predictions and every rule scored here:
results/predictions_amend1b.json (refuses to start without it).

Amendment 1 gave Γ FT's inputs before layer l and W0's after. Here Γ reads H(l) itself, FT's weights in layers < l
and W0's after, so every input it sees comes from one model: K_H(l) = KL(FT ‖ masked), the masks from `importances`
run on H(l), the masked model on W0's weights, original subcomponents, Δ off. That is amend1's `Run.k_plain(l)`,
reused unchanged, as are its weights, capture, readings, stop rules and δW check.

Stage 1 (controls, alone): the floor (K_H(0) twice on row 0) and K(0), K(28) against Amendment 1's values;
amend1b_controls.json; any miss stops the job. Stage 2: K_H(16) (the verdict), K_H(24), then the unscored
KL_self(0, 16, 24); amend1b_stage2.json and amend1b_score.json after each. Every number also goes to run.log.
Usage (repo root, L40): python experiments/qwen_maps/amend1b.py --decomp DIR [--e40 DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from amend1 import JOB_CFG, NAN, Run, Weights, check_dw, div, ok, r9
from job import DEV, RESULTS, Job, JobOpts, write_json

from commute.qwen_vpd import importances, masked_logits
from commute.vpd import kl_per_position

PRED_B, OUT = RESULTS / "predictions_amend1b.json", RESULTS / "amend1b"


@dataclass(frozen=True)
class B:
    """The filed constants and thresholds (predictions_amend1b.json `rules`)."""

    step: int = 10
    floor_max: float = 1e-3
    rep_tol: float = 1e-3
    k0_a1: float = 1.4956748775221875  # Amendment 1's K_0 = Kplain_W0 (amend1_controls.json)
    k28_a1: float = 0.10243869165192843  # Amendment 1's K_28 = Kplain_FT
    u: tuple[tuple[int, float], ...] = ((16, 0.92620087), (24, 0.99403834))  # Amendment 1's u(l), filed constants
    hi: float = 0.5
    lo: float = 0.2
    self_cuts: tuple[int, ...] = (0, 16, 24)


P = B()
U = dict(P.u)


def controls(v: dict[str, float]) -> dict[str, Any]:
    """control_floor and control_reproduce; NaN or a missing value misses."""
    g = lambda k: float(v.get(k, NAN))
    f = abs(div(g("floor_a"), g("floor_b")) - 1)
    r0, r28 = abs(g("K_0") / P.k0_a1 - 1), abs(g("K_28") / P.k28_a1 - 1)
    out = {"f": f, "rep_K0": r0, "rep_K28": r28, "bitwise": r0 == 0 and r28 == 0,
           "control_floor": ok(f) and r9(f) <= P.floor_max,
           "control_reproduce": ok(r0) and ok(r28) and r9(r0) <= P.rep_tol and r9(r28) <= P.rep_tol}
    out["holds"] = out["control_floor"] and out["control_reproduce"]
    return out


def bin_of(m: float, u: float) -> str:
    if not ok(m):
        return "unreadable"
    if r9(m) >= r9(P.hi * u):
        return "reproduces from consistent inputs"
    return "no reproduction from consistent inputs" if r9(m) <= r9(P.lo * u) else "partial"


def score(v: dict[str, float]) -> dict[str, Any]:
    """m_H(l) from this job's K(0), K(28); the verdict at l = 16 (scored), l = 24 reported."""
    g = lambda k: float(v.get(k, NAN))
    gap = g("K_0") - g("K_28")
    m = {ell: div(g("K_0") - g(f"K_H{ell}"), gap) if gap > 0 else NAN for ell in U}
    return {"m_H": m, "u": U, "verdict": bin_of(m[16], U[16]), "bin_24": bin_of(m[24], U[24]),
            "flags": [f"overshoot {ell}" for ell in U if ok(m[ell]) and m[ell] > 1],
            "KL_self": {ell: g(f"KLself_{ell}") for ell in P.self_cuts} | {28: g("K_28")}}


def kl_self(run: Run, ell: int) -> float:
    """KL(H(ell) ‖ masked with H(ell)'s masks), H(ell)'s logits from the same `importances` call (unscored)."""
    job, sums_ = run.job, []
    for tokens in run.rows:
        t = tokens[None].to(DEV)
        run.W.use(ell)
        h, g = importances(job.model, t, job.gamma(), job.comps())
        run.W.use(0)
        q = masked_logits(job.model, t, job.comps(), g)
        sums_.append(float(kl_per_position(h, q).double().sum()))
        del h, g, q
    v = sum(sums_) / run.rows.numel()
    run.per_row[f"KLself_{ell}"] = [x / run.rows.shape[1] for x in sums_]
    job.log(f"[KLself_{ell}] {v:.9g}  rows {[round(x, 6) for x in run.per_row[f'KLself_{ell}']]}  {job.peaks()}")
    return v


def stage1(run: Run) -> dict[str, float]:
    v: dict[str, float] = {}
    v["floor_a"] = run.reading("floor_a", run.k_plain(0), n_rows=1)
    v["floor_b"] = run.reading("floor_b", run.k_plain(0), n_rows=1)
    v["K_0"] = run.reading("K_0", run.k_plain(0))
    v["K_28"] = run.reading("K_28", run.k_plain(28))
    c = controls(v)
    write_json(OUT / "amend1b_controls.json",
               {"values": v, "controls": c, "per_row": run.per_row, "s": time.time() - run.job.t0})
    run.job.log(f"[controls] {json.dumps(c)}")
    for name in ("control_floor", "control_reproduce"):
        if not c[name]:
            run.job.stop(name, f"{name} missed: {json.dumps(c)}")
    return v


def stage2(run: Run, v: dict[str, float]) -> None:
    run.stage = "2"
    todo = [(f"K_H{ell}", lambda ell=ell: run.reading(f"K_H{ell}", run.k_plain(ell))) for ell in (16, 24)]
    todo += [(f"KLself_{ell}", lambda ell=ell: kl_self(run, ell)) for ell in P.self_cuts]
    s2: dict[str, float] = {}
    for i, (name, read) in enumerate(todo):
        s2[name] = read()
        run.W.use(0)
        write_json(OUT / "amend1b_stage2.json", {"values": s2, "per_row": run.per_row, "s": time.time() - run.job.t0})
        sc = score(v | s2)
        write_json(OUT / "amend1b_score.json", {"controls": controls(v), "score": sc, "complete": i == len(todo) - 1})
        run.job.log(f"[score] {json.dumps(sc)}")
    run.job.log(f"[done] verdict {score(v | s2)['verdict']}, {run.job.peaks()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    ap.add_argument("--e40", default=str(Path.home() / "commute-data/results-large/qwen_sft/nowarmup_regen"))
    ap.add_argument("--sums", default=str(RESULTS / "regen_SHA256SUMS"))
    a = ap.parse_args()
    if not PRED_B.is_file():
        sys.exit(f"refusing to start: {PRED_B} missing")
    if (OUT / "stopped.json").is_file():
        sys.exit(f"refusing to start: {OUT / 'stopped.json'} exists (a stop rule fired; move it aside to rerun)")
    job = Job(JobOpts(out=str(OUT), e40_large="", decomp=a.decomp, results=str(OUT), e40=a.e40), JOB_CFG,
              lambda res, pred: {})
    job.stage = "1"
    job.log(f"[config] {P} rows {JOB_CFG.n_rows} x {JOB_CFG.row_len}, e40 {a.e40}")
    check_dw(job, Path(a.e40) / "dW" / f"step_{P.step}.safetensors", Path(a.sums))
    with torch.no_grad():
        W = Weights(job.model, job.dw(P.step))
        job.forget()
        job.gamma()
        run = Run(job, W, job.rows("maths"))
        run.capture()  # FT's clean logits, the KL reference, exactly as Amendment 1 made them
        run.ft_in.clear()
        run.w0_in.clear()  # Γ's spliced inputs are not read here
        v = stage1(run)
        stage2(run, v)


if __name__ == "__main__":
    main()
