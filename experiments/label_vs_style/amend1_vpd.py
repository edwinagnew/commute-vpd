"""E107 Amendment 1 (label_vs_style), 2026-11-06: E107's wrong-answer matched states through E94's VPD reads, so that
their switched-on sets can be set beside the known-facts and fictional arms'. Plan:
reports/plans/2026-11-06_plan_e107-amend1-vpd-sets.md; rules: results/predictions_amend1.json (the run refuses to start
without it); the set reading and the rule: amend1_sets.py.

  python experiments/label_vs_style/amend1_vpd.py --states wrong_s42_matched,wrong_s142_matched

Each state is read exactly as E94 Amendment 1 read the known-facts states (qwen_d3_rows `run.finish`: its norm against
the filed fp64 norm, the split, the 128 maths rows with the gate counts, the overlap and the placement), without the
own-row passes: the switched-on set comes from the maths rows alone, and no prediction reads the own rows. Outputs, as
E94's: experiments/qwen_d3_rows/results/<state>/{maths.json,state.json} and results/large/<state>/counts_pe.safetensors.
Needs E94's stage 0 and unit nulls (written). A missed check writes qwen_d3_rows/results/stopped.json and exits 1;
exit 3: resubmit the same command, written states are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

from commute import qwen_d3 as d3
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment
from commute.qwen_sft import load, sha256

run = load_experiment("qwen_d3_rows", "run")
HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions_amend1.json"
DW = HERE / "results" / "large" / "dW"
WRONG = {"wrong_s42_matched": "WRONG42_matched", "wrong_s142_matched": "WRONG142_matched"}  # name -> file stem in DW


def path(name: str) -> Path:
    return DW / f"{WRONG[name]}.safetensors"


def read_wrong(d: d3.Decomp, rows: dict[str, d3.RowSet], name: str, P: dict[str, Any], want: dict[str, Any]) -> None:
    """amend1.read_known's checks and record, with this experiment's file and filed norm, and no own rows."""
    p = path(name)
    dW = load(p)
    dtypes = {str(x.dtype) for x in dW.values()}
    nrm = d3.norm(dW)
    run.check(f"{name}_norm", dtypes == {"torch.float32"} and run.rel(nrm, want["norm"]) <= P["first_minutes"]["norm_rel"],
              {"norm": nrm, "filed": want["norm"], "dtypes": sorted(dtypes)})
    state = {"name": name, "step": want["step"], "file": str(p.relative_to(EXPERIMENTS)), "norm": nrm, "sha256": sha256(p)}
    d.weights.put("FT", dW)
    run.finish(d, rows, name, dW, (), False, state, P)


def main() -> None:
    ap = argparse.ArgumentParser(description="E107 Amendment 1: the wrong-answer states through E94's reads")
    ap.add_argument("--states", required=True, help=f"comma-separated names from {sorted(WRONG)}")
    ap.add_argument("--decomp", default=os.environ.get("COMMUTE_QWEN_DIR", ""))
    a = ap.parse_args()
    names = a.states.split(",")
    if not PRED.is_file() or "status" in json.loads(PRED.read_text()):
        sys.exit(f"refusing to start: {PRED} missing or has a 'status' key")
    if run.STOPPED.is_file():
        sys.exit(f"refusing to start: {run.STOPPED} exists (a check missed; the lead decides)")
    if any(n not in WRONG for n in names):
        sys.exit(f"give --states from {sorted(WRONG)}")
    if not (json.loads(run.CONTROL.read_text()).get("passed") and run.NULLS.is_file()):
        sys.exit("refusing: E94's stage 0 and unit nulls must be written first")
    P, P1 = json.loads(run.PRED.read_text()), json.loads(PRED.read_text())
    run.log(f"[start e107 amend1] {vars(a)}; config {run.C}")
    rows = run.build_rows(P)
    with torch.no_grad():
        d = d3.Decomp(Path(a.decomp), run.log)
        run.stage_nulls(d, P)  # read back: written by E94
        run.each_state(names, lambda n: read_wrong(d, rows, n, P, P1["states"][n]))
    run.log(f"[done e107 amend1] {names}; FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        hard_exit(e.code if isinstance(e.code, int) else 1 if e.code else 0)
