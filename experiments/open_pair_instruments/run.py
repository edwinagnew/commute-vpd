"""open_pair_instruments (E55 Stage C3): pruning revival and reverse fine-tuning toward W0's distribution, and the
possession probe (reference), on E58's four 67M key -> value maps (rank-1, cipher, elicit, copy), eight seeds each;
each column's elicit position on the rank-1-to-cipher axis. Plan: reports/plans/2026-10-31_plan_e55-stage-c-open-pair.md.
Rules, constants and predictions: results/predictions.json (two filings; read, never edited here). Instruments and
rules: src/commute/revival_kv.py.

  --stage measure      job 1, after filing A: stop rows, D / prior / MLP share, R's floor, the planted wrapper ->
                       results/measure.json (with `filing_B_draft`). measure.py.
  --stage seeds        job 2, after filing B: seed-0 reproduction control, then seeds 3-7 -> results/seeds.json and
                       results/large/<map>_s<seed>.pt. seeds.py.
  --stage instruments  job 3 (and 4), after job 2: stage 0, the 32 states, q, scores -> results/run.json. instruments.py.
  --stage ladder       the step ladder (plan reports/plans/2026-10-31_plan_e55-c3-ladder.md), after
                       results/predictions_ladder.json is filed: C3's states at K up to 320, then rank-1, cipher and
                       elicit x seeds 0-2 at steps 10-300 -> results/ladder.json. ladder.py.

Filing B (appended to predictions.json after job 1) must hold, under "filing_B": status "filed", date, and the keys the
code reads: D_void, noise_floor_R, planted {units, alpha, floor} (Amendment 1), prior (per map), plus the predictions C0b, P1-P3, V.
Exit codes: 0 done; 1 a stop row (tell the user); 2 run out of order (a filing missing, or results written under other
predictions); 3 deadline: saved, not finished, resubmit the same stage (compute nodes have no sbatch).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from common import (
    ALIVE,
    REPO,
    Clock,
    e58_thresholds,
    header,
    load_predictions,
    load_tasks,
    out_of_order,
)
from instruments import stage_instruments
from ladder import stage_ladder
from measure import OUT as MEASURE_OUT
from measure import stage_measure
from seeds import OUT as SEEDS_OUT
from seeds import stage_seeds

from commute import keydep as KD
from commute import steering as ST

HERE = Path(__file__).resolve().parent
CODE = [str(p.relative_to(REPO)) for p in (*sorted(HERE.glob("*.py")), REPO / "src/commute/revival_kv.py", REPO / "src/commute/finetune.py")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("measure", "seeds", "instruments", "ladder"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--deadline-min", type=float, default=25.0, help="save and exit 3 after this many minutes")
    a = ap.parse_args()
    clock = Clock(a.deadline_min)
    P = load_predictions(need_b=a.stage != "measure")
    C = {**P["constants"], "planted": {**P["constants"]["planted"], **P["amendment_1"]["planted"]}}  # Amendment 1 replaces planted keys
    out = header(CODE)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if a.stage == "ladder":
        return stage_ladder(a.device, C, out, clock)
    if a.stage == "measure":
        if MEASURE_OUT.is_file():
            out_of_order(f"{MEASURE_OUT} exists: job 1 ran; move it aside by hand to rerun")
        return stage_measure(ST.Net(a.device), C, out, clock.end)
    if a.stage == "seeds":
        old = json.loads(SEEDS_OUT.read_text()) if SEEDS_OUT.is_file() else {}
        if old and old["predictions_sha256"] != out["predictions_sha256"]:
            out_of_order(f"{SEEDS_OUT} was written under other predictions: move it aside by hand")
        out = {**old, **out}
        ctx = KD.make_ctx(a.device, ALIVE, {"thresholds": e58_thresholds()}, HERE / "results" / "large")
        return stage_seeds(ctx, load_tasks(ctx.net), C, out, clock)
    net = ST.Net(a.device)
    return stage_instruments(net, load_tasks(net), C, P["filing_B"], out, clock)


if __name__ == "__main__":
    sys.exit(main())
