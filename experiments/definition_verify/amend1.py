"""E99 Amendment 1: R-net clause (ii) read as the plan words it, on cased rows only, then stage 3 if it passes.
Plan: reports/plans/2026-11-05_plan_e99-refine-verify-definition.md (R-net). Rules and expected values:
results/predictions_amend1.json (committed first; the run refuses otherwise; never edited). The base filing
results/predictions.json gives the settings, thresholds, P3 and P4, unchanged; S0-ii stays a miss in results/stage0.json.

  gate (no model): rules.moves on stage0.json's stored cased-row move of W0's lowercase-minus-capitalised gap and its
       random-pair band (S0-ii-A1). The stored numbers must equal the filed inputs. The lowercased-row reading is printed
       beside, not part of the rule. A miss writes the record and exits 0 with nothing trained.
  stage 3: run.py's forced-part then forbidden-part lowercase arms, seeds 0 and 1; a forced net share off 1 stops the job.

Writes results/stage3_amend1.json after every arm; <large>/dW_{forced,forbidden}_lowercase_s{0,1}.pt and reads_*.pt as
run.py. Every scored number is printed. Stages 0-2 and their files are not touched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import rules as R
import run

from commute.jobs import hard_exit
from commute.keydep import log, sha256
from commute.locate import from_saved

PRED1, OUT = run.RESULTS / "predictions_amend1.json", run.RESULTS / "stage3_amend1.json"


def gate(s0: dict[str, Any], filed: dict[str, Any]) -> dict[str, Any]:
    """S0-ii-A1 on stage 0's stored cased-row move and band; the lowercased-row reading beside."""
    c, lo = s0["clause_ii"]["cased"], s0["clause_ii"]["e53_lowercase"]
    rec = {"rule": "S0-ii-A1", "move_gap": c["move_gap"], "band": c["band"], "passes": R.moves(c["move_gap"], c["band"]),
           "beside_lowercased_rows": {"move_gap": lo["move_gap"], "band": lo["band"], "moves": R.moves(lo["move_gap"], lo["band"])}}  # fmt: skip
    if (c["move_gap"], c["band"]) != (filed["cased_move_gap"], filed["cased_band"]):
        sys.exit(f"stage0.json's cased move and band {c['move_gap']}, {c['band']} are not the filed inputs {filed}")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    P, P1 = run.filed(run.PRED), run.filed(PRED1)
    s0 = from_saved(json.loads(run.STAGE[0].read_text()))
    if s0.get("status") != "pass":
        sys.exit("stage 0 has not passed")
    out = {"clause_ii_amend1": gate(s0, P1["predictions"][0]["inputs"])}
    run.say("amend1 clause (ii), cased rows", out["clause_ii_amend1"])
    run.dump(OUT, out)
    if not out["clause_ii_amend1"]["passes"]:
        log("stage 3 does not run: amended clause (ii) failed")
        hard_exit(0)
    J = run.start(P, args.device, f"amend1 {sha256(PRED1)} stage 3")
    t0 = time.monotonic()
    run.stage_arms(J, 3, run.trainers(J, {3}), OUT, out)
    log(f"[stage 3 amend1 done] {time.monotonic() - t0:.0f} s")
    hard_exit(0)


if __name__ == "__main__":
    main()
