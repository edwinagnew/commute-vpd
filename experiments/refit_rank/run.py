"""refit_rank (E171, cell P2.25): the effective rank of each saved update on the four 96-key rows of the E133 figure,
the column's last empty cells. E141's reading (`commute.update_spectrum.state`: erank weighted by ||dW_m||^2 over the 24
matrices) and E141's pair rule (separates iff |mean gap| > the larger within-side range) on E147's refit updates
(experiments/refit_96key/results/large/dW_<map>_s<seed>.pt, step 30, seeds 0-2, bf16). Weights only: no model, CPU.
Filed before the run: results/predictions.json. Descriptive: no bar, no verdict on reuse.

  control, first and alone (a miss writes, logs FAIL and exits 1): rank-10 and rank-1 equal-singular-value updates read
    erank 10 and 1; a zero update reads None. Reported only: E58's own saved rank1_s0 step-30 update beside the refit's.
  then each pair: per state erank and norm2 with file sha256, printed as read; the pair's gap, range and call.
Output: results/cells.json (.control, .states, .cells["<pair>/erank"] in the method table's shape).

    ~/commute-agents/fleet/cpu-run -c 2 --mem 4G python experiments/refit_rank/run.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

from commute import update_spectrum as US
from commute.jobs import hard_exit, require_filed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFIT = HERE.parent / "refit_96key" / "results"
E58_SAVED = HERE.parent / "open_pair" / "results" / "large" / "amend3" / "rank1_s0_step30.pt"
OUT = HERE / "results" / "cells.json"


def log(msg: str) -> None:
    print(msg, flush=True)


def control(shapes: dict[str, tuple[int, int]]) -> dict[str, Any]:
    r10, r1 = US.state(US.equal_rank(shapes, 10, 0)), US.state(US.equal_rank(shapes, 1, 1))
    zero = US.state({m: torch.zeros(s, dtype=torch.float64) for m, s in shapes.items()})
    checks = {
        "rank10_erank": abs(float(r10["erank"]) - 10) <= 1e-6,  # type: ignore[arg-type]
        "rank1_erank": abs(float(r1["erank"]) - 1) <= 1e-9,  # type: ignore[arg-type]
        "zero": zero["erank"] is None and zero["norm2"] == 0.0,
    }
    return {"rank10": r10, "rank1": r1, "zero": zero, "checks": checks, "pass": all(checks.values())}


def read(path: Path) -> dict[str, Any]:
    s = US.state(US.load(path))
    return {"erank": s["erank"], "norm2": s["norm2"], "file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def pair_rule(reuse: list[float], new: list[float]) -> dict[str, Any]:
    """E141's rule (E51's): separates iff |mean gap| > the larger within-side range, strict."""
    gap = sum(reuse) / len(reuse) - sum(new) / len(new)
    rng = max(max(reuse) - min(reuse), max(new) - min(new))
    return {"gap": gap, "range": rng, "separates": abs(gap) > rng}


def main() -> None:
    require_filed(HERE / "results" / "predictions.json")
    torch.set_num_threads(2)
    inv = json.loads((REFIT / "inventory.json").read_text())
    path = lambda m, s: REFIT / "large" / f"dW_{m}_s{s}.pt"
    missing = [str(path(p[k], s)) for p in inv["pairs"] for k in ("reuse", "new") for s in inv["seeds"] if not path(p[k], s).is_file()]
    if missing:
        sys.exit(f"missing saved states: {missing}")
    ref = US.load(path("rank1", 0))
    out: dict[str, Any] = {"control": control({m: (v.shape[0], v.shape[1]) for m, v in ref.items()}), "states": {}, "cells": {}}
    log(f"control: {json.dumps(out['control'])}")
    OUT.write_text(json.dumps(out, indent=1))
    if not out["control"]["pass"]:
        log("FAIL: control missed its filed values")
        sys.exit(1)
    out["e58_saved_rank1_s0"] = {"saved": read(E58_SAVED), "refit": read(path("rank1", 0))}
    log(f"e58 saved vs refit rank1_s0: {json.dumps(out['e58_saved_rank1_s0'])}")
    for p in inv["pairs"]:
        sides = {}
        for k in ("reuse", "new"):
            names = [f"{p[k]}_s{s}" for s in inv["seeds"]]
            for n, s in zip(names, inv["seeds"], strict=True):
                out["states"][n] = read(path(p[k], s))
                log(f"{p['id']} {k} {n}: erank {out['states'][n]['erank']:.3f} norm2 {out['states'][n]['norm2']:.6g}")
            sides[k] = [float(out["states"][n]["erank"]) for n in names]
        call = pair_rule(sides["reuse"], sides["new"])
        out["cells"][f"{p['id']}/erank"] = {"reuse": sides["reuse"], "new": sides["new"], "filed": call["separates"], **call,
                                           "source": f"experiments/refit_rank/results/cells.json .states[<state>].erank"}
        log(f"{p['id']}: {json.dumps(call)}")
        OUT.write_text(json.dumps(out, indent=1))
    hard_exit(0)


if __name__ == "__main__":
    main()
