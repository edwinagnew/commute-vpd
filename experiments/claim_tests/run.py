"""claim_tests (E47): do the five red-teamed claims survive several adversary starts and the missing controls?
Plan: reports/plans/2026-10-28_plan_claim-tests.md (section 3; Amendment 1 for t1b, t2b, t2c, t4b, t5b).

The released 67M decomposition, Gamma and Delta frozen. A state is read at the stage's adversarial starts by
`faithfulness.multi_start` ("h<k>" = the Pile harness at adversary seed k, "budget" = E34 Amendments 4-5's start), on
256 Pile val rows (rows 256-511 for t2c), each ratio over W0 read at the same start in the same job (C0, first and
alone). Lines from W0's floor f: pass <= 1.5 / (1 + f), fail >= 1.5 (1 + f).

  t1   claim (i): r1 SGD do-nothing (positive control), r1's copy, rand32 do-nothing, rand32's copy; budget, h0, h1
  t1b  t1's four states at h2
  t2   claim (ii): Adam teacher-trained (positive control, h0), rank-8 and rank-128 LoRA teacher-trained; h0, h1
  t2b  t2 with rank 32 in place of rank 128
  t2c  t2's states on Pile val rows 256-511, W0 and f read there
  t3   claim (iii): keep of the scaled adjoint split, r1 SGD, seeds 0 and 1; no adversary
  t4   claim (iv): emoticon Adam fine-tune at step 1000, do-nothing and teacher training plus remainder; h0, h1, h2
  t4b  t4 at step 10
  t5   claim (v): arm S (plain SGD) cells (0.03, step 100), (0.01, step 1000) on r1's Adam and LoRA teachers; budget, h0
  t5b  t5's cell (0.003, step 1000)
  combine   no model: the combined verdicts (T1 with t1b, T2 with t2b and t2c, T5 with t5b) -> results/combined.json

Each stage writes results/<stage>.json after every state (a rerun on the same card and W0 skips finished states) and
logs every number to results/large/<stage>.log. Usage (repo root, project venv, GPU):  run.py <stage> [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import rules
import stages
from job import RESULTS, STARTS, Job, c0, log

PRED = RESULTS / "predictions.json"
RUN = {"t1": stages.t1, "t1b": stages.t1, "t2": stages.t2, "t2b": stages.t2, "t2c": stages.t2, "t3": stages.t3,
       "t4": stages.t4, "t4b": stages.t4, "t5": stages.t5, "t5b": stages.t5}
assert set(RUN) == set(STARTS)


def filed() -> set[str]:
    """The stage ids filed in results/predictions.json (T1-T5, and Amendment 1's), lower case."""
    pred = json.loads(PRED.read_text())
    return {p["id"].lower() for p in [*pred["predictions"], *pred.get("amendment_1", {}).get("predictions", [])]}


def run(stage: str, device: str) -> None:
    import torch

    if not PRED.is_file() or stage not in filed():
        sys.exit(f"file the predictions for {stage} first: {PRED}")
    t0 = time.time()
    job = Job(stage, device)
    log(f"[{stage}] {job.card}; starts {job.starts}; rows {job.out['rows']}")
    c0(job)
    RUN[stage](job)
    job.out["seconds"] = time.time() - t0
    if torch.cuda.is_available():
        job.out["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    job.out["done"] = True
    job.dump()
    log(f"[{stage} score] {json.dumps(job.out['score'], default=str)}")
    log(f"[{stage}] done in {job.out['seconds']:.0f} s; GPU peak {job.out.get('gpu_peak_gb', 0):.1f} GB")


def load(stage: str) -> dict[str, Any] | None:
    p = RESULTS / f"{stage}.json"
    out = json.loads(p.read_text()) if p.is_file() else None
    return out if out and out.get("done") else None


def combine() -> None:
    """The combined verdicts of Amendment 1 (predictions.json amendment_1.combined), from the finished stage files;
    a missing stage leaves its combination unscored."""
    o = {s: load(s) for s in STARTS}
    res: dict[str, object] = {}
    if o["t1"] and o["t1b"]:
        a, b = o["t1"], o["t1b"]
        floor = {**a["W0"]["floor_starts"], **b["W0"]["floor_starts"]}
        f = max(floor.values()) / min(floor.values()) - 1
        res["T1"] = rules.t1({n: rules.merge(a["states"][n], b["states"][n]) for n in a["states"]}, f) | {"floor": floor}
    if o["t2"] and o["t2b"] and o["t2c"]:
        res["T2"] = rules.t2_combined(o["t2"]["score"], o["t2b"]["score"], o["t2c"]["score"])
    t5, t5b = o["t5"], o["t5b"]
    if t5 and t5b:
        parts = [stages.t5_cells(x, x["W0"]["f"]) for x in (t5, t5b)]
        cells = {k: {**parts[0][0][k], **parts[1][0][k]} for k in ("adam", "lora")}
        res["T5"] = rules.t5(cells, {k: parts[0][1][k] + parts[1][1][k] for k in ("adam", "lora")})
    res["missing"] = [s for s, v in o.items() if v is None]
    (RESULTS / "combined.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    print(json.dumps(res, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=[*RUN, "combine"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    combine() if a.stage == "combine" else run(a.stage, a.device)


if __name__ == "__main__":
    main()
