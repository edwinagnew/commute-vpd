"""Cell P2.68: whole-group head CMAP (cross-model activation patching after Prakash et al., ICLR 2024) on the 67M.

For each item (a fine-tune W0 + c dW: E237's 15 key items and both sides of E187's nine pairs), the head circuit is
found on the fine-tune by the authors' path patching (their `apply_pp`, unchanged, from a pinned read-only checkout)
and the paper-first minimality rule, its faithfulness read, and then W0 is run on the definition's rows D with the
circuit's heads given the fine-tune's outputs (their `cmap_out`, unchanged). The compared scalar is the whole-circuit
CMAP log-odds share on D; the A+B share is reported for every item. Spec reports/plans/2026-11-23_plan_p2-68-cmap-spec.md
(revision 4); rules and expected values in results/predictions.json (the run refuses unless it is committed unchanged).

  0  every task's rows, the hook floor per format, the gate per task and scale, the do-nothing, the two-layer path,
     the planted head through the whole pipeline, the output-weights blind spot, one timed item and the price
     estimate (exit 1 on any miss)                                                         -> results/stage0.json
  1  every item (sharded), each written as it finishes; an item with a pool over max_pool waits for pricing
     (--large-pools runs it); stops after 4 items if they run over the estimate by rerun_over -> results/items/

    sbatch --comment=E### cluster/cmap.sbatch --stage 0
    sbatch --comment=E### cluster/cmap.sbatch --stage 1 --shard 0 4     (and 1 4, 2 4, 3 4)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from job import ITEMS, PRED, STAGE0, Job, item_path, load_config, log, run_listed, stop
from stage0 import stage0

from commute.jobs import hard_exit, require_filed, run_items


def stage1(job: Job, shard: tuple[int, int], large_ok: bool) -> int:
    s0 = json.loads(STAGE0.read_text()) if STAGE0.is_file() else {}
    if not s0.get("done"):
        log("STOP: stage 0 has not passed")
        return 1
    job.gates, est = s0["gates"], s0["estimate"]["items"]
    todo = [it for i, it in enumerate(job.items.values()) if i % shard[1] == shard[0]]
    ran = {"n": 0, "seconds": 0.0, "estimate": 0.0}

    def one(name: str) -> None:
        it, path = job.items[name], item_path(name)
        if path.is_file() and "waits" not in json.loads(path.read_text()) and "stopped" not in json.loads(path.read_text()):
            log(f"{name}: done before, skipped")
            return
        t0 = time.time()
        rec = run_listed(job, it, large_ok)
        if "waits" in rec:
            return
        ran["n"] += 1
        ran["seconds"] += time.time() - t0
        ran["estimate"] += est[name]
        log(f"ITEM {name} DONE in {time.time() - t0:.0f} s (estimate {est[name]:.0f} s)")

    stopped = []
    for it in todo:
        stopped += [it.name] if run_items([it.name], one, log) else []  # an item's stop ends that item only
        if ran["n"] == 4 and ran["seconds"] > job.cfg.rerun_over * ran["estimate"]:
            stop(ITEMS / f"overrun_shard{shard[0]}.json", ran, f"the first 4 items took {ran['seconds']:.0f} s against {ran['estimate']:.0f} s estimated")
    log(f"STAGE 1 shard {shard} DONE ({ran}; stopped items {stopped})")
    return 1 if stopped else 0


def main(stage: str, authors: Path, shard: tuple[int, int], large_ok: bool) -> int:
    cfg = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout
    log(f"P2.68 CMAP start, stage {stage}, shard {shard}, authors' checkout {authors}; settings {asdict(cfg)}")
    job = Job(cfg, authors)
    log(f"{len(job.items)} items")
    if stage == "0":
        stage0(job)
        code = 0
    else:
        code = stage1(job, shard, large_ok)
    log(f"DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["0", "1"])
    ap.add_argument("--authors", type=Path, required=True, help="the read-only checkout of Nix07/finetuning at 6d6c098")
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="stage 1: every N-th item from the I-th")
    ap.add_argument("--large-pools", action="store_true", help="stage 1: run items whose pool exceeds max_pool (after pricing)")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stage, args.authors, (args.shard[0], args.shard[1]), args.large_pools)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
