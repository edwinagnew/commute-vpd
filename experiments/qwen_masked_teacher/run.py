"""E76: teacher training with masks in its loss, on u1 (E40 step 10), about 200 often-on columns per site ($0, one cluster L40). Plan:
reports/plans/2026-11-03_plan_e76-masked-teacher.md; rules: rules.py, results/predictions.json (the run refuses to start without
it, and while results/stopped.json exists). Stages, in order: 0 (stage0.py), C the control, A clean term alone, B clean + masked
term with all 196 sites masked, L with the library's routing (arms.py). Readings on 8 fixed rows every 50 steps into results/<arm>.json,
final offsets bf16 in results/large/. A stage is skipped when its file says done; an arm resumes from its checkpoint. Past --wall-min,
at a reading: exit 0, resubmit. Each stop: results/stopped.json, its reading in results/run.log, exit 1.

Usage (repo root): .venv/bin/python experiments/qwen_masked_teacher/run.py --decomp DIR --out /ephemeral/$USER/e76 --wall-min 280
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import arms as arms_mod
import rules
import stage0 as stage0_mod
import torch
from common import (
    ARMS,
    DEV,
    DW_FILE,
    DW_SUMS,
    LARGE,
    PRED,
    RESULTS,
    C,
    load_g,
    log,
    peak,
    read_json,
    rows_all,
    stop,
    write_json,
)
from safetensors.torch import load_file

from commute.jobs import hard_exit
from commute.qwen_e40_files import sums
from commute.qwen_sft import load, sha256
from commute.qwen_teacher import Offsets
from commute.qwen_vpd import SITES, fp32_matmuls, load_components, load_target


def score() -> None:
    arms = {a.name: read_json(RESULTS / f"{a.name}.json") for a in ARMS}
    out = rules.score(read_json(RESULTS / "stage0.json"), arms, read_json(RESULTS / "stopped.json"))
    write_json(RESULTS / "score.json", out)
    log(f"[score] {json.dumps(out)}")


def load_dw() -> dict[str, torch.Tensor]:
    got, want = sha256(DW_FILE), sums(DW_SUMS)[DW_FILE.name]
    log(f"[dW] {DW_FILE}: sha256 {got}, want {want}")
    if got != want:
        stop("dW_file", f"{got} != {want}")
    return {s: v for s, v in load(DW_FILE).items() if s in SITES}


def state(model: object, rows: torch.Tensor, decomp: Path, out: Path, wall_end: float) -> arms_mod.State:
    """The arms' residents: the offsets of the cut's columns (fp32 copies of V, U), components in bf16; fp32 stays on the host only to build them."""
    cut = load_file(str(LARGE / "cut.safetensors"))
    host = load_components(decomp, "cpu", torch.float32)
    off = Offsets(host, cut, DEV)
    comps = host.to(DEV, torch.bfloat16)
    del host
    gc.collect()
    log(f"[arms] {off.n_params() / 1e6:.1f} M offsets, {peak()}")
    return arms_mod.State(model, comps, off, rows, [load_g(out, i) for i in range(len(rows))], load_dw(), wall_end)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    ap.add_argument("--out", required=True, help="the node's disk: g per row, ~25 GB")
    ap.add_argument("--wall-min", type=float, required=True)
    a = ap.parse_args()
    if not PRED.is_file() or (RESULTS / "stopped.json").is_file():
        sys.exit("refusing to start: predictions.json missing, or stopped.json exists")
    wall_end, decomp, out = time.time() + a.wall_min * 60, Path(a.decomp), Path(a.out)
    e74_rows, rows = rows_all()
    fp32_matmuls()
    model = load_target(DEV, torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    log(f"[load] {torch.cuda.get_device_name()}, {rows.shape} rows, {peak()}; config {C}")
    with torch.no_grad():
        s0 = stage0_mod.stage0(model, e74_rows, rows, decomp, out)
    log(f"[stage 0] {json.dumps({k: s0[k] for k in ('K0', 'share', 'floor', 'identity_max_abs')})}")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    S = state(model, rows, decomp, out, wall_end)
    for arm in ARMS:
        arms_mod.run_arm(S, arm)
        score()
    log(f"[done] {peak()}")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        code = e.code if isinstance(e.code, int) else 1 if e.code else 0
    except BaseException:  # noqa: BLE001  (an out-of-memory or the like: the same exit, never a hang in teardown)
        traceback.print_exc()
        code = 1
    try:
        score()
    finally:
        hard_exit(code)
