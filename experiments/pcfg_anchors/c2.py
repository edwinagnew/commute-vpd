"""E55 Stage C2: VPD on the C1b truth set. Plan: reports/plans/2026-10-31_plan_e55-c2-vpd.md. Config:
c2_config.yaml (param-decomp's ss_llama_simple_mlp-2L recipe; C from the released 67M decomposition's C per unit rank,
14,592 subcomponents). Draft predictions: results/predictions_c2.json.

One decomposition of LOW on LOW's own pretraining rows (pcfg.pretrain_batch, prior LOW; 258-token inputs), trained by
`commute.decompose` (param-decomp's step, with saves and resume). Each C1b state is then read by the route split's
re-gating share r and the stand-in ratio (`commute.pcfg_vpd`) on natural 'C b 40' rows labelled with the state's task.

Stages (repo root, GPU):
  rehearse   $0, never scored (plan gate 3), the full config's shapes at REHEARSAL's 400 steps. (1) Readings on an
             untrained decomposition: the matrix share, outside norm share, do-nothing cells and T0 - LOW are the target
             model's and real; r and the ratio exercise the code only. (2) Run A to step 200 and its save. Then, in a
             fresh process (`rehearse-resume`, as a resubmitted job): (3) run A resumed to 400, (4) run B unbroken to
             400 and the comparison, (5) the alive count of A's result. results/c2_rehearsal.json is rewritten after
             each part; a CUDA OOM is recorded with its part (exit 1). Files in results/large/c2_rehearsal/.
  decompose  the full decomposition, resumed from results/large/c2/decomp_train.pt; exit 3 at the deadline (resubmit
             by hand), exit 0 with results/large/c2/decomp_final.pt. Refuses (exit 2) while predictions_c2.json is a
             draft.
`read` and `score` are written after the rehearsal's numbers are in the plan (plan, "Hand-off").

Usage: python experiments/pcfg_anchors/c2.py {rehearse,decompose} [--deadline-min M]  (cluster/pcfg_c2.sbatch)
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from amend3 import texts_of
from c1b import task_rows
from run import (
    EVAL_SEED,
    EXIT_RESUME,
    LARGE,
    RESULTS,
    device,
    load_model,
    log,
    write_json,
)
from torch import Tensor

from commute import decompose as D
from commute import pcfg, pcfg_vpd, vpd
from commute.anchors import lift_context
from commute.pcfg import A
from commute.pcfg_train import Batches

HERE = Path(__file__).resolve().parent
CONFIG, PRED_C2 = HERE / "c2_config.yaml", RESULTS / "predictions_c2.json"
LOW_PATH, C1B_JSON = LARGE / "a1" / "pt_LOW.pt", RESULTS / "c1b.json"
DECOMP_SEED, EVAL_DECOMP_SEED, ALIVE_SEED, READ_TAIL = 200, 201, 202, 60  # data streams; READ_TAIL: [EVAL_SEED, 60]
CHUNK = 50  # batches per generated chunk
REHEARSAL_STATES = ("c1b/P_1e-4_s0/10000", "a2/N1/10000")  # one calibration state per task, both at 10K


@dataclass(frozen=True)
class Sizes:
    """Every size that differs between the rehearsal and the scored run."""

    steps: int  # config.steps (the schedules' length)
    stop_first: int  # 0, or the step where run A stops and is resumed from its file (the rehearsal)
    save_every: int
    log_every: int
    alive_rows: int  # pretraining rows for the alive set (mean causal importance > 1e-6)
    read_windows: int  # natural 'C b 40' windows per reading
    read_chunk: int  # windows per route-cell chunk (host memory: ~40 MB of gates per window at 38K subcomponents)
    read_batch: int

    def __post_init__(self) -> None:
        assert self.steps % self.save_every == 0 and self.save_every % self.log_every == 0, self
        assert self.stop_first == 0 or (0 < self.stop_first < self.steps and self.stop_first % self.log_every == 0), self
        assert self.read_windows % self.read_chunk == 0 and self.read_chunk % self.read_batch == 0, self


REHEARSAL = Sizes(steps=400, stop_first=200, save_every=200, log_every=50, alive_rows=512, read_windows=256, read_chunk=64, read_batch=32)
FULL = Sizes(steps=400_000, stop_first=0, save_every=5_000, log_every=200, alive_rows=8192, read_windows=1024, read_chunk=64, read_batch=32)


def inputs(rows: np.ndarray) -> np.ndarray:
    return rows[:, :-1]  # 258 tokens: the minGPT's block (SEQ - 1)


def pretrain_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    return inputs(pcfg.pretrain_batch(rng, n, pcfg.PRIORS["LOW"]))


def data(cfg: Any) -> tuple[Batches, Any]:
    """Training batches by step and an endless iterator of eval batches (fresh streams, LOW's pretraining rows)."""
    train = Batches(pretrain_rows, DECOMP_SEED, cfg.batch_size, CHUNK, device())
    ev = Batches(pretrain_rows, EVAL_DECOMP_SEED, cfg.eval_batch_size, CHUNK, device())

    def eval_iter() -> Any:
        i = 0
        while True:
            yield ev.get(i)
            i += 1

    return train, eval_iter()


def fresh(cfg: Any) -> D.Decomp:
    return D.build(load_model(LOW_PATH), cfg, (cfg.batch_size, pcfg.SEQ - 1), device())


def decomp_run(cfg: Any, ckpt: Path, plan: D.Plan, with_eval: bool) -> tuple[D.Decomp, int, list[dict[str, Any]]]:
    d = fresh(cfg)
    train, ev = data(cfg)
    step, recs = D.run(d, train.get, ckpt, plan, ev if with_eval else None, log=log)
    train.close()
    return d, step, recs


def free(d: D.Decomp) -> float:
    """Release d's GPU state (decompose.release); returns the GiB still allocated afterwards (should be ~LOW only)."""
    D.release(d)
    return torch.cuda.memory_allocated() / 2**30


# ---- rehearsal ----

REH_JSON, REH_DIR = RESULTS / "c2_rehearsal.json", LARGE / "c2_rehearsal"


def state_path(sid: str) -> Path:
    p = Path(json.loads(C1B_JSON.read_text())["meta"][sid]["path"])
    return LARGE / p.relative_to(p.parents[1])  # rebased on this clone's results/large


def answers(task: str, rows: np.ndarray) -> np.ndarray:
    """Token ids of each row's answer under the task: count(a) + 1 (P) or count(b) (N)."""
    t = texts_of(rows)
    return pcfg.NUM0 + (pcfg.count(t, A) + 1 if task == "P" else pcfg.count(t, pcfg.B))


def alive_set(model: Any, n_rows: int) -> dict[str, Tensor]:
    batches = [torch.from_numpy(pretrain_rows(np.random.default_rng([ALIVE_SEED, i]), 64).astype(np.int64)) for i in range(n_rows // 64)]
    return {k: v.cpu().bool() for k, v in vpd.alive_masks(vpd.mean_ci(model, batches)).items()}


def read_states(model: Any, sz: Sizes, ratio_floor: float) -> dict[str, Any]:
    """Both readings, the matrix share, the outside norm share and T0 - LOW on the zero update and REHEARSAL_STATES.
    The T cells, hence the matrix share and T0 - LOW, are the target model's: real numbers on any decomposition."""
    low = load_model(LOW_PATH)
    t = time.perf_counter()
    keys = vpd.module_keys(model)
    alive = alive_set(model, sz.alive_rows)
    lift = lift_context(model, alive)
    comps0, W0 = vpd.component_weights(model), vpd.target_weights(model)
    rec: dict[str, Any] = {"alive": {k: int(v.sum()) for k, v in alive.items()}, "setup_seconds": time.perf_counter() - t, "states": {}}
    log(f"rehearsal read: alive total {sum(rec['alive'].values())} of {sum(model.module_to_c.values())}; setup {rec['setup_seconds']:.0f} s")
    rows = task_rows("N", np.random.default_rng([EVAL_SEED, READ_TAIL]), sz.read_windows)
    for sid, task in (("zero_update", "P"), (REHEARSAL_STATES[0], "P"), (REHEARSAL_STATES[1], "N")):
        t = time.perf_counter()
        st = low if sid == "zero_update" else load_model(state_path(sid))
        w = pcfg_vpd.windows(rows, answers(task, rows))
        dW = pcfg_vpd.matrix_update(st, low, keys)
        cells = pcfg_vpd.cells_of(model, lift, comps0, W0, dW, w, sz.read_chunk, sz.read_batch)
        lo_low = pcfg_vpd.answer_log_odds(low, w)
        full = pcfg_vpd.answer_log_odds(st, w) - lo_low
        sh = pcfg_vpd.shares(cells, ratio_floor)
        rec["states"][sid] = {"task": task, "cells": cells, **sh, "full_gain": full,
                              "matrix_share": sh["gain"] / full if abs(full) >= pcfg_vpd.GAIN_MIN else float("nan"),
                              "outside_norm_share": pcfg_vpd.outside_norm_share(st, low, keys), "T0_minus_LOW": cells["T0"] - lo_low,
                              "seconds": time.perf_counter() - t, **D.memory()}  # fmt: skip
        log(f"rehearsal read {sid}: {rec['states'][sid]}")
    return rec


def guarded(res: dict[str, Any], part: str, fn: Callable[[], None]) -> bool:
    """Run one part; a CUDA OOM is written into the rehearsal file with the part it hit. False on an OOM."""
    try:
        fn()
    except torch.OutOfMemoryError as e:
        res["oom"] = {"part": part, "error": str(e).splitlines()[0], **D.memory()}
        write_json(REH_JSON, res)
        log(f"rehearsal: CUDA OOM in {part}: {res['oom']['error']} (exit 1)")
        return False
    write_json(REH_JSON, res)
    return True


def stage_rehearse() -> int:
    """Readings first (an untrained decomposition: the target-model numbers are real, r and the ratio are not), then
    run A to stop_first and its save; the resume and run B run in a fresh process (`rehearse-resume`), as a resubmitted
    job would. results/c2_rehearsal.json is rewritten after each part."""
    sz = REHEARSAL
    REH_DIR.mkdir(parents=True, exist_ok=True)
    for p in REH_DIR.glob("*_train.pt"):
        p.unlink()  # stale training files of earlier rehearsals (8.4 GB each at 38K)
    cfg = D.load_config(CONFIG, steps=sz.steps)
    floor = json.loads(PRED_C2.read_text())["constants"]["ratio_floor"]
    res: dict[str, Any] = {"note": "rehearsal ($0, never scored): the full config's shapes, REHEARSAL's steps", "sizes": asdict(sz)}

    def read_untrained() -> None:
        torch.cuda.reset_peak_memory_stats()
        model = D.component_model(load_model(LOW_PATH), cfg, device()).requires_grad_(False).eval()
        res["n_subcomponents"] = sum(model.module_to_c.values())
        res["read_untrained"] = read_states(model, sz, floor)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    def first_half() -> None:
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        d, step, recs = decomp_run(cfg, REH_DIR / "A_train.pt", D.Plan(sz.stop_first, sz.save_every, sz.log_every), with_eval=True)
        res["A_first"] = {"step": step, "seconds": time.perf_counter() - t, "records": recs,
                          "ckpt_gb": (REH_DIR / "A_train.pt").stat().st_size / 1e9, **D.memory()}  # fmt: skip
        res["A_first"]["gpu_left_after_release_gib"] = free(d)

    if not (guarded(res, "read_untrained", read_untrained) and guarded(res, "A_first", first_half)):
        return 1
    log("rehearsal: resuming run A in a fresh process")
    return subprocess.run([sys.executable, __file__, "rehearse-resume"], check=False).returncode


def stage_rehearse_resume() -> int:
    """Run A resumed from its file to the end, then run B unbroken, the comparison and the alive count of A's result."""
    sz, res = REHEARSAL, json.loads(REH_JSON.read_text())
    cfg = D.load_config(CONFIG, steps=sz.steps)
    ck_a, ck_b, final = REH_DIR / "A_train.pt", REH_DIR / "B_train.pt", REH_DIR / "A_final.pt"
    sd: dict[str, dict[str, Tensor]] = {}
    last: dict[str, float] = {}

    def train_part(name: str, ck: Path, with_eval: bool) -> None:
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        d, step, recs = decomp_run(cfg, ck, D.Plan(sz.steps, sz.save_every, sz.log_every), with_eval=with_eval)
        res[name] = {"step": step, "seconds": time.perf_counter() - t, "records": recs, **D.memory()}
        if name == "A_resumed":
            D.save_final(final, d, step)
        sd[name] = {k: v.detach().float().cpu() for k, v in d.model.state_dict().items()}
        last[name] = [r for r in recs if "loss/total" in r][-1]["loss/total"]
        res[name]["gpu_left_after_release_gib"] = free(d)
        ck.unlink(missing_ok=True)  # GB-sized; the readings use A_final.pt

    def alive_after() -> None:
        model, step = D.load_final(final, load_model(LOW_PATH), cfg, device())
        res["alive_after"] = {"step": step, "alive": {k: int(v.sum()) for k, v in alive_set(model, sz.alive_rows).items()}}

    ok = guarded(res, "A_resumed", lambda: train_part("A_resumed", ck_a, True)) and guarded(res, "B_unbroken", lambda: train_part("B_unbroken", ck_b, False))
    if not ok:
        return 1
    diffs = {k: float((sd["A_resumed"][k] - sd["B_unbroken"][k]).abs().max()) for k in sd["A_resumed"]}
    res["resume_vs_unbroken"] = {"max_abs_param_diff": max(diffs.values()), "worst_param": max(diffs, key=lambda k: diffs[k]),
                                 "loss_total_A": last["A_resumed"], "loss_total_B": last["B_unbroken"]}  # fmt: skip
    log(f"rehearsal resume vs unbroken: {res['resume_vs_unbroken']}")
    return 0 if guarded(res, "alive_after", alive_after) else 1


# ---- the full decomposition ----


def stage_decompose(deadline_min: float) -> int:
    if json.loads(PRED_C2.read_text()).get("status", "").startswith("DRAFT"):
        log(f"{PRED_C2.name} is a draft: the full decomposition waits for the filed predictions (exit 2)")
        return 2
    deadline = time.monotonic() + 60 * deadline_min
    cfg = D.load_config(CONFIG, steps=FULL.steps)
    out = LARGE / "c2"
    torch.cuda.reset_peak_memory_stats()
    d, step, recs = decomp_run(cfg, out / "decomp_train.pt", D.Plan(FULL.steps, FULL.save_every, FULL.log_every, deadline), with_eval=True)
    with (out / "decomp_log.jsonl").open("a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    if step < FULL.steps:
        log(f"decompose: step {step} of {FULL.steps}; resubmit to resume (exit {EXIT_RESUME})")
        return EXIT_RESUME
    D.save_final(out / "decomp_final.pt", d, step)
    log(f"decompose: finished at step {step}: {out / 'decomp_final.pt'}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("rehearse", "rehearse-resume", "decompose"))
    ap.add_argument("--deadline-min", type=float, default=22.0)
    a = ap.parse_args()
    assert PRED_C2.exists(), f"{PRED_C2} is missing: predictions are filed before any run"
    log(f"c2 {a.stage} on {torch.cuda.get_device_name()}")
    stages = {"rehearse": stage_rehearse, "rehearse-resume": stage_rehearse_resume}
    sys.exit(stages[a.stage]() if a.stage in stages else stage_decompose(a.deadline_min))


if __name__ == "__main__":
    main()
