"""E55 Stage C2: VPD on the C1b truth set. Plan: reports/plans/2026-10-31_plan_e55-c2-vpd.md. Config:
c2_config.yaml (param-decomp's ss_llama_simple_mlp-2L recipe; C scaled per matrix rank). Draft predictions:
results/predictions_c2.json.

One decomposition of LOW on LOW's own pretraining rows (pcfg.pretrain_batch, prior LOW; 258-token inputs), trained by
`commute.decompose` (param-decomp's step, with saves and resume). Each C1b state is then read by the route split's
re-gating share r and the stand-in ratio (`commute.pcfg_vpd`) on natural 'C b 40' rows labelled with the state's task.

Stages (repo root, GPU):
  rehearse   $0, never scored (plan gate 3). First 100 steps at the 14,592-subcomponent alternative (cost only). With the full config's shapes (C, batch, CI function) but REHEARSAL's step
             count: run A trains to stop_first, saves, is deleted, is rebuilt from the file and trains on to the end;
             run B trains the same steps unbroken; A and B are compared. Then the alive set and LOW's lift context
             from A's decomposition, and both readings on the zero update (do-nothing) and two calibration states.
             Logs ms/step, peak GPU memory, host RSS, checkpoint size and save/load seconds, eval cost, alive counts
             and per-state reading seconds -> results/c2_rehearsal.json. Files in results/large/c2_rehearsal/.
  decompose  the full decomposition, resumed from results/large/c2/decomp_train.pt; exit 3 at the deadline (resubmit
             by hand), exit 0 with results/large/c2/decomp_final.pt. Refuses (exit 2) while predictions_c2.json is a
             draft.
`read` and `score` are written after the rehearsal's numbers are in the plan (plan, "Hand-off").

Usage: python experiments/pcfg_anchors/c2.py {rehearse,decompose} [--deadline-min M]  (cluster/pcfg_c2.sbatch)
"""

from __future__ import annotations

import argparse
import json
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


Dump = Callable[[], None]


def rehearse_decompose(sz: Sizes, out_dir: Path, rec: dict[str, Any], dump: Dump) -> None:
    """Fills `rec` part by part, calling `dump` after each, so a crash keeps what was measured."""
    cfg = D.load_config(CONFIG, steps=sz.steps)
    rec["sizes"] = asdict(sz)
    ck_a, ck_b = out_dir / "A_train.pt", out_dir / "B_train.pt"
    for p in (ck_a, ck_b):
        p.unlink(missing_ok=True)  # incl. a stale file from an earlier rehearsal (job 46971's 8.4 GB A_train.pt)
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    d, step, recs_a1 = decomp_run(cfg, ck_a, D.Plan(sz.stop_first, sz.save_every, sz.log_every), with_eval=True)
    rec["n_subcomponents"] = sum(d.model.module_to_c.values())
    rec["A_first"] = {"step": step, "seconds": time.perf_counter() - t, "records": recs_a1, "ckpt_gb": ck_a.stat().st_size / 1e9, **D.memory()}
    rec["A_first"]["gpu_left_after_release_gib"] = free(d)
    del d
    dump()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    d, step, recs_a2 = decomp_run(cfg, ck_a, D.Plan(sz.steps, sz.save_every, sz.log_every), with_eval=True)
    rec["A_resumed"] = {"step": step, "seconds": time.perf_counter() - t, "records": recs_a2, **D.memory()}
    D.save_final(out_dir / "A_final.pt", d, step)
    sd_a = {k: v.detach().float().cpu() for k, v in d.model.state_dict().items()}
    rec["A_resumed"]["gpu_left_after_release_gib"] = free(d)
    del d
    ck_a.unlink(missing_ok=True)  # ~8 GB; the readings use A_final.pt
    dump()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    d, step, recs_b = decomp_run(cfg, ck_b, D.Plan(sz.steps, sz.save_every, sz.log_every), with_eval=False)
    rec["B_unbroken"] = {"step": step, "seconds": time.perf_counter() - t, "records": recs_b, **D.memory()}
    sd_b = {k: v.detach().float().cpu() for k, v in d.model.state_dict().items()}
    free(d)
    del d
    ck_b.unlink(missing_ok=True)
    diffs = {k: float((sd_a[k] - sd_b[k]).abs().max()) for k in sd_a}
    last_a = [r for r in recs_a2 if "loss/total" in r][-1]
    last_b = [r for r in recs_b if "loss/total" in r][-1]
    rec["resume_vs_unbroken"] = {"max_abs_param_diff": max(diffs.values()), "worst_param": max(diffs, key=lambda k: diffs[k]),
                                 "loss_total_A": last_a["loss/total"], "loss_total_B": last_b["loss/total"]}  # fmt: skip
    log(f"rehearsal resume vs unbroken: {rec['resume_vs_unbroken']}")
    dump()


# The plan's alternative C (section "Settings"): the released 67M decomposition's C per unit rank (c_fc 3072 / 768,
# down_proj 3584 / 768, q, k 512 / 768, v, o 1024 / 768; artifacts/vpd/s-55ea3f9b/final_config.yaml) times 192.
C67_MODULE_INFO = [{"module_pattern": "blocks.*.mlp.0", "C": 768}, {"module_pattern": "blocks.*.mlp.2", "C": 896},
                   {"module_pattern": "blocks.*.attn.qkv", "C": 512}, {"module_pattern": "blocks.*.attn.proj", "C": 256}]  # fmt: skip


def variant_cost(sz: Sizes, out_dir: Path) -> dict[str, Any]:
    """Step time and memory only, at the alternative C (14,592 subcomponents): 2 x log_every steps, no eval, no resume."""
    cfg = D.load_config(CONFIG, steps=sz.steps, module_info=C67_MODULE_INFO)
    ck = out_dir / "C67_train.pt"
    ck.unlink(missing_ok=True)
    torch.cuda.reset_peak_memory_stats()
    d, step, recs = decomp_run(cfg, ck, D.Plan(2 * sz.log_every, 2 * sz.log_every, sz.log_every), with_eval=False)
    out = {"n_subcomponents": sum(d.model.module_to_c.values()), "step": step, "ckpt_gb": ck.stat().st_size / 1e9,
           "ms_per_step": [r["ms_per_step"] for r in recs if "ms_per_step" in r], **D.memory()}  # fmt: skip
    free(d)
    del d
    ck.unlink(missing_ok=True)
    log(f"rehearsal C67 variant: {out}")
    return out


def state_path(sid: str) -> Path:
    p = Path(json.loads(C1B_JSON.read_text())["meta"][sid]["path"])
    return LARGE / p.relative_to(p.parents[1])  # rebased on this clone's results/large


def answers(task: str, rows: np.ndarray) -> np.ndarray:
    """Token ids of each row's answer under the task: count(a) + 1 (P) or count(b) (N)."""
    t = texts_of(rows)
    return pcfg.NUM0 + (pcfg.count(t, A) + 1 if task == "P" else pcfg.count(t, pcfg.B))


def rehearse_read(sz: Sizes, out_dir: Path, ratio_floor: float) -> dict[str, Any]:
    cfg = D.load_config(CONFIG, steps=sz.steps)
    low = load_model(LOW_PATH)
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    model, step = D.load_final(out_dir / "A_final.pt", load_model(LOW_PATH), cfg, device())
    keys = vpd.module_keys(model)
    alive_b = [torch.from_numpy(pretrain_rows(np.random.default_rng([ALIVE_SEED, i]), 64).astype(np.int64)) for i in range(sz.alive_rows // 64)]
    alive = vpd.alive_masks(vpd.mean_ci(model, alive_b))
    lift = lift_context(model, {k: v.cpu().bool() for k, v in alive.items()})
    comps0, W0 = vpd.component_weights(model), vpd.target_weights(model)
    rec: dict[str, Any] = {"decomp_step": step, "alive": {k: int(v.sum()) for k, v in alive.items()},
                           "setup_seconds": time.perf_counter() - t, "states": {}}  # fmt: skip
    log(f"rehearsal read: alive total {sum(rec['alive'].values())} of {sum(model.module_to_c.values())}; setup {rec['setup_seconds']:.0f} s")
    rows = task_rows("N", np.random.default_rng([EVAL_SEED, READ_TAIL]), sz.read_windows)
    for sid, task in (("zero_update", "P"), (REHEARSAL_STATES[0], "P"), (REHEARSAL_STATES[1], "N")):
        t = time.perf_counter()
        st = low if sid == "zero_update" else load_model(state_path(sid))
        w = pcfg_vpd.windows(rows, answers(task, rows))
        dW = pcfg_vpd.matrix_update(st, low, keys)
        cells = pcfg_vpd.cells_of(model, lift, comps0, W0, dW, w, sz.read_chunk, sz.read_batch)
        full = pcfg_vpd.answer_log_odds(st, w) - pcfg_vpd.answer_log_odds(low, w)
        sh = pcfg_vpd.shares(cells, ratio_floor)
        rec["states"][sid] = {"task": task, "cells": cells, **sh, "full_gain": full,
                              "matrix_share": sh["gain"] / full if abs(full) >= pcfg_vpd.GAIN_MIN else float("nan"),
                              "outside_norm_share": pcfg_vpd.outside_norm_share(st, low, keys),
                              "T0_minus_LOW": cells["T0"] - pcfg_vpd.answer_log_odds(low, w),
                              "seconds": time.perf_counter() - t, **D.memory()}  # fmt: skip
        log(f"rehearsal read {sid}: {rec['states'][sid]}")
    return rec


def stage_rehearse() -> int:
    """The 14,592-subcomponent cost first (small; a number even if the full shapes do not fit), then the full shapes'
    decomposition parts, then the readings; results/c2_rehearsal.json is rewritten after each part, and a CUDA OOM is
    recorded there (with the part it hit) before exit 1."""
    out_dir = LARGE / "c2_rehearsal"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "c2_rehearsal.json"
    res: dict[str, Any] = {"note": "rehearsal ($0, never scored): shapes of the full config, REHEARSAL's steps", "decompose": {}}

    def dump() -> None:
        write_json(path, res)

    part = "C67_cost"
    try:
        res["C67_cost"] = variant_cost(REHEARSAL, out_dir)
        dump()
        part = "decompose"
        rehearse_decompose(REHEARSAL, out_dir, res["decompose"], dump)
        part = "read"
        floor = json.loads(PRED_C2.read_text())["constants"]["ratio_floor"]
        res["read"] = rehearse_read(REHEARSAL, out_dir, floor)
        dump()
    except torch.OutOfMemoryError as e:
        res["oom"] = {"part": part, "at": [k for k in ("A_first", "A_resumed", "B_unbroken") if k in res["decompose"]],
                      "error": str(e).splitlines()[0], **D.memory()}  # fmt: skip
        dump()
        log(f"rehearsal: CUDA OOM in {part} after {res['oom']['at']}: {res['oom']['error']} (exit 1)")
        return 1
    return 0


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
    ap.add_argument("stage", choices=("rehearse", "decompose"))
    ap.add_argument("--deadline-min", type=float, default=22.0)
    a = ap.parse_args()
    assert PRED_C2.exists(), f"{PRED_C2} is missing: predictions are filed before any run"
    log(f"c2 {a.stage} on {torch.cuda.get_device_name()}")
    sys.exit(stage_rehearse() if a.stage == "rehearse" else stage_decompose(a.deadline_min))


if __name__ == "__main__":
    main()
