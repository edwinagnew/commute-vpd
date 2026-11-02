"""E55 Stage C2's $0 stages, never scored (plan reports/plans/2026-10-31_plan_e55-c2-vpd.md, sections 8 and 8a). The
stages that count are in c2.py.

Stages (repo root, GPU):
  rehearse   the full config's shapes at REHEARSAL's 400 steps. (1) Readings on an untrained decomposition: the matrix
             share, outside norm share, do-nothing cells and T0 - LOW are the target model's and real; r and the ratio
             exercise the code only. (2) Run A to step 200 and its save. Then, in a fresh process (`rehearse-resume`, as
             a resubmitted job): (3) run A resumed to 400, (4) run B unbroken to 400 and the comparison, (5) the alive
             count of A's result. results/c2_rehearsal.json is rewritten after each part; a CUDA OOM is recorded with
             its part (exit 1). Files in results/large/c2_rehearsal/.
  noise      same-seed unbroken arms U1 and U2, E with evaluation, R with a resume, 400 steps each in fresh processes;
             results/c2_noise.json.
  firststeps 2,000 steps on the full schedules; results/c2_firststeps.json sets the early checks.

Usage: python experiments/pcfg_anchors/c2_rehearse.py {rehearse,noise,firststeps}  (cluster/pcfg_c2.sbatch)
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from c2 import (
    CONFIG,
    FULL,
    LOW_PATH,
    PRED_C2,
    REHEARSAL,
    Sizes,
    alive_set,
    data,
    fresh,
    read_one,
    reader,
    state_path,
)
from run import LARGE, RESULTS, device, load_model, log, write_json
from torch import Tensor

from commute import decompose as D

REH_JSON, REH_DIR = RESULTS / "c2_rehearsal.json", LARGE / "c2_rehearsal"
REHEARSAL_STATES = ("c1b/P_1e-4_s0/10000", "a2/N1/10000")  # one calibration state per task, both at 10K


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


def read_states(model: Any, sz: Sizes, ratio_floor: float) -> dict[str, Any]:
    """`c2.read_one` on the zero update and REHEARSAL_STATES. The T cells, hence the matrix share and T0 - LOW, are the
    target model's: real numbers on any decomposition."""
    t = time.perf_counter()
    alive = alive_set(model, sz.alive_rows)
    rd = reader(model, alive, sz, ratio_floor)
    rec: dict[str, Any] = {"alive": {k: int(v.sum()) for k, v in alive.items()}, "setup_seconds": time.perf_counter() - t, "states": {}}
    log(f"rehearsal read: alive total {sum(rec['alive'].values())} of {sum(model.module_to_c.values())}; setup {rec['setup_seconds']:.0f} s")
    for sid, task in (("zero_update", "P"), (REHEARSAL_STATES[0], "P"), (REHEARSAL_STATES[1], "N")):
        st = rd.low if sid == "zero_update" else load_model(state_path(sid))
        rec["states"][sid] = read_one(rd, st, task)
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


# ---- GPU noise floor and first steps (plan section 8a; $0, never scored) ----

NOISE_JSON, NOISE_DIR = RESULTS / "c2_noise.json", LARGE / "c2_noise"
# (arm, with_eval, end): each leg is its own process. U1 against U2 is the noise floor; E adds evaluation, R a resume.
NOISE_LEGS = (("U1", False, 400), ("U2", False, 400), ("E", True, 400), ("R", False, 200), ("R", False, 400))
FIRST_JSON, FIRST_STEPS = RESULTS / "c2_firststeps.json", 2000


def stage_noise_leg(leg: int) -> int:
    """One leg of NOISE_LEGS at REHEARSAL's schedule: its losses and final parameters go to files for `stage_noise`."""
    arm, with_eval, end = NOISE_LEGS[leg]
    sz = REHEARSAL
    cfg = D.load_config(CONFIG, steps=sz.steps)
    _, step, recs = decomp_run(cfg, NOISE_DIR / f"{arm}_train.pt", D.Plan(end, sz.save_every, sz.log_every), with_eval)
    assert step == end
    losses = NOISE_DIR / f"{arm}_losses.json"
    old = json.loads(losses.read_text()) if losses.exists() else {}
    write_json(losses, {**old, **{str(r["step"]): r["loss/total"] for r in recs if "loss/total" in r}})
    return 0


def param_diff(a: str, b: str) -> dict[str, Any]:
    """Largest absolute parameter difference between two arms' last saves, and each logged total loss difference."""
    sd_a, sd_b = (torch.load(NOISE_DIR / f"{x}_train.pt", map_location="cpu", weights_only=False)["model"] for x in (a, b))
    diffs = {k: float((sd_a[k].float() - sd_b[k].float()).abs().max()) for k in sd_a}
    la, lb = (json.loads((NOISE_DIR / f"{x}_losses.json").read_text()) for x in (a, b))
    return {"max_abs_param_diff": max(diffs.values()), "worst_param": max(diffs, key=lambda k: diffs[k]),
            "abs_loss_total_diff": {s: abs(la[s] - lb[s]) for s in la}}  # fmt: skip


def stage_noise() -> int:
    """Every leg in a fresh process, then the three comparisons against U1. Plan section 8a holds the reading rule."""
    NOISE_DIR.mkdir(parents=True, exist_ok=True)
    for p in NOISE_DIR.iterdir():
        p.unlink()  # a stale training file would be resumed from
    for leg in range(len(NOISE_LEGS)):
        log(f"noise: leg {NOISE_LEGS[leg]}")
        rc = subprocess.run([sys.executable, __file__, "noise-leg", "--leg", str(leg)], check=False).returncode
        if rc != 0:
            return rc
    res = {f"U1_vs_{b}": param_diff("U1", b) for b in ("U2", "E", "R")}
    write_json(NOISE_JSON, res)
    log(f"noise: {res}")
    for p in NOISE_DIR.glob("*_train.pt"):
        p.unlink()  # 3.35 GB each
    return 0


def stage_firststeps() -> int:
    """FIRST_STEPS steps on the full run's schedules, evaluated at 0, 1,000 and the end: the values the scored run's
    early checks are set from. Same seed and streams as the scored run; its file is not resumed from."""
    cfg = D.load_config(CONFIG, steps=FULL.steps)
    ck = LARGE / "c2_firststeps_train.pt"
    ck.unlink(missing_ok=True)
    torch.cuda.reset_peak_memory_stats()
    d = fresh(cfg)
    train, ev = data(cfg)
    step, recs = D.run(d, train.get, ck, D.Plan(FIRST_STEPS, FIRST_STEPS, FULL.log_every), ev, log=log)
    train.close()
    recs.append({"step": step, "eval": D.evaluate(d, ev, step)})
    write_json(FIRST_JSON, {"note": "$0, never scored; the full run's schedules", "records": recs, **D.memory()})
    ck.unlink()
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("rehearse", "rehearse-resume", "noise", "noise-leg", "firststeps"))
    ap.add_argument("--deadline-min", type=float, default=22.0, help="not read: the sbatch passes it to every stage; the job's limit is the stop")
    ap.add_argument("--leg", type=int, default=0)
    a = ap.parse_args()
    assert PRED_C2.exists(), f"{PRED_C2} is missing: predictions are filed before any run"
    log(f"c2 {a.stage} on {torch.cuda.get_device_name()}")
    stages: dict[str, Callable[[], int]] = {"rehearse": stage_rehearse, "rehearse-resume": stage_rehearse_resume, "noise": stage_noise,
                                            "noise-leg": lambda: stage_noise_leg(a.leg), "firststeps": stage_firststeps}  # fmt: skip
    sys.exit(stages[a.stage]())


if __name__ == "__main__":
    main()
