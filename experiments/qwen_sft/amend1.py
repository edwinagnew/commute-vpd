"""E40 Amendment 1 (qwen_sft): the fine-tune with the library's warm-up and cosine schedule, teacher training on a
141 GB card. Plan: reports/plans/2026-10-28_plan_qwen-sft.md, "Amendment 1" and its Gates; rules, first-minutes checks
and thresholds in results/predictions_amend1.json (scored by run.py's `score`). Results in results/amend1/ (a stage
whose file exists there is refused), large files on --out (default /workspace/e40a1), small per-state files in
results/large/amend1/. Every stage is run.py's or defined here; nothing reads another card's results.

Stages, in this order (logical stage in run.py's terms in brackets):
  B   [B]  the benchmark, first on the paid card and within ~10 min: disk pre-flight against the full save list
           (offsets at the alive cap), host RAM and its cgroup limit, a timed W&B upload (with --wandb-upload), one
           generation batch, W0's masked KL, g statistics and the alive set (recomputed), two teacher-training steps
           at the full alive set (step time, peak GPU memory), one adversary pass; the projection against
           --budget-left gives the number of read steps D1 and T1 can afford. Stops: peak, host, disk, budget.
  G1  [G]  W0's adversary on FineWeb (seed 0) and maths rows (seeds 0-2, the maths reference), G_control first.
  L   [L]  one AdamW step from W0 on F's first batch at 1e-7, 1e-6, 1e-5; KL to start on the maths rows. Runs alone
           on a cluster L40 (no decomposition): --stages L.
  A0  [A0] as E40 (P0), its first batch compared with the earlier run's answers (reported).
  F1  [F]  SFT 100 steps: transformers.get_cosine_schedule_with_warmup (10 warm-up steps; step 1 runs at rate 0),
           gradients clipped at 1.0; no Adam moments saved; read steps chosen after it (`run.read_rule`).
  T0  [T0] the teacher-training positive control: the teacher is W0 plus the change of small random offsets on the
           alive columns, sized so KL(teacher ‖ W0) is F1's KL to start at its first read step; trained t0_steps.
  D1  [D]  do-nothing at the read steps (run.stage_D).
  T1  [T]  teacher training at the read steps (run.stage_T); offsets bf16 before the last read step, fp32 at it.
  TR  [TR] rehearsal only: teacher-training rate check, the positive control on 2 layers at 1e-6 and 1e-5.

--rehearse runs every stage at the tiny sizes of A1_TINY on a cluster L40 (46 GB), F1 and L at full size, with stops
logged instead of taken, into results/amend1_rehearse/. Stages that need the decomposition fit there because the
rehearsal loads U, V in bf16 (9.8 GB instead of 19.5) and cuts the rows to 128 tokens; teacher training uses 16 alive
columns per site. --kill-after-save ends the job after T1's first saved offsets, so the next job resumes from them.

Memory, GB = 1e9 B, H200 141 GB (estimates; `mem_gb`, checked against x the filed tolerance): target fp32 6.9, U, V
fp32 19.5, Γ bf16 17.1, alive 3.65e9 parameters (E40), 20 B each in training (73 GB). B: 72.7 building the offsets
with the decomposition resident, ~88 training with it parked (host: U, V and Γ, 36.6). G1, D1: ~64 (adversary with one
state's matrices). F1: ~45 (fp32 copy, gradients, Adam, activations). T0, T1: ~91 training, ~64 reading. Host peak
~70 (U, V and Γ parked plus the offsets after training, 29.2).

Usage (repo root):
  cluster L40, L alone:   .venv/bin/python experiments/qwen_sft/amend1.py --stages L --out /ephemeral/$USER/e40a1
  rehearsal (L40):        cluster/qwen_sft_amend1.sbatch (three jobs; see its header)
  pod (H200):             python experiments/qwen_sft/amend1.py --budget-left <usd> [--wandb-upload]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import run
import torch
from run import (
    BF16,
    CFG,
    DEV,
    LARGE,
    RESULTS,
    Job,
    Layout,
    Stage,
    chosen,
    disk_sizes,
    fall,
    free,
    log,
    peaks,
    run_stages,
    score,
    start,
    teacher_checks,
    w0_adversaries,
    w0_importance,
)

from commute.qwen_eval import load_model, prompt
from commute.qwen_sft import (
    DISK_MARGIN,
    Uploader,
    adamw,
    answer_ids,
    free_bytes,
    mean_kl,
    plus,
    set_volume,
    sft_step,
    trainable,
)
from commute.qwen_teacher import Offsets, random_change, train
from commute.qwen_vpd import SITES

A1 = replace(
    CFG, warmup=10, schedule="hf_cosine", clip=1.0, save_adam=False, read_fixed=(10, 100), read_kl_min=0.01,
    drop_order=("dW_3", "dW_20"), offsets_bf16_before_last=True, offsets_gb_max=110.0, min_card_gb=130.0,
    t0_steps=300, a0_compare="large/nowarmup/step0_test.jsonl",
    mem_gb=(("B", 95.0), ("G", 70.0), ("L", 42.0), ("A0", 30.0), ("F", 50.0), ("T0", 100.0), ("D", 72.0),
            ("T", 100.0)),
)
A1_TINY = replace(  # the rehearsal: F1 and L at full size, the rest tiny; L40 46 GB estimates
    A1, n_rows=4, row_len=128, eval_problems=16, gen_batch=8, alive_max=16, comps_dtype="bfloat16", adv_steps=2,
    t_steps=3, t0_steps=3, t_rows_each=1, t_micro=1, web_pool=8, micro=4, min_card_gb=40.0,
    upload_max_bytes=50_000_000, upload_probe_bytes=10_000_000,
    mem_gb=(("B", 40.0), ("G", 40.0), ("L", 42.0), ("A0", 20.0), ("F", 42.0), ("T0", 40.0), ("D", 44.0),
            ("T", 44.0), ("TR", 30.0)),
)
NAMES = {"B": "B", "G": "G1", "L": "L", "A0": "A0", "F": "F1", "T0": "T0", "D": "D1", "T": "T1", "TR": "TR"}
PRED = RESULTS / "predictions_amend1.json"
LAYOUT = Layout(RESULTS / "amend1", PRED, NAMES, LARGE / "amend1", guard=True)
REHEARSAL = Layout(RESULTS / "amend1_rehearse", PRED, NAMES, LARGE / "amend1_rehearse", guard=True)


# ----------------------------------------------------------------------------- B, the benchmark


def host_memory() -> dict[str, float | None]:
    """Physical RAM and the cgroup's limit (v2 memory.max or v1 limit_in_bytes; None if unlimited), in GB."""
    phys = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    limit = None
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(f).read_text().strip()
        except OSError:
            continue
        limit = None if v == "max" or int(v) > 1e15 else int(v) / 1e9
        break
    return {"physical_gb": phys, "cgroup_gb": limit, "usable_gb": min(phys, limit) if limit else phys}


def preflight(job: Job, may_delete: Path | None) -> dict[str, Any]:
    """Free space on --out (`qwen_sft.free_bytes`: the volume's quota minus its files) against every large file of the
    run, offsets sized at the alive cap. Short of space, `may_delete` (the one file the user allowed to go) is deleted
    and the space read again."""
    cfg = job.cfg
    sizes = disk_sizes(cfg, sum(w.numel() for w in job.W0.values()), int(cfg.offsets_gb_max * 1e9 / 20))
    need = sum(sizes.values()) + DISK_MARGIN + (cfg.upload_probe_bytes if job.uploader else 0)
    job.out.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"free_gb": free_bytes(job.out) / 1e9, "need_gb": need / 1e9,
                           "sizes_gb": {k: v / 1e9 for k, v in sizes.items()}}
    if out["free_gb"] * 1e9 < need and may_delete is not None and may_delete.is_file():
        log(f"[B] deleting {may_delete} ({may_delete.stat().st_size / 1e9:.1f} GB) for space")
        may_delete.unlink()
        out |= {"deleted": str(may_delete), "free_gb": free_bytes(job.out) / 1e9}
    job.check("B disk", out["free_gb"] * 1e9 >= need, {"free_gb": out["free_gb"], "need_gb": need / 1e9})
    return out


def probe_upload(job: Job) -> dict[str, float]:
    """A random file of upload_probe_bytes uploaded to W&B and waited for: the uplink's rate."""
    assert job.uploader is not None
    path = job.out / "upload_probe.bin"
    with path.open("wb") as f:
        for _ in range(0, job.cfg.upload_probe_bytes, 1 << 26):
            f.write(os.urandom(min(1 << 26, job.cfg.upload_probe_bytes)))
    t0 = time.time()
    job.uploader.add(path, wait=True)
    dt = time.time() - t0
    size = path.stat().st_size
    path.unlink()
    return {"bytes": size, "s": dt, "mb_per_s": size / 1e6 / dt}


def projection(job: Job, b: dict[str, Any]) -> dict[str, Any]:
    """Time of every later stage from B's measurements, at 3 and 2 read steps; the worst case of the re-reads (every
    FineWeb ratio and P5 in their bands) beside it. n_read_max: the most read steps whose base spend fits
    --budget-left (0: not even 2)."""
    cfg = job.cfg
    start_s = (cfg.adv_steps + 1) * b["adv_pass_s"]
    n_test, n_1k = (len(job.sets[n][: cfg.eval_problems]) for n in ("test", "gsm1k"))
    gen_all = (math.ceil(n_test / cfg.gen_batch) + math.ceil(n_1k / cfg.gen_batch)) * b["eval_batch_s"]
    step, maths = b["teacher_step_s"], job.pools()[0]
    sft = step * maths.numel() / cfg.steps / (2 * cfg.t_rows_each * cfg.row_len)  # rough: by tokens per step

    def stages(k: int) -> dict[str, float]:
        return {"G1": 4 * start_s, "A0": gen_all, "F1": cfg.steps * sft + len(cfg.saved) * 60 + gen_all,
                "T0": cfg.t0_steps * step + start_s + 120, "D1": (k + 1) * start_s + 300,
                "T1": k * (cfg.t_steps * step + start_s + 120) + start_s + math.ceil(n_test / cfg.gen_batch)
                * b["eval_batch_s"] + 300}

    usd = {k: (time.time() - run.T_START + sum(stages(k).values())) / 3600 * job.usd_per_h for k in (3, 2)}
    worst = {k: (4 * k + 2 + 4) * start_s / 3600 * job.usd_per_h for k in (3, 2)}
    n_max = 3 if usd[3] <= job.budget_usd else 2 if usd[2] <= job.budget_usd else 0
    return {"stages_s": {k: stages(k) for k in (3, 2)}, "projected_usd": usd, "rereads_worst_usd": worst,
            "budget_usd": job.budget_usd, "guard_usd": job.guard_usd, "n_read_max": n_max}


def stage_B(job: Job) -> None:
    cfg, bench = job.cfg, job.early("bench") or {}
    out: dict[str, Any] = {"disk": preflight(job, job.may_delete), "host": host_memory()}
    job.check("B host RAM", out["host"]["usable_gb"] >= bench.get("host_gb_min", 0), out["host"])
    if job.uploader is not None:
        out["upload"] = probe_upload(job)
    tok, gen = load_model()
    texts = [prompt(tok, it["question"]) for it in job.sets["test"][: cfg.gen_batch]]
    t0 = time.time()
    answer_ids(tok, gen, texts, cfg.gen_batch)
    out["eval_batch_s"] = time.time() - t0
    del gen
    free()
    w0_importance(job, out)
    job.put("B", out)
    if out["alive"]["offsets_gb"] > cfg.offsets_gb_max:
        job.halt(f"alive offsets need {out['alive']['offsets_gb']:.1f} GB (20 B x {out['alive']['n_params']})")
    off = Offsets(job.decomp()[0], job.alive())
    job.decomp("cpu")
    curve = train(job.model, off, None, *job.pools(), cfg.t_lr, 2, cfg.t_rows_each, cfg.t_micro, cfg.seed, log,
                  teacher_checks(job, "B", zero=True))
    out |= {"teacher_step_s": curve[-1]["s"], "teacher_peak_gb": torch.cuda.max_memory_allocated() / 1e9}
    del off
    free()
    job.check("B peak memory", out["teacher_peak_gb"] <= bench.get("peak_gb_max", math.inf),
              {"peak_gb": out["teacher_peak_gb"]})
    first = job.pred["three_starts"]["seeds"][0]
    job.adv(nullcontext, job.web, first, "B_fineweb", "fineweb", steps=1)
    out |= {"adv_pass_s": job.last_pass_s, "adv_step0": job.last_step0}
    out |= projection(job, out)
    job.put("B", out)
    if out["n_read_max"] == 0:
        job.halt(f"projected spend over ${job.budget_usd:.1f} even with 2 read steps: {out['projected_usd']}")


# ----------------------------------------------------------------------------- G1, L, F1


def stage_G1(job: Job) -> None:
    """W0's adversary, G_control first: FineWeb at the first seed, maths rows at every seed (the maths reference)."""
    b, seeds = job.res["B"], tuple(job.pred["three_starts"]["seeds"])
    out: dict[str, Any] = {"fineweb": {"masked_kl": b["fineweb"]["masked_kl"]}, "alive": b["alive"],
                           "maths": {"masked_kl": b["maths"]["masked_kl"]}}
    w0_adversaries(job, out, {"fineweb": seeds[:1], "maths": seeds})
    out["maths"]["mean_over_seeds"] = statistics.mean(out["maths"]["by_seed"].values())
    job.put("G", out)


def stage_L(job: Job) -> None:
    """From W0, one AdamW step (no clipping, as E40's F) on F's first batch at each rate; KL to start on the maths
    rows, fp32. The 1e-5 reading reproduces E40's step 1 (the L control)."""
    cfg = job.cfg
    job.park()
    maths, seqs = job.maths(full=True), [job.seq(i) for i in job.train_idx[: cfg.batch]]
    out: dict[str, Any] = {"rates": {}}
    for lr in cfg.l_rates:
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        with trainable(job.model, job.W0) as params:
            opt = adamw(params, lr)
            opt.zero_grad(set_to_none=True)
            job.model.train()
            loss = sft_step(job.model, seqs, cfg.l_micro, DEV)
            job.model.eval()
            opt.step()
            del opt
            kl = mean_kl(job.model, maths, job.W0, None)
        out["rates"][f"{lr:g}"] = {"kl_to_start": kl, "loss": loss, "s": time.time() - t0,
                                   "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
        log(f"[L] rate {lr:g}: KL to start {kl:.4e}, loss {loss:.4f}, {time.time() - t0:.0f} s, {peaks()}")
        job.finite(f"L {lr:g}", kl=kl, loss=loss)
        job.put("L", out)
    if not score(job.res, job.pred, cfg.steps)["L_control"]["holds"]:
        job.halt(f"L control: KL at 1e-5 {out['rates']['1e-05']['kl_to_start']:.4f} off E40's step 1")


def stage_F1(job: Job) -> None:
    if "L" not in job.res or not score(job.res, job.pred, job.cfg.steps)["L_control"]["holds"]:
        job.halt("F1 needs stage L's control to hold first (results/amend1/L.json)")
    run.stage_F(job)


# ----------------------------------------------------------------------------- the positive control, its rate check


def positive_control(job: Job, off: Offsets, target: float, lr: float, steps: int, name: str
                     ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """The teacher W0 + `random_change` at the α whose KL(teacher ‖ W0) on the first pc_rows maths rows is `target`
    (KL ∝ α², pc_iters rounds from α = 1e-3), then teacher training of `off` (zero offsets on the GPU) towards it.
    Returns the teacher's change (bf16) and the record."""
    cfg, rows = job.cfg, job.maths()[: job.cfg.pc_rows]
    alpha, search = 1e-3, []
    for _ in range(cfg.pc_iters):
        kl = mean_kl(job.model, rows, plus(job.W0, random_change(off, alpha, cfg.pc_seed, BF16)), None)
        search.append({"alpha": alpha, "kl": kl})
        job.finite(f"{name} size search", kl=kl)
        if not kl > 0:
            break
        alpha *= math.sqrt(target / kl)
    teacher = random_change(off, alpha, cfg.pc_seed, BF16)
    curve = train(job.model, off, teacher, *job.pools(), lr, steps, cfg.t_rows_each, cfg.t_micro, cfg.seed, log,
                  teacher_checks(job, name, zero=False))
    kls = [c["kl"] for c in curve]
    return teacher, {"target_kl": target, "alpha": alpha, "search": search, "lr": lr, "t_steps": steps,
                     "curve": kls, "first_batch_kl_fp32": curve[0]["kl_fp32_teacher_w0"], "fall": fall(kls)}


def first_read_kl(job: Job) -> float:
    """F1's KL to start at its first read step (the positive control's size)."""
    F = job.res["F"]
    return F["saved"][str(min(F["read_steps"]))]["kl_to_start"]


def stage_T0(job: Job) -> None:
    """The positive control: trained, then behaviour kept on maths rows and the FineWeb adversary of the student."""
    cfg = job.cfg
    off = Offsets(job.decomp()[0], job.alive())
    job.decomp("cpu")
    teacher, out = positive_control(job, off, first_read_kl(job), cfg.t_lr, cfg.t0_steps, "T0")
    off.to("cpu")
    with torch.no_grad():
        Wt, Ws = plus(job.W0, teacher), {s: job.W0[s] + off.change(s, DEV) for s in SITES}
    del teacher
    out["kl_teacher_w0"] = mean_kl(job.model, job.maths(), Wt, None)
    out["kl_teacher_student"] = mean_kl(job.model, job.maths(), Wt, Ws)
    out["max_abs_offset"] = off.max_abs()
    del Wt, Ws
    free()
    out["fineweb"] = job.fineweb(lambda: off.state(job.model, job.decomp()[0], job.W0), "T0")
    job.put("T0", out)
    if not score(job.res, job.pred, cfg.steps)["T0"]["holds"]:
        job.halt(f"T0 positive control: fall {out['fall']:.3f}, ratio {out['fineweb']['ratio']:.3f}")


def stage_TR(job: Job) -> None:
    """The rehearsal's teacher-training rate check: the positive control with offsets on tr_layers only, at each of
    tr_rates for tr_steps, so the rate on Qwen is measured rather than carried from the 67M."""
    cfg = job.cfg
    target = first_read_kl(job) if "read_steps" in job.res.get("F", {}) else cfg.tr_target_kl
    out: dict[str, Any] = {"layers": list(cfg.tr_layers), "rates": {}}
    for lr in cfg.tr_rates:
        off = Offsets(job.decomp()[0], job.alive(cfg.tr_layers))
        job.decomp("cpu")
        _, out["rates"][f"{lr:g}"] = positive_control(job, off, target, lr, cfg.tr_steps, f"TR {lr:g}")
        job.put("TR", out)
        del off
        free()


STAGES: tuple[Stage, ...] = (("B", "B", stage_B), ("G1", "G", stage_G1), ("L", "L", stage_L),
                             ("A0", "A0", run.stage_A0), ("F1", "F", stage_F1), ("T0", "T0", stage_T0),
                             ("D1", "D", run.stage_D), ("T1", "T", run.stage_T), ("TR", "TR", stage_TR))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="B,G1,A0,F1,T0,D1,T1")
    ap.add_argument("--out", default="/workspace/e40a1")
    ap.add_argument("--decomp", default="/workspace/qwen3-1.7b-p-e9fcffd5-torch")
    ap.add_argument("--usd-per-h", type=float, default=4.5)
    ap.add_argument("--budget-left", type=float, default=None, help="dollars left for this run (B's projection)")
    ap.add_argument("--wandb-upload", action="store_true", help="upload every large file as it is written")
    ap.add_argument("--rehearse", action="store_true", help="tiny sizes (A1_TINY), stops logged, own directory")
    ap.add_argument("--kill-after-save", action="store_true", help="rehearsal: end the job after T1's first save")
    ap.add_argument("--volume", default="/workspace", help="the quota'd network volume's mount (none: 'none')")
    ap.add_argument("--volume-gb", type=float, default=150.0, help="its quota: free = quota - du")
    ap.add_argument("--may-delete", default="/workspace/e40_nowarmup/adam_step100.safetensors",
                    help="the one file B may delete when space is short")
    a = ap.parse_args()
    set_volume(None if a.volume == "none" else Path(a.volume), None if a.volume == "none" else a.volume_gb)
    cfg, layout = (A1_TINY, REHEARSAL) if a.rehearse else (A1, LAYOUT)
    stages = chosen(a.stages, STAGES)
    start(layout, stages)
    pred = json.loads(PRED.read_text())
    rule = pred["rules"]["read_steps"]
    assert (tuple(rule["fixed"]), rule["kl_min"]) == (cfg.read_fixed, cfg.read_kl_min), "read rule: file vs config"
    only_l = {s[0] for s in stages} <= {"L"}
    need = cfg.l_card_gb if only_l else cfg.min_card_gb
    assert torch.cuda.get_device_properties(0).total_memory > need * 1e9, f"needs a {need:.0f} GB card"
    if a.budget_left is None and not only_l and not a.rehearse:
        sys.exit("--budget-left (dollars) is needed for B's projection and the re-read guard")
    budget = a.budget_left if a.budget_left is not None else math.inf
    uploader = None
    if a.wandb_upload:
        uploader = Uploader("e40a1-rehearse" if a.rehearse else "e40a1", cfg.upload_max_bytes, Path(a.out) / "wandb", log)
    job = Job(a, cfg, layout, budget_usd=budget, guard_usd=pred["three_starts"]["guard_share"] * budget,
              rehearse=a.rehearse, uploader=uploader, kill_after_save=a.kill_after_save)
    job.may_delete = Path(a.may_delete) if a.may_delete else None
    run_stages(job, stages)


if __name__ == "__main__":
    main()
