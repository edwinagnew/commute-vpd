"""E40 Amendment 1 (qwen_sft): the fine-tune with the library's warm-up and cosine schedule, teacher training on a
141 GB card. Plan: reports/plans/2026-10-28_plan_qwen-sft.md, "Amendment 1" and its Gates; rules, first-minutes checks
and thresholds in results/predictions_amend1.json (scored by run.py's `score`). Results in results/amend1/ (a stage
marked finished there is refused; a partial file is overwritten, D and T resume theirs), large files on --out
(default /workspace/e40a1), small per-state files in results/large/amend1/. Every stage is run.py's or defined here;
nothing reads another card's results.

Before any model loads (`before_loading`): the read rule and T0's steps as filed; for F1, L's results and a finished
B; --volume a mount point; the decomposition fetched (`qwen_files.fetch`, ~20 min) when a stage needs it and it is
incomplete, after checking the volume has room for it (53.65 GB) and the run's files.

Stages, in this order (logical stage in run.py's terms in brackets):
  B   [B]  the benchmark, first on the paid card: host RAM and its cgroup limit, one generation batch, W0's masked
           KL, g statistics and the alive set (recomputed), the disk pre-flight at that alive count, the positive
           control at the full alive set at each filed rate (rules.B_rate: 100 steps, fp32 KL(teacher ‖ student) on
           the fixed rows at steps 0, 50, 100; `qualifies`, `pick_rate`) giving step time, peak GPU memory, the rate
           T0 and T1 use and T0's steps (`t0_steps`), one adversary pass; the projection against --budget-left gives
           the number of read steps D1 and T1 can afford. Stops: host, disk, peak memory, no qualifying rate, budget.
  G1  [G]  W0's adversary on FineWeb (seed 0) and maths rows (seeds 0-2, the maths reference), G_control first.
  L   [L]  one AdamW step from W0 on F's first batch at 1e-7, 1e-6, 1e-5; KL to start on the maths rows. Runs alone
           on a cluster L40 (no decomposition): --stages L. Done (results/amend1/L.json: 29.5 GB GPU, 10.9 host).
  A0  [A0] as E40 (P0), its first batch compared with the earlier run's answers (reported).
  F1  [F]  SFT 100 steps: transformers.get_cosine_schedule_with_warmup (10 warm-up steps; step 1 runs at rate 0),
           gradients clipped at 1.0; no Adam moments saved; read steps chosen after it among the saved δW files
           (`run.read_rule`).
  T0  [T0] the teacher-training positive control: the teacher is W0 plus the change of small random offsets on the
           alive columns, sized so KL(teacher ‖ W0) on the fixed rows is F1's KL to start at its first read step; B's
           t0_steps at B's rate; the fall is the fixed-row KL at the end over step 0's (rules.T0).
  D1  [D]  do-nothing at the read steps (run.stage_D), at step 100 the maths pass test (rule M).
  T1  [T]  teacher training at the read steps (run.stage_T); offsets bf16 before the last read step, fp32 at it, each
           state read back from its file; resumed only when δW, rate, steps and alive set match.

Teacher training runs the fp32 target, matmuls in TF32, with gradient checkpointing (`qwen_teacher` docstring: bf16
autocast was 18-44% off fp32 on the first batch, TF32 2.4e-4). With --wandb-upload only results JSONs, the log and T1's
last offsets are uploaded, at the job's end; δW and the other large files stay on the volume, which outlives the pod.

--rehearse runs every stage at the tiny sizes of A1_TINY on a cluster L40 (46 GB), F1 and L at full size, with stops
logged instead of taken, into results/amend1_rehearse/, and writes rehearsal.json from checks.json: every miss of
every job so far, the ones expected at tiny size and any beyond them. Stages that need the decomposition fit there
because the rehearsal loads U, V in bf16 (9.8 GB instead of 19.5) and cuts the rows to 128 tokens; teacher training
uses 16 alive columns per site. --kill-after-save ends the job after T1's first saved offsets, so the next job resumes
from them.

Memory, GB = 1e9 B, H200 141 GB (estimates from the code; the one filed limit is 125): target fp32 6.9, U, V fp32
19.5, Γ bf16 17.1; alive 2.39e9 parameters in the rehearsal's B (328,041 subcomponents; E40 counted 3.65e9), 20 B each
in training (47.9 GB; 73 at E40's count). Teacher training, decomposition parked: target 6.9 + offsets 47.9 + δW bf16
2.8 + two fp32 copies of the 196 matrices while a fixed-row reading or the step-1 reference runs (11.3) + logits and
KL of a 2-row micro-batch (~3) + checkpointed fp32 activations and one layer's changes (<1): ~72 (~97 at E40's count).
B: 72.7 building the offsets with the decomposition resident. G1, D1: ~60 (first pod). F1: ~40 (first pod 39.9).
A0: ~24 (first pod 23.7). Host peak ~70 (U, V and Γ parked plus the offsets after training), of 251 on the first pod.

Usage (repo root):
  cluster L40, L alone:   cluster/qwen_sft_amend1.sbatch --stages L (see its header)
  rehearsal (L40):        cluster/qwen_sft_amend1.sbatch (five jobs; see its header)
  pod (H200):             python experiments/qwen_sft/amend1.py --budget-left <usd>
"""

from __future__ import annotations

import argparse
import itertools
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
    fixed_fall,
    free,
    log,
    peaks,
    r9,
    run_stages,
    score,
    start,
    teacher_checks,
    w0_adversaries,
    w0_importance,
)

from commute import qwen_files
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
from commute.qwen_teacher import Offsets, kl_to_teacher, random_change, train
from commute.qwen_vpd import SITES

A1 = replace(
    CFG, warmup=10, schedule="hf_cosine", clip=1.0, save_adam=False, read_fixed=(10, 100), read_kl_min=0.01,
    drop_order=("dW_20",), offsets_bf16_before_last=True, offsets_gb_max=110.0, min_card_gb=130.0,
    t0_steps=300, a0_compare="large/nowarmup/step0_test.jsonl",
)
A1_TINY = replace(  # the rehearsal: F1 and L at full size, the rest tiny, on an L40 (46 GB)
    A1, n_rows=4, row_len=128, eval_problems=16, gen_batch=8, alive_max=16, comps_dtype="bfloat16", adv_steps=2,
    t_steps=3, t_rows_each=1, t_micro=1, web_pool=8, micro=4, min_card_gb=40.0, upload_max_bytes=50_000_000,
    expected_misses=(  # tiny rows against full-size references, 16 problems, 3 teacher steps, 16 columns per site
        "B_fineweb seed 0 step-0 KL", "G_fineweb seed 0 step-0 KL", "G_maths seed 0 step-0 KL", "G control",
        "P0:", "T0 positive control", "pass time"),
)
DIMS = {"q": (2048, 2048), "k": (2048, 1024), "v": (2048, 1024), "o": (2048, 2048), "gate": (2048, 6144),
        "up": (2048, 6144), "down": (6144, 2048)}  # (d_in, d_out) of Qwen3-1.7B's decomposed matrices
MATRIX_NUMEL = 28 * sum(i * o for i, o in DIMS.values())  # 1,409,286,144: δW's entries, before the model loads
NAMES = {"B": "B", "G": "G1", "L": "L", "A0": "A0", "F": "F1", "T0": "T0", "D": "D1", "T": "T1"}
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


def disk_need(cfg: Any, n_alive_params: int, extra: int) -> tuple[int, dict[str, int]]:
    """Bytes the run will write (`run.disk_sizes` at this alive count) plus `extra` (a decomposition still to fetch)
    and the margin."""
    sizes = disk_sizes(cfg, MATRIX_NUMEL, n_alive_params)
    return sum(sizes.values()) + extra + DISK_MARGIN, sizes


def preflight(job: Job, may_delete: Path | None) -> dict[str, Any]:
    """Free space on --out (`qwen_sft.free_bytes`: the volume's quota minus its files, or `df` if smaller) against every
    large file of the run at the real alive count. Short of space, `may_delete` (a file the user allowed to go) is
    deleted and the space read again."""
    need, sizes = disk_need(job.cfg, job.alive_info()["n_params"], 0)
    out: dict[str, Any] = {"free_gb": free_bytes(job.out) / 1e9, "need_gb": need / 1e9,
                           "sizes_gb": {k: v / 1e9 for k, v in sizes.items()}}
    if out["free_gb"] * 1e9 < need and may_delete is not None and may_delete.is_file():
        log(f"[B] deleting {may_delete} ({may_delete.stat().st_size / 1e9:.1f} GB) for space")
        may_delete.unlink()
        out |= {"deleted": str(may_delete), "free_gb": free_bytes(job.out) / 1e9}
    job.check("B disk", out["free_gb"] * 1e9 >= need, {"free_gb": out["free_gb"], "need_gb": need / 1e9})
    return out


def projection(job: Job, b: dict[str, Any]) -> dict[str, Any]:
    """Time of every later stage from B's measurements, at 3 and 2 read steps (no upload time: large files stay on
    the volume); the worst case of the re-reads (every FineWeb ratio, the maths pass test and P5 in their bands)
    beside it. n_read_max: the most read steps whose base spend fits --budget-left (0: not even 2)."""
    cfg = job.cfg
    start_s = (cfg.adv_steps + 1) * b["adv_pass_s"]
    n_test, n_1k = (len(job.sets[n][: cfg.eval_problems]) for n in ("test", "gsm1k"))
    gen_all = (math.ceil(n_test / cfg.gen_batch) + math.ceil(n_1k / cfg.gen_batch)) * b["eval_batch_s"]
    step, maths = b["teacher_step_s"], job.pools()[0]
    sft = step * maths.numel() / cfg.steps / (2 * cfg.t_rows_each * cfg.row_len)  # rough: by tokens per step

    def stages(k: int) -> dict[str, float]:
        return {"G1": 4 * start_s, "A0": gen_all, "F1": cfg.steps * sft + len(cfg.saved) * 60 + gen_all,
                "T0": b["t0_steps"] * step + start_s + 120, "D1": (k + 1) * start_s + 300,
                "T1": k * (cfg.t_steps * step + start_s + 120) + start_s + math.ceil(n_test / cfg.gen_batch)
                * b["eval_batch_s"] + 300}

    usd = {k: (time.time() - run.T_START + sum(stages(k).values())) / 3600 * job.usd_per_h for k in (3, 2)}
    worst = {k: (4 * k + 2 + 4 + 4) * start_s / 3600 * job.usd_per_h for k in (3, 2)}
    n_max = 3 if usd[3] <= job.budget_usd else 2 if usd[2] <= job.budget_usd else 0
    return {"stages_s": {k: stages(k) for k in (3, 2)}, "projected_usd": usd, "rereads_worst_usd": worst,
            "budget_usd": job.budget_usd, "guard_usd": job.guard_usd, "n_read_max": n_max}


def qualifies(kl: dict[str, float], reads: list[int], ratio_max: float) -> bool:
    """The filed rate rule for one rate: fixed-row KL readings at `reads` (step -> KL) all present and finite, a
    positive start, strictly falling from each reading to the next (a tie does not), and last / first <= ratio_max."""
    v = [kl.get(str(t), math.nan) for t in reads]
    return (all(map(math.isfinite, v)) and v[0] > 0 and all(a > b for a, b in itertools.pairwise(v))
            and r9(v[-1] / v[0]) <= ratio_max)


def pick_rate(readings: dict[float, dict[str, float]], reads: list[int], ratio_max: float) -> float | None:
    """The largest rate that `qualifies`; None if none does."""
    return max((lr for lr, kl in readings.items() if qualifies(kl, reads, ratio_max)), default=None)


def t0_steps(kl: dict[str, float], frac: float, cap: int) -> int:
    """T0's steps from B's readings at the chosen rate: the step at which the fixed-row KL reaches `frac` of its
    start, extrapolated linearly in log KL through step 0 and B's last reading N; N when that reading is already at or
    below `frac` (0 included), at most `cap`. Needs a rate that `qualifies` (a positive start, last / first < 1)."""
    n = max(map(int, kl))
    r = kl[str(n)] / kl["0"]
    return n if r <= frac else min(cap, math.ceil(n * math.log(frac) / math.log(r)))


def stage_B(job: Job) -> None:
    """The benchmark (module docstring). The positive control at the full alive set gives teacher training's step
    time, peak memory, its rate (`pick_rate`) and T0's steps (`t0_steps`)."""
    cfg, bench, rule = job.cfg, job.early("bench") or {}, job.pred["rules"]["B_rate"]
    out: dict[str, Any] = {"host": host_memory()}
    job.check("B host RAM", out["host"]["usable_gb"] >= bench.get("host_gb_min", 0), out["host"])
    tok, gen = load_model()
    texts = [prompt(tok, it["question"]) for it in job.sets["test"][: cfg.gen_batch]]
    t0 = time.time()
    answer_ids(tok, gen, texts, cfg.gen_batch)
    out["eval_batch_s"] = time.time() - t0
    del gen
    free()
    w0_importance(job, out)
    job.put("B", out)
    out["disk"] = preflight(job, job.may_delete)
    job.put("B", out)
    if out["alive"]["offsets_gb"] > cfg.offsets_gb_max:
        job.halt(f"alive offsets need {out['alive']['offsets_gb']:.1f} GB (20 B x {out['alive']['n_params']})")
    off = Offsets(job.decomp()[0], job.alive())
    job.decomp("cpu")
    out["rates"] = {}
    reads, steps = rule["reads"], max(rule["reads"])
    for lr in rule["rates"]:
        off.reset()
        _, rec = positive_control(job, off, rule["target_kl"], lr, steps, f"B {lr:g}", tuple(reads), None)
        rec["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        out["rates"][f"{lr:g}"] = rec | {"qualifies": qualifies(rec["kl"], reads, rule["ratio_max"])}
        job.put("B", out)
    del off
    free()
    out["teacher_step_s"] = statistics.mean(r["step_s"] for r in out["rates"].values())
    out["teacher_peak_gb"] = max(r["peak_gb"] for r in out["rates"].values())
    job.mem_check("teacher training")
    t_lr = pick_rate({float(k): r["kl"] for k, r in out["rates"].items()}, reads, rule["ratio_max"])
    job.check("B rate", t_lr is not None, {k: {"kl": r["kl"], "qualifies": r["qualifies"]}
                                           for k, r in out["rates"].items()})
    out["t_lr"] = t_lr if t_lr is not None else cfg.t_lr  # a rehearsal goes on at the config's rate and T0's cap
    out["t0_steps"] = (t0_steps(out["rates"][f"{t_lr:g}"]["kl"], job.pred["rules"]["T0"]["fall_max"], cfg.t0_steps)
                       if t_lr is not None else cfg.t0_steps)
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
    if not score(job.res, job.pred, job.cfg.steps)["L_control"]["holds"]:
        job.halt("F1 needs stage L's control to hold first (results/amend1/L.json)")
    run.stage_F(job)


# ----------------------------------------------------------------------------- the positive control, its rate check


def positive_control(job: Job, off: Offsets, target: float, lr: float, steps: int, name: str,
                     reads: tuple[int, ...], end_stops: bool | None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """The teacher W0 + `random_change` at the α whose KL(teacher ‖ W0) on the fixed rows (`Job.fixed_rows`) is
    `target` (KL ∝ α², pc_iters rounds from α = 1e-3), then teacher training of `off` (zero offsets on the GPU)
    towards it, the fixed-row KL(teacher ‖ student) read at step 0 and at `reads` (`run.teacher_checks`, with its
    end check stopping per `end_stops`). Returns the teacher's change (bf16) and the record; "fall" is the last
    reading over step 0's."""
    cfg, rows = job.cfg, job.fixed_rows()
    alpha, search = 1e-3, []
    for _ in range(cfg.pc_iters):
        kl = mean_kl(job.model, rows, plus(job.W0, random_change(off, alpha, cfg.pc_seed, BF16)), None)
        search.append({"alpha": alpha, "kl": kl})
        job.finite(f"{name} size search", kl=kl)
        if not kl > 0:
            break
        alpha *= math.sqrt(target / kl)
    teacher = random_change(off, alpha, cfg.pc_seed, BF16)
    on_step, fixed = teacher_checks(job, name, lambda: kl_to_teacher(job.model, rows, job.W0, teacher, off), reads,
                                    end_stops)
    curve = train(job.model, off, teacher, *job.pools(), lr, steps, cfg.t_rows_each, cfg.t_micro, cfg.seed, log,
                  on_step)
    step_s = statistics.mean(c["s"] for c in curve[1:]) if len(curve) > 1 else curve[0]["s"]
    return teacher, {"target_kl": target, "alpha": alpha, "search": search, "lr": lr, "t_steps": steps,
                     "kl": fixed, "fall": fixed_fall(fixed), "curve": [c["kl"] for c in curve],
                     "first_batch_kl_fp32": curve[0]["kl_fp32_teacher_w0"], "step_s": step_s}


def first_read_kl(job: Job) -> float:
    """F1's KL to start at its first read step (the positive control's size)."""
    F = job.res["F"]
    return F["saved"][str(min(F["read_steps"]))]["kl_to_start"]


def stage_T0(job: Job) -> None:
    """The positive control: trained, then behaviour kept on maths rows and the FineWeb adversary of the student."""
    cfg = job.cfg
    off = Offsets(job.decomp()[0], job.alive())
    job.decomp("cpu")
    steps = job.res["B"]["t0_steps"]
    teacher, out = positive_control(job, off, first_read_kl(job), job.t_lr(), steps, "T0", (steps,), True)
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


STAGES: tuple[Stage, ...] = (("B", "B", stage_B), ("G1", "G", stage_G1), ("L", "L", stage_L),
                             ("A0", "A0", run.stage_A0), ("F1", "F", stage_F1), ("T0", "T0", stage_T0),
                             ("D1", "D", run.stage_D), ("T1", "T", run.stage_T))


def before_loading(a: argparse.Namespace, cfg: Any, layout: Layout, stages: list[Stage]) -> None:
    """Checks that need no model: the read rule and T0's step cap as filed, F1's inputs (L's results, a finished B),
    a mounted volume, and the decomposition fetched when a stage needs it and it is not complete (the space for it
    and the run's files checked first, at E40's alive count)."""
    pred, names = json.loads(PRED.read_text()), {s[0] for s in stages}
    rule = pred["rules"]["read_steps"]
    assert (tuple(rule["fixed"]), rule["kl_min"]) == (cfg.read_fixed, cfg.read_kl_min), "read rule: file vs config"
    assert pred["rules"]["T0"]["steps_max"] == cfg.t0_steps, "T0 step cap: file vs config"
    if "F1" in names:
        if "L" not in names and not layout.result("L").is_file():
            sys.exit(f"F1 needs stage L's results first: {layout.result('L')}")
        if "B" not in names and "B" not in run.done(layout):
            sys.exit("F1 needs a finished stage B (its read-step budget)")
    if a.volume != "none":
        assert os.path.ismount(a.volume), f"--volume {a.volume} is not a mount point"
    decomp = Path(a.decomp)
    if names & {"B", "G1", "T0", "D1", "T1"} and not qwen_files.complete(decomp):
        prior = pred["early"]["bench"]["alive_params_prior"]
        need, _ = disk_need(cfg, prior, pred["early"]["bench"]["decomp_bytes"])
        free_b = free_bytes(decomp.parent if decomp.parent.exists() else Path(a.out))
        log(f"[fetch] {decomp} incomplete: {free_b / 1e9:.1f} GB free for {need / 1e9:.1f} GB (decomposition + run)")
        if free_b < need:
            sys.exit(f"no room to fetch the decomposition and run: {free_b / 1e9:.1f} GB free, {need / 1e9:.1f} needed")
        qwen_files.fetch(decomp, log)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="B,G1,A0,F1,T0,D1,T1")
    ap.add_argument("--out", default="/workspace/e40a1")
    ap.add_argument("--decomp", default="/workspace/qwen3-1.7b-p-e9fcffd5-torch")
    ap.add_argument("--usd-per-h", type=float, default=4.5)
    ap.add_argument("--budget-left", type=float, default=None, help="dollars left for this run (B's projection)")
    ap.add_argument("--wandb-upload", action="store_true", help="upload results JSONs, the log and T1's last offsets to W&B (off)")
    ap.add_argument("--wandb-dir", default=str(Path.home() / "wandb"), help="W&B's run files, on the container disk")
    ap.add_argument("--rehearse", action="store_true", help="tiny sizes (A1_TINY), stops logged, own directory")
    ap.add_argument("--kill-after-save", action="store_true", help="rehearsal: end the job after T1's first save")
    ap.add_argument("--volume", default="/workspace", help="the quota'd network volume's mount point ('none': no volume)")
    ap.add_argument("--volume-gb", type=float, default=150.0, help="its quota: free = min(quota - du, df)")
    ap.add_argument("--may-delete", default="", help="a file B may delete when space is short (none by default)")
    a = ap.parse_args()
    set_volume(None if a.volume == "none" else Path(a.volume), None if a.volume == "none" else a.volume_gb)
    cfg, layout = (A1_TINY, REHEARSAL) if a.rehearse else (A1, LAYOUT)
    stages = chosen(a.stages, STAGES)
    start(layout, stages)
    before_loading(a, cfg, layout, stages)
    pred = json.loads(PRED.read_text())
    only_l = {s[0] for s in stages} <= {"L"}
    need = cfg.l_card_gb if only_l else cfg.min_card_gb
    assert torch.cuda.get_device_properties(0).total_memory > need * 1e9, f"needs a {need:.0f} GB card"
    if a.budget_left is None and not only_l and not a.rehearse:
        sys.exit("--budget-left (dollars) is needed for B's projection and the re-read guard")
    budget = a.budget_left if a.budget_left is not None else math.inf
    uploader = None
    if a.wandb_upload:
        # its messages to stdout: run.log is itself uploaded (immutable) at the end, so nothing may append to it then
        uploader = Uploader("e40a1-rehearse" if a.rehearse else "e40a1", cfg.upload_max_bytes, Path(a.wandb_dir), print)
    job = Job(a, cfg, layout, budget_usd=budget, guard_usd=pred["three_starts"]["guard_share"] * budget,
              rehearse=a.rehearse, uploader=uploader, kill_after_save=a.kill_after_save)
    job.may_delete = Path(a.may_delete) if a.may_delete else None
    run_stages(job, stages)


if __name__ == "__main__":
    main()
