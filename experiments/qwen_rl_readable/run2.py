"""P3.4 run 2: E69's GRPO loop on Qwen3-1.7B (experiments/qwen_rl) at peak 3e-6 after 10 warm-up steps (the user's
choice, decisions.md 2026-11-12), arms R, N and F on seeds 0 (E69's draw seed 69) and 70, one arm and seed per pod, to
step 400, with δW saved at every save step. Plan: reports/plans/2026-11-12_plan_qwen-rl-readable-run2.md; rules in
results/predictions_run2.json (refuses to start unless it is committed unchanged; on a pod, E120_PRED_SHA).

  --pod R_s0|N_s0|F_s0|R_s70|N_s70|F_s70   one pod: results/run2/<pod>.json and .log, large/run2/<pod>/
  --chosen                                  no model: the matched steps from the R files, into results/run2/chosen.json
  --size tiny --large DIR                   the rehearsal (E69's tiny config, 2 steps, a pause and resume at step 1)

At each save step (every 25 from 50; tiny: 1 and 2), after the update: a checkpoint (large/run2/<pod>/ckpt, so the same
command resumes), δW in fp32 (large/run2/<pod>/dW/step_<t>.safetensors, E69's layout and keys) with its size and sha256
in the results file under "saves", then the reads. R reads the output change every 5 steps and the GSM8K pass rate every
50 (for the stop rule only); N and F read the output change at their save steps only. Each read is run 1's
(`run.read`). R's file also carries "chosen" (`match.chosen`) and its log a CONFIRMED line when a size's step is fixed:
the runner downloads that step from R, N and F of the seed. F trains on W0's rollouts of R's problems and step seeds:
E69's (seed 0, steps 1-100, extracted into large/run2/F_s0/rollouts/F by the set-up) or sampled here at W0 just before
the step. Past step 114 the problem order starts again with new draws (run 1's `ReadJob.step_pids`).
Exit 0 done or ended by the drift stop ("drift"); 1 a refusal; 2 a check missed ("stopped"); 3 paused (tiny only).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import match
import torch

from commute import qwen_d3
from commute import qwen_rl as q
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.qwen_sft import adamw, save, set_weights, sha256, trainable
from commute.qwen_train import checkpoint
from commute.qwen_updates import ok

E69 = load_experiment("qwen_rl", "run")  # `update` (inside run 1's train_step) and `start_arm`
JOB = load_experiment("qwen_rl", "job")
RUN1 = load_experiment("qwen_rl_readable", "run")  # train_step, read, drift_stop and their checks, unchanged
HERE = Path(__file__).resolve().parent
RESULTS, PRED = HERE / "results", HERE / "results" / "predictions_run2.json"
LARGE = RESULTS / "large"
TAG, SEEDS, ARMS = "3e-6w10", {0: 69, 70: 70}, ("R", "N", "F")  # seed name -> E69's draw seed
GB, DW_BYTES, CKPT_BYTES = 10**9, 5_637_164_512, 17 * 10**9  # E69's dW file; run 1's checkpoint (~17 GB)
STRUCTURAL = ("loss and gradient norm finite", "first save: E69's size, sha256 recorded", "GSM8K read finite")
D = dict[str, Any]


@dataclass(frozen=True)
class Size:
    name: str  # E69's config size (job.SIZES): problems and samples per step, GSM8K problems read
    steps: int
    first_save: int
    save_every: int
    kl_every: int  # R: an output-change read every this many steps (N and F: at the save steps)
    gsm8k_every: int  # R: a GSM8K read every this many steps, for the stop rule
    pause_at: int | None = None  # exit 3 after this step's save and reads: the rehearsal of the resume
    margin_gb: float = 5.0  # disk left free after every save and checkpoint

    def __post_init__(self) -> None:
        assert self.name in JOB.SIZES and 1 <= self.first_save <= self.steps
        assert self.first_save % self.save_every == 0 and self.steps % self.save_every == 0
        assert self.save_every % self.kl_every == 0 and self.gsm8k_every % self.save_every == 0
        assert self.steps % self.gsm8k_every == 0 and set(self.gsm8k) <= set(self.saves)
        assert self.pause_at is None or self.pause_at in self.saves

    @property
    def saves(self) -> tuple[int, ...]:
        return tuple(range(self.first_save, self.steps + 1, self.save_every))

    @property
    def gsm8k(self) -> tuple[int, ...]:
        return tuple(range(self.gsm8k_every, self.steps + 1, self.gsm8k_every))


SIZES = {"full": Size("full", 400, 50, 25, 5, 50), "tiny": Size("tiny", 2, 1, 1, 1, 2, pause_at=1)}


@dataclass(frozen=True)
class Pod:
    arm: str
    seed: int
    size: Size
    kl_batch: int = 4  # run 1's `output_change`: 4 maths rows a forward
    reproduces: bool = False  # run 1's `train_step`: the step-20 reward change is logged, not a stop

    def __post_init__(self) -> None:
        assert self.arm in ARMS and self.seed in SEEDS

    @property
    def name(self) -> str:
        return f"{self.arm}_s{self.seed}"

    @property
    def kl_steps(self) -> set[int]:
        s = self.size
        return set(range(s.kl_every, s.steps + 1, s.kl_every)) if self.arm == "R" else set(s.saves)

    @property
    def gsm8k_steps(self) -> set[int]:
        return set(self.size.gsm8k) if self.arm == "R" else set()


def results_dir(size: str) -> Path:
    return RESULTS / ("run2" if size == "full" else "run2_tiny")


class Job(JOB.Job):
    """E69's job with this pod's own files; E69's H and W0's GSM8K answers are read where E69 keeps them, at every size."""

    def __init__(self, pod: Pod, large: Path) -> None:
        self.pod, self.dir = pod, results_dir(pod.size.name)
        self.dir.mkdir(parents=True, exist_ok=True)
        if LARGE.is_symlink() and not LARGE.exists():  # a link synced to a pod arrives dangling
            LARGE.unlink()
        super().__init__(JOB.config(pod.size.name, TAG, draw_seed=SEEDS[pod.seed]), pod.name)
        self.large, self.shared_large = large / pod.name, JOB.LARGE
        self.large.mkdir(parents=True, exist_ok=True)
        self.epoch = len(self.train) // self.cfg.problems

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (self.dir / f"{self.stage}.log").open("a") as f:
            f.write(line + "\n")

    def where(self, name: str) -> Path:
        return self.dir / f"{name}.json"

    def step_pids(self, t: int) -> list[int]:
        return super().step_pids((t - 1) % self.epoch + 1)

    def rollout_dir(self, arm: str) -> Path:
        return self.large / "rollouts" / arm

    def check(self, out: D, name: str, holds: bool, detail: D, band: bool = False) -> None:
        """A miss stops the job; at the tiny size only a structural check stops (a value's band means nothing there)."""
        super().check(out, name, holds, detail, band=band or not name.startswith(STRUCTURAL))


# ----------------------------------------------------------------------------- saves, checks, the chosen steps


def save_state(job: Job, out: D, t: int, params: D) -> None:
    """δW at step t in fp32, E69's layout; its size and sha256 into "saves" (the runner's manifest)."""
    path, t0 = job.large / "dW" / f"step_{t}.safetensors", time.time()
    save(path, q.delta_fp32(params, job.W0), torch.float32, droppable=False, log=job.log)
    rec = {"file": str(path.relative_to(job.large.parent)), "bytes": path.stat().st_size, "sha256": sha256(path),
           "save_s": time.time() - t0}
    out["saves"][str(t)] = rec
    job.put(job.stage, out)
    job.log(f"[{job.stage}] SAVED step {t} {json.dumps(rec)}")
    if len(out["saves"]) == 1:
        job.check(out, "first save: E69's size, sha256 recorded", rec["bytes"] == DW_BYTES and len(rec["sha256"]) == 64, rec)


def room(job: Job, t: int) -> None:
    """Refuse before training unless the disk holds what this size still writes: every save to come, one more checkpoint
    (the .tmp beside the one on disk, which is already counted as used) and the margin. Full size at step 0: ~107 GB;
    tiny: ~33 GB."""
    s = job.pod.size
    need = sum(x > t for x in s.saves) * DW_BYTES + CKPT_BYTES + s.margin_gb * GB
    free = shutil.disk_usage(job.large).free
    job.log(f"[{job.stage}] disk: {free / GB:.0f} GB free, {need / GB:.0f} GB needed")
    if free < need:
        sys.exit(f"{free / GB:.0f} GB free under {job.large}, {need / GB:.0f} GB needed: give the pod a larger disk")


def bands(job: Job, out: D, t: int, E: D) -> None:
    """The filed bands at step t: |δW| at step 20 on every arm (a rate not applied), R seed 0's multiple at 50 and 150."""
    filed = {"20": ("dw_norm", E["dw_norm_step20"][job.pod.arm])}
    if job.pod.name == "R_s0":
        filed |= {s: ("multiple", b) for s, b in E["multiple_R_s0"].items()}
    if str(t) not in filed:
        return
    key, band = filed[str(t)]
    got = out["steps"][-1]["dw_norm"] if key == "dw_norm" else out["reads"].get(str(t), {}).get("multiple")
    job.check(out, f"step {t}: {key} in its filed band", q.in_band(got, band), {key: got, "band": band})


def note_chosen(job: Job, out: D, P: D) -> None:
    """R: the matched steps so far into its file, and a CONFIRMED line for each size newly fixed."""
    if job.pod.arm != "R":
        return
    before = {k for k, v in out.get("chosen", {}).get("targets", {}).items() if v["status"] == "confirmed"}
    out["chosen"] = c = match.chosen(out, list(job.pod.size.saves), P["match"]["targets"])
    for k, v in c["targets"].items():
        if v["status"] == "confirmed" and k not in before:
            job.log(f"CONFIRMED {k}x at step {v['step']} (crossing {v['crossing']}, multiple {v['multiple']:.3f}): "
                    f"download step {v['step']} of R, N and F seed {job.pod.seed}")


def early(P: D, pod: Pod) -> D:
    """Run 1's early checks, with E69's step-1 rollouts of this seed."""
    return P["early"] | {"e69_rollout_sha_1": P["early"]["rollout_sha_1"][str(pod.seed)]}


# ----------------------------------------------------------------------------- the arm


def f_rollouts(job: Job, t: int) -> None:
    """F's batch of step t: W0's rollouts of R's problems and step seed; sampled here at W0 if the set-up did not bring
    E69's."""
    p = job.rollout_dir("F") / f"step_{t}.jsonl"
    if p.is_file():
        return
    t0 = time.time()
    with set_weights(job.model, job.W0):
        JOB.save_rollouts(p, job.sampled(job.step_pids(t), job.cfg.samples, t))
    job.log(f"[{job.stage}] W0's rollouts of step {t} sampled here, {time.time() - t0:.0f} s")


def after_update(job: Job, out: D, t: int, params: D, rows: torch.Tensor, w0: q.Records, P: D) -> bool:
    """Step t's save and reads (each unless already recorded: a resume at t), its bands, the stop rule and the chosen
    steps. True: the drift stop ended the arm."""
    pod = job.pod
    if t in pod.size.saves and str(t) not in out["saves"]:
        save_state(job, out, t, params)
    if t in pod.kl_steps and str(t) not in out["reads"]:
        r = RUN1.read(job, out, t, rows, w0, P["refs"], t in pod.gsm8k_steps)
        if t in pod.gsm8k_steps:
            job.check(out, "GSM8K read finite", ok(r.get("pass")) and ok(r.get("fits")), {"step": t, "pass": r.get("pass")})
    ended = t in pod.gsm8k_steps and RUN1.drift_stop(job, out, t, P["refs"])
    note_chosen(job, out, P)  # before the bands, so a band stop never hides a confirmation
    job.put(job.stage, out)
    bands(job, out, t, P["early"])
    return ended


def step(job: Job, out: D, params: D, opt: Any, t: int, rows: torch.Tensor, w0: q.Records, P: D) -> bool:
    if job.pod.arm == "F":
        f_rollouts(job, t)
    RUN1.train_step(job, out, params, opt, t, early(P, job.pod))
    if t in job.pod.size.saves:
        t0 = time.time()
        checkpoint(job.large / "ckpt" / f"{job.pod.arm}.pt", t, params, job.W0, opt, None, set(), set())
        job.log(f"[{job.stage}] checkpoint at step {t}, {time.time() - t0:.0f} s")
    ended = after_update(job, out, t, params, rows, w0, P)
    if t == job.pod.size.pause_at and not ended:
        job.log(f"[{job.stage}] paused at step {t}: run the same command")
        sys.exit(3)
    return ended


def run_pod(job: Job, P: D) -> None:
    pod, out = job.pod, job.begin()
    if out is None:
        return
    w0_file = job.shared_large / "acc" / "W0_gsm8k.jsonl"
    if missing := [str(p) for p in (job.shared_large / "rollouts" / "H.jsonl", w0_file) if not p.is_file()]:
        sys.exit(f"missing {missing}: fetch them from W&B (the plan's set-up) before the arm")
    if pod.name == "F_s0" and pod.size.name == "full" and not all(
        (job.rollout_dir("F") / f"step_{t}.jsonl").is_file() for t in range(1, 101)
    ):
        sys.exit(f"E69's F rollouts step_1..100.jsonl are not all in {job.rollout_dir('F')}: extract F.tar there "
                 "(the plan's set-up)")
    out |= {"pod": pod.name, "arm": pod.arm, "seed": pod.seed, "draw_seed": job.cfg.draw_seed, "tag": TAG,
            "size": pod.size.name, "device": torch.cuda.get_device_name()}
    out.setdefault("saves", {})
    job.fp32()
    rows, w0 = qwen_d3.maths_rows().tokens, JOB.load_rollouts(w0_file)
    assert rows.shape == (qwen_d3.MATHS_ROWS, qwen_d3.ROW_LEN), rows.shape
    with trainable(job.model, job.W0) as params:
        opt = adamw(params, job.cfg.lr)
        t = E69.start_arm(job, pod.arm, out, params, opt)
        out["saves"] = {s: v for s, v in out["saves"].items() if int(s) <= t}
        room(job, t)
        ended = t > 0 and after_update(job, out, t, params, rows, w0, P)  # a resume finishes its checkpoint step first
        while not ended and t < pod.size.steps:
            t += 1
            ended = step(job, out, params, opt, t, rows, w0, P)
    out |= {"done": True, "last": out["steps"][-1]["step"], "peaks": job.peaks()}
    note_chosen(job, out, P)
    job.put(job.stage, out)
    (job.large / "ckpt" / f"{pod.arm}.pt").unlink(missing_ok=True)
    job.log(f"[{job.stage}] done at step {out['last']}; {json.dumps(out['peaks'])}")


def chosen(P: D, size: str) -> D:
    """Per seed: the matched steps (`match.chosen` on R's file) and, per arm, the file and sha256 of each kept step as
    saved so far. Written to <results>/chosen.json; read2.py reads the states listed there."""
    d, s = results_dir(size), SIZES[size]
    out: D = {}
    for seed in SEEDS:
        arms = {a: JOB.read_json(d / f"{a}_s{seed}.json") for a in ARMS}
        c = match.chosen(arms["R"], list(s.saves), P["match"]["targets"])
        c["files"] = {f"{a}_s{seed}": {str(t): arms[a].get("saves", {}).get(str(t), "not saved yet") for t in c["keep"]}
                      for a in ARMS}
        out[str(seed)] = c
    JOB.write_text(d / "chosen.json", json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pod", choices=[f"{a}_s{s}" for s in SEEDS for a in ARMS])
    g.add_argument("--chosen", action="store_true")
    ap.add_argument("--size", default="full", choices=tuple(SIZES))
    ap.add_argument("--large", type=Path, default=None, help="states root (default results/large/run2; the rehearsal: /ephemeral)")
    a = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"{PRED} is missing: predictions are filed before any run")
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    if a.chosen:
        print(json.dumps(chosen(P, a.size), indent=1), flush=True)
        return
    arm, seed = a.pod.split("_s")
    job = Job(Pod(arm, int(seed), SIZES[a.size]), a.large or LARGE / "run2")
    run_pod(job, P)
    hard_exit(0)


if __name__ == "__main__":
    main()
