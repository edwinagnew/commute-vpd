"""P3.4 run 1 (E163): E69's GRPO loop on Qwen3-1.7B (experiments/qwen_rl, recipe 1e-6w10, draw seed 69, "seed 0")
reproduced for R, N and F to step 100, R continued at the same rate to step 250, and R at two higher peak rates: how far
each grows the output change while the held-out pass rate holds. Plan: reports/plans/2026-11-11_plan_qwen-rl-readable-run1.md;
rules in results/predictions.json (refuses to start unless it is committed unchanged; on a pod, E120_PRED_SHA).

  --pod control|N|F|r3e-6|r1e-5   one pod's arm (PODS below), writing results/<pod>.json and results/<pod>.log
  --score                         no model: the selection rule on the three R pods' files, into results/selection.json

At each read step: |δW|; the output change, KL(W_t ‖ W0) per token over every position of the 128 maths rows at length 256
(E93's clean.W0.K), and its multiple of the decomposition's masked error; at a GSM8K read step also the greedy GSM8K pass
rate (strict_dropped, 1319 problems), the share of answers that fit the window, and the paired gain over W0. R pods read
the output change alone every 5 steps besides. Each read is written and logged as it finishes. Nothing is uploaded and no
δW is saved; a checkpoint at each of the pod's checkpoint steps (results/large/<pod>/ckpt), written right after that
step's update and before its read, lets the same command resume after a crash (a crash in a read replays only the read).
The control logs GATE_LINE once its step-25 read is logged and its checkpoint written: pods 2-5 start then.
E69's job and run modules are used unchanged; its shared large files must be on the pod (rollouts/H.jsonl,
acc/W0_gsm8k.jsonl, and for F rollouts/F/, else F samples them first). Past step 114 (7345 problems / 64) the problem
order starts again at step 1's, with new draws (the step seed counts on).
Exit 0 done, or ended by the drift stop ("drift" in the JSON); 1 a refusal; 2 a check missed ("stopped" in the JSON).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pick
import torch

from commute import qwen_d3
from commute import qwen_rl as q
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.qwen_eval import score as gsm_score
from commute.qwen_eval import summary
from commute.qwen_sft import adamw, set_weights, trainable
from commute.qwen_train import checkpoint
from commute.qwen_updates import ok
from commute.vpd import kl_per_position

E69 = load_experiment("qwen_rl", "run")  # its `update` and `start_arm`; it imports job.py, the module below
JOB = load_experiment("qwen_rl", "job")
HERE = Path(__file__).resolve().parent
RESULTS, PRED = HERE / "results", HERE / "results" / "predictions.json"
LARGE = RESULTS / "large"
REF_LR = q.RATES[q.DEFAULT_TAG]  # the rate E69's first-step KL band is filed for
D = dict[str, Any]


GATE_LINE = "GATE OPEN: control step 25 read logged with finite kl and pass, checkpoint written; start pods 2-5"


@dataclass(frozen=True)
class Pod:
    arm: str  # R, N or F (`qwen_rl.arm_weights`)
    tag: str  # the rate (`qwen_rl.RATES`): peak, and warm-up steps after the w
    steps: int
    reads: tuple[int, ...]  # GSM8K read steps (with the output change); the last is `steps`
    ckpts: tuple[int, ...]  # checkpoint steps, written before that step's read
    reproduces: bool  # E69's recipe: its reward-change check at step 20 stops (else it is logged)
    kl_every: int = 0  # R pods: an output-change-only read every this many steps (0: none)
    gate: int | None = None  # the control: the step whose read and checkpoint open the stage gate
    kl_batch: int = 4  # maths rows per output-change forward: 4 x 256 x 151936 fp32 logits, 0.6 GB a copy

    def __post_init__(self) -> None:
        assert self.arm in ("R", "N", "F") and self.tag in q.RATES
        assert list(self.reads) == sorted(set(self.reads)) and self.reads[-1] == self.steps
        assert list(self.ckpts) == sorted(set(self.ckpts)) and set(self.ckpts) <= set(range(1, self.steps + 1))
        assert self.reproduces == (self.tag == "1e-6w10") and (self.arm == "R" or self.steps == 100)
        assert (self.kl_every > 0) == (self.arm == "R")
        assert self.gate is None or (self.gate in self.reads and self.gate in self.ckpts)

    @property
    def kl_steps(self) -> set[int]:
        """Output-change-only read steps: every `kl_every`th step that has no GSM8K read."""
        return set(range(self.kl_every, self.steps + 1, self.kl_every)) - set(self.reads) if self.kl_every else set()


R25 = (25, 50, 100, 150, 200, 250)
PODS = {"control": Pod("R", "1e-6w10", 250, R25, R25, True, kl_every=5, gate=25),
        "N": Pod("N", "1e-6w10", 100, (100,), (20, 50, 100), True),
        "F": Pod("F", "1e-6w10", 100, (100,), (20, 50, 100), True),
        "r3e-6": Pod("R", "3e-6w10", 200, (10, *R25[:5]), (10, *R25[:5]), False, kl_every=5),
        "r1e-5": Pod("R", "1e-5w10", 100, (10, 25, 50, 100), (10, 25, 50, 100), False, kl_every=5)}


class ReadJob(JOB.Job):
    """E69's job with this run's own files (results/<pod>.json and .log, results/large/<pod>/); E69's shared files (H, W0's
    accuracy, F's rollouts) are read where E69 keeps them."""

    def __init__(self, name: str) -> None:
        self.pod = PODS[name]
        if LARGE.is_symlink() and not LARGE.exists():  # a link synced to a pod arrives dangling
            LARGE.unlink()
        RESULTS.mkdir(parents=True, exist_ok=True)
        super().__init__(JOB.config("full", self.pod.tag), name)
        self.res, self.large = RESULTS, LARGE / name
        self.large.mkdir(parents=True, exist_ok=True)
        self.epoch = len(self.train) // self.cfg.problems

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / f"{self.stage}.log").open("a") as f:
            f.write(line + "\n")

    def where(self, name: str) -> Path:
        return RESULTS / f"{name}.json"

    def step_pids(self, t: int) -> list[int]:
        return super().step_pids((t - 1) % self.epoch + 1)


# ----------------------------------------------------------------------------- the reads


@torch.no_grad()
def output_change(job: ReadJob, rows: torch.Tensor) -> float:
    """KL(W_t ‖ W0) per position, the mean over every position of `rows`: E93's clean.W0.K (`qwen_d3.clean` reads
    `kl_per_position(FT, W0)`), batched since every row is ROW_LEN long with no padding."""
    total = 0.0
    for chunk in rows.split(job.pod.kl_batch):
        t = chunk.to(JOB.DEV)
        after = job.model(t, use_cache=False).logits
        with set_weights(job.model, job.W0):
            before = job.model(t, use_cache=False).logits
        total += float(kl_per_position(after, before).double().sum())
        del after, before
    return total / rows.numel()


def pass_rate(job: ReadJob, label: str, w0: q.Records) -> D:
    """Greedy GSM8K test accuracy as E69's `Job.accuracy` reads it (GSM8K only), and the paired gain over W0's answers."""
    sink = job.large / "acc" / f"{label}_gsm8k.jsonl"
    sink.parent.mkdir(parents=True, exist_ok=True)
    recs = gsm_score(job.tok, job.model, job.sets["test"][: job.cfg.n_gsm8k], math.inf, sink, f"{label} gsm8k", job.cfg.gen_batch)
    s = summary(recs)
    return {"pass": s["strict_dropped"], "fits": s["fits_share"], "k_fits": s["k_fits"],
            "truncation_share": s["truncation_share"], "gain": q.gain(w0, recs)}


def read(job: ReadJob, out: D, t: int, rows: torch.Tensor, w0: q.Records, refs: D, gsm8k: bool) -> D:
    """The output change at step t and, if `gsm8k`, the pass rate; written and logged as soon as it is read."""
    t0 = time.time()
    kl = output_change(job, rows)
    r = {"dw_norm": out["steps"][-1]["dw_norm"], "kl": kl, "multiple": kl / refs["masked_error"], "kl_s": time.time() - t0}
    if gsm8k:
        t1 = time.time()
        r |= pass_rate(job, f"{job.stage}_{t}", w0) | {"acc_s": time.time() - t1}
    out["reads"][str(t)] = r
    job.put(job.stage, out)
    job.log(f"[{job.stage}] step {t} {'read' if gsm8k else 'output-change read'} {json.dumps(r, default=float)}")
    return r


# ----------------------------------------------------------------------------- checks


def first_checks(job: ReadJob, out: D, recs: q.Records, E: D) -> None:
    """Step 1: E69's reward-rate band; whether the rollouts equal E69's token for token is logged, not a stop (a
    different card or driver may draw differently; the bands below catch a broken loop)."""
    sha, rate = q.rollout_hash(recs), q.rollout_summary(recs)["reward_rate"]
    out |= {"rollout_sha_1": sha, "same_rollouts_as_e69": sha == E["e69_rollout_sha_1"]}
    job.log(f"[{job.stage}] step 1's rollouts identical to E69's R: {out['same_rollouts_as_e69']}")
    job.check(out, "step 1's reward rate", q.in_band(rate, E["reward_step1"]), {"rate": rate, "band": E["reward_step1"]})


def live_checks(job: ReadJob, out: D, rec: D, E: D) -> None:
    """E69's first live step: Adam's first step is rate x sign, and KL(W0 ‖ W_1) on H is in E69's band for a step at
    1e-6 scaled by (rate / 1e-6) squared (E69's `live_checks` scales by its own lr, which is 1e-6 only at its tags)."""
    rate = rec["rate"]
    ratio = rec["dw_norm"] / (rate * math.sqrt(job.numel))
    job.check(out, "first live step: |dW| / (rate sqrt(numel))", q.in_band(ratio, E["dw_ratio"]),
              {"step": rec["step"], "ratio": ratio, "band": E["dw_ratio"]})
    out["first_live_read"] = h = job.read_state()
    band = [x * (rate / REF_LR) ** 2 for x in E["kl_first"]]
    job.check(out, "first live step: KL(W0 || W_1) on H", q.in_band(h["kl"], band), {"step": rec["step"], "kl": h["kl"], "band": band})


def reproduce(job: ReadJob, out: D, t: int, P: D) -> None:
    """The filed bands at step t (E69's values, or a rate-scaled |δW|; a stop on a miss), and the logged expectations
    under "expect" (not a stop)."""
    ref = P["reproduce"].get(job.stage, {}).get(str(t), {})
    got = {"dw_norm": out["steps"][-1]["dw_norm"], **{k: out["reads"].get(str(t), {}).get(k) for k in ("pass", "kl")}}
    for key in ("dw_norm", "pass"):
        if key in ref:
            job.check(out, f"step {t}: {key} in its filed band", q.in_band(got[key], ref[key]), {key: got[key], "band": ref[key]})
    for key, band in ref.get("expect", {}).items():
        holds = q.in_band(got[key], band)
        out.setdefault("expected", {})[f"step {t} {key}"] = {"holds": holds, key: got[key], "band": band}
        job.log(f"[expected] step {t} {key} in {band}: {'ok' if holds else 'MISSED (not a stop: tell the lead)'} {got[key]}")


def drift_stop(job: ReadJob, out: D, t: int, refs: D) -> bool:
    """R arms: the filed drift stop at a GSM8K read; the arm ends there and its last good read is the GSM8K read before."""
    steps = sorted(s for s in pick.gsm8k_reads(out) if s < t)
    prev_kl = out["reads"][str(steps[-1])]["kl"] if steps else 0.0
    if job.pod.arm != "R" or not pick.drift(out["reads"][str(t)], prev_kl, refs["w0_pass"], pick.fits_bar(refs)):
        return False
    out["drift"] = {"step": t, "last_good": steps[-1] if steps else None}
    job.log(f"DRIFT STOP at step {t}: pass rate below W0's ({refs['w0_pass']}) or fewer answers fit than "
            f"{pick.fits_bar(refs)}, while the output change grows; {json.dumps(out['drift'])}")
    return True


def gate(job: ReadJob, out: D, t: int) -> None:
    """The control's stage gate: its read at t has a finite kl and pass rate, and its checkpoint is on disk."""
    r, ckpt = out["reads"].get(str(t), {}), job.large / "ckpt" / f"{job.pod.arm}.pt"
    job.check(out, f"step {t}: gate read finite and checkpoint written", ok(r.get("kl")) and ok(r.get("pass")) and ckpt.is_file(),
              {"kl": r.get("kl"), "pass": r.get("pass"), "ckpt": ckpt.is_file()})
    job.log(GATE_LINE)


# ----------------------------------------------------------------------------- the arm


def after_update(job: ReadJob, out: D, t: int, rows: torch.Tensor, w0: q.Records, P: D) -> bool:
    """Step t's read (unless already recorded: a resume at t), its filed bands, the drift stop and the gate. True: the
    drift stop ended the arm."""
    pod = job.pod
    if (t in pod.reads or t in pod.kl_steps) and str(t) not in out["reads"]:
        read(job, out, t, rows, w0, P["refs"], t in pod.reads)
    reproduce(job, out, t, P)
    ended = t in pod.reads and drift_stop(job, out, t, P["refs"])
    job.put(job.stage, out)
    if t == pod.gate and not ended:
        gate(job, out, t)
    return ended


def step(job: ReadJob, out: D, params: D, opt: Any, t: int, rows: torch.Tensor, w0: q.Records, P: D) -> bool:
    """One step, its checks, its checkpoint and its read. True: the drift stop ended the arm."""
    train_step(job, out, params, opt, t, P["early"])
    if t in job.pod.ckpts:
        checkpoint(job.large / "ckpt" / f"{job.pod.arm}.pt", t, params, job.W0, opt, None, set(), set())
        job.log(f"[{job.stage}] checkpoint at step {t}")
    return after_update(job, out, t, rows, w0, P)


def train_step(job: ReadJob, out: D, params: D, opt: Any, t: int, E: D) -> None:
    """Step t's rollouts and update, with E69's step-1, first-live and reward-change checks."""
    pod, cfg, t0 = job.pod, job.cfg, time.time()
    if pod.arm == "F":
        recs = JOB.load_rollouts(job.rollout_dir("F") / f"step_{t}.jsonl")
    else:
        recs = job.sampled(job.step_pids(t), cfg.samples, t)
    if t == 1:
        first_checks(job, out, recs, E)
    rec = {"step": t, "sample_s": time.time() - t0, **q.rollout_summary(recs)}
    rec |= E69.update(job, params, opt, recs, q.arm_weights(pod.arm, recs, cfg.draw_seed, t), cfg.rate(t))
    first_live = rec["live"] and not any(s["live"] for s in out["steps"])
    out["steps"].append(rec)
    job.put(job.stage, out)
    job.log(f"[{job.stage}] {json.dumps(rec)}")
    job.check(out, "loss and gradient norm finite", ok(rec["loss"]) and ok(rec["grad_norm"]), {"step": t})
    if first_live:
        live_checks(job, out, rec, E)
    if t == cfg.drop_at and pod.arm != "F":
        change = q.reward_change([s["reward_rate"] for s in out["steps"]], t)
        if pod.reproduces:
            job.check(out, f"reward change by step {t}", q.above(change, E["reward_change_above"]), {"change": change})
        else:
            job.log(f"[{job.stage}] reward change by step {t}: {change} (logged only at this rate)")
    job.put(job.stage, out)


def f_data(job: ReadJob, steps: int) -> None:
    """F's batches, W0's rollouts of each step: E69's (its rollouts/F/, from F.tar), or sampled here first if absent."""
    for t in range(1, steps + 1):
        p = job.rollout_dir("F") / f"step_{t}.jsonl"
        if not p.is_file():
            JOB.save_rollouts(p, job.sampled(job.step_pids(t), job.cfg.samples, t))
            job.log(f"[F] W0's rollouts of step {t} sampled here")


def run_pod(job: ReadJob, P: D) -> None:
    pod, cfg = job.pod, job.cfg
    out = job.begin()
    if out is None:
        return
    w0_file = job.shared_large / "acc" / "W0_gsm8k.jsonl"
    if missing := [str(p) for p in (job.shared_large / "rollouts" / "H.jsonl", w0_file) if not p.is_file()]:
        sys.exit(f"missing {missing}: fetch them from W&B (the plan's commands) before the arm")
    out |= {"pod": job.stage, "arm": pod.arm, "tag": pod.tag, "device": torch.cuda.get_device_name()}
    job.fp32()
    if pod.arm == "F":
        f_data(job, pod.steps)
    rows, w0 = qwen_d3.maths_rows().tokens, JOB.load_rollouts(w0_file)
    assert rows.shape == (qwen_d3.MATHS_ROWS, qwen_d3.ROW_LEN), rows.shape
    with trainable(job.model, job.W0) as params:
        opt = adamw(params, cfg.lr)
        t = E69.start_arm(job, pod.arm, out, params, opt)
        ended = t > 0 and after_update(job, out, t, rows, w0, P)  # a resume finishes its checkpoint step's read first
        while not ended and t < pod.steps:
            t += 1
            ended = step(job, out, params, opt, t, rows, w0, P)
    out |= {"done": True, "last": out["steps"][-1]["step"], "peaks": job.peaks()}
    job.put(job.stage, out)
    (job.large / "ckpt" / f"{pod.arm}.pt").unlink(missing_ok=True)
    job.log(f"[{job.stage}] done at step {out['last']}; {json.dumps(out['peaks'])}")


def score(P: D) -> None:
    arms = {n: JOB.read_json(RESULTS / f"{n}.json") for n in P["selection"]["recipes"]}
    out = pick.select(P["selection"], P["refs"]["w0_pass"], pick.fits_bar(P["refs"]), arms)
    JOB.write_text(RESULTS / "selection.json", json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pod", choices=tuple(PODS))
    g.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"{PRED} is missing: predictions are filed before any run")
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    if a.score:
        score(P)
        return
    job = ReadJob(a.pod)
    run_pod(job, P)
    hard_exit(0)


if __name__ == "__main__":
    main()
