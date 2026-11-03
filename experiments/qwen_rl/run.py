"""E69 (qwen_rl): the GRPO loop on Qwen3-1.7B: RL, its shuffled-reward null and own-samples SFT, with δW saved. Plan:
reports/plans/2026-11-03_plan_e69-grpo-loop.md; rules in results/predictions.json (the run refuses to start without it);
order and commands in RUN_BRIEF.md.

Stages (--stage, each refuses until the one before has finished; --size tiny writes under results/tiny and large/tiny):
  A0     step 0: greedy accuracy; the held-out samples H from W0; ΔR(0) and KL(0); P0
  S      the sampler against `generate` on one batch; two RL steps, nothing saved; the cost rule sets N's steps
  R N F  100 steps each. δW (fp32) and readouts at the saved steps, accuracy at the last; every number in results/<arm>.json
  score  no model: P0-P7 from the results files
Exit 0 done; 1 a refusal; 2 an early check missed (results/<stage>.json "stopped", and that stage refuses from then on);
3 paused at a checkpoint (tiny only, to rehearse the resume): the same command goes on.
No autocast anywhere: every forward here is fp32 with TF32 off.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tarfile
import time
from typing import Any

import torch
from job import (
    F32,
    GB,
    PRED,
    RESULTS,
    SIZES,
    Config,
    Job,
    load_rollouts,
    read_json,
    save_rollouts,
    write_text,
)

from commute.qwen_rl import (
    SAMPLING_T1,
    Records,
    above,
    agreement,
    arm_weights,
    cost_rule,
    delta_fp32,
    gain,
    generate_rollouts,
    in_band,
    p0,
    pool,
    projected_usd,
    read_h,
    reward_change,
    rl_weights,
    rollout_hash,
    rollout_summary,
    rows_for,
    score_all,
    sft_weights,
    step_seed,
    weighted_update,
)
from commute.qwen_sft import Uploader, adamw, save, trainable
from commute.qwen_train import checkpoint, norms, resume
from commute.qwen_updates import ok

# ----------------------------------------------------------------------------- A0


def stage_A0(job: Job) -> None:
    cfg, out = job.cfg, job.begin()
    if out is None:
        return
    job.fp32()
    up = job.uploader()
    try:
        t0 = time.time()
        acc = job.accuracy("W0", up)
        out |= {"accuracy": {k: v["summary"] for k, v in acc.items()}, "accuracy_s": time.time() - t0}
        job.put("A0", out)
        t0 = time.time()
        recs = job.sampled(job.held, cfg.held_samples, 0)
        h = job.large / "rollouts" / "H.jsonl"
        save_rollouts(h, recs)
        job.publish(up, h)
        out |= {"held": rollout_summary(recs), "held_sha": rollout_hash(recs), "held_s": time.time() - t0, "pool": len(pool(recs))}
        job.log(f"[A0] H {json.dumps(out['held'])}; pool {out['pool']} problems; {out['held_s']:.0f} s")
        t0 = time.time()
        out |= {"read": job.read_state(), "read_s": time.time() - t0, "peaks": job.peaks()}
        job.log(f"[A0] {json.dumps(out['read'])}; {json.dumps(out['peaks'])}")
        out["P0"] = p0(job.pred["P0"], out["accuracy"], out["read"], out["pool"])
        job.check(out, "P0", out["P0"]["holds"], out["P0"], band=True)
        out["done"] = True
        job.put("A0", out)
    finally:
        up.finish()


# ----------------------------------------------------------------------------- one update


def update(job: Job, params: dict[str, Any], opt: Any, recs: Records, w: list[float]) -> dict[str, Any]:
    t0 = time.time()
    rec = weighted_update(job.model, params, opt, job.seqs(recs), w, job.cfg.divisor, job.cfg.train_tokens)
    dw = math.sqrt(sum(v * v for v in norms(params, job.W0).values()))
    return rec | {"dw_norm": dw, "grad_rms": rec["grad_norm"] / math.sqrt(job.numel), "train_s": time.time() - t0,
                  "live_tokens": sum(len(r["ids"]) for r, x in zip(recs, w, strict=True) if x)}


# ----------------------------------------------------------------------------- S


def compare_sampler(job: Job, out: dict[str, Any]) -> None:
    """One step's batch by the sampler and by `generate` at W0: length and reward within the filed SEs."""
    cfg, pids = job.cfg, job.step_pids(1)
    t0 = time.time()
    a = job.sampled(pids, cfg.samples, 1)
    t1 = time.time()
    torch.manual_seed(step_seed(cfg.seed, 1))
    total = torch.cuda.get_device_properties(0).total_memory
    rows = rows_for(total, cfg.generate_reserved_gb * GB, cfg.generate_tokens_per_row, cfg.max_rows)
    b = job.graded(pids, generate_rollouts(job.model, [job.prompt_ids(p) for p in pids], cfg.samples, cfg.window, rows,
                                           job.tok.pad_token_id, **SAMPLING_T1), cfg.samples)
    mult = job.pred["early"]["sampler_se_mult"]
    out["sampler"] = {"sampler_s": t1 - t0, "generate_s": time.time() - t1, "generate_rows": rows,
                      "length": agreement(a, b, lambda r: len(r["ids"]), mult),
                      "reward": agreement(a, b, lambda r: r["reward"], mult),
                      "peaks": job.peaks()}
    job.log(f"[S] {json.dumps(out['sampler'], default=float)}")
    job.check(out, "sampler agrees with generate: length", out["sampler"]["length"]["holds"], out["sampler"]["length"], band=True)
    job.check(out, "sampler agrees with generate: reward", out["sampler"]["reward"]["holds"], out["sampler"]["reward"], band=True)


def stage_S(job: Job, usd_per_h: float) -> None:
    """Reruns whole; the pod's run replaces the rehearsal's file (RUN_BRIEF.md)."""
    cfg, A0 = job.cfg, job.get("A0")
    if usd_per_h and not A0.get("done"):  # the cost rule reads A0's accuracy time; the rehearsal's slice (rate 0) does not
        sys.exit("stage A0 must finish first")
    if job.get("S").get("stopped"):
        sys.exit("S stopped: the user decides; move its file to run it again")
    out: dict[str, Any] = {"device": torch.cuda.get_device_name()}
    job.fp32()
    compare_sampler(job, out)
    job.put("S", out)
    torch.cuda.reset_peak_memory_stats()
    sample_s, train_s, tokens, f_tokens, first = [], [], 0, 0, None
    with trainable(job.model, job.W0) as params:
        opt = adamw(params, cfg.lr)
        for t in (1, 2):
            t0 = time.time()
            recs = job.sampled(job.step_pids(t), cfg.samples, t)
            sample_s.append(time.time() - t0)
            first = first or recs
            rec = update(job, params, opt, recs, rl_weights(recs))
            train_s.append(rec["train_s"])
            tokens += rec["live_tokens"]
            f_tokens += sum(len(r["ids"]) for r, x in zip(recs, sft_weights(recs), strict=True) if x)
            job.log(f"[S] step {t}: {sample_s[-1]:.1f} s sampling, {json.dumps(rec)}, {json.dumps(rollout_summary(recs))}")
        assert first is not None
        t0 = time.time()
        read_h(job.model, job.W0, job.seqs(first), cfg.read_tokens)
        read_s = (time.time() - t0) * cfg.held * cfg.held_samples / (cfg.problems * cfg.samples)  # H's size
    t = {"sample": sum(sample_s) / 2, "train": sum(train_s) / 2, "f_train": sum(train_s) / 2 * f_tokens / max(1, tokens),
         "read": read_s, "acc": A0.get("accuracy_s", 0.0)}
    reads = {n: sum(s <= n for s in cfg.saves) for n in (cfg.steps, cfg.n_cut)}
    usd = {n: projected_usd(t, cfg.steps, n, reads, cfg.setup_h, usd_per_h) for n in (cfg.steps, cfg.n_cut)}
    rule = cost_rule(usd[cfg.steps], usd[cfg.n_cut], job.pred["cost"]["usd_max"], cfg.steps, cfg.n_cut)
    out |= {"seconds": t, "usd_per_h": usd_per_h, "projected_usd": usd, "n_steps": rule["n_steps"], "peaks": job.peaks()}
    job.log(f"[S] {json.dumps(out, default=float)}")
    job.check(out, "projected cost within the limit", not rule["stop"], {"usd": usd})
    out["done"] = True
    job.put("S", out)


# ----------------------------------------------------------------------------- arms


def start_arm(job: Job, arm: str, out: dict[str, Any], params: dict[str, Any], opt: Any) -> int:
    """After a crash: back to the checkpoint and its record; no checkpoint: from the start."""
    ckpt = job.large / "ckpt" / f"{arm}.pt"
    if not ckpt.is_file():
        out |= {"steps": [], "reads": {}, "resumes": []}
        return 0
    start, _, _ = resume(ckpt, params, job.W0, opt, None)
    out |= {"steps": [s for s in out["steps"] if s["step"] <= start],
            "reads": {t: r for t, r in out["reads"].items() if int(t) <= start}}
    out["resumes"].append(start)
    job.log(f"[{arm}] resuming after step {start}")
    return start


def f_rollouts(job: Job, out: dict[str, Any], steps: int) -> None:
    """F's data: W0's rollouts of every step, sampled once before its first update, with the same seeds as the RL arms;
    step 1's checks at once, not after the hours of sampling."""
    for t in range(1, steps + 1):
        p = job.large / "rollouts" / "F" / f"step_{t}.jsonl"
        if not p.is_file():
            recs = job.sampled(job.step_pids(t), job.cfg.samples, t)
            if t == 1:
                first_checks(job, "F", out, recs)
            save_rollouts(p, recs)
            job.log(f"[F] W0's rollouts of step {t}: {json.dumps(rollout_summary(recs))}")


def first_checks(job: Job, arm: str, out: dict[str, Any], recs: Records) -> None:
    """Step 1's rollouts: identical across the arms, reward rate in its band."""
    E, sha, rate = job.pred["early"], rollout_hash(recs), rollout_summary(recs)["reward_rate"]
    out["rollout_sha_1"] = sha
    if arm != "R":
        ref = job.get("R")["rollout_sha_1"]
        job.check(out, "step 1's rollouts identical to R's", sha == ref, {"sha": sha, "R": ref})
    job.check(out, "step 1's reward rate", in_band(rate, E["reward_step1"]), {"rate": rate, "band": E["reward_step1"]}, band=True)


def live_checks(job: Job, out: dict[str, Any], rec: dict[str, Any], read: dict[str, Any]) -> None:
    """The first step that moved the weights: Adam's first step is lr x sign of the gradient."""
    E = job.pred["early"]
    ratio = rec["dw_norm"] / (job.cfg.lr * math.sqrt(job.numel))
    job.check(out, "first live step: |dW| / (lr sqrt(numel))", in_band(ratio, E["dw_ratio"]),
              {"step": rec["step"], "ratio": ratio, "band": E["dw_ratio"]}, band=True)
    job.check(out, "first live step: KL(W0 || W_1) on H", in_band(read["kl"], E["kl_first"]),
              {"step": rec["step"], "kl": read["kl"], "band": E["kl_first"]}, band=True)


def save_dw(job: Job, arm: str, t: int, params: dict[str, Any], up: Uploader) -> None:
    path = job.large / "dW" / arm / f"step_{t}.safetensors"
    save(path, delta_fp32(params, job.W0), F32, droppable=False, log=job.log)
    job.publish(up, path)


def arm_step(job: Job, arm: str, out: dict[str, Any], params: dict[str, Any], opt: Any, t: int, saves: set[int],
             up: Uploader) -> None:
    cfg, t0, path = job.cfg, time.time(), job.large / "rollouts" / arm / f"step_{t}.jsonl"
    if arm == "F":
        recs = load_rollouts(path)
    else:
        recs = job.sampled(job.step_pids(t), cfg.samples, t)
        save_rollouts(path, recs)
    if t == 1:
        first_checks(job, arm, out, recs)
    rec = {"step": t, "sample_s": time.time() - t0, **rollout_summary(recs)}
    rec |= update(job, params, opt, recs, arm_weights(arm, recs, cfg.seed, t))
    first_live = rec["live"] and not any(s["live"] for s in out["steps"])
    out["steps"].append(rec)
    job.put(arm, out)
    job.log(f"[{arm}] {json.dumps(rec)}")
    job.check(out, "loss and gradient norm finite", ok(rec["loss"]) and ok(rec["grad_norm"]), {"step": t})
    if first_live or t in saves:
        out["reads"][str(t)] = read = job.read_state()
        job.log(f"[{arm}] step {t} read {json.dumps(read)}")
        if first_live:
            live_checks(job, out, rec, read)
    if t in saves:
        save_dw(job, arm, t, params, up)
    if t == cfg.drop_at and arm != "F":
        change = reward_change([s["reward_rate"] for s in out["steps"]], t)
        job.check(out, f"reward change by step {t}", above(change, job.pred["early"]["reward_change_above"]),
                  {"change": change}, band=True)
    job.put(arm, out)
    if t in cfg.ckpt_steps:
        checkpoint(job.large / "ckpt" / f"{arm}.pt", t, params, job.W0, opt, None, set(), set())
        job.log(f"[{arm}] checkpoint at step {t}")
    if t == cfg.pause_at:
        job.log(f"[{arm}] paused at step {t}: run the same command")
        sys.exit(3)


def finish_arm(job: Job, arm: str, out: dict[str, Any], steps: int, up: Uploader) -> None:
    acc = job.accuracy(arm, up)
    w0 = {s: load_rollouts(job.large / "acc" / f"W0_{s}.jsonl") for s in acc}
    out |= {"accuracy": {s: v["summary"] for s, v in acc.items()}, "last": steps, "peaks": job.peaks(),
            "gain": {s: gain(w0[s], acc[s]["recs"]) for s in acc}}
    job.log(f"[{arm}] gain {json.dumps(out['gain'])}; {json.dumps(out['peaks'])}")
    tar = job.large / "rollouts" / f"{arm}.tar"
    with tarfile.open(tar, "w") as f:
        f.add(job.large / "rollouts" / arm, arcname=arm)
    job.publish(up, tar)
    out["done"] = True
    job.put(arm, out)
    (job.large / "ckpt" / f"{arm}.pt").unlink(missing_ok=True)


def stage_arm(job: Job, arm: str) -> None:
    cfg = job.cfg
    out = job.begin("A0", "S") if arm == "R" else job.begin("R")
    if out is None:
        return
    S = job.get("S")
    if S["device"] != torch.cuda.get_device_name():
        sys.exit(f"S ran on {S['device']}: rerun it on this card (the rehearsal's file is not the pod's)")
    steps = S["n_steps"] if arm == "N" else cfg.steps
    saves = {t for t in cfg.saves if t <= steps}
    job.fp32()
    if arm == "F":
        f_rollouts(job, out, steps)
    up = job.uploader()
    try:
        with trainable(job.model, job.W0) as params:
            opt = adamw(params, cfg.lr)
            start = start_arm(job, arm, out, params, opt)
            for t in range(start + 1, steps + 1):
                arm_step(job, arm, out, params, opt, t, saves, up)
            finish_arm(job, arm, out, steps, up)
    finally:
        up.finish()


# ----------------------------------------------------------------------------- score, main


def stage_score(cfg: Config) -> None:
    res = RESULTS / cfg.out
    A0, R, N, f = (read_json(res / f"{s}.json") for s in ("A0", "R", "N", "F"))
    scored = score_all(json.loads(PRED.read_text()), A0, R, N, f)
    write_text(res / "score.json", json.dumps(scored, indent=1, default=float))
    print(json.dumps(scored, indent=1, default=float), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("A0", "S", "R", "N", "F", "score"))
    ap.add_argument("--size", required=True, choices=tuple(SIZES))
    ap.add_argument("--usd-per-h", type=float, default=3.99, help="S's cost rule: the pod's rate (0 on the cluster)")
    a = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"{PRED} is missing: predictions are filed before any run")
    cfg = SIZES[a.size]
    (RESULTS / cfg.out).mkdir(parents=True, exist_ok=True)
    if a.stage == "score":
        stage_score(cfg)
        return
    job = Job(cfg, a.stage)
    if a.stage == "A0":
        stage_A0(job)
    elif a.stage == "S":
        stage_S(job, a.usd_per_h)
    else:
        stage_arm(job, a.stage)
    job.log(f"[{a.stage}] finished in {time.time() - job.t0:.0f} s; {json.dumps(job.peaks())}")


if __name__ == "__main__":
    main()
