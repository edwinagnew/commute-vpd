"""E69's $0 recipe probe: is Adam's first step at 1e-6 too large, and which rate and warm-up should the paid run use? Rules:
results/predictions_probe.json (refuses to start without it). Writes results/probe_<name>.json as it goes, every number also
printed, and the shared results/large/probe/pool.jsonl; uploads nothing, saves no δW, touches none of the run's files.
Submit: sbatch cluster/qwen_rl_probe.sbatch <arguments>

  --arm R|N --lr 1e-6 --warmup 10   20 steps of the run's loop; KL and ΔR on the pool and the step's own gain at the read steps
  --f-step --lr 3e-7                one F step from W0 on step 1's batch, read as sanity.py's check 3 F
  --score                           no model: the filed rules from the results files, into results/probe_score.json
The name is the arm in lower case, the rate and any warm-up: r1e-6w10, f3e-7. Exit 0 done; 1 a refusal.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any

import torch
from job import (
    DEV,
    LARGE,
    RESULTS,
    Job,
    config,
    load_rollouts,
    read_json,
    save_rollouts,
    write_text,
)
from run import update
from sanity import SanityJob, mean_se, per_problem

from commute import qwen_rl as q
from commute import qwen_sft as sft
from commute.jobs import hard_exit

D = dict[str, Any]
FILED, POOL = RESULTS / "predictions_probe.json", LARGE / "probe" / "pool.jsonl"


def result(name: str) -> Path:
    return RESULTS / f"probe_{name}.json"


def keep(name: str, out: D) -> None:
    write_text(result(name), json.dumps(out, indent=1, default=float))


def say(tag: str, row: D) -> None:
    print(f"[{tag}] {json.dumps(row, default=float)}", flush=True)


def held_pool(job: Job, P: D) -> q.Records:
    """The pool every job reads: the first of H's list, by H's seed, from W0 (call before any update). Each job that finds
    no file draws it, and the first to link its finished copy into place wins; the others read that copy. A link either
    takes the name or fails, on NFS too."""
    if not POOL.is_file():
        recs = job.sampled(job.held[: P["pool_problems"]], P["pool_samples"], 0)
        tmp = POOL.with_name(f"pool.{os.getpid()}.{time.time_ns()}")
        save_rollouts(tmp, recs)
        try:
            os.link(tmp, POOL)
        except FileExistsError:
            pass
        tmp.unlink()
    for _ in range(60):  # another node's new file can take a moment to show here
        if POOL.is_file():
            break
        time.sleep(1)
    recs = load_rollouts(POOL)
    assert len(recs) == P["pool_problems"] * P["pool_samples"] and q.pool(recs), "the pool file is not the filed pool"
    return recs


def split_means(changes: list[float], recs: q.Records) -> D:
    """Mean change of the right samples and of the wrong ones; None for a side with no sample."""
    right = [c for c, r in zip(changes, recs, strict=True) if r["reward"]]
    wrong = [c for c, r in zip(changes, recs, strict=True) if not r["reward"]]
    return {"right": st.mean(right) if right else None, "n_right": len(right), "wrong": st.mean(wrong) if wrong else None, "n_wrong": len(wrong)}


def read_step(job: Job, before: dict[str, torch.Tensor], recs: q.Records, pool: q.Records) -> D:
    """KL and ΔR on the pool against W0; the last update's own gain on its own batch, against the weights before it."""
    budget = job.cfg.read_tokens
    changes, kl = q.read_h(job.model, job.W0, job.seqs(pool), budget)
    own, _ = q.read_h(job.model, {s: w.to(DEV) for s, w in before.items()}, job.seqs(recs), budget)
    return {"kl": kl, "delta_r": q.delta_r(changes, pool), "own": split_means(own, recs)}


def run_arm(job: Job, P: D, name: str, arm: str, lr: float, warmup: int) -> None:
    cfg, pool = job.cfg, held_pool(job, P)
    out: D = {"name": name, "arm": arm, "lr": lr, "warmup": warmup, "card": torch.cuda.get_device_name(), "rows": job.rows,
              "pool_sha": q.rollout_hash(pool), "pool_mixed": len(q.pool(pool)), "steps": [], "reads": {}}
    say(name, {k: out[k] for k in ("card", "rows", "pool_sha", "pool_mixed")})
    with sft.trainable(job.model, job.W0) as params:
        opt = sft.adamw(params, lr)
        for t in range(1, P["steps"] + 1):
            t0 = time.time()
            recs = job.sampled(job.step_pids(t), cfg.samples, t)
            sample_s = time.time() - t0
            reading = t in P["read_steps"]
            before = {s: p.detach().cpu() for s, p in params.items()} if reading else {}  # off the card through the update
            rec = update(job, params, opt, recs, q.arm_weights(arm, recs, cfg.seed, t), q.rate_at(lr, warmup, t))
            row = {"step": t, "sample_s": sample_s, **q.rollout_summary(recs), **rec, "seconds": time.time() - t0}
            if t == 1:
                row["rollout_sha"] = q.rollout_hash(recs)
            out["steps"].append(row)
            keep(name, out)
            say(f"{name} step {t}", row)
            if reading:
                out["reads"][str(t)] = read_step(job, before, recs, pool)
                keep(name, out)
                say(f"{name} read {t}", out["reads"][str(t)])
    out["done"] = True
    keep(name, out)


def run_f(job: Job, name: str, lr: float) -> None:
    """Sanity's check 3 F at this rate: a fresh optimiser from W0, one update, the same mean and SE of the kept samples."""
    cfg = job.cfg
    recs = job.sampled(job.step_pids(1), cfg.samples, 1)
    w = q.sft_weights(recs)
    with sft.trainable(job.model, job.W0) as params:
        opt = sft.adamw(params, lr)
        rec = update(job, params, opt, recs, w, lr)
        del opt
        gc.collect()
        torch.cuda.empty_cache()
        changes, kl = q.read_h(job.model, job.W0, job.seqs(recs), cfg.read_tokens)
    kept = [(c, r) for c, r, x in zip(changes, recs, w, strict=True) if x]
    mean, se = mean_se(per_problem([c for c, _ in kept], [r for _, r in kept]))
    out = {"name": name, "lr": lr, "card": torch.cuda.get_device_name(), "rows": job.rows, "rollout_sha": q.rollout_hash(recs),
           "mean": mean, "se": se, "kept": len(kept), "kl_on_batch": kl, **rec, "done": True}
    keep(name, out)
    say(name, out)


def score(P: D) -> None:
    runs = {n: read_json(result(n)) for n in P["names"]}
    reading = {n: q.probe_reading(runs[n], P["steps"]) for n in ("r1e-6", "r1e-6w10", "r3e-7", "n1e-6")}
    out = {"cause": q.cause_test(P["cause"], runs["f3e-7"], runs["f1e-7"], runs["f1e-6"]),
           "f": {n: {k: runs[n].get(k) for k in ("mean", "se", "kept", "kl_on_batch")} for n in ("f3e-7", "f1e-7", "f1e-6")},
           "recipe": q.recipe_rule(P["recipe"], reading["r1e-6"], reading["r1e-6w10"], reading["r3e-7"]),
           "readings": reading, "pools": sorted({str(r.get("pool_sha")) for r in runs.values() if "steps" in r})}
    write_text(result("score"), json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--arm", choices=("R", "N"))
    mode.add_argument("--f-step", action="store_true")
    mode.add_argument("--score", action="store_true")
    ap.add_argument("--lr", help="the rate as filed, as in 3e-7")
    ap.add_argument("--warmup", type=int, default=0)
    a = ap.parse_args()
    if not FILED.is_file():
        sys.exit(f"{FILED} is missing: predictions are filed before this runs")
    P = json.loads(FILED.read_text())
    if a.score:
        score(P)
        return
    name = f"{'f' if a.f_step else a.arm.lower()}{a.lr}{f'w{a.warmup}' if a.warmup else ''}"
    if name not in P["names"]:
        sys.exit(f"{name} is not one of the filed runs {P['names']}")
    if read_json(result(name)).get("done"):
        sys.exit(f"{result(name)} is done: move it to run it again")
    job = SanityJob(config("full", q.DEFAULT_TAG), "probe")
    job.fp32()
    if a.f_step:
        run_f(job, name, float(a.lr))
    else:
        run_arm(job, P, name, a.arm, float(a.lr), a.warmup)
    print(f"=== {name}: {time.time() - job.t0:.0f} s; {json.dumps(job.peaks())} ===", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    main()
