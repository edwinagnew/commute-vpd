# Run brief: E69 (qwen_rl), the GRPO loop on Qwen3-1.7B

Plan `reports/plans/2026-11-03_plan_e69-grpo-loop.md` (the spec). Rules: `results/predictions.json` (the run refuses to start without it). Code: `run.py`, `job.py`, `fetch.py`, `src/commute/qwen_rl.py`. Treated as gated: the paid run needs the review and the user's go-ahead on the rehearsal's numbers (plan, Gates 5 and 6).

Exit codes: 0 done; 1 a refusal; 2 an early check missed (`results/<stage>.json` has `"stopped"`, the stage refuses until the file is moved: tell the lead); 3 paused at a checkpoint (tiny only): the same command goes on.

## 1. Rehearsal, $0, cluster `debug`, one L40

Needs `WANDB_API_KEY` in the submitting shell. Submit from the clone root; record job IDs.

**(a) Tiny, every stage end to end.** 4 problems x 2 samples, 3 steps per arm, H of 4 problems x 8, 8-problem accuracy, δW at steps 1 and 3. At this size a value's band is logged as `MISSED` and does not stop (the numbers mean nothing on 8 samples); identity, finiteness and code errors stop. Results go to `results/tiny/` and `results/large/tiny/`.
```
sbatch cluster/qwen_rl.sbatch A0 tiny
sbatch cluster/qwen_rl.sbatch S tiny
sbatch cluster/qwen_rl.sbatch R tiny      # exit 3 at step 2, after the checkpoint
sbatch cluster/qwen_rl.sbatch R tiny      # resumes: log "resuming after step 2"; ends done
sbatch cluster/qwen_rl.sbatch N tiny      # the same two jobs
sbatch cluster/qwen_rl.sbatch N tiny
sbatch cluster/qwen_rl.sbatch F tiny      # the same two jobs
sbatch cluster/qwen_rl.sbatch F tiny
sbatch cluster/qwen_rl.sbatch score tiny
sbatch cluster/qwen_rl_fetch.sbatch --size tiny --only rollouts/      # the upload and fetch of a small file
```
Each waits for the one before (`--dependency=afterany:<id>`, or submit in turn). Read: `step 1's rollouts identical to R's` ok in N and F; `R`'s |δW| at step 3 after the resume continuing the first job's; `fetch` ends done (δW over the tiny size's 100 MB cap are skipped on upload by design). Then free the space: `rm -r ~/commute-data/results-large/qwen_rl/tiny` (δW 34 GB) and `rm -r experiments/qwen_rl/results/tiny`.

**(b) Full-size slice on the L40.** `sbatch cluster/qwen_rl.sbatch S full`. One batch of 512 by the sampler and by `generate` (length and reward within 2 SE, over problems; a miss stops: tell the lead), two RL steps, nothing saved. Rate 0 keeps the cost rule off, and A0 is not needed at this rate. Numbers to write into the plan's "Rehearsal record": `sampler.sampler_s`, `sampler.generate_s` (the plan expects 2-3 x), `seconds.sample`, `seconds.train`, `seconds.f_train`, `seconds.read`, `peaks`, `sampler.peaks`, and `grad_rms` per step in the log (Adam's eps is 1e-8). Then `git mv`/rename `results/S.json` to `results/S_rehearsal_L40.json`: stage R refuses an S from another card, and the pod's S writes its own.

## 2. Pod (after the go-ahead)

One RunPod H100 80 GB, any region, **no network volume** (`remote-kernels.toml` names none before `start()`; the lead edits it, never while a pod of the session runs), container disk **250 GB**. Label `commute-qwen_rl-<id>`. Pass `--usd-per-h` at the pod's real rate. Needs `WANDB_API_KEY`.

Set-up: `uv sync --locked --no-default-groups --group dev --group gpu`, `export HF_HOME=/root/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1`. No decomposition is needed. `results/large` arrives as a dangling link; `run.py` replaces it with a real directory.

**Commands**, from the repo root, in this order, one background subprocess each, teed to the console. A crash: the same command resumes from the checkpoint (steps 20, 50); only R, N, F resume, A0 and S rerun whole.
```
.venv/bin/python experiments/qwen_rl/run.py --stage A0 --size full --usd-per-h <rate> 2>&1 | tee -a experiments/qwen_rl/results/console.log /proc/1/fd/1
.venv/bin/python experiments/qwen_rl/run.py --stage S  --size full --usd-per-h <rate> 2>&1 | tee -a experiments/qwen_rl/results/console.log /proc/1/fd/1
.venv/bin/python experiments/qwen_rl/run.py --stage R  --size full --usd-per-h <rate> 2>&1 | tee -a experiments/qwen_rl/results/console.log /proc/1/fd/1
.venv/bin/python experiments/qwen_rl/run.py --stage N  --size full --usd-per-h <rate> 2>&1 | tee -a experiments/qwen_rl/results/console.log /proc/1/fd/1
.venv/bin/python experiments/qwen_rl/run.py --stage F  --size full --usd-per-h <rate> 2>&1 | tee -a experiments/qwen_rl/results/console.log /proc/1/fd/1
.venv/bin/python experiments/qwen_rl/run.py --stage score --size full 2>&1 | tee -a experiments/qwen_rl/results/console.log /proc/1/fd/1
```

| stage | time (plan's estimate) | first values to see | stop (exit 2) | download when it ends |
|---|---|---|---|---|
| A0 | ~30 min | `[accuracy W0 gsm8k]` strict_dropped ~0.777, gsm1k ~0.782; `[A0]` pool non-empty, `delta_r` and `kl` ~0 | P0 | `results/A0.json`, `SHA256SUMS`, `run.log` |
| S | ~15 min | `[S]` sampler vs generate; `projected_usd` | sampler disagrees with `generate`; projected over $50 even with N at 50 steps | `S.json`, `run.log` |
| R | ~2.4 h | `[check]` reward rate 0.60-0.88; first live step |δW| ratio 0.85-1.05, KL 0.0015-0.02; `grad_norm` far above 1e-8 x sqrt(1.41e9) | a check missed; at step 20 reward change at or below -0.05 (branch R3: N and F do not start) | `R.json`, `SHA256SUMS`, `run.log` after each saved step |
| N | ~2.4 h (half if S cut it) | `step 1's rollouts identical to R's` ok | as R, and the identity | as R |
| F | ~3 h, of which the first ~1.7 h samples W0's rollouts | identity at the first step of sampling, then as R | as R (no reward-change check) | as R |

δW and rollouts go to W&B as written (5.64 GB per δW, 21 in all if N is not cut; `Uploader.finish` waits for the queue at each stage's end, so the arm's last minutes are uploads). Download JSONs into `experiments/qwen_rl/results/`, never into the `results/large` link. `results/large/acc/` and `rollouts/` also reach W&B; nothing else from the disk is needed once `SHA256SUMS` and the JSONs are home. Terminate once `score.json`'s and `SHA256SUMS`'s sha256 match on both sides, before any local move. Each stage's end and any stop go to the lead before the next paid stage.

If a process dies with the pod alive, `wandb` may have dropped a queued file: after the resume, compare `SHA256SUMS` with the artifacts by the fetch below.

## 3. Fetch to the cluster

Commit the run's `results/` (JSONs, `SHA256SUMS`). Then, from the clone root:
```
sbatch cluster/qwen_rl_fetch.sbatch
```
Each file lands in `~/commute-data/results-large/qwen_rl/` at its `SHA256SUMS` path (`dW/<arm>/step_<t>.safetensors`, `rollouts/<arm>.tar`, `rollouts/H.jsonl`, `acc/<label>_<set>.jsonl`), checked against the sum; exit 0 means all match. Artifacts: `edwinagnew-mats/commute/qwen_rl-<parent>-<file>:latest`, type `qwen_rl`. A second run of the fetch skips the files that already match. Tell `qwen-decomp-lead` when each arm's δW land (plan, R1).
