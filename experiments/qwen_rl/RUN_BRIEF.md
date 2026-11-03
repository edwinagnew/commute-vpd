# Run brief: E69 (qwen_rl), the GRPO loop on Qwen3-1.7B

Plan `reports/plans/2026-11-03_plan_e69-grpo-loop.md` (the spec). Rules: `results/predictions.json` (the run refuses to start without it). Code: `run.py`, `job.py`, `fetch.py`, `src/commute/qwen_rl.py`. Treated as gated: the paid run needs the review and the user's go-ahead on the rehearsal's numbers (plan, Gates 5 and 6).

Exit codes: 0 done; 1 a refusal; 2 an early check missed (`results/<stage>.json` has `"stopped"`, the stage refuses until the file is moved: tell the lead); 3 paused at a checkpoint (tiny only): the same command goes on; 4 `verify` found a file missing from W&B.

Do not trust the exit status alone: read `"stopped"` and `"done"` in the stage's JSON (`results/<stage>.json`, or `results/lr<tag>/<stage>.json` at another rate) after every stage. `"done": true` and no `"stopped"` is a finished stage; anything else is not.

## 1. Rehearsal, $0, cluster `debug`, one L40

Needs `WANDB_API_KEY` in the submitting shell. Submit from the clone root; record job IDs.

**(a) Tiny, every stage end to end.** 4 problems x 2 samples, 3 steps per arm, H of 4 problems x 8, 8-problem accuracy, δW at steps 1 and 3. At this size a value's band is logged as `MISSED` and does not stop (the numbers mean nothing on 8 samples); identity, finiteness and code errors stop. Results go to `results/tiny/` and `results/large/tiny/`.
```
sbatch cluster/qwen_rl.sbatch A0 tiny
sbatch cluster/qwen_rl.sbatch S tiny
sbatch cluster/qwen_rl.sbatch R tiny --upload-cap-gb 6   # exit 3 at step 2, after the checkpoint (the cap flag: see (a2))
sbatch cluster/qwen_rl.sbatch R tiny --upload-cap-gb 6   # resumes: log "resuming after step 2"; ends done
sbatch cluster/qwen_rl.sbatch verify tiny --upload-cap-gb 6   # (a2): before N and F; every row ok
sbatch cluster/qwen_rl.sbatch N tiny      # the same two jobs
sbatch cluster/qwen_rl.sbatch N tiny
sbatch cluster/qwen_rl.sbatch F tiny      # the same two jobs
sbatch cluster/qwen_rl.sbatch F tiny
sbatch cluster/qwen_rl.sbatch score tiny
sbatch cluster/qwen_rl_fetch.sbatch --size tiny --only rollouts/      # the upload and fetch of a small file
```
**(a2) One large file, rehearsed once.** The tiny size's cap (100 MB) skips every δW, so R's two jobs above run with `--upload-cap-gb 6`: the 5.6 GB δW of steps 1 and 3 go to W&B, and `verify` (with the same flag, straight after R and before N and F, whose δW stay under the default cap and would read as missed) must print every row `ok`. Write down the upload minutes and any W&B error. A `verify` miss here is the finding.

Each waits for the one before (`--dependency=afterany:<id>`, or submit in turn). Read: `step 1's rollouts identical to R's` ok in N and F; `R`'s |δW| at step 3 after the resume continuing the first job's; `fetch` ends done (δW over the tiny size's 100 MB cap are skipped on upload by design). Then free the space: `rm -r ~/commute-data/results-large/qwen_rl/tiny` (δW 34 GB) and `rm -r experiments/qwen_rl/results/tiny`.

**(b) Full-size slice on the L40.** `sbatch cluster/qwen_rl.sbatch S full`. One batch of 512 by the sampler and by `generate` (length and reward within 3 SE, over problems; a miss stops: tell the lead), two RL steps, nothing saved. Rate 0 keeps the cost rule off, and A0 is not needed at this rate. Numbers to write into the plan's "Rehearsal record": `sampler.sampler_s`, `sampler.generate_s` (the plan expects 2-3 x), `seconds.sample`, `seconds.train`, `seconds.f_train`, `seconds.read`, `peaks`, `sampler.peaks`, and `grad_rms` per step in the log (Adam's eps is 1e-8). Then `git mv`/rename `results/S.json` to `results/S_rehearsal_L40.json`: stage R refuses an S from another card, and the pod's S writes its own.

## 2. Pod (after the go-ahead)

One RunPod H100 80 GB, any region, **no network volume** (`remote-kernels.toml` names none before `start()`; the lead edits it, never while a pod of the session runs), container disk **250 GB**. Label `commute-qwen_rl-<id>`. Pass `--usd-per-h` at the pod's real rate. Needs `WANDB_API_KEY`.

Set-up: `uv sync --locked --no-default-groups --group dev --group gpu`, `export HF_HOME=/root/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1`. No decomposition is needed. `results/large` arrives as a dangling link; `run.py` replaces it with a real directory.

**Commands**, from the repo root, one background subprocess each, teed to the console, each beginning `set -o pipefail;` so that `tee` does not turn a failure into exit 0. A crash: the same command resumes from the checkpoint (steps 20, 50, 100); only R, N, F resume, A0 and S rerun whole. `<tag>` is the rate: `1e-6` (default; omit `--lr-tag`), `3e-6`, `3e-7` or `1e-6w10` (1e-6 after 10 steps of warm-up, as `probe.py`).
```
set -o pipefail; .venv/bin/python experiments/qwen_rl/run.py --stage A0 --size full --usd-per-h <rate> 2>&1 | tee -a experiments/qwen_rl/results/console.log /proc/1/fd/1
set -o pipefail; .venv/bin/python experiments/qwen_rl/run.py --stage S  --size full --usd-per-h <rate> 2>&1 | tee -a experiments/qwen_rl/results/console.log /proc/1/fd/1
set -o pipefail; .venv/bin/python experiments/qwen_rl/run.py --stage R  --size full --usd-per-h <rate> 2>&1 | tee -a experiments/qwen_rl/results/console.log /proc/1/fd/1
# after R: verify (below), then ask which arm comes next and run exactly what it prints:
.venv/bin/python experiments/qwen_rl/run.py --stage next --size full --lr-tag <tag> --usd-per-h <rate>
```
`--stage next` reads R's result and the plan's decision tree and prints JSON: `stage` (R, N, F, score or `lead`), `tag`, `why`, and `command`. Run `command` as printed (it carries `--lr-tag`, `--usd-per-h` and the tee). `stage: lead` or an empty `command`: stop and tell the lead; apply no judgment of your own. Order it gives: R at 1e-6; then, from R's result, N and F at 1e-6 (R1), or R at 3e-6 (R2, P1 no gain) or R at 3e-7 (R3, reward fell by step 20); after any rerun, call `next` with that rerun's `--lr-tag`, which gives its N and F, then `score`. The 1e-6 files are never overwritten: another rate has its own `results/lr<tag>/`, `results/large/lr<tag>/` and W&B prefix `qwen_rl_lr<tag>`; A0, S, W0's accuracy, H and F's rollouts are shared, so F at a second rate does not resample.

**Verify, after R (before N and F start, so an upload problem shows before more money goes) and after every later arm, and again before `terminate()`:**
```
set -o pipefail; .venv/bin/python experiments/qwen_rl/run.py --stage verify --size full --lr-tag <tag> 2>&1 | tee -a experiments/qwen_rl/results/console.log /proc/1/fd/1
```
It checks that every `SHA256SUMS` row of that tag is a committed W&B artifact whose size and md5 equal the file on the pod's disk, prints each row `ok` or `MISSED`, and exits 4 on any miss. A miss: stop and tell the lead at once (the file is still on the pod's disk; do not terminate).

| stage | time (plan's estimate) | first values to see | stop (exit 2) | download when it ends |
|---|---|---|---|---|
| A0 | ~30 min | `[accuracy W0 gsm8k]` strict_dropped ~0.777, gsm1k ~0.782; `[A0]` pool non-empty, `delta_r` and `kl` ~0 | P0 | `results/A0.json`, `SHA256SUMS`, `run.log` |
| S | ~15 min | `[S]` sampler vs generate; `projected_usd` | sampler disagrees with `generate` (beyond 3 SE); projected over $50 even with N at 50 steps | `S.json`, `run.log` |
| R | ~2.4 h | `[check]` reward rate 0.60-0.88; first live step |δW| ratio 0.85-1.05, KL 0.0015-0.02; `grad_norm` far above 1e-8 x sqrt(1.41e9) | a check missed; at step 20 reward change at or below -0.05 (branch R3: N and F do not start) | `R.json`, `SHA256SUMS`, `run.log` after each saved step |
| N | ~2.4 h (half if S cut it) | `step 1's rollouts identical to R's` ok | as R, and the identity | as R |
| F | ~3 h, of which the first ~1.7 h samples W0's rollouts | identity at the first step of sampling, then as R | as R (no reward-change check) | as R |

For a rerun at another rate, the same files are under `results/lr<tag>/` (`R.json`, `N.json`, `F.json`, `SHA256SUMS`, `run.log`, `score.json`); download them into the same path locally.

δW and rollouts go to W&B as written (5.64 GB per δW, 21 in all if N is not cut; `Uploader.finish` waits for the queue at each stage's end, so the arm's last minutes are uploads). Download JSONs into `experiments/qwen_rl/results/`, never into the `results/large` link. `results/large/acc/` and `rollouts/` also reach W&B; nothing else from the disk is needed once `SHA256SUMS` and the JSONs are home. Terminate only when `--stage verify` has passed for every tag run, and `score.json`'s and `SHA256SUMS`'s sha256 match on both sides, before any local move. Each stage's end and any stop go to the lead before the next paid stage.

**Pod lost.** A new pod has none of `rollouts/H.jsonl` or `acc/W0_*.jsonl`, though `A0.json` and `S.json` say done; without them an arm crashes after its whole run. Do this before restarting the arm (`stage_arm` now refuses at once if they are missing):
1. Set up the pod as above, then sync the local `experiments/qwen_rl/results/` (JSONs and `SHA256SUMS`, including `lr<tag>/`) to it.
2. `WANDB_API_KEY` set: `.venv/bin/python experiments/qwen_rl/fetch.py --root experiments/qwen_rl/results/large --only rollouts/H.jsonl`, then the same with `--only acc/W0_`.
If a process dies with the pod alive, run `--stage verify` after the resume: `wandb` may have dropped a queued file.

3. Run the arm in flight again (it restarts from step 0: the checkpoint died with the pod). F also needs `rollouts/F/` back: re-sampled, about 1.7 h, unless fetched from W&B via `rollouts/F.tar` of an earlier F.

## 3. Fetch to the cluster

Commit the run's `results/` (JSONs, `SHA256SUMS`). Then, from the clone root:
```
sbatch cluster/qwen_rl_fetch.sbatch
```
Each file lands in `~/commute-data/results-large/qwen_rl/` at its `SHA256SUMS` path (`dW/<arm>/step_<t>.safetensors`, `rollouts/<arm>.tar`, `rollouts/H.jsonl`, `acc/<label>_<set>.jsonl`), checked against the sum; exit 0 means all match. Artifacts: `edwinagnew-mats/commute/qwen_rl-<parent>-<file>:latest`, type `qwen_rl`. A second run of the fetch skips the files that already match. Tell `qwen-decomp-lead` when each arm's δW land (plan, R1).
