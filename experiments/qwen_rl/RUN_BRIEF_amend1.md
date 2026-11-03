# Run brief: E69 Amendment 1, a second seed of RL and the null, and the $0 reads

Plan `reports/plans/2026-11-03_plan_e69-amend1.md`. Rules: `results/predictions_amend1.json` (P1-P7 stand). Under the gate. Code: `run.py --seed`, `amend1.py`, `relaxed.py`, `score_amend1.py`. Control G0 has passed (`results/amend1_G0.json`). Exit codes as `RUN_BRIEF.md`: 0 done, 1 a refusal, 2 a check missed (read `"stopped"` in the stage's JSON, tell the lead; the stage refuses until the file is moved). Read `"done"` and `"stopped"` after every stage; do not trust the exit status alone.

The two parts do not wait for each other. Part B's seed-1 jobs can start at once; its seed-2 jobs need Part A's δW on the cluster.

## A. R2 then N2 on one RunPod H100: ~5 h, ~$20; hand back past $30

One RunPod H100 80 GB, any region, no network volume, container disk 250 GB. Label `commute-qwen_rl-e69a1`. `WANDB_API_KEY` needed. The main clone must hold this commit before `sync()`: sync is rooted there. Same pod for both arms: N2's step-1 identity check compares with R2's hash, and rows come from the card's memory.

**On the pod.** Set-up, about 30 min, from the repo root:
```
uv sync --locked --no-default-groups --group dev --group gpu
export HF_HOME=/root/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
if [ -L experiments/qwen_rl/results/large ]; then rm experiments/qwen_rl/results/large; fi; mkdir -p experiments/qwen_rl/results/large
.venv/bin/python experiments/qwen_rl/fetch.py --root experiments/qwen_rl/results/large --only rollouts/H.jsonl
.venv/bin/python experiments/qwen_rl/fetch.py --root experiments/qwen_rl/results/large --only acc/W0_
```
The link line comes first: `results/large` arrives as a dangling link, and `fetch.py` runs before anything replaces it, so its `mkdir` would fail. The two fetches restore H and A0's per-problem answers, each checked against the committed `results/SHA256SUMS` (3 small files). `results/A0.json` and `results/S.json` arrive by `sync()` and are not rerun: `run.py` refuses `--seed` on A0 and S. Stage R refuses if `S.json`'s device (`NVIDIA H100 80GB HBM3`) is not this card's: stop and tell the lead, do not rerun S without asking.

**Commands**, one background subprocess each, re-armed with reads of at most 5 min (no `execute()` or `wait()` longer). A crash with the pod alive: the same command resumes from the checkpoint at steps 20, 50, 100.
```
set -o pipefail; .venv/bin/python experiments/qwen_rl/run.py --stage R --size full --lr-tag 1e-6w10 --seed 70 --usd-per-h <rate> 2>&1 | tee -a experiments/qwen_rl/results/console.log /proc/1/fd/1
set -o pipefail; .venv/bin/python experiments/qwen_rl/run.py --stage verify --size full --lr-tag 1e-6w10 --seed 70 2>&1 | tee -a experiments/qwen_rl/results/console.log /proc/1/fd/1
set -o pipefail; .venv/bin/python experiments/qwen_rl/run.py --stage N --size full --lr-tag 1e-6w10 --seed 70 --usd-per-h <rate> 2>&1 | tee -a experiments/qwen_rl/results/console.log /proc/1/fd/1
set -o pipefail; .venv/bin/python experiments/qwen_rl/run.py --stage verify --size full --lr-tag 1e-6w10 --seed 70 2>&1 | tee -a experiments/qwen_rl/results/console.log /proc/1/fd/1
```
Do not call `--stage next` and do not run F: this amendment's tree is these four commands. Everything goes to `results/lr1e-6w10_s70/` and `results/large/lr1e-6w10_s70/`, W&B prefix `qwen_rl_lr1e-6w10_s70`; `results/lr1e-6w10/` is never written. If the path printed is anything else, stop.

| stage | time (E69's seconds) | first values to see | stop (exit 2) |
|---|---|---|---|
| R | ~1.95 h, of which the last 13 min is accuracy | step 1 reward rate 0.60-0.88 (seed 1: 0.80); first live step: dW ratio 0.85-1.05 (0.92), KL 3e-5 to 2e-3 at the warm-up's rate (0.00018); `reward change by step 20` above -0.05 (seed 1: positive) | any of these |
| verify | minutes | every row `ok` | exit 4: a file missing from W&B. Stop, tell the lead, do not terminate |
| N | ~2.5 h | `step 1's rollouts identical to R's` ok; then as R. Seed 1's first KL was 5e-5, near the band's floor: a miss here is a stop at step 1, cheap | as R, and the identity |

Each arm: 100 steps, δW at steps 1, 2, 5, 10, 20, 50, 100 (5.64 GB each, 14 files, 79 GB), then GSM8K and GSM1k accuracy.

**On the pod, and home when.** Nothing on the pod outlives it.
| what | where on the pod | home how, when |
|---|---|---|
| `R.json`, `N.json`, `SHA256SUMS`, `run.log` | `experiments/qwen_rl/results/lr1e-6w10_s70/` | `download()` file by file to the same path, after each saved step and at each arm's end |
| per-problem answers `acc/{R,N}_gsm{8k,1k}.jsonl` | `results/large/lr1e-6w10_s70/acc/` | `download()` file by file to `experiments/qwen_rl/results/lr1e-6w10_s70_dl/acc/` at each arm's end; also on W&B. Never into `results/large`, the local symlink refuses it |
| δW `dW/{R,N}/step_*.safetensors`, `rollouts/{R,N}.tar` | `results/large/lr1e-6w10_s70/` | to W&B as written; `--stage verify` proves size and md5, and part A.2 fetches them. Too large for `download()`. Only if verify still misses after one retry and the lead says so: `download()` the missed file to `results/lr1e-6w10_s70_dl/` |
| H, W0's answers, `A0.json`, `S.json` | as fetched and synced | already home |

**Terminate** when N is done, the second verify has exited 0, and `N.json`'s and `SHA256SUMS`' sha256 match on both sides. Terminate before any local move or commit. Tell the lead at each arm's end and at any stop.

**Pod lost** (connection refused, port moved): stop and tell the lead at once; a `/mcp` reconnect may fix it. A new pod: the set-up above, with `sync()` carrying the local `results/lr1e-6w10_s70/` (it is not gitignored). The arm in flight restarts from step 0, since its checkpoint died with the pod; N needs R's finished `R.json` for the identity check, and a miss of that check on the new pod is a stop. At most ~$10 lost.

**A.2 After the pod, on the cluster**, from the main clone root, `WANDB_API_KEY` set, the JSONs and `SHA256SUMS` committed with named paths:
```
sbatch cluster/qwen_rl_fetch.sbatch --lr-tag 1e-6w10 --seed 70
```
About 79 GB, ~30 min, every file checked against the sum. Then `rm -r experiments/qwen_rl/results/lr1e-6w10_s70_dl`.

## B. Saved-step accuracy and forced openings: cluster `debug`, one L40, $0

Submit from the main clone root, after this commit is in it. One state per job, because a state is 1,319 + 1,197 problems, 13 min on E69's H100 and an estimated 20 min on an L40 with loading; the limit is 30 min. Chain with `--dependency=afterok:<id>` and record the IDs. **Read the first job's seconds:** if it passes 28 min, stop and tell the lead. Memory: ~13 GB of weights plus ~22 GB of cache at batch 128, on 46 GB. If a job waits in the queue over 30 min, tell the lead. Do not move a state to another card: gains are read against W0 on the same card.

**Controls, first and alone.**
```
sbatch cluster/qwen_rl_amend1.sbatch --stage saved --arm W0
sbatch --dependency=afterok:<W0 id> cluster/qwen_rl_amend1.sbatch --stage saved --arm N --step 100
```
W0 reads the card's floor against A0: `results/lr1e-6w10/amend1/W0card.json`, `floor.<set>`: `gap`, `flips`, `f`. Stop if gap exceeds 1.5 points. N100 rebuilt must read `N.json`'s `strict_dropped` within `f`, per set. A miss exits 2: tell the lead, with both numbers.

**Seed 1's saved steps, after N100.** Six jobs, each `--dependency=afterok:<N100 id>`:
```
sbatch --dependency=afterok:<N100 id> cluster/qwen_rl_amend1.sbatch --stage saved --arm N --step 50
sbatch --dependency=afterok:<N100 id> cluster/qwen_rl_amend1.sbatch --stage saved --arm N --step 20
sbatch --dependency=afterok:<N100 id> cluster/qwen_rl_amend1.sbatch --stage saved --arm N --step 10
sbatch --dependency=afterok:<N100 id> cluster/qwen_rl_amend1.sbatch --stage saved --arm R --step 50
sbatch --dependency=afterok:<N100 id> cluster/qwen_rl_amend1.sbatch --stage saved --arm R --step 20
sbatch --dependency=afterok:<N100 id> cluster/qwen_rl_amend1.sbatch --stage saved --arm R --step 10
```

**Openings, after W0 and N100.** GSM8K only, about 10 min each. The control first, then the rest after it:
```
sbatch --dependency=afterok:<W0 id> cluster/qwen_rl_amend1.sbatch --stage open --arm W0 --opening own
sbatch --dependency=afterok:<own id> cluster/qwen_rl_amend1.sbatch --stage open --arm W0 --opening "We are"
sbatch --dependency=afterok:<own id> cluster/qwen_rl_amend1.sbatch --stage open --arm W0 --opening "We need"
sbatch --dependency=afterok:<own id>:<N100 id> cluster/qwen_rl_amend1.sbatch --stage open --arm N --step 100 --opening own
```
The control compares response, length and truncation with `W0card` on every problem. If any differs, the same job measures the floor with plain W0 at batch 127 (10 min more) and passes only if the count is within it: `differs` in `results/lr1e-6w10/amend1/W0_own.json`. A miss exits 2: a code error until shown otherwise.

**Seed 2's saved steps, after A.2.** The same six commands, each with `--seed 70` and `--dependency=afterok:<fetch id>`. They need seed 1's W0 and N100 done on the same card.

**Files.** Numbers in `results/<run>/amend1/<label>.json` and the log `results/<run>/amend1.log`; commit both with named paths. Per-problem answers in `results/large/<run>/acc/<label>_<set>.jsonl`, labels `W0card`, `N100`, `N50`, ..., `W0_own`, `W0_we_are`, `W0_we_need`, `N100_own`.

## C. Scoring, $0, any time
```
~/commute-agents/fleet/cpu-run --mem 6G -t 00:10:00 python experiments/qwen_rl/score_amend1.py
```
Run from the main clone, where the results are. It reads whatever exists, a file not there reads `missing`, and it writes `results/amend1_score.json`. S1 and S1r read seed 2's `acc/` files, which A.2 brings; for those alone, `sbatch cluster/qwen_rl_fetch.sbatch --lr-tag 1e-6w10 --seed 70 --only acc/`.
