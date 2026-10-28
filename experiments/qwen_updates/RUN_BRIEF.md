# Run brief: E45 (qwen_updates), Job 1 — the updates ($0, MATS cluster `debug`, one L40 per job)

Plan `reports/plans/2026-10-28_plan_qwen-q2.md` (§3-§6, §8). Rules and first-minute values: `results/predictions.json` (run refuses to start without it). Code: `run.py`, `job.py`, `src/commute/qwen_{train,updates,facts,anchors}.py`. Job 2 (E44's table) is gated and not here.

**Before the first job**, in the lead's clone after `git pull`: `ls -l experiments/qwen_updates/results/large` must be a link into `~/commute-data/results-large/qwen_updates` (else `~/commute-agents/fleet/link-data qwen-decomp-lead`). Submit every job from the clone root, one at a time, each when the one before has ended. Record job IDs.

**Exit codes.** 0 done. **2 = STOP**: a first-minute check missed (`[check] ... MISSED`, `"stopped"` in the stage's JSON): report to the lead before any further job. **3 = budget**: the job stopped before 25 min with its work saved; resubmit the same command (P, S, E resume; T never returns 3).

| # | command (`sbatch` + args) | L40 time | first values to see in `cluster/logs/commute-qwen_updates-<id>.out` | stop rule (code) | output |
|---|---|---|---|---|---|
| 1 | `--mem=32G cluster/qwen_updates.sbatch --stage control` | ~12 min | within 2 min: `control: KL of one gold step at 1e-5: ok {"kl": ~0.10679, ..., "loss": ~2.407}`; then invented facts ≤ 0.02 (both draws), known ~0.8-1.0; last: W0 accuracy ~0.779, `same_as_baseline` near 1 | KL off 0.10679 by > 1%; invented > 0.02; accuracy off 0.779 by > 0.03 | `results/control.json`, `large/W0_test.jsonl` |
| 2 | `--mem=40G cluster/qwen_updates.sbatch --stage T --arm gold` | ~16 min | step 1: lr 0, loss ~2.4070, \|dW\| 0, KL 0; step 2: band ok, KL ~0.0037; steps 3, 5, 10: KL ~0.022, 0.110, 0.356 | loss off by > 1%; any curve step off by > 5%; step-2 KL > 0.005 | `results/T_gold_s0.json`, `large/dW/gold_s0/` |
| 3 | `--mem=32G cluster/qwen_updates.sbatch --stage E --arm gold` | ~20 min (+1 resubmit) | `[E gold_s0] state 0`, then one line per saved state (~50 s each), `generation at 100` (~7 min, full test) | — | `results/E_gold_s0.json`, `score.json` |
| 4 | `T --arm a2 --seed 0`, then `E --arm a2 --seed 0` | ~15 + 10 min | T: `data` logs `rows_per_step` (~30-40), step 2 KL ≤ 0.005 | as 2, no curve | `T_/E_a2_s0.json` |
| — | **read `results/score.json` → `anchor2_learns_s0`** | | `holds` true → go on; false → **stop, report to the lead** (plan §8: one 300-step amendment, `RUN_BRIEF_amend1.md`; it replaces row 8's `--arm a2 --seed 1`) | | |
| 5 | `--stage P` | ~15-20 min (+1) | `[P] none: strict ~0.77`, then fewshot, recheck, givens; `[P] rules ... chosen <name>` | none qualifies and the format fails → exit 2 | `results/P.json`, `large/P/` |
| 6 | `--stage S --set own`, resubmit until exit 0 | 2-3 jobs | first `[S own] batch 1/N ...: projected T s in all; peak GPU` (peak ≤ 44 GB) | out of memory → report (`prompts_per_batch` is fixed in `results/S_<set>.json` at the first batch: to lower `kv_tokens`, delete that file and `large/S/<set>/` first) | `large/S/own/`, `results/S_own.json` |
| 7 | `--stage S --set a1`, resubmit until exit 0 | 2-6 jobs (few-shot prompts are slower) | as 6 | as 6 | `large/S/a1/`, `results/S_a1.json` |
| 8 | `T`, then `E`, for `--arm own`, `--arm null`, `--arm a1 --seed 0`, `--arm a1 --seed 1`, `--arm a2 --seed 1` (T with `--mem=40G`) | ~15 + 20 min each | T: `data` (own: `n` kept ≥ 4,000), step 1 at rate 0, step 2 band and KL ≤ 0.005 | own kept < 4,000; step-2 checks | `T_/E_<arm>_s<seed>.json`, `large/dW/<run>/` |

**What decides the next job.** Job 1's checks gate everything. Gold's curve (job 2) gates every other T: a miss means the training code changed. `anchor2_learns_s0` gates every other anchor run. P's `chosen` gates S a1 (`format` means anchor 1 is a format fine-tune: tell the lead). `S_own.json` `train_kept_correct` < 4,000 stops own. After E own: if `E_own_s0.json` step 100 `task.drop` < 0.02 (not learned), tell the lead: the plan's one rerun is `T/E --arm own2` (rate 2e-5), the lead's call.

**Reporting.** After each job: job ID, exit code, wall time, peak GPU/host (last log line), and the stage's key numbers (control: KL, facts, accuracy; T: KL at steps 2, 5, 10 and 100, saves; E: `chosen` with matched KLs, learned flags, step-100 accuracy and length; P: gains and chosen; S: strict and truncated shares, kept). Any STOP or surprise goes to the lead before the next job. Commit `experiments/qwen_updates/results/*.json` and `run.log` as they appear (never `large/`).

Expected totals: ~16 `debug` jobs, ~5 h, $0, ~150-250 GB of δW on `commute-data`.
