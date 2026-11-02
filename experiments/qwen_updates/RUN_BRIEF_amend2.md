# Run brief: E45 Amendment 2 — format anchor 1, matched decisive point, step-2 check ($0, MATS cluster `debug`, one L40 per job)

Plan: `reports/plans/2026-10-29_plan_e45-amendment2.md` (approved 2026-10-30). Rules and first-minute values:
`results/predictions_amend2.json`; stage P and every T and E of `own`, `own2`, `null` and `a1` read it and refuse to
start without it. Already done and not rerun: control, gold, `a2long` (both seeds), S own, check_format. `a2_s1` is
not run (Amendment 1 replaced it). E45 Job 2 is dropped.

**Before the first job**, as in `RUN_BRIEF.md`: `git pull` in the lead's clone; `ls -l experiments/qwen_updates/results/large`
is the link into `~/commute-data/results-large/qwen_updates`; `results/control.json`, `results/P.json`,
`results/S_own.json` and `large/P/*.jsonl` are there. Submit from the clone root, one job at a time, every job with
`sbatch cluster/qwen_updates_amend2.sbatch <args>` (48G, 30 min, debug). Record job IDs.

**Exit codes.** 0 done. **2 = STOP**: a first-minute check missed (`[check] ... MISSED`): report to the lead before any
further job. **3 = resubmit the same command** (P, S and E resume). Any other exit = STOP, report. A T started without
a checkpoint is a fresh start: it deletes that run's old `T_`/`E_` files and `large/dW/<run>/`, `large/E/<run>/`
(for `own_s0` and `null_s0` this is intended: their step-2 stops are replaced). Never resubmit a finished T.

| # | args | L40 time | first values to see in `cluster/logs/commute-qwen_updates_amend2-<id>.out` | stop rule (code) | output |
|---|---|---|---|---|---|
| 1 | `--stage P` | ~6 min | within a minute, before any generation: **`[check] P: the amended format rule reproduces 474/512 and 0/512: ok`** (`with` 0.92578125, `without` 0, `with_filed_regex` 0.033203125); `[P] rules ...; chosen format`; then the greedy answers to test 256-511; exit 0 | not exactly 474/512 and 0/512 → exit 2 | `results/P.json` (`format`, recomputed `runs.*.format_follow`), `large/P/a1_rows.json` |
| 2 | `--stage T --arm own --seed 0` | ~15 min | `own samples kept` n 5939; step 1 at rate 0; step 2: band ok (stopped run 0.633-0.819), **`step 2 KL on the maths rows reproduces the stopped run: ok`** (ref 0.0056601, tol 1%), `step 2 KL on the maths rows: ok` (max 0.010679); `[save]` lines at the grid brackets and where the maths tracker first passes 0.618 (and the read before), and 100 | reproduction off by > 1% → exit 2 (the control: code or data changed since the stop; nothing else runs) | `results/T_own_s0.json` (`grid` shows maths with 0.618...), `large/dW/own_s0/` |
| 3 | `--stage E --arm own --seed 0`, resubmit on exit 3 | ~20 min (+1) | one `state` line per saved state; `generation at` each chosen step and 100 (~7 min at 100, full test) | — | `results/E_own_s0.json`, `score.json` `amend2_decisive.own_s0` |
| 4 | rows 2-3 with `--arm null` | 15 + 20 min | as rows 2-3; reproduction ref 0.0066907 | as row 2 | `T_/E_null_s0.json` |
| 5 | `--stage S --set a1`, resubmit until exit 0 | 3 jobs, ~65 min | first `[S a1] batch 1/N ...: projected ... s in all; peak GPU` (≤ 44 GB) | out of memory → report (`RUN_BRIEF.md` row 6) | `large/S/a1/`, `results/S_a1.json` (`complete` true) |
| 6 | `--stage T --arm a1 --seed 0`, then `--stage E --arm a1 --seed 0` | 15 + 20 min | T step 2: band ok, KL ≤ 0.010679 (**expected between gold's 0.0037 and the null's 0.0067**); saves on the maths axis at 0.618 as row 2 | code: step-1, step-2 checks. **Manual stop**: T a1 s0's step-2 KL (`checks["step 2 KL on the maths rows"].kl` in `T_a1_s0.json`) outside 0.0037-0.0067 → report to the lead before row 7 | `T_/E_a1_s0.json` |
| 7 | row 6 with `--seed 1` | 15 + 20 min | as row 6 | step-1, step-2 checks | `T_/E_a1_s1.json` |
| 8 | `--stage score` (every job also ends with `score`, so only if the last E did not); read `results/score.json` → `amend2_decisive` | ~1 min | per run: `read` {step, kl, log_dist, matched}, `nearest_unlearned`, `dw_norm`, `filed_point`. Expected: gold_s0 step 100 (log 0.1194), a2long_s0/_s1 step 100 (0.0173, 0.0176), matched | — | `results/score.json` |

**What decides the next job.** Row 1's check gates everything (the format rule is the filed one). Row 2's
reproduction is the control: a miss means nothing else runs. After E own: if `E_own_s0.json` step 100 `task.drop` <
0.02 (not learned), tell the lead: the approved branch is `T/E --arm own2` (rate 2e-5, step-2 limit 0.0427, no
reproduction check). Row 6's manual stop gates row 7. An `amend2_decisive.a1_s<seed>.read.matched` false (anchor 1's
maths KL never in 0.412-0.927 by step 100) → report: the lead sends a card to the user (plan appendix C).

**Reporting.** After each job: job ID, exit code, wall time, peak GPU/host (last log line), and the stage's key
numbers (P: the check line and `chosen`; T: KL at steps 2, 5, 10 and 100, the step-2 checks, saves, `du -sh
large/dW/<run>`; E: `chosen` with matched KLs, `learned` flags, step-100 accuracy and length, a1's
`format_unprompted`; score: `amend2_decisive.<run>` whole). Commit `experiments/qwen_updates/results/*.json` and
`run.log` as they appear (never `large/`).

**Disk.** About 8-10 δW states × 2.82 GB per run, four runs: ~100-120 GB on `commute-data` (3.9 TB free).

Expected totals: ~11-13 `debug` jobs, ~4 h, $0.
