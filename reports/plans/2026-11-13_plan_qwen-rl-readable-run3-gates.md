# Gates appendix: P3.4 run 3 (E178)

Appendix to `reports/plans/2026-11-13_plan_qwen-rl-readable-run3.md`. Gates 1-4 of `.claude/CLAUDE.md` "Gated runs";
gate 5 (one fresh Opus review of plan and code together) and gate 6 (the user's go-ahead) are the lead's.
Code: `experiments/qwen_rl_readable/{run3.py,read3.py,match3.py}` (new; they call `run2.py`, `read2.py`, `match.py` and
run 1's `run.py` through `load_experiment`, and change none of them); rules: `results/predictions_run3.json`; jobs:
`cluster/qwen_rl_run3_{rehearsal,read}.sbatch`. Frozen callers touched: none (no file in `src/commute/` changed).

## Pod set-up and run lines (for the pod-runner brief)
```
bash cluster/pod_setup.sh
.venv/bin/python experiments/qwen_rl/fetch.py --root experiments/qwen_rl/results/large --only rollouts/H.jsonl
E120_PRED_SHA=<sha> .venv/bin/python experiments/qwen_rl_readable/run3.py --pod N_s0     # or N_s70; the same command resumes
```
`<sha>` = `git show HEAD:experiments/qwen_rl_readable/results/predictions_run3.json | sha256sum`. W0's GSM8K answers and
F's rollouts are not needed (no GSM8K read on the pod, no F). The log's last line names the end: `MATCHED at step t ...
download N_s<seed>/dW/step_<t>.safetensors`, `CAP ...` (same), or `COLLAPSED ...; no state saved`. Download the state
to `experiments/qwen_rl_readable/results/run3_dl/N_s<seed>/dW/step_<t>.safetensors` (gitignored project path), check its
sha256 against `saves` in `results/run3/N_s<seed>.json`, download that JSON and the log, then terminate.

## Gate 1: pre-mortem

**Recipe and hyperparameters.** E69's loop and recipe `1e-6w10`, unchanged, as run 1's null pod (which reproduced E69's
null token for token at steps 1-5). New: steps 101-500, never run at this rate for the null. Risks: the null collapses
later, as run 2's did at |δW| ~0.5 (this null reaches that near step 190 if it grows like run 1's RL at 1e-6); the
update stalls once no problem has a mix of right and wrong answers. Guards: the filed collapse rule after every step;
|δW| at step 20 and seed 0's multiple at step 100 in filed bands; the reward-change check at step 20.

**Data.** E69's split (seed 69), prompt order and template. 7345 training problems / 64 = 114 steps a pass, so step 500
is the fifth pass, with new draws each pass (step seed 1000 x draw seed + t, run 1's `step_pids`). Shuffles are seeded by
(draw seed, step, problem), so the null's shuffles at steps 1-100 equal E69's. No leakage: H, GSM8K test and the 128
maths rows are never trained on. Guard: step-1 rollouts' sha256 against E69's of the same seed, steps 1-5 reward and
length against E69's null (logged).

**Precision.** fp32, TF32 off, as E69, runs 1 and 2; δW saved in fp32. The match compares the pod's own read with R's pod
read on the same function and rows; run 2 measured R's pod and L40 reads of the same state equal to 1e-7.

**Memory and host RAM.** Run 2's null pods: 69.4 GB on the card, 11.6 GB host (`results/run2/N_s70.json` peaks); a δW
save adds 5.64 GB host. H100 80 GB only: the training state does not fit an L40's 48 GB. Cluster reads: run 2's (L40
48 GB; `--mem 32G`).

**Disk on the pod.** One δW 5.64 GB, a checkpoint 17 GB and its .tmp 17 GB, venv and caches ~15 GB: ~55 GB of 250.
Guard: run 2's `room` refuses before training under ~28 GB free.

**Time and cost per stage.** In the plan's table: ~62 s a step (sources there), set-up 40 min, one download 9.4 min at
10 MB/s. The training cap is 8.3 h of training and read time from the step records, so two capped pods cost $73.7.
Unknown until the pod: the H100 step time at this rate and length past step 100 (gate 3 below says where it is read).

**What is saved, where, its size, whether it outlives the pod.** Per pod: `results/run3/N_s<seed>.json` and `.log`
(written after every step and read; downloaded at the end, committed), and at most one δW (5.64 GB, matched or capped),
downloaded to `results/run3_dl/` before terminating. Checkpoints stay on the pod and are deleted at the end. R's inputs to
the verdict already on disk: `results/run2_reads/states/R_s*@75.json` (committed) and
`results/run2_dl/reads/acc/R_s*@75_gsm8k.jsonl` (gitignored; their sha256 is filed and `read3.py` refuses another).
Do not move `run2_dl/reads/acc/` before the verdict is written.

**Downloads.** One state per pod, when its end line is logged; the pod ends itself after the save (exit 0), so no
training runs on during it. Terminate once the JSON and the state match their sha256 on both sides.

**Pod loss and resume.** The same command resumes from the last checkpoint (at most 25 steps, ~26 min, lost). The hours
cap sums the step records, so a resume does not reset it. A lost pod's disk is gone: its seed restarts from step 1 on a
new pod (cost only; the match is by output change, so the restart's matched step may differ). A plugin disconnect:
stop and tell the lead.

**Controls and references (same card, same rows).** Both pods on H100 80 GB (an H200 is accepted, as run 1's null ran on
one: the trajectory may change, the match does not, since it is by output change). Every state is read on the
cluster's L40, the card run 2's R states were read on, with read2's controls first (W0, E69's R at step 100, run 2's
bands).

**Predictions (complete, not degenerate).** Edge cases below. The do-nothing state never reaches a target ("cap", "no
call"); taken as N it would read "apart", which is why the null is matched by output change, and why the collapse stop
exists (run 2's collapsed nulls also read "apart").

**Code paths never run on a GPU.** `run3.py`: all of it (the rehearsal runs every path except the step-20 and step-100
bands and the hours cap; those ran on a stub on CPU, below). `read3.py`: all of it (the rehearsal runs every stage).
`match3.py`: CPU only, run on its edge cases.

## Gate 2: every setting sourced
| setting | value | source |
|---|---|---|
| loop, batch, samples, window, Adam, fp32 | E69's | `experiments/qwen_rl/job.py` `Config`, unchanged |
| peak rate, warm-up | 1e-6, 10 steps (`1e-6w10`) | decision card option (a) (decisions.md 2026-11-13); E69's recipe; run 1's null pod |
| seeds | 0 (draw 69), 70 (draw 70) | run 2 (`run2.SEEDS`) |
| targets | 2.08818212, 3.056994076 | run 2's R pods' reads at step 75 (`results/run2/R_s*.json`), rounded to 9 places as every comparison is |
| read cadence | output change every 5 steps | run 1 and run 2's R pods (every 5); bounds the overshoot to 5 steps of growth |
| checkpoint | every 25 steps, 15 s each | run 2's save cadence; 15 s measured on run 2's null pods (`N_s70.log`) |
| cap | step 500, 8.3 h | decision card (500); the $75 budget (8.3 h) |
| collapse rule | reward < 0.5 or length < 100, 5 steps in a row | decision card option (a); just below the three 1e-6 nulls' minima (0.529, 124.9 tokens; verified with jq) |
| early bands | step-1 reward [0.6, 0.88], first-step ratio [0.85, 1.05], KL on H [0.003, 0.2] x (rate/1e-6)^2, reward change > -0.05 | E69's, as run 1's null pod |
| step-20 |δW| | [0.5x, 2x] of E69's null of the same seed | run 1's and run 2's rule, at E69's own rate (no x3) |
| seed-0 multiple at step 100 | [0.5825, 1.0489] | run 1's null 0.78167 ÷/x 1.3419, run 2's measured run-to-run ratio (`predictions_run2.json`) |
| steps 1-5 tolerance | |δW| within 0.1% | run 1's null against E69's at step 5: 1.4e-7 relative |
| read controls | W0, E69 R100 | `predictions_run2.json` controls, unchanged |
| verdict margin | max(seed spread, 2 x paired se) | run 2's filed rule (`match.size_call`) |
| null-held floor | 0.7597 | W0's control band low end (0.7817 − 2 binomial se) |

Departures from the cell: the null's rate (a third of RL's) and the match by output change; both are option (a), chosen
under the user's delegation (decisions.md 2026-11-13).

## Gate 3: rehearsal for $0
After `predictions_run3.json` is committed:
```
sbatch --comment=E178 cluster/qwen_rl_run3_rehearsal.sbatch                 # one debug L40 job, ~30 min
sbatch --comment=E178 cluster/qwen_rl_run3_read.sbatch --controls-only      # the real controls job at full size; resubmit once on exit 3
```
The rehearsal job runs, at E69's tiny size (4 problems x 2 samples a step, 8 GSM8K problems; the output change on all
128 rows): `N_s70 --size tiny_collapse` (a forced collapse rule ends it at step 2, exit 0, no save); `N_s0 --size tiny`
(pause after step 1's checkpoint, exit 3; resume, exit 0; its first read at step 2 meets a target of 0, saves δW, ends
"matched"); `read3.py --size tiny` controls, the state `N_s0@2`, and the verdict ("no call": seed 70 collapsed).
States go to `~/commute-data/scratch/qwen-rl-lead/run3_tiny_<job>` and are removed; JSON and logs to `results/run3_tiny/`.

What the runner measures and writes here (TODO):
1. The ends: `COLLAPSED at step 2` for N_s70, `MATCHED at step 2` for N_s0, each pod's `end` in its JSON. TODO
2. The resume: N_s0's `resumes` is [1], step 1 recorded once, step 2 run. TODO
3. Save and checkpoint seconds on NFS (log `SAVED ... save_s`, `checkpoint at step`), an upper bound for the pod. TODO
4. The disk line and `du` of the scratch folder. TODO
5. Card and host peaks (`peaks`). TODO
6. Read stage: controls logged (bands are not stops at the tiny size), the state's sha256 check passed, the verdict file
   written with `"verdict": "no call", "why": {"70": "collapsed"}`. TODO
7. From the full-size controls job: W0 and E69 R100 inside their bands, and the seconds per read on an L40 (run 2:
   918-960 s). TODO
Not rehearsable on the cluster: the H100 step time at full size, the pod disk, the W&B fetch. Each pod's first 20 steps
(~20 min, ~$1.4) are its paid first stage: the runner reports the mean step time over steps 1-20 and the projected
spend (0.67 h + min(500 x step, 8.3 h) + 0.26 h at $3.99/h) before letting it run on; over 75 s a step goes to the lead.

## Gate 4: early checks in code (filed values)
| when | check | filed value | on a miss |
|---|---|---|---|
| step 1 | rollouts' sha256 equal E69's of the seed | seed 0 `3b5fc36b...`, seed 70 `2c318920...` | logged |
| step 1 | reward rate | [0.6, 0.88] (E69: 0.799, 0.816) | stop |
| steps 1-5 | reward, length, |δW| as E69's null | seed 0: 0.799/190.2/0.00347 ... 0.8125/181.9/0.02492; seed 70: 0.816/192.1/0.00346 ... 0.805/180.3/0.02533 | logged |
| first live step (1) | |δW| / (rate x sqrt(numel)) | [0.85, 1.05] (run 1's null 0.924) | stop |
| first live step | KL(W0 ‖ W1) on H | [3e-5, 2e-3] (run 1's null 4.99e-5) | stop |
| step 20 | reward change, steps 11-20 minus 1-10 | above −0.05 (E69: +0.012, −0.010) | stop |
| step 20 | |δW| | seed 0 [0.06669, 0.26677], seed 70 [0.06693, 0.26773] (E69: 0.1334, 0.1339) | stop |
| step 100 | seed 0's multiple | [0.5825, 1.0489] (run 1's null 0.7817) | stop |
| every step | collapse rule | reward < 0.5 or length < 100, 5 in a row | the pod ends, "collapsed" |
| every step | loss and gradient finite | | stop |
| first save | E69's size 5637164512 bytes, sha256 recorded | | stop |

## Edge cases run (2026-11-13, `~/commute-data/scratch/qwen-rl-lead/run3-writer/edge.py`, output `edge.out` beside it)
- **Collapse rule** on saved nulls: run 2's N_s0 fires at 31, N_s70 at 26; E69's two nulls and run 1's never. Cases:
  empty → none; 4 bad then good → none; 5 bad on reward, on length, or mixed → 5; exactly 0.5 and 100 → none (not
  under); a missing value → counts as bad; a run broken at 4 then 5 more → 10.
- **Ending**: below the target → none; exactly at the target → matched; at a step with no read → none; seed 70 at seed
  0's target → none; a missing multiple → none; step 500 → cap; step 499 at 60 s a step → cap (the hours cap binds near
  step 485 at 60 s); collapse and target on one step → collapsed; the do-nothing state at 500 → cap; the two rehearsal
  sizes → matched and collapsed.
- **Verdict**: N 0.86/0.80 → together (margin 0.06); N 0.70/0.71 → apart, null not held; gap equal to the margin →
  together (a tie is not apart); N above R on both → "below", read together; one seed collapsed, capped, running or
  unread → no call with the reason; a missing standard error (an empty pool) → no call; W0's reads as N → apart (margin
  0.019), the case the match keeps out; E69's step-100 stand-in → together (margin 0.076); run 2's collapsed nulls → apart.
- **Bands and expected on a stub job**: step-20 |δW| 0.1332 holds, 0 misses (a stop at full size); seed 0's multiple at
  100 0.78 holds, 1.2 misses; seed 70 has no step-100 band; steps 1-5 equal to E69's hold, step 6 is not checked.
