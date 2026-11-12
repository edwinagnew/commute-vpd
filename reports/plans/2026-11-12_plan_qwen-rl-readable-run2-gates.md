# Gates appendix: P3.4 run 2 (E167)

Appendix to `reports/plans/2026-11-12_plan_qwen-rl-readable-run2.md`. Gates 1-4 of `.claude/CLAUDE.md` "Gated runs" in
full; gate 5 (one fresh Opus review of plan and code together) and gate 6 (the user's go-ahead) are the lead's.
Code: `experiments/qwen_rl_readable/{run2.py,match.py,read2.py}`; rules: `results/predictions_run2.json`; jobs:
`cluster/qwen_rl_run2_{rehearsal,read,share}.sbatch`.

## Gate 1: pre-mortem

**Recipe and hyperparameters.** E69's loop unchanged (64 problems x 8 samples, window 512, AdamW betas (0.9, 0.999), no
decay, fp32 master weights), rate tag `3e-6w10` (`qwen_rl.RATES`, run 1's r3e-6 pod). New against run 1: steps 200 -> 400,
so the second half is untested at this rate. Guard: the drift stop at every GSM8K read (every 50 steps); R seed 0's
multiple band at steps 50 and 150; step-20 |δW| bands on every arm (a rate not applied).

**Data.** E69's split (seed 69), prompt order, chat template and window. 7345 training problems / 64 = 114 steps a pass,
so step 400 is the fourth pass, with new draws each pass (step seed 1000 x draw seed + t, run 1's `step_pids`). F: W0's
rollouts of R's problems and step seeds at every step (E69's `F.tar` for seed 0 steps 1-100, sampled in the loop at W0
otherwise), so F differs from R only in the reward. No leakage: `Job.step_pids` asserts no training problem is in H;
GSM8K test and the 128 maths rows are never trained on. Guard: step-1 rollouts' sha256 against E69's R of the same draw
seed (logged, as run 1: seed 0 `3b5fc36b...`, seed 70 `2c318920...`); step-1 reward rate in [0.6, 0.88] (stop).

**Precision.** fp32 with TF32 off throughout (`fp32_matmuls`), as E69 and run 1; δW saved in fp32 (bf16 would round a
1e-6-scale step away, `qwen_rl.delta_fp32`). Cluster reads in fp32 too. Guard: finite loss and gradient every step;
finite GSM8K pass and fits at every R GSM8K read (structural stops).

**Memory and host RAM per stage.** Run 1's peaks (`results/*.json` `peaks`): R and N 69.4 GB on the card, F 46.9 GB;
host 11.4 GB. F now samples inside the training context, as R does, with the sampler sized for that state (`rows_for`,
30 GB reserved), so F's card peak rises to about R's 69.4 GB, inside 80 GB; the rehearsal runs this path. Saving δW adds
5.64 GB of host memory per save (`delta_fp32` builds it on the host): ~17 GB host at a save. Cluster reads on an L40
(48 GB): fp32 model 6.8 GB, W0 + δW 5.6 GB, GSM8K cache at batch 128 x 512 tokens ~15 GB; host 11.4 + 5.6 GB (`--mem 32G`).

**Disk on the pod.** Per pod: 15 states x 5.64 GB = 84.6 GB, a checkpoint 17 GB plus its .tmp 17 GB while it is
rewritten, a δW .tmp 5.6 GB, the venv, HF cache and tree ~15 GB: ~139 GB of the 250 GB container disk
(`remote-kernels.toml` `container-disk-gb = 250`). Guard: `run2.py` refuses before training (exit 1) unless the disk
holds every save still to come, two checkpoints and 5 GB (`room`); `qwen_sft.save` raises rather than write a part file.

**Time and cost per stage.** From run 1's logs on H100 80GB HBM3 (`results/r3e-6.json` steps: sample 42.5 s, train 10.8 s;
`N.json` 40.7 + 14.6 s on an H200; `F.json` train 34.5 s; GSM8K read 388 s, output change 5.5 s, `r3e-6.json` reads);
save and sha256 (~40 s) and checkpoint (~60 s) are not measured: the rehearsal measures them on the L40 and each pod's
first save at step 50 measures them on its card.

| stage | time | $ |
|---|---|---|
| set-up per pod (venv, HF, W&B fetch of H, W0's answers, F.tar) | 40 min | 2.7 |
| first 20 steps, to the step-20 band (every pod) | ~22-27 min | ~1.6 |
| R_s0, R_s70 to 400 | 7.32 h train + reads, 8.19 h with set-up and tail | 32.7 each |
| N_s0, N_s70 | 6.58 h, 7.45 h | 29.7 each |
| F_s0 | 7.81 h, 8.68 h | 34.6 |
| F_s70 | 9.00 h, 9.86 h | 39.3 |
| pods | 49.8 pod-hours; wall ~9.9 h | ~199 |
| downloads (dev node, ~10 MB/s): 12 states x 5.64 GB = 67.7 GB / 10 MB/s = 6,765 s = 1.9 h (18 states 2.8 h), 9.4 min a state, overlapping the run | | in the 0.2 h tails |
| cluster reads (L40, debug): controls 2 + ~12 states x ~15 min (E104 9-20 min, to be measured by the controls job) | ~3.5 L40-h, side by side | 0 |
| outside share (CPU, 8 cores): E152 ~40 min for 15 updates; here ~19 files | ~1 h in 2-3 jobs | 0 |

Session budget: `REMOTE_KERNELS_BUDGET` $300; the pods' ~$199 leaves ~34% headroom, above the 15% rule, if this session
has spent nothing else on pods.

**What is saved, where, its size, and whether it outlives the pod.** On the pod only: 15 δW per pod (84.6 GB), the
checkpoint, GSM8K answers and F's rollouts. Downloaded: `results/run2/<pod>.json` and `.log` (small, after every read;
committed), and only the confirmed states, ~12 x 5.64 GB = 68 GB (18 at most, 102 GB) into `results/run2_dl/<pod>/` (a
project path, CLAUDE.md 2026-10-26), sha256 checked against the pod's `saves`, then moved after the pods are gone into
`results/large/run2/<pod>/dW/` (on `/mnt/nw`, 3.3 TB free) and listed in `experiments/SAVED.md`.

**Downloads.** A `CONFIRMED` log line names the step; the runner downloads that step from R, N and F of the seed at once,
one file at a time, sha256 on both sides. 12 states take ~1.9 h in all, spread over the run; the last (5x, if confirmed
near step 350) land about 10-30 min after their GSM8K read. Terminate a pod only when its final JSON's sha256 matches on
both sides and every confirmed state it holds is home.

**Pod loss and resume.** The same command resumes from the last checkpoint (at most 25 steps, ~27-34 min, lost); a crash in
a save or read replays only that. A lost pod's disk is gone: its arm restarts from step 1 on a new pod. Matching across
pods: steps are fixed by R's file alone and N and F save at every 25th step, so any step R picks exists on N and F. If an
R pod is lost, its restart defines that seed's steps (runs diverge at a fixed step, run 1), and states downloaded from the
lost R are stale (moved to `~/commute-data/stale/qwen-rl-lead/`); N and F of the seed run on. If N or F is lost, its restart
saves the same steps; cost only. A pod that loses its plugin connection: stop and tell the lead (CLAUDE.md lessons).

**Controls and references (same card, same rows).** All six pods on H100 80GB (an H200 is accepted: run 1's N ran on
one; the training trajectory may change with the card, the comparison does not, since every state is read again on one
card type). The cluster reads use one card type for every state and read W0 and E69's saved R step 100 first (bands
below). The output change uses the same 128 maths rows as run 1 and E152.

**Predictions (complete, not degenerate).** Each rule run against its edge cases and the do-nothing state (below):
W0 reads "not reached" for every size and "together"; E69's step-100 numbers read "together", so the rule is not passed by
a small gap when N's seed spread is wide.

**Code paths never run on a GPU.** `run2.py`: `save_state`, `room`, `note_chosen`, F's W0 sampling inside the training
context (`f_rollouts`), the pause and resume at a save step, `--chosen`. `read2.py`: every stage. `bands` at steps 20, 50
and 150 is never reached at the tiny size; it was run on a stub job on CPU (below). All of the others run in the rehearsal.

## Gate 2: every setting sourced
| setting | value | source |
|---|---|---|
| loop, batch, samples, window, Adam, fp32 | E69's | `experiments/qwen_rl/job.py` `Config`, unchanged (run 1) |
| peak rate, warm-up | 3e-6, 10 steps | user, decisions.md 2026-11-12 (run 1's r3e-6, `reports/analyses/2026-11-12_analysis_e163-qwen-rl-readable-run1.md`) |
| steps | 400 | user, same row |
| seeds | 0 (draw 69), 70 (draw 70) | cell brief; E69 Amendment 1's second seed |
| save steps | every 25 from 50 to 400 | lead's design: crossings are read every 5 steps, one save per 25 bounds the step error to 25 at 15 x 5.64 GB of disk |
| R reads | output change every 5, GSM8K every 50 | run 1 (every 5); the stop rule's reads every 50 (run 1 read at 50, 100, 150, 200) |
| checkpoint | at each save step | run 1 (checkpoint at each read step) |
| stop rule, fits bar 0.9215 | `pick.drift` | the brief; run 1's filed rule |
| targets | 2, 5, 10 x 0.0932 | the cell; E152 step0.json |
| step-20 |δW| bands | [0.5, 2] x 3 x E69's seed-0 step 20 per arm | run 1's r3e-6 rule |
| R seed 0 multiple bands | run 1's r3e-6 reads (1.331 at 50, 2.647 at 150) / and x 1.3419 | the run-to-run ratio at fixed step and norm, 0.06431 (E69's update, `kl_check.json`) / 0.04793 (run 1's control, `control.json` reads.100.kl) |
| F's data | fresh W0 draws at every step | Fable, 2026-11-12: F then differs from R only in the reward; reusing one draw per problem past step 114 would add multi-epoch memorisation as a second difference |
| cluster read controls | W0 kl 0, pass 0.7817 +-0.022; E69 R100 kl 0.064313 +-1%, pass 0.8807 +-0.022, |δW| 0.350884 +-0.1% | E69 A0 (H100); `kl_check.json` (L40, job 59301); run 1's control bands |
| share control | E69 R100 ratio 0.98047 +-1e-5 | E152 `step0.json` table1.updates.R_s0 (same file, same code) |
| worked floor | 2 x paired se | binomial noise of R against N on ~1140-1230 paired problems (se 0.0099, 0.0108 at E69 step 100) |

No departure from E69's recipe beyond the rate and steps the user chose.

## Gate 3: rehearsal for $0
`cluster/qwen_rl_run2_rehearsal.sbatch`, three debug jobs on L40s side by side, each under 30 min, after
`predictions_run2.json` is committed (the run refuses otherwise):
```
sbatch --comment=E167 cluster/qwen_rl_run2_rehearsal.sbatch R_s70   # pod stage, then --chosen, read controls, one state, share on 7 matrices, verdict
sbatch --comment=E167 cluster/qwen_rl_run2_rehearsal.sbatch N_s70
sbatch --comment=E167 cluster/qwen_rl_run2_rehearsal.sbatch F_s70
sbatch --comment=E167 cluster/qwen_rl_run2_read.sbatch --controls-only   # full size: the read stage's controls on an L40
```
Tiny size: E69's tiny config (4 problems x 2 samples a step, 8 GSM8K problems), 2 steps, saves at 1 and 2, the pause
after step 1's checkpoint, save and reads (exit 3) and a resume from that checkpoint (exit 0); the output change on the
full 128 rows; states on the node's `/ephemeral`, removed at the end. N's path is R's with other weights; F's in-loop W0
sampling is new, so F gets its own job.

What it must measure and write into this file before the review:
1. save + sha256 seconds for one 5.64 GB δW and the checkpoint's seconds (log `SAVED ... save_s`, `checkpoint ... s`), to
   replace the 100 s per save step assumed above (on the pod's card, the first save at step 50 measures it again);
2. that the resume works: step 1 recorded once, its save and read not repeated, step 2 runs (each pod's JSON `resumes`);
3. F's W0 sampling inside the training context: runs, logs its seconds, and its step-1 rollouts' reward check;
4. the disk line (`run2.py` prints free and needed) and `du` of a pod's folder;
5. host peak and card peak per pod (`peaks`), and the read stage's;
6. the read stage on an L40: controls inside their bands (E69 R100's output change at full rows, 0.0643 +-1%), the tiny
   state's sha256 check, and, from the full-size `--controls-only` job, **the seconds per state on an L40** that sizes the
   read jobs (one or two states each);
7. `--chosen` writes `chosen.json`; the share stage on 7 matrices and the verdict run end to end.
What the cluster cannot rehearse: H100 step times (run 1 measured them), the pod's disk (`room` refuses at once), the
W&B fetches (run 1's set-up). Each pod's first 20 steps (~25 min, ~$1.6) are its paid-card first stage, with a stop.

## Gate 4: early checks in code (every stop is exit 2 with "stopped" in the pod's JSON; the runner tells the lead)
| when | check | band | stop? |
|---|---|---|---|
| start | disk room for the saves to come + 2 checkpoints + 5 GB | ~124 GB at step 0 | refusal, exit 1 |
| step 1 | step-1 rollouts equal E69's R of the same draw seed | sha256 per seed | logged (as run 1) |
| step 1 | reward rate | [0.6, 0.88] | yes |
| first live step | |δW| / (rate sqrt(numel)) | [0.85, 1.05] | yes |
| first live step | KL(W0 ‖ W_1) on H, scaled (rate / 1e-6)² | step 1 at 3e-7: [2.7e-4, 0.018]; run 1 read 1.56e-3 | yes |
| every step | loss and gradient norm finite | | yes, at every size |
| step 20 | |δW| | R [0.2047, 0.8187], N [0.2001, 0.8003], F [0.1693, 0.6773] | yes |
| step 20 | reward change, steps 11-20 minus 1-10 | logged | no |
| first save (step 50) | file at E69's size 5,637,164,512 bytes, sha256 recorded | | yes, at every size |
| R seed 0, step 50 | multiple | [0.9920, 1.7862] | yes |
| R seed 0, step 150 | multiple | [1.9729, 3.5525] | yes |
| every R GSM8K read | pass and fits finite | | yes, at every size |
| every R GSM8K read | drift stop | `pick.drift` | ends R (exit 0, "drift") |
| cluster, first | W0: kl, pass, |δW|; E69 R100: kl, pass, |δW| | predictions "controls" | yes: the read stage refuses after |
| cluster, each state | file sha256 equals the pod's | | yes |

The multiple bands derive from one measured pair: run 1's control and E69's own update at step 100 differ by 1.3419 in
output change at the same seed, recipe, norm (0.352 vs 0.351) and step-1 rollouts. Seed 0 of run 2 is a rerun of run 1's
r3e-6 pod, so its step-50 and step-150 reads should sit within that factor of run 1's (1.331, 2.647). Seed 70 gets no band:
E69's two seeds differ by 1.52x at step 100. Risk: one pair is a thin base; a miss stops R_s0 only and goes to the lead,
while N_s0 and F_s0 run on.

## Edge cases run (2026-11-12, `~/commute-data/scratch/qwen-rl-lead/run2-writer/edge.py`, output `edge.out` beside it)
Matched steps (`match.chosen`, saves 50..400 by 25): expected shape (2x crossed at 125 and confirmed by the GSM8K read at
150; 5x pending while running, confirmed at 325 once done); a crossing on a save step keeps that step; a crossing before
the first save goes to step 50; 2x and 5x in one save interval share a step; crossed but not yet vouched: pending, not
kept; a dip after the crossing does not undo it; drift at 300 (last good 250): 2x at 250 confirmed, 5x crossed at 270
not reached; drift on the first GSM8K read: nothing kept, N and F may stop; drift at 400 with 5x crossed at 355 (save 375
past last good 350): not reached; stopped on a check at 50 after a crossing at 45: confirmed at 50 (a stop goes to the lead
before any download); a missing multiple skipped; exactly 2.0 counts; W0 running: open; W0 done: not reached; no R file:
open. Worked (`match.worked`): apart at 2x with 5x read on one seed only: apart; N's seed spread wide: together; gap
exactly the margin: together; N above R on both: "below", verdict together; one seed apart, the other not: together;
tiny spread but gap under 2 se: together; a missing se: unread, no call; a missing N state: no call; no confirmed step:
no call; 10x read on both seeds only: apart; W0 (R = N, se 0): together; E69's step-100 stand-in: together (margin 0.076).
Bands on a stub job: R's step-20 |δW| 0.3996 (run 1) holds, 0.136 (rate not applied) stops; N 0.40 holds; F 0.0 stops;
R_s0 multiple 1.331 at 50 holds, 0.95 stops, 3.6 at 150 stops, a missing multiple stops; seed 70 and N have no multiple band.

## Rehearsal results
(to be filled by the lead from the rehearsal jobs before the review)
