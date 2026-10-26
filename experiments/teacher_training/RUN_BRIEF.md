# Pod-runner brief: teacher_training (scored)

Plan: `reports/2026-10-23_plan_teacher-training.md`. Predictions: `experiments/teacher_training/results/predictions.json`
(the run refuses to start without it, or if its `config` differs from `run.py`'s `Config`).

```
Experiment: teacher_training   Run: scored (the scored run is its own smoke: C1-C3 and the first A arm run in the first ~15 min)
Command: (.venv/bin/python experiments/teacher_training/run.py cuda; echo "=== FINISHED (exit $?) — results in <repo>/experiments/teacher_training/results ===") 2>&1 | tee -a experiments/teacher_training/results/large/run.log /proc/1/fd/1
         (mkdir -p experiments/teacher_training/results/large first; run it as a background subprocess, not a blocking execute())
Expected: about 80 min of run (plan's compute table, H100), + ~10 min set-up and downloads; GPU memory under 40 GB
          (arm B holds two 8 x 512 x vocab logit tensors and their softmaxes); CPU RAM ~8 GB
Outputs: results/teacher_training_controls.json (C1, C2, C3)              -> bring home
         results/teacher_training_{r1_t534_v369,r0_t352_v310,emoticon}.json -> bring home (updated after every state)
         results/large/r0_t352_v310/C3_student_step1000.pt (~60 MB)        -> bring home
         results/large/<task>/A_{sgd,adam}_step1000.pt (~60 MB each, 6)    -> bring home
         results/large/run.log                                             -> bring home
Inputs not in git (sync with include=[<directory>], then check sha256 both sides):
         experiments/transport_test/results/large            (alive.pt, windows.pt; 1.6 MB)
         experiments/task_ladder/results/large               (windows_r0_t352_v310.pt, windows_r1_t534_v369.pt; 1.7 MB)
         experiments/ladder_second_seed/results/large/r0_t352_v310/adam_1e-05_seed1_step1000.pt (C3's teacher, 60 MB; only
         this file, not its 231 MB directory. File names alone did not send on 2026-10-19: check it arrived by sha256)
         Pile val rows 0..255 and train rows 32,768..40,959 are streamed from Hugging Face in-job (cached under COMMUTE_CACHE_DIR)
Platform: RunPod H100 directly (over 1 h on one card); pod commute-teacher_training-<id>; set up with `bash cluster/pod_setup.sh`
Pod: fresh; terminate when done
Budget: $15 cap for this run; expect about $6-8 (1.5-2 h at about $4/h)
Watch for: see stages and stop rules below
```

## Stages, in order, with expected time (H100)

| Stage | What | Expected |
|---|---|---|
| Load + C1 | model, harness, W0 on Pile adversary seeds 0, 1, 2 | ~5 min |
| C2 | teacher = W0, 100 steps on r1 | ~15 s |
| C3 | E26's r0 state as teacher: its harness reading, 1000 steps, two scored states | ~4 min |
| A arms | per task r1, r0, emoticon; SGD teacher then Adam teacher: fine-tune (~30 s), do-nothing (~55 s), adjoint (~10 s), arm A 1000 steps + 2 states (~190 s) | ~5 min per teacher, ~28 min |
| B arms | Pile rows (first B arm streams 41k rows, a few min), 1000 steps + 2 states per teacher | ~5 min per teacher, ~31 min |
| S arms | SGD teachers only, 1000 steps + 2 states | ~3 min per teacher, ~10 min |

Each teacher is regenerated once per process (~30 s) and kept in memory for its B and S arms.

## Downloads

- After C3 passes: `teacher_training_controls.json` and `results/large/r0_t352_v310/`.
- After each task's A arms (log lines `[<task> adam A@1000]`): that task's `teacher_training_<task>.json` and
  `results/large/<task>/`. Download into the matching parent directory and check the local layout after the first.
- After the B arms and after the S arms: the three task JSONs again (they grow in place).
- Before `terminate()`: the three task JSONs, the controls JSON and `run.log`, sha256 checked on both sides.

## Stop rules (the run halts itself with a `STOP:` line; report that line and the numbers named here, then terminate)

- **C1**: `STOP: C1 ...`. Report the `[C1]` line (pgd_rel, ci_rel, pgd_floor). Expected floor about 0.075.
- **C2**: `STOP: C2 ...`. Report the step-1 loss and the largest offset from the line; the lead decides whether the two
  forward paths need unifying. Expected: both exactly 0.
- **C3**: `STOP: C3 ...`. Report the `[C3]` lines: gap at W0, hook-path floor, the teacher's and the student's
  `p_target` and `teacher_gap` at 100 and 1000, offset recovery. Pass needs gap <= 10% of W0's and P(target) within 0.05
  of the teacher's (about 0.92).
- **First A arm** (r1, SGD teacher): `STOP: first A arm ...`. Report the `[r1_t534_v369 sgd A@1000]` line and the
  `teacher gap at W0` line.

A rerun continues the result files only on the same card with the same W0 floor; otherwise it exits naming both
(move the files aside to start again). No stop rule fires after that; a code error is reported with its traceback, not retried.

## What to copy into the report

Every line starting `[C1]`, `[C2]`, `[C3]`, `[<task> teacher ...] reproduction`, and every state line
`[<task> <sgd|adam> <do-nothing|adjoint|A@s|B@s|S@s>] pgd ... verdict ...` (the seven numbers and the verdict), plus
the `[setup ...]` lines and the time from `start()` to the first cell.

## Stall rule

Read the log every 1-2 min. If the log has not grown for 3 minutes outside a known long step (the Pile streaming before
the first B arm prints `[data] train rows ...` when done), check `nvidia-smi` and the process; if it is hung, stop,
report, and terminate. Never sleep blindly.
