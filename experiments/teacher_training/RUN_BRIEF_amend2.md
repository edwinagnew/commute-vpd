# Pod-runner brief: teacher_training Amendment 2 (scored)

Plan: `reports/2026-10-23_plan_teacher-training.md` → "Amendment 2". Predictions:
`experiments/teacher_training/results/predictions_amend2.json` (the run refuses to start without it, or if its `config`
or `amend2` differ from `run.py`'s `CFG` and `amend2.py`'s `AM`).

```
Experiment: teacher_training   Run: Amendment 2, scored (its own smoke: C1 and the first G1 run in the first ~10 min)
Command: (.venv/bin/python experiments/teacher_training/amend2.py cuda; echo "=== FINISHED (exit $?) — results in <repo>/experiments/teacher_training/results ===") 2>&1 | tee -a experiments/teacher_training/results/large/run_amend2.log /proc/1/fd/1
         (mkdir -p experiments/teacher_training/results/large first; run it as a background subprocess, not a blocking execute())
Expected: about 15-20 min of run (H100), + ~10 min set-up and downloads; GPU memory under 40 GB; CPU RAM ~8 GB
Outputs: results/teacher_training_amend2.json (updated after every stage)                 -> bring home
         results/large/emoticon/Bdropped01_{sgd,adam}_step1000.pt (~60 MB each, 2)        -> bring home
         results/large/run_amend2.log                                                     -> bring home
Inputs not in git (sync with include=[<directory>], then check sha256 both sides):
         experiments/teacher_training/                       (amend2.py, results/predictions_amend2.json: untracked;
                                                              results/teacher_training_amend1.json is tracked, a plain
                                                              sync carries it)
         experiments/transport_test/results/large            (alive.pt, windows.pt; 1.6 MB) and transport_test/results/windows.json
         NOT needed: any .pt under teacher_training/results/large (Amendment 1's references are read from its JSON),
         the C3 teacher, task_ladder windows, r0/r1 states.
         Pile train rows 32,768..40,959 and the spare rows 40,960..43,007 stream from Hugging Face in-job (two streams,
         ~3 min each; cached under COMMUTE_CACHE_DIR). Optional: sync artifacts/sets/train_32768_8192.npy into the
         pod's cache dir to skip the first stream (check sha256). The spare set is new (2,048 rows), so it streams.
Platform: RunPod H100 directly: the comparison is to Amendment 1's H100 readings, and three amendment pods run in
          parallel; pod commute-teacher_training-amend2-<id>; set up with `bash cluster/pod_setup.sh`
Pod: fresh; terminate when done
Budget: $5 cap for this run; expect about $2 (~30 min at about $4/h)
Watch for: see stages and stop rules below
```

## Stages, in order, with expected time (H100, from Amendment 1's log)

| Stage | What | Expected |
|---|---|---|
| Load + C1 | model, harness, W0 on Pile adversary seeds 0, 1, 2 | ~3 min |
| Rows | stream the 8,192 arm-B rows and 2,048 spare rows (`[data] train rows ...` each, ~3 min each); g of the trigger subcomponent on all 10,240 rows; `[trigger]` line | ~8 min |
| SGD teacher | fine-tune (~30 s), `[G1 sgd]`, training (~100 s), harness read (~55 s) | ~3.5 min |
| Adam teacher | the same, `[G1 adam]` | ~3.5 min |
| Score | `[score]` line, then `done` | — |

## Downloads

- After the `[emoticon sgd B-dropped-0.1@1000]` line: `teacher_training_amend2.json` and
  `results/large/emoticon/Bdropped01_sgd_step1000.pt` (into the matching parent; check the local layout after the
  first download).
- After `done`: the JSON again, `Bdropped01_adam_step1000.pt` and `run_amend2.log`, sha256 checked on both sides,
  then `terminate()`.

## Stop rules (the run halts itself with a `STOP:` line; report that line and the numbers named here, then terminate)

- **C1**: `STOP: C1 ...`. Report the `[C1]` line (pgd_rel, ci_rel, pgd_floor). Expected floor about 0.075 on an H100.
- **G1**: `STOP: G1 ...`. Report the `[G1 <opt>]` line and the `[emoticon teacher <opt> <lr>] reproduction` line. Pass
  needs the teacher's P(target) within 0.01 and size within 0.002 of Amendment 1's (SGD 0.9980 / 0.0122, Adam
  0.9996 / 0.0321).
- An `AssertionError` on `max_g_left` (a position above 0.1 left after replacement) or in `replace_rows` (`... clean
  spare rows`: fewer clean spare rows than rows to replace) is a code stop: report the `[trigger]` line and the
  traceback; do not retry.

A rerun continues the results file only on the same card with the same W0 floor; otherwise it exits naming both.
A code error is reported with its traceback, not retried.

## What to copy into the report

Every line starting `[C1]`, `[trigger]`, `[G1 <opt>]`, `[emoticon teacher <opt> <lr>] reproduction`,
`[emoticon <opt> B-dropped-0.1] trained in`, `[emoticon <opt> B-dropped-0.1] P(target)@100 ...`, the state line
`[emoticon <opt> B-dropped-0.1@1000] pgd ... verdict ...`, and `[score]`, plus the `[setup ...]` lines and the time
from `start()` to the first cell.

## Stall rule

Read the log every 1-2 min. If it has not grown for 3 minutes outside a known long step (the two Pile streams print
`[data] train rows ...` when done, ~3 min each; the g pass prints `[trigger]` when done, ~2 min), check `nvidia-smi`
and the process; if it is hung, stop, report, and terminate. Set-up deadline: if the first cell has not run 15 min
after `start()`, report and terminate. Never sleep blindly.
