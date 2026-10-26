# Pod-runner brief: teacher_training Amendment 1 (scored)

Plan: `reports/2026-10-23_plan_teacher-training.md` → "Amendment 1". Predictions:
`experiments/teacher_training/results/predictions_amend1.json` (the run refuses to start without it, or if its `config`
or `amend1` differ from `run.py`'s `CFG` and `AM`).

```
Experiment: teacher_training   Run: Amendment 1, scored (its own smoke: C1 and G1 run in the first ~10 min)
Command: (.venv/bin/python experiments/teacher_training/run.py cuda --amend1; echo "=== FINISHED (exit $?) — results in <repo>/experiments/teacher_training/results ===") 2>&1 | tee -a experiments/teacher_training/results/large/run_amend1.log /proc/1/fd/1
         (mkdir -p experiments/teacher_training/results/large first; run it as a background subprocess, not a blocking execute())
Expected: about 35 min of run (H100), + ~10 min set-up and downloads; GPU memory under 40 GB; CPU RAM ~8 GB
Outputs: results/teacher_training_amend1.json (updated after every state)                -> bring home
         results/large/emoticon/{B,Bmasked,Bdropped}_{sgd,adam}_step1000.pt (~60 MB each, 6) -> bring home
         results/large/run_amend1.log                                                    -> bring home
Inputs not in git (sync with include=[<directory>], then check sha256 both sides):
         experiments/teacher_training/                       (run.py, results/predictions_amend1.json: untracked)
         src/commute/{teacher,data,finetune,component_training,harness}.py (the plain sync() after the clone carries the whole working tree)              (teacher.py untracked; data.py, finetune.py changed)
         experiments/transport_test/results/large            (alive.pt, windows.pt; 1.6 MB) and transport_test/results/windows.json
         experiments/teacher_training/results/large/emoticon/A_{sgd,adam}_step1000.pt (E34's arm-A students, 60 MB each;
         only these two files. File names alone did not send on 2026-10-19: check they arrived by sha256)
         NOT needed: the 60 MB C3 teacher (ladder_second_seed/...), task_ladder windows, r0/r1 states.
         Pile train rows 32,768..40,959 and the spare rows 40,960..41,983 stream from Hugging Face in-job (two streams,
         ~2-3 min each; cached under COMMUTE_CACHE_DIR). Optional: sync artifacts/sets/train_32768_8192.npy into the
         pod's cache dir to skip the first stream (check sha256).
Platform: RunPod H100 (E34's card: G1 then compares like with like); pod commute-teacher_training-amend1-<id>;
          set up with `bash cluster/pod_setup.sh`. A 4090 or L40 is acceptable if no H100 is free (G1 is read relative).
Pod: fresh; terminate when done
Budget: $5 cap for this run; expect about $3 (~45 min at about $4/h)
Watch for: see stages and stop rules below
```

## Stages, in order, with expected time (H100, from E34's log)

| Stage | What | Expected |
|---|---|---|
| Load + C1 | model, harness, W0 on Pile adversary seeds 0, 1, 2 | ~193 s |
| Rows | stream Pile train rows and 1,024 spare rows; g of the trigger subcomponent on all 9,216 rows; `[trigger]` line | ~5-7 min |
| SGD teacher | fine-tune (~30 s), cosine references (~15 s), then B (= G1), B-masked, B-dropped: each ~100 s training + ~55 s harness | ~8.5 min |
| G1 | after the SGD teacher's B: `[G1]` line | — |
| Adam teacher | the same | ~8.5 min |

## Downloads

- After the `[G1]` line: `teacher_training_amend1.json` and `results/large/emoticon/` (into the matching parent;
  check the local layout after the first download).
- After each later `[emoticon <opt> <variant>@1000]` line: the JSON again, and the new `.pt`.
- Before `terminate()`: the JSON, `run_amend1.log` and the six `.pt` files, sha256 checked on both sides.

## Stop rules (the run halts itself with a `STOP:` line; report that line and the numbers named here, then terminate)

- **C1**: `STOP: C1 ...`. Report the `[C1]` line (pgd_rel, ci_rel, pgd_floor). Expected floor about 0.075 on an H100.
- **G1**: `STOP: G1 ...`. Report the `[G1]` line and the `[emoticon sgd B@1000]` line. Pass needs the adversarial
  ratio within 25% of E34's 0.8954 and P(target) within 0.05 of 0.9603.
- An `AssertionError` on `max_g_dropped` or on spare rows (`... clean spare rows`) is a code stop: report the
  `[trigger]` line and the traceback; do not retry.

A rerun continues the results file only on the same card with the same W0 floor; otherwise it exits naming both.
A code error is reported with its traceback, not retried.

## What to copy into the report

Every line starting `[C1]`, `[trigger]`, `[G1]`, `[emoticon teacher <opt> <lr>] reproduction`, `[emoticon <opt>] cosine
with delta-W`, `[emoticon <opt> <variant>] P(target)@100 ...`, and every state line `[emoticon <opt> <variant>@1000] pgd
... verdict ...`, plus the `[setup ...]` lines and the time from `start()` to the first cell.

## Stall rule

Read the log every 1-2 min. If it has not grown for 3 minutes outside a known long step (the two Pile streams print
`[data] train rows ...` when done; the g pass prints `[trigger]` when done, ~1-2 min), check `nvidia-smi` and the
process; if it is hung, stop, report, and terminate. Never sleep blindly.
