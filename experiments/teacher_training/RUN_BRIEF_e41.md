# Pod-runner brief: teacher_training E41, teacher training along the fine-tune (scored)

Plan: `reports/plans/2026-10-28_plan_e41-teacher-path.md`. Code: `experiments/teacher_training/path.py` (imports
`run.py` beside it). Predictions: `experiments/teacher_training/results/predictions_e41.json`. The run refuses to start
without it, and also if its `config` or `e41` entries differ from `run.py`'s `CFG` or `path.py`'s `AM`.

```
Experiment: teacher_training   Run: E41, scored (its own smoke: C1, G1, the do-nothing reads and G3 come in its first
            ~10 min, C2 by ~17 min)
Command: mkdir -p experiments/teacher_training/results/large
         (.venv/bin/python experiments/teacher_training/path.py cuda; echo "=== FINISHED (exit $?) — results in <repo>/experiments/teacher_training/results ===") 2>&1 | tee -a experiments/teacher_training/results/large/run_e41.log /proc/1/fd/1
         (start it as a background subprocess; no execute() or wait() over 5 min, re-arm instead)
Expected: about 50 min of run on the H100 (E34 and Amendment 3 timings, see stages), + set-up and downloads;
          GPU memory under 40 GB; CPU RAM ~8 GB
Outputs: results/teacher_training_e41.json (rewritten after every reading; "score" and "done" at the end) -> bring home
         results/large/run_e41.log -> bring home
         No .pt files: the run saves no states.
Inputs not in git (sync with include=[<the gitignored directory itself>], then check sha256 on both sides):
         experiments/transport_test/results/large            (alive.pt 361 KB, windows.pt 156 KB)
         experiments/task_ladder/results/large/windows_r1_t534_v369.pt   (263 KB). Every results/large is a symlink
                  that arrives dangling: hard-link this one file into a staging path ending in
                  task_ladder/results/large/ and name that path, replace the symlink on the pod with the real
                  directory, then delete the staging directory on both sides
         artifacts/vpd: let pod_setup.sh fetch it from W&B on the pod (works, ~65 s, 2026-10-27); only if that
                  fails, stage it by hard link as RUN_BRIEF_amend5.md describes
         artifacts/sets/val_0_256.npy (1 MB; optional, otherwise it streams in ~1 min). Pile train rows
                  32,768..40,959 (arm B) stream in-job, ~3 min
         NOT needed: any .pt under teacher_training/results/large (no saved state is read)
Platform: RunPod "NVIDIA H100 80GB HBM3" directly (over 30 min; C1's floor 0.0748, G1, G3 and the noise floor are
          that card's; an L40 reads ~0.21 and stops at C1). Pod commute-teacher_training-e41-<id>, label set through
          start()'s argument; set up with `bash cluster/pod_setup.sh`
Pod: fresh; terminate when done. If another machine of the session runs: pass instance=... to every machine-scoped
     call, and never edit remote-kernels.toml while any machine runs
Budget: $7 cap for this run; expect about $5 (~65 min of pod at about $4-4.7/h)
```

## Stages, in order, with expected time (H100)

Per state, `run.read_state` prints `[<label>] p_target ... pgd ... verdict ...` (after the harness's own
`[<label>] gain ...` line). Then `teach` prints `[<label>] gap share ... start KL ... P(target)@100 ... (<s> s)`.

| Stage | What | Expected |
|---|---|---|
| Load + C1 | model, harness, W₀ on Pile adversary seeds 0, 1, 2 (`[pile W0 seed s]`); run.py's `[C1] <s> s; {...}`, then path.py's `[C1] {"pass": ..., "pgd_floor": ...}` | ~3 min |
| emoticon teachers | fine-tune (`[ft] checkpoint 10/100/1000`), `[emoticon T10]`, `[emoticon T100]`, `[emoticon T1000]` (P, damage, size, gap_w0), `[G1 emoticon]` | ~0.7 min |
| emoticon do-nothing | `[emoticon T10 do-nothing]`, `T100`, `T1000` (expected pgd 2.52, 1.14, 1.50) | ~3 min |
| emoticon arm A | `[emoticon A oneshot T1000]`, then `[G3 emoticon]`; `oneshot T10`, `oneshot T100`; `[C2] {...}`; `chain T100`, `chain T1000`; `[emoticon A] cosine ...` | ~12 min (each copy ~85 s training + ~55 s read) |
| emoticon arm B | the Pile train stream (`[data] train rows 32768..40960`, ~3 min, once); the same five copies and the cosine line | ~16 min (each copy ~100 s + ~55 s) |
| r1 | task context (~1 min, quiet), fine-tune, `[r1_t534_v369 T*]`, `[G1 r1_t534_v369]`, three do-nothing reads, arm A as emoticon's with `[G3 r1_t534_v369]` | ~16 min |
| Score | `[score] {...}` then `done` | seconds |

## Early stop for the control

C1 runs first and alone. If its `[C1] {"pass": false, ...}` line appears, the run halts with a `STOP:` line. Nothing
else runs, so report the line and terminate. The same goes for G1, G3 and C2, which stop the run themselves.

## Downloads (into the matching parent; check the local layout after the first, since download() of a directory nested it, 2026-10-19)

- After `[G3 emoticon]`: `teacher_training_e41.json`.
- After each `[<task> <arm>] cosine ...` line (a group finished): the JSON and `run_e41.log`.
- Before `terminate()`: the JSON (with `"done": true` and `score`) and `run_e41.log`. Check sha256 on both sides,
  then terminate. Terminate before any local file move.

## Stop rules (the run halts itself with a `STOP:` line; report that line and the numbers named here, then terminate)

- **C1**: `STOP: C1: W0 check <bool>, floor <f> (expected 0.0748 within 10%)`. Report both `[C1]` lines.
- **G1**: `STOP: G1 <task>: T1000 P <p> size <s>; E34 <p>, <s>`. Report the `[<task> T1000]` and `[G1 <task>]` lines.
- **G3**: `STOP: G3 <task>: one-shot T1000 copy P <p> ratio <r>; E34 <p>, <r>`. Report the copy's two lines and `[G3]`.
- **C2**: `STOP: C2: warm start KL <x> (max 0.005), delta-W moved <y> (max 0.001)`. Report `[C2]`.
- A results file written on another card or W₀ floor makes the run exit at once, naming both. Move the file aside
  only if the lead says so.
- A Python traceback is a code stop: report it with the last ten log lines. Do not retry.

A rerun on the same card keeps C1, the finished do-nothing reads and every (task, arm) group marked done. It redoes an
unfinished group from its first copy (states are in memory only; at most ~16 min) and regenerates the fine-tunes
(~30 s each).

## What to copy into the report

- every `[C1]`, `[pile W0 seed`, `[G1`, `[G3`, `[C2]` line;
- every `[<task> T<k>]` teacher line and every `do-nothing` read line;
- for every copy, the read line (p_target, pgd, stochastic, teacher_gap, damage, size, delta_share, verdict) and the
  `gap share ... start KL ... P(target)@100` line;
- every cosine line; the `[score]` line; any `STOP:` line;
- the `[setup ...]` lines, the VPD fetch's time and rate, and the time from `start()` to the first cell;
- the card `nvidia-smi` names.

## Stall rule

Read the log every 1-2 min. The known quiet steps are the Pile train stream, the r1 task context, and each copy's
training. Training logs at component-training steps 1-30 and every 500, so a copy can be quiet for ~40 s. Outside
those, if the log has not grown for 3 minutes, check `nvidia-smi` and the process. If it is hung, stop, report and
terminate. Never sleep blindly. If the plugin loses the pod (connection refused, port moved), stop and tell the lead at
once. On a WebSocket closure, run `status(instance=...)`, then `create_kernel(instance=...)` at once.
