# Pod-runner brief: teacher_training Amendment 3, a LoRA teacher (scored)

Plan: `reports/plans/2026-10-23_plan_teacher-training.md` → "Amendment 3: a LoRA teacher". Predictions:
`experiments/teacher_training/results/predictions_amend3.json` (the run refuses to start without it, or if its `config`
or `amend3` differ from `run.py`'s `CFG` and `amend3.py`'s `AM`).

```
Experiment: teacher_training   Run: Amendment 3, scored (its own smoke: C1 and G1 run in the first ~8 min)
Command: mkdir -p experiments/teacher_training/results/large
         (.venv/bin/python experiments/teacher_training/amend3.py cuda; echo "=== FINISHED (exit $?) — results in <repo>/experiments/teacher_training/results ===") 2>&1 | tee -a experiments/teacher_training/results/large/run_amend3.log /proc/1/fd/1
         (run it as a background subprocess, not a blocking execute())
Expected: about 35-40 min of run (H100), + ~10 min set-up and downloads; GPU memory under 40 GB; CPU RAM ~8 GB
Outputs: results/teacher_training_amend3.json (one file for all tasks, updated after every state)        -> bring home
         results/large/<task>/lora_{A,B}_step1000.pt (bf16 offsets, ~60 MB each, 6 at most)                -> bring home
         results/large/<task>/lora_teacher_factors.pt (the LoRA's (B, A) at step 1000, a few MB, 3)        -> bring home
         results/large/run_amend3.log                                                                       -> bring home
Inputs not in git (sync with include=[<directory>], then check sha256 both sides):
         experiments/teacher_training/                     (amend3.py and results/predictions_amend3.json: untracked)
         src/commute/                                      (the plain sync() after the clone carries the working tree)
         experiments/transport_test/results/large          (alive.pt 361 KB, windows.pt 156 KB) and transport_test/results/windows.json
         experiments/task_ladder/results/large/windows_{r0_t352_v310,r1_t534_v369}.pt (263 KB each; run.py's LADDER
         path, read by Job.task for the ladder tasks. The directory holds other large files: hard-link these two into a
         staging path ending in results/large/ if the whole directory is too big to send)
         NOT needed: any .pt under teacher_training/results/large (E34's students, the C3 teacher), r0/r1 states.
         Pile train rows 32,768..40,959 stream from Hugging Face in-job before the first arm B (~3 min; cached under
         COMMUTE_CACHE_DIR). Optional: sync artifacts/sets/train_32768_8192.npy into the pod's cache dir to skip it.
Platform: RunPod H100 directly (over 30 min; E34's card, so the W0 floor and verdict bands compare like with like);
          pod commute-teacher_training-amend3-<id>; set up with `bash cluster/pod_setup.sh`
Pod: fresh; terminate when done
Budget: $6 cap for this run; expect about $3 (~45-50 min at about $4/h)
Watch for: see stages and stop rules below
```

## Stages, in order, with expected time (H100, from E34's and E27's logs)

| Stage | What | Expected |
|---|---|---|
| Load + C1 | model, harness, W0 on Pile adversary seeds 0, 1, 2; `[C1]` line | ~3 min |
| emoticon teacher | LoRA fine-tune (`[lora r8] checkpoint 100/1000` lines, ~1 min), `[emoticon LoRA teacher]`, gap at W0, do-nothing read (~1 min, Pile), adjoint read (~0.5 min, no Pile), `[emoticon LoRA] cosine with delta-W` | ~3 min |
| G1 | `[G1]` line right after the adjoint read, before any student | — |
| emoticon arm A | ~85 s training + two harness reads (~55 s each) | ~3.5 min |
| emoticon arm B | Pile stream (~3 min, once), ~100 s training + two reads | ~6.5 min |
| r1, r0 | each: windows' input moments (~1 min, `Job.task`), then as emoticon without G1 | ~9-10 min each |
| score | `score {...}` and `done` | — |

About 35-40 min in all. If a task's LoRA teacher reads P(target) < 0.9 the log says `[<task> LoRA] teacher P <0.9
(...): students skipped` and the task costs ~3 min instead of ~10.

## Downloads

- After the `[G1]` line: `teacher_training_amend3.json` and `results/large/emoticon/lora_teacher_factors.pt`.
- After each `[<task> LoRA <arm>@1000]` line: the JSON again and the new `.pt` (`results/large/<task>/lora_<arm>_step1000.pt`,
  into the matching parent; check the local layout after the first download).
- Before `terminate()`: the JSON (with `"done": true` and `score`), `run_amend3.log` and every `.pt` written, sha256
  checked on both sides.

## Stop rules (the run halts itself with a `STOP:` line; report that line and the numbers named here, then terminate)

- **C1**: `STOP: C1 ...`. Report the `[C1]` line (pgd_rel, ci_rel, pgd_floor). Expected floor about 0.075 on an H100.
- **G1**: `STOP: G1 ...`. Report the `[G1]` line, the `[emoticon LoRA teacher]` line and the `[emoticon LoRA adjoint]`
  line. Pass needs P(target) within 0.01 of 0.9999, KL elsewhere within 0.005 of 0.0048, size within 0.005 of 0.0900,
  and the adjoint state's P(target) within 0.05 of 0.063.
- **The task gate is a skip, not a stop**: a teacher below P 0.9 is reported and the run moves on. Report the line.
- A code error (traceback, no `STOP:` line) is reported with its traceback and the last state line, not retried.

A rerun continues the results file only on the same card with the same W0 floor; otherwise it exits naming both.
Finished sub-records (do-nothing, adjoint, each arm at step 1000) are skipped; the LoRA teacher itself is regenerated
(deterministic, ~1 min).

## What to copy into the report

Every line starting `[C1]`, `[G1]`, `[<task> LoRA teacher]`, `[<task> LoRA] teacher gap at W0`, `[<task> LoRA] cosine
with delta-W`, `[<task> LoRA <arm>@1000] cosine with delta-W`, every state line `[<task> LoRA <state>] pgd ... verdict
...` (do-nothing, adjoint, A@100, A@1000, B@100, B@1000), any `students skipped` line, the final `score {...}` block,
plus the `[setup ...]` lines and the time from `start()` to the first cell.

## Stall rule

Read the log every 1-2 min. If it has not grown for 3 minutes outside a known long step (the Pile stream prints
`[data] train rows ...` when done; a harness read prints its state line after ~55 s; the ladder tasks' input moments
take ~1 min with no output), check `nvidia-smi` and the process; if it is hung, stop, report, and terminate. Never
sleep blindly.
