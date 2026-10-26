# Pod-runner brief: teacher_training Amendment 4 (scored)

Plan: `reports/2026-10-23_plan_teacher-training.md` → "Amendment 4: a second batch order, read under a longer
adversary". Predictions: `experiments/teacher_training/results/predictions_amend4.json` (the run refuses to start
without it, or if its `config` or `amend4` differ from `run.py`'s `CFG` and `amend4.py`'s `AM`).

```
Experiment: teacher_training   Run: Amendment 4, scored (its own smoke: C1, C1b and G1 run in the first ~12 min)
Command: mkdir -p experiments/teacher_training/results/large
         (.venv/bin/python experiments/teacher_training/amend4.py cuda; echo "=== FINISHED (exit $?) — results in <repo>/experiments/teacher_training/results ===") 2>&1 | tee -a experiments/teacher_training/results/large/run_amend4.log /proc/1/fd/1
         (run it as a background subprocess, not a blocking execute())
Expected: about 75-90 min of run (H100), + ~10 min set-up and downloads; GPU memory under 40 GB; CPU RAM ~8 GB
Outputs: results/teacher_training_amend4.json (updated after every state; out["score"] and "done" at the end) -> bring home
         results/large/<task>/{A,B}_{sgd,adam}_seed1_step1000.pt (~60 MB each, 12; tasks r1_t534_v369, r0_t352_v310, emoticon) -> bring home per task, as written
         results/large/run_amend4.log -> bring home
Inputs not in git (sync with include=[<directory>], then check sha256 both sides):
         experiments/teacher_training/                       (amend4.py and results/predictions_amend4.json: untracked)
         src/commute/                                        (the plain sync() after the clone carries the working tree)
         experiments/transport_test/results/large            (alive.pt, windows.pt; 1.6 MB) and transport_test/results/windows.json
         experiments/task_ladder/results/large/windows_{r0_t352_v310,r1_t534_v369}.pt
         NOT needed: any .pt under teacher_training/results/large (no saved state is read), the C3 teacher.
         Pile train rows 32,768..40,959 (arm B) and val rows 0..127 (the budget sweep) stream from Hugging Face in-job
         (~3 min and ~1 min; cached under COMMUTE_CACHE_DIR). Optional: sync artifacts/sets/train_32768_8192.npy and
         artifacts/sets/val_0_128.npy into the pod's cache dir to skip the streams (check sha256).
Platform: RunPod H100 directly (over 1 h; E34's card, so G1 and P2 compare like with like); pod
          commute-teacher_training-amend4-<id>; set up with `bash cluster/pod_setup.sh`
Pod: fresh; terminate when done
Budget: $10 cap for this run; expect about $6 (~90 min at about $4/h)
Watch for: see stages and stop rules below
```

## Stages, in order, with expected time (H100; E34's log for the harness steps, E31's L40 log for the budget steps)

| Stage | What | Expected |
|---|---|---|
| Load + C1 | model, harness, W0 on Pile adversary seeds 0, 1, 2; `[C1]` line | ~3 min |
| Val rows | stream Pile val rows 0..127 (`[data] val rows 0..128` when cached) | ~1 min |
| C1b | budget sweep at W0: seeds 0 and 1 × (20 + 160 steps); four `[budget W0 seed s]` lines, then `[C1b]` | ~4 min (estimate, see below) |
| Per teacher (×6) | fine-tune ~30 s → `reproduction` line (G1 after the first); do-nothing harness read ~55 s + budget ~1.5 min; A: train ~85 s + read ~55 s + budget ~1.5 min; B: train ~100 s + read ~55 s + budget ~1.5 min (the first B streams the Pile train rows first, ~3 min) | ~11 min each, ~66 min |
| Score | `[score] {...}` then `done` | seconds |

**The 160-step budget time is an estimate.** On E31's L40 one state's sweep took 33 s for 20 steps and a further
226 s for 160 steps (`[budget ...] (seconds)` is cumulative within a state there; here each budget line carries its
own seconds). An H100 is assumed to take about half. Copy the first `[budget W0 seed 0] 160 steps ...` seconds into
the report. If a 160-step run takes over 3 min, the run will take about 2 h: tell the lead at once and keep going
unless told otherwise (the $10 cap still holds; the run resumes per state from its results file on the same card).

## Downloads

- After the `[C1b]` line: `teacher_training_amend4.json`.
- After each task's last state line (`[<task> adam B@1000] ...` followed by its budget lines): the JSON again and
  `results/large/<task>/` (four `.pt` files; download into the matching parent and check the local layout after the
  first task, as the 2026-10-19 lesson: `download()` of a directory nested it).
- Before `terminate()`: the JSON (with `"done": true` and `score`), `run_amend4.log` and any `.pt` not yet home,
  sha256 checked on both sides.

## Stop rules (the run halts itself with a `STOP:` line; report that line and the numbers named here, then terminate)

- **C1**: `STOP: C1 ...`. Report the `[C1]` line (pgd_rel, ci_rel, pgd_floor). Expected floor about 0.075 on an H100.
- **G1**: `STOP: G1 ...`. Report the `[G1]` line and the `[r1_t534_v369 teacher sgd 0.03] reproduction` line. Pass
  needs P(target) within 0.01 of 0.983 and size within 0.002 of 0.0287 (E34's reading of the same recipe on an H100).
- A Python traceback is a code stop: report it with the last ten log lines; do not retry.

A rerun continues the results file only on the same card with the same W0 floor; otherwise it exits naming both.
Finished records (W0 budgets per seed, each teacher's do-nothing, each arm) are skipped; the teacher fine-tunes
(~30 s each) are regenerated.

## What to copy into the report

Every line starting `[C1]`, `[C1b]`, `[G1]`, `[budget ...]` (with its seconds and ratio), `[<task> teacher <opt> <lr>]
reproduction`, `[<task> <opt>] teacher gap at W0`, every state line `[<task> <opt> do-nothing] pgd ...` and
`[<task> <opt> <arm>@1000] pgd ... verdict ...`, the `P(target)@100 (free reading)` lines, the `[score]` line, plus the
`[setup ...]` lines and the time from `start()` to the first cell.

## Stall rule

Read the log every 1-2 min. The known long steps are the 160-step budget runs (~2 min each on an H100, up to ~4 min
on a slower card; the line prints when the budget finishes) and the two Pile streams (`[data] ... rows ...` when done).
Outside those, if the log has not grown for 3 minutes, check `nvidia-smi` and the process; if it is hung, stop,
report, and terminate. Never sleep blindly.
