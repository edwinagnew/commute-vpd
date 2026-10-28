# Pod-runner brief: E42, teacher training on a behaviour the model lacks (scored)

Plan: `reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md`. Predictions:
`experiments/teacher_training/results/predictions_e42.json` (filed 2026-10-28 11:57; the run refuses to start without
it, or if its `config` or `e42` differ from `run.py`'s `CFG` and `newskill.py`'s `AM`).

```
Experiment: teacher_training   Run: E42, scored (its own smoke: C1 in the first ~8 min, G1 a minute later)
Command:    job 1   sbatch cluster/teacher_e42.sbatch rand32        (from the clone root; record the job id)
            jobs 2, 3 after job 1 ends with C1 and G1 passed (whatever else it read), in parallel:
                    sbatch cluster/teacher_e42.sbatch ident32
                    sbatch cluster/teacher_e42.sbatch rand1
            Each runs  .venv/bin/python experiments/teacher_training/newskill.py <task>  and logs to
            cluster/logs/commute-teacher-e42-<jobid>.out
Expected:   about 30 min per job on an L40 (debug QoS, 1-h wall time); GPU memory under 40 GB; CPU RAM ~8 GB
Outputs:    experiments/teacher_training/results/teacher_training_e42_<task>.json   (updated after every state)
            experiments/teacher_training/results/large/e42/<task>/{A,B}_step1000.pt (bf16 offsets, ~60 MB each)
            cluster/logs/commute-teacher-e42-<jobid>.out
            On the cluster all of these land on NFS directly: nothing to download. Check the JSON ends with
            "done": true and a "score" block.
Platform:   MATS cluster, account c11, partition compute, QoS debug ($0). If a job has not started within 10 min,
            cancel it and run on a RunPod H100 instead (see "RunPod fallback").
Budget:     $0 on the cluster; RunPod fallback cap $5 for all three tasks
```

Optional, before job 1 (saves ~3 min of GPU time): cache arm B's Pile rows on NFS with a CPU job,
`~/commute-agents/fleet/cpu-run --mem 6G -t 00:30:00 -- .venv/bin/python -c "from commute.data import pile_rows; pile_rows('train', 32768, 8192)"`
(writes `artifacts/sets/train_32768_8192.npy`). If skipped, the first job streams them before its pile reading.

## Stages, in order, per job (L40 minutes, from E21's and E10's cluster logs)

| Stage | Log lines | Minutes |
|---|---|---|
| load, bases | `bases and input moments ...` | 1 |
| C1 | `[pile W0 seed 0/1/2] ~137 s` each, then `[C1] ...` | 7 |
| teacher | `[ft] checkpoint 10 / 300 / 1000`, `[<task> teacher] {...}` | 1 |
| G1 | `[<task> G1] {...}` | 0 |
| references | `[<task>] teacher gap at W0`, state lines `[<task> do-nothing] ...`, `[<task> least-squares] ...`, `[<task> least-squares] power ...` | 5 |
| Pile rows | a data-loading line (3 min if not cached) | 0-3 |
| pile reading | `[<task> pile reading] {...}` | 0.5 |
| arm A | `[component training ...] step ...`, `[<task> A] trained in ... s`, `[<task> A@100] ...`, `[<task> A@1000] ...` | 6 |
| arm B | as arm A with B | 8 |
| score | `[<task> score] {...}`, `done` | 0 |

## Stop rules (the run halts itself with a `STOP:` line; report that line and the numbers named, do not resubmit)

- **C1 (early stop for the control)**: `STOP: C1: W0 is not at its known values`. Report the `[C1]` line (`pgd_rel`,
  `ci_rel`, `pgd_floor`). Expected: `pgd_floor` about 0.21 on an L40. Do not submit jobs 2 and 3.
- **G1**: `STOP: G1 <task>: teacher P(target) at step 10 ...`. Report the `[<task> G1]` and `[<task> teacher]` lines.
  Pass needs step-10 P within 0.02 of 0.5847 (rand32), 0.9595 (ident32), 0.9982 (rand1). Do not submit further jobs.
- **The task gate is a skip, not a stop**: `[<task>] teacher P < 0.9: students skipped`. Report the line.
- A traceback without a `STOP:` line: report it with the last state line; do not retry.

A rerun of the same task continues its results file only on the same card with the same W0 floor; otherwise it exits
naming both. Finished records (do-nothing, least squares, pile reading, each arm) are skipped; the teacher is
regenerated (40 s). Move a file aside only on the lead's word.

## RunPod fallback (only if the cluster does not start a job within 10 min)

One H100 pod, `commute-teacher_training-e42-<id>`, set up with `bash cluster/pod_setup.sh`; all three tasks in one
process: `(.venv/bin/python experiments/teacher_training/newskill.py rand32 ident32 rand1; echo "=== FINISHED (exit $?) ===") 2>&1 | tee -a experiments/teacher_training/results/large/run_e42.log /proc/1/fd/1`
as a background subprocess. About 35 min of run (H100: C1 ~3 min, ~10 min per task) plus set-up; about $3.5.
Inputs not in git: `experiments/teacher_training/` (newskill.py, predictions_e42.json untracked), `src/commute/`
(kv_windows.py untracked), `experiments/transport_test/results/large/{alive,windows}.pt` with
`transport_test/results/windows.json`, and `experiments/kv_task/results/large/windows_{rand32,ident32,rand1}.pt`
(580 KB each). Every `results/large` is a symlink and arrives dangling: hard-link the needed files into a staging
path ending in `results/large/`, sync that directory, replace the symlink on the pod with the real directory, check
sha256 both sides. Download each task's JSON and `.pt` files to a project path (`experiments/teacher_training/results/e42_dl/`)
as soon as `[<task> B@1000]` is logged; terminate as soon as the last JSON's sha256 matches, then move files.

## What to copy into the report

Every line starting `[C1]`, `[<task> teacher]`, `[<task> G1]`, `[<task>] teacher gap at W0`, every state line
(`[<task> do-nothing]`, `[<task> least-squares]`, `[<task> A@100]`, `[<task> A@1000]`, `[<task> B@100]`,
`[<task> B@1000]`), `[<task> least-squares] power`, `[<task> pile reading]`, any `students skipped` line, each
`[<task> score]` block, the job ids, start and end times and the card (`torch.cuda.get_device_name` is in the JSON's
`card`).

## Stall rule

Read the log every 1-2 min. A Pile-harness read prints nothing for ~140 s on an L40, and the Pile row stream prints
only when done (~3 min). If the log has not grown for 4 minutes outside those, check `squeue` and the job's state;
if it is hung, `scancel`, report, and do not resubmit. Never sleep blindly.
