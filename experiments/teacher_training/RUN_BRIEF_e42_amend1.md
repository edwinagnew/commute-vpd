# Pod-runner brief: E42 Amendment 1, teacher training with the masked students matched to the teacher (scored)

Plan: `reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md` → "Amendment 1". Predictions:
`experiments/teacher_training/results/predictions_e42_amend1.json` (filed 2026-10-28 13:13:54, sha256 `d01c266d…`).
The run refuses to start without it, or if its `config`, `e42` or `amend1` differ from `run.py`'s `CFG`, `newskill.py`'s
`AM` and `newskill_amend1.py`'s `AM`.

```
Experiment: teacher_training   Run: E42 Amendment 1, scored (its own smoke: C1 in the first ~8 min, G2 seconds later,
            M0's training curve from minute ~8 and its verdict at ~minute 14)
Command:    job 1   sbatch cluster/teacher_e42_amend1.sbatch rand32      (from the clone root; record the job id)
            job 2   after job 1 has passed C1, G2, M0 and G1 (whatever arm M then reads):
                    sbatch cluster/teacher_e42_amend1.sbatch rand1
            Each runs  .venv/bin/python experiments/teacher_training/newskill_amend1.py <task>  and logs to
            cluster/logs/commute-teacher-e42a1-<jobid>.out
Expected:   about 21 min per job on an L40 (debug QoS, 45-min wall time); GPU memory under 30 GB; CPU RAM under 16 GB
Outputs:    experiments/teacher_training/results/teacher_training_e42_amend1_<task>.json   (updated after every stage)
            experiments/teacher_training/results/large/e42_amend1/<task>/M_step1000.pt     (bf16 offsets, ~60 MB)
            cluster/logs/commute-teacher-e42a1-<jobid>.out
            On the cluster these land on NFS directly: nothing to download. Check the JSON ends with "done": true and a
            "score" block.
Platform:   MATS cluster, account c11, partition compute, QoS debug ($0). If a job has not started within 10 min,
            cancel it and use RunPod (see "RunPod fallback").
Budget:     $0 on the cluster; RunPod fallback cap $3 for both tasks
```

## Stages, in order, per job (L40 minutes)

| Stage | Log lines | Minutes |
|---|---|---|
| load, bases | `bases and input moments ...` | 1 |
| C1 | `[pile W0 seed 0/1/2] ...` each ~140 s, then `[C1] ...` | 7 |
| G2 | `[G2] {"ci_masked_w0": ..., "path_kl": ..., "pass": ...}` | 0.1 |
| M0 training | `[masked copy coeff 0.5] step 1 / 2 / 5 / 10 / 50 / 100 / 200 ...`, then `[<task> M0 train] {...}` | 3-4 |
| M0 read | `[<task> M0@1000] gain ...` (route line), a silent Pile harness read, then the record line | 2.3 |
| teacher, G1 | `[ft] checkpoint 10 / 300 / 1000`, `[<task> teacher] {...}`, `[<task> G1] {...}` | 0.7 |
| references | `[<task> W0 trigger] {...}; E42's floor ..., this job's ...`, `[<task> E42 copy] ...` (two lines) | 0.2 |
| arm M | `[masked copy coeff 0.5] step ...`, `[<task> M train] {...}`, `[<task> M@100] ...`, `[<task> M@1000] ...` | 6.5 |
| score | `[<task> score] {...}`, `done` | 0 |

The first masked-copy line should show `unmasked` about 0 in M0 (the teacher is W₀; G2's `path_kl`) and about 10-13 in
arm M at step 1 (E42's teacher gap at W₀: 13.2 rand32, 10.5 rand1). If M0's training takes more than 8 minutes, say so
in the report. It is not a stop.

## Stop rules (the run halts itself with a `STOP:` line; report that line and the numbers named, do not resubmit)

- **C1 (early stop for the control)**: `STOP: C1: W0 is not at its known values`. Report the `[C1]` line. Expected
  `pgd_floor` 0.2088 on an L40. Do not submit job 2.
- **G2**: `STOP: G2: the trainer's unmasked student at W0 is ... nats from the target model`. Report the `[G2]` line
  (pass needs `path_kl` ≤ 0.01; expected about 1e-3). Do not submit job 2.
- **M0**: `STOP: M0: with nothing to copy the masked recipe fails or breaks the Pile check ...`. Report the
  `[<task> M0 train]` line and both `[<task> M0@1000]` lines. Do not submit job 2. This stop is a result the lead
  needs at once (E37 Amendment 1's version of this control failed), so tell the lead in the hand-back's first line.
- **G1**: `STOP: G1 <task>: teacher P(target) at step 10 ...`. Report the `[<task> G1]` and `[<task> teacher]` lines.
  Do not submit job 2.
- A traceback without a `STOP:` line: report it with the last state line, and do not retry.

A rerun of the same task continues its results file only on the same card with the same W₀ floor; otherwise it exits
naming both. Finished stages (G2 passed, M0, references, arm M) are skipped, and the teacher is regenerated (40 s).
Move a file aside only on the lead's word.

## RunPod fallback (only if the cluster does not start a job within 10 min)

One H100 pod, `commute-teacher_training-e42a1-<id>`, set up with `bash cluster/pod_setup.sh`. Both tasks run in
sequence: `(for t in rand32 rand1; do .venv/bin/python experiments/teacher_training/newskill_amend1.py $t || break; done; echo "=== FINISHED (exit $?) ===") 2>&1 | tee -a experiments/teacher_training/results/large/run_e42_amend1.log /proc/1/fd/1`
as a background subprocess. That is about 20 min of run plus set-up, about $2.

Inputs not in git, all needed on the pod:
- `src/commute/masked_copy.py` and `experiments/teacher_training/newskill_amend1.py` (untracked);
  `results/predictions_e42_amend1.json`.
- `newskill.py`, `src/commute/kv_windows.py` and E42's `results/teacher_training_e42_{rand32,rand1}.json`. These may
  be committed by then; check with `git status`.
- `experiments/transport_test/results/large/{alive,windows}.pt` with `transport_test/results/windows.json`.
- `experiments/kv_task/results/large/windows_{rand32,rand1}.pt`.
- `experiments/teacher_training/results/large/e42/{rand32,rand1}/A_step1000.pt` (60 MB each; the references stage
  reads them).

Every `results/large` is a symlink and arrives dangling. Hard-link the needed files into staging paths that end in
`results/large/`, sync each of those directories, replace each symlink on the pod with the real directory, and check
sha256 on both sides.

On an H100 the W₀ floor differs from E42's L40 floor. The JSON records both (`e42_floor`), and the lead reads E42's
references as from another card.

Download each task's JSON and `M_step1000.pt` to a project path (`experiments/teacher_training/results/e42_amend1_dl/`)
as soon as `[<task> M@1000]` is logged. Terminate as soon as the last JSON's sha256 matches, then move the files.

## What to copy into the report

- Every line starting `[C1]`, `[G2]`, `[<task> M0 train]`, `[<task> M0@1000]`, `[<task> teacher]`, `[<task> G1]`,
  `[<task> W0 trigger]`, `[<task> E42 copy]`, `[<task> M train]`, `[<task> M@100]`, `[<task> M@1000]` and
  `[<task> score]`.
- The masked-copy lines at steps 1, 100, 500 and 1000 of M0 and of arm M.
- Any `STOP:` line.
- The job ids, start and end times, and the card (the JSON's `card`).

## Stall rule

Read the log every 1-2 min. A Pile-harness read prints nothing for ~140 s on an L40, and C1 prints once per seed. The
masked trainer prints at steps 1, 2, 5, 10, 50 and every 100. If the log has not grown for 4 minutes outside the
silent reads, check `squeue` and the job's state. If it is hung, `scancel`, report, and do not resubmit. Never sleep
blindly.
