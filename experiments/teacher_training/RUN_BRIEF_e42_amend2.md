# Pod-runner brief: E42 Amendment 2, each masked term alone with nothing to copy, then the copy (scored)

Plan: `reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md` → "Amendment 2". Predictions:
`experiments/teacher_training/results/predictions_e42_amend2.json` (filed 2026-10-28 14:13:50, sha256 `1b3928be…`).
The run refuses to start without it, or if its `config`, `e42`, `amend1` or `amend2` differ from `run.py`'s `CFG`,
`newskill.py`'s `AM`, `newskill_amend1.py`'s `AM` and `newskill_amend2.py`'s `AM`.

```
Experiment: teacher_training   Run: E42 Amendment 2, scored (its own smoke: C1 in the first ~7 min, G2 seconds later,
            the first control's training curve from minute ~8 and its verdict at ~minute 13)
Command:    sbatch cluster/teacher_e42_amend2.sbatch      (from the clone root, no arguments; record the job id)
            It runs  .venv/bin/python experiments/teacher_training/newskill_amend2.py  and logs to
            cluster/logs/commute-teacher-e42a2-<jobid>.out
Expected:   25 min on an L40 if a single-term control holds; 24 min if all three controls fail (stop); 31 min if both
            single-term controls fail and the trigger-only one holds. debug QoS, 55-min wall time; GPU memory under
            30 GB; CPU RAM under 16 GB
Outputs:    experiments/teacher_training/results/teacher_training_e42_amend2_rand32.json   (updated after every stage)
            experiments/teacher_training/results/large/e42_amend2/rand32/{M0_ci,M0_st[,M0_trig]}_step1000.pt and
              M_<recipe>_step1000.pt   (bf16 offsets, ~60 MB each)
            cluster/logs/commute-teacher-e42a2-<jobid>.out
            On the cluster these land on NFS directly: nothing to download. Check the JSON ends with "done": true and a
            "score" block, or with a "stopped" line.
Platform:   MATS cluster, account c11, partition compute, QoS debug ($0). If the job has not started within 10 min,
            cancel it and use RunPod (see "RunPod fallback").
Budget:     $0 on the cluster; RunPod fallback cap $2
```

## Stages, in order (L40 minutes; job 40338's timings)

| Stage | Log lines | Minutes |
|---|---|---|
| load, bases | `bases and input moments ...` | 0.5 |
| C1 | `[pile W0 seed 0/1/2] ...` each ~124 s, then `[C1] ...` | 7 |
| G2 | `[G2] {"ci_masked_w0": ..., "path_kl": ..., "pass": ...}` | 0.1 |
| M0_ci | `[masked terms ci 0.5 stoch 0] step 1 / 2 / 5 / 10 / 50 / 100 / 200 ...`, `[rand32 M0_ci@1000] ...` (two lines, after a silent ~2-min Pile read), `[rand32 M0_ci train] {...}`, `[M0_ci] control: pass / inconclusive / fail` | 5.4 |
| M0_st | the same with `[masked terms ci 0 stoch 0.5]` and `M0_st` | 5.4 |
| M0_trig (only if both read `fail`) | the same with `[masked terms ci 0.5 stoch 0.5 trigger]` and `M0_trig` | 5.4 |
| tree | `[tree] controls {...}; copy ci / st / trig / None` | 0 |
| teacher, G1 | `[ft] checkpoint 10 / 300 / 1000`, `[rand32 teacher] {...}`, `[rand32 G1] {...}` | 0.7 |
| references | `[rand32 W0 trigger] {...}; E42's floor ..., this job's ...`, `[rand32 E42 copy] ...` | 0.3 |
| copy M_<recipe> | `[masked terms ...] step ...`, `[rand32 M_<recipe> train] {...}`, `[rand32 M_<recipe>@100] ...`, `[rand32 M_<recipe>@1000] ...` | 5.7 |
| score | `[rand32 score] {...}`, `done` | 0 |

A control that reads `fail` does **not** stop the job: the tree reads it. The first masked-terms line of each control
should show `unmasked` about 0.0016 (G2's `path_kl`, the teacher is W₀) and the trained term near Amendment 1's step-1
value (CI-masked 1.13, stochastic 0.74). In the copy, `unmasked` at step 1 should be about 13 (E42's teacher gap at
W₀). If a control's training takes more than 6 minutes, say so in the report. It is not a stop.

## Stop rules (the run halts itself with a `STOP:` line; report that line and the numbers named, do not resubmit)

- **C1 (early stop for the control)**: `STOP: C1: W0 is not at its known values`. Report the `[C1]` line. Expected
  `pgd_floor` 0.2088 on an L40.
- **G2**: `STOP: G2: the trainer's unmasked student at W0 is ... nats from the target model`. Report the `[G2]` line
  (pass needs `path_kl` ≤ 0.01; Amendment 1 read 0.0016).
- **tree**: `STOP: tree: every masked control failed ...`. This is a filed outcome, not an error: report every
  `[rand32 M0_*@1000]` record line, every `[rand32 M0_* train]` line and the `[tree]` line. Tell the lead in the
  hand-back's first line.
- **G1**: `STOP: G1 rand32: teacher P(target) at step 10 ...`. Report the `[rand32 G1]` and `[rand32 teacher]` lines.
- A traceback without a `STOP:` line: report it with the last state line, and do not retry.

A rerun continues its results file only on the same card with the same W₀ floor; otherwise it exits naming both.
Finished stages (G2 passed, each control, references, the copy) are skipped, and the teacher is regenerated (40 s).
Move a file aside only on the lead's word.

## RunPod fallback (only if the cluster does not start the job within 10 min)

One H100 pod, `commute-teacher_training-e42a2-<id>`, set up with `bash cluster/pod_setup.sh`. Run
`(.venv/bin/python experiments/teacher_training/newskill_amend2.py; echo "=== FINISHED (exit $?) ===") 2>&1 | tee -a experiments/teacher_training/results/large/run_e42_amend2.log /proc/1/fd/1`
as a background subprocess. That is about 20 min of run plus set-up, about $1.5. On an H100 the lines are 1.40 / 1.61
and E42's references are from another card (the JSON records `e42_floor`).

Inputs not in git, all needed on the pod:
- `src/commute/masked_terms.py`, `src/commute/masked_copy.py`, `experiments/teacher_training/newskill_amend2.py`,
  `newskill_amend1.py` (check `git status`); `results/predictions_e42_amend2.json`.
- `newskill.py`, `src/commute/kv_windows.py` and E42's `results/teacher_training_e42_rand32.json`.
- `experiments/transport_test/results/large/{alive,windows}.pt` with `transport_test/results/windows.json`.
- `experiments/kv_task/results/large/windows_rand32.pt`.
- `experiments/teacher_training/results/large/e42/rand32/A_step1000.pt` (60 MB; the references stage reads it).

Every `results/large` is a symlink and arrives dangling. Hard-link the needed files into staging paths that end in
`results/large/`, sync each of those directories, replace each symlink on the pod with the real directory, and check
sha256 on both sides. Download the JSON and each `*_step1000.pt` to `experiments/teacher_training/results/e42_amend2_dl/`
as each is written. Terminate as soon as the final JSON's sha256 matches, then move the files.

## What to copy into the report

- Every line starting `[C1]`, `[G2]`, `[rand32 M0_ci@1000]`, `[rand32 M0_st@1000]`, `[rand32 M0_trig@1000]`,
  `[rand32 M0_* train]`, `[M0_*] control:`, `[tree]`, `[rand32 teacher]`, `[rand32 G1]`, `[rand32 W0 trigger]`,
  `[rand32 E42 copy]`, `[rand32 M_* train]`, `[rand32 M_*@100]`, `[rand32 M_*@1000]` and `[rand32 score]`.
- The masked-terms lines at steps 1, 100, 500 and 1000 of each control and of the copy.
- Any `STOP:` line.
- The job id, start and end times, and the card (the JSON's `card`).

## Stall rule

Read the log every 1-2 min. A Pile-harness read prints nothing for ~124 s on an L40, and C1 prints once per seed. The
masked trainer prints at steps 1, 2, 5, 10, 50 and every 100. If the log has not grown for 4 minutes outside the
silent reads, check `squeue` and the job's state. If it is hung, `scancel`, report, and do not resubmit. Never sleep
blindly.
