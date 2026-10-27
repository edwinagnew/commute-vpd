# Pod-runner brief: teacher_spread Amendment 1 (E37 A1, scored)

Plan: `reports/plans/2026-10-26_plan_teacher-spread.md`, section "Amendment 1 (2026-10-27)". Code:
`experiments/teacher_spread/amend1.py` (imports `run.py`), `src/commute/masked_teacher.py`. Predictions:
`experiments/teacher_spread/results/amend1_predictions.json`, filed by the lead. The run will not start without that
file, or if its `config` or `amend1` block differs from the code.

```
Experiment: teacher_spread   Run: amend1, scored (its own smoke: C1 and C2 by ~7 min, C3 by ~8 min)
Command: mkdir -p experiments/teacher_spread/results/large dl_amend1
         (.venv/bin/python experiments/teacher_spread/amend1.py --device cuda; echo "=== FINISHED (exit $?) ===") 2>&1 | tee -a experiments/teacher_spread/results/large/stdout.log /proc/1/fd/1
         (start it as a background subprocess, not a blocking execute(); amend1.py also writes results/large/run.log)
Expected: 40-55 min of run on an H100 plus ~7 min of set-up and the VPD upload. GPU memory: arms 1 and 3 about 30 GB
          (E37 peaked at 30.5 GB), arm 2 and M0 40-50 GB estimated; every state line prints "peak N GB". CPU RAM ~10 GB
Outputs: experiments/teacher_spread/results/amend1.json (rewritten after every read; "status": "done" at the end)
         experiments/teacher_spread/results/large/amend1_arm1_step1000.pt, amend1_arm1_step4000.pt,
           amend1_arm3_step2000.pt, amend1_arm2_step2000.pt (or _step1000.pt if the time rule cut arm 2; none if arm 2
           did not run): bf16 offsets, ~60 MB each
         experiments/teacher_spread/results/large/run.log and stdout.log (on the pod they hold this job only)
Inputs not in git (sync with include=[<directory>], then check sha256 both sides), as E37's brief (RUN_BRIEF.md):
         experiments/transport_test/results/large: only alive.pt and windows.pt (~1.6 MB). The directory also holds A/ and B/
           (433 MB each): hard-link the two files into stage_tt/experiments/transport_test/results/large/ and
           sync(include=["stage_tt/experiments/transport_test/results/large"]); on the pod move them into place.
         artifacts/vpd (3.2 GB): hard-link into stage_vpd/artifacts/vpd/ (same layout), sync(include=["stage_vpd/artifacts"])
           and start it BEFORE set-up; on the pod mv it to /root/artifacts/vpd before `bash cluster/pod_setup.sh`.
           Check sha256 of model_400000.pth and model_step_99999.pt on both sides, then delete the local stage_vpd/ and
           the pod's copy. If the upload will take over 20 min, tell the lead at once with the rate.
         Optional: artifacts/sets/val_0_256.npy (1 MB) into /root/artifacts/sets/.
         Pile train rows 0..4095 stream from Hugging Face in-job (~1.5 min). No saved state from another run is read.
Platform: RunPod "NVIDIA H100 80GB HBM3" only: start(gpu_type="NVIDIA H100 80GB HBM3", ...) with the pod label
          commute-teacher_spread-a1 passed through start()'s argument. C1 stops on any other card (H100 PCIe and NVL
          included). Never edit remote-kernels.toml while any machine of the session runs. Pass instance=... to every
          machine-scoped call. Set up with `bash cluster/pod_setup.sh`.
Pod: fresh; terminate when done (rules below)
Budget: expect $3-4.5 ($5.5 with M0's retry) at about $4/h; cap $8: past it, download what is written and terminate
Watch for: no execute() or wait() over 5 min: re-arm instead. On a WebSocket closure, status(instance=...) then
           create_kernel(instance=...) at once. If the plugin loses the pod (connection refused, port moved), stop and
           tell the lead; do not retry in a loop.
```

## Stages, in order, with expected time on an H100

| Stage | Log lines to see | Expected |
|---|---|---|
| Load, C1 | `[pile W0 seed s] N s` ×3, `[load] NVIDIA H100 80GB HBM3, ...`, `[C1] floor ~0.0748 (known 0.0748) ... pass True` | ~3.5 min |
| Rows | `[rows] sample: ...` ×2 (one lowercase, one plain) | ~2 min |
| Teacher, C2 | `[ft] ...` to step 300, `[C2] held-out gain ~0.130 (E36 0.1301) pass True` | ~30 s |
| W0 | `[W0] logp ~-3.146 gap ~0.272 damage 0 ...`, `[score] {}` | ~20 s |
| do-nothing, C3 | `[do-nothing] logp ~-3.016 gap 0 damage ~0.235 ...`, `[score] ...`, `[C3] {... "pass": true}` | ~20 s |
| probe | `[masked teacher coeff None] warmup 1 ...` ×3, `[probe] {"sec_per_step": ..., "peak_gb": ..., "m0_arm2_train_minutes": ...}`: the masked trainer's first GPU run (nothing moves). Not a stop, but a traceback here is a code stop; if `m0_arm2_train_minutes` is over 40 or `peak_gb` over 70, tell the lead at once | ~10 s |
| arm 3, step 0 | `[arm3_0] second-order share ...` (read it with the step-0 numbers: if it is not small, s and check may differ from E37's 0.80 and 1.05 for that known reason; report it, it is not a failed control), `[faith] ...`, `[arm3_0] logp ~-3.041 ... pile {...}`, `[score]` with s ~0.80, check ~1.05 | ~1.5 min |
| arm 1 | `[component training lr 1e-05] step 1 ... 500 ... 4000`, `[arm1] trained 4000 steps in N s; train KL step 1 ~0.290`, `[saved] amend1_arm1_step1000.pt`, `[saved] amend1_arm1_step4000.pt`, then `[arm1_1000]`, `[arm1_2000]`, `[arm1_4000]` each with `[score]` | ~7 min + 3 min of reads |
| arm 3 | the same with `[arm3]`, 2000 steps, `[saved] amend1_arm3_step2000.pt`, `[arm3_1000]`, `[arm3_2000]` | ~3.5 min + 2 min |
| G2 | `[G2] {"kl": ~0.290, "arm1_kl_step1": ~0.290, "path_kl": ~0, "pass": true}` (path_kl: the two forward paths at W0 alone; on a miss it says whether rounding or a real mismatch caused it) | seconds |
| M0 | `[masked teacher coeff None] warmup 1/2/5/10/50 ...`, `step 50 ... 1000`, `[M0] trained in N s (N s per step); teacher KL at loop step 1 ~0; masked terms ...`, `[M0_1000] ...`, `[M0] {... "pass": true}`, `[M0] pass at coeff None; arm 2 projected N min ...: arm 2 runs 2000 steps` | 5-11 min + 1 min |
| (M0 retry) | only if M0 misses: the same as `M0b` at `coeff 0.05` | as M0 |
| arm 2 | `[masked teacher coeff ...] ...` to step 2000 (or 1000), `[arm2] trained in ...`, `[saved] amend1_arm2_step2000.pt`, `[arm2_1000]`, `[arm2_2000]` | 10-21 min + 2 min |
| End | `[score] {... "P1": ..., "P2": ..., "P3": ..., "branch": ...}`, `FINISHED wrote ...`, `=== FINISHED (exit 0) ===` | seconds |

Copy the seconds per step of M0's first 50 warm-up steps into the report (the time between the `warmup 1` and
`warmup 50` lines). The code applies the plan's time rule itself after M0: if M0 plus arm 2's projection passes 40
min, arm 2 runs 1000 steps and says so in the `[M0] pass ...` line. Do not change the steps by hand.

## Downloads (through a staging directory)

`download()` refuses local paths through the `results/large` symlink, so every file goes through `dl_amend1/` in the
clone. On the pod, `cp` the file into `dl_amend1/`; download that one file (never a directory: a directory download
nests it) into the local `dl_amend1/`; check its sha256 on both sides. Check the local layout after the first one.

- After `[C3]`, and after every `[score]` line that follows a state line: `results/amend1.json`.
- After each `[saved] <file>` line: that `.pt` from `results/large/`.
- After `[arm1_4000]`, `[arm3_2000]`, the `[M0] ...` verdict and the arm 2 reads: `amend1.json` again.
- At the end: `amend1.json` with `"status": "done"`, then `run.log` and `stdout.log` (rename them `amend1_run.log`
  and `amend1_stdout.log` in `dl_amend1/` on the pod first), and any `.pt` not yet home.

**Terminate the pod as soon as the final `amend1.json`'s sha256 matches on both sides** (the logs and offsets should
already be home; if one is not, bring it first only if that takes under a minute). Do not move files locally first.
After `terminate()`: move `dl_amend1/amend1.json` to `experiments/teacher_spread/results/amend1.json`, and the `.pt`
files and the two logs to `experiments/teacher_spread/results/large/` (they land in
`~/commute-data/results-large/teacher_spread/`; do not overwrite E37's `run.log` or `stdout.log`). Check sha256
again after the move, then delete `dl_amend1/`. Never `git add` it.

## Stop rules

The run halts itself on C1-C3 with a `STOP:` line and `"status": "stopped: ..."` in the JSON. Report the line and the
numbers named here, download the JSON and logs, then terminate.

- **C1**, `STOP: C1: W0 is not at its known values on this card`. Report the `[C1]` line. The floor must be within 5% of
  0.0748; on any card but the H100 80GB HBM3 it is None and stops.
- **C2**, `STOP: C2: the teacher's held-out gain ...`. Report `[C2]` and the `[ft] checkpoint 300` line.
- **C3**, `STOP: C3: the do-nothing state does not read as the teacher`. Report `[C3]` and `[do-nothing]`.
- **G2 or M0 missing** is not a stop: the run logs `[arm2] not run: ...` and finishes. Report the `[G2]` and `[M0]` lines.
- **Reading M0**: read M0's stored loop-step-1 teacher KL (`train.M0.kl_step1`, in the `[M0] trained` line) before its damage. It should be about 0; if not, the two forward paths differ and that, not the masked losses, may move the model. Report both.
- **A Python traceback** (an out-of-memory error in M0 or arm 2 included) is a code stop: report it with the last ten
  log lines, download what is written, terminate; do not retry. Arms 1 and 3 are already in the JSON by then.

A rerun on the same card resumes from the JSON: finished stages are skipped (load, C1, rows and teacher are redone,
about 7 min). Only on the lead's say-so.

## What to copy into the report

- every `[load]`, `[C1]`, `[C2]`, `[C3]`, `[probe]`, `[G2]`, `[M0]`, `[arm2]` and `[saved]` line, the `[ft] checkpoint 300` line,
  both `[rows] sample` lines and the `[arm3_0] second-order share` line;
- every state line (`[W0]`, `[do-nothing]`, `[arm3_0]`, `[arm1_N]`, `[arm3_N]`, `[M0_1000]`, `[M0b_1000]`,
  `[arm2_N]`), with its seconds and peak GB, and the `[score]` line after each;
- every `[arm1] trained`, `[arm3] trained`, `[M0] trained` and `[arm2] trained` line;
- the last `[score]` line, the `[setup ...]` lines, the time from `start()` to the first cell, the VPD upload's time
  and rate, M0's seconds per step over its first 50 warm-up steps;
- the pod id, card, $ spent, and the sha256 of every file brought home.

## Stall rule

Read the log every 1-2 min. Known silent stretches: the Pile stream (~1.5 min, until `[data] train rows ...`); each
harness read, which prints `[faith]` per block (~25 s each); component training between its 500-step lines (~50 s);
masked training between its 100-step lines (30-60 s). Outside those, if the log has not grown for 3 minutes, check
`nvidia-smi` and the process; if it is hung, report and terminate. Never sleep blindly. Set-up deadline: if the first
cell has not run 15 min after `start()`, not counting the VPD upload, report and terminate.
