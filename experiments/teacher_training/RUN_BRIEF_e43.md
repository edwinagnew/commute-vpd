# Pod-runner brief: teacher_training E43, teacher training plus remainder (scored)

Plan: `reports/plans/2026-10-28_plan_e43-teacher-remainder.md` ("Scope as approved"). Code:
`experiments/teacher_training/remainder.py` (imports `run.py` and `path.py` beside it). Predictions:
`experiments/teacher_training/results/predictions_e43.json` (sha256
`74124bbdc9be6b9b97e0d40f58e11374aba13f7512142070f506ce29a3eee172`). The run refuses to start without the predictions
file. It also refuses if the file's `config` or `e43` entries differ from `run.py`'s `CFG` or `remainder.py`'s `AM`.

```
Experiment: teacher_training   Run: E43, scored (its own smoke: C1 in ~3 min, G1 and PC by ~5 min, G3 and G2 by ~7 min)
Command: [ -L experiments/teacher_training/results/large ] && rm experiments/teacher_training/results/large
         mkdir -p experiments/teacher_training/results/large
         (.venv/bin/python experiments/teacher_training/remainder.py cuda; echo "=== FINISHED (exit $?) — results in <repo>/experiments/teacher_training/results ===") 2>&1 | tee -a experiments/teacher_training/results/large/run_e43.log /proc/1/fd/1
         (start it as a background subprocess; no execute() or wait() over 5 min, re-arm instead)
Expected: about 15 min of run on the H100, plus set-up and downloads; GPU memory under 40 GB (one teacher copy of the
          67M model at a time, on top of E41's reads); CPU RAM ~8 GB
Outputs: results/teacher_training_e43.json (rewritten after every gate and state; "score" and "done" at the end) -> bring home
         results/large/run_e43.log -> bring home
         No .pt files: the run saves no states.
Download target: the JSON into experiments/teacher_training/results/ (a real directory). The log into
         experiments/teacher_training/results/e43_dl/run_e43.log, never into results/large (a symlink into commute-data:
         download() there is refused, and a mv waits at a prompt). Move it into results/large only after the pod is
         terminated.
Inputs not in git (each results/large below is a symlink that arrives on the pod dangling): stage them by hard link
         into a path ending in results/large/. On the pod, replace each symlink, including
         experiments/teacher_training/results/large, with a real directory before any mkdir or tee. Use
         sync(include=[<path ending in results/large>]), check sha256 on both sides, and delete the staging
         directories on both sides afterwards. Start the sync before set-up (the uplink is ~80 KB/s).
         experiments/transport_test/results/large/{alive.pt 361 KB, windows.pt 156 KB}
           alive.pt   49746222ec4cfaef330e84a7d6c093d5740f764d6dc9096709d30f177e92c018
           windows.pt c19a880cdb3c7ee267a454512c3c91d5270ac9a6407124110a13d5719fba9f22
           e.g. stage_e43/experiments/transport_test/results/large/ -> sync(include=["stage_e43/experiments/transport_test/results/large"]);
           on the pod: rm the symlink experiments/transport_test/results/large, mkdir it, mv the two files in.
         artifacts/vpd: let pod_setup.sh fetch it from W&B on the pod (works, ~65 s, 2026-10-27). Only if that
                  fails, stage it by hard link as RUN_BRIEF_amend5.md describes.
         artifacts/sets/val_0_256.npy (1 MB; optional, otherwise it streams in ~1 min)
         remainder.py and predictions_e43.json are committed: check the predictions sha256 on the pod.
         NOT needed: the r1 windows, Pile train rows, any .pt under teacher_training/results/large
Platform: RunPod "NVIDIA H100 80GB HBM3" directly. C1's floor 0.0748 and seed-0 KL 0.69882, PC's 2.5159 and 1.4994,
          and G3's E41 copies are that card's. An L40 or an H100 PCIe/NVL would likely stop at C1 or PC.
          Pod commute-teacher_training-e43-<id>, label set through start()'s argument; set up with `bash cluster/pod_setup.sh`.
Pod: fresh; terminate when done. Another pod of this session may be running, so pass instance=... to every
     machine-scoped call from the first one on. Never edit remote-kernels.toml.
Budget: expect about $2 (~25-30 min of pod at about $4.4/h); cap $4 for this run.
```

## Stages, in order, with expected time (H100)

Each state prints `[<label>] pgd ... stochastic ... p_target ... damage ... size ... delta_share ... verdict ... e_delta
... faith ...` after the harness's `[<label>] gain ...` line and two `[faith]` lines. A harness read takes about 53 s
per state. After the four states of a step, the run prints `[T<k> <name>] removal share task ... pile ...` for
do-nothing, plus remainder and least squares, then `[score after T<k>] {...}`.

| Stage | What | Expected |
|---|---|---|
| Load + C1 | `[pile W0 seed s]` for s = 0, 1, 2, then `[C1] {"pass": true, "pgd_floor": 0.0748..., "w0_pgd_kl": 0.69882..., ...}` | ~3 min |
| fine-tune + G1 | `[ft] ...` to step 1000, `[G1 T10] {... "pass": true}` (P 0.99931, size 0.01125), `[G1 T1000]` (0.99961, 0.03209) | ~0.5 min |
| T10 do-nothing + PC | `[T10 do-nothing] pgd 2.5159 ... verdict fail`, `[PC T10] {"pgd": 2.5159..., "kl_task_w0": 5.078..., "kl_pile_w0": ..., "pass": true}` | ~1 min |
| T10 copy + G3, G2 | component training to step 1000, `[T10 teacher training] pgd 1.5465 ... p_target 0.9994`, `[G3 T10] {... "pass": true}`, `[G2 T10] {"removal_vs_copy": ~1e-5, "pass": true}` | ~2.2 min |
| T10 plus remainder, least squares | `[T10 plus remainder] ...` (delta_share about 0.9-1.1), `[T10 least squares] ...` (pgd about 7.0), the three removal lines, `[score after T10]` | ~2 min |
| T1000 | the same: do-nothing 1.4994, copy 0.9591 and P 0.99675, then plus remainder and least squares (about 2.1) | ~5.5 min |
| Score | `[score] {...}` then `done` | seconds |

## Early stop for the controls

C1 runs first and alone. Every gate halts the run itself with a `STOP: <name>: {...}` line and `"stopped"` in the JSON.
Report the line and the numbers named here, download the JSON and the log, then terminate.

- **C1**: `STOP: C1: {...}`. Report the `[C1]` lines and the three `[pile W0 seed s]` lines: the card or harness differs.
- **G1**: `STOP: G1 T<k>: {...}`. Report `[G1 T10]`, `[G1 T1000]`: the fine-tune does not reproduce E41's.
- **PC**: `STOP: PC T<k>: {...}`. Report the do-nothing lines and `[PC T<k>]`: E41's references do not hold here.
- **G3**: `STOP: G3 T<k>: {...}`. Report the `[T<k> teacher training]` line: the copy does not reproduce E41's (or Δ moved).
- **G2**: `STOP: G2 T<k>: {...}`: removing the remainder does not land on the copy (a code bug).
- A results file written on another card or W₀ floor makes the run exit at once, naming both. Move the file aside only
  if the lead says so.
- A Python traceback is a code stop: report it with the last ten log lines. Do not retry.

A rerun on the same card redoes every step not marked done (about 6 min per step).

## Downloads (check the local layout after the first: download() of a directory nested it, 2026-10-19)

- After `[score after T10]`: `teacher_training_e43.json` (to `results/`) and `run_e43.log` (to `results/e43_dl/`).
- Before `terminate()`: the JSON (with `"done": true` and `score`). Check its sha256 on both sides, then terminate.
  **Terminate as soon as the final JSON's sha256 matches.** Bring the log first only if that takes under a minute.
- Terminate before any local file move. Afterwards move `results/e43_dl/run_e43.log` into `results/large/`, check its
  sha256, then delete `e43_dl/` and the `stage_e43/` staging directory. Never `git add` them.

## What to copy into the report

- every `[C1]`, `[pile W0 seed`, `[G1`, `[PC`, `[G3`, `[G2` line;
- every state's read line (pgd, stochastic, p_target, damage, size, delta_share, verdict, e_delta, faith) and its
  harness `gain ...` line;
- the six removal-share lines and both `[score after T<k>]` lines;
- the final `[score]` line, and any `STOP:` line;
- the `[setup ...]` lines, the VPD fetch's time and rate, and the time from `start()` to the first cell;
- the card's `nvidia-smi` name, the pod ID, the minutes and dollars of pod, and the sha256 of every file brought home.

## Stall rule

Read the log every 1-2 min. The known quiet stretches are each harness read (~53 s) and each copy's training between
its logged steps. Outside those, if the log has not grown for 3 minutes, check `nvidia-smi` and the process. If it is
hung, stop, report and terminate. Never sleep blindly.

Set-up deadline: if the first cell has not run 15 min after `start()`, report and terminate.

If the plugin loses the pod (connection refused, port moved), stop and tell the lead at once; do not retry in a loop.
On a WebSocket closure, run `status(instance=...)`, then `create_kernel(instance=...)` at once.
