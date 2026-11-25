# Pod-runner brief: teacher_spread (E37, scored)

Plan: `reports/plans/2026-10-26_plan_teacher-spread.md`. Predictions: `experiments/teacher_spread/results/predictions.json`.
The run will not start without that file, or if its `config` differs from `run.py`'s `CFG`.

```
Experiment: teacher_spread   Run: scored (its own smoke: C1 and C2 run in the first ~7 min, C3 by ~10 min)
Command: mkdir -p experiments/teacher_spread/results/large
         (.venv/bin/python experiments/teacher_spread/run.py --device cuda; echo "=== FINISHED (exit $?) — results in <repo>/experiments/teacher_spread/results ===") 2>&1 | tee -a experiments/teacher_spread/results/large/stdout.log /proc/1/fd/1
         (start it as a background subprocess, not a blocking execute(); run.py also writes results/large/run.log itself)
Expected: about 22 min of run on an H100, plus set-up and the VPD upload (see Inputs); GPU memory estimated 15-25 GB
          (each state line prints "peak N GB"); CPU RAM ~8 GB
Outputs: results/teacher_spread.json (rewritten after every state; "status": "done", "score" and "uniqueness" at the end) -> bring home
         results/large/copy_{lower_seed0,plain_seed0,lower_seed1}_step1000.pt (bf16 offsets, ~60 MB each) -> bring home each as written
         results/large/run.log and results/large/stdout.log -> bring home
Inputs not in git (sync with include=[<directory>], then check sha256 both sides):
         experiments/transport_test/results/large            (alive.pt, windows.pt; ~1.6 MB; round 1's alive masks and windows)
         artifacts/vpd (3.2 GB: s-55ea3f9b/{model_400000.pth 2.9 GB, final_config.yaml}, t-9d2b8f02/{final_config.yaml,
           model_config.yaml, ckpt/model_step_99999.pt 268 MB}). W&B may refuse the dev node's key for goodfire/spd
           (403), so send the files: hard-link them into a staging dir `stage_vpd/artifacts/vpd/` (same layout), then
           sync(include=["stage_vpd/artifacts"]) and start it BEFORE set-up. On the pod, mv stage_vpd/artifacts/vpd to
           /root/artifacts/vpd (COMMUTE_CACHE_DIR is /root/artifacts) before `bash cluster/pod_setup.sh`; its fetch
           keeps files already in place without asking W&B. Check sha256 of the two large files on both sides, then
           delete the local stage_vpd/ and the pod's copy of it. Time the first minute of the upload: if 3.2 GB will
           take over 20 min, tell the lead at once and say how long, before paying for the wait.
         Optional: artifacts/sets/val_0_256.npy (1 MB) into /root/artifacts/sets/ skips one stream.
         Pile train rows 0..4095 stream from Hugging Face in-job (~1.5 min, cached under /root/artifacts/sets).
         The code (run.py, src/commute/row_copy.py) and results/predictions.json are in the working tree: the plain
         sync() after the clone carries them. No saved state from another experiment is read.
Platform: RunPod "NVIDIA H100 80GB HBM3" only: pass start(gpu_type="NVIDIA H100 80GB HBM3"). C1 stops on any card
          not in its table, H100 PCIe and H100 NVL included. Pod label commute-teacher_spread-<id>, set by start()'s
          argument. Do NOT edit remote-kernels.toml: another pod of this session (E34 Amendment 5) runs in parallel.
          Pass instance=... to every machine-scoped call (status, sync, download, create_kernel, stop, terminate).
          Set up with `bash cluster/pod_setup.sh`.
Pod: fresh; terminate when done
Budget: $4 cap for this run; expect about $2 (~35 min with set-up at about $3-4/h), more if the VPD upload is slow
Watch for: stages, stop rules and downloads below. No execute() or wait() over 5 min: re-arm instead. On a WebSocket
           closure, call status(instance=...) and then create_kernel(instance=...) at once. If the plugin loses the pod,
           stop and tell the lead; do not retry in a loop.
```

## Stages, in order, with expected time on an H100 (E34's H100 briefs; E36's teacher log)

| Stage | Log lines to see | Expected |
|---|---|---|
| Load | `bases and input moments ...`, `[faith] ...` ×6, `[pile W0 seed s] N s` ×3, `[load] <card>, N s, peak N GB` | ~4 min |
| C1 | `[C1] floor 0.0748 (known 0.0748) ... pass True` | seconds |
| Rows | `[data] train rows 0..4096: cached at ...` (on first stream), `[rows] sample: ...` ×2 (one lowercase, one plain) | ~2 min |
| Teacher, C2 | `[ft] step N ...` to 300, `[ft] checkpoint 300: ce ~3.016 ...`, `[C2] held-out gain 0.130x (E36 0.1301) pass True` | ~30 s |
| W0 | `[W0] logp ~-3.146 gap ... damage 0 budget_kl ...`, then `[score] {}` | ~30 s |
| do-nothing, C3 | `[faith] ...` ×2, `[do-nothing] logp ~-3.016 gap ~0 ... pile {...}`, `[score] ...`, `[C3] {... "pass": true}` | ~1.5 min |
| adjoint | `[adjoint] logp ... pile {...}`, `[score] ...` | ~2 min |
| lower_s0 | `[component training lr 1e-05] step 1/2/3/5/10/20/30/1000 ...`, `[copy lower seed 0] trained in N s; train KL ...`, `[lower_s0] ...`, `[score] ...` | ~3 min |
| plain_s0 | the same with `plain seed 0` / `[plain_s0]` | ~3 min |
| lower_s1 | the same with `lower seed 1` / `[lower_s1]` (no budget reading) | ~3 min |
| Uniqueness, end | `[uniqueness] cosine between batch orders ...`, `[score] {...}`, `FINISHED wrote ...`, then `=== FINISHED (exit 0)` | seconds |

The budget adversary, 20 steps on 128 rows, runs inside the W0, do-nothing, adjoint, lower_s0 and plain_s0 reads. It
takes about 17 s on an H100, 33 s on an L40. Copy the first state line's seconds into the report.

## Downloads

- After `[C3]`: `teacher_spread.json`.
- After each `[copy <text> seed <s>] trained in` line: that copy's `.pt`. It is written before the state is read.
  Download the files themselves into `experiments/teacher_spread/results/large/`, not the directory: `download()` of a
  directory nests it. Check the local layout after the first one. After each `[<text>_s<seed>]` line, the JSON again.
- Before `terminate()`: the JSON with `"status": "done"`, `run.log`, `stdout.log` and any `.pt` not yet home, each
  checked by sha256 on both sides. Then terminate.

## Stop rules

The run halts itself with a `STOP:` line and `"status": "stopped: ..."` in the JSON. Report that line and the numbers
named here, download the JSON and logs, then terminate.

- **C1**, `STOP: C1: W0 is not at its known values on this card`. Report the `[C1]` line: floor, known floor,
  pgd_rel, ci_rel. On an H100 80GB HBM3 the floor must be within 5% of 0.0748. On any other card the known floor is
  None and it stops: you got the wrong card.
- **C2**, `STOP: C2: the teacher's held-out gain ...`. Report the `[C2]` line and the `[ft] checkpoint 300` line.
  The gain must be within 0.005 of 0.1301 and at least 0.02.
- **C3**, `STOP: C3: the do-nothing state does not read as the teacher`. Report the `[C3]` line and the `[do-nothing]`
  line. Its logp must be within 1e-4 of the teacher net's, and its gap within 1e-4 of 0. This stops before any copy.
- **Resume check**, an exit without `STOP:` that ends "... is missing: bring it back from the earlier run first".
  On a rerun, every copy state already in the JSON needs its `.pt` in `results/large/`. Sync the saved copies back
  up first. A JSON written on another card also exits, naming both cards.
- A Python traceback is a code stop. Report it with the last ten log lines; do not retry.

A rerun on the same card resumes from the JSON: finished states are skipped. Load, C1, the rows and the teacher are
redone each time, about 7 min.

## What to copy into the report

Copy these lines from the log:
- every line starting `[load]`, `[C1]`, `[C2]`, `[C3]` and `[uniqueness]`, the `[ft] checkpoint 300` line, and both
  `[rows] sample` lines;
- every state line (`[W0]`, `[do-nothing]`, `[adjoint]`, `[lower_s0]`, `[plain_s0]`, `[lower_s1]`), with its seconds
  and peak GB;
- every `[copy ...] trained in` line, with the mean train KL at steps 451-500 and 951-1000;
- the last `[score]` line, the `[setup ...]` lines, the time from `start()` to the first cell, and the VPD upload's
  time and rate;
- the pod id, card, $ spent, and sha256 of every file brought home.

## Stall rule

Read the log every 1-2 min. The known silent stretches are:
- the Pile train stream, until `[data] train rows ...` prints, about 1.5 min;
- each harness read, which prints `[faith]` per block, about 25 s each;
- copy training between its step-30 and step-1000 lines, about 100 s.

Outside those, if the log has not grown for 3 minutes, check `nvidia-smi` and the process. If it is hung, stop,
report and terminate. Never sleep blindly. Set-up deadline: if the first cell has not run 15 min after `start()`,
not counting the VPD upload, report and terminate.
