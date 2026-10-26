# Pod-runner brief: teacher_training Amendment 5 (scored)

Plan: `reports/plans/2026-10-23_plan_teacher-training.md` → "Amendment 5 (2026-10-26, lead-2): why the two adversaries
disagree, and is a copy unique?". Code: `experiments/teacher_training/amend5.py` (commit a16d366; its docstring lists
the differences between the two adversaries and every reading). Predictions:
`experiments/teacher_training/results/predictions_amend5.json`. The run refuses to start without it, and also if its
`config` or `amend5` entries differ from `run.py`'s `CFG` or `amend5.py`'s `AM`.

```
Experiment: teacher_training   Run: Amendment 5, scored (its own smoke: C1 and G1 run in the first ~7 min)
Command: mkdir -p experiments/teacher_training/results/large
         (.venv/bin/python experiments/teacher_training/amend5.py cuda; echo "=== FINISHED (exit $?) — results in <repo>/experiments/teacher_training/results ===") 2>&1 | tee -a experiments/teacher_training/results/large/run_amend5.log /proc/1/fd/1
         (start it as a background subprocess; no execute() or wait() over 5 min, re-arm instead)
Expected: about 27 min of run on the H100 (estimate: E34/A4 timings, see stages), + set-up and downloads; GPU memory
          ~33 GB at the 256-row reading (rows256 caches g and logits for 32 batches): H100 only; CPU RAM ~8 GB
Outputs: results/teacher_training_amend5.json (rewritten after every reading; out["score"] and "done" at the end) -> bring home
         results/large/r1_t534_v369/amend5_{A,B}_sgd_seed{0,1}_step1000.pt (4 files, ~60 MB each, bf16 offsets) -> bring home as written
         results/large/run_amend5.log -> bring home
Inputs not in git (sync with include=[<the gitignored directory itself>], then check sha256 on both sides):
         experiments/transport_test/results/large            (alive.pt 361 KB, windows.pt 156 KB)
         experiments/task_ladder/results/large/windows_r1_t534_v369.pt   (263 KB; its directory is the gitignored one:
                  hard-link this one file into a staging path ending in task_ladder/results/large/ and name that path,
                  or sync the directory)
         artifacts/vpd (3.0 GB: s-55ea3f9b/{final_config.yaml, model_400000.pth}, t-9d2b8f02/{model_config.yaml,
                  final_config.yaml, ckpt/model_step_99999.pt}). FIRST let pod_setup.sh fetch them from W&B on the pod
                  (its "VPD files" step; the user says the goodfire/spd 403 is fixed) and report the fetch's time and
                  rate. ONLY if that fetch fails, send the files: `mkdir -p stage_vpd/artifacts && cp -al
                  "$(readlink -f artifacts)/vpd" stage_vpd/artifacts/` (hard links; artifacts is a symlink into
                  commute-data on the same NFS), then sync(include=["stage_vpd/artifacts"]) (".gitignore: artifacts"
                  makes stage_vpd/artifacts the gitignored directory itself), on the pod `mkdir -p /root/artifacts &&
                  mv stage_vpd/artifacts/vpd /root/artifacts/vpd` (COMMUTE_CACHE_DIR = /root/artifacts,
                  remote-kernels.toml), sha256 the two .pth/.pt files on both sides, then delete stage_vpd on both
                  sides. The uplink has measured ~80 KB/s (CLAUDE.md, 2026-10-23): time the first minute, report the
                  measured rate, and tell the lead at once if 3 GB will take more than ~20 min. pod_setup.sh's VPD
                  fetch skips files that are already there.
         artifacts/sets/val_0_256.npy (1 MB; Pile val rows 0..255, the harness blocks) -> /root/artifacts/sets/ (optional:
                  without it the rows stream in ~1 min). Pile train rows 32,768..40,959 (arm B) stream in-job, ~3 min.
         NOT needed: any other .pt under teacher_training/results/large (no saved state is read), the C3 teacher.
         Tracked and arriving with the plain sync(): amend5.py, run.py, amend4.py, predictions_amend5.json,
         transport_test/results/windows.json, src/commute/.
Platform: RunPod "NVIDIA H100 80GB HBM3" directly. C1's floor (0.075) and G1's references are that card's (E34, A4);
          an L40 reads a floor of ~0.21 and stops at C1. Pod commute-teacher_training-amend5-<id>, label set through
          start()'s argument; set up with `bash cluster/pod_setup.sh`
Pod: fresh; terminate when done. Another pod (E37) runs from the same session: pass instance=... to every
     machine-scoped call, and never edit remote-kernels.toml while either machine runs
Budget: $4 cap for this run; expect about $2-2.5 (~35-45 min of pod at about $3-4/h)
Watch for: see stages and stop rules below
```

## Stages, in order, with expected time (H100; harness and training from E34/A4's logs, budget from A4's brief)

Every reading line is `[part1 <state> <reading>] KL_<kind> <v> ratio_<kind> <v> ... (<s> s)`, with state `W0` or
`teacher` and kind `bf16` and/or `fp32`.

| Stage | What | Expected |
|---|---|---|
| Load + C1 | model, harness, W0 on Pile adversary seeds 0, 1, 2 (`[pile W0 seed s]` lines), run.py's `[C1] <s> s; {...}`, then amend5's `[C1] {"pass": ..., "pgd_floor": ...}` | ~3 min |
| W0 defaults | `[part1 W0 harness_b0]`, `[part1 W0 harness_b1]`, `[part1 W0 budget]` | ~1 min |
| Teacher | fine-tune (`[ft] step ...`) → `[r1_t534_v369 teacher sgd 0.03] reproduction {...}` | ~0.5 min |
| Teacher defaults + G1 | `[part1 teacher harness_b0/harness_b1/budget]`, then `[G1] {...}` | ~1.5 min |
| Zero-step check | `[zero_step W0] {"harness": ..., "pgd_sources_from_harness_start": ..., "rel_diff": ...}` (reported; expected rel_diff ~1e-6) | ~0.5 min |
| 20-step variants | `[part1 W0 start]`, `[part1 W0 rows256]`, `[part1 teacher start]`, `[part1 teacher rows256]` | ~2-3 min |
| Part 2 | `[part2 A seed 0]`, `[part2 A seed 1]` (~85 s each); the Pile train stream (`[data] train rows 32768..40960 ...`, ~3 min); `[part2 B seed 0]`, `[part2 B seed 1]` (~100 s each); `[part2 cos] {...}` | ~10 min |
| 160-step readings | `[part1 W0 budget160]` (~2 min), `[part1 W0 harness160_b0]` (~2.5 min, an estimate: the authors' loop recomputes g every step), then the same two for `teacher` | ~9 min |
| Score | `[score] {...}` then `done` | seconds |

The 160-step times are estimates. If one 160-step reading takes over 4 min, tell the lead and keep going: the run
resumes per reading from its results file on the same card, and the $4 cap holds.

## Downloads (into the matching parent; check the local layout after the first, since download() of a directory nested it, 2026-10-19)

- After `[G1]`: `teacher_training_amend5.json`.
- After `[part1 teacher rows256]`: the JSON.
- After each `[part2 <arm> seed <s>]` line: that copy's `.pt` (`results/large/r1_t534_v369/amend5_<arm>_sgd_seed<s>_step1000.pt`)
  and the JSON.
- After `[part2 cos]`: the JSON.
- Before `terminate()`: the JSON (with `"done": true` and `score`), `run_amend5.log`, and all four
  `results/large/r1_t534_v369/amend5_*.pt` (a resumed run rebuilds the cosines from them, so they must be home).
  Check sha256 on both sides, then terminate.

## Stop rules (the run halts itself with a `STOP:` line; report that line and the numbers named here, then terminate)

- **C1**: `STOP: C1: W0 check <bool>, floor <f> (expected 0.075 within 5%)`. Report both `[C1]` lines (pgd_rel,
  ci_rel, pgd_floor). Pass needs faith_score.w0_check to pass and a floor in 0.07125-0.07875 (A4 read 0.0748 on this card).
- **G1**: `STOP: G1: {checks}; P <p> size <s> harness <r> budget <r>`. Report the `[G1]` line and the
  `reproduction` line. Pass needs P(target) within 0.01 of 0.983, size within 0.002 of 0.0287, harness ratio within
  0.01 of 1.588, and budget ratio at 20 steps within 5% of 9.24.
- A results file written on another card or W0 floor makes the run exit at once, naming both; move the file aside
  only if the lead says so.
- A Python traceback is a code stop: report it with the last ten log lines; do not retry.

A rerun on the same card skips finished readings and copies (the teacher fine-tune, ~30 s, is regenerated).

## What to copy into the report

Every line starting `[C1]`, `[pile W0 seed`, `[r1_t534_v369 teacher sgd 0.03] reproduction`, `[G1]`, `[zero_step W0]`, every `[part1 ...]`
line (KL, ratio and seconds), every `[part2 ...]` line, both `[score]` lines (the second: P2 as filed, implicates_agree, identity_ok), any `STOP:` line, plus the `[setup ...]` lines,
the measured uplink rate and duration of the VPD sync, and the time from `start()` to the first cell. Name the card
`nvidia-smi` reports.

## Stall rule

Read the log every 1-2 min. The known long steps are the 160-step readings (a line prints only when each finishes)
and the Pile train stream. Outside those, if the log has not grown for 3 minutes, check `nvidia-smi` and the process;
if it is hung, stop, report, and terminate. Never sleep blindly. If the plugin loses the pod (connection refused,
port moved), stop and tell the lead at once; on a WebSocket closure run `status(instance=...)`, then
`create_kernel(instance=...)` at once.
