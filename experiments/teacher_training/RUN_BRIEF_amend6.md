# Pod-runner brief: teacher_training Amendment 6, teacher training coverage (scored)

Plan: `reports/plans/2026-10-23_plan_teacher-training.md` → "Amendment 6 (2026-10-28, commute lead): teacher training
coverage". Code: `experiments/teacher_training/amend6.py` (its docstring lists the stages). Predictions:
`experiments/teacher_training/results/predictions_amend6.json`. The run refuses to start without it, and also if its
`config` or `amend6` entries differ from `run.py`'s `CFG` or `amend6.py`'s `AM`.

```
Experiment: teacher_training   Run: Amendment 6, scored (its own smoke: C1, the W0 plain-text control and G1 run in the first ~7 min)
Command: [ -L experiments/teacher_training/results/large ] && rm experiments/teacher_training/results/large
         mkdir -p experiments/teacher_training/results/large
         (.venv/bin/python experiments/teacher_training/amend6.py cuda; echo "=== FINISHED (exit $?) — results in <repo>/experiments/teacher_training/results ===") 2>&1 | tee -a experiments/teacher_training/results/large/run_amend6.log /proc/1/fd/1
         (start it as a background subprocess; no execute() or wait() over 5 min, re-arm instead)
Expected: about 85 min of run on the H100 (stages below), + ~15 min set-up and downloads; GPU memory under 40 GB; CPU RAM ~10 GB
Outputs: results/teacher_training_amend6.json (rewritten after every state; "score" and "done" at the end) -> bring home as it grows
         results/large/run_amend6.log -> bring home
         8 files results/large/<task>/amend6_<teacher>_<arm>_seed<s>_step<n>.pt (bf16 offsets, ~60 MB each) -> bring home as written:
           r2_t275_v2667/amend6_sgd_{A,B}_seed0_step1000.pt
           {emoticon,r1_t534_v369,r0_t352_v310}/amend6_lora_B_seed0_step5000.pt
           {emoticon,r1_t534_v369,r0_t352_v310}/amend6_lora_B_seed1_step2500.pt
         NOT needed: results/large/<task>/lora_teacher_factors.pt (amend3.lora_teacher writes them; Amendment 3 has them)
Download target: every results/large file into a project path, experiments/teacher_training/results/amend6_dl/<same subpath>,
         never into results/large (a symlink into commute-data: download() there is refused and a mv waits at a prompt).
         Move them into results/large only after the pod is terminated. The JSON goes to its own place, results/.
Inputs not in git (each results/large below is a symlink that arrives on the pod dangling: stage by hard link, then on
         the pod replace the symlink with the real directory; sync(include=[<path ending in results/large>]); check
         sha256 on both sides; delete the staging directory on both sides afterwards):
         experiments/transport_test/results/large/{alive.pt 361 KB, windows.pt 156 KB}
         experiments/task_ladder/results/large/windows_{r0_t352_v310,r1_t534_v369,r2_t275_v2667}.pt (263 KB each; r2 is
                  new to this experiment, read by Job.task for stage b)
         artifacts/vpd: let cluster/pod_setup.sh fetch it from W&B on the pod (its "VPD files" step; worked 2026-10-27,
                  ~45 MB/s). Only if that fails, stage it by hard link as in RUN_BRIEF_amend5.md and tell the lead the rate.
         artifacts/sets/val_0_256.npy (1 MB, optional; the harness rows stream in ~1 min without it). Pile train rows
                  32,768..40,959 (students B) stream in-job, ~3 min, before the first student B.
         amend6.py and predictions_amend6.json are untracked unless the lead commits them first: check they are on the pod.
Platform: RunPod "NVIDIA H100 80GB HBM3" directly (over 30 min; G1's references, C1's floor 0.075 and every comparison
          are that card's; another H100 variant or an L40 will stop at G1 or C1). Pod commute-teacher_training-amend6-<id>,
          label set through start()'s argument; set up with `bash cluster/pod_setup.sh`
Pod: fresh; terminate when done. If another machine of the session runs, pass instance=... to every machine-scoped
     call and never edit remote-kernels.toml while either runs
Budget: $10 cap for this run; expect about $8 (~100 min at about $4.7/h). If the lead says the session budget is short,
        stop the process after the line `[r0_t352_v310/lora/B0@5000] pile_damage ...` (end of stage d's seed 0, about
        $6.5 so far): every state up to there is in the JSON
Watch for: stages, stop rules and skips below
```

## Stages, in order, with expected time (H100; from E34's, A3's and A5's logs)

State lines come in pairs: run.read_state's `[<label>] pgd ... p_target ... verdict ...` (all E34 numbers; pgd is
None without the adversary) and `[<label>] pile_damage <x> pile_gap <y> [faith <v>]`. Labels are
`<task>/<teacher>/<state>`, the state being `nothing`, `adjoint` or `<arm><seed>@<step>`. A teacher prints
`[<task>/<teacher> teacher] {"reading": ..., "matches": ..., "gap_w0": ...}`. Training prints
`[component training lr ...] step 500: ...` every 500 steps, then `[<task>/<teacher>/<arm><seed>] <n> steps in <s> s; curve [...]`.
Each stage ends with `[stage_<x>] done; score so far {...}`.

| Stage | What | Expected |
|---|---|---|
| Load + C1 | model, harness, `[pile W0 seed 0/1/2]`, `[C1] <s> s; {...}` | ~4 min |
| W0 control | `[W0] {"pile_damage": ...}`: must be <= 1e-4 (expected 0) | seconds |
| G1 | r1 task context (~1 min); `[r1_t534_v369/sgd teacher]`; `r1_t534_v369/sgd/nothing` with the adversary (~1 min); `.../adjoint`; `[G1] {...}` | ~3 min |
| a | Pile train stream (`[data] train rows 32768..40960`, ~3 min); sgd A0, B0; the Adam teacher, its nothing, adjoint, A0, B0; the LoRA teacher, its nothing, adjoint, A0 (all `@1000`, no adversary) | ~13 min |
| c | `r1_t534_v369/sgd/S0@1000` with the adversary, then `[G3] {...}`; then adam S0 and lora S0 | ~7 min |
| b | r2 task context (~1 min); `[r2_t275_v2667/sgd teacher]` (gate); nothing (adversary), adjoint, A0@1000 and B0@1000 (adversary each) | ~8 min |
| d seed 0 | LoRA teachers emoticon (~40 s) and r0 (context ~1 min + ~40 s); per task `<task>/lora/B0`, 5,000 steps (~8.5 min), states @1000, @2500 and @5000 (adversary at 2500 and 5000) | ~33 min |
| d seed 1 | per task `<task>/lora/B1`, 2,500 steps (~4.2 min), states @1000 and @2500 (adversary at 2500) | ~17 min |
| Score | `[score] {...}` then `done` | seconds |

## Downloads (check the local layout after the first one; download() of a directory nested it, 2026-10-19)

- After `[G1]` and after each `[stage_<x>] done` line: `results/teacher_training_amend6.json`.
- After the state lines of `r2_t275_v2667/sgd/A0@1000` and `.../B0@1000`: that student's `.pt`.
- After each `<task>/lora/B0@5000` and `<task>/lora/B1@2500` state line: that student's `.pt`.
- Before `terminate()`: the JSON (with `"done": true` and `score`), `run_amend6.log` and all 8 `.pt` files. Check
  sha256 on both sides, terminate, and only then move the `.pt` files from `results/amend6_dl/` into `results/large/`.

## Stop rules (the run halts itself with a `STOP:` line; report that line and the numbers named here, then terminate)

- **C1**: `STOP: C1: W0 is not at its known values; nothing read`. Report the `[C1]` line (pgd_rel, ci_rel, pgd_floor).
- **W0 control**: `STOP: W0 plain-text damage <x> > 0.0001`. Report the `[W0]` line.
- **G1**: `STOP: G1: r1 SGD teacher {...}, do-nothing ratio <r>`. Report the `[G1]` line and the teacher line. Pass needs
  P(target) within 0.01 of 0.9834, size within 0.003 of 0.02875, and the do-nothing adversarial ratio within 0.03 of 1.588.
- A results file written on another card or W0 floor makes the run exit at once, naming both; move it aside only if
  the lead says so.
- A Python traceback is a code stop: report it with the last ten log lines; do not retry.

## Skips (the run goes on; tell the lead at once, in one line, and keep watching)

- **G3**: `[G3] {...}: arm S does not reproduce E34; S arms skipped` (needs verdict 'fail' and P within 0.02 of 0.9857).
- **r2 gate**: `[r2_t275_v2667/sgd] teacher P <p> < 0.5: students skipped`.
- Any teacher line with `"matches": false` (only G1's halts): report the reading against its reference.

A rerun on the same card continues the JSON and skips every state already in it; teachers are regenerated (~30-40 s
each) and a student with any state missing is retrained in full.

## What to copy into the report

Every line starting `[C1]`, `[pile W0 seed`, `[W0]`, `[G1]`, `[G3]`, every `[<task>/<teacher> teacher]` line, every
state line pair, every `<n> steps in <s> s` line (without the curve if it is long), every `[stage_<x>] done` line, the
final `[score]` line, any `STOP:` or skip line, plus the `[setup ...]` lines, the VPD fetch's time, the time from
`start()` to the first cell, and the card `nvidia-smi` reports.

## Stall rule

Read the log every 1-2 min. Known long steps: the Pile train stream (~3 min, prints `[data] train rows ...` when done),
task contexts (~1 min), each adversary read (~1 min, prints its state line when done). During a 5,000-step student a
`[component training ...] step <n>` line appears every 500 steps (~50 s). Outside those, if the log has not grown for
3 minutes, check `nvidia-smi` and the process; if it is hung, stop, report, and terminate. Never sleep blindly. If
the plugin loses the pod (connection refused, port moved), stop and tell the lead at once; on a WebSocket closure run
`status(instance=...)`, then `create_kernel(instance=...)` at once.
