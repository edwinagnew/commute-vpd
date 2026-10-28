# Pod-runner brief: teacher_training Amendment 7, the LoRA's Pile-text copies on r1 and r0 to 10,000 steps (scored)

Plan: `reports/plans/2026-10-23_plan_teacher-training.md` → "Amendment 7 (2026-10-28, commute lead)". Code:
`experiments/teacher_training/amend7.py` (its docstring lists the stages; it imports `run.py`, `amend3.py`, `amend4.py`
and `amend6.py` beside it). Predictions: `experiments/teacher_training/results/predictions_amend7.json`, sha256
`b42d8e6ae4ace67dc70c0fd5cfd41b75ff2fbe6c457de28dc7562f5f033d7748`. The run refuses to start without that file, or if
its `config` or `amend7` entries differ from `run.py`'s `CFG` or `amend7.py`'s `AM`.

```
Experiment: teacher_training   Run: Amendment 7, scored (its own smoke: C1, the W0 plain-text control, G1 and G2 run in the first ~12 min)
Command: [ -L experiments/teacher_training/results/large ] && rm experiments/teacher_training/results/large
         mkdir -p experiments/teacher_training/results/large
         (.venv/bin/python experiments/teacher_training/amend7.py cuda; echo "=== FINISHED (exit $?) ===") 2>&1 | tee -a experiments/teacher_training/results/large/run_amend7.log /proc/1/fd/1
         (start it as a background subprocess; no execute() or wait() over 5 min, re-arm instead)
Expected: about 50 min of run on the H100 (stages below), + ~15 min set-up and downloads; GPU memory under 40 GB
          (Amendment 6 ran the same students); CPU RAM ~10 GB
Outputs: results/teacher_training_amend7.json (rewritten after every state; "score" and "done" at the end)
         results/large/run_amend7.log
         results/large/r1_t534_v369/amend7_lora_B_seed0_step10000.pt and
         results/large/r0_t352_v310/amend7_lora_B_seed0_step10000.pt (bf16 offsets, ~60 MB each)
         NOT needed: results/large/<task>/lora_teacher_factors.pt (amend3.lora_teacher writes them; Amendment 3 has them)
Download target: experiments/teacher_training/results/amend7_dl/<same subpath as under results/>, one file at a time
         (a directory download nests it). Never into results/large: locally it is a symlink into commute-data, so
         download() there is refused and a mv waits at a permission prompt. The JSON also lands in amend7_dl/ first.
Inputs not in git (each results/large below is a symlink locally and arrives on the pod dangling: stage by hard link
         into a path ending in results/large/, sync(include=[<that path>]), check sha256 on both sides; on the pod
         replace each dangling symlink with a real directory before any mkdir or tee; delete the staging directories
         on both sides afterwards):
         experiments/transport_test/results/large/alive.pt    49746222ec4cfaef330e84a7d6c093d5740f764d6dc9096709d30f177e92c018
         experiments/transport_test/results/large/windows.pt  c19a880cdb3c7ee267a454512c3c91d5270ac9a6407124110a13d5719fba9f22
           e.g. stage_a7/experiments/transport_test/results/large/ -> sync(include=["stage_a7/experiments/transport_test/results/large"])
         experiments/task_ladder/results/large/windows_r1_t534_v369.pt  e38bc9b9bbbfccd037d224ae42b88f7238ce82aedcbd7ea93a10215a9557f0bd
         experiments/task_ladder/results/large/windows_r0_t352_v310.pt  3883c4156dc69668a9c3019a0521aa5cf0e2e55246ff92be81e5582819ca6f3d
           e.g. stage_a7/experiments/task_ladder/results/large/ -> sync(include=["stage_a7/experiments/task_ladder/results/large"])
           on the pod: rm each symlink, mkdir the directory, mv the files in.
         artifacts/vpd: let cluster/pod_setup.sh fetch it from W&B on the pod (its "VPD files" step; ~65 s on
           2026-10-27). Only if that fails, stage it by hard link as RUN_BRIEF_amend5.md describes and tell the lead the rate.
         artifacts/sets/val_0_256.npy (1 MB, optional; the harness rows stream in ~1 min without it). Pile train rows
           32,768..40,959 stream in-job, ~3 min, before G2.
         NOT needed: any saved state (E34's and Amendment 6's states are not used; the teachers are regenerated in-job).
         amend7.py and predictions_amend7.json are committed; check both are on the pod and the sha256 above matches.
Platform: RunPod "NVIDIA H100 80GB HBM3" directly (over 30 min; C1's floor 0.0748, G1's references and G2/G3's
          Amendment 6 readings are that card's; any other card, H100 PCIe and NVL included, stops at C1 or G2).
          Pod commute-teacher_training-amend7-<id>, label set through start()'s argument; set up with
          `bash cluster/pod_setup.sh`
Pod: fresh; terminate when done. If another machine of the session runs, pass instance=... to every machine-scoped
     call and never edit remote-kernels.toml while any machine runs
Budget: expect about $5 (~65 min of pod at about $4.7/h); cap $7: past it, download what is written and terminate.
        Before start(), check the session's spend against REMOTE_KERNELS_BUDGET ($40) and tell the lead if under $8 remains
Watch for: stages, stop rules below
```

## Stages, in order, with expected time (H100; Amendment 6's log: 5,000 student steps in 445 s)

State lines come in pairs: run.read_state's `[<label>] pgd ... p_target ... verdict ...` (pgd is None without the
adversary) and `[<label>] pile_damage <x> pile_gap <y> [faith <v>]`. Labels are `<task>/lora/nothing` and
`<task>/lora/B0@<step>`. Training prints `[component training lr 1e-05] step <n>: ...` at steps 1-30 and every 500,
then `[<task>/lora/B0] <n> steps in <s> s; P-bar {...}`.

| Stage | Log lines | Expected |
|---|---|---|
| Load + C1 | `[pile W0 seed 0/1/2]`, `[C1] <s> s; {...}` (pgd_floor about 0.0748) | ~4 min |
| W0 control | `[W0] {"pile_damage": ...}`: must be <= 1e-4 (expected 0) | seconds |
| r1 teacher (G1) | r1 task context (~1 min); `[r1_t534_v369 LoRA teacher]`; `[r1_t534_v369/lora teacher] {... "matches": true ...}` | ~1.5 min |
| r1 do-nothing | `r1_t534_v369/lora/nothing` pair, no adversary | seconds |
| Pile stream | `[data] train rows 32768..40960` | ~3 min |
| G2 | 1,000 steps (~90 s); `r1_t534_v369/lora/B0@1000` pair; `[G2 r1_t534_v369] {... "pass": true}` (P 0.87530, size 0.01634) | ~2 min |
| r1 to 10,000 | `step 500 ... 10000` lines (~45 s apart); `10000 steps in ~900 s`; states @5000 (adversary, ~1 min), @7500, @10000 (adversary); `[saved] r1_t534_v369/amend7_lora_B_seed0_step10000.pt`; `[G3 r1_t534_v369] {... "pass": true}`; `[r1_t534_v369] done; score so far {...}` | ~18 min |
| r0 | context (~1 min), teacher (G1), do-nothing, then as r1 without G2; `[G3 r0_t352_v310]`; `[r0_t352_v310] done` | ~20 min |
| Score | `[score] {...}` then `done` | seconds |

## Early stop for the controls (the run halts itself with a `STOP:` line; report that line and the numbers named, download the JSON and the log, terminate)

- **C1**: `STOP: C1: W0 is not at its known values; nothing read`. Report the `[C1]` line and the three `[pile W0 seed` lines.
- **W0 control**: `STOP: W0 plain-text damage <x> > 0.0001`. Report the `[W0]` line.
- **G1**: `STOP: G1: <task> LoRA teacher {...} against {...}`. Report the teacher line (needs P within 0.01, damage
  within 0.005, size within 0.003 of the reference).
- **G2**: `STOP: G2: r1_t534_v369 student B at step 1000 reads {...}, Amendment 6 {...}` (needs P within 0.002 of
  0.87530 and size within 0.0005 of 0.01634). This is the first-minutes check of the training code path.
- **G3**: `STOP: G3: <task> student B at step 5000 reads {...}` (needs P within 0.002 and ratio within 0.03 of
  Amendment 6's: r1 0.88704 and 1.0711, r0 0.81925 and 0.9555). On r1 it stops before r0 starts.
- A results file written on another card or W0 floor makes the run exit at once, naming both; move it aside only if
  the lead says so.
- A Python traceback is a code stop: report it with the last ten log lines, download what is written, terminate, and
  do not retry.

Not stops, but tell the lead at once in one line and keep watching: any `faith` other than `pass` on a `B0@5000` or
`B0@10000` line (a qualitative surprise the plan's tree sends to the user).

A rerun on the same card continues the JSON and skips every state already in it; teachers are regenerated (~40 s
each) and a student with any state missing is retrained in full (~15 min). Only rerun if the lead says so.

## Downloads (all into `experiments/teacher_training/results/amend7_dl/`, one file at a time; check the local layout after the first)

- After `[G2 ...]`, and after each `[<task>] done` line: `results/teacher_training_amend7.json`.
- After each `[saved] <task>/amend7_lora_B_seed0_step10000.pt` line: that file from `results/large/<task>/`, while the
  run goes on.
- Before `terminate()`: the final JSON (`"done": true`, with `score`), then `results/large/run_amend7.log`. Check the
  sha256 of every file on both sides. **Terminate as soon as the final JSON's sha256 matches**; bring the log or a
  `.pt` still missing first only if that takes under a minute.

After `terminate()`, never before: move `amend7_dl/teacher_training_amend7.json` to
`experiments/teacher_training/results/`. Move the log to `experiments/teacher_training/results/large/amend7/` and each
`.pt` to `experiments/teacher_training/results/large/amend7/<task>/`; they land in
`~/commute-data/results-large/teacher_training/`. Check sha256 again after the move, then delete `amend7_dl/` and the
`stage_a7/` staging directory. Never `git add` them.

## What to copy into the report

Every line starting `[C1]`, `[pile W0 seed`, `[W0]`, `[G2`, `[G3`, every `[<task>/lora teacher]` and `[<task> LoRA
teacher]` line, every state line pair, every `steps in <s> s; P-bar` line, every `[saved]` line, every `[<task>] done`
line, the final `[score]` line, any `STOP:` line, plus the `[setup ...]` lines, the VPD fetch's time, the time from
`start()` to the first cell, the card `nvidia-smi` names, the pod ID, the minutes and dollars of pod, and the sha256
of every file brought home.

## Stall rule

Read the log every 1-2 min. Known quiet stretches: the Pile train stream (~3 min, ends with `[data] train rows ...`),
task contexts (~1 min), each adversary read (~1 min, ends with its state line), and the 500-step stretches of
training (~45 s each). Outside those, if the log has not grown for 3 minutes, check `nvidia-smi` and the process; if it
is hung, report and terminate. Never sleep blindly. Set-up deadline: if the first cell has not run 15 min after
`start()`, report and terminate. If the plugin loses the pod (connection refused, port moved), stop and tell the lead
at once; do not retry in a loop. On a WebSocket closure, run `status(instance=...)`, then `create_kernel(instance=...)`
at once.
