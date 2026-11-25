# Pod-runner brief: broad_fragility E39 Amendment 2, why the broad failure happens (scored)

Plan: `reports/plans/2026-10-27_plan_broad-fragility.md` → "Amendment 2 (2026-10-28)". Code:
`experiments/broad_fragility/amend2.py` (imports `run.py` beside it), `src/commute/rank_cap.py` (new). Predictions:
`experiments/broad_fragility/results/predictions_amend2.json`, sha256
`ff4cce5c14b5bcb018e65747e33e865a7da7f4821d50f3305d2cac0d21cfd961`. The run refuses to start without that file, or if
its `config` differs from the code's.

```
Experiment: broad_fragility   Run: Amendment 2, scored (its own smoke: stage 0 is three stop-gates in the first ~8 min)
Command: [ -L experiments/broad_fragility/results/large ] && rm experiments/broad_fragility/results/large
         mkdir -p experiments/broad_fragility/results/large experiments/broad_fragility/results/a2_dl
         (.venv/bin/python experiments/broad_fragility/amend2.py --device cuda; echo "=== FINISHED (exit $?) ===") 2>&1 | tee -a experiments/broad_fragility/results/large/amend2_stdout.log /proc/1/fd/1
         (start it as a background subprocess; no execute() or wait() over 5 min, re-arm instead; amend2.py also
         writes results/large/amend2.log)
Expected: about 40 min of run on the H100 (stage 0 ~8, move ~3, label_adam ~4, label_sgd ~5, lora ~11, cap 6-10),
          plus set-up and downloads. GPU memory under 40 GB (E39 ran the same reads); CPU RAM ~10 GB
Outputs: results/amend2.json (rewritten after every read; "status": "done" and "score" with "tree" at the end)
         results/large/amend2.log and amend2_stdout.log
         results/large/a2_label_adam.pt, a2_label_sgd.pt, a2_tt_lora_s0.pt, a2_tt_lora_s1.pt, a2_tt_sgd_r8.pt
           (bf16 offsets, ~60 MB each; a label file is missing if every state of that optimiser diverged, and
           a2_tt_sgd_r8.pt if every capped rate diverged)
         results/large/trained_step1000.pt is written too (E39's state, already home): do NOT download it
Download target: everything into experiments/broad_fragility/results/a2_dl/, one file at a time (a directory download
         nests it). Never into results/large: locally it is a symlink into commute-data, so download() there is
         refused and a mv waits at a permission prompt. Files are moved into place only after the pod is terminated.
Inputs not in git (every results/large below is a symlink locally and arrives on the pod dangling: stage by hard
         link into a path ending in results/large/, sync(include=[<that path>]), check sha256 on both sides; on the
         pod replace each dangling symlink with a real directory before any mkdir or tee; delete the staging
         directories on both sides afterwards):
         experiments/transport_test/results/large/{alive.pt 361 KB, windows.pt 156 KB}
           alive.pt   49746222ec4cfaef330e84a7d6c093d5740f764d6dc9096709d30f177e92c018
           windows.pt c19a880cdb3c7ee267a454512c3c91d5270ac9a6407124110a13d5719fba9f22
           e.g. stage_a2/experiments/transport_test/results/large/ -> sync(include=["stage_a2/experiments/transport_test/results/large"]);
           on the pod: rm the symlink experiments/transport_test/results/large, mkdir it, mv the two files in.
         artifacts/vpd: let pod_setup.sh fetch it from W&B on the pod (works, ~65 s, 2026-10-27); only if that
           fails, stage it by hard link as experiments/teacher_training/RUN_BRIEF_amend5.md describes
         artifacts/sets/val_0_256.npy (1 MB; optional). Pile train rows 0-4095 and val rows 256-511 stream from
           Hugging Face in-job (~1.5 min and ~1 min).
         amend2.py, rank_cap.py and predictions_amend2.json are untracked unless the lead commits them first: check
           all three are on the pod; the predictions sha256 must match the one above.
         NOT needed: any saved state; E39's results/large files.
Platform: RunPod "NVIDIA H100 80GB HBM3" directly (stage 0's W0 per-block readings 0.6794 / 0.7182 and spread 0.0748
          are that card's; any other card, H100 PCIe and NVL included, stops at stage 0 (i)). Over 30 min, so not the
          cluster. Pod commute-broad_fragility-a2-<id>, label set through start()'s argument; set up with
          `bash cluster/pod_setup.sh`
Pod: fresh; terminate when done. If another machine of the session runs: pass instance=... to every machine-scoped
     call, and never edit remote-kernels.toml while any machine runs
Budget: expect about $4 (~50-55 min of pod at about $4.4/h); cap $6: past it, download what is written and terminate
```

## Stages, in order, with expected time (H100)

Every state read prints one line, `[<name>] logp ... gap ... damage ... s ... check ... stochastic ... size ...
verdict <pass|between|fail> | pile {...} | N s` (~53 s each; the harness prints `[faith]`-style lines per block while
it works). After each stage: `[score after <stage>, N s] {...}`.

| Stage | Log lines to see | Expected |
|---|---|---|
| Load | `[pile W0 seed s] N s` ×3, `[load] NVIDIA H100 80GB HBM3; logp_W0 -3.1456` | ~5 min incl. data streams |
| 0 (i) | `[stage0 i] spread 0.0748 blocks [0.6794, 0.7182] (E37 ...) pass True` | ~1 min |
| 0 (ii) | `[ft] ...` to step 300; `[trained] ~90 s; {...}`; `[saved] trained_step1000.pt`; `[trained] ... check 2.2011 ... verdict fail`; `[stage0 ii] gain 0.1300 ... check 2.2011 ... pass True` | ~2.5 min |
| 0 (iii) | `[nothing_adam] ... check ~4.868 ... verdict fail`; `[stage0 iii] do-nothing check ~4.868 (E37 4.8678) pass True` | ~1 min |
| move (D) | per seed 0 then 1: `[move_s<s>] x <x> damage <d> (target ~0.234)` 4-12 times (each ~4 s), `[move_s<s>] matched True: x ~5 ...`, then `[move_s<s>] ... check ...` | ~3 min |
| label_adam (A) | `[component training lr 1e-05] step 1 ... 500 ... 1000`; `[label_adam lr 1e-05] 1000 steps in N s`; three `[label_adam lr 1e-05 step <100/300/1000>] gain ... gap ... damage ... diverged ...`; the same at 3e-05; `[label_adam] chosen lr ... step ...: gain ... (Adam fine-tune 0.1300)`; `[saved] a2_label_adam.pt`; `[label_adam] ... check ...` | ~4 min |
| label_sgd (A) | the same with `sgd` at 0.01, 0.03, 0.1 | ~5 min |
| lora (C) | `[lora r8] checkpoint 300 ...`; `[lora_s0] held-out gain 0.1479 (E39 seed 0: 0.1479)`; teacher training to step 1000; `[tt_lora_s0] trained in ~90 s; {...}`; `[saved] a2_tt_lora_s0.pt`; `[tt_lora_s0] ... check ~1.2726 ...`; the same for `lora_s1` and `tt_lora_s1`; `[data] val rows 256..512: cached ...`; `[rows2 W0 seed s] {...} N s` ×3; `[rows2] W0 spread over adversary seeds 0-2: ...`; `[trained_rows2]`, `[tt_lora_s0_rows2]`, `[tt_lora_s1_rows2]` | ~11 min |
| cap (B) | per rate 0.01, 0.03, 0.1: `[ft] step 1/10/20 ... 300` and `[sgd_r8 lr <r>] held-out gain ... diverged ...`; `[sgd_r8] chosen rate ...`; `[sgd plain lr 0.01] gain ~0.1367 (Amendment 1 0.1367); share beyond rank 8: {...}`; `[sgd_r8] gain ...; share beyond rank 8: {"pooled": ~1e-10, "max": ...}`; `[tt_sgd_r8] trained ...`; `[saved] a2_tt_sgd_r8.pt`; `[tt_sgd_r8] ... check ...` | 6-10 min |
| End | `[tree] {...}`, `FINISHED wrote ...`, `=== FINISHED (exit 0) ===` | seconds |

The capped fine-tune cuts all 24 matrices to rank 8 after every step, and its speed has never been measured. Copy
the time from `[ft] step 1` to `[ft] checkpoint 300` of the first capped rate into the report. If one rate takes
over 10 min, say so in the report. Keep the run going unless the $6 cap comes near.

## Early stop for the controls

Stage 0 runs first and alone. The run halts itself on any of its three gates with a `STOP:` line and
`"status": "stopped: ..."` in the JSON. Report the line and the numbers named here, download the JSON and both logs,
then terminate.

- **(i)** `STOP: stage 0 (i): W0 is not at E37's readings on this card`. Report the `[stage0 i]` line and the three
  `[pile W0 seed s]` lines.
- **(ii)** `STOP: stage 0 (ii): the regenerated fine-tune or its teacher training does not read as in E37`. Report
  `[stage0 ii]`, `[trained]` and the `[ft] checkpoint 300` line.
- **(iii)** `STOP: stage 0 (iii): the Adam fine-tune's do-nothing does not read as in E37`. Report `[nothing_adam]`
  and `[stage0 iii]`.

Not stops, but put them at the top of the report:

- **G_C**: `[tt_lora_s0]`'s check is not within 3% of 1.2726, or `[lora_s0] held-out gain` is not within 0.005 of
  0.1479. The `[score after lora]` line then shows `"G_C": false`.
- **The cap failed**: the `[sgd_r8] ... share beyond rank 8` line's `max` is over 1e-4. That is a code fault; the
  score marks P11 unscored.
- **A seed of the move did not match**: `[move_s<s>] matched False`.

A Python traceback is a code stop. Report it with the last ten log lines, download what is written (the JSON, both
logs, any `.pt` not yet home), terminate, and do not retry. A rerun on the same card resumes: stage 0 is redone
(~8 min) and finished stages are skipped. Only rerun if the lead says so.

## Downloads (all into `experiments/broad_fragility/results/a2_dl/`, one file at a time; check the local layout after the first)

- After `[stage0 iii]`: `results/amend2.json`.
- After each `[score after <stage>]` line: `results/amend2.json`.
- After each `[saved] a2_<name>.pt` line: that file from `results/large/`, while the next stage runs. Do not
  download `trained_step1000.pt`.
- Before `terminate()`: the final `amend2.json` (`"status": "done"`), then `results/large/amend2.log` and
  `results/large/amend2_stdout.log`. Check the sha256 of every file on both sides. **Terminate as soon as the final
  JSON's sha256 matches**; bring a log or `.pt` still missing first only if that takes under a minute.

After `terminate()`, never before: move `a2_dl/amend2.json` to `experiments/broad_fragility/results/amend2.json`. Move
the `.pt` files and the two logs to `experiments/broad_fragility/results/large/`. They land in
`~/commute-data/results-large/broad_fragility/`; do not overwrite E39's `run.log` or `amend1.log`. Check sha256 again
after the move, then delete `a2_dl/` and the `stage_a2/` staging directory. Never `git add` them.

## What to copy into the report

- every `[load]`, `[pile W0 seed`, `[stage0 i/ii/iii]`, `[rows2 W0 seed`, `[rows2]` and `[saved]` line, and the
  `[ft] checkpoint 300` line of each fine-tune;
- every state line (`[trained]`, `[nothing_adam]`, `[move_s0]`, `[move_s1]`, `[label_adam]`, `[label_sgd]`,
  `[tt_lora_s0]`, `[tt_lora_s1]`, the three `_rows2` reads, `[tt_sgd_r8]`) with its seconds;
- each move's `matched` line and its x/damage trail; the `[label_<opt> lr <r> step <n>]` gain lines and the `chosen`
  lines; `[lora_s0]`/`[lora_s1]` gains; the `[sgd_r8 lr <r>]` grid lines, `[sgd_r8] chosen`, the two `share beyond
  rank 8` lines; every `trained in N s` line;
- every `[score after ...]` line and the final `[tree]` block; any `STOP:` line;
- the time per capped rate (above); the `[setup ...]` lines, the VPD fetch's time and rate, the time from `start()`
  to the first cell;
- the card `nvidia-smi` names, the pod ID, the minutes and dollars of pod, and the sha256 of every file brought home.

## Stall rule

Read the log every 1-2 min. Known quiet stretches: the data streams (~1.5 min before `[data] train rows ...`, ~1 min
before `[data] val rows 256..512`); each harness read (~53 s); component training and teacher training between their
step-30 and step-500 lines (~40-50 s); a W0 harness pass on the second rows (~55 s per seed). Outside those, if the log
has not grown for 3 minutes, check `nvidia-smi` and the process. If it is hung, report and terminate. Never sleep
blindly. Set-up deadline: if the first cell has not run 15 min after `start()`, report and terminate. If the plugin
loses the pod (connection refused, port moved), stop and tell the lead at once; do not retry in a loop. On a
WebSocket closure, run `status(instance=...)`, then `create_kernel(instance=...)` at once.
