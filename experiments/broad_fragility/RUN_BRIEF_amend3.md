# Pod-runner brief (DRAFT, for the lead): broad_fragility E39 Amendment 3, a rank-8 LoRA trained by plain SGD (scored)

Plan: `reports/plans/2026-10-27_plan_broad-fragility.md` → "Amendment 3 (2026-10-29)". Code:
`experiments/broad_fragility/amend3.py` (imports `run.py` beside it), `src/commute/harness.py` (`on_rows`, new).
Predictions: `experiments/broad_fragility/results/predictions_amend3.json`, sha256
`54351edeb6c4b1388bbe6879f5efa6f711d6ca5eabc2673790048abf405152df` (filed 2026-10-29 10:58:55). The run
refuses to start without that file, or if its `config` differs from the code's.

```
Experiment: broad_fragility   Run: Amendment 3, scored (its own smoke: stage 0 is two stop-gates in the first ~8 min)
Command: [ -L experiments/broad_fragility/results/large ] && rm experiments/broad_fragility/results/large
         mkdir -p experiments/broad_fragility/results/large experiments/broad_fragility/results/a3_dl
         (.venv/bin/python experiments/broad_fragility/amend3.py --device cuda; echo "=== FINISHED (exit $?) ===") 2>&1 | tee -a experiments/broad_fragility/results/large/amend3_stdout.log /proc/1/fd/1
         (start it as a background subprocess; no execute() or wait() over 5 min, re-arm instead; amend3.py also
         writes results/large/amend3.log)
Expected: about 28 min of run on the H100 (load ~4, stage 0 ~4, rows2 ~11, label ~3, lora_sgd ~7), plus set-up and
          downloads. GPU memory under 40 GB (A2 ran the same reads with more teachers); CPU RAM ~10 GB
Outputs: results/amend3.json (rewritten after every read; "status": "done" and "score" with "tree" at the end)
         results/large/amend3.log and amend3_stdout.log
         results/large/a3_label_sgd_1000.pt, a3_tt_lora_sgd.pt (bf16 offsets, ~60 MB each; a3_tt_lora_sgd.pt is
           missing if no rate of the grid reached the gain, or if the job stopped earlier)
         results/large/trained_step1000.pt is written too (E39's state, already home): do NOT download it
Download target: everything into experiments/broad_fragility/results/a3_dl/, one file at a time (a directory download
         nests it). Never into results/large: locally it is a symlink into commute-data, so download() there is
         refused and a mv waits at a permission prompt. Files are moved into place only after the pod is terminated.
Inputs not in git (every results/large below is a symlink locally and arrives on the pod dangling: stage by hard
         link into a path ending in results/large/, sync(include=[<that path>]), check sha256 on both sides; on the
         pod replace each dangling symlink with a real directory before any mkdir or tee; delete the staging
         directories on both sides afterwards):
         experiments/transport_test/results/large/{alive.pt 361 KB, windows.pt 156 KB}
           alive.pt   49746222ec4cfaef330e84a7d6c093d5740f764d6dc9096709d30f177e92c018
           windows.pt c19a880cdb3c7ee267a454512c3c91d5270ac9a6407124110a13d5719fba9f22
           e.g. stage_a3/experiments/transport_test/results/large/ -> sync(include=["stage_a3/experiments/transport_test/results/large"]);
           on the pod: rm the symlink experiments/transport_test/results/large, mkdir it, mv the two files in.
         artifacts/vpd: let pod_setup.sh fetch it from W&B on the pod (~65 s, 2026-10-27); only if that fails, stage
           it by hard link as experiments/teacher_training/RUN_BRIEF_amend5.md describes
         artifacts/sets/val_0_256.npy (1 MB; optional). Pile train rows 0-4095 and val rows 256-511 stream from
           Hugging Face in-job (~1.5 min and ~1 min).
         NOT needed: any saved state. Every state is regenerated in the job (the plan says why: 241 MB of saved
           states would take ~50 min at the ~80 KB/s uplink).
         amend3.py, harness.py and predictions_amend3.json must be the committed versions: check the predictions
           sha256 on the pod matches the one above.
Platform: RunPod "NVIDIA H100 80GB HBM3" directly (stage 0's W0 per-block readings 0.6794 / 0.7182 and spread 0.0748
          are that card's; any other card, H100 PCIe and NVL included, stops at stage 0 (i)). Over 30 min, so not the
          cluster. Pod commute-broad_fragility-a3-<id>, label set through start()'s argument; set up with
          `bash cluster/pod_setup.sh`
Pod: fresh; terminate when done. If another machine of the session runs: pass instance=... to every machine-scoped
     call, and never edit remote-kernels.toml while any machine runs
Budget: expect about $1.8 (~40 min of pod at about $2.7/h); cap $3: past it, download what is written and terminate
```

## Stages, in order, with expected time and cost (H100, about $2.7/h)

Every state read prints one line, `[<name>] logp ... gap ... damage ... s ... check ... stochastic ... size ...
verdict <pass|between|fail> | pile {...} | N s` (~54 s each). After each stage: `[score after <stage>, N s] {...}`.

| Stage | Log lines to see | Expected | Time, cost |
|---|---|---|---|
| Load | `[pile W0 seed s] N s` ×3, `[load] NVIDIA H100 80GB HBM3; logp_W0 -3.1456` | as A2 | ~4 min, $0.18 |
| 0 (i) | `[stage0 i] spread 0.0748 blocks [0.6794, 0.7182] (E37 ...) pass True` | as A2 | ~1 min, $0.05 |
| 0 (ii) | `[ft] ...` to step 300; `[trained] ~90 s`; `[saved] trained_step1000.pt`; `[trained] ... check 2.2011 ... verdict fail`; `[stage0 ii] gain 0.1300 ... pass True` | check 2.2011 | ~3 min, $0.13 |
| rows2 | `[data] val rows 256..512 ...`; `[pile W0 seed s, val rows 256..512] {...} N s` ×3; `[rows2] W0 spread ... 0.0621`; `[trained_rows2] ... check ~1.8826 ... verdict fail`; `[rows2 control] ... pass True`; `[sgd] regenerated: held-out gain 0.1367 (filed 0.13671)`; `[sgd_r8] regenerated: held-out gain 0.1314 (filed 0.13142)`; `[tt_sgd] trained in ~95 s`, `[tt_sgd_rows2] ... check ...`; the same for `sgd_r8` | P15: both 1.4-2.2 | ~11 min, $0.50 |
| label | `[label_sgd lr 0.03] 1000 steps in ~75 s`; `[label_sgd step 100] gain 0.1253 ...`; `[label_sgd step 1000] gain ~0.1537 ... damage ~0.229`; `[label_sgd G_label] ... pass True`; `[saved] a3_label_sgd_1000.pt`; `[label_sgd_1000] ... check ...` | P16: 1.6-2.6 | ~3 min, $0.14 |
| lora_sgd | `[lora r8] checkpoint 300 ...`; `[tt_lora_adam] trained in ~90 s`; `[tt_lora_adam] ... check ~1.2726 ... verdict pass`; `[lora_ref] gain 0.1479 ... pass True`; per rate 0.03, 0.1, 0.3, 1, 3: `[lora r8] checkpoint 300 ...`, `[lora_sgd lr <r>] held-out gain ... diverged ...`; `[lora_sgd] chosen rate ...`; `[lora_sgd] grid gains {...} range [...]`; `[nothing_lora_sgd] ... check ...`; `[tt_lora_sgd] trained ...`; `[saved] a3_tt_lora_sgd.pt`; `[tt_lora_sgd] ... check ...` | do-nothing 2-4; P17: 1.15-1.5 | ~7 min, $0.32 |
| End | `[tree] {...}`, `FINISHED wrote ...`, `=== FINISHED (exit 0) ===` | | seconds |

Set-up (~10 min, $0.45) and downloads (~3 min, $0.14) bring the total to about 40 min, $1.8.

## Stops

The run halts itself with a `STOP:` line, `"status": "stopped: ..."` in the JSON, and a non-zero exit. Report the line
and the numbers named here, download the JSON and both logs (and any `a3_*.pt` written), then terminate.

- **Stage 0 (i)** `STOP: stage 0 (i): W0 is not at E37's readings on this card`: report `[stage0 i]` and the three
  `[pile W0 seed s]` lines.
- **Stage 0 (ii)** `STOP: stage 0 (ii): ...`: report `[stage0 ii]`, `[trained]` and the `[ft] checkpoint 300` line.
- **rows2 control** `STOP: rows2 control: ...`: report `[rows2]`, `[trained_rows2]` and `[rows2 control]`.
- **A regenerated gain** `STOP: the regenerated sgd (or sgd_r8) fine-tune's gain ...`: report the `[... regenerated]`
  line.
- **The filed stop rule** `STOP: rows2: both SGD teacher-trained decompositions pass on the second rows (the plan's stop
  rule)`. This is a filed outcome, not a fault: report `[tt_sgd_rows2]`, `[tt_sgd_r8_rows2]` and the score line.
- **The LoRA control** `STOP: lora_sgd control: ...`: report `[tt_lora_adam]` and `[lora_ref]`.

Not stops, but put them at the top of the report: `[label_sgd G_label] ... pass False`; `[lora_sgd] no rate within
30% ...` (then copy the grid gains line); a `[nothing_lora_sgd]` check under 1.61 (G_nothing fails).

A Python traceback is a code stop. Report it with the last ten log lines, download what is written, terminate, and do
not retry. A rerun on the same card repeats stage 0 (~8 min) and skips finished stages. Only rerun if the lead says so.

## Downloads (all into `experiments/broad_fragility/results/a3_dl/`, one file at a time; check the local layout after the first)

- After `[stage0 ii]`: `results/amend3.json`.
- After each `[score after <stage>]` line: `results/amend3.json`.
- After each `[saved] a3_<name>.pt` line: that file from `results/large/`, while the next stage runs. Do not
  download `trained_step1000.pt`.
- Before `terminate()`: the final `amend3.json` (`"status": "done"` or `"stopped: ..."`), then
  `results/large/amend3.log` and `results/large/amend3_stdout.log`. Check the sha256 of every file on both sides.
  **Terminate as soon as the final JSON's sha256 matches**; bring a log or `.pt` still missing first only if that
  takes under a minute.

After `terminate()`, never before: move `a3_dl/amend3.json` to `experiments/broad_fragility/results/amend3.json`. Move
the `.pt` files and the two logs to `experiments/broad_fragility/results/large/`. They land in
`~/commute-data/results-large/broad_fragility/`; do not overwrite any existing file there. Check sha256 again after the
move, then delete `a3_dl/` and the `stage_a3/` staging directory. Never `git add` them.

## What to copy into the report

- every `[load]`, `[pile W0 seed`, `[stage0 i/ii]`, `[rows2]`, `[rows2 control]`, `[... regenerated]`, `[lora_ref]`,
  `[label_sgd G_label]` and `[saved]` line, and the `[ft]` / `[lora r8] checkpoint 300` line of each fine-tune;
- every state line (`[trained]`, `[trained_rows2]`, `[tt_sgd_rows2]`, `[tt_sgd_r8_rows2]`, `[label_sgd_1000]`,
  `[tt_lora_adam]`, `[nothing_lora_sgd]`, `[tt_lora_sgd]`) with its seconds, and the `[faith]` per-block lines of each
  second-rows read;
- the `[label_sgd step <n>]` lines; each `[lora_sgd lr <r>]` grid line, `[lora_sgd] chosen` and the grid gains line;
  every `trained in N s` line;
- every `[score after ...]` line and the final `[tree]` block; any `STOP:` line;
- the `[setup ...]` lines, the VPD fetch's time and rate, the time from `start()` to the first cell;
- the card `nvidia-smi` names, the pod ID, the minutes and dollars of pod, and the sha256 of every file brought home.

## Stall rule

Read the log every 1-2 min. Known quiet stretches: the data streams (~1.5 min before `[data] train rows ...`, ~1 min
before `[data] val rows 256..512`); each harness read (~54 s); each W0 pass on the second rows (~53 s per seed);
component training and teacher training between their step lines (~40-50 s). Outside those, if the log has not grown
for 3 minutes, check `nvidia-smi` and the process. If it is hung, report and terminate. Never sleep blindly. Set-up
deadline: if the first cell has not run 15 min after `start()`, report and terminate. If the plugin loses the pod
(connection refused, port moved), stop and tell the lead at once; do not retry in a loop. On a WebSocket closure, run
`status(instance=...)`, then `create_kernel(instance=...)` at once.
