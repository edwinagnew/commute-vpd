# Pod-runner brief: teacher_training E41 Amendment 1, the emoticon step-10 copy at batch orders 1 and 2 (scored)

Plan: `reports/plans/2026-10-28_plan_e41-teacher-path.md` → "Amendment 1". Code:
`experiments/teacher_training/path_amend1.py` (imports `run.py` and `amend4.py` beside it). Predictions:
`experiments/teacher_training/results/predictions_e41_amend1.json` (sha256 `c4a7f6601bc6bca520e82caacbf06e0cfb786cfbd815b8f05a663c9db00b4dd4`).
The run refuses to start without the predictions file, and also if its `config` or `e41_amend1` entries differ from
`run.py`'s `CFG` or `path_amend1.py`'s `AM`.

```
Experiment: teacher_training   Run: E41 Amendment 1, scored (its own smoke: C1, G1, the do-nothing read and G2 come in
            its first ~7 min, before either scored copy)
Command: [ -L experiments/teacher_training/results/large ] && rm experiments/teacher_training/results/large
         mkdir -p experiments/teacher_training/results/large
         (.venv/bin/python experiments/teacher_training/path_amend1.py cuda; echo "=== FINISHED (exit $?) — results in <repo>/experiments/teacher_training/results ===") 2>&1 | tee -a experiments/teacher_training/results/large/run_e41_amend1.log /proc/1/fd/1
         (start it as a background subprocess; no execute() or wait() over 5 min, re-arm instead)
Expected: about 12 min of run on the H100, + set-up and downloads; GPU memory under 40 GB; CPU RAM ~8 GB
Outputs: results/teacher_training_e41_amend1.json (rewritten after every reading; "score" and "done" at the end) -> bring home
         results/large/run_e41_amend1.log -> bring home
         No .pt files: the run saves no states.
Download target: results/large/run_e41_amend1.log goes to experiments/teacher_training/results/e41a1_dl/run_e41_amend1.log,
         never into results/large (a symlink into commute-data: download() there is refused and a mv waits at a
         prompt). Move it into results/large only after the pod is terminated. The JSON goes to its own place, results/.
Inputs not in git (each results/large below is a symlink that arrives on the pod dangling: stage by hard link into a
         path ending in results/large/, then on the pod replace each symlink, including
         experiments/teacher_training/results/large, with a real directory before any mkdir or tee;
         sync(include=[<path ending in results/large>]); check sha256 on both sides; delete the staging directories on
         both sides afterwards):
         experiments/transport_test/results/large/{alive.pt 361 KB, windows.pt 156 KB}
         artifacts/vpd: let pod_setup.sh fetch it from W&B on the pod (works, ~65 s, 2026-10-27); only if that
                  fails, stage it by hard link as RUN_BRIEF_amend5.md describes
         artifacts/sets/val_0_256.npy (1 MB; optional, otherwise it streams in ~1 min)
         path_amend1.py and predictions_e41_amend1.json are untracked unless the lead commits them first: check they
                  are on the pod (sha256 of the predictions file as above).
         NOT needed: the r1 windows, Pile train rows (no Pile-text copy), any .pt under teacher_training/results/large
Platform: RunPod "NVIDIA H100 80GB HBM3" directly (C1's floor 0.0748, G1, G2 and E41's seed-0 reading are that card's;
          an L40 reads ~0.21 and stops at C1). Pod commute-teacher_training-e41a1-<id>, label set through start()'s
          argument; set up with `bash cluster/pod_setup.sh`
Pod: fresh; terminate when done. If another machine of the session runs: pass instance=... to every machine-scoped
     call, and never edit remote-kernels.toml while any machine runs
Budget: $3 cap for this run; expect about $1.9 (~25 min of pod at about $4.4/h)
```

## Stages, in order, with expected time (H100)

Per state, `run.read_state` prints `[<label>] p_target ... pgd ... verdict ...` (after the harness's own
`[<label>] gain ...` line and two `[faith]` lines, one per 128-row Pile block). Then `path_amend1.read` prints
`[<label>] adversarial ratio per 128-row block <a> <b>`. For copies, `teach` prints `[<label>] gap share ... start KL
... P(target)@100 ... (<s> s)`.

| Stage | What | Expected |
|---|---|---|
| Load + C1 | model, harness, W₀ on Pile adversary seeds 0, 1, 2 (`[pile W0 seed s]`, six `[faith]` lines); run.py's `[C1] <s> s; {...}`, then path_amend1's `[C1] {"pass": ..., "pgd_floor": ...}; W0 seed-0 block KLs [0.6794, 0.7182]` | ~3 min |
| teachers | fine-tune (`[ft] checkpoint 10/1000`), `[emoticon T10]`, `[emoticon T1000]` (P, damage, size), `[G1] {...}`, `[emoticon T10] gap at W0 5.078` | ~0.7 min |
| positive control | `[emoticon T10 do-nothing]` (expected pgd 2.516, verdict fail; blocks about 1.17 / 3.79) | ~1 min |
| G2 | `[emoticon A T1000 seed 1]` (expected P 0.99790, pgd 0.990), then `[G2] ... pass True` | ~2.2 min |
| scored | `[emoticon A T10 seed 1]`, then `[emoticon A T10 seed 2]` (expected P ~0.999, pgd ~1.45-1.65, gap share ~0.001) | ~4.5 min |
| Score | `[score] {...}` then `done` | seconds |

## Early stop for the control

C1 runs first and alone. If its `[C1] {"pass": false, ...}` line appears, the run halts with a `STOP:` line. Nothing
else runs, so report the line and terminate. The same goes for G1 and G2, which stop the run themselves.

## Downloads (the JSON into results/, the log into results/e41a1_dl/, never results/large; check the local layout after the first, since download() of a directory nested it, 2026-10-19)

- After `[G2] ...`: `teacher_training_e41_amend1.json` (to `results/`) and `run_e41_amend1.log` (to `results/e41a1_dl/`).
- After each `[emoticon A T10 seed <s>] gap share ...` line: the JSON (to `results/`).
- Before `terminate()`: the JSON (with `"done": true` and `score`) and `run_e41_amend1.log`. Check sha256 on both
  sides, then terminate. Terminate before any local file move; move `results/e41a1_dl/run_e41_amend1.log` into
  `results/large/` afterwards.

## Stop rules (the run halts itself with a `STOP:` line; report that line and the numbers named here, then terminate)

- **C1**: `STOP: C1: W0 check <bool>, floor <f> (expected 0.0748 within 10%)`. Report both `[C1]` lines.
- **G1**: `STOP: G1: the regenerated teachers differ from E41's: {...}`. Report the `[emoticon T10]`,
  `[emoticon T1000]` and `[G1]` lines.
- **G2**: `STOP: G2: T1000 copy at seed 1 P <p> ratio <r>; Amendment 4 0.9979, 0.99007`. Report the copy's lines and
  `[G2]`.
- **Seed not applied (review 2026-10-28; G2's tolerance cannot catch it)**: stop, download and terminate if the G2 copy
  reads about P 0.99675 / ratio 0.959 (E41's seed-0 values) instead of 0.99790 / 0.990, or if either T10 copy's
  `start KL` reads 5.42720 (E41 seed 0's first batch). The H100 path reproduces to four digits, so a match means the
  batch-order seed was ignored.
- An `AssertionError` naming `[faith] lines` or `block KLs` is the per-block reading going wrong (the harness logged a
  different number of blocks, or the parsed values do not give the stored ratio): a code stop. Report it with the last
  ten log lines.
- A results file written on another card or W₀ floor makes the run exit at once, naming both. Move the file aside
  only if the lead says so.
- A Python traceback is a code stop: report it with the last ten log lines. Do not retry.

A rerun on the same card keeps C1, the do-nothing read, G2's copy and every finished seed, and regenerates the
fine-tune (~30 s).

## What to copy into the report

- every `[C1]`, `[pile W0 seed`, `[G1]`, `[G2]` line, and the W₀ seed-0 block KLs;
- the `[emoticon T10]` and `[emoticon T1000]` teacher lines and the do-nothing read line with its per-block line;
- for every copy, the read line (p_target, pgd, stochastic, teacher_gap, damage, size, delta_share, verdict), its
  per-block line and its `gap share ... start KL ... P(target)@100` line;
- the `[score]` line; any `STOP:` line;
- the `[setup ...]` lines, the VPD fetch's time and rate, and the time from `start()` to the first cell;
- the card `nvidia-smi` names; the pod ID; the minutes and dollars of pod.

## Stall rule

Read the log every 1-2 min. The known quiet steps are each copy's training: it logs at component-training steps 1-30
and every 500, so a copy can be quiet for ~40 s. Outside those, if the log has not grown for 3 minutes, check
`nvidia-smi` and the process. If it is hung, stop, report and terminate. Never sleep blindly. If the plugin loses the
pod (connection refused, port moved), stop and tell the lead at once. On a WebSocket closure, run
`status(instance=...)`, then `create_kernel(instance=...)` at once.
