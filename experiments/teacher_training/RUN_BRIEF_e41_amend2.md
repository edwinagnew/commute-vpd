# Pod-runner brief: teacher_training E41 Amendment 2, short links along the emoticon fine-tune (scored)

Plan: `reports/plans/2026-10-28_plan_e41-teacher-path.md` → "Amendment 2". Code:
`experiments/teacher_training/path_amend2.py` (imports `run.py` and `path.py` beside it). Predictions:
`experiments/teacher_training/results/predictions_e41_amend2.json` (sha256
`08e5e05bb11029a9df88db04b7dfcfe8f25c4f722e2be7aa5e7394c9d6963db8`). The run refuses to start without the predictions
file, and also if its `config` or `e41_amend2` entries differ from `run.py`'s `CFG` or `path_amend2.py`'s `AM`.

```
Experiment: teacher_training   Run: E41 Amendment 2, scored (its own smoke: C1, G1 and PC come in its first ~5 min,
            the first link and C2 by ~7 min)
Command: [ -L experiments/teacher_training/results/large ] && rm experiments/teacher_training/results/large
         mkdir -p experiments/teacher_training/results/large
         (.venv/bin/python experiments/teacher_training/path_amend2.py cuda; echo "=== FINISHED (exit $?) — results in <repo>/experiments/teacher_training/results ===") 2>&1 | tee -a experiments/teacher_training/results/large/run_e41_amend2.log /proc/1/fd/1
         (start it as a background subprocess; no execute() or wait() over 5 min, re-arm instead)
Expected: about 17 min of run on the H100, + set-up and downloads; GPU memory under 40 GB (seven teacher copies of the
          67M model, ~2 GB, on top of E41's reads); CPU RAM ~8 GB
Outputs: results/teacher_training_e41_amend2.json (rewritten after every reading and every link; "score" and "done" at
           the end) -> bring home
         results/large/run_e41_amend2.log -> bring home
         No .pt files: the run saves no states.
Download target: results/large/run_e41_amend2.log goes to experiments/teacher_training/results/e41a2_dl/run_e41_amend2.log,
         never into results/large (a symlink into commute-data: download() there is refused and a mv waits at a
         prompt). Move it into results/large only after the pod is terminated. The JSON goes to its own place, results/
         (a real directory).
Inputs not in git (each results/large below is a symlink that arrives on the pod dangling: stage by hard link into a
         path ending in results/large/, then on the pod replace each symlink, including
         experiments/teacher_training/results/large, with a real directory before any mkdir or tee;
         sync(include=[<path ending in results/large>]); check sha256 on both sides; delete the staging directories on
         both sides afterwards):
         experiments/transport_test/results/large/{alive.pt 361 KB, windows.pt 156 KB}
           alive.pt   49746222ec4cfaef330e84a7d6c093d5740f764d6dc9096709d30f177e92c018
           windows.pt c19a880cdb3c7ee267a454512c3c91d5270ac9a6407124110a13d5719fba9f22
           e.g. stage_e41a2/experiments/transport_test/results/large/ -> sync(include=["stage_e41a2/experiments/transport_test/results/large"]);
           on the pod: rm the symlink experiments/transport_test/results/large, mkdir it, mv the two files in.
         artifacts/vpd: let pod_setup.sh fetch it from W&B on the pod (works, ~65 s, 2026-10-27); only if that
                  fails, stage it by hard link as RUN_BRIEF_amend5.md describes
         artifacts/sets/val_0_256.npy (1 MB; optional, otherwise it streams in ~1 min)
         path_amend2.py and predictions_e41_amend2.json are committed: check the predictions sha256 on the pod.
         NOT needed: the r1 windows, Pile train rows (no Pile-text copy), any .pt under teacher_training/results/large
Platform: RunPod "NVIDIA H100 80GB HBM3" directly (C1's floor 0.0748 and seed-0 KL 0.69882, PC's 2.5159 and every E41
          reference are that card's; an L40 reads a floor of ~0.21 and stops at C1; H100 PCIe or NVL would likely
          stop at C1 or PC). Pod commute-teacher_training-e41a2-<id>, label set through start()'s argument; set up
          with `bash cluster/pod_setup.sh`
Pod: fresh; terminate when done. If another machine of the session runs: pass instance=... to every machine-scoped
     call, and never edit remote-kernels.toml while any machine runs
Budget: $3 cap for this run; expect about $1.7 (~23 min of pod at about $4.4/h)
```

## Stages, in order, with expected time (H100)

Per state, `run.read_state` prints `[<label>] pgd ... stochastic ... p_target ... verdict ...` after the harness's own
`[<label>] gain ...` line and two `[faith]` lines (one per 128-row Pile block), about 53 s per state. Each link then
prints `[link T<k>] check <pass|at the line|fail> pgd ... teacher gap ... gap at W0 ... gap share ... size ... damage ... start KL ... seed <i>
student steps <n> (<s> s)` and `[score after T<k>] {...}`.

| Stage | What | Expected |
|---|---|---|
| Load + C1 | model, harness, W₀ on Pile adversary seeds 0, 1, 2 (`[pile W0 seed s]`); run.py's `[C1] <s> s; {...}`, then `[C1] {"pass": true, "pgd_floor": 0.0748..., "w0_pgd_kl": 0.69882..., ...}` | ~3 min |
| teachers | fine-tune (`[ft] checkpoint 1/3/10/30/100/300/1000`), seven `[emoticon T<k>] {...}` lines (P, damage, size, gap_w0), `[G1] {...  "pass": true}` | ~1 min |
| PC | `[emoticon T10 do-nothing] ... pgd 2.516 ... verdict fail`, then `[PC] {"pgd": 2.5159..., "pass": true, ...}` | ~1 min |
| link T1 | `[emoticon T1 do-nothing]`, component training steps 1-30 and 100, `[emoticon link T1] ...`, `[link T1] ...`, `[score after T1]` | ~2 min |
| C2 | `[C2] {"loss_step1": ~0.0005, "dw_rel": 0, "pass": true}` | seconds |
| links T3, T10, T30, T100, T300, T1000 | the same per link (T10's do-nothing is already read) | ~1-2 min each, ~10 min |
| Score | `[score] {...}` then `done` | seconds |

Expected readings: links from T10 on hold P(target) about 0.999 and gap share under 0.01; ratios at the line or pass at
T1-T10, about 1.0-1.1 from T100 on; Δ share ~1e-9. From the T30 link on, the start KL should be under 0.5 nats (E41's
chain links started 0.13 and 0.065 nats from their teachers).

## Early stop for the controls

C1 runs first and alone. Every gate halts the run itself with a `STOP: <name>: {...}` line and `"stopped"` in the JSON.
Report the line and the numbers named here, download the JSON and the log, then terminate.

- **C1**: `STOP: C1: {...}`. Report both `[C1]` lines and the three `[pile W0 seed s]` lines.
- **G1**: `STOP: G1: {...}`. Report the seven `[emoticon T<k>]` lines and `[G1]`.
- **PC**: `STOP: PC: {...}`. Report the `[emoticon T10 do-nothing]` lines and `[PC]`. This means E41's references do
  not hold on this card.
- **C2**: `STOP: C2: {...}`. Report `[C2]` and the `[link T1]` line: the warm start does not reproduce its start.
- **Δ**: `STOP: link T<k>: Delta share ... (Delta is frozen: a bug)`. Report that link's lines.
- **Warm start lost (not in code)**: if any link from T30 on reads a start KL above 4 nats (a start from W₀ reads
  5.3-5.4), the warm start was dropped: stop, download and terminate.
- A results file written on another card or W₀ floor makes the run exit at once, naming both. Move the file aside
  only if the lead says so.
- A Python traceback is a code stop: report it with the last ten log lines. Do not retry.

A rerun on the same card keeps the do-nothing reads and redoes the fine-tune and the whole chain (~12 min).

## Downloads (the JSON into results/, the log into results/e41a2_dl/, never results/large; check the local layout after the first, since download() of a directory nested it, 2026-10-19)

- After `[C2] ...`: `teacher_training_e41_amend2.json` (to `results/`) and `run_e41_amend2.log` (to `results/e41a2_dl/`).
- After each `[score after T<k>]` line: the JSON (to `results/`).
- Before `terminate()`: the JSON (with `"done": true` and `score`) and `run_e41_amend2.log`. Check sha256 on both
  sides, then terminate. **Terminate as soon as the final JSON's sha256 matches**; bring the log first only if that
  takes under a minute. Terminate before any local file move; move `results/e41a2_dl/run_e41_amend2.log` into
  `results/large/` afterwards, check its sha256, then delete `e41a2_dl/` and the `stage_e41a2/` staging directory.
  Never `git add` them.

## What to copy into the report

- every `[C1]`, `[pile W0 seed`, `[G1]`, `[PC]`, `[C2]` line;
- the seven `[emoticon T<k>]` teacher lines and every do-nothing read line;
- for every link, its read line (p_target, pgd, stochastic, teacher_gap, damage, size, delta_share, verdict), its
  `[link T<k>]` line and its `[score after T<k>]` line;
- the final `[score]` line; any `STOP:` line;
- the `[setup ...]` lines, the VPD fetch's time and rate, and the time from `start()` to the first cell;
- the card `nvidia-smi` names; the pod ID; the minutes and dollars of pod; the sha256 of every file brought home.

## Stall rule

Read the log every 1-2 min. The known quiet stretches are each harness read (~53 s) and each link's training between
its step-30 and step-100 lines (a few seconds). Outside those, if the log has not grown for 3 minutes, check
`nvidia-smi` and the process. If it is hung, stop, report and terminate. Never sleep blindly. Set-up deadline: if the
first cell has not run 15 min after `start()`, report and terminate. If the plugin loses the pod (connection refused,
port moved), stop and tell the lead at once; do not retry in a loop. On a WebSocket closure, run
`status(instance=...)`, then `create_kernel(instance=...)` at once.
