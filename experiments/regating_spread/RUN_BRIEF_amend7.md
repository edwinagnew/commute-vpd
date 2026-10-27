# Pod-runner brief: regating_spread (E36) Amendment 7 (scored)

Plan: `reports/plans/2026-10-26_plan_regating-without-cues.md` → "Amendment 9". Code: `experiments/regating_spread/amend7.py`
(commit ca328bc; its docstring lists every rule and expected value). Predictions: `results/predictions_amend7.json`,
filed and committed (51b9aa2) before the run; the run refuses to start without it and refuses to refile it.

```
Experiment: regating_spread   Run: Amendment 7, scored (its own smoke: gate A0 at step 10 comes ~1 min after load)
Command: mkdir -p experiments/regating_spread/results/large
         (.venv/bin/python experiments/regating_spread/amend7.py --device cuda; echo "=== FINISHED (exit $?) — results in <repo>/experiments/regating_spread/results ===") 2>&1 | tee -a experiments/regating_spread/results/large/run_amend7.log /proc/1/fd/1
         (start it as a background subprocess; no execute() or wait() over 5 min, re-arm instead)
Expected: A6 (same harness, same fine-tune, same 262,144 fitted positions) took 123 s after load on an H100; A7 adds
          two bin rules and a 32-neighbour text rule, so expect ~3-6 min of run, + set-up and downloads. GPU memory as
          A6 (fits an H100 80 GB); CPU RAM ~8 GB
Outputs: results/regating_spread_amend7.json (rewritten after every step; "score" and status "done" at the end) -> bring home
         results/large/amend7.log (the script's own log) and results/large/run_amend7.log (tee) -> bring home
         No state files are written.
Inputs not in git (sync with include=[<the gitignored directory itself>], then check sha256 on both sides):
         experiments/transport_test/results/large   (alive.pt 361 KB; windows.pt is fine to send too)
         artifacts/vpd (3.0 GB: s-55ea3f9b/{final_config.yaml, model_400000.pth}, t-9d2b8f02/{model_config.yaml,
                  final_config.yaml, ckpt/model_step_99999.pt}). FIRST let pod_setup.sh fetch them from W&B (its "VPD
                  files" step) and report time and rate. ONLY if that fetch fails (a goodfire/spd 403 was seen on
                  2026-10-26), send them: `mkdir -p stage_vpd/artifacts && cp -al "$(readlink -f artifacts)/vpd"
                  stage_vpd/artifacts/`, sync(include=["stage_vpd/artifacts"]), on the pod `mkdir -p /root/artifacts
                  && mv stage_vpd/artifacts/vpd /root/artifacts/vpd`, sha256 the .pth/.pt files on both sides, then
                  delete stage_vpd on both sides. Time the first minute of upload; tell the lead at once if 3 GB will
                  take over ~20 min.
         Pile rows (train and held-out) stream in-job via commute.data.pile_rows.
         Tracked and arriving with the plain sync(): amend1-7.py, run.py, results/regating_spread_amend6.json (gate A0's
         reference), results/predictions_amend7.json, src/commute/.
Platform: RunPod "NVIDIA H100 80GB HBM3" (or H100 NVL/PCIe) directly: gate A0 compares against A6's H100 numbers
          within 0.02, so another card can stop the run at A0 for a reason that is not a bug. Pod
          commute-regating_spread-amend7-<id>, label through start()'s argument; set up with `bash cluster/pod_setup.sh`
Pod: fresh; terminate when done. Only machine of this session.
Budget: $1.50 cap for this run; expect under $1 (~15-25 min of pod)
Watch for: see below
```

## Stages and log lines

| Stage | Log | Expected |
|---|---|---|
| Set-up | pod_setup.sh's "idle watchdog started; runpodctl present, RUNPOD_POD_ID set, RUNPOD_API_KEY set" | copy this line into your report verbatim: the watchdog (commit 218ea63) is untested |
| Load + task | `[task] sample: ...` (lowercased Pile text), `[ft] step ...`, `[ft] checkpoint 10/100/300` | ~1-2 min |
| Step 10 | `[step 10] R_near_g0 ... R_bin16q ... R_bin16w ... R_fix ... R_text4_k32 ... A0 True C_bin ... C_fit ...` | R_near_g0 ≈ 0.89 (A0) |
| Step 100 | `[step 100] ...` | R_near_g0 ≈ 0.78 |
| Step 300 | `[step 300] ...`, then `score {...}`, `FINISHED wrote ...` | R_near_g0 ≈ 0.76 |

Stop rules:
- If `A0 False` (the run stops itself with "stopped at A0 (step N)"): bring home the JSON and both logs, terminate,
  report the R_near_g0 it read against A6's (0.89, 0.78, 0.76). Do not rerun or edit code.
- A Python error: bring home the logs, terminate, report the traceback. Do not edit code.
- C_bin or C_fit false is NOT a stop: the run records B1 as "unreadable" and finishes. Let it.
- No new log line for 5 min after the step-10 line: read `nvidia-smi` and the log once, tell the lead, keep going
  unless it is clearly hung (then bring home what exists and terminate).

## Downloads (into the matching parent; check the local layout after the first, since download() of a directory nested it)

- After `[step 10]`: `regating_spread_amend7.json`.
- Before `terminate()`: the JSON (status "done" and "score", or the stop status), `results/large/amend7.log`,
  `results/large/run_amend7.log`. `results/large` is a symlink into `~/commute-data` and download() there is refused:
  download the two logs to `experiments/regating_spread/results/amend7_dl/`, check sha256 of all three files on both
  sides, terminate, and only then move the logs into `results/large/` (a refused `mv` must never hold a live pod).

## Report back

The three `[step N]` lines verbatim, the `score` line, the set-up watchdog line, the pod's total cost, and the local
paths of the three files. Commit nothing; the lead commits.
