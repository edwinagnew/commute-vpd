# Pod-runner brief: recruitment_test (E38) Amendment 2 (scored)

Plan: `reports/plans/2026-10-27_plan_e38-amend2.md`. Code: `experiments/recruitment_test/amend2.py` (its docstring
and `Cfg` give every rule and expected value; shared steps in `src/commute/{recruitment,census}.py`). Predictions:
`experiments/recruitment_test/results/predictions_amend2.json`, filed and committed (0064bd5) before the code; the run
refuses to start without it. Never edit it.

```
Experiment: recruitment_test   Run: Amendment 2, scored (stage 0 is the control and its own smoke; it runs alone first)
Commands (from the repo root on the pod; start each as a background subprocess, no execute() or wait() over 5 min):
  1. (.venv/bin/python experiments/recruitment_test/amend2.py --stages 0 --device cuda; echo "=== FINISHED (exit $?) ===") 2>&1 | tee -a experiments/recruitment_test/results/run_amend2_stage0.log /proc/1/fd/1
     Download results/amend2_stage0.json and run_amend2_stage0.log at once. Go on only if it ends "[a2 stage0] PASS"
     and exit 0.
  2. (.venv/bin/python experiments/recruitment_test/amend2.py --stages 1,2 --device cuda; echo "=== FINISHED (exit $?) ===") 2>&1 | tee -a experiments/recruitment_test/results/run_amend2_stage12.log /proc/1/fd/1
Expected: from E38's logs on an RTX 4090 (run_stage0.log, run_stage123.log): E38's stage 0 took 28 s after load;
          amendment stage 0 adds ~15 forward passes per task, so ~1 min. Step 2: loading the cached states ~10 s,
          stage 1 ~92 removals per state (E38 did ~50 in ~1.5 s) so ~30 s, stage 2 one Pile pass of 2,000 rows ~30 s
          (E38's census: 22 s) plus the row download (seconds). Under 5 min of run in all; ~15 min of pod with
          set-up. GPU memory under 10 GB; CPU RAM ~8 GB.
Outputs (bring home, into experiments/recruitment_test/results/):
          amend2_stage0.json (after step 1), amend2.json (rewritten after each state; readings P5-P7 at the end),
          amend2_census.json (small: counters per piece of the sets, P8), run_amend2_stage0.log, run_amend2_stage12.log
          results/large/amend2.log (the script's own log; download to results/amend2_dl/, move into large/ after the
          pod is gone). Do NOT download results/large/states/ (nine full weight changes, ~150 MB each, regenerated in-job) or
          results/large/full.log.
Inputs not in git (every results/large is a symlink: stage by hard link into a path ending in results/large/, sync
          with include=[that directory], replace the dangling symlink on the pod with the real directory, sha256 both
          sides; ~2.3 MB, start the sync before set-up):
          experiments/transport_test/results/large/{alive.pt, windows.pt}
          experiments/task_ladder/results/large/windows_{r1_t534_v369,r2_t359_v816}.pt
          experiments/regating_leak/results/large/{emoticon,r1_t534_v369,r2_t359_v816}_s{0,1}_shift.pt
          artifacts/vpd: let pod_setup.sh fetch it from W&B (works since 2026-10-27, ~65 s).
          Pile val rows 0..2000 stream in-job (commute.data.pile_rows).
          Tracked and arriving with the plain sync(): run.py, amend2.py, results/predictions_amend2.json,
          experiments/regating_leak/results/regating_leak.json, experiments/transport_test/results/windows.json, src/commute/.
Platform: RunPod RTX 4090 (E38 ran there; cluster GPU jobs were broken on 2026-10-27 by the shared venv). Pod
          commute-recruitment_test-amend2-<id>, label through start()'s argument; set up with `bash cluster/pod_setup.sh`.
          A cluster `debug` job is fine instead if the shared venv has been rebuilt with CUDA torch.
Pod: fresh; terminate when done. Only machine of this session.
Budget: $0.60 cap; expect ~$0.20.
```

## Log lines and expected values

| Stage | Log | Expected |
|---|---|---|
| Stage 0, per state | `[stage0 <task>/s0] T1 ... | ok` (E38's lines, nine of them) | `ok` on all nine (C0) |
| Stage 0, per task | `[stage0 <task>] empty removal diffs ... | ok` (E38's apparatus) | `ok` (C0) |
| Stage 0, per task | `[a2 stage0 <task>] C1 lo diffs {...} True | C2 max |logit diff| {... "pre": ..} True | C3 ... True | C4 ... True` | C1, C2, C3 diffs 0 exactly (tolerances 1e-3, 1e-4, 1e-5); C2 "at" > 0; C4 KL trigger >= 1 on emoticon (|busy| 210) |
| Stage 0 | `[a2 stage0] PASS` | PASS |
| Stage 1, per state | `[a2 stage1 <task>/s<k>] T0 ... T1 ... G ...` then a `trigger` and an `all` line | G 10-14 nats; the `all` line's S rho_ft equals E38's rho_S (stage1.json: 0.50, 0.54, 0.32, 0.36, 0.17, 0.22) to ~1e-3 |
| Stage 1 | `[a2 stage1] P5 ...`, `P5_all ...`, `P5_null ... P6 ... P7 ...` | readings, not stop rules |
| Stage 2 | `[census] rows ...`, `[a2 stage2 <task>] ... control True ...`, `[a2 stage2] ... P8 {...}` | control True on all three |

## Stop rules

- Any C0-C4 fail (step 1 ends "FAIL: stop and report", exit 1): bring home amend2_stage0.json and the log, terminate,
  report which check failed and its numbers. Do not run step 2, do not rerun, do not edit code.
- A Python error: bring home the logs, terminate, report the traceback. Do not edit code.
- The `all` line's S rho_ft differing from E38's rho_S by more than 0.01: not a stop (the states are regenerated and
  E38's own seeds differ by ~2%), but quote both numbers in the report.
- P5-P8 readings and the P8 control are results, not stop rules: let the run finish.
- No new log line for 3 min: read `nvidia-smi` and the log once, tell the lead; if clearly hung, bring home what exists
  and terminate.

## Before `terminate()`

Download amend2.json and amend2_census.json, check their sha256 on both sides, and check that amend2.json has the P5
key (the end of stage 1) and amend2_census.json the P8 key. Then terminate, then move amend2.log into results/large/.
