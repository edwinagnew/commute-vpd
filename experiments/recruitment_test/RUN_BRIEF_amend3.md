# Pod-runner brief: recruitment_test (E38) Amendment 3, the restore test (scored)

Plan: `reports/plans/2026-10-27_plan_e38-amend3.md`. Code: `experiments/recruitment_test/amend3.py` (its docstring
and `Cfg` give every rule and tolerance; the hook and readings in `src/commute/recruitment.py`). Predictions:
`experiments/recruitment_test/results/predictions_amend3.json`, filed and committed (53ce9f4) before the code; the run
refuses to start without it. The rule for a state whose active pool is still too small at m0 >= 0.2 in some layer
(P9 seed class 'unscored: active pool too small', counting toward no branch) was filed before any run in
`results/predictions_amend3_pool.json`; the run refuses to start without that file too. Never edit either.

```
Experiment: recruitment_test   Run: Amendment 3, scored (the controls R0-R3 run first in the same job and stop it on a fail)
Command (from the clone root on the cluster controller):
  sbatch cluster/recruitment_amend3.sbatch
  Record the job ID. Log: cluster/logs/commute-recruitment-amend3-<id>.out (and results/large/amend3.log).
Expected: no fine-tuning. Amendment 2's stage-1,2 job (38809, L40) took 87 s from start to done, of which its
          stage 1 (~550 forward passes over the held-out windows) took 16 s. This job: model and state load ~15 s,
          the controls ~50 passes, the grid ~260 passes: ~1-2 min of run, ~3 min with queueing on an idle node. GPU
          memory under 10 GB.
Outputs (already on NFS; nothing to download): experiments/recruitment_test/results/amend3.json (rewritten after the
          controls of each task and after each grid state; P9, P10 and "reported" at the end),
          results/large/amend3.log. Copy the job's .out to experiments/recruitment_test/results/run_amend3.log.
Inputs not in git, already on NFS: results/large/states/*.pt (Amendment 2's, sha256-checked in-job against
          results/amend2_stage0.json), transport_test and task_ladder windows, artifacts/vpd.
Platform: MATS cluster, debug QoS, 1 L40 (as Amendment 2). RunPod only if not started within 10 min (then the
          states must go up by hard-link staging, ~1.4 GB: ask the lead first).
Budget: $0.
```

## Log lines and expected values

| Phase | Log | Expected |
|---|---|---|
| Start | `[a3 start] on NVIDIA L40; cfg Cfg(...)` | |
| Controls, nine lines | `[a3 controls <task>/<s0,s1,null>] T0 ... T1 ... |S| ... | R0 lo - T1 ... | R1 lo - T0 ... | R2 lo - T0 ... | R3 max |logit diff| pre ... at ... | {...} ok` | R0, R1, R3 pre exactly 0 (tolerances 1e-5, 1e-4, 1e-4); R2 ~0 (tolerance 1e-2); R3 at > 0; T1 as Amendment 2 (emoticon/s0 8.4070) |
| Controls | `[a3 controls] PASS` | PASS |
| Grid, three lines per state (six states) | `[a3 grid <task>/s<k>] T0 ... G ... | active threshold ... short ...`, then `rho_r S ... null_minus_S ... | random ... | active ...`, then the 20 + 20 draws | G 10-14 nats; active threshold 0.5 (low confidence), short False (True: that seed is unscored for P9, not a stop); rho_r(S) 0.10-0.50 on emoticon and r1 (` which`), below 0.10 on r2 (` we`), low confidence |
| End | `[a3] P9 seeds ... tasks ... reading '...'` and `[a3] P10 {...} | reported {...}` | readings, not stop rules |
| Done | `[a3 done] (... s)` | |

## Stop rules

- Any R0-R3 fail (`[a3 controls] FAIL: stop and report`, exit 1): report which check failed on which state and its
  numbers. Do not rerun, do not edit code.
- A Python error: report the traceback. Do not edit code.
- P9, P10, the active threshold and rho_r(S)'s range are results, not stop rules: let the run finish.
- No new log line for 3 min after `[a3 start]`: read `squeue` and the log once, tell the lead; if clearly hung, cancel.

## Before reporting

Check that amend3.json has the `P9` and `P10` keys (the end of the grid) and `controls_pass: true`; quote the nine
control lines' R0-R3 numbers, the six `rho_r S` lines, and the two `[a3]` lines.
