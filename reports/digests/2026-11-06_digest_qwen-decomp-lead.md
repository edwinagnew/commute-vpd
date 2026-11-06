# qwen-decomp-lead digest, 2026-11-06 (evening)

**D1, day 4 of 8** (D1: a map on Qwen3-1.7B that keeps the decomposition faithful for a real update). Finished for
the day because the only next step for D1 is a paid run that waits on the user's go-ahead (ask 36).

## What ran
- **Teacher training on RL seed 70 (E111 stage 1, RunPod, $2.30).** At the planned rate it repaired the update that
  least squares had missed: it passes every behaviour check (clean-model K 0.0006 nats against a line of 0.02;
  least squares 0.024). The 10x-rate positive control stopped at its first reading and is dropped, not rerun
  (commit 8b0affb0); the code now lets stages 2-5 start from the planned-rate arm alone.
- **The adjoint split as a column of the table (E93 Amendment 1, cluster, $0).** It fails all 11 updates, as filed.
  Two weak filed leans reversed on all 11: it leaves out a larger share of the update than of a random change
  (0.63-0.73 of the squared size against 0.59-0.61), and under its own masks it sits above doing nothing. No cause is
  named. Report: `reports/analyses/2026-11-06_analysis_e93-d1-table.md`, section "Amendment 1"; log card E93 (v278).
- **D3 labels:** definition-lead's step 2 says no Qwen example update can get a reuse verdict, so D3's rows are
  readings with the definition's label printed beside them (`reports/decisions/2026-11-06_decision_definition-step2-qwen-anchors.md`).

## Pending decision for the user
- **Ask 36:** run teacher training on the other 10 updates (E111 stages 2-5, about 5 h of A100, about $10-15,
  RunPod)? The plan makes this go-ahead yours, as a gated run's last gate.

## Money and jobs
RunPod this session: $2.30 (stage 1, earlier). Cluster jobs 54583, 54598, 54656 (E93 Amendment 1), all done. No
pod or job of this lead is live.

## Friction noted
scancel and scontrol update of my own pending job were refused by auto mode (the user cancelled it); a 30-min debug
limit I briefed killed a job whose early-exit check uses a fixed 2-h limit; sacct is down on the dev node; worktree
agents fail when the lead's shell has drifted out of the clone; the log publisher needed three sends.
