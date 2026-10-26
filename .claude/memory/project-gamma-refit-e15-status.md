---
name: project-gamma-refit-e15-status
description: "E15 Γ refit ONGOING (reopened by the user 2026-10-21 after a same-day shelving; fix = attacker matched to the harness, W₀ control first): both the filed recipe and the v2 protocol (adversary warm-up, lr 5e-6) took W₀ off the adversarial floor by step 100; the refit's attacker (0.42 nats) is weaker than the harness's (0.70)"
metadata:
  node_type: memory
  type: project
  originSessionId: e503c159-7c9b-4929-ba85-bd36d8d40bcc
  modified: 2026-10-21
---

E15 (`experiments/gamma_refit`, analysis `reports/analyses/2026-10-20_analysis_gamma-refit.md`, notebook `notebooks/E15_gamma_refit.ipynb`)
ran two refits on 2026-10-20 on H100s ($10.07 + $9.10). Split state, Pile-only: adversarial ratio 7 → 24 → 23 → 7.6 → 1.06
over 3 000 steps, sparsity ratio 1.60 (filed "cheating" branch). W₀ control under the same refit: 1.0 → 2.57 → 3.53 → 6.27 →
2.63, sparsity 0.94–1.09, every other check within 8% of 1: P1 failed at every checkpoint, so the recipe (AdamW restarted at
the released START lr 5e-5, persistent adversary from random sources) drifts on its own and the split's fall is not a repair.
Stage 0: cancellation reading (median ablation ratio 0.98). The control was folded into the report, notebook, TRIED, README
and the research log (v14) on 2026-10-21. The design check for the protocol fix is `reports/plans/2026-10-21_plan_gamma-refit-protocol.md`
(300 steps at lr 0 so the adversary and Adam's moments warm up, then lr 5e-6; W₀-only test, ~$9.5, early stop at step 300);
the user confirmed it and the W₀-only run went 2026-10-21 ($3.35, early-stopped at step 300 as RunPod credits ran out):
adversarial 1.00 → 2.06 → 7.76 at Γ steps 0/100/300 with the adversary warmed (flat at 0.42 nats) and no spike, so P1 failed
again and refitting Γ alone is SHELVED (decisions.md 2026-10-21). Key mechanism: the refit's persistent attacker reaches
0.42 nats on the released Γ where the harness's 20-step attacker reaches 0.70 (9.4 on one block after 300 steps): the
objective as reconstructed does not see what the harness measures. Results `results/gamma_refit_v2.json` (partial.w0_P),
logs `results/large/full_v2.log`, `scored_stdout_v2.log`. Folded into the report (§ 5), notebook, TRIED, README, log v16. Results JSON still has no `score` key (scoring runs only after all six refits).

**Update 2026-10-21:** the user reopened it ("ongoing since we're trying to fix the issues"), `reports/decisions/2026-10-21_decision_gamma-refit-reopened.md`; every state file now says ongoing.

**Why:** the remaining arms must not run before a retraining with an attacker matched to the harness's holds W₀ at the floor, and the split's 1.06 is not a repair.
**How to apply:** next per decisions.md 2026-10-21 are the cancellation tests on the saved split state and the re-gating line;
never edit E15's filed predictions or results. Rule learned (CLAUDE.md): the control that validates the apparatus runs first. Related: [[project-gamma-refit-ppgd-scope]],
[[feedback-never-refile-predictions]], [[feedback-compute-budget]].
