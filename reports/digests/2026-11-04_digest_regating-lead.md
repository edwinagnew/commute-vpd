# Digest, `regating-lead`, 2026-11-04

**Why finished.** The task was the user's one-off for Q3 of the update notes, and it is done. The re-gating line stays closed to new work.

**What ran.** E83: the logit-shift bars on the 67M, Slurm job 51367, one L40, about 4 minutes, $0. No RunPod.

**What it found.** Masks shrink a whole update's effect on the 67M's logits, as they do on Qwen3-1.7B, but it keeps about twice as much of its size.

| Masks | Adam | Rank-8 LoRA | Qwen, first update |
| --- | --- | --- | --- |
| Random masks, random subset | 0.55 | 0.57 | 0.26 |
| Random masks, all matrices | 0.31 | 0.30 | 0.13 |
| Fixed masks, all matrices | 0.20 | 0.15 | 0.10 |

All four controls passed and both filed predictions held. The bars measure size only and come from one draw. Figure `experiments/logit_shift_67m/figures/logit_shift.png`, report `reports/analyses/2026-11-04_analysis_e83-logit-shift-67m.md`.

**Pending decisions for the user.** One, open since 2026-11-02: close E62 or locate the unexplained 0.32 (Fable's advice, `reports/plans/2026-11-02_advice_e62-after-amend6-FABLE.md`, is to close and log the limit). Nothing here depends on it.

**Friction.** `fleet/ecode` handed out E82 although commit 03ddd3af already used it without a claim, so this run took E83. The E82 row in the claims ledger now holds this run's name, not that commit's. The runner's `wait-job` exited 126 when re-armed, and it fell back on its own loops.

**Live jobs at finish.** None of mine. Jobs 51344 and 51345 (`commute-qwen_tofu`) belong to another lead.
