# qwen-rl-lead digest, 2026-11-12 evening

P3.4, due 11-16: run 2 is read and written up, and the cell's last run waits on the user (ask 16). Reinforcement
learning on Qwen3-1.7B kept its GSM8K gain at twice the decomposition's error. But the shuffled-reward null collapsed
during training, so beating it says nothing yet.

**Cells.** P3.4 open, 2 of 3 runs used, due 11-16. Finished because everything left waits on ask 16.

**What ran.**
- Run 2 trained on six RunPod pods: $174.73. The cell has spent about $226.
- The 9 saved states were read on the cluster at $0 (jobs 59649-59657 for the states, 59673 for the outside share).

**What it found** (`reports/analyses/2026-11-12_analysis_e167-qwen-rl-readable-run2.md`):
- By the filed rule, RL is apart from the shuffled-reward null at 2x on both seeds. 5x was reached on seed 70 only and
  10x on neither, so both are unread.
- The null collapsed on both seeds: its pass rate fell to 0.039 and 0.000, against RL's 0.882 and 0.884. So "apart"
  carries no claim. The cause was not checked.
- Every update's outside share reads like a random change of the same size (0.979-0.993 of the random change's share).
  The control read 0.98047, in its band.

**Ask waiting on the user.** Ask 16 (card `reports/decisions/2026-11-12_decision_qwen-rl-last-run.md`, in the log's
Pending decisions):
- (a), recommended: two pods retrain the null at 1e-6 until its output change matches RL's saved 2x states. It stops if
  under half its answers are right or answers fall under 100 tokens. Up to $75 and about 9 h; the cell would reach
  about $300, about $38 over approval. Gated.
- (b): RL and the null at 1e-6 to step 500, about $150.
- (c): close the cell as read, $0.

**Log.** Version 338 is published with the card and the updated P3.4 experiment card.

**Blocked agents.** definition-lead has been waiting at a question since 18:00.

**Friction noted.**
- 768: the outside-share job's runner reported a 74 GB host-memory peak under a 16 GB request.
- 769: my clerk brief gave two wrong paths.
- 770: `fleet/premise` refuses P3.4 because qwen-rl-lead has no ladder part, so `cell fill` or `close` will refuse.

**Not run.** No run 3. No premise check (refused, 770). No independent check, since no headline claim was made.
