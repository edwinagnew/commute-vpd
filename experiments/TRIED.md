# What has been tried

Living catalogue. Check here before proposing an experiment, and update it in the same commit as any new experiment,
result, or proposal that gets dropped. Each row says what was found and why we moved on.

## Nearest prior work (not ours)

| # | What | Found | Relevance |
|---|---|---|---|
| 0a | VPD paper, future work "Interpreting parameter gradients to understand training and finetuning" (`../papers/pd/vpd_blog_post.md`, Limitations) | Proposes projecting gradients or a fine-tune weight diff onto the subcomponent basis. Not run | The exact idea; nobody has executed it |
| 0b | VPD paper, "Editing a language model's parameters by hand" | Emoticon → `o` behaviour is expressed by one subcomponent (`2.mlp.down:2359`, five siblings); a hand edit of its write vector competes with a LoRA on ±20-token windows (n = 10 or 947, KL penalty λ) | Shows a one-subcomponent answer exists for this behaviour: the most favourable test case for commuting |
| 0c | VPD paper, "Geometric consistency across seeds" | Cross-seed mean max cosine similarity of rank-1 subcomponents is 0.28 (U 0.48, V 0.52); random init 0.01 | Re-decomposing from scratch does not recover the same subcomponents: any re-decomposition test needs a same-W control |
| 0d | vpd-audit (LessWrong 2026, `github.com/angsten/vpd-audit`) | Aggregating explanations across tokens drifts the model (0.34 nats own explanation → 0.80 at 64 donor tokens); deleting the 28,946 never-needed subcomponents moves the model 1.28 nats | Tests aggregation, not updates. Warns that "unimportant" subcomponents and Δ are not inert; loader pinned to `param-decomp` commit `74146b5` works on one 40 GB GPU |

## Trained / run

| # | Experiment | Set-up | Found | Why we moved on |
|---|---|---|---|---|
| (none yet) | | | | |
