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
| 1 | Transport test, round 1 (`experiments/transport_test`, 2026-10-14; `reports/2026-10-14_analysis_transport-test.md`) | Released 67M Pile model + VPD decomposition, frozen Γ and Δ. Arm A: full-weight fine-tune on 365 emoticon windows (P(o) 0.999 by step 10); arm B: continued pretraining (held-out effect 0.014 nats). Transports: Frobenius projection on alive subcomponents, functional rescaling of all alive, k-sparse write/read edits on top-k by Frobenius contribution vs k random alive | δW is not in the span of the rank-1 subcomponents: Frobenius fraction 0.05–0.06% vs 0.035% for a random direction (predicted ≥ 10×; reversed). Functionally the emoticon behaviour is reproduced by 10 edited subcomponents, but 10 *random* ones do as well (0.987 vs 1.000; predicted margin ≥ 0.3; reversed); at k = 3 top beats random by 0.16. The paper's `2359` ranks 2nd in its matrix at step 10 and disperses by step 30. Decomposition stays faithful under every transport by the stochastic metric (ratios 0.94–1.02); adversarial excursion to 2.4 at one checkpoint, one block. Arm B functional fits overfit (capture < 0) | The null (free write vectors) is too expressive and the Frobenius selector picks the wrong subcomponents; next round needs a scale-only or constrained null, a functional selector, checkpoints at steps 1–5, a larger arm-B effect, and selectivity on generic text. Through `/design` first |
