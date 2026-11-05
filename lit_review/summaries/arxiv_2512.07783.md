# On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models

## Metadata
- **Authors**: Charlie Zhang, Graham Neubig, Xiang Yue (Carnegie Mellon University, LTI)
- **Year**: 2025 (arXiv v1, 8 Dec 2025)
- **Source**: arXiv preprint, cs.CL
- **DOI/URL**: arXiv:2512.07783

## Summary
The paper trains small (100M, Qwen2.5-style) decoder models from scratch on synthetic GSM-Infinite-style arithmetic word problems. The problems come from dependency graphs, so the training distributions for pre-, mid- and RL (GRPO) stages are disjoint and known by construction. It uses this control to say when RL gives "true capability gains" (pass@128 up) and when it only sharpens. Answer given: RL extends capability only when pre-training leaves headroom and the RL data sit at the "edge of competence", and a minimal (about 1%) seed of a primitive or context in pre-training is needed for RL to transfer. All evidence is behavioural (pass@k, process-verified accuracy, graph-topology similarity of outputs). Nothing looks at activations, weights or circuits.

## Key Findings
(All are the authors' claims, from behavioural evaluations.)
- ID tasks (op=2-10): RL raises pass@1 but not pass@128 under every RL data regime, read by the authors as sharpening. On OOD-edge (op=11-14) and OOD-hard (op=15-20), RL on edge-of-competence data (op=11-14) raises pass@128 (Sec. 3, Obs. 1, Fig. 3).
- Contextual transfer: with 0% or 0.1% pre-training exposure to a long-tail context B, RL fails to transfer to B. With 1% or more of atomic op=2 examples in B, RL generalises up to op=20 (Sec. 4, Obs. 2, Fig. 4).
- "Replication or creation": the topological similarity of correct context-B solution graphs to gold context-A graphs is high for op=2-10 (replication) and lower for op=11-20, especially with more B exposure (Sec. 4, Discussion 2, Fig. 5). The authors read the low-similarity cases as novel reasoning structures.
- Under a fixed FLOP-normalised budget, mid-training plus RL beats RL alone. Light RL is best for OOD-edge pass@1, and heavy RL is needed for OOD-hard (Sec. 5, Fig. 6). The authors cite external work that Qwen bases respond to RL better than LLaMA, possibly because of mid-training (Discussion 3).
- Mixing process-verification reward into the outcome reward gives about +4-5% pass@1 on op=15-20 and reduces structural errors (Sec. 6, Figs. 7-8). Fig. 7 text is garbled, so I give only the headline claim.

## Methodology
Three-stage pipeline on a 30B-token synthetic corpus split into disjoint pre/mid/post data. Pre-training is on 10B tokens at 100M parameters. Post-training is GRPO on 200K samples. Tasks are DAG-defined with a parseable step-by-step solution. Correctness is "process-verified": the predicted graph must match the gold graph and the final answer. Generalisation is measured along depth (number of operations) and breadth (surface-context template). Metrics are pass@1 and pass@128, process accuracy, and graph similarity. The mid-training vs RL compute comparison uses an approximate token-equivalent formula for RL cost (Eq. 1).

## Relevance to Proposal
**Score: 4**

1. Evidence is purely behavioural: pass@k and parsed-output graph similarity. There is no interpretability, activation, weight or circuit analysis, and no SFT arm (GRPO only).
2. Yes, the two can come apart in either direction. The paper's "sharpening" label (pass@128 flat) is a behavioural criterion of the kind the project rejected, since the base model already gives the right answers. So ID-task RL could be "sharpening" here while a direction-removal check shows the fine-tuned behaviour runs through a new route, or the reverse. Their "genuine gain" cases (pass@128 up on op=11-14) say nothing about whether the new behaviour reuses a readable pre-existing capability C. The 1% seed result is closer in spirit: RL needs pre-existing primitives, which is loosely an elicitation or reuse story, but it is inferred from behaviour. The graph-similarity "replication vs creation" result is the nearest thing to a mechanism claim, but it compares output structure and not internals (my inference).
3. Usable pieces:
   - A controlled setting where what the base model possesses is fixed by construction (op range, and context B at 0, 0.1, 1 or 10%). It could supply anchors with known novelty if one pretrained a tiny model, though it is not Qwen3-1.7B or GSM8K.
   - Their pass@1 vs pass@128 split is a ready behavioural comparator to put beside our readings.
   - The finding that RL on tasks already well covered gives no pass@128 gain is evidence for the behavioural sharpening notion.
   - Qwen-specific content is thin: the mid-training explanation for Qwen's RL responsiveness is cited from other work, and their own models are small Qwen2.5-style architectures trained from scratch.
   - Nothing about early training steps or about delta-W.

The connection to the weight-update decomposition question is thin. The paper is a useful contrast case for how "existing capability" is usually defined in the literature (behavioural), not a test of the mechanistic claim.

## Limitations
- Behavioural evidence only. "Sharpening" and "true capability gain" are defined by pass@128 movement, which depends on k and on a base model with near-saturated pass@128 on ID tasks.
- 100M models trained from scratch on synthetic arithmetic. Transfer to a pretrained 1.7B model on GSM8K is not shown.
- GRPO only, no SFT comparison. Process-verification relies on a parseable, gold-graph format that real tasks lack.
- Several claims about Qwen vs LLaMA and mid-training are citations to other work, not tested here.
- Appendices (A.1-A.10, including training dynamics and the extra contextual and extrapolation experiments) were not in the extracted text, so I did not read them. Figure data are garbled in extraction; I do not quote figure numbers beyond the stated headline claims.

## Key Quotes
> "RL produces true capability gains (pass@128) beyond base models only when two conditions hold: (i) The task is not heavily covered during pre-training, leaving sufficient headroom for exploration; and (ii) the RL data is calibrated to the model's edge of competence, neither too easy (in-distribution) nor too hard (out-of-distribution)." (Sec. 3, Takeaway 1)

> "For ID tasks (op=2-10), there are obvious performance gains on pass@1 but no improvement on pass@128 regardless of the RL data regime, which indicates that RL only sharpens existing capabilities without extending them." (Sec. 3, Observation 1)

> "RL cannot synthesize capabilities from a void; it requires latent "seeds" to amplify." (Sec. 4, Practical Guidance 2)

## Comments Summary
Not applicable (arXiv paper, no comments). Not read: the appendices and the full reference list.
