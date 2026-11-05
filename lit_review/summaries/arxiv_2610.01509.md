# Sharpening Tax in Post-Training

## Metadata
- **Authors**: Changdae Oh, Qi Zeng, Qi Qi, Andrey Zhmoginov, Deren Lei, Yun He, Hoang Phan, Hangoo Kang, Azalia Mirhoseini, Sharon Li (Meta Superintelligence Labs, UW-Madison, NYU, Stanford)
- **Year**: 2026
- **Source**: arXiv:2610.01509v1 [cs.AI], 1 Oct 2026
- **DOI/URL**: arXiv:2610.01509; project page https://changdaeoh.github.io/sharpening-tax/

## Summary
The paper tests the "RL only sharpens the base model" hypothesis on agentic tasks (BFCL v4 multi-turn, WebShop, ACEBench) using 14 open base/instruct checkpoint pairs. With a light prompting harness, base models often overtake their post-trained versions in pass@K at large K, even though pass@1 is far lower. The authors define a scalar "Sharpening Tax" (the base-minus-post difference in area under the pass@K curve) and propose PTGS, a per-prompt adaptive rollout temperature for RL, which they report reduces the tax in two toy RAGEN environments. All evidence is behavioural, from sampling outcomes.

## Key Findings
(All of these are the authors' claims unless marked as inference.)
- Post-trained models win at small k, base models (with harness) scale more steeply and cross over, mostly for larger backbones. The crossover budget k* shrinks with scale (WebShop, Gemma-4: k*>128 at 4B to k*~3 at 31B). For the smallest backbones (3-4B) the base curve "never catches up", because post-training supplies basic skills such as well-formed tool calls (App. C.1).
- Post-training "bimodalizes" per-task success rates. Example in the text: gemma-4-31B on WebShop, "pass given compute" tasks fall from 87.6% to 30.0%, always-pass rises 0.0% to 26.0%, always-fail rises 12.4% to 44.0% (main text, Sec. 3, Fig. 4).
- Sharpening Tax: A(K) = sum_{k=1}^{K-1}(pass@K - pass@k); S(K) normalises by (K-1)(1-pass@1); Tax = base minus post (Eqs. 1-3). TaxS(128) > 0 in 36 of 42 model-benchmark cases. Negative cases are small models on BFCL, where post-training repairs tool-call formatting (Sec. 4, App. D.1).
- Tax estimated from 8 rollouts predicts the 32-rollout tax (Spearman 0.85) and correlates with the consistency gap (0.66) over 42 combinations (Fig. 7).
- Theory (Thm 2): if sharpening sends a fraction lambda of tasks to p=0 or p=1 independently of base p, then TaxA = lambda*A_Base, and the calibrated tax stays positive even when pass@1 improves. This is a toy model, not evidence of what the weights do.
- App. D.2, Table 8 splits TaxA(128) into "lost" (solved only by base), "gained" (only by post), and "shape" (both solve). Lost is positive in every combination. Gained is large only for small backbones on BFCL and WebShop. Fig. 27 pools per-task movement: for the largest backbones, BFCL "sharpened 57%, lost 12%, softened 7%, gained 10%" (inset text, figure garbled in extraction).
- App. D.3: the highest-tax tasks are ones where the post-trained policy is deterministic (flat at 0% or 100%) while the base succeeds about once in 128 rollouts.
- PTGS on Qwen2.5-7B-Instruct (RAGEN; Sokoban, FrozenLake; PPO and GRPO; 200 steps; 5 seeds): task-specific RL also pays the tax. PTGS gives higher pass@1 and pass@128 and a lower tax than fixed-temperature baselines. Table 2 is garbled by extraction (environment labels interleaved with rows), so I do not quote cells. Qualitatively, GRPO pass@128 is about 55% vs about 72% with PTGS on Sokoban, as I read the garbled table.
- Conclusion (Sec. 8): post-training "mainly changes how reliably a model solves tasks rather than which tasks it can solve in the agentic domain", while "Whether heavier RL truly moves beyond elicitation remains open".

## Methodology
Observational, black-box study of public base/post-trained checkpoint pairs (Gemma-4, Ministral-3, Qwen2.5, Qwen3.5; 3B-35B). N=128 rollouts per task, with the unbiased pass@k estimator and paired task bootstrap CIs. Base models get a text-completion harness; post-trained models get the default chat template and tool parser (asymmetric scaffolding, App. A.3). Post-trained models run with thinking mode off. The post-training recipes are undisclosed (SFT, DPO, and RL mixed). The authors treat SFT, DPO and distillation as all "sharpening" and make no recipe distinction. The RL training experiment (PTGS) is separate, small-scale and in toy grid environments. Theory is a stylised mixture model of per-task success probabilities. No activations, weights, probes or circuits are examined anywhere.

## Relevance to Proposal
**Score: 3**

1. Evidence type: purely behavioural (pass@k, per-task success-rate distributions, rollout counts). Nothing looks inside the model, and no parameter-space or representation analysis appears in the main text or in Appendices A, B, D.1-D.3 that I read. Their "mechanism" in Sec. 3 means a change in the distribution of per-task success probabilities, not a circuit.
2. Could their result hold while our mechanistic verdict goes the other way? Yes, in both directions (this is my inference). (a) Their finding that a base model "covers" tasks at large k says the behaviour is in W0's sampling support. That is the behavioural notion the user rejected, and it is compatible with a fine-tuned behaviour that no longer runs through a readable direction of W0 ("not through C" / "unreadable"). (b) The reverse also holds: a post-trained model can lose coverage (their "lost" tasks, 10-30% on large backbones) while its remaining behaviour still reuses a linear-readable capability of W0 (Jain Def 1/2). Pass@k bimodalization does not distinguish reuse from new mechanism. Their small-model result is mildly suggestive: where post-training "repairs" the base (tool-call formatting, 3-4B), the base curve never catches up. That is behavioural evidence of capability acquisition, not mechanistic. The paper does not claim to settle new-vs-reused; it says it remains open.
3. Directly usable: weakly.
   - A behavioural comparator. Sharpening Tax and the lost/gained/shape split (Table 8) are cheap, base-anchored scalars that could sit beside our readings. The tax could be computed on GSM8K for the SFT and GRPO arms and checked against the causal direction-removal verdict. Divergence between the two would itself be informative. This is a suggestion, not something the authors do. I checked that per-step or early-step results exist: none are in what I read.
   - Their theory (Thm 2) offers a clean null for "pure sharpening": tasks move only to p=0 or p=1. Our anchor updates with known novelty could be probed against this.
   - GRPO-specific: in RAGEN, GRPO validation success "peaks before step 200 and then sometimes collapses" (App. A.5), and Thm 3 argues GRPO only learns from mixed groups. Their GRPO runs are on Sokoban/FrozenLake with Qwen2.5-7B-Instruct, not GSM8K.
   - Qwen: Qwen2.5 and Qwen3.5 pairs only; no Qwen3-1.7B, and no GSM8K-style task. Qwen3.5 shows the softest sharpening and Gemma-4 the steepest (Sec. 3). Qwen3.5 base models already produce parseable native tool calls, which hints that base models of that family are already post-training-contaminated (my inference, not stated).
   - No controlled task with novelty fixed by construction. The authors list this as a limitation: "our analysis remains observational rather than interventional" (Sec. 9).
Connection to our project is thin: same debate ("sharpening vs new capability"), different level of evidence, different tasks and scale.

## Limitations
- Observational, undisclosed training data and recipes. SFT, DPO and RL are not separated (authors, Sec. 9, App. A.2).
- Asymmetric scaffolding: base gets a tuned harness, post-trained gets default scaffolding, and the harness hurts post-trained models (Table 4). Thinking mode is off by default. Both choices can move the crossover (authors partly ablate thinking, Fig. 10).
- Agentic benchmarks with small task sets (200 BFCL, 500 WebShop, 770 ACEBench). Pass@128 at K=N is noisy (App. A.4).
- Coverage at large k is not "possession in W0": the success probability can be tiny (often 1 in 128, App. D.3), which is compatible with luck or a different mechanism (my inference).
- Binary success only. Tax is not defined for open-ended generation (Sec. 9).
- PTGS evidence is two toy environments on one model, 5 seeds; GRPO uses best-validation checkpoint selection.
- Text was extracted from PDF. Tables 2, 6 and the figure insets are partly garbled; I did not guess cell values.

## Key Quotes
> "post-training pushes tasks toward two extremes, always solved or never solved, and thereby improves sampling efficiency and consistency at the cost of solution coverage." (Abstract)

> "our analysis remains observational rather than interventional. Replicating this in a controlled setup ... where both data mixes are fully known would allow a causal account of when and how sharpening emerges exactly." (Sec. 9)

> "Whether heavier RL truly moves beyond elicitation remains open, and base-anchored diagnostics like Sharpening Tax offer a direct way to check it as post-training continues to scale." (Sec. 8)

> "For the smallest backbones (3-4B), post-training mainly supplies basic skills that the base model lacks, e.g., well-formed tool calls, and thereby expands coverage." (App. C.1)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper, no comments).

## What I did not read
Read in full: abstract, Secs. 1-9, App. A, App. B, App. C.1 and the start of C.2, App. D.1-D.3 (Tables 7, 8, text of Figs. 26-28). Skimmed only: the rest of App. C (per-family curves, C.3-C.4 temperature results). Not read: App. D.4 (tax-guided routing), App. E (PTGS training dynamics and ablations), App. F (PTGS analysis). All figure content is available only as axis text, so any numbers from plots are the authors' captions or insets.
