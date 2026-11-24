---
slug: zhao2025-echo-chamber
title: "Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining"
authors: Zhao et al.
year: 2025
id: 2504.07912
tags: [EH, KNOWN-ANSWER, DEFINITION, SUPPORTS, DYNAMICS]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation, training-dynamics]
read_from: "local PDF arxiv_2504.07912.pdf (COLM 2025, v2) via pdf-text: abstract, §1-5 (pp. 1-10) incl. setup, §3.1-3.6, Table 1; appendices not read; figure values not readable in extraction"
digest_buckets: [2-known-answer-case, 3-definition, 4-supports]
importance: 4
---

# Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining (Zhao 2025)

**What it does.** Pretrains 150M and 1B OLMo-style models from scratch on FineMath-3+ and Algebraic-Stack plus chosen
mixtures of three instruction sets with distinct, string-detectable output formats: TinyGSM (code), OpenMathInstruct1
(code) and OpenMathInstruct2 (natural language). It then runs PPO, GRPO or Expert Iteration on GSM8K with a 0/1
correctness reward. It tracks, through training, what share of generations falls in each pretraining format and the
accuracy within each format. Checkpoints for two 1B mixtures are released.

**Key result.** RL quickly collapses the output onto one pretraining format, usually within the first epoch, and the
biggest pass@1 gain comes with that shift. pass@64 dips late in training. The winning format is not always the most
common one at initialisation, and once (8× OMI1) it is the less accurate one, which lowers final accuracy. The winner
depends on scale: 150M models pick TinyGSM code, and 1B models pick natural language. Outputs within the winning
format are also refined, and 1B models trained on GSM8K gain on MATH-500 (e.g. 33.4% to 43.8% pass@1). The theory
section restates the known KL-regularised optimum as a re-weighting of a mixture of policies.

**Quote.** "RL fine-tuning amplifies a specific mode from the pretraining mixture while collapsing the others." (§5)

**Relevance to our paper.** This is the cleanest behavioural setting where what the base model learned is known by
construction: each candidate behaviour is a tagged pretraining source. That makes it a candidate known-answer testbed
for reuse at a scale near ours (150M). PPO that selects the TinyGSM mode should, by intuition, reuse the base model's
TinyGSM mechanism, and the paper's own evidence is only the output-format share. Nothing in it looks at activations or
weights. The within-format refinement and the MATH-500 transfer show that "amplify a mode" is not the whole update, so
the known answer would be "mostly reuse", not pure reuse. Its mixture-reweighting formula, π* ∝ π_ref·exp(r/β), is a
policy-level definition of elicitation that we can contrast with ours. Ours asks whether the mechanisms are kept, and
two updates with the same output re-weighting can differ under it. Worth one sentence in related work under "post-training as selection".

**For the digest.**
- 2-known-answer-case: 150M/1B from-scratch models with tagged pretraining formats and released RL checkpoints. RL selects a known pretraining mode, a behavioural known answer that the paper does not check mechanistically. Our featurizations would need fitting to these models.
- 3-definition: elicitation as KL-optimal re-weighting of a mixture of pretraining policies, a behavioural definition to contrast with our mechanistic one.
- 4-supports: post-training amplifies a pretraining behaviour rather than creating one, behaviourally, with caveats (refinement within the format, transfer to MATH).
