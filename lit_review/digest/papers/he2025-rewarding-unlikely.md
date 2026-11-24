---
slug: he2025-rewarding-unlikely
title: "Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening"
authors: He et al.
year: 2025
id: 2506.02355
tags: [EH, SUPPORTS, BACKGROUND]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: "local PDF arxiv_2506.02355.pdf (v2) via pdf-text: abstract, §1-5 (setup, rank-bias analysis, unlikeliness reward, PPO-epochs, Tables 2-3), related work; Appendix A toy model not read"
digest_buckets: [4-supports]
importance: 3
---

# Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening (He 2025)

**What it does.** Studies whether GRPO only sharpens, in Lean formal theorem proving with a perfect verifier. The base
model is DeepSeek-Prover-V1.5-SFT, trained on about 9.6K Lean Workbook theorems. It measures an "uplift rate": for each
correct sample, ranked by base-model probability within its group, whether GRPO raised its probability. It then adds an
"unlikeliness reward" that down-weights correct samples the policy already rates likely, and studies the number of
PPO epochs per batch.

**Key result.** Standard GRPO almost only uplifts already high-probability correct proofs, which they call "rank
bias". It improves pass@1 but can hurt pass@N at large N. The unlikeliness reward reverses the rank pattern, raises
proof diversity and solves the most training problems in one epoch (8065 / 9600 against 7860 for default GRPO and 7707
for no training). Their full recipe matches DeepSeek-Prover-V1.5-RL on miniF2F-test.

**Quote.** "GRPO is more likely to increase the probability of already high-probability correct solutions." (§3.5)

**Relevance to our paper.** It is behavioural and probability-level support for the sharpening reading of GRPO, and it
gives an optimiser-level reason (rank bias) alongside Shao et al.'s clipping bias. It also shows that the recipe
decides how much an RL update sharpens versus extends, which is relevant if our Qwen RL runs are compared across
recipes. No internal analysis, no known answer. One citation in an "elicitation vs sharpening" paragraph at most.

**For the digest.**
- 4-supports: standard GRPO mostly reinforces the base model's already likely correct outputs (rank bias), behaviourally, in Lean theorem proving.
