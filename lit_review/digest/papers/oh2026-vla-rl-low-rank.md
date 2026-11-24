---
slug: oh2026-vla-rl-low-rank
title: The Low-Rank Structure of VLA Reinforcement Learning
authors: Oh et al.
year: 2026
id: 2609.34599
tags: [FT-INTERP, DIFFING, SUPPORTS, BACKGROUND]
primary_tag: FT-INTERP
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, model-diffing]
read_from: arXiv HTML abstract+intro+sections 3-5 (settings, results, takeaways)+conclusion; appendix not read
digest_buckets: [4-supports]
importance: 3
---

# The Low-Rank Structure of VLA Reinforcement Learning (Oh 2026)

**What it does.** Asks where and how RL (PPO) post-training changes flow-based vision-language-action robot policies
(pi0.5, GR00T N1.5/N1.6, SmolVLA) on LIBERO, ManiSkill, MetaWorld and CALVIN. It measures per-module update density
and effective rank of ΔW (singular directions for 95% of squared Frobenius norm), runs module-replacement experiments
(swap a module back to base weights), compares each AdaRMS output's base value with its RL change by absolute cosine,
then probes and steers along the RL shift-update directions.

**Key result.** RL updates to the Timestep Modules (the layers mapping the denoising timestep to scale, shift and gate
vectors) are dense but low-rank, roughly an order of magnitude below behaviour cloning (BC ~250-650), while attention
and MLP updates stay high-rank; discrete-timestep BC reproduces the low rank, so the cause is training on few timesteps,
not RL as such. Keeping only the RL Timestep Modules preserves more of the RL gain than keeping only attention+MLP.
The RL change to scale and gate is mostly aligned with the base output, the change to shift is not. Shift directions
predict success (ROC-AUC up to 99.6%) and steering along them adds success over matched-norm random directions.

**Quote.** "This suggests that RL largely rescales the existing scale and gate patterns while introducing new directions
predominantly in the shift vectors" (Sec. 4.2)

**Relevance to our paper.** A different model class (robot policies, not LMs) and no causal-abstraction or
decomposition machinery, so not a baseline or known-answer case. Its one point of contact is the rescale-versus-new-
direction question: it reads, per output vector, whether the RL change is parallel to the base value (a rescale) or
not, which is a crude, input-free version of our kept-up-to-a-gain clause, and it finds both kinds in one update. That
mildly supports our finding that trained updates are not pure rescalings, and that a gain-only kept clause will miss
part of what a realistic update does. Its effective-rank and module-swap readings are the same family as our update
effective-rank reading; worth a one-line citation in the RL-update-structure discussion at most.

**For the digest.**
4-supports: in VLA RL, part of the update rescales base modulation vectors and part adds new directions (shift), consistent with "trained updates turn as well as scale".
