---
slug: lingam2024-svft
title: "SVFT: Parameter-Efficient Fine-Tuning with Singular Vectors"
authors: Lingam et al.
year: 2024
id: 2405.19597
tags: [EH, SUPPORTS, BACKGROUND]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation, other]
read_from: arXiv HTML abstract, method (3.1-3.2), results 5.1, 5.4-5.5, Table 1
digest_buckets: [4-supports, 1-missing-method]
importance: 3
---

# SVFT: Parameter-Efficient Fine-Tuning with Singular Vectors (Lingam 2024)

**What it does.** A PEFT method that writes the update in the frozen weight's own SVD basis, ΔW = U M Vᵀ, and trains
only a sparse set of entries of M. The "Plain" variant trains only the diagonal, so it rescales the base matrix's own
singular directions. The other variants (banded, random, top-k) also mix neighbouring directions. Tested on GSM-8K/MATH
(MetaMathQA-40K), commonsense reasoning, GLUE and vision tasks with Gemma-2B/7B and LLaMA-3-8B.

**Key result.** The diagonal-only variant, at 0.43M parameters on Gemma-7B, reaches 73.50 on GSM-8K against 74.67 for
full fine-tuning and 76.57 for LoRA r=32 (Table 1). On Gemma-2B the same comparison is 40.34 against 52.69 for full
fine-tuning. The off-diagonal variants recover up to 96% of full fine-tuning. SVFT gains more than LoRA from a
better-trained Pythia checkpoint.

**Quote.** "we constrain M to be a diagonal matrix, which can be interpreted as adapting singular values and
reweighting the frozen singular vectors." (Sec. 3.1)

**Relevance to our paper.** It shows that a rescale-only update on the base model's own rank-one directions can
reach most of a math fine-tune's gain at 7B, though not at 2B. This is behavioural support for re-weighting. It also
bears on our open question of whether "kept up to a gain" is too strict: rescaling is enough to reach the behaviour,
yet ordinary trained updates turn the writes instead. A diagonal-SVD refit is a natural extra gain-only refit beside
our IA³-style one: the share of F's gain recovered by refitting only the singular values of M's matrices. It is cheap
on our 67M. It says nothing about which mechanism the full fine-tune actually uses.

**For the digest.**
- 4-supports: rescaling only the base SVD directions reaches 73.5 vs 74.7 (full FT) GSM-8K on Gemma-7B; re-weighting can suffice behaviourally.
- 1-missing-method: an SVFT-Plain refit (singular values only) is a gain-only reading we could add to our IA³-style refits.
