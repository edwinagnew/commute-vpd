---
slug: neurons-domain-ft-2510-09359
title: Understanding the Effects of Domain Finetuning on LLMs
authors: Eshaan Tanwar et al.
year: 2025
id: 2510.09359
tags: [DIFFING, BASELINE, FT-INTERP, SUPPORTS]
primary_tag: DIFFING
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, interpreting-finetunes, baselines]
read_from: arXiv HTML full text (abstract, sections 2-4, conclusion, appendices A.6 patching and A.7 choice of k)
digest_buckets: [1-missing-method, 4-supports]
importance: 4
---

# Understanding the Effects of Domain Finetuning on LLMs (Tanwar 2025)

**What it does.** The paper studies released medical fine-tunes of open models: Llama3-Aloe-8B-Alpha against Llama-3-8B, Qwen2.5-Aloe-Beta-7B against Qwen2.5-7B, and MediPhi against Phi-3.5-Instruct. It compares which MLP neurons fire on 100M PubMed tokens (normalised edit distance of the binary on/off patterns) and weight cosine similarity. It then defines "tuning vectors" (θ_ft − θ_pre, i.e. task vectors) and runs negation and addition tests. Last, it scores each weight matrix's update by "SubSpace Alignment" (SSA): the share of the update that lies in the top-k left singular subspace of the base weight. A high SSA is read as "amplifying existing directions" and a low one as "new directions".

**Key result.** Neuron activation patterns barely change (small edit distance in every layer). Subtracting the tuning vector cuts medical-benchmark accuracy by 65% (Llama Aloe) and 7% (Qwen Aloe). On SSA, attention updates sit mostly inside the base subspace (about 0.75-0.99), and MLP gate and up updates mostly outside it (about 0.20-0.46), while down projections stay above 0.97. The same pattern holds for Qwen2.5-Coder and Qwen2.5-Math tuning vectors.

**Quote.** "we find these vectors primarily write new directional information into the MLP layers of the model, while amplifying existing directions in attention heads." (Abstract)

**Relevance to our paper.** This is a cheap weight-space "reuse vs new" reading of realistic fine-tunes, and it was never validated against known answers. It is the kind of tool our second claim says fails or goes untested. SSA could run on our setting at no cost (project ΔW onto the base weight's top-k singular subspace, per matrix) and could be scored on our built pairs and key tasks. It has two weaknesses worth noting. First, k is chosen to keep 99.75% of the weight's squared singular mass (ε = 0.05), so for most matrices the subspace is close to full rank and a high SSA may be close to trivial; no random-update floor is reported. Second, a rotation inside the subspace counts as "amplifying", which is exactly what our rescale-only kept clause does not allow. Their finding that activation patterns and patched regions stay the same fits our observation that neuron-level readings call almost everything reuse. The only causal evidence is one attribution-patching prompt pair (appendix), so it is not a known-answer case.

**For the digest.**
- 1-missing-method: SSA (share of ΔW inside the base weight's top singular subspace) is a free weight-space reuse score we could add as a baseline; it needs a random-update floor and a smaller k to say anything.
- 4-supports: neuron activation patterns barely change under real domain fine-tunes, as in our finding that neurons read everything as reuse.
