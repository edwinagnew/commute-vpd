---
slug: braun2024-e2e-sae
title: "Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning"
authors: Braun et al.
year: 2024
id: 2405.12241
tags: [BACKGROUND, BASELINE]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [baselines, decompositions]
read_from: "arXiv HTML v2: abstract, section 1, section 3 headline results and conclusion; appendices not read"
digest_buckets: []
importance: 2
---

# Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning (Braun 2024)

**What it does.** Trains SAEs on GPT-2 small (and TinyStories-1M) residual streams to minimise the KL divergence
between the original model's output and the output with SAE reconstructions spliced in (SAE_e2e). A variant also
matches activations at all downstream layers (SAE_e2e+ds), so the reconstruction takes the model's own pathway. Both
are compared with standard MSE-trained SAEs (SAE_local) on CE-loss increase, L0, alive latents, geometry and automated
interpretability.

**Key result.** At a given L0, both e2e variants have a CE-loss increase "less than 45%" of SAE_local's, and need
fewer alive latents, with no loss of automated interpretability. They cost 2 to 3.5 times the compute. The
downstream-loss variant explains about as much as plain e2e while keeping later-layer activations close to the
original. Local SAEs show more feature splitting.

**Quote.** "These results suggest that standard, locally trained SAEs are capturing information about dataset
structure that is not maximally useful for explaining the algorithm implemented by the network." (Conclusion)

**Relevance to our paper.** Background for our SAE featurization: our BatchTopK SAE latents are MSE-trained, which this
paper argues pick up dataset structure rather than the network's functional variables. An e2e+ds SAE fitted on the
base model would be a featurization closer in spirit to VPD's (output-faithful) and could be a fairer SAE arm. It is
also the lineage of the APD/SPD/VPD loss design. Low priority: one citation at most where the SAE featurization is
introduced.

**For the digest.**
- (no bucket; background on featurization choice)
