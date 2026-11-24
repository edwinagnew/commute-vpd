---
slug: cao2026-aspd
title: "Weights Read and Write Features: Scalable Parameter Decomposition Grounded in Activation Space (ASPD)"
authors: Cao et al.
year: 2026
id: 2609.37731
tags: [DECOMP, BASELINE, CITE-FIX]
primary_tag: DECOMP
in_our_bib: no
cited_in_draft: no
themes: [decompositions, baselines, model-diffing]
read_from: "arXiv HTML v2: abstract, sections 1-6 (method, experiments, IOI and semantic case studies); appendices not read"
digest_buckets: [1-missing-method]
importance: 5
---

# Weights Read and Write Features: Scalable Parameter Decomposition Grounded in Activation Space (Cao 2026)

**What it does.** Proposes ASPD, a parameter decomposition in which each rank-one weight component u_c v_c^T is gated
by the same coordinate of a BatchTopK encoder that is also trained to reconstruct the residual stream, so each
component "reads" and "writes" an activation feature. It drops VPD's model-output (KL) objective and its
parameter/ablation losses for a local reconstruction loss at the decomposed matrix's own output. It is compared with
VPD, VPD plus the local loss, and a "PD Transcoder" on one matrix each of GPT-2 small, Gemma-2-2B and Qwen-3-8B (2B
tokens, L0 = 32), then trained model-wide on GPT-2 small to trace IOI at the weight level.

**Key result.** Intruder interpretability (chance 0.2): ASPD 0.62 against VPD 0.20 on Gemma-2-2B, and 0.57 against
0.22 on Qwen-3-8B. Causal weight editing localisation ratio (random = 1): ASPD 6.7x (single target) and 11.1x (multi)
on Qwen-3-8B, VPD 0.9x and 0.8x. On GPT-2 small, removing an identified QK component pair in head 8.6 suppresses the
S-inhibition attention pattern. "Parameter Diffing" of a fine-tune against its base is named as future work.

**Quote.** "One potential application is Parameter Diffing (identifying what changed in the mechanisms of the
fine-tuned compared to the base model) ... We leave this direction for future work." (Section 3.3)

**Relevance to our paper.** It is the nearest scalable alternative to VPD as a featurization, and it reports VPD
components near chance on interpretability and at or below random on editing at Qwen-3-8B. That bears directly on our
Qwen3-1.7B result, where the re-gating share saturates near 0.99 for every update: a reviewer can ask whether the
saturation comes from the decomposition being poor at that scale rather than from our readings. ASPD's components are
input-gated rank-one pieces like VPD's, so our kept clause, needed set and ρ apply to them unchanged; an ASPD
featurization of Qwen would be a direct test. It also supports our novelty claim that no decomposition has been used to
diff a fine-tune, since the authors list that as future work; the draft should cite it there and in the
parameter-decomposition paragraph. Caveat: their VPD comparison uses their own training of VPD on single matrices,
not Goodfire's released decompositions.

**For the digest.**
- 1-missing-method: an alternative rank-one, input-gated decomposition that claims to scale where VPD fails (Qwen-3-8B); our definition runs on it as is, and it is the natural check on the Qwen "no call".
