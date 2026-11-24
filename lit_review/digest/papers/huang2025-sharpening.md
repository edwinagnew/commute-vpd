---
slug: huang2025-sharpening
title: "Self-Improvement in Language Models: The Sharpening Mechanism"
authors: Huang et al.
year: 2025
id: 2412.01951
tags: [DEFINITION, EH, KNOWN-ANSWER]
primary_tag: DEFINITION
in_our_bib: no
cited_in_draft: no
themes: [elicitation, reuse-definitions]
read_from: "local PDF arxiv_2412.01951.pdf (v2) via pdf-text: abstract, §1 (sharpening definition, contributions), §5.2-5.4 experiments incl. Table 1 (SFT-Sharpening); theory sections 3-4 and proofs not read in detail"
digest_buckets: [2-known-answer-case, 3-definition]
importance: 4
---

# Self-Improvement in Language Models: The Sharpening Mechanism (Huang 2025)

**What it does.** A theory paper on self-improvement. It defines "sharpening" as tilting a base policy toward the
responses it itself rates most certain, with the self-reward taken as sequence log-likelihood (maximum-likelihood
sharpening). It frames this as amortising an expensive inference-time search, and proves sample-complexity results for
an SFT-based method (fine-tune on the base model's best-of-N most-likely samples) and an RLHF-based method. Experiments
test best-of-N at inference across 7 models and 6 tasks. They also run SFT-Sharpening with LoRA on a few model-task
pairs.

**Key result.** SFT-Sharpening is minimax-optimal when the base model has enough coverage. RLHF-Sharpening can do
without coverage by exploring. Best-of-50 by likelihood often beats greedy decoding. SFT-Sharpening with LoRA amortises
it, for example Phi3.5-Mini on MATH +19.2% relative accuracy over greedy and Mistral-7B on MATH +8.9% (Table 1).

**Quote.** "We refer to sharpening as any process that tilts πbase toward responses that are more certain in the sense
that they enjoy greater self-reward rself." (§1.1)

**Relevance to our paper.** This is the formal source for "sharpening", the behavioural counterpart of elicitation
that the planned "elicitation-versus-sharpening" related-work paragraph needs. Sharpening is defined on the output
distribution, so two fine-tunes that sharpen equally can differ under our definition, which is our point. The paper also
says sharpening is "a computational phenomenon rather than a statistical one". A sharpened model has no new
information but may need new computation to produce the argmax directly, so "no new information" does not imply
"reused mechanisms". That is a useful distinction for our discussion. SFT-Sharpening (LoRA on the base model's own
most-likely samples) is a cheap constructed fine-tune whose training targets come entirely from the base model. It is a
candidate "elicitation by construction" case that we could build on the 67M model, though its mechanistic status is
not known in advance.

**For the digest.**
- 2-known-answer-case: SFT on the base model's own maximum-likelihood best-of-N samples is a fine-tune with no external information, buildable on our 67M or Qwen. Its status is known only at the information level, not the mechanism level.
- 3-definition: "sharpening" = tilting the base policy toward its own high-certainty outputs, a distribution-level definition of elicitation to cite and contrast with ours.
