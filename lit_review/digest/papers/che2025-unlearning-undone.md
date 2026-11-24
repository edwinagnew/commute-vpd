---
slug: che2025-unlearning-undone
title: Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities
authors: Zora Che et al.
year: 2025
id: 2502.05209
tags: [ALIGNMENT, EH, BACKGROUND, SUPPORTS]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [elicitation, other]
read_from: arXiv HTML abstract, sections 1-2, fine-tuning-attack results (Figure 7 text) and limitations
digest_buckets: [4-supports]
importance: 3
---

# Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities (Che 2025)

**What it does.** Benchmarks 8 unlearning methods (64 released models unlearning WMDP-Bio knowledge) and 9 safety
fine-tuned models, all derived from Llama-3-8B-Instruct, against 5 input-space attacks and 6 "model tampering" attacks
(latent-space perturbations, LoRA and full fine-tuning, benign fine-tuning on WikiText). It asks whether tampering
attacks predict held-out input-space attacks, including a proprietary UK AISI attack.

**Key result.** Robustness across the 11 attacks lies on a low-dimensional subspace (first principal component
separates the adversarial fine-tuning attacks from the rest). Tampering-attack success correlates with held-out
input-space attack success and tends to upper-bound it. Every unlearning method is reversed by fine-tuning: WMDP-Bio
rises by 10 points or more "within 16 gradient steps, with 128 examples or fewer", sometimes in one step.

**Quote.** "State-of-the-art unlearning can reliably be reversed within 16 fine-tuning steps – sometimes in a single
step." (Section on fine-tuning attacks)

**Relevance to our paper.** Behavioural evidence for the background claim that a post-training change can leave the
underlying capability intact and only suppress its use (the reverse direction of elicitation), which our "why
mechanisms" notes use Deeb and Roger for. It adds scale (64 models, 8 methods) to that citation. It does not look
inside the model, so it is not a known-answer case or a baseline. Optional citation beside Deeb and Roger.

**For the digest.**
- 4-supports: behaviour after a fine-tune understates what the weights still hold (few-step relearning), the premise for reading reuse inside the model.
