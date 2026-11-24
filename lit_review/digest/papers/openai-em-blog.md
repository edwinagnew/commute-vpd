---
slug: openai-em-blog
title: "Toward understanding and preventing misalignment generalization (OpenAI blog, companion to Wang et al. 2025)"
authors: OpenAI
year: 2025
id: https://openai.com/index/emergent-misalignment/
tags: [KNOWN-ANSWER, DIFFING, EH, ALIGNMENT, SUPPORTS]
primary_tag: KNOWN-ANSWER
in_our_bib: "no (the companion paper is wang2025persona, 2506.19823)"
cited_in_draft: "no (wang2025persona is cited)"
themes: [model-diffing, interpreting-finetunes, elicitation]
read_from: "blog post full (fetched 2026-11-23 with a browser user agent; the earlier 403 no longer occurs); companion paper not re-read"
digest_buckets: [2-known-answer-case, 4-supports]
importance: 3
---

# Toward understanding and preventing misalignment generalization (OpenAI 2025)

**What it does.** A blog summary of Wang et al. 2025 ("Persona Features Control Emergent Misalignment"). GPT-4o and
o3-mini are fine-tuned (SFT, and RL against a grader) to give wrong answers in narrow domains such as insecure code or
bad car advice, which produces broad misalignment. An SAE trained on the base model under GPT-4o is used to diff
activations before and after fine-tuning; the latents whose activity rises are inspected on pre-training text and
steered.

**Key result.** One "misaligned persona" latent, active on quotes from morally questionable characters in pre-training
text, rises more after incorrect-data fine-tuning than after correct-data fine-tuning and predicts the misalignment
score. Steering it up makes the original model misaligned; steering it down suppresses misalignment in the fine-tuned
models, almost completely for insecure code but less for bad legal advice. Re-alignment takes 30 SFT steps (120
examples) of correct data.

**Quote.** "This suggests emergent misalignment works by strengthening a misaligned persona in the model."

**Relevance to our paper.** The substance is in Wang et al., which the draft already cites; cite the paper, not the
blog. The blog states the claim in our terms more plainly than the paper's title: a realistic fine-tune is read as
strengthening a base-model variable, with evidence from a base-model SAE, the latent's pre-training activations, and
two-way steering. That is evidence about the variable's value and its causal role, not about whether F computes it with
M's mechanism, so it is a known-answer candidate only in the weak, by-value sense, and on closed models we cannot run.
An open replication (e.g. Qwen insecure-code fine-tunes with a base SAE) would be the runnable version.

**For the digest.**
- 2-known-answer-case: GPT-4o narrow wrong-answer fine-tunes; base-SAE persona latent rises and steers both ways (value-level reuse; closed model).
- 4-supports: post-training "strengthens" a pre-existing variable, the re-weighting reading of EH.
