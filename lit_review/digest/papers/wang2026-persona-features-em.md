---
slug: wang2026-persona-features-em
title: "Persona Features Control Emergent Misalignment"
authors: Wang et al. (OpenAI)
year: 2025
id: 2506.19823
tags: [KNOWN-ANSWER, BASELINE, DIFFING, ALIGNMENT, EH, SUPPORTS]
primary_tag: KNOWN-ANSWER
in_our_bib: wang2025persona
cited_in_draft: yes
themes: [elicitation, model-diffing, interpreting-finetunes, baselines]
read_from: "whip-lead's digest + arXiv abstract (export.arxiv.org API, v2 Oct 2025), plus the local summary lit_review/summaries/Personafeaturescontrol_emergent_misalignment.md"
digest_buckets: [1-missing-method, 2-known-answer-case, 4-supports]
importance: 7
---

# Persona Features Control Emergent Misalignment (Wang 2025)

Note: the slug says 2026 (the ICLR 2026 version); the arXiv paper and our bib key are 2025.

**What it does.** Fine-tunes GPT-4o (and runs RL on o3-mini) on narrow wrong data (insecure code, bad advice) that
makes the model broadly misaligned. It applies a sparse autoencoder trained on the pre-trained model (2.1M latents,
middle layer) before and after the fine-tune, ranks latents by their rise in mean activation, and steers the top 1000
to find the causal ones.

**Key result.** Ten latents steer misalignment; one "toxic persona" latent (#10) induces it when added to GPT-4o and
suppresses it when subtracted from the fine-tunes. Its activation change separates aligned from misaligned models and
rises at about 5% wrong data, before misalignment is visible in behaviour. About 120 benign samples re-align the model.

**Quote.** "This approach reveals several "misaligned persona" features in activation space, including a toxic persona
feature which most strongly controls emergent misalignment" (abstract)

**Relevance to our paper.** The draft cites it (main.tex l.205) as a fixed-dictionary diffing method. It is also a
realistic fine-tune with outside evidence of reuse: a pre-training latent the fine-tune turns up, shown causal by
steering. That evidence is activation-level (value plus steering), not mechanism-level: it does not show that what
drives latent #10 in the fine-tune is still the base model's mechanism, which is exactly the gap our definition names.
whip-lead's reading: "activation-level reuse only (steering), SAE-diff recipe cheap baseline (~30 min debug) likely
fails built-new control". The recipe (base SAE, rank by mean-activation rise, steer) is cheap to run on our 67M with
the BatchTopK SAE we already have; a fixed dictionary has no unit for a new feature, so it should fail the built-new
pairs. Closed models, so the case cannot be re-read mechanistically by us; open-model replications (Soligo et al.,
Nadaf et al.) are the usable versions.

**For the digest.**
- 1-missing-method: base-SAE mean-activation diff + steering; runnable on 67M with our SAE; expected to miss built-new edits.
- 2-known-answer-case: GPT-4o EM fine-tune reuses a pre-training persona latent; activation/steering-level evidence only; closed weights.
- 4-supports: fine-tune works through an existing base feature, turned up, as our "rescale" picture predicts at the value level.
