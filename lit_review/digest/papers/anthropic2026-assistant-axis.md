---
slug: anthropic2026-assistant-axis
title: "The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models"
authors: Lu et al. (Anthropic / MATS)
year: 2026
id: 2601.10387
tags: [EH, SUPPORTS, ALIGNMENT, DIFFING]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation, model-diffing]
read_from: "arXiv HTML v1 (linked from the Anthropic research page, which was JS-only): abstract, sections 2-3 incl. 3.2.2 base models, appendix B.3 base vs instruct"
digest_buckets: [4-supports]
importance: 4
---

# The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models (Lu 2026)

**What it does.** The paper extracts 275 role vectors (mean activations when the model is system-prompted as each character) in Gemma 2 27B, Qwen 3 32B and Llama 3.3 70B, and runs PCA on them. The "Assistant Axis" is defined as the default-Assistant vector minus the mean of the others, and it aligns with the first principal component. The paper steers along it, uses projections onto it to track "persona drift", and caps activations along it as a defence. For Gemma 2 27B and Llama 3.1 70B it compares against the base models.

**Key result.** Steering toward the axis makes models more Assistant-like and resistant to persona jailbreaks; steering away makes them adopt other identities. Running the instruct rollouts through base Gemma 2 27B gives nearly the same principal components, with cosine "> 0.99" for every role vector between base and instruct. Steering base models with the instruct-derived axis yields completions from helpful human archetypes (consultants, coaches) and suppresses spiritual ones. The paper concludes that the axis "mainly inherits" from personas already in the base model, and that post-training adds associations such as being an AI.

**Quote.** "Our results suggest that post-training steers models toward a particular region of persona space but only loosely tethers them to it" (abstract)

**Relevance to our paper.** This is another activation-level piece of evidence that post-training selects a region of a persona space already present in the base model. It fits the planned persona-selection paragraph next to Marks et al. and Moskvoretskii et al. One caveat matters for how we cite it: the base-versus-instruct comparison runs both models on the instruct model's rollouts, so the >0.99 similarity is of representations on identical text, not of mechanisms. That is the "reads activations" kind of evidence our gap sentence describes. It is not usable as a known-answer case, because no fine-tune is traced at the level of mechanisms.

**For the digest.**
- 4-supports: an instruct-derived Assistant direction is already present in, and steers, the base model (Gemma 2 27B, Llama 3.1 70B); post-training adds associations on top of an inherited axis, by an activation-level reading.
