---
slug: lindsey2024-crosscoders
title: "Sparse Crosscoders for Cross-Layer Features and Model Diffing"
authors: Jack Lindsey et al.
year: 2024
id: https://transformer-circuits.pub/2024/crosscoders/index.html
tags: [BASELINE, DIFFING, FT-INTERP, SUPPORTS]
primary_tag: BASELINE
in_our_bib: lindsey2024crosscoders
cited_in_draft: yes
themes: [model-diffing, baselines, interpreting-finetunes]
read_from: "Transformer Circuits note full text (prose extracted from HTML): motivation, model-diffing section, Claude 3 Sonnet fine-tune experiment, future directions; figures not seen"
digest_buckets: [1-missing-method, 4-supports]
importance: 7
---

# Sparse Crosscoders for Cross-Layer Features and Model Diffing (Lindsey 2024)

**What it does.** Introduces crosscoders: one sparse dictionary that reads from and writes to several layers or several
models, so a latent can have a decoder in each. For model diffing it trains a 1M-latent crosscoder on the middle-layer
residual stream of Claude 3 Sonnet and its base model, and sorts latents by the relative norm of their two decoders into
base-specific, fine-tune-specific and shared. Also applies crosscoders to cross-layer superposition and to scaling.

**Key result.** Latents fall into three clear groups, with 4,000-5,000 model-specific latents per model out of 1M; some
fine-tune-specific ones are interpretable (refusal, code review, personal questions to the Assistant), but "the majority
of the model-exclusive features are not immediately interpretable". For shared latents the two decoders are almost
always highly aligned, but for a few thousand the correlation is very low or negative, which the authors suspect marks
a base concept used in a new way. They call the diffing results preliminary and "a bit mixed".

**Quote.** "We suspect that these indicate cases where the finetuned model uses a concept that was present in the base
model, but in a new way." (model-diffing section)

**Relevance to our paper.** The original crosscoder diffing method; we run its BatchTopK successor (Minder et al.) as a
baseline, and our related-work paraphrase is accurate. Three things to use. (1) The low-correlation shared latents are
close to our main finding that trained fine-tunes turn writes rather than scale them: a variable kept but rotated. The
draft should say that Lindsey et al. saw this in a crosscoder and set it aside, which is direct prior evidence for the
rotation we report. (2) They contrast crosscoders with applying a base SAE to the fine-tune (Kissane et al.), which
shows "how the finetune model recruits abstractions that were already present in the base model in novel contexts",
and they say crosscoders suit fine-tunes with much compute; our crosscoder failing at fine-tune size fits that. (3) The
decoder-cosine reading of shared latents is a cheap diffing signal we do not run; it could be computed from our
existing crosscoder and set against the known-answer pairs at no cost.

**For the digest.**
- 1-missing-method: decoder-cosine of shared crosscoder latents (concept reused "in a new way") is a reading we could take from our trained crosscoder and score on the nine built pairs.
- 4-supports: shared latents with low or negative decoder correlation in Claude 3 Sonnet vs base are prior evidence that fine-tunes rotate reused variables rather than rescale them.
