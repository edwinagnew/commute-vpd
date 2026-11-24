---
slug: aranguri2025-tied-crosscoders
title: "Tied Crosscoders: Explaining Chat Behavior from Base Model"
authors: Aranguri
year: 2025
id: https://www.lesswrong.com/posts/3T8eKyaPvDDm2wzor/research-question
tags: [DIFFING, BASELINE, EH, SUPPORTS, CITE-FIX]
primary_tag: BASELINE
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines, elicitation]
read_from: "LessWrong post full (markdown via LessWrong GraphQL; figures not viewed), posted 2025-03-22"
digest_buckets: [1-missing-method, 4-supports, 5-cite-fix]
importance: 5
---

# Tied Crosscoders: Explaining Chat Behavior from Base Model (Aranguri 2025)

**What it does.** Argues that standard crosscoders split one concept into a base-exclusive and a chat-exclusive latent
when fine-tuning changes when the concept fires. It proposes a tied crosscoder: latents come in triples (shared, base,
chat) with one decoder direction per model. The shared latent fires for both, and the base or chat latent lets the
same concept fire at different times in one model; a penalty η = 2 makes exclusive firing cost more. It is trained on
layer 14 of Gemma-2-2b and Gemma-2-2b-it (16,384 x 3 latents, 250M tokens of Pile plus LMSys). A toy fine-tune of
Gemma-2-2b swaps " blue"/" green" targets half the time to correlate two existing features; a standard crosscoder is
trained on it.

**Key result.** The ratio of chat-latent to shared-latent firing counts is offered as a novelty score. Chat-side
latents such as refusal (10615) have base-side partners firing on refusals and Q/A in Pile text, and a second refusal
latent (14643) pairs with legal and medical disclaimers in the base model. In the blue/green toy, a standard
crosscoder gives one exclusive latent per model for the same concept, with decoder cosine 0.97. In a standard
crosscoder on Gemma, exclusive latents are closer to some other latent than shared latents are. The author found no
chat latent without a more frequent shared partner.

**Quote.** "An explanation for this is that since fine-tuning enhances existing mechanisms, the chat model reuses the
features from the base model in new contexts." (Abstract)

**Relevance to our paper.** It is a crosscoder built for our question: each chat latent gets a tied base partner and a
graded novelty ratio, where a standard crosscoder gives only an exclusive/shared split. It could run on our 67M
fine-tunes and built pairs at the cost of a crosscoder, and it is the obvious strengthened version of the crosscoder
baseline that fails our planted control. A reviewer will ask whether a tied crosscoder fixes it. The blue/green toy is
a constructed reuse case (an existing feature fires in new contexts) that a standard crosscoder misreads as two
exclusive latents. That independently supports our claim that crosscoders cannot tell reuse from new, and the draft
should cite it there. Its evidence for "chat reuses base features" is top-activation inspection only, with no causal
test and one η; and "firing at different times" is a change in when a feature is used, which our kept clause would
count as reuse only if the downstream mechanism is kept. Note: ASPD cites "Aranguri and McGrath, 2025" for model
diffing, possibly a later version of this work; check before citing.

**For the digest.**
- 1-missing-method: tied crosscoder with a per-latent novelty ratio, a direct crosscoder-family reuse reading we do not compare against; cheap to run on the 67M.
- 4-supports: its blue/green toy shows a standard crosscoder splitting one reused concept into two exclusive latents (cos 0.97), backing our crosscoder failure; qualitatively, chat refusal latents are built from base features.
