---
slug: lambert2025-elicitation-post
title: "Elicitation, the simplest way to understand post-training (Interconnects blog)"
authors: Nathan Lambert
year: 2025
id: https://www.interconnects.ai/p/elicitation-theory-of-post-training
tags: [EH, DEFINITION]
primary_tag: DEFINITION
in_our_bib: lambert2025elicitation
cited_in_draft: yes
themes: [elicitation, reuse-definitions]
read_from: "blog post full (Interconnects, 2025-03-10), via WebFetch summary with quotes checked against the page HTML"
digest_buckets: [3-definition]
importance: 5
---

# Elicitation, the simplest way to understand post-training (Lambert 2025)

**What it does.** An essay proposing the "elicitation interpretation" of post-training (everything after most of
pretraining: annealing, instruction tuning, preference tuning, RLVR). Uses a Formula 1 analogy (same chassis, large
in-season gains) and examples: OLMoE Instruct's post-training average rising from 35 to 48 without new pretraining,
the Superficial Alignment Hypothesis, large instruction sets such as OpenMathInstruct 2, and a few thousand RL prompts
beating millions of SFT samples on MATH.

**Key result.** No experiments of its own. Argues that post-training mostly extracts and amplifies base behaviours, that
LIMA's hypothesis is right "for the wrong reasons" (post-training is not superficial), and that new skills can be
taught in post-training but are hard to tell apart from skills seen in pretraining.

**Quote.** "...the elicitation interpretation of post-training, where all we are doing is extracting and amplifying
valuable behaviors in the base model." Also: "it is clear that you can teach new skills to models in post-training."

**Relevance to our paper.** The source of our hypothesis's name, and our draft cites it for the term only, which is
fine. Lambert's notion is behavioural ("behaviors", "latent in the pretraining dataset"), so our EH is a mechanistic
sharpening of it, a contrast the draft could make explicit. Two useful points: his "amplifying" matches our kept clause
(a mechanism kept up to a positive scale), and he concedes that genuinely new skills exist and that the hard part is
telling them from seen ones, which is the motivation for a definition and known-answer tests.

**For the digest.**
- 3-definition: elicitation = "extracting and amplifying valuable behaviors in the base model" (behavioural); our EH is its mechanistic version and his "amplifying" matches our rescale clause.
