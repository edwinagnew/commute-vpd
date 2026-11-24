---
slug: marks2026-persona-selection-model
title: "The persona selection model"
authors: Marks et al. (Lindsey, Olah)
year: 2026
id: https://www.lesswrong.com/posts/dfoty34sT7CSKeJNn/the-persona-selection-model
tags: [EH, DEFINITION, ALIGNMENT, CITE-FIX]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation, reuse-definitions]
read_from: "LessWrong post full text (via LW GraphQL, posted 2026-02-23); key passages grepped and quoted from it"
digest_buckets: [3-definition, 4-supports]
importance: 6
---

# The persona selection model (Marks 2026)

**What it does.** An essay that states and surveys evidence for the persona selection model (PSM). Pre-training teaches
an LLM a distribution over personas, and post-training refines one, the Assistant, as a posterior updated on training
episodes. It reports no new controlled experiments ("no originality for the ideas presented here"). The evidence is
cited work: generalization effects (emergent misalignment, inoculation prompting), base-to-chat SAE transfer, persona
features that steer behaviour, and the "Assistant Axis".

**Key result.** No new numbers. The interpretability argument is that representations are largely reused: SAEs
"typically transfer well" from pre-trained to post-trained models, and post-trained-only features are few ("under 1% in
Minder et al's setting"). The authors limit their own claim. PSM "does not rule out learning of new capabilities during
post-training" (tool-call syntax is their example). They also warn of a streetlight effect: reused representations may
be easier to interpret, so the evidence may lean toward PSM.

**Quote.** "Not all representations in post-trained models are reused from pre-training, as we discuss below. Additionally, it may be the case that reused representations are systematically more interpretable than representations that are learned from scratch during post-training." (Interpretability evidence, Caveat)

**Relevance to our paper.** The draft lists persona selection as planned related work but has no entry for it. This is
the source to cite, as the persona-level form of EH. Its definition is about *which character* is selected and its
evidence about *representations*, so it is the clearest example of our gap sentence: no test of whether the mechanisms
are kept. The streetlight caveat is a useful citation for why our known answers must include new-by-construction cases:
a method that reads only base-interpretable directions would read reuse by default, as our neuron featurization does
(≥ 0.999). Its tool-syntax example (a capability learned in post-training, explained as the Assistant knowing it) shows
that "new" at the behaviour level can be "selection" at the persona level, which our featurization-relative definition
can state precisely.

**For the digest.**
- 3-definition: PSM is EH at the persona level; contrast with our mechanism-level definition; cite (draft has it planned, not in refs.bib).
- 4-supports: summarises representation reuse (SAE transfer, <1% chat-only features) and states the streetlight caveat our known answers address.
