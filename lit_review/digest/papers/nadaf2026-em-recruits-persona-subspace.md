---
slug: nadaf2026-em-recruits-persona-subspace
title: Emergent Misalignment Recruits a Pre-existing Persona Subspace
authors: Mohammed Suhail B Nadaf
year: 2026
id: 2607.21356
tags: [KNOWN-ANSWER, FT-INTERP, ALIGNMENT, EH, SUPPORTS, CITE-FIX]
primary_tag: KNOWN-ANSWER
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, elicitation, reuse-definitions]
read_from: arXiv HTML: abstract, sections 1-5, 7.1 (opening), 8 Discussion, 9 Limitations, start of 10 (numbers stripped from inline math in places)
digest_buckets: [2-known-answer-case, 4-supports, 5-cite-fix]
importance: 6
---

# Emergent Misalignment Recruits a Pre-existing Persona Subspace (Nadaf 2026)

**What it does.** Asks why narrow bad-advice fine-tuning makes Qwen2.5-14B-Instruct broadly misaligned, and pits an
"inference" account (the update is evidence about a pre-existing cross-domain author latent) against an "accumulation"
account (breadth is optimiser residue). It extracts rank-4 persona subspaces per domain from the frozen model by
contrastive teacher forcing (same response tokens, "reckless" vs "cautious" system prompt), then tests them causally on
Turner et al.'s LoRA organisms and on its own insecure-code fine-tunes: projecting the subspace out of the residual
stream during training, injecting it into the untouched model, projecting it out of the weight gradient, and post-hoc
weight edits.

**Key result.** Four domains share one low-rank persona core (overlap 0.513 vs a random null, ratio 657), 82% outside a
matched style core. Projecting it out of activations throughout fine-tuning takes judged broad misalignment from 27.7%
to 0.0% (matched-rank random subspace: 27.5%); injecting it into the never-fine-tuned model rises with dose to 45.4%.
The same projection on the weight gradient is inert (26.6% vs 26.7%), the persona component is about 0.3% of the
organisms' weight change, and three post-hoc weight edits leave the disposition in place. Removing the carrier at
inference from an already fine-tuned organism removes only 36% of the misalignment. The ablation also kills the narrow
trained behaviour (adherence 0.902 to 0.000), and the extraction is from an instruct, not a base, checkpoint.

**Quote.** "On this reading the disposition is read loudly and written obliquely." (Section 8)

**Relevance to our paper.** This is the cleanest realistic case we have seen of a fine-tuned behaviour that, by the
paper's own causal evidence, needs a variable present in the model before the fine-tune: a needed-variable result
(training-time necessity plus injection sufficiency, each with a random control) in the spirit of our Def. 2. What it
does not show is our kept clause: it never asks whether the mechanism computing that variable is kept, and its
weight-side findings (gradient projection inert, persona share 0.3% of the update, re-formation after edits) say the
update reaches the old activation subspace through weights that are not a rescaling of the old ones. That is the
"old representation, new mechanism" case, and it fits our finding that trained fine-tunes turn writes rather than scale
them; it is a ready example for the discussion of why "reuse" read in activations and read in mechanisms can differ.
As a known-answer case it is weak: the base model is an instruct checkpoint at 14B, the label is about a representation
not a mechanism, and the author flags provenance as open. The draft's planned "persona selection" related-work
paragraph should cite it beside Wang et al. 2025.

**For the digest.**
- 2-known-answer-case: Qwen2.5-14B-Instruct, LoRA EM organisms; a pre-existing persona subspace is causally needed for the fine-tuned behaviour (activation-side), with mechanism-level reuse untested and the weight side pointing to new writes.
- 4-supports: "read loudly, written obliquely": the update reaches an old subspace by weights the naive lift misses, consistent with our "turns rather than scales".
