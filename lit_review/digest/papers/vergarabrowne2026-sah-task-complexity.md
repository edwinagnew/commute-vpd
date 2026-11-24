---
slug: vergarabrowne2026-sah-task-complexity
title: Operationalising the Superficial Alignment Hypothesis via Task Complexity
authors: Vergara-Browne et al.
year: 2026
id: 2602.15829
tags: [EH, DEFINITION, CITE-FIX]
primary_tag: DEFINITION
in_our_bib: no
cited_in_draft: no
themes: [elicitation, reuse-definitions]
read_from: arXiv HTML abstract+section 3 (definitions 3.1-3.7)+experimental setup+sections 6-9 (results, pre/post-training checkpoints, conclusion, limitations); section 4 method list only skimmed
digest_buckets: [3-definition, 4-supports]
importance: 6
---

# Operationalising the Superficial Alignment Hypothesis via Task Complexity (Vergara-Browne 2026)

**What it does.** Gives the superficial alignment hypothesis (SAH) a precise, information-theoretic definition. Task
complexity C(T_δ) is the length of the shortest program reaching score δ on task T; conditional complexity C(T_δ | θ) is
the same with free access to the weights θ. A model is (b, δ)-adaptable if C(T_δ | θ) ≤ b, and the SAH says many complex
tasks are adaptable with small b at high δ. The paper upper-bounds C(T_δ | θ) by Pareto curves of program length
against score over three families of adaptation (inference control: ICL, URIAL; data: subset training; parameters:
LoRA, Bayesian-LoRA), on SmolLM3-3B, Olmo3-7B and Olmo3-32B (stage-1 pre-training checkpoints, plus final pre- and
post-training checkpoints) for GSM8K, English-to-French FLORES and IFEval.

**Key result.** Short programs suffice: 4,358 bits take Olmo3-32B on GSM8K to 72.2%, and 3,992 bits take Olmo3-7B from
22.63 to 34.43 BLEU. Pre-training makes strong performance reachable (Olmo3-7B best GSM8K 1.1% from random init vs 67.6%
from pre-trained) but at high complexity (megabytes to gigabytes); post-training "collapses" the complexity of reaching
that performance by orders of magnitude. Training only the output layer (Chen et al. 2025) is far from the Pareto curve.
It revisits Raghavendra et al. (2024) and Lambert (2025), arguing their few-data saturation test conflates the SAH with
a fixed target performance.

**Quote.** "the SAH simply claims that pre-trained models drastically reduce the complexity of achieving high
performance on many tasks." (Abstract)

**Relevance to our paper.** This is the cleanest recent definition of the elicitation side of the question, and it is
exactly the kind our gap sentence describes: behavioural (program length to reach a score), with no look inside the
model. Under it two fine-tunes with the same behaviour and the same short adaptation program are equally superficial,
whereas our definition can tell them apart by whether the needed variables keep M's mechanisms; that contrast is worth
one or two sentences in "Post-training as selection" or "Defining reuse". It also bears directly on two papers the
draft already cites (Raghavendra et al., Lambert), so the related-work paragraph on the SAH is incomplete without it.
Its result (post-training lowers the cost of reaching performance that pre-training made reachable) is behavioural
support for EH, not for or against our mechanism-level readings. Note its setting: a LoRA update counts as a long
program, so our small fine-tunes would be "superficial" here regardless of their ρ.

**For the digest.**
3-definition: an information-theoretic, behavioural definition of the SAH (bits needed to adapt), to contrast with our mechanism-level definition of reuse.
4-supports: behavioural support for EH: post-training collapses the complexity of reaching performance that pre-training already made accessible.
