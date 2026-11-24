---
slug: venhoff2025-base-know-how
title: "Base Models Know How to Reason, Thinking Models Learn When"
authors: Venhoff et al.
year: 2025
id: 2510.07364
tags: [EH, BASELINE, DIFFING, FT-INTERP, KNOWN-ANSWER, SUPPORTS, CITE-FIX]
primary_tag: EH
in_our_bib: venhoff2025base
cited_in_draft: yes
themes: [elicitation, model-diffing, interpreting-finetunes, baselines]
read_from: "arXiv HTML v4 (7 Jul 2026): abstract, intro, sections 2-3 (taxonomy, constructive model diffing, Tables 1-2, ablations), related work, discussion, conclusion"
digest_buckets: [1-missing-method, 2-known-answer-case, 4-supports, 5-cite-fix]
importance: 7
---

# Base Models Know How to Reason, Thinking Models Learn When (Venhoff 2025)

**What it does.** Asks what thinking models learn that their base models lack. Small Top-K SAEs (5-50 latents) on
sentence-mean activations of the thinking model's traces give a taxonomy of reasoning behaviours. "Constructive model
diffing" then builds a hybrid: the base model generates; where base and thinking model disagree on the next token,
the thinking model's SAE picks a category, and a category vector trained in the base model (plus a small MLP for the
coefficient) is added. Nine base/thinking pairs, 0.5B-32B: four RL-trained (Open-Reasoner-Zero on Qwen2.5), four
SFT-distilled (DeepSeek-R1-Distill), one mixed (QwQ-32B).

**Key result.** In v4 the hybrid recovers "roughly 76%" of the base-to-thinking accuracy gap for the RL-trained pairs
but "only 11%" for the SFT-distilled ones (GSM8K, MATH500, held-out Hendrycks-MATH), steering 5-12% of tokens for the
RL pairs. Category vectors also reach lower loss for RL-derived taxonomies. Negative controls (random category 20-28%,
norm-matched random vectors -28% and -13%, magnitude only, random positions) all cut recovery.

**Quote.** "RL primarily teaches heuristics for orchestrating pre-existing base mechanisms, whereas SFT-distillation
installs new ones" (Abstract)

**Relevance to our paper.** This is the closest published claim to our question on realistic post-training: it
states a reuse-versus-new verdict that depends on the training regime, from a steering reading. Our draft cites it
only for the gap-recovery comparison borrowed by our mean-difference steering share; its headline finding (RL reuses,
SFT installs) belongs in the related-work paragraph on post-training as selection and bears on our dropped claim that
RL on the repeat task reads higher (0.05-0.22). Its evidence is a steering reading, so by our standard it reads
activations, not mechanisms: two fine-tunes with the same steerable directions could differ under our definition. It
also uses the thinking model at inference (SAE gating, coefficient MLP), so "recovery" credits the base model with a
direction while the fine-tune supplies when and how strongly; the authors note the low SFT recovery "could reflect
either genuine mechanism modification or limitations in our category vector optimization". The ORZ pairs (Qwen2.5-0.5B
and 1.5B) are small, public, and come with a labelled answer by an independent method, so they are a candidate
known-answer pair for our Qwen line (we hold no VPD of Qwen2.5; SAE or neuron featurizations would run).

**For the digest.**
- 1-missing-method: constructive model diffing (category vectors in the base model plus a heuristic from the fine-tune, scored by gap recovered) is a fuller version of the steering baseline we only partly borrow.
- 2-known-answer-case: ORZ (RL) vs R1-Distill (SFT) on Qwen2.5 bases, labelled reuse vs new by steering recovery with ablation controls; usable as a soft label, not a by-construction one.
- 4-supports: RL elicits more than SFT, by an independent reading; supports the RL-reads-higher observation the draft no longer claims.
