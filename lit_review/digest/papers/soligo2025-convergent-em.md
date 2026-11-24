---
slug: soligo2025-convergent-em
title: Convergent Linear Representations of Emergent Misalignment
authors: Soligo et al. (Turner, Rajamanoharan, Nanda)
year: 2025
id: 2506.11618
tags: [ALIGNMENT, FT-INTERP, BASELINE, KNOWN-ANSWER, DIFFING]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, baselines, model-diffing]
read_from: "local PDF text (ICML 2025 Actionable Interp workshop, arXiv v2): abstract, Sec. 1-3 (model organism, mean-diff directions, steering, ablation, transfer, B-vector vs mean-diff), Sec. 4 adapter interpretation (skimmed)"
digest_buckets: [1-missing-method, 2-known-answer-case]
importance: 5
---

# Convergent Linear Representations of Emergent Misalignment (Soligo 2025)

**What it does.** Builds a minimal emergent-misalignment (EM) model: Qwen2.5-14B-Instruct fine-tuned with 9 rank-1 LoRA
adapters on bad medical advice. It extracts a "misalignment direction" from the fine-tune as the mean difference of
residual activations between misaligned and aligned answers. It steers the original chat model with that direction,
ablates it from the EM model and from other EM fine-tunes (rank-32 LoRA, other datasets), and reads the rank-1 adapters'
scalar activations to see which adapters are general and which are medical-specific.

**Key result.** Adding the direction to the chat model gives up to 50% EM responses, four times the fine-tune's own
rate. Ablating it from the EM model takes EM from 11.25% to 0% (layer-wise) or to 1% (one direction). Transfer ablation
cuts misalignment in other EM fine-tunes by 90% and 78%, with coherence above 99%, and random directions change it by at
most 10%. But for a single rank-1 LoRA at layer 24, the adapter's B vector has cosine only 0.04 with the mean-diff
direction, although ablating the latter removes over 98% of the EM. Six adapters drive general misalignment and two
drive medical-only misalignment.

**Quote.** "different emergently misaligned models converge to similar representations of misalignment" (abstract)

**Relevance to our paper.** This is a realistic fine-tune with partial independent evidence for a base mechanism. A
direction taken from the fine-tune steers the unfine-tuned chat model into the same behaviour, so the chat model already
contains a pathway the fine-tune uses. In value terms that is reuse; whether the mechanism is kept is untested. It is
the strongest public case for our mean-difference steering share as a reading, and a candidate realistic test: the
rank-1 adapters are tiny, open-sourced and act on a known direction. The B-vector result (cosine 0.04, the same
downstream effect) is a warning that matches our "turned writes" finding: a weight update can reach a base direction's
downstream effect without lying along it, so weight-direction readings and activation-direction readings can disagree.
Running our readings on a 14B model would need a decomposition we do not have; a small-model EM organism (from the
companion Turner et al. 2025) would be the feasible version.

**For the digest.**
- 1-missing-method: mean-diff steering and directional ablation with transfer across fine-tunes; we have a steering share but not cross-fine-tune transfer ablation.
- 2-known-answer-case: Qwen2.5-14B-Instruct, rank-1 LoRA EM fine-tune on bad medical advice; steering the chat model with the fine-tune's direction reproduces EM (evidence of a reused pathway), ablation removes it.
