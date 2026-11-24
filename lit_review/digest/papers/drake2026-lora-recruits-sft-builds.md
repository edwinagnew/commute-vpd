---
slug: drake2026-lora-recruits-sft-builds
title: "Transplanting, inverting, and preventing a misalignment persona: method-conditional emergent misalignment in Qwen2.5"
authors: Lyndon Drake et al.
year: 2026
id: 2607.04510
tags: [KNOWN-ANSWER, FT-INTERP, DIFFING, EH, ALIGNMENT, CITE-FIX]
primary_tag: KNOWN-ANSWER
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, model-diffing, elicitation, baselines]
read_from: arXiv HTML abstract, sections 1-3, Part I-II (sections 8-12) and Part III (sections 13-14) by grep; Part IV only via abstract
digest_buckets: [1-missing-method, 2-known-answer-case, 4-supports]
importance: 6
---

# Transplanting, inverting, and preventing a misalignment persona (Drake 2026)

**What it does.** Asks when a narrow fine-tune "recruits" a pre-existing misalignment-persona direction versus building
a localised solution. On Qwen2.5-32B (base and instruct) it fine-tunes on Betley et al.'s insecure-code data with
rs-LoRA at several ranks and with full SFT (same data, template, weights), plus overt inducers (bad medical, financial,
extreme sports). It reads each update by projecting the activation drift and the weight update onto a persona axis and
an alignment ("erosion") axis, and tests the persona direction causally by ablation and by transplanting it into a
model that shares only pretraining with the source.

**Key result.** LoRA at low rank on insecure code gives broad EM (3.4% misaligned) and moves along the persona axis
(drift-persona cosine +0.17 at rank 1), while full SFT stays at the floor (0.3%) and moves against it (-0.10);
weight-update enrichment on the persona axis is ~3.2-3.7 for insecure LoRA against ~1.3-1.7 for secure-code and SFT
cells. Transplanting the direction induces broad EM (2.83% against a ~1.1% random floor) and ablating it halves an
overt inducer's broadcast (21% to 10%). The direction is readable in the base frame (classifier AUC ~0.89), but steering
the base model along it mostly damages coherence: "represented in the base, expressed only after instruction-tuning".
Erosion along the alignment axis is near-uniform (cos ≈ -0.75 to -0.80) across all instruct cells and does not predict
EM. Several results are single-seed.

**Quote.** "full SFT, with sufficient parametric freedom to do so (at least on this model family), builds a dedicated,
localised solution for the narrow training task, while LoRA, constrained by rank and technique, recruits the EM
persona" (Part I)

**Relevance to our paper.** The closest thing in this batch to a realistic known-answer pair: the same data trained two
ways, one shown (by transplant, ablation and base-frame readability) to run through a pre-existing direction, the other
not. It is weaker than our built pairs on the "new" side: "builds a dedicated circuit" is inferred from the absence of
recruitment, rank and norm controls, not from a traced mechanism, and the scale (32B) is out of reach for our VPD
setting; a scaled-down replication on Qwen3-1.7B would be needed to use it. Its persona-axis projection of the weight
update is a cheap reading of "what an update reused" that we neither run nor cite; it is close to our mean-difference
steering share and could be added as a baseline wherever a behaviour-derived direction exists. Its finding that
full SFT can move against a base direction while LoRA amplifies it is a concrete case where method, not data, sets
reuse (on the training task both fit insecure code; their broad behaviour differs, so it is not a same-behaviour pair). Its "represented but not
expressed" gap is an EH-style finding at the activation level that the related-work section on post-training as
selection could cite. Should be cited next to Wang et al. 2025.

**For the digest.**
- 1-missing-method: projection of the weight update and activation drift onto a behaviour-derived persona axis (and
  cross-model transplant of the direction) as a reading of reuse; runnable on Qwen3-1.7B if an EM direction exists there.
- 2-known-answer-case: Qwen2.5-32B insecure code, low-rank LoRA = recruits a pre-existing persona (transplant +
  ablation + base readability) vs full SFT = no recruitment, anti-persona drift (inferred localised solution).
- 4-supports: two fine-tunes of the same data differ in whether they go through a base mechanism, so the method, not
  the task, sets reuse; supports treating reuse as a property of the update, read inside the model.
