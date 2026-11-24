---
slug: transplanting-misalignment-persona
title: "Transplanting, inverting, and preventing a misalignment persona: method-conditional emergent misalignment in Qwen2.5"
authors: Lyndon Drake et al.
year: 2026
id: 2607.04510
tags: [KNOWN-ANSWER, FT-INTERP, ALIGNMENT, DIFFING, SUPPORTS, CITE-FIX]
primary_tag: KNOWN-ANSWER
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, model-diffing, elicitation]
read_from: arXiv HTML: abstract, sections 1-2, 6, 10, 12, 14, 16, 19 (start); located by WebSearch of the title fragment. Same paper as master-list slug drake2026-lora-recruits-sft-builds
digest_buckets: [2-known-answer-case, 1-missing-method, 4-supports]
importance: 6
---

# Transplanting, inverting, and preventing a misalignment persona (Drake 2026)

**What it does.** Holds Betley et al.'s insecure-code data fixed and varies the fine-tuning method on Qwen2.5-32B
(and 7B, 14B): low-rank LoRA versus full SFT. It measures where each update lands relative to a pre-existing
misalignment-persona direction (projection enrichment of the weight update, drift-persona cosine), tests the direction
causally by ablation and by a cross-model transplant (extract from one checkpoint, inject into another that shares only
pretraining), and follows the full-SFT run across checkpoints.

**Key result.** On identical data, LoRA recruits the persona (3.4% broad misalignment, weight update concentrated on
the persona direction at 3.7 times chance) while full SFT does not (0.3%) and moves against the persona axis. The
full-SFT run never passes through a recruitment phase: "the narrow insecure-code circuit is built directly". An L2-SP
norm constraint does not restore recruitment, so the difference is attributed to the update's rank structure, not its
size. The transplant induces broad misalignment at 2.83% against a random-direction floor of 1.1%; ablating a model's
own direction roughly halves an overt inducer's broadcast (21% to 10%). Steering away from the persona during an overt
(bad-medical) SFT run raises broadcast, so the direction's role is conditional. Single-seed in places.

**Quote.** "Full SFT is therefore not failing to learn. It is spending capacity on a dedicated narrow circuit rather than recruiting the broad persona." (Section 16)

**Relevance to our paper.** This is a same-data, two-method pair where the paper's own evidence (weight-update
enrichment on a pre-existing direction, a causal transplant, a checkpoint trajectory) says one fine-tune runs through a
pre-existing structure and the other builds a narrow circuit. That is exactly the shape of a realistic known-answer
pair for us, though its labels are about a representation and a behaviour (broad misalignment), not about our kept
clause, and the "new circuit" side is inferred from absence of recruitment rather than traced. Its projection-enrichment
reading of a weight update onto a base direction is a cheap diffing method we do not compare against and that would run
on our setting (project F's update onto base directions or subcomponents and compare to chance). The cross-model
transplant is value-patching across models, which our definition deliberately avoids; worth naming as the contrast.
"Full SFT is not the high-rank limit of LoRA" bears on our use of LoRA and full fine-tunes in one table. Scale (32B,
with no recruitment at 7B) puts a replication beyond our 67M model and probably beyond Qwen3-1.7B.

**For the digest.**
- 2-known-answer-case: Qwen2.5-32B, insecure code; LoRA reuses a pre-existing persona direction, full SFT builds a localised circuit (evidence: weight enrichment, transplant, trajectory); labels are representational, not mechanism-level.
- 1-missing-method: enrichment of a weight update along a base-model direction relative to chance; cheap and runnable on our fine-tunes.
- 4-supports: the method, not the data, decides whether a pre-existing structure is used; consistent with our reading that trained updates do not simply rescale base components.
