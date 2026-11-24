---
slug: que2026-em-see-say-sorted
title: "See It, Say It, Sorted: Mechanistic Diagnosis and Parameter-Space Mitigation of Emergent Misalignment in LLMs"
authors: Que et al.
year: 2026
id: 2609.34970
tags: [ALIGNMENT, FT-INTERP, DYNAMICS, EH]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, training-dynamics, elicitation]
read_from: "arXiv HTML: abstract, sections 1-5 (setup, curvature, subspace overlap, interventions, discussion); appendices not read"
digest_buckets: [4-supports]
importance: 3
---

# See It, Say It, Sorted: Mechanistic Diagnosis and Parameter-Space Mitigation of Emergent Misalignment (Que 2026)

**What it does.** Studies emergent misalignment (EM) from narrow fine-tunes (Turner et al.'s risky-finance,
extreme-sports and bad-medical datasets) on Qwen2.5-3B/7B/14B-IT, Llama-3.2-3B/3.1-8B-IT, Gemma-3-4B/12B-IT and
GPT-OSS-20B, mostly with a rank-1 LoRA on one mid-to-late MLP down_proj. Along training it tracks directional Hessian
curvature on "pivot" tokens and the Grassmannian overlap of pivot-token gradients with rank-4 SVD subspaces of
harmful-data and safe-data training gradients. It then projects the harmful gradient subspace out of the final LoRA
update (or amplifies it 1.5x), against norm-matched random subspaces.

**Key result.** Pivot-token curvature ends 2.6x to 21x the neutral-token curvature. In most settings the pivot
gradients' overlap with the harmful subspace plateaus while overlap with the safe subspace declines, with named
exceptions. Removing the harmful component cuts free-generation EM by 55% to 80% on Qwen2.5-14B-IT (p < 0.05), and in
models with near-zero behavioural EM the same edit moves teacher-forced log-probabilities of EM responses both ways
beyond random controls.

**Quote.** "EM is not driven by continuous rotation toward a de novo malicious circuit. Instead, in most settings,
gradient update establishes early alignment with harmful directions that stabilizes or varies non-monotonically,
while overlap with safe subspace declines." (Section 1)

**Relevance to our paper.** It is an elicitation-style reading of EM ("allowing pre-existing ... unaligned
pre-training behaviors to reassert themselves"), but the evidence is about gradient subspaces of a rank-1 adapter's
parameters, not about base-model mechanisms. It does not show what the misaligned outputs are computed by, so it is
not a known-answer case. "Rotation" here means rotation of gradient subspaces over training, not our sense of a
fine-tune turning a mechanism's writes, so it neither supports nor contradicts our rescale-vs-rotate finding. Its
ablate/amplify-a-subspace-of-the-update test, with norm-matched Haar random controls, is a cheap control design close to
our random-update floor. At most one line in related work on EM; Wang et al. and Soligo et al. carry the mechanistic
claim it relies on.

**For the digest.**
- 4-supports: weak, behavioural-and-gradient support for "narrow fine-tuning erodes constraints rather than builds a new circuit" in EM; not mechanism-level.
