---
slug: nguyen2026-beyond-shallow-alignment
title: "Beyond Shallow Alignment: How Post-Training Methods Determine Refusal Circuits And Steering Robustness"
authors: Nguyen et al.
year: 2026
id: 2609.03887
tags: [FT-INTERP, DIFFING, CAUSAL, SUPPORTS, ALIGNMENT]
primary_tag: FT-INTERP
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, model-diffing, elicitation]
read_from: arXiv HTML abstract, sections 1-6 and limitations; appendix figure captions on base-vs-post cosine similarity
digest_buckets: [4-supports]
importance: 4
---

# Beyond Shallow Alignment: How Post-Training Methods Determine Refusal Circuits And Steering Robustness (Nguyen 2026)

**What it does.** Fine-tunes three base models (Llama-3.1-8B, Gemma-2-9B, Qwen3-8B, full fine-tuning from the
pretrained checkpoints) to refuse harmful requests with three objectives on matched data (Alpaca plus BeaverTails):
plain SFT, reasoning-augmented SFT (Ra-SFT, reasoning chains before the decision) and ORPO. It then compares the
fine-tunes' refusal directions (difference in means over 256 harmful/harmless pairs, Arditi et al.), localises
refusal by activation patching and attribution patching over layers, heads and MLPs, and tests the localised parts
with ActAdd and ITI steering.

**Key result.** Every objective moves the refusal direction away from the base model's and from the other
objectives'; SFT and ORPO stay closer to each other than either is to Ra-SFT, in all three models. Circuit makeup
depends on the objective: in Llama, SFT refusal is carried by one attention head (L30 H25), Ra-SFT by the last MLP,
ORPO by diffuse MLPs. All three objectives lower attack success below the base's, by different amounts
(WildJailbreak ASR on Llama: base 55.5%, Ra-SFT 45.3%, ORPO 21.8%), and steering and attack vulnerability differ by
circuit type. Patching shows causal effects from the first
layer, which the authors read as harm concepts inherited from pre-training.

**Quote.** "training method, not just data, reshapes how refusal is computed internally" (abstract)

**Relevance to our paper.** This supports the motivation of claim 1: fine-tunes on the same data, all trained to refuse,
implement refusal with different components (their behaviour is not identical, so this is suggestive only), so a reuse test has to look inside. It also hints that
trained fine-tunes turn the base direction rather than scale it, which is close to our finding that trained updates
turn writes. But the evidence is cosine similarity of difference-in-means directions, and the circuits are traced in
the fine-tunes only, never in the base. So it is not a known-answer case. Its patching plus steering pipeline is the
same family as our neuron CMAP and steering baselines, not a new method. It points to Du et al. 2025 (post-training
keeps knowledge representations but changes the refusal direction), which is not in our bibliography and should be
checked. Credibility caution: single runs, and 18 of 256 harmless prompts (7%) flip refusal label between two identical runs.

**For the digest.**
- 4-supports: same data, different objectives give different refusal circuits for one trained behaviour, which supports "two fine-tunes with one input-output behaviour can differ"; the refusal direction also moves away from the base's.
