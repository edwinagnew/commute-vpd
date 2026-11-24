---
slug: prakash2024-finetuning-existing
title: "Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking"
authors: Prakash et al. (Rott Shaham, Haklay, Belinkov, Bau)
year: 2024
id: 2402.14811
tags: [KNOWN-ANSWER, EH, FT-INTERP, CAUSAL, BASELINE, DIFFING, SUPPORTS]
primary_tag: KNOWN-ANSWER
in_our_bib: prakash2024
cited_in_draft: yes
themes: [elicitation, interpreting-finetunes, causal-abstraction, baselines, model-diffing]
read_from: "local PDF text (ICLR 2024, arXiv v1): abstract, Sec. 1, 4 (circuit, Table 1), 5 (DCM functionality), 6 (CMAP, Fig. 4)"
digest_buckets: [1-missing-method, 2-known-answer-case, 3-definition, 4-supports]
importance: 9
---

# Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking (Prakash 2024)

**What it does.** Finds the entity-tracking circuit in Llama-7B by path patching (72 heads in four groups) and tests it,
graph unchanged, in three fine-tunes: Vicuna-7B (ShareGPT chat), Goat-7B (arithmetic, LoRA) and FLoat-7B (same data,
full fine-tune). It uses Desiderata-based Component Masking (DCM) to label each group's function, and introduces
cross-model activation patching (CMAP): fine-tuned head outputs are patched into the base model on the same input.

**Key result.** The base circuit's faithfulness in the fine-tunes is 0.97 (Vicuna), 0.89 (Goat) and 0.88 (FLoat).
Accuracy goes from 0.66 to 0.82 for full Goat/FLoat, against 0.72-0.73 for the circuit alone, and random circuits score
0.00-0.01 (Table 1). Every model uses the same positional algorithm (Position Detector, then Position Transmitter, then
Value Fetcher). With CMAP, patching only the Value Fetcher heads from Goat or FLoat into Llama brings Llama up to the
fine-tune's performance. The gain is attributed to better handling of "augmented positional information".

**Quote.** "Our findings suggest that fine-tuning enhances, rather than fundamentally alters, the mechanistic operation
of the model." (abstract)

**Relevance to our paper.** This is the canonical realistic known-answer case for reuse. It has independent evidence by
circuit faithfulness, DCM labelling and CMAP in three real fine-tunes of Llama-7B, with Goat (LoRA) and FLoat (full)
trained on the same data. The draft cites it and adapts CMAP at neuron level as a baseline (it reverses 3-5 of 9 pairs).
Two cautions for how we use it. (i) "Enhances" is shown by value patching across models, which our definition rules out
("by mechanism, not by value"), so Prakash's evidence is that F's circuit computes better values with the same
components. It is not evidence that the heads' mechanisms are kept up to a gain. Under our Def. 1 those Value Fetcher
heads could well read as "changed", like our copy head (0.09-0.14 rescale). That makes it the most direct
real-model test of whether our kept clause is too strict. (ii) The 88-89% faithfulness also means extra components
outside the base circuit carry part of the gain. If the cheap readings could run on Llama-7B / Goat, this is the
known-answer case the reviewers will ask for. At minimum the paper should say why it was not run (scale, no
decomposition of Llama-7B).

**For the digest.**
- 1-missing-method: CMAP (cross-model activation patching) and DCM; we use a neuron-level CMAP adaptation; DCM is not used.
- 2-known-answer-case: Llama-7B to Vicuna/Goat(LoRA)/FLoat(full) on entity tracking: same circuit (faithfulness 0.88-0.97), same algorithm, gain localized to Value Fetcher heads by CMAP = reuse with enhancement.
- 3-definition: "enhances existing mechanism" = same circuit and same functionality (DCM), better values (CMAP); value-based, contrast with our by-mechanism kept clause.
- 4-supports: supports EH in a real model; may also contradict our "turns rather than scales" reading unless the enhanced heads would also read as rotated, which is untested.
