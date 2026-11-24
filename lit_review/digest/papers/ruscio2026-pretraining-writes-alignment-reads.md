---
slug: ruscio2026-pretraining-writes-alignment-reads
title: "Where Pretraining writes and Alignment reads: the asymmetry of Transformer weight space"
authors: Ruscio et al.
year: 2026
id: 2605.16600
tags: [DIFFING, BASELINE, DYNAMICS, EH]
primary_tag: DIFFING
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, training-dynamics, elicitation, baselines]
read_from: "arXiv HTML: abstract, sec. 1-4 (frame, RSF, gradient analysis), sec. 5-7 (static cohort, Pythia branches, gradient statistics, rank-1 edits, LoRA placement)"
digest_buckets: [1-missing-method]
importance: 4
---

# Where Pretraining writes and Alignment reads: the asymmetry of Transformer weight space (Ruscio 2026)

**What it does.** Asks in which directions alignment changes the weights, not how much. It measures the "relative
subspace fraction" (RSF): the share of a weight delta's energy in a reference subspace, divided by the share expected
for a random orientation. Read matrices (W_Q, W_K) are scored against the principal subspace of their input
activations, and write matrices (W_O, W_2) against the top unembedding directions. Data: 14 base-to-aligned
comparisons over seven open bases (Llama 3 incl. Tulu 3, OLMo 2, Qwen2.5, Mistral; SFT, DPO, ORPO). Also controlled
Pythia-410M branches (continued cross-entropy, SFT, DPO from one anchor), a late pretraining interval, gradient
statistics, rank-1 unembedding-direction edits and LoRA placement tests.

**Key result.** Alignment deltas to the read matrices are enriched in the input subspace (mean RSF 2.08, positive in all
14 comparisons), while write deltas are near random relative to the prediction subspace (1.10). The same signature
appears in late Pythia pretraining (read RSF 3.30) and under all three objectives from one checkpoint. The authors
conclude it belongs to the training stage, not the objective. Alignment's write-side gradients are concentrated (55-59%
of covariance in 50 of 2048 directions) but point outside the prediction subspace. Base-to-SFT deltas are strongly
enriched in the leading singular subspaces of the base weights (8.63 reads, 4.22 writes). A rank-1 write edit along
unembedding directions still suppresses targets by about 2-3 nats in most aligned models.

**Quote.** "what distinguishes alignment in weight space is when it happens rather than what it optimises." (sec. 2)

**Relevance to our paper.** A weight-space diffing reading on realistic alignment fine-tunes that costs almost nothing
and could run on our 67M and Qwen pairs: RSF of the update against base-model activation and unembedding subspaces,
and against the base weights' own singular subspaces. Like our ρ, it is a reading with no known answer behind it. Its
main finding, that the alignment signature equals that of any late update, matches ours that the cheap readings "tell
an update from noise, not reuse from new". Two numbers bear on the rescale-versus-turn question, weakly: write deltas
leave the prediction subspace, yet the updates sit well inside the base weights' top singular subspaces. The paper
frames itself as a weight-space counterpart of the superficial-alignment hypothesis, but measures only orientation, so it
cannot say whether a mechanism is kept.

**For the digest.**
- 1-missing-method: RSF, an orientation reading of the weight delta against base activation, unembedding and base-weight singular subspaces; cheap, runnable on our updates; it would need our known-answer pairs to show it separates reuse from new (likely not, since it reads the same for any late update).
