---
slug: meloux2025-many-algorithms-one-network
title: "Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable?"
authors: Maxime Méloux et al.
year: 2025
id: 2502.20914
tags: [DEFINITION, CAUSAL]
primary_tag: DEFINITION
in_our_bib: meloux2025identifiable
cited_in_draft: yes
themes: [causal-abstraction, reuse-definitions]
read_from: "arXiv HTML (2502.20914): abstract, intro, the two strategies, the XOR example and the headline counts; appendix not read. Identified as the paper behind this row from Sutter et al. 2025 (p1 and its references: 'Méloux ... Everything, everywhere, all at once ... ICLR 2025'). Same paper as master-list row meloux2025-identifiable (batch b23)."
digest_buckets: [3-definition]
importance: 4
---

# Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable? (Méloux 2025)

**Duplicate.** This row is the paper that Sutter et al. cite for "many algorithms can be aligned to the same neural
network". It is the same paper as the row `meloux2025-identifiable` (ICLR 2025). Merge the two entries.

**What it does.** It asks whether, for a fixed behaviour and under the criteria mechanistic interpretability sets
itself, the explanation is unique. It trains MLPs small enough to enumerate every candidate explanation (e.g. two hidden
layers of 3 neurons computing XOR, and other Boolean functions). It then tests two strategies exhaustively. In
"where-then-what", find a circuit and then interpret it as logic gates. In "what-then-where", choose an algorithm and
then search for neuron subsets aligned with it by IIA.

**Key result.** Every stage is non-identifiable. In the XOR example they find 85 circuits with perfect accuracy, with on
average 535.8 logic-gate interpretations per circuit. For the what-then-where strategy, they test only two candidate
algorithms and find 159 perfect minimal mappings, with perfect mappings for both. In an appendix experiment with 59
circuits they count 114,230 interpretations. They argue that sparsity alone (Occam's razor) does not resolve this.

**Quote.** "multiple circuits can replicate model behavior, multiple interpretations can exist for a circuit, several
algorithms can be causally aligned with the neural network, and a single algorithm can be causally aligned with
different subspaces of the network." (Abstract)

**Relevance to our paper.** The draft cites it as showing that explanations of one network need not be unique, "which
bears on fixing one featurization from the base model". That is accurate. Our definition sidesteps uniqueness by
design: the featurization is fixed from M first, and reuse is then a relation at that featurization, so a different
featurization may give a different verdict. That is the price, and our neuron result (≥ 0.999 reuse) shows it. The paper
works on one network at a time and does not treat fine-tuning or the comparison of two networks.

**For the digest.**
- `3-definition`: non-identifiability of circuits and of algorithm alignments, cited correctly. It is the reason our definition is stated relative to a fixed featurization and does not claim a featurization-free verdict.
