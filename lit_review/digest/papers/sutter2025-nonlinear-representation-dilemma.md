---
slug: sutter2025-nonlinear-representation-dilemma
title: "The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?"
authors: Denis Sutter et al.
year: 2025
id: 2507.08802
tags: [CAUSAL, DEFINITION, SUPPORTS]
primary_tag: DEFINITION
in_our_bib: sutter2025
cited_in_draft: yes
themes: [causal-abstraction, reuse-definitions]
read_from: "PDF text (v2, NeurIPS 2025), local copy: abstract, intro, definitions and Theorem 1 (pp1-6), experimental setup and results (pp7-9), discussion and limitations (p10); proofs in the appendix not read"
digest_buckets: [3-definition, 4-supports]
importance: 6
---

# The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability? (Sutter 2025)

**What it does.** It asks what causal abstraction claims mean once the alignment map from network to algorithm may be
non-linear. It restates the Beckers-Halpern definitions for a network and an algorithm. It then proves (Theorem 1) that,
under five assumptions, any algorithm is an input-restricted distributed abstraction of any network that solves the
task. The assumptions are a countable input space, input-injective hidden layers, strict output-surjectivity, matchable
orderings and task success. It tests this with DAS-style training, using identity maps, orthogonal (linear) maps and
reversible residual networks (non-linear). The tests are run on a 3-layer MLP for hierarchical equality and on Pythia
31M-410M for IOI, across training checkpoints.

**Key result.** With non-linear maps the layer-dependent drop that linear DAS shows on hierarchical equality vanishes:
IIA is near-optimal at every layer. On a randomly initialised MLP the most complex map reaches over 80% IIA. On IOI the
ABAB-ABBA algorithm aligns with all Pythia sizes, including the 31M and 70M models that fail the task, and "this
alignment is perfect even for randomly initialised models across all sizes". During Pythia-410M training, an 8-layer map
gives near-perfect IIA at every checkpoint. Simpler maps reach it only as training proceeds, and linear maps only in the
fully trained model.

**Quote.** "we prove that under reasonable assumptions, any neural network can be mapped to any algorithm, rendering
this unrestricted notion of causal abstraction trivial and uninformative." (Abstract)

**Relevance to our paper.** The draft cites this paper correctly: with an unrestricted map, abstraction is vacuous, and
our answer is to fix one featurization from the base model in advance. Two further points bear on our results.
- The paper's control is a random network, and our random-update floor plays the same role. Our Qwen result fits its
  warning: the re-gating share sits at about 0.99 or higher with random changes included, so a reading without a floor
  under it says nothing. That reading uses VPD's fitted gate function Γ on F's activations, and the least-squares carry of
  F's update onto subcomponents is also a fitted map, so the "fix it in advance" answer covers τ but not every fitted step.
  This is a reader's inference, not the paper's claim.
- The training-checkpoint result shows that how simple the map can be, rather than whether a map exists, is what tracks
  learning. That echoes our finding that ρ "tells an update from noise, not reuse from new".

The paper does not compare two trained networks or treat fine-tuning.

**For the digest.**
- `3-definition`: the vacuity result our fixed featurization answers. It is cited correctly; the draft could add that our fitted steps (the least-squares carry, re-gating with Γ) are where the dilemma could re-enter.
- `4-supports`: a random-model control can reach perfect alignment under expressive maps, which matches our need for a random-update floor and the saturation of the re-gating share on Qwen.
