---
slug: makelov2023-subspace-illusion
title: "Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching"
authors: Aleksandar Makelov et al.
year: 2023
id: 2311.17030
tags: [CAUSAL, BACKGROUND]
primary_tag: CAUSAL
in_our_bib: no
cited_in_draft: no
themes: [causal-abstraction, reuse-definitions]
read_from: "PDF text (v2), local copy: abstract, intro, the illusion and its toy example (pp1-8), IOI MLP results table and the v_resid success case (pp9-12, Table 1), factual-recall and rank-1 editing section (pp19-20); appendices only grepped"
digest_buckets: []
importance: 4
---

# Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching (Makelov 2023)

**What it does.** It shows that patching along a 1-D subspace found by optimisation (DAS, or a gradient direction) can
change a model's output in the intended way without that subspace being what the model uses. The patch can move a
direction that is the sum of a "causally disconnected" part (it varies with the concept but lies in the nullspace of
downstream weights) and a "dormant" part (it can steer the output but is constant on the data). Patching the sum
activates a pathway the model never uses. This is shown in a toy example, on IOI in GPT-2 small (layer-8 MLP against the
residual stream), and on factual recall in GPT-2 XL. The factual-recall case is linked to ROME's rank-1 edits.

**Key result.** On IOI, patching the DAS direction in the MLP gives a 46.7% fractional logit-difference decrease. This
drops to 13.5% when only the part of the direction that reaches the output is patched, and to -8% when the whole MLP
activation is patched. The residual-stream direction is the success case. Its three corresponding patches read 140.7%,
127.5% and 123.6%, and it has cosine ≈ 0.78 with the gradient direction that downstream heads read. For facts, the
authors show "an approximate equivalence between 1-dimensional subspace interventions and rank-1 editing". This explains
why ROME works in layers where the fact is not stored.

**Quote.** "even if a subspace intervention makes the model's output behave as if the value of a feature was changed,
this effect may be achieved by activating a dormant parallel pathway" (Abstract)

**Relevance to our paper.** Our needed test interchanges variables fixed from the base model (VPD subcomponents, SAE
latents, neurons), not directions optimised to fit an algorithm, so the illusion as stated does not hit our definition.
It bears on three places. (1) Our mean-difference steering share is a 1-D intervention, and the same dormant-pathway
reading could make it "recover" a gain through a path F does not use. (2) Rank-1 weight edits can work by building a
dormant pathway, which is relevant to our built edits of the weights: an edit labelled "strengthen" could act through a
new path, so the head-ablation check on labels matters. (3) It is a precedent for a basis-relative verdict: in a rotated
basis the illusory direction looks meaningful. The paper studies one model only, with no fine-tune, so it is not a
known-answer case or a diffing baseline. Optional cite if the draft discusses why a steering reading can mislead.

**For the digest.** No bucket. It is a caution for our 1-D steering reading and for rank-1 built edits, not a method,
definition or known answer.
