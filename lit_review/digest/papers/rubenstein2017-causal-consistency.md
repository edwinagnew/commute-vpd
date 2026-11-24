---
slug: rubenstein2017-causal-consistency
title: "Causal Consistency of Structural Equation Models"
authors: Paul K. Rubenstein et al.
year: 2017
id: 1707.00819
tags: [DEFINITION, BACKGROUND]
primary_tag: DEFINITION
in_our_bib: no
cited_in_draft: no
themes: [causal-abstraction, reuse-definitions]
read_from: "arXiv abstract page and PDF text (UAI 2017; Secs 1, 4.3-4.4: Definition 3 and the causal-consistency discussion; the rest skimmed). Identified from Beckers and Halpern 2019 (the 'RW+' reference) and Geiger et al. 2024 (p1)."
digest_buckets: [3-definition]
importance: 2
---

# Causal Consistency of Structural Equation Models (Rubenstein 2017)

**What it does.** It formalises when two structural equation models (SEMs) of one system, at different levels of
detail, are consistent: they agree on the effects of interventions. It defines an exact τ-transformation from one SEM to
another. There is a state map τ and a surjective, order-preserving map of interventions, such that intervening low and
then mapping up gives the same distribution as mapping the intervention up and intervening high. The cases covered are
marginalising out variables, aggregating micro-variables into macro-variables, and replacing a time-series model by its
stationary behaviour.

**Key result.** A theory paper. Exact transformations are transitive. If one SEM is an exact transformation of another,
both "can sensibly be thought of as causal models of the same system" (Sec. 4.4). The paper stresses well-specified
interventions and sheds light on cyclic SEMs.

**Quote.** "Ideally, causal models of the same system should be consistent with one another in the sense that they agree
in their predictions of the effects of interventions." (Abstract)

**Relevance to our paper.** It is the origin of the commuting-diagram notion that Beckers and Halpern tighten and that
Geiger et al. 2025 inherit, so it is part of the ancestry of our featurized models. It relates a model to a coarser
model of the same system, not two trained networks, so it does not define reuse. Beckers and Halpern show the notion is
too weak (unrelated models can be exact transformations of each other). Citing Geiger 2025, and optionally Beckers and
Halpern, covers what we need; this paper is optional.

**For the digest.**
- `3-definition`: the original exact-transformation (consistency) notion behind causal abstraction. It is ancestry only, because it relates two levels of one system, not two networks.
