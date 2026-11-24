---
slug: li2026-elicitation-vs-creation
title: "On Distinguishing Capability Elicitation from Capability Creation in Post-Training: A Free-Energy Perspective"
authors: Li & Liu
year: 2026
id: 2605.08368
tags: [DEFINITION, EH, CITE-FIX]
primary_tag: DEFINITION
in_our_bib: no
cited_in_draft: no
themes: [elicitation, reuse-definitions]
read_from: arXiv HTML full main text (abstract, sections 1-6); appendix A derivation not read
digest_buckets: [3-definition, 5-cite-fix]
importance: 5
---

# On Distinguishing Capability Elicitation from Capability Creation in Post-Training: A Free-Energy Perspective (Li 2026)

**What it does.** A position paper with no experiments. It argues the SFT-versus-RL framing ("imitation" vs
"discovery") confounds the training signal with what the training does, and proposes the axis elicitation versus
creation. Both SFT and KL-regularized RL are written as minimizing a free energy E_q[E(x,y)] + β·KL(q‖p₀), whose optimum
q* ∝ p₀·exp(−E/β) reweights the base distribution. "Accessible support" (behaviours the base can practically produce
under finite sampling, decoding, search and divergence budgets) is the dividing line: reweighting within it is
elicitation, changing it is creation.

**Key result.** A four-regime hierarchy: demonstration-covered elicitation (stabilizing a high-probability basin),
tail reweighting (amplifying behaviours rare at pass@1 but present under pass@N), barrier-crossing discovery (behaviours
needing a chain of low-probability intermediate steps, reachable only when reward is coupled with search, interaction
or tools), and unsupported regimes (p₀ = 0, where reweighting is undefined and new information is needed). It
concludes that creation should be credited to search, tools or new information coupled with reward, not to "RL" as a
label, and that capability claims "need to specify their mechanism".

**Quote.** "Post-training that reweights behaviors within this support is capability elicitation; whereas changing the
support itself corresponds to capability creation." (Abstract)

**Relevance to our paper.** A recent, explicit definition of elicitation vs creation to cite and contrast in "Defining
reuse" or "Post-training as selection". It is defined entirely on the output distribution (what the base model can
sample), so it is exactly the behavioural kind of definition our gap sentence says cannot look inside the model: two
fine-tunes that reweight the same accessible behaviour count as the same here, while ours can call one reuse and one
new by whether the needed variables keep M's mechanisms. Its "mechanism" means the training process behind the gain
(stabilization, tail amplification, search), not an internal computation, so citing its call to "specify the
mechanism" as support for a mechanistic test should be worded carefully. Its tail-reweighting regime is the theory
behind Yue et al. (pass@k), which the draft cites, and it maps cleanly onto our cued-copy / prior-matched-cipher
construction, where the base prior on the answer is pulled apart from the mechanism: a fine-tune can raise an output
the base already assigns probability to (elicitation here) while needing a new computation (new under ours), and the
cued copy shows the converse. That example is a sharp way to state the
difference between the two definitions.

**For the digest.**
3-definition: behavioural elicitation-vs-creation by "accessible support" of the base output distribution; contrast with our mechanism-level reuse (our prior-vs-mechanism pairs are cases where the two disagree).
