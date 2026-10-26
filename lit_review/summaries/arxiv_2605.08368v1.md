# On Distinguishing Capability Elicitation from Capability Creation in Post-Training: A Free-Energy Perspective

## Metadata
- **Authors**: Yuhao Li, Shengchao Liu (Department of Computer Science and Engineering, The Chinese University of Hong Kong)
- **Year**: 2026 (arXiv ID 2605.08368, so posted May 2026; exact date not verified from the text)
- **Source**: arXiv preprint (v1). Position paper, no experiments.
- **DOI/URL**: https://arxiv.org/abs/2605.08368 (URL inferred from the arXiv ID)

## Summary
The paper argues that post-training research should separate capability elicitation (reweighting behaviours a base model can already reach under finite budgets) from capability creation (changing the set of reachable behaviours). It frames SFT and KL-regularised RL as reweightings of a pretrained reference distribution under a demonstration-defined or reward-defined "energy", and uses this to sort post-training gains into four regimes: demonstration-covered elicitation, tail reweighting, barrier-crossing discovery, and unsupported regimes.

## Key Findings
- The SFT vs RL framing confounds algorithm with mechanism. Either objective can elicit or create, depending on where its candidate behaviours come from (demonstration source, verifier selection, search, tools, new information).
- Under a strong KL constraint, SFT and RL both reduce to a Boltzmann reweighting of the base model. The paper's free-energy derivation (Sec. 3, App. A) is standard maximum-entropy / KL-regularised RL algebra restated, not a new result.
- "Accessible support" is a diagnostic notion (behaviours reachable at pass@1, pass@N, or under search and tool budgets), not a formal object with an estimator. Reachability is relative to the base model, decoding, evaluation signal and compute budget.
- Four regimes: basin (stabilise), tail (amplify rare but reachable behaviour, which can look like creation in benchmarks), barrier (needs a changed trajectory-generation process), unsupported (base model assigns ~zero probability, so reweighting has nothing to act on).
- The paper's practical test is whether the base model produces the target behaviour under larger sampling budgets. Performance gains alone do not establish creation.

## Methodology
Conceptual and theoretical. The authors define accessible support informally, map SFT and RL onto a free-energy / Boltzmann-reweighting form, derive the optimum with Lagrange multipliers (App. A), and classify regimes by reachability. They respond to four counterarguments (RL can create capabilities; SFT is only imitation; the physics language is metaphor; optimisation dynamics matter). No empirical tests, no models trained, no parameter-space analysis.

## Relevance to Proposal
**Score: 4**

The proposal's core question includes a fork: when a small post-training update is not faithfully carried by a decomposition, is that because post-training creates new mechanisms or because decompositions are brittle. This paper supplies the vocabulary for the "creates vs strengthens" side (elicitation vs creation, tail vs barrier vs unsupported), which could help pre-register what a "creation" outcome would look like. It does not address parameter decomposition, weight-space updates, LoRA geometry, or interpretability, so it does not bear on the commuting map itself.

## Limitations
- No experiments. The regime classification is not tested on any model, and the paper does not say how to measure accessible support in practice beyond pass@N-style budgets.
- Accessible support is acknowledged to be a diagnostic rather than a formal object with a unique estimator, so it cannot be applied to a weight-space claim directly.
- The free-energy framing is, by the authors' own admission (View 3), largely an organising metaphor; the derivation adds no new predictive content.
- Mechanistic (internal-representation) evidence is absent. "Reachable" is defined behaviourally, so the paper says nothing about whether internal mechanisms are reused or rebuilt, which is the question the proposal cares about.
- The regime boundaries (tail vs barrier, barrier vs unsupported) are described qualitatively; no criterion separates them operationally.

## Key Quotes
> "We argue that post-training research should distinguish between capability elicitation and capability creation." (Abstract)

> "Performance improvement is evidence that post-training changed what the model tends to do. But performance improvement alone is not sufficient evidence that post-training changed what the model can practically reach." (Section 6, Conclusion, p. 10)

> "Crucially, for an autoregressive language model, the softmax distributions assign nonzero probability to many token sequences, so strict mathematical support is too weak for capability analysis. What matters is whether a behavior can be practically produced under finite sampling, decoding, search, optimization, and divergence budgets." (Section 4.1, p. 6)
