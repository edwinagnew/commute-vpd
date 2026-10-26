# Sparse Autoencoders Trained on the Same Data Learn Different Features

## Metadata
- **Authors**: Gonçalo Paulo, Nora Belrose (EleutherAI)
- **Year**: 2025 (arXiv v2; preprint, January 2025 first version)
- **Source**: arXiv preprint (not peer-reviewed in the text provided)
- **DOI/URL**: https://arxiv.org/abs/2501.16615

## Summary
The paper asks whether sparse autoencoders (SAEs) recover a unique, objective set of features in a language model. Training SAEs on identical activations and data order, differing only in the random initialisation seed, the authors match latents across seeds with the Hungarian algorithm and find that only about 30% (Llama 3 8B, 131K latents) to 42% (Pythia 160M, 32K latents) of features are shared. They conclude that an SAE's feature set is a useful decomposition of activation space, not a canonical list of what the model "truly uses".

## Key Findings
- Matched-latent cosine similarities (encoder and decoder) are bimodal: a high-similarity "shared" mode and a low-similarity "orphan" mode. Only 42% of latents are shared between two seeds of a 32K Pythia-160M SAE (threshold 0.7 on both encoder and decoder). For Llama 3 8B only about 30% are shared.
- The shared fraction falls as the number of latents grows and as the number of active latents (TopK k) grows. It rises with training time. Smaller models and smaller SAEs show higher overlap.
- With 9 seeds, about 35% of latents are still found in only one SAE. A power law with an offset fits the decline in orphan fraction better than one without, so a residual set of orphans appears to persist at small seed counts.
- Orphan latents are often interpretable: the mean detection score across all 32K latents is 0.72, and some high-scoring explanations occur among low-alignment pairs (Table 1). Latents shared across more seeds score higher on average.
- Standard L1 ReLU and Gated SAEs are more seed-stable than TopK SAEs at matched sparsity. The authors interpret this as TopK SAEs being more seed-dependent.
- Prior reports of high cross-seed overlap (>90% on GPT-2 SAEs) used fewer latents and ReLU architectures. The authors also note that the Balagansky et al. layer-alignment result may depend on Gemma Scope using a single shared seed.

## Methodology
- Two or more SAEs are trained on the same activations (sixth MLP of Pythia 160M, 2^15 latents, first 8B Pile tokens, identical data order), varying only the initialisation seed. Llama 3 8B, SmolLM, and GPT-2 are used for further checks, plus several architectures (ReLU, Gated, TopK).
- Latents are aligned with the Hungarian algorithm, maximising mean cosine similarity of matched encoder and decoder vectors. A latent counts as shared if its match is the same in both encoder and decoder matchings and both cosines are at least 0.7. The authors check that max-cosine matching gives nearly identical results.
- Feature meaning is scored with the automated detection-scoring pipeline of Paulo et al. (2024), using 100 active and 100 inactive sequences per explanation.
- Ablations vary latent count, k, training tokens, and architecture. Asymptotic trends are estimated from 9 seeds (36 pairwise matchings).

## Relevance to Proposal
**Score: 5/10**

This is directly relevant to strand 1 (stability, seed dependence and identifiability of decompositions, and matching components across runs) and to the proposal's "too brittle or non-identifiable" alternative. It is an SAE result, not a parameter-decomposition result, so it bears on the proposal only by analogy: if features from a dictionary-learning method are not unique across seeds, a warm-started or commuted decomposition needs a matching-based definition of "the same component". It also suggests that comparing a commuted decomposition against a fresh re-decomposition of W+δW should use the same matching-and-threshold rigour, and report an across-seed noise floor.

## Limitations
- Main quantitative claims use one model and one layer (Pythia 160M, layer 6 MLP). Llama 3 8B is the only larger model shown, and it is measured on a single SAE pair.
- The "shared" fraction depends on a chosen 0.7 cosine threshold, applied to both encoder and decoder. The authors show the trend is robust, not that the absolute number is.
- Seed is the only varied factor; data order is held fixed, so the result does not separate seed effects from data-order effects.
- Interpretability is judged with an automated detection score, not human evaluation. The authors note that their absorption metric was tuned for residual-stream SAEs and found no absorption evidence in MLP SAEs, which they admit may be a measurement artefact.
- The orphan-fraction extrapolation rests on nine seeds and a fitted power law with an offset. Asymptotic claims are not tested beyond that.
- The paper makes no claim about functional equivalence of the features, only about matched weight directions. Whether orphan latents change model behaviour is not tested.
- Several comparison works are cited as "Anonymous" (under review at the time).

## Key Quotes
> "only 30% of the features were shared across different seeds" (Abstract)

> "the set of features uncovered by an SAE should be viewed as a pragmatically useful decomposition of activation space, rather than an exhaustive and universal list of features 'truly used' by the model" (Abstract)

> "We use this to formally define a shared latent: a latent is shared if it has the same counterpart latent in both the encoder and decoder matchings, and in both of these matchings it has a cosine similarity of 0.7 or greater." (Section 3, Methods)

> "SAEs seem to diverge, rather than converge, with increasing scale." (Section 7, Conclusion)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is an arXiv preprint and no comment thread was provided.
