# [Paper] Stochastic Parameter Decomposition

## Metadata
- **Authors**: Lee Sharkey (LessWrong post author). The post body does not list the paper's authors; the research proposal attributes SPD to Bushnaq et al. 2025.
- **Year**: 2025 (post dated 2025-06-27; arXiv 2506.20790)
- **Source**: LessWrong (score 47; one top-level comment thread with nested replies)
- **DOI/URL**: https://www.lesswrong.com/posts/yjrpmCmqurDmbMztW/paper-stochastic-parameter-decomposition ; arXiv: https://arxiv.org/abs/2506.20790 ; code: https://github.com/goodfire-ai/spd

## Summary
The post announces Stochastic Parameter Decomposition (SPD), a linear parameter decomposition method meant to replace Attribution-based Parameter Decomposition (APD) as the practical method in that framework. The authors claim SPD is cheaper, less sensitive to hyperparameters, avoids shrinkage of learned parameters, and recovers ground-truth mechanisms better in toy models, and that it scales to slightly larger models than APD could handle. The post itself is a short announcement: the body is the abstract plus links, so the method details below come from the abstract and the comment thread, not from the paper text.

## Key Findings
- Authors' claim: SPD scales better and is more robust to hyperparameters than APD, and decomposes models slightly larger and more complex than APD could.
- Authors' claim: SPD avoids parameter shrinkage and better identifies ground-truth mechanisms in toy models.
- Commenter critique (J Bostock): because the faithfulness loss drives SPD, the method may simply return the whole linear transformation of a weight matrix when the subcomponents are overcomplete, so the decomposition may carry little information. The critique is based on the toy model's compressed-computation example, where the input-side matrix is found to be the input basis and the MLP middle layer is found to be the whole linear map. This was not resolved in the thread.
- Lucius Bushnaq (a co-author of the framework, per the thread) disputes the critique: nonlinearities after a matrix cause SPD to split it into many subcomponents, and the stochastic reconstruction loss ablates all matrices at once, not one at a time. He also argues that output matrices are not expected to decompose into overcomplete circuit sums, which is why SPD may struggle there.

## Methodology
Linear parameter decomposition: a weight matrix is written as a sum of sparsely used rank-one subcomponents in parameter space. SPD learns these subcomponents with a stochastic-layerwise reconstruction loss (each matrix is replaced by a randomly ablated version during forward passes) and a faithfulness loss so that the summed components reproduce the original network. Evaluation is on toy models (including a compressed-computation model with known ground truth) and on somewhat larger models than APD could handle. The post gives no quantitative results.

## Relevance to Proposal
**Score: 6**

SPD is the sibling of VPD, the decomposition the proposal's "commuting" step would have to act on, so the stability, hyperparameter-robustness and faithfulness-loss claims bear directly on whether a parameter decomposition is stable enough for an update map Φ(D, δW) to be meaningful (strand 1, and the "too brittle" branch of the core question). The comment thread's argument that the faithfulness objective can collapse to the full linear map is a concrete risk to check for VPD before trusting any projection-based Φ.

## Limitations
- The post is an announcement and contains no experimental numbers; all claims come from the authors' abstract.
- Evaluated only on toy models and modestly larger networks; no evidence on LLM-scale or post-trained models.
- Commenter critique (unresolved in thread) that the faithfulness loss makes overcomplete decompositions degenerate to the full linear transformation.
- Disagreement over whether output-side (W_out) matrices can be decomposed meaningfully, including the unembedding interaction.
- No seed-stability or component-matching results are visible in the post itself; these would need checking in the paper.

## Key Quotes
> "In this work, we introduce *Stochastic Parameter Decomposition* (SPD), a method that is more scalable and robust to hyperparameters than APD" (Abstract)

> "I don't see how this method does anything at all, and the culprit is the faithfulness loss." (Comment by J Bostock, top-level)

> "SPD will decompose you a matrix into a single linear transformation *if* what the network is doing with that matrix really is just applying one global linear transformation. If e.g. there are non-linearities right after the matrix that aren't just always switched on, SPD will usually decompose the matrix into many sub-components." (Comment by Lucius Bushnaq, reply)

## Comments Summary
The single top-level comment (J Bostock, score 2) argues SPD mainly returns the full linear transformation when the subcomponent basis is overcomplete, citing the compressed-computation toy model: the input matrix is decomposed into the 100-element input basis and the MLP middle layer into the full linear map, with the output matrices left undecomposed. He also argues that since the toy model's noise is not optimised (embedding and unembedding are fixed), it differs from real networks in a way that matters for superposition-based claims, and that the premise of true computation in superposition does not hold there (his later reply retracts the word "superposition" in favour of "compressed computation").

Lucius Bushnaq (score 4 on the top reply, with follow-ups) disputes the central claim: nonlinearities break the single-linear-map collapse, the stochastic loss ablates all matrices simultaneously, and he expects W_in to decompose into co-activating circuits but W_out to be a global linear read-out, because circuit-level W_out sums are noisy. He notes that a prior circuits-in-superposition construction had this error and was corrected. The thread ends with him declining to continue the longer argument on whether superposition noise can be helpful, and with the point that a rotation or identity over superposed features should not be split into rank-one pieces. No resolution is reached; the thread leaves the question of whether SPD (and by extension VPD) genuinely decomposes overcomplete structure open.
