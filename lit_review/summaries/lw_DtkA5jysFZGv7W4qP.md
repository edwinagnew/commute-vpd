# Training Process Transparency through Gradient Interpretability: Early experiments on toy language models

## Metadata
- **Authors**: robertzk (SERI MATS 3.1; thanks to Evan Hubinger for feedback)
- **Year**: 2023
- **Source**: LessWrong (score 56 at capture; 1 comment)
- **DOI/URL**: https://www.lesswrong.com/posts/DtkA5jysFZGv7W4qP/training-process-transparency-through-gradient

## Summary
The author records the per-datum parameter change produced by every SGD step while training a small 3-layer transformer on WikiText2 (about 12M parameters, roughly 292k datum-level updates, about 882 GB stored). Large parameter shifts are attributed to the training tokens that caused them, and a token-frequency heuristic then proposes which tokens each neuron "learned" from. Zero-weight ablation of the neuron is used to check whether the proposed tokens actually flip predictions.

## Key Findings
- Per-datum attribution works better than per-batch attribution: batch-level shifts showed no discernible relation to the batch contents, while datum-level shifts (recovered by unrolling the batch gradient with `reduction=none`) gave interpretable token patterns.
- Of 46 MLP neurons with some attributed training data, 23 (about 1.9% of MLP neurons) showed attribution to specific tokens. Examples: a "war" neuron (L3.2 N96), a sports neuron (L3.1 N78, "season/league/game"), and a quantity neuron (L3.2 N181, "number"). Ablation flip rates for the attributed tokens are sometimes high (e.g. 56% for "war") but are often low (e.g. 13.8% for "season"), and the authors note many flipped tokens were not attributed.
- Attribution was clearest for late-layer MLP neurons. No early-layer (layers 0-1) neurons were attributed, and the early-layer neuron tested (L1.2 N156) produced diffuse, unhelpful flips.
- Activation-based inspection of the same neurons did not reveal the attributed tokens, which the author takes as evidence that the parameter-shift method adds information beyond the activations.
- The authors report that attention heads could not be attributed with this method, and that the storage cost of full gradient capture is large but, by their estimate, affordable relative to training.

## Methodology
- Model: 3-layer transformer (attention plus MLP, positional encodings, layer norm, dropout 0.2), embedding and hidden dimension 200, 4 heads, context 35, trained for 5 epochs with SGD (lr 5.0, step decay), gradient clipping 0.5. Embedding and unembedding were excluded from recording.
- Recording: the per-datum update `Delta w_{t,j} = -(alpha / (nC)) * sum_k grad` is computed for every datum in every batch and stored as (epoch, batch, datum, unit, index, diff).
- Selection: for each step, the top k=1000 absolute shifts across non-embedding 2-tensors (about 0.13% of parameters) are kept.
- Token ranking: a univariate heuristic (tokens with count at least 20, top 100 by relative frequency against the full training distribution, then top 3 by count). The author reports that Mahalanobis distance, log-scaled counts and KL-based scores all surfaced spurious rare tokens or the most common tokens.
- Validation: zero-weight ablation of the neuron, then a full forward pass over all attributed tokens to measure prediction flips. The author concedes this ground truth is imperfect.
- Single-run, single-model, qualitative; no seeds, no error bars, no held-out evaluation of the attributions.

## Relevance to Proposal
**Score: 3**

The post is a very early, hand-scored attempt at "training-process interpretability": tracking how parameter updates build up over training and attributing them to features. It touches strand 2 (interpretability that tracks training) and strand 3 (structure of updates), but it works in the neuron basis, has no parameter decomposition (no VPD/SPD/APD components, no causal-importance function), and does not test whether a parameter update can be pushed through an existing decomposition. Its useful contribution to the proposal is the cautionary evidence about neuron-basis attribution and the explicit note that non-standard-basis features are hard to recover from parameter shifts, which bears on the "is the update carried by components or by the residual" question.

## Limitations
- Single small model (about 12M parameters), single dataset (WikiText2), single run; no replication or seed variance reported.
- Attribution is qualitative and selected by hand (neurons were chosen because they gave clean tokens). No quantitative precision/recall for attributed tokens.
- Ablation validation is partial: many flipped tokens were not attributed, and the author admits the ground truth is weak for early layers and attention.
- Analysis is in the standard (neuron) basis; the author acknowledges features in superposition would be missed or confounded with noise.
- Attention heads could not be attributed with the method.
- Storage numbers (882 GB for a 12M model, extrapolated to "490 exabytes" for GPT-3) are rough and the extrapolation is not carefully costed.
- Predates Developmental Interpretability; the author notes the framing may be superseded.

## Key Quotes
> "Most attempts at mechanistic interpretability (mechint) focus on taking a completed trained model and performing a static analysis of specific aspects of its behavior and internals." (Introduction)

> "Initially we attempted to record attribution at the level of each SGD batch. However, this proved to be too noisy: there was no discernible relationship between the parameter shifts in a given batch and all of the training data in that batch." (Accounting for datum-level attribution)

> "Note that we are primarily looking at training data attribution along the standard basis, that is, the neuron basis." (Examples of results)

> "Features are directions in activation space, so changes to features should be directions in parameter change space." (Limitations of the method: Explanations outside the standard basis)

## Comments Summary (if LessWrong/AF post)
The post has one comment, from TurnTrout (score 3). The text captured here is mostly MathJax/CSS residue from the original page and adds little. The substantive part is a pointer to a shortform comment on convergently-ordered developmental milestones for AI (e.g., syntax before semantics, object permanence in embodied tasks). The comment is a suggestion for further work rather than a critique of the method. No author replies are present in the captured text.
