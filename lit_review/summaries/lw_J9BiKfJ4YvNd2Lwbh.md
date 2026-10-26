# Reasoning-Finetuning Repurposes Latent Representations in Base Models

## Metadata
- **Authors**: Jake Ward, Chuqiao Lin, Constantin Venhoff, Neel Nanda (equal contribution for first two; completed during Neel Nanda's MATS 8.0 training phase)
- **Year**: 2025 (posted 2025-07-23)
- **Source**: LessWrong (score 36 at time of reading; 1 comment)
- **DOI/URL**: https://www.lesswrong.com/posts/J9BiKfJ4YvNd2Lwbh/reasoning-finetuning-repurposes-latent-representations-in

## Summary
The authors compute a difference-of-means steering vector for "backtracking" (e.g. "Wait", "But") from activations of the base model Llama-3.1-8B at layer 10, using token positions just before backtracking sentences in DeepSeek-R1 reasoning traces. Added to the reasoning-finetuned model DeepSeek-R1-Distill-Llama-8B, the vector induces backtracking; added to the base model, it does not. The authors read this as evidence that the base model already represents an upstream concept that reasoning finetuning learns to use for backtracking, rather than backtracking being wholly new.

## Key Findings
- A steering vector computed from base-model activations induces backtracking tokens in the reasoning-finetuned model, with a substantially larger effect than mean-activation steering, random noise, activation-magnitude scaling, or vectors for other reasoning behaviours (deduction, initialisation).
- The base-model-derived vector and the reasoning-model-derived vector have cosine similarity above 0.7 and behave similarly.
- The base model never exhibits backtracking, even when steered with the reasoning-model-derived vector. The effect is therefore conditional on the finetuned model.
- Logit lens shows the base-derived vectors do not map directly onto backtracking logits at early-to-mid layers, so the effect is not trivially a logit boost on "Wait".
- Authors are explicitly unsure what concept the vector represents, and whether it is monosemantic. An "uncertainty" hypothesis was investigated and was inconclusive.

## Methodology
Generate reasoning rollouts with DeepSeek-R1; annotate backtracking sentences with GPT-4o; compute difference-of-means steering vectors from a 5-token window starting 12 tokens before each backtracking sentence; sample these activations from a forward pass of the base model (Llama-3.1-8B, layer 10) and steer the distilled reasoning model. Controls: logit lens, and comparisons to mean-activation, random-noise, magnitude-scaling, and other-behaviour vectors. The evidence is qualitative and figure-based in the post; effect sizes are not given in the text.

## Relevance to Proposal
**Score: 5**

This is a small, single-model-pair case for the proposal's strand 4 question (does fine-tuning reuse existing representations or build new ones): a feature present in the base model is repurposed by RL/reasoning finetuning, which bears on "strengthen existing vs create new". It does not involve parameter decomposition, weight-space updates (δW), or any map from a decomposition of W to one of W + δW, so it does not address the core commuting question directly.

## Limitations
- Single model family and pair (Llama-3.1-8B base vs DeepSeek-R1-Distill-Llama-8B); no replication across models or scales shown.
- Reported as an informal LessWrong post; quantitative results appear only in figures and the text gives no statistics, seeds, or variance.
- The concept carried by the vector is unidentified; the "upstream concept" interpretation is the authors' hypothesis, not a demonstrated mechanism.
- Steering with random noise also induces some backtracking, though much less than the vector; the size of the gap is not stated numerically in the text.
- Operates on a single layer (10) in the text; layer sweep beyond the logit-lens comment is not described.
- The "base model never backtracks" result is a steering result; it does not rule out other base-model routes to backtracking.

## Key Quotes
> "We interpret this direction as representing some concept upstream of backtracking which the base model has learned to keep track of, while only the reasoning model has learned to use this concept *for* backtracking." (Introduction)

> "Note that the base model never exhibits backtracking behavior, even when steered with the reasoning model-derived backtracking-inducing vector." (Introduction)

> "We take this as evidence that backtracking isn't a totally new behavior learned during finetuning. Instead, base models already contain useful concepts which the reasoning-finetuning process finds and repurposes for emergent reasoning behavior." (Conclusion)

## Comments Summary
One comment, from TurnTrout (score 5): "Nice work. What a cool use of steering vectors!" No methodological pushback or extensions are raised in the thread.
