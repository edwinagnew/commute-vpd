# Tied Crosscoders: Explaining Chat Behavior from Base Model

## Metadata
- **Authors**: Santiago Aranguri
- **Year**: 2025
- **Source**: LessWrong (posted 2025-03-22; described by the author as a research update, not a finished paper)
- **DOI/URL**: https://www.lesswrong.com/posts/3T8eKyaPvDDm2wzor/research-question (code: https://github.com/aranguri/tied-crosscoder)

## Summary
The author proposes a "tied crosscoder" for model-diffing between a base model and its chat fine-tune (Gemma 2 2B vs Gemma 2 2B it, layer 14). Each latent group has three members (base-only, shared, chat-only) that share a decoder direction, so the same concept can fire at different times in the two models. The author argues this yields more monosemantic base and chat latents and lets one trace which base-model features chat behaviours (e.g., refusal) are built from.

## Key Findings
- In a standard crosscoder, base-exclusive and chat-exclusive latents are qualitatively very similar. The author reads this as the chat model reusing base features in new contexts, which forces duplicate exclusive latents (Experiment 4 shows exclusive latents are, on average, more similar to other latents than shared latents are).
- The tied crosscoder's base and chat latents look more monosemantic on inspection. Top-left (chat-heavy) latents are mostly interpretable as desirable chat behaviours (refusal, instruction detection, step-by-step lists, names). Latent 10615 fires on chat refusals and on base-model refusal and disclaimer-like text, which the author presents as evidence that chat refusal is assembled from several base refusal features.
- A toy experiment (Experiment 2) shows that inducing correlation between two features by fine-tuning produces near-identical exclusive latents (decoder cosine 0.97) in a standard crosscoder.
- On Gemma 2 9B with separate SAEs, the Jaccard overlap between "cannot" and "harmful" latents is about 0.19 for base and 0.39 for chat (Experiment 3). The author calls this "potential evidence" that fine-tuning increases correlation between these features.
- The author states open questions: whether any chat latent is entirely new (not present in the base model), and whether a smaller regularisation weight eta gives more interpretable results.

## Methodology
Training a crosscoder on concatenated base and chat residual-stream activations at one layer, with 16384 x 3 latents organised in tied triples: a shared encoder and decoder direction per triple, plus base-only and chat-only encoder rows that tie to the same decoder vectors. The regulariser weights base-only and chat-only firing by eta = 2 so that the shared latent captures co-firing. Training used 250M tokens of Pile and LMSys chat data. Evaluation is by counting firing ratios across datasets (base/shared and chat/shared), by reading top activations, and by cosine similarities between decoder vectors. Experiments 2 and 3 use a toy model and a fine-tuning set-up that is only loosely matched to the main experiment. No quantitative reconstruction or interpretability benchmark is reported for the tied versus standard crosscoder.

## Relevance to Proposal
**Score: 5**

The post is about model-diffing and reuse versus creation of features under fine-tuning, which matches strands 2 (base-to-chat feature transfer, model diffing) and 4 (does fine-tuning reuse existing mechanisms) of the proposal. Its central claim, that fine-tuning enhances existing mechanisms rather than creating new ones, is the kind of hypothesis the proposal wants to test, though the work uses SAE latents rather than a parameter decomposition, so it does not address whether a decomposition can be carried across an update.

## Limitations
- The author labels this a research update. Evidence is largely qualitative (hand-picked latents, top-activation inspection) with no held-out or quantitative interpretability comparison against the standard crosscoder.
- The main model pair is a single one (Gemma 2 2B vs 2B it, one layer). Experiment 3 uses a different model size (9B) and different SAEs, so it does not test the same pair.
- Experiment 3 is one comparison of set overlaps with thresholds not fully specified in the post, and the author notes "there could be some other phenomenon going on."
- Only eta = 2 was tried for the main result; the author states that the choice of eta changes how "new" latents can appear.
- The tied architecture has about twice the parameters of the standard crosscoder (encoder has 3x, decoder the same), which the author notes complicates a fair comparison.
- The claim that chat latents are more localised in the tied crosscoder is listed as an untested next step.
- Uses SAE latents, not weight-space decomposition. Nothing here checks whether an update to the weights can be carried through the decomposition.

## Key Quotes
> "since fine-tuning enhances existing mechanisms, the chat model reuses the features from the base model in new contexts. Hence, the base and chat models would like to fire the same latent at different times, forcing the crosscoder to have two exclusive latents for the same concept." (Abstract)

> "This evidence suggests that fine-tuning enhances existing mechanisms. Further, it serves as a proof-of-concept demonstrating how this modified crosscoder can precisely identify which base model capabilities enable specific chat behaviors." (Experiment 1, "Explaining chat behavior from base model")

> "this is potential evidence that the Cannot and Harmful features are more correlated in the chat than in the base model." (Experiment 3)

## Comments Summary (if LessWrong/AF post)
The file contains no comments section, so no commenter discussion is summarised. The post invites collaboration through a linked Arbor Project discussion, which is not included in the file.
