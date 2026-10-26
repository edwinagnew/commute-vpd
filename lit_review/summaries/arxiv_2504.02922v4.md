# Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning

## Metadata
- **Authors**: Julian Minder, Clément Dumas (equal contribution), Caden Juang, Bilal Chughtai, Neel Nanda
- **Year**: 2025 (arXiv 2504.02922v4; NeurIPS 2025)
- **Source**: arXiv preprint / NeurIPS 2025 (MATS-affiliated project)
- **DOI/URL**: https://arxiv.org/abs/2504.02922

## Summary
The paper examines whether crosscoders (shared-dictionary SAEs trained on base and fine-tuned model activations) can identify concepts that chat-tuning introduces. It argues that the standard L1-trained crosscoder produces many spurious "chat-only" latents because of two loss-induced artifacts (Complete Shrinkage and Latent Decoupling), and proposes a diagnostic (Latent Scaling) plus a switch to BatchTopK sparsity that substantially reduces these artifacts. Using the BatchTopK crosscoder on Gemma-2-2B base vs. chat, the authors find a set of chat-specific latents that are interpretable and causally effective, including refusal-related latents and latents tied to chat template tokens.

## Key Findings
- The L1 crosscoder's decoder-norm metric (Delta_norm, from Lindsey et al. 2024) labels many latents as chat-only that still carry base-model information: reconstruction ratios overlap heavily with shared latents, and error ratios reach about 0.5 (Section 3.1, Figure 2a).
- BatchTopK crosscoders show almost no such overlap (nu_r and nu_eps near zero for chat-only latents). Delta_norm correlates with the Latent Scaling ratios (Pearson 0.73 for nu_r, 0.87 for nu_eps) (Section 3.1).
- Causal test (KL divergence on 512 chat interactions, Section 3.2): for BatchTopK, the top 50% by Delta_norm reduce KL far more than the bottom 50%; for L1, Delta_norm fails, but Latent Scaling recovers causally important latents. Full replacement gives ~59% KL reduction over all tokens and ~78% over the first 9 tokens for both architectures.
- The reconstruction-error term carries much of the chat behaviour, especially in full responses. For the first 9 tokens, the error term does far worse than the reconstruction, so the crosscoder captures early-token framing better than long generations.
- About 40% of BatchTopK chat-only latents fire mainly on chat template tokens. Replacing base activations with chat activations only at template tokens reproduces KL of 0.239 (full response) and 0.507 (first 9 tokens), roughly matching the top-50% latents (Section 3.3).
- Chat-only latents include refusal triggers, personal-experience/emotion questions, detection of user false information, and joke detection, which the authors describe as interpretable (largely from qualitative examples; autointerpretability scores are reported only as a comparison, around 0.60-0.65, Figure 5).
- The authors note the KL difference between base and chat is about three times larger on the first tokens than over full responses, consistent with Qi et al. (2024).

## Methodology
Trains L1 and BatchTopK crosscoders (L0 target 100) on the layer-13 residual stream of Gemma-2-2B and Gemma-2-2B-it, using web and chat data. Introduces Latent Scaling: closed-form least-squares scales beta for each latent to reconstruct base vs. chat activations. Uses nu = beta_base / beta_chat, plus error (nu_eps) and reconstruction (nu_r) variants to separate Complete Shrinkage from Latent Decoupling. Validates with a hybrid-model KL intervention (patch chat-decoder contributions of a latent set into base activations, run through the rest of the chat model). Autointerpretability and qualitative examples support interpretability claims. Replications on Llama 3 chat models and RL-tuned medical/reasoning models are reported in appendices (not read in detail for this summary).

## Relevance to Proposal
**Score: 6**

This is the most direct prior work on tracking what fine-tuning adds using a shared sparse dictionary across base and tuned models (strand 2: model diffing, feature transfer, training dynamics), and its warning that L1 penalties make norm-based "new feature" labels unreliable matters for any commuting or decomposition-tracking method that reads off new components. It is not a parameter-decomposition paper: it works on activations, with a single fixed checkpoint pair and no update-level map Phi(D, deltaW), so the link to the commuting question is indirect.

## Limitations
- Authors note they study only small models (Gemma-2-2B) at one middle layer, and cannot claim the findings transfer to larger models, other layers, or other widths and sparsities.
- Focus is on chat-only latents; base-only and shared latents are left largely unexplored.
- The paper admits it cannot distinguish genuinely new latents from existing latents whose activation pattern shifted during tuning. This is the central question for the proposal, and the method does not answer it.
- Error terms still hold substantial chat-behaviour information (Figure 4), so the dictionary does not fully capture the behavioural difference.
- Interpretability is largely judged from hand-picked examples and an autointerp score comparison, not a systematic human evaluation.
- Latent Scaling thresholds (nu < 0.2 and nu_r < 0.5 for "chat-specific") are chosen by the authors, and the sensitivity to them is not shown in the main text (from reading, not verified against the appendix).
- Appendix material (Sections E-N, training details) was not read in full for this summary.

## Key Quotes
> "the sparsity loss can force base latent directions to zero norm, even when they contribute to base model reconstruction" (Section 1, Introduction, item 1)

> "a significant limitation is our inability to distinguish between truly novel latents learned during chat-tuning and existing latents that have merely shifted their activation patterns, as the crosscoder architecture does not provide a mechanism to make this distinction" (Section 5, Discussion and limitations)

> "the error term performs more than twice worse than the reconstruction for both crosscoders. This contrast demonstrates that our crosscoders excel at capturing crucial early-token behavior" (Section 3.2)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper).
